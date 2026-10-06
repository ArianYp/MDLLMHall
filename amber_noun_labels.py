"""Per-noun AMBER-g labels for one response, following third_party/AMBER/inference.py.

The official scorer only prints totals. This applies the same per-noun rules
(NLTK noun extraction + lemmatisation, the relation.json object vocabulary,
global safe words, truth / hallu association lists, spaCy en_core_web_lg
similarity > 0.8) and returns a label for every object noun with its
character span, so it can be joined to token-level traces:

  grounded      in the truth list or its associations, or similar to one
  hallucinated  an object word that is neither (counts toward CHAIR)
  ignored       an object word in the global safe-word list

`annotated_absent` additionally marks nouns matching the image's 'hallu' list
(the objects the Cog metric looks for).
"""

import json
import os

import nltk
from nltk.stem import WordNetLemmatizer

ROOT = os.path.dirname(os.path.abspath(__file__))
AMBER_DATA = os.path.join(ROOT, "third_party", "AMBER", "data")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class AmberLabeler:
    def __init__(self, similarity_score=0.8, use_spacy=True):
        self.association = read_json(os.path.join(AMBER_DATA, "relation.json"))
        self.annotations = read_json(os.path.join(AMBER_DATA, "annotations.json"))
        self.object_words = set(self.association)
        for values in self.association.values():
            self.object_words.update(values)
        with open(os.path.join(AMBER_DATA, "safe_words.txt"), "r", encoding="utf-8") as handle:
            self.global_safe = {line.split("\n")[0] for line in handle}
        self.lemmatizer = WordNetLemmatizer()
        self.similarity_score = similarity_score
        self.nlp = None
        if use_spacy:
            import spacy

            self.nlp = spacy.load("en_core_web_lg")
        self._docs = {}
        self._similar = {}

    def similar(self, word1, word2):
        if self.nlp is None:
            return False
        key = (word1, word2)
        if key not in self._similar:
            for word in key:
                if word not in self._docs:
                    self._docs[word] = self.nlp(word)
            self._similar[key] = self._docs[word1].similarity(self._docs[word2]) > self.similarity_score
        return self._similar[key]

    def nouns(self, text):
        """Every NN* token with its lemma and character span in `text` (None if not locatable)."""
        found = []
        cursor = 0
        for word, tag in nltk.pos_tag(nltk.word_tokenize(text)):
            start = text.find(word, cursor)
            span = None
            if start != -1:
                span = (start, start + len(word))
                cursor = span[1]
            if tag.startswith("NN"):
                found.append(dict(word=word, lemma=self.lemmatizer.lemmatize(word), span=span))
        return found

    def word_lists(self, item_id):
        truth = self.annotations[item_id - 1]
        safe_words = [w for t in truth["truth"] for w in self.association[t]] + truth["truth"]
        hallu_words = [w for t in truth["hallu"] for w in self.association[t]] + truth["hallu"]
        return truth, safe_words, hallu_words

    def label(self, item_id, text):
        truth, safe_words, hallu_words = self.word_lists(item_id)
        if truth["type"] != "generative":
            raise ValueError(f"AMBER id {item_id} is not a generative item.")
        rows = []
        for noun in self.nouns(text):
            lemma = noun["lemma"]
            if lemma not in self.object_words:
                continue
            if lemma in self.global_safe:
                label = "ignored"
            elif lemma in safe_words or any(self.similar(lemma, w) for w in safe_words):
                label = "grounded"
            else:
                label = "hallucinated"
            absent = lemma in hallu_words or (
                label != "ignored" and any(self.similar(lemma, w) for w in hallu_words)
            )
            rows.append(dict(noun, label=label, annotated_absent=absent))
        return rows

    def has_annotated_absent_object(self, item_id, text):
        """Selection rule used to pick the study captions (exact matches only, no spaCy)."""
        truth, safe_words, _ = self.word_lists(item_id)
        truth_set = set(truth["truth"]) | {w for t in truth["truth"] for w in self.association.get(t, [])}
        absent = set()
        for word in truth["hallu"]:
            absent.add(word)
            absent.update(self.association.get(word, []))
        for noun in self.nouns(text):
            lemma = noun["lemma"]
            if lemma not in self.global_safe and lemma not in truth_set and lemma in absent:
                return True
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select AMBER-g captions that name an annotated-absent object.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labeler = AmberLabeler(use_spacy=False)
    rows = read_json(args.predictions)
    ids = [int(r["id"]) for r in rows if labeler.has_annotated_absent_object(int(r["id"]), r["response"])]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(dict(source=os.path.abspath(args.predictions), n_scanned=len(rows), ids=ids), handle)
    print(f"{len(ids)} / {len(rows)} captions selected -> {args.output}")
