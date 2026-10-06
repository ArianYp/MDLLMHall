"""Install-time check for the official AMBER generative scorer.

AMBER inference.py imports nltk, spacy, and tqdm, loads en_core_web_lg, and
calls NLTK tokenization, POS tagging, and WordNet lemmatization. This script
only downloads missing resources. It does not change metric code.
"""

import importlib.util


NLTK_RESOURCES = (
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
)


def require_module(name, install_hint):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"Missing Python package '{name}'. Install it with: {install_hint}")


def missing_nltk_resource(exc):
    message = str(exc)
    for line in message.splitlines():
        if "Resource" in line and "not found" in line:
            parts = line.replace("Resource", "").strip().split()
            if parts:
                return parts[0].strip("'\"")
    raise SystemExit(
        "NLTK reported a missing resource, but the resource name could not be read. "
        f"Original error:\n{message}"
    ) from exc


def ensure_nltk():
    import nltk

    for package, probe in NLTK_RESOURCES:
        try:
            nltk.data.find(probe)
            print(f"nltk resource present: {package}")
        except LookupError:
            print(f"downloading nltk resource: {package}")
            if not nltk.download(package, quiet=False):
                raise SystemExit(f"nltk.download({package!r}) failed.")

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    for _attempt in range(6):
        try:
            tokens = nltk.word_tokenize("a dog sits")
            nltk.pos_tag(tokens)
            lemmatizer.lemmatize("dogs")
            return
        except LookupError as exc:
            missing = missing_nltk_resource(exc)
            print(f"downloading additional nltk resource required by this install: {missing}")
            if not nltk.download(missing, quiet=False):
                raise SystemExit(f"nltk.download({missing!r}) failed.") from exc
    raise SystemExit("NLTK still cannot tokenize, tag, or lemmatize after downloading missing resources.")


def ensure_spacy():
    import spacy

    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError as exc:
        raise SystemExit(
            "spaCy model en_core_web_lg is not installed. Run:\n"
            "  python -m spacy download en_core_web_lg"
        ) from exc
    left = nlp("dog")
    right = nlp("puppy")
    _ = left.similarity(right)
    print("spacy model present: en_core_web_lg")


def main():
    require_module("nltk", "pip install -U nltk")
    require_module("spacy", "pip install -U spacy")
    require_module("tqdm", "pip install -U tqdm")
    ensure_nltk()
    ensure_spacy()
    print("AMBER scorer dependencies are ready.")


if __name__ == "__main__":
    main()
