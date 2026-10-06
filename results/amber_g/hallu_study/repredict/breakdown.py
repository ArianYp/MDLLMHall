import csv, numpy as np, sys
sys.path.insert(0, '.')
from analyze_hallu_study import auroc
STUFF = {"sky","cloud","wall","floor","ground","grass","tree","bush","plant","road","street","path","sidewalk","water","sea","ocean","river","lake","beach","sand","snow","mountain","hill","field","forest","dirt","court","building","fence","rock","leaf","sun"}
rows = list(csv.DictReader(open('results/amber_g/hallu_study/repredict/repredict.csv')))
clip = {(r['id'], r['pos']): r for r in csv.DictReader(open('results/amber_g/hallu_study/rag/clip_relation/nouns_clip.csv'))}
y = np.array([r['label'] == 'hallucinated' for r in rows]); stuff = np.array([r['lemma'] in STUFF for r in rows])
p = np.array([float(r['plain_nb_p']) for r in rows]); rk = np.array([float(clip[(r['id'], r['pos'])]['rank_full']) for r in rows])
first = np.array([float(r['repeat_mention']) == 0 for r in rows])
rk_ = lambda v: np.argsort(np.argsort(v))
spearmanr = lambda a, b: (np.corrcoef(rk_(a), rk_(b))[0, 1],)
print('spearman(p_plain_nb, rank_full)', spearmanr(p, rk)[0])
for name, m in [('objects', ~stuff), ('background', stuff), ('first mentions', first)]:
    print(f'{name:15s} n={m.sum()} hallu={y[m].sum()}  AUROC 1-p_nb {auroc(1-p[m], y[m]):.3f}  CLIP {auroc(1-rk[m], y[m]):.3f}  mean p_nb grounded {p[m & ~y].mean():.2f} halluc {p[m & y].mean():.2f}')
print('share of grounded with p_nb<0.5 that are background:', (stuff & ~y & (p < .5)).sum() / (~y & (p < .5)).sum(), ' (background share of grounded overall', (stuff & ~y).sum() / (~y).sum(), ')')
