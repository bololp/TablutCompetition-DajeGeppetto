import numpy as np, csv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
z = np.load('cand.npz'); C = z['cand']; F = z['feats']; game = z['game']; names = list(z['names'])
rows = list(csv.DictReader(open('lab.csv')))
idx = np.array([int(r['idx']) for r in rows]); rl = np.array([int(r['role']) for r in rows])
st = np.array([float(r['static']) for r in rows]); se = np.array([float(r['search']) for r in rows])
sg = np.where(rl == 0, 1.0, -1.0); s_m = st*sg/1000.0; t_m = np.clip(se/1000.0, -1, 1); res = t_m - s_m
col = lambda n: C[idx, names.index(n)]
d = col('d')
base_cols = [i for i in range(44) if i not in (38, 39, 40)]
Xb = np.column_stack([s_m, rl, F[idx][:, base_cols]])
cands = {
  'esc_prox': np.where(d <= 3, 1.0/np.maximum(d, 1), 0.0), 'd3cells': col('d3cells'), 'reach2': col('reach2'), 'edge3': col('edge3'),
  'safe_forks': col('safe_forks'), 'gates_closed': 16 - col('gates_open'), 'gate_run': col('gate_run'),
  'dup_king_ring': F[idx, 2], 'noise': np.random.RandomState(3).rand(len(idx)),
}
groups = game[idx]
def cv_r2(X, reps=3):
    out = []
    for rep in range(reps):
        # permuta i gruppi per cambiare le pieghe
        ug = np.unique(groups); rs = np.random.RandomState(rep); mp = dict(zip(ug, rs.permutation(len(ug))))
        g2 = np.array([mp[g] for g in groups])
        pred = np.zeros(len(res))
        for tr, te in GroupKFold(5).split(X, res, g2):
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=20, l2_regularization=1.0, random_state=rep)
            m.fit(X[tr], res[tr]); pred[te] = m.predict(X[te])
        out.append(1 - np.sum((res - pred)**2)/np.sum((res - res.mean())**2))
    return np.array(out)
print('target: residuo (ricerca prof.3 - statica CNN), campioni', len(res), '| R2 = quota di residuo spiegata fuori campione (CV per partita, 3 ripetizioni)')
cfgs = [
 ('base (statica + ruolo + 41 feature)', []),
 ('+ ESCAPE [esc_prox,d3cells,reach2,edge3]', ['esc_prox', 'd3cells', 'reach2', 'edge3']),
 ('+ SAFE_FORK', ['safe_forks']),
 ('+ GATES', ['gates_closed', 'gate_run']),
 ('+ ESCAPE+SAFE_FORK', ['esc_prox', 'd3cells', 'reach2', 'edge3', 'safe_forks']),
 ('controllo: + duplicato F_KING_RING', ['dup_king_ring']),
 ('controllo: + rumore', ['noise']),
]
base = None
for name, add in cfgs:
    X = np.column_stack([Xb] + [cands[a] for a in add]) if add else Xb
    r = cv_r2(X)
    if base is None: base = r
    print(f'{name:44s} R2 = {r.mean():+.4f}  (rip.: {", ".join(f"{v:+.4f}" for v in r)})   delta vs base = {r.mean()-base.mean():+.4f}')
# solo dove il Re e' vicino alla fuga
sub = (d <= 3)
print('\nsolo posizioni con fuga entro 3 mosse (n =', int(sub.sum()), '):')
def cv_sub(X, reps=3):
    out = []
    for rep in range(reps):
        ug = np.unique(groups[sub]); rs = np.random.RandomState(rep); mp = dict(zip(ug, rs.permutation(len(ug))))
        g2 = np.array([mp[g] for g in groups[sub]]); Xs = X[sub]; ys = res[sub]; pred = np.zeros(len(ys))
        for tr, te in GroupKFold(5).split(Xs, ys, g2):
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=20, l2_regularization=1.0, random_state=rep)
            m.fit(Xs[tr], ys[tr]); pred[te] = m.predict(Xs[te])
        out.append(1 - np.sum((ys - pred)**2)/np.sum((ys - ys.mean())**2))
    return np.array(out)
base = None
for name, add in cfgs:
    X = np.column_stack([Xb] + [cands[a] for a in add]) if add else Xb
    r = cv_sub(X)
    if base is None: base = r
    print(f'{name:44s} R2 = {r.mean():+.4f}   delta vs base = {r.mean()-base.mean():+.4f}')
