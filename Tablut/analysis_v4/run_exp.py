import numpy as np, time, sys
from nn import train
z = np.load('cand.npz'); C = z['cand']; F = z['feats']; role = z['role'].astype(int); y = z['outc'].astype(float); game = z['game']; names = list(z['names'])
col = lambda n: C[:, names.index(n)]
base_cols = [i for i in range(44) if i not in (38, 39, 40)]   # le 41 feature del CNN/MLP v3 di default
Xb = F[:, base_cols].astype(np.float64)
d = col('d')
new = {
  'esc_prox': np.where(d <= 3, 1.0/np.maximum(d, 1), 0.0),
  'd3cells': col('d3cells')/8.0, 'reach2': col('reach2')/16.0, 'edge3': col('edge3')/8.0,
  'safe_forks': col('safe_forks')/4.0,
  'gates_closed': (16 - col('gates_open'))/4.0, 'gate_run': col('gate_run')/100.0,
  'dup_king_ring': F[:, 2].astype(np.float64),             # controllo: duplicato esatto (zero informazione)
  'noise': np.random.RandomState(5).rand(len(F)),          # controllo: rumore puro
}
test = (game % 12 == 0); tr = ~test
print('posizioni', len(y), 'train', int(tr.sum()), 'test', int(test.sum()), '| predittore costante MSE test:', round(float(np.mean((y[tr].mean()-y[test])**2)), 4))
configs = [
  ('base (41 feature)', []),
  ('+ ESCAPE [esc_prox,d3cells,reach2,edge3]', ['esc_prox', 'd3cells', 'reach2', 'edge3']),
  ('+ SAFE_FORK [safe_forks]', ['safe_forks']),
  ('+ GATES [gates_closed,gate_run]', ['gates_closed', 'gate_run']),
  ('+ ESCAPE+SAFE_FORK', ['esc_prox', 'd3cells', 'reach2', 'edge3', 'safe_forks']),
  ('+ tutte le candidate', ['esc_prox', 'd3cells', 'reach2', 'edge3', 'safe_forks', 'gates_closed', 'gate_run']),
  ('controllo: + duplicato di F_KING_RING', ['dup_king_ring']),
  ('controllo: + rumore casuale', ['noise']),
]
seeds = [1, 2, 3]
subs = {
  'tutte': np.ones(len(y), bool),
  'Re: nessuna via/forchetta ma fuga entro 3 mosse': (col('lanes') == 0) & (col('f2') == 0) & (d <= 3),
  'Re: fuga NON entro 3 mosse': d == 7,
  'Re: via immediata o forchetta': (col('lanes') > 0) | (col('f2') > 0),
}
res = {}
t0 = time.time()
for name, add in configs:
    X = np.hstack([Xb] + [new[a][:, None] for a in add]) if add else Xb
    ms = []; sub_ms = {k: [] for k in subs}
    for s in seeds:
        net, mse, pred = train(X[tr], role[tr], y[tr], X[test], role[test], y[test], s)
        ms.append(mse)
        for k, msk in subs.items():
            mm = msk[test]
            sub_ms[k].append(float(np.mean((pred[mm] - y[test][mm])**2)))
    res[name] = (ms, sub_ms)
    print(f'{name:45s} MSE held-out {np.mean(ms):.4f}  (seed: {", ".join(f"{m:.4f}" for m in ms)})   [{time.time()-t0:.0f}s]', flush=True)
print('\nMSE per sottoinsiemi di posizioni (media 3 seed; n test):')
for k, msk in subs.items(): print('  ', k, '| n test =', int(msk[test].sum()))
b = res['base (41 feature)'][1]
for name, (ms, sm) in res.items():
    print(f'{name:45s}', ' | '.join(f'{np.mean(sm[k]):.4f} ({np.mean(sm[k])-np.mean(b[k]):+.4f})' for k in subs))
