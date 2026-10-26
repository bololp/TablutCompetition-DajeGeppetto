import numpy as np
exec(open('resid_cv.py').read().split("print('target")[0])
fs = lambda cols: np.column_stack([F[idx][:, cols]]) if cols else np.zeros((len(idx), 0))
def run(name, X):
    r = cv_r2(X); print(f'{name:60s} R2 = {r.mean():+.4f}', flush=True); return r
run('solo statica (calibrazione)', s_m[:, None])
run('statica + ruolo', np.column_stack([s_m, rl]))
run('statica + ruolo + solo materiale (2 feature)', np.column_stack([s_m, rl, F[idx][:, [0, 1]]]))
run('statica + ruolo + minacce di cattura (3 feature)', np.column_stack([s_m, rl, F[idx][:, [41, 42, 43]]]))
run('statica + ruolo + fuga/lanes (4 feature)', np.column_stack([s_m, rl, F[idx][:, [34, 35, 36, 37]]]))
run('statica + ruolo + minacce + fuga (7 feature)', np.column_stack([s_m, rl, F[idx][:, [34, 35, 36, 37, 41, 42, 43]]]))
run('statica + ruolo + tutte le 41 feature', Xb)
run('SENZA statica: ruolo + 41 feature', np.column_stack([rl, F[idx][:, base_cols]]))
