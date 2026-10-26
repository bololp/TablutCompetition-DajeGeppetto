import numpy as np, csv
z = np.load('cand.npz'); C = z['cand']; F = z['feats']; role_all = z['role'].astype(int); game = z['game']; names = list(z['names'])
rows = list(csv.DictReader(open('lab.csv')))
idx = np.array([int(r['idx']) for r in rows]); rl = np.array([int(r['role']) for r in rows])
st = np.array([float(r['static']) for r in rows]); se = np.array([float(r['search']) for r in rows])
assert (role_all[idx] == rl).all(), 'ordine dataset diverso'
print('campioni', len(idx), '| ruolo coincide col dataset Python: OK')
sg = np.where(rl == 0, 1.0, -1.0)           # il punteggio della ricerca e' dal punto di vista di chi muove (verificato sotto)
s_m = st*sg/1000.0                          # statica, vista da chi muove
t_m = np.clip(se/1000.0, -1, 1)             # ricerca prof. 3, vista da chi muove (le vittorie forzate = +-1)
for r in (0, 1):
    m = rl == r
    print(f'ruolo {r}: corr(statica, ricerca) = {np.corrcoef(s_m[m], t_m[m])[0,1]:.3f}  (se fosse negativa la prospettiva sarebbe sbagliata)  n={m.sum()}')
res = t_m - s_m
print(f'residuo (ricerca - statica): media {res.mean():+.3f}  sd {res.std():.3f}  |r|>0.5: {(abs(res)>0.5).mean():.1%}   risultato forzato in 3 piani (|ricerca|=1): {(abs(t_m)>=0.999).mean():.1%}')
col = lambda n: C[idx, names.index(n)]
d = col('d'); lanes = col('lanes'); f2 = col('f2'); sf = col('safe_forks')
kth = F[idx, 43]; wth = F[idx, 41]*8; bth = F[idx, 42]*8
print('\n== dove sbaglia la valutazione statica (rispetto alla sua stessa ricerca a prof. 3) ==')
def rep(name, m):
    if m.sum() == 0: print(f'{name:48s} n=0'); return
    print(f'{name:48s} n={int(m.sum()):5d} ({m.mean():5.1%})  media|r|={abs(res[m]).mean():.3f}  quota |r|>0.5 = {(abs(res[m])>0.5).mean():5.1%}  risultato forzato = {(abs(t_m[m])>=0.999).mean():5.1%}')
rep('tutte', np.ones(len(idx), bool))
rep('Re: via di fuga immediata (d=1)', d == 1)
rep('Re: fuga in 2 mosse (d=2)', d == 2)
rep('Re: fuga in 3 mosse (d=3)', d == 3)
rep('Re: nessuna fuga entro 3 mosse (d>3)', d == 7)
rep('minaccia di cattura del Re (F_KING_THREAT)', kth > 0)
rep('pedine catturabili (bianche o nere) e Re non minacciato', ((wth > 0) | (bth > 0)) & (kth == 0))
rep('nessun flag (d>3, nessuna minaccia)', (d == 7) & (kth == 0) & (wth == 0) & (bth == 0))
rep('forchetta sicura (safe_forks>0)', sf > 0)
print('\n== segno sbagliato sui risultati forzati (ricerca vede vittoria/sconfitta in <=3 piani) ==')
fz = abs(t_m) >= 0.999
wrong = fz & (np.sign(s_m) != np.sign(t_m))
print(f'risultati forzati: {fz.sum()}; la statica ha il segno OPPOSTO in {wrong.sum()} ({wrong.sum()/max(1,fz.sum()):.1%}); in {(fz & (abs(s_m)<0.2)).sum()} e\' quasi neutra (|v|<0.2)')
for nm, m in (('d=1', d==1), ('d=2', d==2), ('d=3', d==3), ('d>3', d==7), ('Re minacciato', kth>0)):
    mm = fz & m
    print(f'   tra i forzati con {nm:14s}: {int(mm.sum()):4d}  di cui statica sbagliata/neutra: {int((mm & ((np.sign(s_m)!=np.sign(t_m)) | (abs(s_m)<0.2))).sum())}')
