#!/usr/bin/env python3
"""Legge i gameLog.txt del server Tablut e scrive un file .states (una riga per posizione: 'ply turno scacchiera81', riga 1 in alto) piu' un riepilogo
(catture, ultima cattura, posizione finale). Il file .states e' l'ingresso di StallDiag / StallCalib / StallMini (package experiments).
Uso: python3 tools/parse_game_logs.py LOG.txt [LOG2.txt ...]   ->  LOG.txt.states"""
import re, sys

def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read().split('\n')
    states, cur = [], None
    for i, l in enumerate(txt):
        m = re.match(r'BUONO: Turn: (\w) Pawn from (\w\d) to (\w\d)', l)
        if m:
            cur = m.groups()
        if l.startswith('BUONO: Stato:') and i + 11 < len(txt):
            states.append((cur, txt[i + 1:i + 10], txt[i + 11].strip()))
    return states

def counts(b):
    s = ''.join(b)
    return s.count('W'), s.count('B'), s.count('K')

for p in sys.argv[1:]:
    st = parse(p)
    with open(p + '.states', 'w') as f:
        f.write('\n'.join(f"{k} {s[2]} {''.join(s[1])}" for k, s in enumerate(st)))
    prev, last = None, 0
    for k, s in enumerate(st):
        c = counts(s[1])[:2]
        if prev is not None and c != prev:
            print(f"  ply {k}: cattura -> bianchi={c[0]} neri={c[1]} (mossa {s[0]})")
            last = k
        prev = c
    print(f"{p}: {len(st)} posizioni, ultima cattura al ply {last}, pedine finali W,B,K = {counts(st[-1][1])}, turno {st[-1][2]}  -> {p}.states")
