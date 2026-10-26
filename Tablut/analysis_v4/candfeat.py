import numpy as np, struct, sys, time

NF = 44
def load(path, game_off=0):
    with open(path, 'rb') as f:
        data = f.read()
    n = struct.unpack('>i', data[:4])[0]
    p = 4
    cells = np.frombuffer(data, dtype=np.uint8, count=n*81, offset=p).reshape(n, 81); p += n*81
    role = np.frombuffer(data, dtype=np.uint8, count=n, offset=p); p += n
    outc = np.frombuffer(data, dtype=np.int8, count=n, offset=p); p += n
    game = np.frombuffer(data, dtype='>i4', count=n, offset=p).astype(np.int64); p += 4*n
    feats = np.frombuffer(data, dtype='>f4', count=n*NF, offset=p).reshape(n, NF).astype(np.float32)
    return cells, role, outc, game + game_off, feats

# ---- geometria (stessa dell'estrattore Java: scacchiera 11x11 con bordo sentinella)
W = 11
EMPTY, WHITE, BLACK, THRONE, KING, OUT, CEMPTY = 0, 1, 2, 3, 4, 5, 8
def idx(r, c): return (r+1)*W + (c+1)
THRONE_P = idx(4, 4)
DIRS = (-W, W, -1, 1)
CITS = [(3,0),(4,0),(5,0),(4,1),(0,3),(0,4),(0,5),(1,4),(3,8),(4,8),(5,8),(4,7),(8,3),(8,4),(8,5),(7,4)]
CIT = [False]*(W*W); CIT_CELLS = []
for r, c in CITS: CIT[idx(r,c)] = True; CIT_CELLS.append(idx(r,c))
EDGE = [False]*(W*W); MAN = [0]*(W*W)
for r in range(9):
    for c in range(9):
        p = idx(r, c)
        EDGE[p] = r in (0, 8) or c in (0, 8)
        MAN[p] = abs(r-4) + abs(c-4)
# 16 "gate": celle di fuga raggiungibili lungo le righe/colonne 1,2,6,7 (i 4 angoli non si raggiungono dall'interno)
GATES = []
for L in (1, 2, 6, 7):
    GATES += [(idx(L, 0), 1), (idx(L, 8), -1), (idx(0, L), W), (idx(8, L), -W)]  # (cella di gate, passo verso l'interno)

def board(cells81):
    b = [OUT]*(W*W)
    for r in range(9):
        for c in range(9):
            v = int(cells81[r*9+c])
            b[idx(r, c)] = KING if v == 3 else v
    if b[THRONE_P] == EMPTY: b[THRONE_P] = THRONE
    for p in CIT_CELLS:
        if b[p] == EMPTY: b[p] = CEMPTY
    return b

def is_free(v): return (v & 7) == 0

def scan_black(b):
    reach = [0]*(W*W)
    for p in range(W*W):
        if b[p] != BLACK: continue
        if CIT[p]:
            for d in DIRS:
                q = p; step = 0
                while True:
                    q += d; step += 1
                    v = b[q]
                    if v != EMPTY and (v != CEMPTY or step > 5): break
                    reach[q] += 1
        else:
            for d in DIRS:
                q = p + d
                while b[q] == EMPTY:
                    reach[q] += 1; q += d
    return reach

def lanes_from(b, c):
    n = 0
    for d in DIRS:
        q = c
        while True:
            q += d
            if b[q] != EMPTY: break
            if EDGE[q]: n += 1; break
    return n

def king_capturable(b, reachB, kp):
    hostile = lambda p: b[p] == BLACK or CIT[p]
    if kp == THRONE_P or MAN[kp] == 1:
        missing = -1
        for d in DIRS:
            s = kp + d
            if s == THRONE_P: continue
            if b[s] != BLACK:
                if missing >= 0: return False
                missing = s
        return missing >= 0 and is_free(b[missing]) and reachB[missing] > 0
    for axis in (0, 1):
        a = kp + (-W if axis == 0 else -1); c = kp + (W if axis == 0 else 1)
        if is_free(b[a]) and reachB[a] > 0 and hostile(c): return True
        if is_free(b[c]) and reachB[c] > 0 and hostile(a): return True
    return False

def king_pos(b):
    for p in range(W*W):
        if b[p] == KING: return p
    return -1

def king_bfs(b, kp, maxd=3):
    """strati di mosse del Re (scacchiera statica, il Re ha lasciato la sua cella). Ritorna (layers, edges_by_depth)."""
    saved = b[kp]
    b[kp] = THRONE if kp == THRONE_P else EMPTY
    visited = {kp}
    frontier = [kp]
    layers = []          # celle NON di bordo raggiunte a profondita' k (k=1..)
    edges = []           # celle di fuga raggiunte a profondita' k
    for k in range(maxd):
        new = []; e = []
        for c in frontier:
            for d in DIRS:
                q = c
                while True:
                    q += d
                    if b[q] != EMPTY: break
                    if q not in visited:
                        visited.add(q)
                        if EDGE[q]: e.append(q)
                        else: new.append(q)
                    if EDGE[q]: break
        layers.append(new); edges.append(e)
        frontier = new
    b[kp] = saved
    return layers, edges

def cand(cells81):
    b = board(cells81)
    kp = king_pos(b)
    out = {}
    if kp < 0: return None
    layers, edges = king_bfs(b, kp, 3)
    # validazione contro le feature esistenti
    lanes = len(edges[0])
    saved = b[kp]; b[kp] = THRONE if kp == THRONE_P else EMPTY
    c2 = f2 = 0
    for c in layers[0]:
        n = lanes_from(b, c)
        if n >= 1: c2 += 1
        if n >= 2: f2 += 1
    # celle a 2 mosse (strato 2) con almeno una via: fuga in 3 mosse
    d3 = 0
    for c in layers[1]:
        if lanes_from(b, c) >= 1: d3 += 1
    b[kp] = saved
    out['lanes'] = lanes; out['c2'] = c2; out['f2'] = f2
    d = 7
    for k in range(3):
        if edges[k]: d = k+1; break
    out['d'] = d
    out['d3cells'] = d3
    out['reach2'] = len(layers[0]) + len(layers[1])
    out['edge3'] = len(edges[0]) + len(edges[1]) + len(edges[2])
    # forchette sicure (il Re arriva in c con >=2 vie e li' non e' catturabile al volo)
    safe = 0
    if f2 > 0:
        b[kp] = THRONE if kp == THRONE_P else EMPTY
        for c in layers[0]:
            if lanes_from(b, c) >= 2:
                b[c] = KING
                reach = scan_black(b)
                if not king_capturable(b, reach, c): safe += 1
                b[c] = EMPTY
        b[kp] = saved
    out['safe_forks'] = safe
    # gate aperti: cella di gate vuota con corsa libera verso l'interno
    gopen = 0; grun = 0
    for gp, st in GATES:
        run = 0; q = gp
        while b[q] == EMPTY:
            run += 1; q += st
        if run >= 1: gopen += 1
        grun += run
    out['gates_open'] = gopen; out['gate_run'] = grun
    return out

if __name__ == '__main__':
    names = ['lanes','c2','f2','d','d3cells','reach2','edge3','safe_forks','gates_open','gate_run']
    allc = []; allr = []; allo = []; allg = []; allf = []
    off = 0
    for path in sys.argv[1:-1]:
        cells, role, outc, game, feats = load(path, off)
        off = int(game.max()) + 1
        t0 = time.time()
        C = np.zeros((len(role), len(names)), dtype=np.float32)
        for i in range(len(role)):
            o = cand(cells[i])
            if o is None: continue
            C[i] = [o[k] for k in names]
        print(path, 'candidate feature time', round(time.time()-t0, 1), 's', flush=True)
        allc.append(C); allr.append(role); allo.append(outc); allg.append(game); allf.append(feats)
    np.savez_compressed(sys.argv[-1], cand=np.concatenate(allc), role=np.concatenate(allr), outc=np.concatenate(allo),
                        game=np.concatenate(allg), feats=np.concatenate(allf), names=np.array(names))
