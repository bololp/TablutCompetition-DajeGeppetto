#!/usr/bin/env python3
"""
Mutation test dell'MCTS (aiplayer/Mcts.java, EvalCache.java, MctsConfig.java): inietta UN bug alla volta in una COPIA del sorgente, ricompila quel
solo file in una cartella separata messa PRIMA del build nel classpath, lancia i test e si aspetta che FALLISCANO. Un mutante che i test non
rilevano e' un buco dei test (o un mutante equivalente: si verifica a mano e si annota qui sotto).

Uso (dalla cartella Tablut, con il progetto gia' compilato in $BUILD):
  python3 tools/mutation_mcts.py [indiceIniziale indiceFinale | lista]    (lista: indici e intervalli, es. 5,20,26-68)
Per ogni mutante si lancia prima MctsSearchCheck e, se sopravvive, MctsConcurrencyCheck, entrambi con -Dmcts.failfast=true (il primo fallimento
termina il test: i mutanti rilevati costano pochi secondi, quelli sopravvissuti un'esecuzione completa).
  Variabili: BUILD (cartella delle classi, default build), JAVAC (comando del compilatore, default javac).

MUTANTI EQUIVALENTI NOTI (non si mutano, verificato a ragionamento):
  - il trono che blocca il Re: oltre il trono, lungo la riga 4 o la colonna 4, si incontra sempre una citadel ((4,1),(4,7),(1,4),(7,4)) prima di ogni cella
    di bordo (e i bordi di quelle linee sono a loro volta citadel), quindi per il predicato "il Re puo' fuggire" che il trono blocchi o no e' indifferente;
  - il limite di 5 passi per un Re che parte da una citadel (5 -> 3..7): lungo una riga o colonna libera una citadel ne vede un'altra solo a 1, 2 o 8 passi
    (calcolato sulle coordinate dei campi, senza attraversare il trono), quindi ogni soglia da 3 a 7 e' equivalente; contano 5 -> 8 e "sempre bloccata";
  - azzerare la storia di ripetizione alla cattura (w.histBase = w.histLen) e' un'ottimizzazione di costo: dopo una cattura nessuna posizione
    precedente puo' ripresentarsi, quindi la memoria prima della cattura non puo' mai coincidere con una posizione successiva;
  - ripulire la sentinella EXPANDING dopo un'eccezione: dopo un errore l'albero viene comunque scartato;
  - non fermarsi a radice risolta quando la radice e' gia' risolta all'espansione: la ricerca non parte nemmeno.
"""
import os, shutil, subprocess, sys

ROOT = "src/it/unibo/ai/didattica/competition/tablut/"
BUILD = os.environ.get("BUILD", "build")
CP = BUILD + ":lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
T = "it.unibo.ai.didattica.competition.tablut."
CHECKS = [(T + "aiplayer.MctsSearchCheck", 240), (T + "aiplayer.MctsConcurrencyCheck", 180)]  # ~3x la durata del baseline (48 s e 32 s): un loop e' un rilevamento
JAVAC = os.environ.get("JAVAC", "javac").split()
M = "aiplayer/Mcts.java"
C = "aiplayer/EvalCache.java"
G = "aiplayer/MctsConfig.java"

MUT = [  # (descrizione, file relativo a ROOT, vecchio, nuovo[, indice dell'occorrenza])
 # ---- propagazione dei valori
 ("backprop: nessun cambio di segno", M, "\t\t\tv = -v;\n\t\t\tboolean changed = false;", "\t\t\tboolean changed = false;"),
 ("backprop: media aggiornata dividendo per n+1", M, "n.q[e] += (float) ((v - n.q[e]) / nv);", "n.q[e] += (float) ((v - n.q[e]) / (nv + 1));"),
 ("backprop: perdita virtuale non rimossa", M, "\t\t\t\tn.vloss[e]--;\n\t\t\t\tn.totalVloss--;\n\t\t\t\tfinal int nv = ++n.visits[e];", "\t\t\t\tfinal int nv = ++n.visits[e];"),
 ("backprop: visite totali del nodo non incrementate", M, "n.totalVisits++;\n\t\t\t\tif (n.lo[e] != n.hi[e]) {", "if (n.lo[e] != n.hi[e]) {"),
 ("backprop: la sintesi non sale ai livelli superiori", M, "childNode = changed ? n : null;", "childNode = null;"),
 ("backprop: la media si aggiorna anche per gli archi esatti", M, "\t\t\t\tif (n.lo[e] != n.hi[e]) {\n\t\t\t\t\tn.q[e] += (float)", "\t\t\t\tif (true) {\n\t\t\t\t\tn.q[e] += (float)"),
 # ---- limiti dimostrati e minimax implicito
 ("absorb: limite inferiore preso dal limite sbagliato del figlio", M, "n.lo[e] = (byte) -cHi;", "n.lo[e] = (byte) -cLo;"),
 ("absorb: limite superiore senza cambio di segno", M, "n.hi[e] = (byte) -cLo;", "n.hi[e] = (byte) cLo;"),
 ("absorb: minimax implicito senza cambio di segno", M, "n.im[e] = -cIm;", "n.im[e] = cIm;"),
 ("absorb: la media dell'arco risolto non diventa il valore esatto", M, "\t\tif (n.lo[e] == n.hi[e]) {\n\t\t\tn.q[e] = n.lo[e];\n\t\t}", ""),
 ("absorb: limite superiore del nodo mai ricalcolato", M, "if (n.hi[e] < oldHi && oldHi == n.nodeHi) {", "if (false) {"),
 ("absorb: minimax del nodo mai ricalcolato al ribasso", M, "} else if (n.im[e] < oldIm && oldIm == n.nodeIm) {", "} else if (false) {"),
 ("absorb: limite inferiore del nodo aggiornato al contrario", M, "if (n.lo[e] > n.nodeLo) {", "if (n.lo[e] < n.nodeLo) {"),
 ("absorb: minimax del nodo mai aggiornato al rialzo", M, "if (n.im[e] > n.nodeIm) {\n\t\t\tn.nodeIm = n.im[e];\n\t\t\tchanged = true;", "if (false) {\n\t\t\tn.nodeIm = n.im[e];\n\t\t\tchanged = true;"),
 # ---- selezione
 ("select: perdita virtuale ignorata", M, "if (vl > 0 && n.lo[i] != n.hi[i]) {", "if (false) {"),
 ("select: pesi del minimax implicito scambiati", M, "qv = (1 - a) * qv + a * n.im[i];", "qv = a * qv + (1 - a) * n.im[i];"),
 ("select: nessuna potatura degli archi dominati", M, "if (n.hi[i] < lo || n.child[i] == EXPANDING) {", "if (n.child[i] == EXPANDING) {"),
 ("select: gli archi in espansione non sono saltati", M, "if (n.hi[i] < lo || n.child[i] == EXPANDING) {", "if (n.hi[i] < lo) {"),
 ("select: nessun bonus di esplorazione", M, "final double score = qv + cp * n.prior[i] / (1 + nv + vw * vl);", "final double score = qv;"),
 ("select: bonus senza il denominatore delle visite", M, "cp * n.prior[i] / (1 + nv + vw * vl)", "cp * n.prior[i]"),
 ("select: la perdita virtuale non conta nel denominatore", M, "(1 + nv + vw * vl)", "(1 + nv)"),
 # ---- espansione
 ("expand: la vittoria terminale vale come sconfitta", M, "v = moverWins ? 1f : -1f;\n\t\t\t\t\tl = (byte) (moverWins ? 1 : -1);", "v = moverWins ? -1f : 1f;\n\t\t\t\t\tl = (byte) (moverWins ? -1 : 1);"),
 ("expand: ripetizione non rilevata", M, "} else if (!u.wasCapture() && historyContains(w, st.zobristHash)) {", "} else if (false) {"),
 ("expand: valutazione statica con la prospettiva invertita", M, "v = whiteToMove ? white : -white;", "v = whiteToMove ? -white : white;"),
 ("expand: la fuga del Re vale come vittoria del Nero", M, "v = -1f; // il Bianco e' di turno", "v = 1f; // il Bianco e' di turno"),
 ("expand: prior invertiti", M, "p[i] = Math.exp((n.q[i] - qmax) * invTau);", "p[i] = Math.exp((qmax - n.q[i]) * invTau);"),
 ("expand: il nodo senza mosse non e' una sconfitta dimostrata", M, "\t\t\tn.nodeLo = -1;\n\t\t\tn.nodeHi = -1;\n\t\t\tn.nodeIm = -1;\n\t\t\treturn n;", "\t\t\tn.nodeLo = -1;\n\t\t\tn.nodeHi = 1;\n\t\t\tn.nodeIm = -1;\n\t\t\treturn n;"),
 ("expand: rilevatore di fuga spento", M, "} else if (cfg.escapeScan && kingCanEscapeNow(st)) {", "} else if (false) {"),
 # ---- rilevatore di fuga
 ("fuga: dalla citadel il limite dei passi e' 8 invece di 5", M, "if (BoardConstants.IS_CITADEL[to] && (!originCitadel || step > 5)) {", "if (BoardConstants.IS_CITADEL[to] && (!originCitadel || step > 8)) {"),
 ("fuga: una citadel blocca sempre il Re, anche se parte da una citadel", M, "if (BoardConstants.IS_CITADEL[to] && (!originCitadel || step > 5)) {", "if (BoardConstants.IS_CITADEL[to]) {"),
 ("fuga: le citadel non bloccano il Re", M, "if (BoardConstants.IS_CITADEL[to] && (!originCitadel || step > 5)) {\n\t\t\t\t\tbreak;\n\t\t\t\t}", ""),
 ("fuga: segnalata anche con il Nero di turno", M, "if (st.turn != BoardConstants.WHITE_TURN || st.winner != FastState.WINNER_NONE) {", "if (st.winner != FastState.WINNER_NONE) {"),
 ("fuga: una direzione non esaminata", M, "for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {\n\t\t\tfinal int steps = BoardConstants.STEPS[d][ks];", "for (int d = 0; d < BoardConstants.NUM_DIRS - 1; d++) {\n\t\t\tfinal int steps = BoardConstants.STEPS[d][ks];"),
 # ---- simulazione
 ("simulate: profondita' massima off-by-one", M, "d + 1 >= cfg.maxDepth", "d >= cfg.maxDepth"),
 ("simulate: espansione una visita prima del dovuto", M, "node.visits[edge] < cfg.expandAfter", "node.visits[edge] < cfg.expandAfter - 1"),
 ("simulate: foglia statica/esatta con il segno sbagliato", M, "\t\t\tif (kind == K_EXACT || kind == K_STATIC) {\n\t\t\t\tleafV = -sv;", "\t\t\tif (kind == K_EXACT || kind == K_STATIC) {\n\t\t\t\tleafV = sv;"),
 ("simulate: valore della foglia espansa con il segno sbagliato", M, "leafV = fresh.nodeIm;", "leafV = -fresh.nodeIm;"),
 ("simulate: perdita virtuale non rilasciata quando la simulazione si annulla", M, "\t\t\t\trelease(w, d);\n\t\t\t\treturn false;", "\t\t\t\treturn false;"),
 ("simulate: tetto ai nodi con > invece di >=", M, "if (nodeCount.incrementAndGet() >= c.maxNodes) {", "if (nodeCount.incrementAndGet() > c.maxNodes) {"),
 ("simulate: array del percorso non fatti crescere", M, "\t\t\tw.ensurePath(d + 1);\n", ""),
 ("simulate: array della storia non fatti crescere", M, "\t\t\tw.ensureHist(w.histLen + 2);\n", ""),
 ("runWorker: non si ferma quando la radice e' risolta", M, "if (c.root.nodeLo == c.root.nodeHi) { // radice risolta: la risposta non puo' cambiare\n\t\t\t\t\tc.stop = true;\n\t\t\t\t\tbreak;\n\t\t\t\t}", ""),
 ("runWorker: le simulazioni annullate contano come fatte", M, "\t\t\t\t\tc.simsStarted.decrementAndGet(); // simulazione annullata: lo slot si restituisce\n", ""),
 ("runWorker: oltre il tetto di simulazioni si ferma tutti subito (perde una simulazione annullata)", M, "\t\t\t\t\tif (c.simsDone.get() >= c.maxSims || System.nanoTime() >= c.deadlineNanos) {\n\t\t\t\t\t\tc.stop = true;\n\t\t\t\t\t\tbreak;\n\t\t\t\t\t}\n\t\t\t\t\tThread.yield();\n\t\t\t\t\tcontinue;", "\t\t\t\t\tc.stop = true;\n\t\t\t\t\tbreak;"),
 ("runWorker: lo slot oltre il tetto non e' restituito", M, "\t\t\t\t\tc.simsStarted.decrementAndGet(); // slot non assegnato\n", ""),
 ("runWorker: oltre il tetto non si attende (si esce senza fermare gli altri)", M, "\t\t\t\t\tThread.yield();\n\t\t\t\t\tcontinue;", "\t\t\t\t\tbreak;"),
 ("runWorker: l'eccezione di un thread non e' registrata", M, "if (c.error == null) {\n\t\t\t\tc.error = t;\n\t\t\t}\n\t\t\tc.stop = true;", "c.stop = true;"),
 # ---- scelta della mossa e statistiche
 ("scelta: tra le vittorie dimostrate la piu' lunga", M, "if (best < 0 || d < bestDist || (d == bestDist && r.visits[i] > r.visits[best])) {", "if (best < 0 || d > bestDist || (d == bestDist && r.visits[i] > r.visits[best])) {"),
 ("scelta: le mosse dimostrate perse non sono escluse", M, "if (anyNotLost && r.hi[i] == -1) {", "if (false) {"),
 ("scelta: la meno visitata", M, "return r.visits[i] > r.visits[j];", "return r.visits[i] < r.visits[j];"),
 ("valore della radice: prospettiva invertita", M, "lastRootValue = whiteToMove ? qv : -qv;", "lastRootValue = whiteToMove ? -qv : qv;"),
 ("storia: la radice non viene aggiunta", M, "final long[] h = Arrays.copyOf(history, n + 1);\n\t\th[n] = rootState.zobristHash;\n\t\treturn h;", "return Arrays.copyOf(history, n);"),
 ("tempo: margine di sicurezza non sottratto", M, "Math.max(MIN_SEARCH_MILLIS, timeBudgetMillis - safetyMarginMillis)", "Math.max(MIN_SEARCH_MILLIS, timeBudgetMillis)"),
 ("tempo: scadenza mille volte piu' corta", M, "t0 + budget * 1_000_000L", "t0 + budget * 1_000L"),
 ("una sola mossa: si cerca comunque", M, "if (r.nMoves > 1 && !r.solved()) {", "if (!r.solved()) {"),
 ("errore: l'albero di una ricerca con errore viene conservato", M, "if (lastError != null) {\n\t\t\tclearTree(); // un thread", "if (false) {\n\t\t\tclearTree(); // un thread"),
 ("helper bloccato: non rilevato", M, "} catch (TimeoutException e) {\n\t\t\t\tstuck = true;", "} catch (TimeoutException e) {"),
 ("helper: attesa di 1 ms invece di 2000 (dichiarati bloccati)", M, "f.get(HELPER_JOIN_MILLIS, TimeUnit.MILLISECONDS);", "f.get(1, TimeUnit.MILLISECONDS);"),
 # ---- riuso
 ("riuso: la storia non e' confrontata (nodo discendente)", M, "if (pos.samePosition(target) && Arrays.equals(h2, targetHist)) {", "if (pos.samePosition(target)) {"),
 ("riuso: la storia non e' confrontata (stessa radice)", M, "if (pos.samePosition(newPos) && Arrays.equals(rootHist, newHist)) {", "if (pos.samePosition(newPos)) {"),
 ("riuso: profondita' massima 2 invece di 4", M, "static final int REUSE_MAX_DEPTH = 4;", "static final int REUSE_MAX_DEPTH = 2;"),
 ("riuso: la cattura non azzera la storia del cammino", M, "if (capture) {\n\t\t\t\th2 = new long[] { pos.zobristHash };", "if (false) {\n\t\t\t\th2 = new long[] { pos.zobristHash };"),
 ("riuso: i nodi ereditati non sono contati", M, "nodeCount.set(countNodes(r));", "nodeCount.set(1);"),
 ("memoria: il vecchio albero resta raggiungibile durante la ricerca", M, "\t\troot = null;\n\t\trootPos = null;\n\t\trootHist = null;\n\t\tfor (int i = 0; workers", "\t\trootPos = null;\n\t\trootHist = null;\n\t\tfor (int i = 0; workers"),
 ("memoria: il percorso dell'ultima simulazione tiene vivo il vecchio albero", M, "Arrays.fill(workers[i].pNode, null);", ";"),
 # ---- cache, configurazione, scala
 ("cache: la chiave non e' verificata", C, "if ((check ^ data) == hash && (data >>> 32) == 0x1L) {", "if ((data >>> 32) == 0x1L) {"),
 ("cache: voce vuota accettata", C, "if ((check ^ data) == hash && (data >>> 32) == 0x1L) {", "if ((check ^ data) == hash) {"),
 ("cache: la chiave si scrive senza lo xor", C, "slots.set(i, hash ^ data);", "slots.set(i, hash);"),
 ("config: tau=0 ammesso", G, "c.tau = num(val, 0.005, 100, key);", "c.tau = num(val, 0, 100, key);"),
 ("config: cache da 1 a 3 ammessa", G, "if (c.evalCacheLog2 != 0 && c.evalCacheLog2 < 4) {", "if (false) {"),
 ("config: mm oltre 1 ammesso", G, "c.mmWeight = num(val, 0, 1, key);", "c.mmWeight = num(val, 0, 10, key);"),
 ("scala: Simple con scala 10", M, "\t\t\treturn 100.0;", "\t\t\treturn 10.0;"),
 ("compressione: limite euristico 1.0", M, "static final double HEURISTIC_LIMIT = 0.98;", "static final double HEURISTIC_LIMIT = 1.0;"),
 ("compressione: NaN non gestito", M, "if (Double.isNaN(v)) {\n\t\t\treturn 0f;\n\t\t}", ""),
]

def main():
    if not os.path.isdir("src") or not os.path.isdir(BUILD):
        sys.exit("lancia dalla cartella Tablut (serve src/ e la cartella di build gia' compilata, $BUILD)")
    if len(sys.argv) == 2:  # lista di indici e intervalli: 5,20,26-68
        idxs = []
        for part in sys.argv[1].split(","):
            a, _, b = part.partition("-")
            idxs.extend(range(int(a), int(b or a) + 1))
    else:
        lo = int(sys.argv[1]) if len(sys.argv) > 2 else 0
        hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(MUT)
        idxs = list(range(lo, hi))
    for check, to in CHECKS:  # BASELINE: i test devono PASSARE sul codice non mutato
        b = subprocess.run(["java", "-Xss8m", "-Dmcts.failfast=true", "-cp", CP, check], capture_output=True, text=True, timeout=to)
        if b.returncode != 0:
            sys.exit("BASELINE FALLITA (%s): i risultati sui mutanti non avrebbero senso.\n%s" % (check, "\n".join(b.stdout.splitlines()[-6:])))
    print("baseline OK (i test passano sul codice non mutato)", flush=True)
    survivors, killed = [], 0
    for idx in idxs:
        desc, rel, old, new = MUT[idx][:4]
        nth = MUT[idx][4] if len(MUT[idx]) > 4 else 0
        base = os.path.basename(rel)
        work = "/tmp/mutm_%d" % idx
        shutil.rmtree(work, ignore_errors=True); os.makedirs(work + "/out")
        src = open(ROOT + rel, encoding="utf-8").read()
        if src.count(old) <= nth:
            print("[%2d] ERRORE: frammento da mutare non trovato: %s" % (idx, desc), flush=True); survivors.append(desc + " (frammento assente)"); continue
        pos = -1
        for _ in range(nth + 1):
            pos = src.index(old, pos + 1)
        open("%s/%s" % (work, base), "w", encoding="utf-8").write(src[:pos] + new + src[pos + len(old):])
        c = subprocess.run(JAVAC + ["-encoding", "iso-8859-1", "-nowarn", "-d", work + "/out", "-cp", CP, "%s/%s" % (work, base)], capture_output=True, text=True)
        if c.returncode != 0:
            print("[%2d] il mutante non compila: %s\n%s" % (idx, desc, c.stderr[:300]), flush=True); survivors.append(desc + " (non compila)"); continue
        failed, first, by = False, "", ""
        for check, to in CHECKS:
            try:
                r = subprocess.run(["java", "-Xss8m", "-Dmcts.failfast=true", "-cp", "%s/out:%s" % (work, CP), check], capture_output=True, text=True, timeout=to)
                if r.returncode != 0:
                    failed = True
                    first = next((l.strip() for l in r.stdout.splitlines() if "FALLITO" in l and "===" not in l), "") or (r.stderr.splitlines()[-1] if r.stderr else "")
                    by = check.split(".")[-1]
                    break
            except subprocess.TimeoutExpired:
                failed, first, by = True, "(timeout: considerato rilevato)", check.split(".")[-1]
                break
        print("[%2d] %s - %s" % (idx, "RILEVATO     " if failed else "SOPRAVVISSUTO", desc) + (("\n        -> [%s] %s" % (by, first[:150])) if failed else ""), flush=True)
        if failed: killed += 1
        else: survivors.append(desc)
        shutil.rmtree(work, ignore_errors=True)
    print("\n%d/%d mutanti rilevati" % (killed, len(idxs)))
    if survivors:
        print("NON rilevati (buchi dei test o mutanti equivalenti da verificare):"); [print("  -", s) for s in survivors]; sys.exit(1)

main()
