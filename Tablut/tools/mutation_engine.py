#!/usr/bin/env python3
"""
Mutation test della rappresentazione leggera (pacchetto engine) e della ricerca che la usa: inietta UN bug alla volta in una COPIA
del sorgente, ricompila quel solo file in una cartella separata messa PRIMA del build nel classpath, lancia il test di
riferimento e si aspetta che FALLISCA. Un mutante che il test non rileva e' un buco del test, non del mutante.

Uso (dalla cartella Tablut, con il progetto gia' compilato in build/ oppure in $BUILD):
  python3 tools/mutation_engine.py [state|search|order|tactical|tt2] [indiceIniziale indiceFinale]
    state    -> mutanti di FastState/ZobristHash/Move, verificati da FastStateCheck (default)
    search   -> mutanti di Negamax (make/unmake, storia, ordinamento, PVS), verificati da SearchEquivalenceCheck
    order    -> mutanti dell'ordinamento ESTESO di Negamax (wins/caps/counter/killer-carry/ordine radice, newSearch, margine), da OrderingExtCheck
    tactical -> mutanti di FastState.tacticalValue, verificati da FastStateCheck
    tt2      -> mutanti della TT a due slot, verificati da TwoSlotTTCheck
  Variabili: BUILD (cartella delle classi), JAVAC (comando del compilatore, default javac), CHECK_ARGS (argomenti del test).
  Un mutante puo' avere un quinto elemento: l'indice (da 0) dell'occorrenza del frammento da mutare (default 0, la prima).
"""
import os, shutil, subprocess, sys

ROOT = "src/it/unibo/ai/didattica/competition/tablut/"
BUILD = os.environ.get("BUILD", "build")
CP = BUILD + ":lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
T = "it.unibo.ai.didattica.competition.tablut."
CHECK_STATE = T + "selftest.FastStateCheck"
CHECK_SEARCH = T + "aiplayer.SearchEquivalenceCheck"
CHECK_ORDER = T + "aiplayer.OrderingExtCheck"
CHECK_TT2 = T + "aiplayer.TwoSlotTTCheck"
JAVAC = os.environ.get("JAVAC", "javac").split()

S = [  # (descrizione, file relativo a ROOT, vecchio, nuovo)
 ("Bianco: le punte delle citadel contano come muro", "engine/FastState.java", "|| (BoardConstants.IS_CITADEL[wall] && !BoardConstants.IS_TIP[wall])) {", "|| BoardConstants.IS_CITADEL[wall]) {"),
 ("Bianco: il Re non e' un muro", "engine/FastState.java", "if (w == BoardConstants.WHITE || w == BoardConstants.KING || (wall", "if (w == BoardConstants.WHITE || (wall"),
 ("Bianco: il trono vuoto non e' un muro", "engine/FastState.java", "|| (wall == BoardConstants.THRONE && w == BoardConstants.EMPTY)", "|| false"),
 ("Bianco: la cattura ignora il muro (cattura sempre)", "engine/FastState.java", "if (w == BoardConstants.WHITE || w == BoardConstants.KING", "if (true || w == BoardConstants.KING"),
 ("Bianco: conteggio nere non decrementato", "engine/FastState.java", "undo.addCapture(adj, BoardConstants.BLACK);\n\t\t\t\tblackCount--;", "undo.addCapture(adj, BoardConstants.BLACK);"),
 ("Bianco: hash non aggiornato alla cattura", "engine/FastState.java", "zobristHash ^= ZobristHash.KEY[adj][BoardConstants.BLACK];", ""),
 ("Nero: il trono non e' un muro", "engine/FastState.java", "if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {", "if (b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {"),
 ("Nero: le citadel non sono un muro", "engine/FastState.java", "if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {", "if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK) {"),
 ("Nero: conteggio bianche non decrementato", "engine/FastState.java", "undo.addCapture(adj, BoardConstants.WHITE);\n\t\t\t\t\twhiteCount--;", "undo.addCapture(adj, BoardConstants.WHITE);"),
 ("Nero: cattura registrata con il codice sbagliato (unmake rimette una nera)", "engine/FastState.java", "undo.addCapture(adj, BoardConstants.WHITE);", "undo.addCapture(adj, BoardConstants.BLACK);"),
 # NOTA: due mutanti sono EQUIVALENTI (verificato a mano): togliere l'esenzione del "lato del pezzo appena mosso" nella cattura del Re sul trono /
 # adiacente al trono non cambia nulla, perche' makeMove ha gia' scritto la pedina nera in quella cella prima di valutare le catture (b[m] == BLACK).
 # Lo stesso vale per AshtonRules, dove l'esenzione e' ridondante.
 ("Re sul trono: bastano 3 lati (uno ignorato)", "engine/FastState.java", "for (int n : BoardConstants.THRONE_NEIGHBORS) {\n\t\t\t\tif (n != m", "for (int n : BoardConstants.THRONE_NEIGHBORS) {\n\t\t\t\tif (n == 49 || n != m"),
 ("Re adiacente al trono: il trono non conta come lato ostile", "engine/FastState.java", "if (n == BoardConstants.THRONE || n == m) {", "if (n == m) {"),
 ("Re altrove: la citadel non e' un lato ostile", "engine/FastState.java", "return b[f] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[f];", "return b[f] == BoardConstants.BLACK;"),
 ("Re altrove: serve sempre una pedina nera, la citadel conta solo vuota (bug sul muro)", "engine/FastState.java", "return b[f] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[f];", "return b[f] == BoardConstants.BLACK || (BoardConstants.IS_CITADEL[f] && b[f] != BoardConstants.EMPTY);"),
 ("Re altrove: cattura senza cella oltre (fuori scacchiera) ammessa", "engine/FastState.java", "if (!hasCellBeyond) {\n\t\t\treturn false;\n\t\t}", ""),
 ("Re catturato: cella del Re non azzerata", "engine/FastState.java", "kingSquare = -1;\n\t\t\t\t\tkingCaptured = true;", "kingCaptured = true;"),
 ("Re catturato sul trono: hash senza il marcatore di trono vuoto", "engine/FastState.java", "if (adj == BoardConstants.THRONE) {\n\t\t\t\t\t\tzobristHash ^= ZobristHash.THRONE_EMPTY_KEY; // la cella 40 e' ora vuota\n\t\t\t\t\t}", ""),
 ("fuga del Re: anche le celle di citadel/nessun bordo (IS_EDGE ignorato)", "engine/FastState.java", "escaped = piece == BoardConstants.KING && BoardConstants.IS_EDGE[to];", "escaped = piece == BoardConstants.KING && BoardConstants.IS_EDGE[to] && BoardConstants.ROW[to] != 0;"),
 ("makeMove: cella del Re non aggiornata", "engine/FastState.java", "if (piece == BoardConstants.KING) {\n\t\t\tkingSquare = (byte) to;\n\t\t}", ""),
 ("makeMove: hash senza il marcatore del trono quando il Re parte", "engine/FastState.java", "if (from == BoardConstants.THRONE) {\n\t\t\th ^= ZobristHash.THRONE_EMPTY_KEY; // il Re lascia il trono: ora e' vuoto\n\t\t}", ""),
 ("makeMove: contributo del turno anche sugli stati terminali", "engine/FastState.java", "(newWinner == WINNER_NONE && turn == BoardConstants.BLACK_TURN)", "(turn == BoardConstants.BLACK_TURN)"),
 ("makeMove: turno non alternato", "engine/FastState.java", "turn = (byte) (turn ^ 1);", ""),
 ("makeMove: vittoria nera ignorata", "engine/FastState.java", "final byte newWinner = escaped ? WINNER_WHITE : (kingCaptured ? WINNER_BLACK : WINNER_NONE);", "final byte newWinner = escaped ? WINNER_WHITE : WINNER_NONE;"),
 ("unmake: bianche catturate non rimesse nel conteggio", "engine/FastState.java", "if (cp == BoardConstants.WHITE) {\n\t\t\t\twhiteCount++;", "if (cp == 99) {\n\t\t\t\twhiteCount++;"),
 ("unmake: cella del Re non ripristinata", "engine/FastState.java", "kingSquare = undo.oldKingSquare;\n\t\twinner = WINNER_NONE;", "winner = WINNER_NONE;"),
 ("unmake: pedine catturate non rimesse sulla scacchiera", "engine/FastState.java", "b[undo.capturedSquares[i]] = cp;", ""),
 ("unmake: hash non ripristinato", "engine/FastState.java", "zobristHash = undo.oldZobristHash;\n\t}\n\n\t// ----", "\n\t}\n\n\t// ----"),
 ("mosse: le citadel si possono attraversare da qualunque origine", "engine/FastState.java", "if (cit[to] && (!originCitadel || step > 5)) {", "if (cit[to] && (step > 5)) {"),
 ("mosse: nessun tetto di 5 passi tra citadel", "engine/FastState.java", "if (cit[to] && (!originCitadel || step > 5)) {", "if (cit[to] && (!originCitadel)) {"),
 ("mosse: il trono vuoto si puo' attraversare", "engine/FastState.java", "if (b[to] != BoardConstants.EMPTY || to == BoardConstants.THRONE) {\n\t\t\t\t\t\tbreak; // pedina", "if (b[to] != BoardConstants.EMPTY) {\n\t\t\t\t\t\tbreak; // pedina"),
 ("mosse: direzioni scambiate (ordine di generazione diverso)", "engine/BoardConstants.java", "public static final int[] DIR_OFFSET = { -SIZE, SIZE, -1, 1 };", "public static final int[] DIR_OFFSET = { SIZE, -SIZE, -1, 1 };"),
 ("hasLegalMove: ignora il vincolo delle citadel", "engine/FastState.java", "if (cit[to] && !originCitadel) {\n\t\t\t\t\tcontinue; // il primo passo", "if (false) {\n\t\t\t\t\tcontinue; // il primo passo"),
 ("fromState: conteggio nere sbagliato", "engine/FastState.java", "code = BoardConstants.BLACK;\n\t\t\t\t\tbl++;", "code = BoardConstants.BLACK;"),
 ("fromState: turno terminale BLACKWIN trattato come in corso", "engine/FastState.java", "turn = BoardConstants.WHITE_TURN;\n\t\t\twinner = WINNER_BLACK;", "turn = BoardConstants.WHITE_TURN;\n\t\t\twinner = WINNER_NONE;"),
 ("toState: il trono vuoto diventa EMPTY", "engine/FastState.java", "p = (sq == BoardConstants.THRONE) ? Pawn.THRONE : Pawn.EMPTY;", "p = Pawn.EMPTY;"),
 ("Zobrist: chiavi di bianco e nero scambiate", "engine/ZobristHash.java", "KEY[sq][BoardConstants.WHITE] = rnd.nextLong();\n\t\t\t\tKEY[sq][BoardConstants.BLACK] = rnd.nextLong();", "KEY[sq][BoardConstants.BLACK] = rnd.nextLong();\n\t\t\t\tKEY[sq][BoardConstants.WHITE] = rnd.nextLong();"),
 ("Zobrist: chiave del turno a Bianco invece che Nero", "engine/ZobristHash.java", "if (winner == FastState.WINNER_NONE && turn == BoardConstants.BLACK_TURN) {", "if (winner == FastState.WINNER_NONE && turn == BoardConstants.WHITE_TURN) {"),
 ("Zobrist: trono vuoto non incluso", "engine/ZobristHash.java", "if (board[BoardConstants.THRONE] == BoardConstants.EMPTY) {\n\t\t\th ^= THRONE_EMPTY_KEY;\n\t\t}", ""),
 ("Move: indice con da e a scambiati", "engine/Move.java", "return (move & 127) * BoardConstants.NUM_SQUARES + (move >>> 7);", "return (move >>> 7) * BoardConstants.NUM_SQUARES + (move & 127);"),
 ("Move: conversione in Action con riga/colonna scambiate", "engine/BoardConstants.java", "return \"\" + (char) ('a' + COL[sq]) + (ROW[sq] + 1);", "return \"\" + (char) ('a' + ROW[sq]) + (COL[sq] + 1);"),
]

N = [  # mutanti della ricerca (Negamax su FastState): (descrizione, file, vecchio, nuovo)
 ("exploreChild: la mossa non viene annullata (unmake mancante)", "aiplayer/Negamax.java", "\t\ts.unmakeMove(move, u);\n\t\treturn score;", "\t\treturn score;"),
 ("exploreChild: unmake con la mossa sbagliata (scambia da/a)", "aiplayer/Negamax.java", "\t\ts.unmakeMove(move, u);\n\t\treturn score;", "\t\ts.unmakeMove(Move.of(Move.getFrom(move), Move.getFrom(move)), u);\n\t\treturn score;"),
 ("storia: nodo non-cattura non rimosso al ritorno (histLen-- mancante)", "aiplayer/Negamax.java", "\t\t\thistLen--; // backtrack", "\t\t\t// backtrack mancante"),
 ("storia: la cattura non ripristina la lunghezza", "aiplayer/Negamax.java", "\t\t\thistLen = savedLen;\n\t\t\thistBase = savedBase;", "\t\t\thistBase = savedBase;"),
 ("storia: la cattura non ripristina la base del segmento", "aiplayer/Negamax.java", "\t\t\thistLen = savedLen;\n\t\t\thistBase = savedBase;", "\t\t\thistLen = savedLen;"),
 ("storia: la ripetizione non e' contata", "aiplayer/Negamax.java", "\t\t\trepetitionEventCounter++;\n\t\t\tscore = 0;", "\t\t\tscore = 0;"),
 ("storia: ripetizione valutata come vittoria invece che pareggio", "aiplayer/Negamax.java", "\t\t\tscore = 0; // seconda occorrenza", "\t\t\tscore = 1; // seconda occorrenza"),
 # NOTA: "historyContains salta l'ultima voce" e' un mutante EQUIVALENTE (verificato a mano): l'ultima voce della storia e' la posizione GENITORE e un
 # figlio non puo' mai coincidere con il genitore (ogni mossa cambia la scacchiera), quindi saltarla non cambia alcun risultato.
 ("storia: historyContains ignora la prima voce del segmento", "aiplayer/Negamax.java", "for (int i = histBase; i < histLen; i++) {", "for (int i = histBase + 1; i < histLen; i++) {"),
 ("radice: la storia del chiamante non e' copiata (ultima voce persa)", "aiplayer/Negamax.java", "System.arraycopy(history, 0, histKeys, 0, historyLen);\n\t\thistLen = historyLen;", "System.arraycopy(history, 0, histKeys, 0, historyLen);\n\t\thistLen = Math.max(0, historyLen - 1);"),
 ("radice: la posizione di lavoro non viene copiata dalla radice", "aiplayer/Negamax.java", "s.copyFrom(rootState); //", "//"),
 ("terminale: vittoria nera con segno sbagliato", "aiplayer/Negamax.java", "return perspective * (-(WIN_SCORE + depth));\n\t\t\t}\n\t\t\treturn 0; // pareggio", "return perspective * (WIN_SCORE + depth);\n\t\t\t}\n\t\t\treturn 0; // pareggio"),
 ("terminale: la vittoria non dipende dalla profondita' (non preferisce la piu' rapida)", "aiplayer/Negamax.java", "return perspective * (WIN_SCORE + depth); // vince prima", "return perspective * WIN_SCORE; // vince prima"),
 ("stallo: segno invertito", "aiplayer/Negamax.java", "double absolute = (s.turn == BoardConstants.WHITE_TURN) ? -(WIN_SCORE + depth) : (WIN_SCORE + depth);", "double absolute = (s.turn == BoardConstants.WHITE_TURN) ? (WIN_SCORE + depth) : -(WIN_SCORE + depth);"),
 ("TT: un valore LOWER_BOUND usato come esatto", "aiplayer/Negamax.java", "if (pr.flag == TranspositionTable.EXACT) {\n\t\t\t\t\t\treturn pr.score;", "if (pr.flag == TranspositionTable.EXACT || pr.flag == TranspositionTable.LOWER_BOUND) {\n\t\t\t\t\t\treturn pr.score;"),
 ("TT: profondita' insufficiente accettata", "aiplayer/Negamax.java", "if (pr.depthSufficient) {", "if (true) {"),
 ("TT: flag UPPER/LOWER scambiati nel salvataggio", "aiplayer/Negamax.java", "if (best <= alphaOriginal) {\n\t\t\t\tflag = TranspositionTable.UPPER_BOUND;\n\t\t\t} else if (best >= beta) {\n\t\t\t\tflag = TranspositionTable.LOWER_BOUND;", "if (best <= alphaOriginal) {\n\t\t\t\tflag = TranspositionTable.LOWER_BOUND;\n\t\t\t} else if (best >= beta) {\n\t\t\t\tflag = TranspositionTable.UPPER_BOUND;"),
 ("TT: salva anche i nodi che hanno incontrato ripetizioni (GHI ignorata)", "aiplayer/Negamax.java", "if (useTT && !timeUp && repetitionEventCounter == repCounterAtEntry) {", "if (useTT && !timeUp) {"),
 ("TT: salva anche a tempo scaduto", "aiplayer/Negamax.java", "if (useTT && !timeUp && repetitionEventCounter == repCounterAtEntry) {", "if (useTT && repetitionEventCounter == repCounterAtEntry) {"),
 ("TT: mossa migliore non salvata", "aiplayer/Negamax.java", "tt.store(hash, depth, best, flag, bestMoveFound == Move.NONE ? -1 : bestMoveFound);", "tt.store(hash, depth, best, flag, -1);"),
 ("ordinamento: parita' di punteggio risolta con l'ultima invece che la prima (selectBest >=)", "aiplayer/Negamax.java", "if (sc[j] > bv) {", "if (sc[j] >= bv) {"),
 ("ordinamento: scambio senza aggiornare i punteggi", "aiplayer/Negamax.java", "\t\t\tsc[best] = sc[i];\n\t\t\tsc[i] = bv;", "\t\t\tsc[i] = bv;"),
 ("ordinamento: seconda killer ignorata", "aiplayer/Negamax.java", "} else if (idx == k1) {\n\t\t\t\tsc[i] = Integer.MAX_VALUE - 2;", "} else if (idx == -5) {\n\t\t\t\tsc[i] = Integer.MAX_VALUE - 2;"),
 ("ordinamento: la mossa TT non ha priorita' massima", "aiplayer/Negamax.java", "if (idx == ttIdx) {\n\t\t\t\tsc[i] = Integer.MAX_VALUE;", "if (idx == ttIdx) {\n\t\t\t\tsc[i] = 0;"),
 ("ordinamento radice: la mossa preferita non e' portata in testa", "aiplayer/Negamax.java", "\t\t\t\tif (i > 0) {\n\t\t\t\t\tfinal int tmp = moves[0];", "\t\t\t\tif (i > 99999) {\n\t\t\t\t\tfinal int tmp = moves[0];"),
 ("killer: aggiornate anche se gia' prima", "aiplayer/Negamax.java", "if ((orderingMode & ORDER_KILLERS) != 0 && killers[ply][0] != idx) {", "if ((orderingMode & ORDER_KILLERS) != 0) {"),
 ("history: incremento lineare invece che quadratico", "aiplayer/Negamax.java", "hist[idx] += depth * depth;", "hist[idx] += depth;"),
 ("history: lato (Bianco/Nero) invertito nel punteggio", "aiplayer/Negamax.java", "final int[] hist = historyScore[sideIdx];\n\t\tfor (int i = 0; i < n; i++) {", "final int[] hist = historyScore[1 - sideIdx];\n\t\tfor (int i = 0; i < n; i++) {"),
 ("history: nessun invecchiamento tra una ricerca e l'altra", "aiplayer/Negamax.java", "h[i] >>= 1; // invecchiamento", "h[i] += 0; // invecchiamento"),
 ("killer: ply calcolato senza la profondita' della radice", "aiplayer/Negamax.java", "Math.min(killers.length - 1, rootIterDepth - depth));", "Math.min(killers.length - 1, depth));"),
 ("taglio beta: soglia > invece di >=", "aiplayer/Negamax.java", "if (alpha >= beta) {\n\t\t\t\tif (orderScore != null && !timeUp) {", "if (alpha > beta) {\n\t\t\t\tif (orderScore != null && !timeUp) {"),
 ("PVS: la ri-ricerca avviene solo se score < beta E score > alpha+1", "aiplayer/Negamax.java", "if (!timeUp && score > alpha && score < beta) {\n\t\t\t\t\t// la mossa migliora", "if (!timeUp && score > alpha + 1 && score < beta) {\n\t\t\t\t\t// la mossa migliora"),
 ("PVS: la ri-ricerca non avviene mai (valori inesatti)", "aiplayer/Negamax.java", "if (!timeUp && score > alpha && score < beta) {\n\t\t\t\t\t// la mossa migliora", "if (false) {\n\t\t\t\t\t// la mossa migliora"),
 ("radice: la vittoria forzata non interrompe l'approfondimento", "aiplayer/Negamax.java", "if (bestScore >= WIN_SCORE) {\n\t\t\t\t\tbreak;", "if (bestScore >= WIN_SCORE * 10) {\n\t\t\t\t\tbreak;"),
 ("radice: la migliore mossa dell'iterazione non e' aggiornata", "aiplayer/Negamax.java", "\t\t\t\tbest = iterationBest;\n\t\t\t\tbestScore = iterationBestScore;", "\t\t\t\tbestScore = iterationBestScore;"),
 ("Move: conversione a Action con turno sbagliato", "aiplayer/Negamax.java", "return mv == Move.NONE ? null : Move.toAction(mv, rootState.getTurn());", "return mv == Move.NONE ? null : Move.toAction(mv, rootState.getTurn() == it.unibo.ai.didattica.competition.tablut.domain.State.Turn.WHITE ? it.unibo.ai.didattica.competition.tablut.domain.State.Turn.BLACK : it.unibo.ai.didattica.competition.tablut.domain.State.Turn.WHITE);"),
]

NF = "aiplayer/Negamax.java"
O = [  # mutanti dell'ordinamento ESTESO (punteggio esteso, countermove, killer-carry, ordine radice, newSearch, margine): (descrizione, file, vecchio, nuovo)
 ("ext: la vittoria immediata non ha priorita'", NF, "if (wins && tv == FastState.TACT_WIN) {\n\t\t\t\tsc[i] = SC_WIN;", "if (false && tv == FastState.TACT_WIN) {\n\t\t\t\tsc[i] = SC_WIN;"),
 ("ext: con catture-prima una cattura che e' anche killer non e' una cattura", NF, "} else if (capsFirst && caps && tv > 0) {", "} else if (false) {"),
 ("ext: il rango della vittoria nelle catture non e' limitato a 4 (overflow)", NF, "final int rank = tv >= FastState.TACT_WIN ? 4 : tv;\n\t\t\t\tsc[i] = capBase + rank * SC_CAPTURE_STEP + Math.min(useHistory", "final int rank = tv;\n\t\t\t\tsc[i] = capBase + rank * SC_CAPTURE_STEP + Math.min(useHistory"),
 ("ext: le catture non hanno priorita' sulle quiete (dopo le killer)", NF, "if (caps && tv > 0) {\n\t\t\t\t\tfinal int rank", "if (false) {\n\t\t\t\t\tfinal int rank"),
 ("ext: countermove letta con il lato sbagliato", NF, "ctr = counterMove[sideIdx][prevIdx];", "ctr = counterMove[1 - sideIdx][prevIdx];"),
 ("ext: la killer 0 e' sostituita dalla killer 1", NF, "} else if (idx == k0) {\n\t\t\t\tsc[i] = s0;", "} else if (idx == k1) {\n\t\t\t\tsc[i] = s0;"),
 ("ext: la history e' ignorata nelle quiete", NF, "\t\t\t\t} else {\n\t\t\t\t\tsc[i] = h;", "\t\t\t\t} else {\n\t\t\t\t\tsc[i] = 0;"),
 ("ext: la mossa TT non ha priorita' massima", NF, "if (idx == ttIdx) {\n\t\t\t\tsc[i] = SC_TT;", "if (idx == ttIdx) {\n\t\t\t\tsc[i] = 0;"),
 ("killer carry: slittamento di 1 ply invece di 2", NF, "final int q = p + 2;", "final int q = p + 1;"),
 ("killer carry: non avviene (azzeramento)", NF, "if ((orderingMode & ORDER_KILLER_CARRY) != 0) {\n\t\t\tfor (int p = 0", "if (false) {\n\t\t\tfor (int p = 0"),
 ("killer: l'azzeramento di default non avviene", NF, "\t\t} else {\n\t\t\tfor (int[] k : killers) {\n\t\t\t\tk[0] = -1;\n\t\t\t\tk[1] = -1;\n\t\t\t}\n\t\t}\n\t}", "\t\t} else {\n\t\t}\n\t}"),
 ("ordine radice: ordinamento non stabile (<=)", NF, "while (j >= 0 && scores[j] < v) {", "while (j >= 0 && scores[j] <= v) {"),
 ("ordine radice: ordine crescente invece che decrescente", NF, "while (j >= 0 && scores[j] < v) {", "while (j >= 0 && scores[j] > v) {"),
 ("ordine radice: la migliore non e' portata in testa", NF, "if (at > 0) {\n\t\t\t\tfinal int mb", "if (false) {\n\t\t\t\tfinal int mb"),
 ("ordine radice: i punteggi delle mosse non sono registrati", NF, "\t\t\t\trootScores[mi] = score;\n", ""),
 ("ordine radice: il riordino dopo l'iterazione non avviene", NF, "if (rootByScores) {\n\t\t\t\t\tsortRootByScores(rootMoves, rootScores, nRoot, best);\n\t\t\t\t}", ""),
 ("countermove: mossa precedente registrata nel ply sbagliato", NF, "prevMoveAt[sp + 1] = Move.index(move);", "prevMoveAt[sp] = Move.index(move);"),
 ("countermove: al taglio si registra la mossa precedente invece della risposta", NF, "counterMove[sideIdx][prevMoveAt[sp]] = Move.index(a);", "counterMove[sideIdx][prevMoveAt[sp]] = prevMoveAt[sp];"),
 ("countermove: scritta nella riga del lato sbagliato", NF, "counterMove[sideIdx][prevMoveAt[sp]] = Move.index(a);", "counterMove[1 - sideIdx][prevMoveAt[sp]] = Move.index(a);"),
 ("countermove: la tabella non viene mai allocata", NF, "if ((orderingMode & ORDER_COUNTER) == 0) {\n\t\t\treturn null;", "if (true) {\n\t\t\treturn null;"),
 ("dispatch: il punteggio esteso non e' mai usato", NF, "if ((orderingMode & ORDER_EXT_MASK) != 0) {\n\t\t\t\tcomputeOrderScoresExt", "if (false) {\n\t\t\t\tcomputeOrderScoresExt"),
 ("dispatch: il punteggio esteso e' usato sempre", NF, "if ((orderingMode & ORDER_EXT_MASK) != 0) {\n\t\t\t\tcomputeOrderScoresExt", "if (true) {\n\t\t\t\tcomputeOrderScoresExt"),
 ("TT: newSearch non chiamato a ogni mossa", NF, "\t\t\ttt.newSearch();", ""),
 ("margine: il margine di istanza e' ignorato (resta quello storico)", NF, "timeBudgetMillis - safetyMarginMillis);", "timeBudgetMillis - SAFETY_MARGIN_MILLIS);"),
]

FS = "engine/FastState.java"
TAC = [  # mutanti di FastState.tacticalValue (fondo del file: i frammenti che compaiono anche in makeMove sono la SECONDA occorrenza)
 ("tactical: la fuga del Re non e' una vittoria", FS, "if (b[from] == BoardConstants.KING && BoardConstants.IS_EDGE[to]) {", "if (false) {"),
 ("tactical: Bianco, nessun controllo che la cella oltre esista (STEPS < 1)", FS, "if (BoardConstants.STEPS[d][to] < 2) {", "if (BoardConstants.STEPS[d][to] < 1) {"),
 ("tactical: Bianco, cattura anche pedine bianche", FS, "if (b[adj] != BoardConstants.BLACK) {\n\t\t\t\t\tcontinue;\n\t\t\t\t}\n\t\t\t\tfinal int wall = adj + off;\n\t\t\t\tfinal byte w = b[wall];", "if (b[adj] == BoardConstants.EMPTY) {\n\t\t\t\t\tcontinue;\n\t\t\t\t}\n\t\t\t\tfinal int wall = adj + off;\n\t\t\t\tfinal byte w = b[wall];"),
 ("tactical: Bianco, le punte delle citadel contano come muro", FS, "|| (BoardConstants.IS_CITADEL[wall] && !BoardConstants.IS_TIP[wall])) {", "|| BoardConstants.IS_CITADEL[wall]) {", 1),
 ("tactical: Bianco, il Re non e' un muro", FS, "if (w == BoardConstants.WHITE || w == BoardConstants.KING || (wall", "if (w == BoardConstants.WHITE || (wall", 1),
 ("tactical: Bianco, il trono vuoto non e' un muro", FS, "|| (wall == BoardConstants.THRONE && w == BoardConstants.EMPTY)", "|| false", 1),
 ("tactical: Bianco, conta due per cattura", FS, "\t\t\t\t\tcount++;\n\t\t\t\t}\n\t\t\t}\n\t\t\treturn count;", "\t\t\t\t\tcount += 2;\n\t\t\t\t}\n\t\t\t}\n\t\t\treturn count;"),
 ("tactical: Nero, il trono non e' un muro", FS, "if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {", "if (b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {", 1),
 ("tactical: Nero, le citadel non sono un muro", FS, "if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {", "if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK) {", 1),
 ("tactical: Nero, la cattura del Re non e' vista", FS, "} else if (p == BoardConstants.KING && isKingCaptured(adj, to, off, steps >= 2)) {", "} else if (false) {"),
 ("tactical: Nero, la cattura del Re vale una pedina e non una vittoria", FS, "isKingCaptured(adj, to, off, steps >= 2)) {\n\t\t\t\treturn TACT_WIN;", "isKingCaptured(adj, to, off, steps >= 2)) {\n\t\t\t\tcount++;"),
 ("tactical: Nero, Re al bordo catturabile senza cella oltre", FS, "isKingCaptured(adj, to, off, steps >= 2)", "isKingCaptured(adj, to, off, true)"),
]

TF = "aiplayer/TranspositionTable.java"
TT2 = [  # mutanti della TT a due slot
 ("tt2: nuova posizione, slot 0 solo se strettamente piu' profonda (> invece di >=)", TF, "if (pEmpty || depth >= pDepth || pStale) {", "if (pEmpty || depth > pDepth || pStale) {"),
 ("tt2: nessun aging (lo slot 0 vecchio non e' sostituibile)", TF, "if (pEmpty || depth >= pDepth || pStale) {", "if (pEmpty || depth >= pDepth) {"),
 ("tt2: aging invertito (sostituisce solo la voce della ricerca corrente)", TF, "final boolean pStale = ((int) ((pMeta >>> 10) & 63)) != generation;", "final boolean pStale = ((int) ((pMeta >>> 10) & 63)) == generation;"),
 ("tt2: stessa posizione nello slot 0, aggiorna solo se piu' profonda (> invece di >=)", TF, "if (pSame) {\n\t\t\tif (depth >= pDepth) {", "if (pSame) {\n\t\t\tif (depth > pDepth) {"),
 ("tt2: stessa posizione nello slot 1, aggiorna sempre (anche con profondita' minore)", TF, "if (depth >= ((int) (aMeta & 0xFF)) - 1) {", "if (true) {"),
 ("tt2: il probe non guarda lo slot 1", TF, "if ((meta & 0xFF) != 0 && (check ^ meta ^ scoreBits) == hash) {\n\t\t\t\t\tfillProbe", "if (false) {\n\t\t\t\t\tfillProbe"),
 ("tt2: il probe accetta lo slot 1 senza verificare la chiave", TF, "if ((meta & 0xFF) != 0 && (check ^ meta ^ scoreBits) == hash) {\n\t\t\t\t\tfillProbe", "if ((meta & 0xFF) != 0) {\n\t\t\t\t\tfillProbe"),
 ("tt2: newSearch non avanza la generazione", TF, "generation = (generation + 1) & 63;", ""),
 ("tt2: le voci sono scritte sempre con generazione 0", TF, "packMeta(depth, flag, packedMove, generation)", "packMeta(depth, flag, packedMove, 0)"),
 ("tt2: indice del bucket senza il fattore 2 (bucket sovrapposti)", TF, "(int) (hash & mask) * 2 : (int) (hash & mask);", "(int) (hash & mask) : (int) (hash & mask);"),
 ("tt2: la posizione che non va nello slot 0 sovrascrive lo slot 0 invece del 1", TF, "} else {\n\t\t\twriteAt(b1, hash, depth, score, flag, packedMove);\n\t\t}\n\t}", "} else {\n\t\t\twriteAt(b0, hash, depth, score, flag, packedMove);\n\t\t}\n\t}"),
 ("tt2: i bit della generazione si sovrappongono al flag", TF, "((long) (gen & 63) << 10)", "((long) (gen & 63) << 8)"),
 ("tt2: occupancy conta i vuoti", TF, "if ((slots.get(i * ENTRY_LONGS + 1) & 0xFF) != 0) {\n\t\t\t\tused++;", "if ((slots.get(i * ENTRY_LONGS + 1) & 0xFF) == 0) {\n\t\t\t\tused++;"),
 ("tt2: maschera dei bucket uguale a quella dello slot singolo", TF, "this.mask = twoSlot ? (n >> 1) - 1 : n - 1;", "this.mask = n - 1;"),
 ("tt2: capacity sbagliata", TF, "public int capacity() {\n\t\treturn size;", "public int capacity() {\n\t\treturn size - 1;"),
]

def main():
    if not os.path.isdir("src") or not os.path.isdir(BUILD):
        sys.exit("lancia dalla cartella Tablut (serve src/ e la cartella di build gia' compilata, $BUILD)")
    suite = sys.argv[1] if len(sys.argv) > 1 else "state"
    muts, check, to, default_args = {"search": (N, CHECK_SEARCH, 600, ""), "order": (O, CHECK_ORDER, 280, "2024 6 unit+mech+visit"),
                                     "tactical": (TAC, CHECK_STATE, 300, ""), "tt2": (TT2, CHECK_TT2, 120, "")}.get(suite, (S, CHECK_STATE, 300, ""))
    extra = os.environ.get("CHECK_ARGS", default_args).split()
    lo = int(sys.argv[2]) if len(sys.argv) > 3 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(muts)
    # BASELINE: il test deve PASSARE sul codice non mutato, altrimenti ogni mutante risulterebbe "rilevato" per il motivo sbagliato
    b = subprocess.run(["java", "-cp", CP, check] + extra, capture_output=True, text=True, timeout=to)
    if b.returncode != 0:
        sys.exit("BASELINE FALLITA: il test non passa sul codice non mutato; i risultati sui mutanti non avrebbero senso.\n" + "\n".join(b.stdout.splitlines()[-6:]))
    print("baseline OK (il test passa sul codice non mutato)")
    survivors, killed = [], 0
    for idx in range(lo, hi):
        desc, rel, old, new = muts[idx][:4]
        nth = muts[idx][4] if len(muts[idx]) > 4 else 0
        base = os.path.basename(rel)
        work = f"/tmp/mute_{suite}_{idx}"; shutil.rmtree(work, ignore_errors=True); os.makedirs(work + "/out")
        src = open(ROOT + rel, encoding="utf-8").read()
        if src.count(old) <= nth:
            print(f"[{idx:2d}] ERRORE: frammento da mutare non trovato: {desc}"); survivors.append(desc + " (frammento assente)"); continue
        pos = -1
        for _ in range(nth + 1):
            pos = src.index(old, pos + 1)
        open(f"{work}/{base}", "w", encoding="utf-8").write(src[:pos] + new + src[pos + len(old):])
        c = subprocess.run(JAVAC + ["-encoding", "iso-8859-1", "-nowarn", "-d", work + "/out", "-cp", CP, f"{work}/{base}"], capture_output=True, text=True)
        if c.returncode != 0:
            print(f"[{idx:2d}] il mutante non compila: {desc}\n{c.stderr[:300]}"); survivors.append(desc + " (non compila)"); continue
        try:
            r = subprocess.run(["java", "-cp", f"{work}/out:{CP}", check] + extra, capture_output=True, text=True, timeout=to)
            failed = r.returncode != 0
            first = next((l for l in r.stdout.splitlines() if l.startswith("FAIL")), "") or (r.stderr.splitlines()[-1] if r.stderr else "")
        except subprocess.TimeoutExpired:
            failed, first = True, "(timeout: considerato rilevato)"
        print(f"[{idx:2d}] {'RILEVATO ' if failed else 'SOPRAVVISSUTO'} - {desc}" + (f"\n        -> {first[:150]}" if failed else ""))
        if failed: killed += 1
        else: survivors.append(desc)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{killed}/{hi - lo} mutanti rilevati ({suite})")
    if survivors:
        print("NON rilevati (buchi del test):"); [print("  -", s) for s in survivors]; sys.exit(1)

main()
