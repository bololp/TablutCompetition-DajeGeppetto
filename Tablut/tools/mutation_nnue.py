#!/usr/bin/env python3
"""
Mutation test dei test NNUE: inietta UN bug alla volta in una COPIA di NnueNetwork / NnueSession / NnueEvaluator / Negamax /
EvaluatorLoader / CachingEvaluator / TablutNegamaxClient, ricompila in una cartella separata messa PRIMA delle classi gia' compilate
nel classpath, lancia il test che deve accorgersene e si aspetta che FALLISCA. Un mutante che il test non rileva e' un buco del test.
Stesso metodo di tools/mutation_cnn.py.

Uso (dalla cartella Tablut):  python3 tools/mutation_nnue.py [unit|search|client|training|data|selfplay|trainer|all] [indiceIniziale indiceFinale]
  unit   -> NnueCheck        (rete, evaluator, sessione, file dei pesi)
  search -> NnueSearchCheck  (integrazione con Negamax; ~25 s per mutante)
  client -> NnueClientCheck  (selezione del backend eval=)
  training -> NnueTrainingCheck (distillatore: gradiente, quantizzazione, dati, determinismo)
  data     -> NnueDataCheck     (formato dati, importatori dei log veri / dati.gz / DistillSet, NnueDataTool)
  selfplay -> NnueSelfPlayCheck (auto-gioco, rietichettatura, CLI dei dati)
  trainer  -> NnueTrainerCheck  (addestramento da dati di gioco)
Variabili d'ambiente: BUILD = cartella delle classi gia' compilate (default 'build').
"""
import os, shutil, subprocess, sys

ROOT = "src/it/unibo/ai/didattica/competition/tablut/"
BUILD = os.environ.get("BUILD", "build")
CP = f"{BUILD}:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
T = "it.unibo.ai.didattica.competition.tablut."
CHECKS = {"unit": T + "aiplayer.NnueCheck", "search": T + "aiplayer.NnueSearchCheck", "client": T + "client.NnueClientCheck", "training": T + "training.NnueTrainingCheck", "data": T + "training.NnueDataCheck",
          "selfplay": T + "training.NnueSelfPlayCheck", "trainer": T + "training.NnueTrainerCheck"}
TIMEOUT = {"unit": 240, "search": 420, "client": 300, "training": 300, "data": 300, "selfplay": 300, "trainer": 300}

NET, SES, EVA = "aiplayer/NnueNetwork.java", "aiplayer/NnueSession.java", "aiplayer/NnueEvaluator.java"
NEG, LOAD, CACHE, CLI = "aiplayer/Negamax.java", "aiplayer/EvaluatorLoader.java", "aiplayer/CachingEvaluator.java", "client/TablutNegamaxClient.java"
TRN = "training/NnueDistiller.java"
NDS, NDI, NSP, NRL, NTR, NDT = ("training/NnueDataset.java", "training/NnueDataImport.java", "training/NnueSelfPlay.java", "training/NnueRelabel.java",
                                "training/NnueTrainer.java", "training/NnueDataTool.java")

M = {"unit": [  # (descrizione, file, vecchio, nuovo)
    # ---- NnueNetwork: feature e aritmetica
    ("la feature del Re coincide con quelle del Nero", NET, "private static final int[] BASE = { -1, 0, NUM_SQUARES, -1, 2 * NUM_SQUARES };", "private static final int[] BASE = { -1, 0, NUM_SQUARES, -1, NUM_SQUARES };"),
    ("refresh: bias dell'accumulatore non copiato", NET, "\t\t\tacc[j] = ftB[j];\n\t\t}\n\t\tlong sig = 0L;", "\t\t\tacc[j] = 0;\n\t\t}\n\t\tlong sig = 0L;"),
    ("refresh: firma non aggiornata", NET, "\t\t\tsig ^= FEATURE_KEY[f];\n\t\t}\n\t\treturn sig;", "\t\t}\n\t\treturn sig;"),
    ("applyMove: segni scambiati", NET, "dst[j] = src[j] - w[s + j] + w[a + j];", "dst[j] = src[j] + w[s + j] - w[a + j];"),
    ("applyMove: la colonna di arrivo e' quella di partenza", NET, "final int a = addFeature * h;", "final int a = subFeature * h;"),
    ("subtract: somma invece di sottrarre", NET, "acc[j] -= w[o + j];", "acc[j] += w[o + j];"),
    ("clipped ReLU 1: nessun limite superiore", NET, "\t\t\tif (a > QA) {\n\t\t\t\ta = QA;\n\t\t\t}\n", ""),
    ("clipped ReLU 1: nessun limite inferiore (i neuroni negativi non si saltano)", NET, "\t\t\tif (a <= 0) {\n\t\t\t\tcontinue;\n\t\t\t}\n", ""),
    ("strato 1: contributo senza il fattore di attivazione", NET, "work[i] += wt[o + i] * a;", "work[i] += wt[o + i];"),
    ("strato 1: matrice trasposta male (indice dell'uscita)", NET, "this.l1WT[r][j * l2 + i] = l1W[r][i * hidden + j];", "this.l1WT[r][j * l2 + i] = l1W[r][j * l2 + i];"),
    ("strato 1: passo della matrice trasposta sbagliato", NET, "for (int j = 0, o = 0; j < h; j++, o += l) {", "for (int j = 0, o = 0; j < h; j++, o += l + 1) {"),
    ("strato 1: l'ultimo neurone nascosto non e' sommato", NET, "for (int j = 0, o = 0; j < h; j++, o += l) {", "for (int j = 0, o = 0; j < h - 1; j++, o += l) {"),
    ("clipped ReLU 2: nessun limite superiore", NET, "c = c < 0 ? 0 : (c > QA ? QA : c);", "c = c < 0 ? 0 : c;"),
    ("clipped ReLU 2: nessun limite inferiore", NET, "c = c < 0 ? 0 : (c > QA ? QA : c);", "c = c > QA ? QA : c;"),
    ("riscalamento senza arrotondamento", NET, "int c = (work[i] + (QW >> 1)) >> QW_SHIFT;", "int c = work[i] >> QW_SHIFT;"),
    ("riscalamento con shift sbagliato", NET, "int c = (work[i] + (QW >> 1)) >> QW_SHIFT;", "int c = (work[i] + (QW >> 1)) >> (QW_SHIFT - 1);"),
    ("strato 1: pesi sempre della testa 0", NET, "final short[] wt = l1WT[role];", "final short[] wt = l1WT[0];"),
    ("strato 1: bias sempre della testa 0", NET, "final int[] b1 = l1B[role];", "final int[] b1 = l1B[0];"),
    ("strato 2: pesi sempre della testa 0", NET, "final short[] w2 = l2W[role];", "final short[] w2 = l2W[0];"),
    ("uscita: bias sempre della testa 0", NET, "int z = l2B[role];", "int z = l2B[0];"),
    ("strato 1: bias ignorato", NET, "work[i] = b1[i];", "work[i] = 0;"),
    ("uscita: bias ignorato", NET, "int z = l2B[role];", "int z = 0;"),
    ("tanh: nessuna interpolazione", NET, "return lo + (TANH[idx + 1] - lo) * ((t & ((1 << LUT_SHIFT) - 1)) * (1.0 / (1 << LUT_SHIFT)));", "return lo;"),
    ("tanh: intervallo dimezzato", NET, "private static final int Z_RANGE = 8 * QA * QW;", "private static final int Z_RANGE = 4 * QA * QW;"),
    ("tanh: saturazione superiore sbagliata", NET, "(z > Z_RANGE - 1 ? Z_RANGE - 1 : z);", "(z > Z_RANGE - 1 ? Z_RANGE - 2 : z);"),
    # ---- firma di coerenza
    ("firma: manca il termine del trono vuoto", NET, "h ^= ZobristHash.THRONE_EMPTY_KEY;", "h ^= 0L;"),
    ("firma: manca il termine del turno", NET, "h ^= ZobristHash.TURN_KEY;", "h ^= 0L;"),
    ("firma: chiave Zobrist della pedina sbagliata", NET, "FEATURE_KEY[BASE[p] + sq] = ZobristHash.KEY[sq][p];", "FEATURE_KEY[BASE[p] + sq] = ZobristHash.KEY[sq][BoardConstants.WHITE];"),
    # ---- NnueNetwork: file e validazione
    ("overflow dello strato 1 non verificato", NET, "\t\t\t\tif (sum > limit) {\n\t\t\t\t\tthrow new IllegalArgumentException(\"l1W/l1B[\"", "\t\t\t\tif (false) {\n\t\t\t\t\tthrow new IllegalArgumentException(\"l1W/l1B[\""),
    ("overflow dell'uscita non verificato", NET, "\t\t\tif (sum > limit) {\n\t\t\t\tthrow new IllegalArgumentException(\"l2W/l2B[\"", "\t\t\tif (false) {\n\t\t\t\tthrow new IllegalArgumentException(\"l2W/l2B[\""),
    ("checksum non verificato", NET, "if (declared != checksum(all)) {", "if (false) {"),
    ("scale di quantizzazione non verificate", NET, "if (qa != QA || qw != QW) {", "if (false) {"),
    ("intervallo short non verificato", NET, "if (v < Short.MIN_VALUE || v > Short.MAX_VALUE) {", "if (false) {"),
    ("insieme di feature non verificato", NET, "if (!features.equals(FEATURE_SET)) {", "if (false) {"),
    ("versione del formato non verificata", NET, "if (!first.equals(MAGIC)) {", "if (false) {"),
    ("params non verificato contro le dimensioni", NET, "if (np != paramCount(hidden, l2)) {", "if (false) {"),
    ("dimensioni non verificate al caricamento", NET, "if (hidden < 1 || hidden > MAX_HIDDEN || l2 < 1 || l2 > MAX_L2) {", "if (false) {"),
    ("file troncato non rilevato", NET, "if (k != np) {", "if (false) {"),
    ("valori in eccesso accettati", NET, "if (k >= np) {", "if (false) {"),
    ("contenuto dopo il checksum ignorato", NET, "if (!extra.trim().isEmpty()) {", "if (false) {"),
    ("salvataggio: ftB scritto prima di ftW", NET, "\t\tfor (short v : ftW) {\n\t\t\tall[k++] = v;\n\t\t}\n\t\tfor (short v : ftB) {\n\t\t\tall[k++] = v;\n\t\t}", "\t\tfor (short v : ftB) {\n\t\t\tall[k++] = v;\n\t\t}\n\t\tfor (short v : ftW) {\n\t\t\tall[k++] = v;\n\t\t}"),
    ("costruttore: ftW non copiato (rete non immutabile)", NET, "this.ftW = ftW.clone();", "this.ftW = ftW;"),
    ("ftWeights restituisce l'array interno", NET, "public short[] ftWeights() {\n\t\treturn ftW.clone();", "public short[] ftWeights() {\n\t\treturn ftW;"),
    ("numero di teste non verificato", NET, "if (n != ROLES) {", "if (false) {"),
    ("isNnueFile non riconosce l'intestazione", NET, "return first != null && first.startsWith(MAGIC_PREFIX);", "return false;"),
    ("EvaluatorLoader non riconosce l'NNUE", LOAD, "if (NnueNetwork.isNnueFile(path)) {", "if (false) {"),
    # ---- NnueEvaluator
    ("evaluate completo: sempre la testa 0", EVA, "net.logit(c.acc, c.act, s.turn == BoardConstants.BLACK_TURN ? 1 : 0)", "net.logit(c.acc, c.act, 0)"),
    ("evaluate completo: outputScale ignorato", EVA, ")) * outputScale;", ")) * 1000.0;"),
    # ---- NnueSession
    ("applyDelta: catture ignorate", SES, "net.subtract(dst, f);", "// catture ignorate"),
    ("applyDelta: firma delle catture dimenticata (il valore resta giusto)", SES, "g ^= NnueNetwork.FEATURE_KEY[f];", ""),
    ("applyDelta: parte dal frame 0 invece che dal genitore", SES, "net.applyMove(acc[j - 1], dst, fromF, toF);", "net.applyMove(acc[0], dst, fromF, toF);"),
    ("applyDelta: firma parte dal frame 0", SES, "long g = sig[j - 1] ^", "long g = sig[0] ^"),
    ("push: pedina mossa registrata sempre come Bianco", SES, "mvPiece[ply] = piece;", "mvPiece[ply] = 1;"),
    ("push: frame non marcato da calcolare (riuso del frame di un fratello)", SES, "\t\tstate[ply] = PENDING;", ""),
    ("push: pedina catturata registrata sempre come Bianco", SES, "capPc[o + i] = pc;", "capPc[o + i] = 1;"),
    ("push: cella catturata registrata come la cella di arrivo", SES, "capSq[o + i] = (byte) sq;", "capSq[o + i] = (byte) to;"),
    ("push: la pila non cresce", SES, "allocate(Math.max(ply + 1, cap * 2));", "allocate(cap);"),
    ("materialize: riapplica la differenza dell'antenato", SES, "for (int j = k + 1; j <= ply; j++) {\n\t\t\tapplyDelta(j);", "for (int j = k; j <= ply; j++) {\n\t\t\tapplyDelta(j);"),
    ("materialize: catena con frame inutilizzabili accettata", SES, "ok = state[j] == PENDING;", "ok = true;"),
    ("evaluate: controllo di coerenza rimosso", SES, "if (sig[ply] != NnueNetwork.boardSignature(s)) {", "if (false) {"),
    ("evaluate: testa del lato al tratto ignorata", SES, "final int role = s.turn == BoardConstants.BLACK_TURN ? 1 : 0;", "final int role = 0;"),
    ("reset: radice non ricostruita", SES, "rebuild(0, root);\n\t\tfullRebuilds++;", "fullRebuilds++;"),
    ("rebuild: frame ricostruito non marcato come calcolato", SES, "sig[ply] = net.refresh(s.board, acc[ply]);\n\t\tstate[ply] = COMPUTED;", "sig[ply] = net.refresh(s.board, acc[ply]);"),
    ("fallbackRebuilds non conta il ripiego da firma", SES, "rebuild(ply, s);\n\t\t\tfallbackRebuilds++;\n\t\t}\n\t\tfinal int role", "rebuild(ply, s);\n\t\t}\n\t\tfinal int role"),
], "search": [
    ("searchRoot: sessione non reinizializzata alla radice", NEG, "\t\tif (inc != null) {\n\t\t\tinc.reset(s); // la radice e' l'unico punto in cui l'accumulatore si ricostruisce da zero\n\t\t}\n", ""),
    ("exploreChild: push dimenticato", NEG, "\t\t\tinc.push(sp + 1, move, s, u); // registra la differenza di feature; l'unmake non richiede nulla (i frame sono una pila)\n", ""),
    ("exploreChild: push sul ply sbagliato", NEG, "inc.push(sp + 1, move, s, u);", "inc.push(sp + 2, move, s, u);"),
    ("negamax: valuta il frame del genitore", NEG, "inc.evaluate(sp, s)", "inc.evaluate(Math.max(0, sp - 1), s)"),
    ("installForTest: radice non reinstallata", NEG, "\t\tif (inc != null) {\n\t\t\tinc.reset(st);\n\t\t}\n\t\tensureHist", "\t\tensureHist"),
    ("helper Lazy SMP senza sessione propria", NEG, "this.inc = main.evaluator instanceof IncrementalEvaluator ? ((IncrementalEvaluator) main.evaluator).newSession(main.maxDepth + 3) : null;", "this.inc = null;"),
    ("helper Lazy SMP condivide la sessione del principale", NEG, "this.inc = main.evaluator instanceof IncrementalEvaluator ? ((IncrementalEvaluator) main.evaluator).newSession(main.maxDepth + 3) : null;", "this.inc = main.inc;"),
    ("principale senza sessione (percorso completo, solo piu' lento)", NEG, "this.inc = evaluator instanceof IncrementalEvaluator ? ((IncrementalEvaluator) evaluator).newSession(maxDepth + 3) : null;", "this.inc = null;"),
], "client": [
    ("eval esplicito ripiega in silenzio su Simple", CLI, "throw new IOException(\"eval=\" + backend + \": impossibile caricare i pesi da '\" + weightsFile + \"' (\" + e.getMessage() + \")\", e);", "return new SimpleEvaluator();"),
    ("tipo del file non verificato nelle scelte esplicite", CLI, "if (!matches) {", "if (false) {"),
    ("eval esplicito senza pesi non e' un errore", CLI, "\t\tif (weightsFile == null) {\n\t\t\tthrow new IOException(\"eval=\" + backend + \" richiede", "\t\tif (false) {\n\t\t\tthrow new IOException(\"eval=\" + backend + \" richiede"),
    ("eval sconosciuto ignorato", CLI, "if (!isKnownEvalBackend(evalBackend)) {", "if (false) {"),
    ("eval non normalizzato in minuscolo", CLI, "evalBackend = args[i].substring(\"eval=\".length()).trim().toLowerCase();", "evalBackend = args[i].substring(\"eval=\".length()).trim();"),
    ("la cache avvolge anche l'NNUE", CACHE, "return e instanceof NeuralEvaluator || e instanceof CnnEvaluator;", "return e instanceof NeuralEvaluator || e instanceof CnnEvaluator || e instanceof NnueEvaluator;"),
    ("auto scarta i pesi NNUE (ripiega su Simple)", CLI, "evaluator = EvaluatorLoader.load(weightsFile); // NNUE, CNN o MLP, riconosciuto dall'intestazione del file", "evaluator = EvaluatorLoader.load(weightsFile); if (evaluator instanceof NnueEvaluator) { evaluator = new SimpleEvaluator(); }"),
], "training": [
    ("gradiente: derivata della clipped ReLU 2 senza il limite superiore", TRN, "if (ws.u[i] > 0 && ws.u[i] < 1) {", "if (ws.u[i] > 0) {"),
    ("gradiente: derivata della clipped ReLU 1 senza il limite superiore", TRN, "if (ws.h[j] > 0 && ws.h[j] < 1) {", "if (ws.h[j] > 0) {"),
    ("gradiente: pesi di uscita senza l'attivazione", TRN, "g[n.offL2W(role) + i] += dz * ws.c[i];", "g[n.offL2W(role) + i] += dz;"),
    ("gradiente: derivata della tanh dimenticata", TRN, "backward(n, d.role[i], 2 * err * (1 - y * y), ws, g);", "backward(n, d.role[i], 2 * err, ws, g);"),
    ("gradiente: bias dello strato 1 non accumulato", TRN, "g[n.offL1B(role) + i] += du;", ""),
    ("gradiente: bias dell'accumulatore non accumulato", TRN, "g[n.oFtB + j] += dh;", ""),
    ("gradiente: colonne FT accumulate sulla feature sbagliata", TRN, "g[ws.act[k] * H + j] += dh;", "g[ws.act[0] * H + j] += dh;"),
    ("quantizzazione: ftW con la scala sbagliata", TRN, "ftW[i] = toShort(f.p[i] * NnueNetwork.QA, clipped);", "ftW[i] = toShort(f.p[i] * NnueNetwork.QW, clipped);"),
    ("quantizzazione: bias dello strato 1 con la scala dei pesi", TRN, "l1B[r][i] = toInt(f.p[f.offL1B(r) + i] * sb, clipped);", "l1B[r][i] = toInt(f.p[f.offL1B(r) + i] * NnueNetwork.QA, clipped);"),
    ("quantizzazione: bias di uscita con la scala sbagliata", TRN, "l2B[r] = toInt(f.p[f.offL2B(r)] * sb, clipped);", "l2B[r] = toInt(f.p[f.offL2B(r)] * NnueNetwork.QA, clipped);"),
    ("quantizzazione: saturazione dei pesi non contata", TRN, "\t\t\tclipped[0]++;\n\t\t\treturn (short) Math.max(Short.MIN_VALUE", "\t\t\treturn (short) Math.max(Short.MIN_VALUE"),
    ("split: per campione invece che per posizione di partenza", TRN, "final long h = (d.group[i] * 0x9E3779B97F4A7C15L)", "final long h = (i * 0x9E3779B97F4A7C15L)"),
    ("dati: 7 simmetrie invece di 8", TRN, "final int syms = cfg.augment ? BoardSymmetry.COUNT : 1;", "final int syms = cfg.augment ? BoardSymmetry.COUNT - 1 : 1;"),
    ("dati: etichetta della posizione originale invece di quella trasformata", TRN, "final double v = teacher.evaluate(t) / cfg.teacherScale;", "final double v = teacher.evaluate(fs) / cfg.teacherScale;"),
    ("dati: bersaglio non limitato a [-1,1]", TRN, "targets.add(Math.max(-1.0, Math.min(1.0, v)));", "targets.add(v);"),
    ("addestramento: seme dell'inizializzazione ignorato", TRN, "final Random rnd = new Random(cfg.seed);\n\t\tfinal FloatNet net", "final Random rnd = new Random(1234);\n\t\tfinal FloatNet net"),
    ("ottimizzatore: passo con segno sbagliato (la perdita non cala)", TRN, "w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);", "w[i] += lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);"),
    ("ottimizzatore: gradiente non azzerato fra un batch e il successivo", TRN, "\t\t\t\tjava.util.Arrays.fill(g, 0.0);\n\t\t\t\tfor (int k = b; k < end; k++) {", "\t\t\t\tfor (int k = b; k < end; k++) {"),
], "data": [
    # ---- NnueDataset: invarianti, I/O, unione, split
    ("posizione: Re sul bordo accettato", NDS, "if (r == 0 || r == 8 || c == 0 || c == 8) {", "if (false) {"),
    ("posizione: Re assente accettato", NDS, "if (kings == 0) {", "if (false) {"),
    ("posizione: due Re accettati", NDS, "if (kings > 1) {", "if (false) {"),
    ("posizione: troppe pedine accettate", NDS, "if (white > MAX_WHITE || black > MAX_BLACK) {", "if (false) {"),
    ("posizione: pedina bianca sul trono accettata", NDS, "return \"pedina bianca sul trono\";", "break;"),
    ("posizione: codice sconosciuto (il 3 dei dati legacy) accettato", NDS, "return \"codice pedina \" + p + \" nella cella \" + sq + \" (ammessi 0, 1, 2, 4: il Re e' 4)\";", "break;"),
    ("campione: ruolo non verificato", NDS, "if (role != 0 && role != 1) {", "if (false) {"),
    ("campione: esito non canonico con outValid = 0 accettato", NDS, "if (outValid == 0 && outcome != 0) {", "if (false) {"),
    ("campione: valore > 1 accettato", NDS, "if (Float.isNaN(value) || value < -1f || value > 1f) {", "if (Float.isNaN(value) || value < -1f) {"),
    ("campione: valore non canonico con valValid = 0 accettato", NDS, "if (valValid == 0 && value != 0f) {", "if (false) {"),
    ("campione: toEnd senza esito valido accettato", NDS, "if (toEnd >= 0 && outValid == 0) {", "if (false) {"),
    ("Builder: valore non riportato a 0 quando non valido", NDS, "this.value[n] = valValid ? (float) value : 0f;", "this.value[n] = (float) value;"),
    ("unione: chiavi uguali di file diversi fuse", NDS, "final long key = ((long) part << 32) | (d.game[i] & 0xFFFFFFFFL);", "final long key = (d.game[i] & 0xFFFFFFFFL);"),
    ("subset: ruolo preso dall'indice sbagliato", NDS, "r[k] = role[i];", "r[k] = role[k];"),
    ("split: off-by-one nella percentuale", NDS, "return Math.floorMod(h, 100L) < pct;", "return Math.floorMod(h, 100L) <= pct;"),
    ("split: per campione invece che per partita", NDS, "if (isHeldOut(game[i], salt, pct)) {", "if (isHeldOut(i, salt, pct)) {"),
    ("salvataggio: esito e outValid scambiati", NDS, "\t\t\to.write(outcome);\n\t\t\to.write(outValid);", "\t\t\to.write(outValid);\n\t\t\to.write(outcome);"),
    ("caricamento: byte in piu' non rilevati (niente verifica del CRC)", NDS, "if (in.read() != -1) { // leggere fino alla fine fa verificare il CRC di gzip", "if (false) {"),
    ("caricamento: versione del formato non verificata", NDS, "if (!magic.equals(MAGIC)) {", "if (false) {"),
    ("caricamento: dimensione non limitata", NDS, "if (n < 0 || n > MAX_SAMPLES) {", "if (n < 0) {"),
    # ---- NnueDataImport
    ("coordinate: riga e colonna scambiate", NDI, "final int c = Character.toLowerCase(s.charAt(0)) - 'a', r = s.charAt(1) - '1';", "final int r = Character.toLowerCase(s.charAt(0)) - 'a', c = s.charAt(1) - '1';"),
    ("Re catturato: la cella del Re del server non e' controllata", NDI, "if (kings != 1 || at != kingBefore) {", "if (kings != 1) {"),
    ("Re catturato: server che toglie il Re non gestito", NDI, "return Arrays.equals(engine, b); // un server che toglie il Re dalla scacchiera: stessa rappresentazione del motore", "return false;"),
    ("mossa illegale applicata dal server scambiata per forfeit", NDI, "end = Arrays.equals(p.boards.get(k), p.boards.get(k - 1)) ? \"ILLEGAL\" : \"DIVERGENCE\";", "end = \"ILLEGAL\";"),
    ("vittoria su scacchiera non terminale accettata come esito", NDI, "} else if (last.equals(\"WW\") || last.equals(\"BW\")) {", "} else if (false) {"),
    ("patta del server ignorata", NDI, "} else if (tracker.isRepetitionDraw() || last.equals(\"D\")) {", "} else if (tracker.isRepetitionDraw()) {"),
    ("token D scambiato per divergenza", NDI, "(token.equals(\"D\") && (t == Turn.DRAW || t == Turn.WHITE || t == Turn.BLACK))", "(token.equals(\"D\") && t == Turn.DRAW)"),
    ("toEnd sbagliato di uno", NDI, "r.outValid ? r.applied - i : -1", "r.outValid ? r.applied - i - 1 : -1"),
    ("doppioni non scartati", NDI, "if (dedupe && !seen.add(moveKey(p))) {", "if (false) {"),
    ("posizioni senza etichetta tenute", NDI, "if (!r.outValid && !keepUnlabeled) {", "if (false) {"),
    ("mosse rifiutate non contate", NDI, "\t\t\t\tif (pendingSet) {\n\t\t\t\t\tp.rejectedMoves++;\n\t\t\t\t}\n", ""),
    ("Re dei dati legacy non convertito (3 -> 4)", NDI, "return code == 3 ? BoardConstants.KING : code;", "return code;"),
    ("DistillSet: outValid ignorato", NDI, "final boolean valid = outValid == null || outValid[i] == 1;", "final boolean valid = true;"),
    ("DistillSet: sorgente non mappata", NDI, "source != null && source[i] == 1 ? NnueDataset.SRC_SELFPLAY : NnueDataset.SRC_IMPORT", "NnueDataset.SRC_IMPORT"),
    # ---- NnueDataTool
    ("filter: l'esito richiesto non e' controllato", NDT, "(!needOutcome || d.outValid[i] == 1)", "true"),
    ("split: percentuale oltre 100 accettata", NDT, "if (train == null || held == null || pct < 0 || pct > 100) {", "if (train == null || held == null || pct < 0) {"),
], "selfplay": [
    ("scoreToValue: segno non invertito per il Nero", NSP, "final double white = whiteToMove ? moverScore : -moverScore;", "final double white = moverScore;"),
    ("scoreToValue: soglia delle vittorie forzate sbagliata", NSP, "if (white >= PROVEN_SCORE) {", "if (white >= 2 * PROVEN_SCORE) {"),
    ("scoreToValue: valore non limitato a [-1,1]", NSP, "return Math.max(-1.0, Math.min(1.0, white / scale));", "return white / scale;"),
    ("auto-gioco: l'esplorazione eps non gioca mosse casuali", NSP, "if (chosen == null || rnd.nextDouble() < cfg.eps) {", "if (chosen == null) {"),
    ("auto-gioco: nessuna apertura casuale", NSP, "if (ply < cfg.openRandom) {", "if (ply < 0) {"),
    ("auto-gioco: esito valido anche per partite troncate", NSP, "g.outValid = !GameTracker.LABEL_MAXPLIES.equals(g.result.label);", "g.outValid = true;"),
    ("dataset: toEnd sbagliato di uno", NSP, "g.applied - ply,", "g.applied - ply - 1,"),
    ("dataset: tutte le partite con la stessa chiave", NSP, "g.values.get(k), true, i, ply,", "g.values.get(k), true, 0, ply,"),
    ("generate: tutte le partite con lo stesso seme", NSP, "final long gs = master.nextLong();", "final long gs = cfg.seed;"),
    ("rietichettatura: lato al tratto invertito", NRL, "d.role[i] == 0, scale);", "d.role[i] == 1, scale);"),
    ("rietichettatura: onlyMissing ignorato", NRL, "if (onlyMissing && d.valValid[i] == 1) {", "if (false) {"),
], "trainer": [
    ("bersaglio: lambda scambiato fra valore ed esito", NTR, "return cfg.lambda * d.value[i] + (1.0 - cfg.lambda) * o;", "return (1.0 - cfg.lambda) * d.value[i] + cfg.lambda * o;"),
    ("bersaglio: sconto costante invece che per distanza dalla fine", NTR, "o *= Math.pow(cfg.gamma, d.toEnd[i]);", "o *= cfg.gamma;"),
    ("bersaglio: sconto applicato anche con distanza ignota", NTR, "if (cfg.gamma < 1.0 && d.toEnd[i] >= 0) {", "if (cfg.gamma < 1.0) {"),
    ("usable: peso di sorgente 0 non esclude", NTR, "cfg.sourceWeight[d.source[i]] > 0)", "cfg.sourceWeight[d.source[i]] >= 0)"),
    ("addestramento: la rete di partenza non e' una candidata (puo' regredire)", NTR, "double bestVal = initial, lastTrain = 0;\n\t\tint bestEpoch = 0, run = 0;", "double bestVal = Double.MAX_VALUE, lastTrain = 0;\n\t\tint bestEpoch = -1, run = 0;"),
    ("addestramento: la copia della migliore epoca e' un alias", NTR, "bestEpoch = e + 1;\n\t\t\t\tbest = net.p.clone();", "bestEpoch = e + 1;\n\t\t\t\tbest = net.p;"),
    ("addestramento: nessun arresto anticipato", NTR, "if (e + 1 - bestEpoch >= cfg.patience) {", "if (false) {"),
    ("addestramento: simmetrie non applicate", NTR, "BoardSymmetry.MAP[cfg.augment ? rnd.nextInt(BoardSymmetry.COUNT) : 0]", "BoardSymmetry.MAP[0]"),
    ("addestramento: peso di sorgente ignorato nel gradiente", NTR, "2 * w * err * (1 - y * y)", "2 * err * (1 - y * y)"),
    ("addestramento: validazione esterna col lambda dell'addestramento", NTR, "final Config xcfg = cfg.extLambda >= 0 ? cfg.withLambda(cfg.extLambda) : cfg;", "final Config xcfg = cfg;"),
    ("addestramento: maxSamples ignorato", NTR, "order = Arrays.copyOf(order, cfg.maxSamples);", "order = order;"),
    ("passo dell'ottimizzatore: pesi non saturati", NTR, "\t\tadam.step(net.p, grad, lr, gradScale);\n\t\tclampToQuantRange(net);", "\t\tadam.step(net.p, grad, lr, gradScale);"),
    ("saturazione: bias non limitati", NTR, "net.p[b2] = Math.max(-BIAS_LIMIT, Math.min(BIAS_LIMIT, net.p[b2]));", ""),
    ("metriche: segno dell'esito confrontato al contrario", NTR, "if ((y > 0) == (o > 0)) {", "if ((y > 0) == (o < 0)) {"),
    ("metriche: linea di base 'predire 0' sbagliata", NTR, "se0 += o * o;", "se0 += o;"),
    ("fromQuantized: bias dello strato 1 con la scala sbagliata", TRN, "f.p[f.offL1B(r) + i] = b1[i] / sb;", "f.p[f.offL1B(r) + i] = b1[i] / NnueNetwork.QA;"),
    ("fromQuantized: ftB con la scala sbagliata", TRN, "f.p[f.oFtB + j] = (double) ftB[j] / NnueNetwork.QA;", "f.p[f.oFtB + j] = (double) ftB[j] / NnueNetwork.QW;"),
    ("forward: la permutazione di simmetria e' ignorata", TRN, "NnueNetwork.featureIndex(piece, map == null ? sq : map[sq])", "NnueNetwork.featureIndex(piece, sq)"),
]}


def main():
    if not os.path.isdir("src") or not os.path.isdir(BUILD):
        sys.exit(f"lancia dalla cartella Tablut (serve src/ e {BUILD}/ gia' compilata; BUILD=cartella per cambiarla)")
    suites = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(M) if suites == "all" else [suites]
    lo = int(sys.argv[2]) if len(sys.argv) > 3 else 0
    hi_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    total_killed, total, survivors = 0, 0, []
    for suite in names:
        muts = M[suite]
        hi = hi_arg if hi_arg is not None else len(muts)
        killed = 0
        for idx in range(lo, min(hi, len(muts))):
            desc, rel, old, new = muts[idx]
            base = os.path.basename(rel)
            work = f"/tmp/mut_nnue_{suite}_{idx}"
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work + "/out")
            src = open(ROOT + rel, encoding="iso-8859-1").read()
            n = src.count(old)
            if n != 1:
                print(f"[{suite} {idx:2d}] ERRORE: frammento da mutare trovato {n} volte (ne serve 1): {desc}")
                survivors.append(desc + f" (frammento trovato {n} volte)")
                total += 1
                continue
            open(f"{work}/{base}", "w", encoding="iso-8859-1").write(src.replace(old, new, 1))
            c = subprocess.run(["javac", "-encoding", "iso-8859-1", "--release", "8", "-nowarn", "-d", work + "/out", "-cp", CP, f"{work}/{base}"], capture_output=True, text=True)
            total += 1
            if c.returncode != 0:
                print(f"[{suite} {idx:2d}] il mutante non compila: {desc}\n{c.stderr[:300]}")
                survivors.append(desc + " (non compila)")
                continue
            try:
                r = subprocess.run(["java", "-Xss4m", "-Xmx2g", "-cp", f"{work}/out:{CP}", CHECKS[suite]], capture_output=True, text=True, timeout=TIMEOUT[suite])
                failed = r.returncode != 0
                first = next((l for l in r.stdout.splitlines() if l.startswith("FAIL")), "") or (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
            except subprocess.TimeoutExpired:
                failed, first = True, "(timeout: considerato rilevato)"
            print(f"[{suite} {idx:2d}] {'RILEVATO ' if failed else 'SOPRAVVISSUTO'} - {desc}" + (f"\n        -> {first[:150]}" if failed else ""))
            if failed:
                killed += 1
            else:
                survivors.append(desc)
            shutil.rmtree(work, ignore_errors=True)
        print(f"--- {suite}: {killed} rilevati")
        total_killed += killed
    print(f"\n{total_killed}/{total} mutanti rilevati")
    if survivors:
        print("NON rilevati (buchi del test):")
        for s in survivors:
            print("  -", s)
        sys.exit(1)


main()
