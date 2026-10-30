#!/usr/bin/env python3
"""
Mutation test di DistillCheck (feature "fine-tuning verso il valore di ricerca"): inietta UN bug alla volta in una COPIA del
sorgente, ricompila solo quel file in una cartella separata messa PRIMA del classpath, lancia il controllo indicato (di norma la
sezione di DistillCheck che deve accorgersene) e si aspetta che FALLISCA. Un mutante che sopravvive e' un buco del test.
Uso (dalla cartella Tablut, con le classi gia' compilate in build/ o in MUT_CP):
    python3 tools/mutation_distill.py [indiceIniziale indiceFinale]
Variabile d'ambiente MUT_CP = directory delle classi compilate (default: build).
Mutanti EQUIVALENTI o non rilevabili per costruzione sono elencati in fondo come commento, con il motivo.
"""
import os, shutil, subprocess, sys

ROOT = "src/it/unibo/ai/didattica/competition/tablut/"
LIB = "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
CP = os.environ.get("MUT_CP", "build") + ":" + LIB
PKG = "it.unibo.ai.didattica.competition.tablut."
DC = PKG + "training.DistillCheck"
CNNCHECK = PKG + "selftest.ConvNetCheck"

# (descrizione, file relativo a ROOT, vecchio, nuovo, classe di controllo, argomenti)
M = [
 # ---------------- SearchLabeler
 ("etichetta: nessun cambio di segno per il Nero", "training/SearchLabeler.java", "return whiteToMove ? sideToMoveScore : -sideToMoveScore;", "return sideToMoveScore;", DC, ["conv", "labels"]),
 ("etichetta: chi muove forzato a Bianco", "training/SearchLabeler.java", "boolean whiteToMove = root.isWhiteToMove();", "boolean whiteToMove = true;", DC, ["labels"]),
 ("etichetta: clip a +-1.5 invece di +-0.98", "training/SearchLabeler.java", "Math.min(CLIP, whiteScore / SCALE)", "Math.min(1.5, whiteScore / SCALE)", DC, ["conv"]),
 ("etichetta: scala 100 invece di 1000", "training/SearchLabeler.java", "SCALE = 1000.0;", "SCALE = 100.0;", DC, ["conv"]),
 ("terminali: chi muove senza mosse non e' terminale", "training/SearchLabeler.java", "s.winner != FastState.WINNER_NONE || !s.hasLegalMove()", "s.winner != FastState.WINNER_NONE", DC, ["data", "repro"]),
 ("codici->stato: tocca sempre al Bianco", "training/SearchLabeler.java", "role == 0 ? Turn.WHITE : Turn.BLACK", "Turn.WHITE", DC, ["data"]),
 ("codici->stato: nero letto come bianco", "training/SearchLabeler.java", "code == 2 ? Pawn.BLACK", "code == 2 ? Pawn.WHITE", DC, ["data", "repro"]),
 ("labelAll: risultato scritto all'indice di posizione invece che alla posizione nell'elenco", "training/SearchLabeler.java", "out.y[k] = r.y;", "out.y[i] = r.y;", DC, ["repro"]),
 # ---------------- DistillSet
 ("split: 20% invece di pct%", "training/DistillSet.java", "Long.remainderUnsigned(x, 100) < pct", "Long.remainderUnsigned(x, 100) < pct * 2", DC, ["data"]),
 ("split: per posizione invece che per partita", "training/DistillSet.java", "h[i] = isHeldOut(m[3], salt, heldPct);", "h[i] = isHeldOut(i, salt, heldPct);", DC, ["data"]),
 ("dataset: i terminali non vengono esclusi", "training/DistillSet.java", "if (SearchLabeler.isTerminal(SearchLabeler.fastStateFromCodes(d.cells, i * 81, d.role[i]))) {", "if (false) {", DC, ["data"]),
 ("mix: tutta la richiesta dal dataset", "training/DistillSet.java", "int takeA = Math.min(ia.length, want / 2), takeB", "int takeA = Math.min(ia.length, want), takeB", DC, ["data"]),
 ("mix: nessun riempimento dalla sorgente abbondante", "training/DistillSet.java", "int extraA = Math.min(ia.length - takeA, Math.max(0, missing));", "int extraA = 0;", DC, ["data"]),
 ("campionamento: flag held-out non conservato", "training/DistillSet.java", "held[k++] = a.heldOut[i];", "held[k++] = !a.heldOut[i];", DC, ["data"]),
 ("engine: esito con segno invertito", "training/DistillSet.java", "roles.get(i), (int) end.outcome, valid, key, SRC_ENGINE);", "roles.get(i), -(int) end.outcome, valid, key, SRC_ENGINE);", DC, ["data"]),
 ("engine: ruolo registrato sempre 0", "training/DistillSet.java", "roles.add(NeuralEvaluator.roleOf(s));", "roles.add(0);", DC, ["data"]),
 # ---------------- CnnDistiller: loss
 ("loss: pesi (1-lambda) e lambda scambiati", "training/CnnDistiller.java", "return (1 - lambda) * ls + lambda * lo;", "return lambda * ls + (1 - lambda) * lo;", DC, ["loss"]),
 ("loss: la ricerca non e' piu' nel dominio logit", "training/CnnDistiller.java", "kind == SearchLoss.LOGIT ? sq(z - logitTarget(y)) : sq(th - y);", "sq(th - y);", DC, ["loss"]),
 ("loss: atanh senza il fattore 1/2", "training/CnnDistiller.java", "return 0.5 * Math.log((1 + y) / (1 - y));", "return Math.log((1 + y) / (1 - y));", DC, ["conv", "loss"]),
 ("loss: termine d'esito sempre attivo", "training/CnnDistiller.java", "final double lo = outValid ? sq(th - o) : 0.0;", "final double lo = sq(th - o);", DC, ["loss"]),
 ("gradiente: ramo logit usa il gradiente su tanh", "training/CnnDistiller.java", "kind == SearchLoss.LOGIT ? 2 * (z - logitTarget(y)) : 2 * (th - y) * dth;", "2 * (th - y) * dth;", DC, ["loss"]),
 ("gradiente: esito senza il fattore (1 - tanh^2)", "training/CnnDistiller.java", "final double dout = outValid ? 2 * (th - o) * dth : 0.0;", "final double dout = outValid ? 2 * (th - o) : 0.0;", DC, ["loss"]),
 ("gradiente: lambda ignorato nel gradiente", "training/CnnDistiller.java", "return (1 - lambda) * ds + lambda * dout;", "return ds + dout;", DC, ["loss"]),
 ("ancoraggio: senza il fattore batch", "training/CnnDistiller.java", "final double c = 2.0 * mu * batchSize;", "final double c = 2.0 * mu;", DC, ["loss"]),
 ("ancoraggio: segno invertito (allontana dal riferimento)", "training/CnnDistiller.java", "g[i] += (float) (c * (w[i] - ref[i]));", "g[i] += (float) (c * (ref[i] - w[i]));", DC, ["loss"]),
 ("ancoraggio: penalita' senza il quadrato", "training/CnnDistiller.java", "double d = w[i] - ref[i];\n\t\t\ts += d * d;", "double d = w[i] - ref[i];\n\t\t\ts += Math.abs(d);", DC, ["loss"]),
 # ---------------- CnnDistiller: minibatch / ciclo
 ("minibatch: backward sull'uscita tanh invece che sul logit", "training/CnnDistiller.java", "net.backwardLogit(ws, sampleDz(z, y[i], d.outcome[i], ov, kind, lambda), grad);", "net.backward(ws, sampleDz(z, y[i], d.outcome[i], ov, kind, lambda), grad);", DC, ["grad"]),
 ("minibatch: augmentation ignorata", "training/CnnDistiller.java", "augment ? CnnPretrainer.symOf(seed, epoch, i) : 0", "0", DC, ["sym", "grad"]),
 ("minibatch: colonne di feature spostate", "training/CnnDistiller.java", "fbuf[j] = d.feats[i * NF + cols[j]];\n\t\t\t\t}\n\t\t\t\tnet.forward(ws, fbuf, d.role[i]);\n\t\t\t\tfinal double z = ws.logit();\n\t\t\t\tfinal boolean ov", "fbuf[j] = d.feats[i * NF + (cols[j] + 1) % NF];\n\t\t\t\t}\n\t\t\t\tnet.forward(ws, fbuf, d.role[i]);\n\t\t\t\tfinal double z = ws.logit();\n\t\t\t\tfinal boolean ov", DC, ["grad"]),
 ("minibatch: ruolo del campione ignorato", "training/CnnDistiller.java", "net.forward(ws, fbuf, d.role[i]);\n\t\t\t\tfinal double z = ws.logit();\n\t\t\t\tfinal boolean ov", "net.forward(ws, fbuf, 0);\n\t\t\t\tfinal double z = ws.logit();\n\t\t\t\tfinal boolean ov", DC, ["grad"]),
 ("minibatch: l'ultimo campione di ogni fetta e' perso", "training/CnnDistiller.java", "b = from + (int) ((long) len * (t + 1) / T);", "b = from + (int) ((long) len * (t + 1) / T) - 1;", DC, ["grad"]),
 ("train: ancoraggio mai applicato", "training/CnnDistiller.java", "addAnchorGradient(net.params(), anchorRef, anchorMu, batch, g);", "/* nessun ancoraggio */", DC, ["train"]),
 ("train: anchorTo=init ignorato", "training/CnnDistiller.java", "float[] ref = P.anchorToInit ? w0 : cur.params().clone();", "float[] ref = cur.params().clone();", DC, ["train"]),
 ("ciclo: la rete di partenza viene modificata in place", "training/CnnDistiller.java", "ConvNet cur = baseline.net.copy();", "ConvNet cur = baseline.net;", DC, ["train"]),
 ("ciclo: l'iterazione e' sempre tenuta", "training/CnnDistiller.java", "st.kept = st.endHeld.tanhMse < st.startHeld.tanhMse;", "st.kept = true;", DC, ["train"]),
 ("ciclo: riferimento della porta (i) preso dall'ultima iterazione", "training/CnnDistiller.java", "if (it == 0) {\n\t\t\t\tres.baselineTargets = y.clone();", "if (it >= 0) {\n\t\t\t\tres.baselineTargets = y.clone();", DC, ["train"]),
 ("ciclo: stessa profondita' per tutte le iterazioni", "training/CnnDistiller.java", "return depths[Math.min(iter, depths.length - 1)];", "return depths[0];", DC, ["train"]),
 ("porta (i): passa anche a parita'", "training/CnnDistiller.java", "g.pass = g.n > 0 && g.candMse < g.baseMse;", "g.pass = g.n > 0 && g.candMse <= g.baseMse;", DC, ["train"]),
 ("porta (i): guadagno con segno invertito", "training/CnnDistiller.java", "double diff = eb[k] - ec[k];", "double diff = ec[k] - eb[k];", DC, ["train"]),
 ("deriva: normalizzata male", "training/CnnDistiller.java", "return Math.sqrt(a / Math.max(1e-30, b));", "return Math.sqrt(a);", DC, ["train"]),
 # ---------------- ConvNet (logit)
 ("logit: ultimo strato senza bias", "aiplayer/ConvNet.java", "return p[bOff] + (s0 + s1);", "return (s0 + s1);", DC, ["logit"]),
 ("logit: coda dispari ignorata", "aiplayer/ConvNet.java", "if (i < in) {\n\t\t\ts0 += p[wOff + i] * x[i];\n\t\t}\n\t\treturn p[bOff]", "if (false) {\n\t\t\ts0 += p[wOff + i] * x[i];\n\t\t}\n\t\treturn p[bOff]", DC, ["logit"]),
 ("logit: secondo accumulatore sbagliato", "aiplayer/ConvNet.java", "s1 += p[wOff + i + 1] * x[i + 1];\n\t\t}\n\t\tif (i < in) {\n\t\t\ts0 += p[wOff + i] * x[i];\n\t\t}\n\t\treturn p[bOff]", "s1 += p[wOff + i + 1] * x[i];\n\t\t}\n\t\tif (i < in) {\n\t\t\ts0 += p[wOff + i] * x[i];\n\t\t}\n\t\treturn p[bOff]", DC, ["logit"]),
 ("logit: logit non memorizzato", "aiplayer/ConvNet.java", "ws.logit = z;", "ws.logit = 0f;", DC, ["logit"]),
 ("logit: uscita senza tanh", "aiplayer/ConvNet.java", "ws.hd[role][j][0] = (float) Math.tanh(z);", "ws.hd[role][j][0] = z;", DC, ["logit"]),
 ("backwardLogit: gradiente dei pesi finali senza dz", "aiplayer/ConvNet.java", "grad[wOff + i] += dz * x[i];\n\t\t\tdx[i] += p[wOff + i] * dz;", "grad[wOff + i] += x[i];\n\t\t\tdx[i] += p[wOff + i] * dz;", DC, ["logit", "grad"]),
 ("backwardLogit: gradiente verso l'ingresso senza dz", "aiplayer/ConvNet.java", "dx[i] += p[wOff + i] * dz;\n\t\t}\n\t}\n\n\tprivate static float act", "dx[i] += p[wOff + i];\n\t\t}\n\t}\n\n\tprivate static float act", DC, ["grad"]),
 ("backwardLogit: bias finale non accumulato", "aiplayer/ConvNet.java", "grad[bOff] += dz;\n\t\tfor (int i = 0; i < in; i++) {\n\t\t\tgrad[wOff + i] += dz * x[i];", "grad[bOff] += 0;\n\t\tfor (int i = 0; i < in; i++) {\n\t\t\tgrad[wOff + i] += dz * x[i];", DC, ["logit"]),
 ("backwardLogit: la modalita' logit non viene usata", "aiplayer/ConvNet.java", "if (logitMode && j == H - 1) {", "if (false) {", DC, ["logit", "grad"]),
 ("backward (uscita) e' diventato backwardLogit", "aiplayer/ConvNet.java", "backwardImpl(ws, outGrad, false, grad);", "backwardImpl(ws, outGrad, true, grad);", CNNCHECK, []),
 # ---------------- CnnPretrainer init=
 ("init=: i pesi non vengono caricati (rete casuale)", "training/CnnPretrainer.java", "ConvNet net = initEv != null ? initEv.net.copy() : new ConvNet(cfg, seed);", "ConvNet net = new ConvNet(cfg, seed);", DC, ["cli"]),
 ("init=: gruppi di feature del file ignorati", "training/CnnPretrainer.java", "groups = initEv.featureGroups;", "", DC, ["cli"]),
 # ---------------- HangingGate
 ("hanging: passa solo se TUTTE e due le misure non peggiorano -> basta una", "training/HangingGate.java", "return new Verdict(true, shareOk && lossOk, t);", "return new Verdict(true, shareOk || lossOk, t);", DC, ["gates"]),
 ("hanging: a parita' peggiora", "training/HangingGate.java", "boolean shareOk = cand.hangAnyShare() <= base.hangAnyShare() + tol;", "boolean shareOk = cand.hangAnyShare() < base.hangAnyShare() + tol;", DC, ["gates"]),
 ("hanging: tolleranza ignorata", "training/HangingGate.java", "boolean lossOk = cand.whiteLoss3() <= base.whiteLoss3() + tol;", "boolean lossOk = cand.whiteLoss3() <= base.whiteLoss3();", DC, ["gates"]),
 ("hanging: nessun minimo di campione", "training/HangingGate.java", "if (base.pawnMoves < minPawnMoves || cand.pawnMoves < minPawnMoves || base.events3 == 0 || cand.events3 == 0) {", "if (false) {", DC, ["gates"]),
 ("hanging: casella con righe e colonne scambiate", "training/HangingGate.java", "return new int[] { t.charAt(1) - '1', t.charAt(0) - 'a' };\n\t}\n\n\tstatic int[] cellFrom", "return new int[] { t.charAt(0) - 'a', t.charAt(1) - '1' };\n\t}\n\n\tstatic int[] cellFrom", DC, ["gates"]),
 ("hanging: la cattura piu' grande non viene tenuta", "training/HangingGate.java", "if (loss > maxLoss) {", "if (loss < maxLoss) {", DC, ["gates"]),
 ("hanging: materiale perso con segno invertito", "training/HangingGate.java", "st.whiteLost3 += hist.get(p)[0] - hist.get(p + 3)[0];", "st.whiteLost3 += hist.get(p + 3)[0] - hist.get(p)[0];", DC, ["gates"]),
 ("hanging: la pedina mossa non e' controllata", "training/HangingGate.java", "if (loss > 0 && dr >= 0 && s2.getBoard()[dr][dc] == Pawn.EMPTY) {", "if (false) {", DC, ["gates"]),
 # ---------------- DistillAcceptance
 ("arena: basta una delle due condizioni", "training/DistillAcceptance.java", "boolean pass = score >= 0.5 && lo >= 0.5 - margin;", "boolean pass = score >= 0.5 || lo >= 0.5 - margin;", DC, ["gates"]),
 ("arena: le patte non contano", "training/DistillAcceptance.java", "double score = (wdl[0] + 0.5 * wdl[1]) / n,", "double score = (wdl[0] + 0.0 * wdl[1]) / n,", DC, ["gates"]),
 ("arena: margine ignorato", "training/DistillAcceptance.java", "lo >= 0.5 - margin;", "lo >= 0.5;", DC, ["gates"]),
 ("adozione: la porta arena a profondita' fissa non conta", "training/DistillAcceptance.java", "&& arenaTimeVerified && arenaTimePass && arenaDepthVerified && arenaDepthPass;", "&& arenaTimeVerified && arenaTimePass;", DC, ["gates"]),
 ("adozione: una porta non verificata vale come superata", "training/DistillAcceptance.java", "return gate1 && gate2Verified && gate2Pass", "return gate1 && gate2Pass", DC, ["gates"]),
 ("adozione: nessuna copia di sicurezza", "training/DistillAcceptance.java", "Files.copy(t.toPath(), new File(target + \".bak\").toPath(), StandardCopyOption.REPLACE_EXISTING);", "", DC, ["gates"]),
 ("verdetto: la porta (i) non pesa sul codice di uscita", "training/DistillAcceptance.java", "g1 = g.pass;", "g1 = true;", DC, ["gates"]),
 ("verdetto: nessuna porta fallita -> codice 0 comunque", "training/DistillAcceptance.java", "return any_fail ? 1 : 4;", "return 0;", DC, ["gates"]),
]

# Mutanti EQUIVALENTI o non rilevabili per costruzione (non inclusi):
#  - togliere il controllo "ricerca incompleta" in SearchLabeler.label: il tetto di un'ora per posizione non scatta mai nei test;
#  - setSafetyMarginMillis(0) nel labeler: con budget di un'ora il margine non cambia nulla;
#  - Negamax a 2 thread nel labeler: la ricerca resta esatta e il valore alla radice non cambia (si perde solo la riproducibilita' dei nodi);
#  - "ply >= 4" -> ">= 0" nelle partite dell'engine: cambia solo quali posizioni si registrano (nessuna proprieta' testabile oltre al conteggio);
#  - outValid per partite troncate (MAXPLIES): non succede nelle partite brevi dei test.

def main():
    if not os.path.isdir("src"):
        sys.exit("lancia dalla cartella Tablut")
    lo = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    hi = min(int(sys.argv[2]), len(M)) if len(sys.argv) > 2 else len(M)
    survivors, killed = [], 0
    for idx in range(lo, hi):
        desc, rel, old, new, check, args = M[idx]
        base = os.path.basename(rel)
        work = f"/tmp/mutd_{idx}"
        shutil.rmtree(work, ignore_errors=True); os.makedirs(work + "/out")
        src = open(ROOT + rel, encoding="iso-8859-1").read()
        if old not in src:
            print(f"[{idx:2d}] ERRORE: frammento da mutare non trovato: {desc}"); survivors.append(desc + " (frammento assente)"); continue
        open(f"{work}/{base}", "w", encoding="iso-8859-1").write(src.replace(old, new, 1))
        c = subprocess.run(["javac", "-encoding", "iso-8859-1", "-nowarn", "-d", work + "/out", "-cp", CP, f"{work}/{base}"], capture_output=True, text=True)
        if c.returncode != 0:
            print(f"[{idx:2d}] il mutante non compila: {desc}\n{c.stderr[:300]}"); survivors.append(desc + " (non compila)"); continue
        try:
            r = subprocess.run(["java", "-cp", f"{work}/out:{CP}", check] + args, capture_output=True, text=True, timeout=400)
            failed = r.returncode != 0
            first = next((l for l in r.stdout.splitlines() if l.startswith("FAIL")), "") or (r.stderr.splitlines()[-1] if r.stderr else "")
        except subprocess.TimeoutExpired:
            failed, first = True, "(timeout: considerato rilevato)"
        print(f"[{idx:2d}] {'RILEVATO ' if failed else 'SOPRAVVISSUTO'} - {desc}" + (f"\n        -> {first[:140]}" if failed else ""), flush=True)
        if failed: killed += 1
        else: survivors.append(desc)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{killed}/{hi - lo} mutanti rilevati")
    if survivors:
        print("NON rilevati (buchi del test):"); [print("  -", s) for s in survivors]; sys.exit(1)

main()
