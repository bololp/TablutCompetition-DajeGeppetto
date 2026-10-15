#!/usr/bin/env python3
"""
Mutation test di ConvNetCheck: inietta UN bug alla volta in una COPIA di ConvNet.java / BoardSymmetry.java, ricompila in una
cartella separata messa PRIMA di 'build' nel classpath, lancia ConvNetCheck e si aspetta che FALLISCA.
Un mutante che il test non rileva e' un buco del test, non del mutante.
Uso (dalla cartella Tablut):  python3 tools/mutation_cnn.py [model|training] [indiceIniziale indiceFinale]
  model    -> mutanti di ConvNet/BoardSymmetry, verificati da ConvNetCheck (default)
  training -> mutanti del trainer/pretrainer/Adam, verificati da CnnTrainingCheck (piu' lento: ~40 s per mutante)
"""
import os, shutil, subprocess, sys

ROOT = "src/it/unibo/ai/didattica/competition/tablut/"
PKG = ROOT + "aiplayer/"
CP = "build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
CHECK_MODEL = "it.unibo.ai.didattica.competition.tablut.selftest.ConvNetCheck"
CHECK_TRAIN = "it.unibo.ai.didattica.competition.tablut.training.CnnTrainingCheck"

M = [  # (descrizione, file, vecchio, nuovo)
 ("tap OFF scambiato (-PW+1 <-> -PW-1) nello scatter del livello 0", "ConvNet.java", "{ -PW - 1, -PW, -PW + 1, -1, 0, 1, PW - 1, PW, PW + 1 }", "{ -PW + 1, -PW, -PW - 1, -1, 0, 1, PW - 1, PW, PW + 1 }"),
 ("coda del kernel conv (canali dispari): ultimo tap sbagliato", "ConvNet.java", "+ a8 * in[i + PW + 1];", "+ a8 * in[i + PW];"),
 ("kernel conv a 2 canali: tap x5 della seconda uscita sbagliato", "ConvNet.java", "(b4 * x4 + b5 * x5)", "(b4 * x4 + b5 * x4)"),
 ("pesi trasposti senza il ribaltamento (8-k)", "ConvNet.java", "wt[dstw + 8 - k] = p[srcw + k];", "wt[dstw + k] = p[srcw + k];"),
 ("derivata leaky = 0 sul ramo negativo (conv)", "ConvNet.java", "final float v = dz[ob + i] * (act[ob + i] > 0 ? 1f : LEAK);", "final float v = dz[ob + i] * (act[ob + i] > 0 ? 1f : 0f);"),
 ("bordo non azzerato dopo i livelli conv (forward)", "ConvNet.java", "\t\t\t\tleakyRange(out, o * PS);\n\t\t\t}\n\t\t\tzeroBorders(out, cout);", "\t\t\t\tleakyRange(out, o * PS);\n\t\t\t}\n"),
 ("bordo non azzerato dopo il livello 0 (forward)", "ConvNet.java", "\t\tzeroBorders(a0, c0);\n\n\t\t// ---- livelli 1..L-1", "\n\n\t\t// ---- livelli 1..L-1"),
 ("bias dei conv non accumulato nel gradiente", "ConvNet.java", "grad[convB[l] + o] += db;", "grad[convB[l] + o] += 0;"),
 ("gradiente dx del denso senza il termine dz", "ConvNet.java", "dx[i] += p[wb + i] * dz;", "dx[i] += p[wb + i];"),
 ("backward usa sempre la head 0", "ConvNet.java", "denseBack(p, headW[role][j], headB[role][j], x, hin,", "denseBack(p, headW[0][j], headB[0][j], x, hin,"),
 ("forward usa sempre la head 0", "ConvNet.java", "dense(p, headW[role][j], headB[role][j], x, in, ws.hd[role][j], hout, j == H - 1);", "dense(p, headW[0][j], headB[0][j], x, in, ws.hd[role][j], hout, j == H - 1);"),
 ("flatten con colonne invertite (solo forward)", "ConvNet.java", "ws.flat[d + c] = src[s + c];", "ws.flat[d + c] = src[s + (N - 1 - c)];"),
 ("outGrad ignorato (sempre 1)", "ConvNet.java", "ws.dhd[role][H - 1][0] = (float) outGrad;", "ws.dhd[role][H - 1][0] = 1f;"),
 ("gradiente dei pesi conv: ultimo tap sbagliato", "ConvNet.java", "s8 += d * in[i + PW + 1];", "s8 += d * in[i + PW];"),
 ("livello 0: mappa di bias non scritta sull'ultima riga (stato residuo/valore errato)", "ConvNet.java", "for (int r = 0; r < N; r++) {\n\t\t\t\tfinal int dst = zb + (r + 1) * PW + 1;", "for (int r = 0; r < N - 1; r++) {\n\t\t\t\tfinal int dst = zb + (r + 1) * PW + 1;"),
 # NOTA: togliere "Arrays.fill(a0, ..., 0f)" e' un mutante EQUIVALENTE (verificato a mano): l'interno e' sovrascritto dalla mappa e
 # il bordo e' gia' zero all'ingresso (zeroBorders a fine forward); il fill e' solo una difesa ridondante.
 ("scatter livello 0 srotolato: due tap scambiati", "ConvNet.java", "a0[q + PW + 1] += p[wb];\n\t\t\t\ta0[q + PW] += p[wb + 1];", "a0[q + PW + 1] += p[wb + 1];\n\t\t\t\ta0[q + PW] += p[wb];"),
 ("denso a blocchi: quarta uscita legge la riga della terza", "ConvNet.java", "a3 += p[w3 + i] * xa;\n\t\t\t\tb3 += p[w3 + i + 1] * xb;", "a3 += p[w2 + i] * xa;\n\t\t\t\tb3 += p[w3 + i + 1] * xb;"),
 ("denso a blocchi: coda dispari ignora l'ultimo ingresso", "ConvNet.java", "if (i < in) {\n\t\t\t\tfinal float xa = x[i];\n\t\t\t\ta0 += p[w0 + i] * xa;", "if (false) {\n\t\t\t\tfinal float xa = x[i];\n\t\t\t\ta0 += p[w0 + i] * xa;"),
 ("denso scalare (uscite residue): secondo accumulatore sbagliato", "ConvNet.java", "s1 += p[wb + i + 1] * x[i + 1];", "s1 += p[wb + i + 1] * x[i];"),
 ("denso: bias non sommato", "ConvNet.java", "y[o] = act(p[bOff + o] + (a0 + b0), tanh);", "y[o] = act((a0 + b0), tanh);"),
 ("livello 0: gradiente della mappa di bias sulla riga sbagliata", "ConvNet.java", "grad[bb + r * N + c] += dz0[s + c];", "grad[bb + c * N + r] += dz0[s + c];"),
 ("conv 1x1: gradiente verso l'ingresso senza il peso", "ConvNet.java", "dIn[cb + i] += w * dz;", "dIn[cb + i] += dz;"),
 ("layout: bias del livello 0 sovrapposto ai pesi", "ConvNet.java", "off += cfg.conv[0] * PIECE_TYPES * 9;\n\t\tconvB[0] = off;", "off += cfg.conv[0] * PIECE_TYPES * 9 - 1;\n\t\tconvB[0] = off;"),
 ("salvataggio: parametri in float troncati a 6 cifre", "ConvNet.java", "sb.append(Float.toString(P[i]));", "sb.append(String.format(java.util.Locale.ROOT, \"%.6g\", P[i]));"),
 ("copy() condivide l'array dei parametri", "ConvNet.java", "this.P = params.clone();", "this.P = params;"),
 ("simmetria rot90 uguale alla trasposta (non distinte)", "BoardSymmetry.java", "MAP[1][i] = c * N + (N - 1 - r);", "MAP[1][i] = c * N + r;"),
 ("simmetria riflessione sbagliata (non e' un'isometria della scacchiera)", "BoardSymmetry.java", "MAP[4][i] = r * N + (N - 1 - c);", "MAP[4][i] = r * N + (N - 2 - c + (c == N - 1 ? N : 0));"),
]

MT = [  # suite "training": (descrizione, file relativo a ROOT, vecchio, nuovo)
 ("TD: traccia di eleggibilita' mai decaduta", "training/CnnSelfPlayTrainer.java", "trace[k] *= lam;", "trace[k] *= 1f;"),
 ("TD: segno dell'errore invertito", "training/CnnSelfPlayTrainer.java", "final double delta = target - v;", "final double delta = v - target;"),
 ("TD: bersaglio = valore della stessa posizione (non la successiva)", "training/CnnSelfPlayTrainer.java", "ws.setCodes(traj.cells.get(i + 1), 0);\n\t\t\t\ttarget = net.forward(ws, traj.feats.get(i + 1), traj.roles[i + 1]);", "ws.setCodes(traj.cells.get(i), 0);\n\t\t\t\ttarget = net.forward(ws, traj.feats.get(i), traj.roles[i]);"),
 ("TD: passo senza il fattore delta", "training/CnnSelfPlayTrainer.java", "net.addScaled(trace, alpha * delta);", "net.addScaled(trace, alpha);"),
 ("TD: ruolo del bersaglio sbagliato (usa quello della posizione corrente)", "training/CnnSelfPlayTrainer.java", "target = net.forward(ws, traj.feats.get(i + 1), traj.roles[i + 1]);", "target = net.forward(ws, traj.feats.get(i + 1), traj.roles[i]);"),
 ("TD: esito finale con segno invertito", "training/CnnSelfPlayTrainer.java", "target = traj.outcome;", "target = -traj.outcome;"),
 ("self-play: feature a mano non registrate (zeri)", "training/CnnSelfPlayTrainer.java", "mine.extractFeatures(s, fws, full, f);", "// feature saltate"),
 ("self-play: ruolo registrato sbagliato", "training/CnnSelfPlayTrainer.java", "roleList.add(NeuralEvaluator.roleOf(s));", "roleList.add(0);"),
 ("minibatch: la perdita ignora una parte del gradiente (segno)", "training/CnnPretrainer.java", "net.backward(ws, err, grad);", "net.backward(ws, -err, grad);"),
 ("minibatch: simmetria di augmentation ignorata", "training/CnnPretrainer.java", "augment ? symOf(seed, epoch, i) : 0", "0"),
 ("minibatch: l'ultimo campione di ogni fetta e' perso", "training/CnnPretrainer.java", "b = from + (int) ((long) len * (t + 1) / T);", "b = from + (int) ((long) len * (t + 1) / T) - 1;"),
 ("minibatch: ruolo del campione ignorato", "training/CnnPretrainer.java", "final double err = net.forward(ws, fbuf, d.role[i]) - d.outcome[i];", "final double err = net.forward(ws, fbuf, 0) - d.outcome[i];"),
 ("minibatch: colonne di feature sbagliate (spostate di una)", "training/CnnPretrainer.java", "fbuf[j] = d.feats[i * NF + cols[j]];\n\t\t\t\t}\n\t\t\t\tfinal double err", "fbuf[j] = d.feats[i * NF + (cols[j] + 1) % NF];\n\t\t\t\t}\n\t\t\t\tfinal double err"),
 ("dataset: ruolo registrato sempre 0", "training/CnnPretrainer.java", "rec.role.add((byte) NeuralEvaluator.roleOf(s));", "rec.role.add((byte) 0);"),
 ("dataset: feature registrate spostate", "training/CnnPretrainer.java", "f[i] = (float) full[i];", "f[i] = (float) full[(i + 1) % NF];"),
 ("dataset: esito con segno invertito", "training/CnnPretrainer.java", "rec.outcome = (int) end.outcome;", "rec.outcome = -(int) end.outcome;"),
 ("Adam: correzione del bias del primo momento omessa", "training/Adam.java", "final double a = lr / c1, sq = 1.0 / Math.sqrt(c2);", "final double a = lr, sq = 1.0 / Math.sqrt(c2);"),
 ("Adam: aggiornamento nel verso del gradiente", "training/Adam.java", "w[i] -= (float)", "w[i] += (float)"),
 ("self-play: esito della traiettoria con segno invertito", "training/CnnSelfPlayTrainer.java", "new Traj(cells, feats, roles, game.result.outcome,", "new Traj(cells, feats, roles, -game.result.outcome,"),
]

def main():
    if not os.path.isdir("src") or not os.path.isdir("build"):
        sys.exit("lancia dalla cartella Tablut (serve src/ e build/ gia' compilata)")
    suite = sys.argv[1] if len(sys.argv) > 1 else "model"
    muts, check, to = (MT, CHECK_TRAIN, 400) if suite == "training" else (M, CHECK_MODEL, 240)
    lo = int(sys.argv[2]) if len(sys.argv) > 3 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(muts)
    survivors, killed = [], 0
    for idx in range(lo, hi):
        desc, fname, old, new = muts[idx]
        rel = fname if "/" in fname else "aiplayer/" + fname
        base = os.path.basename(rel)
        work = f"/tmp/mut_{suite}_{idx}"; shutil.rmtree(work, ignore_errors=True); os.makedirs(work + "/out")
        src = open(ROOT + rel, encoding="utf-8").read()
        if old not in src:
            print(f"[{idx:2d}] ERRORE: frammento da mutare non trovato: {desc}"); survivors.append(desc + " (frammento assente)"); continue
        open(f"{work}/{base}", "w", encoding="utf-8").write(src.replace(old, new, 1))
        c = subprocess.run(["javac", "-encoding", "iso-8859-1", "--release", "8", "-nowarn", "-d", work + "/out", "-cp", CP, f"{work}/{base}"], capture_output=True, text=True)
        if c.returncode != 0:
            print(f"[{idx:2d}] il mutante non compila: {desc}\n{c.stderr[:300]}"); survivors.append(desc + " (non compila)"); continue
        try:
            r = subprocess.run(["java", "-cp", f"{work}/out:{CP}", check], capture_output=True, text=True, timeout=to)
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
