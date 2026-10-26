"""Mutation test della v4 delle feature (dalla cartella Tablut):
   python3 tools/mutation_features_v4.py feat   # 14 bug iniettati in TablutFeatures (ESCAPE/SAFEFORK/GATES): FeaturesCheck deve rilevarli tutti
   python3 tools/mutation_features_v4.py conv   # bug iniettati in WeightConverter/FeatureRegistry/DatasetUpgrade: FeatureConversionCheck
Richiede il progetto gia' compilato in ./build (o TABLUT_BUILD=...). Un mutante 'SOPRAVVISSUTO' e' un buco del test oppure un mutante equivalente
(C9, ruolo invertito in DatasetUpgrade, e' equivalente: l'estrattore non legge il turno)."""
import subprocess, shutil, os, sys, tempfile
ROOT = os.environ.get('TABLUT_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
SRC = ROOT + '/src/it/unibo/ai/didattica/competition/tablut/'
BUILD = os.environ.get('TABLUT_BUILD', ROOT + '/build')
LIBS = ROOT + '/lib/gson-2.2.2.jar:' + ROOT + '/lib/commons-cli-1.4.jar'
T = 'it.unibo.ai.didattica.competition.tablut'

def run_mutant(name, relpath, old, new, test, args):
    path = SRC + relpath
    orig = open(path).read()
    assert orig.count(old) >= 1, f'{name}: pattern non trovato'
    mutated = orig.replace(old, new, 1)
    out = tempfile.mkdtemp(prefix='mutout')
    tmp = tempfile.mkdtemp(prefix='mutsrc')
    mpath = os.path.join(tmp, os.path.basename(path))
    open(mpath, 'w').write(mutated)
    c = subprocess.run(['javac', '-nowarn', '-encoding', 'iso-8859-1', '-d', out, '-cp', BUILD + ':' + LIBS, mpath], capture_output=True, text=True)
    if c.returncode != 0:
        return name, 'NON COMPILA', c.stderr.strip().splitlines()[:2]
    r = subprocess.run(['java', '-cp', out + ':' + BUILD + ':' + LIBS, f'{T}.selftest.{test}'] + args, capture_output=True, text=True, timeout=160, cwd=ROOT)
    detected = r.returncode != 0
    fails = [l for l in r.stdout.splitlines() if l.startswith('FAIL')][:1]
    return name, 'RILEVATO' if detected else 'SOPRAVVISSUTO', (fails[0][:110] if fails else (r.stderr.strip().splitlines()[:1] or ['']))

conv = [
 ('C1b DIV ignorato nel ripiegamento', 'aiplayer/WeightConverter.java', 't[jd][j] += e.getValue() * to.div[newCols[jd]] / divOld;', 't[jd][j] += e.getValue();'),
 ('C1 DIV ignorato nel rapporto', 'aiplayer/WeightConverter.java', 't[jn][j] += to.div[newCols[jn]] / divOld;', 't[jn][j] += 1.0;'),
 ('C2 segno della identita BLACK_RING4', 'aiplayer/FeatureRegistry.java', '"BLACK_RING1", -1', '"BLACK_RING1", 1'),
 ('C3 ripiegamento non applicato', 'aiplayer/WeightConverter.java', 't[jd][j] += e.getValue() * to.div[newCols[jd]] / divOld;', 't[jd][j] += 0.0 * e.getValue() * to.div[newCols[jd]] / divOld;'),
 ('C4 colonna ruolo MLP non spostata', 'aiplayer/WeightConverter.java', 'np[tn[l][0] + o * tn[l][2] + plan.newIds.length] = op[to[l][0] + o * to[l][2] + plan.oldIds.length];', ';'),
 ('C5 parte spaziale CNN copiata corta di 1', 'aiplayer/WeightConverter.java', 'np, w0 + o * newIn, fs);', 'np, w0 + o * newIn, fs - 1);'),
 ('C6 colonna nuova non a zero', 'aiplayer/WeightConverter.java', 'newW[newOff + jn] = acc;', 'newW[newOff + jn] = acc + (jn == 0 ? 1e-2 : 0.0);'),
 ('C7 id v3 scambiati nel registro', 'aiplayer/FeatureRegistry.java', '"KING_SUPPORT", "BLACK_NEAR_KING"', '"BLACK_NEAR_KING", "KING_SUPPORT"'),
 ('C8 bias MLP non copiati', 'aiplayer/WeightConverter.java', 'System.arraycopy(op, to[l][1], np, tn[l][1], out); // bias', '// bias non copiati'),
 ('C9 DatasetUpgrade ruolo invertito', 'training/DatasetUpgrade.java', 's.setTurn(role[i] == 0 ? Turn.WHITE : Turn.BLACK);', 's.setTurn(role[i] == 0 ? Turn.BLACK : Turn.WHITE);'),
 ('C10 strati interni MLP copiati corti', 'aiplayer/WeightConverter.java', 'System.arraycopy(op, to[l][0], np, tn[l][0], to[l][2] * out);', 'System.arraycopy(op, to[l][0], np, tn[l][0], to[l][2] * out - 1);'),
 ('C11 DERIVED KING_AXIS_MAX sparita', 'aiplayer/FeatureRegistry.java', 'd.put("KING_AXIS_MAX", terms("KING_RING", 1));', ''),
]
feat = [
 ('F1 BFS ferma a profondita 2', 'aiplayer/TablutFeatures.java', 'final int maxDepth = wantEsc ? 3 : 1;', 'final int maxDepth = wantEsc ? 2 : 1;'),
 ('F2 fuga-in-3 contata dallo strato 1', 'aiplayer/TablutFeatures.java', 'if (depth == 2) {\n\t\t\t\tfor (int i = 0; i < nNxt; i++) {', 'if (depth == 1) {\n\t\t\t\tfor (int i = 0; i < nNxt; i++) {'),
 ('F3 forchette: cattura ignorata', 'aiplayer/TablutFeatures.java', 'if (!kingCapturable(b, rt, c)) {\n\t\t\t\tsafe++;', 'if (true) {\n\t\t\t\tsafe++;'),
 ('F4 forchette: Re non piazzato', 'aiplayer/TablutFeatures.java', '\t\t\tb[c] = KING;\n\t\t\tArrays.fill(rt, (byte) 0);', '\t\t\tArrays.fill(rt, (byte) 0);'),
 ('F5 gate: passo col segno sbagliato', 'aiplayer/TablutFeatures.java', 'GATE_CELL[n] = idx(l, 8);\n\t\t\tGATE_STEP[n++] = -1;', 'GATE_CELL[n] = idx(l, 8);\n\t\t\tGATE_STEP[n++] = 1;'),
 ('F6 gate: chiuso se corsa <= 1', 'aiplayer/TablutFeatures.java', 'if (run == 0) {\n\t\t\t\tclosed++;', 'if (run <= 1) {\n\t\t\t\tclosed++;'),
 ('F7 distanza: ultima non prima', 'aiplayer/TablutFeatures.java', 'if (dist == 0 && edgesHere > 0) {', 'if (edgesHere > 0) {'),
 ('F8 origine non marcata come vista', 'aiplayer/TablutFeatures.java', 'cur[0] = kp;\n\t\tseen[kp] = stamp;', 'cur[0] = kp;'),
 ('F9 cella di partenza non ripristinata', 'aiplayer/TablutFeatures.java', '\t\tb[kp] = saved;\n\t\tif (wantEsc) {', '\t\tif (wantEsc) {'),
 ('F10 ESC_DIST = d invece di 1/d', 'aiplayer/TablutFeatures.java', '(1.0 / dist) * INV[F_ESC_DIST]', '(double) dist * INV[F_ESC_DIST]'),
 ('F11 fuga-in-3 richiede 2 vie', 'aiplayer/TablutFeatures.java', 'if (lanesFrom(b, nxt[i]) >= 1) {\n\t\t\t\t\t\tthreeStep++;', 'if (lanesFrom(b, nxt[i]) >= 2) {\n\t\t\t\t\t\tthreeStep++;'),
 ('F12 reachTmp non azzerata', 'aiplayer/TablutFeatures.java', '\t\t\tb[c] = KING;\n\t\t\tArrays.fill(rt, (byte) 0);', '\t\t\tb[c] = KING;'),
 ('F13 ESC_EDGE3 conta solo strato 1', 'aiplayer/TablutFeatures.java', '\t\t\tedgeCells += edgesHere;', '\t\t\tedgeCells += depth == 1 ? edgesHere : 0;'),
 ('F14 forchette: soglia >=1 vie', 'aiplayer/TablutFeatures.java', 'if (lanesFrom(b, c) < 2) {\n\t\t\t\tcontinue;', 'if (lanesFrom(b, c) < 1) {\n\t\t\t\tcontinue;'),
]
which = sys.argv[1]
res = []
if which == 'conv':
    for m in conv: res.append(run_mutant(*m, 'FeatureConversionCheck', ['600', '5', '/nonexistent']))
else:
    for m in feat: res.append(run_mutant(*m, 'FeaturesCheck', ['2500', '31']))
for r in res: print(f'{r[0]:42s} {r[1]:14s} {r[2]}')
print('rilevati', sum(1 for r in res if r[1] == 'RILEVATO'), 'su', len(res))
