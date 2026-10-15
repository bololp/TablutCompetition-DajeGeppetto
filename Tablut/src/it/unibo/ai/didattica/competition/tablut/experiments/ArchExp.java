package it.unibo.ai.didattica.competition.tablut.experiments;

import java.io.*;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP.Act;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP.Config;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.*;
import static it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures.*;

/** Confronto di architetture / dimensioni / attivazioni / gruppi di feature su un compito supervisionato: predire l'esito della partita da una posizione. */
public class ArchExp {
    static class Data {
        float[][] x; byte[] role; float[] z; float[] simple; byte[][] board; int[] game; int[] pliesToEnd; int n;
    }
    static Data load(String file) throws IOException {
        List<float[]> X = new ArrayList<>(); List<Byte> R = new ArrayList<>(); List<Float> Z = new ArrayList<>(), S = new ArrayList<>();
        List<byte[]> B = new ArrayList<>(); List<Integer> G = new ArrayList<>(), P = new ArrayList<>();
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file), 1 << 20))) {
            int nf = in.readInt();
            while (true) {
                int gid; try { gid = in.readInt(); } catch (EOFException e) { break; }
                int outcome = in.readInt(); boolean trunc = in.readBoolean(); int n = in.readInt();
                for (int i = 0; i < n; i++) {
                    byte role = in.readByte(); float sc = in.readFloat(); byte[] bd = new byte[81]; in.readFully(bd);
                    float[] f = new float[nf]; for (int k = 0; k < nf; k++) f[k] = in.readFloat();
                    if (trunc) continue;
                    X.add(f); R.add(role); Z.add((float) outcome); S.add(sc); B.add(bd); G.add(gid); P.add(n - i);
                }
            }
        }
        Data d = new Data(); d.n = X.size();
        d.x = X.toArray(new float[0][]); d.role = new byte[d.n]; d.z = new float[d.n]; d.simple = new float[d.n];
        d.board = B.toArray(new byte[0][]); d.game = new int[d.n]; d.pliesToEnd = new int[d.n];
        for (int i = 0; i < d.n; i++) { d.role[i] = R.get(i); d.z[i] = Z.get(i); d.simple[i] = S.get(i); d.game[i] = G.get(i); d.pliesToEnd[i] = P.get(i); }
        return d;
    }
    static State toState(byte[] bd, int role) {
        State s = new StateTablut(); Pawn[][] b = s.getBoard(); Pawn[] vals = Pawn.values();
        for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) b[r][c] = vals[bd[r * 9 + c]];
        s.setTurn(role == 1 ? Turn.BLACK : Turn.WHITE); return s;
    }

    /** Variante di ingresso: quali colonne tenere (le altre azzerate) oppure sorgente alternativa. */
    static class Spec {
        String name; Config cfg; String source = "new"; boolean[] keep;
        Spec(String n, Config c) { name = n; cfg = c; }
        Spec source(String s) { source = s; return this; }
        Spec drop(int... cols) { keep = new boolean[NUM_FEATURES]; Arrays.fill(keep, true); for (int c : cols) keep[c] = false; return this; }
        Spec only(int... cols) { keep = new boolean[NUM_FEATURES]; for (int c : cols) keep[c] = true; return this; }
    }

    static double[][] buildInputs(Data d, Spec sp) {
        double[][] in = new double[d.n][];
        for (int i = 0; i < d.n; i++) {
            if (sp.source.equals("new")) {
                double[] v = new double[NUM_FEATURES];
                for (int k = 0; k < NUM_FEATURES; k++) v[k] = (sp.keep == null || sp.keep[k]) ? d.x[i][k] : 0.0;
                in[i] = v;
            } else if (sp.source.equals("old")) {
                in[i] = OldFeatures.extract(toState(d.board[i], d.role[i]));
            } else { // simple
                in[i] = new double[] { d.simple[i] / 1000.0 };
            }
        }
        return in;
    }

    static class Result { double mse, r2, acc, mseW, mseB, cons, consWB, consBW, laneMse; int laneN; int params, onPath; double fwdNs; }

    static Result run(Data d, double[][] inp, Spec sp, long seed, int epochs, double lr0, int batch, double zMean, double zVar) {
        Random rnd = new Random(seed);
        MLP net = new MLP(sp.cfg, seed * 7919 + 13);
        double[] p = net.params(), g = new double[p.length], m = new double[p.length], v = new double[p.length];
        MLP.Scratch sc = net.newScratch();
        List<Integer> tr = new ArrayList<>(), te = new ArrayList<>();
        double frac = Double.parseDouble(System.getProperty("frac", "1.0"));
        for (int i = 0; i < d.n; i++) { if (d.game[i] % 5 == 0) te.add(i); else if ((d.game[i] * 2654435761L % 1000) / 1000.0 < frac) tr.add(i); }
        int[] trIdx = new int[tr.size()]; for (int i = 0; i < trIdx.length; i++) trIdx[i] = tr.get(i);
        long step = 0;
        for (int ep = 0; ep < epochs; ep++) {
            for (int i = trIdx.length - 1; i > 0; i--) { int j = rnd.nextInt(i + 1); int t = trIdx[i]; trIdx[i] = trIdx[j]; trIdx[j] = t; }
            double lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs));
            for (int b = 0; b + batch <= trIdx.length; b += batch) {
                Arrays.fill(g, 0);
                for (int k = 0; k < batch; k++) {
                    int i = trIdx[b + k];
                    double out = net.forward(inp[i], d.role[i], sc);
                    net.backward(sc, out - d.z[i], g);
                }
                step++;
                double c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
                for (int k = 0; k < p.length; k++) {
                    double gk = g[k] / batch;
                    m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk;
                    p[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
                }
            }
        }
        Result r = new Result(); double se = 0, seW = 0, seB = 0; int nW = 0, nB = 0, ok = 0, dec = 0;
        for (int i : te) {
            double out = net.forward(inp[i], d.role[i], sc); double e = out - d.z[i]; se += e * e;
            if (d.role[i] == 0) { seW += e * e; nW++; } else { seB += e * e; nB++; }
            if (d.z[i] != 0) { dec++; if (Math.signum(out) == Math.signum(d.z[i])) ok++; }
        }
        double[] pred = new double[te.size()];
        for (int k = 0; k < te.size(); k++) pred[k] = net.forward(inp[te.get(k)], d.role[te.get(k)], sc);
        double cs = 0, csWB = 0, csBW = 0; int cn = 0, cwb = 0, cbw = 0; double ls = 0; int ln = 0;
        for (int k = 0; k < te.size(); k++) {
            int i = te.get(k);
            if (d.x[i][F_KING_LANES] > 0 || d.x[i][F_KING_2STEP_FORKS] > 0) { double e = pred[k] - d.z[i]; ls += e * e; ln++; }
            if (k + 1 < te.size()) { int j = te.get(k + 1); if (d.game[j] == d.game[i] && j == i + 1) {
                double dv = pred[k + 1] - pred[k]; cs += Math.abs(dv); cn++;
                if (d.role[i] == 0) { csWB += dv; cwb++; } else { csBW += dv; cbw++; } } }
        }
        r.cons = cs / Math.max(1, cn); r.consWB = csWB / Math.max(1, cwb); r.consBW = csBW / Math.max(1, cbw); r.laneMse = ls / Math.max(1, ln); r.laneN = ln;
        r.mse = se / te.size(); r.mseW = seW / Math.max(1, nW); r.mseB = seB / Math.max(1, nB);
        r.r2 = 1 - r.mse / zVar; r.acc = ok / (double) Math.max(1, dec);
        r.params = net.numParams(); r.onPath = net.paramsOnPath(); r.fwdNs = fwdNs(net, inp);
        return r;
    }

    static volatile double sink;
    static double fwdNs(MLP net, double[][] inp) {
        MLP.Scratch sc = net.newScratch(); int n = Math.min(inp.length, 4096);
        for (int w = 0; w < 100000; w++) sink += net.forward(inp[w % n], w & 1, sc);
        double best = 1e18;
        for (int round = 0; round < 3; round++) {
            long t0 = System.nanoTime(); int reps = 300000;
            for (int w = 0; w < reps; w++) sink += net.forward(inp[w % n], w & 1, sc);
            best = Math.min(best, (System.nanoTime() - t0) / (double) reps);
        }
        return best;
    }

    public static void main(String[] a) throws Exception {
        Data d = load(a[0]); String exp = a[1]; int seeds = a.length > 2 ? Integer.parseInt(a[2]) : 2; int epochs = a.length > 3 ? Integer.parseInt(a[3]) : 12;
        double zm = 0; int nt = 0; for (int i = 0; i < d.n; i++) if (d.game[i] % 5 != 0) { zm += d.z[i]; nt++; } zm /= nt;
        double zv = 0, nte = 0; for (int i = 0; i < d.n; i++) if (d.game[i] % 5 == 0) { zv += (d.z[i] - zm) * (d.z[i] - zm); nte++; } zv /= nte;
        int dec = 0; for (int i = 0; i < d.n; i++) if (d.z[i] != 0) dec++;
        System.out.printf("posizioni=%d (partite decisive=100%% non troncate) test=%.0f  varianza esito test=%.3f  media esito train=%.3f%n", d.n, nte, zv, zm);
        int F = NUM_FEATURES; int[] sh = {32}, hd = {16};
        List<Spec> specs = new ArrayList<>();
        Act RELU = Act.RELU;
        if (exp.equals("base")) {
            specs.add(new Spec("Simple (1 ingresso, 2 head=calibrazione per ruolo)", new Config(1, new int[0], new int[0], 2, false, Act.TANH)).source("simple"));
            specs.add(new Spec("VECCHIO: feature v1(16) rete 24-12 tanh", new Config(16, new int[0], new int[]{24,12}, 1, false, Act.TANH)).source("old"));
            specs.add(new Spec("lineare su " + F + " feature (2 head)", new Config(F, new int[0], new int[0], 2, false, RELU)));
            specs.add(new Spec("A  " + F + "->32->16->1 ruolo come input", Config.singleHeadRoleInput(F, sh, hd, RELU)));
            specs.add(new Spec("B  tronco 32 + 2 head(16)", Config.sharedTrunk(F, sh, hd, RELU)));
            specs.add(new Spec("C  2 reti separate 32-16", Config.separateNets(F, new int[]{32,16}, RELU)));
        } else if (exp.equals("cap")) {
            for (int[] s : new int[][]{{8,4},{16,8},{24,8},{32,16},{48,16},{64,32},{128,64}})
                specs.add(new Spec("B  tronco " + s[0] + " + 2 head(" + s[1] + ")", Config.sharedTrunk(F, new int[]{s[0]}, new int[]{s[1]}, RELU)));
            specs.add(new Spec("B  tronco 32 + 2 head(solo uscita)", Config.sharedTrunk(F, new int[]{32}, new int[0], RELU)));
            specs.add(new Spec("B  tronco 32-16 + 2 head(8)", Config.sharedTrunk(F, new int[]{32,16}, new int[]{8}, RELU)));
        } else if (exp.equals("role")) {
            specs.add(new Spec("A  " + F + "->32->16->1 ruolo come input", Config.singleHeadRoleInput(F, sh, hd, RELU)));
            specs.add(new Spec("B  tronco 32 + 2 head(16)", Config.sharedTrunk(F, sh, hd, RELU)));
            specs.add(new Spec("C  2 reti separate 32-16", Config.separateNets(F, new int[]{32,16}, RELU)));
            specs.add(new Spec("A' 1 head SENZA ruolo (cieca al turno)", new Config(F, sh, hd, 1, false, RELU)));
        } else if (exp.equals("lanes")) {
            Config c = Config.sharedTrunk(F, new int[]{24}, new int[]{8}, Act.LEAKY);
            specs.add(new Spec("tutte le " + F, c));
            specs.add(new Spec("senza LANES", c).drop(F_KING_LANES, F_KING_LANES_UNBLOCKABLE, F_KING_2STEP_CELLS, F_KING_2STEP_FORKS));
            specs.add(new Spec("senza LANES ne' FLOOD", c).drop(F_KING_LANES, F_KING_LANES_UNBLOCKABLE, F_KING_2STEP_CELLS, F_KING_2STEP_FORKS, F_FLOOD_REGION, F_FLOOD_EDGE, F_FLOOD_SIDES));
        } else if (exp.equals("act")) {
            for (Act ac : Act.values()) specs.add(new Spec("B 32/16 act=" + ac, Config.sharedTrunk(F, sh, hd, ac)));
        } else if (exp.equals("abl")) {
            Config c = Config.sharedTrunk(F, sh, hd, RELU);
            specs.add(new Spec("tutte le " + F, c));
            specs.add(new Spec("senza MOB (6)", c).drop(F_WHITE_MOVES, F_BLACK_MOVES, F_KING_MOVES, F_WHITE_BLOCKED, F_BLACK_BLOCKED, F_BLACK_EDGE_REACH));
            specs.add(new Spec("senza LANES (4)", c).drop(F_KING_LANES, F_KING_LANES_UNBLOCKABLE, F_KING_2STEP_CELLS, F_KING_2STEP_FORKS));
            specs.add(new Spec("senza FLOOD (3)", c).drop(F_FLOOD_REGION, F_FLOOD_EDGE, F_FLOOD_SIDES));
            specs.add(new Spec("senza THREATS (3)", c).drop(F_WHITE_THREATENED, F_BLACK_THREATENED, F_KING_THREAT));
            specs.add(new Spec("senza MOB+THREATS (le due passate costose)", c).drop(F_WHITE_MOVES, F_BLACK_MOVES, F_KING_MOVES, F_WHITE_BLOCKED, F_BLACK_BLOCKED, F_BLACK_EDGE_REACH, F_WHITE_THREATENED, F_BLACK_THREATENED, F_KING_THREAT));
            specs.add(new Spec("senza MOB+THREATS+FLOOD (solo BASE+LANES)", c).drop(F_WHITE_MOVES, F_BLACK_MOVES, F_KING_MOVES, F_WHITE_BLOCKED, F_BLACK_BLOCKED, F_BLACK_EDGE_REACH, F_WHITE_THREATENED, F_BLACK_THREATENED, F_KING_THREAT, F_FLOOD_REGION, F_FLOOD_EDGE, F_FLOOD_SIDES));
            int[] base = TablutFeatures.columnsFor(0); // BASE = feature con FEATURE_GROUP 0 (indici simbolici: non si sposta se ne aggiungono)
            specs.add(new Spec("solo BASE (" + base.length + ")", c).only(base));
            int[] struct = new int[F_BLACK_LINES - F_BLACK_CENTRAL + 1]; for (int i = 0; i < struct.length; i++) struct[i] = F_BLACK_CENTRAL + i;
            specs.add(new Spec("senza struttura (" + F_BLACK_CENTRAL + ".." + F_BLACK_LINES + ")", c).drop(struct));
        }
        System.out.printf("%-58s %7s %7s %7s | %-17s %6s %6s %-11s %7s %7s %7s %8s%n", "modello", "params", "onPath", "fwd ns", "MSE test (±sd)", "R2", "acc", "MSE W/B", "|dV|ply", "bias W>B", "bias B>W", "MSE lane");
        for (Spec sp : specs) {
            double[][] inp = buildInputs(d, sp);
            double[] mses = new double[seeds]; Result last = null; double r2 = 0, acc = 0, mw = 0, mb = 0, cons = 0, cwb = 0, cbw = 0, lm = 0;
            for (int s = 0; s < seeds; s++) { last = run(d, inp, sp, 100 + s, epochs, 3e-3, 128, zm, zv); mses[s] = last.mse; r2 += last.r2; acc += last.acc; mw += last.mseW; mb += last.mseB; cons += last.cons; cwb += last.consWB; cbw += last.consBW; lm += last.laneMse; }
            double mean = Arrays.stream(mses).average().orElse(0); double sd = Math.sqrt(Arrays.stream(mses).map(x -> (x - mean) * (x - mean)).sum() / Math.max(1, seeds - 1));
            System.out.printf("%-58s %7d %7d %7.0f | %.4f ± %.4f    %6.3f %6.3f %.3f/%.3f %7.4f %+7.4f %+7.4f %8.4f%n", sp.name, last.params, last.onPath, last.fwdNs, mean, sd, r2 / seeds, acc / seeds, mw / seeds, mb / seeds, cons / seeds, cwb / seeds, cbw / seeds, lm / seeds);
            System.out.flush();
        }
    }
}
