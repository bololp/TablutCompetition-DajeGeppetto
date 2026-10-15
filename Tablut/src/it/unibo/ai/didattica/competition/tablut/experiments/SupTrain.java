package it.unibo.ai.didattica.competition.tablut.experiments;

import java.io.*;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP.*;
import static it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures.*;

/** Addestra (supervisionato, esito della partita) i pesi di un tier e li salva nel formato vero di NeuralEvaluator. */
public class SupTrain {
    public static void main(String[] a) throws Exception {
        ArchExp.Data d = ArchExp.load(a[0]); int groups = Integer.parseInt(a[1]); String outFile = a[2];
        int epochs = Integer.parseInt(a[3]); String kind = a.length > 4 ? a[4] : "new";
        long seed = 42;
        Config cfg; double[][] inp = new double[d.n][];
        if (kind.equals("old")) {
            cfg = new Config(16, new int[0], new int[]{24, 12}, 1, false, Act.TANH);
            for (int i = 0; i < d.n; i++) inp[i] = OldFeatures.extract(ArchExp.toState(d.board[i], d.role[i]));
        } else {
            int[] cols = columnsFor(groups);
            cfg = NeuralEvaluator.defaultConfig(groups);
            for (int i = 0; i < d.n; i++) { double[] v = new double[cols.length]; for (int k = 0; k < cols.length; k++) v[k] = d.x[i][cols[k]]; inp[i] = v; }
        }
        MLP net = new MLP(cfg, seed);
        double[] p = net.params(), g = new double[p.length], m = new double[p.length], v = new double[p.length];
        MLP.Scratch sc = net.newScratch(); Random rnd = new Random(seed);
        int[] idx = new int[d.n]; for (int i = 0; i < d.n; i++) idx[i] = i;
        int batch = 128; long step = 0; double lr0 = 3e-3;
        for (int ep = 0; ep < epochs; ep++) {
            for (int i = idx.length - 1; i > 0; i--) { int j = rnd.nextInt(i + 1); int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
            double lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs)); double loss = 0;
            for (int b = 0; b + batch <= idx.length; b += batch) {
                Arrays.fill(g, 0);
                for (int k = 0; k < batch; k++) { int i = idx[b + k]; double out = net.forward(inp[i], d.role[i], sc); double e = out - d.z[i]; loss += e * e; net.backward(sc, e, g); }
                step++; double c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
                for (int k = 0; k < p.length; k++) { double gk = g[k] / batch; m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk; p[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8); }
            }
            if (ep % 5 == 4 || ep == epochs - 1) System.out.printf("  epoca %d  train MSE=%.4f%n", ep + 1, loss / idx.length);
        }
        net.save(outFile, kind.equals("old") ? 1 : TablutFeatures.VERSION, groups);
        System.out.println("salvato " + outFile + " (" + cfg + ", " + net.numParams() + " parametri)");
    }
}
