package it.unibo.ai.didattica.competition.tablut.experiments;

import java.io.*; import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
public class NetStats {
    public static void main(String[] a) throws Exception {
        NeuralEvaluator ev = NeuralEvaluator.fromFile(a[0]);
        double[] p = ev.net.params(); double mx = 0, norm = 0; boolean fin = true;
        for (double v : p) { if (Double.isNaN(v) || Double.isInfinite(v)) fin = false; mx = Math.max(mx, Math.abs(v)); norm += v * v; }
        ArchExp.Data d = ArchExp.load(a[1]); int n = Math.min(d.n, 20000);
        double s = 0, s2 = 0; int sat = 0; double[] byRole = new double[2]; int[] cnt = new int[2];
        for (int i = 0; i < n; i++) { State st = ArchExp.toState(d.board[i], d.role[i]); double v = ev.evaluate(st) / 1000.0; s += v; s2 += v * v; if (Math.abs(v) > 0.95) sat++; byRole[d.role[i]] += v; cnt[d.role[i]]++; }
        double mean = s / n, sd = Math.sqrt(s2 / n - mean * mean);
        System.out.printf("pesi finiti=%b  max|w|=%.3f  ||w||=%.3f | uscita su %d posizioni: media=%.3f sd=%.3f saturate(|v|>.95)=%.1f%%  media(W di turno)=%.3f media(B di turno)=%.3f%n", fin, mx, Math.sqrt(norm), n, mean, sd, 100.0 * sat / n, byRole[0] / cnt[0], byRole[1] / cnt[1]);
    }
}
