package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
public class PvsDepthBench {
    public static void main(String[] a) throws Exception {
        String spec = a[0]; int maxD = Integer.parseInt(a[1]); long ms = Long.parseLong(a[2]); int n = Integer.parseInt(a[3]);
        Evaluator ev = spec.equals("simple") ? new SimpleEvaluator() : NeuralEvaluator.fromFile(spec);
        List<State> ps = Bench0.positions(300, 9); List<State> sel = new ArrayList<>(); for (int i = 0; i < n; i++) sel.add(ps.get(i * 23 % ps.size()));
        // 1) profondita' fissa: nodi e tempo per arrivarci (TT accesa, come in partita)
        for (int d = 4; d <= maxD; d++) {
            long nAB = 0, nPVS = 0; double tAB = 0, tPVS = 0;
            for (State s : sel) {
                for (int pass = 0; pass < 2; pass++) {
                    boolean pvs = pass == 1; Negamax ng = new Negamax(ev, d, true, 1 << 20, pvs);
                    long t0 = System.nanoTime(); ng.findBestMove(s, 600000); double t = (System.nanoTime() - t0) / 1e9;
                    if (pvs) { nPVS += ng.lastNodeCount; tPVS += t; } else { nAB += ng.lastNodeCount; tAB += t; }
                }
            }
            System.out.printf("profondita' %d (iterative deepening fino a %d): alpha-beta %.1fM nodi %.1fs | PVS %.1fM nodi %.1fs  => nodi %+.1f%%, tempo %+.1f%%%n", d, d, nAB / 1e6, tAB, nPVS / 1e6, tPVS, 100.0 * (nPVS - nAB) / nAB, 100.0 * (tPVS - tAB) / tAB);
        }
        // 2) tempo fisso: profondita' raggiunta
        double dAB = 0, dPVS = 0;
        for (State s : sel) for (int pass = 0; pass < 2; pass++) { Negamax ng = new Negamax(ev, 40, true, 1 << 20, pass == 1); ng.findBestMove(s, ms); if (pass == 1) dPVS += ng.lastDepthReached; else dAB += ng.lastDepthReached; }
        System.out.printf("tempo fisso %d ms/mossa: profondita' media alpha-beta %.2f, PVS %.2f (%d posizioni)%n", ms, dAB / sel.size(), dPVS / sel.size(), sel.size());
    }
}
