package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures.Workspace;
import it.unibo.ai.didattica.competition.tablut.domain.*;
public class FeatBench1 {
    static volatile double sink;
    public static void main(String[] a) {
        int mask = Integer.parseInt(a[0]);
        List<State> ps = Bench0.positions(800, 3);
        Workspace ws = TablutFeatures.newWorkspace(); double[] out = new double[TablutFeatures.NUM_FEATURES];
        for (int w = 0; w < 40; w++) for (State s : ps) { TablutFeatures.extract(s, ws, out, mask); sink += out[3]; }
        double best = 1e9;
        for (int round = 0; round < 5; round++) {
            long t0 = System.nanoTime(); int reps = 20;
            for (int r = 0; r < reps; r++) for (State s : ps) { TablutFeatures.extract(s, ws, out, mask); sink += out[3]; }
            best = Math.min(best, (System.nanoTime() - t0) / (double) (reps * ps.size()));
        }
        System.out.printf("mask=%2d  %6.0f ns%n", mask, best);
    }
}
