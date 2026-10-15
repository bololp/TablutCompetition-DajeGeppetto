package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
public class EvalBench {
    static volatile double sink;
    public static void main(String[] a) throws Exception {
        Evaluator ev = Arena.build(a[0]);
        List<State> ps = Bench0.positions(800, 3);
        for (int w = 0; w < 60; w++) for (State s : ps) sink += ev.evaluate(s);
        double best = 1e18;
        for (int r = 0; r < 5; r++) { long t0 = System.nanoTime(); for (int k = 0; k < 20; k++) for (State s : ps) sink += ev.evaluate(s); best = Math.min(best, (System.nanoTime() - t0) / (20.0 * ps.size())); }
        System.out.printf("%-62s %6.0f ns/valutazione%n", a[1], best);
    }
}
