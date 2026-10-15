package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/** Baseline: dove va il tempo in un nodo di ricerca? */
public class Bench0 {
    static volatile double sink;
    static List<State> positions(int n, long seed) {
        Random rnd = new Random(seed);
        List<State> out = new ArrayList<>();
        Negamax nm = new Negamax(new SimpleEvaluator(), 2);
        while (out.size() < n) {
            State s = new StateTablut(); s.setTurn(Turn.WHITE);
            for (int ply = 0; ply < 120 && !AshtonRules.isTerminal(s); ply++) {
                List<Action> mv = AshtonRules.generateMoves(s);
                if (mv.isEmpty()) break;
                Action a = (rnd.nextDouble() < 0.5) ? mv.get(rnd.nextInt(mv.size())) : nm.findBestMove(s, 60);
                if (a == null) break;
                s = AshtonRules.applyMove(s, a);
                if (ply >= 4 && !AshtonRules.isTerminal(s) && rnd.nextInt(3) == 0) out.add(s);
                if (out.size() >= n) break;
            }
        }
        return out;
    }
    interface Task { void run(State s); }
    static double timeNs(List<State> ps, Task t, int reps) {
        for (int w = 0; w < 3; w++) for (State s : ps) t.run(s);
        long t0 = System.nanoTime();
        for (int r = 0; r < reps; r++) for (State s : ps) t.run(s);
        return (System.nanoTime() - t0) / (double) (reps * ps.size());
    }
    public static void main(String[] a) throws Exception {
        List<State> ps = positions(600, 1);
        double avgMoves = 0; for (State s : ps) avgMoves += AshtonRules.generateMoves(s).size(); avgMoves /= ps.size();
        System.out.printf("posizioni=%d  mosse legali medie=%.1f%n", ps.size(), avgMoves);

        double gen = timeNs(ps, s -> sink = AshtonRules.generateMoves(s).size(), 20);
        System.out.printf("generateMoves (per posizione)      : %8.0f ns  => %.0f ns/mossa generata%n", gen, gen / avgMoves);
        // applyMove + wasCapture + hash su tutte le mosse: costo per mossa
        double child = timeNs(ps, s -> {
            long h = ZobristHasher.fullHash(s);
            for (Action m : AshtonRules.generateMoves(s)) {
                State c = AshtonRules.applyMove(s, m);
                h ^= ZobristHasher.updateHash(h, s, c, m);
                if (AshtonRules.wasCapture(s, c, m)) h++;
            }
            sink = h;
        }, 10);
        System.out.printf("gen+apply+hash+wasCapture (posiz.) : %8.0f ns  => %.0f ns per figlio%n", child, child / avgMoves);
        double ev = timeNs(ps, s -> sink = new SimpleEvaluator().evaluate(s), 200);
        SimpleEvaluator se = new SimpleEvaluator();
        ev = timeNs(ps, s -> sink = se.evaluate(s), 200);
        System.out.printf("SimpleEvaluator.evaluate           : %8.0f ns%n", ev);
        NeuralEvaluator ne = NeuralEvaluator.randomInit(7);
        double nev = timeNs(ps, s -> sink = ne.evaluate(s), 200);
        System.out.printf("NeuralEvaluator (16-24-12-1)       : %8.0f ns%n", nev);
        double feat = timeNs(ps, s -> sink = TablutFeatures.extract(s)[3], 200);
        System.out.printf("  di cui TablutFeatures.extract    : %8.0f ns%n", feat);

        // ricerca completa a profondita' fissa, con strumentazione sul costo dell'evaluator
        for (String name : new String[]{"Simple", "Neural16-24-12-1"}) {
            final Evaluator base = name.equals("Simple") ? new SimpleEvaluator() : NeuralEvaluator.randomInit(7);
            final long[] ns = {0, 0};
            Evaluator timed = st -> { long t0 = System.nanoTime(); double v = base.evaluate(st); ns[0] += System.nanoTime() - t0; ns[1]++; return v; };
            // calibrazione overhead di nanoTime
            long c0 = System.nanoTime(); for (int i = 0; i < 200000; i++) { long x = System.nanoTime(); sink = x; } double nanoOv = (System.nanoTime() - c0) / 200000.0;
            long totNodes = 0, totTime = 0;
            int N = 12;
            for (int i = 0; i < N; i++) {
                State s = ps.get(i * 37 % ps.size());
                Negamax nm = new Negamax(timed, 4, false, 1 << 10); // TT off: costo puro per nodo
                long t0 = System.nanoTime();
                nm.findBestMove(s, 600000);
                totTime += System.nanoTime() - t0;
                totNodes += nm.lastNodeCount;
            }
            double evalShare = (ns[0] - ns[1] * nanoOv) / (double) totTime;
            System.out.printf("%-18s depth<=4 TT off: nodi=%d tempo=%.2fs => %.0f ns/nodo | eval chiamate=%d (%.0f%% dei nodi) | quota tempo in eval ~%.0f%%%n",
                name, totNodes, totTime / 1e9, totTime / (double) totNodes, ns[1], 100.0 * ns[1] / totNodes, 100 * evalShare);
        }
    }
}
