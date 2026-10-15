package it.unibo.ai.didattica.competition.tablut.experiments;

import java.io.*;
import java.lang.reflect.*;
import java.net.*;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/** Partite a tempo fisso per mossa tra evaluator diversi, con aperture casuali accoppiate (ogni apertura giocata a colori invertiti). */
public class Arena {
    /** Carica le classi aiplayer ORIGINALI (build0) in un classloader isolato: stesso nome, codice vecchio. */
    static Evaluator legacyEvaluator(String build0, String weights) throws Exception {
        final String PKG = "it.unibo.ai.didattica.competition.tablut.aiplayer.";
        URLClassLoader cl = new URLClassLoader(new URL[]{new File(build0).toURI().toURL()}, Arena.class.getClassLoader()) {
            @Override protected synchronized Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
                if (name.startsWith(PKG)) {
                    Class<?> c = findLoadedClass(name);
                    if (c == null) c = findClass(name);
                    if (resolve) resolveClass(c);
                    return c;
                }
                return super.loadClass(name, resolve);
            }
        };
        Class<?> ne = cl.loadClass(PKG + "NeuralEvaluator");
        final Object inst = ne.getMethod("fromFile", String.class).invoke(null, weights);
        final Method ev = ne.getMethod("evaluate", State.class);
        return s -> { try { return (Double) ev.invoke(inst, s); } catch (Exception e) { throw new RuntimeException(e); } };
    }

    static Evaluator build(String spec) throws Exception {
        if (spec.equals("simple")) return new SimpleEvaluator();
        if (spec.startsWith("legacy:")) return legacyEvaluator("/home/claude/build0", spec.substring(7));
        if (spec.startsWith("old16:")) { // vecchie feature v1 + rete 24-12 tanh, sul MLP veloce
            final MLP net = MLP.load(spec.substring(6)).net; final MLP.Scratch sc = net.newScratch();
            return s -> net.forward(OldFeatures.extract(s), 0, sc) * 1000.0;
        }
        return NeuralEvaluator.fromFile(spec);
    }

    static class Stats { int w, d, l; double depthSum, nodeSum, timeSum; long moves; }

    public static void main(String[] a) throws Exception {
        String specA = a[0], specB = a[1]; int pairs = Integer.parseInt(a[2]); long ms = Long.parseLong(a[3]); long seed = Long.parseLong(a[4]);
        Evaluator evA = build(specA), evB = build(specB);
        Random rnd = new Random(seed);
        Stats sa = new Stats(), sb = new Stats(); int aw = 0, ad = 0, al = 0;
        for (int p = 0; p < pairs; p++) {
            // apertura casuale: 4 semi-mosse random
            State open = new StateTablut(); open.setTurn(Turn.WHITE);
            for (int i = 0; i < 4; i++) { List<Action> mv = AshtonRules.generateMoves(open); open = AshtonRules.applyMove(open, mv.get(rnd.nextInt(mv.size()))); }
            for (int swap = 0; swap < 2; swap++) {
                Evaluator white = swap == 0 ? evA : evB, black = swap == 0 ? evB : evA;
                Stats sw = swap == 0 ? sa : sb, sbk = swap == 0 ? sb : sa;
                Negamax nw = new Negamax(white, 30, true, 1 << 18), nb = new Negamax(black, 30, true, 1 << 18);
                State s = open; int ply = 4;
                for (; ply < 160 && !AshtonRules.isTerminal(s); ply++) {
                    boolean wt = s.getTurn() == Turn.WHITE; Negamax n = wt ? nw : nb; Stats st = wt ? sw : sbk;
                    long t0 = System.nanoTime();
                    Action m = n.findBestMove(s, ms);
                    st.timeSum += (System.nanoTime() - t0) / 1e9; st.depthSum += n.lastDepthReached; st.nodeSum += n.lastNodeCount; st.moves++;
                    if (m == null) break;
                    s = AshtonRules.applyMove(s, m);
                }
                Turn t = s.getTurn(); int res; // dal punto di vista di A
                if (t == Turn.WHITEWIN) res = swap == 0 ? 1 : -1; else if (t == Turn.BLACKWIN) res = swap == 0 ? -1 : 1;
                else if (AshtonRules.generateMoves(s).isEmpty() && !AshtonRules.isTerminal(s)) res = (s.getTurn() == Turn.WHITE) == (swap == 0) ? -1 : 1; else res = 0;
                if (res > 0) aw++; else if (res < 0) al++; else ad++;
            }
            if ((p + 1) % 5 == 0) System.out.printf("  [%d/%d coppie] A: +%d =%d -%d%n", p + 1, pairs, aw, ad, al);
        }
        int n = aw + ad + al; double score = (aw + 0.5 * ad) / n;
        double z = 1.96, ph = score; double den = 1 + z * z / n; double centre = (ph + z * z / (2 * n)) / den; double half = z * Math.sqrt(ph * (1 - ph) / n + z * z / (4.0 * n * n)) / den;
        System.out.printf("%s  VS  %s  @%dms/mossa: %d partite  A +%d =%d -%d  punteggio A = %.1f%% [IC95 %.1f%% - %.1f%%]%n", specA, specB, ms, n, aw, ad, al, 100 * score, 100 * (centre - half), 100 * (centre + half));
        System.out.printf("   A: profondita' media %.2f, nodi/mossa %.0f, nodi/s %.0f%n   B: profondita' media %.2f, nodi/mossa %.0f, nodi/s %.0f%n",
            sa.depthSum / sa.moves, sa.nodeSum / sa.moves, sa.nodeSum / sa.timeSum, sb.depthSum / sb.moves, sb.nodeSum / sb.moves, sb.nodeSum / sb.timeSum);
    }
}
