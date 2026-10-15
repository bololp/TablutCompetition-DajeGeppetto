package it.unibo.ai.didattica.competition.tablut.experiments;

import java.io.*;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures.Workspace;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/** Genera partite (stili misti) e salva per ogni posizione: ruolo, 43 feature (float), punteggio SimpleEvaluator, esito. */
public class GenData {
    public static void main(String[] a) throws Exception {
        String out = a[0]; int games = Integer.parseInt(a[1]); long seed = Long.parseLong(a[2]);
        Random rnd = new Random(seed);
        Workspace ws = TablutFeatures.newWorkspace(); double[] f = new double[TablutFeatures.NUM_FEATURES];
        SimpleEvaluator simple = new SimpleEvaluator();
        long t0 = System.currentTimeMillis(); int wins = 0, bw = 0, trunc = 0; long plies = 0;
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(out), 1 << 20))) {
            o.writeInt(TablutFeatures.NUM_FEATURES);
            for (int g = 0; g < games; g++) {
                int type = rnd.nextInt(10);
                int dW, dB; double epsW, epsB;
                if (type < 4) { dW = 2; dB = 2; epsW = 0.10; epsB = 0.10; }
                else if (type < 7) { boolean sw = rnd.nextBoolean(); dW = sw ? 1 : 3; dB = sw ? 3 : 1; epsW = epsB = 0.05; }
                else { boolean sw = rnd.nextBoolean(); dW = 2; dB = 2; epsW = sw ? 0.5 : 0.03; epsB = sw ? 0.03 : 0.5; }
                Negamax nW = new Negamax(simple, dW, true, 1 << 16), nB = new Negamax(simple, dB, true, 1 << 16);
                State s = new StateTablut(); s.setTurn(Turn.WHITE);
                List<float[]> feats = new ArrayList<>(); List<Integer> roles = new ArrayList<>(); List<Float> sc = new ArrayList<>(); List<byte[]> boards = new ArrayList<>();
                int ply = 0;
                for (; ply < 200 && !AshtonRules.isTerminal(s); ply++) {
                    List<Action> mv = AshtonRules.generateMoves(s);
                    if (mv.isEmpty()) break;
                    TablutFeatures.extract(s, ws, f, TablutFeatures.G_ALL);
                    float[] ff = new float[f.length]; for (int i = 0; i < f.length; i++) ff[i] = (float) f[i];
                    byte[] bb = new byte[81]; for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) bb[r * 9 + c] = (byte) s.getBoard()[r][c].ordinal(); boards.add(bb);
                    feats.add(ff); roles.add(s.getTurn() == Turn.BLACK ? 1 : 0); sc.add((float) simple.evaluate(s));
                    boolean w = s.getTurn() == Turn.WHITE;
                    Action m;
                    if (rnd.nextDouble() < (w ? epsW : epsB)) m = mv.get(rnd.nextInt(mv.size()));
                    else m = (w ? nW : nB).findBestMove(s, 5000);
                    if (m == null) m = mv.get(rnd.nextInt(mv.size()));
                    s = AshtonRules.applyMove(s, m);
                }
                Turn t = s.getTurn();
                int outcome = t == Turn.WHITEWIN ? 1 : (t == Turn.BLACKWIN ? -1 : 0);
                boolean truncated = !AshtonRules.isTerminal(s) && AshtonRules.generateMoves(s).size() > 0;
                if (outcome == 1) wins++; else if (outcome == -1) bw++; if (truncated) trunc++;
                plies += feats.size();
                o.writeInt(g); o.writeInt(outcome); o.writeBoolean(truncated); o.writeInt(feats.size());
                for (int i = 0; i < feats.size(); i++) {
                    o.writeByte(roles.get(i)); o.writeFloat(sc.get(i)); o.write(boards.get(i));
                    for (float v : feats.get(i)) o.writeFloat(v);
                }
                if ((g + 1) % 50 == 0) System.out.printf("game %d  W:%d B:%d trunc:%d  avgPlies=%.1f  %.1fs%n", g + 1, wins, bw, trunc, plies / (double) (g + 1), (System.currentTimeMillis() - t0) / 1000.0);
            }
        }
        System.out.printf("DONE games=%d W:%d B:%d trunc:%d positions=%d time=%.1fs%n", games, wins, bw, trunc, plies, (System.currentTimeMillis() - t0) / 1000.0);
    }
}
