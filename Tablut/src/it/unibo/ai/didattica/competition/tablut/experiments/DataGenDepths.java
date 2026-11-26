package it.unibo.ai.didattica.competition.tablut.experiments;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.engine.*;
import it.unibo.ai.didattica.competition.tablut.training.NnueDataset;
public class DataGenDepths {
  public static void main(String[] a) throws Exception {
    Evaluator ev = NnueEvaluator.fromFile(a[0]); NnueDataset d = NnueDataset.load(a[1]); int n = Integer.parseInt(a[2]);
    Random rnd = new Random(21); List<FastState> pos = new ArrayList<>(); byte[] bb = new byte[81];
    while (pos.size() < n) { int i = rnd.nextInt(Math.min(d.n, 600000)); if (d.ply(i) < 8 || d.ply(i) > 70 || Math.abs(d.value(i)) > 0.9) continue; d.copyBoard(i, bb, 0); FastState f = DataGenSpeed.make(bb, d.role(i)==1); if (f.winner != FastState.WINNER_NONE || !Arrays.equals(f.board, bb)) continue; pos.add(f); }
    int[] depths = {4,5,6,7,8}; double[][] v = new double[depths.length][n]; double[] secs = new double[depths.length];
    for (int k = 0; k < depths.length; k++) { long t0 = System.nanoTime();
      for (int i = 0; i < n; i++) { Negamax ng = new Negamax(ev, depths[k], true, 1 << 22, true, Negamax.ORDER_RECOMMENDED, 1); ng.setSafetyMarginMillis(0); ng.setPruning(PruningOptions.parse("all"));
        ng.findBestMove(FastState.copyOf(pos.get(i)), 1_000_000_000L, PositionHistory.empty()); double sc = ng.lastScore; boolean white = pos.get(i).turn == BoardConstants.WHITE_TURN; double w = white ? sc : -sc; v[k][i] = Math.max(-1, Math.min(1, w / 1000.0)); }
      secs[k] = (System.nanoTime() - t0) / 1e9 / n; }
    for (int k = 0; k < depths.length - 1; k++) { double se = 0; for (int i = 0; i < n; i++) se += (v[k][i] - v[4][i]) * (v[k][i] - v[4][i]); System.out.printf("profondita' %d (%.2f s/pos): RMS rispetto alla 8 = %.3f%n", depths[k], secs[k], Math.sqrt(se / n)); }
    System.out.printf("profondita' 8 (%.2f s/pos)%n", secs[4]);
  }
}
