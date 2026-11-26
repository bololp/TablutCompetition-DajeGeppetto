package it.unibo.ai.didattica.competition.tablut.experiments;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.*;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
import it.unibo.ai.didattica.competition.tablut.training.NnueDataset;
public class DataGenSpeed {
  static FastState make(byte[] b, boolean blackTurn) {
    State st = TestPositions.blank(); Pawn[][] bd = st.getBoard();
    for (int i = 0; i < 81; i++) { int c = b[i]; bd[i/9][i%9] = c==1?Pawn.WHITE : c==2?Pawn.BLACK : c==4?Pawn.KING : Pawn.EMPTY; }
    st.setTurn(blackTurn ? Turn.BLACK : Turn.WHITE); return FastState.fromState(st);
  }
  public static void main(String[] a) throws Exception {
    Evaluator ev = NnueEvaluator.fromFile(a[0]);
    NnueDataset d = NnueDataset.load(a[1]);
    int n = Integer.parseInt(a[2]); int depth = Integer.parseInt(a[3]);
    Random rnd = new Random(7); List<FastState> pos = new ArrayList<>();
    byte[] bb = new byte[81];
    while (pos.size() < n) { int i = rnd.nextInt(Math.min(d.n, 600000)); if (d.ply(i) < 8 || d.ply(i) > 70 || Math.abs(d.value(i)) > 0.8) continue; d.copyBoard(i, bb, 0); FastState f = make(bb, d.role(i)==1); if (f.winner != FastState.WINNER_NONE) continue; if (!Arrays.equals(f.board, bb)) continue; pos.add(f); }
    String[] cfgs = a[4].split(";");
    double[][] scores = new double[cfgs.length][n]; int[][] moves = new int[cfgs.length][n];
    for (int c = 0; c < cfgs.length; c++) {
      String[] p = cfgs[c].split(","); // prune, ttbits
      PruningOptions po = p[0].equals("off") ? PruningOptions.OFF : PruningOptions.parse(p[0]);
      long nodes = 0; long t0 = System.nanoTime();
      for (int i = 0; i < n; i++) {
        Negamax ng = new Negamax(ev, depth, true, 1 << Integer.parseInt(p[1]), true, Negamax.ORDER_RECOMMENDED, 1); ng.setSafetyMarginMillis(0); ng.setPruning(po);
        moves[c][i] = ng.findBestMove(FastState.copyOf(pos.get(i)), 1_000_000_000L, PositionHistory.empty());
        scores[c][i] = ng.lastScore; nodes += ng.lastNodeCount;
      }
      double s = (System.nanoTime() - t0) / 1e9;
      System.out.printf("d%d %-12s tt2^%s: %6.2f s/posizione, %.1f Mnodi totali%n", depth, p[0], p[1], s / n, nodes / 1e6);
    }
    for (int c = 1; c < cfgs.length; c++) {
      double se = 0; int same = 0; for (int i = 0; i < n; i++) { double x = Math.max(-1, Math.min(1, scores[c][i]/1000.0)), y = Math.max(-1, Math.min(1, scores[0][i]/1000.0)); se += (x-y)*(x-y); if (moves[c][i]==moves[0][i]) same++; }
      System.out.printf("  config %d vs config 0: stessa mossa %d/%d, MSE del valore %.5f%n", c, same, n, se / n);
    }
  }
}
