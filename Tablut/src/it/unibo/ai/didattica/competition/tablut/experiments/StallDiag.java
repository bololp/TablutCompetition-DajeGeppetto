package it.unibo.ai.didattica.competition.tablut.experiments;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import java.io.*; import java.nio.file.*; import java.util.*;
import it.unibo.ai.didattica.competition.tablut.domain.*; import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn; import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.*;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
public class StallDiag {
  static String sq(int s){ return "" + (char)('a'+ s%9) + (1+ s/9); }
  public static void main(String[] a) throws Exception {
    Evaluator ev = NnueEvaluator.fromFile(a[0]);
    List<String> lines = Files.readAllLines(Paths.get(a[1]));
    int depth = Integer.parseInt(a[2]); int from = Integer.parseInt(a[3]), step = Integer.parseInt(a[4]);
    int[] buf = new int[FastState.MAX_MOVES]; UndoInfo u = new UndoInfo();
    for (int idx = from; idx < lines.size(); idx += step) {
      String[] p = lines.get(idx).split(" "); // ply turn board81
      if (!p[1].equals("B")) continue;
      State st = TestPositions.blank(); Pawn[][] b = st.getBoard();
      for (int i = 0; i < 81; i++) { char c = p[2].charAt(i); b[i/9][i%9] = c=='W'?Pawn.WHITE : c=='B'?Pawn.BLACK : c=='K'?Pawn.KING : c=='T'?Pawn.THRONE : Pawn.EMPTY; }
      st.setTurn(Turn.BLACK);
      FastState root = FastState.fromState(st);
      int n = root.generateMoves(buf); int caps = 0; List<Integer> capMoves = new ArrayList<>();
      for (int k = 0; k < n; k++) { FastState c = FastState.copyOf(root); c.makeMove(buf[k], u); if (u.capturedCount() > 0) { caps++; capMoves.add(buf[k]); } }
      Negamax ng = new Negamax(ev, depth, true, 1<<20, true, Negamax.ORDER_DEFAULT, 1);
      int best = ng.findBestMove(FastState.copyOf(root), 100_000_000L, PositionHistory.empty());
      System.out.printf("ply %s: %d mosse nere, %d catturano; ricerca d%d: %s valore %.1f (statico %.1f)%n", p[0], n, caps, depth, sq(best&127)+sq(best>>>7), ng.lastScore, ev.evaluate(root));
      for (int m : capMoves) {
        FastState c = FastState.copyOf(root); c.makeMove(m, u);
        Negamax ng2 = new Negamax(ev, depth-1, true, 1<<20, true, Negamax.ORDER_DEFAULT, 1);
        ng2.findBestMove(FastState.copyOf(c), 100_000_000L, PositionHistory.empty());
        System.out.printf("     cattura %s%s: statico dopo %.1f, ricerca d%d (visto dal Nero) %.1f%n", sq(m&127), sq(m>>>7), ev.evaluate(c), depth, -ng2.lastScore);
      }
    }
  }
}
