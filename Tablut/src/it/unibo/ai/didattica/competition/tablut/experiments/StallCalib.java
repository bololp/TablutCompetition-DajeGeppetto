package it.unibo.ai.didattica.competition.tablut.experiments;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import java.nio.file.*; import java.util.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.*;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
public class StallCalib {
  static FastState make(String b, boolean blackTurn, int removeIdx) {
    State st = TestPositions.blank(); Pawn[][] bd = st.getBoard();
    for (int i = 0; i < 81; i++) { char c = b.charAt(i); if (i == removeIdx) c = 'O'; bd[i/9][i%9] = c=='W'?Pawn.WHITE : c=='B'?Pawn.BLACK : c=='K'?Pawn.KING : c=='T'?Pawn.THRONE : Pawn.EMPTY; }
    st.setTurn(blackTurn ? Turn.BLACK : Turn.WHITE); return FastState.fromState(st);
  }
  public static void main(String[] a) throws Exception {
    NnueEvaluator ev = NnueEvaluator.fromFile(a[0]);
    List<String> lines = Files.readAllLines(Paths.get(a[1]));
    int lo = Integer.parseInt(a[2]), hi = Integer.parseInt(a[3]);
    double sumW = 0, sumB = 0, sumBase = 0; int nW = 0, nB = 0, n = 0; double sumZ = 0;
    for (int k = lo; k < Math.min(hi, lines.size()); k++) {
      String[] p = lines.get(k).split(" "); boolean bt = p[1].equals("B");
      double base = ev.evaluate(make(p[2], bt, -1)); // prospettiva Bianco
      sumBase += -base; n++;
      for (int i = 0; i < 81; i++) {
        char c = p[2].charAt(i);
        if (c == 'W') { sumW += (-ev.evaluate(make(p[2], bt, i))) - (-base); nW++; }
        if (c == 'B') { sumB += (-ev.evaluate(make(p[2], bt, i))) - (-base); nB++; }
      }
    }
    System.out.printf("plies %d-%d: valore medio per il Nero %.0f | togliere UNA pedina bianca: %+.1f punti (media su %d) | togliere UNA pedina nera: %+.1f punti (media su %d)%n", lo, hi, sumBase/n, sumW/nW, nW, sumB/nB, nB);
  }
}
