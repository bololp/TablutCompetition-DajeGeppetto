import java.util.*;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.domain.*;

public class PawnValue {
  public static void main(String[] a) throws Exception {
    CnnPretrainer.Dataset d = CnnPretrainer.Dataset.load(a[0]);
    CnnEvaluator ev = CnnEvaluator.fromFile(a[1]);
    Random rnd = new Random(5);
    double sw = 0, sb = 0, sw2 = 0, sb2 = 0; int nw = 0, nb = 0;
    double sK = 0; int nK = 0;
    for (int t = 0; t < 6000; t++) {
      int i = rnd.nextInt(d.n);
      State s = SearchLabelsLike.stateFrom(d.cells, i * 81, d.role[i]);
      FastState fs = FastState.fromState(s); double v0 = ev.evaluate(fs);
      for (int rep = 0; rep < 2; rep++) {
        // rimuove una pedina a caso del colore scelto (non il Re)
        int want = rep == 0 ? 1 : 2; List<Integer> L = new ArrayList<>();
        for (int c = 0; c < 81; c++) if (fs.board[c] == want) L.add(c);
        if (L.isEmpty()) continue;
        int c = L.get(rnd.nextInt(L.size())); FastState f2 = FastState.copyOf(fs); f2.board[c] = 0;
        double dv = ev.evaluate(f2) - v0;
        if (rep == 0) { sw += dv; sw2 += dv*dv; nw++; } else { sb += dv; sb2 += dv*dv; nb++; }
      }
    }
    System.out.printf("togliere una pedina BIANCA: variazione media del valore (prospettiva Bianco, scala 1000) = %+.1f (sd %.1f, n=%d)%n", sw/nw, Math.sqrt(sw2/nw - (sw/nw)*(sw/nw)), nw);
    System.out.printf("togliere una pedina NERA:   variazione media del valore (prospettiva Bianco, scala 1000) = %+.1f (sd %.1f, n=%d)%n", sb/nb, Math.sqrt(sb2/nb - (sb/nb)*(sb/nb)), nb);
  }
}
class SearchLabelsLike {
  static State stateFrom(byte[] cells, int off, int role) {
    State s = new StateTablut(); State.Pawn[][] b = s.getBoard();
    for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) { int code = cells[off + r*9 + c];
      b[r][c] = code == 1 ? State.Pawn.WHITE : code == 2 ? State.Pawn.BLACK : code == 3 ? State.Pawn.KING : (r == 4 && c == 4 ? State.Pawn.THRONE : State.Pawn.EMPTY); }
    s.setTurn(role == 0 ? State.Turn.WHITE : State.Turn.BLACK); return s;
  }
}
