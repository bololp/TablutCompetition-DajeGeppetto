import java.io.*;
import java.util.*;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

public class SearchLabels {
  static State stateFrom(byte[] cells, int off, int role) {
    State s = new StateTablut();
    Pawn[][] b = s.getBoard();
    for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) {
      int code = cells[off + r*9 + c];
      b[r][c] = code == 1 ? Pawn.WHITE : code == 2 ? Pawn.BLACK : code == 3 ? Pawn.KING : (r == 4 && c == 4 ? Pawn.THRONE : Pawn.EMPTY);
    }
    s.setTurn(role == 0 ? Turn.WHITE : Turn.BLACK);
    return s;
  }
  public static void main(String[] a) throws Exception {
    CnnPretrainer.Dataset d = CnnPretrainer.Dataset.load(a[0]);
    CnnEvaluator ev = CnnEvaluator.fromFile(a[1]);
    int depth = Integer.parseInt(a[2]); int from = Integer.parseInt(a[3]), count = Integer.parseInt(a[4]);
    long seed = 99; double tmax = Double.parseDouble(a[6]);
    Random rnd = new Random(seed);
    int[] perm = new int[d.n]; for (int i = 0; i < d.n; i++) perm[i] = i;
    for (int i = d.n-1; i > 0; i--) { int k = rnd.nextInt(i+1); int t = perm[i]; perm[i] = perm[k]; perm[k] = t; }
    Negamax ng = new Negamax(ev, depth, true, 1 << 18); ng.setThreads(1); ng.setSafetyMarginMillis(0);
    long t0 = System.nanoTime(); int done = 0;
    try (PrintWriter w = new PrintWriter(new FileWriter(a[5], from > 0))) {
      if (from == 0) w.println("idx,role,static,search,depth");
      for (int k = from; k < from + count; k++) {
        if ((System.nanoTime() - t0) / 1e9 > tmax) break;
        int i = perm[k];
        State s = stateFrom(d.cells, i*81, d.role[i]);
        FastState fs = FastState.fromState(s);
        double st = ev.evaluate(fs);
        ng.findBestMove(fs, 120000);
        w.println(i + "," + d.role[i] + "," + st + "," + ng.lastScore + "," + ng.lastDepthReached);
        done++;
      }
    }
    System.out.printf("fatte %d posizioni in %.1f s (%.3f s/pos)%n", done, (System.nanoTime()-t0)/1e9, (System.nanoTime()-t0)/1e9/Math.max(1,done));
  }
}
