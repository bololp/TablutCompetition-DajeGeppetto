import java.util.*;
import it.unibo.ai.didattica.competition.tablut.training.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

public class BenchV4b {
  static double sink;
  static double time(CnnEvaluator ev, FastState[] fs, int reps) { long t = System.nanoTime(); for (int r = 0; r < reps; r++) for (FastState f : fs) sink += ev.evaluate(f); return (System.nanoTime()-t)/1e3/(reps*(double)fs.length); }
  public static void main(String[] a) throws Exception {
    CnnPretrainer.Dataset d = DatasetUpgrade.loadAndRecompute(a[0]);
    int n = 20000; Random rnd = new Random(1); FastState[] fs = new FastState[n];
    for (int i = 0; i < n; i++) { int k = rnd.nextInt(d.n); fs[i] = FastState.fromState(SearchLabelsLike.stateFrom(d.cells, k*81, d.role[k])); }
    int[] groups = { 11, 11|16, 11|32, 11|64, 123 };
    CnnEvaluator[] ev = new CnnEvaluator[groups.length];
    for (int g = 0; g < groups.length; g++) { String tmp = "/tmp/cnn_v4_g" + groups[g] + ".txt"; WeightConverter.convertFile(a[1], tmp, groups[g]); ev[g] = CnnEvaluator.fromFile(tmp); }
    double[][] res = new double[groups.length][4];
    for (int rep = 0; rep < 4; rep++) for (int g = 0; g < groups.length; g++) { res[g][rep] = time(ev[g], fs, 4); }
    for (int g = 0; g < groups.length; g++) {
      double best = Math.min(Math.min(res[g][2], res[g][3]), Math.min(res[g][0], res[g][1]));
      System.out.printf("CNN completa, gruppi %-4d (%2d feature): %.2f us/valutazione (min di 4 giri; ultimi due: %.2f %.2f)%n", groups[g], ev[g].net.config().nFeat, best, res[g][2], res[g][3]);
    }
  }
}
