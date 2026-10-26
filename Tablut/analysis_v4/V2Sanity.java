import java.util.*;
import it.unibo.ai.didattica.competition.tablut.training.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

public class V2Sanity {
  public static void main(String[] a) throws Exception {
    CnnPretrainer.Dataset d = DatasetUpgrade.loadAndRecompute(a[0]);
    int n = 30000; Random rnd = new Random(3); int[] idx = new int[n]; FastState[] fs = new FastState[n];
    for (int i = 0; i < n; i++) { idx[i] = rnd.nextInt(d.n); fs[i] = FastState.fromState(SearchLabelsLike.stateFrom(d.cells, idx[i]*81, d.role[idx[i]])); }
    double base = 0, mean = 0; for (int i = 0; i < n; i++) mean += d.outcome[idx[i]]; mean /= n; for (int i = 0; i < n; i++) base += Math.pow(d.outcome[idx[i]] - mean, 2); 
    System.out.printf("predittore costante: MSE %.4f%n", base / n);
    for (int k = 1; k < a.length; k++) {
      NeuralEvaluator ev = NeuralEvaluator.fromFile(a[k]);
      double mse = 0, sxy = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
      for (int i = 0; i < n; i++) { double v = ev.evaluate(fs[i]) / 1000.0, y = d.outcome[idx[i]]; mse += (v-y)*(v-y); sx += v; sy += y; sxy += v*y; sxx += v*v; syy += y*y; }
      double corr = (sxy/n - sx/n*sy/n) / Math.sqrt((sxx/n - sx*sx/n/n) * (syy/n - sy*sy/n/n));
      System.out.printf("%-58s MSE %.4f  corr %+.3f%n", a[k].substring(a[k].lastIndexOf('/')+1), mse / n, corr);
    }
  }
}
