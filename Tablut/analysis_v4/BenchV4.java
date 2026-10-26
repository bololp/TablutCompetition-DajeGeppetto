import java.util.*;
import it.unibo.ai.didattica.competition.tablut.training.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.domain.*;

public class BenchV4 {
  static double sink;
  public static void main(String[] a) throws Exception {
    long t0 = System.nanoTime();
    CnnPretrainer.Dataset d = DatasetUpgrade.loadAndRecompute(a[0]);
    System.out.printf("DatasetUpgrade su dataset reale: %d posizioni in %.1f s%n", d.n, (System.nanoTime()-t0)/1e9);
    int n = 20000; Random rnd = new Random(1);
    FastState[] fs = new FastState[n]; State[] st = new State[n];
    for (int i = 0; i < n; i++) { int k = rnd.nextInt(d.n); st[i] = SearchLabelsLike.stateFrom(d.cells, k*81, d.role[k]); fs[i] = FastState.fromState(st[i]); }
    String tmp = "/tmp/cnn_v4_bench.txt";
    WeightConverter.convertFile(a[1], tmp, -1);
    CnnEvaluator evNew = CnnEvaluator.fromFile(tmp);
    ConvNet.Loaded old = ConvNet.load(a[1]);
    double[] out = new double[47]; double[] lout = new double[44];
    TablutFeatures.Workspace ws = TablutFeatures.newWorkspace(); LegacyTablutFeaturesV3.Workspace lws = LegacyTablutFeaturesV3.newWorkspace();
    int[] masks = { 11, 123, 11 | 16, 11 | 32, 11 | 64, 11 | 16 | 32, 123 };
    String[] nm = { "v3 default (11)", "v4 default (123)", "v4 11+ESCAPE", "v4 11+SAFEFORK", "v4 11+GATES", "v4 11+ESCAPE+SAFEFORK", "v4 default (123) bis" };
    for (int pass = 0; pass < 2; pass++) {
      for (int m = 0; m < masks.length; m++) {
        int reps = 5; long t = System.nanoTime();
        for (int r = 0; r < reps; r++) for (int i = 0; i < n; i++) { if (m == 0) LegacyTablutFeaturesV3.extract(fs[i], lws, lout, masks[m]); else TablutFeatures.extract(fs[i], ws, out, masks[m]); sink += out[3]; }
        double us = (System.nanoTime() - t) / 1e3 / (reps * (double) n);
        if (pass == 1) System.out.printf("estrazione feature  %-24s %.2f us/posizione%n", nm[m], us);
      }
    }
    // valutazione completa
    ConvNet.Workspace wo = old.net.newWorkspace(); float[] fo = new float[41]; int[] oc = LegacyTablutFeaturesV3.columnsFor(old.featureGroups);
    for (int pass = 0; pass < 3; pass++) {
      long t = System.nanoTime(); for (int r = 0; r < 5; r++) for (int i = 0; i < n; i++) sink += evNew.evaluate(fs[i]);
      double usNew = (System.nanoTime() - t) / 1e3 / (5.0 * n);
      t = System.nanoTime();
      for (int r = 0; r < 5; r++) for (int i = 0; i < n; i++) { wo.setBoard(fs[i]); LegacyTablutFeaturesV3.extract(fs[i], lws, lout, old.featureGroups); for (int j = 0; j < 41; j++) fo[j] = (float) lout[oc[j]]; sink += old.net.forward(wo, fo, NeuralEvaluator.roleOf(fs[i])); }
      double usOld = (System.nanoTime() - t) / 1e3 / (5.0 * n);
      if (pass == 2) System.out.printf("valutazione completa CNN: v3 (41 feature) %.2f us | v4 convertita (44 feature) %.2f us  (%+.0f%%)%n", usOld, usNew, 100.0*(usNew-usOld)/usOld);
    }
    System.out.println("(sink " + (sink > 0) + ")");
  }
}
