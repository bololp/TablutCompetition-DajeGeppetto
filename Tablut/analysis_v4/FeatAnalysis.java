import java.util.*;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;

public class FeatAnalysis {
  public static void main(String[] a) throws Exception {
    CnnPretrainer.Dataset d = CnnPretrainer.Dataset.load(a[0]);
    ConvNet.Loaded L = ConvNet.load(a[1]);
    ConvNet net = L.net;
    int[] cols = TablutFeatures.columnsFor(L.featureGroups);
    int nF = cols.length, NF = TablutFeatures.NUM_FEATURES;
    int n = Math.min(d.n, 40000);
    System.out.println("n=" + d.n + " usate=" + n + " nFeat rete=" + nF + " cnn params=" + net.numParams());

    // --- statistiche per feature (tutte le 44)
    double[] mean = new double[NF], sd = new double[NF], nz = new double[NF];
    for (int j = 0; j < NF; j++) {
      double s = 0, s2 = 0; int z = 0;
      for (int i = 0; i < d.n; i++) { double v = d.feats[i*NF+j]; s += v; s2 += v*v; if (v != 0) z++; }
      mean[j] = s / d.n; sd[j] = Math.sqrt(Math.max(0, s2/d.n - mean[j]*mean[j])); nz[j] = (double) z / d.n;
    }

    // --- redondanza: R^2 di ogni feature (tra quelle usate) sulle altre, normal equations + ridge minuscolo
    int[] use = cols;
    int m = use.length;
    double[][] C = new double[m+1][m+1]; // con intercetta
    for (int i = 0; i < d.n; i++) {
      double[] x = new double[m+1]; x[m] = 1;
      for (int j = 0; j < m; j++) x[j] = d.feats[i*NF+use[j]];
      for (int p = 0; p <= m; p++) for (int q = 0; q <= m; q++) C[p][q] += x[p]*x[q];
    }
    double[] r2 = new double[m];
    for (int t = 0; t < m; t++) {
      // regredisci feature t sulle altre
      int k = m; // m-1 altre + intercetta
      double[][] A = new double[k][k]; double[] bb = new double[k];
      int[] idx = new int[k]; int c = 0;
      for (int j = 0; j <= m; j++) if (j != t) idx[c++] = j;
      for (int p = 0; p < k; p++) { bb[p] = C[idx[p]][t]; for (int q = 0; q < k; q++) A[p][q] = C[idx[p]][idx[q]]; A[p][p] += 1e-6 * d.n; }
      double[] w = solve(A, bb);
      // R2 = 1 - SSres/SStot ; SSres = C[t][t] - w.b (per le normal equations)
      double ssres = C[t][t]; for (int p = 0; p < k; p++) ssres -= w[p]*bb[p];
      double sstot = C[t][t] - C[t][m]*C[t][m]/d.n;
      r2[t] = sstot > 1e-9 ? 1 - ssres / sstot : Double.NaN;
    }

    // --- importanza per permutazione sulla rete
    ConvNet.Workspace ws = net.newWorkspace();
    float[][] F = new float[n][nF];
    double[] base = new double[n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < nF; j++) F[i][j] = d.feats[i*NF+use[j]];
      ws.setCodes(d.cells, i*81, 0);
      base[i] = net.forward(ws, F[i], d.role[i]);
    }
    double bs = 0, bm = 0; for (double v : base) { bm += v; } bm /= n; for (double v : base) bs += (v-bm)*(v-bm);
    System.out.printf("uscita rete: media %.3f sd %.3f%n", bm, Math.sqrt(bs/n));
    Random rnd = new Random(7);
    double[] imp = new double[nF];
    for (int j = 0; j < nF; j++) {
      int[] perm = new int[n]; for (int i = 0; i < n; i++) perm[i] = i;
      for (int i = n-1; i > 0; i--) { int k = rnd.nextInt(i+1); int t = perm[i]; perm[i] = perm[k]; perm[k] = t; }
      double s = 0;
      float[] f = new float[nF];
      for (int i = 0; i < n; i++) {
        System.arraycopy(F[i], 0, f, 0, nF);
        f[j] = F[perm[i]][j];
        ws.setCodes(d.cells, i*81, 0);
        double v = net.forward(ws, f, d.role[i]);
        s += (v-base[i])*(v-base[i]);
      }
      imp[j] = Math.sqrt(s/n);
    }
    // colonne denso: norma L2 per feature (in uscita dal primo denso) * sd feature
    ConvNet.Config cfg = net.config();
    float[] P = net.params();
    int in = cfg.denseIn(), out = cfg.dense[0], fs = cfg.flatSize();
    double[] colNorm = new double[nF];
    for (int j = 0; j < nF; j++) { double s = 0; for (int o = 0; o < out; o++) { double w = P[net.denseW[0] + o*in + fs + j]; s += w*w; } colNorm[j] = Math.sqrt(s); }

    Integer[] ord = new Integer[nF]; for (int j = 0; j < nF; j++) ord[j] = j;
    Arrays.sort(ord, (x,y) -> Double.compare(imp[y], imp[x]));
    System.out.println("\nrank | importanza(RMS dV)| |w| denso | sd   | %!=0 | R2 vs altre | feature");
    int rk = 1;
    for (int jj : ord) {
      int f = use[jj];
      System.out.printf("%3d  | %.4f (x1000=%.0f) | %.3f | %.3f | %.2f | %.4f | %s%n", rk++, imp[jj], imp[jj]*1000, colNorm[jj], sd[f], nz[f], r2[jj], TablutFeatures.NAMES[f]);
    }
  }
  static double[] solve(double[][] A, double[] b) {
    int n = b.length; double[][] M = new double[n][n+1];
    for (int i = 0; i < n; i++) { System.arraycopy(A[i], 0, M[i], 0, n); M[i][n] = b[i]; }
    for (int c = 0; c < n; c++) {
      int p = c; for (int r = c+1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      double[] t = M[c]; M[c] = M[p]; M[p] = t;
      double pv = M[c][c]; if (Math.abs(pv) < 1e-12) continue;
      for (int r = 0; r < n; r++) if (r != c) { double f = M[r][c]/pv; if (f != 0) for (int k = c; k <= n; k++) M[r][k] -= f*M[c][k]; }
    }
    double[] x = new double[n]; for (int i = 0; i < n; i++) x[i] = Math.abs(M[i][i]) < 1e-12 ? 0 : M[i][n]/M[i][i]; return x;
  }
}
