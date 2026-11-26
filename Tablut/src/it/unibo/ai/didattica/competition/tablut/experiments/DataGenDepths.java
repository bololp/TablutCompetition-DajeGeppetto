package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.engine.*;
import it.unibo.ai.didattica.competition.tablut.training.NnueDataset;

public class DataGenDepths {
  public static void main(String[] a) throws Exception {
    if (a.length < 5 || a.length > 6) {
      System.err.println(
          "Uso: DataGenDepths <file-modello> <file-dataset> <numero-posizioni> "
              + "<profondita-min> <profondita-max> [numero-thread]");
      return;
    }

    NnueDataset d = NnueDataset.load(a[1]);
    int n = Integer.parseInt(a[2]);
    int minDepth = Integer.parseInt(a[3]);
    int maxDepth = Integer.parseInt(a[4]);
    int threads = a.length == 6
        ? Integer.parseInt(a[5])
        : Runtime.getRuntime().availableProcessors();

    if (n <= 0 || minDepth <= 0 || maxDepth < minDepth || threads <= 0) {
      throw new IllegalArgumentException(
          "Numero di posizioni, profondità e numero di thread devono essere validi.");
    }

    int[] depths = new int[maxDepth - minDepth + 1];
    for (int k = 0; k < depths.length; k++) {
      depths[k] = minDepth + k;
    }

    ExecutorService pool = Executors.newFixedThreadPool(threads);

    try {
      // Genera le posizioni in parallelo.
      FastState[] positions = new FastState[n];
      AtomicInteger generated = new AtomicInteger();
      List<Future<?>> generationJobs = new ArrayList<>();

      for (int worker = 0; worker < threads; worker++) {
        final int workerId = worker;

        generationJobs.add(pool.submit(() -> {
          Random rnd = new Random(21L + workerId);
          byte[] bb = new byte[81];

          while (generated.get() < n) {
            int i = rnd.nextInt(Math.min(d.n, 600000));

            if (d.ply(i) < 8 || d.ply(i) > 70 || Math.abs(d.value(i)) > 0.9) {
              continue;
            }

            d.copyBoard(i, bb, 0);
            FastState f = DataGenSpeed.make(bb, d.role(i) == 1);

            if (f.winner != FastState.WINNER_NONE || !Arrays.equals(f.board, bb)) {
              continue;
            }

            int slot = generated.getAndIncrement();
            if (slot < n) {
              positions[slot] = f;
            }
          }
        }));
      }

      for (Future<?> job : generationJobs) {
        job.get();
      }

      List<FastState> pos = Arrays.asList(positions);

      // Ogni worker mantiene il proprio evaluator.
      Evaluator[] evaluators = new Evaluator[threads];
      for (int worker = 0; worker < threads; worker++) {
        evaluators[worker] = NnueEvaluator.fromFile(a[0]);
      }

      double[][] v = new double[depths.length][n];
      double[] secs = new double[depths.length];

      // Per ogni profondità, i thread si dividono le posizioni.
      for (int k = 0; k < depths.length; k++) {
        long t0 = System.nanoTime();
        AtomicInteger nextPosition = new AtomicInteger();
        List<Future<?>> searchJobs = new ArrayList<>();

        for (int worker = 0; worker < threads; worker++) {
          final int workerId = worker;
          final int depthIndex = k;

          searchJobs.add(pool.submit(() -> {
            while (true) {
              int i = nextPosition.getAndIncrement();
              if (i >= n) {
                break;
              }

              Negamax ng = new Negamax(
                  evaluators[workerId],
                  depths[depthIndex],
                  true,
                  1 << 22,
                  true,
                  Negamax.ORDER_RECOMMENDED,
                  1);

              ng.setSafetyMarginMillis(0);
              ng.setPruning(PruningOptions.parse("all"));
              ng.findBestMove(
                  FastState.copyOf(pos.get(i)),
                  1_000_000_000L,
                  PositionHistory.empty());

              double sc = ng.lastScore;
              boolean white = pos.get(i).turn == BoardConstants.WHITE_TURN;
              double w = white ? sc : -sc;
              v[depthIndex][i] = Math.max(-1, Math.min(1, w / 1000.0));
            }
          }));
        }

        for (Future<?> job : searchJobs) {
          job.get();
        }

        secs[k] = (System.nanoTime() - t0) / 1e9 / n;
      }

      int referenceIndex = depths.length - 1;
      int referenceDepth = depths[referenceIndex];

      for (int k = 0; k < depths.length - 1; k++) {
        double se = 0;
        for (int i = 0; i < n; i++) {
          double diff = v[k][i] - v[referenceIndex][i];
          se += diff * diff;
        }

        System.out.printf(
            "profondita' %d (%.2f s/pos): RMS rispetto alla %d = %.3f%n",
            depths[k], secs[k], referenceDepth, Math.sqrt(se / n));
      }

      System.out.printf(
          "profondita' %d (%.2f s/pos)%n",
          referenceDepth, secs[referenceIndex]);

    } finally {
      pool.shutdown();
    }
  }
}

