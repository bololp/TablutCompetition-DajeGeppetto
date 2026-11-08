package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Costo di UNA valutazione: CNN contro NNUE (valutazione completa e incrementale) sullo STESSO carico.
 * <p>
 * Carico: per ciascuna posizione di un insieme (posizioni di partite reali, non terminali) e per ciascuna mossa legale:
 * {@code makeMove -> valuta il figlio -> unmakeMove}. E' quello che fa la ricerca alle foglie (un padre, tanti figli).
 * <ul>
 * <li><b>make/unmake</b>: solo l'esecuzione e l'annullamento delle mosse (costo comune, da sottrarre per avere il costo netto della valutazione);</li>
 * <li><b>CNN</b>: {@code evaluate(FastState)} dell'evaluator caricato, senza cache;</li>
 * <li><b>NNUE completa</b>: {@code evaluate(FastState)}: ricostruisce l'accumulatore dalla scacchiera;</li>
 * <li><b>NNUE incrementale</b>: una {@link NnueSession}: {@code reset} una volta per padre, poi per ogni figlio {@code push} + {@code evaluate}.</li>
 * </ul>
 * Le tre valutazioni NNUE danno lo stesso numero (si verifica: la somma dei valori deve coincidere bit per bit). I carichi sono eseguiti a
 * turno, dopo il riscaldamento del JIT, e si riporta il minimo e la mediana per figlio; un solo thread. In piu' si misurano i componenti
 * interni dell'NNUE (ricostruzione, aggiornamento incrementale, strati densi) per capire dove va il tempo.
 * <p>
 * <p>
 * Poi la RICERCA ({@link Negamax}, un thread, TT 2^20, stessi parametri di {@code SearchSpeedBench}) sulle stesse posizioni, con quattro
 * valutatori: CNN nuda; CNN con la cache delle valutazioni del client (2^22 voci, FREDDA a ogni passata: si scalda solo dentro la passata);
 * NNUE con la sola valutazione completa (involucro che nasconde il percorso incrementale); NNUE incrementale. A profondita' fissa si
 * misurano nodi e nodi/s (i nodi differiscono fra valutatori: pesi diversi potano diversamente); a tempo fisso la profondita' raggiunta.
 * <p>
 * Uso: {@code NnueBench [pesiCNN] [pesiNNUE|random] [posizioni=300] [ripetizioni=9] [seme=1] [posizioniRicerca=16] [msRicerca=500]}.
 */
public class NnueBench {

	private static double sink;

	public static void main(String[] args) throws Exception {
		final String cnnPath = args.length > 0 ? args[0] : "pesi_v4/cnn_t1500000_v4.txt";
		final String nnuePath = args.length > 1 ? args[1] : "random";
		final int nPos = args.length > 2 ? Integer.parseInt(args[2]) : 300;
		final int reps = args.length > 3 ? Integer.parseInt(args[3]) : 9;
		final long seed = args.length > 4 ? Long.parseLong(args[4]) : 1L;
		final int searchPos = args.length > 5 ? Integer.parseInt(args[5]) : 16;
		final int searchMs = args.length > 6 ? Integer.parseInt(args[6]) : 500;

		final Evaluator cnn = EvaluatorLoader.load(cnnPath);
		final NnueEvaluator nnue = nnuePath.equals("random") ? new NnueEvaluator(NnueNetwork.random(64, 16, 1)) : NnueEvaluator.fromFile(nnuePath);
		System.out.println("CNN : " + EvaluatorLoader.describe(cnn) + " (" + cnnPath + ")");
		System.out.println("NNUE: " + nnue.describe() + " (" + nnuePath + ")");
		System.out.println("JVM : " + System.getProperty("java.version") + ", " + Runtime.getRuntime().availableProcessors() + " processore/i visibile/i");

		final List<FastState> roots = new ArrayList<>();
		final Random rnd = new Random(seed);
		while (roots.size() < nPos) {
			final List<it.unibo.ai.didattica.competition.tablut.domain.State> ps = TestPositions.playGame(rnd, rnd.nextBoolean(), 90).positions;
			if (ps.size() > 8) {
				roots.add(FastState.fromState(ps.get(6 + rnd.nextInt(ps.size() - 6))));
			}
		}
		final int[][] moves = new int[nPos][];
		long children = 0;
		final int[] buf = new int[FastState.MAX_MOVES];
		for (int i = 0; i < nPos; i++) {
			final int n = roots.get(i).generateMoves(buf);
			moves[i] = Arrays.copyOf(buf, n);
			children += moves[i].length;
		}
		System.out.println(nPos + " posizioni, " + children + " figli per passata (" + String.format("%.1f", children / (double) nPos) + " mosse/posizione), " + reps + " ripetizioni");

		final UndoInfo u = new UndoInfo();
		final NnueSession ses = (NnueSession) nnue.newSession(4);
		final String[] names = { "make/unmake (comune)", "CNN, evaluate(FastState)", "NNUE completa, evaluate(FastState)", "NNUE incrementale (push+evaluate)" };
		final double[][] ns = new double[4][reps];
		// riscaldamento
		for (int w = 0; w < 4; w++) {
			for (int k = 0; k < 4; k++) {
				pass(k, roots, moves, cnn, nnue, ses, u);
			}
		}
		final double[] sums = new double[4];
		for (int r = 0; r < reps; r++) {
			for (int k = 0; k < 4; k++) {
				final long t0 = System.nanoTime();
				sums[k] = pass(k, roots, moves, cnn, nnue, ses, u);
				ns[k][r] = (System.nanoTime() - t0) / (double) children;
			}
			if (Double.doubleToLongBits(sums[2]) != Double.doubleToLongBits(sums[3])) {
				System.out.println("ERRORE: la somma dei valori NNUE completi (" + sums[2] + ") e incrementali (" + sums[3] + ") non coincide");
				System.exit(1);
			}
		}
		System.out.println("somma dei valori NNUE completa == incrementale (bit per bit): " + (sums[2] == sums[3]) + "; fallbackRebuilds=" + ses.fallbackRebuilds());
		final double base = median(ns[0]);
		System.out.println();
		System.out.printf("%-38s %12s %12s %14s%n", "carico (per figlio)", "min ns", "mediana ns", "netto ns (-mk)");
		for (int k = 0; k < 4; k++) {
			System.out.printf("%-38s %12.0f %12.0f %14s%n", names[k], min(ns[k]), median(ns[k]), k == 0 ? "-" : String.format("%.0f", median(ns[k]) - base));
		}
		final double cnnNet = median(ns[1]) - base, fullNet = median(ns[2]) - base, incNet = median(ns[3]) - base;
		System.out.println();
		System.out.printf("costo netto per valutazione: CNN %.0f ns | NNUE completa %.0f ns | NNUE incrementale %.0f ns%n", cnnNet, fullNet, incNet);
		System.out.printf("rapporti (mediane nette): CNN / NNUE incrementale = %.1fx, CNN / NNUE completa = %.1fx, NNUE completa / incrementale = %.2fx%n", cnnNet / incNet, cnnNet / fullNet, fullNet / incNet);
		System.out.printf("rapporti (mediane lorde, con make/unmake): CNN / NNUE incrementale = %.1fx%n", median(ns[1]) / median(ns[3]));

		// ---- dove va il tempo dell'NNUE
		final int[] acc = new int[nnue.net.hidden()];
		final int[] act = new int[nnue.net.scratchSize()];
		final int[] acc2 = new int[nnue.net.hidden()];
		final int reps2 = 2_000_000;
		for (int w = 0; w < 3; w++) {
			micro(nnue.net, roots.get(0), acc, acc2, act, 300_000);
		}
		final double[] mic = micro(nnue.net, roots.get(1), acc, acc2, act, reps2);
		System.out.println();
		System.out.printf("componenti NNUE (una posizione, %d iterazioni): refresh completo %.0f ns | spostamento incrementale %.0f ns | strati densi (%dx%d + %d, solo neuroni attivi) %.0f ns%n", reps2, mic[0],
				mic[1], nnue.net.l2(), nnue.net.hidden(), nnue.net.l2(), mic[2]);
		sink += acc[0] + act[0] + acc2[0];
		if (searchPos > 0) {
			searchBench(roots.subList(0, Math.min(searchPos, roots.size())), cnn, nnue, searchMs);
		}
		if (sink == 42.4242) {
			System.out.println(sink); // impedisce l'eliminazione del codice morto
		}
	}

	/** Espone la sola valutazione completa: {@link Negamax} non crea la sessione incrementale. */
	private static final class FullOnly implements Evaluator {
		private final Evaluator inner;

		FullOnly(Evaluator inner) {
			this.inner = inner;
		}

		@Override
		public double evaluate(FastState state) {
			return inner.evaluate(state);
		}
	}

	private static void searchBench(List<FastState> pos, Evaluator cnn, NnueEvaluator nnue, int ms) {
		final String[] names = { "CNN nuda", "CNN + cache valutazioni (2^22, fredda)", "NNUE, sola valutazione completa", "NNUE incrementale" };
		final int depth = 4;
		System.out.println();
		System.out.println("=== Ricerca: " + pos.size() + " posizioni, profondita' fissa " + depth + ", poi tempo fisso " + ms + " ms/posizione ===");
		final long[] nodes = new long[4];
		final double[][] nps = new double[4][3];
		final double[] hit = new double[4];
		for (int w = 0; w < 2; w++) { // riscaldamento del JIT (cache fredda, risultati scartati)
			for (int k = 0; k < 4; k++) {
				fixedDepth(evaluatorFor(k, cnn, nnue), pos, depth);
			}
		}
		for (int r = 0; r < 3; r++) {
			for (int k = 0; k < 4; k++) {
				final Evaluator ev = evaluatorFor(k, cnn, nnue);
				final double[] res = fixedDepth(ev, pos, depth);
				nodes[k] = (long) res[0];
				nps[k][r] = res[0] / res[1];
			}
		}
		final CachingEvaluator withStats = new CachingEvaluator(cnn, 22, true); // una passata a parte: le statistiche costano, il client non le attiva
		fixedDepth(withStats, pos, depth);
		hit[1] = withStats.hitRate();
		System.out.printf("%-42s %10s %12s %12s %12s%n", "profondita' fissa " + depth, "nodi", "nodi/s min", "mediana", "max");
		for (int k = 0; k < 4; k++) {
			System.out.printf("%-42s %10d %12.0f %12.0f %12.0f%s%n", names[k], nodes[k], min(nps[k]), median(nps[k]), max(nps[k]),
					k == 1 ? String.format("   (hit rate cache %.1f%%)", 100 * hit[k]) : "");
		}
		System.out.printf("nodi/s mediani: NNUE incrementale / CNN nuda = %.1fx, / CNN con cache = %.1fx, / NNUE completa = %.2fx%n", median(nps[3]) / median(nps[0]),
				median(nps[3]) / median(nps[1]), median(nps[3]) / median(nps[2]));
		System.out.println();
		System.out.printf("%-42s %16s %12s%n", "tempo fisso " + ms + " ms", "profondita' media", "nodi/s");
		for (int k = 0; k < 4; k++) {
			timed(evaluatorFor(k, cnn, nnue), pos, Math.min(ms, 200), 3); // riscaldamento
		}
		for (int k = 0; k < 4; k++) {
			final double[] t = timed(evaluatorFor(k, cnn, nnue), pos, ms, pos.size());
			System.out.printf("%-42s %16.2f %12.0f%n", names[k], t[0], t[1]);
		}
	}

	private static Evaluator evaluatorFor(int k, Evaluator cnn, NnueEvaluator nnue) {
		switch (k) {
		case 0:
			return cnn;
		case 1:
			return new CachingEvaluator(cnn, 22);
		case 2:
			return new FullOnly(nnue);
		default:
			return nnue;
		}
	}

	private static double[] fixedDepth(Evaluator ev, List<FastState> pos, int depth) {
		long nodes = 0;
		final long t0 = System.nanoTime();
		for (FastState root : pos) {
			final Negamax nm = new Negamax(ev, depth, true, 1 << 20).setThreads(1);
			nm.findBestMove(FastState.copyOf(root), 100_000_000L, PositionHistory.empty());
			nodes += nm.lastNodeCount;
		}
		return new double[] { nodes, (System.nanoTime() - t0) / 1e9 };
	}

	/** Profondita' media raggiunta e nodi/s, una istanza (TT fresca) per posizione; +300 ms: il margine di sicurezza del motore, come in SearchSpeedBench. */
	private static double[] timed(Evaluator ev, List<FastState> pos, int ms, int n) {
		long nodes = 0, elapsed = 0;
		double sumDepth = 0;
		for (int i = 0; i < Math.min(n, pos.size()); i++) {
			final Negamax nm = new Negamax(ev, 40, true, 1 << 20).setThreads(1);
			final long t0 = System.nanoTime();
			nm.findBestMove(FastState.copyOf(pos.get(i)), ms + 300, PositionHistory.empty());
			elapsed += (System.nanoTime() - t0) / 1_000_000;
			nodes += nm.lastNodeCount;
			sumDepth += nm.lastDepthReached;
		}
		final int m = Math.min(n, pos.size());
		return new double[] { sumDepth / m, nodes / Math.max(0.001, elapsed / 1000.0) };
	}

	private static double max(double[] a) {
		double m = -Double.MAX_VALUE;
		for (double v : a) {
			m = Math.max(m, v);
		}
		return m;
	}

	private static double pass(int kind, List<FastState> roots, int[][] moves, Evaluator cnn, NnueEvaluator nnue, NnueSession ses, UndoInfo u) {
		double s = 0;
		for (int i = 0; i < roots.size(); i++) {
			final FastState fs = roots.get(i);
			if (kind == 3) {
				ses.reset(fs);
			}
			final int[] mv = moves[i];
			for (int j = 0; j < mv.length; j++) {
				final int m = mv[j];
				fs.makeMove(m, u);
				switch (kind) {
				case 1:
					s += cnn.evaluate(fs);
					break;
				case 2:
					s += nnue.evaluate(fs);
					break;
				case 3:
					ses.push(1, m, fs, u);
					s += ses.evaluate(1, fs);
					break;
				default:
					s += fs.zobristHash & 1;
				}
				fs.unmakeMove(m, u);
			}
		}
		sink += s;
		return s;
	}

	private static double[] micro(NnueNetwork net, FastState fs, int[] acc, int[] acc2, int[] act, int n) {
		final int bP = NnueNetwork.featureIndex(1, 0), bQ = NnueNetwork.featureIndex(1, 5);
		long t0 = System.nanoTime();
		long x = 0;
		for (int i = 0; i < n; i++) {
			x += net.refresh(fs.board, acc);
		}
		final double refresh = (System.nanoTime() - t0) / (double) n;
		t0 = System.nanoTime();
		for (int i = 0; i < n; i++) {
			net.applyMove(acc, acc2, bP, bQ);
			x += acc2[i & 7];
		}
		final double delta = (System.nanoTime() - t0) / (double) n;
		t0 = System.nanoTime();
		for (int i = 0; i < n; i++) {
			acc[i & 7] ^= 1;
			x += net.logit(acc, act, i & 1);
		}
		final double dense = (System.nanoTime() - t0) / (double) n;
		sink += x;
		return new double[] { refresh, delta, dense };
	}

	private static double min(double[] a) {
		double m = Double.MAX_VALUE;
		for (double v : a) {
			m = Math.min(m, v);
		}
		return m;
	}

	private static double median(double[] a) {
		final double[] c = a.clone();
		Arrays.sort(c);
		return c.length % 2 == 1 ? c[c.length / 2] : 0.5 * (c[c.length / 2 - 1] + c[c.length / 2]);
	}
}
