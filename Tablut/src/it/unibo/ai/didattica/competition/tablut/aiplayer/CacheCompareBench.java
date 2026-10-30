package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Confronto nella STESSA sessione fra: nessuna cache, cache a solo hash ({@link HashCachingEvaluator}, versione originale) e cache a
 * chiave esatta ({@link CachingEvaluator}, versione finale) a parita' di nodi e di decisioni.
 * <pre>
 *   CacheCompareBench game  &lt;simple|mlp|cnn&gt; &lt;profondita'&gt; [ripetizioni=3] [plyMax=60]   una partita con Negamax persistente per lato
 *   CacheCompareBench micro                                                              costo del singolo lookup (hit/miss) delle due cache
 * </pre>
 * Pesi: proprieta' di sistema {@code -Dtablut.weights=cartella/} (default {@code pesi_v4/}, deve finire con '/').
 * Le modalita' sono ruotate a ogni ripetizione per attenuare deriva e rumore; si riportano minimo e mediana.
 */
public class CacheCompareBench {

	static final String W = System.getProperty("tablut.weights", "pesi_v4/");
	static volatile double sink;

	static final String[] NAMES = { "nessuna cache", "HASH  2^22 (64 MiB)", "ESATTA 2^21 (64 MiB)", "ESATTA 2^22 (128 MiB)" };

	static Evaluator make(int mode, Evaluator base) {
		switch (mode) {
		case 1:
			return new HashCachingEvaluator(base, 22, true);
		case 2:
			return new CachingEvaluator(base, 21, true);
		case 3:
			return new CachingEvaluator(base, 22, true);
		default:
			return base;
		}
	}

	/** @return {secondi, nodi, hit%, ply, checksum delle mosse} */
	static double[] game(Evaluator base, int depth, int mode, int order, int maxPly) {
		FastState cur = new FastState();
		Evaluator[] ev = { make(mode, base), make(mode, base) };
		Negamax[] nm = { new Negamax(ev[0], depth, true, 1 << 22, true, order, 1), new Negamax(ev[1], depth, true, 1 << 22, true, order, 1) };
		ArrayList<Long> hist = new ArrayList<>();
		hist.add(cur.zobristHash);
		int ply = 0;
		long nodes = 0, sum = 0;
		final long t0 = System.nanoTime();
		while (!cur.isGameOver() && ply < maxPly) {
			int side = cur.turn;
			long[] h = new long[hist.size()];
			for (int i = 0; i < h.length; i++) {
				h[i] = hist.get(i);
			}
			int mv = nm[side].findBestMove(cur, 1_000_000_000L, h, h.length);
			if (mv == Move.NONE) {
				break;
			}
			nodes += nm[side].lastNodeCount;
			sum = sum * 31 + mv;
			int before = cur.whiteCount + cur.blackCount;
			cur.makeMove(mv, new UndoInfo());
			if (cur.whiteCount + cur.blackCount < before) {
				hist.clear();
			}
			hist.add(cur.zobristHash);
			ply++;
		}
		final double secs = (System.nanoTime() - t0) / 1e9;
		long calls = 0, hits = 0;
		for (Evaluator e : ev) {
			if (e instanceof HashCachingEvaluator) {
				calls += ((HashCachingEvaluator) e).calls();
				hits += ((HashCachingEvaluator) e).hits();
			} else if (e instanceof CachingEvaluator) {
				calls += ((CachingEvaluator) e).calls();
				hits += ((CachingEvaluator) e).hits();
			}
		}
		return new double[] { secs, nodes, calls == 0 ? 0 : 100.0 * hits / calls, ply, sum };
	}

	static void gameMode(String which, int depth, int reps, int maxPly) throws Exception {
		Evaluator base = which.equals("simple") ? new SimpleEvaluator()
			: EvaluatorLoader.load(W + (which.equals("cnn") ? "cnn_conv4_hybrid_pretrained_v4.txt" : "mlp_v3_pretrained_v4.txt"));
		final int rec = OrderingSpec.parse("rec");
		for (int m = 0; m < 4; m++) {
			game(base, Math.min(depth, 3), m, rec, 20); // scaldamento JIT di tutte le classi
		}
		double[][] t = new double[4][reps];
		double[] hit = new double[4], nodes = new double[4], chk = new double[4];
		for (int r = 0; r < reps; r++) {
			for (int k = 0; k < 4; k++) {
				int m = (k + r) % 4;
				double[] x = game(base, depth, m, rec, maxPly);
				t[m][r] = x[0];
				hit[m] = x[2];
				nodes[m] = x[1];
				chk[m] = x[4];
			}
		}
		System.out.printf("%s d=%d, order=rec, %d ripetizioni, plyMax=%d (a parita' di nodi e decisioni)%n", which, depth, reps, maxPly);
		double[] off = t[0].clone();
		Arrays.sort(off);
		for (int m = 0; m < 4; m++) {
			double[] s = t[m].clone();
			Arrays.sort(s);
			System.out.printf("  %-22s min %6.2fs  med %6.2fs  (%+5.1f%% min, %+5.1f%% med)  hit %5.1f%%  nodi %d%n", NAMES[m], s[0], s[reps / 2],
				100 * (s[0] - off[0]) / off[0], 100 * (s[reps / 2] - off[reps / 2]) / off[reps / 2], hit[m], (long) nodes[m]);
		}
		boolean same = true;
		for (int m = 1; m < 4; m++) {
			same &= nodes[m] == nodes[0] && chk[m] == chk[0];
		}
		System.out.println("  stesse mosse e stessi nodi in tutte le modalita': " + (same ? "si" : "NO (differenza: da indagare!)"));
	}

	// ---- micro: costo del lookup con evaluator interno costante -------------------------------------------------------------------
	static final class Const implements Evaluator {
		@Override
		public double evaluate(FastState s) {
			return 1.5;
		}
	}

	static void fill(FastState s, byte[] base, long i) {
		System.arraycopy(base, 0, s.board, 0, 81);
		long h = (i + 1) * 0x9E3779B97F4A7C15L;
		h ^= h >>> 31;
		h *= 0xBF58476D1CE4E5B9L;
		final byte[] codes = { 0, 1, 2, 4 };
		for (int k = 0; k < 6; k++) {
			int pos = (int) (((h >>> (k * 9)) & 0x1FF) % 81);
			s.board[pos] = codes[(int) ((h >>> (50 + k)) & 3)];
		}
		s.turn = (byte) (i & 1);
		s.zobristHash = s.computeHash(); // la versione a solo hash ne ha bisogno; costo identico per tutte, sottratto nel baseline
	}

	static void micro() {
		final byte[] base = new FastState().board.clone();
		final FastState s = new FastState();
		final int N = 1 << 21, ITER = 3_000_000;
		Evaluator[] ev = { new HashCachingEvaluator(new Const(), 22), new CachingEvaluator(new Const(), 21), new CachingEvaluator(new Const(), 22) };
		String[] nm = { "HASH  2^22 (64 MiB)", "ESATTA 2^21 (64 MiB)", "ESATTA 2^22 (128 MiB)" };
		double fillNs = 1e18;
		for (int rep = 0; rep < 5; rep++) {
			long x = 12345;
			long t0 = System.nanoTime();
			double acc = 0;
			for (int k = 0; k < ITER; k++) {
				x = x * 6364136223846793005L + 1442695040888963407L;
				fill(s, base, (x >>> 33) % N);
				acc += s.board[3];
			}
			sink += acc;
			fillNs = Math.min(fillNs, (System.nanoTime() - t0) / (double) ITER);
		}
		System.out.printf("preparazione chiave (da sottrarre): %.0f ns%n", fillNs);
		for (int e = 0; e < ev.length; e++) {
			for (int i = 0; i < N; i++) { // precarica N chiavi distinte
				fill(s, base, i);
				ev[e].evaluate(s);
			}
			double hitDram = 1e18, hitRes = 1e18, miss = 1e18;
			for (int rep = 0; rep < 5; rep++) {
				long x = 12345;
				long t0 = System.nanoTime();
				double acc = 0;
				for (int k = 0; k < ITER; k++) {
					x = x * 6364136223846793005L + 1442695040888963407L;
					fill(s, base, (x >>> 33) % N);
					acc += ev[e].evaluate(s);
				}
				sink += acc;
				hitDram = Math.min(hitDram, (System.nanoTime() - t0) / (double) ITER - fillNs);
				x = 12345;
				t0 = System.nanoTime();
				acc = 0;
				for (int k = 0; k < ITER; k++) {
					x = x * 6364136223846793005L + 1442695040888963407L;
					fill(s, base, (x >>> 33) % 2000); // 2000 chiavi: restano nella cache della CPU
					acc += ev[e].evaluate(s);
				}
				sink += acc;
				hitRes = Math.min(hitRes, (System.nanoTime() - t0) / (double) ITER - fillNs);
				Evaluator fresh = e == 0 ? new HashCachingEvaluator(new Const(), 22) : new CachingEvaluator(new Const(), e == 1 ? 21 : 22);
				t0 = System.nanoTime();
				acc = 0;
				for (int k = 0; k < ITER; k++) {
					fill(s, base, (long) N + k);
					acc += fresh.evaluate(s);
				}
				sink += acc;
				miss = Math.min(miss, (System.nanoTime() - t0) / (double) ITER - fillNs);
			}
			System.out.printf("  %-22s hit (accesso casuale in tabella) %4.0f ns | hit (voce in cache CPU) %4.0f ns | miss + scrittura %4.0f ns%n", nm[e], hitDram, hitRes, miss);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length >= 1 && args[0].equals("micro")) {
			micro();
		} else if (args.length >= 3 && args[0].equals("game")) {
			gameMode(args[1], Integer.parseInt(args[2]), args.length > 3 ? Integer.parseInt(args[3]) : 3, args.length > 4 ? Integer.parseInt(args[4]) : 60);
		} else {
			System.out.println("uso: CacheCompareBench game <simple|mlp|cnn> <profondita'> [ripetizioni] [plyMax]  |  CacheCompareBench micro");
		}
	}
}
