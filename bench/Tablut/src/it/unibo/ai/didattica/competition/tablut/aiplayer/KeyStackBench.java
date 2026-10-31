package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Confronto nella STESSA sessione, a parita' di nodi e di decisioni, fra le combinazioni di transposition table e cache delle valutazioni:
 * <pre>
 *   KeyStackBench game &lt;simple|mlp|cnn&gt; &lt;profondita'&gt; [ripetizioni=3] [plyMax=60]    una partita con Negamax persistente per lato
 *   KeyStackBench micro                                                              costo del singolo probe/store della TT e del lookup della cache
 * </pre>
 * Modalita' (TT, cache): 0 hash/nessuna | 1 hash/cache SOLO HASH (versione originale) | 2 hash/cache a chiave esatta ricodificata (versione
 * consegnata prima) | 3 hash/cache con chiave dallo stato | 4 TT esatta/nessuna | 5 TT esatta/cache con chiave dallo stato (chiave unica).
 * Pesi: {@code -Dtablut.weights=cartella/} (default {@code pesi_v4/}). Le modalita' sono ruotate a ogni ripetizione; si riportano minimo e mediana.
 */
public class KeyStackBench {

	static final String W = System.getProperty("tablut.weights", "pesi_v4/");
	static volatile double sink;

	static final String[] NAMES = {
		"TT hash,   nessuna cache",
		"TT hash,   cache SOLO HASH 2^22 (64 MiB)       [originale]",
		"TT hash,   cache esatta ricodificata 2^22 (128 MiB) [consegnata]",
		"TT hash,   cache esatta dallo stato 2^22 (128 MiB)",
		"TT esatta, nessuna cache",
		"TT esatta, cache esatta dallo stato 2^22 (128 MiB) [chiave unica]" };
	static final boolean[] EXACT_TT = { false, false, false, false, true, true };

	static Evaluator cache(int mode, Evaluator base) {
		switch (mode) {
		case 1:
			return new HashCachingEvaluator(base, 22, true);
		case 2:
			return new CachingEvaluator(base, 22, true, false);
		case 3:
		case 5:
			return new CachingEvaluator(base, 22, true, true);
		default:
			return base;
		}
	}

	/** @return {secondi, nodi, hit%, ply, checksum delle mosse} */
	static double[] game(Evaluator base, int depth, int mode, int order, int maxPly) {
		FastState cur = new FastState();
		Evaluator[] ev = { cache(mode, base), cache(mode, base) };
		Negamax[] nm = new Negamax[2];
		for (int i = 0; i < 2; i++) {
			TranspositionTable table = EXACT_TT[mode] ? new ExactTranspositionTable(1 << 22, false) : null;
			nm[i] = new Negamax(ev[i], depth, true, 1 << 22, true, order, 1, false, table);
		}
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
		final int M = NAMES.length;
		for (int m = 0; m < M; m++) {
			game(base, Math.min(depth, 3), m, rec, 20); // scaldamento JIT di tutte le classi
		}
		double[][] t = new double[M][reps];
		double[] hit = new double[M], nodes = new double[M], chk = new double[M];
		for (int r = 0; r < reps; r++) {
			for (int k = 0; k < M; k++) {
				int m = (k + r) % M;
				double[] x = game(base, depth, m, rec, maxPly);
				t[m][r] = x[0];
				hit[m] = x[2];
				nodes[m] = x[1];
				chk[m] = x[4];
			}
		}
		System.out.printf("%s d=%d, order=rec, %d ripetizioni, plyMax=%d (TT 2^22 voci: 96 MiB hash, 160 MiB esatta)%n", which, depth, reps, maxPly);
		double[] off = t[0].clone();
		Arrays.sort(off);
		for (int m = 0; m < M; m++) {
			double[] s = t[m].clone();
			Arrays.sort(s);
			System.out.printf("  %-70s min %6.2fs  med %6.2fs  (%+5.1f%% min, %+5.1f%% med)  hit %5.1f%%%n", NAMES[m], s[0], s[reps / 2],
				100 * (s[0] - off[0]) / off[0], 100 * (s[reps / 2] - off[reps / 2]) / off[reps / 2], hit[m]);
		}
		boolean same = true;
		for (int m = 1; m < M; m++) {
			same &= nodes[m] == nodes[0] && chk[m] == chk[0];
		}
		System.out.println("  nodi totali " + (long) nodes[0] + "; stesse mosse e stessi nodi in tutte le modalita': " + (same ? "si" : "NO (da indagare!)"));
	}

	// ---- micro ---------------------------------------------------------------------------------------------------------------------------------------------
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
		s.zobristHash = s.computeHash();
		s.computeKeys();
	}

	static void micro() {
		final byte[] base = new FastState().board.clone();
		final FastState s = new FastState();
		final int N = 1 << 21, ITER = 2_000_000;
		double fillNs = 1e18;
		for (int rep = 0; rep < 4; rep++) {
			long x = 12345, t0 = System.nanoTime();
			double acc = 0;
			for (int k = 0; k < ITER; k++) {
				x = x * 6364136223846793005L + 1442695040888963407L;
				fill(s, base, (x >>> 33) % N);
				acc += s.board[3];
			}
			sink += acc;
			fillNs = Math.min(fillNs, (System.nanoTime() - t0) / (double) ITER);
		}
		System.out.printf("preparazione posizione (da sottrarre): %.0f ns%n", fillNs);
		// ---- TT: probe (hit, accesso casuale) e store (voce nuova) ----
		TranspositionTable h = new TranspositionTable(1 << 22, false);
		ExactTranspositionTable x2 = new ExactTranspositionTable(1 << 22, false);
		TranspositionTable.ProbeResult pr = new TranspositionTable.ProbeResult();
		for (int i = 0; i < N; i++) {
			fill(s, base, i);
			h.store(s, 6, 1.0, TranspositionTable.EXACT, 5);
			x2.store(s, 6, 1.0, TranspositionTable.EXACT, 5);
		}
		TranspositionTable[] tts = { h, x2 };
		String[] tn = { "TT hash  (2^22 voci, 96 MiB) ", "TT esatta (2^22 voci, 160 MiB)" };
		for (int e = 0; e < 2; e++) {
			double probeHit = 1e18, probeMiss = 1e18, storeNew = 1e18;
			for (int rep = 0; rep < 4; rep++) {
				long x = 12345, t0 = System.nanoTime();
				double acc = 0;
				for (int k = 0; k < ITER; k++) {
					x = x * 6364136223846793005L + 1442695040888963407L;
					fill(s, base, (x >>> 33) % N);
					tts[e].probeInto(s, 0, pr);
					acc += pr.score;
				}
				sink += acc;
				probeHit = Math.min(probeHit, (System.nanoTime() - t0) / (double) ITER - fillNs);
				t0 = System.nanoTime();
				acc = 0;
				for (int k = 0; k < ITER; k++) {
					fill(s, base, (long) N * 4 + k); // mai viste
					tts[e].probeInto(s, 0, pr);
					acc += pr.score;
				}
				sink += acc;
				probeMiss = Math.min(probeMiss, (System.nanoTime() - t0) / (double) ITER - fillNs);
				TranspositionTable fresh = e == 0 ? new TranspositionTable(1 << 22, false) : new ExactTranspositionTable(1 << 22, false);
				t0 = System.nanoTime();
				for (int k = 0; k < ITER; k++) {
					fill(s, base, k);
					fresh.store(s, 6, 1.0, TranspositionTable.EXACT, 5);
				}
				storeNew = Math.min(storeNew, (System.nanoTime() - t0) / (double) ITER - fillNs);
			}
			System.out.printf("  %s probe hit %4.0f ns | probe miss %4.0f ns | store (voce nuova) %4.0f ns%n", tn[e], probeHit, probeMiss, storeNew);
		}
		// ---- cache: hit casuale / miss+scrittura, con chiave ricodificata o dallo stato ----
		Evaluator[] ev = { new HashCachingEvaluator(new Const(), 22), new CachingEvaluator(new Const(), 22, false, false), new CachingEvaluator(new Const(), 22, false, true) };
		String[] cn = { "cache SOLO HASH (64 MiB)          ", "cache esatta ricodificata (128 MiB)", "cache esatta dallo stato (128 MiB) " };
		for (int e = 0; e < ev.length; e++) {
			for (int i = 0; i < N; i++) {
				fill(s, base, i);
				ev[e].evaluate(s);
			}
			double hit = 1e18, miss = 1e18;
			for (int rep = 0; rep < 4; rep++) {
				long x = 12345, t0 = System.nanoTime();
				double acc = 0;
				for (int k = 0; k < ITER; k++) {
					x = x * 6364136223846793005L + 1442695040888963407L;
					fill(s, base, (x >>> 33) % N);
					acc += ev[e].evaluate(s);
				}
				sink += acc;
				hit = Math.min(hit, (System.nanoTime() - t0) / (double) ITER - fillNs);
				Evaluator fresh = e == 0 ? new HashCachingEvaluator(new Const(), 22) : new CachingEvaluator(new Const(), 22, false, e == 2);
				t0 = System.nanoTime();
				acc = 0;
				for (int k = 0; k < ITER; k++) {
					fill(s, base, (long) N + k);
					acc += fresh.evaluate(s);
				}
				sink += acc;
				miss = Math.min(miss, (System.nanoTime() - t0) / (double) ITER - fillNs);
			}
			System.out.printf("  %s hit %4.0f ns | miss + scrittura %4.0f ns%n", cn[e], hit, miss);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length >= 1 && args[0].equals("micro")) {
			micro();
		} else if (args.length >= 3 && args[0].equals("game")) {
			gameMode(args[1], Integer.parseInt(args[2]), args.length > 3 ? Integer.parseInt(args[3]) : 3, args.length > 4 ? Integer.parseInt(args[4]) : 60);
		} else {
			System.out.println("uso: KeyStackBench game <simple|mlp|cnn> <profondita'> [ripetizioni] [plyMax]  |  KeyStackBench micro");
		}
	}
}
