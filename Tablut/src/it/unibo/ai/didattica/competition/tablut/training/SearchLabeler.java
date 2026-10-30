package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Etichettatore "valore di ricerca": per ogni posizione calcola il valore che darebbe {@link Negamax} a profondita' FISSA D con
 * lo stesso evaluator, e lo converte nel bersaglio di distillazione {@code y = clip(score / 1000, -0.98, 0.98)} dal punto di vista
 * del BIANCO. E' il generatore dei bersagli di {@link CnnDistiller} (idea TD-leaf / distillazione della ricerca).
 *
 * <h3>Convenzione di segno e prospettiva (verificata nel codice di {@link Negamax}, e dai test di {@code DistillCheck})</h3>
 * <ul>
 * <li>{@link Evaluator#evaluate} e' SEMPRE dal punto di vista del Bianco; nelle foglie Negamax usa {@code perspective * evaluate}.</li>
 * <li>Alla radice {@code searchRoot} calcola {@code score = -exploreChild(..., -mySign)}: {@code Negamax.lastScore} e' quindi dal
 * punto di vista di CHI MUOVE (+1 Bianco, -1 Nero). Il valore dal punto di vista del Bianco e' {@code mySign * lastScore}: la
 * conversione e' in {@link #whiteScore}. (Un'analisi che confrontasse {@code lastScore} con il valore statico senza girare il
 * segno per il Nero avrebbe i bersagli invertiti sulla meta' delle posizioni.)</li>
 * <li>Vittoria/sconfitta forzata: {@code +-(1e6 + profondita' residua)}; il clip a {@code +-0.98} le porta a un bersaglio finito
 * (atanh(0.98) = 2.30). Chi non ha mosse legali perde (come in {@code Negamax}).</li>
 * </ul>
 *
 * <h3>Riproducibilita' e parallelismo</h3>
 * Ogni posizione e' cercata da UN solo thread ({@code Negamax} a 1 thread: deterministico, a differenza di Lazy SMP) con un'istanza
 * FRESCA di {@code Negamax} (TT, killer e history vuote): senza, una voce di TT a profondita' maggiore lasciata dalla posizione
 * precedente potrebbe filtrare un valore piu' profondo di D, e l'etichetta dipenderebbe dall'ordine di elaborazione. Il
 * parallelismo e' solo TRA posizioni e il risultato e' scritto per indice: le etichette non dipendono dal numero di thread.
 * La ricerca non ha limite di tempo pratico (un'ora): se non completa la profondita' D (e non ha trovato una vittoria forzata, che
 * ferma l'approfondimento) e' un errore, non un'etichetta a profondita' ridotta.
 */
public final class SearchLabeler {

	/** Scala dei punteggi di {@link Evaluator}/{@link Negamax} (uscita rete in (-1,1) per 1000). */
	public static final double SCALE = 1000.0;
	/** Clip del bersaglio: tiene atanh(y) finito. */
	public static final double CLIP = 0.98;
	/** Soglia oltre la quale un punteggio e' una vittoria/sconfitta forzata (WIN_SCORE di {@link Negamax}). */
	static final double FORCED = 1_000_000;
	/** Tetto di tempo per posizione: non deve mai scattare (la profondita' e' fissa). */
	static final long MAX_MS = 3_600_000L;

	private SearchLabeler() {
	}

	// ====================================================================== conversioni (pure, testate)

	/** {@code y = clip(score / 1000, -0.98, 0.98)}; {@code whiteScore} e' dal punto di vista del Bianco. */
	public static double clippedTarget(double whiteScore) {
		return Math.max(-CLIP, Math.min(CLIP, whiteScore / SCALE));
	}

	/** Da punteggio "chi muove" ({@code Negamax.lastScore}) a punteggio dal punto di vista del Bianco. */
	public static double whiteScore(double sideToMoveScore, boolean whiteToMove) {
		return whiteToMove ? sideToMoveScore : -sideToMoveScore;
	}

	/** Vera se la posizione non e' un'etichetta valida: partita finita, oppure chi muove non ha mosse legali. */
	public static boolean isTerminal(FastState s) {
		return s.winner != FastState.WINNER_NONE || !s.hasLegalMove();
	}

	/** {@code FastState} da 81 codici (0 vuoto, 1 bianco, 2 nero, 3 re) e ruolo (0 = muove il Bianco). */
	public static FastState fastStateFromCodes(byte[] cells, int off, int role) {
		State s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				int code = cells[off + r * 9 + c];
				b[r][c] = code == 1 ? Pawn.WHITE : code == 2 ? Pawn.BLACK : code == 3 ? Pawn.KING : (r == 4 && c == 4 ? Pawn.THRONE : Pawn.EMPTY);
			}
		}
		s.setTurn(role == 0 ? Turn.WHITE : Turn.BLACK);
		return FastState.fromState(s);
	}

	// ====================================================================== una posizione

	/** Esito dell'etichettatura di una posizione. */
	public static final class Result {
		/** Valore di ricerca dal punto di vista del Bianco, scala evaluator (+-1e6 per le vittorie forzate). */
		public final double whiteScore;
		/** Bersaglio {@code clip(whiteScore / 1000, -0.98, 0.98)}. */
		public final double y;
		public final int depthReached;
		public final long nodes;

		Result(double whiteScore, double y, int depthReached, long nodes) {
			this.whiteScore = whiteScore;
			this.y = y;
			this.depthReached = depthReached;
			this.nodes = nodes;
		}

		/** Vero se la ricerca ha trovato una vittoria/sconfitta forzata (il bersaglio e' allora il clip). */
		public boolean forced() {
			return Math.abs(whiteScore) >= FORCED;
		}
	}

	/**
	 * Etichetta UNA posizione con una ricerca a 1 thread, profondita' fissa {@code depth}, istanza fresca di {@link Negamax}.
	 * La storia passata alla ricerca e' la sola radice, come fa il client all'inizio di un segmento senza catture.
	 *
	 * @param ttLog2 log2 del numero di voci della TT (16 e' abbondante fino a D=4)
	 * @throws IllegalArgumentException posizione terminale
	 * @throws IllegalStateException    la ricerca non ha raggiunto la profondita' richiesta
	 */
	public static Result label(Evaluator ev, FastState root, int depth, int ttLog2) {
		if (depth < 1) {
			throw new IllegalArgumentException("profondita' < 1");
		}
		if (isTerminal(root)) {
			throw new IllegalArgumentException("posizione terminale: nessuna etichetta di ricerca");
		}
		Negamax ng = new Negamax(ev, depth, true, 1 << ttLog2, true, Negamax.ORDER_DEFAULT, 1, false);
		ng.setSafetyMarginMillis(0);
		int mv = ng.findBestMove(root, MAX_MS, new long[] { root.zobristHash }, 1);
		if (mv == Move.NONE) {
			throw new IllegalArgumentException("nessuna mossa legale");
		}
		boolean whiteToMove = root.isWhiteToMove();
		double white = whiteScore(ng.lastScore, whiteToMove);
		boolean forced = Math.abs(ng.lastScore) >= FORCED;
		if (ng.lastDepthReached < depth && !(forced && ng.lastScore > 0)) {
			// una vittoria forzata di chi muove ferma l'approfondimento; in tutti gli altri casi D deve essere completata
			throw new IllegalStateException("ricerca incompleta: profondita' " + ng.lastDepthReached + " < " + depth);
		}
		return new Result(white, clippedTarget(white), ng.lastDepthReached, ng.lastNodeCount);
	}

	// ====================================================================== molte posizioni

	/** Etichette di un insieme di posizioni, nello stesso ordine di {@code idx}. */
	public static final class Labels {
		public final double[] y;
		public final double[] whiteScore;
		public final byte[] depthReached;
		public long nodes;
		public double seconds;
		public int forced;

		Labels(int n) {
			y = new double[n];
			whiteScore = new double[n];
			depthReached = new byte[n];
		}
	}

	/**
	 * Etichetta le posizioni {@code idx[k]} di un array piatto di scacchiere ({@code cells}, 81 codici per posizione) e ruoli.
	 * UNA ricerca per thread, parallelismo tra posizioni; il risultato non dipende da {@code threads}.
	 */
	public static Labels labelAll(final Evaluator ev, final byte[] cells, final byte[] role, final int[] idx, final int depth, int threads,
			final int ttLog2, final PrintStream log) throws Exception {
		final Labels out = new Labels(idx.length);
		final AtomicInteger next = new AtomicInteger();
		final AtomicInteger done = new AtomicInteger();
		final long t0 = System.nanoTime();
		final int T = Math.max(1, threads);
		ExecutorService pool = Executors.newFixedThreadPool(T);
		try {
			List<Future<long[]>> fs = new ArrayList<>();
			for (int t = 0; t < T; t++) {
				fs.add(pool.submit(new Callable<long[]>() {
					@Override
					public long[] call() {
						long nodes = 0, forced = 0;
						for (int k = next.getAndIncrement(); k < idx.length; k = next.getAndIncrement()) {
							final int i = idx[k];
							Result r = label(ev, fastStateFromCodes(cells, i * 81, role[i]), depth, ttLog2);
							out.y[k] = r.y;
							out.whiteScore[k] = r.whiteScore;
							out.depthReached[k] = (byte) r.depthReached;
							nodes += r.nodes;
							forced += r.forced() ? 1 : 0;
							int d = done.incrementAndGet();
							if (log != null && d % 5000 == 0) {
								log.printf("  etichettate %d/%d (%.0f s)%n", d, idx.length, (System.nanoTime() - t0) / 1e9);
							}
						}
						return new long[] { nodes, forced };
					}
				}));
			}
			for (Future<long[]> f : fs) {
				long[] r = f.get();
				out.nodes += r[0];
				out.forced += (int) r[1];
			}
		} finally {
			pool.shutdown();
		}
		out.seconds = (System.nanoTime() - t0) / 1e9;
		return out;
	}

	// ====================================================================== riga di comando

	/**
	 * Etichetta un campione di posizioni NON terminali di un dataset e scrive un CSV
	 * ({@code idx,role,static_white,search_white,y,depth}, punteggi su scala 1000, dal punto di vista del Bianco).
	 * <pre>
	 * SearchLabeler data=dati.gz weights=cnn.txt depth=3 n=20000 threads=4 seed=1 out=labels.csv
	 * </pre>
	 */
	public static void main(String[] args) throws Exception {
		Map<String, String> p = new HashMap<>();
		for (String a : args) {
			int eq = a.indexOf('=');
			if (eq > 0) {
				p.put(a.substring(0, eq), a.substring(eq + 1));
			}
		}
		if (!p.containsKey("data") || !p.containsKey("weights")) {
			System.out.println("uso: SearchLabeler data=dati.gz weights=pesi.txt [depth=3] [n=20000] [threads=tutti] [seed=1] [tt=16] [out=labels.csv]");
			System.exit(2);
		}
		int depth = Integer.parseInt(p.getOrDefault("depth", "3"));
		int n = Integer.parseInt(p.getOrDefault("n", "20000"));
		int threads = Integer.parseInt(p.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
		long seed = Long.parseLong(p.getOrDefault("seed", "1"));
		int tt = Integer.parseInt(p.getOrDefault("tt", "16"));
		String out = p.getOrDefault("out", "labels.csv");
		CnnPretrainer.Dataset d = DistillSet.loadDataset(p.get("data"));
		Evaluator ev = EvaluatorLoader.load(p.get("weights"));
		Random rnd = new Random(seed);
		int[] perm = new int[d.n];
		for (int i = 0; i < d.n; i++) {
			perm[i] = i;
		}
		for (int i = d.n - 1; i > 0; i--) {
			int j = rnd.nextInt(i + 1);
			int t = perm[i];
			perm[i] = perm[j];
			perm[j] = t;
		}
		List<Integer> sel = new ArrayList<>();
		for (int k = 0; k < d.n && sel.size() < n; k++) {
			if (!isTerminal(fastStateFromCodes(d.cells, perm[k] * 81, d.role[perm[k]]))) {
				sel.add(perm[k]);
			}
		}
		int[] idx = new int[sel.size()];
		for (int k = 0; k < idx.length; k++) {
			idx[k] = sel.get(k);
		}
		System.out.println("=== SearchLabeler === " + EvaluatorLoader.describe(ev) + " | D=" + depth + " | " + idx.length + " posizioni non terminali | threads=" + threads);
		Labels L = labelAll(ev, d.cells, d.role, idx, depth, threads, tt, System.out);
		double absGap = 0;
		try (PrintWriter w = new PrintWriter(new FileWriter(out))) {
			w.println("idx,role,static_white,search_white,y,depth");
			for (int k = 0; k < idx.length; k++) {
				double st = ev.evaluate(fastStateFromCodes(d.cells, idx[k] * 81, d.role[idx[k]]));
				absGap += Math.abs(st - Math.max(-SCALE * CLIP, Math.min(SCALE * CLIP, L.whiteScore[k])));
				w.println(idx[k] + "," + d.role[idx[k]] + "," + st + "," + L.whiteScore[k] + "," + L.y[k] + "," + L.depthReached[k]);
			}
		}
		System.out.printf("fatte %d etichette in %.1f s (%.1f ms/posizione/thread, %d nodi/posizione, %d vittorie/sconfitte forzate) -> %s%n", idx.length, L.seconds,
				1000.0 * L.seconds * threads / Math.max(1, idx.length), L.nodes / Math.max(1, idx.length), L.forced, out);
		System.out.printf("scarto medio |statico - ricerca| (scala 1000, ricerca clippata a +-980): %.1f%n", absGap / Math.max(1, idx.length));
	}
}
