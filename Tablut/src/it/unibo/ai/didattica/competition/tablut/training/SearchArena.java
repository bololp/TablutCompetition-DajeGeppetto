package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Mcts;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MctsConfig;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NegamaxSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Arena che confronta due ALGORITMI DI RICERCA (Negamax e/o MCTS, con qualunque evaluator) a TEMPO REALE uguale per mossa, con le regole del server:
 * una posizione che si ripete dall'ultima cattura e' un pareggio, chi non ha mosse perde, e a entrambi i lati si passa la storia di ripetizione (come fa
 * il client). Le regole di fine partita sono quelle di {@link GameTracker} (gia' verificate da {@code SelfPlayRulesCheck}).
 * <p>
 * Lo stesso protocollo di {@link EqualTimeArena} (che resta invariata: i suoi risultati precedenti restano riproducibili): ogni coppia di partite parte da
 * 4 semi-mosse casuali e viene giocata due volte a colori scambiati, tetto a 160 semi-mosse = pareggio. Differenze: le regole di ripetizione e la storia,
 * e il TEMPO. Qui {@code ms} e' il tempo di ricerca REALE di ogni mossa per entrambi i lati (il margine di sicurezza e' 0 a meno di {@code m=}; con
 * {@code EqualTimeArena} il budget e' ridotto di 400 ms dal margine di Negamax, con minimo 50 ms: una trappola gia' costata un risultato sbagliato nel
 * README_CNN). Per questo {@code ms} dovrebbe essere almeno 50.
 *
 * <pre>
 * Uso: SearchArena &lt;A&gt; &lt;B&gt; [pairs=30] [ms=200] [seed=1] [par=1] [threads=1]
 *   A, B = [modificatori:]evaluator
 *     evaluator    simple | percorso di un file di pesi (MLP o CNN)
 *     modificatori mcts | mcts=chiave=valore,...  (MctsConfig)   negamax | neg   ord=NOME (ordinamento di Negamax)   tN (thread)   m=MS (margine)
 *   senza 'mcts' si gioca Negamax con l'ordinamento di default del client.
 *   par = partite giocate in parallelo (usare al piu' i core liberi, con threads=1 per lato).
 * Esempio: SearchArena mcts:simple negamax:simple pairs=30 ms=200
 * </pre>
 * Stampa il punteggio di A con intervallo di Wilson al 95% (con 60 partite e' largo, circa +-12 punti), il dettaglio per colore e, per ogni lato,
 * profondita' media e lavoro per mossa (nodi per Negamax, valutazioni statiche per MCTS).
 */
public final class SearchArena {

	private static final int PLY_CAP = 160;
	private static final int OPENING_PLIES = 4;

	/** Un lato: l'algoritmo, il suo evaluator e i parametri. */
	static final class Lane {
		final String spec;
		final boolean mcts;
		final String mctsSpec;
		final int ordering;
		final int threads;
		final long margin;
		final Evaluator evaluator;

		Lane(String spec, boolean mcts, String mctsSpec, int ordering, int threads, long margin, Evaluator ev) {
			this.spec = spec;
			this.mcts = mcts;
			this.mctsSpec = mctsSpec;
			this.ordering = ordering;
			this.threads = threads;
			this.margin = margin;
			this.evaluator = ev;
		}

		/** Una ricerca nuova (cosi' la TT e l'albero non passano da una partita all'altra). */
		MoveSearcher newSearcher() {
			if (mcts) {
				final MctsConfig cfg = MctsConfig.parse(mctsSpec);
				return new Mcts(evaluator, cfg, threads).setSafetyMarginMillis(margin);
			}
			return new NegamaxSearcher(new Negamax(evaluator, 30, true, 1 << 20, true, ordering, threads, false).setThreads(threads).setSafetyMarginMillis(margin));
		}
	}

	/**
	 * Interpreta una specifica di lato.
	 *
	 * @throws IllegalArgumentException modificatore sconosciuto o valore non valido
	 */
	static Lane parseLane(String spec, int defaultThreads) throws IOException {
		final String[] parts = spec.split(":");
		if (parts.length < 1 || parts[parts.length - 1].isEmpty()) {
			throw new IllegalArgumentException("specifica vuota: '" + spec + "'");
		}
		boolean mcts = false;
		String mctsSpec = "";
		int ordering = Negamax.ORDER_DEFAULT;
		int threads = defaultThreads;
		long margin = 0;
		for (int i = 0; i < parts.length - 1; i++) {
			final String p = parts[i].trim();
			final String low = p.toLowerCase();
			if (low.equals("mcts")) {
				mcts = true;
			} else if (low.startsWith("mcts=")) {
				mcts = true;
				mctsSpec = p.substring(5);
				MctsConfig.parse(mctsSpec);
			} else if (low.equals("negamax") || low.equals("neg")) {
				mcts = false;
			} else if (low.startsWith("ord=")) {
				ordering = OrderingSpec.parse(p.substring(4));
			} else if (low.matches("t\\d+")) {
				threads = Integer.parseInt(low.substring(1));
			} else if (low.startsWith("m=")) {
				margin = Long.parseLong(low.substring(2));
			} else {
				throw new IllegalArgumentException("modificatore sconosciuto '" + p + "' in '" + spec + "'");
			}
		}
		final String ev = parts[parts.length - 1];
		final Evaluator evaluator = ev.equals("simple") ? new SimpleEvaluator() : EvaluatorLoader.load(ev);
		return new Lane(spec, mcts, mctsSpec, ordering, Math.max(1, threads), margin, evaluator);
	}

	/** Statistiche di un lato su tutte le sue mosse. */
	static final class Stats {
		double depthSum, workSum, timeSum, simsSum;
		long moves, reused;

		synchronized void add(MoveSearcher s, double seconds) {
			depthSum += s.lastDepth();
			workSum += s.lastWork();
			timeSum += seconds;
			moves++;
			if (s instanceof Mcts) {
				simsSum += ((Mcts) s).lastSims;
				if (((Mcts) s).lastReused) {
					reused++;
				}
			}
		}
	}

	/** Esito di una partita. */
	static final class GameResult {
		/** dal punto di vista del Bianco: +1, -1, 0 */
		double whiteOutcome;
		String label;
		int plies;
		int illegal;
	}

	/**
	 * Gioca una partita dalla posizione {@code opening} (con la storia delle sue aperture gia' dentro {@code tracker}).
	 *
	 * @param statsW statistiche del lato che gioca il Bianco; {@code statsB} quelle del Nero
	 */
	static GameResult play(State opening, GameTracker tracker, Lane white, Lane black, long ms, Stats statsW, Stats statsB) {
		final MoveSearcher sw = white.newSearcher();
		final MoveSearcher sb = black.newSearcher();
		final GameResult gr = new GameResult();
		State s = opening;
		int ply = OPENING_PLIES;
		try {
			for (; ply < PLY_CAP && !AshtonRules.isTerminal(s) && !tracker.isRepetitionDraw(); ply++) {
				final List<Action> legal = AshtonRules.generateMoves(s);
				if (legal.isEmpty()) {
					break; // chi non puo' muovere perde: lo stabilisce GameTracker.result
				}
				final boolean whiteToMove = s.getTurn() == Turn.WHITE;
				final MoveSearcher who = whiteToMove ? sw : sb;
				final List<Long> h = tracker.history();
				final long[] hist = new long[h.size()];
				for (int i = 0; i < hist.length; i++) {
					hist[i] = h.get(i);
				}
				final long t0 = System.nanoTime();
				final int mv = who.findBestMove(FastState.fromState(s), ms, hist, hist.length);
				(whiteToMove ? statsW : statsB).add(who, (System.nanoTime() - t0) / 1e9);
				if (mv == Move.NONE) {
					gr.illegal++;
					gr.whiteOutcome = whiteToMove ? -1 : 1; // nessuna mossa restituita pur avendone: forfait
					gr.label = "FORFEIT";
					gr.plies = ply;
					return gr;
				}
				final Action a = Move.toAction(mv, s.getTurn());
				boolean ok = false;
				for (Action x : legal) {
					if (x.getFrom().equals(a.getFrom()) && x.getTo().equals(a.getTo())) {
						ok = true;
						break;
					}
				}
				if (!ok) {
					gr.illegal++;
					gr.whiteOutcome = whiteToMove ? -1 : 1;
					gr.label = "ILLEGAL";
					gr.plies = ply;
					return gr;
				}
				s = tracker.apply(s, a);
			}
			final GameTracker.Result r = tracker.result(s);
			gr.whiteOutcome = r.outcome;
			gr.label = r.label;
			gr.plies = ply;
			return gr;
		} finally {
			sw.shutdown();
			sb.shutdown();
		}
	}

	/** Intervallo di Wilson al 95% per un punteggio medio {@code score} su {@code n} partite. */
	static double[] wilson(double score, int n) {
		final double z = 1.96;
		final double centre = score + z * z / (2.0 * n);
		final double half = z * Math.sqrt(score * (1 - score) / n + z * z / (4.0 * n * n));
		final double denom = 1 + z * z / n;
		return new double[] { (centre - half) / denom, (centre + half) / denom };
	}

	/** Una apertura casuale di {@link #OPENING_PLIES} semi-mosse con la storia di ripetizione gia' tracciata. */
	private static State randomOpening(Random rnd, GameTracker[] trackerOut) {
		State open = new StateTablut();
		open.setTurn(Turn.WHITE);
		final GameTracker t = new GameTracker(open);
		for (int i = 0; i < OPENING_PLIES; i++) {
			final List<Action> mv = AshtonRules.generateMoves(open);
			open = t.apply(open, mv.get(rnd.nextInt(mv.size())));
		}
		trackerOut[0] = t;
		return open;
	}

	/** Risultati di un confronto A contro B. */
	static final class Summary {
		int aw, ad, al, games, illegal;
		int awWhite, adWhite, alWhite, awBlack, adBlack, alBlack;
		final Stats sa = new Stats(), sb = new Stats();
		final java.util.Map<String, Integer> labels = new java.util.TreeMap<>();

		double score() {
			return (aw + 0.5 * ad) / Math.max(1, games);
		}
	}

	static Summary run(final Lane a, final Lane b, int pairs, final long ms, long seed, int par) throws Exception {
		final Summary sum = new Summary();
		final Random rnd = new Random(seed);
		final List<State> openings = new ArrayList<>();
		final List<GameTracker> trackers = new ArrayList<>();
		for (int p = 0; p < pairs; p++) {
			final GameTracker[] t = new GameTracker[1];
			openings.add(randomOpening(rnd, t));
			trackers.add(t[0]);
		}
		final ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, par));
		final List<Future<Object[]>> futures = new ArrayList<>();
		for (int p = 0; p < pairs; p++) {
			for (int swap = 0; swap < 2; swap++) {
				final int pp = p, sw = swap;
				futures.add(pool.submit(new java.util.concurrent.Callable<Object[]>() {
					@Override
					public Object[] call() {
						// il tracker dell'apertura va clonato: ogni partita lo modifica
						final GameTracker src = trackers.get(pp);
						final GameTracker t = new GameTracker(new StateTablut());
						t.history().clear();
						t.history().addAll(src.history());
						final Stats stW = new Stats(), stB = new Stats();
						final Lane white = sw == 0 ? a : b, black = sw == 0 ? b : a;
						final GameResult r = play(openings.get(pp), t, white, black, ms, stW, stB);
						return new Object[] { sw, r, sw == 0 ? stW : stB, sw == 0 ? stB : stW };
					}
				}));
			}
		}
		for (Future<Object[]> f : futures) {
			final Object[] o = f.get();
			final int swap = (Integer) o[0];
			final GameResult r = (GameResult) o[1];
			final Stats mineA = (Stats) o[2], mineB = (Stats) o[3];
			merge(sum.sa, mineA);
			merge(sum.sb, mineB);
			final double aOutcome = swap == 0 ? r.whiteOutcome : -r.whiteOutcome; // dal punto di vista di A
			sum.games++;
			sum.illegal += r.illegal;
			final Integer prev = sum.labels.get(r.label);
			sum.labels.put(r.label, prev == null ? 1 : prev + 1);
			if (aOutcome > 0) {
				sum.aw++;
				if (swap == 0) {
					sum.awWhite++;
				} else {
					sum.awBlack++;
				}
			} else if (aOutcome < 0) {
				sum.al++;
				if (swap == 0) {
					sum.alWhite++;
				} else {
					sum.alBlack++;
				}
			} else {
				sum.ad++;
				if (swap == 0) {
					sum.adWhite++;
				} else {
					sum.adBlack++;
				}
			}
		}
		pool.shutdown();
		return sum;
	}

	private static void merge(Stats into, Stats from) {
		into.depthSum += from.depthSum;
		into.workSum += from.workSum;
		into.timeSum += from.timeSum;
		into.simsSum += from.simsSum;
		into.moves += from.moves;
		into.reused += from.reused;
	}

	private static String lane(String name, Lane l, Stats st) {
		if (st.moves == 0) {
			return name + " " + l.spec + ": nessuna mossa";
		}
		final StringBuilder sb = new StringBuilder();
		sb.append(String.format("%s %s: tempo/mossa %.0f ms, profondita' %.2f, %s/mossa %.0f", name, l.spec, st.timeSum / st.moves * 1000, st.depthSum / st.moves,
				l.mcts ? "valutazioni" : "nodi", st.workSum / st.moves));
		if (l.mcts) {
			sb.append(String.format(", simulazioni/mossa %.0f, riuso albero %.0f%%", st.simsSum / st.moves, 100.0 * st.reused / st.moves));
		}
		return sb.toString();
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 2) {
			System.out.println("Uso: SearchArena <A> <B> [pairs=30] [ms=200] [seed=1] [par=1] [threads=1]");
			System.out.println("  lato = [mcts|mcts=chiave=valore,...|negamax|ord=NOME|tN|m=MS:]*<simple|pesi.txt>");
			System.exit(2);
		}
		int pairs = 30, par = 1, threads = 1;
		long ms = 200, seed = 1;
		for (int i = 2; i < args.length; i++) {
			final String x = args[i].toLowerCase();
			if (x.startsWith("pairs=")) {
				pairs = Integer.parseInt(x.substring(6));
			} else if (x.startsWith("ms=")) {
				ms = Long.parseLong(x.substring(3));
			} else if (x.startsWith("seed=")) {
				seed = Long.parseLong(x.substring(5));
			} else if (x.startsWith("par=")) {
				par = Integer.parseInt(x.substring(4));
			} else if (x.startsWith("threads=")) {
				threads = Integer.parseInt(x.substring(8));
			} else {
				throw new IllegalArgumentException("argomento sconosciuto: " + args[i]);
			}
		}
		if (ms < 50) {
			System.out.println("ATTENZIONE: ms < 50: Negamax cerca comunque almeno 50 ms (il confronto non sarebbe a tempo uguale).");
		}
		final Lane a = parseLane(args[0], threads), b = parseLane(args[1], threads);
		final long t0 = System.nanoTime();
		final Summary sum = run(a, b, pairs, ms, seed, par);
		final double score = sum.score();
		final double[] ci = wilson(score, sum.games);
		System.out.printf("A=%s  VS  B=%s  @%d ms reali/mossa: %d partite  A +%d =%d -%d  punteggio A = %.1f%% [IC95 %.1f%% - %.1f%%]%n", a.spec, b.spec, ms, sum.games,
				sum.aw, sum.ad, sum.al, 100 * score, 100 * ci[0], 100 * ci[1]);
		System.out.printf("  A come Bianco: +%d =%d -%d | A come Nero: +%d =%d -%d%n", sum.awWhite, sum.adWhite, sum.alWhite, sum.awBlack, sum.adBlack, sum.alBlack);
		System.out.println("  " + lane("A", a, sum.sa));
		System.out.println("  " + lane("B", b, sum.sb));
		System.out.println("  esiti: " + sum.labels + (sum.illegal > 0 ? "  MOSSE ILLEGALI: " + sum.illegal : "") + "  (" + (System.nanoTime() - t0) / 1_000_000_000L + " s)");
	}
}
