package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Partite di AUTO-GIOCO del motore ({@link Negamax} con un valutatore da entrambi i lati, o con due valutatori diversi) raccolte in un
 * {@link NnueDataset}: per ogni posizione non terminale il lato al tratto, il VALORE DI RICERCA e, a partita finita, l'ESITO vero.
 *
 * <h3>Come si gioca</h3>
 * <ul>
 * <li>Le prime {@code openRandom} semi-mosse sono casuali (varieta' di apertura): quelle posizioni non si registrano.</li>
 * <li>Poi ogni posizione e' CERCATA a profondita' fissa {@code depth} (iterative deepening fino a {@code depth}, un thread: il risultato non
 * dipende dal carico della macchina ne' dal numero di thread del generatore) oppure, con {@code msPerMove > 0}, per un tempo reale;
 * il punteggio della radice, dal punto di vista del Bianco, diventa il valore etichetta ({@link #scoreToValue}). Con probabilita' {@code eps}
 * la mossa giocata e' invece casuale (esplorazione): l'etichetta resta quella della posizione, non della mossa giocata.</li>
 * <li>Le regole di fine partita sono quelle di {@link GameTracker} (le stesse di ricerca e server): patta per ripetizione, nessuna mossa =
 * sconfitta, vittoria per Re catturato o in fuga. Una partita troncata a {@code maxPlies} NON ha esito ({@code outValid = 0}).</li>
 * </ul>
 * Le partite sono indipendenti, con un seme per partita derivato dal seme principale: lo stesso seme da' lo stesso dataset, bit per bit, con
 * qualunque numero di thread (a profondita' fissa).
 * <p>
 * Uso: {@code NnueSelfPlay eval=PESI out=FILE.gz [black=PESI] [games=100] [depth=4] [ms=0] [eps=0.05] [open=4] [maxplies=200] [seed=1] [threads=1]
 * [scale=1000]}.
 */
public final class NnueSelfPlay {

	private NnueSelfPlay() {
	}

	/** Un punteggio di ricerca da questa soglia in su e' una vittoria/sconfitta forzata (Negamax usa 1 000 000 + profondita'). */
	static final double PROVEN_SCORE = 500_000;

	public static final class Config {
		public int games = 100;
		public int depth = 4;
		public long msPerMove = 0;
		public double eps = 0.05;
		public int openRandom = 4;
		public int maxPlies = 200;
		public double valueScale = 1000.0;
		public long seed = 1;
		public int threads = 1;

		void validate() {
			if (games < 1 || depth < 1 || depth > 40 || eps < 0 || eps > 1 || openRandom < 0 || maxPlies < 2 || maxPlies > 30000 || valueScale <= 0 || threads < 1
					|| msPerMove < 0) {
				throw new IllegalArgumentException("configurazione non valida: games=" + games + " depth=" + depth + " eps=" + eps + " open=" + openRandom + " maxplies="
						+ maxPlies + " scale=" + valueScale + " threads=" + threads + " ms=" + msPerMove);
			}
		}
	}

	/**
	 * Punteggio di ricerca DEL LATO AL TRATTO (come lo restituisce {@link Negamax#lastScore}) -> valore in [-1,1] dal punto di vista del Bianco:
	 * una vittoria/sconfitta forzata vale +-1, altrimenti {@code punteggio / scala} limitato a [-1,1].
	 */
	public static double scoreToValue(double moverScore, boolean whiteToMove, double scale) {
		final double white = whiteToMove ? moverScore : -moverScore;
		if (white >= PROVEN_SCORE) {
			return 1.0;
		}
		if (white <= -PROVEN_SCORE) {
			return -1.0;
		}
		return Math.max(-1.0, Math.min(1.0, white / scale));
	}

	/** Una partita giocata: le posizioni registrate e come e' finita. */
	public static final class Game {
		public final List<byte[]> boards = new ArrayList<>();
		public final List<Integer> roles = new ArrayList<>();
		public final List<Integer> plies = new ArrayList<>();
		public final List<Double> values = new ArrayList<>();
		/** Tutte le mosse giocate, aperture casuali comprese. */
		public final List<Action> moves = new ArrayList<>();
		public int applied;
		public GameTracker.Result result;
		/** Falso se la partita e' stata troncata: l'esito non vale. */
		public boolean outValid;
	}

	private static Negamax newSearcher(Evaluator ev, Config cfg) {
		final Negamax nm = cfg.msPerMove > 0 ? new Negamax(ev, 64, true, 1 << 18, true, Negamax.ORDER_RECOMMENDED, 1) : new Negamax(ev, cfg.depth, true, 1 << 18, true,
				Negamax.ORDER_RECOMMENDED, 1);
		nm.setSafetyMarginMillis(0);
		return nm;
	}

	/** Gioca una partita (white/black possono essere lo stesso evaluator) con il seme dato. */
	public static Game playGame(Evaluator white, Evaluator black, Config cfg, long gameSeed) {
		cfg.validate();
		final Random rnd = new Random(gameSeed);
		final Negamax nw = newSearcher(white, cfg);
		final Negamax nb = white == black ? nw : newSearcher(black, cfg);
		final Game g = new Game();
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		final GameTracker tracker = new GameTracker(s);
		final long budget = cfg.msPerMove > 0 ? cfg.msPerMove : 1_000_000_000L;
		int ply = 0;
		for (; ply < cfg.maxPlies && !AshtonRules.isTerminal(s) && !tracker.isRepetitionDraw(); ply++) {
			final List<Action> legal = AshtonRules.generateMoves(s);
			if (legal.isEmpty()) {
				break;
			}
			final boolean whiteToMove = s.getTurn() == Turn.WHITE;
			Action chosen;
			if (ply < cfg.openRandom) {
				chosen = legal.get(rnd.nextInt(legal.size()));
			} else {
				final Negamax nm = whiteToMove ? nw : nb;
				chosen = nm.findBestMove(s, budget, tracker.history());
				g.boards.add(FastState.fromState(s).board.clone());
				g.roles.add(whiteToMove ? 0 : 1);
				g.plies.add(ply);
				g.values.add(scoreToValue(nm.lastScore, whiteToMove, cfg.valueScale));
				if (chosen == null || rnd.nextDouble() < cfg.eps) {
					chosen = legal.get(rnd.nextInt(legal.size()));
				}
			}
			s = tracker.apply(s, chosen);
			g.moves.add(chosen);
			g.applied++;
		}
		g.result = tracker.result(s);
		g.outValid = !GameTracker.LABEL_MAXPLIES.equals(g.result.label);
		return g;
	}

	/** Gioca {@code cfg.games} partite e ne raccoglie le posizioni (sorgente {@link NnueDataset#SRC_SELFPLAY}, chiave di partita = indice). */
	public static NnueDataset generate(final Evaluator white, final Evaluator black, final Config cfg, PrintStream log) throws Exception {
		cfg.validate();
		final ExecutorService pool = Executors.newFixedThreadPool(cfg.threads);
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final long t0 = System.currentTimeMillis();
		int w = 0, bl = 0, dr = 0, tr = 0;
		long plySum = 0;
		try {
			final List<Future<Game>> fs = new ArrayList<>();
			final Random master = new Random(cfg.seed * 131 + 17);
			for (int i = 0; i < cfg.games; i++) {
				final long gs = master.nextLong();
				fs.add(pool.submit(new Callable<Game>() {
					@Override
					public Game call() {
						return playGame(white, black, cfg, gs);
					}
				}));
			}
			for (int i = 0; i < fs.size(); i++) {
				final Game g = fs.get(i).get();
				for (int k = 0; k < g.boards.size(); k++) {
					final int ply = g.plies.get(k);
					b.add(g.boards.get(k), 0, g.roles.get(k), (int) g.result.outcome, g.outValid, g.values.get(k), true, i, ply, g.applied - ply,
							NnueDataset.SRC_SELFPLAY);
				}
				if (!g.outValid) {
					tr++;
				} else if (g.result.outcome > 0) {
					w++;
				} else if (g.result.outcome < 0) {
					bl++;
				} else {
					dr++;
				}
				plySum += g.applied;
				if (log != null && ((i + 1) % 10 == 0 || i + 1 == fs.size())) {
					log.printf("  partite %d/%d: Bianco %d, Nero %d, patte %d, troncate %d; semi-mosse medie %.1f; posizioni %d; %.0f s%n", i + 1, cfg.games, w, bl, dr, tr,
							plySum / (double) (i + 1), b.size(), (System.currentTimeMillis() - t0) / 1000.0);
				}
			}
		} finally {
			pool.shutdown();
		}
		return b.build();
	}

	// ====================================================================== main

	public static void main(String[] args) throws Exception {
		final Config cfg = new Config();
		String eval = null, black = null, out = null;
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + a);
				System.exit(2);
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "eval": eval = v; break;
				case "black": black = v; break;
				case "out": out = v; break;
				case "games": cfg.games = Integer.parseInt(v); break;
				case "depth": cfg.depth = Integer.parseInt(v); break;
				case "ms": cfg.msPerMove = Long.parseLong(v); break;
				case "eps": cfg.eps = Double.parseDouble(v); break;
				case "open": cfg.openRandom = Integer.parseInt(v); break;
				case "maxplies": cfg.maxPlies = Integer.parseInt(v); break;
				case "seed": cfg.seed = Long.parseLong(v); break;
				case "threads": cfg.threads = Integer.parseInt(v); break;
				case "scale": cfg.valueScale = Double.parseDouble(v); break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (NumberFormatException e) {
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (eval == null || out == null) {
			System.out.println("Uso: NnueSelfPlay eval=PESI out=FILE.gz [black=PESI] [games=100] [depth=4] [ms=0] [eps=0.05] [open=4] [maxplies=200] [seed=1] [threads=1] [scale=1000]");
			System.exit(2);
		}
		try {
			cfg.validate();
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
			System.exit(2);
		}
		try {
			final Evaluator white = "simple".equals(eval) ? new it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator() : EvaluatorLoader.load(eval);
			final Evaluator blk = black == null ? white : "simple".equals(black) ? new it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator()
					: EvaluatorLoader.load(black);
			System.out.println("Bianco: " + EvaluatorLoader.describe(white) + (black == null ? " (anche Nero)" : "; Nero: " + EvaluatorLoader.describe(blk)));
			final NnueDataset d = generate(white, blk, cfg, System.out);
			d.save(out);
			System.out.println("scritto " + out + ": " + d.describe());
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}
}
