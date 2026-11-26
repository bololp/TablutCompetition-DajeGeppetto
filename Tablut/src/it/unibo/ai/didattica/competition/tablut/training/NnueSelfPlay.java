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
import it.unibo.ai.didattica.competition.tablut.aiplayer.PruningOptions;
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
 *
 * <h3>Generare piu' in fretta a profondita' alta (tutte le opzioni sono spente di default: senza di esse il dataset e' identico, bit per bit, a prima)</h3>
 * <ul>
 * <li>{@code prune=SPEC} (stessa sintassi del client, per es. {@code all}): null-move, LMR e aspirazione. Misurato sulle posizioni di partita: ~5x piu' veloce a
 * profondita' 7, con il valore di radice che cambia di ~0,03 (RMS) e la stessa mossa nell'80% dei casi; e' la ricerca con cui si gioca davvero.</li>
 * <li>{@code tt=N}: la tabella di trasposizione ha {@code 2^N} voci (predefinito 18). A profondita' 8 conviene 20-22 (24 byte per voce, per ogni partita in corso).</li>
 * <li>{@code movedepth=D} (con {@code D < depth}): la MOSSA si sceglie con una ricerca a profondita' D (economica), l'ETICHETTA e' una seconda ricerca a {@code depth}
 * solo sulle posizioni scelte da {@code labelevery=K} (una ogni K, con fase casuale per partita). La qualita' dell'etichetta e' quella di {@code depth}; la partita e'
 * quella di {@code D}, che serve solo a coprire posizioni sensate.</li>
 * <li>{@code skipsat=X}: le posizioni la cui etichetta ha {@code |valore| >= X} (per es. 0,97) non si registrano: sono quelle "saturate" dove il valore non distingue piu' nulla.</li>
 * <li>{@code adj=X:N}: se per N ricerche consecutive il valore e' {@code >= X} in modulo e dello stesso segno, la partita si interrompe e NON ha esito
 * ({@code outValid = 0}, come le troncate): le etichette di valore restano valide. Evita di spendere ricerche profonde in partite che si sono bloccate.</li>
 * </ul>
 * <p>
 * Uso: {@code NnueSelfPlay eval=PESI out=FILE.gz [black=PESI] [games=100] [depth=4] [ms=0] [eps=0.05] [open=4] [maxplies=200] [seed=1] [threads=1]
 * [scale=1000] [prune=off] [tt=18] [movedepth=0] [labelevery=1] [skipsat=0] [adj=off]}.
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
		/** Potatura del Negamax (null-move/LMR/aspirazione); {@link PruningOptions#OFF} = ricerca storica. */
		public PruningOptions pruning = PruningOptions.OFF;
		/** La TT ha {@code 2^ttLog2} voci. */
		public int ttLog2 = 18;
		/** 0 (o {@code >= depth}) = la mossa e l'etichetta vengono dalla stessa ricerca; altrimenti la mossa e' scelta a questa profondita'. */
		public int moveDepth = 0;
		/** Si registra una posizione ogni {@code labelEvery} (fase casuale per partita). */
		public int labelEvery = 1;
		/** 0 = spento; altrimenti non si registrano le posizioni con {@code |valore| >= skipSat}. */
		public double skipSat = 0;
		/** Aggiudicazione: {@code adjPlies} ricerche consecutive con {@code |valore| >= adjThreshold} dello stesso segno chiudono la partita (0 = spento). */
		public double adjThreshold = 0;
		public int adjPlies = 0;

		boolean split() {
			return moveDepth > 0 && moveDepth < depth;
		}

		void validate() {
			if (pruning == null || ttLog2 < 10 || ttLog2 > 28 || moveDepth < 0 || moveDepth > 40 || (moveDepth > 0 && msPerMove > 0) || labelEvery < 1 || skipSat < 0 || skipSat > 1
					|| adjPlies < 0 || adjThreshold < 0 || adjThreshold > 1 || (adjPlies > 0 && adjThreshold <= 0)) {
				throw new IllegalArgumentException("configurazione non valida: tt=" + ttLog2 + " movedepth=" + moveDepth + " ms=" + msPerMove + " labelevery=" + labelEvery + " skipsat=" + skipSat
						+ " adj=" + adjThreshold + ":" + adjPlies);
			}
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
		/** Falso se la partita e' stata troncata o aggiudicata: l'esito non vale. */
		public boolean outValid;
		/** Vero se la partita e' stata chiusa da {@code adj=}: nessun esito, le etichette di valore restano valide. */
		public boolean adjudicated;
	}

	private static Negamax newSearcher(Evaluator ev, Config cfg, int depth) {
		final Negamax nm = cfg.msPerMove > 0 ? new Negamax(ev, 64, true, 1 << cfg.ttLog2, true, Negamax.ORDER_RECOMMENDED, 1)
				: new Negamax(ev, depth, true, 1 << cfg.ttLog2, true, Negamax.ORDER_RECOMMENDED, 1);
		nm.setSafetyMarginMillis(0);
		if (cfg.pruning != PruningOptions.OFF) {
			nm.setPruning(cfg.pruning);
		}
		return nm;
	}

	/** Gioca una partita (white/black possono essere lo stesso evaluator) con il seme dato. */
	public static Game playGame(Evaluator white, Evaluator black, Config cfg, long gameSeed) {
		cfg.validate();
		final Random rnd = new Random(gameSeed);
		final boolean split = cfg.split();
		final Negamax nw = newSearcher(white, cfg, split ? cfg.moveDepth : cfg.depth);
		final Negamax nb = white == black ? nw : newSearcher(black, cfg, split ? cfg.moveDepth : cfg.depth);
		// con movedepth < depth l'etichetta viene da una seconda ricerca, a depth, fatta solo sulle posizioni scelte
		final Negamax lw = split ? newSearcher(white, cfg, cfg.depth) : null;
		final Negamax lb = split ? (white == black ? lw : newSearcher(black, cfg, cfg.depth)) : null;
		final int phase = cfg.labelEvery > 1 ? rnd.nextInt(cfg.labelEvery) : 0; // nessun numero casuale consumato se spento: le partite di default non cambiano
		int satRun = 0, satSign = 0;
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
				final double vMove = scoreToValue(nm.lastScore, whiteToMove, cfg.valueScale);
				double vLabel = vMove;
				final boolean label = (ply - cfg.openRandom + phase) % cfg.labelEvery == 0;
				if (split && label) {
					final Negamax lm = whiteToMove ? lw : lb;
					if (lm.findBestMove(s, budget, tracker.history()) != null) {
						vLabel = scoreToValue(lm.lastScore, whiteToMove, cfg.valueScale);
					}
				}
				if (label && !(cfg.skipSat > 0 && Math.abs(vLabel) >= cfg.skipSat)) {
					g.boards.add(FastState.fromState(s).board.clone());
					g.roles.add(whiteToMove ? 0 : 1);
					g.plies.add(ply);
					g.values.add(vLabel);
				}
				if (cfg.adjPlies > 0) {
					final int sign = vMove >= cfg.adjThreshold ? 1 : vMove <= -cfg.adjThreshold ? -1 : 0;
					satRun = sign != 0 && sign == satSign ? satRun + 1 : (sign != 0 ? 1 : 0);
					satSign = sign;
				}
				if (chosen == null || rnd.nextDouble() < cfg.eps) {
					chosen = legal.get(rnd.nextInt(legal.size()));
				}
			}
			s = tracker.apply(s, chosen);
			g.moves.add(chosen);
			g.applied++;
			if (cfg.adjPlies > 0 && satRun >= cfg.adjPlies && !AshtonRules.isTerminal(s) && !tracker.isRepetitionDraw()) {
				g.adjudicated = true;
				break;
			}
		}
		g.result = tracker.result(s);
		g.outValid = !g.adjudicated && !GameTracker.LABEL_MAXPLIES.equals(g.result.label);
		return g;
	}

	/** Gioca {@code cfg.games} partite e ne raccoglie le posizioni (sorgente {@link NnueDataset#SRC_SELFPLAY}, chiave di partita = indice). */
	public static NnueDataset generate(final Evaluator white, final Evaluator black, final Config cfg, PrintStream log) throws Exception {
		cfg.validate();
		final ExecutorService pool = Executors.newFixedThreadPool(cfg.threads);
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final long t0 = System.currentTimeMillis();
		int w = 0, bl = 0, dr = 0, tr = 0, aj = 0;
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
				if (g.adjudicated) {
					aj++;
				} else if (!g.outValid) {
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
					log.printf("  partite %d/%d: Bianco %d, Nero %d, patte %d, troncate %d%s; semi-mosse medie %.1f; posizioni %d; %.0f s%n", i + 1, cfg.games, w, bl, dr, tr,
							cfg.adjPlies > 0 ? ", aggiudicate " + aj : "", plySum / (double) (i + 1), b.size(), (System.currentTimeMillis() - t0) / 1000.0);
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
				case "prune": cfg.pruning = PruningOptions.parse(v); break;
				case "tt": cfg.ttLog2 = Integer.parseInt(v); break;
				case "movedepth": cfg.moveDepth = Integer.parseInt(v); break;
				case "labelevery": cfg.labelEvery = Integer.parseInt(v); break;
				case "skipsat": cfg.skipSat = Double.parseDouble(v); break;
				case "adj":
					if (v.equalsIgnoreCase("off")) {
						cfg.adjPlies = 0;
					} else {
						final String[] p = v.split(":");
						if (p.length != 2) {
							throw new NumberFormatException(v);
						}
						cfg.adjThreshold = Double.parseDouble(p[0]);
						cfg.adjPlies = Integer.parseInt(p[1]);
					}
					break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (IllegalArgumentException e) { // include NumberFormatException e PruningOptions.parse
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (eval == null || out == null) {
			System.out.println("Uso: NnueSelfPlay eval=PESI out=FILE.gz [black=PESI] [games=100] [depth=4] [ms=0] [eps=0.05] [open=4] [maxplies=200] [seed=1] [threads=1] [scale=1000] [prune=off] [tt=18] [movedepth=0] [labelevery=1] [skipsat=0] [adj=off]");
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
