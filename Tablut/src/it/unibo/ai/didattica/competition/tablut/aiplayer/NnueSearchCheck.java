package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Integrazione dell'NNUE incrementale con la ricerca ({@link Negamax}).
 * <p>
 * Il confronto centrale e' a TRE VIE, con la STESSA rete:
 * <ol>
 * <li>{@link Negamax} con {@link NnueEvaluator}: usa la sessione incrementale (verificato: {@code incrementalSessionForTest() != null});</li>
 * <li>{@link Negamax} con {@link FullOnly}, un involucro che espone la sola valutazione completa: stessa ricerca, nessuna sessione;</li>
 * <li>{@link LegacyNegamax} (motore storico a {@code State}): ricerca indipendente, sempre con valutazione completa.</li>
 * </ol>
 * Con aritmetica a interi i tre devono dare la STESSA mossa, lo STESSO punteggio (bit per bit), la stessa profondita' e lo stesso numero di
 * nodi: qualunque differenza vuol dire che un accumulatore e' stato aggiornato male lungo il cammino della ricerca. In piu' si verifica che
 * nessuna sessione di nessun worker abbia mai ricostruito per ripiego ({@code incrementalFallbacksForTest() == 0}), cioe' che l'integrazione
 * rispetti il protocollo (reset alla radice, push dopo ogni makeMove) e non si affidi al controllo di coerenza.
 * <p>
 * Uso: {@code NnueSearchCheck [partite] [posizioniCasuali] [seed]}.
 */
public class NnueSearchCheck {

	private static final long BIG_BUDGET = 100_000_000L;
	private static int failures = 0;
	private static long checks = 0;
	private static long comparisons = 0, legacyComparisons = 0, nodesTotal = 0, wins = 0, repetitions = 0, deltaSteps = 0;

	private static void fail(String msg) {
		failures++;
		if (failures <= 30) {
			System.out.println("FAIL - " + msg);
		}
	}

	private static void check(boolean cond, String msg) {
		checks++;
		if (!cond) {
			fail(msg);
		}
	}

	/** Espone la sola valutazione completa di un altro evaluator: NON e' un {@link IncrementalEvaluator}, quindi {@link Negamax} non crea sessioni. */
	static final class FullOnly implements Evaluator {
		private final Evaluator inner;

		FullOnly(Evaluator inner) {
			this.inner = inner;
		}

		@Override
		public double evaluate(FastState state) {
			return inner.evaluate(state);
		}
	}

	private static final class Cfg {
		final String name;
		final boolean tt, pvs;
		final int order;

		/**
		 * {@link LegacyNegamax} (il motore storico) implementa solo gli ordinamenti di base (nessuno, killer, history): con i flag piu'
		 * recenti (WINS, CAPTURES, ROOT_SCORES...) cerca in un altro ordine e visita altri nodi, quindi non e' un riferimento valido.
		 */
		final boolean legacyComparable;

		Cfg(String name, boolean tt, boolean pvs, int order) {
			this.name = name;
			this.tt = tt;
			this.pvs = pvs;
			this.order = order;
			this.legacyComparable = (order & ~Negamax.ORDER_DEFAULT) == 0;
		}
	}

	private static final Cfg[] CFGS = { new Cfg("produzione (TT+PVS+killer+history)", true, true, Negamax.ORDER_DEFAULT),
			new Cfg("alpha-beta puro", false, false, Negamax.ORDER_NONE), new Cfg("raccomandato (TT+PVS+wins+caps+rootsc)", true, true, Negamax.ORDER_RECOMMENDED) };

	public static void main(String[] args) {
		// asserzioni interne di Negamax attive senza -ea
		ClassLoader.getSystemClassLoader().setClassAssertionStatus("it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax", true);
		final int games = args.length > 0 ? Integer.parseInt(args[0]) : 8;
		final int randomPositions = args.length > 1 ? Integer.parseInt(args[1]) : 30;
		final long seed = args.length > 2 ? Long.parseLong(args[2]) : 777L;
		final Random rnd = new Random(seed);
		final NnueEvaluator main = new NnueEvaluator(NnueNetwork.random(64, 16, 21));
		final NnueEvaluator small = new NnueEvaluator(NnueNetwork.random(8, 4, 22));

		// ---- 1) posizioni di partite reali, con storia: tutte le configurazioni di ricerca
		for (int g = 0; g < games; g++) {
			final TestPositions.Game game = TestPositions.playGame(rnd, rnd.nextInt(3) != 0, 120);
			final int n = game.positions.size();
			if (n == 0) {
				continue;
			}
			for (int k = 0; k < 4; k++) {
				final int i = k == 0 ? Math.min(n - 1, 2 + rnd.nextInt(4)) : rnd.nextInt(n);
				final FastState fs = FastState.fromState(game.positions.get(i));
				final PositionHistory h = game.histories.get(i);
				for (Cfg cfg : CFGS) {
					compare("partita " + g + " ply " + i + " [" + cfg.name + "]", fs, h, k % 2 == 0 ? main : small, cfg, cfg.tt ? 4 : 3);
				}
			}
		}
		System.out.println("Posizioni di partita: " + comparisons + " ricerche confrontate, " + nodesTotal + " nodi");

		// ---- 2) posizioni casuali dense: catture, vittorie in 1, mosse mancanti
		final long before = comparisons;
		for (int i = 0; i < randomPositions; i++) {
			final State s = TestPositions.randomPosition(rnd);
			final PositionHistory h = new PositionHistory();
			h.add(s);
			final FastState fs = FastState.fromState(s);
			for (Cfg cfg : CFGS) {
				compare("casuale " + i + " [" + cfg.name + "]", fs, h, i % 3 == 0 ? small : main, cfg, cfg.tt ? 4 : 3);
			}
		}
		System.out.println("Posizioni casuali: " + (comparisons - before) + " ricerche confrontate, vittorie forzate viste " + wins);

		forcedResults(main);
		reusedInstanceGames(main, rnd);
		interruptedSearches(main, rnd);
		siblingsWithExploreChild(main, rnd);
		reinstallRoot(main, rnd);
		lazySmp(main, rnd);
		mctsWithNnue(main, rnd);
		check(comparisons > 150, "copertura: troppi pochi confronti (" + comparisons + ")");
		check(legacyComparisons > 100, "copertura: troppi pochi confronti con il motore storico (" + legacyComparisons + ")");
		check(nodesTotal > 300_000, "copertura: troppi pochi nodi (" + nodesTotal + ")");
		check(wins > 0, "copertura: nessuna vittoria forzata vista");
		check(repetitions > 0, "copertura: nessuna ripetizione mai incontrata dalla ricerca");
		check(deltaSteps > 100_000, "copertura: l'aggiornamento incrementale e' stato esercitato troppo poco (" + deltaSteps + " passi)");

		if (failures == 0) {
			System.out.println("=== NNUE SEARCH CHECK OK: " + checks + " verifiche (" + comparisons + " ricerche confrontate, di cui " + legacyComparisons
					+ " anche con il motore storico, " + nodesTotal + " nodi, "
					+ deltaSteps + " passi incrementali) ===");
		} else {
			System.out.println("=== NNUE SEARCH CHECK: " + failures + " FALLIMENTI su " + checks + " verifiche ===");
			System.exit(1);
		}
	}

	// ====================================================================== confronto a tre vie

	private static void compare(String label, FastState root, PositionHistory hist, NnueEvaluator ev, Cfg cfg, int depth) {
		final Evaluator full = new FullOnly(ev);
		final Negamax inc = new Negamax(ev, depth, cfg.tt, 1 << 14, cfg.pvs, cfg.order, 1);
		final Negamax ref = new Negamax(full, depth, cfg.tt, 1 << 14, cfg.pvs, cfg.order, 1);
		final LegacyNegamax legacy = new LegacyNegamax(full, depth, cfg.tt, 1 << 14, cfg.pvs, cfg.order);
		check(inc.incrementalSessionForTest() != null, label + ": Negamax con NnueEvaluator deve avere una sessione incrementale");
		check(ref.incrementalSessionForTest() == null, label + ": il controllo (FullOnly) non deve avere una sessione");

		final FastState before = FastState.copyOf(root);
		final int mi = inc.findBestMove(FastState.copyOf(root), BIG_BUDGET, hist);
		final int mr = ref.findBestMove(FastState.copyOf(root), BIG_BUDGET, hist);
		final boolean withLegacy = cfg.legacyComparable;
		int ml = Move.NONE;
		if (withLegacy) {
			final it.unibo.ai.didattica.competition.tablut.domain.Action la = legacy.findBestMove(root.toState(), BIG_BUDGET, hist);
			ml = la == null ? Move.NONE : Move.fromAction(la);
		}
		comparisons++;
		nodesTotal += inc.lastNodeCount;

		check(mi == mr && (!withLegacy || mi == ml), label + ": mossa incrementale " + Move.toString(mi) + " vs completa " + Move.toString(mr) + " vs storico " + Move.toString(ml));
		check(same(inc.lastScore, ref.lastScore) && (!withLegacy || same(inc.lastScore, legacy.lastScore)),
				label + ": punteggio incrementale " + inc.lastScore + " vs completo " + ref.lastScore + " vs storico " + legacy.lastScore);
		check(inc.lastDepthReached == ref.lastDepthReached && (!withLegacy || inc.lastDepthReached == legacy.lastDepthReached),
				label + ": profondita' " + inc.lastDepthReached + " / " + ref.lastDepthReached + " / " + legacy.lastDepthReached);
		check(inc.lastNodeCount == ref.lastNodeCount && (!withLegacy || inc.lastNodeCount == legacy.lastNodeCount),
				label + ": nodi " + inc.lastNodeCount + " / " + ref.lastNodeCount + " / " + legacy.lastNodeCount);
		check(inc.lastPvsScouts == ref.lastPvsScouts && inc.lastPvsReSearches == ref.lastPvsReSearches, label + ": scout/ri-ricerche PVS diversi");
		check(inc.repetitionEventCounter == ref.repetitionEventCounter && (!withLegacy || inc.repetitionEventCounter == legacy.repetitionEventCounter),
				label + ": ripetizioni incontrate " + inc.repetitionEventCounter + " / " + ref.repetitionEventCounter + " / " + legacy.repetitionEventCounter);
		if (withLegacy) {
			legacyComparisons++;
		}
		repetitions += inc.repetitionEventCounter;
		if (Math.abs(inc.lastScore) >= Negamax.WIN_SCORE) {
			wins++;
		}
		check(inc.incrementalFallbacksForTest() == 0, label + ": la sessione ha ricostruito per ripiego " + inc.incrementalFallbacksForTest() + " volte");
		deltaSteps += ((NnueSession) inc.incrementalSessionForTest()).deltaSteps();
		check(inc.workingState().samePosition(before) && inc.workingState().zobristHash == before.zobristHash,
				label + ": la posizione di lavoro non e' tornata alla radice dopo la ricerca");
	}

	private static boolean same(double a, double b) {
		return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
	}

	// ====================================================================== vittorie forzate (posizioni terminali nell'albero)

	/**
	 * Un Re che puo' fuggire e un Re che puo' essere catturato in una mossa: i figli terminali vengono registrati con push come ogni altro,
	 * ma la ricerca non li valuta mai. Tutte e tre le ricerche devono vedere la stessa vittoria e la stessa mossa.
	 */
	private static void forcedResults(NnueEvaluator ev) {
		// (a) Bianco: il Re in (2,2) fugge in (0,2); un solo altro bianco e due neri lontani
		State s = TestPositions.blank();
		s.getBoard()[2][2] = Pawn.KING;
		s.getBoard()[6][6] = Pawn.BLACK;
		s.getBoard()[7][2] = Pawn.BLACK;
		s.getBoard()[5][5] = Pawn.WHITE;
		s.setTurn(Turn.WHITE);
		forced("fuga del Re in 1", s, Move.of(2 * 9 + 2, 0 * 9 + 2), true, ev);
		// (b) Nero: re in (2,2), nero in (2,1); un nero da (2,7) entra in (2,3) e lo cattura a tenaglia
		s = TestPositions.blank();
		s.getBoard()[2][2] = Pawn.KING;
		s.getBoard()[2][1] = Pawn.BLACK;
		s.getBoard()[2][7] = Pawn.BLACK;
		s.getBoard()[6][6] = Pawn.WHITE;
		s.setTurn(Turn.BLACK);
		forced("cattura del Re in 1", s, Move.of(2 * 9 + 7, 2 * 9 + 3), false, ev);
	}

	private static void forced(String label, State s, int winningMove, boolean whiteMoves, NnueEvaluator ev) {
		final FastState fs = FastState.fromState(s);
		final PositionHistory h = new PositionHistory();
		h.add(s);
		for (int depth = 1; depth <= 4; depth++) {
			compare("forzata '" + label + "' d" + depth, fs, h, ev, CFGS[0], depth);
			final Negamax inc = new Negamax(ev, depth, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
			final int m = inc.findBestMove(FastState.copyOf(fs), BIG_BUDGET, h);
			check(m == winningMove, "forzata '" + label + "' d" + depth + ": mossa " + Move.toString(m) + " invece di " + Move.toString(winningMove));
			check(inc.lastScore >= Negamax.WIN_SCORE, "forzata '" + label + "' d" + depth + ": punteggio " + inc.lastScore + " (atteso una vittoria)");
		}
	}

	// ====================================================================== istanze riusate per partite intere

	/**
	 * Due istanze (incrementale e di controllo) riusate per ogni mossa di una partita, come nel client: TT, killer, history e SESSIONE
	 * persistono da una ricerca alla successiva. Devono scegliere sempre la stessa mossa con gli stessi numeri.
	 */
	private static void reusedInstanceGames(NnueEvaluator ev, Random rnd) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		for (int g = 0; g < 3; g++) {
			final Negamax inc = new Negamax(ev, 3, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 1);
			final Negamax ref = new Negamax(new FullOnly(ev), 3, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 1);
			final FastState fs = new FastState();
			final PositionHistory hist = new PositionHistory();
			hist.add(fs);
			// apertura casuale (uguale per le due istanze: una sola posizione corrente)
			for (int i = 0; i < 4 + g; i++) {
				final int n = fs.generateMoves(buf);
				final int m = buf[rnd.nextInt(n)];
				fs.makeMove(m, u);
				hist.add(fs);
			}
			int plies = 0;
			for (; plies < 40 && fs.winner == FastState.WINNER_NONE; plies++) {
				final int mi = inc.findBestMove(FastState.copyOf(fs), BIG_BUDGET, hist);
				final int mr = ref.findBestMove(FastState.copyOf(fs), BIG_BUDGET, hist);
				comparisons++;
				nodesTotal += inc.lastNodeCount;
				check(mi == mr && same(inc.lastScore, ref.lastScore) && inc.lastNodeCount == ref.lastNodeCount && inc.lastDepthReached == ref.lastDepthReached,
						"partita riusata " + g + " ply " + plies + ": " + Move.toString(mi) + "/" + inc.lastScore + "/" + inc.lastNodeCount + " vs " + Move.toString(mr) + "/"
								+ ref.lastScore + "/" + ref.lastNodeCount);
				if (mi == Move.NONE) {
					break;
				}
				fs.makeMove(mi, u);
				if (u.wasCapture()) {
					hist.clear();
				}
				if (fs.winner == FastState.WINNER_NONE) {
					hist.add(fs);
				}
			}
			check(inc.incrementalFallbacksForTest() == 0, "partita riusata " + g + ": ricostruzioni di ripiego " + inc.incrementalFallbacksForTest());
			check(plies >= 10, "partita riusata " + g + ": troppo corta (" + plies + " ply)");
			deltaSteps += ((NnueSession) inc.incrementalSessionForTest()).deltaSteps();
		}
	}

	// ====================================================================== ricerche interrotte a meta' albero

	/**
	 * Una ricerca interrotta a un numero di nodi fisso (deterministico) lascia la pila degli accumulatori a meta': la ricerca successiva,
	 * dalla stessa istanza e da un'altra posizione, deve dare gli stessi numeri di quella dell'istanza di controllo che ha subito lo stesso
	 * trattamento. Una sessione che si portasse dietro frame dell'albero interrotto darebbe valori diversi.
	 */
	private static void interruptedSearches(NnueEvaluator ev, Random rnd) {
		for (int r = 0; r < 6; r++) {
			final Negamax inc = new Negamax(ev, 5, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 1);
			final Negamax ref = new Negamax(new FullOnly(ev), 5, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 1);
			for (int round = 0; round < 3; round++) {
				final State s = TestPositions.randomPosition(rnd);
				final PositionHistory h = new PositionHistory();
				h.add(s);
				final FastState root = FastState.fromState(s);
				final FastState keep = FastState.copyOf(root);
				final int limit = 1500 * (1 + rnd.nextInt(3));
				inc.forceTimeUpAfterNodes = limit;
				ref.forceTimeUpAfterNodes = limit;
				final int mi = inc.findBestMove(FastState.copyOf(root), BIG_BUDGET, h);
				final int mr = ref.findBestMove(FastState.copyOf(root), BIG_BUDGET, h);
				comparisons++;
				nodesTotal += inc.lastNodeCount;
				check(mi == mr && same(inc.lastScore, ref.lastScore) && inc.lastNodeCount == ref.lastNodeCount,
						"ricerca interrotta " + r + "." + round + ": " + Move.toString(mi) + "/" + inc.lastScore + "/" + inc.lastNodeCount + " vs " + Move.toString(mr) + "/"
								+ ref.lastScore + "/" + ref.lastNodeCount);
				check(inc.workingState().samePosition(keep) && inc.workingState().zobristHash == keep.zobristHash,
						"ricerca interrotta " + r + "." + round + ": la posizione di lavoro non e' tornata alla radice");
				check(inc.incrementalFallbacksForTest() == 0, "ricerca interrotta " + r + "." + round + ": ricostruzioni di ripiego " + inc.incrementalFallbacksForTest());
				deltaSteps += ((NnueSession) inc.incrementalSessionForTest()).deltaSteps();
			}
		}
	}

	// ====================================================================== rami fratelli, uno per uno

	/**
	 * Esplora direttamente ogni mossa della radice con {@code exploreChild} (hook white-box) a profondita' 1..3: i figli della radice sono
	 * fratelli che riscrivono lo stesso frame; il punteggio di ciascuno deve coincidere con quello della ricerca di controllo, e dopo ogni
	 * esplorazione la posizione di lavoro deve essere tornata alla radice.
	 */
	private static void siblingsWithExploreChild(NnueEvaluator ev, Random rnd) {
		final int[] buf = new int[FastState.MAX_MOVES];
		for (int p = 0; p < 12; p++) {
			final State s = p % 2 == 0 ? TestPositions.randomPosition(rnd) : somePlayed(rnd);
			final FastState root = FastState.fromState(s);
			final PositionHistory h = new PositionHistory();
			h.add(s);
			final int depth = 1 + p % 3;
			final Negamax inc = new Negamax(ev, depth + 1, false, 1 << 10, false, Negamax.ORDER_NONE, 1);
			final Negamax ref = new Negamax(new FullOnly(ev), depth + 1, false, 1 << 10, false, Negamax.ORDER_NONE, 1);
			inc.installForTest(root, h, depth + 1);
			ref.installForTest(root, h, depth + 1);
			final int n = root.generateMoves(buf);
			for (int i = 0; i < n; i++) {
				final double a = inc.exploreChild(buf[i], depth, -1e9, 1e9, 1, 0);
				final double b = ref.exploreChild(buf[i], depth, -1e9, 1e9, 1, 0);
				check(same(a, b), "fratelli posizione " + p + " mossa " + Move.toString(buf[i]) + " d" + depth + ": " + a + " vs " + b);
				check(inc.workingState().samePosition(root) && inc.workingState().zobristHash == root.zobristHash, "fratelli posizione " + p + ": posizione di lavoro non ripristinata");
			}
			check(inc.incrementalFallbacksForTest() == 0, "fratelli posizione " + p + ": ricostruzioni di ripiego " + inc.incrementalFallbacksForTest());
			deltaSteps += ((NnueSession) inc.incrementalSessionForTest()).deltaSteps();
		}
	}

	private static State somePlayed(Random rnd) {
		final List<State> ps = TestPositions.playGame(rnd, true, 60).positions;
		return ps.get(ps.size() > 4 ? 4 + rnd.nextInt(ps.size() - 4) : 0);
	}

	/** Reinstallare un'altra radice sulla stessa istanza deve azzerare l'accumulatore: i risultati sono quelli di un'istanza nuova. */
	private static void reinstallRoot(NnueEvaluator ev, Random rnd) {
		final Negamax reused = new Negamax(ev, 4, false, 1 << 10, false, Negamax.ORDER_NONE, 1);
		final int[] buf = new int[FastState.MAX_MOVES];
		for (int p = 0; p < 8; p++) {
			final State s = TestPositions.randomPosition(rnd);
			final FastState root = FastState.fromState(s);
			final PositionHistory h = new PositionHistory();
			h.add(s);
			final Negamax fresh = new Negamax(ev, 4, false, 1 << 10, false, Negamax.ORDER_NONE, 1);
			reused.installForTest(root, h, 3);
			fresh.installForTest(root, h, 3);
			final int n = root.generateMoves(buf);
			for (int i = 0; i < Math.min(n, 12); i++) {
				check(same(reused.exploreChild(buf[i], 2, -1e9, 1e9, 1, 0), fresh.exploreChild(buf[i], 2, -1e9, 1e9, 1, 0)),
						"radice reinstallata " + p + " mossa " + Move.toString(buf[i]) + ": istanza riusata diversa da quella nuova");
			}
			check(reused.incrementalFallbacksForTest() == 0, "radice reinstallata " + p + ": ricostruzioni di ripiego");
		}
	}

	// ====================================================================== MCTS

	/**
	 * MCTS non ha un percorso incrementale: valuta le foglie con {@code evaluate(FastState)} (la valutazione completa). Con l'NNUE deve
	 * funzionare comunque, anche con piu' thread che condividono l'evaluator: mossa legale, nessuna eccezione, posizione di partenza intatta.
	 */
	private static void mctsWithNnue(NnueEvaluator ev, Random rnd) {
		final int[] buf = new int[FastState.MAX_MOVES];
		for (int threads = 1; threads <= 2; threads++) {
			final Mcts mcts = new Mcts(ev, MctsConfig.parse(""), threads).setSafetyMarginMillis(0);
			for (int p = 0; p < 4; p++) {
				final State s = p == 0 ? TestPositions.initial() : somePlayed(rnd);
				final FastState root = FastState.fromState(s);
				final FastState keep = FastState.copyOf(root);
				final PositionHistory h = new PositionHistory();
				h.add(s);
				final int m = mcts.findBestMove(FastState.copyOf(root), 150, h);
				final int n = keep.generateMoves(buf);
				boolean legal = false;
				for (int i = 0; i < n; i++) {
					legal |= buf[i] == m;
				}
				check(legal, "MCTS con NNUE (" + threads + " thread), posizione " + p + ": mossa non legale " + Move.toString(m));
				check(root.samePosition(keep), "MCTS con NNUE: la posizione di partenza e' stata modificata");
			}
		}
	}

	// ====================================================================== Lazy SMP

	/**
	 * Piu' thread (helper = altre istanze di Negamax con la STESSA rete e la stessa TT, ciascuno con la propria sessione): la mossa e'
	 * legale, il risultato valido, nessun helper e' andato in errore e nessuna sessione ha ricostruito per ripiego.
	 */
	private static void lazySmp(NnueEvaluator ev, Random rnd) {
		final int[] buf = new int[FastState.MAX_MOVES];
		for (int threads = 2; threads <= 4; threads++) {
			final Negamax mt = new Negamax(ev, 6, true, 1 << 18, true, Negamax.ORDER_DEFAULT, threads);
			for (int p = 0; p < 4; p++) {
				final State s = p == 0 ? TestPositions.initial() : somePlayed(rnd);
				final FastState root = FastState.fromState(s);
				final FastState keep = FastState.copyOf(root);
				final PositionHistory h = new PositionHistory();
				h.add(s);
				final int m = mt.findBestMove(FastState.copyOf(root), 250, h);
				final int n = keep.generateMoves(buf);
				boolean legal = false;
				for (int i = 0; i < n; i++) {
					legal |= buf[i] == m;
				}
				check(legal, "Lazy SMP " + threads + " thread, posizione " + p + ": mossa non legale " + Move.toString(m));
				check(!Double.isNaN(mt.lastScore) && mt.lastDepthReached >= 1, "Lazy SMP " + threads + " thread: risultato non valido");
				check(mt.incrementalSessionsForTest() == threads,
						"Lazy SMP " + threads + " thread: ogni worker deve avere una propria sessione incrementale (trovate " + mt.incrementalSessionsForTest() + ")");
				check(mt.lastHelperErrors == 0, "Lazy SMP " + threads + " thread, posizione " + p + ": " + mt.lastHelperErrors + " helper andati in errore");
				check(mt.incrementalFallbacksForTest() == 0, "Lazy SMP " + threads + " thread, posizione " + p + ": ricostruzioni di ripiego " + mt.incrementalFallbacksForTest());
				check(mt.workingState().samePosition(keep), "Lazy SMP " + threads + " thread: posizione di lavoro del principale non ripristinata");
			}
		}
	}
}
