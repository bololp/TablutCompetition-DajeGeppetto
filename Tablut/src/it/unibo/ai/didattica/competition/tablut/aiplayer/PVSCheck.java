package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Test "white-box" (stesso package di {@link Negamax}) della Principal
 * Variation Search. Una PVS sbagliata produce mosse leggermente errate SENZA
 * mai un errore visibile, quindi qui si verifica a fondo che:
 * <ol>
 * <li>il valore alla radice sia IDENTICO a quello dell'alpha-beta puro (senza
 * TT, dove l'equivalenza teorica e' esatta), con un evaluator a valori quasi
 * interi ({@link SimpleEvaluator}: molti pareggi) e con uno a valori continui
 * (rete neurale con pesi casuali);</li>
 * <li>la mossa scelta sia davvero OTTIMA: il suo valore esatto, ricalcolato con
 * un alpha-beta indipendente a finestra piena, deve coincidere col valore di
 * radice (non basta confrontare il punteggio: con pareggi la mossa puo'
 * cambiare);</li>
 * <li>lo stesso valga con la transposition table accesa e con una storia di
 * ripetizione reale (ciclo di mosse che ricrea la posizione);</li>
 * <li>con budget di tempo minuscoli la mossa restituita sia sempre legale
 * (le ri-ricerche non devono mai usare valori di un'iterazione interrotta);</li>
 * </ol>
 * Stampa anche nodi e tasso di ri-ricerca (informativo, non asserito).
 */
public class PVSCheck {

	static int failures = 0;
	static final double INF = 1e12;

	public static void main(String[] args) {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 2024L;
		int numPositions = args.length > 1 ? Integer.parseInt(args[1]) : 30;
		Random rnd = new Random(seed);
		List<State> positions = collectPositions(rnd, numPositions);
		System.out.println("Posizioni di test: " + positions.size());

		Evaluator simple = new SimpleEvaluator();
		Evaluator neural = NeuralEvaluator.randomInit(5); // valori continui, quasi tutti distinti

		checkEquivalence("Simple", simple, positions, 4);
		checkEquivalence("Rete (valori continui)", neural, positions, 3);
		checkWithTT("Simple", simple, positions, 4);
		checkWithTT("Rete (valori continui)", neural, positions, 3);
		checkRepetitionHistory();
		checkTimeUpLegal(simple, positions, rnd);

		if (failures == 0) {
			System.out.println("=== PVS CHECK OK ===");
		} else {
			System.out.println("=== PVS CHECK FALLITO: " + failures + " discrepanze ===");
			System.exit(1);
		}
	}

	static void fail(String m) {
		failures++;
		System.out.println("FAIL - " + m);
	}

	// ------------------------------------------------------------------ oracolo

	/**
	 * Valore ESATTO della mossa m alla radice, a profondita' d, per il lato di turno: oracolo INDIPENDENTE da
	 * {@link Negamax} (alpha-beta puro scritto qui, senza TT, senza PVS, senza orologio) che ripete la semantica del
	 * motore: punteggi di vittoria/sconfitta con bonus di profondita', "nessuna mossa legale" = sconfitta,
	 * ripetizione = pareggio (storia azzerata dalle catture).
	 * (Non si puo' usare una seconda istanza di Negamax come oracolo: il suo orologio interno interrompe la ricerca
	 * dopo 2048 nodi se non e' stata avviata da findBestMove.)
	 */
	static double exactValueOfMove(Evaluator ev, State root, Action m, int d, List<Long> history) {
		int mySign = root.getTurn() == Turn.WHITE ? 1 : -1;
		State child = AshtonRules.applyMove(root, m);
		return -oracleChild(ev, root, child, m, d - 1, -INF, INF, -mySign, new ArrayList<>(history));
	}

	static double oracleChild(Evaluator ev, State parent, State child, Action a, int depth, double alpha, double beta,
			int persp, List<Long> history) {
		long h = ZobristHasher.fullHash(child);
		if (AshtonRules.wasCapture(parent, child, a)) {
			List<Long> fresh = new ArrayList<>();
			fresh.add(h);
			return oracle(ev, child, depth, alpha, beta, persp, fresh);
		}
		if (history.contains(h)) {
			return 0; // ripetizione = pareggio
		}
		history.add(h);
		double v = oracle(ev, child, depth, alpha, beta, persp, history);
		history.remove(history.size() - 1);
		return v;
	}

	static double oracle(Evaluator ev, State s, int depth, double alpha, double beta, int persp, List<Long> history) {
		Turn t = s.getTurn();
		if (t == Turn.WHITEWIN) {
			return persp * (Negamax.WIN_SCORE + depth);
		}
		if (t == Turn.BLACKWIN) {
			return persp * -(Negamax.WIN_SCORE + depth);
		}
		if (t == Turn.DRAW) {
			return 0;
		}
		if (depth == 0) {
			return persp * ev.evaluate(s);
		}
		List<Action> moves = AshtonRules.generateMoves(s);
		if (moves.isEmpty()) {
			double absolute = (t == Turn.WHITE) ? -(Negamax.WIN_SCORE + depth) : (Negamax.WIN_SCORE + depth);
			return persp * absolute;
		}
		double best = Double.NEGATIVE_INFINITY;
		for (Action a : moves) {
			State c = AshtonRules.applyMove(s, a);
			double v = -oracleChild(ev, s, c, a, depth - 1, -beta, -alpha, -persp, history);
			if (v > best) {
				best = v;
			}
			if (best > alpha) {
				alpha = best;
			}
			if (alpha >= beta) {
				break;
			}
		}
		return best;
	}

	static boolean sameMove(Action a, Action b) {
		return a.getRowFrom() == b.getRowFrom() && a.getColumnFrom() == b.getColumnFrom()
				&& a.getRowTo() == b.getRowTo() && a.getColumnTo() == b.getColumnTo();
	}

	// ------------------------------------------------- 1+2: PVS == alpha-beta puro

	static void checkEquivalence(String name, Evaluator ev, List<State> positions, int maxDepth) {
		long nodesAB = 0, nodesPVS = 0, scouts = 0, reSearches = 0;
		int comparisons = 0, before = failures;
		for (int p = 0; p < positions.size(); p++) {
			State pos = positions.get(p);
			for (int d = 1; d <= maxDepth; d++) {
				Negamax ab = new Negamax(ev, d, false, 0, false);
				Negamax pvs = new Negamax(ev, d, false, 0, true);
				Action mAB = ab.findBestMove(pos, 600_000L);
				Action mPVS = pvs.findBestMove(pos, 600_000L);
				comparisons++;
				nodesAB += ab.lastNodeCount;
				nodesPVS += pvs.lastNodeCount;
				scouts += pvs.lastPvsScouts;
				reSearches += pvs.lastPvsReSearches;

				if (ab.lastDepthReached != pvs.lastDepthReached) {
					fail(name + " pos " + p + " d=" + d + ": profondita' diverse " + ab.lastDepthReached + " vs " + pvs.lastDepthReached);
					continue;
				}
				if (Math.abs(ab.lastScore - pvs.lastScore) > 1e-9) {
					fail(name + " pos " + p + " d=" + d + ": VALORE DIVERSO alpha-beta=" + ab.lastScore + " PVS=" + pvs.lastScore
							+ "\n" + pos);
					continue;
				}
				// l'alpha-beta di Negamax (baseline) deve coincidere con l'oracolo indipendente
				double vab = exactValueOfMove(ev, pos, mAB, ab.lastDepthReached, new ArrayList<Long>());
				if (Math.abs(vab - ab.lastScore) > 1e-9) {
					fail(name + " pos " + p + " d=" + d + ": l'alpha-beta di Negamax diverge dall'oracolo (" + ab.lastScore + " vs " + vab + ")");
				}
				// la mossa di PVS deve essere ottima: valore esatto == valore di radice
				double v = exactValueOfMove(ev, pos, mPVS, pvs.lastDepthReached, new ArrayList<Long>());
				if (Math.abs(v - pvs.lastScore) > 1e-9) {
					fail(name + " pos " + p + " d=" + d + ": la mossa di PVS " + mPVS.getFrom() + "->" + mPVS.getTo()
							+ " vale " + v + " ma il valore di radice e' " + pvs.lastScore);
				}
				// e nessuna altra mossa deve valere di piu' (verifica indipendente del massimo)
				if (d <= 3) {
					for (Action m : AshtonRules.generateMoves(pos)) {
						double vm = exactValueOfMove(ev, pos, m, pvs.lastDepthReached, new ArrayList<Long>());
						if (vm > pvs.lastScore + 1e-9) {
							fail(name + " pos " + p + " d=" + d + ": esiste una mossa migliore " + m.getFrom() + "->" + m.getTo()
									+ " (" + vm + " > " + pvs.lastScore + ")");
							break;
						}
					}
				}
			}
		}
		System.out.printf("%s %-24s %d confronti, alpha-beta %d nodi vs PVS %d nodi (%.1f%%), scout %d, ri-ricerche %d (%.1f%%)%n",
				failures == before ? "OK  " : "FAIL", "[" + name + ", TT off]", comparisons, nodesAB, nodesPVS,
				100.0 * (nodesPVS - nodesAB) / nodesAB, scouts, reSearches, 100.0 * reSearches / Math.max(1, scouts));
	}

	// ----------------------------------------------------------- 3: con la TT

	/**
	 * Con la TT accesa l'equivalenza esatta puo' rompersi per cause che NON dipendono da PVS (una voce di profondita'
	 * maggiore riusata a profondita' minore): quindi si confronta PVS+TT con PVS senza TT come fa TTCorrectnessCheck per
	 * l'alpha-beta, e si riporta in modo separato quante volte il valore differisce per l'alpha-beta classico.
	 */
	static void checkWithTT(String name, Evaluator ev, List<State> positions, int maxDepth) {
		int comparisons = 0, mismatchPVS = 0, mismatchAB = 0;
		long nodesABtt = 0, nodesPVStt = 0;
		for (int p = 0; p < positions.size(); p++) {
			State pos = positions.get(p);
			for (int d = 1; d <= maxDepth; d++) {
				Negamax noTT = new Negamax(ev, d, false, 0, false);
				noTT.findBestMove(pos, 600_000L);
				Negamax abTT = new Negamax(ev, d, true, 1 << 16, false);
				Negamax pvsTT = new Negamax(ev, d, true, 1 << 16, true);
				abTT.findBestMove(pos, 600_000L);
				Action m = pvsTT.findBestMove(pos, 600_000L);
				comparisons++;
				nodesABtt += abTT.lastNodeCount;
				nodesPVStt += pvsTT.lastNodeCount;
				if (Math.abs(abTT.lastScore - noTT.lastScore) > 1e-9) {
					mismatchAB++;
				}
				if (Math.abs(pvsTT.lastScore - noTT.lastScore) > 1e-9) {
					mismatchPVS++;
					System.out.println("  (info) " + name + " pos " + p + " d=" + d + ": PVS+TT=" + pvsTT.lastScore + " senza TT="
							+ noTT.lastScore);
				}
				// qualunque sia il valore, la mossa di PVS+TT deve essere legale
				boolean legal = false;
				for (Action a : AshtonRules.generateMoves(pos)) {
					legal |= sameMove(a, m);
				}
				if (!legal) {
					fail(name + ": mossa illegale con PVS+TT");
				}
			}
		}
		// l'invariante forte e' che PVS+TT non sia PEGGIO dell'alpha-beta+TT in coerenza con la ricerca senza TT
		if (mismatchPVS > mismatchAB) {
			fail(name + " [TT on]: PVS+TT diverge dalla ricerca senza TT piu' spesso dell'alpha-beta+TT (" + mismatchPVS + " vs "
					+ mismatchAB + " su " + comparisons + ")");
		}
		System.out.printf("%s %-24s %d confronti, divergenze dal TT-off: alpha-beta+TT=%d, PVS+TT=%d | nodi alpha-beta+TT %d vs PVS+TT %d (%.1f%%)%n",
				mismatchPVS <= mismatchAB ? "OK  " : "FAIL", "[" + name + ", TT on]", comparisons, mismatchAB, mismatchPVS,
				nodesABtt, nodesPVStt, 100.0 * (nodesPVStt - nodesABtt) / nodesABtt);
	}

	// --------------------------------------------- 3b: storia di ripetizione reale

	private static State initial() {
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		return s;
	}

	/** Come in RepetitionAndStalemateCheck: cerca (sulle mosse LEGALI) un ciclo di 4 semi-mosse senza catture che ricrea la posizione. */
	static Object[] buildCycle() {
		for (long seed = 1; seed < 80; seed++) {
			State p0 = initial();
			Random rnd = new Random(seed);
			for (int i = 0; i < 6; i++) {
				List<Action> moves = AshtonRules.generateMoves(p0);
				Action chosen = moves.get(rnd.nextInt(moves.size()));
				State next = AshtonRules.applyMove(p0, chosen);
				if (AshtonRules.wasCapture(p0, next, chosen) || AshtonRules.isTerminal(next)) {
					continue;
				}
				p0 = next;
			}
			if (p0.getTurn() != Turn.WHITE) {
				continue;
			}
			for (Action m1 : AshtonRules.generateMoves(p0)) {
				State p1 = AshtonRules.applyMove(p0, m1);
				if (AshtonRules.wasCapture(p0, p1, m1)) {
					continue;
				}
				for (Action m2 : AshtonRules.generateMoves(p1)) {
					State p2 = AshtonRules.applyMove(p1, m2);
					if (AshtonRules.wasCapture(p1, p2, m2)) {
						continue;
					}
					Action m3 = reverse(p2, m1);
					if (m3 == null) {
						continue;
					}
					State p3 = AshtonRules.applyMove(p2, m3);
					if (AshtonRules.wasCapture(p2, p3, m3)) {
						continue;
					}
					Action m4 = reverse(p3, m2);
					if (m4 == null) {
						continue;
					}
					State p4 = AshtonRules.applyMove(p3, m4);
					if (ZobristHasher.fullHash(p4) == ZobristHasher.fullHash(p0)) {
						return new Object[] { p0, p1, p2, p3 };
					}
				}
			}
		}
		throw new IllegalStateException("nessun ciclo di ripetizione trovato");
	}

	static Action reverse(State s, Action orig) {
		for (Action a : AshtonRules.generateMoves(s)) {
			if (a.getFrom().equals(orig.getTo()) && a.getTo().equals(orig.getFrom())) {
				return a;
			}
		}
		return null;
	}

	static void checkRepetitionHistory() {
		Object[] cyc = buildCycle();
		State p0 = (State) cyc[0], p1 = (State) cyc[1], p2 = (State) cyc[2], p3 = (State) cyc[3];
		List<Long> history = new ArrayList<>();
		history.add(ZobristHasher.fullHash(p0));
		history.add(ZobristHasher.fullHash(p1));
		history.add(ZobristHasher.fullHash(p2));
		history.add(ZobristHasher.fullHash(p3));
		Evaluator ev = new SimpleEvaluator();
		int before = failures;
		long reps = 0;
		for (int d = 2; d <= 5; d++) {
			Negamax ab = new Negamax(ev, d, false, 0, false);
			Negamax pvs = new Negamax(ev, d, false, 0, true);
			Action mAB = ab.findBestMove(p3, 600_000L, history);
			long repAB = ab.repetitionEventCounter;
			Action mPVS = pvs.findBestMove(p3, 600_000L, history);
			reps += repAB;
			if (Math.abs(ab.lastScore - pvs.lastScore) > 1e-9) {
				fail("storia di ripetizione, d=" + d + ": alpha-beta=" + ab.lastScore + " PVS=" + pvs.lastScore);
			}
			double v = exactValueOfMove(ev, p3, mPVS, pvs.lastDepthReached, history);
			if (Math.abs(v - pvs.lastScore) > 1e-9) {
				fail("storia di ripetizione, d=" + d + ": la mossa di PVS vale " + v + " ma la radice " + pvs.lastScore);
			}
		}
		if (reps == 0) {
			fail("lo scenario non ha mai incontrato una ripetizione: il test sarebbe vuoto");
		}
		System.out.printf("%s %-24s profondita' 2..5 con storia reale di 4 posizioni (%d ripetizioni incontrate dall'alpha-beta)%n",
				failures == before ? "OK  " : "FAIL", "[ripetizione]", reps);
	}

	// ------------------------------------------------------------ 4: timeUp

	static void checkTimeUpLegal(Evaluator ev, List<State> positions, Random rnd) {
		int before = failures, runs = 0;
		long[] budgets = { 1, 20, 60, 120, 450 };
		for (State pos : positions) {
			for (long b : budgets) {
				Negamax n = new Negamax(ev, 40, true, 1 << 14, true);
				Action m = n.findBestMove(pos, b);
				runs++;
				boolean legal = false;
				for (Action a : AshtonRules.generateMoves(pos)) {
					legal |= m != null && sameMove(a, m);
				}
				if (!legal) {
					fail("budget " + b + " ms: mossa non legale o null");
				}
			}
		}
		System.out.printf("%s %-24s %d ricerche con budget da 1 a 450 ms: mossa sempre legale%n",
				failures == before ? "OK  " : "FAIL", "[timeUp]", runs);
	}

	// ---------------------------------------------------------- posizioni

	/** Posizioni a stadi diversi di partita, con catture e vie di fuga (gioco casuale con preferenza per mosse "tattiche"). */
	static List<State> collectPositions(Random rnd, int count) {
		List<State> positions = new ArrayList<>();
		positions.add(initial());
		while (positions.size() < count) {
			State state = initial();
			int targetPly = 3 + rnd.nextInt(45);
			double greedy = rnd.nextDouble() * 0.7;
			for (int ply = 0; ply < targetPly; ply++) {
				if (AshtonRules.isTerminal(state)) {
					break;
				}
				List<Action> moves = AshtonRules.generateMoves(state);
				if (moves.isEmpty()) {
					break;
				}
				Action chosen = null;
				if (rnd.nextDouble() < greedy) {
					List<Action> good = new ArrayList<>();
					for (Action m : moves) {
						State c = AshtonRules.applyMove(state, m);
						if (AshtonRules.isTerminal(c) || AshtonRules.wasCapture(state, c, m)) {
							good.add(m);
						}
					}
					if (!good.isEmpty()) {
						chosen = good.get(rnd.nextInt(good.size()));
					}
				}
				state = AshtonRules.applyMove(state, chosen != null ? chosen : moves.get(rnd.nextInt(moves.size())));
			}
			if (!AshtonRules.isTerminal(state) && !AshtonRules.generateMoves(state).isEmpty()) {
				positions.add(state);
			}
		}
		return positions;
	}
}
