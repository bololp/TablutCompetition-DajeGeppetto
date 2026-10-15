package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Test "white-box" (stesso package di {@link Negamax}) delle ottimizzazioni della ricerca: Principal Variation
 * Search, killer moves, history heuristic. Sono tutte ottimizzazioni che NON devono cambiare il risultato (solo il
 * numero di nodi), ma se sbagliate producono mosse leggermente errate SENZA mai un errore visibile: quindi si verifica a
 * fondo che, per OGNI configurazione (PVS da solo, +killer, +history, +entrambi, alpha-beta+ordinamento):
 * <ol>
 * <li>il valore alla radice sia IDENTICO a quello dell'alpha-beta puro (senza TT, dove l'equivalenza teorica e' esatta),
 * con un evaluator a valori quasi interi ({@link SimpleEvaluator}: molti pareggi) e con uno a valori continui (rete
 * neurale con pesi casuali);</li>
 * <li>la mossa scelta sia davvero OTTIMA: il suo valore esatto, ricalcolato con un alpha-beta INDIPENDENTE scritto qui
 * (senza TT, PVS, ordinamento ne' orologio), coincida col valore di radice;</li>
 * <li>lo stesso valga con la transposition table accesa, con una storia di ripetizione reale e con una istanza di
 * Negamax riusata per piu' mosse consecutive (come fa il client: TT e history persistono, le killer no);</li>
 * <li>con budget di tempo minuscoli la mossa restituita sia sempre legale;</li>
 * <li>l'ordinamento sia una PERMUTAZIONE delle mosse (nessuna persa o duplicata), con punteggi non crescenti, mossa TT
 * per prima e killer subito dopo; l'indice mossa coincida con i getter di {@link Action}; killer e history dopo una
 * ricerca vera contengano valori ben formati (e non siano vuote, altrimenti il test sarebbe vacuo).</li>
 * </ol>
 * Stampa anche nodi e tasso di ri-ricerca (informativo, non asserito).
 */
public class SearchOptimizationsCheck {

	static int failures = 0;
	static final double INF = 1e12;

	/** Una configurazione di ricerca da verificare contro l'alpha-beta puro. */
	static final class Cfg {
		final String name;
		final boolean pvs;
		final int order;

		Cfg(String name, boolean pvs, int order) {
			this.name = name;
			this.pvs = pvs;
			this.order = order;
		}

		Negamax make(Evaluator ev, int depth, boolean tt) {
			return new Negamax(ev, depth, tt, tt ? 1 << 16 : 0, pvs, order).setThreads(1);
		}
	}

	static final Cfg BASE = new Cfg("alpha-beta puro", false, Negamax.ORDER_NONE);
	static final Cfg[] CFGS = { new Cfg("PVS", true, Negamax.ORDER_NONE),
			new Cfg("PVS+killer", true, Negamax.ORDER_KILLERS),
			new Cfg("PVS+history", true, Negamax.ORDER_HISTORY),
			new Cfg("PVS+killer+history", true, Negamax.ORDER_DEFAULT),
			new Cfg("alpha-beta+killer+history", false, Negamax.ORDER_DEFAULT) };

	public static void main(String[] args) {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 2024L;
		int numPositions = args.length > 1 ? Integer.parseInt(args[1]) : 16;
		Random rnd = new Random(seed);
		List<State> positions = collectPositions(rnd, numPositions);
		System.out.println("Posizioni di test: " + positions.size());

		Evaluator simple = new SimpleEvaluator();
		Evaluator neural = NeuralEvaluator.randomInit(5); // valori continui, quasi tutti distinti

		checkMoveIndex(positions);
		checkOrderingUnits(positions, rnd);
		checkAllChildrenVisited(positions, rnd);
		checkSearchTables(simple, positions);
		for (Cfg c : CFGS) {
			checkEquivalence(c, "Simple", simple, positions, 4);
			checkEquivalence(c, "Rete (continui)", neural, positions, 3);
		}
		for (Cfg c : CFGS) {
			checkWithTT(c, "Simple", simple, positions, 4);
		}
		checkWithTT(CFGS[3], "Rete (continui)", neural, positions, 3);
		for (Cfg c : CFGS) {
			checkRepetitionHistory(c);
		}
		checkReusedInstance(simple, positions);
		checkTimeUpLegal(simple, positions);

		if (failures == 0) {
			System.out.println("=== SEARCH OPTIMIZATIONS CHECK OK ===");
		} else {
			System.out.println("=== SEARCH OPTIMIZATIONS CHECK FALLITO: " + failures + " discrepanze ===");
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

	// ------------------------------------------------------- indice mossa e ordinamento

	/** moveIndex (due charAt) deve coincidere con la definizione via getter di Action (riga*9+colonna). */
	static void checkMoveIndex(List<State> positions) {
		int checked = 0, before = failures;
		for (State s : positions) {
			for (Turn t : new Turn[] { Turn.WHITE, Turn.BLACK }) {
				State c = s.clone();
				c.setTurn(t);
				for (Action a : AshtonRules.generateMoves(c)) {
					int expected = (a.getRowFrom() * 9 + a.getColumnFrom()) * 81 + a.getRowTo() * 9 + a.getColumnTo();
					if (Negamax.moveIndex(a) != expected) {
						fail("moveIndex " + Negamax.moveIndex(a) + " != " + expected + " per " + a.getFrom() + "->" + a.getTo());
						return;
					}
					checked++;
				}
			}
		}
		System.out.printf("%s %-30s %d mosse: moveIndex coincide con i getter di Action%n", failures == before ? "OK  " : "FAIL",
				"[indice mossa]", checked);
	}

	/** Punteggio atteso, calcolato in modo indipendente da computeOrderScores. */
	static long expectedScore(Action a, int ttIdx, int k0, int k1, int[] hist) {
		int idx = (a.getRowFrom() * 9 + a.getColumnFrom()) * 81 + a.getRowTo() * 9 + a.getColumnTo();
		if (idx == ttIdx) {
			return Integer.MAX_VALUE;
		}
		if (idx == k0) {
			return Integer.MAX_VALUE - 1L;
		}
		if (idx == k1) {
			return Integer.MAX_VALUE - 2L;
		}
		return hist[idx];
	}

	static void checkOrderingUnits(List<State> positions, Random rnd) {
		int before = failures, checks = 0;
		for (State s : positions) {
			for (int side = 0; side < 2; side++) {
				State c = s.clone();
				c.setTurn(side == 0 ? Turn.WHITE : Turn.BLACK);
				List<Action> moves = AshtonRules.generateMoves(c);
				if (moves.size() < 4) {
					continue;
				}
				Negamax n = new Negamax(new SimpleEvaluator(), 6, false, 0, true, Negamax.ORDER_DEFAULT).setThreads(1);
				int ply = 1 + rnd.nextInt(4);
				// stato "sporco": killer e history casuali, anche con valori uguali tra mosse diverse (pareggi)
				Action ka = moves.get(rnd.nextInt(moves.size())), kb = moves.get(rnd.nextInt(moves.size()));
				n.killers[ply][0] = Negamax.moveIndex(ka);
				n.killers[ply][1] = Negamax.moveIndex(kb);
				for (Action a : moves) {
					n.historyScore[side][Negamax.moveIndex(a)] = rnd.nextInt(4) == 0 ? 0 : rnd.nextInt(1000);
				}
				int ttPacked = rnd.nextBoolean() ? TranspositionTable.packMove(moves.get(rnd.nextInt(moves.size()))) : -1;
				int ttIdx = -1;
				if (ttPacked >= 0) {
					ttIdx = (TranspositionTable.packedFromRow(ttPacked) * 9 + TranspositionTable.packedFromCol(ttPacked)) * 81
							+ TranspositionTable.packedToRow(ttPacked) * 9 + TranspositionTable.packedToCol(ttPacked);
				}
				List<Action> original = new ArrayList<>(moves);
				int[] sc = n.computeOrderScores(moves, side, ply, ttPacked);
				List<Action> seq = new ArrayList<>();
				for (int i = 0; i < moves.size(); i++) {
					Negamax.selectBest(moves, sc, i);
					seq.add(moves.get(i));
				}
				checks++;
				// (a) permutazione: stesso insieme di mosse, nessuna duplicata (si confrontano i riferimenti)
				List<Action> a1 = new ArrayList<>(seq), a2 = new ArrayList<>(original);
				boolean perm = a1.size() == a2.size();
				for (Action a : a2) {
					perm &= a1.remove(a);
				}
				perm &= a1.isEmpty();
				if (!perm) {
					fail("l'ordinamento NON e' una permutazione delle mosse generate");
					return;
				}
				// (b) punteggi non crescenti lungo la sequenza (ricalcolati in modo indipendente)
				int k0 = n.killers[ply][0], k1 = n.killers[ply][1];
				long prev = Long.MAX_VALUE;
				for (Action a : seq) {
					long e = expectedScore(a, ttIdx, k0, k1, n.historyScore[side]);
					if (e > prev) {
						fail("ordine non decrescente violato: " + e + " dopo " + prev);
						return;
					}
					prev = e;
				}
				// (c) la mossa TT e' la prima se presente
				if (ttPacked >= 0) {
					Action f = seq.get(0);
					int fi = (f.getRowFrom() * 9 + f.getColumnFrom()) * 81 + f.getRowTo() * 9 + f.getColumnTo();
					if (fi != ttIdx) {
						fail("la mossa TT non e' la prima");
						return;
					}
				}
			}
		}
		System.out.printf("%s %-30s %d ordinamenti: permutazione, punteggi non crescenti, TT per prima, killer subito dopo%n",
				failures == before ? "OK  " : "FAIL", "[ordinamento]", checks);
	}

	/**
	 * A profondita' 1 con finestra piena (nessun taglio possibile) OGNI figlio non terminale deve essere valutato, per
	 * qualunque configurazione e con killer/history popolate a caso: intercetta un ciclo che salta mosse o un ordinamento che
	 * ne perde/duplica, anche quando quella mossa non sarebbe stata la migliore (e quindi il valore non cambierebbe).
	 */
	static void checkAllChildrenVisited(List<State> positions, Random rnd) {
		int before = failures, checks = 0;
		List<Cfg> all = new ArrayList<>();
		all.add(BASE);
		for (Cfg c : CFGS) {
			all.add(c);
		}
		for (State s : positions) {
			for (int side = 0; side < 2; side++) {
				final State root = s.clone();
				root.setTurn(side == 0 ? Turn.WHITE : Turn.BLACK);
				if (AshtonRules.isTerminal(root)) {
					continue;
				}
				Set<Long> expected = new HashSet<>();
				for (Action m : AshtonRules.generateMoves(root)) {
					State c = AshtonRules.applyMove(root, m);
					if (!AshtonRules.isTerminal(c)) { // i figli terminali non vengono valutati dall'evaluator
						expected.add(ZobristHasher.fullHash(c));
					}
				}
				for (Cfg cfg : all) {
					final Set<Long> seen = new HashSet<>();
					Evaluator recording = st -> {
						seen.add(ZobristHasher.fullHash(st));
						return 0.0;
					};
					Negamax n = cfg.make(recording, 4, false);
					// killer e history casuali (stato "sporco") per esercitare davvero la selezione
					for (int[] k : n.killers) {
						List<Action> mv = AshtonRules.generateMoves(root);
						k[0] = Negamax.moveIndex(mv.get(rnd.nextInt(mv.size())));
						k[1] = Negamax.moveIndex(mv.get(rnd.nextInt(mv.size())));
					}
					for (int sd = 0; sd < 2; sd++) {
						for (int i = 0; i < n.historyScore[sd].length; i++) {
							n.historyScore[sd][i] = rnd.nextInt(5);
						}
					}
					int persp = side == 0 ? 1 : -1;
					n.negamax(root, 1, -INF, INF, persp, ZobristHasher.fullHash(root), new ArrayList<Long>());
					checks++;
					if (!seen.equals(expected)) {
						fail(cfg.name + ": a profondita' 1 sono stati valutati " + seen.size() + " figli distinti su " + expected.size() + " attesi");
						return;
					}
				}
			}
		}
		System.out.printf("%s %-30s %d verifiche: a profondita' 1, finestra piena, ogni figlio viene valutato (tutte le configurazioni)%n",
				failures == before ? "OK  " : "FAIL", "[tutti i figli visitati]", checks);
	}

	/** Dopo una ricerca vera le tabelle devono essere ben formate e NON vuote (altrimenti l'ottimizzazione non farebbe nulla). */
	static void checkSearchTables(Evaluator ev, List<State> positions) {
		int before = failures;
		int withKillers = 0, nonZeroHist = 0, searches = 0;
		for (State s : positions) {
			Negamax n = new Negamax(ev, 4, true, 1 << 16, true, Negamax.ORDER_DEFAULT).setThreads(1);
			n.findBestMove(s, 600_000L);
			searches++;
			for (int[] k : n.killers) {
				if (k[0] != -1 || k[1] != -1) {
					withKillers++;
				}
				if (k[0] < -1 || k[0] >= 6561 || k[1] < -1 || k[1] >= 6561) {
					fail("killer fuori range: " + k[0] + "," + k[1]);
				}
				if (k[0] != -1 && k[0] == k[1]) {
					fail("le due killer di uno stesso ply coincidono");
				}
				if (k[0] == -1 && k[1] != -1) {
					fail("seconda killer valorizzata con la prima vuota");
				}
			}
			for (int side = 0; side < 2; side++) {
				for (int i = 0; i < n.historyScore[side].length; i++) {
					int h = n.historyScore[side][i];
					if (h < 0) {
						fail("history negativa");
						return;
					}
					if (h > 0) {
						nonZeroHist++;
						int from = i / 81, to = i % 81;
						boolean sameLine = (from / 9 == to / 9) || (from % 9 == to % 9);
						if (!sameLine || from == to) {
							fail("history su una mossa che non e' una mossa a torre: da " + from + " a " + to);
							return;
						}
					}
				}
			}
		}
		if (withKillers == 0 || nonZeroHist == 0) {
			fail("killer o history mai popolate: il test sarebbe vacuo");
		}
		System.out.printf("%s %-30s %d ricerche: tabelle ben formate (%d ply con killer, %d voci history non nulle)%n",
				failures == before ? "OK  " : "FAIL", "[tabelle dopo ricerca]", searches, withKillers, nonZeroHist);
	}

	// ----------------------------------------------- equivalenza con l'alpha-beta puro

	static void checkEquivalence(Cfg cfg, String name, Evaluator ev, List<State> positions, int maxDepth) {
		long nodesBase = 0, nodesCfg = 0;
		int comparisons = 0, before = failures;
		for (int p = 0; p < positions.size(); p++) {
			State pos = positions.get(p);
			for (int d = 1; d <= maxDepth; d++) {
				Negamax base = BASE.make(ev, d, false);
				Negamax cand = cfg.make(ev, d, false);
				Action mBase = base.findBestMove(pos, 600_000L);
				Action mCand = cand.findBestMove(pos, 600_000L);
				comparisons++;
				nodesBase += base.lastNodeCount;
				nodesCfg += cand.lastNodeCount;
				String where = cfg.name + " / " + name + " pos " + p + " d=" + d;
				if (base.lastDepthReached != cand.lastDepthReached) {
					fail(where + ": profondita' diverse " + base.lastDepthReached + " vs " + cand.lastDepthReached);
					continue;
				}
				if (Math.abs(base.lastScore - cand.lastScore) > 1e-9) {
					fail(where + ": VALORE DIVERSO alpha-beta=" + base.lastScore + " candidato=" + cand.lastScore + "\n" + pos);
					continue;
				}
				// l'alpha-beta di Negamax deve coincidere con l'oracolo indipendente (valida anche la baseline)
				if (cfg == CFGS[0]) {
					double vb = exactValueOfMove(ev, pos, mBase, base.lastDepthReached, new ArrayList<Long>());
					if (Math.abs(vb - base.lastScore) > 1e-9) {
						fail(where + ": l'alpha-beta di Negamax diverge dall'oracolo (" + base.lastScore + " vs " + vb + ")");
					}
				}
				// la mossa del candidato deve essere ottima: valore esatto == valore di radice
				double v = exactValueOfMove(ev, pos, mCand, cand.lastDepthReached, new ArrayList<Long>());
				if (Math.abs(v - cand.lastScore) > 1e-9) {
					fail(where + ": la mossa " + mCand.getFrom() + "->" + mCand.getTo() + " vale " + v + " ma il valore di radice e' "
							+ cand.lastScore);
				}
				// e nessuna altra mossa deve valere di piu' (verifica indipendente del massimo)
				if (d <= 3 && cfg == CFGS[3]) {
					for (Action m : AshtonRules.generateMoves(pos)) {
						double vm = exactValueOfMove(ev, pos, m, cand.lastDepthReached, new ArrayList<Long>());
						if (vm > cand.lastScore + 1e-9) {
							fail(where + ": esiste una mossa migliore " + m.getFrom() + "->" + m.getTo() + " (" + vm + " > "
									+ cand.lastScore + ")");
							break;
						}
					}
				}
			}
		}
		System.out.printf("%s %-26s %-16s TT off: %d confronti, nodi alpha-beta %d -> %d (%+.1f%%)%n",
				failures == before ? "OK  " : "FAIL", "[" + cfg.name + "]", name, comparisons, nodesBase, nodesCfg,
				100.0 * (nodesCfg - nodesBase) / nodesBase);
	}

	// ----------------------------------------------------------- con la TT

	/**
	 * Con la TT accesa l'equivalenza esatta puo' rompersi per cause che NON dipendono dalle ottimizzazioni (una voce di
	 * profondita' maggiore riusata a profondita' minore): quindi si confronta il candidato+TT con la ricerca SENZA TT come
	 * fa TTCorrectnessCheck, e il candidato non deve divergere piu' spesso dell'alpha-beta+TT.
	 */
	static void checkWithTT(Cfg cfg, String name, Evaluator ev, List<State> positions, int maxDepth) {
		int comparisons = 0, mismatchCand = 0, mismatchBase = 0;
		long nodesBaseTT = 0, nodesCandTT = 0;
		for (int p = 0; p < positions.size(); p++) {
			State pos = positions.get(p);
			for (int d = 1; d <= maxDepth; d++) {
				Negamax noTT = BASE.make(ev, d, false);
				noTT.findBestMove(pos, 600_000L);
				Negamax baseTT = BASE.make(ev, d, true);
				Negamax candTT = cfg.make(ev, d, true);
				baseTT.findBestMove(pos, 600_000L);
				Action m = candTT.findBestMove(pos, 600_000L);
				comparisons++;
				nodesBaseTT += baseTT.lastNodeCount;
				nodesCandTT += candTT.lastNodeCount;
				if (Math.abs(baseTT.lastScore - noTT.lastScore) > 1e-9) {
					mismatchBase++;
				}
				if (Math.abs(candTT.lastScore - noTT.lastScore) > 1e-9) {
					mismatchCand++;
					System.out.println("  (info) " + cfg.name + " / " + name + " pos " + p + " d=" + d + ": con TT=" + candTT.lastScore
							+ " senza TT=" + noTT.lastScore);
				}
				boolean legal = false;
				for (Action a : AshtonRules.generateMoves(pos)) {
					legal |= sameMove(a, m);
				}
				if (!legal) {
					fail(cfg.name + ": mossa illegale con la TT accesa");
				}
			}
		}
		if (mismatchCand > mismatchBase) {
			fail(cfg.name + " / " + name + " [TT on]: diverge dalla ricerca senza TT piu' spesso dell'alpha-beta+TT (" + mismatchCand
					+ " vs " + mismatchBase + " su " + comparisons + ")");
		}
		System.out.printf("%s %-26s %-16s TT on: %d confronti, divergenze dal TT-off: alpha-beta+TT=%d, candidato+TT=%d | nodi %d -> %d (%+.1f%%)%n",
				mismatchCand <= mismatchBase ? "OK  " : "FAIL", "[" + cfg.name + "]", name, comparisons, mismatchBase, mismatchCand,
				nodesBaseTT, nodesCandTT, 100.0 * (nodesCandTT - nodesBaseTT) / nodesBaseTT);
	}

	/**
	 * Una stessa istanza di Negamax riusata per mosse consecutive (come fa il client): la TT e la history persistono tra una
	 * ricerca e l'altra, le killer no. Il valore di ogni mossa deve coincidere con quello di una ricerca fresca senza TT
	 * (ammettendo, come sopra, solo divergenze che l'alpha-beta+TT riusata mostra a sua volta).
	 */
	static void checkReusedInstance(Evaluator ev, List<State> positions) {
		int before = failures, compared = 0, mismatchCand = 0, mismatchBase = 0;
		for (int p = 0; p < Math.min(6, positions.size()); p++) {
			State s = positions.get(p);
			Negamax cand = new Negamax(ev, 3, true, 1 << 16, true, Negamax.ORDER_DEFAULT).setThreads(1);
			Negamax baseTT = new Negamax(ev, 3, true, 1 << 16, false, Negamax.ORDER_NONE).setThreads(1);
			for (int ply = 0; ply < 8 && !AshtonRules.isTerminal(s); ply++) {
				Negamax fresh = BASE.make(ev, 3, false);
				Action mf = fresh.findBestMove(s, 600_000L);
				Action m = cand.findBestMove(s, 600_000L);
				baseTT.findBestMove(s, 600_000L);
				compared++;
				if (Math.abs(cand.lastScore - fresh.lastScore) > 1e-9) {
					mismatchCand++;
				}
				if (Math.abs(baseTT.lastScore - fresh.lastScore) > 1e-9) {
					mismatchBase++;
				}
				s = AshtonRules.applyMove(s, mf != null ? mf : m); // si avanza lungo la linea dell'alpha-beta puro
			}
		}
		if (mismatchCand > mismatchBase) {
			fail("istanza riusata: " + mismatchCand + " divergenze contro " + mismatchBase + " dell'alpha-beta+TT riusata");
		}
		System.out.printf("%s %-30s %d mosse consecutive: divergenze dal fresco senza TT: alpha-beta+TT=%d, PVS+killer+history=%d%n",
				failures == before ? "OK  " : "FAIL", "[istanza riusata]", compared, mismatchBase, mismatchCand);
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

	static void checkRepetitionHistory(Cfg cfg) {
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
			Negamax base = BASE.make(ev, d, false);
			Negamax cand = cfg.make(ev, d, false);
			base.findBestMove(p3, 600_000L, history);
			long repBase = base.repetitionEventCounter;
			Action m = cand.findBestMove(p3, 600_000L, history);
			reps += repBase;
			if (Math.abs(base.lastScore - cand.lastScore) > 1e-9) {
				fail(cfg.name + ", storia di ripetizione, d=" + d + ": alpha-beta=" + base.lastScore + " candidato=" + cand.lastScore);
			}
			double v = exactValueOfMove(ev, p3, m, cand.lastDepthReached, history);
			if (Math.abs(v - cand.lastScore) > 1e-9) {
				fail(cfg.name + ", storia di ripetizione, d=" + d + ": la mossa vale " + v + " ma la radice " + cand.lastScore);
			}
		}
		if (reps == 0) {
			fail("lo scenario non ha mai incontrato una ripetizione: il test sarebbe vuoto");
		}
		System.out.printf("%s %-26s ripetizione: profondita' 2..5 con storia reale (%d ripetizioni incontrate dall'alpha-beta)%n",
				failures == before ? "OK  " : "FAIL", "[" + cfg.name + "]", reps);
	}

	// ------------------------------------------------------------ timeUp

	static void checkTimeUpLegal(Evaluator ev, List<State> positions) {
		int before = failures, runs = 0;
		long[] budgets = { 1, 20, 60, 120, 450 };
		for (State pos : positions) {
			for (long b : budgets) {
				Negamax n = new Negamax(ev, 40, true, 1 << 14, true, Negamax.ORDER_DEFAULT).setThreads(1);
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
		System.out.printf("%s %-30s %d ricerche con budget da 1 a 450 ms: mossa sempre legale%n",
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
