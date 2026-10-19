package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.aiplayer.SearchOptimizationsCheck.Cfg;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Verifica le opzioni di ordinamento ESTESE di {@link Negamax} ({@link Negamax#ORDER_WINS}, {@code ORDER_CAPTURES},
 * {@code ORDER_CAPTURES_FIRST}, {@code ORDER_COUNTER}, {@code ORDER_KILLER_CARRY}, {@code ORDER_ROOT_SCORES}).
 * <p>
 * Sono solo ordinamento: NON devono cambiare il valore di nessun nodo. Quindi:
 * <ol>
 * <li><b>punteggi</b> ({@code computeOrderScoresExt}): per ognuna delle combinazioni dei 4 bit che usano il punteggio esteso,
 * con killer/history/countermove/mossa TT casuali ("stato sporco"), la sequenza prodotta dalla selezione pigra deve
 * rispettare i LIVELLI ricostruiti da un oracolo indipendente (il valore tattico di ogni mossa viene dal motore storico
 * {@code AshtonRules}: vittoria immediata e pedine perse), con il sotto-ordine (numero di catture, history) e come permutazione;</li>
 * <li><b>valore</b>: per ogni configurazione, valore di radice e profondita' identici all'alpha-beta puro (TT spenta), mossa
 * ottima, con TT, con una storia di ripetizione reale, con una istanza riusata per mosse consecutive (la' dove killer carry e
 * ordine radice hanno effetto), a profondita' 1 ogni figlio viene valutato, con 4 thread (Lazy SMP) mossa sempre legale;</li>
 * <li><b>meccanismi</b>: la mossa che porta a un nodo e' registrata nel ply giusto (si guarda la scacchiera di ogni foglia), la
 * tabella countermove si popola ed e' ben formata, lo slittamento delle killer e' di 2 ply, l'ordinamento della radice e'
 * stabile, decrescente e porta la mossa migliore in testa.</li>
 * </ol>
 * Uso: {@code OrderingExtCheck [seme=2024] [posizioni=8] [parti=all|unit|eq|tt|rep|reuse|visit|smp|mech]}.
 */
public class OrderingExtCheck {

	static int failures = 0;

	static void fail(String m) {
		failures++;
		if (failures <= 40) {
			System.out.println("FAIL - " + m);
		}
	}

	static final int BASE = Negamax.ORDER_DEFAULT;

	/** Le configurazioni di ricerca da verificare contro l'alpha-beta puro. */
	static Cfg[] cfgs() {
		final int all = BASE | Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES | Negamax.ORDER_COUNTER | Negamax.ORDER_KILLER_CARRY
				| Negamax.ORDER_ROOT_SCORES;
		return new Cfg[] { new Cfg("+wins", true, BASE | Negamax.ORDER_WINS, true),
				new Cfg("+caps", true, BASE | Negamax.ORDER_CAPTURES),
				new Cfg("+capsFirst", true, BASE | Negamax.ORDER_CAPTURES_FIRST),
				new Cfg("+counter", true, BASE | Negamax.ORDER_COUNTER),
				new Cfg("+kcarry", true, BASE | Negamax.ORDER_KILLER_CARRY),
				new Cfg("+rootsc", true, BASE | Negamax.ORDER_ROOT_SCORES, true),
				new Cfg("wins+caps (senza killer/history)", true, Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES),
				new Cfg("ALL", true, all, true),
				new Cfg("ALL, capsFirst, alpha-beta", false, (all & ~Negamax.ORDER_CAPTURES) | Negamax.ORDER_CAPTURES_FIRST) };
	}

	public static void main(String[] args) {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 2024L;
		int numPositions = args.length > 1 ? Integer.parseInt(args[1]) : 8;
		String parts = args.length > 2 ? args[2] : "all";
		Random rnd = new Random(seed);
		List<State> positions = SearchOptimizationsCheck.collectPositions(rnd, numPositions);
		System.out.println("Posizioni di test: " + positions.size() + " (seme " + seed + ", parti: " + parts + ")");
		Evaluator simple = new SimpleEvaluator();
		Evaluator neural = NeuralEvaluator.randomInit(5);
		Cfg[] cfgs = cfgs();
		boolean all = parts.equals("all");

		if (all || parts.contains("unit")) {
			checkScoresExt(positions, rnd);
			checkSortRoot(rnd);
		}
		if (all || parts.contains("mech")) {
			checkKillerRoll(rnd);
			checkPrevMoveAndCounter(positions);
			checkRootOrder(positions);
			checkDispatch(positions);
			checkNewSearchAndMargin(positions);
		}
		if (all || parts.contains("visit")) {
			checkAllChildrenVisited(cfgs, positions, rnd);
		}
		if (all || parts.contains("eq")) {
			for (Cfg c : cfgs) {
				SearchOptimizationsCheck.checkEquivalence(c, "Simple", simple, positions, 4);
				SearchOptimizationsCheck.checkEquivalence(c, "Rete (continui)", neural, positions, 3);
			}
		}
		if (all || parts.contains("tt")) {
			for (Cfg c : cfgs) {
				SearchOptimizationsCheck.checkWithTT(c, "Simple", simple, positions, 4);
			}
		}
		if (all || parts.contains("rep")) {
			for (Cfg c : cfgs) {
				SearchOptimizationsCheck.checkRepetitionHistory(c);
			}
		}
		if (all || parts.contains("reuse")) {
			for (Cfg c : cfgs) {
				checkReusedInstance(c, simple, positions);
			}
		}
		if (all || parts.contains("smp")) {
			checkParallel(cfgs, simple, positions);
		}

		int total = failures + SearchOptimizationsCheck.failures;
		if (total == 0) {
			System.out.println("=== ORDERING EXT CHECK OK ===");
		} else {
			System.out.println("=== ORDERING EXT CHECK FALLITO: " + total + " discrepanze ===");
			System.exit(1);
		}
	}

	/** Posizione vuota (solo il trono) con le pedine indicate {@code {riga, colonna, tipo}}, tipo: 'W', 'B', 'K'. */
	static State handBuilt(int[][] pieces, char[] kinds) {
		State s = new it.unibo.ai.didattica.competition.tablut.domain.StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = (r == 4 && c == 4) ? Pawn.THRONE : Pawn.EMPTY;
			}
		}
		for (int i = 0; i < pieces.length; i++) {
			b[pieces[i][0]][pieces[i][1]] = kinds[i] == 'W' ? Pawn.WHITE : (kinds[i] == 'B' ? Pawn.BLACK : Pawn.KING);
		}
		s.setTurn(Turn.WHITE);
		return s;
	}

	/**
	 * Posizioni COSTRUITE (non dipendono dal campione casuale): (1) il Re ha due fughe immediate (colonna 2 verso l'alto e verso il
	 * basso) e il Bianco ha anche una cattura; (2) il Nero puo' catturare il Re spostando una pedina in (2,3) con una nera gia' in (2,1),
	 * e ha altre catture.
	 */
	static List<State> constructedPositions() {
		List<State> out = new ArrayList<>();
		out.add(handBuilt(new int[][] { { 4, 2 }, { 6, 6 }, { 7, 6 }, { 6, 7 }, { 3, 6 }, { 2, 6 } }, new char[] { 'K', 'W', 'B', 'B', 'W', 'B' }));
		out.add(handBuilt(new int[][] { { 2, 2 }, { 2, 1 }, { 2, 6 }, { 6, 4 }, { 6, 3 }, { 7, 3 }, { 6, 2 } }, new char[] { 'K', 'B', 'B', 'W', 'B', 'W', 'B' }));
		return out;
	}

	// ------------------------------------------------------------------------------------------ oracolo del valore tattico

	static int countPawns(State s, Pawn p) {
		int c = 0;
		for (Pawn[] row : s.getBoard()) {
			for (Pawn x : row) {
				if (x == p) {
					c++;
				}
			}
		}
		return c;
	}

	/** Valore tattico di una mossa calcolato SOLO con il motore storico: vittoria immediata, altrimenti pedine avversarie perse (Re escluso). */
	static int tacticalOracle(State before, Action a) {
		State after = AshtonRules.applyMove(before, a);
		if (after.getTurn() == Turn.WHITEWIN || after.getTurn() == Turn.BLACKWIN) {
			return FastState.TACT_WIN;
		}
		Pawn victim = before.getTurn() == Turn.WHITE ? Pawn.BLACK : Pawn.WHITE;
		return countPawns(before, victim) - countPawns(after, victim);
	}

	// ------------------------------------------------------------------------------------------ 1. punteggi

	static final int T_TT = 0, T_WIN = 1, T_CAP = 2, T_K0 = 3, T_K1 = 4, T_CTR = 5, T_QUIET = 6;

	/** Livello di una mossa secondo la SPECIFICA (non secondo le costanti del codice): primo livello della lista il cui predicato vale. */
	static int tierOf(int mode, boolean isTT, int tv, boolean isK0, boolean isK1, boolean isCtr) {
		final boolean capsOn = (mode & (Negamax.ORDER_CAPTURES | Negamax.ORDER_CAPTURES_FIRST)) != 0;
		final boolean capsFirst = (mode & Negamax.ORDER_CAPTURES_FIRST) != 0;
		final boolean winsOn = (mode & Negamax.ORDER_WINS) != 0;
		final boolean[] pred = new boolean[7];
		pred[T_TT] = isTT;
		pred[T_WIN] = winsOn && tv == FastState.TACT_WIN;
		pred[T_CAP] = capsOn && tv > 0;
		pred[T_K0] = isK0;
		pred[T_K1] = isK1;
		pred[T_CTR] = isCtr;
		pred[T_QUIET] = true;
		final int[] order = capsFirst ? new int[] { T_TT, T_WIN, T_CAP, T_K0, T_K1, T_CTR, T_QUIET }
				: new int[] { T_TT, T_WIN, T_K0, T_K1, T_CTR, T_CAP, T_QUIET };
		for (int rank = 0; rank < order.length; rank++) {
			if (pred[order[rank]]) {
				return rank * 10 + (order[rank] == T_CAP ? 0 : 0);
			}
		}
		throw new IllegalStateException();
	}

	static void checkScoresExt(List<State> positions, Random rnd) {
		int before = failures, checks = 0, withCapture = 0, withWin = 0, withCounterHit = 0, withKillerCapture = 0;
		final int[] bits = { Negamax.ORDER_WINS, Negamax.ORDER_CAPTURES, Negamax.ORDER_CAPTURES_FIRST, Negamax.ORDER_COUNTER };
		final List<State> samples = new ArrayList<>(positions);
		samples.addAll(constructedPositions());
		for (int combo = 1; combo < 16; combo++) {
			int ext = 0;
			for (int b = 0; b < 4; b++) {
				if ((combo >> b & 1) != 0) {
					ext |= bits[b];
				}
			}
			for (int baseSel = 0; baseSel < 2; baseSel++) {
				final int mode = ext | (baseSel == 0 ? BASE : 0);
				final boolean useKillers = (mode & Negamax.ORDER_KILLERS) != 0, useHist = (mode & Negamax.ORDER_HISTORY) != 0;
				for (State s : samples) {
					for (int side = 0; side < 2; side++) {
						State c = s.clone();
						c.setTurn(side == 0 ? Turn.WHITE : Turn.BLACK);
						if (AshtonRules.isTerminal(c)) {
							continue;
						}
						List<Action> moves = AshtonRules.generateMoves(c);
						final int cnt = moves.size();
						if (cnt < 4) {
							continue;
						}
						Negamax n = new Negamax(new SimpleEvaluator(), 6, false, 0, true, mode).setThreads(1);
						n.installForTest(FastState.fromState(c), new ArrayList<Long>(), 4);
						final int ply = 1 + rnd.nextInt(4);
						int[] mv = new int[cnt];
						int[] tv = new int[cnt];
						for (int i = 0; i < cnt; i++) {
							mv[i] = Move.fromAction(moves.get(i));
							tv[i] = tacticalOracle(c, moves.get(i));
							if (tv[i] == FastState.TACT_WIN) {
								withWin++;
							} else if (tv[i] > 0) {
								withCapture++;
							}
						}
						// stato sporco, con pareggi e con killer/countermove che coincidono apposta con mosse tattiche
						final int ka = rnd.nextInt(cnt), kb = rnd.nextInt(4) == 0 ? ka : rnd.nextInt(cnt);
						n.killers[ply][0] = Move.index(mv[ka]);
						n.killers[ply][1] = Move.index(mv[kb]);
						int prevIdx = rnd.nextInt(4) == 0 ? -1 : rnd.nextInt(6561);
						int ctrIdx = -1;
						if (n.counterMove != null && prevIdx >= 0) {
							ctrIdx = Move.index(mv[rnd.nextInt(cnt)]);
							n.counterMove[side][prevIdx] = ctrIdx;
							withCounterHit++;
						}
						final int[] hist = n.historyScore[side];
						for (int i = 0; i < cnt; i++) {
							hist[Move.index(mv[i])] = rnd.nextInt(4) == 0 ? 0 : rnd.nextInt(1000);
						}
						final int ttPacked = rnd.nextBoolean() ? mv[rnd.nextInt(cnt)] : -1;
						final int ttIdx = ttPacked >= 0 ? Move.index(ttPacked) : -1;

						// specifica: livello (e sotto-ordine) di ogni mossa
						final int[] tier = new int[cnt], sub1 = new int[cnt], sub2 = new int[cnt];
						for (int i = 0; i < cnt; i++) {
							final int idx = Move.index(mv[i]);
							final boolean isK0 = useKillers && idx == n.killers[ply][0];
							final boolean isK1 = useKillers && idx == n.killers[ply][1];
							final boolean isCtr = ctrIdx >= 0 && idx == ctrIdx;
							tier[i] = tierOf(mode, idx == ttIdx, tv[i], isK0, isK1, isCtr);
							final boolean isCapTier = tier[i] == tierOfCap(mode);
							sub1[i] = isCapTier ? -(tv[i] == FastState.TACT_WIN ? 4 : tv[i]) : 0; // piu' catture prima
							sub2[i] = (isCapTier || tier[i] == tierOfQuiet(mode)) ? -(useHist ? hist[idx] : 0) : 0; // history decrescente
							if (isCapTier && (idx == n.killers[ply][0] || idx == n.killers[ply][1])) {
								withKillerCapture++;
							}
						}
						int[] sc = new int[cnt];
						int[] copy = mv.clone();
						n.computeOrderScoresExt(copy, cnt, side, ply, ttPacked, prevIdx, sc);
						checks++;
						// ordine prodotto dalla selezione pigra
						int[] seq = new int[cnt];
						for (int i = 0; i < cnt; i++) {
							Negamax.selectBest(copy, sc, i, cnt);
							seq[i] = copy[i];
						}
						int[] a = seq.clone(), b = mv.clone();
						Arrays.sort(a);
						Arrays.sort(b);
						if (!Arrays.equals(a, b)) {
							fail("mode=" + mode + ": l'ordinamento esteso NON e' una permutazione");
							return;
						}
						int[] pos = new int[6561 + 1];
						for (int i = 0; i < cnt; i++) {
							pos[Move.index(mv[i])] = i;
						}
						long prevKey = Long.MIN_VALUE;
						for (int i = 0; i < cnt; i++) {
							final int j = pos[Move.index(seq[i])];
							final long key = ((long) tier[j] << 40) + ((long) (sub1[j] + 100) << 20) + (sub2[j] + 1_000_000);
							if (key < prevKey) {
								fail("mode=" + mode + " side=" + side + ": ordine violato alla posizione " + i + " (livello " + tier[j] + ", catture "
										+ -sub1[j] + ", history " + -sub2[j] + ") dopo una chiave piu' alta\n" + c);
								return;
							}
							prevKey = key;
						}
					}
				}
			}
		}
		if (withCapture < 50 || withWin < 5 || withCounterHit < 50) {
			fail("il test dei punteggi e' vacuo: catture " + withCapture + ", vittorie " + withWin + ", countermove " + withCounterHit);
		}
		System.out.printf("%s %-30s %d ordinamenti (15 combinazioni x 2 basi): permutazione e livelli dell'oracolo (mosse con catture %d, vittorie %d, countermove %d, killer-cattura %d)%n",
				failures == before ? "OK  " : "FAIL", "[punteggi estesi]", checks, withCapture, withWin, withCounterHit, withKillerCapture);
	}

	/** Livello dell'oracolo delle catture / delle quiete (dipende dalla modalita': stessa lista di tierOf). */
	static int tierOfCap(int mode) {
		return ((mode & Negamax.ORDER_CAPTURES_FIRST) != 0 ? 2 : 5) * 10;
	}

	static int tierOfQuiet(int mode) {
		return 6 * 10;
	}

	static void checkSortRoot(Random rnd) {
		int before = failures, checks = 0;
		for (int t = 0; t < 400; t++) {
			final int n = 1 + rnd.nextInt(60);
			int[] moves = new int[n];
			double[] scores = new double[n];
			for (int i = 0; i < n; i++) {
				moves[i] = Move.of(i % 81, (i * 7 + 3) % 81 == i % 81 ? 0 : (i * 7 + 3) % 81) + (i << 14); // valori distinti
				scores[i] = rnd.nextInt(5) == 0 ? 0 : rnd.nextInt(8) - 4; // molti pareggi
			}
			final int bestAt = rnd.nextInt(n);
			double top = Double.NEGATIVE_INFINITY;
			for (double d : scores) {
				top = Math.max(top, d);
			}
			scores[bestAt] = top + 1; // la migliore ha il punteggio piu' alto (come dopo una vera iterazione)
			final int best = moves[bestAt];
			int[] expMoves = moves.clone();
			double[] expScores = scores.clone();
			// riferimento: ordinamento stabile di Java su oggetti
			Integer[] order = new Integer[n];
			for (int i = 0; i < n; i++) {
				order[i] = i;
			}
			final double[] sc = scores;
			Arrays.sort(order, (x, y) -> Double.compare(sc[y], sc[x])); // decrescente, stabile
			for (int i = 0; i < n; i++) {
				expMoves[i] = moves[order[i]];
				expScores[i] = scores[order[i]];
			}
			Negamax.sortRootByScores(moves, scores, n, best);
			checks++;
			if (!Arrays.equals(moves, expMoves) || !Arrays.equals(scores, expScores)) {
				fail("sortRootByScores diverge dall'ordinamento stabile di riferimento (n=" + n + ")");
				return;
			}
			if (moves[0] != best) {
				fail("sortRootByScores: la mossa migliore non e' in testa");
				return;
			}
		}
		// caso in cui la migliore NON ha il punteggio piu' alto (non deve capitare, ma la funzione la porta comunque in testa)
		int[] m = { 10, 20, 30, 40 };
		double[] sv = { 5, 4, 3, 9 };
		Negamax.sortRootByScores(m, sv, 4, 20);
		if (m[0] != 20 || m[1] != 40 || m[2] != 10 || m[3] != 30) {
			fail("sortRootByScores: la migliore forzata in testa non rispetta l'ordine dei rimanenti: " + Arrays.toString(m));
		}
		System.out.printf("%s %-30s %d ordinamenti: stabile, decrescente, migliore in testa (come Arrays.sort su oggetti)%n",
				failures == before ? "OK  " : "FAIL", "[ordine radice]", checks);
	}

	// ------------------------------------------------------------------------------------------ 3. meccanismi

	static void checkKillerRoll(Random rnd) {
		int before = failures;
		for (int mode : new int[] { BASE, BASE | Negamax.ORDER_KILLER_CARRY }) {
			Negamax n = new Negamax(new SimpleEvaluator(), 6, false, 0, true, mode).setThreads(1);
			final int len = n.killers.length;
			int[][] snapshot = new int[len][2];
			for (int p = 0; p < len; p++) {
				n.killers[p][0] = 100 + p;
				n.killers[p][1] = 500 + p;
				snapshot[p][0] = n.killers[p][0];
				snapshot[p][1] = n.killers[p][1];
			}
			n.rollKillersForNewSearch();
			final boolean carry = (mode & Negamax.ORDER_KILLER_CARRY) != 0;
			for (int p = 0; p < len; p++) {
				int e0 = -1, e1 = -1;
				if (carry && p + 2 < len) {
					e0 = snapshot[p + 2][0];
					e1 = snapshot[p + 2][1];
				}
				if (n.killers[p][0] != e0 || n.killers[p][1] != e1) {
					fail("killer " + (carry ? "carry" : "azzeramento") + ": ply " + p + " = " + n.killers[p][0] + "," + n.killers[p][1] + " atteso " + e0
							+ "," + e1);
				}
			}
		}
		System.out.printf("%s %-30s azzeramento (default) e slittamento di 2 ply (carry) delle killer%n", failures == before ? "OK  " : "FAIL",
				"[killer tra mosse]");
	}

	static boolean rookShaped(int idx) {
		final int from = idx / 81, to = idx % 81;
		return from != to && (from / 9 == to / 9 || from % 9 == to % 9);
	}

	/**
	 * (a) La mossa registrata per il nodo e' quella che lo ha davvero prodotto: in ogni foglia la cella di partenza e' vuota e
	 * quella di arrivo contiene una pedina di chi ha appena mosso. (b) Dopo la ricerca la tabella countermove e' popolata e le
	 * sue voci sono mosse "a torre" valide.
	 */
	static void checkPrevMoveAndCounter(List<State> positions) {
		int before = failures;
		long leaves = 0;
		int entries = 0;
		for (int p = 0; p < Math.min(6, positions.size()); p++) {
			final Negamax[] holder = new Negamax[1];
			final long[] cnt = { 0 };
			final int[] bad = { 0 };
			Evaluator ev = st -> {
				final Negamax nm = holder[0];
				final int idx = nm.prevMoveAt[nm.rootIterDepth]; // un nodo a profondita' 0 ha indice di stack = profondita' dell'iterazione
				cnt[0]++;
				final int from = idx / 81, to = idx % 81;
				final byte piece = idx >= 0 ? st.board[to] : 0;
				final boolean moverBlack = st.turn == BoardConstants.WHITE_TURN; // al turno il Bianco: ha appena mosso il Nero
				final boolean ok = idx >= 0 && st.board[from] == BoardConstants.EMPTY && rookShaped(idx)
						&& (moverBlack ? piece == BoardConstants.BLACK : (piece == BoardConstants.WHITE || piece == BoardConstants.KING));
				if (!ok) {
					bad[0]++;
				}
				return 0.0;
			};
			Negamax n = new Negamax(ev, 4, false, 0, true, BASE | Negamax.ORDER_COUNTER).setThreads(1);
			n.debugVerifyCounter = true; // a ogni taglio la risposta registrata deve essere una mossa possibile del lato al tratto
			holder[0] = n;
			n.findBestMove(positions.get(p), 600_000L);
			if (n.debugCounterViolations != 0) {
				fail("posizione " + p + ": " + n.debugCounterViolations + " risposte countermove registrate che il lato al tratto non puo' fare");
			}
			leaves += cnt[0];
			if (bad[0] != 0) {
				fail("posizione " + p + ": " + bad[0] + " foglie su " + cnt[0] + " la cui mossa registrata non e' quella che le ha prodotte");
			}
			for (int side = 0; side < 2; side++) {
				for (int prev = 0; prev < n.counterMove[side].length; prev++) {
					final int r = n.counterMove[side][prev];
					if (r == -1) {
						continue;
					}
					entries++;
					if (r < 0 || r >= 6561 || !rookShaped(r) || !rookShaped(prev)) {
						fail("countermove non valida: lato " + side + " prev " + prev + " -> " + r);
					}
				}
			}
		}
		if (leaves < 1000 || entries == 0) {
			fail("il test countermove e' vacuo: foglie " + leaves + ", voci " + entries);
		}
		System.out.printf("%s %-30s %d foglie con la mossa precedente giusta; %d voci countermove ben formate%n", failures == before ? "OK  " : "FAIL",
				"[mossa precedente/counter]", leaves, entries);
	}

	/**
	 * Con ORDER_ROOT_SCORES, al termine di una ricerca completa le mosse della radice sono una PERMUTAZIONE di quelle legali, hanno
	 * punteggi non crescenti e la migliore e' in testa (quella restituita); i punteggi sono quelli veri (il primo coincide con il
	 * valore di radice) e non tutti uguali.
	 */
	static void checkRootOrder(List<State> positions) {
		int before = failures, searched = 0, varied = 0;
		for (State s : positions) {
			if (AshtonRules.isTerminal(s)) {
				continue;
			}
			Negamax n = new Negamax(new SimpleEvaluator(), 3, false, 0, true, BASE | Negamax.ORDER_ROOT_SCORES).setThreads(1);
			Action best = n.findBestMove(s, 600_000L);
			searched++;
			final int cnt = n.rootMoveCountForTest;
			final int[] rm = n.rootMovesForTest();
			List<Action> legal = AshtonRules.generateMoves(s);
			if (cnt != legal.size()) {
				fail("ordine radice: " + cnt + " mosse alla radice contro " + legal.size() + " legali");
				continue;
			}
			int[] a = Arrays.copyOf(rm, cnt), b = new int[cnt];
			for (int i = 0; i < cnt; i++) {
				b[i] = Move.fromAction(legal.get(i));
			}
			Arrays.sort(a);
			Arrays.sort(b);
			if (!Arrays.equals(a, b)) {
				fail("ordine radice: le mosse alla radice non sono una permutazione di quelle legali");
			}
			boolean desc = true, distinct = false;
			for (int i = 1; i < cnt; i++) {
				desc &= n.rootScores[i - 1] >= n.rootScores[i];
				distinct |= n.rootScores[i] != n.rootScores[0];
			}
			if (!desc) {
				fail("ordine radice: punteggi non decrescenti dopo la ricerca");
			}
			if (distinct) {
				varied++;
			}
			if (best == null || rm[0] != Move.fromAction(best)) {
				fail("ordine radice: la mossa restituita non e' in testa");
			}
			if (Math.abs(Math.abs(n.rootScores[0]) - Math.abs(n.lastScore)) > 1e-9) {
				fail("ordine radice: il primo punteggio " + n.rootScores[0] + " non e' il valore di radice " + n.lastScore);
			}
		}
		if (searched < 4 || varied < searched / 2) {
			fail("ordine radice: test vacuo (ricerche " + searched + ", con punteggi diversi " + varied + ")");
		}
		System.out.printf("%s %-30s %d ricerche complete: permutazione, punteggi non crescenti, migliore in testa, primo punteggio = valore di radice%n",
				failures == before ? "OK  " : "FAIL", "[ordine radice dopo la ricerca]", searched);
	}

	/** Il punteggio esteso e' usato se e solo se e' accesa una delle opzioni che lo richiedono (killer-carry e ordine radice non lo richiedono). */
	static void checkDispatch(List<State> positions) {
		int before = failures;
		final int[] modes = { BASE, BASE | Negamax.ORDER_KILLER_CARRY, BASE | Negamax.ORDER_ROOT_SCORES, BASE | Negamax.ORDER_WINS, BASE | Negamax.ORDER_CAPTURES,
				BASE | Negamax.ORDER_CAPTURES_FIRST, BASE | Negamax.ORDER_COUNTER, Negamax.ORDER_WINS };
		for (int mode : modes) {
			Negamax n = new Negamax(new SimpleEvaluator(), 3, false, 0, true, mode).setThreads(1);
			for (int p = 0; p < 3; p++) {
				n.findBestMove(positions.get(p), 600_000L);
			}
			final boolean ext = (mode & Negamax.ORDER_EXT_MASK) != 0;
			if (ext && n.extScoreCalls == 0) {
				fail("modalita' " + mode + ": il punteggio esteso non e' mai stato usato");
			}
			if (!ext && n.extScoreCalls != 0) {
				fail("modalita' " + mode + ": il punteggio esteso e' stato usato senza opzioni che lo richiedano (" + n.extScoreCalls + " chiamate)");
			}
		}
		System.out.printf("%s %-30s punteggio esteso usato solo con wins/caps/capsfirst/counter (8 modalita')%n", failures == before ? "OK  " : "FAIL",
				"[dispatch]");
	}

	/** {@code newSearch} della TT a ogni mossa; il margine di sicurezza configurabile cambia davvero il tempo usato. */
	static void checkNewSearchAndMargin(List<State> positions) {
		int before = failures;
		Negamax g = new Negamax(new SimpleEvaluator(), 2, true, 1 << 10, true, BASE, 1, true);
		final int g0 = g.transpositionTable().generationForTest();
		g.findBestMove(positions.get(0), 600_000L);
		g.findBestMove(positions.get(1), 600_000L);
		if (g.transpositionTable().generationForTest() != ((g0 + 2) & 63)) {
			fail("la generazione della TT non avanza di 1 per ogni findBestMove");
		}
		if (g.getSafetyMarginMillis() != Negamax.defaultSafetyMarginMillis()) {
			fail("il margine di default non e' quello storico");
		}
		long e0 = -1, e600 = -1;
		for (int p = 0; p < positions.size() && e0 < 0; p++) {
			State s = positions.get(p);
			if (AshtonRules.isTerminal(s)) {
				continue;
			}
			Negamax a = new Negamax(new SimpleEvaluator(), 40, true, 1 << 16, true, BASE, 1, false).setSafetyMarginMillis(600);
			long t = System.currentTimeMillis();
			a.findBestMove(s, 900L);
			long d600 = System.currentTimeMillis() - t;
			if (d600 < 200) {
				continue; // vittoria forzata trovata subito: questa posizione non misura il tempo
			}
			Negamax b = new Negamax(new SimpleEvaluator(), 40, true, 1 << 16, true, BASE, 1, false).setSafetyMarginMillis(0);
			t = System.currentTimeMillis();
			b.findBestMove(s, 900L);
			e0 = System.currentTimeMillis() - t;
			e600 = d600;
		}
		if (e0 < 0) {
			fail("margine: nessuna posizione utile per misurare il tempo");
		} else if (e0 < 800 || e0 > 1100 || e600 < 250 || e600 > 500 || e0 - e600 < 450) {
			fail("margine: con budget 900 ms il tempo usato dovrebbe essere ~900 (margine 0) e ~300 (margine 600); misurati " + e0 + " e " + e600);
		}
		System.out.printf("%s %-30s generazione TT +1 per mossa; margine 0/600 ms con budget 900: %d / %d ms usati%n", failures == before ? "OK  " : "FAIL",
				"[newSearch e margine]", e0, e600);
	}

	/** Come SearchOptimizationsCheck.checkAllChildrenVisited, per le nuove configurazioni (con countermove sporca e mossa precedente casuale). */
	static void checkAllChildrenVisited(Cfg[] cfgs, List<State> positions, Random rnd) {
		int before = failures, checks = 0;
		for (State s : positions) {
			for (int side = 0; side < 2; side++) {
				final State root = s.clone();
				root.setTurn(side == 0 ? Turn.WHITE : Turn.BLACK);
				if (AshtonRules.isTerminal(root)) {
					continue;
				}
				Set<Long> expected = new HashSet<>();
				List<Action> rootMoves = AshtonRules.generateMoves(root);
				for (Action m : rootMoves) {
					State c = AshtonRules.applyMove(root, m);
					if (!AshtonRules.isTerminal(c)) {
						expected.add(ZobristHasher.fullHash(c));
					}
				}
				for (Cfg cfg : cfgs) {
					final Set<Long> seen = new HashSet<>();
					Evaluator recording = st -> {
						seen.add(st.zobristHash);
						return 0.0;
					};
					Negamax n = cfg.make(recording, 4, false);
					for (int[] k : n.killers) {
						k[0] = Move.index(Move.fromAction(rootMoves.get(rnd.nextInt(rootMoves.size()))));
						k[1] = Move.index(Move.fromAction(rootMoves.get(rnd.nextInt(rootMoves.size()))));
					}
					for (int sd = 0; sd < 2; sd++) {
						for (int i = 0; i < n.historyScore[sd].length; i++) {
							n.historyScore[sd][i] = rnd.nextInt(5);
						}
						if (n.counterMove != null) {
							for (int i = 0; i < n.counterMove[sd].length; i++) {
								n.counterMove[sd][i] = rnd.nextInt(3) == 0 ? Move.index(Move.fromAction(rootMoves.get(rnd.nextInt(rootMoves.size())))) : -1;
							}
						}
					}
					n.installForTest(FastState.fromState(root), new ArrayList<Long>(), 1);
					n.prevMoveAt[0] = rnd.nextInt(3) == 0 ? -1 : rnd.nextInt(6561);
					n.negamax(1, -SearchOptimizationsCheck.INF, SearchOptimizationsCheck.INF, side == 0 ? 1 : -1, 0);
					checks++;
					if (!seen.equals(expected)) {
						fail(cfg.name + ": a profondita' 1 sono stati valutati " + seen.size() + " figli distinti su " + expected.size() + " attesi");
						return;
					}
				}
			}
		}
		System.out.printf("%s %-30s %d verifiche: a profondita' 1, finestra piena, ogni figlio viene valutato (stato sporco)%n",
				failures == before ? "OK  " : "FAIL", "[tutti i figli visitati]", checks);
	}

	// ------------------------------------------------------------------------------------------ 2. istanza riusata e Lazy SMP

	/**
	 * Una stessa istanza per mosse consecutive (killer carry, history, countermove, TT e ordine radice persistono secondo le
	 * regole di ogni opzione): il valore di ogni mossa deve coincidere con quello di una ricerca fresca senza TT, ammettendo solo
	 * le divergenze che l'alpha-beta+TT riusata mostra a sua volta (si veda SearchOptimizationsCheck.checkReusedInstance).
	 */
	static void checkReusedInstance(Cfg cfg, Evaluator ev, List<State> positions) {
		int before = failures, compared = 0, mismatchCand = 0, mismatchBase = 0;
		for (int p = 0; p < Math.min(6, positions.size()); p++) {
			State s = positions.get(p);
			Negamax cand = cfg.make(ev, 3, true);
			Negamax baseTT = new Negamax(ev, 3, true, 1 << 16, false, Negamax.ORDER_NONE).setThreads(1);
			for (int ply = 0; ply < 8 && !AshtonRules.isTerminal(s); ply++) {
				Negamax fresh = SearchOptimizationsCheck.BASE.make(ev, 3, false);
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
				// la mossa deve valere il valore di radice (ottima), anche se il tie-break e' cambiato
				if (m != null) {
					double v = SearchOptimizationsCheck.exactValueOfMove(ev, s, m, cand.lastDepthReached, new ArrayList<Long>());
					if (Math.abs(v - cand.lastScore) > 1e-9 && Math.abs(cand.lastScore - fresh.lastScore) <= 1e-9) {
						fail(cfg.name + " [istanza riusata]: la mossa scelta vale " + v + " ma il valore di radice e' " + cand.lastScore);
					}
				}
				s = AshtonRules.applyMove(s, mf != null ? mf : m);
			}
		}
		if (mismatchCand > mismatchBase) {
			fail(cfg.name + " [istanza riusata]: " + mismatchCand + " divergenze contro " + mismatchBase + " dell'alpha-beta+TT riusata");
		}
		System.out.printf("%s %-26s %d mosse consecutive: divergenze dal fresco senza TT: alpha-beta+TT=%d, candidato=%d%n",
				failures == before ? "OK  " : "FAIL", "[" + cfg.name + "]", compared, mismatchBase, mismatchCand);
	}

	/** Con 4 thread (Lazy SMP) e tutte le opzioni: mossa sempre legale, nessun errore negli helper. */
	static void checkParallel(Cfg[] cfgs, Evaluator ev, List<State> positions) {
		int before = failures, runs = 0;
		for (Cfg cfg : new Cfg[] { cfgs[cfgs.length - 2], cfgs[cfgs.length - 1] }) {
			Negamax n = new Negamax(ev, 40, true, 1 << 16, cfg.pvs, cfg.order).setThreads(4);
			try {
				for (int p = 0; p < Math.min(10, positions.size()); p++) {
					State s = positions.get(p);
					for (int step = 0; step < 3 && !AshtonRules.isTerminal(s); step++) { // stessa istanza: tabelle degli helper persistenti
						Action m = n.findBestMove(s, 400L + 100L * step);
						runs++;
						if (m == null) {
							fail(cfg.name + " [4 thread]: nessuna mossa");
							break;
						}
						boolean legal = false;
						for (Action a : AshtonRules.generateMoves(s)) {
							legal |= SearchOptimizationsCheck.sameMove(a, m);
						}
						if (!legal) {
							fail(cfg.name + " [4 thread]: mossa illegale");
						}
						if (n.lastHelperErrors != 0) {
							fail(cfg.name + " [4 thread]: " + n.lastHelperErrors + " helper in errore");
						}
						s = AshtonRules.applyMove(s, m);
					}
				}
			} finally {
				n.shutdown();
			}
		}
		System.out.printf("%s %-30s %d ricerche con 4 thread: mossa legale, nessun errore negli helper%n", failures == before ? "OK  " : "FAIL",
				"[Lazy SMP + opzioni]", runs);
	}
}
