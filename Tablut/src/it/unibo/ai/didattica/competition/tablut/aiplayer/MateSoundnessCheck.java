package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Verifica che i punteggi di vittoria/sconfitta della ricerca siano SEMPRE dimostrati dalla ricerca corrente e che la
 * transposition table non possa ne' crearli, ne' nasconderli, ne' alterarne la distanza (si veda la sezione "Punteggi
 * di vittoria e transposition table" del javadoc di {@link Negamax}).
 * <p>
 * Nasce da una partita vera (negamax bianco contro un altro player, pareggio per ripetizione): il bianco aveva una
 * vittoria in 1 ({@code K c7->c9}) e per due volte ha giocato altro perche' la TT dava a una mossa lenta un valore di
 * vittoria piu' alto ({@code 1000001} contro {@code 1000000}); la posizione che ne e' uscita ricreava un'altra gia'
 * vista e il server ha dichiarato pareggio. Le posizioni 16..36 di quella partita sono incorporate qui sotto, con la
 * storia reale (dall'ultima cattura).
 * <ol>
 * <li><b>A. Le due vittorie in 1 della partita vera</b>, con TT avvelenata (ogni figlio della radice con una voce di
 * vittoria falsa, tutti i flag e margini, TT a hash e a chiave esatta), con e senza la scorciatoia alla radice.</li>
 * <li><b>B. Vittorie immediate sintetiche</b>: piu' fughe del Re per il bianco; cattura del Re per il NERO (la
 * scorciatoia e' simmetrica).</li>
 * <li><b>C. Ricerca contro oracolo</b>: su posizioni reali, sintetiche e casuali, una ricerca senza TT (oracolo: alpha-beta
 * puro, esatta per costruzione), una con TT pulita e una con TT avvelenata devono concordare su "c'e' una matta?", sul suo
 * punteggio (cioe' la distanza) e sulla profondita'; la mossa scelta deve davvero vincere in quella distanza.</li>
 * <li><b>D. Nessun giro in tondo</b>: partendo da posizioni vinte, il bianco gioca la mossa della ricerca ogni turno
 * (TT avvelenata a ogni turno) contro un avversario che ripete una posizione appena puo' (pareggio) e altrimenti
 * resiste: deve vincere, senza ripetizioni, e la distanza deve scendere di almeno 2 semi-mosse per turno.</li>
 * <li><b>E. Valutazione fuori scala</b>: valutatori che restituiscono 1e12, -1e12, infinito, NaN non producono mai un
 * punteggio di matta.</li>
 * </ol>
 */
public class MateSoundnessCheck {

	private static int failures = 0;
	private static final double WIN = Negamax.WIN_SCORE;

	/** Stati 16..36 della partita vera (81 caratteri, riga 1 in alto; O vuoto, W bianco, B nero, K re, T trono vuoto). Tocca al bianco negli stati pari. */
	private static final int FIRST_STATE = 16;
	private static final String[] GAME = {
		/* 16 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWBBOOBOOOOOBOOWKOWOOOOOWOOOOOOOOOOOBOOO",
		/* 17 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWBBOOBOOOOOBWOOKOWOOOOOWOOOOOOOOOOOBOOO",
		/* 18 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWBBOOOOOOOOBWOBKOWOOOOOWOOOOOOOOOOOBOOO",
		/* 19 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWBBOOOOOWOOBWOBKOOOOOOOWOOOOOOOOOOOBOOO",
		/* 20 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOOOOWOOBWOBKOOOBOOOWOOOOOOOOOOOBOOO",
		/* 21 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOOKOOOBOOOWOOOOOOOOOOOBOOO",
		/* 22 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOOKOOOBOOOWOOOOOOOBOOOOOOO",
		/* 23 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOOKOOOBOOWOOOOOOOOBOOOOOOO",
		/* 24 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOOKOOOBOOWOOOOOOOOBOOOOOOO",
		/* 25 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOKOOOOBOOWOOOOOOOOBOOOOOOO",
		/* 26 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOKOOOOBOOWOOOOOOOOOBOOOOOO",
		/* 27 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOKOOOOBOOOOOOOOOOOWOOOOOOO",
		/* 28 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOKOOOOBOOOOOOOOOOOWOOOOOOO",
		/* 29 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOOKOOOBOOOOOOOOOOOWOOOOOOO",
		/* 30 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOOKOOOBOOOOOOOOOOOWOOOOOOO",
		/* 31 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWWOKOOOBOOOOOOOOOOOOOOOOOOO",
		/* 32 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOOBOOTOWOBOOWOOOOOBWWOKOOOBOOOOOOOOOOOOOOOOOOO",
		/* 33 */ "OOOBOBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOOBOOTOWOBOOWOOOOOBWWKOOOOBOOOOOOOOOOOOOOOOOOO",
		/* 34 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOOBOOTOWOBOOWOOOOOBWWKOOOOBOOOOOOOOOOOOOOOOOOO",
		/* 35 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOOBOOTOWOBOOWOOOOOBWOKOOOOBOOOOOOOOOOOWOOOOOOO",
		/* 36 */ "OOOOBBOOOOOOOBOOOOOOOBOOOOOOOBOBOOOOBOOOTOWOBOOWOOOOOBWOKOOOOBOOOOOOOOOOOWOOOOOOO"
	};

	private static void check(String label, boolean ok) {
		if (ok) {
			System.out.println("OK   - " + label);
		} else {
			System.out.println("FAIL - " + label);
			failures++;
		}
	}

	public static void main(String[] args) {
		long t0 = System.currentTimeMillis();
		testRealGameBugPositions();
		testSyntheticImmediateWins();
		testAgainstOracle();
		testNoCycling();
		testEvalClamp();
		System.out.println("(durata " + (System.currentTimeMillis() - t0) / 1000 + " s)");
		if (failures == 0) {
			System.out.println("=== MateSoundnessCheck: TUTTI I TEST OK ===");
		} else {
			System.out.println("=== MateSoundnessCheck: " + failures + " FALLIMENTI ===");
			System.exit(1);
		}
	}

	// ------------------------------------------------------------------------------------------ posizioni

	static FastState fromRows(String flat, char turn) {
		State s = new StateTablut();
		Pawn[][] b = new Pawn[9][9];
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				switch (flat.charAt(r * 9 + c)) {
				case 'W': b[r][c] = Pawn.WHITE; break;
				case 'B': b[r][c] = Pawn.BLACK; break;
				case 'K': b[r][c] = Pawn.KING; break;
				case 'T': b[r][c] = Pawn.THRONE; break;
				default: b[r][c] = Pawn.EMPTY;
				}
			}
		}
		s.setBoard(b);
		s.setTurn(turn == 'W' ? Turn.WHITE : Turn.BLACK);
		return FastState.fromState(s);
	}

	static FastState real(int stateIdx) {
		return fromRows(GAME[stateIdx - FIRST_STATE], stateIdx % 2 == 0 ? 'W' : 'B');
	}

	/** Storia reale: dall'ultima cattura (stati 16, 21 e 27 della partita) fino a {@code stateIdx} compreso. */
	static PositionHistory realHistory(int stateIdx) {
		int from = stateIdx >= 27 ? 27 : (stateIdx >= 21 ? 21 : 16);
		PositionHistory h = new PositionHistory();
		for (int i = from; i <= stateIdx; i++) {
			h.add(real(i));
		}
		return h;
	}

	/** Una mossa che vince subito per chi muove (stessa condizione dei nodi terminali della ricerca), o {@link Move#NONE}. */
	static int immediateWin(FastState p) {
		int[] mv = new int[256];
		int n = p.generateMoves(mv);
		FastState w = new FastState();
		w.copyFrom(p);
		UndoInfo u = new UndoInfo();
		byte me = p.turn == BoardConstants.WHITE_TURN ? FastState.WINNER_WHITE : FastState.WINNER_BLACK;
		for (int i = 0; i < n; i++) {
			w.makeMove(mv[i], u);
			boolean win = w.winner == me;
			w.unmakeMove(mv[i], u);
			if (win) {
				return mv[i];
			}
		}
		return Move.NONE;
	}

	// ------------------------------------------------------------------------------------------ motori e avvelenamento

	/** useTT=false: oracolo (alpha-beta puro). exact: TT a chiave esatta (come {@code exactkey=true} del client) o a hash. */
	static Negamax engine(int maxDepth, boolean useTT, boolean exact, boolean shortcut) {
		Negamax n;
		if (useTT && exact) {
			n = new Negamax(new SimpleEvaluator(), maxDepth, true, 1 << 16, true, OrderingSpec.parse("rec"), 1, false,
					new ExactTranspositionTable(1 << 16, false));
		} else {
			n = new Negamax(new SimpleEvaluator(), maxDepth, useTT, 1 << 16);
		}
		n.setThreads(1);
		n.immediateWinShortcut = shortcut;
		return n;
	}

	static boolean isMate(double score) {
		return Math.abs(score) >= WIN;
	}

	/**
	 * Avvelena la TT di {@code n}: per ogni posizione raggiungibile dalla radice in 1..{@code plies} semi-mosse scrive una
	 * voce di vittoria/sconfitta FALSA (segno, margine 0..30, flag EXACT/LOWER/UPPER, profondita' 0..40 e mossa casuali).
	 * Se {@code claimRootSideWins} i figli diretti ricevono sempre "vince chi ha mosso alla radice, con margine {@code k}"
	 * (il caso della partita vera: una mossa lenta che sembra vincere piu' di una vittoria in 1).
	 */
	static void poison(Negamax n, FastState root, Random rnd, int plies, boolean claimRootSideWins, double fixedMargin) {
		FastState w = new FastState();
		w.copyFrom(root);
		poisonRec(n.transpositionTable(), w, rnd, plies, claimRootSideWins, fixedMargin, true);
	}

	private static void poisonRec(TranspositionTable tt, FastState w, Random rnd, int plies, boolean claim, double fixedMargin, boolean direct) {
		if (plies == 0) {
			return;
		}
		int[] mv = new int[256];
		int nm = w.generateMoves(mv);
		UndoInfo u = new UndoInfo();
		for (int i = 0; i < nm; i++) {
			w.makeMove(mv[i], u);
			if (w.winner == FastState.WINNER_NONE) {
				double k = fixedMargin >= 0 ? fixedMargin : rnd.nextInt(31);
				// punteggio dal punto di vista di chi muove nel figlio: negativo = chi ha mosso alla radice vince
				double sign = (claim && direct) ? -1 : (rnd.nextBoolean() ? 1 : -1);
				int[] cm = new int[256];
				int cn = w.generateMoves(cm);
				int packed = (cn > 0 && rnd.nextBoolean()) ? cm[rnd.nextInt(cn)] : -1;
				byte flag = (claim && direct && fixedMargin >= 0) ? (byte) (rnd.nextInt(3)) : (byte) rnd.nextInt(3);
				tt.store(w, direct ? 40 : rnd.nextInt(41), sign * (WIN + k), flag, packed);
				poisonRec(tt, w, rnd, plies - 1, claim, fixedMargin, false);
			}
			w.unmakeMove(mv[i], u);
		}
	}

	// ------------------------------------------------------------------------------------------ A

	private static void testRealGameBugPositions() {
		System.out.println("--- A: le due vittorie in 1 saltate nella partita vera ---");
		int[] roots = { 28, 34 }; // turni 29 e 35 della partita: K c7 e colonna c libera
		for (int r : roots) {
			FastState p = real(r);
			int win = immediateWin(p);
			check("stato " + r + ": esiste la vittoria immediata K c7->c9 (56->74)", win != Move.NONE && Move.getFrom(win) == 56 && Move.getTo(win) == 74);
			boolean allOk = true;
			String firstBad = "";
			for (boolean shortcut : new boolean[] { true, false }) {
				for (boolean exact : new boolean[] { false, true }) {
					for (double k : new double[] { 0, 1, 3, 7, 15, 40 }) {
						Negamax n = engine(40, true, exact, shortcut);
						poison(n, p, new Random(1000 + (long) k), 2, true, k);
						int m = n.findBestMove(p, 600000, realHistory(r));
						boolean ok = m == win && n.lastScore == WIN && n.lastDepthReached == 1;
						if (!ok && allOk) {
							allOk = false;
							firstBad = "scorciatoia=" + shortcut + " exact=" + exact + " margine=" + k + " -> " + Move.getFrom(m) + "->" + Move.getTo(m)
									+ " valore=" + n.lastScore + " prof=" + n.lastDepthReached;
						}
					}
				}
			}
			check("stato " + r + ": con TT avvelenata (24 varianti) gioca sempre K c7->c9, valore esatto 1000000, profondita' 1" + (allOk ? "" : " [" + firstBad + "]"), allOk);
		}
	}

	// ------------------------------------------------------------------------------------------ B

	/** Re in c3, nero in c2, nero in g4 che arriva in c4: cattura del Re (verticale). Tocca al nero. */
	private static final String BLACK_CAPTURES_KING = "OOOOOOOOO" + "OOBOOOOOO" + "OOKOOOOOO" + "OOOOOOBOO" + "OOOOTOOOO"
			+ "OOOOOOOOO" + "OOOOOOOOO" + "OOOOOOOWO" + "OOOOOOOOO";
	/** Re in c3 con piu' vie verso il bordo. Tocca al bianco. */
	private static final String WHITE_MANY_ESCAPES = "OOOOOOOOO" + "OOOOOOOOO" + "OOKOOOOOO" + "OOOOOOBOO" + "OOOOTOOOO"
			+ "OOOOOOOOO" + "OOOOOOOOO" + "OOOOOOOWO" + "OOOOOOOOO";

	private static void testSyntheticImmediateWins() {
		System.out.println("--- B: vittorie immediate sintetiche (bianco e NERO) ---");
		String[] names = { "bianco: fughe multiple del Re", "nero: cattura del Re (33->29)" };
		FastState[] roots = { fromRows(WHITE_MANY_ESCAPES, 'W'), fromRows(BLACK_CAPTURES_KING, 'B') };
		for (int i = 0; i < 2; i++) {
			FastState p = roots[i];
			int win = immediateWin(p);
			check(names[i] + ": la posizione ha una vittoria immediata", win != Move.NONE);
			boolean ok = true;
			for (boolean shortcut : new boolean[] { true, false }) {
				for (double k : new double[] { 0, 2, 9 }) {
					Negamax n = engine(40, true, true, shortcut);
					poison(n, p, new Random(7 + (long) k), 2, true, k);
					int m = n.findBestMove(p, 600000, PositionHistory.of(p));
					// con fughe multiple basta una qualunque che vinca davvero
					FastState w = new FastState();
					w.copyFrom(p);
					UndoInfo u = new UndoInfo();
					w.makeMove(m, u);
					byte me = p.turn == BoardConstants.WHITE_TURN ? FastState.WINNER_WHITE : FastState.WINNER_BLACK;
					ok &= (w.winner == me) && n.lastScore == WIN && n.lastDepthReached == 1;
				}
			}
			check(names[i] + ": con TT avvelenata, con e senza scorciatoia, gioca una mossa che vince subito (valore 1000000, prof. 1)", ok);
		}
	}

	// ------------------------------------------------------------------------------------------ C

	private static void testAgainstOracle() {
		System.out.println("--- C: ricerca con TT (pulita e avvelenata) contro oracolo senza TT ---");
		int comparisons = 0, mates = 0;
		boolean allOk = true;
		String firstBad = "";
		// (1) posizioni reali 16..36, ciascuna con la propria storia; piu' profondo dove le vittorie sono vicine
		for (int r = FIRST_STATE; r <= 36; r++) {
			int depth = (r == 24 || r == 26 || r == 28 || r == 30 || r == 32 || r == 34) ? 5 : 4;
			for (boolean shortcut : new boolean[] { false, true }) {
				if (shortcut && depth == 4) {
					continue; // con scorciatoia cambia solo dove c'e' una vittoria immediata: basta la profondita' 5
				}
				String res = compare("reale " + r + (shortcut ? " (scorciatoia)" : ""), real(r), realHistory(r), depth, shortcut, 31L * r + (shortcut ? 1 : 0));
				comparisons++;
				if (res == null) {
					continue;
				}
				if (res.startsWith("MATE")) {
					mates++;
				} else if (allOk) {
					allOk = false;
					firstBad = res;
				}
			}
		}
		// (2) la stessa posizione 35 vista dal NERO: con la storia reale la risposta b5->a5 e' pareggio
		// (3) posizioni casuali: nessuna matta inventata
		Random rnd = new Random(20240607L);
		for (int i = 0; i < 24; i++) {
			PositionHistory h = new PositionHistory();
			FastState p = fromRows("OOOBBBOOO" + "OOOOBOOOO" + "OOOOWOOOO" + "BOOOWOOOB" + "BBWWKWWBB" + "BOOOWOOOB" + "OOOOWOOOO" + "OOOOBOOOO" + "OOOBBBOOO", 'W');
			h.add(p);
			int plies = 8 + rnd.nextInt(40);
			UndoInfo u = new UndoInfo();
			int[] mv = new int[256];
			for (int k = 0; k < plies && p.winner == FastState.WINNER_NONE; k++) {
				int n = p.generateMoves(mv);
				if (n == 0) {
					break;
				}
				p.makeMove(mv[rnd.nextInt(n)], u);
				if (u.wasCapture()) {
					h.clear();
				}
				h.add(p);
			}
			if (p.winner != FastState.WINNER_NONE || p.generateMoves(mv) == 0) {
				continue;
			}
			String res = compare("casuale " + i, p, h, 3, false, 99L + i);
			comparisons++;
			if (res != null && res.startsWith("MATE")) {
				mates++;
			} else if (res != null && allOk) {
				allOk = false;
				firstBad = res;
			}
		}
		check("oracolo/TT pulita/TT avvelenata concordano su matta si'/no, distanza, profondita' e sulla mossa scelta (" + comparisons
				+ " confronti, " + mates + " con matta)" + (allOk ? "" : " [" + firstBad + "]"), allOk);
		check("il confronto e' significativo: ci sono posizioni con matta", mates >= 6);
	}

	/** Ritorna null se nessuna matta e tutto coerente, "MATE ..." se c'era una matta ed e' tutto coerente, altrimenti la descrizione dell'errore. */
	private static String compare(String label, FastState p, PositionHistory h, int maxDepth, boolean shortcut, long seed) {
		Negamax oracle = engine(maxDepth, false, false, shortcut);
		int mO = oracle.findBestMove(p, 600000, h);
		double sO = oracle.lastScore;
		int dO = oracle.lastDepthReached;
		String err = null;
		for (int variant = 0; variant < 3 && err == null; variant++) {
			Negamax n = engine(maxDepth, true, (seed & 1) == 0, shortcut);
			if (variant == 1) {
				poison(n, p, new Random(seed), 2, false, -1);
			} else if (variant == 2) {
				poison(n, p, new Random(seed + 1), 2, true, -1);
			}
			int m = n.findBestMove(p, 600000, h);
			double s = n.lastScore;
			String tag = label + " variante " + (variant == 0 ? "TT pulita" : variant == 1 ? "TT avvelenata" : "TT avvelenata (vittorie false)");
			if (isMate(sO) != isMate(s)) {
				err = tag + ": oracolo " + sO + " ma motore " + s;
			} else if (isMate(sO) && (s != sO || n.lastDepthReached != dO)) {
				err = tag + ": distanza diversa, oracolo " + sO + "@" + dO + " motore " + s + "@" + n.lastDepthReached;
			} else if (sO >= WIN && !moveReallyWins(p, h, m, dO)) {
				err = tag + ": la mossa " + Move.getFrom(m) + "->" + Move.getTo(m) + " non vince davvero in " + dO + " semi-mosse";
			}
		}
		if (err != null) {
			return err;
		}
		return isMate(sO) ? "MATE " + label : null;
	}

	/** La mossa {@code m} (distanza di vittoria {@code dist} dalla radice) vince davvero: lo si verifica con l'oracolo sul figlio. */
	private static boolean moveReallyWins(FastState p, PositionHistory h, int m, int dist) {
		FastState c = new FastState();
		c.copyFrom(p);
		UndoInfo u = new UndoInfo();
		c.makeMove(m, u);
		byte me = p.turn == BoardConstants.WHITE_TURN ? FastState.WINNER_WHITE : FastState.WINNER_BLACK;
		if (c.winner == me) {
			return dist == 1;
		}
		if (c.winner != FastState.WINNER_NONE || dist < 2) {
			return false;
		}
		PositionHistory ch = new PositionHistory(h);
		if (u.wasCapture()) {
			ch.clear();
		} else if (ch.contains(c)) {
			return false; // la mossa ripete una posizione: e' pareggio, non vittoria
		}
		ch.add(c);
		Negamax o = engine(dist - 1, false, false, false);
		o.findBestMove(c, 600000, ch);
		// chi muove nel figlio deve perdere in dist-1 semi-mosse: valore -(WIN_SCORE + 0)
		return o.lastScore == -WIN;
	}

	// ------------------------------------------------------------------------------------------ D

	private static void testNoCycling() {
		System.out.println("--- D: dalle posizioni vinte il bianco vince, senza ripetizioni, distanza in calo ---");
		int[][] starts = { { 24, 1 }, { 30, 1 }, { 24, 0 }, { 30, 0 }, { 26, 0 } }; // {stato, scorciatoia}
		for (int[] st : starts) {
			String res = simulate(st[0], st[1] == 1, 555L + st[0]);
			check("da " + st[0] + (st[1] == 1 ? " (con scorciatoia)" : " (solo ricerca)") + ": " + (res == null ? "vittoria bianca, nessuna ripetizione, distanza in calo" : res), res == null);
		}
	}

	private static String simulate(int startState, boolean shortcut, long seed) {
		Random rnd = new Random(seed);
		FastState cur = real(startState);
		PositionHistory h = realHistory(startState);
		Negamax white = engine(14, true, true, shortcut);
		Negamax blackOracle = engine(3, false, false, false);
		int prevDist = Integer.MAX_VALUE;
		UndoInfo u = new UndoInfo();
		for (int ply = 0; ply < 40; ply++) {
			int m;
			if (cur.turn == BoardConstants.WHITE_TURN) {
				poison(white, cur, rnd, 2, true, -1);
				m = white.findBestMove(cur, 600000, h);
				if (white.lastScore < WIN) {
					return "al ply " + ply + " il bianco non vede piu' la vittoria (valore " + white.lastScore + ")";
				}
				int dist = white.lastDepthReached;
				if (dist > prevDist - 2) {
					return "al ply " + ply + " la distanza non scende di 2: " + prevDist + " -> " + dist;
				}
				prevDist = dist;
			} else {
				m = repeatingMove(cur, h);
				if (m == Move.NONE) {
					m = blackOracle.findBestMove(cur, 600000, h);
				}
			}
			cur.makeMove(m, u);
			if (cur.winner != FastState.WINNER_NONE) {
				return cur.winner == FastState.WINNER_WHITE ? null : "la partita finisce con esito " + cur.winner + " (3 = pareggio)";
			}
			if (u.wasCapture()) {
				h.clear();
			} else if (h.contains(cur)) {
				return "ripetizione al ply " + ply + ": pareggio";
			}
			h.add(cur);
		}
		return "nessun esito in 40 semi-mosse";
	}

	/** Una mossa del lato al tratto che ricrea una posizione della storia (cioe' provoca il pareggio), o NONE. */
	private static int repeatingMove(FastState p, PositionHistory h) {
		int[] mv = new int[256];
		int n = p.generateMoves(mv);
		FastState w = new FastState();
		w.copyFrom(p);
		UndoInfo u = new UndoInfo();
		for (int i = 0; i < n; i++) {
			w.makeMove(mv[i], u);
			boolean rep = !u.wasCapture() && w.winner == FastState.WINNER_NONE && h.contains(w);
			w.unmakeMove(mv[i], u);
			if (rep) {
				return mv[i];
			}
		}
		return Move.NONE;
	}

	// ------------------------------------------------------------------------------------------ E

	private static void testEvalClamp() {
		System.out.println("--- E: valutazione statica fuori scala ---");
		double[] values = { 1e12, -1e12, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, WIN, -WIN, WIN + 5 };
		FastState p = real(16);
		for (double v : values) {
			Evaluator ev = (FastState s) -> v;
			Negamax n = new Negamax(ev, 3, true, 1 << 12).setThreads(1);
			n.immediateWinShortcut = false;
			int m = n.findBestMove(p, 600000, realHistory(16));
			check("valutatore costante " + v + ": restituisce una mossa e nessuna matta (valore " + n.lastScore + ")",
					m != Move.NONE && !Double.isNaN(n.lastScore) && Math.abs(n.lastScore) < WIN);
		}
		check("clampEval: NaN -> 0, +/-infinito e 1e12 -> +/-(WIN-1), valori normali invariati",
				Negamax.clampEval(Double.NaN) == 0 && Negamax.clampEval(Double.POSITIVE_INFINITY) == WIN - 1
						&& Negamax.clampEval(-1e12) == -(WIN - 1) && Negamax.clampEval(123.5) == 123.5 && Negamax.clampEval(-WIN) == -(WIN - 1));
	}
}
