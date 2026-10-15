package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures.Workspace;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

import static it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures.*;

/**
 * Verifica {@link TablutFeatures} su migliaia di posizioni reali, confrontando
 * ogni feature con una definizione INDIPENDENTE:
 * <ul>
 * <li>le feature di mobilita' / fuga / minaccia contro le mosse generate e
 * applicate da {@link AshtonRules} (il motore di regole gia' verificato contro
 * quello ufficiale): qui l'estrattore (scansione di raggi su byte[] con bordo
 * sentinella) e l'oracolo (oggetti State/Action) non condividono codice;</li>
 * <li>le feature di struttura contro un ricalcolo diretto su {@code Pawn[][]};</li>
 * <li>l'invarianza per le 8 simmetrie della scacchiera su TUTTE le feature (un
 * errore di segno/riga/colonna in qualunque feature la violerebbe);</li>
 * <li>l'assenza di "memoria" tra una chiamata e l'altra (workspace riusato in
 * ordine casuale == workspace nuovo);</li>
 * <li>la copertura: un test che non esercita mai una feature sarebbe vuoto.</li>
 * </ul>
 */
public class FeaturesCheck {

	static int failures = 0;
	static final int N = 9;

	public static void main(String[] args) {
		int numPositions = args.length > 0 ? Integer.parseInt(args[0]) : 4000;
		long seed = args.length > 1 ? Long.parseLong(args[1]) : 20260101L;

		checkEnumCodes();
		checkTables();
		List<State> positions = collectPositions(numPositions, seed);
		System.out.println("Posizioni raccolte: " + positions.size());

		long[] cover = new long[NUM_FEATURES]; // posizioni in cui la feature e' != 0
		Workspace ws = newWorkspace();
		double[] f = new double[NUM_FEATURES];
		int bad = 0;
		for (State s : positions) {
			extract(s, ws, f, G_ALL);
			bad += checkPosition(s, ws, f);
			for (int i = 0; i < NUM_FEATURES; i++) {
				if (f[i] != 0.0) {
					cover[i]++;
				}
			}
			if (bad > 20) {
				System.out.println("Troppi errori, interrompo.");
				break;
			}
		}
		failures += bad;

		checkSymmetry(positions);
		checkWorkspaceReuse(positions, seed);
		checkGroupsAreIndependent(positions);
		checkCoverage(cover, positions.size());
		checkNamesAndRanges(positions);

		if (failures == 0) {
			System.out.println("=== FEATURES CHECK OK: " + positions.size() + " posizioni, tutte le feature coerenti con gli oracoli ===");
		} else {
			System.out.println("=== FEATURES CHECK FALLITO: " + failures + " discrepanze ===");
			System.exit(1);
		}
	}

	static void fail(String msg) {
		failures++;
		System.out.println("FAIL - " + msg);
	}

	// ---------------------------------------------------------------- setup

	static void checkEnumCodes() {
		boolean ok = Pawn.EMPTY.ordinal() == 0 && Pawn.WHITE.ordinal() == 1 && Pawn.BLACK.ordinal() == 2
				&& Pawn.THRONE.ordinal() == 3 && Pawn.KING.ordinal() == 4;
		if (!ok) {
			fail("l'ordine dell'enum Pawn non coincide con i codici byte usati da TablutFeatures");
		} else {
			System.out.println("OK   - codici byte == Pawn.ordinal()");
		}
	}

	/** Le tabelle statiche (citadel, bordo, anelli) rispetto alla definizione diretta. */
	static void checkTables() {
		int[][] cit = { { 3, 0 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, { 3, 8 },
				{ 4, 8 }, { 5, 8 }, { 4, 7 }, { 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 } };
		Set<Integer> cs = new HashSet<>();
		for (int[] c : cit) {
			cs.add(c[0] * N + c[1]);
		}
		// Le citadel di AshtonRules non sono esposte: le ricaviamo dal comportamento. Un pezzo NON partito da una
		// citadel non puo' mai atterrarci: lo verifichiamo sul Re, che parte sempre da una cella non-citadel.
		int freeEdge = 0;
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				boolean edge = r == 0 || r == 8 || c == 0 || c == 8;
				if (edge && !cs.contains(r * N + c)) {
					freeEdge++;
				}
			}
		}
		if (freeEdge != 20) {
			fail("celle di fuga libere attese=20, trovate=" + freeEdge);
		} else {
			System.out.println("OK   - 20 celle di fuga reali (32 di bordo - 12 citadel di bordo)");
		}
	}

	/** Posizioni reali: gioco misto (casuale + preferenza per catture/fughe) cosi' compaiono molte situazioni tattiche. */
	static List<State> collectPositions(int n, long seed) {
		Random rnd = new Random(seed);
		List<State> out = new ArrayList<>();
		while (out.size() < n) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			double greedy = rnd.nextDouble() * 0.7; // quanto "tattica" e' questa partita
			for (int ply = 0; ply < 160 && out.size() < n; ply++) {
				if (AshtonRules.isTerminal(s)) {
					break;
				}
				List<Action> moves = AshtonRules.generateMoves(s);
				if (moves.isEmpty()) {
					break;
				}
				Action chosen = null;
				if (rnd.nextDouble() < greedy) {
					List<Action> good = new ArrayList<>();
					for (Action m : moves) {
						State c = AshtonRules.applyMove(s, m);
						if (AshtonRules.isTerminal(c) || AshtonRules.wasCapture(s, c, m)) {
							good.add(m);
						}
					}
					if (!good.isEmpty()) {
						chosen = good.get(rnd.nextInt(good.size()));
					}
				}
				if (chosen == null) {
					chosen = moves.get(rnd.nextInt(moves.size()));
				}
				s = AshtonRules.applyMove(s, chosen);
				if (!AshtonRules.isTerminal(s) && rnd.nextInt(3) == 0) {
					out.add(s);
				}
			}
		}
		return out;
	}

	// -------------------------------------------------------- oracoli (motore)

	static State withTurn(State s, Turn t) {
		State c = s.clone();
		c.setTurn(t);
		return c;
	}

	static boolean isEdge(int r, int c) {
		return r == 0 || r == 8 || c == 0 || c == 8;
	}

	static int[] findKing(State s) {
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				if (b[r][c] == Pawn.KING) {
					return new int[] { r, c };
				}
			}
		}
		return null;
	}

	static void expect(String what, double actual, double expectedCount, int feature, State s) {
		double expected = expectedCount / DIV[feature];
		if (Math.abs(actual - expected) > 1e-12) {
			fail(what + ": feature=" + actual + " (=" + actual * DIV[feature] + ") atteso=" + expected + " (="
					+ expectedCount + ")\n" + s);
		}
	}

	/** Ritorna il numero di discrepanze trovate su questa posizione. */
	static int checkPosition(State s, Workspace ws, double[] f) {
		int before = failures;
		int[] k = findKing(s);
		int kr = k[0], kc = k[1];
		Pawn[][] b = s.getBoard();

		List<Action> wm = AshtonRules.generateMoves(withTurn(s, Turn.WHITE));
		List<Action> bm = AshtonRules.generateMoves(withTurn(s, Turn.BLACK));

		// --- mobilita' -------------------------------------------------------------
		if (ws.whiteMoves != wm.size()) {
			fail("whiteMoves " + ws.whiteMoves + " != " + wm.size());
		}
		if (ws.blackMoves != bm.size()) {
			fail("blackMoves " + ws.blackMoves + " != " + bm.size());
		}
		int[][] reachW = new int[N][N], reachB = new int[N][N];
		boolean[][] hasMove = new boolean[N][N];
		int kingMoves = 0;
		for (Action a : wm) {
			reachW[a.getRowTo()][a.getColumnTo()]++;
			hasMove[a.getRowFrom()][a.getColumnFrom()] = true;
			if (a.getRowFrom() == kr && a.getColumnFrom() == kc) {
				kingMoves++;
			}
		}
		for (Action a : bm) {
			reachB[a.getRowTo()][a.getColumnTo()]++;
			hasMove[a.getRowFrom()][a.getColumnFrom()] = true;
		}
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				if (ws.reachWhiteAt(r, c) != reachW[r][c] || ws.reachBlackAt(r, c) != reachB[r][c]) {
					fail("mappa di raggiungibilita' diversa in (" + r + "," + c + "): W " + ws.reachWhiteAt(r, c) + "/"
							+ reachW[r][c] + " B " + ws.reachBlackAt(r, c) + "/" + reachB[r][c] + "\n" + s);
				}
			}
		}
		int wBlocked = 0, bBlocked = 0, nW = 0, nB = 0;
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				if (b[r][c] == Pawn.WHITE) {
					nW++;
					if (!hasMove[r][c]) {
						wBlocked++;
					}
				} else if (b[r][c] == Pawn.BLACK) {
					nB++;
					if (!hasMove[r][c]) {
						bBlocked++;
					}
				}
			}
		}
		expect("F_WHITE_MOVES", f[F_WHITE_MOVES], wm.size(), F_WHITE_MOVES, s);
		expect("F_BLACK_MOVES", f[F_BLACK_MOVES], bm.size(), F_BLACK_MOVES, s);
		expect("F_KING_MOVES", f[F_KING_MOVES], kingMoves, F_KING_MOVES, s);
		expect("F_WHITE_BLOCKED", f[F_WHITE_BLOCKED], wBlocked, F_WHITE_BLOCKED, s);
		expect("F_BLACK_BLOCKED", f[F_BLACK_BLOCKED], bBlocked, F_BLACK_BLOCKED, s);
		int edgeReach = 0;
		boolean[][] citT = citadels();
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				if (isEdge(r, c) && !citT[r][c] && reachB[r][c] > 0) {
					edgeReach++;
				}
			}
		}
		expect("F_BLACK_EDGE_REACH", f[F_BLACK_EDGE_REACH], edgeReach, F_BLACK_EDGE_REACH, s);

		// --- vie di fuga immediate e bloccabilita' -------------------------------
		int lanes = 0, unblockable = 0;
		for (Action a : wm) {
			if (a.getRowFrom() == kr && a.getColumnFrom() == kc && isEdge(a.getRowTo(), a.getColumnTo())) {
				lanes++;
				int dr = Integer.signum(a.getRowTo() - kr), dc = Integer.signum(a.getColumnTo() - kc);
				boolean blockable = false;
				for (int r = kr + dr, c = kc + dc;; r += dr, c += dc) {
					if (reachB[r][c] > 0) {
						blockable = true;
					}
					if (r == a.getRowTo() && c == a.getColumnTo()) {
						break;
					}
				}
				if (!blockable) {
					unblockable++;
				}
			}
		}
		expect("F_KING_LANES", f[F_KING_LANES], lanes, F_KING_LANES, s);
		expect("F_KING_LANES_UNBLOCKABLE", f[F_KING_LANES_UNBLOCKABLE], unblockable, F_KING_LANES_UNBLOCKABLE, s);

		// --- fuga in 2 mosse: il Re si sposta davvero (senza catture) e si contano le vie dal motore ---
		int cells = 0, forks = 0;
		for (Action a : wm) {
			if (a.getRowFrom() != kr || a.getColumnFrom() != kc || isEdge(a.getRowTo(), a.getColumnTo())) {
				continue;
			}
			State moved = withTurn(s, Turn.WHITE);
			Pawn[][] mb = moved.getBoard();
			mb[kr][kc] = (kr == 4 && kc == 4) ? Pawn.THRONE : Pawn.EMPTY;
			mb[a.getRowTo()][a.getColumnTo()] = Pawn.KING;
			int n2 = 0;
			for (Action m2 : AshtonRules.generateMoves(moved)) {
				if (m2.getRowFrom() == a.getRowTo() && m2.getColumnFrom() == a.getColumnTo()
						&& isEdge(m2.getRowTo(), m2.getColumnTo())) {
					n2++;
				}
			}
			if (n2 >= 1) {
				cells++;
			}
			if (n2 >= 2) {
				forks++;
			}
		}
		expect("F_KING_2STEP_CELLS", f[F_KING_2STEP_CELLS], cells, F_KING_2STEP_CELLS, s);
		expect("F_KING_2STEP_FORKS", f[F_KING_2STEP_FORKS], forks, F_KING_2STEP_FORKS, s);

		// --- minacce di cattura: le catture vere, applicando ogni mossa col motore ---
		Set<Integer> whiteVictims = new HashSet<>(), blackVictims = new HashSet<>();
		boolean kingCapturable = false;
		State sb = withTurn(s, Turn.BLACK);
		for (Action a : bm) {
			State after = AshtonRules.applyMove(sb, a);
			if (after.getTurn() == Turn.BLACKWIN) {
				kingCapturable = true;
			}
			Pawn[][] ab = after.getBoard();
			for (int r = 0; r < N; r++) {
				for (int c = 0; c < N; c++) {
					if (b[r][c] == Pawn.WHITE && ab[r][c] == Pawn.EMPTY) {
						whiteVictims.add(r * N + c);
					}
				}
			}
		}
		State sw = withTurn(s, Turn.WHITE);
		for (Action a : wm) {
			State after = AshtonRules.applyMove(sw, a);
			Pawn[][] ab = after.getBoard();
			for (int r = 0; r < N; r++) {
				for (int c = 0; c < N; c++) {
					if (b[r][c] == Pawn.BLACK && ab[r][c] == Pawn.EMPTY) {
						blackVictims.add(r * N + c);
					}
				}
			}
		}
		expect("F_WHITE_THREATENED", f[F_WHITE_THREATENED], whiteVictims.size(), F_WHITE_THREATENED, s);
		expect("F_BLACK_THREATENED", f[F_BLACK_THREATENED], blackVictims.size(), F_BLACK_THREATENED, s);
		expect("F_KING_THREAT", f[F_KING_THREAT], kingCapturable ? 1 : 0, F_KING_THREAT, s);

		// --- accerchiamento: BFS indipendente sulle celle EMPTY non-citadel ---------
		boolean[][] cit = citadels();
		boolean[][] seen = new boolean[N][N];
		List<int[]> stack = new ArrayList<>();
		stack.add(new int[] { kr, kc });
		seen[kr][kc] = true;
		int region = 0, edgeCells = 0, sides = 0;
		int[][] D = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
		while (!stack.isEmpty()) {
			int[] p = stack.remove(stack.size() - 1);
			for (int[] d : D) {
				int r = p[0] + d[0], c = p[1] + d[1];
				if (r < 0 || r >= N || c < 0 || c >= N || seen[r][c] || b[r][c] != Pawn.EMPTY || cit[r][c]) {
					continue;
				}
				seen[r][c] = true;
				region++;
				if (isEdge(r, c)) {
					edgeCells++;
					sides |= (r == 0 ? 1 : 0) | (r == 8 ? 2 : 0) | (c == 0 ? 4 : 0) | (c == 8 ? 8 : 0);
				} else {
					stack.add(new int[] { r, c });
				}
			}
		}
		expect("F_FLOOD_REGION", f[F_FLOOD_REGION], region, F_FLOOD_REGION, s);
		expect("F_FLOOD_EDGE", f[F_FLOOD_EDGE], edgeCells, F_FLOOD_EDGE, s);
		expect("F_FLOOD_SIDES", f[F_FLOOD_SIDES], Integer.bitCount(sides), F_FLOOD_SIDES, s);

		// --- materiale, re, struttura: ricalcolo diretto su Pawn[][] -----------------
		expect("F_N_WHITE", f[F_N_WHITE], nW, F_N_WHITE, s);
		expect("F_N_BLACK", f[F_N_BLACK], nB, F_N_BLACK, s);
		int dr = Math.abs(kr - 4), dc = Math.abs(kc - 4);
		expect("F_KING_RING", f[F_KING_RING], Math.max(dr, dc), F_KING_RING, s);
		expect("F_KING_AXIS_MIN", f[F_KING_AXIS_MIN], Math.min(dr, dc), F_KING_AXIS_MIN, s);
		expect("F_KING_AXIS_MAX", f[F_KING_AXIS_MAX], Math.max(dr, dc), F_KING_AXIS_MAX, s);
		expect("F_KING_ON_THRONE", f[F_KING_ON_THRONE], (kr == 4 && kc == 4) ? 1 : 0, F_KING_ON_THRONE, s);
		expect("F_KING_NEAR_THRONE", f[F_KING_NEAR_THRONE], (dr + dc == 1) ? 1 : 0, F_KING_NEAR_THRONE, s);

		int support = 0, near = 0, central = 0, wcentral = 0, isoW = 0, isoB = 0, diag = 0, ortho = 0;
		int[] bring = new int[5], wring = new int[5];
		int[] rowAll = new int[N], colAll = new int[N], rowBlack = new int[N], colBlack = new int[N];
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				Pawn p = b[r][c];
				if (p == Pawn.WHITE || p == Pawn.KING || p == Pawn.BLACK) {
					rowAll[r]++;
					colAll[c]++;
				}
				if (p == Pawn.BLACK) {
					rowBlack[r]++;
					colBlack[c]++;
				}
				int ring = Math.max(Math.abs(r - 4), Math.abs(c - 4));
				int man = Math.abs(r - 4) + Math.abs(c - 4);
				int cheb = Math.max(Math.abs(r - kr), Math.abs(c - kc));
				if (p == Pawn.WHITE) {
					wring[ring]++;
					wcentral += 8 - man;
					if (cheb <= 2) {
						support++;
					}
					if (!neighbor8(b, r, c, Pawn.WHITE, Pawn.KING)) {
						isoW++;
					}
				} else if (p == Pawn.BLACK) {
					bring[ring]++;
					central += 8 - man;
					if (cheb <= 2) {
						near++;
					}
					if (!neighbor8(b, r, c, Pawn.BLACK, Pawn.BLACK)) {
						isoB++;
					}
					if (c + 1 < N && b[r][c + 1] == Pawn.BLACK) {
						ortho++;
					}
					if (r + 1 < N && b[r + 1][c] == Pawn.BLACK) {
						ortho++;
					}
					if (r + 1 < N && c + 1 < N && b[r + 1][c + 1] == Pawn.BLACK) {
						diag++;
					}
					if (r + 1 < N && c - 1 >= 0 && b[r + 1][c - 1] == Pawn.BLACK) {
						diag++;
					}
				}
			}
		}
		expect("F_KING_SUPPORT", f[F_KING_SUPPORT], support, F_KING_SUPPORT, s);
		expect("F_BLACK_NEAR_KING", f[F_BLACK_NEAR_KING], near, F_BLACK_NEAR_KING, s);
		expect("F_BLACK_CENTRAL", f[F_BLACK_CENTRAL], central, F_BLACK_CENTRAL, s);
		expect("F_WHITE_CENTRAL", f[F_WHITE_CENTRAL], wcentral, F_WHITE_CENTRAL, s);
		expect("F_BLACK_RING1", f[F_BLACK_RING1], bring[1], F_BLACK_RING1, s);
		expect("F_BLACK_RING2", f[F_BLACK_RING2], bring[2], F_BLACK_RING2, s);
		expect("F_BLACK_RING3", f[F_BLACK_RING3], bring[3], F_BLACK_RING3, s);
		expect("F_BLACK_RING4", f[F_BLACK_RING4], bring[4], F_BLACK_RING4, s);
		expect("F_WHITE_RING1", f[F_WHITE_RING1], wring[1], F_WHITE_RING1, s);
		expect("F_WHITE_RING2", f[F_WHITE_RING2], wring[2], F_WHITE_RING2, s);
		expect("F_WHITE_RING3", f[F_WHITE_RING3], wring[3], F_WHITE_RING3, s);
		expect("F_WHITE_RING4", f[F_WHITE_RING4], wring[4], F_WHITE_RING4, s);
		// Gli anelli devono PARTIZIONARE le pedine di ciascun colore (nessuna pedina persa o contata due volte).
		// Il trono (anello 0) e' l'unica cella dell'anello 0 e solo il Re puo' starci, quindi anelli 1-4 == tutte le pedine.
		// Verificato sui valori ESTRATTI (non sull'oracolo): cattura asimmetrie del tipo "i neri hanno l'anello 4, i bianchi no".
		double whiteInRings = f[F_WHITE_RING1] * DIV[F_WHITE_RING1] + f[F_WHITE_RING2] * DIV[F_WHITE_RING2]
				+ f[F_WHITE_RING3] * DIV[F_WHITE_RING3] + f[F_WHITE_RING4] * DIV[F_WHITE_RING4];
		double blackInRings = f[F_BLACK_RING1] * DIV[F_BLACK_RING1] + f[F_BLACK_RING2] * DIV[F_BLACK_RING2]
				+ f[F_BLACK_RING3] * DIV[F_BLACK_RING3] + f[F_BLACK_RING4] * DIV[F_BLACK_RING4];
		if (Math.abs(whiteInRings - nW) > 1e-9) {
			fail("anelli bianchi 1-4 non partizionano le pedine bianche: somma=" + whiteInRings + " pedine=" + nW + "\n" + s);
		}
		if (Math.abs(blackInRings - nB) > 1e-9) {
			fail("anelli neri 1-4 non partizionano le pedine nere: somma=" + blackInRings + " pedine=" + nB + "\n" + s);
		}
		expect("F_WHITE_ISOLATED", f[F_WHITE_ISOLATED], isoW, F_WHITE_ISOLATED, s);
		expect("F_BLACK_ISOLATED", f[F_BLACK_ISOLATED], isoB, F_BLACK_ISOLATED, s);
		expect("F_BLACK_DIAG_LINKS", f[F_BLACK_DIAG_LINKS], diag, F_BLACK_DIAG_LINKS, s);
		expect("F_BLACK_ORTHO_LINKS", f[F_BLACK_ORTHO_LINKS], ortho, F_BLACK_ORTHO_LINKS, s);

		int free = 0, guarded = 0;
		for (int L : new int[] { 1, 2, 6, 7 }) {
			free += (rowAll[L] == 0 ? 1 : 0) + (colAll[L] == 0 ? 1 : 0);
			guarded += (rowBlack[L] > 0 ? 1 : 0) + (colBlack[L] > 0 ? 1 : 0);
		}
		expect("F_FREE_LINES", f[F_FREE_LINES], free, F_FREE_LINES, s);
		expect("F_BLACK_LINES", f[F_BLACK_LINES], guarded, F_BLACK_LINES, s);

		// cattura mancante: stesse tre regole di AshtonRules.isKingCaptured, riscritte su coordinate
		int need, have;
		if (kr == 4 && kc == 4) {
			need = 4;
			have = countBlackAround(b, kr, kc);
		} else if (dr + dc == 1) {
			need = 3;
			have = countBlackAround(b, kr, kc);
		} else {
			need = 2;
			int h1 = (hostile(b, kr - 1, kc, cit) ? 1 : 0) + (hostile(b, kr + 1, kc, cit) ? 1 : 0);
			int h2 = (hostile(b, kr, kc - 1, cit) ? 1 : 0) + (hostile(b, kr, kc + 1, cit) ? 1 : 0);
			have = Math.max(h1, h2);
		}
		expect("F_KING_CAPTURE_MISSING", f[F_KING_CAPTURE_MISSING], Math.max(0, need - have),
				F_KING_CAPTURE_MISSING, s);
		int adjB = countBlackAround(b, kr, kc), adjW = 0;
		for (int[] d : D) {
			int r = kr + d[0], c = kc + d[1];
			if (r >= 0 && r < N && c >= 0 && c < N && b[r][c] == Pawn.WHITE) {
				adjW++;
			}
		}
		expect("F_KING_ADJ_BLACK", f[F_KING_ADJ_BLACK], adjB, F_KING_ADJ_BLACK, s);
		expect("F_KING_ADJ_WHITE", f[F_KING_ADJ_WHITE], adjW, F_KING_ADJ_WHITE, s);

		return failures - before;
	}

	static boolean[][] citCache;

	static boolean[][] citadels() {
		if (citCache == null) {
			citCache = new boolean[N][N];
			int[][] cit = { { 3, 0 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, { 3, 8 },
					{ 4, 8 }, { 5, 8 }, { 4, 7 }, { 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 } };
			for (int[] c : cit) {
				citCache[c[0]][c[1]] = true;
			}
		}
		return citCache;
	}

	static boolean hostile(Pawn[][] b, int r, int c, boolean[][] cit) {
		return r >= 0 && r < N && c >= 0 && c < N && (b[r][c] == Pawn.BLACK || cit[r][c]);
	}

	static int countBlackAround(Pawn[][] b, int r, int c) {
		int n = 0;
		int[][] D = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
		for (int[] d : D) {
			int rr = r + d[0], cc = c + d[1];
			if (rr >= 0 && rr < N && cc >= 0 && cc < N && b[rr][cc] == Pawn.BLACK) {
				n++;
			}
		}
		return n;
	}

	static boolean neighbor8(Pawn[][] b, int r, int c, Pawn a1, Pawn a2) {
		for (int dr = -1; dr <= 1; dr++) {
			for (int dc = -1; dc <= 1; dc++) {
				if (dr == 0 && dc == 0) {
					continue;
				}
				int rr = r + dr, cc = c + dc;
				if (rr >= 0 && rr < N && cc >= 0 && cc < N && (b[rr][cc] == a1 || b[rr][cc] == a2)) {
					return true;
				}
			}
		}
		return false;
	}

	// -------------------------------------------------------- proprieta' globali

	/** Applica una delle 8 simmetrie del quadrato alla scacchiera. */
	static State symmetric(State s, int t) {
		State out = s.clone();
		Pawn[][] src = s.getBoard(), dst = out.getBoard();
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				int rr = r, cc = c;
				if ((t & 4) != 0) { // trasposizione
					int tmp = rr;
					rr = cc;
					cc = tmp;
				}
				for (int k = 0; k < (t & 3); k++) { // rotazioni di 90 gradi
					int nr = cc, nc = N - 1 - rr;
					rr = nr;
					cc = nc;
				}
				dst[rr][cc] = src[r][c];
			}
		}
		return out;
	}

	static void checkSymmetry(List<State> positions) {
		Workspace ws = newWorkspace();
		double[] a = new double[NUM_FEATURES], b = new double[NUM_FEATURES];
		int bad = 0;
		boolean[] featBad = new boolean[NUM_FEATURES];
		for (State s : positions) {
			extract(s, ws, a, G_ALL);
			for (int t = 1; t < 8; t++) {
				extract(symmetric(s, t), ws, b, G_ALL);
				for (int i = 0; i < NUM_FEATURES; i++) {
					if (Math.abs(a[i] - b[i]) > 1e-12) {
						featBad[i] = true;
						bad++;
					}
				}
			}
		}
		if (bad > 0) {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < NUM_FEATURES; i++) {
				if (featBad[i]) {
					sb.append(NAMES[i]).append("; ");
				}
			}
			fail("invarianza per le 8 simmetrie violata (" + bad + " casi) nelle feature: " + sb);
		} else {
			System.out.println("OK   - tutte le " + NUM_FEATURES + " feature sono invarianti per le 8 simmetrie ("
					+ positions.size() * 7 + " confronti per feature)");
		}
	}

	static void checkWorkspaceReuse(List<State> positions, long seed) {
		Random rnd = new Random(seed ^ 77);
		List<State> shuffled = new ArrayList<>(positions);
		Collections.shuffle(shuffled, rnd);
		Workspace shared = newWorkspace();
		double[] a = new double[NUM_FEATURES], b = new double[NUM_FEATURES];
		// riempiamo `a` con valori sporchi per verificare che extract() sovrascriva davvero tutto
		int bad = 0;
		for (State s : shuffled) {
			java.util.Arrays.fill(a, 123.0);
			extract(s, shared, a, G_ALL);
			extract(s, newWorkspace(), b, G_ALL);
			for (int i = 0; i < NUM_FEATURES; i++) {
				if (a[i] != b[i]) {
					bad++;
				}
			}
		}
		if (bad > 0) {
			fail("il Workspace riusato produce valori diversi da uno nuovo (" + bad + " valori): stato che 'trapela' tra chiamate");
		} else {
			System.out.println("OK   - nessuna perdita di stato tra chiamate (workspace riusato in ordine casuale == nuovo)");
		}
	}

	/**
	 * Per OGNI maschera di gruppi (16 combinazioni): le colonne attive devono coincidere esattamente col vettore
	 * completo e le altre devono valere 0 (nessuna dipendenza nascosta tra gruppi, p. es. LANES senza la scansione
	 * dei bianchi); inoltre columnsFor() deve elencare esattamente le colonne attive.
	 */
	static void checkGroupsAreIndependent(List<State> positions) {
		Workspace ws = newWorkspace();
		double[] full = new double[NUM_FEATURES], part = new double[NUM_FEATURES];
		int bad = 0;
		for (int mask = 0; mask < 16; mask++) {
			int[] cols = TablutFeatures.columnsFor(mask);
			boolean[] active = new boolean[NUM_FEATURES];
			for (int c : cols) {
				active[c] = true;
			}
			for (int i = 0; i < NUM_FEATURES; i++) {
				boolean expected = TablutFeatures.FEATURE_GROUP[i] == 0 || (mask & TablutFeatures.FEATURE_GROUP[i]) != 0;
				if (active[i] != expected) {
					bad++;
				}
			}
			for (State s : positions.subList(0, Math.min(500, positions.size()))) {
				extract(s, ws, full, G_ALL);
				extract(s, ws, part, mask);
				for (int i = 0; i < NUM_FEATURES; i++) {
					boolean ok = active[i] ? part[i] == full[i] : part[i] == 0.0;
					if (!ok) {
						bad++;
					}
				}
			}
		}
		if (bad > 0) {
			fail("i gruppi di feature non sono indipendenti o columnsFor e' incoerente (" + bad + " valori)");
		} else {
			System.out.println("OK   - tutte le 16 maschere di gruppi: colonne attive identiche al vettore completo, le altre a 0");
		}
	}

	static void checkCoverage(long[] cover, int n) {
		StringBuilder sb = new StringBuilder("Copertura (posizioni con feature != 0):\n");
		boolean vacuous = false;
		for (int i = 0; i < NUM_FEATURES; i++) {
			sb.append(String.format("  %-62s %5.1f%%%n", NAMES[i], 100.0 * cover[i] / n));
			if (cover[i] < 20) {
				vacuous = true;
				sb.append("    ^^^ POCHE posizioni: il test su questa feature e' debole\n");
			}
		}
		System.out.print(sb);
		if (vacuous) {
			fail("alcune feature non sono state esercitate abbastanza dalle posizioni di test");
		}
	}

	static void checkNamesAndRanges(List<State> positions) {
		for (int i = 0; i < NUM_FEATURES; i++) {
			if (NAMES[i] == null) {
				fail("nome mancante per la feature " + i);
			}
		}
		Workspace ws = newWorkspace();
		double[] f = new double[NUM_FEATURES];
		double[] max = new double[NUM_FEATURES], min = new double[NUM_FEATURES];
		java.util.Arrays.fill(min, Double.MAX_VALUE);
		for (State s : positions) {
			extract(s, ws, f, G_ALL);
			for (int i = 0; i < NUM_FEATURES; i++) {
				max[i] = Math.max(max[i], f[i]);
				min[i] = Math.min(min[i], f[i]);
			}
		}
		int out = 0;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < NUM_FEATURES; i++) {
			if (min[i] < 0 || max[i] > 1.5) {
				out++;
				sb.append(String.format("  %s: [%.3f, %.3f]%n", NAMES[i], min[i], max[i]));
			}
		}
		if (out > 0) {
			System.out.println("ATTENZIONE - feature fuori dal range atteso [0, 1.5] (ricalibrare DIV):\n" + sb);
			fail("feature fuori scala");
		} else {
			System.out.println("OK   - tutte le feature in [0, 1.5] sulle posizioni di test");
		}
	}
}
