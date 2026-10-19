package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ZobristHasher;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;

/**
 * Generatori di posizioni per i test. Usano SOLO il motore storico ({@link AshtonRules}, {@link ZobristHasher}), mai
 * {@code FastState}: cosi' le posizioni (e le storie di ripetizione) sono un input indipendente dal codice che si sta verificando.
 */
public final class TestPositions {

	private TestPositions() {
	}

	/** Posizione iniziale, Bianco di turno. */
	public static State initial() {
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		return s;
	}

	/** Scacchiera vuota con il solo trono (e5) al suo posto, Bianco di turno: base per le posizioni costruite a mano. */
	public static State blank() {
		StateTablut s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE;
		s.setTurn(Turn.WHITE);
		return s;
	}

	/**
	 * Posizione casuale DENSA (fino a 25 pedine, addensate attorno a un centro casuale): stressa catture, Re circondato e
	 * mosse bloccate molto piu' delle partite. Mai terminale (turno Bianco o Nero); puo' non avere mosse legali.
	 */
	public static State randomPosition(Random rnd) {
		return randomPosition(rnd, false);
	}

	/**
	 * Come {@link #randomPosition(Random)}; con {@code kingMayBeOnEdge} il Re puo' trovarsi sul bordo (posizione non
	 * raggiungibile in partita, perche' sarebbe gia' una vittoria del Bianco, ma lecita per le regole: serve a verificare che
	 * le catture accanto a un Re di bordo, dove manca la cella "oltre", si comportino come in {@code AshtonRules}).
	 */
	public static State randomPosition(Random rnd, boolean kingMayBeOnEdge) {
		StateTablut s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		int nW = rnd.nextInt(9);
		int nB = 1 + rnd.nextInt(16);
		int cr = 2 + rnd.nextInt(5), cc = 2 + rnd.nextInt(5);
		boolean kingOnThrone = rnd.nextInt(4) == 0;
		int kr, kc;
		if (kingOnThrone) {
			kr = 4;
			kc = 4;
		} else {
			do {
				if (rnd.nextBoolean()) {
					kr = Math.max(0, Math.min(8, cr + rnd.nextInt(3) - 1));
					kc = Math.max(0, Math.min(8, cc + rnd.nextInt(3) - 1));
				} else {
					kr = rnd.nextInt(9);
					kc = rnd.nextInt(9);
				}
			} while (BoardConstants.IS_CITADEL[kr * 9 + kc] || (kr == 4 && kc == 4) || (!kingMayBeOnEdge && BoardConstants.IS_EDGE[kr * 9 + kc]));
		}
		b[kr][kc] = Pawn.KING;
		if (!kingOnThrone) {
			b[4][4] = Pawn.THRONE;
		}
		place(b, rnd, nW, Pawn.WHITE, kr, kc, false);
		place(b, rnd, nB, Pawn.BLACK, kr, kc, true);
		s.setTurn(rnd.nextBoolean() ? Turn.WHITE : Turn.BLACK);
		return s;
	}

	private static void place(Pawn[][] b, Random rnd, int n, Pawn p, int kr, int kc, boolean allowCitadel) {
		for (int i = 0; i < n; i++) {
			for (int tries = 0; tries < 50; tries++) {
				int r, c;
				if (rnd.nextInt(3) != 0) {
					r = Math.max(0, Math.min(8, kr + rnd.nextInt(5) - 2));
					c = Math.max(0, Math.min(8, kc + rnd.nextInt(5) - 2));
				} else {
					r = rnd.nextInt(9);
					c = rnd.nextInt(9);
				}
				if (b[r][c] != Pawn.EMPTY) {
					continue;
				}
				if (BoardConstants.IS_CITADEL[r * 9 + c] && !(allowCitadel || rnd.nextInt(10) == 0)) {
					continue;
				}
				b[r][c] = p;
				break;
			}
		}
	}

	/** Una partita giocata a caso (o "avida": preferisce catture e vittorie) con le regole storiche. */
	public static final class Game {
		/** Posizioni NON terminali visitate, nell'ordine. */
		public final List<State> positions = new ArrayList<>();
		/** Per ciascuna posizione, gli hash ({@code ZobristHasher}) delle posizioni dall'ultima cattura, ATTUALE INCLUSA (come fa il client). */
		public final List<List<Long>> histories = new ArrayList<>();
	}

	public static Game playGame(Random rnd, boolean greedy, int maxPlies) {
		Game g = new Game();
		State s = initial();
		List<Long> history = new ArrayList<>();
		history.add(ZobristHasher.fullHash(s));
		for (int ply = 0; ply < maxPlies && !AshtonRules.isTerminal(s); ply++) {
			List<Action> moves = AshtonRules.generateMoves(s);
			if (moves.isEmpty()) {
				break;
			}
			g.positions.add(s);
			g.histories.add(new ArrayList<>(history));
			Action chosen = moves.get(rnd.nextInt(moves.size()));
			if (greedy) {
				for (int tries = 0; tries < 6; tries++) {
					Action cand = moves.get(rnd.nextInt(moves.size()));
					State next = AshtonRules.applyMove(s, cand);
					if (AshtonRules.isTerminal(next) || AshtonRules.wasCapture(s, next, cand)) {
						chosen = cand;
						break;
					}
				}
			}
			State next = AshtonRules.applyMove(s, chosen);
			if (!AshtonRules.isTerminal(next)) {
				if (AshtonRules.wasCapture(s, next, chosen)) {
					history.clear();
				}
				history.add(ZobristHasher.fullHash(next));
			}
			s = next;
		}
		return g;
	}
}
