package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.GameAshtonTablut;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

public class CrossCheck {

	static int mismatches = 0;

	public static void main(String[] args) throws Exception {
		int games = Integer.parseInt(args.length > 0 ? args[0] : "40");
		int maxPlies = Integer.parseInt(args.length > 1 ? args[1] : "200");
		long seed = args.length > 2 ? Long.parseLong(args[2]) : 12345L;
		Random rnd = new Random(seed);

		// UNA sola istanza del motore ufficiale, riusata per tutte le partite:
		// e' esattamente il tipo di uso "una volta per validare/testare", non
		// "migliaia di volte dentro la ricerca", che le slide raccomandano.
		GameAshtonTablut official = new GameAshtonTablut(99, 0, "/home/claude/work/testlogs", "w", "b");

		int totalPlies = 0;
		for (int g = 0; g < games; g++) {
			totalPlies += playOneGame(official, rnd, maxPlies, g);
		}

		System.out.println("=== FINE TEST: " + games + " partite, " + totalPlies + " semi-mosse totali, "
				+ mismatches + " MISMATCH ===");
		if (mismatches > 0) {
			System.exit(1);
		}
	}

	private static int playOneGame(GameAshtonTablut official, Random rnd, int maxPlies, int gameIdx)
			throws Exception {
		State state = new StateTablut();
		state.setTurn(Turn.WHITE);

		int ply = 0;
		for (; ply < maxPlies; ply++) {
			Turn t = state.getTurn();
			if (t != Turn.WHITE && t != Turn.BLACK) {
				break; // partita finita (WHITEWIN/BLACKWIN/DRAW)
			}

			// 1) confronto l'insieme delle mosse legali generate dal mio motore
			// con quelle "vere" (validate una per una dal motore ufficiale via
			// checkMove, per esclusione/inclusione).
			List<Action> fastMoves = AshtonRules.generateMoves(state);
			List<Action> officialMoves = bruteForceLegalMoves(official, state);

			if (!sameMoveSet(fastMoves, officialMoves)) {
				mismatches++;
				System.out.println("--- MISMATCH mosse legali (game " + gameIdx + ", ply " + ply + ") ---");
				System.out.println("Stato:\n" + state);
				printDiff(fastMoves, officialMoves);
				return ply;
			}

			if (fastMoves.isEmpty()) {
				break; // nessuna mossa legale: la partita finisce qui (perde chi deve muovere)
			}

			// 2) scelgo una mossa a caso e confronto il risultato (board + turno)
			// tra AshtonRules.applyMove e GameAshtonTablut.checkMove
			Action chosen = fastMoves.get(rnd.nextInt(fastMoves.size()));

			State fastResult = AshtonRules.applyMove(state, chosen);

			State officialInput = state.clone();
			Action officialAction = new Action(chosen.getFrom(), chosen.getTo(), t);
			State officialResult = official.checkMove(officialInput, officialAction);

			// Nota: quando la mossa e' l'ultima della partita perche' cattura il Re,
			// il motore ufficiale imposta il turno a BLACKWIN ma NON rimuove lo
			// sprite del Re dalla board (guarda checkCaptureBlackKing* in
			// GameAshtonTablut: chiamano solo setTurn, mai removePawn). E' un
			// dettaglio cosmetico innocuo (la partita e' comunque finita, nessuno
			// generera' piu' mosse da quello stato), quindi per gli stati
			// terminali confrontiamo SOLO il vincitore, non il contenuto della board.
			boolean bothTerminal = fastResult.getTurn() != Turn.WHITE && fastResult.getTurn() != Turn.BLACK
					&& officialResult.getTurn() != Turn.WHITE && officialResult.getTurn() != Turn.BLACK;
			boolean mismatch = bothTerminal ? (fastResult.getTurn() != officialResult.getTurn())
					: (!boardsEqual(fastResult.getBoard(), officialResult.getBoard())
							|| fastResult.getTurn() != officialResult.getTurn());

			if (mismatch) {
				mismatches++;
				System.out.println("--- MISMATCH risultato mossa (game " + gameIdx + ", ply " + ply + ") ---");
				System.out.println("Stato di partenza:\n" + state);
				System.out.println("Mossa: " + chosen);
				System.out.println("Risultato FAST (AshtonRules), turno=" + fastResult.getTurn() + ":\n"
						+ fastResult.boardString());
				System.out.println("Risultato OFFICIAL (GameAshtonTablut), turno=" + officialResult.getTurn() + ":\n"
						+ officialResult.boardString());
				return ply;
			}

			state = fastResult;
		}
		return ply;
	}

	/** Genera le mosse legali "vere" provando ogni possibile (from,to) con checkMove. */
	private static List<Action> bruteForceLegalMoves(GameAshtonTablut official, State state) throws Exception {
		List<Action> legal = new ArrayList<>();
		Turn t = state.getTurn();
		Pawn[][] b = state.getBoard();
		for (int rf = 0; rf < 9; rf++) {
			for (int cf = 0; cf < 9; cf++) {
				Pawn p = b[rf][cf];
				boolean mine = (t == Turn.WHITE) ? (p == Pawn.WHITE || p == Pawn.KING) : (p == Pawn.BLACK);
				if (!mine) {
					continue;
				}
				for (int rt = 0; rt < 9; rt++) {
					for (int ct = 0; ct < 9; ct++) {
						if (rt == rf && ct == cf) {
							continue;
						}
						if (rt != rf && ct != cf) {
							continue; // scarto subito le diagonali, inutile generare l'eccezione
						}
						try {
							Action a = new Action(state.getBox(rf, cf), state.getBox(rt, ct), t);
							official.checkMove(state.clone(), a);
							legal.add(a);
						} catch (Exception e) {
							// mossa illegale: ignorata
						}
					}
				}
			}
		}
		return legal;
	}

	private static boolean sameMoveSet(List<Action> a, List<Action> b) {
		if (a.size() != b.size()) {
			return false;
		}
		List<String> sa = toKeys(a);
		List<String> sb = toKeys(b);
		Collections.sort(sa);
		Collections.sort(sb);
		return sa.equals(sb);
	}

	private static List<String> toKeys(List<Action> list) {
		List<String> keys = new ArrayList<>();
		for (Action a : list) {
			keys.add(a.getFrom() + "->" + a.getTo());
		}
		return keys;
	}

	private static void printDiff(List<Action> fast, List<Action> official) {
		List<String> sf = toKeys(fast);
		List<String> so = toKeys(official);
		System.out.println("Solo nel motore FAST: " + minus(sf, so));
		System.out.println("Solo nel motore OFFICIAL: " + minus(so, sf));
	}

	private static List<String> minus(List<String> a, List<String> b) {
		List<String> r = new ArrayList<>(a);
		r.removeAll(b);
		return r;
	}

	private static boolean boardsEqual(Pawn[][] a, Pawn[][] b) {
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				if (a[r][c] != b[r][c]) {
					return false;
				}
			}
		}
		return true;
	}
}
