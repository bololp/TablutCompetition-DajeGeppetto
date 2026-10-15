package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

public class WasCaptureCheck {

	public static void main(String[] args) {
		int games = Integer.parseInt(args.length > 0 ? args[0] : "400");
		int maxPlies = Integer.parseInt(args.length > 1 ? args[1] : "300");
		long seed = args.length > 2 ? Long.parseLong(args[2]) : 314159L;
		Random rnd = new Random(seed);

		long totalMoves = 0, captureMovesFound = 0;
		int mismatches = 0;

		for (int g = 0; g < games && mismatches == 0; g++) {
			State state = new StateTablut();
			state.setTurn(Turn.WHITE);

			for (int ply = 0; ply < maxPlies; ply++) {
				if (AshtonRules.isTerminal(state)) {
					break;
				}
				List<Action> moves = AshtonRules.generateMoves(state);
				if (moves.isEmpty()) {
					break;
				}
				Action chosen = moves.get(rnd.nextInt(moves.size()));
				State next = AshtonRules.applyMove(state, chosen);
				totalMoves++;

				boolean reference = countPieces(state) != countPieces(next);
				boolean underTest = AshtonRules.wasCapture(state, next, chosen);

				if (reference) {
					captureMovesFound++;
				}

				if (reference != underTest) {
					mismatches++;
					System.out.println("--- MISMATCH (game " + g + ", ply " + ply + ") ---");
					System.out.println("Mossa: " + chosen);
					System.out.println("Stato prima:\n" + state);
					System.out.println("Stato dopo:\n" + next);
					System.out.println("riferimento(conteggio pedine)=" + reference + " wasCapture()=" + underTest);
					break;
				}

				state = next;
			}
		}

		System.out.println("=== FINE: " + games + " partite, " + totalMoves + " mosse confrontate, "
				+ captureMovesFound + " con cattura, " + mismatches + " MISMATCH ===");
		if (mismatches > 0) {
			System.exit(1);
		}
	}

	private static int countPieces(State s) {
		int n = 0;
		for (Pawn[] row : s.getBoard()) {
			for (Pawn p : row) {
				if (p == Pawn.WHITE || p == Pawn.BLACK || p == Pawn.KING) {
					n++;
				}
			}
		}
		return n;
	}
}
