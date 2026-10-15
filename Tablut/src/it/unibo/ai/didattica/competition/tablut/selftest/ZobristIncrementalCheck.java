package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ZobristHasher;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Confronta {@link ZobristHasher#updateHash} (incrementale, esamina solo le
 * celle potenzialmente cambiate) con {@link ZobristHasher#fullHash}
 * (ricalcolo completo) dopo OGNI mossa di molte partite casuali. Se anche una
 * sola volta i due valori divergono, l'hashing incrementale ha un bug: dato
 * che una transposition table si fida ciecamente dell'hash per riconoscere
 * posizioni gia' viste, un bug qui produrrebbe mosse scelte in modo scorretto
 * senza alcun crash - il tipo di errore piu' pericoloso da lasciare
 * inosservato.
 */
public class ZobristIncrementalCheck {

	public static void main(String[] args) {
		int games = Integer.parseInt(args.length > 0 ? args[0] : "200");
		int maxPlies = Integer.parseInt(args.length > 1 ? args[1] : "300");
		long seed = args.length > 2 ? Long.parseLong(args[2]) : 999L;
		Random rnd = new Random(seed);

		long totalMoves = 0;
		long captureMoves = 0; // mosse in cui e' sparita almeno una pedina (cattura avvenuta)
		int whiteWins = 0, blackWins = 0, maxPliesHit = 0;
		int mismatches = 0;

		for (int g = 0; g < games && mismatches == 0; g++) {
			State state = new StateTablut();
			state.setTurn(Turn.WHITE);
			long hash = ZobristHasher.fullHash(state);

			if (hash != ZobristHasher.fullHash(state)) {
				System.out.println("MISMATCH sullo stato iniziale (non dovrebbe mai succedere)");
				mismatches++;
				break;
			}

			int ply = 0;
			for (; ply < maxPlies; ply++) {
				if (AshtonRules.isTerminal(state)) {
					break;
				}
				List<Action> moves = AshtonRules.generateMoves(state);
				if (moves.isEmpty()) {
					break;
				}
				Action chosen = moves.get(rnd.nextInt(moves.size()));
				State next = AshtonRules.applyMove(state, chosen);

				long incremental = ZobristHasher.updateHash(hash, state, next, chosen);
				long full = ZobristHasher.fullHash(next);
				totalMoves++;

				if (countPieces(state) != countPieces(next)) {
					captureMoves++;
				}

				if (incremental != full) {
					mismatches++;
					System.out.println("--- MISMATCH (game " + g + ", ply " + ply + ") ---");
					System.out.println("Mossa: " + chosen);
					System.out.println("Stato prima:\n" + state);
					System.out.println("Stato dopo:\n" + next);
					System.out.println("hash incrementale=" + incremental + " hash completo=" + full);
					break;
				}

				state = next;
				hash = incremental;
			}

			if (AshtonRules.isTerminal(state)) {
				if (state.getTurn() == Turn.WHITEWIN) {
					whiteWins++;
				} else if (state.getTurn() == Turn.BLACKWIN) {
					blackWins++;
				}
			} else if (ply >= maxPlies) {
				maxPliesHit++;
			}
		}

		System.out.println("=== FINE: " + games + " partite, " + totalMoves + " mosse confrontate, " + mismatches
				+ " MISMATCH ===");
		System.out.println("Copertura: " + captureMoves + " mosse con cattura, " + whiteWins + " vittorie bianco (fuga), "
				+ blackWins + " vittorie nero (cattura Re), " + maxPliesHit + " partite troncate per maxPlies");
		if (mismatches > 0) {
			System.exit(1);
		}
	}

	private static int countPieces(State s) {
		int n = 0;
		for (State.Pawn[] row : s.getBoard()) {
			for (State.Pawn p : row) {
				if (p == State.Pawn.WHITE || p == State.Pawn.BLACK || p == State.Pawn.KING) {
					n++;
				}
			}
		}
		return n;
	}
}
