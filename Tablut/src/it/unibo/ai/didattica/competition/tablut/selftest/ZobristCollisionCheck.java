package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ZobristHasher;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Raccoglie l'hash di moltissime posizioni DISTINTE generate da partite
 * casuali e verifica che non ci siano collisioni spurie (stesso hash per
 * board effettivamente diverse). Con un hash a 64 bit ben distribuito, tra
 * centinaia di migliaia di posizioni distinte non dovrebbe comparire NESSUNA
 * collisione vera (il paradosso del compleanno suggerisce che ne servirebbero
 * dell'ordine di 2^32 per aspettarsene una per caso). Se ne trovassimo anche
 * una sola, indicherebbe quasi certamente un bug nella funzione hash stessa
 * (es. una simmetria non voluta che riduce l'entropia effettiva), non
 * sfortuna statistica.
 */
public class ZobristCollisionCheck {

	public static void main(String[] args) {
		int games = Integer.parseInt(args.length > 0 ? args[0] : "500");
		int maxPlies = Integer.parseInt(args.length > 1 ? args[1] : "200");
		long seed = args.length > 2 ? Long.parseLong(args[2]) : 4242L;
		Random rnd = new Random(seed);

		Map<Long, String> seen = new HashMap<>(); // hash -> rappresentazione testuale della board (per il confronto)
		int collisions = 0;
		int distinctPositions = 0;

		for (int g = 0; g < games; g++) {
			State state = new StateTablut();
			state.setTurn(Turn.WHITE);

			for (int ply = 0; ply < maxPlies; ply++) {
				if (AshtonRules.isTerminal(state)) {
					break;
				}
				long hash = ZobristHasher.fullHash(state);
				String boardKey = boardKey(state);

				String existing = seen.get(hash);
				if (existing == null) {
					seen.put(hash, boardKey);
					distinctPositions++;
				} else if (!existing.equals(boardKey)) {
					collisions++;
					System.out.println("--- COLLISIONE VERA TROVATA ---");
					System.out.println("hash=" + hash);
					System.out.println("board A: " + existing);
					System.out.println("board B: " + boardKey);
				}

				List<Action> moves = AshtonRules.generateMoves(state);
				if (moves.isEmpty()) {
					break;
				}
				state = AshtonRules.applyMove(state, moves.get(rnd.nextInt(moves.size())));
			}
		}

		System.out.println("=== FINE: " + distinctPositions + " posizioni distinte raccolte, " + collisions
				+ " collisioni vere trovate ===");
		if (collisions > 0) {
			System.exit(1);
		}
	}

	/** Rappresentazione testuale univoca della board+turno, per riconoscere se
	 *  due stati con lo stesso hash sono davvero la stessa posizione o una
	 *  vera collisione. */
	private static String boardKey(State s) {
		StringBuilder sb = new StringBuilder();
		for (State.Pawn[] row : s.getBoard()) {
			for (State.Pawn p : row) {
				sb.append(p.toString());
			}
		}
		sb.append('|').append(s.getTurn());
		return sb.toString();
	}
}
