package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Confronta Negamax CON transposition table e SENZA, sulle stesse posizioni
 * e alla stessa profondita' fissa (nessun limite di tempo, cosi' il confronto
 * e' deterministico). Se la TT e' implementata correttamente (gestione
 * corretta dei bound EXACT/LOWER/UPPER), il punteggio e la mossa scelta
 * DEVONO essere identici in entrambi i casi - una TT corretta cambia SOLO
 * quanti nodi servono per arrivare alla risposta, mai la risposta stessa. Se
 * anche un solo confronto differisse, ci sarebbe un bug nella gestione dei
 * bound (il tipo di bug che fa scegliere mosse leggermente sbagliate in
 * silenzio, senza mai un errore/crash visibile).
 */
public class TTCorrectnessCheck {

	public static void main(String[] args) {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 55L;
		int numPositions = Integer.parseInt(args.length > 1 ? args[1] : "8");
		int maxDepthToTest = Integer.parseInt(args.length > 2 ? args[2] : "4");
		long generousBudget = 300_000L; // enorme apposta: non deve MAI scadere in questo test

		Random rnd = new Random(seed);
		List<State> positions = collectPositions(rnd, numPositions);

		int mismatches = 0;
		long totalNodesWithTT = 0, totalNodesWithoutTT = 0;

		for (int pIdx = 0; pIdx < positions.size(); pIdx++) {
			State pos = positions.get(pIdx);
			for (int depth = 1; depth <= maxDepthToTest; depth++) {
				SimpleEvaluator evalA = new SimpleEvaluator();
				SimpleEvaluator evalB = new SimpleEvaluator();

				Negamax withTT = new Negamax(evalA, depth, true, 1 << 16).setThreads(1);
				Negamax withoutTT = new Negamax(evalB, depth, false, 0).setThreads(1);

				Action moveWithTT = withTT.findBestMove(pos, generousBudget);
				Action moveWithoutTT = withoutTT.findBestMove(pos, generousBudget);

				totalNodesWithTT += withTT.lastNodeCount;
				totalNodesWithoutTT += withoutTT.lastNodeCount;

				boolean scoreMatches = Math.abs(withTT.lastScore - withoutTT.lastScore) < 1e-9;
				boolean moveMatches = sameMove(moveWithTT, moveWithoutTT);

				if (!scoreMatches || !moveMatches) {
					mismatches++;
					System.out.println("--- MISMATCH (posizione " + pIdx + ", profondita' " + depth + ") ---");
					System.out.println("Stato:\n" + pos);
					System.out.println("Con TT:    mossa=" + moveWithTT + " punteggio=" + withTT.lastScore
							+ " nodi=" + withTT.lastNodeCount);
					System.out.println("Senza TT:  mossa=" + moveWithoutTT + " punteggio=" + withoutTT.lastScore
							+ " nodi=" + withoutTT.lastNodeCount);
				}
			}
		}

		System.out.println("=== FINE: " + positions.size() + " posizioni x profondita' 1.." + maxDepthToTest + " = "
				+ (positions.size() * maxDepthToTest) + " confronti, " + mismatches + " MISMATCH ===");
		System.out.println("Nodi totali CON TT: " + totalNodesWithTT + " | SENZA TT: " + totalNodesWithoutTT
				+ " | riduzione: " + String.format("%.1f%%", 100.0 * (1.0 - (double) totalNodesWithTT / totalNodesWithoutTT)));

		if (mismatches > 0) {
			System.exit(1);
		}
	}

	private static boolean sameMove(Action a, Action b) {
		if (a == null || b == null) {
			return a == b;
		}
		return a.getRowFrom() == b.getRowFrom() && a.getColumnFrom() == b.getColumnFrom()
				&& a.getRowTo() == b.getRowTo() && a.getColumnTo() == b.getColumnTo();
	}

	/** Raccoglie posizioni distinte (a stadi diversi di partita, con catture gia'
	 *  avvenute) da giocate casuali, cosi' il confronto non riguarda solo la
	 *  posizione iniziale. */
	private static List<State> collectPositions(Random rnd, int count) {
		List<State> positions = new ArrayList<>();
		positions.add(freshInitial());

		while (positions.size() < count) {
			State state = freshInitial();
			int targetPly = 3 + rnd.nextInt(40);
			for (int ply = 0; ply < targetPly; ply++) {
				if (AshtonRules.isTerminal(state)) {
					break;
				}
				List<Action> moves = AshtonRules.generateMoves(state);
				if (moves.isEmpty()) {
					break;
				}
				state = AshtonRules.applyMove(state, moves.get(rnd.nextInt(moves.size())));
			}
			if (!AshtonRules.isTerminal(state)) {
				positions.add(state);
			}
		}
		return positions;
	}

	private static State freshInitial() {
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		return s;
	}
}
