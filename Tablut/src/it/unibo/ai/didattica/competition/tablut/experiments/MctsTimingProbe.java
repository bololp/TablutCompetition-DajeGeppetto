package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Mcts;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MctsConfig;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NegamaxSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Sonda dei TEMPI dell'{@link Mcts} (non e' un test: stampa numeri). Gioca {@code games} partite dell'MCTS (margine 0, budget {@code budget} ms, riuso
 * dell'albero) contro Negamax e registra, per ogni mossa dell'MCTS, di quanto il tempo effettivo supera il budget. Serve a vedere i "sforamenti"
 * dovuti al garbage collector: un albero grande e' memoria viva, e una pausa che attraversa la scadenza fa perdere per timeout.
 * Va lanciata con un heap realistico e il log del GC, ad esempio:
 * <pre>java -Xmx1000m -Xlog:gc:file=gc.log -cp ... MctsTimingProbe games=3 budget=1700 opp=100 [mcts=chiave=valore,...]</pre>
 */
public final class MctsTimingProbe {

	private MctsTimingProbe() {
	}

	public static void main(String[] args) {
		int games = 3;
		long budget = 1700, oppMs = 100;
		String spec = "";
		for (String a : args) {
			if (a.startsWith("games=")) {
				games = Integer.parseInt(a.substring(6));
			} else if (a.startsWith("budget=")) {
				budget = Long.parseLong(a.substring(7));
			} else if (a.startsWith("opp=")) {
				oppMs = Long.parseLong(a.substring(4));
			} else if (a.startsWith("mcts=")) {
				spec = a.substring(5);
			}
		}
		final List<Long> overs = new ArrayList<>();
		for (int g = 0; g < games; g++) {
			final boolean mctsWhite = g % 2 == 0;
			final Mcts mcts = new Mcts(new SimpleEvaluator(), MctsConfig.parse(spec), 1).setSafetyMarginMillis(0);
			final MoveSearcher opp = new NegamaxSearcher(new Negamax(new SimpleEvaluator(), 30, true, 1 << 20, true, Negamax.ORDER_DEFAULT, 1, false).setThreads(1)
					.setSafetyMarginMillis(0));
			final FastState cur = FastState.fromState(new StateTablut());
			final List<Long> hist = new ArrayList<>();
			hist.add(cur.zobristHash);
			long maxOver = Long.MIN_VALUE, maxNodes = 0;
			int moves = 0, late = 0;
			for (int ply = 0; ply < 160 && cur.winner == FastState.WINNER_NONE; ply++) {
				final boolean mctsTurn = (cur.turn == BoardConstants.WHITE_TURN) == mctsWhite;
				final long[] h = new long[hist.size()];
				for (int i = 0; i < h.length; i++) {
					h[i] = hist.get(i);
				}
				final long t0 = System.nanoTime();
				final int mv = mctsTurn ? mcts.findBestMove(cur, budget, h, h.length) : opp.findBestMove(cur, oppMs, h, h.length);
				final long ms = (System.nanoTime() - t0) / 1_000_000L;
				if (mv == Move.NONE) {
					break;
				}
				if (mctsTurn) {
					moves++;
					final long over = ms - budget;
					overs.add(over);
					maxOver = Math.max(maxOver, over);
					maxNodes = Math.max(maxNodes, mcts.lastNodes);
					if (over > 100) {
						late++;
					}
				}
				final UndoInfo u = new UndoInfo();
				cur.makeMove(mv, u);
				if (u.wasCapture()) {
					hist.clear();
				}
				if (hist.contains(cur.zobristHash)) {
					break; // ripetizione: pareggio
				}
				hist.add(cur.zobristHash);
			}
			System.out.printf("partita %d (MCTS %s): %d mosse MCTS, sforamento massimo %d ms, %d mosse oltre +100 ms, nodi massimi %d%n", g + 1, mctsWhite ? "Bianco" : "Nero", moves,
					maxOver, late, maxNodes);
			mcts.shutdown();
			opp.shutdown();
		}
		long worst = Long.MIN_VALUE;
		int over100 = 0, over300 = 0;
		for (long o : overs) {
			worst = Math.max(worst, o);
			if (o > 100) {
				over100++;
			}
			if (o > 300) {
				over300++;
			}
		}
		System.out.printf("TOTALE: %d mosse MCTS, sforamento massimo %d ms, %d oltre +100 ms, %d oltre +300 ms (budget %d ms)%n", overs.size(), worst, over100, over300, budget);
	}
}
