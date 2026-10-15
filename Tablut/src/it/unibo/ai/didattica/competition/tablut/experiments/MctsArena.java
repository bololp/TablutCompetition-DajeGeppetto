package it.unibo.ai.didattica.competition.tablut.experiments;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * MCTS ({@link MctsPlayer}) contro Negamax A TEMPO UGUALE con lo STESSO file di pesi: isola la domanda "l'algoritmo di ricerca"
 * dalla qualita' della rete. Stesso protocollo di {@code EqualTimeArena} (apertura casuale di 4 semi-mosse giocata due volte a colori
 * invertiti, tetto di 160 semi-mosse = pareggio, nessuna ripetizione, 1 thread per lato).
 * <p>
 * Uso: {@code MctsArena pesi.txt [coppie=20] [ms=100] [seed=1] [cPuct=1.0] [tau=0.2]} (punteggio dal punto di vista dell'MCTS).
 */
public class MctsArena {

	/** Margine che {@code Negamax.findBestMove} toglie al budget ricevuto (SAFETY_MARGIN_MILLIS): senza compensarlo, a budget bassi Negamax avrebbe meta' del tempo dell'MCTS. */
	static final long NEGAMAX_SAFETY_MARGIN_MS = 400;

	public static void main(String[] a) throws IOException {
		if (a.length < 1) {
			System.out.println("Uso: MctsArena pesi.txt [coppie=20] [ms=100] [seed=1] [cPuct=1.0] [tau=0.2]");
			return;
		}
		Evaluator ev = EvaluatorLoader.load(a[0]);
		int pairs = a.length > 1 ? Integer.parseInt(a[1]) : 20;
		long ms = a.length > 2 ? Long.parseLong(a[2]) : 100;
		long seed = a.length > 3 ? Long.parseLong(a[3]) : 1;
		double c = a.length > 4 ? Double.parseDouble(a[4]) : 1.0, tau = a.length > 5 ? Double.parseDouble(a[5]) : 0.2;
		MctsPlayer mcts = new MctsPlayer(ev, 1000.0, c, tau);
		Negamax nega = new Negamax(ev, 30, true, 1 << 18).setThreads(1);
		Random rnd = new Random(seed);
		int mw = 0, md = 0, ml = 0;
		double sims = 0, simTime = 0, depth = 0, nodes = 0, negTime = 0, mctsDepth = 0, evals = 0;
		long mMoves = 0, nMoves = 0;
		for (int p = 0; p < pairs; p++) {
			State open = new StateTablut();
			open.setTurn(Turn.WHITE);
			for (int i = 0; i < 4; i++) {
				List<Action> mv = AshtonRules.generateMoves(open);
				open = AshtonRules.applyMove(open, mv.get(rnd.nextInt(mv.size())));
			}
			for (int swap = 0; swap < 2; swap++) { // swap 0: MCTS e' il Bianco
				State s = open;
				for (int ply = 4; ply < 160 && !AshtonRules.isTerminal(s); ply++) {
					boolean mctsMoves = (s.getTurn() == Turn.WHITE) == (swap == 0);
					long t0 = System.nanoTime();
					Action m;
					if (mctsMoves) {
						m = mcts.findBestMove(s, ms);
						simTime += (System.nanoTime() - t0) / 1e9;
						sims += mcts.lastSims;
						evals += mcts.lastEvals;
						mctsDepth += mcts.lastMaxDepth;
						mMoves++;
					} else {
						m = nega.findBestMove(s, ms + NEGAMAX_SAFETY_MARGIN_MS); // Negamax sottrae 400 ms di margine di sicurezza (e ha un minimo di 50 ms): lo si compensa
						negTime += (System.nanoTime() - t0) / 1e9;
						depth += nega.lastDepthReached;
						nodes += nega.lastTotalNodeCount;
						nMoves++;
					}
					if (m == null) {
						break;
					}
					s = AshtonRules.applyMove(s, m);
				}
				Turn t = s.getTurn();
				int res;
				if (t == Turn.WHITEWIN) {
					res = swap == 0 ? 1 : -1;
				} else if (t == Turn.BLACKWIN) {
					res = swap == 0 ? -1 : 1;
				} else if (!AshtonRules.isTerminal(s) && AshtonRules.generateMoves(s).isEmpty()) {
					res = (s.getTurn() == Turn.WHITE) == (swap == 0) ? -1 : 1;
				} else {
					res = 0;
				}
				if (res > 0) {
					mw++;
				} else if (res < 0) {
					ml++;
				} else {
					md++;
				}
			}
		}
		int n = mw + md + ml;
		double score = (mw + 0.5 * md) / n, z = 1.96, den = 1 + z * z / n;
		double centre = (score + z * z / (2 * n)) / den, half = z * Math.sqrt(score * (1 - score) / n + z * z / (4.0 * n * n)) / den;
		System.out.printf("MCTS(cPuct=%.2f, tau=%.2f) VS Negamax, stessi pesi %s @%dms/mossa: %d partite  MCTS +%d =%d -%d  punteggio MCTS = %.1f%% [IC95 %.1f%% - %.1f%%]%n",
				c, tau, a[0], ms, n, mw, md, ml, 100 * score, 100 * (centre - half), 100 * (centre + half));
		System.out.printf("   tempo realmente usato per mossa: MCTS %.1f ms, Negamax %.1f ms (budget %d ms)%n", 1000 * simTime / Math.max(1, mMoves), 1000 * negTime / Math.max(1, nMoves), ms);
		System.out.printf("   MCTS: %.0f simulazioni/mossa, %.0f valutazioni/s, profondita' massima media %.1f | Negamax: profondita' media %.2f, %.0f nodi/s%n",
				sims / Math.max(1, mMoves), evals / Math.max(1e-9, simTime), mctsDepth / Math.max(1, mMoves), depth / Math.max(1, nMoves), nodes / Math.max(1e-9, negTime));
	}
}
