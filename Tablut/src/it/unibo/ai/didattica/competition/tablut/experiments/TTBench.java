package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.State;

/**
 * Misura l'effetto della DIMENSIONE e della POLITICA della transposition table a profondita' fissa (deterministico, un thread):
 * stesse posizioni di partite vere di {@link OrderingBench}, una istanza per lato riusata per le mosse consecutive (la TT persiste
 * come in partita). Con una TT piccola il numero di nodi sale: il rapporto con la TT grande misura quanto costa la saturazione.
 * <pre>
 *   TTBench &lt;simple|pesi.txt&gt; &lt;profondita'&gt; &lt;partite&gt; &lt;posizioniPerLato&gt; &lt;ordinamento&gt; cfg...    cfg = log2(voci)[:two]   (es. 22 14 14:two)
 * </pre>
 * Stampa nodi, tempo, nodi per ricerca, riempimento finale medio della TT e quante ricerche danno un valore diverso dalla prima
 * configurazione (con TT piccola o riusata puo' capitare: e' il comportamento noto di TT + ripetizioni).
 */
public class TTBench {

	public static void main(String[] a) throws Exception {
		Evaluator ev = a[0].equals("simple") ? new SimpleEvaluator() : EvaluatorLoader.load(a[0]);
		int depth = Integer.parseInt(a[1]);
		int games = Integer.parseInt(a[2]);
		int perSide = Integer.parseInt(a[3]);
		int order = OrderingSpec.parse(a[4]);
		List<List<State>> lines = OrderingBench.recordGames(games, perSide);
		int total = 0;
		for (List<State> l : lines) {
			total += l.size();
		}
		System.out.printf("evaluator=%s profondita'=%d ordinamento=%s: %d linee, %d ricerche%n", a[0], depth, a[4], lines.size(), total);
		double[] ref = null;
		for (int c = 5; c < a.length; c++) {
			String[] parts = a[c].split(":");
			int log2 = Integer.parseInt(parts[0]);
			boolean two = parts.length > 1 && parts[1].equals("two");
			double[] scores = new double[total];
			long nodes = 0, nanos = 0;
			double occSum = 0;
			int k = 0;
			for (List<State> line : lines) {
				Negamax n = new Negamax(ev, depth, true, 1 << log2, true, order, 1, two);
				for (State s : line) {
					long t0 = System.nanoTime();
					n.findBestMove(s, 100_000_000L, new ArrayList<Long>());
					nanos += System.nanoTime() - t0;
					nodes += n.lastNodeCount;
					scores[k++] = n.lastScore;
				}
				occSum += n.transpositionTable().occupancy(1 << 16);
			}
			int diff = 0;
			if (ref == null) {
				ref = scores;
			} else {
				for (int i = 0; i < total; i++) {
					if (Math.abs(scores[i] - ref[i]) > 1e-9) {
						diff++;
					}
				}
			}
			System.out.printf("2^%-2d %-9s nodi %10d (%7.0f/ricerca)  tempo %6.1f s  TT piena al %5.1f%%  valori diversi dal primo: %d%n", log2,
					two ? "due slot" : "slot sing.", nodes, (double) nodes / total, nanos / 1e9, 100 * occSum / lines.size(), diff);
		}
	}
}
