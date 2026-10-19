package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Misura quanti nodi e quanto tempo servono varie modalita' di ordinamento delle mosse, a profondita' FISSA, su posizioni di
 * partite vere (giocate da {@code Simple} a profondita' 3 con il 10% di mosse casuali), con UNA istanza di ricerca riusata per
 * le mosse consecutive di un lato (come nel client: TT, history, killer-carry, countermove persistono). Il valore di radice di
 * ogni ricerca viene confrontato con quello della prima configurazione (con TT riusata e ripetizioni possono esserci rare
 * differenze: si contano, non sono un errore).
 * <pre>
 *   OrderingBench &lt;simple|pesi.txt&gt; &lt;profondita'&gt; &lt;partite&gt; &lt;posizioniPerLato&gt; cfg1 cfg2 ...   (es. base wins caps counter all)
 * </pre>
 * Un thread di ricerca, TT 2^20. La prima configurazione viene ripetuta alla fine (ultima riga) per stimare la deriva del tempo.
 */
public class OrderingBench {

	/** {@code -Dbench.fresh=true}: una istanza NUOVA per ogni posizione (niente TT/history/killer riusate tra ricerche consecutive). */
	static final boolean FRESH = Boolean.getBoolean("bench.fresh");

	public static void main(String[] a) throws Exception {
		String evSpec = a[0];
		int depth = Integer.parseInt(a[1]);
		int games = Integer.parseInt(a[2]);
		int perSide = Integer.parseInt(a[3]);
		List<String> cfgs = new ArrayList<>();
		for (int i = 4; i < a.length; i++) {
			cfgs.add(a[i]);
		}
		Evaluator ev = evSpec.equals("simple") ? new SimpleEvaluator() : EvaluatorLoader.load(evSpec);
		List<List<State>> lines = recordGames(games, perSide);
		int total = 0;
		for (List<State> l : lines) {
			total += l.size();
		}
		System.out.printf("evaluator=%s profondita'=%d: %d linee, %d ricerche per configurazione%n", evSpec, depth, lines.size(), total);

		run(ev, depth, lines, cfgs.get(0), null, true); // riscaldamento del JIT
		double[][] ref = new double[1][];
		List<String> order = new ArrayList<>(cfgs);
		order.add(cfgs.get(0));
		boolean first = true;
		for (String c : order) {
			double[] scores = new double[total];
			long[] r = run(ev, depth, lines, c, scores, false);
			int diff = 0;
			if (first) {
				ref[0] = scores;
				first = false;
			} else {
				for (int i = 0; i < total; i++) {
					if (Math.abs(scores[i] - ref[0][i]) > 1e-9) {
						diff++;
					}
				}
			}
			System.out.printf("%-22s nodi %10d  tempo %7.2f s  %8.0f nodi/s  valori diversi dal primo: %d%n", c, r[0], r[1] / 1e9, r[0] / (r[1] / 1e9), diff);
		}
	}

	/** @return {nodi totali, nanosecondi} */
	static long[] run(Evaluator ev, int depth, List<List<State>> lines, String cfg, double[] scores, boolean quick) {
		final int mode = OrderingSpec.parse(cfg);
		long nodes = 0, nanos = 0;
		int k = 0;
		for (List<State> line : lines) {
			Negamax n = new Negamax(ev, depth, true, 1 << 20, true, mode).setThreads(1);
			for (State s : line) {
				if (FRESH) {
					n = new Negamax(ev, depth, true, 1 << 20, true, mode).setThreads(1); // TT, history e killer vuote a ogni ricerca
				}
				long t0 = System.nanoTime();
				n.findBestMove(s, 100_000_000L, new ArrayList<Long>());
				nanos += System.nanoTime() - t0;
				nodes += n.lastNodeCount;
				if (scores != null) {
					scores[k] = n.lastScore;
				}
				k++;
			}
			if (quick && k > 40) {
				break;
			}
		}
		return new long[] { nodes, nanos };
	}

	/** Partite di Simple(depth 3) con il 10% di mosse casuali; per ogni partita e per ogni lato, le posizioni in cui quel lato muove (dal ply 6). */
	static List<List<State>> recordGames(int games, int perSide) {
		Random rnd = new Random(777);
		List<List<State>> out = new ArrayList<>();
		for (int g = 0; g < games; g++) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			for (int i = 0; i < 4; i++) {
				List<Action> mv = AshtonRules.generateMoves(s);
				s = AshtonRules.applyMove(s, mv.get(rnd.nextInt(mv.size())));
			}
			Negamax w = new Negamax(new SimpleEvaluator(), 3, true, 1 << 18).setThreads(1);
			Negamax b = new Negamax(new SimpleEvaluator(), 3, true, 1 << 18).setThreads(1);
			List<State> white = new ArrayList<>(), black = new ArrayList<>();
			for (int ply = 4; ply < 120 && !AshtonRules.isTerminal(s); ply++) {
				boolean wt = s.getTurn() == Turn.WHITE;
				if (ply >= 6) {
					List<State> dst = wt ? white : black;
					if (dst.size() < perSide) {
						dst.add(s);
					}
				}
				Action m;
				if (rnd.nextInt(10) == 0) {
					List<Action> mv = AshtonRules.generateMoves(s);
					if (mv.isEmpty()) {
						break;
					}
					m = mv.get(rnd.nextInt(mv.size()));
				} else {
					m = (wt ? w : b).findBestMove(s, 100_000_000L);
				}
				if (m == null) {
					break;
				}
				s = AshtonRules.applyMove(s, m);
			}
			if (white.size() > 2) {
				out.add(white);
			}
			if (black.size() > 2) {
				out.add(black);
			}
		}
		return out;
	}
}
