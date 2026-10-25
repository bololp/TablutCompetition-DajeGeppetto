package it.unibo.ai.didattica.competition.tablut.experiments;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Mcts;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MctsConfig;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Misure di costo dell'{@link Mcts} (non e' un test: stampa numeri).
 * <ul>
 * <li><b>Throughput</b>: simulazioni e valutazioni statiche al secondo, per evaluator e numero di thread, dalla posizione iniziale.</li>
 * <li><b>Cache</b>: quota di valutazioni servite dalla {@link it.unibo.ai.didattica.competition.tablut.aiplayer.EvalCache}.</li>
 * <li><b>Memoria</b>: byte reali per nodo espanso, misurati come variazione dell'heap dopo aver costruito un albero di {@code N} nodi e aver forzato
 * il garbage collector. E' il numero da cui dipende {@code Mcts.BYTES_PER_NODE_ESTIMATE}, cioe' il tetto automatico dei nodi.</li>
 * </ul>
 * Uso: {@code MctsBench [--only=throughput|memory|cache] [pesi.txt ...]} (senza pesi solo {@code SimpleEvaluator}; senza {@code --only} tutte le sezioni).
 * Ogni misura dura ~1-2 s, le sezioni con la CNN molto di piu'.
 */
public final class MctsBench {

	private MctsBench() {
	}

	static long usedHeap() {
		for (int i = 0; i < 4; i++) {
			System.gc();
			try {
				Thread.sleep(50);
			} catch (InterruptedException e) {
				return -1;
			}
		}
		final Runtime r = Runtime.getRuntime();
		return r.totalMemory() - r.freeMemory();
	}

	public static void main(String[] args) throws Exception {
		final java.util.List<String> names = new java.util.ArrayList<>();
		final java.util.List<Evaluator> evals = new java.util.ArrayList<>();
		names.add("simple");
		evals.add(new SimpleEvaluator());
		String only = "";
		for (String a : args) {
			if (a.startsWith("--only=")) {
				only = a.substring(7);
				continue;
			}
			names.add(a);
			evals.add(EvaluatorLoader.load(a));
		}
		final boolean doThroughput = only.isEmpty() || only.equals("throughput");
		final boolean doMemory = only.isEmpty() || only.equals("memory");
		final boolean doCache = only.isEmpty() || only.equals("cache");
		final FastState start = FastState.fromState(new StateTablut());
		final long[] hist = { start.zobristHash };

		System.out.println("== throughput dalla posizione iniziale (1000 ms per misura, margine 0) ==");
		for (int e = 0; doThroughput && e < evals.size(); e++) {
			for (String cfg : new String[] { "expand=1", "expand=0", "expand=1,cache=0" }) {
				for (int threads : new int[] { 1, 2 }) {
					final Mcts m = new Mcts(evals.get(e), MctsConfig.parse(cfg + ",reuse=false"), threads).setSafetyMarginMillis(0);
					m.findBestMove(start, 300, hist, 1); // riscaldamento del JIT
					m.clearTree();
					final long t0 = System.nanoTime();
					final int mv = m.findBestMove(start, 1000, hist, 1);
					final double s = (System.nanoTime() - t0) / 1e9;
					final long total = m.lastEvals + m.lastCacheHits;
					System.out.printf("%-8s %-18s thread=%d: %8.0f sim/s  %9.0f valutazioni/s (+%2.0f%% dalla cache)  %7d nodi  mossa=%s%n", names.get(e), cfg, threads, m.lastSims / s,
							m.lastEvals / s, total > 0 ? 100.0 * m.lastCacheHits / total : 0.0, m.lastNodes, Move.toString(mv));
					m.shutdown();
				}
			}
		}

		System.out.println("\n== memoria: byte per nodo espanso (heap dopo gc, albero di N nodi) ==");
		for (int e = 0; doMemory && e < evals.size(); e++) {
			for (String cfg : new String[] { "expand=1", "expand=0", "expand=1,mm=0.5" }) {
				final long n = 60_000;
				final Mcts m = new Mcts(evals.get(e), MctsConfig.parse(cfg + ",reuse=false,cache=0,maxnodes=" + n), 1).setSafetyMarginMillis(0);
				final long before = usedHeap();
				m.findBestMove(start, 600000, hist, 1);
				final long after = usedHeap();
				final long nodes = m.lastNodes;
				System.out.printf("%-8s %-18s: %d nodi, %.1f MB -> %.0f byte/nodo (sim %d)%n", names.get(e), cfg, nodes, (after - before) / 1048576.0, nodes > 0 ? (double) (after - before) / nodes : 0.0,
						m.lastSims);
				m.clearTree();
			}
		}
		System.out.println("\n== cache A/B a parita' di albero (simulazioni fisse, 1 thread: l'albero e' IDENTICO con e senza cache, quindi la differenza di tempo e' solo la cache) ==");
		final java.util.Random rnd = new java.util.Random(3);
		final it.unibo.ai.didattica.competition.tablut.selftest.TestPositions.Game game = it.unibo.ai.didattica.competition.tablut.selftest.TestPositions.playGame(rnd, true, 80);
		final java.util.List<FastState> pos = new java.util.ArrayList<>();
		for (int i = 8; i < game.positions.size() && pos.size() < 4; i += 9) {
			final FastState f = FastState.fromState(game.positions.get(i));
			if (f.winner == FastState.WINNER_NONE) {
				pos.add(f);
			}
		}
		for (int e = 0; doCache && e < evals.size(); e++) {
			final long sims = e == 0 ? 30000 : (names.get(e).contains("cnn") ? 2500 : 8000);
			double base = 0;
			for (int log2 : new int[] { 0, 12, 16, 18, 22 }) {
				double best = Double.MAX_VALUE;
				double hit = 0;
				for (int rep = 0; rep < 4; rep++) { // la prima ripetizione scalda il JIT, si prende il minimo
					double t = 0;
					long hits = 0, total = 0;
					for (FastState p : pos) {
						final Mcts m = new Mcts(evals.get(e), MctsConfig.parse("expand=1,reuse=false,cache=" + log2), 1).setSafetyMarginMillis(0);
						m.setMaxSims(sims);
						final long t0 = System.nanoTime();
						m.findBestMove(p, 600000, new long[] { p.zobristHash }, 1);
						t += (System.nanoTime() - t0) / 1e6;
						hits += m.lastCacheHits;
						total += m.lastCacheHits + m.lastEvals;
					}
					if (rep > 0 && t < best) {
						best = t;
						hit = total > 0 ? 100.0 * hits / total : 0;
					}
				}
				if (log2 == 0) {
					base = best;
				}
				System.out.printf("%-8s cache=%-2d (%d simulazioni x %d posizioni): %8.0f ms  %5.2fx rispetto a nessuna cache  hit %4.1f%%%n", names.get(e).replaceAll(".*/", ""), log2, sims, pos.size(), best,
						best / base, hit);
			}
		}

		System.out.println("\nStima in uso (Mcts.BYTES_PER_NODE_ESTIMATE): " + Mcts.BYTES_PER_NODE_ESTIMATE + " byte/nodo");
	}
}
