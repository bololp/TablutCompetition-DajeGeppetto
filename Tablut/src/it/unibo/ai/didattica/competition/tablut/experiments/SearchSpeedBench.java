package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Misura la velocita' della ricerca. Usa SOLO l'API pubblica con {@code State}/{@code Action} (che esiste identica prima e
 * dopo la migrazione a {@code FastState}), quindi lo STESSO sorgente si compila e si lancia contro la build storica e contro
 * quella nuova: si confrontano i due output a parita' di posizioni (generate con il motore storico e un seme fisso).
 *
 * <pre>
 *   java ... SearchSpeedBench depth  [evaluator] [profondita'] [posizioni]   // tempo a profondita' fissa (stessi nodi nei due motori)
 *   java ... SearchSpeedBench time   [evaluator] [ms]          [posizioni]   // profondita' raggiunta in un tempo fisso
 * </pre>
 * {@code evaluator} = "simple" oppure il percorso di un file di pesi (MLP o CNN). Un thread di ricerca, TT 2^20, warm-up
 * del JIT prima di misurare.
 */
public class SearchSpeedBench {

	public static void main(String[] a) throws Exception {
		String mode = a.length > 0 ? a[0] : "depth";
		String evSpec = a.length > 1 ? a[1] : "simple";
		int param = a.length > 2 ? Integer.parseInt(a[2]) : (mode.equals("depth") ? 5 : 1000);
		int nPos = a.length > 3 ? Integer.parseInt(a[3]) : 16;
		Evaluator ev = evSpec.equals("simple") ? new SimpleEvaluator() : EvaluatorLoader.load(evSpec);

		List<State> positions = positions(nPos);
		if (mode.equals("depth")) {
			run(ev, positions, param, true); // warm-up
			run(ev, positions, param, true);
			Result r = run(ev, positions, param, false);
			System.out.printf("[%s] profondita' fissa %d, %d posizioni: %d nodi in %.2f s  ->  %.0f nodi/s%n", evSpec, param, positions.size(),
					r.nodes, r.seconds, r.nodes / r.seconds);
		} else {
			double depth = timed(ev, positions, 200, true);
			double d = timed(ev, positions, param, false);
			System.out.printf("[%s] tempo fisso %d ms, %d posizioni: profondita' media raggiunta %.2f, nodi/s medi %.0f%n", evSpec, param,
					positions.size(), d, lastNps);
		}
	}

	static final class Result {
		long nodes;
		double seconds;
	}

	static Result run(Evaluator ev, List<State> positions, int depth, boolean quick) {
		Result r = new Result();
		long t0 = System.nanoTime();
		int n = quick ? Math.min(positions.size(), 6) : positions.size();
		for (int i = 0; i < n; i++) {
			Negamax nm = new Negamax(ev, depth, true, 1 << 20).setThreads(1);
			nm.findBestMove(positions.get(i), 100_000_000L, new ArrayList<Long>());
			r.nodes += nm.lastNodeCount;
		}
		r.seconds = (System.nanoTime() - t0) / 1e9;
		return r;
	}

	static double lastNps;

	/** Profondita' media raggiunta con una sola istanza per posizione (TT fresca), budget in ms (il motore ne toglie un margine di sicurezza: stesso per entrambe le versioni). */
	static double timed(Evaluator ev, List<State> positions, int budgetMs, boolean quick) {
		long nodes = 0;
		long ms = 0;
		double sumDepth = 0;
		int n = quick ? Math.min(positions.size(), 4) : positions.size();
		for (int i = 0; i < n; i++) {
			Negamax nm = new Negamax(ev, 40, true, 1 << 20).setThreads(1);
			long t0 = System.nanoTime();
			nm.findBestMove(positions.get(i), budgetMs + 300, new ArrayList<Long>()); // +300: il margine di sicurezza del motore
			ms += (System.nanoTime() - t0) / 1_000_000;
			nodes += nm.lastNodeCount;
			sumDepth += nm.lastDepthReached;
		}
		lastNps = nodes / Math.max(0.001, ms / 1000.0);
		return sumDepth / n;
	}

	/** Posizioni di partite casuali "avide" (seme fisso), dal ply 6 in poi: stesso insieme per entrambe le build. */
	static List<State> positions(int count) {
		Random rnd = new Random(20260101L);
		List<State> out = new ArrayList<>();
		while (out.size() < count) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			int target = 6 + rnd.nextInt(30);
			for (int ply = 0; ply < target && !AshtonRules.isTerminal(s); ply++) {
				List<Action> mv = AshtonRules.generateMoves(s);
				if (mv.isEmpty()) {
					break;
				}
				Action ch = mv.get(rnd.nextInt(mv.size()));
				for (int t = 0; t < 4; t++) {
					Action c = mv.get(rnd.nextInt(mv.size()));
					State nx = AshtonRules.applyMove(s, c);
					if (AshtonRules.isTerminal(nx) || AshtonRules.wasCapture(s, nx, c)) {
						ch = c;
						break;
					}
				}
				s = AshtonRules.applyMove(s, ch);
			}
			if (!AshtonRules.isTerminal(s) && !AshtonRules.generateMoves(s).isEmpty()) {
				out.add(s);
			}
		}
		return out;
	}
}
