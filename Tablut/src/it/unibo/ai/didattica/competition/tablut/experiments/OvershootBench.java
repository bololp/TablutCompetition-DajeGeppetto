package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.State;

/**
 * Misura di quanto {@code Negamax.findBestMove} SFORA il tempo che gli e' stato assegnato: con margine 0 e budget B, lo sforo e'
 * {@code tempoUsato - B} (il tempo per fermare la ricerca, attendere gli helper e restituire la mossa). Il margine di sicurezza deve
 * stare sopra lo sforo peggiore + la latenza di rete/JSON + le pause del GC. Usa posizioni di partite vere ({@link OrderingBench}),
 * evaluator e thread a scelta, TT 2^22 come in gara.
 * <pre>
 *   OvershootBench &lt;simple|pesi.txt&gt; &lt;budgetMs&gt; &lt;ricerche&gt; &lt;thread&gt; [ordinamento=base]
 * </pre>
 * Attenzione: con piu' thread dei core il sandbox alterna i thread a fette di tempo e lo sforo del principale puo' peggiorare
 * (caso sfavorevole); su una macchina a 4 core con 4 thread va misurato di nuovo.
 */
public class OvershootBench {
	public static void main(String[] a) throws Exception {
		Evaluator ev = a[0].equals("simple") ? new SimpleEvaluator() : EvaluatorLoader.load(a[0]);
		long budget = Long.parseLong(a[1]);
		int searches = Integer.parseInt(a[2]);
		int threads = Integer.parseInt(a[3]);
		int order = OrderingSpec.parse(a.length > 4 ? a[4] : "base");
		List<State> positions = new ArrayList<>();
		for (List<State> l : OrderingBench.recordGames(6, 12)) {
			positions.addAll(l);
		}
		Negamax n = new Negamax(ev, 40, true, Negamax.DEFAULT_TT_SIZE, true, order, threads, false).setThreads(threads).setSafetyMarginMillis(0);
		try {
			// riscaldamento del JIT (due ricerche brevi)
			n.findBestMove(positions.get(0), 300L, new ArrayList<Long>());
			n.findBestMove(positions.get(1), 300L, new ArrayList<Long>());
			List<Long> over = new ArrayList<>();
			int short_ = 0;
			for (int i = 0; i < searches && i < positions.size(); i++) {
				long t0 = System.nanoTime();
				n.findBestMove(positions.get(i), budget, new ArrayList<Long>());
				long used = (System.nanoTime() - t0) / 1_000_000L;
				if (used < budget - 100) {
					short_++; // trovata una vittoria forzata: non misura il tempo
					continue;
				}
				over.add(used - budget);
			}
			long[] o = over.stream().mapToLong(Long::longValue).sorted().toArray();
			System.out.printf("evaluator=%s thread=%d budget=%d ms: %d ricerche misurate (%d chiuse prima per vittoria forzata)%n", a[0], threads, budget,
					o.length, short_);
			if (o.length > 0) {
				System.out.printf("sforo (ms): min %d  mediana %d  p90 %d  max %d   (profondita' ultima ricerca %d, helper in errore %d)%n", o[0], o[o.length / 2],
						o[(int) Math.min(o.length - 1, Math.ceil(0.9 * o.length) - 1)], o[o.length - 1], n.lastDepthReached, n.lastHelperErrors);
			}
		} finally {
			n.shutdown();
		}
	}
}
