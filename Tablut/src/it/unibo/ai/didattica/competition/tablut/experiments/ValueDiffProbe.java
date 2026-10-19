package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.State;

/**
 * Per le posizioni di {@link OrderingBench} in cui il valore di radice a profondita' fissa (TT accesa, istanza nuova) dipende
 * dall'ordinamento, confronta i valori con la TT SPENTA (alpha-beta esatto: non puo' dipendere dall'ordinamento) e con la TT accesa.
 * Serve a distinguere "l'ordinamento cambia il valore" (un bug) da "la TT dipende dall'ordine in cui riempie le voci" (profondita'
 * residue diverse per la stessa posizione raggiunta per trasposizione).
 * <pre> ValueDiffProbe &lt;profondita'&gt; &lt;partite&gt; &lt;posizioniPerLato&gt; &lt;ordinamentoA&gt; &lt;ordinamentoB&gt; </pre>
 */
public class ValueDiffProbe {
	public static void main(String[] a) {
		int depth = Integer.parseInt(a[0]);
		List<List<State>> lines = OrderingBench.recordGames(Integer.parseInt(a[1]), Integer.parseInt(a[2]));
		int oa = OrderingSpec.parse(a[3]), ob = OrderingSpec.parse(a[4]);
		SimpleEvaluator ev = new SimpleEvaluator();
		int total = 0, diffTT = 0, diffNoTT = 0, ttMatchesExact = 0;
		for (List<State> line : lines) {
			for (State s : line) {
				total++;
				double ta = value(ev, depth, true, oa, s), tb = value(ev, depth, true, ob, s);
				if (Math.abs(ta - tb) < 1e-9) {
					continue;
				}
				diffTT++;
				double ea = value(ev, depth, false, oa, s), eb = value(ev, depth, false, ob, s);
				if (Math.abs(ea - eb) > 1e-9) {
					diffNoTT++;
				}
				boolean aOk = Math.abs(ta - ea) < 1e-9, bOk = Math.abs(tb - eb) < 1e-9;
				System.out.printf("posizione %d: con TT %s=%.1f %s=%.1f | senza TT (esatto) %s=%.1f %s=%.1f | TT di %s %s, TT di %s %s%n", total, a[3], ta, a[4], tb, a[3],
						ea, a[4], eb, a[3], aOk ? "= esatto" : "diverso dall'esatto", a[4], bOk ? "= esatto" : "diverso dall'esatto");
				if (aOk || bOk) {
					ttMatchesExact++;
				}
			}
		}
		System.out.printf("%d posizioni; valore con TT dipendente dall'ordinamento in %d; di queste, valore ESATTO (senza TT) dipendente dall'ordinamento in %d%n", total,
				diffTT, diffNoTT);
	}

	static double value(SimpleEvaluator ev, int depth, boolean tt, int order, State s) {
		Negamax n = new Negamax(ev, depth, tt, 1 << 20, true, order, 1, false).setThreads(1);
		n.findBestMove(s, 100_000_000L, new ArrayList<Long>());
		return n.lastScore;
	}
}
