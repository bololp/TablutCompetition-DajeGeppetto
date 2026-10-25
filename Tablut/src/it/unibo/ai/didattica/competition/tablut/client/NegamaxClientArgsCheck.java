package it.unibo.ai.didattica.competition.tablut.client;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Mcts;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NegamaxSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Verifica le funzioni di configurazione del client: margine di default, dimensione della TT, adattamento all'heap, e la scelta dell'algoritmo
 * di ricerca ({@code algo=negamax|mcts}).
 */
public class NegamaxClientArgsCheck {
	static int failures = 0;

	static void check(boolean ok, String msg) {
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	public static void main(String[] args) {
		check(TablutNegamaxClient.defaultMarginMillis(60) == 1500, "60 s -> 1500 ms");
		check(TablutNegamaxClient.defaultMarginMillis(20) == 1500, "20 s -> 1500 ms");
		check(TablutNegamaxClient.defaultMarginMillis(19) == 300 && TablutNegamaxClient.defaultMarginMillis(2) == 300, "timeout brevi -> margine storico 300 ms");
		check(TablutNegamaxClient.ttBytes(22) == 100_663_296L, "2^22 voci = 96 MiB");
		check(TablutNegamaxClient.ttBytes(25) == 805_306_368L, "2^25 voci = 768 MiB");
		final long GiB = 1L << 30;
		check(TablutNegamaxClient.fitTtToHeap(22, 2 * GiB) == 22, "2^22 sta in 2 GiB");
		check(TablutNegamaxClient.fitTtToHeap(25, 2 * GiB) == 25, "2^25 (768 MiB) sta in 2 GiB: serve meta' dell'heap, 1 GiB");
		check(TablutNegamaxClient.fitTtToHeap(26, 2 * GiB) == 25, "2^26 (1,5 GiB) non sta in 2 GiB: scende a 2^25");
		check(TablutNegamaxClient.fitTtToHeap(25, GiB / 2) == 23, "in 512 MiB il limite e' 256 MiB: 2^23 (192 MiB) ci sta, 2^24 (384 MiB) no");
		check(TablutNegamaxClient.fitTtToHeap(22, 1L << 20) == 14, "heap di 1 MiB: il limite e' 512 KiB, 2^14 (384 KiB) ci sta e 2^15 (768 KiB) no");
		check(TablutNegamaxClient.fitTtToHeap(22, 1L << 16) == 12, "heap di 64 KiB: non scende sotto il pavimento 2^12");
		for (int log2 = 12; log2 <= 26; log2++) {
			final long heap = 1L << 31;
			final int f = TablutNegamaxClient.fitTtToHeap(log2, heap);
			check(f <= log2 && TablutNegamaxClient.ttBytes(f) <= heap / 2, "fit 2^" + log2 + " -> 2^" + f + " rispetta il limite");
			check(f == log2 || TablutNegamaxClient.ttBytes(f + 1) > heap / 2, "fit 2^" + log2 + " -> 2^" + f + " e' il massimo possibile");
		}
		// ---- scelta dell'algoritmo
		check(TablutNegamaxClient.normalizeAlgo("MCTS").equals("mcts") && TablutNegamaxClient.normalizeAlgo(" Negamax ").equals("negamax"), "algo: maiuscole e spazi tollerati");
		for (String bad : new String[] { "foo", "", "mcts2", null }) {
			boolean threw = false;
			try {
				TablutNegamaxClient.normalizeAlgo(bad);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "algo sconosciuto rifiutato: '" + bad + "'");
		}
		check(TablutNegamaxClient.mctsExtraMarginMillis(60) == 500 && TablutNegamaxClient.mctsExtraMarginMillis(20) == 500 && TablutNegamaxClient.mctsExtraMarginMillis(19) == 0
				&& TablutNegamaxClient.mctsExtraMarginMillis(2) == 0, "margine extra dell'MCTS: 500 ms da 20 s di timeout in su, 0 sotto");
		check(TablutNegamaxClient.defaultMarginMillis(60) + TablutNegamaxClient.mctsExtraMarginMillis(60) == 2000, "margine di default MCTS a 60 s: 2000 ms");
		final SimpleEvaluator ev = new SimpleEvaluator();
		final MoveSearcher neg = TablutNegamaxClient.buildSearcher("negamax", "cpuct=9", ev, 2, 777, Negamax.ORDER_DEFAULT, 12, false);
		check(neg instanceof NegamaxSearcher, "algo=negamax -> NegamaxSearcher (i parametri MCTS non contano)");
		check(neg instanceof NegamaxSearcher && ((NegamaxSearcher) neg).negamax().getSafetyMarginMillis() == 777, "negamax: margine passato");
		final MoveSearcher mc = TablutNegamaxClient.buildSearcher("mcts", "cpuct=2,expand=0", ev, 3, 555, Negamax.ORDER_DEFAULT, 12, false);
		check(mc instanceof Mcts, "algo=mcts -> Mcts");
		if (mc instanceof Mcts) {
			final Mcts m = (Mcts) mc;
			check(m.config().cPuct == 2 && m.config().expandAfter == 0, "mcts: parametri letti");
			check(m.getThreads() == 3 && m.getSafetyMarginMillis() == 555, "mcts: thread e margine passati");
			check(m.config().maxNodes == Mcts.autoMaxNodes(Runtime.getRuntime().maxMemory()), "mcts: tetto nodi ricavato dall'heap se non e' dato");
			check(((Mcts) TablutNegamaxClient.buildSearcher("mcts", "maxnodes=1234", ev, 1, 0, Negamax.ORDER_DEFAULT, 12, false)).config().maxNodes == 1234,
					"mcts: tetto nodi esplicito rispettato");
			final FastState root = FastState.fromState(new StateTablut());
			final int mv = m.setThreads(1).setSafetyMarginMillis(0).findBestMove(root, 150, new long[] { root.zobristHash }, 1);
			check(mv != Move.NONE && m.lastSims > 0, "mcts dal client: restituisce una mossa dopo aver simulato");
			m.shutdown();
		}
		for (String bad : new String[] { "cpuct=abc", "foo=1", "tau=0" }) {
			boolean threw = false;
			try {
				TablutNegamaxClient.buildSearcher("mcts", bad, ev, 1, 0, Negamax.ORDER_DEFAULT, 12, false);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "parametri MCTS non validi rifiutati: '" + bad + "'");
		}
		if (failures == 0) {
			System.out.println("=== NEGAMAX CLIENT ARGS CHECK OK ===");
		} else {
			System.out.println("=== FALLITO: " + failures + " ===");
			System.exit(1);
		}
	}
}
