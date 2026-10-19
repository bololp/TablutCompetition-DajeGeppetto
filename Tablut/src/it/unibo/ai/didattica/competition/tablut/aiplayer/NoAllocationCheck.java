package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Custodisce la promessa centrale della migrazione a {@link FastState}: la ricerca NON ALLOCA memoria per nodo. Misura i byte
 * allocati dal thread corrente (HotSpot: {@code com.sun.management.ThreadMXBean.getThreadAllocatedBytes}) durante una ricerca
 * a profondita' fissa su una istanza GIA' scaldata (TT e stack gia' allocati), con tre evaluator, e richiede meno di
 * {@value #MAX_BYTES_PER_NODE} byte per nodo (il rumore residuo sono allocazioni per CHIAMATA, non per nodo). Per confronto
 * stampa il consumo della versione storica ({@link LegacyNegamax}), che alloca uno {@code State} e un'{@code Action} per figlio.
 * <p>
 * Se la JVM non espone il contatore (JVM non HotSpot) il test stampa SKIP ed esce con successo: non e' un errore del codice.
 */
public class NoAllocationCheck {

	static final double MAX_BYTES_PER_NODE = 0.5;

	/** Modalita' di ordinamento da verificare: {@code -Dnoalloc.order=rec} (nomi di {@link OrderingSpec}; default: quello storico). */
	static final int ORDER = OrderingSpec.parse(System.getProperty("noalloc.order", "base"));

	public static void main(String[] args) throws Exception {
		Object bean = ManagementFactory.getThreadMXBean();
		Method m;
		try {
			m = Class.forName("com.sun.management.ThreadMXBean").getMethod("getThreadAllocatedBytes", long.class);
			m.setAccessible(true);
			m.invoke(bean, Thread.currentThread().getId());
		} catch (Throwable t) {
			System.out.println("SKIP: la JVM non espone getThreadAllocatedBytes (" + t + ")");
			return;
		}
		final Method alloc = m;
		final Object mx = bean;

		Random rnd = new Random(99);
		TestPositions.Game game = TestPositions.playGame(rnd, true, 60);
		State[] states = new State[5];
		FastState[] fast = new FastState[5];
		for (int i = 0; i < 5; i++) {
			states[i] = game.positions.get(Math.min(game.positions.size() - 1, 4 + i * 4));
			fast[i] = FastState.fromState(states[i]);
		}

		Evaluator[] evs = { new SimpleEvaluator(), NeuralEvaluator.randomInit(3), CnnEvaluator.randomInit(CnnEvaluator.defaultConfig(TablutFeatures.G_DEFAULT), TablutFeatures.G_DEFAULT, 5) };
		String[] names = { "Simple", "Rete MLP", "Rete CNN" };
		int[] depths = { 5, 4, 3 };
		int failures = 0;
		for (int e = 0; e < evs.length; e++) {
			// scaldamento su un'altra istanza: JIT e ThreadLocal degli evaluator. La misura usa un'istanza NUOVA (TT vuota: altrimenti
			// le stesse posizioni verrebbero risolte dalla TT in poche centinaia di nodi), creata PRIMA di leggere il contatore.
			Negamax warm = new Negamax(evs[e], depths[e], true, 1 << 18, true, ORDER, 1);
			for (int pass = 0; pass < 2; pass++) {
				for (FastState f : fast) {
					warm.findBestMove(f, 1_000_000L);
				}
			}
			Negamax fresh = new Negamax(evs[e], depths[e], true, 1 << 18, true, ORDER, 1);
			long nodes = 0;
			long before = bytes(alloc, mx);
			for (FastState f : fast) {
				fresh.findBestMove(f, 1_000_000L);
				nodes += fresh.lastNodeCount;
			}
			long used = bytes(alloc, mx) - before;
			double perNode = (double) used / Math.max(1, nodes);

			Evaluator legacyEv = evs[e] instanceof SimpleEvaluator ? new LegacySimpleEvaluator() : evs[e];
			LegacyNegamax legacyWarm = new LegacyNegamax(legacyEv, depths[e], true, 1 << 18, true, Negamax.ORDER_DEFAULT);
			for (State s : states) {
				legacyWarm.findBestMove(s, 1_000_000L, new ArrayList<Long>());
			}
			LegacyNegamax legacy = new LegacyNegamax(legacyEv, depths[e], true, 1 << 18, true, Negamax.ORDER_DEFAULT);
			long ln = 0;
			long lb = bytes(alloc, mx);
			for (State s : states) {
				legacy.findBestMove(s, 1_000_000L, new ArrayList<Long>());
				ln += legacy.lastNodeCount;
			}
			double legacyPerNode = (double) (bytes(alloc, mx) - lb) / Math.max(1, ln);

			boolean ok = perNode < MAX_BYTES_PER_NODE;
			System.out.printf("%s %-9s nuovo: %7.3f B/nodo (%d nodi, %d B in totale) | storico: %7.1f B/nodo%n", ok ? "OK  " : "FAIL", names[e], perNode, nodes, used,
					legacyPerNode);
			if (!ok) {
				failures++;
			}
		}
		if (failures == 0) {
			System.out.println("=== NO ALLOCATION CHECK OK ===");
		} else {
			System.out.println("=== NO ALLOCATION CHECK: " + failures + " FALLIMENTI ===");
			System.exit(1);
		}
	}

	private static long bytes(Method m, Object bean) throws Exception {
		return (Long) m.invoke(bean, Thread.currentThread().getId());
	}
}
