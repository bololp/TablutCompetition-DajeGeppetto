package it.unibo.ai.didattica.competition.tablut.client;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CachingEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ExactTranspositionTable;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Mcts;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NegamaxSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

import java.lang.reflect.Field;

/**
 * Verifica le funzioni di configurazione del client: margine di default, dimensione della TT, adattamento all'heap, e la scelta dell'algoritmo
 * di ricerca ({@code algo=negamax|mcts}), l'ordinamento delle mosse di default ({@code rec}) e la possibilita' di sceglierne un altro, la
 * dimensione di default e il tetto di memoria della cache delle valutazioni. Il comportamento del {@code main} reale e' verificato da
 * {@link ClientDefaultsCheck}.
 */
public class NegamaxClientArgsCheck {
	static int failures = 0;

	static void check(boolean ok, String msg) {
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	/** Il valore di {@code orderingMode} con cui e' stato costruito il Negamax del searcher (campo privato: reflection). */
	static int orderingModeOf(MoveSearcher searcher) throws Exception {
		final Field f = Negamax.class.getDeclaredField("orderingMode");
		f.setAccessible(true);
		return f.getInt(((NegamaxSearcher) searcher).negamax());
	}

	public static void main(String[] args) throws Exception {
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
		// ---- ordinamento di default (rec) e scelta esplicita di un altro
		check(TablutNegamaxClient.DEFAULT_ORDER_NAME.equals("rec"), "l'ordinamento di default del client e' rec");
		final int defOrder = OrderingSpec.parse(TablutNegamaxClient.DEFAULT_ORDER_NAME);
		check(defOrder == (Negamax.ORDER_DEFAULT | Negamax.ORDER_RECOMMENDED), "il default equivale a ORDER_DEFAULT | ORDER_RECOMMENDED");
		check(defOrder != Negamax.ORDER_DEFAULT, "il default NON e' piu' l'ordinamento storico");
		check(OrderingSpec.parse("base") == Negamax.ORDER_DEFAULT, "order=base seleziona ancora l'ordinamento storico");
		check(OrderingSpec.parse("rec") == defOrder, "order=rec e' il default");
		check(orderingModeOf(TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, defOrder, 12, false)) == defOrder, "il Negamax costruito con il default riceve rec");
		check(orderingModeOf(TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, OrderingSpec.parse("base"), 12, false)) == Negamax.ORDER_DEFAULT,
			"il Negamax costruito con order=base riceve l'ordinamento storico");
		final int custom = OrderingSpec.parse("wins+counter");
		check(custom != defOrder && custom != Negamax.ORDER_DEFAULT
			&& orderingModeOf(TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, custom, 12, false)) == custom, "un ordinamento esplicito diverso (wins+counter) viene rispettato");
		for (String name : new String[] { "none", "wins", "caps", "capsfirst", "counter", "kcarry", "rootsc", "all", "allfirst" }) {
			check(orderingModeOf(TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, OrderingSpec.parse(name), 12, false)) == OrderingSpec.parse(name), "order=" + name + " continua a funzionare");
		}
		// ---- cache delle valutazioni: dimensione di default e tetto di memoria
		check(TablutNegamaxClient.DEFAULT_EVAL_CACHE_LOG2 == 22, "cache delle valutazioni di default: 2^22 voci");
		check(CachingEvaluator.bytesFor(TablutNegamaxClient.DEFAULT_EVAL_CACHE_LOG2) == 128L << 20, "la cache di default occupa 128 MiB");
		check(CachingEvaluator.fitToHeap(TablutNegamaxClient.DEFAULT_EVAL_CACHE_LOG2, GiB) == 22, "la cache di default sta in un heap di 1 GiB");
		check(CachingEvaluator.fitToHeap(TablutNegamaxClient.DEFAULT_EVAL_CACHE_LOG2, GiB / 2) == 21, "in 512 MiB la cache scende a 2^21 (64 MiB = un ottavo)");
		// ---- wiring della cache: withEvalCache (usata da main) avvolge solo le reti, solo con Negamax, e rispetta il tetto di memoria
		final it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator mlp = it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator.randomInit(1);
		final it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator wrapped = TablutNegamaxClient.withEvalCache("negamax", mlp, 22, 2 * GiB);
		check(wrapped instanceof CachingEvaluator && ((CachingEvaluator) wrapped).inner() == mlp && ((CachingEvaluator) wrapped).log2Entries() == 22, "MLP + negamax: avvolta in una cache di 2^22 voci");
		check(TablutNegamaxClient.withEvalCache("negamax", mlp, 22, GiB / 2) instanceof CachingEvaluator
			&& ((CachingEvaluator) TablutNegamaxClient.withEvalCache("negamax", mlp, 22, GiB / 2)).log2Entries() == 21, "heap di 512 MiB: cache ridotta a 2^21");
		check(TablutNegamaxClient.withEvalCache("negamax", ev, 22, 2 * GiB) == ev, "SimpleEvaluator: mai avvolto");
		check(TablutNegamaxClient.withEvalCache("mcts", mlp, 22, 2 * GiB) == mlp, "algo=mcts: mai avvolto (ha la sua EvalCache)");
		check(TablutNegamaxClient.withEvalCache("negamax", mlp, 0, 2 * GiB) == mlp, "evcache=0: nessuna cache");
		// ---- chiave esatta (exactkey=): spenta di default; accesa, la TT di Negamax e' davvero la TT esatta e la cache legge la chiave dallo stato
		check(!TablutNegamaxClient.DEFAULT_EXACT_KEY, "exactkey e' spento di default");
		check(!(((NegamaxSearcher) TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, defOrder, 12, false)).negamax().transpositionTable() instanceof ExactTranspositionTable),
			"senza exactkey la TT e' quella a hash");
		check(!(((NegamaxSearcher) TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, defOrder, 12, false, false)).negamax().transpositionTable() instanceof ExactTranspositionTable),
			"exactKeys=false: TT a hash");
		check(((NegamaxSearcher) TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, defOrder, 12, false, true)).negamax().transpositionTable() instanceof ExactTranspositionTable,
			"exactKeys=true: la TT di Negamax e' una ExactTranspositionTable");
		check(((NegamaxSearcher) TablutNegamaxClient.buildSearcher("negamax", "", ev, 1, 0, defOrder, 12, true, true)).negamax().transpositionTable().isTwoSlot(), "exactKeys=true con due slot: TT a due slot");
		check(TablutNegamaxClient.ttBytes(26) == 24L << 26 && TablutNegamaxClient.ttBytes(26, false) == 24L << 26, "TT a hash: 24 byte per voce");
		check(TablutNegamaxClient.ttBytes(26, true) == 40L << 26, "TT esatta: 40 byte per voce");
		check(TablutNegamaxClient.fitTtToHeap(26, 2 * GiB) == 25 && TablutNegamaxClient.fitTtToHeap(26, 2 * GiB, false) == 25, "TT a hash: in 2 GiB di heap scende a 2^25");
		check(TablutNegamaxClient.fitTtToHeap(26, 2 * GiB, true) == 24, "TT esatta: in 2 GiB di heap scende a 2^24 (40 byte per voce)");
		check(((CachingEvaluator) TablutNegamaxClient.withEvalCache("negamax", mlp, 22, 2 * GiB, true)).isStateKey(), "exactkey: la cache legge la chiave dallo stato");
		check(!((CachingEvaluator) TablutNegamaxClient.withEvalCache("negamax", mlp, 22, 2 * GiB, false)).isStateKey()
			&& !((CachingEvaluator) TablutNegamaxClient.withEvalCache("negamax", mlp, 22, 2 * GiB)).isStateKey(), "senza exactkey la cache ricodifica la scacchiera");
		check(TablutNegamaxClient.withEvalCache("negamax", ev, 22, 2 * GiB, true) == ev, "exactkey: SimpleEvaluator comunque mai avvolto");
		if (failures == 0) {
			System.out.println("=== NEGAMAX CLIENT ARGS CHECK OK ===");
		} else {
			System.out.println("=== FALLITO: " + failures + " ===");
			System.exit(1);
		}
	}
}
