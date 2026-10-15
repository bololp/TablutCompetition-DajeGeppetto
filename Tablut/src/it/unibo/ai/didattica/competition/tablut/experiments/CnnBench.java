package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Misura il costo di UNA valutazione (end-to-end: feature a mano comprese) per diverse forme di CNN, con {@link NeuralEvaluator}
 * (MLP di default) e {@link SimpleEvaluator} come riferimento, sulle stesse posizioni di partita. E' lo strumento per scegliere le
 * dimensioni sulla macchina VERA: il costo per nodo domina la profondita' raggiunta (si veda README_CNN.md).
 * <p>
 * Uso: {@code CnnBench [secondiPerConfigurazione=2] [conv=8,8 reduce=2 dense=32 head=8 feat=1 ...]}. Senza altri argomenti
 * confronta una serie di forme predefinite. Con {@code conv=...} (e opzionalmente {@code reduce, dense, head, feat=0|1}) misura
 * una sola forma. L'ultima colonna stima i nodi/s di ricerca assumendo ~0,73 us di costo fisso per nodo (clone + hash,
 * misurato nel README_MLP_REDESIGN).
 */
public class CnnBench {

	static List<State> positions(int n, long seed) {
		Random rnd = new Random(seed);
		List<State> out = new ArrayList<>();
		while (out.size() < n) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			int steps = 4 + rnd.nextInt(70);
			for (int i = 0; i < steps; i++) {
				List<Action> mv = AshtonRules.generateMoves(s);
				if (mv.isEmpty() || AshtonRules.isTerminal(s)) {
					break;
				}
				s = AshtonRules.applyMove(s, mv.get(rnd.nextInt(mv.size())));
			}
			if (!AshtonRules.isTerminal(s) && !AshtonRules.generateMoves(s).isEmpty()) {
				out.add(s);
			}
		}
		return out;
	}

	static double measure(Evaluator ev, List<State> pos, double seconds) {
		double sink = 0;
		long warm = System.nanoTime() + (long) (seconds * 0.6e9);
		while (System.nanoTime() < warm) {
			for (State s : pos) {
				sink += ev.evaluate(s);
			}
		}
		long t0 = System.nanoTime(), end = t0 + (long) (seconds * 1e9);
		long n = 0;
		while (System.nanoTime() < end) {
			for (State s : pos) {
				sink += ev.evaluate(s);
			}
			n += pos.size();
		}
		double us = (System.nanoTime() - t0) / 1e3 / n;
		if (sink == 12345.678) {
			System.out.print(""); // impedisce l'eliminazione del codice morto
		}
		return us;
	}

	static void row(String name, Evaluator ev, long macs, int params, List<State> pos, double secs) {
		double us = measure(ev, pos, secs);
		double nps = 1e6 / (us + 0.73);
		System.out.printf("  %-34s %9s %8s %9.2f us/eval %9.0f nodi/s/thread%n", name, macs > 0 ? macs + " MAC" : "-",
				params > 0 ? params + " par" : "-", us, nps);
	}

	static CnnEvaluator cnn(int[] conv, int reduce, int[] dense, int[] head, boolean feat) {
		int g = TablutFeatures.G_DEFAULT;
		ConvNet.Config c = new ConvNet.Config(conv, reduce, feat ? TablutFeatures.columnsFor(g).length : 0, dense, head);
		return CnnEvaluator.randomInit(c, feat ? g : 0, 1);
	}

	public static void main(String[] a) {
		double secs = a.length > 0 && !a[0].contains("=") ? Double.parseDouble(a[0]) : 2.0;
		List<State> pos = positions(200, 7);
		System.out.println("=== CnnBench: costo di una valutazione (JVM " + System.getProperty("java.version") + ", " + Runtime.getRuntime().availableProcessors()
				+ " core), " + pos.size() + " posizioni, " + secs + " s per riga ===");
		row("SimpleEvaluator", new SimpleEvaluator(), 0, 0, pos, secs);
		NeuralEvaluator mlp = NeuralEvaluator.randomInit(1);
		row("MLP di default (41->24->8->1)", mlp, 1217, mlp.net.numParams(), pos, secs);

		java.util.Map<String, String> kv = new java.util.HashMap<>();
		for (String s : a) {
			int eq = s.indexOf('=');
			if (eq > 0) {
				kv.put(s.substring(0, eq), s.substring(eq + 1));
			}
		}
		if (kv.containsKey("conv")) {
			int[] conv = ConvNet.parseList(kv.get("conv"));
			int[] dense = ConvNet.parseList(kv.getOrDefault("dense", "32"));
			int[] head = ConvNet.parseList(kv.getOrDefault("head", "8"));
			int reduce = Integer.parseInt(kv.getOrDefault("reduce", "2"));
			boolean feat = !kv.getOrDefault("feat", "1").equals("0");
			CnnEvaluator e = cnn(conv, reduce, dense, head, feat);
			row("CNN " + e.net.config(), e, e.net.macsPerEval(), e.net.numParams(), pos, secs);
			return;
		}
		int[][] convs = { { 4 }, { 8 }, { 12 }, { 8, 8 }, { 12, 12 }, { 16, 16 }, { 8, 8, 8 } };
		for (boolean feat : new boolean[] { true, false }) {
			System.out.println(feat ? "-- ibridi (conv + feature a mano) --" : "-- CNN puri (solo scacchiera) --");
			for (int[] c : convs) {
				CnnEvaluator e = cnn(c, 2, new int[] { 32 }, new int[] { 8 }, feat);
				row("conv=" + ConvNet.parseList("").length + java.util.Arrays.toString(c).replace(" ", ""), e, e.net.macsPerEval(),
						e.net.numParams(), pos, secs);
			}
		}
	}
}
