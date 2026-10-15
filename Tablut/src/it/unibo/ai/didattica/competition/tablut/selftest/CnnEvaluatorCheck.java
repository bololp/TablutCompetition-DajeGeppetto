package it.unibo.ai.didattica.competition.tablut.selftest;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Collaudo di {@link CnnEvaluator} come parte della ricerca: la catena completa State -> (feature, pedine) -> rete -> punteggio contro
 * un calcolo indipendente (oracolo double di {@link ConvNetCheck} + feature ricalcolate qui), uso da piu' thread, equivalenza
 * TT/no-TT dentro {@link Negamax} (la ricerca non deve poter distinguere l'evaluator "stateful": se lo fosse, i risultati
 * dipenderebbero dalla storia delle chiamate), Lazy SMP, e {@link EvaluatorLoader} (riconoscimento MLP/CNN, rifiuto di file incoerenti).
 */
public class CnnEvaluatorCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String msg) {
		checks++;
		System.out.println((ok ? "OK   - " : "FAIL - ") + msg);
		if (!ok) {
			failures++;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("=== CnnEvaluatorCheck ===");
		checkPipeline();
		checkThreads();
		checkSearch();
		checkLoader();
		check(constructorRejects(), "CnnEvaluator rifiuta una rete la cui nFeat non corrisponde ai gruppi di feature");
		System.out.println("=== " + (failures == 0 ? "CNN EVALUATOR CHECK OK" : "CNN EVALUATOR CHECK FALLITO (" + failures + ")") + " (" + checks + " verifiche) ===");
		if (failures > 0) {
			System.exit(1);
		}
	}

	static CnnEvaluator make(boolean hybrid, int[] conv, long seed) {
		int g = TablutFeatures.G_DEFAULT;
		ConvNet.Config c = new ConvNet.Config(conv, 2, hybrid ? TablutFeatures.columnsFor(g).length : 0, new int[] { 8 }, new int[] { 4 });
		return new CnnEvaluator(ConvNetCheck.randomNet(c, seed), hybrid ? g : 0);
	}

	/** evaluate(State) == scala * oracolo(scacchiera, feature calcolate indipendentemente, ruolo del turno). */
	static void checkPipeline() {
		List<State> pos = ConvNetCheck.positions(60, 3);
		for (boolean hybrid : new boolean[] { true, false }) {
			CnnEvaluator ev = make(hybrid, new int[] { 4, 3 }, 21);
			double[] Pd = ConvNetCheck.toDouble(ev.net.params());
			TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
			double[] full = new double[TablutFeatures.NUM_FEATURES];
			int[] cols = TablutFeatures.columnsFor(TablutFeatures.G_DEFAULT);
			double worst = 0;
			int white = 0, black = 0;
			for (State s : pos) {
				double[] f = new double[hybrid ? cols.length : 0];
				if (hybrid) {
					TablutFeatures.extract(s, ws, full, TablutFeatures.G_DEFAULT);
					for (int i = 0; i < cols.length; i++) {
						f[i] = (float) full[cols[i]];
					}
				}
				int role = s.getTurn() == Turn.BLACK ? 1 : 0;
				white += role == 0 ? 1 : 0;
				black += role;
				double want = ev.outputScale * ConvNetCheck.oracle(ev.net, Pd, ConvNetCheck.codes(s), f, role);
				worst = Math.max(worst, Math.abs(ev.evaluate(s) - want));
			}
			check(worst < 5e-3 && white > 10 && black > 10, "evaluate() == oracolo indipendente (" + (hybrid ? "ibrido" : "CNN puro") + ", " + pos.size()
					+ " posizioni: " + white + " col Bianco e " + black + " col Nero a muovere, errore max " + String.format("%.1e", worst) + " su scala "
					+ ev.outputScale + ")");
		}
		// stessa scacchiera, turno invertito: gli esiti usano head diverse
		CnnEvaluator ev = make(true, new int[] { 4 }, 5);
		State s = ConvNetCheck.positions(1, 77).get(0);
		double a = ev.evaluate(s);
		State t = s.clone();
		t.setTurn(s.getTurn() == Turn.WHITE ? Turn.BLACK : Turn.WHITE);
		check(a != ev.evaluate(t), "il ruolo (turno) seleziona la head: stessa scacchiera, turno opposto -> valore diverso");
		check(ev.evaluate(s) == a, "evaluate() e' deterministico sulla stessa posizione");
	}

	static void checkThreads() throws Exception {
		final CnnEvaluator ev = make(true, new int[] { 4 }, 8);
		final List<State> pos = ConvNetCheck.positions(50, 9);
		final double[] ref = new double[pos.size()];
		for (int i = 0; i < ref.length; i++) {
			ref[i] = ev.evaluate(pos.get(i));
		}
		ExecutorService pool = Executors.newFixedThreadPool(4);
		List<Future<Boolean>> fs = new ArrayList<>();
		for (int t = 0; t < 4; t++) {
			final int tt = t;
			fs.add(pool.submit(new Callable<Boolean>() {
				@Override
				public Boolean call() {
					Random r = new Random(tt);
					for (int it = 0; it < 4000; it++) {
						int i = r.nextInt(ref.length);
						if (ev.evaluate(pos.get(i)) != ref[i]) {
							return false;
						}
					}
					return true;
				}
			}));
		}
		boolean ok = true;
		for (Future<Boolean> f : fs) {
			ok &= f.get();
		}
		pool.shutdown();
		check(ok, "CnnEvaluator condiviso da 4 thread (16000 valutazioni, ThreadLocal): risultati identici al thread singolo");
	}

	static void checkSearch() {
		List<State> pos = ConvNetCheck.positions(6, 15);
		for (boolean hybrid : new boolean[] { true, false }) {
			boolean same = true, legal = true, smpOk = true;
			for (State s : pos) {
				CnnEvaluator evA = make(hybrid, new int[] { 4 }, 12), evB = make(hybrid, new int[] { 4 }, 12);
				Negamax withTT = new Negamax(evA, 3, true, 1 << 16).setThreads(1);
				Negamax noTT = new Negamax(evB, 3, false, 0).setThreads(1);
				Action m1 = withTT.findBestMove(s, 600000), m2 = noTT.findBestMove(s, 600000);
				same &= m1 != null && m1.toString().equals(m2.toString()) && Math.abs(withTT.lastScore - noTT.lastScore) < 1e-9;
				// Lazy SMP con il CnnEvaluator condiviso fra i thread
				Negamax smp = new Negamax(make(hybrid, new int[] { 4 }, 12), 3, true, 1 << 16).setThreads(4);
				Action m3 = smp.findBestMove(s, 600000);
				boolean isLegal = false;
				for (Action a : AshtonRules.generateMoves(s)) {
					isLegal |= m3 != null && a.toString().equals(m3.toString());
				}
				legal &= isLegal;
				smpOk &= smp.lastHelperErrors == 0 && !Double.isNaN(smp.lastScore) && smp.lastDepthReached >= 3;
			}
			check(same, "ricerca con TT == senza TT (stessa mossa, stesso punteggio) con " + (hybrid ? "CNN ibrido" : "CNN puro") + " su " + pos.size() + " posizioni");
			check(legal && smpOk, "Lazy SMP a 4 thread con evaluator CNN condiviso: mossa legale, nessun errore degli helper (" + (hybrid ? "ibrido" : "puro") + ")");
		}
	}

	static void checkLoader() throws IOException {
		File dir = Files.createTempDirectory("cnn_loader").toFile();
		CnnEvaluator hyb = make(true, new int[] { 4 }, 3), pure = make(false, new int[] { 4 }, 4);
		File fh = new File(dir, "hyb.txt"), fp = new File(dir, "pure.txt"), fm = new File(dir, "mlp.txt");
		hyb.save(fh.getPath());
		pure.save(fp.getPath());
		NeuralEvaluator.randomInit(1).save(fm.getPath());
		State s = ConvNetCheck.positions(1, 5).get(0);
		Evaluator lh = EvaluatorLoader.load(fh.getPath()), lp = EvaluatorLoader.load(fp.getPath()), lm = EvaluatorLoader.load(fm.getPath());
		check(lh instanceof CnnEvaluator && lp instanceof CnnEvaluator && lm instanceof NeuralEvaluator, "EvaluatorLoader: CNN -> CnnEvaluator, MLP -> NeuralEvaluator");
		check(lh.evaluate(s) == hyb.evaluate(s) && lp.evaluate(s) == pure.evaluate(s), "i pesi ricaricati danno esattamente lo stesso valore (ibrido e puro)");
		check(EvaluatorLoader.describe(lh).contains("CnnEvaluator") && EvaluatorLoader.describe(lm).contains("NeuralEvaluator"), "describe() distingue i due tipi");

		String text = new String(Files.readAllBytes(fh.toPath()), "UTF-8");
		check(loadFails(dir, text.replace("featureVersion " + TablutFeatures.VERSION, "featureVersion " + (TablutFeatures.VERSION - 1))),
				"CNN ibrido allenato con un'altra versione delle feature -> IOException (il client ripiega su SimpleEvaluator)");
		check(loadFails(dir, text.replaceFirst("featureGroups \\d+", "featureGroups " + TablutFeatures.G_ALL)) || TablutFeatures.G_ALL == TablutFeatures.G_DEFAULT,
				"CNN ibrido con gruppi di feature incoerenti con nFeat -> IOException");
		String ptext = new String(Files.readAllBytes(fp.toPath()), "UTF-8");
		File ok = new File(dir, "pure_v0.txt");
		Files.write(ok.toPath(), ptext.replace("featureVersion 0", "featureVersion 999").getBytes("UTF-8"));
		boolean loads = true;
		try {
			EvaluatorLoader.load(ok.getPath());
		} catch (IOException e) {
			loads = false;
		}
		check(loads, "il CNN PURO non dipende dalla versione delle feature: si carica comunque");
		boolean missing = false;
		try {
			EvaluatorLoader.load(new File(dir, "non_esiste.txt").getPath());
		} catch (IOException e) {
			missing = true;
		}
		check(missing, "file inesistente -> IOException");
	}

	static boolean loadFails(File dir, String content) throws IOException {
		File f = File.createTempFile("bad", ".txt", dir);
		Files.write(f.toPath(), content.getBytes("UTF-8"));
		try {
			EvaluatorLoader.load(f.getPath());
			return false;
		} catch (IOException e) {
			return true;
		}
	}

	static boolean constructorRejects() {
		ConvNet.Config c = new ConvNet.Config(new int[] { 4 }, 2, 5, new int[] { 8 }, new int[] { 4 }); // 5 feature != quelle dei gruppi di default
		try {
			new CnnEvaluator(new ConvNet(c, 1), TablutFeatures.G_DEFAULT);
			return false;
		} catch (IllegalArgumentException e) {
			return true;
		}
	}
}
