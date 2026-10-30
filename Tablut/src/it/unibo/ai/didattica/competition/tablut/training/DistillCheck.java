package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.ConvNetCheck;
import it.unibo.ai.didattica.competition.tablut.training.CnnDistiller.SearchLoss;

/**
 * Verifica della feature "fine-tuning verso il valore di ricerca" ({@link SearchLabeler}, {@link DistillSet}, {@link CnnDistiller},
 * {@link HangingGate}, {@link DistillAcceptance}, logit di {@link ConvNet}, {@code init=} di {@link CnnPretrainer}).
 * Esecuzione (dalla cartella {@code Tablut}): {@code java -cp build:lib/... it.unibo...training.DistillCheck [sezione ...]}.
 * Senza argomenti esegue tutte le sezioni; con argomenti solo quelle nominate (serve al mutation testing, {@code tools/mutation_distill.py}).
 *
 * <h3>Sezioni</h3>
 * <ul>
 * <li><b>conv</b>: conversioni punteggio -&gt; bersaglio (clip, segno).</li>
 * <li><b>logit</b>: {@code ConvNet.Workspace.logit()} coerente con l'uscita (bit per bit), vivo a tanh satura; {@code backwardLogit}.</li>
 * <li><b>loss</b>: loss per campione, il suo gradiente (differenze finite scalari), proprieta' (minimo in atanh(y), il dominio logit non
 * "muore" a saturazione, il termine d'esito si annulla se non valido).</li>
 * <li><b>grad</b>: gradiente della loss di MINIBATCH completa (mista + ancoraggio + augmentation) rispetto a TUTTI i pesi, confrontato con
 * differenze finite di un oracolo in double.</li>
 * <li><b>labels</b>: segno e prospettiva delle etichette contro un minimax INDIPENDENTE (profondita' 1 e 2, su {@link AshtonRules}).</li>
 * <li><b>repro</b>: riproducibilita' (1 thread; indipendenza dal numero di thread e dall'ordine; nessuna fuga fra posizioni).</li>
 * <li><b>data</b>: esclusione dei terminali, split per partita, mix meta'/meta', partite dell'engine, file.</li>
 * <li><b>sym</b>: invarianza (regole, feature, valore di ricerca) sotto le 8 simmetrie e augmentation realmente applicata.</li>
 * <li><b>train</b>: ancoraggio, ciclo di iterazioni su dati veri, porta (i), determinismo, la rete di partenza non viene modificata.</li>
 * <li><b>gates</b>: logica di {@link HangingGate}, {@link DistillAcceptance} (soglie, tabella di verita', sostituzione con copia).</li>
 * <li><b>cli</b>: {@code CnnPretrainer init=} e {@code CnnDistiller} da riga di comando.</li>
 * </ul>
 */
public final class DistillCheck {

	private static int passed = 0, failed = 0;
	private static final List<String> failures = new ArrayList<>();
	private static final int NF = TablutFeatures.NUM_FEATURES;

	static void check(boolean ok, String msg) {
		if (ok) {
			passed++;
			System.out.println("OK   - " + msg);
		} else {
			failed++;
			failures.add(msg);
			System.out.println("FAIL - " + msg);
		}
	}

	private static boolean near(double a, double b, double tol) {
		return Math.abs(a - b) <= tol;
	}

	public static void main(String[] args) throws Exception {
		Set<String> want = new HashSet<>(Arrays.asList(args));
		System.out.println("=== DistillCheck ===");
		if (run(want, "conv")) {
			checkConversions();
		}
		if (run(want, "logit")) {
			checkLogitExposure();
		}
		if (run(want, "loss")) {
			checkSampleLoss();
		}
		if (run(want, "grad")) {
			checkBatchGradient(7);
		}
		if (run(want, "labels")) {
			checkLabelsVsIndependentMinimax();
		}
		if (run(want, "repro")) {
			checkReproducibility();
		}
		if (run(want, "data")) {
			checkData();
		}
		if (run(want, "sym")) {
			checkSymmetry();
		}
		if (run(want, "train")) {
			checkTraining();
		}
		if (run(want, "gates")) {
			checkGates();
		}
		if (run(want, "cli")) {
			checkCli();
		}
		if (failed == 0) {
			System.out.println("=== DISTILL CHECK OK (" + passed + " verifiche) ===");
		} else {
			System.out.println("=== DISTILL CHECK FALLITO: " + failed + " su " + (passed + failed) + " ===");
			for (String f : failures) {
				System.out.println("  - " + f);
			}
			System.exit(1);
		}
	}

	private static boolean run(Set<String> want, String name) {
		return want.isEmpty() || want.contains(name);
	}

	// ================================================================== fixture

	/** Rete ibrida piccola con le feature di G_DEFAULT, pesi casuali non nulli. */
	static CnnEvaluator smallHybrid(long seed) {
		return smallHybrid(seed, TablutFeatures.G_DEFAULT);
	}

	static CnnEvaluator smallHybrid(long seed, int groups) {
		int nf = TablutFeatures.columnsFor(groups).length;
		ConvNet net = ConvNetCheck.randomNet(new ConvNet.Config(new int[] { 4 }, 2, nf, new int[] { 16 }, new int[] { 8 }), seed);
		return new CnnEvaluator(net, groups);
	}

	/** Insieme sintetico a partire da stati: feature casuali nelle prime {@code NF} colonne, esiti casuali. */
	static DistillSet makeSet(List<State> ps, Random rnd) {
		int n = ps.size();
		byte[] cells = new byte[n * 81], role = new byte[n], oc = new byte[n], ov = new byte[n], so = new byte[n];
		int[] game = new int[n];
		boolean[] held = new boolean[n];
		float[] f = new float[n * NF];
		for (int i = 0; i < n; i++) {
			System.arraycopy(ConvNetCheck.codes(ps.get(i)), 0, cells, i * 81, 81);
			role[i] = (byte) NeuralEvaluator.roleOf(ps.get(i));
			oc[i] = (byte) (rnd.nextInt(3) - 1);
			ov[i] = (byte) (i % 3 == 2 ? 0 : 1);
			game[i] = i;
			for (int j = 0; j < NF; j++) {
				f[i * NF + j] = rnd.nextFloat();
			}
		}
		return new DistillSet(cells, role, oc, ov, game, so, held, f);
	}

	// ================================================================== conv

	static void checkConversions() {
		check(near(SearchLabeler.clippedTarget(0), 0, 1e-15), "conv: score 0 -> y = 0");
		check(near(SearchLabeler.clippedTarget(500), 0.5, 1e-15), "conv: 500 -> 0.5 (divisione per 1000)");
		check(near(SearchLabeler.clippedTarget(-250), -0.25, 1e-15), "conv: -250 -> -0.25");
		check(near(SearchLabeler.clippedTarget(979), 0.979, 1e-15) && near(SearchLabeler.clippedTarget(981), 0.98, 1e-15), "conv: il clip scatta oltre 0.98");
		check(near(SearchLabeler.clippedTarget(5_000_000), 0.98, 1e-15) && near(SearchLabeler.clippedTarget(-5_000_000), -0.98, 1e-15),
				"conv: vittoria/sconfitta forzata -> +-0.98 (simmetrico)");
		check(SearchLabeler.whiteScore(300, true) == 300 && SearchLabeler.whiteScore(300, false) == -300, "conv: lastScore e' di chi muove: Bianco invariato, Nero invertito");
		check(near(CnnDistiller.logitTarget(0.98), Math.log(99) / 2, 1e-12) && near(CnnDistiller.logitTarget(-0.5), -CnnDistiller.logitTarget(0.5), 1e-15),
				"conv: atanh(y) corretto e dispari");
	}

	// ================================================================== logit

	static void checkLogitExposure() {
		ConvNet.Config c = new ConvNet.Config(new int[] { 4, 4 }, 2, 4, new int[] { 8 }, new int[] { 4 });
		// ultimo strato con 4 ingressi (pari), con 5 (dispari: la coda del prodotto scalare conta) e rete SENZA feature ne' strati densi
		ConvNet.Config[] cfgs = { c, new ConvNet.Config(new int[] { 4, 4 }, 2, 4, new int[] { 8 }, new int[] { 5 }), new ConvNet.Config(new int[] { 3, 5 }, 0, 0, new int[] {}, new int[] {}) };
		boolean bitExact = true, matchesOracle = true;
		Random rnd = new Random(5);
		List<State> ps = ConvNetCheck.positions(6, 9);
		for (ConvNet.Config cf : cfgs) {
			ConvNet nn = ConvNetCheck.randomNet(cf, 3);
			ConvNet.Workspace w = nn.newWorkspace();
			for (State s : ps) {
				float[] f = ConvNetCheck.randomFeats(cf.nFeat, rnd);
				for (int role = 0; role < 2; role++) {
					w.setBoard(s);
					double out = nn.forward(w, f, role);
					bitExact &= w.output() == (float) Math.tanh(w.logit()) && out == w.output();
					double o = ConvNetCheck.oracle(nn, ConvNetCheck.toDouble(nn.params()), ConvNetCheck.codes(s), ConvNetCheck.toDouble(f), role);
					matchesOracle &= near(0.5 * Math.log((1 + o) / (1 - o)), w.logit(), 1e-4);
				}
			}
		}
		check(bitExact, "logit: output() == (float) tanh(logit()) bit per bit, come l'inferenza di prima (3 forme di rete)");
		check(matchesOracle, "logit: logit() coincide con atanh dell'oracolo in double (ingressi dell'ultimo strato pari e dispari, CNN pura)");
		ConvNet net = ConvNetCheck.randomNet(c, 3);
		ConvNet.Workspace ws = net.newWorkspace();

		// saturazione: z ~ 12 -> in float tanh(z) == 1.0 e atanh(output) sarebbe infinito; il logit resta utilizzabile
		final int role = 0, last = c.head.length;
		net.params()[net.headB[role][last]] += 12f;
		ws.setBoard(ps.get(0));
		float[] f = ConvNetCheck.randomFeats(4, rnd);
		net.forward(ws, f, role);
		check(ws.output() == 1.0f && ws.logit() > 8, "logit: a tanh satura output()==1 ma logit()=" + ws.logit() + " resta finito");
		float[] gA = new float[net.numParams()], gB = new float[net.numParams()];
		net.backwardLogit(ws, 0.7, gA);
		ws.setBoard(ps.get(0));
		net.forward(ws, f, role);
		net.backward(ws, 0.7, gB);
		check(near(gA[net.headB[role][last]], 0.7, 1e-6), "logit: backwardLogit passa dL/dz senza il fattore (1 - y^2): bias finale riceve esattamente dz");
		check(gB[net.headB[role][last]] == 0f, "logit: il backward sull'uscita tanh si annulla a saturazione (e' il motivo di backwardLogit)");

		// non saturo: backwardLogit(dz) == backward(dz / (1 - y^2))
		ConvNet net2 = ConvNetCheck.randomNet(c, 4);
		ConvNet.Workspace w2 = net2.newWorkspace();
		w2.setBoard(ps.get(1));
		f = ConvNetCheck.randomFeats(4, rnd);
		net2.forward(w2, f, 1);
		double y = w2.output();
		float[] g1 = new float[net2.numParams()], g2 = new float[net2.numParams()];
		net2.backwardLogit(w2, 0.4, g1);
		w2.setBoard(ps.get(1));
		net2.forward(w2, f, 1);
		net2.backward(w2, 0.4 / (1 - y * y), g2);
		double maxd = 0, maxv = 0;
		for (int i = 0; i < g1.length; i++) {
			maxd = Math.max(maxd, Math.abs(g1[i] - g2[i]));
			maxv = Math.max(maxv, Math.abs(g1[i]));
		}
		check(maxd <= 1e-5 * Math.max(1, maxv), "logit: backwardLogit(dz) == backward(dz/(1-y^2)) lontano dalla saturazione (diff max " + maxd + ")");
	}

	// ================================================================== loss per campione

	static void checkSampleLoss() {
		boolean dzOk = true;
		double worst = 0;
		for (SearchLoss kind : SearchLoss.values()) {
			for (double lambda : new double[] { 0.0, 0.2, 0.7, 1.0 }) {
				for (boolean ov : new boolean[] { true, false }) {
					for (double z : new double[] { -3.1, -0.4, 0.0, 0.9, 2.5 }) {
						for (double y : new double[] { -0.9, 0.1, 0.6 }) {
							for (double o : new double[] { -1, 0, 1 }) {
								double h = 1e-6;
								double num = (CnnDistiller.sampleLoss(z + h, y, o, ov, kind, lambda) - CnnDistiller.sampleLoss(z - h, y, o, ov, kind, lambda)) / (2 * h);
								double ana = CnnDistiller.sampleDz(z, y, o, ov, kind, lambda);
								worst = Math.max(worst, Math.abs(num - ana));
								dzOk &= Math.abs(num - ana) < 1e-6;
							}
						}
					}
				}
			}
		}
		check(dzOk, "loss: dL/dz analitico == differenza finita su tutte le combinazioni (kind, lambda, esito valido/no); errore max " + worst);

		// minimo della sola loss di ricerca in z = atanh(y)
		double y = 0.37, t = CnnDistiller.logitTarget(y);
		check(CnnDistiller.sampleLoss(t, y, 0, false, SearchLoss.LOGIT, 0) == 0 && CnnDistiller.sampleLoss(t + 0.1, y, 0, false, SearchLoss.LOGIT, 0) > 0.0099
				&& near(CnnDistiller.sampleDz(t, y, 0, false, SearchLoss.LOGIT, 0), 0, 1e-15), "loss: la loss di ricerca (logit) vale 0 e ha gradiente 0 in z = atanh(y); fuori e' (z - t)^2");
		check(near(CnnDistiller.sampleLoss(1.3, y, 0, false, SearchLoss.LOGIT, 0), Math.pow(1.3 - t, 2), 1e-12), "loss: L_ricerca = (z - atanh(y))^2 senza fattori");
		check(near(CnnDistiller.sampleLoss(1.3, y, 0, false, SearchLoss.TANH, 0), Math.pow(Math.tanh(1.3) - y, 2), 1e-12), "loss: ablation tanh = (tanh(z) - y)^2");

		// perche' il dominio logit: in saturazione il gradiente tanh sparisce, quello logit no
		double yb = 0.98, zs = 7.0;
		double dl = CnnDistiller.sampleDz(zs, yb, 0, false, SearchLoss.LOGIT, 0), dt = CnnDistiller.sampleDz(zs, yb, 0, false, SearchLoss.TANH, 0);
		check(Math.abs(dl) > 4 && Math.abs(dt) < 1e-3, "loss: con z=7 e y=0.98 il gradiente logit e' " + dl + ", quello su tanh " + dt + " (la ragione del dominio logit)");

		// mix
		double l = CnnDistiller.sampleLoss(0.8, 0.3, -1, true, SearchLoss.LOGIT, 0.2);
		double want = 0.8 * Math.pow(0.8 - CnnDistiller.logitTarget(0.3), 2) + 0.2 * Math.pow(Math.tanh(0.8) + 1, 2);
		check(near(l, want, 1e-12), "loss: L = (1 - lambda) L_ricerca + lambda L_esito con lambda=0.2");
		check(CnnDistiller.sampleLoss(0.8, 0.3, -1, false, SearchLoss.LOGIT, 0.2) == 0.8 * Math.pow(0.8 - CnnDistiller.logitTarget(0.3), 2),
				"loss: esito non valido -> termine d'esito omesso (peso 0, la ricerca non viene rinormalizzata)");
		check(CnnDistiller.sampleLoss(0.8, 0.3, -1, true, SearchLoss.LOGIT, 0) == CnnDistiller.sampleLoss(0.8, 0.3, 1, false, SearchLoss.LOGIT, 0)
				&& CnnDistiller.sampleLoss(0.8, 0.3, 1, true, SearchLoss.LOGIT, 1) == CnnDistiller.sampleLoss(0.8, 0.9, 1, true, SearchLoss.LOGIT, 1),
				"loss: lambda=0 ignora l'esito, lambda=1 ignora la ricerca");

		// ancoraggio
		float[] w = { 1f, 2f, 3f }, ref = { 1f, 1f, 5f };
		check(near(CnnDistiller.anchorPenalty(w, ref, 0.5), 0.5 * (0 + 1 + 4), 1e-12) && CnnDistiller.anchorPenalty(w, w, 9) == 0, "loss: penalita' di ancoraggio = mu * ||w - ref||^2");
		float[] g = new float[3];
		CnnDistiller.addAnchorGradient(w, ref, 0.5, 4, g);
		check(near(g[0], 0, 1e-9) && near(g[1], 2 * 0.5 * 4 * 1, 1e-6) && near(g[2], 2 * 0.5 * 4 * -2, 1e-6), "loss: gradiente d'ancoraggio = batch * 2 mu (w - ref) (verso il riferimento)");
	}

	// ================================================================== gradiente di minibatch

	/** Loss totale di minibatch (media dei campioni + ancoraggio) calcolata con l'ORACOLO in double. */
	static double oracleTotalLoss(ConvNet net, double[] P, DistillSet d, int[] cols, double[] y, List<State> ps, boolean augment, long seed, SearchLoss kind,
			double lambda, double[] refD, double mu) {
		double s = 0;
		for (int i = 0; i < d.n; i++) {
			State st = ps.get(i);
			int sym = augment ? CnnPretrainer.symOf(seed, 0, i) : 0;
			double[] f = new double[cols.length];
			for (int j = 0; j < cols.length; j++) {
				f[j] = d.feats[i * NF + cols[j]];
			}
			double out = ConvNetCheck.oracle(net, P, ConvNetCheck.codes(BoardSymmetry.transform(st, sym)), f, d.role[i]);
			double z = 0.5 * Math.log((1 + out) / (1 - out));
			s += CnnDistiller.sampleLoss(z, y[i], d.outcome[i], d.outValid[i] == 1, kind, lambda);
		}
		s /= d.n;
		for (int i = 0; i < P.length; i++) {
			s += mu * (P[i] - refD[i]) * (P[i] - refD[i]);
		}
		return s;
	}

	static void checkBatchGradient(long seed) throws Exception {
		ConvNet.Config cfgA = new ConvNet.Config(new int[] { 4, 4 }, 2, 4, new int[] { 8 }, new int[] { 4 });
		ConvNet.Config cfgB = new ConvNet.Config(new int[] { 3, 5 }, 0, 0, new int[] {}, new int[] {});
		// {config, kind(0 logit/1 tanh), augment, passo sui parametri}
		int[][] plan = { { 0, 0, 1, 1 }, { 0, 0, 0, 3 }, { 0, 1, 0, 3 }, { 1, 0, 1, 2 } };
		for (int k = 0; k < plan.length; k++) {
			ConvNet.Config cfg = plan[k][0] == 0 ? cfgA : cfgB;
			SearchLoss kind = plan[k][1] == 0 ? SearchLoss.LOGIT : SearchLoss.TANH;
			boolean aug = plan[k][2] == 1;
			int stride = plan[k][3];
			Random rnd = new Random(seed * 31 + k);
			ConvNet net = ConvNetCheck.randomNet(cfg, seed + k);
			List<State> ps = ConvNetCheck.positions(3, seed + 50 + k);
			DistillSet d = makeSet(ps, rnd);
			int[] cols = new int[cfg.nFeat];
			for (int j = 0; j < cols.length; j++) {
				cols[j] = j;
			}
			double[] y = new double[d.n];
			for (int i = 0; i < d.n; i++) {
				y[i] = (rnd.nextDouble() * 2 - 1) * 0.98;
			}
			float[] ref = net.params().clone();
			for (int i = 0; i < ref.length; i++) {
				ref[i] += (float) (rnd.nextGaussian() * 0.02);
			}
			double mu = 0.05, lambda = 0.2;
			long gseed = 4242 + k;
			float[] g = new float[net.numParams()];
			CnnDistiller.batchGradient(null, new CnnDistiller.Worker[] { new CnnDistiller.Worker(net) }, d, cols, y, new int[] { 0, 1, 2 }, 0, 3, 0, gseed, aug, kind,
					lambda, g);
			CnnDistiller.addAnchorGradient(net.params(), ref, mu, 3, g);
			for (int i = 0; i < g.length; i++) {
				g[i] /= 3f;
			}
			double[] Pd = ConvNetCheck.toDouble(net.params()), refD = ConvNetCheck.toDouble(ref);
			long checked = 0, skipped = 0, nonZero = 0;
			double worst = 0;
			boolean ok = true;
			for (int i = 0; i < Pd.length && ok; i += stride) {
				double orig = Pd[i];
				double[] est = new double[2], hs = { 1e-5, 1e-7 };
				for (int q = 0; q < 2; q++) {
					Pd[i] = orig + hs[q];
					double up = oracleTotalLoss(net, Pd, d, cols, y, ps, aug, gseed, kind, lambda, refD, mu);
					Pd[i] = orig - hs[q];
					double dn = oracleTotalLoss(net, Pd, d, cols, y, ps, aug, gseed, kind, lambda, refD, mu);
					est[q] = (up - dn) / (2 * hs[q]);
				}
				Pd[i] = orig;
				if (Math.abs(est[0] - est[1]) > 1e-5 + 1e-4 * Math.abs(est[1])) {
					skipped++; // angolo della leaky-ReLU nell'intervallo
					continue;
				}
				double num = est[1], err = Math.abs(g[i] - num);
				worst = Math.max(worst, err);
				checked++;
				if (g[i] != 0) {
					nonZero++;
				}
				if (err > 2e-4 + 2e-3 * Math.abs(num)) {
					check(false, "grad: parametro " + i + " analitico " + g[i] + " vs differenze finite " + num + " (piano " + k + ")");
					ok = false;
				}
			}
			if (ok) {
				check(checked > 0 && skipped * 20 < checked && nonZero * 3 > checked,
						"grad: loss di minibatch (" + kind + ", augmentation=" + aug + ", mista+ancoraggio) == differenze finite su " + checked + " parametri (angoli scartati "
								+ skipped + ", errore max " + String.format("%.2e", worst) + ", " + cfg.conv.length + " livelli, nFeat=" + cfg.nFeat + ")");
			}
		}
	}

	// ================================================================== segno e prospettiva delle etichette

	static final double BIG = 1_000_000;

	/** Minimax INDEPENDENTE (nessun codice di Negamax): valore dal punto di vista del Bianco a profondita' fissa, sulle regole di AshtonRules. */
	static double refWhite(State s, int depth, Evaluator ev) {
		if (AshtonRules.isTerminal(s)) {
			return s.getTurn() == Turn.WHITEWIN ? BIG : s.getTurn() == Turn.BLACKWIN ? -BIG : 0;
		}
		boolean whiteMoves = s.getTurn() == Turn.WHITE;
		List<Action> mv = AshtonRules.generateMoves(s);
		if (mv.isEmpty()) {
			return whiteMoves ? -BIG : BIG; // chi non ha mosse perde
		}
		if (depth == 0) {
			return ev.evaluate(s);
		}
		double best = whiteMoves ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		for (Action a : mv) {
			double v = refWhite(AshtonRules.applyMove(s, a), depth - 1, ev);
			best = whiteMoves ? Math.max(best, v) : Math.min(best, v);
		}
		return best;
	}

	static void checkLabelsVsIndependentMinimax() {
		Evaluator[] evs = { new SimpleEvaluator(), smallHybrid(11) };
		String[] names = { "SimpleEvaluator", "CnnEvaluator" };
		int[] depths = { 1, 2 };
		for (int e = 0; e < evs.length; e++) {
			for (int D : depths) {
				int n = D == 1 ? 60 : (e == 0 ? 36 : 14);
				List<State> ps = ConvNetCheck.positions(n, 100 + 7 * e + D);
				int mismatches = 0, whiteMoves = 0, blackMoves = 0, blackInformative = 0, whiteInformative = 0, forced = 0;
				String first = "";
				for (State s : ps) {
					double ref = refWhite(s, D, evs[e]);
					double refY = SearchLabeler.clippedTarget(ref);
					SearchLabeler.Result r = SearchLabeler.label(evs[e], FastState.fromState(s), D, 12);
					boolean good = near(r.y, refY, 1e-9) && (Math.abs(ref) >= 9e5 || near(r.whiteScore, ref, 1e-6 * Math.max(1, Math.abs(ref))));
					if (!good) {
						mismatches++;
						if (first.isEmpty()) {
							first = "ref " + ref + " vs " + r.whiteScore + " (y " + refY + " vs " + r.y + ", " + s.getTurn() + ")";
						}
					}
					boolean white = s.getTurn() == Turn.WHITE;
					if (white) {
						whiteMoves++;
					} else {
						blackMoves++;
					}
					if (Math.abs(refY) > 0.02 && Math.abs(refY) < 0.98) {
						if (white) {
							whiteInformative++;
						} else {
							blackInformative++;
						}
					}
					if (Math.abs(ref) >= 9e5) {
						forced++;
					}
				}
				check(mismatches == 0, "labels: " + names[e] + " D=" + D + ": etichette == minimax indipendente (AshtonRules) su " + n + " posizioni (" + mismatches + " diverse" + (first.isEmpty() ? "" : "; prima: " + first)
						+ ")");
				check(whiteMoves >= 5 && blackMoves >= 5 && whiteInformative >= 3 && blackInformative >= 3, "labels: " + names[e] + " D=" + D + ": il test discrimina il segno (muove il Bianco " + whiteMoves + ", il Nero "
						+ blackMoves + ", con valore non banale " + whiteInformative + "/" + blackInformative + ", forzate " + forced + ")");
			}
		}
		// esiti forzati non ambigui: il segno del bersaglio e' quello del VINCITORE, qualunque sia chi muove
		State win = emptyBoard();
		win.getBoard()[1][1] = Pawn.KING; // il Re raggiunge il bordo (0,1) fuori dagli accampamenti
		win.setTurn(Turn.WHITE);
		SearchLabeler.Result rw = SearchLabeler.label(new SimpleEvaluator(), FastState.fromState(win), 2, 12);
		check(rw.y == 0.98 && rw.forced(), "labels: Bianco a muovere con Re a una mossa dal bordo -> y = +0.98 (vittoria forzata del Bianco), y=" + rw.y);
		State lose = emptyBoard();
		Pawn[][] b = lose.getBoard();
		b[4][4] = Pawn.KING;
		b[3][4] = Pawn.BLACK;
		b[5][4] = Pawn.BLACK;
		b[4][3] = Pawn.BLACK;
		b[4][7] = Pawn.BLACK; // arriva in (4,5) e chiude i 4 lati del Re sul trono
		lose.setTurn(Turn.BLACK);
		SearchLabeler.Result rl = SearchLabeler.label(new SimpleEvaluator(), FastState.fromState(lose), 2, 12);
		check(rl.y == -0.98 && rl.whiteScore < -9e5, "labels: Nero a muovere che cattura il Re -> y = -0.98 dal punto di vista del Bianco (punteggio di chi muove e' positivo, il Bianco vede il segno opposto), y=" + rl.y);
	}

	static State emptyBoard() {
		State s = new StateTablut();
		for (int r = 0; r < 9; r++) {
			Arrays.fill(s.getBoard()[r], Pawn.EMPTY);
		}
		return s;
	}

	// ================================================================== riproducibilita'

	static void checkReproducibility() throws Exception {
		CnnEvaluator ev = smallHybrid(21);
		List<State> ps = ConvNetCheck.positions(10, 321);
		DistillSet d = makeSet(ps, new Random(1));
		boolean same = true;
		double[] first = new double[ps.size()];
		for (int rep = 0; rep < 2; rep++) {
			for (int i = 0; i < ps.size(); i++) {
				double y = SearchLabeler.label(ev, FastState.fromState(ps.get(i)), 3, 14).y;
				if (rep == 0) {
					first[i] = y;
				} else {
					same &= Double.doubleToLongBits(y) == Double.doubleToLongBits(first[i]);
				}
			}
		}
		check(same, "repro: stessa posizione, stessa rete, 1 thread -> etichetta identica bit per bit (D=3, due esecuzioni)");

		int[] idx = new int[ps.size()];
		for (int i = 0; i < idx.length; i++) {
			idx[i] = i;
		}
		SearchLabeler.Labels l1 = SearchLabeler.labelAll(ev, d.cells, d.role, idx, 3, 1, 14, null);
		SearchLabeler.Labels l3 = SearchLabeler.labelAll(ev, d.cells, d.role, idx, 3, 3, 14, null);
		check(Arrays.equals(l1.y, l3.y) && Arrays.equals(l1.y, first), "repro: labelAll con 1 e 3 thread (parallelismo TRA posizioni) == etichette singole, uguali bit per bit");
		int[] rev = new int[idx.length];
		for (int i = 0; i < rev.length; i++) {
			rev[i] = idx[idx.length - 1 - i];
		}
		SearchLabeler.Labels lr = SearchLabeler.labelAll(ev, d.cells, d.role, rev, 3, 1, 14, null);
		boolean orderFree = true;
		for (int k = 0; k < rev.length; k++) {
			orderFree &= lr.y[k] == l1.y[rev[k]];
		}
		check(orderFree, "repro: l'etichetta non dipende dall'ordine di elaborazione (nessuna TT/history condivisa fra posizioni)");
		// profondita' diverse danno etichette diverse (il test non e' vuoto) e D=3 raggiunge davvero 3
		SearchLabeler.Labels l2 = SearchLabeler.labelAll(ev, d.cells, d.role, idx, 2, 1, 14, null);
		int differ = 0, reached3 = 0;
		for (int i = 0; i < idx.length; i++) {
			differ += l2.y[i] != l1.y[i] ? 1 : 0;
			reached3 += l1.depthReached[i] == 3 ? 1 : 0;
		}
		check(differ >= 3 && reached3 >= idx.length - l1.forced, "repro: D=2 e D=3 danno etichette diverse (" + differ + "/" + idx.length + ") e la profondita' raggiunta e' D (tranne vittorie forzate)");
		boolean threw = false;
		try {
			SearchLabeler.label(ev, SearchLabeler.fastStateFromCodes(boxedKing(), 0, 0), 2, 12);
		} catch (IllegalArgumentException ex) {
			threw = true;
		}
		check(threw, "repro: una posizione terminale e' rifiutata (nessuna etichetta)");
	}

	static byte[] boxedKing() {
		byte[] c = new byte[81];
		c[0] = 3;
		c[1] = 2;
		c[9] = 2;
		return c;
	}

	// ================================================================== dati

	static void checkData() throws Exception {
		CnnPretrainer.Dataset ds = CnnPretrainer.generate(80, 3, 1, null);
		// terminali: aggiungo due righe fittizie non giocabili
		byte[] c = new byte[(ds.n + 2) * 81], r = new byte[ds.n + 2], oc = new byte[ds.n + 2];
		float[] f = new float[(ds.n + 2) * NF];
		int[] g = new int[ds.n + 2];
		System.arraycopy(ds.cells, 0, c, 0, ds.n * 81);
		System.arraycopy(ds.role, 0, r, 0, ds.n);
		System.arraycopy(ds.outcome, 0, oc, 0, ds.n);
		System.arraycopy(ds.feats, 0, f, 0, ds.n * NF);
		System.arraycopy(ds.game, 0, g, 0, ds.n);
		System.arraycopy(boxedKing(), 0, c, ds.n * 81, 81); // il Bianco a muovere non ha mosse
		byte[] blackBoxed = new byte[81];
		blackBoxed[0] = 2;
		blackBoxed[1] = 1;
		blackBoxed[9] = 1;
		System.arraycopy(blackBoxed, 0, c, (ds.n + 1) * 81, 81); // il Nero a muovere non ha mosse
		r[ds.n + 1] = 1;
		g[ds.n] = g[ds.n + 1] = 9999;
		CnnPretrainer.Dataset withTerminal = new CnnPretrainer.Dataset(c, r, f, oc, g);
		check(SearchLabeler.isTerminal(SearchLabeler.fastStateFromCodes(boxedKing(), 0, 0)) && SearchLabeler.isTerminal(SearchLabeler.fastStateFromCodes(blackBoxed, 0, 1)),
				"data: chi muove senza mosse legali e' terminale (Bianco e Nero)");
		DistillSet all = DistillSet.fromDataset(withTerminal, 1, 10);
		boolean noTerminal = true;
		for (int i = 0; i < all.n; i++) {
			noTerminal &= !SearchLabeler.isTerminal(SearchLabeler.fastStateFromCodes(all.cells, i * 81, all.role[i]));
		}
		check(all.n == ds.n && noTerminal, "data: fromDataset esclude le posizioni terminali (" + all.n + " di " + (ds.n + 2) + " righe)");

		// split per partita
		check(all.splitIsDisjointByGame(), "data: nessuna partita compare sia nel train sia nell'held-out");
		int held = all.indices(true).length, train = all.indices(false).length;
		check(held > 0 && train > held && held + train == all.n, "data: train " + train + " / held-out " + held + " non vuoti e complementari");
		int h = 0;
		for (int k = 0; k < 5000; k++) {
			h += DistillSet.isHeldOut(k, 7, 10) ? 1 : 0;
		}
		check(h > 400 && h < 600, "data: isHeldOut assegna ~10% delle partite (" + h + "/5000)");
		boolean det = true, saltMatters = false;
		for (int k = 0; k < 200; k++) {
			det &= DistillSet.isHeldOut(k, 7, 10) == DistillSet.isHeldOut(k, 7, 10);
			saltMatters |= DistillSet.isHeldOut(k, 7, 10) != DistillSet.isHeldOut(k, 8, 10);
		}
		check(det && saltMatters && !DistillSet.isHeldOut(5, 1, 0) && DistillSet.isHeldOut(5, 1, 100), "data: isHeldOut e' deterministico, dipende dal sale e rispetta 0% / 100%");
		boolean sameGameSameSide = true;
		Map<Integer, Boolean> side = new HashMap<>();
		for (int i = 0; i < all.n; i++) {
			Boolean prev = side.put(all.game[i], all.heldOut[i]);
			sameGameSameSide &= prev == null || prev == all.heldOut[i];
		}
		check(sameGameSameSide, "data: tutte le posizioni di una partita stanno dalla stessa parte");

		// partite dell'engine
		DistillSet eng = DistillSet.generateEngineGames(new SimpleEvaluator(), 4, 4, 0.03, 5, 1, 5, 25, null);
		boolean engOk = eng.n > 100, keys = true, nonTerm = true, srcOk = true;
		Map<Integer, Integer> outc = new HashMap<>();
		boolean outcomeConsistent = true;
		for (int i = 0; i < eng.n; i++) {
			keys &= eng.game[i] >= DistillSet.ENGINE_BASE;
			srcOk &= eng.source[i] == DistillSet.SRC_ENGINE;
			nonTerm &= !SearchLabeler.isTerminal(SearchLabeler.fastStateFromCodes(eng.cells, i * 81, eng.role[i]));
			Integer prev = outc.put(eng.game[i], (int) eng.outcome[i]);
			outcomeConsistent &= prev == null || prev == eng.outcome[i];
		}
		check(engOk && keys && nonTerm && srcOk && outcomeConsistent && eng.splitIsDisjointByGame(),
				"data: partite dell'engine: " + eng.n + " posizioni non terminali, chiavi >= ENGINE_BASE, esito coerente per partita, split per partita");

		// segno dell'esito: mappatura INDIPENDENTE delle etichette di fine partita (WW/NOMOVES_WW -> +1, BW/NOMOVES_BW -> -1, il resto 0)
		int signChecked = 0, signBad = 0, decisive = 0, roleBad = 0;
		for (int gi = 0; gi < 8; gi++) {
			DistillSet.Builder gb = DistillSet.playEngineGame(new SimpleEvaluator(), 3, 0.05, 100 + gi, DistillSet.ENGINE_BASE + gi);
			int want = gb.endLabel.equals(GameTracker.LABEL_WHITE_WIN) || gb.endLabel.equals(GameTracker.LABEL_NOMOVES_WHITE_WIN) ? 1
					: gb.endLabel.equals(GameTracker.LABEL_BLACK_WIN) || gb.endLabel.equals(GameTracker.LABEL_NOMOVES_BLACK_WIN) ? -1 : 0;
			decisive += want != 0 ? 1 : 0;
			int k = 0;
			for (int[] m : gb.meta) {
				signChecked++;
				signBad += m[1] != want ? 1 : 0;
				roleBad += m[0] != k % 2 ? 1 : 0; // dal ply 4 (pari) in poi tocca a Bianco, Nero, Bianco, ...
				k++;
			}
		}
		check(roleBad == 0, "data: il ruolo registrato nelle partite dell'engine alterna Bianco/Nero (head giusta per ogni posizione)");
		check(signChecked > 100 && signBad == 0 && decisive >= 2, "data: l'esito registrato nelle partite dell'engine ha il segno del vincitore (" + signChecked + " posizioni, " + decisive + "/8 partite decise)");

		// mix meta'/meta'
		int trainEngine = eng.indices(false).length;
		DistillSet allTrainOnly = all;
		int want = Math.min(200, 2 * Math.min(trainEngine, all.indices(false).length));
		String[] rep = new String[1];
		DistillSet mix = DistillSet.sample(allTrainOnly, eng, want, 20, 9, rep);
		int[] tr = mix.indices(false);
		int nd = 0, ne = 0;
		for (int i : tr) {
			if (mix.source[i] == DistillSet.SRC_DATASET) {
				nd++;
			} else {
				ne++;
			}
		}
		check(tr.length == want && Math.abs(nd - ne) <= 1 && mix.splitIsDisjointByGame(), "data: sample prende meta' dataset / meta' engine (" + nd + " + " + ne + " = " + tr.length + ") e il split per partita resta valido");
		// una sorgente scarsa: il totale richiesto viene comunque raggiunto dall'altra, e la composizione e' dichiarata
		// richiedo piu' del doppio delle posizioni di train dell'engine: la sua meta' non basta e il resto deve venire dal dataset
		DistillSet skew = DistillSet.sample(allTrainOnly, eng, 2 * trainEngine + 60, 10, 9, rep);
		int ne2 = 0;
		for (int i : skew.indices(false)) {
			ne2 += skew.source[i] == DistillSet.SRC_ENGINE ? 1 : 0;
		}
		check(skew.indices(false).length == 2 * trainEngine + 60 && ne2 == trainEngine && rep[0].contains("da engine"),
				"data: se l'engine e' scarso il resto viene dal dataset e il report lo dichiara (" + rep[0] + ")");
		DistillSet onlyA = DistillSet.sample(all, null, 50, 10, 9, rep);
		check(onlyA.indices(false).length == 50, "data: una sola sorgente (engine assente) -> si usa quella disponibile");
		// le righe campionate esistono nelle sorgenti con lo stesso lato
		Set<String> keysA = new HashSet<>();
		for (int i = 0; i < all.n; i++) {
			keysA.add(all.game[i] + ":" + all.heldOut[i] + ":" + Arrays.hashCode(Arrays.copyOfRange(all.cells, i * 81, i * 81 + 81)) + ":" + all.role[i] + ":" + all.outcome[i]);
		}
		boolean preserved = true;
		for (int i = 0; i < onlyA.n; i++) {
			preserved &= keysA.contains(onlyA.game[i] + ":" + onlyA.heldOut[i] + ":" + Arrays.hashCode(Arrays.copyOfRange(onlyA.cells, i * 81, i * 81 + 81)) + ":" + onlyA.role[i] + ":" + onlyA.outcome[i]);
		}
		check(preserved, "data: il campionamento conserva scacchiera, turno, esito, partita e lato train/held-out");

		// file
		File tmp = File.createTempFile("distill", ".gz");
		tmp.deleteOnExit();
		mix.save(tmp.getPath());
		DistillSet back = DistillSet.load(tmp.getPath());
		check(back.n == mix.n && Arrays.equals(back.cells, mix.cells) && Arrays.equals(back.role, mix.role) && Arrays.equals(back.outcome, mix.outcome)
				&& Arrays.equals(back.outValid, mix.outValid) && Arrays.equals(back.game, mix.game) && Arrays.equals(back.heldOut, mix.heldOut)
				&& Arrays.equals(back.source, mix.source) && Arrays.equals(back.feats, mix.feats), "data: salvataggio/caricamento di un DistillSet senza perdite");
		boolean rejected = false;
		try {
			DistillSet.load(File.createTempFile("nogz", ".gz").getPath());
		} catch (IOException ex) {
			rejected = true;
		}
		check(rejected, "data: un file che non e' un DistillSet viene rifiutato");
	}

	// ================================================================== simmetrie

	static void checkSymmetry() {
		List<State> ps = ConvNetCheck.positions(25, 777);
		int badRules = 0, badFeat = 0;
		double worstFeat = 0;
		for (State s : ps) {
			double[] f0 = TablutFeatures.extract(s);
			int m0 = AshtonRules.generateMoves(s).size();
			for (int sym = 0; sym < 8; sym++) {
				State t = BoardSymmetry.transform(s, sym);
				if (AshtonRules.generateMoves(t).size() != m0 || AshtonRules.isTerminal(t) != AshtonRules.isTerminal(s)) {
					badRules++;
				}
				double[] f = TablutFeatures.extract(t);
				for (int j = 0; j < NF; j++) {
					double d = Math.abs(f[j] - f0[j]);
					worstFeat = Math.max(worstFeat, d);
					if (d > 1e-9) {
						badFeat++;
					}
				}
			}
		}
		check(badRules == 0, "sym: le 8 simmetrie conservano numero di mosse legali e terminalita' (regole del gioco invarianti) su 25 posizioni");
		check(badFeat == 0, "sym: tutte le " + NF + " feature a mano sono invarianti sotto le 8 simmetrie (scarto max " + worstFeat + "): si possono riusare per la posizione trasformata");

		// il valore di ricerca con un evaluator simmetrico e' invariante: l'etichetta di una posizione vale per tutte le sue immagini
		Evaluator ev = new SimpleEvaluator();
		int badLabel = 0, tested = 0;
		for (int i = 0; i < 6; i++) {
			State s = ps.get(i);
			double y0 = SearchLabeler.label(ev, FastState.fromState(s), 2, 12).y;
			for (int sym = 1; sym < 8; sym++) {
				double y = SearchLabeler.label(ev, FastState.fromState(BoardSymmetry.transform(s, sym)), 2, 12).y;
				tested++;
				if (!near(y, y0, 1e-9)) {
					badLabel++;
				}
			}
		}
		check(badLabel == 0, "sym: con un evaluator simmetrico (SimpleEvaluator) l'etichetta di ricerca D=2 e' identica sulle 8 immagini (" + tested + " confronti)");

		// l'augmentation usa davvero la simmetria sulla scacchiera e conserva bersaglio, esito e feature
		ConvNet.Config cfg = new ConvNet.Config(new int[] { 4 }, 2, 4, new int[] { 8 }, new int[] { 4 });
		ConvNet net = ConvNetCheck.randomNet(cfg, 5);
		List<State> two = ConvNetCheck.positions(2, 31);
		DistillSet d = makeSet(two, new Random(2));
		int[] cols = { 0, 1, 2, 3 };
		double[] y = { 0.4, -0.3 };
		long seed = 1;
		while (CnnPretrainer.symOf(seed, 0, 0) == 0 || CnnPretrainer.symOf(seed, 0, 1) == 0) {
			seed++;
		}
		float[] ga = new float[net.numParams()], gm = new float[net.numParams()];
		CnnDistiller.Worker w = new CnnDistiller.Worker(net);
		w.run(d, cols, y, new int[] { 0, 1 }, 0, 2, 0, seed, true, SearchLoss.LOGIT, 0.2);
		System.arraycopy(w.grad, 0, ga, 0, ga.length);
		double lossAug = w.loss;
		// riferimento manuale: scacchiera trasformata con BoardSymmetry, stesse feature/bersaglio/esito
		ConvNet.Workspace ws = net.newWorkspace();
		double lossMan = 0;
		for (int i = 0; i < 2; i++) {
			ws.setBoard(BoardSymmetry.transform(two.get(i), CnnPretrainer.symOf(seed, 0, i)));
			float[] f = { d.feats[i * NF], d.feats[i * NF + 1], d.feats[i * NF + 2], d.feats[i * NF + 3] };
			net.forward(ws, f, d.role[i]);
			double z = ws.logit();
			lossMan += CnnDistiller.sampleLoss(z, y[i], d.outcome[i], d.outValid[i] == 1, SearchLoss.LOGIT, 0.2);
			net.backwardLogit(ws, CnnDistiller.sampleDz(z, y[i], d.outcome[i], d.outValid[i] == 1, SearchLoss.LOGIT, 0.2), gm);
		}
		double maxd = 0;
		for (int i = 0; i < ga.length; i++) {
			maxd = Math.max(maxd, Math.abs(ga[i] - gm[i]));
		}
		check(maxd < 1e-5 && near(lossAug, lossMan, 1e-6), "sym: con augment=true il gradiente e' quello della scacchiera TRASFORMATA (diff " + maxd + ")");
		CnnDistiller.Worker w0 = new CnnDistiller.Worker(net);
		w0.run(d, cols, y, new int[] { 0, 1 }, 0, 2, 0, seed, false, SearchLoss.LOGIT, 0.2);
		double diffNoAug = 0;
		for (int i = 0; i < ga.length; i++) {
			diffNoAug = Math.max(diffNoAug, Math.abs(ga[i] - w0.grad[i]));
		}
		check(diffNoAug > 1e-4, "sym: con augment=false il gradiente e' diverso (la trasformazione non e' un no-op su queste posizioni)");
	}

	// ================================================================== addestramento

	static void checkTraining() throws Exception {
		CnnPretrainer.Dataset ds = CnnPretrainer.generate(120, 11, 1, null);
		DistillSet all = DistillSet.fromDataset(ds, 3, 12);
		DistillSet set = DistillSet.sample(all, null, 640, 160, 4, new String[1]);
		CnnEvaluator base = smallHybrid(77);
		float[] baseParams = base.net.params().clone();
		int[] cols = CnnDistiller.colsFor(base.net, base.featureGroups);
		int[] held = set.indices(true), train = set.indices(false);
		CnnDistiller.Params P = new CnnDistiller.Params();
		P.iters = 1;
		P.depths = new int[] { 1 };
		P.epochs = 3;
		P.lr = 3e-3;
		P.batch = 64;
		P.threads = 1;
		P.seed = 5;
		P.anchor = 0.01;
		CnnDistiller.Result r1 = CnnDistiller.run(P, base, set, null);
		check(Arrays.equals(baseParams, base.net.params()), "train: la rete di partenza NON viene modificata dal ciclo");
		check(r1.keptIterations == 1 && r1.stats.get(0).endHeld.tanhMse < r1.stats.get(0).startHeld.tanhMse,
				"train: una iterazione riduce l'MSE held-out contro i suoi bersagli (" + String.format("%.5f -> %.5f", r1.stats.get(0).startHeld.tanhMse, r1.stats.get(0).endHeld.tanhMse) + ")");
		// bersagli dell'iterazione 1 = ricerca della rete di PARTENZA (etichettatura indipendente)
		int[] allIdx = new int[set.n];
		for (int i = 0; i < allIdx.length; i++) {
			allIdx[i] = i;
		}
		SearchLabeler.Labels ref = SearchLabeler.labelAll(base, set.cells, set.role, allIdx, 1, 1, 12, null);
		check(Arrays.equals(ref.y, r1.baselineTargets) && r1.baselineDepth == 1, "train: baselineTargets == ricerca della rete di partenza a D (riferimento fisso della porta (i))");
		CnnDistiller.HeldOutGate g = CnnDistiller.heldOutGate(base.net, r1.net, set, cols, held, r1.baselineTargets);
		check(g.pass && g.candMse < g.baseMse && g.meanGain > 0 && g.n == held.length, "train: porta (i) superata dalla candidata addestrata: " + g);
		CnnDistiller.HeldOutGate same = CnnDistiller.heldOutGate(base.net, base.net.copy(), set, cols, held, r1.baselineTargets);
		check(!same.pass && same.meanGain == 0 && same.se == 0, "train: porta (i) NON superata da una rete identica (richiede un miglioramento stretto)");
		// una rete rovinata non passa la porta (i)
		ConvNet bad = base.net.copy();
		for (int i = 0; i < bad.params().length; i += 7) {
			bad.params()[i] += 0.3f;
		}
		check(!CnnDistiller.heldOutGate(base.net, bad, set, cols, held, r1.baselineTargets).pass, "train: porta (i) NON superata da una rete peggiorata");
		// determinismo (1 thread)
		CnnDistiller.Result r1b = CnnDistiller.run(P, base, set, null);
		check(Arrays.equals(r1.net.params(), r1b.net.params()), "train: stessi dati, stessi parametri, 1 thread -> candidata identica bit per bit");

		// ancoraggio: un mu enorme tiene i pesi vicino al riferimento, mu=0 no
		double[] y = r1.baselineTargets;
		ConvNet free = base.net.copy(), pinned = base.net.copy();
		CnnDistiller.trainIteration(free, set, cols, y, train, held, 3, 3e-3, 64, 1, true, 9, SearchLoss.LOGIT, 0.2, base.net.params().clone(), 0.0, null);
		CnnDistiller.trainIteration(pinned, set, cols, y, train, held, 3, 3e-3, 64, 1, true, 9, SearchLoss.LOGIT, 0.2, base.net.params().clone(), 50.0, null);
		double dFree = CnnDistiller.relativeDrift(free.params(), baseParams), dPinned = CnnDistiller.relativeDrift(pinned.params(), baseParams);
		check(dFree > 0 && dPinned < 0.5 * dFree, "train: l'ancoraggio L2 limita la deriva (mu=0: " + String.format("%.4f", dFree) + ", mu=50: " + String.format("%.4f", dPinned) + ")");
		check(CnnDistiller.relativeDrift(baseParams, baseParams) == 0 && near(CnnDistiller.relativeDrift(new float[] { 3f, 0f }, new float[] { 2f, 0f }), 0.5, 1e-12), "train: deriva relativa ||w-w0||/||w0||");

		// l'ablation tanh gira e anche lei riduce l'errore
		ConvNet th = base.net.copy();
		CnnDistiller.trainIteration(th, set, cols, y, train, held, 3, 3e-3, 64, 1, true, 9, SearchLoss.TANH, 0.2, base.net.params().clone(), 0.01, null);
		check(CnnDistiller.errors(th, set, cols, held, y).tanhMse < CnnDistiller.errors(base.net, set, cols, held, y).tanhMse, "train: ablation loss=tanh addestra (MSE held-out scende)");

		// lambda=1 (solo esito) non impara i bersagli di ricerca come lambda=0
		ConvNet outcomeOnly = base.net.copy(), searchOnly = base.net.copy();
		CnnDistiller.trainIteration(outcomeOnly, set, cols, y, train, held, 3, 3e-3, 64, 1, false, 9, SearchLoss.LOGIT, 1.0, null, 0, null);
		CnnDistiller.trainIteration(searchOnly, set, cols, y, train, held, 3, 3e-3, 64, 1, false, 9, SearchLoss.LOGIT, 0.0, null, 0, null);
		check(CnnDistiller.errors(searchOnly, set, cols, held, y).tanhMse < CnnDistiller.errors(outcomeOnly, set, cols, held, y).tanhMse
				&& CnnDistiller.errors(outcomeOnly, set, cols, held, y).outcomeMse < CnnDistiller.errors(searchOnly, set, cols, held, y).outcomeMse,
				"train: lambda=0 insegue la ricerca, lambda=1 l'esito (i due termini sono davvero collegati a lambda)");

		// piu' iterazioni: D fissa per iterazione, statistiche per iterazione, ancoraggio ai pesi originali vs precedenti
		CnnDistiller.Params P2 = new CnnDistiller.Params();
		P2.iters = 2;
		P2.depths = new int[] { 1, 2 };
		P2.epochs = 2;
		P2.lr = 3e-3;
		P2.batch = 64;
		P2.threads = 1;
		P2.seed = 5;
		P2.anchor = 20;
		P2.anchorToInit = false;
		CnnDistiller.Result prev = CnnDistiller.run(P2, base, set, null);
		P2.anchorToInit = true;
		CnnDistiller.Result init = CnnDistiller.run(P2, base, set, null);
		check(prev.stats.size() == 2 && prev.stats.get(0).depth == 1 && prev.stats.get(1).depth == 2, "train: ogni iterazione usa la sua profondita' fissa (D=1 poi D=2)");
		check(prev.baselineDepth == 1 && Arrays.equals(prev.baselineTargets, r1.baselineTargets), "train: i bersagli di riferimento sono quelli dell'iterazione 1 anche con piu' iterazioni");
		double dPrev = CnnDistiller.relativeDrift(prev.net.params(), baseParams), dInit = CnnDistiller.relativeDrift(init.net.params(), baseParams);
		check(init.stats.size() == 2 && dInit < dPrev, "train: anchorTo=init limita la deriva cumulata piu' di anchorTo=prev (" + String.format("%.4f < %.4f", dInit, dPrev) + ")");

		// arresto: se l'iterazione non migliora viene scartata e il ciclo si ferma
		CnnDistiller.Params P3 = new CnnDistiller.Params();
		P3.iters = 3;
		P3.depths = new int[] { 1 };
		P3.epochs = 1;
		P3.lr = 1e-12;
		P3.batch = 64;
		P3.threads = 1;
		CnnDistiller.Result stuck = CnnDistiller.run(P3, base, set, null);
		check(stuck.keptIterations == 0 && stuck.stats.size() == 1 && !stuck.stats.get(0).kept && Arrays.equals(stuck.net.params(), baseParams),
				"train: con lr trascurabile l'iterazione e' SCARTATA, il ciclo si ferma e la rete resta quella di partenza");
		check(stuck.net != base.net && r1.net != base.net, "train: la rete restituita e' sempre una COPIA, mai lo stesso oggetto della rete di partenza");
	}

	// ================================================================== porte

	static void checkGates() throws Exception {
		// posizione costruita: una pedina bianca in (2,2) tra un nero in (2,1) e un nero che puo' arrivare in (2,3)
		State s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			Arrays.fill(b[r], Pawn.EMPTY);
		}
		b[4][4] = Pawn.KING;
		b[2][2] = Pawn.WHITE;
		b[2][1] = Pawn.BLACK;
		b[2][7] = Pawn.BLACK;
		s.setTurn(Turn.BLACK);
		int[] bb = HangingGate.blackBest(s, 2, 2);
		check(bb[0] == 1 && bb[1] == 1, "gates: blackBest vede la cattura della pedina mossa (persa " + bb[0] + ", mossa catturata " + bb[1] + ")");
		b[2][1] = Pawn.EMPTY;
		bb = HangingGate.blackBest(s, 2, 2);
		check(bb[0] == 0 && bb[1] == 0, "gates: senza il nero che fa da incudine nessuna cattura");
		check(HangingGate.count(s, Pawn.WHITE) == 1 && HangingGate.count(s, Pawn.BLACK) == 1, "gates: conteggio pezzi");
		Action a = new Action("e5", "e6", Turn.WHITE);
		check(HangingGate.cellTo(a)[0] == 5 && HangingGate.cellTo(a)[1] == 4 && HangingGate.cellFrom(a)[0] == 4 && HangingGate.cellFrom(a)[1] == 4, "gates: casella (riga, colonna) da notazione 'e5'");
		double[] w = HangingGate.wilson(50, 100);
		check(w[0] < 0.5 && w[1] > 0.5 && w[0] > 0.39 && w[1] < 0.61 && HangingGate.wilson(0, 0)[1] == 1, "gates: intervallo di Wilson plausibile");

		// partite reali di HangingGate: deterministico a profondita' fissa, contatori coerenti
		HangingGate.Stats s1 = HangingGate.measure(new SimpleEvaluator(), 3, 0, 2, 1, 1, null), s2 = HangingGate.measure(new SimpleEvaluator(), 3, 0, 2, 1, 1, null);
		check(s1.games == 3 && s1.pawnMoves > 20 && s1.hangAny <= s1.pawnMoves && s1.hungMoved <= s1.hangAny && s1.events3 <= s1.pawnMoves, "gates: HangingGate registra mosse di pedina (" + s1.pawnMoves + ") con contatori coerenti");
		check(s1.pawnMoves == s2.pawnMoves && s1.hangAny == s2.hangAny && s1.hungMoved == s2.hungMoved && s1.whiteLost3 == s2.whiteLost3 && s1.blackLost3 == s2.blackLost3,
				"gates: a profondita' fissa e 1 thread la misura e' riproducibile");

		check(s1.whiteLost3 >= 0 && s1.blackLost3 >= 0 && s1.whiteLost3 + s1.blackLost3 > 0, "gates: le pedine perse nei 3 ply sono conteggi non negativi (il materiale non cresce) e ce ne sono (" + s1.whiteLost3 + " bianche, " + s1.blackLost3 + " nere)");
		int[] ar = EqualTimeArena.run("m=0:simple", "m=0:simple", 2, 20, 3, 1, 1);
		check(ar.length == 3 && ar[0] + ar[1] + ar[2] == 4, "gates: EqualTimeArena.run e' richiamabile e restituisce {V, P, S} che sommano alle partite giocate (2 coppie = 4)");

		// regola di decisione
		HangingGate.Stats base = fake(400, 200, 0.30), better = fake(400, 160, 0.20), worseShare = fake(400, 240, 0.30), worseLoss = fake(400, 200, 0.40), tiny = fake(50, 20, 0.1);
		check(HangingGate.compare(base, better, 200, 0).verified && HangingGate.compare(base, better, 200, 0).pass, "gates: HangingGate passa se la candidata non peggiora");
		check(HangingGate.compare(base, base, 200, 0).pass, "gates: HangingGate passa a parita' (non deve PEGGIORARE)");
		check(!HangingGate.compare(base, worseShare, 200, 0).pass && HangingGate.compare(base, worseShare, 200, 0).verified, "gates: HangingGate fallisce se peggiora la quota di pezzi catturabili");
		check(!HangingGate.compare(base, worseLoss, 200, 0).pass, "gates: HangingGate fallisce se peggiora il materiale perso nei 3 ply");
		check(HangingGate.compare(base, worseShare, 200, 0.2).pass && HangingGate.compare(base, worseLoss, 200, 0.2).pass && !HangingGate.compare(base, worseLoss, 200, 0.05).pass,
				"gates: la tolleranza e' rispettata su entrambe le misure (quota e materiale)");
		check(!HangingGate.compare(base, tiny, 200, 0).verified && !HangingGate.compare(base, tiny, 200, 0).pass && !HangingGate.compare(tiny, base, 200, 0).verified,
				"gates: campione insufficiente -> NON VERIFICATA (e non superata)");

		// arena
		DistillAcceptance.ArenaVerdict v = DistillAcceptance.arenaVerdict(new int[] { 130, 0, 70 }, 0.05);
		check(v.verified && v.pass && v.score == 0.65 && v.lo > 0.5, "gates: arena 130-0-70 (65%, 200 partite) supera la soglia");
		check(!DistillAcceptance.arenaVerdict(new int[] { 90, 0, 110 }, 0.05).pass, "gates: arena 45% non passa");
		check(!DistillAcceptance.arenaVerdict(new int[] { 100, 0, 100 }, 0.05).pass, "gates: arena 50% con 200 partite non passa (limite inferiore ~43%, sotto 45%)");
		check(!DistillAcceptance.arenaVerdict(new int[] { 16, 0, 14 }, 0.05).pass, "gates: 30 partite al 53% non bastano (IC95 ~+-17 punti)");
		check(DistillAcceptance.arenaVerdict(new int[] { 16, 0, 14 }, 0.20).pass, "gates: con un margine di non-inferiorita' piu' largo le stesse 30 partite passano (il margine e' configurabile)");
		check(!DistillAcceptance.arenaVerdict(new int[] { 0, 0, 0 }, 0.05).verified, "gates: nessuna partita -> NON VERIFICATA");
		DistillAcceptance.ArenaVerdict dr = DistillAcceptance.arenaVerdict(new int[] { 0, 200, 0 }, 0.05);
		check(!dr.pass && dr.score == 0.5 && dr.lo < 0.45, "gates: 200 patte = 50%: l'IC di Wilson (che non sfrutta la minor varianza delle patte) ha limite inferiore " + String.format("%.3f", dr.lo) + " < 0.45 -> non passa");

		// tabella di verita' dell'adozione
		boolean[] all = { true, true, true, true, true, true, true };
		check(DistillAcceptance.adoptable(true, true, true, true, true, true, true), "gates: tutte le porte superate -> adottabile");
		boolean onlyAll = true;
		for (int k = 0; k < 7; k++) {
			boolean[] x = all.clone();
			x[k] = false;
			onlyAll &= !DistillAcceptance.adoptable(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
		}
		check(onlyAll, "gates: se UNA SOLA porta fallisce o non e' verificata -> NON adottabile (7 casi)");

		// sostituzione con copia di sicurezza
		File dir = Files.createTempDirectory("adopt").toFile();
		File cand = new File(dir, "cand.txt"), target = new File(dir, "net.txt");
		Files.write(cand.toPath(), "NUOVA".getBytes());
		Files.write(target.toPath(), "VECCHIA".getBytes());
		DistillAcceptance.adopt(cand.getPath(), target.getPath());
		check(new String(Files.readAllBytes(target.toPath())).equals("NUOVA") && new String(Files.readAllBytes(new File(dir, "net.txt.bak").toPath())).equals("VECCHIA"),
				"gates: adopt sostituisce il file e conserva l'originale in .bak");

		// DistillAcceptance da riga di comando: con una rete identica non e' adottabile e non scrive nulla
		CnnEvaluator net = smallHybrid(5);
		File nf = new File(dir, "same.txt");
		net.save(nf.getPath());
		CnnPretrainer.Dataset ds = CnnPretrainer.generate(60, 2, 1, null);
		DistillSet set = DistillSet.sample(DistillSet.fromDataset(ds, 1, 20), null, 128, 64, 1, new String[1]);
		File sf = new File(dir, "set.gz");
		set.save(sf.getPath());
		File adopted = new File(dir, "adopted.txt");
		int code = DistillAcceptance.mainImpl(new String[] { "baseline=" + nf, "candidate=" + nf, "set=" + sf, "depth=1", "threads=1", "hgGames=2", "hgDepth=1", "hgMinMoves=1",
				"skipArena=1", "adoptTo=" + adopted });
		check(code == 1 && !adopted.exists(), "gates: baseline==candidata (nessun miglioramento stretto, porta (i) fallita) e arena saltata -> codice 1 e nessuna sostituzione (anche con adoptTo)");
		check(DistillAcceptance.mainImpl(new String[] { "baseline=" + nf }) == 2, "gates: argomenti mancanti -> uso (codice 2)");
	}

	private static HangingGate.Stats fake(int moves, int hang, double loss) {
		HangingGate.Stats s = new HangingGate.Stats();
		s.games = 10;
		s.pawnMoves = moves;
		s.hangAny = hang;
		s.hungMoved = hang / 2;
		s.events3 = moves;
		s.whiteLost3 = loss * moves;
		s.blackLost3 = loss * moves;
		return s;
	}

	// ================================================================== riga di comando

	static void checkCli() throws Exception {
		File dir = Files.createTempDirectory("cli").toFile();
		CnnPretrainer.Dataset ds = CnnPretrainer.generate(90, 4, 1, null);
		File data = new File(dir, "d.gz");
		ds.save(data.getPath());
		CnnEvaluator init = smallHybrid(13, TablutFeatures.G_ALL); // gruppi DIVERSI dal default di CnnPretrainer: init= deve ereditarli dal file
		File initF = new File(dir, "init.txt");
		init.save(initF.getPath());
		CnnEvaluator init2 = smallHybrid(13); // per il distiller (gruppi di default)
		File init2F = new File(dir, "init2.txt");
		init2.save(init2F.getPath());

		// CnnPretrainer init=
		File out = new File(dir, "ft.txt");
		CnnPretrainer.main(new String[] { "data=" + data, "init=" + initF, "epochs=1", "lr=2e-4", "threads=1", "out=" + out });
		CnnEvaluator ft = CnnEvaluator.fromFile(out.getPath());
		double drift = CnnDistiller.relativeDrift(ft.net.params(), init.net.params());
		check(ft.net.config().equals(init.net.config()) && ft.featureGroups == init.featureGroups, "cli: CnnPretrainer init= conserva forma e gruppi di feature del file di partenza");
		check(drift > 0 && drift < 0.5, "cli: CnnPretrainer init= riparte dai pesi dati e li modifica poco con lr basso (deriva " + String.format("%.4f", drift) + ")");

		// CnnDistiller da riga di comando
		File cand = new File(dir, "cand.txt"), setF = new File(dir, "set.gz");
		int code = CnnDistiller.mainImpl(new String[] { "init=" + init2F, "data=" + data, "n=320", "nHeld=96", "iters=1", "depths=1", "epochs=3", "lr=3e-3", "batch=64", "threads=1",
				"out=" + cand, "setOut=" + setF });
		check(code == 0 && cand.exists() && setF.exists(), "cli: CnnDistiller scrive candidata e set (codice " + code + ")");
		if (cand.exists()) {
			CnnEvaluator c = CnnEvaluator.fromFile(cand.getPath());
			check(c.net.config().equals(init2.net.config()) && c.net.numParams() == init2.net.numParams(), "cli: la candidata ha la STESSA rete e dimensione della partenza (nessun costo aggiuntivo a tempo di gioco)");
			DistillSet set = DistillSet.load(setF.getPath());
			check(set.splitIsDisjointByGame() && set.indices(true).length > 0, "cli: il set salvato ha lo split per partita");
		}
		check(CnnDistiller.mainImpl(new String[] {}) == 2, "cli: senza init= stampa l'uso (codice 2)");
	}
}
