package it.unibo.ai.didattica.competition.tablut.training;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.GZIPOutputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.selftest.ConvNetCheck;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer.Dataset;

/**
 * Collaudo dell'addestramento del CNN (sta nel package {@code training} per accedere ai metodi package-private).
 * <ul>
 * <li><b>Adam</b> contro un'implementazione ingenua in double.</li>
 * <li><b>Minibatch parallelo</b>: il gradiente a 1 e a 4 thread coincide con una somma ingenua fatta qui chiamando direttamente
 * {@link ConvNet#forward}/{@link ConvNet#backward}.</li>
 * <li><b>Dataset</b>: round-trip esatto, rifiuto di magic/versione/troncamenti errati; coerenza dei dati GENERATI (ruoli che
 * alternano, un solo re, feature registrate == feature ricalcolate dalla scacchiera ricostruita, esito costante nella partita).</li>
 * <li><b>TD(lambda)</b>: {@link CnnSelfPlayTrainer#trainOnTrajectory} contro un riferimento indipendente in double (valori da un oracolo
 * ingenuo, traccia/bersaglio/passo riscritti qui) su traiettorie con lambda diversi.</li>
 * <li><b>Apprendimento</b>: l'errore held-out scende molto sotto il predittore costante; training con 1 e con 3 thread.</li>
 * <li><b>End-to-end</b>: una mini-esecuzione reale di {@link CnnSelfPlayTrainer#main} (generazione parallela, TD, checkpoint,
 * valutazioni, log) e verifica dei file prodotti.</li>
 * </ul>
 */
public class CnnTrainingCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String msg) {
		checks++;
		System.out.println((ok ? "OK   - " : "FAIL - ") + msg);
		if (!ok) {
			failures++;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("=== CnnTrainingCheck ===");
		checkAdam();
		Dataset d = CnnPretrainer.generate(200, 5, 2, null);
		System.out.println("(dataset di prova: " + d.n + " posizioni da 200 partite)");
		checkGeneratedData(d);
		checkOutcomeSign();
		checkDatasetIO(d);
		checkBatchGradient(d);
		checkTdAgainstReference();
		checkGenerationTask();
		checkLearning(d);
		checkSelfPlayEndToEnd();
		System.out.println("=== " + (failures == 0 ? "CNN TRAINING CHECK OK" : "CNN TRAINING CHECK FALLITO (" + failures + ")") + " (" + checks + " verifiche) ===");
		if (failures > 0) {
			System.exit(1);
		}
	}

	// ------------------------------------------------------------------ Adam

	static void checkAdam() {
		Random rnd = new Random(1);
		int n = 50;
		float[] w = new float[n];
		double[] wd = new double[n], m = new double[n], v = new double[n];
		for (int i = 0; i < n; i++) {
			w[i] = (float) rnd.nextGaussian();
			wd[i] = w[i];
		}
		Adam adam = new Adam(n);
		double lr = 0.01, gs = 1.0 / 32;
		double worst = 0;
		for (int t = 1; t <= 40; t++) {
			float[] g = new float[n];
			for (int i = 0; i < n; i++) {
				g[i] = (float) (rnd.nextGaussian() * (1 + i % 5));
			}
			adam.step(w, g, lr, gs);
			double c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t);
			for (int i = 0; i < n; i++) {
				double gi = g[i] * gs;
				m[i] = 0.9 * m[i] + 0.1 * gi;
				v[i] = 0.999 * v[i] + 0.001 * gi * gi;
				wd[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
			}
		}
		for (int i = 0; i < n; i++) {
			worst = Math.max(worst, Math.abs(w[i] - wd[i]));
		}
		check(worst < 2e-5, "Adam (float) == Adam ingenuo in double dopo 40 passi (errore max " + String.format("%.1e", worst) + ")");
		// il passo sposta davvero i pesi nel verso opposto al gradiente
		float[] w2 = { 1f };
		new Adam(1).step(w2, new float[] { 5f }, 0.1, 1.0);
		check(w2[0] < 1f && Math.abs((1f - w2[0]) - 0.1) < 1e-3, "primo passo Adam = lr nel verso opposto al gradiente (" + (1f - w2[0]) + ")");
	}

	// ------------------------------------------------------------------ dati generati

	static State stateFrom(byte[] cells, int off, int role) {
		State s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				int code = cells[off + r * 9 + c];
				b[r][c] = code == 1 ? Pawn.WHITE : code == 2 ? Pawn.BLACK : code == 3 ? Pawn.KING : (r == 4 && c == 4 ? Pawn.THRONE : Pawn.EMPTY);
			}
		}
		s.setTurn(role == 0 ? Turn.WHITE : Turn.BLACK);
		return s;
	}

	static void checkGeneratedData(Dataset d) {
		check(d.n > 2000, "dataset generato non vuoto: " + d.n + " posizioni");
		boolean outcomesOk = true, rolesOk = true, kingOk = true, featOk = true, constOk = true;
		TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
		double[] full = new double[TablutFeatures.NUM_FEATURES];
		int prevGame = -1, posInGame = 0;
		double worstFeat = 0;
		for (int i = 0; i < d.n; i++) {
			outcomesOk &= d.outcome[i] >= -1 && d.outcome[i] <= 1;
			if (d.game[i] != prevGame) {
				posInGame = 0;
				prevGame = d.game[i];
			}
			rolesOk &= d.role[i] == posInGame % 2;
			constOk &= i == 0 || d.game[i] != d.game[i - 1] || d.outcome[i] == d.outcome[i - 1];
			posInGame++;
			int kings = 0, w = 0, b = 0;
			for (int k = 0; k < 81; k++) {
				int c = d.cells[i * 81 + k];
				kings += c == 3 ? 1 : 0;
				w += c == 1 ? 1 : 0;
				b += c == 2 ? 1 : 0;
			}
			kingOk &= kings == 1 && w <= 8 && b <= 16;
			if (i % 7 == 0) { // le feature registrate devono essere quelle ricalcolate dalla scacchiera
				State s = stateFrom(d.cells, i * 81, d.role[i]);
				TablutFeatures.extract(s, ws, full, TablutFeatures.G_ALL);
				for (int f = 0; f < TablutFeatures.NUM_FEATURES; f++) {
					double diff = Math.abs(d.feats[i * TablutFeatures.NUM_FEATURES + f] - (float) full[f]);
					worstFeat = Math.max(worstFeat, diff);
					featOk &= diff < 1e-6;
				}
			}
		}
		check(outcomesOk, "esiti in {-1,0,+1}");
		check(rolesOk, "in ogni partita i ruoli alternano Bianco, Nero, Bianco... partendo dal Bianco");
		check(constOk, "l'esito e' costante dentro la stessa partita");
		check(kingOk, "ogni posizione ha esattamente un re, al piu' 8 bianchi e 16 neri");
		check(featOk, "feature registrate == feature ricalcolate dalla scacchiera ricostruita (errore max " + String.format("%.1e", worstFeat) + ")");
		// la prima posizione di ogni partita e' quella iniziale
		byte[] init = ConvNetCheck.codes(new StateTablut());
		boolean firstOk = true;
		for (int i = 0; i < d.n; i++) {
			if (i == 0 || d.game[i] != d.game[i - 1]) {
				firstOk &= Arrays.equals(Arrays.copyOfRange(d.cells, i * 81, i * 81 + 81), init) && d.role[i] == 0;
			}
		}
		check(firstOk, "ogni partita comincia dalla posizione iniziale col Bianco a muovere");
	}

	/**
	 * Esito atteso (+1 Bianco vince, -1 Nero vince, 0 pareggio) dedotto SOLO dallo stato finale: re sul bordo/catturato
	 * ({@code Turn.WHITEWIN/BLACKWIN}) o stallo (chi non ha mosse perde). Indipendente da {@code GameTracker}.
	 */
	static double expectedOutcome(State end) {
		if (end.getTurn() == Turn.WHITEWIN) {
			return 1;
		}
		if (end.getTurn() == Turn.BLACKWIN) {
			return -1;
		}
		if (!it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules.isTerminal(end)
				&& it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules.generateMoves(end).isEmpty()) {
			return end.getTurn() == Turn.WHITE ? -1 : 1;
		}
		return 0;
	}

	/** Il segno dell'esito registrato corrisponde al vincitore REALE (un'etichetta invertita si imparerebbe benissimo: serve un controllo diretto). */
	static void checkOutcomeSign() {
		int decisiveW = 0, decisiveB = 0, bad = 0, total = 0;
		for (long seed = 1; seed <= 150; seed++) {
			CnnPretrainer.GameRecord r = CnnPretrainer.playOne(seed);
			if (r.truncated) {
				continue;
			}
			total++;
			double want = expectedOutcome(r.end);
			decisiveW += want > 0 ? 1 : 0;
			decisiveB += want < 0 ? 1 : 0;
			if (r.outcome != want) {
				bad++;
			}
		}
		check(bad == 0 && decisiveW >= 5 && decisiveB >= 5, "esito registrato == vincitore dedotto dallo stato finale su " + total + " partite (Bianco " + decisiveW
				+ ", Nero " + decisiveB + ", discordanze " + bad + ")");
	}

	// ------------------------------------------------------------------ I/O dataset

	static void checkDatasetIO(Dataset d) throws IOException {
		File dir = new File(System.getProperty("java.io.tmpdir"));
		File f = File.createTempFile("cnn_data", ".gz", dir);
		f.deleteOnExit();
		d.save(f.getPath());
		Dataset l = Dataset.load(f.getPath());
		boolean same = l.n == d.n && Arrays.equals(l.cells, d.cells) && Arrays.equals(l.role, d.role) && Arrays.equals(l.outcome, d.outcome)
				&& Arrays.equals(l.game, d.game) && Arrays.equals(l.feats, d.feats);
		check(same, "dataset: salva/carica identico (" + f.length() / 1024 + " KB compressi)");

		check(loadFails(writeRaw(dir, "WRONG MAGIC", TablutFeatures.VERSION, TablutFeatures.NUM_FEATURES, 0)), "dataset: magic errato -> IOException");
		check(loadFails(writeRaw(dir, "TABLUT_CNN_DATA v1", TablutFeatures.VERSION + 1, TablutFeatures.NUM_FEATURES, 0)),
				"dataset: versione delle feature diversa -> IOException");
		check(loadFails(writeRaw(dir, "TABLUT_CNN_DATA v1", TablutFeatures.VERSION, TablutFeatures.NUM_FEATURES + 1, 0)),
				"dataset: numero di feature diverso -> IOException");
		check(loadFails(writeRaw(dir, "TABLUT_CNN_DATA v1", TablutFeatures.VERSION, TablutFeatures.NUM_FEATURES, 1_000_000_000)),
				"dataset: conteggio assurdo -> IOException");
		check(loadFails(writeRaw(dir, "TABLUT_CNN_DATA v1", TablutFeatures.VERSION, TablutFeatures.NUM_FEATURES, 10)),
				"dataset: file troncato (dichiara 10 posizioni, non ce ne sono) -> IOException");
		byte[] raw = Files.readAllBytes(f.toPath());
		File cut = File.createTempFile("cnn_cut", ".gz", dir);
		cut.deleteOnExit();
		Files.write(cut.toPath(), Arrays.copyOf(raw, raw.length / 2));
		check(loadFails(cut), "dataset: gzip tagliato a meta' -> IOException");
	}

	static File writeRaw(File dir, String magic, int ver, int nf, int n) throws IOException {
		File f = File.createTempFile("cnn_raw", ".gz", dir);
		f.deleteOnExit();
		try (DataOutputStream o = new DataOutputStream(new GZIPOutputStream(new FileOutputStream(f)))) {
			o.writeUTF(magic);
			o.writeInt(ver);
			o.writeInt(nf);
			o.writeInt(n);
		}
		return f;
	}

	static boolean loadFails(File f) {
		try {
			Dataset.load(f.getPath());
			return false;
		} catch (IOException e) {
			return true;
		}
	}

	// ------------------------------------------------------------------ minibatch parallelo

	static void checkBatchGradient(Dataset d) throws Exception {
		int[] cols = TablutFeatures.columnsFor(TablutFeatures.G_DEFAULT);
		ConvNet.Config cfg = new ConvNet.Config(new int[] { 4, 3 }, 2, cols.length, new int[] { 8 }, new int[] { 4 });
		ConvNet net = ConvNetCheck.randomNet(cfg, 3);
		int[] idx = new int[96];
		Random rnd = new Random(2);
		for (int i = 0; i < idx.length; i++) {
			idx[i] = rnd.nextInt(d.n);
		}
		final long seed = 99;
		final int epoch = 3;

		// riferimento ingenuo: un solo workspace, chiamate dirette a forward/backward
		float[] ref = new float[net.numParams()];
		ConvNet.Workspace ws = net.newWorkspace();
		float[] fb = new float[cols.length];
		double refLoss = 0;
		for (int k = 0; k < idx.length; k++) {
			int i = idx[k];
			ws.setCodes(d.cells, i * 81, CnnPretrainer.symOf(seed, epoch, i));
			for (int j = 0; j < cols.length; j++) {
				fb[j] = d.feats[i * TablutFeatures.NUM_FEATURES + cols[j]];
			}
			double e = net.forward(ws, fb, d.role[i]) - d.outcome[i];
			refLoss += e * e;
			net.backward(ws, e, ref);
		}
		double mag = 0;
		for (float x : ref) {
			mag = Math.max(mag, Math.abs(x));
		}
		check(mag > 1e-3, "il gradiente di riferimento non e' degenere (|g| max = " + String.format("%.3f", mag) + ")");

		for (int threads : new int[] { 1, 4 }) {
			ExecutorService pool = Executors.newFixedThreadPool(threads);
			CnnPretrainer.Worker[] ws2 = new CnnPretrainer.Worker[threads];
			for (int t = 0; t < threads; t++) {
				ws2[t] = new CnnPretrainer.Worker(net);
			}
			float[] g = new float[net.numParams()];
			double loss = CnnPretrainer.batchGradient(pool, ws2, d, cols, idx, 0, idx.length, epoch, seed, true, g);
			pool.shutdown();
			double worst = 0;
			for (int i = 0; i < g.length; i++) {
				worst = Math.max(worst, Math.abs(g[i] - ref[i]));
			}
			check(worst < 1e-4 * Math.max(1, mag) && Math.abs(loss - refLoss) < 1e-4 * Math.max(1, refLoss),
					"gradiente di minibatch con " + threads + " thread == somma ingenua (errore " + String.format("%.1e", worst) + ", loss "
							+ String.format("%.4f", loss) + " vs " + String.format("%.4f", refLoss) + ")");
		}
		// la simmetria scelta e' deterministica e usa tutte e 8 le simmetrie
		boolean[] seen = new boolean[8];
		for (int i = 0; i < 400; i++) {
			seen[CnnPretrainer.symOf(5, 1, i)] = true;
		}
		boolean all = true;
		for (boolean b : seen) {
			all &= b;
		}
		check(all && CnnPretrainer.symOf(5, 1, 17) == CnnPretrainer.symOf(5, 1, 17), "symOf: deterministica e copre tutte le 8 simmetrie");
	}

	// ------------------------------------------------------------------ TD

	/** Riferimento ingenuo del TD(lambda): valori dall'oracolo double, traccia e passo riscritti, gradiente dal CNN (verificato altrove). */
	static double[] referenceTd(ConvNet layout, ConvNet.Config cfg, double[] P0, CnnSelfPlayTrainer.Traj t, double lambda, double alpha) {
		double[] P = P0.clone();
		double[] trace = new double[P.length];
		int steps = t.cells.size();
		for (int i = 0; i < steps; i++) {
			float[] pf = new float[P.length];
			for (int k = 0; k < P.length; k++) {
				pf[k] = (float) P[k];
			}
			ConvNet cur = new ConvNet(cfg, pf);
			ConvNet.Workspace ws = cur.newWorkspace();
			ws.setCodes(t.cells.get(i), 0);
			cur.forward(ws, t.feats.get(i), t.roles[i]);
			float[] g = new float[P.length];
			cur.backward(ws, 1.0, g);
			double v = ConvNetCheck.oracle(layout, P, t.cells.get(i), toD(t.feats.get(i)), t.roles[i]);
			double target = i + 1 < steps ? ConvNetCheck.oracle(layout, P, t.cells.get(i + 1), toD(t.feats.get(i + 1)), t.roles[i + 1]) : t.outcome;
			for (int k = 0; k < P.length; k++) {
				trace[k] = lambda * trace[k] + g[k];
			}
			double delta = target - v;
			for (int k = 0; k < P.length; k++) {
				P[k] += alpha * delta * trace[k];
			}
		}
		return P;
	}

	static double[] toD(float[] f) {
		double[] d = new double[f.length];
		for (int i = 0; i < f.length; i++) {
			d[i] = f[i];
		}
		return d;
	}

	static CnnSelfPlayTrainer.Traj trajFrom(List<State> pos, double outcome, int nFeat, Random rnd) {
		List<byte[]> cells = new ArrayList<>();
		List<float[]> feats = new ArrayList<>();
		int[] roles = new int[pos.size()];
		for (int i = 0; i < pos.size(); i++) {
			cells.add(ConvNetCheck.codes(pos.get(i)));
			feats.add(ConvNetCheck.randomFeats(nFeat, rnd));
			roles[i] = NeuralEvaluator.roleOf(pos.get(i));
		}
		return new CnnSelfPlayTrainer.Traj(cells, feats, roles, outcome, pos.size(), "TEST", null, "test", 0);
	}

	static void checkTdAgainstReference() {
		Random rnd = new Random(8);
		ConvNet.Config cfg = new ConvNet.Config(new int[] { 3, 2 }, 2, 3, new int[] { 5 }, new int[] { 3 });
		double worstAll = 0, moved = 0;
		for (double lambda : new double[] { 0.0, 0.7, 1.0 }) {
			for (double outcome : new double[] { 1.0, -1.0, 0.0 }) {
				ConvNet net = ConvNetCheck.randomNet(cfg, 40 + (long) (lambda * 10));
				ConvNet layout = net.copy();
				double[] P0 = ConvNetCheck.toDouble(net.params());
				List<State> pos = ConvNetCheck.positions(7, 100 + (long) (outcome * 3 + lambda * 11));
				CnnSelfPlayTrainer.Traj t = trajFrom(pos, outcome, 3, rnd);
				double[] want = referenceTd(layout, cfg, P0, t, lambda, 0.05);
				CnnSelfPlayTrainer.trainOnTrajectory(net, t, lambda, 0.05);
				double worst = 0, mv = 0;
				for (int k = 0; k < want.length; k++) {
					worst = Math.max(worst, Math.abs(net.params()[k] - want[k]));
					mv = Math.max(mv, Math.abs(want[k] - P0[k]));
				}
				worstAll = Math.max(worstAll, worst);
				moved = Math.max(moved, mv);
				check(worst < 5e-5 && mv > 1e-4, "TD(lambda=" + lambda + ", esito " + outcome + ") == riferimento indipendente in double (errore "
						+ String.format("%.1e", worst) + ", spostamento massimo dei pesi " + String.format("%.1e", mv) + ")");
			}
		}
		// casi limite
		ConvNet net = ConvNetCheck.randomNet(cfg, 1);
		float[] before = net.params().clone();
		double e = CnnSelfPlayTrainer.trainOnTrajectory(net, trajFrom(new ArrayList<State>(), 1.0, 3, rnd), 0.7, 0.05);
		check(e == 0 && Arrays.equals(before, net.params()), "traiettoria vuota: nessun errore, nessuna modifica dei pesi");
		CnnSelfPlayTrainer.Traj one = trajFrom(ConvNetCheck.positions(1, 3), 1.0, 3, rnd);
		double[] want1 = referenceTd(net.copy(), cfg, ConvNetCheck.toDouble(net.params()), one, 0.7, 0.05);
		CnnSelfPlayTrainer.trainOnTrajectory(net, one, 0.7, 0.05);
		double w1 = 0;
		for (int k = 0; k < want1.length; k++) {
			w1 = Math.max(w1, Math.abs(net.params()[k] - want1[k]));
		}
		check(w1 < 5e-5, "traiettoria di una sola posizione (bersaglio = esito finale) == riferimento (errore " + String.format("%.1e", w1) + ")");
		// TD spinge V(s_0) verso l'esito: ripetendo molte volte la stessa partita vinta, il valore sale
		ConvNet n2 = ConvNetCheck.randomNet(cfg, 7);
		CnnSelfPlayTrainer.Traj win = trajFrom(ConvNetCheck.positions(5, 55), 1.0, 3, rnd);
		ConvNet.Workspace ws = n2.newWorkspace();
		ws.setCodes(win.cells.get(0), 0);
		double v0 = n2.forward(ws, win.feats.get(0), win.roles[0]);
		for (int it = 0; it < 300; it++) {
			CnnSelfPlayTrainer.trainOnTrajectory(n2, win, 0.7, 0.02);
		}
		ws.setCodes(win.cells.get(0), 0);
		double v1 = n2.forward(ws, win.feats.get(0), win.roles[0]);
		check(v1 > v0 + 0.1, "300 ripetizioni di una partita vinta dal Bianco: V(posizione iniziale) sale da " + String.format("%.3f", v0) + " a "
				+ String.format("%.3f", v1));
		check(Math.abs(CnnSelfPlayTrainer.epsilonSchedule(0, 1000, 0.3, 0.02) - 0.3) < 1e-9
				&& Math.abs(CnnSelfPlayTrainer.epsilonSchedule(700, 1000, 0.3, 0.02) - 0.02) < 1e-9
				&& Math.abs(CnnSelfPlayTrainer.epsilonSchedule(5000, 1000, 0.3, 0.02) - 0.02) < 1e-9, "epsilon: parte da start, arriva a end al 70%, poi resta fisso");
	}

	// ------------------------------------------------------------------ traiettorie del self-play

	/** La traiettoria prodotta da una vera partita di self-play e' coerente: e' cio' che il TD poi usera' come dato. */
	static void checkGenerationTask() throws Exception {
		int groups = TablutFeatures.G_DEFAULT;
		int[] cols = TablutFeatures.columnsFor(groups);
		ConvNet net = ConvNetCheck.randomNet(new ConvNet.Config(new int[] { 4 }, 2, cols.length, new int[] { 8 }, new int[] { 4 }), 9);
		CnnEvaluator ev = new CnnEvaluator(net, groups);
		TablutFeatures.Workspace fws = TablutFeatures.newWorkspace();
		double[] full = new double[TablutFeatures.NUM_FEATURES];
		byte[] init = ConvNetCheck.codes(new StateTablut());
		boolean allOk = true;
		StringBuilder why = new StringBuilder();
		for (int mode = 0; mode < 3; mode++) { // 0 = self-play puro, 1 = contro Simple, 2 = contro il proprio passato (nessun checkpoint: ripiega su self-play)
			CnnSelfPlayTrainer.Traj t = new CnnSelfPlayTrainer.GenerationTask(net.copy(), groups, new it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator(),
					new ArrayList<String>(), 2, 0.1, 5, 12, 1.0, mode == 1 ? 1.0 : 0.0, mode == 2 ? 1.0 : 0.0, 100 + mode).call();
			int n = t.cells.size();
			boolean ok = n > 4 && t.feats.size() == n && t.roles.length == n && t.plies >= n - 1;
			ok &= Arrays.equals(t.cells.get(0), init) && t.roles[0] == 0;
			float[] f = new float[cols.length];
			for (int i = 0; i < n && ok; i++) {
				ok &= t.roles[i] == i % 2;
				State s = stateFrom(t.cells.get(i), 0, t.roles[i]);
				ev.extractFeatures(s, fws, full, f);
				ok &= Arrays.equals(f, t.feats.get(i)); // le feature registrate sono quelle della posizione registrata
				if (i > 0) {
					ok &= !Arrays.equals(t.cells.get(i), t.cells.get(i - 1)); // ogni mossa cambia la scacchiera
				}
			}
			boolean nonZero = false;
			for (float x : t.feats.get(n / 2)) {
				nonZero |= x != 0f;
			}
			ok &= nonZero && Math.abs(t.outcome) <= 1.0 && t.outcomeLabel != null;
			ok &= mode == 1 ? (t.opponentControls != null && t.opponentType.equals("baseline")) : (t.opponentControls == null && t.opponentType.equals("self"));
			if (!ok) {
				why.append(" modo ").append(mode);
			}
			allOk &= ok;
		}
		// l'esito della traiettoria e' il vincitore REALE (si cercano partite decisive: servono almeno 3, altrimenti il controllo sarebbe vuoto)
		int decisive = 0, wrong = 0;
		for (int sd = 0; sd < 40 && decisive < 3; sd++) {
			CnnSelfPlayTrainer.Traj t = new CnnSelfPlayTrainer.GenerationTask(net.copy(), groups, new it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator(),
					new ArrayList<String>(), 2, 0.3, 2, 4, 1.0, 0.0, 0.0, 500 + sd).call();
			double want = expectedOutcome(t.finalState);
			if (want != 0) {
				decisive++;
			}
			if (t.outcome != want) {
				wrong++;
			}
		}
		allOk &= decisive >= 3 && wrong == 0;
		if (decisive < 3 || wrong > 0) {
			why.append(" esito (decisive=" + decisive + ", errate=" + wrong + ")");
		}
		check(allOk, "traiettorie di self-play coerenti (scacchiera iniziale, ruoli alternati, feature registrate == ricalcolate, tipo di avversario)"
				+ (allOk ? "" : " - fallito in:" + why));
	}

	// ------------------------------------------------------------------ apprendimento

	static void checkLearning(Dataset d) throws Exception {
		int[][] sp = CnnPretrainer.split(d);
		double base = CnnPretrainer.constantBaseline(d, sp[0], sp[1]);
		int[] cols = TablutFeatures.columnsFor(TablutFeatures.G_DEFAULT);
		double[] testMse = new double[2];
		int k = 0;
		for (int threads : new int[] { 1, 3 }) {
			ConvNet net = new ConvNet(new ConvNet.Config(new int[] { 6 }, 2, cols.length, new int[] { 16 }, new int[] { 6 }), 11);
			double before = CnnPretrainer.mse(net, d, cols, sp[0]);
			double[] r = CnnPretrainer.train(net, d, cols, sp[0], sp[1], 10, 0.003, 64, threads, true, 11, null);
			double after = CnnPretrainer.mse(net, d, cols, sp[0]);
			testMse[k++] = r[1];
			// (a) l'ottimizzazione funziona: l'errore sul train scende nettamente
			check(after < 0.7 * before, "ottimizzazione (" + threads + " thread): MSE di train " + String.format("%.4f", before) + " -> " + String.format("%.4f", after)
					+ " (< 70%)");
			// (b) generalizza oltre il predittore costante. Con ~6000 posizioni di train la soglia e' prudente: l'overfitting e' reale
			// (la prova di generalizzazione vera e' l'esperimento a 150.000 posizioni, si veda README_CNN.md)
			check(r[1] < base * 0.95, "generalizzazione (" + threads + " thread): MSE held-out " + String.format("%.4f", r[1]) + " < 95% del predittore costante ("
					+ String.format("%.4f", base) + ")");
		}
		check(Math.abs(testMse[0] - testMse[1]) < 0.02, "1 e 3 thread danno lo stesso risultato dell'addestramento (" + String.format("%.4f", testMse[0]) + " vs "
				+ String.format("%.4f", testMse[1]) + ")");
		// CNN puro: impara senza feature a mano
		ConvNet pure = new ConvNet(new ConvNet.Config(new int[] { 6 }, 2, 0, new int[] { 16 }, new int[] { 6 }), 12);
		double pb = CnnPretrainer.mse(pure, d, new int[0], sp[0]);
		double[] r = CnnPretrainer.train(pure, d, new int[0], sp[0], sp[1], 10, 0.003, 64, 2, true, 12, null);
		double pa = CnnPretrainer.mse(pure, d, new int[0], sp[0]);
		check(pa < 0.8 * pb && r[1] < base, "CNN puro (senza feature a mano): MSE di train " + String.format("%.4f", pb) + " -> " + String.format("%.4f", pa)
				+ ", held-out " + String.format("%.4f", r[1]) + " < predittore costante " + String.format("%.4f", base));
		// colonne incoerenti con la rete: errore esplicito
		boolean threw = false;
		try {
			CnnPretrainer.train(pure, d, cols, sp[0], sp[1], 1, 0.003, 64, 1, true, 1, null);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "colonne di feature incoerenti con nFeat -> IllegalArgumentException");
	}

	// ------------------------------------------------------------------ end-to-end

	static void checkSelfPlayEndToEnd() throws Exception {
		File dir = Files.createTempDirectory("cnn_e2e").toFile();
		int[] cols = TablutFeatures.columnsFor(TablutFeatures.G_DEFAULT);
		CnnEvaluator init = CnnEvaluator.randomInit(new ConvNet.Config(new int[] { 4 }, 2, cols.length, new int[] { 8 }, new int[] { 4 }),
				TablutFeatures.G_DEFAULT, 5);
		File w0 = new File(dir, "w0.txt");
		init.save(w0.getPath());
		float[] before = init.net.params().clone();
		File out = new File(dir, "out.txt"), log = new File(dir, "log.csv"), ck = new File(dir, "ck");
		long t0 = System.currentTimeMillis();
		CnnSelfPlayTrainer.main(new String[] { "resumeFrom=" + w0.getPath(), "episodes=4", "threads=2", "minMoveMs=10", "maxMoveMs=25",
				"maxSearchDepthCap=3", "checkpointEvery=2", "evalGames=2", "evalDepths=1", "out=" + out.getPath(), "log=" + log.getPath(),
				"checkpointDir=" + ck.getPath(), "opponentPoolProb=0.5", "historicalOpponentProb=0.3", "seed=3", "alpha=0.002" });
		System.out.println("(mini-training end-to-end: " + (System.currentTimeMillis() - t0) / 1000 + " s)");
		check(out.exists(), "il trainer scrive il file dei pesi");
		CnnEvaluator after = CnnEvaluator.fromFile(out.getPath());
		boolean changed = false, finite = true;
		for (int i = 0; i < before.length; i++) {
			changed |= before[i] != after.net.params()[i];
			finite &= !Float.isNaN(after.net.params()[i]) && !Float.isInfinite(after.net.params()[i]);
		}
		check(changed && finite, "i pesi sono cambiati e sono tutti finiti dopo 4 episodi");
		check(after.net.config().equals(init.net.config()) && after.featureGroups == init.featureGroups, "configurazione e gruppi di feature preservati");
		List<String> lines = Files.readAllLines(log.toPath());
		check(lines.size() == 5 && lines.get(0).startsWith("episode,"), "il log ha intestazione + 4 righe di episodio (" + lines.size() + " righe)");
		boolean logOk = true;
		for (int i = 1; i < lines.size(); i++) {
			String[] f = lines.get(i).split(",");
			logOk &= f.length == 8 && Integer.parseInt(f[0]) == i && !Double.isNaN(Double.parseDouble(f[1])) && Integer.parseInt(f[2]) > 0;
		}
		check(logOk, "righe di log ben formate: episodio progressivo, errore TD finito, mosse > 0");
		File[] cks = ck.listFiles();
		check(cks != null && cks.length == 2, "2 checkpoint scritti (episodi 2 e 4)");
	}
}
