package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Test dell'addestratore da dati di gioco ({@link NnueTrainer}).
 * <ul>
 * <li><b>Bersaglio</b>: tutte le combinazioni di etichette, {@code lambda}, sconto {@code gamma}, nessuna etichetta = nessun bersaglio.</li>
 * <li><b>Aritmetica</b>: {@code fromQuantized} e' esatta (quantizzare di nuovo restituisce la stessa rete); la permutazione degli indici delle feature
 * equivale a trasformare la posizione con {@link NnueDistiller#transformed} (la trasformazione di stato del progetto); il gradiente con simmetria e peso
 * coincide con le differenze finite; la saturazione dei pesi rende la quantizzazione senza perdite.</li>
 * <li><b>Apprende davvero</b>: un esito che e' una funzione nota della posizione (materiale), un valore noto (un maestro NNUE imparabile) e il mix dei due.</li>
 * <li><b>Garanzie</b>: se nessuna epoca migliora la validazione la rete e' restituita INVARIATA (bit per bit); i dati esterni non addestrano (un campione
 * esterno casuale non viene memorizzato); un peso di sorgente 0 esclude quella sorgente; {@code maxSamples}; arresto anticipato; determinismo; validazione
 * per partita.</li>
 * <li>Metriche confrontate con un ricalcolo diretto; configurazioni non valide; {@code main} in un sottoprocesso (partenza da pesi, confronto, errori).</li>
 * </ul>
 */
public class NnueTrainerCheck {

	private static int failures = 0;
	private static long checks = 0;

	private static void check(boolean ok, String msg) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	private static boolean same(double a, double b) {
		return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
	}

	public static void main(String[] args) throws Exception {
		testTarget();
		testFromQuantized();
		testSymmetryPermutation();
		testGradientWithSymmetryAndWeights();
		testClamp();
		testLearnsOutcome();
		testLearnsValueAndMix();
		testNoRegressionAndExternalValidation();
		testExtLambda();
		testBestEpochIsReturned();
		testSourceWeightsAndMaxSamples();
		testDeterminismAndEarlyStopping();
		testMetrics();
		testConfigErrors();
		testCli();
		System.out.println(failures == 0 ? "=== NNUE TRAINER CHECK OK: " + checks + " verifiche ===" : "=== NNUE TRAINER CHECK: " + failures + " FALLIMENTI su " + checks + " ===");
		if (failures != 0) {
			System.exit(1);
		}
	}

	// ====================================================================== dati di prova

	/** Posizioni casuali valide con etichette scelte dal chiamante tramite {@code labeler}. */
	private interface Labeler {
		/** @return {outcome (+1/-1/0), outValid (0/1), value*1000, valValid (0/1)} */
		int[] label(byte[] board, int role, int i);
	}

	private static NnueDataset build(Random rnd, int n, int perGame, int source, Labeler lab) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		while (b.size() < n) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			if (NnueDataset.checkPosition(fs.board, 0) != null) {
				continue;
			}
			final int i = b.size();
			final int role = fs.turn == BoardConstants.BLACK_TURN ? 1 : 0;
			final int[] l = lab.label(fs.board, role, i);
			b.add(fs.board, 0, role, l[0], l[1] == 1, l[2] / 1000.0, l[3] == 1, i / perGame, 0, l[1] == 1 ? 1 : -1, source);
		}
		return b.build();
	}

	private static int count(byte[] board, int code) {
		int c = 0;
		for (byte x : board) {
			c += x == code ? 1 : 0;
		}
		return c;
	}

	private static int[] range(int n) {
		final int[] a = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = i;
		}
		return a;
	}

	/** Campioni sulla posizione iniziale con le sole etichette indicate: ogni riga e' {esito, outValid, valValid, partita, toEnd, sorgente}. */
	private static NnueDataset samples(int[]... rows) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		for (int[] r : rows) {
			b.add(new FastState().board, 0, 0, r[0], r[1] == 1, 0, r[2] == 1, r[3], 0, r[4], r[5]);
		}
		return b.build();
	}

	private static NnueDataset oneSample(int outcome, boolean outValid, double value, boolean valValid, int toEnd) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		b.add(new FastState().board, 0, 0, outcome, outValid, value, valValid, 0, 3, toEnd, 0);
		return b.build();
	}

	// ====================================================================== bersaglio

	private static void testTarget() {
		final NnueTrainer.Config c = new NnueTrainer.Config();
		c.lambda = 0.25;
		check(Math.abs(NnueTrainer.target(c, oneSample(1, true, 0.8, true, 5), 0) - (0.25 * 0.8f + 0.75)) < 1e-12, "entrambe le etichette: lambda*valore + (1-lambda)*esito");
		c.lambda = 1.0;
		check(Math.abs(NnueTrainer.target(c, oneSample(-1, true, 0.5, true, 5), 0) - 0.5) < 1e-12, "lambda = 1: solo il valore (quando ci sono entrambi)");
		c.lambda = 0.0;
		check(Math.abs(NnueTrainer.target(c, oneSample(-1, true, 0.5, true, 5), 0) + 1.0) < 1e-12, "lambda = 0: solo l'esito (quando ci sono entrambi)");
		c.lambda = 0.5;
		check(Math.abs(NnueTrainer.target(c, oneSample(0, false, 0.4, true, -1), 0) - 0.4) < 1e-6, "solo il valore: il bersaglio e' il valore, a prescindere da lambda");
		check(Math.abs(NnueTrainer.target(c, oneSample(-1, true, 0.0, false, 5), 0) + 1.0) < 1e-12, "solo l'esito: il bersaglio e' l'esito, a prescindere da lambda");
		check(Math.abs(NnueTrainer.target(c, oneSample(0, true, 0.0, false, 5), 0)) < 1e-12, "una patta e' un esito valido: bersaglio 0");
		check(Double.isNaN(NnueTrainer.target(c, oneSample(0, false, 0.0, false, -1), 0)), "nessuna etichetta: nessun bersaglio (NaN)");
		// sconto dell'esito con la distanza dalla fine
		c.gamma = 0.9;
		check(Math.abs(NnueTrainer.target(c, oneSample(1, true, 0, false, 10), 0) - Math.pow(0.9, 10)) < 1e-12, "gamma: esito * gamma^toEnd");
		check(Math.abs(NnueTrainer.target(c, oneSample(-1, true, 0, false, 1), 0) + 0.9) < 1e-12, "gamma: ultima mossa prima della fine");
		check(Math.abs(NnueTrainer.target(c, oneSample(1, true, 0.5, true, 4), 0) - (0.5 * 0.5 + 0.5 * Math.pow(0.9, 4))) < 1e-12, "gamma sconta solo l'esito, non il valore");
		check(Math.abs(NnueTrainer.target(c, oneSample(0, false, 0.5, true, -1), 0) - 0.5) < 1e-6, "gamma non tocca un campione senza esito");
		final NnueDataset unknownEnd = samples(new int[] { 1, 1, 0, 0, -1, 0 }); // esito valido ma distanza dalla fine ignota: nessuno sconto
		check(Math.abs(NnueTrainer.target(c, unknownEnd, 0) - 1.0) < 1e-12, "toEnd ignoto: nessuno sconto");
		// usable: filtra senza etichetta e peso 0
		final NnueDataset mix = samples(new int[] { 0, 0, 0, 0, -1, 0 }, new int[] { 1, 1, 0, 1, 1, 1 }, new int[] { 1, 1, 0, 2, 1, 2 });
		final NnueTrainer.Config w = new NnueTrainer.Config();
		w.sourceWeight[2] = 0;
		check(NnueTrainer.usable(w, mix, range(3), true).length == 1 && NnueTrainer.usable(w, mix, range(3), false).length == 2, "usable: senza etichetta fuori, peso 0 fuori solo per l'addestramento");
	}

	// ====================================================================== aritmetica

	private static void testFromQuantized() {
		for (int[] shape : new int[][] { { 64, 16 }, { 8, 4 }, { 1, 1 }, { 3, 2 } }) {
			final NnueNetwork q = NnueNetwork.random(shape[0], shape[1], 100 + shape[0]);
			final NnueDistiller.FloatNet f = NnueDistiller.FloatNet.fromQuantized(q);
			final NnueDistiller.Quantized back = NnueDistiller.quantize(f);
			check(back.clipped == 0 && back.net.equals(q), "fromQuantized -> quantize deve restituire ESATTAMENTE la stessa rete (" + shape[0] + "/" + shape[1] + ")");
			check(f.hidden == shape[0] && f.l2 == shape[1], "dimensioni della rete in virgola mobile");
		}
		// stessi valori in uscita (a meno dell'arrotondamento di tanh vs tabella): la rete float e quella quantizzata concordano sulle stesse posizioni
		final NnueNetwork q = NnueNetwork.random(32, 8, 7);
		final NnueDistiller.FloatNet f = NnueDistiller.FloatNet.fromQuantized(q);
		final NnueEvaluator ev = new NnueEvaluator(q);
		ev.outputScale = 1.0;
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(32, 8);
		final Random rnd = new Random(3);
		double worst = 0;
		for (int i = 0; i < 200; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			final int role = fs.turn == BoardConstants.BLACK_TURN ? 1 : 0;
			// la rete intera arrotonda le attivazioni dello strato 1 (>> 6): concorda con quella float entro l'errore di arrotondamento, non esattamente
			worst = Math.max(worst, Math.abs(Math.tanh(NnueDistiller.forward(f, fs.board, 0, role, ws)) - ev.evaluate(fs)));
		}
		check(worst < 0.05, "la rete float ricavata da una quantizzata concorda con essa entro l'arrotondamento (errore massimo " + worst + ")");
	}

	private static void testSymmetryPermutation() {
		final NnueDistiller.FloatNet f = new NnueDistiller.FloatNet(12, 5);
		f.init(new Random(9));
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(12, 5);
		final Random rnd = new Random(10);
		double worst = 0;
		boolean boardsEqual = true;
		for (int i = 0; i < 40; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			final int role = fs.turn == BoardConstants.BLACK_TURN ? 1 : 0;
			for (int s = 0; s < BoardSymmetry.COUNT; s++) {
				// la posizione trasformata con la trasformazione di STATO del progetto, e la stessa ottenuta permutando l'array
				final byte[] viaState = NnueDistiller.transformed(fs, s).board;
				final byte[] viaMap = new byte[81];
				for (int sq = 0; sq < 81; sq++) {
					viaMap[BoardSymmetry.MAP[s][sq]] = fs.board[sq];
				}
				boardsEqual &= Arrays.equals(viaState, viaMap);
				final double a = NnueDistiller.forward(f, fs.board, 0, role, BoardSymmetry.MAP[s], ws);
				final double b = NnueDistiller.forward(f, viaMap, 0, role, null, ws);
				worst = Math.max(worst, Math.abs(a - b));
			}
			check(same(NnueDistiller.forward(f, fs.board, 0, role, null, ws), NnueDistiller.forward(f, fs.board, 0, role, BoardSymmetry.MAP[0], ws)), "la simmetria 0 e' l'identita'");
		}
		check(boardsEqual, "la permutazione MAP equivale alla trasformazione di stato del progetto (NnueDistiller.transformed)");
		check(worst < 1e-12, "forward con la permutazione = forward sulla posizione trasformata (differenza massima " + worst + ")");
	}

	private static void testGradientWithSymmetryAndWeights() {
		final NnueDistiller.FloatNet net = new NnueDistiller.FloatNet(5, 3);
		net.init(new Random(17));
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(5, 3);
		final Random rnd = new Random(18);
		final int n = 8;
		final byte[][] boards = new byte[n][];
		final int[] roles = new int[n], syms = new int[n];
		final double[] target = new double[n], weight = new double[n];
		for (int i = 0; i < n; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			boards[i] = fs.board.clone();
			roles[i] = fs.turn == BoardConstants.BLACK_TURN ? 1 : 0;
			syms[i] = 1 + rnd.nextInt(7); // sempre una simmetria non banale
			target[i] = rnd.nextDouble() * 1.6 - 0.8;
			weight[i] = 0.5 + 1.5 * rnd.nextDouble();
		}
		final double[] g = new double[net.numParams()];
		for (int i = 0; i < n; i++) {
			final double y = Math.tanh(NnueDistiller.forward(net, boards[i], 0, roles[i], BoardSymmetry.MAP[syms[i]], ws));
			NnueDistiller.backward(net, roles[i], 2 * weight[i] * (y - target[i]) * (1 - y * y), ws, g);
		}
		final double eps = 1e-6;
		int mismatches = 0, nonZero = 0;
		for (int k = 0; k < net.numParams(); k++) {
			final double keep = net.p[k];
			double[] lv = new double[2];
			for (int s = 0; s < 2; s++) {
				net.p[k] = keep + (s == 0 ? eps : -eps);
				double l = 0;
				for (int i = 0; i < n; i++) {
					final double y = Math.tanh(NnueDistiller.forward(net, boards[i], 0, roles[i], BoardSymmetry.MAP[syms[i]], ws));
					l += weight[i] * (y - target[i]) * (y - target[i]);
				}
				lv[s] = l;
			}
			net.p[k] = keep;
			final double fd = (lv[0] - lv[1]) / (2 * eps);
			if (Math.abs(fd - g[k]) > 1e-6 + 1e-5 * Math.max(Math.abs(fd), Math.abs(g[k]))) {
				mismatches++;
			}
			nonZero += Math.abs(g[k]) > 1e-9 ? 1 : 0;
		}
		check(mismatches == 0, "gradiente con simmetria e pesi: " + mismatches + " parametri diversi dalle differenze finite");
		check(nonZero > 150, "gradiente con simmetria e pesi: troppi pochi parametri non nulli (" + nonZero + ")");
	}

	private static void testClamp() {
		// il passo dell'ottimizzatore satura i pesi: un gradiente enorme con un passo enorme porterebbe i pesi ben oltre l'intervallo di short
		final NnueDistiller.FloatNet big = new NnueDistiller.FloatNet(6, 3);
		big.init(new Random(3));
		final double[] grad = new double[big.numParams()];
		Arrays.fill(grad, 1.0);
		NnueTrainer.optimizerStep(new NnueDistiller.AdamD(big.numParams()), big, grad, 1e5, 1.0);
		check(NnueDistiller.quantize(big).clipped == 0, "optimizerStep: dopo un passo enorme i pesi sono ancora quantizzabili senza perdita");
		double maxFt = 0;
		for (int i = 0; i < 243 * 6; i++) {
			maxFt = Math.max(maxFt, Math.abs(big.p[i]));
		}
		check(maxFt > 100 && maxFt < 128, "optimizerStep: i pesi sono stati spinti al limite (" + maxFt + ") e saturati, non lasciati liberi");
		final NnueDistiller.FloatNet f = new NnueDistiller.FloatNet(6, 3);
		f.init(new Random(1));
		java.util.Arrays.fill(f.p, 1e6);
		NnueTrainer.clampToQuantRange(f);
		final NnueDistiller.Quantized q = NnueDistiller.quantize(f);
		check(q.clipped == 0, "dopo la saturazione la quantizzazione non perde pesi (" + q.clipped + ")");
		java.util.Arrays.fill(f.p, -1e6);
		NnueTrainer.clampToQuantRange(f);
		check(NnueDistiller.quantize(f).clipped == 0, "idem per i pesi molto negativi");
		final NnueDistiller.FloatNet g = new NnueDistiller.FloatNet(6, 3);
		g.init(new Random(2));
		final double[] before = g.p.clone();
		NnueTrainer.clampToQuantRange(g);
		check(Arrays.equals(before, g.p), "la saturazione non tocca pesi gia' nell'intervallo");
	}

	// ====================================================================== apprende

	private static NnueTrainer.Config smallCfg(int epochs) {
		final NnueTrainer.Config c = new NnueTrainer.Config();
		c.hidden = 16;
		c.l2 = 4;
		c.epochs = epochs;
		c.batch = 64;
		c.lr = 3e-3;
		c.lrEnd = 3e-4;
		c.valPct = 15;
		c.patience = 100;
		c.seed = 5;
		return c;
	}

	private static void testLearnsOutcome() throws Exception {
		// l'esito e' una funzione NOTA della posizione: chi ha piu' "materiale" (2*bianchi - neri sopra la mediana) vince. Solo l'esito, nessun valore
		final Random rnd = new Random(21);
		final int n = 4000;
		final int[] mat = new int[n];
		final List<byte[]> all = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			all.add(fs.board.clone());
			mat[i] = 2 * count(fs.board, BoardConstants.WHITE) - count(fs.board, BoardConstants.BLACK);
		}
		final int[] sorted = mat.clone();
		Arrays.sort(sorted);
		final int median = sorted[n / 2];
		final NnueDataset.Builder b = new NnueDataset.Builder();
		int plus = 0;
		for (int i = 0; i < n; i++) {
			if (mat[i] == median) {
				continue; // niente pareggi sul confine
			}
			final int o = mat[i] > median ? 1 : -1;
			plus += o > 0 ? 1 : 0;
			b.add(all.get(i), 0, i % 2, o, true, 0, false, i / 4, 0, 1, NnueDataset.SRC_LOG);
		}
		final NnueDataset d = b.build();
		check(plus > d.n / 4 && plus < 3 * d.n / 4, "dati di prova: classi non sbilanciate (" + plus + "/" + d.n + ")");
		final NnueTrainer.Config c = smallCfg(25);
		final int[][] sp = d.splitByGame(c.valSalt, c.valPct);
		final NnueDistiller.FloatNet start = new NnueDistiller.FloatNet(c.hidden, c.l2);
		start.init(new Random(c.seed));
		final double before = NnueTrainer.meanLoss(start, c, d, NnueTrainer.usable(c, d, sp[1], false));
		final NnueTrainer.Result r = NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null);
		final NnueDistiller.Quantized q = NnueDistiller.quantize(r.net);
		final NnueTrainer.Metrics m = NnueTrainer.evaluate(q.net, c, d, sp[1]);
		System.out.println("apprendimento dell'esito (materiale): perdita held-out " + before + " -> " + r.bestVal + "; rete quantizzata: " + m);
		check(r.bestVal < 0.5 * before, "esito: la perdita di validazione deve calare almeno della meta' (" + before + " -> " + r.bestVal + ")");
		check(m.outcomeAccuracy > 0.9, "esito: la rete deve predire il vincitore sul held-out (precisione " + m.outcomeAccuracy + ")");
		check(m.nDecided == m.nOutcome && m.nValue == 0 && r.bestEpoch >= 1, "esito: solo etichette di esito, nessun valore");
		check(q.clipped == 0, "esito: nessun peso saturato");
	}

	private static void testLearnsValueAndMix() throws Exception {
		// un maestro NNUE casuale e' imparabile esattamente dalla stessa struttura: il valore (e il mix con un esito derivato) si impara
		final NnueEvaluator teacher = new NnueEvaluator(NnueNetwork.random(16, 4, 31));
		teacher.outputScale = 1.0;
		final FastState probe = new FastState();
		final Labeler lab = (board, role, i) -> {
			System.arraycopy(board, 0, probe.board, 0, 81);
			probe.turn = role == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
			final double v = teacher.evaluate(probe);
			return new int[] { v > 0 ? 1 : -1, 1, (int) Math.round(v * 1000), 1 };
		};
		final NnueDataset d = build(new Random(22), 6000, 4, NnueDataset.SRC_SELFPLAY, lab);
		for (double lambda : new double[] { 1.0, 0.5 }) {
			final NnueTrainer.Config c = smallCfg(30);
			c.lambda = lambda;
			c.augment = false; // il maestro casuale NON e' invariante per simmetria: l'aumento contraddirebbe le sue etichette
			final int[][] sp = d.splitByGame(c.valSalt, c.valPct);
			final NnueDistiller.FloatNet start = new NnueDistiller.FloatNet(c.hidden, c.l2);
			start.init(new Random(c.seed));
			final double before = NnueTrainer.meanLoss(start, c, d, NnueTrainer.usable(c, d, sp[1], false));
			final NnueTrainer.Result r = NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null);
			final NnueNetwork q = NnueDistiller.quantize(r.net).net;
			final NnueTrainer.Metrics m = NnueTrainer.evaluate(q, c, d, sp[1]);
			System.out.println("apprendimento del valore (lambda " + lambda + "): perdita " + before + " -> " + r.bestVal + "; " + m);
			check(r.bestVal < 0.4 * before, "lambda " + lambda + ": la perdita di validazione deve calare di oltre il 60% (" + before + " -> " + r.bestVal + ")");
			check(m.valueCorr > 0.8, "lambda " + lambda + ": correlazione col valore del maestro " + m.valueCorr);
			check(m.outcomeAccuracy > 0.85, "lambda " + lambda + ": precisione sul segno (esito derivato) " + m.outcomeAccuracy);
		}
	}

	// ====================================================================== garanzie

	private static void testNoRegressionAndExternalValidation() {
		// (1) NESSUNA REGRESSIONE: la rete di partenza predice +1 ovunque; i dati di addestramento dicono -1, la validazione esterna +1. Qualunque
		// addestramento peggiora la validazione: la rete restituita deve essere quella di partenza, bit per bit
		final Random rnd = new Random(40);
		final NnueDataset train = build(rnd, 300, 3, NnueDataset.SRC_SELFPLAY, (b, r, i) -> new int[] { -1, 1, 0, 0 });
		final NnueDataset ext = build(rnd, 120, 3, NnueDataset.SRC_LOG, (b, r, i) -> new int[] { 1, 1, 0, 0 });
		final NnueTrainer.Config c = smallCfg(6);
		final NnueDistiller.FloatNet start = new NnueDistiller.FloatNet(c.hidden, c.l2);
		start.init(new Random(1));
		java.util.Arrays.fill(start.p, 0.0);
		// uscita costante molto positiva: bias di uscita alto in entrambe le teste
		for (int role = 0; role < 2; role++) {
			start.p[start.offL2B(role)] = 3.0;
		}
		final double[] startParams = start.p.clone();
		final NnueTrainer.Result r = NnueTrainer.train(train, range(train.n), new int[0], ext, range(ext.n), c, start, null);
		check(r.bestEpoch == 0, "nessuna regressione: nessuna epoca migliora la validazione esterna, quindi epoca 0 (trovata " + r.bestEpoch + ")");
		check(Arrays.equals(r.net.p, startParams), "nessuna regressione: i pesi restituiti sono ESATTAMENTE quelli di partenza");
		check(same(r.bestVal, r.initialVal), "nessuna regressione: la perdita finale e' quella iniziale");
		check(r.trainSamples == train.n, "i campioni di addestramento sono quelli passati (" + r.trainSamples + " su " + train.n + "), non quelli esterni");

		// (2) I DATI ESTERNI NON ADDESTRANO: 60 posizioni esterne con esito CASUALE, mai memorizzabili se non le si usa per addestrare. Una rete da 17k
		// parametri le memorizzerebbe in poche epoche se le vedesse: la perdita esterna resterebbe invece alta
		final NnueDataset learnable = build(new Random(41), 800, 4, NnueDataset.SRC_SELFPLAY, (b, role, i) -> new int[] { count(b, BoardConstants.WHITE) > 4 ? 1 : -1, 1, 0, 0 });
		final Random coin = new Random(42);
		final NnueDataset noise = build(new Random(43), 60, 3, NnueDataset.SRC_LOG, (b, role, i) -> new int[] { coin.nextBoolean() ? 1 : -1, 1, 0, 0 });
		final NnueTrainer.Config c2 = smallCfg(150);
		c2.hidden = 64;
		c2.l2 = 16;
		c2.patience = 1000;
		c2.augment = false; // senza aumento un campione si vede sempre uguale: e' la condizione in cui una rete cosi' grande memorizza etichette casuali
		c2.lr = 5e-3;
		c2.lrEnd = 1e-3;
		final NnueTrainer.Result r2 = NnueTrainer.train(learnable, range(learnable.n), new int[0], noise, range(noise.n), c2, null, null);
		final double extLoss = NnueTrainer.meanLoss(r2.net, c2, noise, range(noise.n));
		check(r2.trainSamples == learnable.n, "l'addestramento usa solo i propri campioni (" + r2.trainSamples + " su " + learnable.n + ")");
		check(extLoss > 0.6, "i campioni esterni casuali non devono essere memorizzati: perdita esterna " + extLoss + " (se fossero addestrati sarebbe ~0)");
		// controprova: gli stessi campioni, se messi nell'addestramento, si memorizzano
		final NnueDataset both = NnueDataset.merge(Arrays.asList(learnable, noise));
		final int[] tr = range(both.n);
		final NnueTrainer.Result r3 = NnueTrainer.train(both, tr, range(60), null, null, c2, null, null);
		final int[] noiseIdx = new int[60];
		for (int i = 0; i < 60; i++) {
			noiseIdx[i] = learnable.n + i;
		}
		final double memorized = NnueTrainer.meanLoss(r3.net, c2, both, noiseIdx);
		check(memorized < extLoss - 0.2, "controprova: gli stessi campioni nell'addestramento vengono memorizzati (" + memorized + " < " + extLoss + ")");
	}

	/**
	 * Con un learning rate alto la perdita di validazione migliora e poi oscilla: la migliore epoca NON e' l'ultima. La rete restituita deve essere
	 * quella della migliore (perdita esattamente uguale a quella dichiarata), non l'ultima eseguita. Il test esige che lo scenario si verifichi.
	 */
	private static void testBestEpochIsReturned() {
		final Random rnd = new Random(77);
		final NnueDataset d = build(rnd, 2500, 4, NnueDataset.SRC_SELFPLAY, (b, role, i) -> new int[] { 2 * count(b, BoardConstants.WHITE) - count(b, BoardConstants.BLACK) > 0 ? 1 : -1, 1, 0, 0 });
		boolean scenario = false;
		for (double lr : new double[] { 0.05, 0.1, 0.3 }) {
			final NnueTrainer.Config c = smallCfg(14);
			c.lr = lr;
			c.lrEnd = lr;
			c.patience = 100;
			final int[][] sp = d.splitByGame(c.valSalt, c.valPct);
			final NnueTrainer.Result r = NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null);
			if (r.bestEpoch >= 1 && r.bestEpoch < r.epochsRun) {
				scenario = true;
				final double returned = NnueTrainer.meanLoss(r.net, c, d, NnueTrainer.usable(c, d, sp[1], false));
				check(Math.abs(returned - r.bestVal) < 1e-12, "lr " + lr + ": la rete restituita deve avere la perdita della MIGLIORE epoca (" + returned + " vs " + r.bestVal + ", epoca " + r.bestEpoch + " di " + r.epochsRun + ")");
			}
		}
		check(scenario, "scenario di prova: con un lr alto la migliore epoca deve essere diversa dall'ultima (altrimenti il test e' vacuo)");
	}

	private static void testExtLambda() {
		// la validazione esterna ha esito +1 e valore -1: con extLambda = 0 si giudica sul solo esito (bersaglio +1), altrimenti col mix dell'addestramento (bersaglio 0)
		final NnueDataset train = build(new Random(44), 200, 4, NnueDataset.SRC_SELFPLAY, (b, role, i) -> new int[] { 1, 1, 500, 1 });
		final NnueDataset ext = build(new Random(45), 80, 4, NnueDataset.SRC_LOG, (b, role, i) -> new int[] { 1, 1, -1000, 1 });
		final double[] initial = new double[2];
		for (int k = 0; k < 2; k++) {
			final NnueTrainer.Config c = smallCfg(1);
			c.lambda = 0.5;
			c.extLambda = k == 0 ? 0.0 : -1;
			final NnueDistiller.FloatNet start = new NnueDistiller.FloatNet(c.hidden, c.l2);
			start.init(new Random(1));
			java.util.Arrays.fill(start.p, 0.0);
			start.p[start.offL2B(0)] = 3.0; // uscita costante ~ +0.995
			start.p[start.offL2B(1)] = 3.0;
			initial[k] = NnueTrainer.train(train, range(train.n), new int[0], ext, range(ext.n), c, start, null).initialVal;
		}
		check(initial[0] < 1e-3, "extLambda = 0: la validazione esterna si giudica sull'esito (+1): perdita iniziale " + initial[0]);
		check(initial[1] > 0.9, "extLambda = -1: la validazione esterna usa il mix dell'addestramento (bersaglio 0): perdita iniziale " + initial[1]);
		for (double bad : new double[] { 2.0, -0.5 }) {
			final NnueTrainer.Config c = smallCfg(1);
			c.extLambda = bad;
			boolean threw = false;
			try {
				NnueTrainer.train(train, range(train.n), range(10), null, null, c, null, null);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "extLambda " + bad + " non valido");
		}
	}

	private static void testSourceWeightsAndMaxSamples() {
		// le STESSE posizioni con esiti opposti in due sorgenti: vince la sorgente con peso
		final Random rnd = new Random(50);
		final NnueDataset a = build(rnd, 300, 3, 1, (b, r, i) -> new int[] { 1, 1, 0, 0 });
		final NnueDataset bb = build(new Random(50), 300, 3, 2, (b, r, i) -> new int[] { -1, 1, 0, 0 });
		final NnueDataset d = NnueDataset.merge(Arrays.asList(a, bb));
		final double[][] weights = { { 1, 0 }, { 0, 1 }, { 3, 1 }, { 1, 3 } };
		final double[] expectedSign = { 1, -1, 1, -1 };
		for (int k = 0; k < weights.length; k++) {
			final NnueTrainer.Config c = smallCfg(8);
			c.sourceWeight[1] = weights[k][0];
			c.sourceWeight[2] = weights[k][1];
			// la validazione e' sulla sorgente che pesa di piu': in caso contrario la garanzia di non regressione restituirebbe la rete di partenza
			final int[] val = new int[40];
			for (int i = 0; i < 40; i++) {
				val[i] = (weights[k][0] > weights[k][1] ? 0 : 300) + i;
			}
			final NnueTrainer.Result r = NnueTrainer.train(d, range(d.n), val, null, null, c, null, null);
			final NnueNetwork q = NnueDistiller.quantize(r.net).net;
			final NnueEvaluator ev = new NnueEvaluator(q);
			ev.outputScale = 1.0;
			double mean = 0;
			for (int i = 0; i < 100; i++) { // le stesse 100 posizioni, che stanno in entrambe le sorgenti
				final FastState fs = new FastState();
				System.arraycopy(d.boards, 81 * i, fs.board, 0, 81);
				fs.turn = d.role[i] == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
				mean += ev.evaluate(fs) / 100;
			}
			check(mean * expectedSign[k] > 0.1, "pesi di sorgente " + Arrays.toString(weights[k]) + ": la media delle previsioni deve avere segno " + expectedSign[k] + " (" + mean + ")");
			check(k > 1 || r.trainSamples == 300, "peso 0: la sorgente esclusa non si usa (" + r.trainSamples + ")");
		}
		// maxSamples
		final NnueTrainer.Config c = smallCfg(2);
		c.maxSamples = 50;
		final NnueTrainer.Result r1 = NnueTrainer.train(d, range(d.n), range(40), null, null, c, null, null);
		final NnueTrainer.Result r2 = NnueTrainer.train(d, range(d.n), range(40), null, null, c, null, null);
		check(r1.trainSamples == 50, "maxSamples limita l'addestramento (" + r1.trainSamples + ")");
		check(Arrays.equals(r1.net.p, r2.net.p), "maxSamples: il sottoinsieme e' deterministico");
	}

	private static void testDeterminismAndEarlyStopping() {
		final NnueDataset d = build(new Random(60), 600, 3, 1, (b, r, i) -> new int[] { count(b, BoardConstants.BLACK) > 8 ? -1 : 1, 1, 0, 0 });
		final NnueTrainer.Config c = smallCfg(4);
		final int[][] sp = d.splitByGame(c.valSalt, c.valPct);
		final NnueTrainer.Result a = NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null);
		final NnueTrainer.Result b = NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null);
		check(Arrays.equals(a.net.p, b.net.p) && a.bestEpoch == b.bestEpoch, "stesso seme = stessa rete, bit per bit");
		c.seed = 6;
		check(!Arrays.equals(a.net.p, NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null).net.p), "un altro seme da' un'altra rete");
		// senza simmetrie il risultato e' diverso (e il codice le usa davvero)
		c.seed = 5;
		c.augment = false;
		check(!Arrays.equals(a.net.p, NnueTrainer.train(d, sp[0], sp[1], null, null, c, null, null).net.p), "augment=false cambia l'addestramento");
		// arresto anticipato: etichette CASUALI, niente da imparare, la validazione non migliora
		final Random coin = new Random(61);
		final NnueDataset noise = build(new Random(62), 500, 5, 1, (b2, r2, i) -> new int[] { coin.nextBoolean() ? 1 : -1, 1, 0, 0 });
		final NnueTrainer.Config e = smallCfg(40);
		e.patience = 2;
		final int[][] sp2 = noise.splitByGame(e.valSalt, e.valPct);
		final NnueTrainer.Result r = NnueTrainer.train(noise, sp2[0], sp2[1], null, null, e, null, null);
		check(r.epochsRun < 40, "arresto anticipato con etichette casuali (eseguite " + r.epochsRun + " epoche su 40)");
		check(r.bestEpoch <= r.epochsRun && r.bestVal <= r.initialVal + 1e-12, "la perdita restituita non e' mai peggiore di quella iniziale");
		final double lossOfReturned = NnueTrainer.meanLoss(r.net, e, noise, NnueTrainer.usable(e, noise, sp2[1], false));
		check(Math.abs(lossOfReturned - r.bestVal) < 1e-9, "la rete restituita ha la perdita di validazione dichiarata (" + lossOfReturned + " vs " + r.bestVal + ")");
		// nessun campione di addestramento / di validazione: errore esplicito
		final NnueDataset unl = samples(new int[] { 0, 0, 0, 0, -1, 0 });
		for (int k = 0; k < 2; k++) {
			boolean threw = false;
			try {
				if (k == 0) {
					NnueTrainer.train(unl, range(1), range(1), null, null, smallCfg(1), null, null);
				} else {
					NnueTrainer.train(d, range(d.n), new int[0], null, null, smallCfg(1), null, null);
				}
			} catch (IllegalArgumentException ex) {
				threw = true;
			}
			check(threw, "senza campioni etichettati / senza validazione: errore esplicito (caso " + k + ")");
		}
		// rete di partenza di dimensioni diverse
		boolean threw = false;
		try {
			NnueTrainer.train(d, sp[0], sp[1], null, null, smallCfg(1), new NnueDistiller.FloatNet(8, 4), null);
		} catch (IllegalArgumentException ex) {
			threw = true;
		}
		check(threw, "la rete di partenza deve avere le dimensioni della configurazione");
	}

	// ====================================================================== metriche

	private static void testMetrics() {
		final NnueNetwork q = NnueNetwork.random(16, 4, 71);
		final NnueEvaluator ev = new NnueEvaluator(q);
		ev.outputScale = 1.0;
		final Random rnd = new Random(72);
		final NnueDataset d = build(rnd, 300, 3, 0, (b, r, i) -> new int[] { rnd.nextInt(3) - 1, i % 5 == 0 ? 0 : 1, rnd.nextInt(2001) - 1000, i % 3 == 0 ? 0 : 1 });
		final NnueTrainer.Config c = new NnueTrainer.Config();
		c.lambda = 0.5;
		final NnueTrainer.Metrics m = NnueTrainer.evaluate(q, c, d, range(d.n));
		// ricalcolo diretto, indipendente
		double seT = 0, seO = 0, seV = 0, se0 = 0;
		int nT = 0, nO = 0, nV = 0, nD = 0, ok = 0;
		final FastState fs = new FastState();
		for (int i = 0; i < d.n; i++) {
			System.arraycopy(d.boards, 81 * i, fs.board, 0, 81);
			fs.turn = d.role[i] == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
			final double y = ev.evaluate(fs);
			final boolean ho = d.outValid[i] == 1, hv = d.valValid[i] == 1;
			final double o = ho ? d.outcome[i] : 0;
			if (ho) {
				seO += (y - o) * (y - o);
				se0 += o * o;
				nO++;
				if (o != 0) {
					nD++;
					ok += (y > 0) == (o > 0) ? 1 : 0;
				}
			}
			if (hv) {
				seV += (y - d.value[i]) * (y - d.value[i]);
				nV++;
			}
			if (ho || hv) {
				final double t = hv && ho ? 0.5 * d.value[i] + 0.5 * o : hv ? d.value[i] : o;
				seT += (y - t) * (y - t);
				nT++;
			}
		}
		check(m.nTarget == nT && m.nOutcome == nO && m.nValue == nV && m.nDecided == nD, "metriche: conteggi (" + m.nTarget + "/" + nT + ", " + m.nOutcome + "/" + nO + ", " + m.nValue + "/" + nV + ")");
		check(Math.abs(m.mseTarget - seT / nT) < 1e-9 && Math.abs(m.mseOutcome - seO / nO) < 1e-9 && Math.abs(m.mseValue - seV / nV) < 1e-9 && Math.abs(m.zeroMseOutcome - se0 / nO) < 1e-12,
				"metriche: MSE sul bersaglio/esito/valore e linea di base 'predire 0'");
		check(Math.abs(m.outcomeAccuracy - ok / (double) nD) < 1e-12, "metriche: precisione sul segno dell'esito");
		check(Math.abs(m.outcomeCorr) <= 1.0 && Math.abs(m.valueCorr) <= 1.0, "metriche: correlazioni in [-1, 1]");
		// nessuna etichetta: zeri, senza eccezioni (divisioni per zero)
		final NnueDataset unl = samples(new int[] { 0, 0, 0, 0, -1, 0 });
		final NnueTrainer.Metrics z = NnueTrainer.evaluate(q, c, unl, range(1));
		check(z.nTarget == 0 && z.mseTarget == 0 && z.mseOutcome == 0 && z.outcomeAccuracy == 0 && !Double.isNaN(z.outcomeCorr), "metriche senza etichette: zeri, niente NaN");
		check(NnueTrainer.evaluate(q, c, d, new int[0]).nTarget == 0, "metriche su un insieme vuoto");
		check(m.toString().contains("esito:") && m.toString().contains("valore di ricerca"), "toString delle metriche");
		// la correlazione e' quella di Pearson: una rete perfettamente correlata con il bersaglio ha 1
		final NnueDataset.Builder pb = new NnueDataset.Builder();
		for (int i = 0; i < d.n; i++) {
			System.arraycopy(d.boards, 81 * i, fs.board, 0, 81);
			fs.turn = d.role[i] == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
			final double y = ev.evaluate(fs);
			pb.add(d.boards, 81 * i, d.role[i], y > 0 ? 1 : -1, true, y, true, d.game[i], 0, 1, 0);
		}
		final NnueTrainer.Metrics perfect = NnueTrainer.evaluate(q, c, pb.build(), range(d.n));
		check(perfect.valueCorr > 0.9999 && perfect.mseValue < 1e-6 && perfect.outcomeAccuracy == 1.0, "una rete che coincide con i valori: correlazione 1, MSE 0, precisione 100% (" + perfect + ")");
	}

	private static void testConfigErrors() {
		final NnueDataset d = build(new Random(80), 60, 3, 0, (b, r, i) -> new int[] { 1, 1, 0, 0 });
		final List<NnueTrainer.Config> bad = new ArrayList<>();
		for (int k = 0; k < 12; k++) {
			final NnueTrainer.Config c = smallCfg(1);
			switch (k) {
			case 0: c.hidden = 0; break;
			case 1: c.l2 = 0; break;
			case 2: c.epochs = 0; break;
			case 3: c.batch = 0; break;
			case 4: c.lr = 0; break;
			case 5: c.lambda = 1.5; break;
			case 6: c.gamma = 0; break;
			case 7: c.gamma = 1.5; break;
			case 8: c.valPct = 95; break;
			case 9: c.patience = 0; break;
			case 10: c.sourceWeight[3] = -1; break;
			default: c.hidden = NnueNetwork.MAX_HIDDEN + 1;
			}
			bad.add(c);
		}
		for (int k = 0; k < bad.size(); k++) {
			boolean threw = false;
			try {
				NnueTrainer.train(d, range(d.n), range(10), null, null, bad.get(k), null, null);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "configurazione non valida #" + k + " accettata dal trainer");
		}
	}

	// ====================================================================== main in un sottoprocesso

	private static int run(List<String> args, StringBuilder out) throws Exception {
		return runCls("NnueTrainer", args, out);
	}

	private static int runCls(String cls, List<String> args, StringBuilder out) throws Exception {
		final List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.add("-Xmx1g");
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.training." + cls);
		cmd.addAll(args);
		final Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		final java.io.ByteArrayOutputStream bo = new java.io.ByteArrayOutputStream();
		final byte[] buf = new byte[8192];
		int n;
		while ((n = p.getInputStream().read(buf)) > 0) {
			bo.write(buf, 0, n);
		}
		out.append(new String(bo.toByteArray(), "UTF-8"));
		if (!p.waitFor(300, TimeUnit.SECONDS)) {
			p.destroyForcibly();
			return -999;
		}
		return p.exitValue();
	}

	private static File tmp(String suffix) throws IOException {
		final File f = File.createTempFile("trainercli_", suffix);
		f.deleteOnExit();
		return f;
	}

	private static void testCli() throws Exception {
		final Random rnd = new Random(90);
		final NnueDataset train = build(rnd, 600, 4, 1, (b, r, i) -> new int[] { count(b, BoardConstants.WHITE) > 4 ? 1 : -1, 1, (count(b, BoardConstants.WHITE) - 4) * 150, 1 });
		final NnueDataset val = build(rnd, 150, 4, 0, (b, r, i) -> new int[] { count(b, BoardConstants.WHITE) > 4 ? 1 : -1, 1, 0, 0 });
		final File dtr = tmp(".gz"), dva = tmp(".gz"), init = tmp(".nnue"), cmp = tmp(".nnue"), out = tmp(".nnue"), out2 = tmp(".nnue");
		train.save(dtr.getPath());
		val.save(dva.getPath());
		NnueNetwork.random(16, 4, 5).save(init.getPath());
		NnueNetwork.random(16, 4, 6).save(cmp.getPath());

		StringBuilder o = new StringBuilder();
		int rc = run(Arrays.asList("data=" + dtr.getPath(), "val=" + dva.getPath(), "out=" + out.getPath(), "init=" + init.getPath(), "compare=" + cmp.getPath(), "epochs=3", "batch=64",
				"lr=0.003", "seed=1", "weights=1:1,0:2"), o);
		check(rc == 0, "NnueTrainer: uscita " + rc + ". Output: " + o);
		final String s = o.toString();
		check(s.contains("rete addestrata | held-out") && s.contains("rete addestrata | esterno") && s.contains("confronto " + cmp.getPath()), "NnueTrainer: rapporto con rete addestrata e confronto. Output: " + s);
		final NnueNetwork trained = NnueNetwork.load(out.getPath());
		check(trained.hidden() == 16 && trained.l2() == 4, "NnueTrainer: il file scritto ha le dimensioni della rete di partenza");
		check(!trained.equals(NnueNetwork.load(init.getPath())), "NnueTrainer: la rete e' stata addestrata");
		// ripetibile bit per bit
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "val=" + dva.getPath(), "out=" + out2.getPath(), "init=" + init.getPath(), "compare=" + cmp.getPath(), "epochs=3", "batch=64",
				"lr=0.003", "seed=1", "weights=1:1,0:2"), o);
		check(rc == 0 && Arrays.equals(Files.readAllBytes(out.toPath()), Files.readAllBytes(out2.toPath())), "NnueTrainer: lo stesso comando da' lo stesso file");
		// senza miglioramento: lr enorme, la rete resta quella di partenza
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "val=" + dva.getPath(), "out=" + out2.getPath(), "init=" + init.getPath(), "epochs=2", "lr=5", "lrend=5", "seed=1"), o);
		check(rc == 0 && o.toString().contains("NESSUN MIGLIORAMENTO") && NnueNetwork.load(out2.getPath()).equals(NnueNetwork.load(init.getPath())), "NnueTrainer: nessun miglioramento = rete invariata. Output: " + o);
		// extlambda accettato, fuori intervallo rifiutato
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "val=" + dva.getPath(), "out=" + out2.getPath(), "init=" + init.getPath(), "epochs=1", "extlambda=0"), o);
		check(rc == 0, "NnueTrainer extlambda=0: uscita " + rc + ". Output: " + o);
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "val=" + dva.getPath(), "out=" + out2.getPath(), "extlambda=2"), o);
		check(rc == 2, "NnueTrainer extlambda=2: uscita " + rc + " (attesa 2)");
		// NnueEvalNets
		o = new StringBuilder();
		rc = runCls("NnueEvalNets", Arrays.asList("nets=" + init.getPath() + "," + cmp.getPath(), "data=" + dva.getPath()), o);
		check(rc == 0 && o.toString().contains(init.getPath() + " | ") && o.toString().contains(cmp.getPath() + " | ") && o.toString().contains("segno giusto"), "NnueEvalNets: uscita " + rc + ". Output: " + o);
		final String[][] badEval = { {}, { "nets=" + init.getPath() }, { "data=" + dva.getPath() }, { "nets=" + init.getPath(), "data=" + dva.getPath(), "lambda=2" },
				{ "nets=" + init.getPath(), "data=" + dva.getPath(), "boh=1" }, { "nets=" + init.getPath(), "data=" + dva.getPath(), "senzauguale" } };
		for (String[] b : badEval) {
			o = new StringBuilder();
			rc = runCls("NnueEvalNets", Arrays.asList(b), o);
			check(rc == 2, "NnueEvalNets " + Arrays.toString(b) + ": uscita " + rc + " (attesa 2): " + o);
		}
		o = new StringBuilder();
		rc = runCls("NnueEvalNets", Arrays.asList("nets=/nonexistent_dir_xyz/x.nnue", "data=" + dva.getPath()), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueEvalNets con rete inesistente: uscita " + rc + " (attesa 1)");
		o = new StringBuilder();
		rc = runCls("NnueEvalNets", Arrays.asList("nets=" + init.getPath(), "data=/nonexistent_dir_xyz/x.gz"), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueEvalNets con dati inesistenti: uscita " + rc + " (attesa 1)");
		// da zero, con la validazione per partita
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "out=" + out2.getPath(), "hidden=8", "l2=4", "epochs=2", "valpct=20"), o);
		check(rc == 0 && NnueNetwork.load(out2.getPath()).hidden() == 8, "NnueTrainer da zero: uscita " + rc + ". Output: " + o);

		// errori
		final String[][] bad2 = { {}, { "data=" + dtr.getPath() }, { "out=" + out2.getPath() }, { "data=" + dtr.getPath(), "out=" + out2.getPath(), "boh=1" },
				{ "data=" + dtr.getPath(), "out=" + out2.getPath(), "epochs=abc" }, { "data=" + dtr.getPath(), "out=" + out2.getPath(), "senzauguale" },
				{ "data=" + dtr.getPath(), "out=" + out2.getPath(), "lambda=2" }, { "data=" + dtr.getPath(), "out=" + out2.getPath(), "weights=xx" },
				{ "data=" + dtr.getPath(), "out=" + out2.getPath(), "weights=99:1" }, { "data=" + dtr.getPath(), "out=" + out2.getPath(), "valpct=0" },
				{ "data=" + dtr.getPath(), "out=" + out2.getPath(), "init=" + init.getPath(), "hidden=8" }, { "data=" + dtr.getPath(), "out=" + out2.getPath(), "hidden=0" } };
		for (String[] b : bad2) {
			o = new StringBuilder();
			rc = run(Arrays.asList(b), o);
			check(rc == 2, "NnueTrainer " + Arrays.toString(b) + ": uscita " + rc + " (attesa 2): " + o);
		}
		o = new StringBuilder();
		rc = run(Arrays.asList("data=/nonexistent_dir_xyz/x.gz", "out=" + out2.getPath()), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueTrainer con dati inesistenti: uscita " + rc + " (attesa 1): " + o);
		final File junk = tmp(".gz");
		Files.write(junk.toPath(), "non e' gzip".getBytes("UTF-8"));
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + junk.getPath(), "out=" + out2.getPath()), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueTrainer con dati corrotti: uscita " + rc + " (attesa 1): " + o);
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "out=" + out2.getPath(), "init=/nonexistent_dir_xyz/x.nnue"), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueTrainer con pesi di partenza inesistenti: uscita " + rc + " (attesa 1): " + o);
		o = new StringBuilder();
		rc = run(Arrays.asList("data=" + dtr.getPath(), "out=" + out2.getPath(), "init=" + junk.getPath()), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueTrainer con pesi di partenza non validi: uscita " + rc + " (attesa 1): " + o);
	}
}
