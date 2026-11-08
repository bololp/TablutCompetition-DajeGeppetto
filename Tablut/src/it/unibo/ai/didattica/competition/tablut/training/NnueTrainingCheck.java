package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Test del distillatore ({@link NnueDistiller}).
 * <ol>
 * <li><b>Gradiente</b>: la passata indietro scritta a mano contro le differenze finite centrali, su TUTTI i parametri di una rete piccola
 * (e si esige che il gradiente sia non nullo su ogni gruppo di parametri: un test che confronta zeri con zeri non prova niente).</li>
 * <li><b>Quantizzazione</b>: la rete intera ottenuta dalla rete in virgola mobile da' gli stessi valori entro l'errore di arrotondamento;
 * i pesi che non stanno in {@code short} vengono saturati E contati.</li>
 * <li><b>Distillazione</b>: un maestro NNUE casuale (quindi imparabile esattamente dall'architettura) viene appreso: la perdita sul
 * held-out crolla e la rete quantizzata ne e' molto correlata.</li>
 * <li><b>Determinismo</b>: stesso seme = stessa rete, bit per bit; seme diverso = rete diversa.</li>
 * <li><b>Dati</b>: simmetrie corrette (la posizione iniziale e' simmetrica sotto tutte e 8), bersagli = valore del maestro sulla posizione
 * registrata, nessuna posizione senza Re, held-out separato per posizione di partenza.</li>
 * <li>Il {@code main} in un sottoprocesso: scrive un file che l'NNUE sa caricare; argomenti sbagliati = errore.</li>
 * </ol>
 */
public class NnueTrainingCheck {

	private static int failures = 0;
	private static long checks = 0;

	private static void check(boolean ok, String msg) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	public static void main(String[] args) throws Exception {
		testGradient();
		testQuantization();
		testSymmetryAndData();
		testSplit();
		testDistillsKnownTeacher();
		testDeterminism();
		testMainSubprocess();
		System.out.println(failures == 0 ? "=== NNUE TRAINING CHECK OK: " + checks + " verifiche ===" : "=== NNUE TRAINING CHECK: " + failures + " FALLIMENTI su " + checks + " ===");
		if (failures != 0) {
			System.exit(1);
		}
	}

	// ====================================================================== dati di prova

	private static NnueDistiller.Data samplesFromRandomPositions(Random rnd, int n, NnueEvaluator teacher) {
		final byte[] boards = new byte[81 * n];
		final byte[] role = new byte[n];
		final double[] target = new double[n];
		final int[] group = new int[n];
		for (int i = 0; i < n; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			System.arraycopy(fs.board, 0, boards, 81 * i, 81);
			role[i] = (byte) (fs.turn == BoardConstants.BLACK_TURN ? 1 : 0);
			target[i] = teacher == null ? 0 : teacher.evaluate(fs);
			group[i] = i;
		}
		return new NnueDistiller.Data(boards, role, target, group);
	}

	// ====================================================================== 1) gradiente

	private static void testGradient() {
		final NnueDistiller.FloatNet net = new NnueDistiller.FloatNet(5, 3);
		net.init(new Random(17));
		final Random rnd = new Random(18);
		final NnueDistiller.Data d = samplesFromRandomPositions(rnd, 8, null);
		for (int i = 0; i < d.n; i++) {
			d.target[i] = rnd.nextDouble() * 1.6 - 0.8;
		}
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(5, 3);
		final double[] g = new double[net.numParams()];
		for (int i = 0; i < d.n; i++) {
			NnueDistiller.sampleLoss(net, d, i, ws, g);
		}
		final double eps = 1e-6;
		int nonZero = 0, mismatches = 0, checked = 0;
		double worst = 0;
		final int[] groupNonZero = new int[4]; // ftW, ftB, strati 1, strato 2 e bias di uscita
		for (int k = 0; k < net.numParams(); k++) {
			final double keep = net.p[k];
			net.p[k] = keep + eps;
			double lp = 0;
			for (int i = 0; i < d.n; i++) {
				lp += NnueDistiller.sampleLoss(net, d, i, ws, null);
			}
			net.p[k] = keep - eps;
			double lm = 0;
			for (int i = 0; i < d.n; i++) {
				lm += NnueDistiller.sampleLoss(net, d, i, ws, null);
			}
			net.p[k] = keep;
			final double fd = (lp - lm) / (2 * eps);
			checked++;
			final double err = Math.abs(fd - g[k]);
			worst = Math.max(worst, err / (1e-6 + Math.abs(fd) + Math.abs(g[k])));
			if (err > 1e-6 + 1e-5 * Math.max(Math.abs(fd), Math.abs(g[k]))) {
				mismatches++;
				if (mismatches <= 5) {
					System.out.println("  gradiente " + k + ": analitico " + g[k] + " vs differenze finite " + fd);
				}
			}
			if (Math.abs(g[k]) > 1e-9) {
				nonZero++;
				groupNonZero[k < 243 * 5 ? 0 : k < 243 * 5 + 5 ? 1 : k < 243 * 5 + 5 + (15 + 3 + 3 + 1) ? 2 : 3]++;
			}
		}
		check(mismatches == 0, "gradiente: " + mismatches + " parametri su " + checked + " diversi dalle differenze finite (errore relativo massimo " + worst + ")");
		check(nonZero > 200, "gradiente: troppi pochi parametri con gradiente non nullo (" + nonZero + "): il confronto sarebbe vacuo");
		for (int gi = 0; gi < 4; gi++) {
			check(groupNonZero[gi] > 0, "gradiente: il gruppo di parametri " + gi + " ha gradiente sempre nullo (non verificato)");
		}
		System.out.println("gradiente: " + checked + " parametri, " + nonZero + " non nulli, errore relativo massimo " + worst);
	}

	// ====================================================================== 2) quantizzazione

	private static void testQuantization() {
		final NnueDistiller.FloatNet f = new NnueDistiller.FloatNet(32, 8);
		f.init(new Random(5));
		final NnueDistiller.Quantized q = NnueDistiller.quantize(f);
		check(q.clipped == 0, "quantizzazione di pesi normali: nessuna saturazione attesa, trovate " + q.clipped);
		final NnueEvaluator ev = new NnueEvaluator(q.net);
		ev.outputScale = 1.0;
		final Random rnd = new Random(6);
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(32, 8);
		double sum = 0, max = 0;
		final int n = 400;
		for (int i = 0; i < n; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			final int role = fs.turn == BoardConstants.BLACK_TURN ? 1 : 0;
			final double yf = Math.tanh(NnueDistiller.forward(f, fs.board, 0, role, ws));
			final double yq = ev.evaluate(fs);
			sum += Math.abs(yf - yq);
			max = Math.max(max, Math.abs(yf - yq));
		}
		check(sum / n < 0.01, "quantizzazione: errore medio sul valore " + sum / n + " (atteso < 0.01: arrotondamento dei pesi a 1/256 e 1/64)");
		check(max < 0.08, "quantizzazione: errore massimo sul valore " + max + " (atteso < 0.08)");
		System.out.printf("quantizzazione: errore medio %.5f, massimo %.4f sul valore in (-1,1)%n", sum / n, max);

		// pesi enormi: saturati E contati, mai in silenzio
		final NnueDistiller.FloatNet big = new NnueDistiller.FloatNet(4, 2);
		big.init(new Random(7));
		big.p[0] = 1000.0; // ftW * 256 = 256000 > 32767
		big.p[1] = -1000.0;
		final NnueDistiller.Quantized qb = NnueDistiller.quantize(big);
		check(qb.clipped >= 2, "pesi enormi: la saturazione deve essere contata (trovati " + qb.clipped + ")");
		check(qb.net.ftWeights()[0] == Short.MAX_VALUE && qb.net.ftWeights()[1] == Short.MIN_VALUE, "pesi enormi: saturati ai limiti di short");
	}

	// ====================================================================== 5) simmetrie e dati

	private static void testSymmetryAndData() {
		final FastState init = new FastState();
		for (int s = 0; s < 8; s++) {
			check(Arrays.equals(NnueDistiller.transformed(init, s).board, init.board), "la posizione iniziale e' simmetrica: la simmetria " + s + " deve lasciarla invariata");
		}
		final Random rnd = new Random(8);
		for (int i = 0; i < 40; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			final Set<String> distinct = new HashSet<>();
			for (int s = 0; s < 8; s++) {
				final FastState t = NnueDistiller.transformed(fs, s);
				check(count(t, BoardConstants.WHITE) == count(fs, BoardConstants.WHITE) && count(t, BoardConstants.BLACK) == count(fs, BoardConstants.BLACK)
						&& count(t, BoardConstants.KING) == 1, "simmetria " + s + ": i pezzi devono conservarsi");
				check(t.turn == fs.turn, "simmetria " + s + ": il lato al tratto deve conservarsi");
				distinct.add(Arrays.toString(t.board));
			}
			check(distinct.size() >= 2, "le 8 simmetrie di una posizione asimmetrica devono produrre scacchiere diverse");
			check(Arrays.equals(NnueDistiller.transformed(fs, 0).board, fs.board), "simmetria 0 = identita'");
		}

		final NnueEvaluator teacher = new NnueEvaluator(NnueNetwork.random(16, 4, 9));
		final NnueDistiller.Config cfg = new NnueDistiller.Config();
		cfg.games = 6;
		cfg.seed = 3;
		final NnueDistiller.Data d = NnueDistiller.generate(teacher, cfg, null);
		check(d.n > 200 && d.n % 8 == 0, "dati: attesi molti campioni, multipli di 8 (8 simmetrie), trovati " + d.n);
		final FastState probe = new FastState();
		boolean groupsOk = true, kingOk = true, labelOk = true, rangeOk = true;
		for (int i = 0; i < d.n; i++) {
			groupsOk &= d.group[i] == i / 8;
			int kings = 0, w = 0, b = 0;
			for (int sq = 0; sq < 81; sq++) {
				final int p = d.boards[81 * i + sq];
				kings += p == BoardConstants.KING ? 1 : 0;
				w += p == BoardConstants.WHITE ? 1 : 0;
				b += p == BoardConstants.BLACK ? 1 : 0;
			}
			kingOk &= kings == 1 && w <= 8 && b <= 16;
			rangeOk &= d.target[i] >= -1.0 && d.target[i] <= 1.0;
			System.arraycopy(d.boards, 81 * i, probe.board, 0, 81);
			probe.turn = d.role[i] == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
			final double expected = Math.max(-1.0, Math.min(1.0, teacher.evaluate(probe) / cfg.teacherScale));
			labelOk &= Math.abs(expected - d.target[i]) < 1e-12;
		}
		check(groupsOk, "dati: le 8 simmetrie di una stessa posizione devono formare un gruppo consecutivo");
		check(kingOk, "dati: ogni campione ha esattamente un Re (le posizioni terminali per cattura non si registrano) e un numero di pedine legale");
		check(rangeOk, "dati: i bersagli devono stare in [-1, 1]");
		check(labelOk, "dati: il bersaglio deve essere il valore del maestro sulla scacchiera registrata");
		// la generazione e' deterministica
		final NnueDistiller.Data d2 = NnueDistiller.generate(teacher, cfg, null);
		check(d.n == d2.n && Arrays.equals(d.boards, d2.boards) && Arrays.equals(d.target, d2.target), "dati: lo stesso seme deve dare gli stessi dati");
		cfg.augment = false;
		final NnueDistiller.Data d3 = NnueDistiller.generate(teacher, cfg, null);
		check(d3.n > 20 && d3.n < d.n, "senza augment i campioni sono 1/8");

		// un maestro la cui uscita ESCE dalla scala (es. SimpleEvaluator, o una vittoria forzata): i bersagli devono essere limitati a [-1, 1]
		final it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator wild = s -> (s.turn == BoardConstants.WHITE_TURN ? 1 : -1) * 5000.0;
		cfg.augment = true;
		final NnueDistiller.Data dw = NnueDistiller.generate(wild, cfg, null);
		boolean clamped = dw.n > 0;
		for (int i = 0; i < dw.n; i++) {
			clamped &= dw.target[i] == (dw.role[i] == 0 ? 1.0 : -1.0);
		}
		check(clamped, "dati: i bersagli fuori scala devono essere limitati a +-1");
	}

	private static int count(FastState s, int piece) {
		int n = 0;
		for (byte b : s.board) {
			n += b == piece ? 1 : 0;
		}
		return n;
	}

	private static void testSplit() {
		final Random rnd = new Random(10);
		final int n = 8000;
		final byte[] boards = new byte[81 * n];
		final int[] group = new int[n];
		for (int i = 0; i < n; i++) {
			group[i] = i / 8;
		}
		final NnueDistiller.Data d = new NnueDistiller.Data(boards, new byte[n], new double[n], group);
		final int[][] sp = NnueDistiller.split(d, 0.1, 5);
		final Set<Integer> trg = new HashSet<>(), vag = new HashSet<>();
		for (int i : sp[0]) {
			trg.add(d.group[i]);
		}
		for (int i : sp[1]) {
			vag.add(d.group[i]);
		}
		final Set<Integer> inter = new HashSet<>(trg);
		inter.retainAll(vag);
		check(inter.isEmpty(), "split: una posizione di partenza e' sia in train sia in held-out (" + inter.size() + " gruppi)");
		check(sp[0].length + sp[1].length == n, "split: ogni campione in una sola parte");
		final double frac = sp[1].length / (double) n;
		check(frac > 0.05 && frac < 0.16, "split: frazione di held-out " + frac + " lontana da 0.1");
		check(Arrays.equals(sp[1], NnueDistiller.split(d, 0.1, 5)[1]) && !Arrays.equals(sp[1], NnueDistiller.split(d, 0.1, 6)[1]),
				"split: deterministico a parita' di seme e diverso con un altro seme");
		rnd.nextInt();
	}

	// ====================================================================== 3) distilla un maestro imparabile

	private static void testDistillsKnownTeacher() {
		final NnueEvaluator teacher = new NnueEvaluator(NnueNetwork.random(16, 4, 31));
		final NnueDistiller.Config cfg = new NnueDistiller.Config();
		cfg.hidden = 16;
		cfg.l2 = 4;
		cfg.games = 90;
		cfg.epochs = 25;
		cfg.batch = 128;
		cfg.lr = 3e-3;
		cfg.seed = 11;
		final NnueDistiller.Data d = NnueDistiller.generate(teacher, cfg, null);
		final NnueDistiller.Result r = NnueDistiller.train(d, cfg, null);
		check(r.valLoss < 0.3 * r.initialValLoss, "distillazione: la perdita sul held-out deve calare almeno del 70%: " + r.initialValLoss + " -> " + r.valLoss);
		final NnueDistiller.Quantized q = NnueDistiller.quantize(r.net);
		check(q.clipped == 0, "distillazione: nessuna saturazione attesa (" + q.clipped + ")");
		final int[][] sp = NnueDistiller.split(d, cfg.valFraction, cfg.seed);
		final NnueDistiller.Metrics m = NnueDistiller.evaluate(q.net, d, sp[1]);
		System.out.println("distillazione di una NNUE casuale 16/4 (" + d.n + " campioni): perdita held-out " + r.initialValLoss + " -> " + r.valLoss + "; quantizzata: " + m);
		check(m.correlation > 0.8, "distillazione: correlazione sul held-out troppo bassa " + m.correlation);
		check(Math.abs(m.mse - r.valLoss) < 0.02, "distillazione: la rete quantizzata deve avere una perdita vicina a quella in virgola mobile (" + m.mse + " vs " + r.valLoss + ")");
		// il risultato e' una rete valida per il resto del sistema: file, ricarica, evaluator
		try {
			final File f = File.createTempFile("distilled_", ".nnue");
			f.deleteOnExit();
			new NnueEvaluator(q.net).save(f.getPath());
			check(NnueEvaluator.fromFile(f.getPath()).net.equals(q.net), "la rete distillata deve sopravvivere al salvataggio e alla ricarica");
		} catch (IOException e) {
			check(false, "salvataggio della rete distillata: " + e);
		}
	}

	// ====================================================================== 4) determinismo

	private static void testDeterminism() {
		final NnueEvaluator teacher = new NnueEvaluator(NnueNetwork.random(8, 4, 41));
		final NnueDistiller.Config cfg = new NnueDistiller.Config();
		cfg.hidden = 8;
		cfg.l2 = 4;
		cfg.games = 12;
		cfg.epochs = 4;
		cfg.seed = 2;
		final NnueDistiller.Data d = NnueDistiller.generate(teacher, cfg, null);
		final NnueNetwork a = NnueDistiller.quantize(NnueDistiller.train(d, cfg, null).net).net;
		final NnueNetwork b = NnueDistiller.quantize(NnueDistiller.train(d, cfg, null).net).net;
		check(a.equals(b), "determinismo: lo stesso seme deve dare la stessa rete");
		cfg.seed = 3;
		final NnueNetwork c = NnueDistiller.quantize(NnueDistiller.train(d, cfg, null).net).net;
		check(!a.equals(c), "determinismo: un seme diverso deve dare una rete diversa");
		// con valFraction = 0 il seme non puo' cambiare la divisione train/held-out: la differenza viene solo da inizializzazione e mescolamento
		cfg.valFraction = 0.0;
		cfg.seed = 2;
		final NnueNetwork e = NnueDistiller.quantize(NnueDistiller.train(d, cfg, null).net).net;
		cfg.seed = 3;
		final NnueNetwork f = NnueDistiller.quantize(NnueDistiller.train(d, cfg, null).net).net;
		check(!e.equals(f), "determinismo: il seme deve cambiare inizializzazione e mescolamento (anche senza held-out)");
	}

	// ====================================================================== 6) main in un sottoprocesso

	private static int run(List<String> args, StringBuilder out) throws Exception {
		final List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.add("-Xmx1g");
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.training.NnueDistiller");
		cmd.addAll(args);
		final ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
		final Process p = pb.start();
		final byte[] bytes = readAll(p.getInputStream());
		out.append(new String(bytes, "UTF-8"));
		if (!p.waitFor(300, TimeUnit.SECONDS)) {
			p.destroyForcibly();
			return -999;
		}
		return p.exitValue();
	}

	private static byte[] readAll(java.io.InputStream in) throws IOException {
		final java.io.ByteArrayOutputStream bo = new java.io.ByteArrayOutputStream();
		final byte[] buf = new byte[8192];
		int n;
		while ((n = in.read(buf)) > 0) {
			bo.write(buf, 0, n);
		}
		return bo.toByteArray();
	}

	private static void testMainSubprocess() throws Exception {
		final File teacher = File.createTempFile("teacher_", ".nnue");
		teacher.deleteOnExit();
		NnueNetwork.random(8, 4, 51).save(teacher.getPath());
		final File out = File.createTempFile("student_", ".nnue");
		out.deleteOnExit();
		final StringBuilder o = new StringBuilder();
		int rc = run(Arrays.asList("teacher=" + teacher.getPath(), "out=" + out.getPath(), "hidden=8", "l2=4", "games=6", "epochs=2", "seed=1"), o);
		check(rc == 0, "main: uscita " + rc + ". Output: " + o);
		check(o.toString().contains("held-out (rete quantizzata):") && o.toString().contains("scritto "), "main: deve stampare le metriche e il file scritto. Output: " + o);
		try {
			final NnueEvaluator e = NnueEvaluator.fromFile(out.getPath());
			check(e.net.hidden() == 8 && e.net.l2() == 4, "main: il file scritto deve avere le dimensioni richieste");
		} catch (IOException e) {
			check(false, "main: il file scritto non e' caricabile: " + e);
		}
		final String[][] bad = { { "out=x.nnue" }, { "teacher=" + teacher.getPath() }, { "teacher=" + teacher.getPath(), "out=" + out.getPath(), "hidden=0" },
				{ "teacher=" + teacher.getPath(), "out=" + out.getPath(), "epochs=abc" }, { "teacher=" + teacher.getPath(), "out=" + out.getPath(), "boh=1" },
				{ "teacher=" + teacher.getPath(), "out=" + out.getPath(), "senzauguale" } };
		for (String[] b : bad) {
			final StringBuilder ob = new StringBuilder();
			rc = run(Arrays.asList(b), ob);
			check(rc == 2, "main: gli argomenti " + Arrays.toString(b) + " dovrebbero dare uscita 2 (trovata " + rc + "): " + ob);
		}
		final StringBuilder om = new StringBuilder();
		rc = run(Arrays.asList("teacher=/nonexistent_dir_xyz/none.nnue", "out=" + out.getPath()), om);
		check(rc != 0, "main: un maestro inesistente deve fallire (uscita " + rc + ")");
		Files.deleteIfExists(out.toPath());
	}
}
