package it.unibo.ai.didattica.competition.tablut.selftest;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP.Act;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP.Config;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;

/**
 * Test della rete {@link MLP}:
 * <ol>
 * <li><b>Forward</b> contro un'implementazione ingenua scritta qui (cicli
 * semplici sul layout documentato dei parametri): verifica il kernel a blocchi
 * di 4 neuroni, compresi i resti quando la larghezza non e' multipla di 4.</li>
 * <li><b>Gradiente</b> contro differenze finite centrali, su tutte le
 * architetture (A, B, C), tutte le attivazioni e entrambi i ruoli. Per le
 * attivazioni lineari a tratti, una discrepanza e' accettata SOLO se riprovata
 * con un passo molto piu' piccolo scompare (il passo grande aveva attraversato
 * un "ginocchio" della funzione): un vero errore resta in entrambi i casi.</li>
 * <li>Il gradiente di una posizione tocca <b>solo</b> tronco + head del suo
 * ruolo (la head dell'altro ruolo deve restare esattamente a zero).</li>
 * <li>Salvataggio/caricamento esatti; rifiuto dei file legacy e di quelli con
 * versione di feature diversa; copie indipendenti; thread-safety.</li>
 * </ol>
 */
public class MLPGradientCheck {

	static int failures = 0;

	public static void main(String[] args) throws Exception {
		Random rnd = new Random(7);
		List<Config> configs = new ArrayList<>();
		for (Act act : Act.values()) {
			configs.add(Config.sharedTrunk(12, new int[] { 8 }, new int[] { 4 }, act)); // B
			configs.add(Config.singleHeadRoleInput(12, new int[] { 8 }, new int[] { 4 }, act)); // A
			configs.add(Config.separateNets(12, new int[] { 8, 4 }, act)); // C
		}
		configs.add(Config.sharedTrunk(11, new int[] { 7, 5 }, new int[] { 6 }, Act.RELU)); // larghezze NON multiple di 4
		configs.add(Config.sharedTrunk(9, new int[] { 6 }, new int[0], Act.HARDTANH)); // head = solo neurone di uscita
		configs.add(NeuralEvaluator.defaultConfig()); // la rete di default (feature G_DEFAULT)
		configs.add(NeuralEvaluator.defaultConfig(TablutFeatures.G_LEAN)); // il tier leggero
		configs.add(new Config(5, new int[0], new int[0], 1, false, Act.RELU)); // modello lineare + tanh

		for (Config cfg : configs) {
			checkForward(cfg, rnd);
			checkGradient(cfg, rnd);
			checkGradientOnlyOnPath(cfg, rnd);
		}
		System.out.println("OK   - forward e gradiente verificati su " + configs.size() + " architetture");

		checkSaveLoad(rnd);
		checkCopyIndependence(rnd);
		checkThreadSafety(rnd);
		checkTrainingReducesError(rnd);

		if (failures == 0) {
			System.out.println("=== MLP CHECK OK ===");
		} else {
			System.out.println("=== MLP CHECK FALLITO: " + failures + " discrepanze ===");
			System.exit(1);
		}
	}

	static void fail(String m) {
		failures++;
		System.out.println("FAIL - " + m);
	}

	static double[] randomInput(Config cfg, Random rnd) {
		double[] x = new double[cfg.nIn];
		for (int i = 0; i < x.length; i++) {
			x[i] = rnd.nextDouble() * 2 - 0.5;
		}
		return x;
	}

	// ----------------------------------------------------------------- forward

	/** Forward ingenuo sul layout documentato: per strato [nOut][nIn] poi bias; ordine: tronco, head0, head1. */
	static double referenceForward(Config cfg, double[] p, double[] x, int role) {
		double[] a = new double[cfg.effectiveInputs()];
		System.arraycopy(x, 0, a, 0, cfg.nIn);
		if (cfg.roleInput) {
			a[cfg.nIn] = role == 0 ? 1.0 : -1.0;
		}
		int off = 0;
		int in = a.length;
		for (int h : cfg.trunk) {
			a = layer(p, off, in, h, a, cfg.act, true);
			off += in * h + h;
			in = h;
		}
		int trunkOut = in;
		int headParams = 0;
		int hin = trunkOut;
		for (int h : cfg.head) {
			headParams += hin * h + h;
			hin = h;
		}
		headParams += hin * 1 + 1;
		int myHead = cfg.heads == 2 ? role : 0;
		off += myHead * headParams;
		hin = trunkOut;
		for (int h : cfg.head) {
			a = layer(p, off, hin, h, a, cfg.act, true);
			off += hin * h + h;
			hin = h;
		}
		a = layer(p, off, hin, 1, a, cfg.act, false);
		return Math.tanh(a[0]);
	}

	static double[] layer(double[] p, int off, int in, int out, double[] x, Act act, boolean activate) {
		double[] y = new double[out];
		for (int i = 0; i < out; i++) {
			double s = p[off + in * out + i];
			for (int j = 0; j < in; j++) {
				s += p[off + i * in + j] * x[j];
			}
			if (activate) {
				switch (act) {
				case RELU:
					s = Math.max(0, s);
					break;
				case HARDTANH:
					s = Math.max(-1, Math.min(1, s));
					break;
				case LEAKY:
					s = s < 0 ? 0.05 * s : s;
					break;
				default:
					s = Math.tanh(s);
				}
			}
			y[i] = s;
		}
		return y;
	}

	static void checkForward(Config cfg, Random rnd) {
		MLP net = new MLP(cfg, rnd.nextLong());
		randomizeBiases(net, rnd);
		MLP.Scratch s = net.newScratch();
		for (int t = 0; t < 20; t++) {
			double[] x = randomInput(cfg, rnd);
			for (int role = 0; role < 2; role++) {
				double fast = net.forward(x, role, s);
				double ref = referenceForward(cfg, net.params(), x, role);
				if (Math.abs(fast - ref) > 1e-12) {
					fail("forward diverso dal riferimento (" + cfg + ", ruolo " + role + "): " + fast + " vs " + ref);
					return;
				}
			}
		}
	}

	static void randomizeBiases(MLP net, Random rnd) {
		// i bias partono a 0: li perturbiamo per testare anche il loro gradiente e il loro uso nel forward
		double[] p = net.params();
		for (int i = 0; i < p.length; i++) {
			p[i] += (rnd.nextDouble() - 0.5) * 0.3;
		}
	}

	// ---------------------------------------------------------------- gradiente

	static void checkGradient(Config cfg, Random rnd) {
		for (int trial = 0; trial < 3; trial++) {
			MLP net = new MLP(cfg, rnd.nextLong());
			randomizeBiases(net, rnd);
			double[] x = randomInput(cfg, rnd);
			int role = rnd.nextInt(2);
			MLP.Scratch s = net.newScratch();
			net.forward(x, role, s);
			double[] g = new double[net.numParams()];
			net.backward(s, 1.0, g);
			double[] p = net.params();
			int bad = 0;
			for (int i = 0; i < p.length; i++) {
				double num = numeric(net, x, role, i, 1e-6);
				if (!close(g[i], num)) {
					double num2 = numeric(net, x, role, i, 1e-9); // ginocchio attraversato?
					if (!close(g[i], num2)) {
						bad++;
						if (bad <= 3) {
							System.out.println("  param " + i + ": analitico=" + g[i] + " numerico(1e-6)=" + num
									+ " numerico(1e-9)=" + num2);
						}
					}
				}
			}
			if (bad > 0) {
				fail("gradiente errato in " + bad + " parametri (" + cfg + ", ruolo " + role + ")");
				return;
			}
		}
	}

	static boolean close(double a, double b) {
		return Math.abs(a - b) / Math.max(1.0, Math.max(Math.abs(a), Math.abs(b))) < 1e-5;
	}

	static double numeric(MLP net, double[] x, int role, int idx, double eps) {
		double[] p = net.params();
		MLP.Scratch s = net.newScratch();
		double orig = p[idx];
		p[idx] = orig + eps;
		double plus = net.forward(x, role, s);
		p[idx] = orig - eps;
		double minus = net.forward(x, role, s);
		p[idx] = orig;
		return (plus - minus) / (2 * eps);
	}

	static void checkGradientOnlyOnPath(Config cfg, Random rnd) {
		if (cfg.heads != 2) {
			return;
		}
		MLP net = new MLP(cfg, rnd.nextLong());
		randomizeBiases(net, rnd);
		double[] x = randomInput(cfg, rnd);
		MLP.Scratch s = net.newScratch();
		int total = net.numParams();
		double[][] g = new double[2][total];
		for (int role = 0; role < 2; role++) {
			net.forward(x, role, s);
			net.backward(s, 1.0, g[role]);
		}
		// i parametri toccati dai due ruoli devono differire SOLO per le head; il tronco e' in comune
		int touchedBoth = 0, touchedOnly0 = 0, touchedOnly1 = 0;
		for (int i = 0; i < total; i++) {
			boolean t0 = g[0][i] != 0, t1 = g[1][i] != 0;
			if (t0 && t1) {
				touchedBoth++;
			} else if (t0) {
				touchedOnly0++;
			} else if (t1) {
				touchedOnly1++;
			}
		}
		int headParams = (total - trunkParams(cfg)) / 2;
		if (cfg.trunk.length > 0 && touchedBoth == 0 && cfg.act != Act.RELU) {
			fail("il tronco non riceve gradiente da entrambi i ruoli (" + cfg + ")");
		}
		if (touchedOnly0 > headParams || touchedOnly1 > headParams) {
			fail("il gradiente di un ruolo ha toccato parametri della head dell'altro (" + cfg + ")");
		}
		// verifica forte: perturbare un parametro della head 1 non deve cambiare l'uscita del ruolo 0
		double before = net.forward(x, 0, s);
		double[] p = net.params();
		p[total - 1] += 5.0; // ultimo parametro = bias d'uscita della head 1
		double after = net.forward(x, 0, s);
		if (before != after) {
			fail("l'uscita del ruolo 0 dipende dalla head del ruolo 1 (" + cfg + ")");
		}
	}

	static int trunkParams(Config cfg) {
		int n = 0, in = cfg.effectiveInputs();
		for (int h : cfg.trunk) {
			n += in * h + h;
			in = h;
		}
		return n;
	}

	// -------------------------------------------------------------- persistenza

	static void checkSaveLoad(Random rnd) throws IOException {
		Config cfg = NeuralEvaluator.defaultConfig();
		MLP net = new MLP(cfg, 99);
		randomizeBiases(net, rnd);
		File f = File.createTempFile("mlp_check", ".txt");
		f.deleteOnExit();
		net.save(f.getPath(), TablutFeatures.VERSION, TablutFeatures.G_DEFAULT);
		MLP.Loaded l = MLP.load(f.getPath());
		boolean same = l.net.numParams() == net.numParams() && l.featureVersion == TablutFeatures.VERSION
				&& l.featureGroups == TablutFeatures.G_DEFAULT;
		for (int i = 0; same && i < net.numParams(); i++) {
			same = Double.doubleToLongBits(l.net.params()[i]) == Double.doubleToLongBits(net.params()[i]);
		}
		if (!same) {
			fail("salva/carica non e' esatto al bit");
		} else {
			System.out.println("OK   - salva/carica esatto al bit (" + net.numParams() + " parametri)");
		}

		// formato legacy -> rifiuto con messaggio chiaro
		File legacy = File.createTempFile("legacy", ".txt");
		legacy.deleteOnExit();
		try (FileWriter w = new FileWriter(legacy)) {
			w.write("sizes: 16 24 12 1\nW0\n0.1 0.2\n");
		}
		try {
			NeuralEvaluator.fromFile(legacy.getPath());
			fail("un file legacy NON e' stato rifiutato");
		} catch (IOException e) {
			System.out.println("OK   - file legacy rifiutato: " + e.getMessage().substring(0, 60) + "...");
		}

		// versione feature diversa -> rifiuto
		File old = File.createTempFile("oldver", ".txt");
		old.deleteOnExit();
		net.save(old.getPath(), TablutFeatures.VERSION + 1, TablutFeatures.G_DEFAULT);
		try {
			NeuralEvaluator.fromFile(old.getPath());
			fail("pesi con versione di feature diversa NON rifiutati");
		} catch (IOException e) {
			System.out.println("OK   - pesi con altra versione di feature rifiutati: " + e.getMessage());
		}
		// file valido -> carica e valuta
		NeuralEvaluator ev = NeuralEvaluator.fromFile(f.getPath());
		if (ev.net.numParams() != net.numParams()) {
			fail("NeuralEvaluator.fromFile ha caricato una rete diversa");
		}
	}

	static void checkCopyIndependence(Random rnd) {
		Config cfg = NeuralEvaluator.defaultConfig();
		MLP a = new MLP(cfg, 5);
		MLP b = a.copy();
		double[] x = randomInput(cfg, rnd);
		double before = a.forward(x, 0, a.newScratch());
		for (int i = 0; i < b.numParams(); i++) {
			b.params()[i] += 1.0;
		}
		double after = a.forward(x, 0, a.newScratch());
		double bv = b.forward(x, 0, b.newScratch());
		if (before != after) {
			fail("modificare la copia ha alterato l'originale");
		} else if (bv == before) {
			fail("la copia non e' stata davvero modificata (test inefficace)");
		} else {
			System.out.println("OK   - copy() indipendente");
		}
	}

	static void checkThreadSafety(Random rnd) throws Exception {
		Config cfg = NeuralEvaluator.defaultConfig();
		final MLP net = new MLP(cfg, 11);
		randomizeBiases(net, rnd);
		final int n = 2000;
		final double[][] xs = new double[n][];
		final double[] expected = new double[n];
		MLP.Scratch s = net.newScratch();
		for (int i = 0; i < n; i++) {
			xs[i] = randomInput(cfg, rnd);
			expected[i] = net.forward(xs[i], i & 1, s);
		}
		ExecutorService pool = Executors.newFixedThreadPool(4);
		List<Future<Integer>> fs = new ArrayList<>();
		for (int t = 0; t < 4; t++) {
			fs.add(pool.submit((Callable<Integer>) () -> {
				MLP.Scratch mine = net.newScratch();
				int bad = 0;
				for (int rep = 0; rep < 50; rep++) {
					for (int i = 0; i < n; i++) {
						if (net.forward(xs[i], i & 1, mine) != expected[i]) {
							bad++;
						}
					}
				}
				return bad;
			}));
		}
		int bad = 0;
		for (Future<Integer> f : fs) {
			bad += f.get();
		}
		pool.shutdown();
		if (bad > 0) {
			fail("forward concorrente su rete condivisa ha prodotto " + bad + " risultati diversi");
		} else {
			System.out.println("OK   - forward concorrente da 4 thread su una rete condivisa: risultati identici");
		}
	}

	/** Sanity: qualche passo di discesa del gradiente su un bersaglio fisso deve ridurre l'errore (segno del gradiente corretto). */
	static void checkTrainingReducesError(Random rnd) {
		Config cfg = Config.sharedTrunk(12, new int[] { 8 }, new int[] { 4 }, Act.RELU);
		MLP net = new MLP(cfg, 3);
		MLP.Scratch s = net.newScratch();
		double[] x = randomInput(cfg, rnd);
		double target = 0.6;
		double v0 = net.forward(x, 1, s);
		for (int it = 0; it < 200; it++) {
			double v = net.forward(x, 1, s);
			double[] g = new double[net.numParams()];
			net.backward(s, v - target, g);
			net.applyGradients(g, -0.1);
		}
		double v1 = net.forward(x, 1, s);
		if (Math.abs(v1 - target) >= Math.abs(v0 - target) || Math.abs(v1 - target) > 0.05) {
			fail("la discesa del gradiente non converge: " + v0 + " -> " + v1 + " (target " + target + ")");
		} else {
			System.out.println("OK   - discesa del gradiente: " + String.format("%.3f -> %.3f (target %.1f)", v0, v1, target));
		}
	}
}
