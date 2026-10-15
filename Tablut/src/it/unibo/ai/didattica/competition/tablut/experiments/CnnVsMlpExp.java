package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer.Dataset;

/**
 * Confronto A PARITA' DI DATI (stesso dataset, stesso split per partita, stesse epoche) tra MLP con feature a mano e varie forme di
 * CNN: l'errore di previsione dell'esito sulle partite tenute da parte. Misura la QUALITA' dell'evaluator, non la forza di gioco
 * (che dipende anche dal costo per valutazione: si veda {@code CnnBench} ed {@code EqualTimeArena}).
 * <p>
 * Uso: {@code CnnVsMlpExp data.gz [epochs=8] [seed=1] [threads=N] spec...} con spec =
 * <ul>
 * <li>{@code mlp} oppure {@code mlp:trunk=24;head=8}</li>
 * <li>{@code cnn:conv=8;reduce=2;dense=32;head=8;feat=1;aug=1;lr=0.002;save=pesi.txt} (liste con la virgola, campi separati da ';')</li>
 * </ul>
 * Ogni spec puo' avere {@code save=file}: salva i pesi nel formato letto da client e arena.
 */
public class CnnVsMlpExp {

	static Map<String, String> kv(String spec) {
		Map<String, String> m = new HashMap<>();
		int c = spec.indexOf(':');
		if (c < 0) {
			return m;
		}
		for (String s : spec.substring(c + 1).split(";")) {
			int eq = s.indexOf('=');
			if (eq > 0) {
				m.put(s.substring(0, eq), s.substring(eq + 1));
			}
		}
		return m;
	}

	public static void main(String[] a) throws Exception {
		if (a.length < 2) {
			System.out.println("Uso: CnnVsMlpExp data.gz [epochs=8] [seed=1] [threads=N] spec...");
			return;
		}
		Dataset d = Dataset.load(a[0]);
		int epochs = 8;
		long seed = 1;
		int threads = Runtime.getRuntime().availableProcessors();
		java.util.List<String> specs = new java.util.ArrayList<>();
		for (int i = 1; i < a.length; i++) {
			if (a[i].startsWith("epochs=")) {
				epochs = Integer.parseInt(a[i].substring(7));
			} else if (a[i].startsWith("seed=")) {
				seed = Long.parseLong(a[i].substring(5));
			} else if (a[i].startsWith("threads=")) {
				threads = Integer.parseInt(a[i].substring(8));
			} else {
				specs.add(a[i]);
			}
		}
		int[][] sp = splitOf(d);
		System.out.printf("Dataset %d posizioni (train %d / test %d), epoche=%d, seed=%d%n", d.n, sp[0].length, sp[1].length, epochs, seed);
		System.out.printf("%-62s %9s %10s %10s%n", "modello", "parametri", "MSE train", "MSE test");
		int groups = TablutFeatures.G_DEFAULT;
		for (String spec : specs) {
			Map<String, String> p = kv(spec);
			long t0 = System.currentTimeMillis();
			if (spec.startsWith("mlp")) {
				int[] trunk = ConvNet.parseList(p.getOrDefault("trunk", "24"));
				int[] head = ConvNet.parseList(p.getOrDefault("head", "8"));
				int[] cols = TablutFeatures.columnsFor(groups);
				MLP net = new MLP(MLP.Config.sharedTrunk(cols.length, trunk, head, MLP.Act.LEAKY), seed);
				double[] r = trainMlp(net, d, cols, sp[0], sp[1], epochs, Double.parseDouble(p.getOrDefault("lr", "0.003")), 128, seed);
				System.out.printf("%-62s %9d %10.4f %10.4f   (%.0fs)%n", "MLP " + net.config(), net.numParams(), r[0], r[1],
						(System.currentTimeMillis() - t0) / 1000.0);
				if (p.containsKey("save")) {
					new NeuralEvaluator(net, groups).save(p.get("save"));
				}
			} else {
				int[] conv = ConvNet.parseList(p.getOrDefault("conv", "8"));
				int reduce = Integer.parseInt(p.getOrDefault("reduce", "2"));
				int[] dense = ConvNet.parseList(p.getOrDefault("dense", "32"));
				int[] head = ConvNet.parseList(p.getOrDefault("head", "8"));
				boolean feat = !p.getOrDefault("feat", "1").equals("0");
				boolean aug = !p.getOrDefault("aug", "1").equals("0");
				double lr = Double.parseDouble(p.getOrDefault("lr", "0.002"));
				int[] cols = feat ? TablutFeatures.columnsFor(groups) : new int[0];
				ConvNet net = new ConvNet(new ConvNet.Config(conv, reduce, cols.length, dense, head), seed);
				double[] r = CnnPretrainer.train(net, d, cols, sp[0], sp[1], epochs, lr, 128, threads, aug, seed, null);
				System.out.printf("%-62s %9d %10.4f %10.4f   (%.0fs)%n", net.config() + (aug ? "" : " noaug"), net.numParams(), r[0], r[1],
						(System.currentTimeMillis() - t0) / 1000.0);
				if (p.containsKey("save")) {
					new CnnEvaluator(net, feat ? groups : 0).save(p.get("save"));
				}
			}
		}
	}

	/** Stesso split di {@link CnnPretrainer}: ~8% delle partite (indice di partita % 12 == 0) come test. */
	static int[][] splitOf(Dataset d) {
		int nt = 0;
		for (int i = 0; i < d.n; i++) {
			nt += d.game[i] % 12 == 0 ? 1 : 0;
		}
		int[] test = new int[nt], train = new int[d.n - nt];
		int x = 0, y = 0;
		for (int i = 0; i < d.n; i++) {
			if (d.game[i] % 12 == 0) {
				test[x++] = i;
			} else {
				train[y++] = i;
			}
		}
		return new int[][] { train, test };
	}

	/** Addestramento dell'MLP: la procedura di {@code SupervisedPretrainer} (Adam, coseno, batch 128) sullo stesso dataset. */
	static double[] trainMlp(MLP net, Dataset d, int[] cols, int[] train, int[] test, int epochs, double lr0, int batch, long seed) {
		double[] w = net.params(), g = new double[w.length], m = new double[w.length], v = new double[w.length];
		MLP.Scratch sc = net.newScratch();
		Random rnd = new Random(seed);
		int n = train.length;
		int[] idx = train.clone();
		double[] x = new double[cols.length];
		long step = 0;
		double trainMse = 0;
		for (int ep = 0; ep < epochs; ep++) {
			for (int i = n - 1; i > 0; i--) {
				int j = rnd.nextInt(i + 1);
				int t = idx[i];
				idx[i] = idx[j];
				idx[j] = t;
			}
			double lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs)), loss = 0;
			int cnt = 0;
			for (int b = 0; b + batch <= n; b += batch) {
				Arrays.fill(g, 0);
				for (int k = 0; k < batch; k++) {
					int i = idx[b + k];
					for (int j = 0; j < cols.length; j++) {
						x[j] = d.feats[i * TablutFeatures.NUM_FEATURES + cols[j]];
					}
					double e = net.forward(x, d.role[i], sc) - d.outcome[i];
					loss += e * e;
					cnt++;
					net.backward(sc, e, g);
				}
				step++;
				double c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
				for (int k = 0; k < w.length; k++) {
					double gk = g[k] / batch;
					m[k] = 0.9 * m[k] + 0.1 * gk;
					v[k] = 0.999 * v[k] + 0.001 * gk * gk;
					w[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
				}
			}
			trainMse = loss / Math.max(1, cnt);
		}
		double te = 0;
		for (int i : test) {
			for (int j = 0; j < cols.length; j++) {
				x[j] = d.feats[i * TablutFeatures.NUM_FEATURES + cols[j]];
			}
			double e = net.forward(x, d.role[i], sc) - d.outcome[i];
			te += e * e;
		}
		return new double[] { trainMse, te / Math.max(1, test.length) };
	}
}
