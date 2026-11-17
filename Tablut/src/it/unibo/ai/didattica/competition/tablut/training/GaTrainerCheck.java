package it.unibo.ai.didattica.competition.tablut.training;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;

/**
 * Verifica {@link GaTrainer}, {@link GaMatch} e il cancello, in tre livelli:
 * <ol>
 * <li><b>Meccanica della GA</b> su un paesaggio SINTETICO con ottimo noto (fitness = 1 - errore quadratico medio dal bersaglio, con o senza rumore): gli operatori
 * rispettano i limiti, la GA converge e batte una ricerca casuale a pari numero di valutazioni, e' deterministica, rispetta le tre condizioni di arresto,
 * conserva gli elitari, riprende da un checkpoint con lo STESSO risultato di una corsa ininterrotta, rifiuta checkpoint corrotti o di un'altra configurazione.</li>
 * <li><b>Statistica e cancello</b>: riepilogo/verdetto su coppie costruite a mano (limiti inclusi), semi di validazione disgiunti, regola di accettazione, file finali
 * (rifiuto = copia bit per bit della base; accettazione = il campione).</li>
 * <li><b>Con partite vere</b> (profondita' 2-3, reti piccole): due reti identiche fanno 0,5 esatto; la fitness vera vale 0,5 per gli individui neutri senza giocare;
 * l'arena e' identica con 1 o piu' thread; e una corsa completa {@code GaTrainer.run} produce tutti i file e un verdetto coerente.</li>
 * </ol>
 * Non richiede il server ne' porte. Uso: {@code GaTrainerCheck [pesi_nnue/trained_v6.nnue]} (se esiste, verifica anche che l'arena distingua la base da una rete casuale).
 */
public class GaTrainerCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String what) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + what);
		}
	}

	static final PrintStream QUIET = new PrintStream(new ByteArrayOutputStream());

	/** Paesaggio sintetico: raw = 1 - media((g-t)^2), con rumore gaussiano opzionale (deterministico per (gen, individuo)). */
	static final class Landscape implements GaTrainer.Fitness {
		final double[] target;
		final double noise;
		final long delayMs;
		final List<List<GaGenome>> pops = new ArrayList<>();
		final List<double[]> raws = new ArrayList<>();
		int evaluations = 0;

		Landscape(double[] target, double noise, long delayMs) {
			this.target = target;
			this.noise = noise;
			this.delayMs = delayMs;
		}

		double truth(GaGenome g) {
			double s = 0;
			for (int i = 0; i < target.length; i++) {
				s += (g.gene(i) - target[i]) * (g.gene(i) - target[i]);
			}
			return 1.0 - s / target.length;
		}

		@Override
		public double[] evaluate(List<GaGenome> inds, long[] seeds, int gen) throws Exception {
			if (delayMs > 0) {
				Thread.sleep(delayMs);
			}
			pops.add(new ArrayList<>(inds));
			final double[] r = new double[inds.size()];
			for (int i = 0; i < r.length; i++) {
				final Random n = new Random(GaTrainer.mix(gen, i, Arrays.hashCode(seeds)));
				r[i] = Math.max(0, Math.min(1, truth(inds.get(i)) + (noise > 0 ? n.nextGaussian() * noise : 0)));
				evaluations++;
			}
			raws.add(r);
			return r;
		}
	}

	/** Fitness piatta: tutti 0,5 (per la stagnazione). */
	static final class Flat implements GaTrainer.Fitness {
		@Override
		public double[] evaluate(List<GaGenome> inds, long[] seeds, int gen) {
			final double[] r = new double[inds.size()];
			Arrays.fill(r, 0.5);
			return r;
		}
	}

	static File tmpDir() throws IOException {
		return Files.createTempDirectory("gatrainercheck").toFile();
	}

	static void deleteAll(File d) {
		final File[] fs = d.listFiles();
		if (fs != null) {
			for (File f : fs) {
				if (f.isDirectory()) {
					deleteAll(f);
				} else {
					f.delete();
				}
			}
		}
		d.delete();
	}

	static GaTrainer.Config synthCfg(File dir, String base, String name) {
		final GaTrainer.Config c = new GaTrainer.Config();
		c.base = base;
		c.out = new File(dir, name).getPath();
		c.groups = "head,l1b";
		c.pop = 20;
		c.gens = 40;
		c.pairs = 2;
		c.sigma = 0.3;
		c.sigmaDecay = 0.97;
		c.lambda = 0.0;
		c.patience = 1000;
		return c;
	}

	static double[] target(int n, long seed) {
		final Random r = new Random(seed);
		final double[] t = new double[n];
		for (int i = 0; i < n; i++) {
			t[i] = (r.nextBoolean() ? 1 : -1) * (0.3 + 0.4 * r.nextDouble());
		}
		return t;
	}

	static GaMatch.Pair pair(double w, double b) {
		return new GaMatch.Pair(w, b);
	}

	static List<GaMatch.Pair> pairs(double... means) {
		final List<GaMatch.Pair> l = new ArrayList<>();
		for (double m : means) {
			l.add(pair(m, m));
		}
		return l;
	}

	public static void main(String[] args) throws Exception {
		final File dir = tmpDir();
		try {
			run(dir, args.length > 0 ? args[0] : "pesi_nnue/trained_v6.nnue");
		} finally {
			deleteAll(dir);
		}
		if (failures == 0) {
			System.out.println("=== GA TRAINER CHECK OK: " + checks + " verifiche ===");
		} else {
			System.out.println("=== FALLITI: " + failures + " su " + checks + " ===");
			System.exit(1);
		}
	}

	static void run(File dir, String v6Path) throws Exception {
		final NnueNetwork base = NnueNetwork.random(8, 4, 1);
		final File baseFile = new File(dir, "base.nnue");
		base.save(baseFile.getPath());
		final GaGenome.Layout L = GaGenome.Layout.of(base, "head,l1b");
		final int n = L.size();
		final Random rnd = new Random(7);

		// ------------------------------------------------------------ 1. operatori
		for (int t = 0; t < 2000; t++) {
			final double[] a = new double[n], b = new double[n];
			for (int i = 0; i < n; i++) {
				a[i] = rnd.nextDouble() * 2 - 1;
				b[i] = rnd.nextDouble() * 2 - 1;
			}
			final double alpha = rnd.nextDouble();
			final double[] c = GaTrainer.blx(a, b, alpha, rnd);
			boolean ok = true;
			for (int i = 0; i < n; i++) {
				final double lo = Math.min(a[i], b[i]), hi = Math.max(a[i], b[i]), d = hi - lo;
				ok &= c[i] >= Math.max(-1, lo - alpha * d) - 1e-12 && c[i] <= Math.min(1, hi + alpha * d) + 1e-12 && c[i] >= -1 && c[i] <= 1;
			}
			if (!ok) {
				check(false, "BLX: gene fuori da [min-alfa*d, max+alfa*d] o da [-1,1]");
				break;
			}
			if (t == 1999) {
				check(true, "BLX: 2000 incroci entro i limiti");
			}
		}
		{
			final double[] same = new double[n];
			Arrays.fill(same, 0.4);
			final double[] c = GaTrainer.blx(same, same, 0.5, rnd);
			check(Arrays.equals(c, same), "BLX con genitori uguali: figlio uguale (d=0)");
			// con alfa = 0 il figlio sta fra i genitori
			final double[] a = new double[n], b = new double[n];
			Arrays.fill(a, -0.2);
			Arrays.fill(b, 0.6);
			boolean ok = true;
			for (int t = 0; t < 200; t++) {
				for (double v : GaTrainer.blx(a, b, 0.0, rnd)) {
					ok &= v >= -0.2 && v <= 0.6;
				}
			}
			check(ok, "BLX alfa=0: figlio fra i genitori");
		}
		{
			final double[] g = new double[n];
			Arrays.fill(g, 0.99);
			boolean inRange = true;
			for (int t = 0; t < 500; t++) {
				final double[] c = g.clone();
				GaTrainer.mutate(c, 1.0, 5.0, false, rnd);
				for (double v : c) {
					inRange &= v >= -1 && v <= 1;
				}
			}
			check(inRange, "mutazione con sigma enorme: geni sempre in [-1,1]");
			final double[] z = new double[n];
			check(GaTrainer.mutate(z, 0.0, 0.1, false, rnd) == 0 && Arrays.equals(z, new double[n]), "mutazione con pm=0 senza forzatura: nessun cambiamento");
			final double[] f = new double[n];
			check(GaTrainer.mutate(f, 0.0, 0.1, true, rnd) == 1 && !Arrays.equals(f, new double[n]), "mutazione con pm=0 e forzatura: cambia esattamente un gene");
			int changedSum = 0;
			for (int t = 0; t < 400; t++) {
				changedSum += GaTrainer.mutate(new double[n], 0.25, 0.2, false, rnd);
			}
			final double rate = changedSum / (400.0 * n);
			check(rate > 0.20 && rate < 0.30, "mutazione pm=0.25: frequenza osservata " + rate);
			// ampiezza: deviazione standard dei cambiamenti ~ sigma
			double s2 = 0;
			int cnt = 0;
			for (int t = 0; t < 400; t++) {
				final double[] c = new double[n];
				GaTrainer.mutate(c, 1.0, 0.1, false, rnd);
				for (double v : c) {
					s2 += v * v;
					cnt++;
				}
			}
			final double sd = Math.sqrt(s2 / cnt);
			check(sd > 0.09 && sd < 0.11, "mutazione: deviazione osservata " + sd + " (attesa 0.1)");
		}
		{
			final double[] fit = new double[10];
			for (int i = 0; i < 10; i++) {
				fit[i] = i; // il migliore e' l'indice 9
			}
			int best = 0;
			for (int t = 0; t < 200; t++) {
				if (GaTrainer.tournament(fit, 60, rnd) == 9) {
					best++;
				}
			}
			check(best >= 195, "torneo con k grande: sceglie il migliore (" + best + "/200)");
			int[] hist = new int[10];
			for (int t = 0; t < 20000; t++) {
				hist[GaTrainer.tournament(fit, 1, rnd)]++;
			}
			boolean uniform = true;
			for (int h : hist) {
				uniform &= h > 1700 && h < 2300;
			}
			check(uniform, "torneo con k=1: scelta uniforme " + Arrays.toString(hist));
			int k3 = 0;
			for (int t = 0; t < 20000; t++) {
				if (GaTrainer.tournament(fit, 3, rnd) == 9) {
					k3++;
				}
			}
			final double expect = 1 - Math.pow(0.9, 3); // P(il migliore e' fra i 3 estratti)
			check(Math.abs(k3 / 20000.0 - expect) < 0.02, "torneo k=3: P(migliore)=" + k3 / 20000.0 + " attesa " + expect);
		}
		{
			final GaTrainer.Config c = new GaTrainer.Config();
			c.sigma = 0.2;
			c.sigmaDecay = 0.9;
			c.sigmaMin = 0.05;
			boolean mono = true, floor = true;
			for (int g = 1; g < 100; g++) {
				mono &= GaTrainer.sigmaAt(c, g) <= GaTrainer.sigmaAt(c, g - 1);
				floor &= GaTrainer.sigmaAt(c, g) >= 0.05 - 1e-15;
			}
			check(mono && floor && GaTrainer.sigmaAt(c, 0) == 0.2 && GaTrainer.sigmaAt(c, 99) == 0.05, "sigma: decrescente, parte da sigma, mai sotto sigmaMin");
		}
		{
			final GaGenome r = GaTrainer.randomIndividual(L, 0.2, rnd);
			boolean ok = true;
			for (double v : r.genes()) {
				ok &= v >= -1 && v <= 1;
			}
			check(ok && !r.isNeutral(), "individuo casuale: geni validi e non neutro");
		}

		// ------------------------------------------------------------ 2. semi
		{
			final Set<Long> train = new HashSet<>();
			for (int g = 0; g < 300; g++) {
				for (long s : GaTrainer.trainSeeds(1, g, 50)) {
					train.add(s);
				}
			}
			check(train.size() == 300 * 50, "semi di addestramento: 15000 distinti su 300 generazioni (" + train.size() + ")");
			boolean disjoint = true;
			for (long s : GaTrainer.validationSeeds(1, 500)) {
				disjoint &= !train.contains(s);
			}
			check(disjoint, "semi di validazione disgiunti da quelli di addestramento");
			check(Arrays.equals(GaTrainer.trainSeeds(5, 3, 10), GaTrainer.trainSeeds(5, 3, 10)) && !Arrays.equals(GaTrainer.trainSeeds(5, 3, 10), GaTrainer.trainSeeds(6, 3, 10))
					&& !Arrays.equals(GaTrainer.trainSeeds(5, 3, 10), GaTrainer.trainSeeds(5, 4, 10)), "semi: deterministici, diversi per seme e per generazione");
		}

		// ------------------------------------------------------------ 3. convergenza sul paesaggio sintetico
		final double[] tgt = target(n, 99);
		{
			final GaTrainer.Config cfg = synthCfg(dir, baseFile.getPath(), "conv");
			final Landscape ls = new Landscape(tgt, 0.0, 0);
			final GaTrainer.Result r = GaTrainer.run(cfg, base, ls, QUIET, false);
			final double idTruth = ls.truth(GaGenome.identity(L));
			final double champTruth = ls.truth(r.champion);
			System.out.printf("   sintetico senza rumore: identita' %.3f -> campione %.3f (%d valutazioni)%n", idTruth, champTruth, ls.evaluations);
			check(champTruth > idTruth + 0.15, "convergenza: campione " + champTruth + " molto meglio dell'identita' " + idTruth);
			check(champTruth > 0.9, "convergenza: campione vicino al bersaglio (" + champTruth + ")");
			check(r.stopReason.equals("gens") && r.generations == cfg.gens, "arresto per numero di generazioni: " + r.stopReason + " " + r.generations);
			// il miglior raw cresce (media delle ultime 5 > media delle prime 5)
			double first = 0, last = 0;
			for (int i = 0; i < 5; i++) {
				first += r.history.get(i).bestRaw / 5;
				last += r.history.get(r.history.size() - 1 - i).bestRaw / 5;
			}
			check(last > first + 0.1, "il miglior punteggio di generazione cresce: " + first + " -> " + last);
			// baseline: ricerca casuale con lo stesso numero di valutazioni attorno all'identita'
			final Random rr = new Random(1);
			double bestRandom = 0;
			for (int i = 0; i < ls.evaluations; i++) {
				bestRandom = Math.max(bestRandom, ls.truth(GaTrainer.randomIndividual(L, 0.3, rr)));
			}
			System.out.printf("   ricerca casuale a pari valutazioni: %.3f%n", bestRandom);
			check(champTruth > bestRandom + 0.02, "la GA batte la ricerca casuale a pari valutazioni (" + champTruth + " vs " + bestRandom + ")");
			check(new File(cfg.out + ".genome").isFile() && new File(cfg.out + ".final.nnue").isFile() && new File(cfg.out + ".ckpt").isFile(), "file di uscita scritti");
		}
		{
			final GaTrainer.Config cfg = synthCfg(dir, baseFile.getPath(), "noisy");
			cfg.lambda = 0.01;
			final Landscape ls = new Landscape(tgt, 0.05, 0);
			final GaTrainer.Result r = GaTrainer.run(cfg, base, ls, QUIET, false);
			final double idTruth = ls.truth(GaGenome.identity(L)), champTruth = ls.truth(r.champion);
			System.out.printf("   sintetico con rumore 0.05: identita' %.3f -> campione %.3f (valutato %d volte)%n", idTruth, champTruth, r.championEvals);
			check(champTruth > idTruth + 0.1, "con rumore: il campione (verita' " + champTruth + ") resta molto meglio dell'identita' (" + idTruth + ")");
			check(r.championEvals >= 2, "il campione e' scelto fra individui valutati almeno minEvals volte: " + r.championEvals);
		}

		// ------------------------------------------------------------ 4. determinismo
		{
			final GaTrainer.Result a = GaTrainer.run(synthCfg(dir, baseFile.getPath(), "det_a"), base, new Landscape(tgt, 0.03, 0), QUIET, false);
			final GaTrainer.Result b = GaTrainer.run(synthCfg(dir, baseFile.getPath(), "det_b"), base, new Landscape(tgt, 0.03, 0), QUIET, false);
			boolean sameHist = a.history.size() == b.history.size();
			for (int i = 0; sameHist && i < a.history.size(); i++) {
				sameHist &= a.history.get(i).bestRaw == b.history.get(i).bestRaw && a.history.get(i).meanRaw == b.history.get(i).meanRaw;
			}
			check(a.champion.equals(b.champion) && sameHist, "stesso seme: stesso campione e stessa storia, bit per bit");
			final GaTrainer.Config c2 = synthCfg(dir, baseFile.getPath(), "det_c");
			c2.seed = 2;
			final GaTrainer.Result c = GaTrainer.run(c2, base, new Landscape(tgt, 0.03, 0), QUIET, false);
			check(!c.champion.equals(a.champion), "seme diverso: campione diverso");
		}

		// ------------------------------------------------------------ 5. arresto: budget e stagnazione
		{
			final GaTrainer.Config cfg = synthCfg(dir, baseFile.getPath(), "budget");
			cfg.gens = 100;
			cfg.budgetSeconds = 1;
			final long t0 = System.currentTimeMillis();
			final GaTrainer.Result r = GaTrainer.run(cfg, base, new Landscape(tgt, 0, 400), QUIET, false);
			final long el = System.currentTimeMillis() - t0;
			check(r.stopReason.equals("budget") && r.generations >= 1 && r.generations <= 3, "arresto per budget: " + r.stopReason + " dopo " + r.generations + " generazioni, " + el + " ms");
			check(el < 2500, "il budget di 1 s non viene superato di molto (" + el + " ms)");
		}
		{
			final GaTrainer.Config cfg = synthCfg(dir, baseFile.getPath(), "patience");
			cfg.gens = 50;
			cfg.patience = 3;
			cfg.lambda = 0.02;
			final GaTrainer.Result r = GaTrainer.run(cfg, base, new Flat(), QUIET, false);
			check(r.stopReason.equals("patience") && r.generations >= 4 && r.generations <= 6, "arresto per stagnazione (fitness piatta): " + r.stopReason + " dopo " + r.generations);
			check(r.champion.isNeutral(), "senza nessun miglioramento il campione e' il riferimento (neutro)");
			check(r.finalGenome.isNeutral(), "e il genoma finale e' neutro");
		}

		// ------------------------------------------------------------ 6. invarianti della popolazione ed elitismo
		{
			final GaTrainer.Config cfg = synthCfg(dir, baseFile.getPath(), "elit");
			cfg.gens = 12;
			cfg.lambda = 0.02;
			final Landscape ls = new Landscape(tgt, 0.02, 0);
			GaTrainer.run(cfg, base, ls, QUIET, false);
			boolean size = true, bounds = true, elitism = true;
			check(ls.pops.get(0).get(0).isNeutral() && ls.pops.get(0).size() == cfg.pop, "generazione 0: il riferimento (neutro) e' l'individuo 0 e la popolazione ha pop individui");
			int nonNeutral = 0;
			for (GaGenome g : ls.pops.get(0)) {
				nonNeutral += g.isNeutral() ? 0 : 1;
			}
			check(nonNeutral == cfg.pop - 1, "generazione 0: gli altri sono tutti diversi dal riferimento");
			for (int k = 0; k + 1 < ls.pops.size(); k++) {
				size &= ls.pops.get(k).size() == cfg.pop;
				for (GaGenome g : ls.pops.get(k)) {
					for (double v : g.genes()) {
						bounds &= v >= -1 && v <= 1;
					}
				}
				// i `elite` migliori per fitness (raw - lambda*ms) devono comparire nella generazione successiva
				final double[] fit = new double[cfg.pop];
				for (int i = 0; i < cfg.pop; i++) {
					fit[i] = ls.raws.get(k)[i] - cfg.lambda * ls.pops.get(k).get(i).meanSquare();
				}
				final Integer[] ord = new Integer[cfg.pop];
				for (int i = 0; i < ord.length; i++) {
					ord[i] = i;
				}
				final double[] ff = fit;
				Arrays.sort(ord, (x, y) -> Double.compare(ff[y], ff[x]) != 0 ? Double.compare(ff[y], ff[x]) : Integer.compare(x, y));
				for (int e = 0; e < cfg.elite; e++) {
					elitism &= ls.pops.get(k + 1).get(e).equals(ls.pops.get(k).get(ord[e]));
				}
			}
			check(size && bounds, "popolazione di dimensione costante e geni in [-1,1] a ogni generazione");
			check(elitism, "gli elitari passano invariati alla generazione successiva (in testa)");
		}

		// ------------------------------------------------------------ 7. ripresa da checkpoint
		{
			final GaTrainer.Config full = synthCfg(dir, baseFile.getPath(), "res_full");
			full.gens = 8;
			final Landscape lf = new Landscape(tgt, 0.03, 0);
			final GaTrainer.Result rf = GaTrainer.run(full, base, lf, QUIET, false);
			final GaTrainer.Config part = synthCfg(dir, baseFile.getPath(), "res_part");
			part.gens = 4;
			GaTrainer.run(part, base, new Landscape(tgt, 0.03, 0), QUIET, false);
			part.gens = 8;
			part.resume = true;
			final Landscape lr = new Landscape(tgt, 0.03, 0);
			final GaTrainer.Result rr = GaTrainer.run(part, base, lr, QUIET, false);
			boolean same = lr.raws.size() == 4;
			for (int i = 0; same && i < 4; i++) {
				same &= Arrays.equals(lr.raws.get(i), lf.raws.get(4 + i));
			}
			check(same, "ripresa: le generazioni 4..7 ripartono dalla stessa popolazione e danno gli stessi punteggi dell'esecuzione ininterrotta");
			check(rr.champion.equals(rf.champion) && rr.generations == 8, "ripresa: stesso campione dell'esecuzione ininterrotta (" + rr.generations + " generazioni)");
			// checkpoint corrotto
			final File ck = new File(part.out + ".ckpt");
			final byte[] bytes = Files.readAllBytes(ck.toPath());
			Files.write(ck.toPath(), Arrays.copyOf(bytes, bytes.length / 2));
			try {
				GaTrainer.run(part, base, new Landscape(tgt, 0, 0), QUIET, false);
				check(false, "checkpoint troncato accettato");
			} catch (IOException e) {
				check(true, "checkpoint troncato rifiutato: " + e.getMessage());
			}
			final byte[] mod = bytes.clone();
			mod[bytes.length / 3] ^= 0x01;
			Files.write(ck.toPath(), mod);
			try {
				GaTrainer.run(part, base, new Landscape(tgt, 0, 0), QUIET, false);
				check(false, "checkpoint alterato accettato");
			} catch (IOException e) {
				check(true, "checkpoint alterato (checksum) rifiutato");
			}
			Files.write(ck.toPath(), bytes);
			part.pop = 22;
			try {
				GaTrainer.run(part, base, new Landscape(tgt, 0, 0), QUIET, false);
				check(false, "checkpoint di un'altra configurazione accettato");
			} catch (IOException e) {
				check(true, "checkpoint di un'altra configurazione (pop diverso) rifiutato");
			}
		}

		// ------------------------------------------------------------ 8. configurazione
		{
			final String[] bad = { "pop=3", "gens=0", "pairs=0", "depth=0", "elite=0", "pc=1.5", "sigma=0", "sigmamin=0.9", "lambda=2", "patience=0", "threads=0", "margin=0.6" };
			for (String b : bad) {
				final GaTrainer.Config c = new GaTrainer.Config();
				final String[] kv = b.split("=");
				switch (kv[0]) {
				case "pop": c.pop = Integer.parseInt(kv[1]); break;
				case "gens": c.gens = Integer.parseInt(kv[1]); break;
				case "pairs": c.pairs = Integer.parseInt(kv[1]); break;
				case "depth": c.depth = Integer.parseInt(kv[1]); break;
				case "elite": c.elite = Integer.parseInt(kv[1]); break;
				case "pc": c.pc = Double.parseDouble(kv[1]); break;
				case "sigma": c.sigma = Double.parseDouble(kv[1]); break;
				case "sigmamin": c.sigmaMin = Double.parseDouble(kv[1]); break;
				case "lambda": c.lambda = Double.parseDouble(kv[1]); break;
				case "patience": c.patience = Integer.parseInt(kv[1]); break;
				case "threads": c.threads = Integer.parseInt(kv[1]); break;
				default: c.margin = Double.parseDouble(kv[1]); break;
				}
				try {
					c.validate();
					check(false, "configurazione " + b + " accettata");
				} catch (IllegalArgumentException e) {
					check(true, "configurazione " + b + " rifiutata");
				}
			}
			try {
				new GaTrainer.Config().validate();
				check(true, "configurazione di default valida");
			} catch (IllegalArgumentException e) {
				check(false, "la configurazione di default non e' valida: " + e.getMessage());
			}
		}

		// ------------------------------------------------------------ 9. statistica e verdetto
		{
			final GaMatch.Summary s0 = GaMatch.summarize(pairs(), 1.96);
			check(s0.pairs == 0 && s0.mean == 0.5, "riepilogo di 0 coppie: media 0.5");
			final GaMatch.Summary w = GaMatch.summarize(Arrays.asList(pair(1, 0), pair(1, 1), pair(0.5, 0.5), pair(0, 0)), 1.96);
			check(w.wins == 3 && w.losses == 3 && w.draws == 2 && w.pairs == 4, "conteggio partite: +" + w.wins + " =" + w.draws + " -" + w.losses);
			check(Math.abs(w.mean - 0.5) < 1e-12, "media delle coppie " + w.mean);
			// 40 coppie con media nota e sd nota
			final double[] m = new double[40];
			for (int i = 0; i < 40; i++) {
				m[i] = (i % 2 == 0) ? 0.75 : 0.5;
			}
			final GaMatch.Summary s = GaMatch.summarize(pairs(m), 1.96);
			final double mean = 0.625, sd = Math.sqrt(40 * 0.125 * 0.125 / 39);
			check(Math.abs(s.mean - mean) < 1e-12 && Math.abs(s.sd - sd) < 1e-12 && Math.abs(s.se - sd / Math.sqrt(40)) < 1e-12, "media/sd/se corretti: " + s.mean + " " + s.sd + " " + s.se);
			check(Math.abs(s.lo - (mean - 1.96 * s.se)) < 1e-12 && Math.abs(s.hi - (mean + 1.96 * s.se)) < 1e-12, "IC = media +- z*se");
			check(GaMatch.SUPERIORE.equals(GaMatch.verdict(s, 0.02, 30)), "lo>0.5 -> SUPERIORE");
			check(GaMatch.INSUFFICIENTE.equals(GaMatch.verdict(s, 0.02, 41)), "meno di minPairs coppie -> INSUFFICIENTE");
			// tutte a 0.5: sd 0: mai SUPERIORE
			check(GaMatch.NON_INFERIORE.equals(GaMatch.verdict(GaMatch.summarize(pairs(fill(40, 0.5)), 1.96), 0.02, 30)), "40 coppie tutte pari -> NON_INFERIORE, mai SUPERIORE");
			// inferiore
			final double[] lo = new double[40];
			for (int i = 0; i < 40; i++) {
				lo[i] = i % 2 == 0 ? 0.25 : 0.5;
			}
			check(GaMatch.INFERIORE.equals(GaMatch.verdict(GaMatch.summarize(pairs(lo), 1.96), 0.02, 30)), "hi<0.5 -> INFERIORE");
			// non dimostrato: media 0.52, rumorosa
			final double[] nd = new double[40];
			for (int i = 0; i < 40; i++) {
				nd[i] = i % 2 == 0 ? 1.0 : 0.04;
			}
			final GaMatch.Summary snd = GaMatch.summarize(pairs(nd), 1.96);
			check(snd.lo < 0.5 && snd.hi > 0.5 && GaMatch.NON_DIMOSTRATO.equals(GaMatch.verdict(snd, 0.02, 30)), "IC a cavallo di 0.5 e troppo largo -> NON_DIMOSTRATO (" + snd + ")");
			check(GaMatch.NON_INFERIORE.equals(GaMatch.verdict(snd, 0.49, 30)), "con margine larghissimo lo stesso IC e' NON_INFERIORE");
			// regola di accettazione
			check(GaTrainer.isAccepted(GaMatch.SUPERIORE, false) && !GaTrainer.isAccepted(GaMatch.NON_INFERIORE, false) && GaTrainer.isAccepted(GaMatch.NON_INFERIORE, true), "accettazione: SUPERIORE sempre, NON_INFERIORE solo se richiesto");
			boolean rej = true;
			for (String v : new String[] { GaMatch.INSUFFICIENTE, GaMatch.INFERIORE, GaMatch.NON_DIMOSTRATO, "NEUTRO", null }) {
				rej &= !GaTrainer.isAccepted(v, false) && !GaTrainer.isAccepted(v, true);
			}
			check(rej, "INSUFFICIENTE / INFERIORE / NON_DIMOSTRATO / NEUTRO / null: sempre rifiutati");
		}

		// ------------------------------------------------------------ 10. file finali: rifiuto = base, accettazione = campione
		{
			final GaTrainer.Config cfg = synthCfg(dir, baseFile.getPath(), "outs");
			final GaGenome champ = new GaGenome(L, scaled(target(n, 5), 0.5));
			final GaTrainer.Result rej = new GaTrainer.Result();
			rej.champion = champ;
			rej.finalGenome = GaGenome.identity(L);
			rej.accepted = false;
			rej.verdict = GaMatch.NON_DIMOSTRATO;
			rej.stopReason = "gens";
			cfg.out = new File(dir, "out_rej").getPath();
			GaTrainer.writeOutputs(cfg, base, rej);
			check(Arrays.equals(Files.readAllBytes(new File(cfg.out + ".final.nnue").toPath()), Files.readAllBytes(baseFile.toPath())), "rifiuto: final.nnue e' una copia bit per bit della base");
			check(!Arrays.equals(Files.readAllBytes(new File(cfg.out + ".champion.nnue").toPath()), Files.readAllBytes(baseFile.toPath())), "rifiuto: champion.nnue e' comunque il campione (diverso dalla base)");
			final GaTrainer.Result acc = new GaTrainer.Result();
			acc.champion = champ;
			acc.finalGenome = champ;
			acc.accepted = true;
			acc.verdict = GaMatch.SUPERIORE;
			acc.stopReason = "gens";
			cfg.out = new File(dir, "out_acc").getPath();
			GaTrainer.writeOutputs(cfg, base, acc);
			check(Arrays.equals(Files.readAllBytes(new File(cfg.out + ".final.nnue").toPath()), Files.readAllBytes(new File(cfg.out + ".champion.nnue").toPath())), "accettazione: final.nnue = champion.nnue");
			check(GaGenome.load(cfg.out + ".genome", base).equals(champ), "il file .genome ricarica il campione");
			check(GaGenome.fingerprint(NnueNetwork.load(cfg.out + ".final.nnue")) == GaGenome.fingerprint(champ.decode()), "final.nnue contiene la rete del campione");
			final String vt = new String(Files.readAllBytes(new File(cfg.out + ".verdict.txt").toPath()), "UTF-8");
			check(vt.contains("ACCETTATO") && vt.contains("NON dimostra la forza a 60 s"), "verdict.txt riporta l'esito e l'avvertenza sulla forza a 60 s");
		}

		// ------------------------------------------------------------ 11. con partite vere
		{
			final NnueNetwork pb = NnueNetwork.random(16, 4, 3);
			final File pbf = new File(dir, "pb.nnue");
			pb.save(pbf.getPath());
			final NnueEvaluator ref = new NnueEvaluator(pb);
			final GaMatch.Config mc = new GaMatch.Config();
			mc.depth = 2;
			// due reti identiche: esattamente 0.5
			boolean exact = true;
			for (long s = 1; s <= 4; s++) {
				exact &= GaMatch.playPair(new NnueEvaluator(GaGenome.identity(GaGenome.Layout.of(pb, "head,l1b,ftb")).decode()), ref, mc, s).mean() == 0.5;
			}
			check(exact, "reti identiche: la coppia vale esattamente 0.5 (4 semi)");
			// fitness vera: i neutri valgono 0.5 senza giocare
			final GaGenome.Layout pl = GaGenome.Layout.of(pb, "head,l1b");
			final GaTrainer.Config fc = new GaTrainer.Config();
			fc.depth = 2;
			fc.threads = 2;
			final GaTrainer.MatchFitness mf = new GaTrainer.MatchFitness(pb, fc);
			final double[] tiny = new double[pl.size()];
			tiny[0] = 0.1 / pl.range(0);
			final List<GaGenome> inds = Arrays.asList(GaGenome.identity(pl), new GaGenome(pl, tiny), GaTrainer.randomIndividual(pl, 0.3, new Random(2)));
			final double[] raw = mf.evaluate(inds, new long[] { 11, 12 }, 0);
			check(raw[0] == 0.5 && raw[1] == 0.5, "fitness vera: neutro e quasi-neutro valgono 0.5");
			check(raw[2] >= 0 && raw[2] <= 1 && (raw[2] * 8) == Math.rint(raw[2] * 8), "fitness vera: punteggio di un individuo in {0, 1/8, ..., 1} con 2 coppie (" + raw[2] + ")");
			final double[] raw2 = new GaTrainer.MatchFitness(pb, fc).evaluate(inds, new long[] { 11, 12 }, 0);
			check(Arrays.equals(raw, raw2), "fitness vera riproducibile");
			// arena: stesso risultato con 1 o 3 thread
			final NnueEvaluator rndNet = new NnueEvaluator(NnueNetwork.random(16, 4, 77));
			final long[] seeds = { 1, 2, 3, 4 };
			final GaMatch.Summary a1 = GaMatch.summarize(GaMatch.playAll(rndNet, ref, mc, seeds, 1), 1.96), a3 = GaMatch.summarize(GaMatch.playAll(rndNet, ref, mc, seeds, 3), 1.96);
			check(a1.toString().equals(a3.toString()), "arena: stesso risultato con 1 e 3 thread (" + a1 + ")");
			// l'arena distingue davvero: la base vera contro una rete casuale
			if (new File(v6Path).isFile()) {
				final NnueEvaluator v6 = NnueEvaluator.fromFile(v6Path);
				final GaMatch.Config dc = new GaMatch.Config();
				dc.depth = 3;
				final GaMatch.Summary d = GaMatch.summarize(GaMatch.playAll(new NnueEvaluator(NnueNetwork.random(64, 16, 123)), v6, dc, new long[] { 1, 2, 3, 4 }, 1), 1.96);
				check(d.mean < 0.25, "l'arena distingue: una rete casuale contro la base vera fa " + d.mean);
			}
			// corsa completa con partite vere
			final GaTrainer.Config rc = new GaTrainer.Config();
			rc.base = pbf.getPath();
			rc.out = new File(dir, "real").getPath();
			rc.groups = "head,l1b";
			rc.pop = 6;
			rc.gens = 2;
			rc.pairs = 1;
			rc.depth = 2;
			rc.valPairs = 2;
			rc.minPairs = 1;
			rc.threads = 2;
			final ByteArrayOutputStream bo = new ByteArrayOutputStream();
			final GaTrainer.Result r = GaTrainer.run(rc, new PrintStream(bo, true, "UTF-8"));
			for (String ext : new String[] { ".genome", ".champion.nnue", ".final.nnue", ".verdict.txt", ".ckpt" }) {
				check(new File(rc.out + ext).isFile(), "corsa completa: scritto " + ext);
			}
			final byte[] fin = Files.readAllBytes(new File(rc.out + ".final.nnue").toPath());
			if (r.accepted) {
				check(Arrays.equals(fin, Files.readAllBytes(new File(rc.out + ".champion.nnue").toPath())), "corsa completa, accettata: final = campione");
			} else {
				check(Arrays.equals(fin, Files.readAllBytes(pbf.toPath())), "corsa completa, rifiutata: final = base bit per bit");
			}
			check(r.verdict != null && bo.toString("UTF-8").contains("VERDETTO") || "NEUTRO".equals(r.verdict), "corsa completa: verdetto nel log (" + r.verdict + ")");
			rc.out = new File(dir, "real_rej").getPath();
			rc.minPairs = 1000; // impossibile: il cancello deve rifiutare
			final GaTrainer.Result r2 = GaTrainer.run(rc, QUIET);
			check(!r2.accepted && (GaMatch.INSUFFICIENTE.equals(r2.verdict) || "NEUTRO".equals(r2.verdict)), "cancello con minPairs impossibile: rifiuta (" + r2.verdict + ")");
			check(Arrays.equals(Files.readAllBytes(new File(rc.out + ".final.nnue").toPath()), Files.readAllBytes(pbf.toPath())), "rifiuto: final.nnue = base bit per bit");
			check(r2.finalGenome.isNeutral(), "rifiuto: il genoma finale e' neutro");
		}
	}

	static double[] fill(int n, double v) {
		final double[] a = new double[n];
		Arrays.fill(a, v);
		return a;
	}

	static double[] scaled(double[] a, double f) {
		final double[] r = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			r[i] = a[i] * f;
		}
		return r;
	}
}
