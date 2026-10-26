package it.unibo.ai.didattica.competition.tablut.selftest;

import java.io.DataOutputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.FeatureRegistry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.LegacyTablutFeaturesV3;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.aiplayer.WeightConverter;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer;
import it.unibo.ai.didattica.competition.tablut.training.DatasetUpgrade;

/**
 * Verifica {@link WeightConverter}: la rete convertita deve calcolare la STESSA funzione della rete originale.
 *
 * <p>Oracolo indipendente: l'estrattore v3 congelato ({@link LegacyTablutFeaturesV3}, copia del codice com'era prima della v4).
 * La rete VECCHIA viene valutata con le feature v3; la rete CONVERTITA con le feature v4 di {@link TablutFeatures}; le uscite
 * devono coincidere (tolleranza di arrotondamento float). Cosi' si verifica insieme: l'ordine degli id nel registro, i DIV, le
 * identita' lineari usate per ripiegare le colonne rimosse, la mappatura delle colonne nel layout della CNN e dell'MLP, e lo
 * spostamento della colonna del ruolo.
 *
 * <p>Uso: {@code FeatureConversionCheck [posizioni] [seed] [file_pesi_cnn_reali]}.
 */
public class FeatureConversionCheck {

	static int failures = 0;
	static final double TOL = 5e-5; // uscita tanh in (-1,1): differenze da riordino di somme float

	static void fail(String msg) {
		failures++;
		System.out.println("FAIL - " + msg);
	}

	static void ok(String msg) {
		System.out.println("OK   - " + msg);
	}

	public static void main(String[] args) throws Exception {
		int n = args.length > 0 ? Integer.parseInt(args[0]) : 1500;
		long seed = args.length > 1 ? Long.parseLong(args[1]) : 99L;
		String realCnn = args.length > 2 ? args[2] : null;
		List<State> positions = FeaturesCheck.collectPositions(n, seed);
		System.out.println("Posizioni: " + positions.size());

		checkRegistry(positions);
		checkPlans();
		for (int fromGroups : new int[] { 2, 11, 15 }) {
			for (int toGroups : new int[] { -1, TablutFeatures.G_ALL }) {
				checkCnn(positions, fromGroups, toGroups, seed + fromGroups);
			}
		}
		for (int fromGroups : new int[] { 2, 11, 15 }) {
			checkMlp(positions, 3, fromGroups, -1, seed + 100 + fromGroups, true);
		}
		checkMlp(positions, 3, 11, TablutFeatures.G_ALL, seed + 7, false);
		checkMlp(positions, 3, 11, -1, seed + 8, false); // senza ruolo come ingresso
		checkMlpV2(positions, seed + 9);
		checkChangedDiv(positions, seed + 12);
		checkLossy(positions, seed + 10);
		checkFiles(positions, seed + 11);
		checkDatasetUpgrade();
		if (realCnn != null && new File(realCnn).exists()) {
			checkRealCnn(positions, realCnn);
		} else {
			System.out.println("(pesi reali non forniti/non trovati: salto il confronto sul file reale)");
		}

		if (failures == 0) {
			System.out.println("=== FEATURE CONVERSION CHECK OK ===");
		} else {
			System.out.println("=== FEATURE CONVERSION CHECK FALLITO: " + failures + " problemi ===");
			System.exit(1);
		}
	}

	// ---------------------------------------------------------------- registro

	/** Ogni id v3 ha lo stesso valore grezzo nell'estrattore v3 e in quello v4, oppure e' una identita' lineare esatta. */
	static void checkRegistry(List<State> positions) {
		FeatureRegistry.Schema v3 = FeatureRegistry.forVersion(3), cur = FeatureRegistry.current();
		if (v3.size() != LegacyTablutFeaturesV3.NUM_FEATURES) {
			fail("schema v3: " + v3.size() + " id, l'estrattore v3 ne ha " + LegacyTablutFeaturesV3.NUM_FEATURES);
			return;
		}
		if (!Arrays.equals(cur.ids, TablutFeatures.IDS)) {
			fail("lo schema corrente non coincide con TablutFeatures.IDS");
		}
		java.util.Map<String, Integer> pos = new java.util.HashMap<>();
		for (int i = 0; i < cur.size(); i++) {
			pos.put(cur.ids[i], i);
		}
		LegacyTablutFeaturesV3.Workspace lws = LegacyTablutFeaturesV3.newWorkspace();
		TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
		double[] lf = new double[LegacyTablutFeaturesV3.NUM_FEATURES], nf = new double[TablutFeatures.NUM_FEATURES];
		int bad = 0, same = 0, derived = 0;
		for (State s : positions) {
			LegacyTablutFeaturesV3.extract(s, lws, lf, LegacyTablutFeaturesV3.G_ALL);
			TablutFeatures.extract(s, ws, nf, TablutFeatures.G_ALL);
			for (int i = 0; i < v3.size(); i++) {
				double rawOld = lf[i] * v3.div[i];
				Integer j = pos.get(v3.ids[i]);
				double expected;
				if (j != null) {
					expected = nf[j] * cur.div[j];
					same++;
				} else {
					Map<String, Double> der = FeatureRegistry.DERIVED.get(v3.ids[i]);
					if (der == null) {
						fail("l'id v3 " + v3.ids[i] + " non esiste piu' e non ha un'identita' derivata");
						return;
					}
					expected = 0;
					for (Map.Entry<String, Double> e : der.entrySet()) {
						int d = pos.get(e.getKey());
						expected += e.getValue() * nf[d] * cur.div[d];
					}
					derived++;
				}
				if (Math.abs(rawOld - expected) > 1e-9) {
					if (bad++ < 5) {
						fail("id " + v3.ids[i] + ": valore grezzo v3 " + rawOld + " != atteso " + expected);
					}
				}
			}
		}
		if (bad == 0) {
			ok("registro: " + same + " valori grezzi identici fra v3 e v4, " + derived + " ricavati dalle identita' lineari (" + positions.size() + " posizioni)");
		}
		// gruppi congelati coerenti con quelli dell'estrattore v3
		for (int g : new int[] { 0, 1, 2, 3, 8, 11, 15 }) {
			if (!Arrays.equals(v3.columnsFor(g), LegacyTablutFeaturesV3.columnsFor(g))) {
				fail("columnsFor(" + g + ") dello schema v3 diverso da quello dell'estrattore v3");
			}
		}
		for (int g = 0; g <= TablutFeatures.G_ALL; g++) {
			if (!Arrays.equals(cur.columnsFor(g), TablutFeatures.columnsFor(g))) {
				fail("columnsFor(" + g + ") dello schema corrente diverso da TablutFeatures");
			}
		}
	}

	static void checkPlans() {
		WeightConverter.Plan p = new WeightConverter.Plan(3, 11, TablutFeatures.G_DEFAULT);
		boolean good = p.isExact() && p.folded.size() == 3 && p.added.size() == 6 && p.oldIds.length == 41 && p.newIds.length == 44;
		if (!good) {
			fail("piano v3/11 -> default inatteso:\n" + p.describe());
		} else {
			ok("piano v3 (41 col) -> v4 (44 col): 6 aggiunte, 3 ripiegate, 0 perse");
		}
		WeightConverter.Plan q = new WeightConverter.Plan(3, 15, TablutFeatures.G_DEFAULT); // toglie FLOOD
		if (q.isExact() || q.lost.size() != 3) {
			fail("togliendo FLOOD servono 3 colonne perse, trovate: " + q.lost);
		} else {
			ok("togliere un gruppo viene dichiarato come NON esatto (" + q.lost + ")");
		}
		try {
			new WeightConverter.Plan(1, 11, 11);
			fail("la versione 1 non dovrebbe essere accettata");
		} catch (IllegalArgumentException e) {
			ok("versione sconosciuta rifiutata");
		}
	}

	// ---------------------------------------------------------------- CNN

	static ConvNet randomCnn(int nFeat, long seed) {
		ConvNet.Config cfg = new ConvNet.Config(new int[] { 4 }, 2, nFeat, new int[] { 32 }, new int[] { 8 });
		ConvNet net = new ConvNet(cfg, seed);
		Random r = new Random(seed * 31 + 1);
		float[] p = net.params();
		for (int i = 0; i < p.length; i++) { // rumore: tutti i pesi (compresi bias e head) non banali
			p[i] += (float) (r.nextGaussian() * 0.15);
		}
		return net;
	}

	static void checkCnn(List<State> positions, int fromGroups, int toGroups, long seed) {
		ConvNet old = randomCnn(LegacyTablutFeaturesV3.columnsFor(fromGroups).length, seed);
		WeightConverter.Plan plan = new WeightConverter.Plan(3, fromGroups, toGroups >= 0 ? toGroups : (fromGroups | WeightConverter.V4_NEW_GROUPS));
		if (!plan.isExact()) {
			fail("piano non esatto per CNN " + fromGroups + "->" + toGroups + ": " + plan.lost);
			return;
		}
		ConvNet conv = WeightConverter.convertCnn(old, plan);
		double maxDiff = compareCnn(positions, old, fromGroups, conv, plan.toGroups);
		if (maxDiff > TOL) {
			fail("CNN gruppi " + fromGroups + " -> " + plan.toGroups + ": differenza massima " + maxDiff);
		} else {
			ok(String.format("CNN gruppi %d -> %d (%d -> %d feature): stessa funzione su %d posizioni, diff max %.2e", fromGroups,
					plan.toGroups, plan.oldIds.length, plan.newIds.length, positions.size(), maxDiff));
		}
	}

	/** max |old(feature v3) - converted(feature v4)| sull'uscita della rete (prima della scala). */
	static double compareCnn(List<State> positions, ConvNet old, int oldGroups, ConvNet conv, int newGroups) {
		ConvNet.Workspace wo = old.newWorkspace(), wc = conv.newWorkspace();
		LegacyTablutFeaturesV3.Workspace lws = LegacyTablutFeaturesV3.newWorkspace();
		TablutFeatures.Workspace nws = TablutFeatures.newWorkspace();
		int[] oc = LegacyTablutFeaturesV3.columnsFor(oldGroups), nc = TablutFeatures.columnsFor(newGroups);
		double[] lf = new double[LegacyTablutFeaturesV3.NUM_FEATURES], nf = new double[TablutFeatures.NUM_FEATURES];
		float[] fo = new float[oc.length], fc = new float[nc.length];
		double max = 0;
		for (State s : positions) {
			FastState fs = FastState.fromState(s);
			wo.setBoard(fs);
			wc.setBoard(fs);
			LegacyTablutFeaturesV3.extract(s, lws, lf, oldGroups);
			TablutFeatures.extract(s, nws, nf, newGroups);
			for (int i = 0; i < oc.length; i++) {
				fo[i] = (float) lf[oc[i]];
			}
			for (int i = 0; i < nc.length; i++) {
				fc[i] = (float) nf[nc[i]];
			}
			int role = NeuralEvaluator.roleOf(fs);
			max = Math.max(max, Math.abs(old.forward(wo, fo, role) - conv.forward(wc, fc, role)));
		}
		return max;
	}

	// ---------------------------------------------------------------- MLP

	static MLP randomMlp(int nIn, boolean roleInput, long seed) {
		MLP.Config cfg = roleInput ? MLP.Config.singleHeadRoleInput(nIn, new int[] { 24 }, new int[] { 8 }, MLP.Act.LEAKY)
				: MLP.Config.sharedTrunk(nIn, new int[] { 24 }, new int[] { 8 }, MLP.Act.LEAKY);
		MLP net = new MLP(cfg, seed);
		Random r = new Random(seed * 17 + 3);
		double[] p = net.params();
		for (int i = 0; i < p.length; i++) {
			p[i] += r.nextGaussian() * 0.15;
		}
		return net;
	}

	/** v3 -> v4 per un MLP (con ingresso "ruolo" nel primo caso: la sua colonna deve spostarsi in coda). */
	static void checkMlp(List<State> positions, int fromVersion, int fromGroups, int toGroups, long seed, boolean roleInput) {
		MLP old = randomMlp(LegacyTablutFeaturesV3.columnsFor(fromGroups).length, roleInput, seed);
		WeightConverter.Plan plan = new WeightConverter.Plan(fromVersion, fromGroups, toGroups >= 0 ? toGroups : (fromGroups | WeightConverter.V4_NEW_GROUPS));
		MLP conv = WeightConverter.convertMlp(old, plan);
		double max = compareMlp(positions, old, LegacyTablutFeaturesV3.columnsFor(fromGroups), conv, TablutFeatures.columnsFor(plan.toGroups),
				fromGroups, plan.toGroups, -1);
		if (max > TOL) {
			fail("MLP (ruolo come ingresso=" + roleInput + ") gruppi " + fromGroups + " -> " + plan.toGroups + ": diff max " + max);
		} else {
			ok(String.format("MLP %s gruppi %d -> %d: stessa funzione su %d posizioni, diff max %.2e", roleInput ? "(ruolo in ingresso)" : "(2 head)",
					fromGroups, plan.toGroups, positions.size(), max));
		}
	}

	/** Un MLP v2 e' il v3 senza WHITE_RING4: gli ingressi vecchi si ottengono dall'estrattore v3 saltando quella colonna. */
	static void checkMlpV2(List<State> positions, long seed) {
		int groups = 11;
		int[] c3 = LegacyTablutFeaturesV3.columnsFor(groups);
		int skip = Arrays.asList(FeatureRegistry.forVersion(3).ids).indexOf("WHITE_RING4");
		int[] c2 = Arrays.stream(c3).filter(c -> c != skip).toArray();
		MLP old = randomMlp(c2.length, false, seed);
		WeightConverter.Plan plan = new WeightConverter.Plan(2, groups, groups | WeightConverter.V4_NEW_GROUPS);
		if (!plan.isExact() || plan.oldIds.length != c2.length) {
			fail("piano v2 inatteso: " + plan.describe());
			return;
		}
		MLP conv = WeightConverter.convertMlp(old, plan);
		double max = compareMlp(positions, old, c2, conv, TablutFeatures.columnsFor(plan.toGroups), groups, plan.toGroups, skip);
		if (max > TOL) {
			fail("MLP v2 -> v4: diff max " + max);
		} else {
			ok(String.format("MLP v2 (40 col) -> v4 (44 col): stessa funzione su %d posizioni, diff max %.2e", positions.size(), max));
		}
	}

	static double compareMlp(List<State> positions, MLP old, int[] oldCols, MLP conv, int[] newCols, int oldGroups, int newGroups, int unused) {
		MLP.Scratch so = old.newScratch(), sc = conv.newScratch();
		LegacyTablutFeaturesV3.Workspace lws = LegacyTablutFeaturesV3.newWorkspace();
		TablutFeatures.Workspace nws = TablutFeatures.newWorkspace();
		double[] lf = new double[LegacyTablutFeaturesV3.NUM_FEATURES], nf = new double[TablutFeatures.NUM_FEATURES];
		double[] xo = new double[oldCols.length], xc = new double[newCols.length];
		double max = 0;
		for (State s : positions) {
			FastState fs = FastState.fromState(s);
			LegacyTablutFeaturesV3.extract(s, lws, lf, LegacyTablutFeaturesV3.G_ALL);
			TablutFeatures.extract(s, nws, nf, newGroups);
			for (int i = 0; i < oldCols.length; i++) {
				xo[i] = lf[oldCols[i]];
			}
			for (int i = 0; i < newCols.length; i++) {
				xc[i] = nf[newCols[i]];
			}
			int role = NeuralEvaluator.roleOf(fs);
			max = Math.max(max, Math.abs(old.forward(xo, role, so) - conv.forward(xc, role, sc)));
		}
		return max;
	}

	// ---------------------------------------------------------------- casi limite

	/**
	 * Schema "vecchio" sintetico: gli stessi id dello schema corrente ma con DIV diversi (e un id in meno, derivabile). La
	 * rete vecchia legge grezzo/divVecchio, la nuova grezzo/divNuovo: la conversione deve riscalare i pesi per il rapporto.
	 * (Fra v3 e v4 tutti i DIV in comune coincidono, quindi senza questo test il rapporto non sarebbe mai esercitato.)
	 */
	static void checkChangedDiv(List<State> positions, long seed) {
		FeatureRegistry.Schema cur = FeatureRegistry.current();
		Random r = new Random(seed);
		int groups = TablutFeatures.G_DEFAULT;
		double[] div = cur.div.clone();
		for (int i = 0; i < div.length; i++) {
			div[i] *= new double[] { 0.5, 2.0, 3.0, 1.0, 0.25 }[r.nextInt(5)];
		}
		FeatureRegistry.Schema oldS = new FeatureRegistry.Schema(77, cur.ids, cur.group, div);
		WeightConverter.Plan plan = new WeightConverter.Plan(oldS, groups, cur, groups);
		int[] cols = cur.columnsFor(groups);
		MLP old = randomMlp(cols.length, true, seed);
		MLP conv = WeightConverter.convertMlp(old, plan);
		MLP.Scratch so = old.newScratch(), sc = conv.newScratch();
		TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
		double[] nf = new double[TablutFeatures.NUM_FEATURES], xo = new double[cols.length], xc = new double[cols.length];
		double max = 0;
		for (State s : positions) {
			FastState fs = FastState.fromState(s);
			TablutFeatures.extract(s, ws, nf, groups);
			for (int i = 0; i < cols.length; i++) {
				double raw = nf[cols[i]] * cur.div[cols[i]];
				xo[i] = raw / oldS.div[cols[i]]; // cio' che la rete vecchia vedeva
				xc[i] = nf[cols[i]]; // cio' che vede la nuova
			}
			int role = NeuralEvaluator.roleOf(fs);
			max = Math.max(max, Math.abs(old.forward(xo, role, so) - conv.forward(xc, role, sc)));
		}
		if (max > TOL) {
			fail("DIV diversi fra le versioni: diff max " + max);
		} else {
			ok(String.format("stesso id con DIV diverso: i pesi vengono riscalati e la funzione resta identica (diff max %.2e)", max));
		}
		// e con una colonna ripiegata sopra DIV diversi
		FeatureRegistry.Schema v3 = FeatureRegistry.forVersion(3);
		double[] d3 = v3.div.clone();
		for (int i = 0; i < d3.length; i++) {
			d3[i] *= 2.0;
		}
		FeatureRegistry.Schema oldV3 = new FeatureRegistry.Schema(78, v3.ids, v3.group, d3);
		WeightConverter.Plan p3 = new WeightConverter.Plan(oldV3, 11, cur, groups);
		int[] oc = oldV3.columnsFor(11);
		ConvNet oldC = randomCnn(oc.length, seed + 1);
		ConvNet convC = WeightConverter.convertCnn(oldC, p3);
		ConvNet.Workspace wo = oldC.newWorkspace(), wc = convC.newWorkspace();
		LegacyTablutFeaturesV3.Workspace lws = LegacyTablutFeaturesV3.newWorkspace();
		double[] lf = new double[LegacyTablutFeaturesV3.NUM_FEATURES];
		float[] fo = new float[oc.length], fc = new float[cols.length];
		double m2 = 0;
		for (State s : positions) {
			FastState fs = FastState.fromState(s);
			wo.setBoard(fs);
			wc.setBoard(fs);
			LegacyTablutFeaturesV3.extract(s, lws, lf, 11);
			TablutFeatures.extract(s, ws, nf, groups);
			for (int i = 0; i < oc.length; i++) {
				fo[i] = (float) (lf[oc[i]] * v3.div[oc[i]] / d3[oc[i]]); // grezzo v3 / DIV vecchio (doppio)
			}
			for (int i = 0; i < cols.length; i++) {
				fc[i] = (float) nf[cols[i]];
			}
			int role = NeuralEvaluator.roleOf(fs);
			m2 = Math.max(m2, Math.abs(oldC.forward(wo, fo, role) - convC.forward(wc, fc, role)));
		}
		if (m2 > TOL) {
			fail("CNN con DIV v3 raddoppiati: diff max " + m2);
		} else {
			ok(String.format("CNN con DIV diversi e colonne ripiegate: stessa funzione (diff max %.2e)", m2));
		}
	}

	static void checkLossy(List<State> positions, long seed) {
		// togliendo un gruppo la funzione cambia (lo dichiara il piano) e il convertitore non deve inventarsi l'esattezza
		ConvNet old = randomCnn(LegacyTablutFeaturesV3.columnsFor(15).length, seed);
		WeightConverter.Plan plan = new WeightConverter.Plan(3, 15, TablutFeatures.G_DEFAULT);
		ConvNet conv = WeightConverter.convertCnn(old, plan);
		double max = compareCnn(positions, old, 15, conv, plan.toGroups);
		if (plan.isExact() || max < 1e-4) {
			fail("conversione con perdita: il piano dice esatta=" + plan.isExact() + " e la differenza massima e' " + max);
		} else {
			ok(String.format("conversione con perdita: dichiarata non esatta e la funzione cambia davvero (diff max %.3f)", max));
		}
		try {
			WeightConverter.convertCnn(randomCnn(10, seed), new WeightConverter.Plan(3, 11, TablutFeatures.G_DEFAULT));
			fail("una rete con il numero sbagliato di feature dovrebbe essere rifiutata");
		} catch (IllegalArgumentException e) {
			ok("rete con numero di feature incoerente rifiutata");
		}
	}

	static void checkFiles(List<State> positions, long seed) throws IOException {
		File dir = Files.createTempDirectory("convcheck").toFile();
		try {
			// CNN: scrittura/lettura dal file e uso tramite il percorso di produzione
			int fromGroups = 11;
			ConvNet old = randomCnn(41, seed);
			File fin = new File(dir, "old_cnn.txt"), fout = new File(dir, "new_cnn.txt");
			old.save(fin.getPath(), 3, fromGroups);
			try {
				CnnEvaluator.fromFile(fin.getPath());
				fail("i pesi v3 dovrebbero essere rifiutati dal codice v4");
			} catch (IOException e) {
				ok("pesi v3 rifiutati dal loader v4 (e il messaggio indica il convertitore: " + e.getMessage().contains("WeightConverter") + ")");
			}
			WeightConverter.Plan plan = WeightConverter.convertFile(fin.getPath(), fout.getPath(), -1);
			CnnEvaluator ev = CnnEvaluator.fromFile(fout.getPath());
			ConvNet.Loaded l = ConvNet.load(fout.getPath());
			double max = 0;
			ConvNet.Workspace wo = old.newWorkspace();
			LegacyTablutFeaturesV3.Workspace lws = LegacyTablutFeaturesV3.newWorkspace();
			double[] lf = new double[LegacyTablutFeaturesV3.NUM_FEATURES];
			float[] fo = new float[41];
			int[] oc = LegacyTablutFeaturesV3.columnsFor(fromGroups);
			for (State s : positions) {
				FastState fs = FastState.fromState(s);
				wo.setBoard(fs);
				LegacyTablutFeaturesV3.extract(s, lws, lf, fromGroups);
				for (int i = 0; i < 41; i++) {
					fo[i] = (float) lf[oc[i]];
				}
				max = Math.max(max, Math.abs(old.forward(wo, fo, NeuralEvaluator.roleOf(fs)) * ev.outputScale - ev.evaluate(fs)));
			}
			if (l.featureVersion != TablutFeatures.VERSION || l.featureGroups != plan.toGroups || max > TOL * 1000) {
				fail("file CNN convertito: versione " + l.featureVersion + ", gruppi " + l.featureGroups + ", diff max (scala 1000) " + max);
			} else {
				ok(String.format("file CNN: convertito, riletto dal loader di produzione e coerente (diff max %.2e su scala 1000)", max));
			}
			try {
				WeightConverter.convertFile(fout.getPath(), new File(dir, "again.txt").getPath(), -1);
				fail("riconvertire un file gia' attuale dovrebbe essere rifiutato");
			} catch (IOException e) {
				ok("file gia' attuale: conversione rifiutata con messaggio");
			}
			// MLP su file
			MLP mo = randomMlp(41, false, seed + 1);
			File mi = new File(dir, "old_mlp.txt"), mout = new File(dir, "new_mlp.txt");
			mo.save(mi.getPath(), 3, 11);
			WeightConverter.convertFile(mi.getPath(), mout.getPath(), -1);
			NeuralEvaluator nev = NeuralEvaluator.fromFile(mout.getPath());
			double m2 = 0;
			MLP.Scratch so = mo.newScratch();
			double[] xo = new double[41];
			for (State s : positions.subList(0, Math.min(500, positions.size()))) {
				FastState fs = FastState.fromState(s);
				LegacyTablutFeaturesV3.extract(s, lws, lf, 11);
				for (int i = 0; i < 41; i++) {
					xo[i] = lf[oc[i]];
				}
				m2 = Math.max(m2, Math.abs(mo.forward(xo, NeuralEvaluator.roleOf(fs), so) * nev.outputScale - nev.evaluate(fs)));
			}
			if (m2 > TOL * 1000) {
				fail("file MLP convertito: diff max " + m2);
			} else {
				ok(String.format("file MLP: convertito e riletto da NeuralEvaluator.fromFile (diff max %.2e su scala 1000)", m2));
			}
			// CNN puro: nessuna conversione
			ConvNet pure = new ConvNet(new ConvNet.Config(new int[] { 4 }, 2, 0, new int[] { 8 }, new int[] { 4 }), 5L);
			File pf = new File(dir, "pure.txt");
			pure.save(pf.getPath(), 0, 0);
			try {
				WeightConverter.convertFile(pf.getPath(), new File(dir, "pure_out.txt").getPath(), -1);
				fail("un CNN puro non va convertito");
			} catch (IOException e) {
				ok("CNN puro: nessuna conversione necessaria (messaggio chiaro)");
			}
		} finally {
			for (File f : dir.listFiles()) {
				f.delete();
			}
			dir.delete();
		}
	}

	static void checkRealCnn(List<State> positions, String path) throws IOException {
		ConvNet.Loaded l = ConvNet.load(path);
		if (l.featureVersion != 3) {
			System.out.println("(" + path + " non e' v3: salto)");
			return;
		}
		WeightConverter.Plan plan = new WeightConverter.Plan(3, l.featureGroups, l.featureGroups | WeightConverter.V4_NEW_GROUPS);
		ConvNet conv = WeightConverter.convertCnn(l.net, plan);
		double max = compareCnn(positions, l.net, l.featureGroups, conv, plan.toGroups);
		if (!plan.isExact() || max > TOL) {
			fail("pesi reali " + path + ": diff max " + max);
		} else {
			ok(String.format("PESI REALI %s: convertiti, stessa funzione su %d posizioni, diff max %.2e (uscita in (-1,1))", path,
					positions.size(), max));
		}
	}

	// ---------------------------------------------------------------- dataset

	static void checkDatasetUpgrade() throws Exception {
		CnnPretrainer.Dataset d = CnnPretrainer.generate(4, 123L, 1, null);
		File dir = Files.createTempDirectory("dsupgrade").toFile();
		File old = new File(dir, "old.gz");
		try {
			// simula un dataset di una versione precedente: stessi dati, 44 "feature" a zero (verranno scartate)
			try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(old))))) {
				o.writeUTF("TABLUT_CNN_DATA v1");
				o.writeInt(3);
				o.writeInt(44);
				o.writeInt(d.n);
				o.write(d.cells);
				o.write(d.role);
				o.write(d.outcome);
				for (int g : d.game) {
					o.writeInt(g);
				}
				for (int i = 0; i < d.n * 44; i++) {
					o.writeFloat(0f);
				}
			}
			try {
				CnnPretrainer.Dataset.load(old.getPath());
				fail("il caricamento normale dovrebbe rifiutare il dataset v3");
			} catch (IOException e) {
				// atteso
			}
			CnnPretrainer.Dataset up = DatasetUpgrade.loadAndRecompute(old.getPath());
			boolean same = up.n == d.n && Arrays.equals(up.cells, d.cells) && Arrays.equals(up.role, d.role)
					&& Arrays.equals(up.outcome, d.outcome) && Arrays.equals(up.game, d.game) && Arrays.equals(up.feats, d.feats);
			if (!same) {
				fail("DatasetUpgrade: il dataset aggiornato non coincide con quello generato direttamente");
			} else {
				ok("DatasetUpgrade: " + d.n + " posizioni, scacchiere/esiti invariati e feature ricalcolate identiche a quelle generate");
			}
		} finally {
			old.delete();
			dir.delete();
		}
	}
}
