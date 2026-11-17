package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Verifica {@link GaGenome}: (1) il genoma neutro riproduce la base BIT PER BIT (parametri, impronta, valutazione di centinaia di posizioni); (2) OGNI gene muove
 * esattamente il parametro che dice il suo nome, della quantita' {@code round(g*R)} e di nessun altro (oracolo indipendente: si legge il nome, non il codice di
 * decodifica); (3) saturazione dei {@code short}; (4) la base non viene mai modificata; (5) geni agli estremi decodificano sempre; (6) validazione dei geni;
 * (7) formato file: andata e ritorno esatti e rifiuto severo di OGNI difetto (IOException, mai altre eccezioni ne' accettazioni silenziose); (8) la rete evoluta e'
 * una normale rete NNUE (salvataggio / {@code EvaluatorLoader}); (9) una ricerca con rete evoluta dà lo STESSO risultato col percorso incrementale e con la
 * valutazione completa (i bias {@code ftB} cambiano l'accumulatore di base: e' il caso piu' delicato).
 * <p>
 * Uso: {@code GaGenomeCheck [pesi_nnue/trained_v6.nnue]} (se il file esiste viene provato anche come base, oltre a una rete casuale della stessa forma).
 */
public class GaGenomeCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String what) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + what);
		}
	}

	static boolean sameNet(NnueNetwork a, NnueNetwork b) {
		if (a.hidden() != b.hidden() || a.l2() != b.l2() || !Arrays.equals(a.ftWeights(), b.ftWeights()) || !Arrays.equals(a.ftBiases(), b.ftBiases())) {
			return false;
		}
		for (int r = 0; r < 2; r++) {
			if (!Arrays.equals(a.l1Weights(r), b.l1Weights(r)) || !Arrays.equals(a.l1Biases(r), b.l1Biases(r)) || !Arrays.equals(a.l2Weights(r), b.l2Weights(r))
					|| a.l2Bias(r) != b.l2Bias(r)) {
				return false;
			}
		}
		return true;
	}

	/** Tutti i parametri della rete come vettore di long con un'etichetta per elemento: serve a individuare QUALE parametro e' cambiato. */
	static List<String> labels(NnueNetwork n) {
		final List<String> l = new ArrayList<>();
		for (int i = 0; i < n.ftWeights().length; i++) {
			l.add("ftW[" + i + "]");
		}
		for (int i = 0; i < n.ftBiases().length; i++) {
			l.add("ftB[" + i + "]");
		}
		for (int r = 0; r < 2; r++) {
			for (int i = 0; i < n.l1Weights(r).length; i++) {
				l.add("l1W[" + r + "][" + i + "]");
			}
			for (int i = 0; i < n.l1Biases(r).length; i++) {
				l.add("l1B[" + r + "][" + i + "]");
			}
			for (int i = 0; i < n.l2Weights(r).length; i++) {
				l.add("l2W[" + r + "][" + i + "]");
			}
			l.add("l2B[" + r + "]");
		}
		return l;
	}

	static long[] values(NnueNetwork n) {
		final List<Long> v = new ArrayList<>();
		for (short x : n.ftWeights()) {
			v.add((long) x);
		}
		for (short x : n.ftBiases()) {
			v.add((long) x);
		}
		for (int r = 0; r < 2; r++) {
			for (short x : n.l1Weights(r)) {
				v.add((long) x);
			}
			for (int x : n.l1Biases(r)) {
				v.add((long) x);
			}
			for (short x : n.l2Weights(r)) {
				v.add((long) x);
			}
			v.add((long) n.l2Bias(r));
		}
		final long[] out = new long[v.size()];
		for (int i = 0; i < out.length; i++) {
			out[i] = v.get(i);
		}
		return out;
	}

	static double[] randomGenes(int n, double scale, Random rnd) {
		final double[] g = new double[n];
		for (int i = 0; i < n; i++) {
			g[i] = Math.max(-1, Math.min(1, rnd.nextGaussian() * scale));
		}
		return g;
	}

	static void parseFails(String text, NnueNetwork base, String what) {
		try {
			GaGenome.parse(text, base);
			check(false, what + ": doveva essere rifiutato");
		} catch (IOException e) {
			check(true, what);
		} catch (RuntimeException e) {
			check(false, what + ": eccezione NON controllata " + e);
		}
	}

	static void runFor(String name, NnueNetwork base) throws Exception {
		System.out.println("-- base: " + name + " (" + base.describe() + ")");
		final Random rnd = new Random(42);
		final long baseFp = GaGenome.fingerprint(base);
		final long[] baseVals = values(base);
		final List<String> lab = labels(base);

		// (1) Layout e dimensioni
		final int l2 = base.l2(), h = base.hidden();
		final String[] combos = { "head", "l1b", "ftb", "head,l1b", "head,l1b,ftb", "ftb,head" };
		final int[] expect = { 2 * (l2 + 1), 2 * l2, h, 2 * (l2 + 1) + 2 * l2, 2 * (l2 + 1) + 2 * l2 + h, 2 * (l2 + 1) + h };
		for (int c = 0; c < combos.length; c++) {
			final GaGenome.Layout L = GaGenome.Layout.of(base, combos[c]);
			check(L.size() == expect[c], combos[c] + ": " + L.size() + " geni, attesi " + expect[c]);
			for (int i = 0; i < L.size(); i++) {
				check(L.range(i) >= 2, combos[c] + ": ampiezza del gene " + i + " < 2");
			}
		}
		check(GaGenome.Layout.of(base, "ftb,head").groupsText().equals("HEAD,FTB"), "l'ordine dei gruppi e' quello canonico, non quello richiesto");
		try {
			GaGenome.Layout.of(base, "");
			check(false, "nessun gruppo: doveva essere rifiutato");
		} catch (IllegalArgumentException e) {
			check(true, "nessun gruppo rifiutato");
		}
		try {
			GaGenome.Layout.of(base, "head,xyz");
			check(false, "gruppo sconosciuto: doveva essere rifiutato");
		} catch (IllegalArgumentException e) {
			check(true, "gruppo sconosciuto rifiutato");
		}

		final GaGenome.Layout full = GaGenome.Layout.of(base, "head,l1b,ftb");

		// (2) identita': bit per bit
		final GaGenome id = GaGenome.identity(full);
		final NnueNetwork idNet = id.decode();
		check(id.isNeutral() && id.meanSquare() == 0.0, "identita' neutra");
		check(idNet != base && sameNet(idNet, base) && GaGenome.fingerprint(idNet) == baseFp, "identita': rete uguale bit per bit alla base (ma oggetto distinto)");
		final NnueEvaluator evBase = new NnueEvaluator(base), evId = new NnueEvaluator(idNet);
		int same = 0;
		for (int i = 0; i < 300; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			if (Double.doubleToLongBits(evBase.evaluate(fs)) == Double.doubleToLongBits(evId.evaluate(fs))) {
				same++;
			}
		}
		check(same == 300, "identita': valutazioni identiche su 300 posizioni (" + same + ")");
		// geni piccoli che arrotondano tutti a 0: ancora la base
		final double[] tiny = new double[full.size()];
		for (int i = 0; i < tiny.length; i++) {
			tiny[i] = (i % 2 == 0 ? 1 : -1) * 0.49 / full.range(i);
		}
		final GaGenome tg = new GaGenome(full, tiny);
		check(tg.isNeutral() && sameNet(tg.decode(), base) && tg.meanSquare() > 0, "geni non nulli che arrotondano a 0: neutri, rete = base");

		// (3) ogni gene muove il parametro giusto della quantita' giusta
		int mapped = 0;
		for (int gi = 0; gi < full.size(); gi++) {
			for (double sgn : new double[] { 1.0, -1.0, 0.5 }) {
				final double[] g = new double[full.size()];
				g[gi] = sgn;
				final NnueNetwork n = new GaGenome(full, g).decode();
				final long[] v = values(n);
				final int delta = (int) Math.round(sgn * full.range(gi));
				final String target = full.geneName(gi);
				int changed = 0;
				boolean ok = true;
				for (int k = 0; k < v.length; k++) {
					if (v[k] != baseVals[k]) {
						changed++;
						ok &= lab.get(k).equals(target) && v[k] - baseVals[k] == delta;
					}
				}
				check(ok && changed == (delta == 0 ? 0 : 1), "gene " + gi + " (" + target + ", " + sgn + "): deve cambiare solo " + target + " di " + delta + " (cambiati " + changed + ")");
				mapped++;
			}
		}
		System.out.println("   mappatura gene->parametro verificata su " + mapped + " casi");
		// il nome del gene e' coerente con le etichette (nessun nome fuori dalla rete)
		final Pattern pn = Pattern.compile("(l2W|l2B|l1B|ftB)(\\[\\d+\\])+");
		for (int gi = 0; gi < full.size(); gi++) {
			final Matcher m = pn.matcher(full.geneName(gi));
			check(m.matches() && lab.contains(full.geneName(gi)), "nome gene " + gi + " valido: " + full.geneName(gi));
		}

		// (4) base immutata dopo molte decodifiche
		for (int i = 0; i < 20; i++) {
			new GaGenome(full, randomGenes(full.size(), 0.5, rnd)).decode();
		}
		check(GaGenome.fingerprint(base) == baseFp && Arrays.equals(values(base), baseVals), "la base non e' stata modificata dalle decodifiche");

		// (5) estremi
		final double[] plus = new double[full.size()], minus = new double[full.size()];
		Arrays.fill(plus, 1.0);
		Arrays.fill(minus, -1.0);
		try {
			final NnueEvaluator ep = new NnueEvaluator(new GaGenome(full, plus).decode()), em = new NnueEvaluator(new GaGenome(full, minus).decode());
			boolean fin = true;
			for (int i = 0; i < 50; i++) {
				final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
				final double a = ep.evaluate(fs), b = em.evaluate(fs);
				fin &= !Double.isNaN(a) && !Double.isInfinite(a) && !Double.isNaN(b) && !Double.isInfinite(b) && Math.abs(a) <= 1000.0 && Math.abs(b) <= 1000.0;
			}
			check(fin, "geni tutti a +1 / tutti a -1: decodifica valida, valutazioni finite e in [-1000,1000]");
		} catch (RuntimeException e) {
			check(false, "estremi: eccezione " + e);
		}

		// (6) validazione
		for (double[] bad : new double[][] { new double[full.size() - 1], new double[full.size() + 1], null }) {
			try {
				new GaGenome(full, bad);
				check(false, "lunghezza errata accettata");
			} catch (IllegalArgumentException e) {
				check(true, "lunghezza errata rifiutata");
			}
		}
		for (double v : new double[] { Double.NaN, Double.POSITIVE_INFINITY, 1.0000001, -1.0000001, 5 }) {
			final double[] g = new double[full.size()];
			g[3] = v;
			try {
				new GaGenome(full, g);
				check(false, "gene " + v + " accettato");
			} catch (IllegalArgumentException e) {
				check(true, "gene " + v + " rifiutato");
			}
		}
		final double[] src = new double[full.size()];
		final GaGenome cp = new GaGenome(full, src);
		src[0] = 0.5;
		cp.genes()[1] = 0.5;
		check(cp.gene(0) == 0 && cp.gene(1) == 0, "il genoma copia i geni (nessun aliasing)");

		// (7) file
		final GaGenome ev = new GaGenome(full, randomGenes(full.size(), 0.3, rnd));
		final String text = ev.toText();
		try {
			final GaGenome back = GaGenome.parse(text, base);
			check(back.equals(ev) && Arrays.equals(back.genes(), ev.genes()), "andata e ritorno esatti del testo");
			check(GaGenome.parse(GaGenome.identity(full).toText(), base).isNeutral(), "andata e ritorno del genoma neutro");
			final File f = File.createTempFile("gagenome", ".genome");
			try {
				ev.save(f.getPath());
				check(GaGenome.load(f.getPath(), base).equals(ev), "save/load su file");
			} finally {
				f.delete();
			}
		} catch (IOException e) {
			check(false, "andata e ritorno: " + e);
		}
		check(text.startsWith(GaGenome.MAGIC + "\n"), "intestazione " + GaGenome.MAGIC);
		parseFails(text.replace(GaGenome.MAGIC, "TABLUT_GA_GENOME v2"), base, "versione diversa");
		parseFails(text.replace("TABLUT_GA_GENOME", "ALTRO"), base, "intestazione errata");
		parseFails("", base, "file vuoto");
		parseFails(text.substring(0, text.length() / 2), base, "file troncato a meta'");
		final String[] lines = text.split("\n");
		final StringBuilder noChecksum = new StringBuilder();
		for (int i = 0; i < lines.length - 1; i++) {
			noChecksum.append(lines[i]).append('\n');
		}
		parseFails(noChecksum.toString(), base, "manca il checksum");
		parseFails(text + "extra\n", base, "contenuto dopo il checksum");
		parseFails(text.replaceFirst("genes \\d+", "genes " + (full.size() + 1)), base, "numero di geni dichiarato errato");
		parseFails(text.replaceFirst("groups [A-Z0-9,]+", "groups HEAD"), base, "gruppi che non corrispondono al numero di geni");
		parseFails(text.replaceFirst("groups [A-Z0-9,]+", "groups HEAD,ZZZ"), base, "gruppo sconosciuto");
		parseFails(text.replaceFirst("base [0-9a-f]{8}", "base 00000000"), base, "impronta della base diversa");
		parseFails(text.replaceFirst("base [0-9a-f]{8}", "base zzzzzzzz"), base, "impronta non esadecimale");
		final String firstGene = lines[4].split(" ")[0];
		parseFails(text.replaceFirst(Pattern.quote(firstGene), "0.123456789"), base, "gene modificato senza aggiornare il checksum");
		parseFails(text.replaceFirst(Pattern.quote(firstGene), "abc"), base, "gene non numerico");
		parseFails(text.replaceFirst(Pattern.quote(firstGene), "NaN"), base, "gene NaN");
		parseFails(text.replaceFirst(Pattern.quote(firstGene), "1.5"), base, "gene fuori da [-1,1]");
		parseFails(text.replaceFirst("checksum [0-9a-f]{8}", "checksum 00000000"), base, "checksum errato");
		parseFails(text.replace("\n" + lines[4] + "\n", "\n" + lines[4] + " 0.1\n"), base, "geni in eccesso");
		// una base DIVERSA (stessa forma, altri pesi) rifiuta il genoma
		final NnueNetwork other = NnueNetwork.random(h, l2, 999);
		parseFails(text, other, "genoma applicato a un'altra base");
		try {
			GaGenome.load("/nonexistent_dir_xyz/none.genome", base);
			check(false, "file inesistente accettato");
		} catch (IOException e) {
			check(true, "file inesistente: IOException");
		}

		// (8) rete evoluta = normale rete NNUE
		final NnueNetwork evolved = ev.decode();
		final File nf = File.createTempFile("gaevolved", ".nnue");
		try {
			evolved.save(nf.getPath());
			check(NnueNetwork.isNnueFile(nf.getPath()), "la rete evoluta salvata e' riconosciuta come file NNUE");
			final Evaluator loaded = EvaluatorLoader.load(nf.getPath());
			check(loaded instanceof NnueEvaluator && sameNet(((NnueEvaluator) loaded).net, evolved), "EvaluatorLoader ricarica la rete evoluta identica");
			check(evolved.numParams() == base.numParams(), "stessa architettura/numero di parametri della base (" + base.numParams() + ")");
		} finally {
			nf.delete();
		}

		// (9) ricerca con rete evoluta: incrementale == completa
		final Evaluator fullOnly = new Evaluator() {
			final NnueEvaluator inner = new NnueEvaluator(evolved);

			@Override
			public double evaluate(FastState s) {
				return inner.evaluate(s);
			}
		};
		final NnueEvaluator incr = new NnueEvaluator(evolved);
		final TestPositions.Game game = TestPositions.playGame(new Random(3), true, 60);
		int compared = 0;
		for (int i = 0; i < game.positions.size() && compared < 8; i += 5) {
			final State s = game.positions.get(i);
			final Negamax a = new Negamax(incr, 3, true, 1 << 14, true, Negamax.ORDER_RECOMMENDED, 1);
			final Negamax b = new Negamax(fullOnly, 3, true, 1 << 14, true, Negamax.ORDER_RECOMMENDED, 1);
			a.setSafetyMarginMillis(0);
			b.setSafetyMarginMillis(0);
			final Action ma = a.findBestMove(s, 600000), mb = b.findBestMove(s, 600000);
			final boolean eq = ma != null && mb != null && ma.getFrom().equals(mb.getFrom()) && ma.getTo().equals(mb.getTo()) && a.lastScore == b.lastScore
					&& a.lastTotalNodeCount == b.lastTotalNodeCount;
			check(eq, "posizione " + i + ": ricerca incrementale = completa con rete evoluta (" + ma + " " + a.lastScore + "/" + a.lastTotalNodeCount + " vs " + mb + " " + b.lastScore + "/"
					+ b.lastTotalNodeCount + ")");
			compared++;
		}
		check(compared >= 6, "posizioni confrontate nella ricerca: " + compared);
	}

	public static void main(String[] args) throws Exception {
		runFor("rete casuale 64x16", NnueNetwork.random(64, 16, 7));
		runFor("rete casuale piccola 8x4", NnueNetwork.random(8, 4, 11));
		final String v6 = args.length > 0 ? args[0] : "pesi_nnue/trained_v6.nnue";
		if (new File(v6).isFile()) {
			runFor(v6, NnueNetwork.load(v6));
		} else {
			System.out.println("(" + v6 + " non trovato: provo solo le reti casuali)");
		}
		// saturazione dei short (rete costruita a mano con l2W vicino al limite)
		{
			final NnueNetwork r = NnueNetwork.random(4, 2, 5);
			final short[][] l2W = { r.l2Weights(0), r.l2Weights(1) };
			l2W[0][0] = Short.MAX_VALUE - 1;
			l2W[1][1] = Short.MIN_VALUE + 1;
			final short[] ftB = r.ftBiases();
			ftB[0] = Short.MAX_VALUE - 1;
			final NnueNetwork edge;
			try {
				edge = new NnueNetwork(4, 2, r.ftWeights(), ftB, new short[][] { r.l1Weights(0), r.l1Weights(1) }, new int[][] { r.l1Biases(0), r.l1Biases(1) }, l2W,
						new int[] { r.l2Bias(0), r.l2Bias(1) });
				final GaGenome.Layout L = GaGenome.Layout.of(edge, "head,ftb");
				final double[] g = new double[L.size()];
				Arrays.fill(g, 1.0);
				final NnueNetwork d = new GaGenome(L, g).decode();
				check(d.l2Weights(0)[0] == Short.MAX_VALUE && d.ftBiases()[0] == Short.MAX_VALUE, "saturazione a Short.MAX_VALUE");
				Arrays.fill(g, -1.0);
				final NnueNetwork d2 = new GaGenome(L, g).decode();
				check(d2.l2Weights(1)[1] == Short.MIN_VALUE || d2.l2Weights(1)[1] < Short.MIN_VALUE + 1 + 1, "saturazione/limite a Short.MIN_VALUE: " + d2.l2Weights(1)[1]);
			} catch (RuntimeException e) {
				check(false, "saturazione: eccezione " + e);
			}
		}
		if (failures == 0) {
			System.out.println("=== GA GENOME CHECK OK: " + checks + " verifiche ===");
		} else {
			System.out.println("=== FALLITI: " + failures + " su " + checks + " ===");
			System.exit(1);
		}
	}
}
