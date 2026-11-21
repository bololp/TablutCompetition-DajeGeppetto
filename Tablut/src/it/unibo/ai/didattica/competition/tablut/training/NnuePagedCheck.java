package it.unibo.ai.didattica.competition.tablut.training;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;

/**
 * Test del dataset PAGINATO ({@link NnueDataset} con piu' di {@link NnueDataset#FLAT_MAX} campioni) senza dover creare milioni di campioni: le
 * soglie sono abbassate con i costruttori/metodi package-private (blocco unico fino a 10 campioni, pagine da 8 righe). Ogni operazione e' confrontata
 * con la stessa operazione sul dataset a blocco unico, che e' il comportamento di riferimento.
 * <ul>
 * <li>{@link NnueDataset.Builder}, {@code subset}, {@code merge}, {@code loadMerged}, {@code save}/{@code load}, {@code splitByGame}, {@code withValues};</li>
 * <li>file byte per byte identico fra le due disposizioni (il formato non e' cambiato);</li>
 * <li>errori con l'indice GLOBALE del campione, file troncati, byte in piu';</li>
 * <li>{@link NnueTrainer#train} e {@link NnueDataTool#filter} danno gli stessi risultati;</li>
 * <li>l'aritmetica dell'overflow originale ({@code 81 * n} in {@code int}).</li>
 * </ul>
 */
public class NnuePagedCheck {

	private static int failures = 0;
	private static long checks = 0;

	/** Blocco unico fino a FM campioni, pagine da 2^PS righe. */
	private static final int FM = 10, PS = 3;

	private static void check(boolean ok, String msg) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	private static File tmp(String suffix) throws IOException {
		final File f = File.createTempFile("nnupaged_", suffix);
		f.deleteOnExit();
		return f;
	}

	public static void main(String[] args) throws Exception {
		testOverflowArithmetic();
		testBuilderAndAccess();
		testEmptyDatasets();
		testIoAndMerge();
		testSubsetSplitWithValues();
		testLoadErrors();
		testTrainerAndTool();
		System.out.println(failures == 0 ? "=== NNUE PAGED CHECK OK: " + checks + " verifiche ===" : "=== NNUE PAGED CHECK: " + failures + " FALLIMENTI su " + checks + " ===");
		if (failures != 0) {
			System.exit(1);
		}
	}

	// ====================================================================== dati di prova

	/** Una posizione valida e non terminale: 8 bianchi, 10 neri, il Re all'interno, niente sul trono. */
	private static byte[] board(Random r) {
		final byte[] b = new byte[81];
		for (int placed = 0; placed < 8;) {
			final int sq = r.nextInt(81);
			if (sq != BoardConstants.THRONE && b[sq] == 0) {
				b[sq] = BoardConstants.WHITE;
				placed++;
			}
		}
		for (int placed = 0; placed < 10;) {
			final int sq = r.nextInt(81);
			if (sq != BoardConstants.THRONE && b[sq] == 0) {
				b[sq] = BoardConstants.BLACK;
				placed++;
			}
		}
		while (true) {
			final int sq = (1 + r.nextInt(7)) * 9 + 1 + r.nextInt(7);
			if (b[sq] == 0 && sq != BoardConstants.THRONE) {
				b[sq] = BoardConstants.KING;
				return b;
			}
		}
	}

	private static void fill(NnueDataset.Builder bd, int n, long seed, int games) {
		final Random r = new Random(seed);
		for (int i = 0; i < n; i++) {
			final boolean ov = r.nextBoolean(), vv = r.nextBoolean();
			bd.add(board(r), 0, r.nextInt(2), r.nextInt(3) - 1, ov, r.nextDouble() * 2 - 1, vv, i % games, i, ov ? r.nextInt(30) : -1, r.nextInt(3));
		}
	}

	private static NnueDataset flat(int n, long seed, int games) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		fill(b, n, seed, games);
		return b.build();
	}

	private static NnueDataset paged(int n, long seed, int games) {
		final NnueDataset.Builder b = new NnueDataset.Builder(FM, PS);
		fill(b, n, seed, games);
		return b.build();
	}

	private static byte[] gunzip(File f) throws IOException {
		try (InputStream in = new GZIPInputStream(new java.io.FileInputStream(f))) {
			final ByteArrayOutputStream o = new ByteArrayOutputStream();
			final byte[] buf = new byte[1 << 14];
			for (int k; (k = in.read(buf)) > 0;) {
				o.write(buf, 0, k);
			}
			return o.toByteArray();
		}
	}

	private static void writeGz(File f, byte[] raw) throws IOException {
		try (OutputStream o = new GZIPOutputStream(new FileOutputStream(f))) {
			o.write(raw);
		}
	}

	// ====================================================================== test

	private static void testOverflowArithmetic() {
		check(81 * 50_746_746 == -184_480_870, "81 * 50.746.746 in int e' -184480870: e' l'eccezione NegativeArraySizeException originale");
		check((long) NnueDataset.FLAT_MAX * 81 <= Integer.MAX_VALUE && (long) (NnueDataset.FLAT_MAX + 1) * 81 > Integer.MAX_VALUE, "FLAT_MAX e' l'ultimo n per cui 81 * n sta in un int");
		check((long) 81 << NnueDataset.PAGE_SHIFT <= Integer.MAX_VALUE, "una pagina di scacchiere sta in un array");
	}

	private static void testBuilderAndAccess() {
		for (int n : new int[] { 0, 1, 7, 8, 9, 10, 11, 16, 17, 24, 25, 100, 1000 }) {
			final NnueDataset df = flat(n, 5, 7), dp = paged(n, 5, 7);
			check(df.n == n && dp.n == n && df.equals(dp) && dp.equals(df), "Builder paginato == blocco unico (n=" + n + ")");
			check(dp.isPaged() == (n > FM) && !df.isPaged(), "isPaged (n=" + n + ")");
			check(dp.isPaged() ? dp.boards == null && dp.role == null && dp.value == null : dp.boards != null && dp.boards.length == 81 * n && dp.role.length == n, "campi pubblici: null solo se paginato (n=" + n + ")");
			check(dp.describe().equals(df.describe()) && dp.numGames() == df.numGames(), "describe e numGames (n=" + n + ")");
			boolean acc = true;
			final byte[] b1 = new byte[81], b2 = new byte[81];
			for (int i = 0; i < n; i++) {
				df.copyBoard(i, b1, 0);
				dp.copyBoard(i, b2, 0);
				acc &= Arrays.equals(b1, b2) && df.role(i) == dp.role(i) && df.outcome(i) == dp.outcome(i) && df.value(i) == dp.value(i) && df.game(i) == dp.game(i)
						&& df.ply(i) == dp.ply(i) && df.toEnd(i) == dp.toEnd(i) && df.source(i) == dp.source(i) && df.outValid(i) == dp.outValid(i)
						&& df.valValid(i) == dp.valValid(i);
				if (n > 0) {
					acc &= Arrays.equals(Arrays.copyOfRange(dp.boardArray(i), dp.boardOffset(i), dp.boardOffset(i) + 81), b1);
				}
			}
			check(acc, "accessori == campi (n=" + n + ")");
			// il builder resta utilizzabile dopo build() e non altera il dataset gia' costruito (le pagine piene sono condivise)
			final NnueDataset.Builder b = new NnueDataset.Builder(FM, PS);
			fill(b, n, 5, 7);
			final NnueDataset first = b.build();
			fill(b, 3, 9, 7);
			final NnueDataset second = b.build();
			boolean same = second.n == n + 3;
			for (int i = 0; i < n; i++) {
				same &= first.role(i) == second.role(i) && first.game(i) == second.game(i) && first.value(i) == second.value(i);
			}
			check(same && first.equals(dp), "build() ripetuto: il primo dataset non cambia (n=" + n + ")");
			check(df.hashCode() == flat(n, 5, 7).hashCode() && dp.hashCode() == paged(n, 5, 7).hashCode(), "hashCode coerente (n=" + n + ")");
		}
	}

	/** Regressione: l'unione di nessun dataset deve dare colonne vuote, non null (si puo' descrivere e salvare). */
	private static void testEmptyDatasets() throws IOException {
		final NnueDataset m = NnueDataset.merge(new ArrayList<NnueDataset>());
		check(m.n == 0 && m.boards != null && m.boards.length == 0 && m.game != null, "merge di nessuno: colonne vuote, non null");
		check(m.numGames() == 0 && m.describe().contains("0 posizioni"), "merge di nessuno: describe e numGames");
		final File f = tmp(".gz");
		m.save(f.getPath());
		check(NnueDataset.load(f.getPath()).equals(m), "merge di nessuno: si salva e si rilegge");
		check(NnueDataset.loadMerged(new ArrayList<String>()).n == 0 && NnueDataset.loadMerged(Collections.singletonList(f.getPath())).n == 0, "loadMerged di nessuno / di un file vuoto");
		check(flat(5, 1, 2).subset(new int[0]).n == 0 && NnueDataset.merge(Arrays.asList(flat(0, 1, 1), flat(0, 2, 1))).n == 0, "subset vuoto, unione di vuoti");
	}

	private static void testIoAndMerge() throws IOException {
		for (int n : new int[] { 0, 1, 8, 9, 10, 11, 17, 100 }) {
			final NnueDataset df = flat(n, 6, 7), dp = paged(n, 6, 7);
			final File ff = tmp(".gz"), fp = tmp(".gz");
			df.save(ff.getPath());
			dp.save(fp.getPath());
			check(Arrays.equals(gunzip(ff), gunzip(fp)), "file byte per byte identico, paginato e blocco unico (n=" + n + ")");
			check(NnueDataset.loadFiles(Collections.singletonList(fp.getPath()), false, FM, PS).equals(df), "load in pagine == blocco unico (n=" + n + ")");
			check(NnueDataset.load(fp.getPath()).equals(dp) && NnueDataset.load(ff.getPath()).equals(df), "load normale (n=" + n + ")");
		}
		final List<NnueDataset> parts = new ArrayList<>();
		final List<String> paths = new ArrayList<>();
		int total = 0;
		for (int k = 0; k < 4; k++) {
			final int n = 5 + 9 * k;
			total += n;
			final NnueDataset d = flat(n, 100 + k, 4 + k);
			final File f = tmp(".gz");
			d.save(f.getPath());
			parts.add(d);
			paths.add(f.getPath());
		}
		final NnueDataset m1 = NnueDataset.merge(parts), m2 = NnueDataset.merge(parts, FM, PS), m3 = NnueDataset.loadFiles(paths, true, FM, PS), m4 = NnueDataset.loadMerged(paths);
		check(m1.n == total && m1.equals(m2) && m2.equals(m3) && m3.equals(m4), "merge == loadMerged, in pagine e a blocco unico (n=" + total + ")");
		check(m2.isPaged() && m3.isPaged() && !m1.isPaged() && !m4.isPaged(), "la disposizione dipende solo da n");
		int maxG = 0;
		for (int i = 0; i < m2.n; i++) {
			maxG = Math.max(maxG, m2.game(i));
		}
		check(maxG + 1 == m2.numGames() && m2.numGames() == parts.get(0).numGames() + parts.get(1).numGames() + parts.get(2).numGames() + parts.get(3).numGames(),
				"unione: chiavi di partita rinumerate densamente, partite di file diversi distinte");
		check(NnueDataset.loadFiles(paths, false, FM, PS).game(0) == parts.get(0).game(0), "load senza rinumerazione: chiavi del file");
	}

	private static void testSubsetSplitWithValues() {
		for (int n : new int[] { 0, 1, 9, 10, 11, 25, 100, 500 }) {
			final NnueDataset df = flat(n, 3, 9), dp = paged(n, 3, 9);
			final int[][] s1 = df.splitByGame(3, 30), s2 = dp.splitByGame(3, 30);
			check(Arrays.equals(s1[0], s2[0]) && Arrays.equals(s1[1], s2[1]) && s1[0].length + s1[1].length == n, "splitByGame (n=" + n + ")");
			final Random r = new Random(n);
			final int[] idx = new int[n == 0 ? 0 : 2 * n];
			for (int i = 0; i < idx.length; i++) {
				idx[i] = r.nextInt(n);
			}
			check(df.subset(idx).equals(dp.subset(idx, FM, PS)) && dp.subset(idx).equals(dp.subset(idx, FM, PS)), "subset (n=" + n + ")");
			// withValues: nuove etichette, le altre colonne identiche
			final float[] v = new float[n];
			final byte[] vv = new byte[n];
			for (int i = 0; i < n; i++) {
				vv[i] = (byte) (i % 3 == 0 ? 0 : 1);
				v[i] = vv[i] == 0 ? 0f : (i % 11) / 10f - 0.5f;
			}
			final NnueDataset wf = df.withValues(v.clone(), vv.clone()), wp = dp.withValues(v.clone(), vv.clone());
			boolean ok = wf.equals(wp) && wf.n == n && wp.isPaged() == (n > FM);
			for (int i = 0; i < n; i++) {
				ok &= wp.value(i) == v[i] && wp.valValid(i) == vv[i] && wp.role(i) == dp.role(i) && wp.game(i) == dp.game(i) && wp.outcome(i) == dp.outcome(i);
			}
			check(ok, "withValues cambia solo value/valValid (n=" + n + ")");
			check(wp.numGames() == dp.numGames() && wp.describe().contains(n + " posizioni"), "withValues: dataset coerente (n=" + n + ")");
			if (n > 0) {
				final float[] bad = new float[n];
				final byte[] one = new byte[n];
				bad[n - 1] = 2f;
				one[n - 1] = 1;
				try {
					dp.withValues(bad, one);
					check(false, "withValues con valore 2 deve essere rifiutato (n=" + n + ")");
				} catch (IllegalArgumentException e) {
					check(e.getMessage().contains("campione " + (n - 1) + ":"), "withValues: errore con l'indice del campione: " + e.getMessage());
				}
				try {
					dp.withValues(new float[n + 1], new byte[n + 1]);
					check(false, "withValues con lunghezza sbagliata deve essere rifiutato");
				} catch (IllegalArgumentException e) {
					check(e.getMessage().contains("lunghezza"), "withValues: lunghezza sbagliata");
				}
			}
		}
	}

	private static void testLoadErrors() throws IOException {
		final NnueDataset ok = paged(30, 1, 5);
		final File good = tmp(".gz");
		ok.save(good.getPath());
		final byte[] raw = gunzip(good);
		final int hdr = 2 + NnueDataset.MAGIC.length() + 4;
		final File f = tmp(".gz");
		// un codice di cella sbagliato nel campione 21 (terza pagina): l'indice nel messaggio e' globale
		final byte[] bad = raw.clone();
		bad[hdr + 81 * 21 + 3] = 7;
		writeGz(f, bad);
		try {
			NnueDataset.loadFiles(Collections.singletonList(f.getPath()), false, FM, PS);
			check(false, "campione non valido in una pagina: accettato");
		} catch (IOException e) {
			check(e.getMessage().contains("non validi") && e.getMessage().contains("campione 21:"), "campione non valido: " + e.getMessage());
		}
		try {
			final NnueDataset.Builder b = new NnueDataset.Builder(FM, PS);
			fill(b, 20, 2, 3);
			final byte[] brd = new byte[81]; // tutta vuota: senza Re
			b.add(brd, 0, 0, 0, false, 0, false, 0, 0, -1, 0);
			b.build();
			check(false, "Builder paginato: posizione senza Re accettata");
		} catch (IllegalArgumentException e) {
			check(e.getMessage().contains("campione 20:") && e.getMessage().contains("Re"), "Builder paginato: errore con l'indice globale: " + e.getMessage());
		}
		writeGz(f, Arrays.copyOf(raw, raw.length - 10));
		try {
			NnueDataset.loadFiles(Collections.singletonList(f.getPath()), true, FM, PS);
			check(false, "file troncato accettato");
		} catch (IOException e) {
			check(e.getMessage().contains("troncato"), "file troncato: " + e.getMessage());
		}
		writeGz(f, Arrays.copyOf(raw, raw.length + 1));
		try {
			NnueDataset.loadFiles(Collections.singletonList(f.getPath()), true, FM, PS);
			check(false, "byte in piu' accettato");
		} catch (IOException e) {
			check(e.getMessage().contains("in piu'"), "byte in piu': " + e.getMessage());
		}
		// un numero di campioni assurdo nell'intestazione e' un errore di lettura (non un'allocazione enorme)
		final ByteArrayOutputStream o = new ByteArrayOutputStream();
		final java.io.DataOutputStream d = new java.io.DataOutputStream(o);
		d.writeUTF(NnueDataset.MAGIC);
		d.writeInt(NnueDataset.MAX_SAMPLES);
		d.flush();
		writeGz(f, o.toByteArray());
		try {
			NnueDataset.load(f.getPath());
			check(false, "intestazione con 2e9 campioni e nessun dato: accettata");
		} catch (IOException e) {
			check(e.getMessage().contains("troncato"), "2e9 campioni dichiarati e nessun dato: errore pulito, nessuna allocazione enorme: " + e.getMessage());
		}
	}

	private static void testTrainerAndTool() {
		final NnueDataset df = flat(500, 11, 20), dp = paged(500, 11, 20);
		final NnueTrainer.Config cfg = new NnueTrainer.Config();
		cfg.hidden = 4;
		cfg.l2 = 2;
		cfg.epochs = 3;
		cfg.batch = 32;
		final int[][] s1 = df.splitByGame(1, 20), s2 = dp.splitByGame(1, 20);
		final int[] u1 = NnueTrainer.usable(cfg, df, s1[0], true), u2 = NnueTrainer.usable(cfg, dp, s2[0], true);
		check(Arrays.equals(u1, u2) && u1.length > 0 && u1.length < s1[0].length, "usable: stessi indici, i campioni senza etichetta sono filtrati");
		final NnueTrainer.Result r1 = NnueTrainer.train(df, s1[0], s1[1], null, null, cfg, null, null);
		final NnueTrainer.Result r2 = NnueTrainer.train(dp, s2[0], s2[1], null, null, cfg, null, null);
		check(r1.trainSamples == r2.trainSamples && r1.bestVal == r2.bestVal && r1.bestEpoch == r2.bestEpoch && Arrays.equals(r1.net.p, r2.net.p),
				"NnueTrainer.train: stessa rete con dataset paginato e a blocco unico");
		cfg.maxSamples = 100;
		check(NnueTrainer.train(dp, s2[0], s2[1], null, null, cfg, null, null).trainSamples == 100, "train con maxsamples su dataset paginato");
		check(NnueTrainer.meanLoss(r1.net, cfg, df, u1) == NnueTrainer.meanLoss(r1.net, cfg, dp, u2), "meanLoss uguale");
		final boolean[] only1 = new boolean[16];
		only1[1] = true;
		check(NnueDataTool.filter(df, only1, true, false).equals(NnueDataTool.filter(dp, only1, true, false)) && NnueDataTool.filter(dp, null, false, false).equals(dp),
				"NnueDataTool.filter: paginato == blocco unico, senza condizioni = identita'");
	}
}
