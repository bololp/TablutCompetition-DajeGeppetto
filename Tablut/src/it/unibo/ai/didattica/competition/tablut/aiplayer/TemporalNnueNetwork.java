package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.zip.CRC32;

/**
 * NNUE con feature temporali: una {@link NnueNetwork} (le 243 feature di posizione, il bias e le due teste dense) PIU' un blocco di righe
 * {@code tW} che descrivono le ultime {@code K} mosse ({@link TemporalHistory}). Immutabile, condivisibile fra thread.
 *
 * <pre>
 *   acc[j] = ftB[j] + somma_{f di posizione attiva} ftW[f][j] + somma_{g temporale attiva} tW[g][j]
 *   poi identico a NnueNetwork: clipped ReLU, strato 1 (testa per lato al tratto), strato 2, tanh campionata.
 * </pre>
 * Proprieta' volute:
 * <ul>
 * <li><b>Con {@code tW = 0}, o con la finestra vuota, la rete e' ESATTAMENTE la rete di base</b> (stesso intero, bit per bit). Quindi si puo'
 * partire da una NNUE gia' addestrata (blocco temporale a zero) e addestrare solo cio' che la storia aggiunge, e una rete addestrata cosi' resta
 * corretta quando la storia non e' nota (MCTS, valutazione stateless, primi ply di una ricerca senza storia precedente la radice).</li>
 * <li>L'aritmetica e' intera e la somma dell'accumulatore e' associativa: il percorso incrementale ({@link TemporalNnueEvaluator}: accumulatore di
 * posizione per differenza + somma delle righe temporali al momento della valutazione) e la valutazione completa danno lo stesso valore, bit per bit.</li>
 * </ul>
 *
 * <h3>File {@value #MAGIC}</h3>
 * Testo: {@code features psq3x81+hist<K>}, {@code hidden}, {@code l2}, {@code qa}, {@code qw}, {@code hist K}, {@code params}, poi gli interi (16 per
 * riga): i valori di {@link NnueNetwork} nel suo ordine, poi {@code tW[numFeatures(K) * H]} (feature per feature); infine {@code checksum}
 * (CRC32 dei valori come interi big-endian). L'intestazione NON inizia con {@code TABLUT_NNUE}: una rete temporale non puo' essere scambiata per una NNUE.
 */
public final class TemporalNnueNetwork {

	public static final String MAGIC_PREFIX = "TABLUT_TNNUE";
	public static final String MAGIC = MAGIC_PREFIX + " v1";

	private final NnueNetwork base;
	private final int hist;
	private final short[] tW; // [TemporalHistory.numFeatures(hist)][hidden]

	/**
	 * @param base rete di posizione (si usa cosi' com'e': la sua testa e i suoi pesi di posizione)
	 * @param hist lunghezza della finestra, 1..{@link TemporalHistory#MAX_AGES}
	 * @param tW   pesi temporali a scala QA, {@code TemporalHistory.numFeatures(hist) * base.hidden()} valori, feature per feature
	 * @throws IllegalArgumentException dimensioni errate o pesi che potrebbero andare in overflow
	 */
	public TemporalNnueNetwork(NnueNetwork base, int hist, short[] tW) {
		if (base == null) {
			throw new IllegalArgumentException("rete di base nulla");
		}
		if (hist < 1 || hist > TemporalHistory.MAX_AGES) {
			throw new IllegalArgumentException("hist=" + hist + " fuori dall'intervallo 1.." + TemporalHistory.MAX_AGES);
		}
		final int need = TemporalHistory.numFeatures(hist) * base.hidden();
		if (tW == null || tW.length != need) {
			throw new IllegalArgumentException("tW: attesi " + need + " valori, trovati " + (tW == null ? "null" : String.valueOf(tW.length)));
		}
		this.base = base;
		this.hist = hist;
		this.tW = tW.clone();
		verifyNoOverflow();
	}

	/** Rete temporale con blocco temporale a zero: valuta come {@code base} per qualunque storia. */
	public static TemporalNnueNetwork fromBase(NnueNetwork base, int hist) {
		return new TemporalNnueNetwork(base, hist, new short[TemporalHistory.numFeatures(hist) * base.hidden()]);
	}

	/** Come {@link NnueNetwork}: l'accumulatore non puo' andare in overflow nemmeno con la somma peggiore di posizione + finestra completa. */
	private void verifyNoOverflow() {
		final int h = base.hidden();
		final short[] ftW = base.ftWeights(), ftB = base.ftBiases();
		final long limit = Integer.MAX_VALUE - NnueNetwork.QW;
		for (int j = 0; j < h; j++) {
			long sum = Math.abs((long) ftB[j]);
			for (int sq = 0; sq < NnueNetwork.NUM_SQUARES; sq++) {
				int m = 0;
				for (int t = 0; t < 3; t++) {
					m = Math.max(m, Math.abs((int) ftW[(t * NnueNetwork.NUM_SQUARES + sq) * h + j]));
				}
				sum += m;
			}
			for (int a = 0; a < hist; a++) {
				final int o = a * TemporalHistory.PER_AGE;
				int mFrom = 0, mTo = 0;
				for (int k = 0; k < NnueNetwork.NUM_FEATURES; k++) {
					mFrom = Math.max(mFrom, Math.abs((int) tW[(o + k) * h + j]));
					mTo = Math.max(mTo, Math.abs((int) tW[(o + NnueNetwork.NUM_FEATURES + k) * h + j]));
				}
				final int mCap = Math.max(Math.abs((int) tW[(o + 2 * NnueNetwork.NUM_FEATURES) * h + j]), Math.abs((int) tW[(o + 2 * NnueNetwork.NUM_FEATURES + 1) * h + j]));
				sum += mFrom + mTo + mCap;
			}
			if (sum > limit) {
				throw new IllegalArgumentException("tW/ftW: l'accumulatore " + j + " potrebbe andare in overflow (somma massima " + sum + ")");
			}
		}
	}

	public NnueNetwork base() {
		return base;
	}

	public int hist() {
		return hist;
	}

	public int hidden() {
		return base.hidden();
	}

	public int l2() {
		return base.l2();
	}

	public short[] temporalWeights() {
		return tW.clone();
	}

	public static int paramCount(int hidden, int l2, int hist) {
		return NnueNetwork.paramCount(hidden, l2) + TemporalHistory.numFeatures(hist) * hidden;
	}

	public int numParams() {
		return paramCount(base.hidden(), base.l2(), hist);
	}

	public String describe() {
		return "TNNUE " + NnueNetwork.FEATURE_SET + "+hist" + hist + " (" + base.describe() + " + " + TemporalHistory.numFeatures(hist) + " feature temporali, " + numParams()
				+ " parametri)";
	}

	// ====================================================================== aritmetica

	/** Buffer di lavoro di una valutazione completa (un thread). */
	public static final class Scratch {
		final int[] acc, work, feat;

		public Scratch(TemporalNnueNetwork net) {
			acc = new int[net.hidden()];
			work = new int[net.base.scratchSize()];
			feat = new int[3 * net.hist];
		}
	}

	/**
	 * {@code acc += somma delle righe temporali della finestra}.
	 *
	 * @param recs record, il piu' recente per primo; {@code len} quanti ne sono noti (un record nullo termina la finestra, i piu' vecchi di {@code hist} si ignorano)
	 * @param feat buffer di almeno {@code 3 * hist} posti
	 */
	void addTemporal(int[] acc, int[] recs, int off, int len, int[] feat) {
		final int n = TemporalHistory.activeFeatures(recs, off, len, hist, null, feat, 0);
		final int h = base.hidden();
		final short[] w = tW;
		for (int k = 0; k < n; k++) {
			final int o = feat[k] * h;
			for (int j = 0; j < h; j++) {
				acc[j] += w[o + j];
			}
		}
	}

	/** Logit in virgola fissa (scala QA*QW) di una posizione con la sua finestra, calcolato da zero. */
	int logit(byte[] board, int role, int[] recs, int off, int len, Scratch s) {
		base.refresh(board, s.acc);
		addTemporal(s.acc, recs, off, len, s.feat);
		return base.logit(s.acc, s.work, role);
	}

	/**
	 * Valore in (-1,1) (+1 = vince il Bianco) di una posizione con la sua finestra, calcolato da zero.
	 *
	 * @param role 0 = muove il Bianco, 1 = muove il Nero
	 */
	public double value(byte[] board, int role, int[] recs, int off, int len, Scratch s) {
		return NnueNetwork.valueOf(logit(board, role, recs, off, len, s));
	}

	// ====================================================================== file

	public static boolean isTemporalFile(String path) {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			final String first = r.readLine();
			return first != null && first.startsWith(MAGIC_PREFIX);
		} catch (IOException e) {
			return false;
		}
	}

	private int[] flatten() {
		final int[] all = new int[numParams()];
		int k = 0;
		for (short v : base.ftWeights()) {
			all[k++] = v;
		}
		for (short v : base.ftBiases()) {
			all[k++] = v;
		}
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			for (short v : base.l1Weights(r)) {
				all[k++] = v;
			}
			for (int v : base.l1Biases(r)) {
				all[k++] = v;
			}
			for (short v : base.l2Weights(r)) {
				all[k++] = v;
			}
			all[k++] = base.l2Bias(r);
		}
		for (short v : tW) {
			all[k++] = v;
		}
		return all;
	}

	private static long checksum(int[] all) {
		final CRC32 crc = new CRC32();
		final byte[] b = new byte[4];
		for (int v : all) {
			b[0] = (byte) (v >>> 24);
			b[1] = (byte) (v >>> 16);
			b[2] = (byte) (v >>> 8);
			b[3] = (byte) v;
			crc.update(b, 0, 4);
		}
		return crc.getValue();
	}

	public void save(String path) throws IOException {
		try (BufferedWriter w = new BufferedWriter(new FileWriter(path))) {
			w.write(MAGIC + "\n");
			w.write("features " + NnueNetwork.FEATURE_SET + "+hist" + hist + "\n");
			w.write("hidden " + base.hidden() + "\n");
			w.write("l2 " + base.l2() + "\n");
			w.write("qa " + NnueNetwork.QA + "\n");
			w.write("qw " + NnueNetwork.QW + "\n");
			w.write("hist " + hist + "\n");
			w.write("params " + numParams() + "\n");
			final int[] all = flatten();
			final StringBuilder sb = new StringBuilder();
			for (int i = 0; i < all.length; i++) {
				sb.append(all[i]).append(i % 16 == 15 || i == all.length - 1 ? '\n' : ' ');
			}
			w.write(sb.toString());
			w.write("checksum " + checksum(all) + "\n");
		}
	}

	/**
	 * @throws IOException file illeggibile o non valido: intestazione/versione/feature/scale/dimensioni sbagliate, valori mancanti, in eccesso,
	 *                     non numerici o fuori da {@code short}, checksum errato, pesi che andrebbero in overflow
	 */
	public static TemporalNnueNetwork load(String path) throws IOException {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			final String first = r.readLine();
			if (first == null || !first.startsWith(MAGIC_PREFIX)) {
				throw new IOException("Formato file pesi non valido (atteso '" + MAGIC + "'): " + path);
			}
			if (!first.equals(MAGIC)) {
				throw new IOException("Versione del formato non supportata ('" + first + "', questo codice legge '" + MAGIC + "'): " + path);
			}
			try {
				final String features = expect(r, "features").trim();
				final int hidden = Integer.parseInt(expect(r, "hidden").trim());
				final int l2 = Integer.parseInt(expect(r, "l2").trim());
				final int qa = Integer.parseInt(expect(r, "qa").trim());
				final int qw = Integer.parseInt(expect(r, "qw").trim());
				final int hist = Integer.parseInt(expect(r, "hist").trim());
				if (!features.equals(NnueNetwork.FEATURE_SET + "+hist" + hist)) {
					throw new IOException("Insieme di feature '" + features + "' diverso da quello atteso ('" + NnueNetwork.FEATURE_SET + "+hist" + hist + "'): " + path);
				}
				if (qa != NnueNetwork.QA || qw != NnueNetwork.QW) {
					throw new IOException("Scale di quantizzazione qa=" + qa + " qw=" + qw + " diverse da quelle del codice: " + path);
				}
				if (hidden < 1 || hidden > NnueNetwork.MAX_HIDDEN || l2 < 1 || l2 > NnueNetwork.MAX_L2 || hist < 1 || hist > TemporalHistory.MAX_AGES) {
					throw new IOException("Dimensioni non valide hidden=" + hidden + " l2=" + l2 + " hist=" + hist + ": " + path);
				}
				final int np = Integer.parseInt(expect(r, "params").trim());
				if (np != paramCount(hidden, l2, hist)) {
					throw new IOException("params=" + np + " incoerente con le dimensioni (attesi " + paramCount(hidden, l2, hist) + "): " + path);
				}
				final int[] all = new int[np];
				int k = 0;
				String line;
				while (k < np && (line = r.readLine()) != null) {
					if (line.startsWith("checksum ")) {
						break;
					}
					final StringTokenizer tk = new StringTokenizer(line);
					while (tk.hasMoreTokens()) {
						if (k >= np) {
							throw new IOException("Valori in eccesso rispetto a params=" + np + ": " + path);
						}
						all[k++] = Integer.parseInt(tk.nextToken());
					}
				}
				if (k != np) {
					throw new IOException("File pesi troncato: letti " + k + " valori su " + np + " (" + path + ")");
				}
				final long declared = Long.parseLong(expect(r, "checksum").trim());
				if (declared != checksum(all)) {
					throw new IOException("Checksum errato (file corrotto o modificato): " + path);
				}
				String extra;
				while ((extra = r.readLine()) != null) {
					if (!extra.trim().isEmpty()) {
						throw new IOException("Contenuto inatteso dopo il checksum: '" + extra + "' (" + path + ")");
					}
				}
				return build(hidden, l2, hist, all, path);
			} catch (IllegalArgumentException e) {
				throw new IOException("File pesi non valido (" + path + "): " + e.getMessage(), e);
			}
		}
	}

	private static TemporalNnueNetwork build(int hidden, int l2, int hist, int[] all, String path) throws IOException {
		int k = 0;
		final short[] ftW = new short[NnueNetwork.NUM_FEATURES * hidden];
		for (int i = 0; i < ftW.length; i++) {
			ftW[i] = checkedShort(all[k++], "ftW", i, path);
		}
		final short[] ftB = new short[hidden];
		for (int i = 0; i < hidden; i++) {
			ftB[i] = checkedShort(all[k++], "ftB", i, path);
		}
		final short[][] l1W = new short[NnueNetwork.ROLES][l2 * hidden];
		final int[][] l1B = new int[NnueNetwork.ROLES][l2];
		final short[][] l2W = new short[NnueNetwork.ROLES][l2];
		final int[] l2B = new int[NnueNetwork.ROLES];
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			for (int i = 0; i < l1W[r].length; i++) {
				l1W[r][i] = checkedShort(all[k++], "l1W[" + r + "]", i, path);
			}
			for (int i = 0; i < l2; i++) {
				l1B[r][i] = all[k++];
			}
			for (int i = 0; i < l2; i++) {
				l2W[r][i] = checkedShort(all[k++], "l2W[" + r + "]", i, path);
			}
			l2B[r] = all[k++];
		}
		final short[] tW = new short[TemporalHistory.numFeatures(hist) * hidden];
		for (int i = 0; i < tW.length; i++) {
			tW[i] = checkedShort(all[k++], "tW", i, path);
		}
		return new TemporalNnueNetwork(new NnueNetwork(hidden, l2, ftW, ftB, l1W, l1B, l2W, l2B), hist, tW);
	}

	private static short checkedShort(int v, String name, int index, String path) throws IOException {
		if (v < Short.MIN_VALUE || v > Short.MAX_VALUE) {
			throw new IOException("Valore fuori dall'intervallo short in " + name + "[" + index + "] = " + v + " (" + path + ")");
		}
		return (short) v;
	}

	private static String expect(BufferedReader r, String key) throws IOException {
		final String line = r.readLine();
		if (line == null || !line.startsWith(key + " ")) {
			throw new IOException("Atteso '" + key + "', trovato '" + line + "'");
		}
		return line.substring(key.length() + 1);
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof TemporalNnueNetwork)) {
			return false;
		}
		final TemporalNnueNetwork n = (TemporalNnueNetwork) o;
		return hist == n.hist && base.equals(n.base) && Arrays.equals(tW, n.tW);
	}

	@Override
	public int hashCode() {
		return base.hashCode() * 31 + hist + Arrays.hashCode(tW);
	}
}
