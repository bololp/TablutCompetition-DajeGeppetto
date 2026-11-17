package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.zip.CRC32;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;

/**
 * Genoma dell'algoritmo genetico: un INDIVIDUO e' un vettore di numeri reali in [-1, 1] che codifica, come SCOSTAMENTI interi, un
 * sottoinsieme dei parametri della rete NNUE di riferimento (la "base", tipicamente {@code pesi_nnue/trained_v6.nnue}).
 *
 * <h3>Cosa si evolve (e cosa NO)</h3>
 * Si evolvono solo parametri a bassa dimensione e a effetto diretto sul valore, scelti per gruppi ({@link Group}):
 * <ul>
 * <li>{@code HEAD}: per ciascuna delle 2 teste (lato al tratto) i {@code L2} pesi {@code l2W} dell'ultimo strato e il bias {@code l2B}
 * (con L2=16: 34 geni);</li>
 * <li>{@code L1B}: per ciascuna testa i {@code L2} bias {@code l1B} del secondo strato (32 geni);</li>
 * <li>{@code FTB}: i {@code H} bias {@code ftB} dello strato delle feature (64 geni).</li>
 * </ul>
 * NON si evolvono i pesi {@code ftW} (15 552) e {@code l1W} (2 048): troppo numerosi per una GA con partite come fitness, e sono
 * quelli per cui il progetto ha gia' un addestramento a gradiente ({@code NnueTrainer}). L'architettura, la quantizzazione ({@code QA},
 * {@code QW}), il formato file e il percorso incrementale di {@code NnueSession} restano quelli del riferimento: il risultato
 * della decodifica e' una normale {@link NnueNetwork} caricabile da {@code EvaluatorLoader}.
 *
 * <h3>Codifica</h3>
 * Il gene {@code g_i} in [-1,1] diventa lo scostamento intero {@code d_i = round(g_i * R_i)} che si SOMMA al parametro quantizzato della
 * base ({@code R_i} = {@link Layout#range(int)}, derivato dalla deviazione standard dei parametri della base, si veda {@link Layout}).
 * Il genoma nullo ha tutti gli scostamenti a 0 e la decodifica restituisce una rete IDENTICA bit per bit alla base: l'elemento neutro
 * che permette di garantire "mai peggio del riferimento" (si veda {@link GaTrainer}). Piu' geni diversi possono dare la stessa rete
 * (arrotondamento): e' un plateau neutro, non un errore.
 *
 * <h3>File</h3>
 * Testo, versionato, con l'impronta della base (per rifiutare un genoma applicato a una rete diversa da quella per cui e' stato evoluto)
 * e un CRC32 dei geni:
 * <pre>
 * TABLUT_GA_GENOME v1
 * base &lt;impronta CRC32 dei parametri della base, 8 cifre esadecimali&gt;
 * groups HEAD,L1B
 * genes &lt;n&gt;
 * &lt;n numeri, 8 per riga ({@code Double.toString}: andata e ritorno esatti)&gt;
 * checksum &lt;CRC32 dei geni come long IEEE big-endian, 8 cifre esadecimali&gt;
 * </pre>
 * {@link #load}/{@link #parse} sono severi: ogni difetto e' una {@link IOException}, mai uno scarto silenzioso.
 */
public final class GaGenome {

	public static final String MAGIC = "TABLUT_GA_GENOME v1";

	/** Gruppi di parametri evolvibili. L'ordine di dichiarazione e' l'ordine dei geni nel vettore. */
	public enum Group {
		HEAD, L1B, FTB;

		static Group parse(String s) {
			for (Group g : values()) {
				if (g.name().equalsIgnoreCase(s.trim())) {
					return g;
				}
			}
			throw new IllegalArgumentException("gruppo sconosciuto '" + s + "' (ammessi: HEAD, L1B, FTB)");
		}
	}

	private static final int K_L2W = 0, K_L2B = 1, K_L1B = 2, K_FTB = 3;

	/** Frazione della deviazione standard dei pesi {@code l2W} della base usata come ampiezza massima dello scostamento. */
	static final double RANGE_FRAC_L2W = 0.5;
	/** Idem per i bias {@code l1B}. */
	static final double RANGE_FRAC_L1B = 0.25;
	/** Idem per i bias {@code ftB}. */
	static final double RANGE_FRAC_FTB = 0.25;
	/** Ampiezza massima dello scostamento di {@code l2B}: un decimo di logit (QA*QW/10 = 1638 unita' del logit intero). */
	static final int RANGE_L2B = NnueNetwork.QA * NnueNetwork.QW / 10;

	/**
	 * La struttura del vettore genetico per una base e un insieme di gruppi: quanti geni, a quale parametro corrisponde ciascuno e la sua
	 * ampiezza {@code R_i}. Le ampiezze sono DERIVATE dalla base (non scelte a mano): {@code l2W}, {@code l1B}, {@code ftB} usano una frazione
	 * ({@link #RANGE_FRAC_L2W}, {@link #RANGE_FRAC_L1B}, {@link #RANGE_FRAC_FTB}) della deviazione standard dei loro valori nella base
	 * (almeno 2), {@code l2B} un decimo di logit.
	 */
	public static final class Layout {
		private final NnueNetwork base;
		private final Group[] groups;
		private final int[] kind;
		private final int[] role;
		private final int[] pos;
		private final int[] range;
		private final long fingerprint;

		private Layout(NnueNetwork base, Set<Group> chosen) {
			this.base = base;
			final List<Group> gl = new ArrayList<>();
			for (Group g : Group.values()) {
				if (chosen.contains(g)) {
					gl.add(g);
				}
			}
			if (gl.isEmpty()) {
				throw new IllegalArgumentException("nessun gruppo di geni selezionato");
			}
			this.groups = gl.toArray(new Group[0]);
			final int h = base.hidden(), l2 = base.l2();
			final int rW = rangeOf(pooled(base.l2Weights(0), base.l2Weights(1)), RANGE_FRAC_L2W);
			final int rL1B = rangeOf(pooledInt(base.l1Biases(0), base.l1Biases(1)), RANGE_FRAC_L1B);
			final int rFtB = rangeOf(toDouble(base.ftBiases()), RANGE_FRAC_FTB);
			final List<int[]> genes = new ArrayList<>(); // {kind, role, pos, range}
			for (Group g : groups) {
				switch (g) {
				case HEAD:
					for (int r = 0; r < NnueNetwork.ROLES; r++) {
						for (int i = 0; i < l2; i++) {
							genes.add(new int[] { K_L2W, r, i, rW });
						}
						genes.add(new int[] { K_L2B, r, 0, RANGE_L2B });
					}
					break;
				case L1B:
					for (int r = 0; r < NnueNetwork.ROLES; r++) {
						for (int i = 0; i < l2; i++) {
							genes.add(new int[] { K_L1B, r, i, rL1B });
						}
					}
					break;
				case FTB:
					for (int j = 0; j < h; j++) {
						genes.add(new int[] { K_FTB, -1, j, rFtB });
					}
					break;
				default:
					throw new IllegalStateException();
				}
			}
			final int n = genes.size();
			kind = new int[n];
			role = new int[n];
			pos = new int[n];
			range = new int[n];
			for (int i = 0; i < n; i++) {
				final int[] e = genes.get(i);
				kind[i] = e[0];
				role[i] = e[1];
				pos[i] = e[2];
				range[i] = e[3];
			}
			this.fingerprint = fingerprint(base);
		}

		/** @throws IllegalArgumentException base nulla o nessun gruppo */
		public static Layout of(NnueNetwork base, Collection<Group> groups) {
			if (base == null) {
				throw new IllegalArgumentException("rete di base nulla");
			}
			return new Layout(base, groups.isEmpty() ? EnumSet.noneOf(Group.class) : EnumSet.copyOf(groups));
		}

		/** Come {@link #of(NnueNetwork, Collection)} con i gruppi in testo, separati da virgola ({@code "head,l1b"}). */
		public static Layout of(NnueNetwork base, String groupsText) {
			final Set<Group> s = EnumSet.noneOf(Group.class);
			for (String t : groupsText.split(",")) {
				if (!t.trim().isEmpty()) {
					s.add(Group.parse(t));
				}
			}
			return of(base, s);
		}

		public NnueNetwork base() {
			return base;
		}

		public int size() {
			return kind.length;
		}

		/** Ampiezza massima (in unita' quantizzate) dello scostamento del gene {@code i}. */
		public int range(int i) {
			return range[i];
		}

		public long baseFingerprint() {
			return fingerprint;
		}

		public String groupsText() {
			final StringBuilder sb = new StringBuilder();
			for (int i = 0; i < groups.length; i++) {
				sb.append(i == 0 ? "" : ",").append(groups[i].name());
			}
			return sb.toString();
		}

		/** Nome leggibile del gene (per i log): es. {@code l2W[1][7]}. */
		public String geneName(int i) {
			switch (kind[i]) {
			case K_L2W:
				return "l2W[" + role[i] + "][" + pos[i] + "]";
			case K_L2B:
				return "l2B[" + role[i] + "]";
			case K_L1B:
				return "l1B[" + role[i] + "][" + pos[i] + "]";
			default:
				return "ftB[" + pos[i] + "]";
			}
		}

		private static double[] toDouble(short[] a) {
			final double[] r = new double[a.length];
			for (int i = 0; i < a.length; i++) {
				r[i] = a[i];
			}
			return r;
		}

		private static double[] pooled(short[] a, short[] b) {
			final double[] r = new double[a.length + b.length];
			for (int i = 0; i < a.length; i++) {
				r[i] = a[i];
			}
			for (int i = 0; i < b.length; i++) {
				r[a.length + i] = b[i];
			}
			return r;
		}

		private static double[] pooledInt(int[] a, int[] b) {
			final double[] r = new double[a.length + b.length];
			for (int i = 0; i < a.length; i++) {
				r[i] = a[i];
			}
			for (int i = 0; i < b.length; i++) {
				r[a.length + i] = b[i];
			}
			return r;
		}

		private static int rangeOf(double[] v, double frac) {
			double m = 0;
			for (double x : v) {
				m += x;
			}
			m /= v.length;
			double s = 0;
			for (double x : v) {
				s += (x - m) * (x - m);
			}
			final double sd = Math.sqrt(s / v.length);
			return Math.max(2, (int) Math.round(frac * sd));
		}
	}

	private final Layout layout;
	private final double[] genes;

	/**
	 * @throws IllegalArgumentException lunghezza diversa da {@link Layout#size()}, gene non finito o fuori da [-1,1]
	 */
	public GaGenome(Layout layout, double[] genes) {
		if (layout == null || genes == null) {
			throw new IllegalArgumentException("layout o geni nulli");
		}
		if (genes.length != layout.size()) {
			throw new IllegalArgumentException("attesi " + layout.size() + " geni, trovati " + genes.length);
		}
		for (int i = 0; i < genes.length; i++) {
			if (Double.isNaN(genes[i]) || genes[i] < -1.0 || genes[i] > 1.0) {
				throw new IllegalArgumentException("gene " + i + " (" + layout.geneName(i) + ") = " + genes[i] + " fuori da [-1,1] o non finito");
			}
		}
		this.layout = layout;
		this.genes = genes.clone();
	}

	/** L'elemento neutro: tutti i geni a zero, decodifica = rete di base. */
	public static GaGenome identity(Layout layout) {
		return new GaGenome(layout, new double[layout.size()]);
	}

	public Layout layout() {
		return layout;
	}

	public int size() {
		return genes.length;
	}

	public double gene(int i) {
		return genes[i];
	}

	public double[] genes() {
		return genes.clone();
	}

	/** Gli scostamenti interi {@code round(g_i * R_i)} che la decodifica somma ai parametri della base. */
	public int[] deltas() {
		final int[] d = new int[genes.length];
		for (int i = 0; i < d.length; i++) {
			d[i] = (int) Math.round(genes[i] * layout.range[i]);
		}
		return d;
	}

	/** Vero se tutti gli scostamenti sono 0: la rete decodificata e' la base (anche con geni non nulli, per l'arrotondamento). */
	public boolean isNeutral() {
		for (int v : deltas()) {
			if (v != 0) {
				return false;
			}
		}
		return true;
	}

	/** Media dei quadrati dei geni, in [0,1]: la "distanza" dal riferimento, per la pressione di parsimonia della fitness. */
	public double meanSquare() {
		double s = 0;
		for (double g : genes) {
			s += g * g;
		}
		return s / genes.length;
	}

	private static short clampShort(long v) {
		return (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, v));
	}

	/**
	 * La rete corrispondente: i parametri della base con gli scostamenti sommati (i {@code short} saturano, gli {@code int} no). La base non
	 * viene modificata. Il costruttore di {@link NnueNetwork} verifica comunque l'assenza di overflow (con questi intervalli non puo' accadere).
	 */
	public NnueNetwork decode() {
		final NnueNetwork b = layout.base;
		final int h = b.hidden(), l2 = b.l2();
		final short[] ftW = b.ftWeights();
		final short[] ftB = b.ftBiases();
		final short[][] l1W = { b.l1Weights(0), b.l1Weights(1) };
		final int[][] l1B = { b.l1Biases(0), b.l1Biases(1) };
		final short[][] l2W = { b.l2Weights(0), b.l2Weights(1) };
		final int[] l2B = { b.l2Bias(0), b.l2Bias(1) };
		final int[] d = deltas();
		for (int g = 0; g < d.length; g++) {
			if (d[g] == 0) {
				continue;
			}
			switch (layout.kind[g]) {
			case K_L2W:
				l2W[layout.role[g]][layout.pos[g]] = clampShort((long) l2W[layout.role[g]][layout.pos[g]] + d[g]);
				break;
			case K_L2B:
				l2B[layout.role[g]] = Math.addExact(l2B[layout.role[g]], d[g]);
				break;
			case K_L1B:
				l1B[layout.role[g]][layout.pos[g]] = Math.addExact(l1B[layout.role[g]][layout.pos[g]], d[g]);
				break;
			default:
				ftB[layout.pos[g]] = clampShort((long) ftB[layout.pos[g]] + d[g]);
				break;
			}
		}
		return new NnueNetwork(h, l2, ftW, ftB, l1W, l1B, l2W, l2B);
	}

	// ====================================================================== impronta della base

	/** CRC32 dei parametri (dimensioni e tutti gli array, nell'ordine del file, big-endian): identifica la base indipendentemente dal testo del file. */
	public static long fingerprint(NnueNetwork n) {
		final CRC32 crc = new CRC32();
		final ByteBuffer bb = ByteBuffer.allocate(4);
		crc.update(int4(bb, n.hidden()), 0, 4);
		crc.update(int4(bb, n.l2()), 0, 4);
		for (short v : n.ftWeights()) {
			crc.update(int4(bb, v), 0, 4);
		}
		for (short v : n.ftBiases()) {
			crc.update(int4(bb, v), 0, 4);
		}
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			for (short v : n.l1Weights(r)) {
				crc.update(int4(bb, v), 0, 4);
			}
			for (int v : n.l1Biases(r)) {
				crc.update(int4(bb, v), 0, 4);
			}
			for (short v : n.l2Weights(r)) {
				crc.update(int4(bb, v), 0, 4);
			}
			crc.update(int4(bb, n.l2Bias(r)), 0, 4);
		}
		return crc.getValue();
	}

	private static byte[] int4(ByteBuffer bb, int v) {
		bb.clear();
		bb.putInt(v);
		return bb.array();
	}

	private long geneChecksum() {
		final CRC32 crc = new CRC32();
		final ByteBuffer bb = ByteBuffer.allocate(8);
		for (double g : genes) {
			bb.clear();
			bb.putLong(Double.doubleToLongBits(g));
			crc.update(bb.array(), 0, 8);
		}
		return crc.getValue();
	}

	// ====================================================================== file

	public String toText() {
		final StringBuilder sb = new StringBuilder();
		sb.append(MAGIC).append('\n');
		sb.append("base ").append(String.format("%08x", layout.fingerprint)).append('\n');
		sb.append("groups ").append(layout.groupsText()).append('\n');
		sb.append("genes ").append(genes.length).append('\n');
		for (int i = 0; i < genes.length; i++) {
			sb.append(Double.toString(genes[i]));
			sb.append((i % 8 == 7 || i == genes.length - 1) ? '\n' : ' ');
		}
		sb.append("checksum ").append(String.format("%08x", geneChecksum())).append('\n');
		return sb.toString();
	}

	public void save(String path) throws IOException {
		final File f = new File(path);
		final File parent = f.getAbsoluteFile().getParentFile();
		if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
			throw new IOException("impossibile creare la cartella " + parent);
		}
		try (Writer w = new OutputStreamWriter(new FileOutputStream(f), "UTF-8")) {
			w.write(toText());
		}
	}

	/**
	 * @throws IOException file assente, intestazione/versione diverse, gruppi sconosciuti, impronta della base diversa da quella di
	 *         {@code base}, numero di geni sbagliato, valori non numerici / non finiti / fuori da [-1,1], checksum errato, righe in eccesso
	 */
	public static GaGenome load(String path, NnueNetwork base) throws IOException {
		final StringBuilder sb = new StringBuilder();
		try (BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"))) {
			String line;
			while ((line = r.readLine()) != null) {
				sb.append(line).append('\n');
			}
		}
		return parse(sb.toString(), base);
	}

	public static GaGenome parse(String text, NnueNetwork base) throws IOException {
		final String[] lines = text.split("\n", -1);
		int li = 0;
		if (lines.length < 6 || !MAGIC.equals(lines[li++].trim())) {
			throw new IOException("non e' un genoma " + MAGIC);
		}
		final String baseLine = lines[li++].trim();
		if (!baseLine.startsWith("base ")) {
			throw new IOException("manca la riga 'base'");
		}
		final long fp;
		try {
			fp = Long.parseLong(baseLine.substring(5).trim(), 16);
		} catch (NumberFormatException e) {
			throw new IOException("impronta della base non valida: " + baseLine);
		}
		final String groupsLine = lines[li++].trim();
		if (!groupsLine.startsWith("groups ")) {
			throw new IOException("manca la riga 'groups'");
		}
		final Layout layout;
		try {
			layout = Layout.of(base, groupsLine.substring(7));
		} catch (IllegalArgumentException e) {
			throw new IOException("gruppi non validi: " + e.getMessage());
		}
		if (fp != layout.fingerprint) {
			throw new IOException(String.format("il genoma e' stato evoluto per un'altra rete di base (impronta %08x nel file, %08x quella fornita)", fp, layout.fingerprint));
		}
		final String nLine = lines[li++].trim();
		if (!nLine.startsWith("genes ")) {
			throw new IOException("manca la riga 'genes'");
		}
		final int n;
		try {
			n = Integer.parseInt(nLine.substring(6).trim());
		} catch (NumberFormatException e) {
			throw new IOException("numero di geni non valido: " + nLine);
		}
		if (n != layout.size()) {
			throw new IOException("il file dichiara " + n + " geni ma i gruppi " + layout.groupsText() + " ne hanno " + layout.size());
		}
		final double[] g = new double[n];
		int got = 0;
		while (li < lines.length && !lines[li].trim().startsWith("checksum ")) {
			for (String tok : lines[li].trim().split("\\s+")) {
				if (tok.isEmpty()) {
					continue;
				}
				if (got >= n) {
					throw new IOException("geni in eccesso (attesi " + n + ")");
				}
				try {
					g[got] = Double.parseDouble(tok);
				} catch (NumberFormatException e) {
					throw new IOException("gene non numerico: '" + tok + "'");
				}
				got++;
			}
			li++;
		}
		if (got != n) {
			throw new IOException("geni mancanti: trovati " + got + " su " + n);
		}
		if (li >= lines.length) {
			throw new IOException("manca la riga 'checksum'");
		}
		final String cs = lines[li++].trim();
		for (; li < lines.length; li++) {
			if (!lines[li].trim().isEmpty()) {
				throw new IOException("contenuto dopo il checksum");
			}
		}
		final GaGenome genome;
		try {
			genome = new GaGenome(layout, g);
		} catch (IllegalArgumentException e) {
			throw new IOException(e.getMessage());
		}
		final String expected = "checksum " + String.format("%08x", genome.geneChecksum());
		if (!expected.equals(cs)) {
			throw new IOException("checksum errato: nel file '" + cs + "', calcolato '" + expected + "'");
		}
		return genome;
	}

	// ====================================================================== identita'

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof GaGenome)) {
			return false;
		}
		final GaGenome g = (GaGenome) o;
		return layout.fingerprint == g.layout.fingerprint && layout.groupsText().equals(g.layout.groupsText()) && Arrays.equals(genes, g.genes);
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(genes) * 31 + (int) layout.fingerprint;
	}

	/** Chiave di testo esatta dei geni (per le mappe della GA). */
	String key() {
		return Arrays.toString(genes);
	}

	@Override
	public String toString() {
		return "GaGenome[" + layout.groupsText() + ", " + genes.length + " geni, ms=" + String.format("%.4f", meanSquare()) + (isNeutral() ? ", neutro" : "") + "]";
	}
}
