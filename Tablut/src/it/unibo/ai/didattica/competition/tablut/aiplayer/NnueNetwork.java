package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.zip.CRC32;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.ZobristHash;

/**
 * Pesi e aritmetica di una rete NNUE per Tablut. Immutabile dopo la costruzione, quindi condivisibile fra thread: lo stato che cambia
 * (accumulatori) sta altrove ({@link NnueEvaluator}, {@link NnueSession}).
 *
 * <h3>Feature (ingresso sparso)</h3>
 * Una feature per ogni coppia (tipo di pedina, cella): {@code f = base(tipo) + cella}, con {@code base(BIANCO) = 0},
 * {@code base(NERO) = 81}, {@code base(RE) = 162}: {@value #NUM_FEATURES} feature, ne sono attive al piu' 25 (8 + 16 + 1) in una posizione
 * legale. La prospettiva e' UNICA e ASSOLUTA (difensori = Bianco, attaccanti = Nero, mai specchiata per il lato al tratto): in Tablut i
 * due lati sono asimmetrici, quindi non c'e' parametro da condividere fra prospettive, e il Re e' semplicemente un tipo di pedina con i
 * suoi 81 pesi. Conseguenza: una mossa del Re e' un normale aggiornamento (togli (RE, da), aggiungi (RE, a)), non serve mai una
 * ricostruzione. Trono, cittadelle e vie di fuga sono geometria STATICA: sono gia' codificati dalla cella di ogni feature (il peso di
 * una pedina nera su una cittadella e' un parametro proprio), e il trono vuoto e' implicito nell'assenza della feature (RE, trono);
 * non servono feature dedicate. Il lato al tratto NON entra nell'accumulatore: sceglie la testa (strati 1-2) da usare.
 *
 * <h3>Rete (tutto in interi)</h3>
 * <pre>
 *   acc[j]  = ftB[j] + somma_{f attiva} ftW[f][j]                    j = 0..H-1   (int, "accumulatore": si aggiorna per differenza)
 *   a[j]    = clamp(acc[j], 0, QA)                                    clipped ReLU
 *   u[i]    = l1B[r][i] + somma_j l1W[r][i][j] * a[j]                 i = 0..L2-1, r = lato al tratto (0 Bianco, 1 Nero)
 *   c[i]    = clamp((u[i] + QW/2) >> QW_SHIFT, 0, QA)                 clipped ReLU, riportata alla scala QA
 *   z       = l2B[r] + somma_i l2W[r][i] * c[i]                       logit in virgola fissa, scala QA*QW
 *   valore  = tanh(z / (QA*QW))                                       in (-1,1), +1 = vince il Bianco
 * </pre>
 * {@code QA = }{@value #QA}, {@code QW = }{@value #QW}. I pesi di {@code ftW/ftB} sono {@code short} a scala QA; {@code l1W, l2W} sono
 * {@code short} a scala QW; {@code l1B, l2B} sono {@code int} a scala QA*QW. La tanh e' campionata a passo 1/16 di logit su [-8,8] con
 * interpolazione lineare (errore &lt; 4e-4 sul valore): dipende dal solo intero {@code z}, quindi e' ESATTAMENTE riproducibile. Il segno
 * e la scala coincidono con quelli di {@link CnnEvaluator}/{@link NeuralEvaluator} ({@link NnueEvaluator#outputScale} porta il valore
 * sulla scala di {@link SimpleEvaluator}), cosi' {@link Negamax}, MCTS e client non cambiano.
 * <p>
 * Poiche' l'aritmetica e' intera, la somma dell'accumulatore e' associativa e commutativa (anche in caso di overflow, modulo 2^32): lo
 * stato aggiornato per differenze e quello ricostruito da zero sono IDENTICI, qualunque sia l'ordine delle mosse. Il costruttore
 * verifica comunque che nessuna somma possa andare in overflow (limiti conservativi sui pesi), cosi' i clamp operano sempre sul valore vero.
 *
 * <h3>File</h3>
 * Testo, intestazione {@value #MAGIC}: righe {@code features, hidden, l2, qa, qw, params}, poi tutti gli interi (16 per riga) nell'ordine
 * {@code ftW[243*H] (feature per feature), ftB[H]}, poi per ciascun lato (0, 1): {@code l1W[L2*H] (uscita per uscita), l1B[L2], l2W[L2],
 * l2B[1]}; infine {@code checksum} (CRC32 dei valori come interi big-endian). Si rifiuta (IOException) ogni file con intestazione,
 * versione, insieme di feature, scale, dimensioni, numero di valori, intervalli o checksum incoerenti.
 */
public final class NnueNetwork {

	public static final String MAGIC_PREFIX = "TABLUT_NNUE";
	public static final String MAGIC = MAGIC_PREFIX + " v1";
	/** Identifica la codifica delle feature: un file con un altro insieme di feature NON e' leggibile da questo codice. */
	public static final String FEATURE_SET = "psq3x81";

	public static final int NUM_SQUARES = BoardConstants.NUM_SQUARES;
	public static final int NUM_FEATURES = 3 * NUM_SQUARES;
	/** Scala delle attivazioni: la clipped ReLU opera in [0, QA]. */
	public static final int QA = 256;
	public static final int QW_SHIFT = 6;
	/** Scala dei pesi degli strati 1 e 2. */
	public static final int QW = 1 << QW_SHIFT;
	public static final int ROLES = 2;
	public static final int MAX_HIDDEN = 1024;
	public static final int MAX_L2 = 256;

	/** Base della feature per codice di pedina ({@link BoardConstants}): EMPTY e THRONE_MARK non hanno feature. */
	private static final int[] BASE = { -1, 0, NUM_SQUARES, -1, 2 * NUM_SQUARES };

	/** {@code FEATURE_KEY[f]} = chiave Zobrist di (cella, pedina) di quella feature: la firma dell'insieme di feature attive. */
	static final long[] FEATURE_KEY = new long[NUM_FEATURES];

	// tanh campionata: passo 1/16 di logit su [-8, 8]. z ha scala QA*QW = 2^14, quindi un passo = 2^10 unita'.
	private static final int Z_RANGE = 8 * QA * QW; // 131072 = 8 logit
	private static final int LUT_SHIFT = 10;
	private static final double[] TANH = new double[(2 * Z_RANGE >> LUT_SHIFT) + 2];

	static {
		final int[] pieces = { BoardConstants.WHITE, BoardConstants.BLACK, BoardConstants.KING };
		for (int p : pieces) {
			for (int sq = 0; sq < NUM_SQUARES; sq++) {
				FEATURE_KEY[BASE[p] + sq] = ZobristHash.KEY[sq][p];
			}
		}
		for (int k = 0; k < TANH.length; k++) {
			TANH[k] = StrictMath.tanh((k - (Z_RANGE >> LUT_SHIFT)) / 16.0);
		}
	}

	private final int hidden;
	private final int l2;
	private final short[] ftW; // [NUM_FEATURES][hidden]: la colonna di una feature e' contigua
	private final short[] ftB; // [hidden]
	private final short[][] l1W; // [ruolo][l2*hidden], uscita per uscita (forma del file)
	private final short[][] l1WT; // [ruolo][hidden*l2]: la stessa matrice trasposta, per scorrere le uscite di un ingresso in memoria contigua
	private final int[][] l1B; // [ruolo][l2]
	private final short[][] l2W; // [ruolo][l2]
	private final int[] l2B; // [ruolo]

	/**
	 * @throws IllegalArgumentException dimensioni fuori limite, array di forma sbagliata o pesi che potrebbero andare in overflow
	 */
	public NnueNetwork(int hidden, int l2, short[] ftW, short[] ftB, short[][] l1W, int[][] l1B, short[][] l2W, int[] l2B) {
		if (hidden < 1 || hidden > MAX_HIDDEN) {
			throw new IllegalArgumentException("hidden=" + hidden + " fuori dall'intervallo 1.." + MAX_HIDDEN);
		}
		if (l2 < 1 || l2 > MAX_L2) {
			throw new IllegalArgumentException("l2=" + l2 + " fuori dall'intervallo 1.." + MAX_L2);
		}
		requireLength(ftW, NUM_FEATURES * hidden, "ftW");
		requireLength(ftB, hidden, "ftB");
		requireRoles(l1W == null ? -1 : l1W.length, "l1W");
		requireRoles(l1B == null ? -1 : l1B.length, "l1B");
		requireRoles(l2W == null ? -1 : l2W.length, "l2W");
		requireRoles(l2B == null ? -1 : l2B.length, "l2B");
		this.hidden = hidden;
		this.l2 = l2;
		this.ftW = ftW.clone();
		this.ftB = ftB.clone();
		this.l1W = new short[ROLES][];
		this.l1WT = new short[ROLES][];
		this.l1B = new int[ROLES][];
		this.l2W = new short[ROLES][];
		this.l2B = l2B.clone();
		for (int r = 0; r < ROLES; r++) {
			requireLength(l1W[r], l2 * hidden, "l1W[" + r + "]");
			requireLength(l1B[r], l2, "l1B[" + r + "]");
			requireLength(l2W[r], l2, "l2W[" + r + "]");
			this.l1W[r] = l1W[r].clone();
			this.l1WT[r] = new short[l2 * hidden];
			for (int i = 0; i < l2; i++) {
				for (int j = 0; j < hidden; j++) {
					this.l1WT[r][j * l2 + i] = l1W[r][i * hidden + j];
				}
			}
			this.l1B[r] = l1B[r].clone();
			this.l2W[r] = l2W[r].clone();
		}
		verifyNoOverflow();
	}

	private static void requireLength(short[] a, int n, String name) {
		if (a == null || a.length != n) {
			throw new IllegalArgumentException(name + ": attesi " + n + " valori, trovati " + (a == null ? "null" : String.valueOf(a.length)));
		}
	}

	private static void requireLength(int[] a, int n, String name) {
		if (a == null || a.length != n) {
			throw new IllegalArgumentException(name + ": attesi " + n + " valori, trovati " + (a == null ? "null" : String.valueOf(a.length)));
		}
	}

	private static void requireRoles(int n, String name) {
		if (n != ROLES) {
			throw new IllegalArgumentException(name + ": servono " + ROLES + " teste (una per lato al tratto), trovate " + n);
		}
	}

	/**
	 * Limiti CONSERVATIVI: nessuna somma intera puo' superare il campo di un int per nessuna posizione (anche costruita a mano, con una
	 * feature per cella). L'accumulatore e' sommato in modo esatto comunque (aritmetica modulo 2^32), ma con un overflow i clamp
	 * vedrebbero un valore diverso da quello vero.
	 */
	private void verifyNoOverflow() {
		final long limit = Integer.MAX_VALUE - QW;
		for (int j = 0; j < hidden; j++) {
			long sum = Math.abs((long) ftB[j]);
			for (int sq = 0; sq < NUM_SQUARES; sq++) {
				int m = 0;
				for (int t = 0; t < 3; t++) {
					m = Math.max(m, Math.abs((int) ftW[(t * NUM_SQUARES + sq) * hidden + j]));
				}
				sum += m;
			}
			if (sum > limit) {
				throw new IllegalArgumentException("ftW/ftB: l'accumulatore " + j + " potrebbe andare in overflow (somma massima " + sum + ")");
			}
		}
		for (int r = 0; r < ROLES; r++) {
			for (int i = 0; i < l2; i++) {
				long sum = Math.abs((long) l1B[r][i]);
				for (int j = 0; j < hidden; j++) {
					sum += (long) QA * Math.abs((int) l1W[r][i * hidden + j]);
				}
				if (sum > limit) {
					throw new IllegalArgumentException("l1W/l1B[" + r + "]: l'uscita " + i + " potrebbe andare in overflow (somma massima " + sum + ")");
				}
			}
			long sum = Math.abs((long) l2B[r]);
			for (int i = 0; i < l2; i++) {
				sum += (long) QA * Math.abs((int) l2W[r][i]);
			}
			if (sum > limit) {
				throw new IllegalArgumentException("l2W/l2B[" + r + "]: il logit potrebbe andare in overflow (somma massima " + sum + ")");
			}
		}
	}

	public int hidden() {
		return hidden;
	}

	public int l2() {
		return l2;
	}

	/** Numero di valori memorizzati per una rete di queste dimensioni. */
	public static int paramCount(int hidden, int l2) {
		return NUM_FEATURES * hidden + hidden + ROLES * (l2 * hidden + l2 + l2 + 1);
	}

	public int numParams() {
		return paramCount(hidden, l2);
	}

	public String describe() {
		return "NNUE " + FEATURE_SET + " (" + NUM_FEATURES + " -> " + hidden + " -> " + l2 + " -> 1, " + ROLES + " teste per lato al tratto, " + numParams()
				+ " parametri)";
	}

	// ---- copie in sola lettura dei pesi (serializzazione, test, quantizzazione): mai gli array interni ----

	public short[] ftWeights() {
		return ftW.clone();
	}

	public short[] ftBiases() {
		return ftB.clone();
	}

	public short[] l1Weights(int role) {
		return l1W[role].clone();
	}

	public int[] l1Biases(int role) {
		return l1B[role].clone();
	}

	public short[] l2Weights(int role) {
		return l2W[role].clone();
	}

	public int l2Bias(int role) {
		return l2B[role];
	}

	// ====================================================================== feature

	/**
	 * Indice della feature di una pedina in una cella.
	 *
	 * @param piece codice {@link BoardConstants} (WHITE, BLACK, KING)
	 * @throws IllegalArgumentException se {@code piece} non ha feature (vuoto, marcatore del trono, codice sconosciuto) o la cella non esiste
	 */
	public static int featureIndex(int piece, int sq) {
		final int b = piece >= 0 && piece < BASE.length ? BASE[piece] : -1;
		if (b < 0) {
			throw new IllegalArgumentException("il codice pedina " + piece + " non ha una feature");
		}
		if (sq < 0 || sq >= NUM_SQUARES) {
			throw new IllegalArgumentException("cella " + sq + " inesistente");
		}
		return b + sq;
	}

	/** Base della feature per la pedina {@code piece}, o -1 se non ne ha (uso interno: nessun controllo dei limiti sulla cella). */
	static int baseOf(int piece) {
		return piece >= 0 && piece < BASE.length ? BASE[piece] : -1;
	}

	/** Firma dell'insieme di feature attive di {@code s}: la chiave Zobrist della sola scacchiera (senza trono vuoto e senza turno). */
	static long boardSignature(FastState s) {
		long h = s.zobristHash;
		if (s.board[BoardConstants.THRONE] == BoardConstants.EMPTY) {
			h ^= ZobristHash.THRONE_EMPTY_KEY;
		}
		if (s.winner == FastState.WINNER_NONE && s.turn == BoardConstants.BLACK_TURN) {
			h ^= ZobristHash.TURN_KEY;
		}
		return h;
	}

	// ====================================================================== aritmetica

	/**
	 * Ricostruzione COMPLETA dell'accumulatore dalla scacchiera.
	 *
	 * @param acc array di {@link #hidden()} elementi (sovrascritto)
	 * @return la firma dell'insieme di feature attive (XOR delle loro chiavi Zobrist)
	 * @throws IllegalStateException se la scacchiera contiene un codice di pedina senza feature
	 */
	long refresh(byte[] board, int[] acc) {
		final int h = hidden;
		final short[] w = ftW;
		for (int j = 0; j < h; j++) {
			acc[j] = ftB[j];
		}
		long sig = 0L;
		for (int sq = 0; sq < NUM_SQUARES; sq++) {
			final int p = board[sq];
			if (p == BoardConstants.EMPTY) {
				continue;
			}
			final int b = baseOf(p);
			if (b < 0) {
				throw new IllegalStateException("codice pedina non valido " + p + " nella cella " + sq);
			}
			final int f = b + sq;
			final int o = f * h;
			for (int j = 0; j < h; j++) {
				acc[j] += w[o + j];
			}
			sig ^= FEATURE_KEY[f];
		}
		return sig;
	}

	/** {@code dst = src - W[sub] + W[add]}: lo spostamento di una pedina in un solo passaggio sulla memoria. */
	void applyMove(int[] src, int[] dst, int subFeature, int addFeature) {
		final int h = hidden;
		final short[] w = ftW;
		final int s = subFeature * h;
		final int a = addFeature * h;
		for (int j = 0; j < h; j++) {
			dst[j] = src[j] - w[s + j] + w[a + j];
		}
	}

	/** {@code acc -= W[feature]}: una pedina catturata. */
	void subtract(int[] acc, int feature) {
		final int h = hidden;
		final short[] w = ftW;
		final int o = feature * h;
		for (int j = 0; j < h; j++) {
			acc[j] -= w[o + j];
		}
	}

	/** Dimensione del buffer di lavoro che {@link #logit} richiede. */
	int scratchSize() {
		return l2;
	}

	/**
	 * Dall'accumulatore al logit in virgola fissa (scala QA*QW).
	 * <p>
	 * Lo strato 1 e' calcolato "per ingresso": per ogni neurone attivo {@code j} (clipped ReLU {@code > 0}) si somma {@code a[j] * W[j][.]} alle
	 * {@code l2} uscite, sulla matrice trasposta (le uscite di un ingresso sono contigue). Un neurone con {@code acc <= 0} contribuisce
	 * esattamente 0 e si salta; il risultato e' lo STESSO intero del calcolo "per uscita" (somma di interi: l'ordine non conta).
	 *
	 * @param work buffer di {@link #scratchSize()} elementi, solo di lavoro (non conserva nulla fra le chiamate)
	 * @param role 0 = muove il Bianco, 1 = muove il Nero
	 */
	int logit(int[] acc, int[] work, int role) {
		final int h = hidden;
		final int l = l2;
		final short[] wt = l1WT[role];
		final int[] b1 = l1B[role];
		for (int i = 0; i < l; i++) {
			work[i] = b1[i];
		}
		for (int j = 0, o = 0; j < h; j++, o += l) {
			int a = acc[j];
			if (a <= 0) {
				continue;
			}
			if (a > QA) {
				a = QA;
			}
			for (int i = 0; i < l; i++) {
				work[i] += wt[o + i] * a;
			}
		}
		final short[] w2 = l2W[role];
		int z = l2B[role];
		for (int i = 0; i < l; i++) {
			int c = (work[i] + (QW >> 1)) >> QW_SHIFT;
			c = c < 0 ? 0 : (c > QA ? QA : c);
			z += w2[i] * c;
		}
		return z;
	}

	/** Dal logit in virgola fissa al valore in (-1,1): la tanh campionata con interpolazione lineare (dipende dal solo intero {@code z}). */
	static double valueOf(int z) {
		int zc = z < -Z_RANGE ? -Z_RANGE : (z > Z_RANGE - 1 ? Z_RANGE - 1 : z);
		final int t = zc + Z_RANGE;
		final int idx = t >> LUT_SHIFT;
		final double lo = TANH[idx];
		return lo + (TANH[idx + 1] - lo) * ((t & ((1 << LUT_SHIFT) - 1)) * (1.0 / (1 << LUT_SHIFT)));
	}

	// ====================================================================== costruzione di prova

	/**
	 * Rete con pesi casuali (seme fisso) di scala tale che gli accumulatori attraversino tutti e tre i regimi della clipped ReLU
	 * (sotto 0, lineare, saturo): pensata per i test e i benchmark, non per giocare.
	 */
	public static NnueNetwork random(int hidden, int l2, long seed) {
		final Random rnd = new Random(seed);
		final short[] ftW = new short[NUM_FEATURES * hidden];
		for (int i = 0; i < ftW.length; i++) {
			ftW[i] = toShort(rnd.nextGaussian() * 0.30 * QA);
		}
		final short[] ftB = new short[hidden];
		for (int j = 0; j < hidden; j++) {
			ftB[j] = toShort(rnd.nextGaussian() * 0.25 * QA);
		}
		final short[][] l1W = new short[ROLES][l2 * hidden];
		final int[][] l1B = new int[ROLES][l2];
		final short[][] l2W = new short[ROLES][l2];
		final int[] l2B = new int[ROLES];
		final double s1 = 1.0 / Math.sqrt(Math.max(1, hidden / 2));
		for (int r = 0; r < ROLES; r++) {
			for (int i = 0; i < l1W[r].length; i++) {
				l1W[r][i] = toShort(rnd.nextGaussian() * s1 * QW);
			}
			for (int i = 0; i < l2; i++) {
				l1B[r][i] = (int) Math.round(rnd.nextGaussian() * 0.20 * QA * QW);
				l2W[r][i] = toShort(rnd.nextGaussian() * 0.60 * QW);
			}
			l2B[r] = (int) Math.round(rnd.nextGaussian() * 0.10 * QA * QW);
		}
		return new NnueNetwork(hidden, l2, ftW, ftB, l1W, l1B, l2W, l2B);
	}

	private static short toShort(double v) {
		final long r = Math.round(v);
		return (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, r));
	}

	// ====================================================================== file

	/** Vero se il file inizia con l'intestazione dei pesi NNUE (qualunque versione: la versione la controlla {@link #load}). */
	public static boolean isNnueFile(String path) {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			final String first = r.readLine();
			return first != null && first.startsWith(MAGIC_PREFIX);
		} catch (IOException e) {
			return false;
		}
	}

	public void save(String path) throws IOException {
		try (BufferedWriter w = new BufferedWriter(new FileWriter(path))) {
			w.write(MAGIC);
			w.newLine();
			w.write("features " + FEATURE_SET);
			w.newLine();
			w.write("hidden " + hidden);
			w.newLine();
			w.write("l2 " + l2);
			w.newLine();
			w.write("qa " + QA);
			w.newLine();
			w.write("qw " + QW);
			w.newLine();
			w.write("params " + numParams());
			w.newLine();
			final int[] all = flatten();
			final StringBuilder sb = new StringBuilder();
			for (int i = 0; i < all.length; i++) {
				sb.append(all[i]).append(i % 16 == 15 || i == all.length - 1 ? '\n' : ' ');
			}
			w.write(sb.toString());
			w.write("checksum " + checksum(all));
			w.newLine();
		}
	}

	/** Tutti i valori nell'ordine del file. */
	private int[] flatten() {
		final int[] all = new int[numParams()];
		int k = 0;
		for (short v : ftW) {
			all[k++] = v;
		}
		for (short v : ftB) {
			all[k++] = v;
		}
		for (int r = 0; r < ROLES; r++) {
			for (short v : l1W[r]) {
				all[k++] = v;
			}
			for (int v : l1B[r]) {
				all[k++] = v;
			}
			for (short v : l2W[r]) {
				all[k++] = v;
			}
			all[k++] = l2B[r];
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

	/**
	 * Legge un file di pesi NNUE.
	 *
	 * @throws IOException file illeggibile oppure non valido: intestazione/versione/insieme di feature/scale/dimensioni sbagliati, valori
	 *                     mancanti, in eccesso, non numerici o fuori intervallo, checksum errato, pesi che andrebbero in overflow
	 */
	public static NnueNetwork load(String path) throws IOException {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			final String first = r.readLine();
			if (first == null || !first.startsWith(MAGIC_PREFIX)) {
				throw new IOException("Formato file pesi non valido (atteso '" + MAGIC + "'): " + path);
			}
			if (!first.equals(MAGIC)) {
				throw new IOException("Versione del formato NNUE non supportata ('" + first + "', questo codice legge '" + MAGIC + "'): " + path);
			}
			try {
				final String features = expect(r, "features").trim();
				if (!features.equals(FEATURE_SET)) {
					throw new IOException("Insieme di feature '" + features + "' diverso da quello del codice ('" + FEATURE_SET + "'): " + path);
				}
				final int hidden = Integer.parseInt(expect(r, "hidden").trim());
				final int l2 = Integer.parseInt(expect(r, "l2").trim());
				final int qa = Integer.parseInt(expect(r, "qa").trim());
				final int qw = Integer.parseInt(expect(r, "qw").trim());
				if (qa != QA || qw != QW) {
					throw new IOException("Scale di quantizzazione qa=" + qa + " qw=" + qw + " diverse da quelle del codice (qa=" + QA + ", qw=" + QW + "): " + path);
				}
				if (hidden < 1 || hidden > MAX_HIDDEN || l2 < 1 || l2 > MAX_L2) {
					throw new IOException("Dimensioni non valide hidden=" + hidden + " (1.." + MAX_HIDDEN + ") l2=" + l2 + " (1.." + MAX_L2 + "): " + path);
				}
				final int np = Integer.parseInt(expect(r, "params").trim());
				if (np != paramCount(hidden, l2)) {
					throw new IOException("params=" + np + " incoerente con hidden=" + hidden + " l2=" + l2 + " (attesi " + paramCount(hidden, l2) + "): " + path);
				}
				final int[] all = new int[np];
				int k = 0;
				String line;
				while (k < np && (line = r.readLine()) != null) {
					if (line.startsWith("checksum ")) {
						break; // i valori sono finiti prima del previsto: lo segnala il controllo di k subito sotto
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
				return build(hidden, l2, all, path);
			} catch (IllegalArgumentException e) { // include NumberFormatException e le verifiche del costruttore
				throw new IOException("File pesi NNUE non valido (" + path + "): " + e.getMessage(), e);
			}
		}
	}

	private static NnueNetwork build(int hidden, int l2, int[] all, String path) throws IOException {
		int k = 0;
		final short[] ftW = new short[NUM_FEATURES * hidden];
		for (int i = 0; i < ftW.length; i++) {
			ftW[i] = checkedShort(all[k++], "ftW", i, path);
		}
		final short[] ftB = new short[hidden];
		for (int i = 0; i < hidden; i++) {
			ftB[i] = checkedShort(all[k++], "ftB", i, path);
		}
		final short[][] l1W = new short[ROLES][l2 * hidden];
		final int[][] l1B = new int[ROLES][l2];
		final short[][] l2W = new short[ROLES][l2];
		final int[] l2B = new int[ROLES];
		for (int r = 0; r < ROLES; r++) {
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
		return new NnueNetwork(hidden, l2, ftW, ftB, l1W, l1B, l2W, l2B);
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
		if (!(o instanceof NnueNetwork)) {
			return false;
		}
		final NnueNetwork n = (NnueNetwork) o;
		return hidden == n.hidden && l2 == n.l2 && Arrays.equals(ftW, n.ftW) && Arrays.equals(ftB, n.ftB) && Arrays.deepEquals(l1W, n.l1W)
				&& Arrays.deepEquals(l1B, n.l1B) && Arrays.deepEquals(l2W, n.l2W) && Arrays.equals(l2B, n.l2B);
	}

	@Override
	public int hashCode() {
		return hidden * 31 + l2 + Arrays.hashCode(ftW) * 7 + Arrays.hashCode(ftB);
	}
}
