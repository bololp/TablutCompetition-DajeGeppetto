package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipException;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;

/**
 * Posizioni di gioco con le loro ETICHETTE, per addestrare una NNUE ({@link NnueTrainer}). Una riga per posizione:
 * <ul>
 * <li>{@code boards}: 81 codici di {@link BoardConstants} (0 vuoto, 1 bianco, 2 nero, <b>4 re</b>; NB: {@code DistillSet} e i dati del CNN
 * codificano il Re come 3, gli importatori convertono);</li>
 * <li>{@code role}: lato al tratto, 0 = Bianco, 1 = Nero;</li>
 * <li>{@code outcome} (+1 vince il Bianco, -1 vince il Nero, 0 patta) valido solo se {@code outValid = 1}: e' l'esito VERO della partita di
 * provenienza, NON una partita troncata ne' un forfeit (timeout, mossa nulla: l'esito non dice nulla della posizione);</li>
 * <li>{@code value} in [-1, 1], valido solo se {@code valValid = 1}: etichetta di ricerca (o di un maestro), dal punto di vista del Bianco;</li>
 * <li>{@code game}: chiave di partita (tutte le posizioni di una partita hanno la stessa), {@code ply}: semi-mosse dall'inizio (-1 ignoto),
 * {@code toEnd}: semi-mosse da QUI alla fine della partita (-1 ignoto o senza esito), {@code source}: {@link #SRC_LOG}, {@link #SRC_SELFPLAY},
 * {@link #SRC_IMPORT}.</li>
 * </ul>
 *
 * <h3>Invarianti (imposte dal costruttore e dal caricamento: un dato fuori regola e' un errore, mai uno scarto silenzioso)</h3>
 * <ul>
 * <li>Solo posizioni NON terminali: un solo Re, non sul bordo (altrimenti ha gia' vinto il Bianco), al piu' 8 bianchi e 16 neri, nessuna
 * pedina non-Re sul trono, nessun codice sconosciuto ({@link #checkPosition}).</li>
 * <li>Forma canonica: se un'etichetta non e' valida il suo valore e' 0 (cosi' il file e' deterministico e il confronto esatto).</li>
 * <li>Train e held-out si separano PER PARTITA ({@link #isHeldOut}): posizioni consecutive di una stessa partita sono quasi identiche e
 * non devono stare da parti diverse.</li>
 * </ul>
 *
 * <h3>Memoria: blocco unico o pagine</h3>
 * Un array Java ha al piu' 2^31 - 1 elementi, e la colonna {@code boards} ne ha 81 per campione: un dataset a BLOCCO UNICO (le colonne pubbliche
 * {@code boards}, {@code role}, ... sono un solo array ciascuna) non puo' superare {@link #FLAT_MAX} campioni (circa 26,5 milioni). Oltre, il
 * dataset e' PAGINATO: ogni colonna e' spezzata in pagine di {@code 2^}{@link #PAGE_SHIFT} righe (l'ultima piu' corta). La scelta dipende solo da
 * {@code n}, quindi:
 * <ul>
 * <li>fino a {@link #FLAT_MAX} campioni nulla cambia rispetto a prima: gli array pubblici ci sono e sono gli stessi di sempre;</li>
 * <li>oltre, gli array pubblici sono {@code null} ({@link #isPaged()}): si accede ai dati con {@link #role(int)}, {@link #value(int)},
 * {@link #boardArray(int)} + {@link #boardOffset(int)}, {@link #copyBoard}, ..., che funzionano in entrambi i casi (e vanno preferiti in codice
 * nuovo).</li>
 * </ul>
 * Il costo in RAM e' di circa 98 byte per campione (81 di scacchiera + 17 di colonne); per i dataset grandi usare {@link #loadMerged}, che
 * legge i file direttamente nelle pagine finali senza mai tenere due copie dei dati.
 *
 * <h3>File</h3>
 * Flusso gzip: {@code TABLUT_NNUE_DATA v1}, n, poi per colonne: boards, role, outcome, outValid, valValid, source (byte), game (int), ply,
 * toEnd (short), value (float). Il formato NON e' cambiato (i file vecchi si leggono, quelli nuovi con n piccolo si leggono col codice vecchio).
 * Il caricamento esige di arrivare alla fine del flusso (cosi' il CRC di gzip viene verificato) e che non restino byte in piu'; ogni anomalia
 * e' una {@link IOException}.
 */
public final class NnueDataset {

	public static final String MAGIC = "TABLUT_NNUE_DATA v1";
	public static final byte SRC_LOG = 0, SRC_SELFPLAY = 1, SRC_IMPORT = 2;
	/** Massimo di campioni di un dataset a BLOCCO UNICO: 81 * FLAT_MAX deve stare in un array di byte (indice int). */
	public static final int FLAT_MAX = Integer.MAX_VALUE / 81;
	/** Massimo assoluto di campioni (indici int; il limite pratico e' la RAM: ~98 byte per campione, piu' ~8 byte per gli indici dell'addestramento). */
	public static final int MAX_SAMPLES = 2_000_000_000;
	/** Le pagine di un dataset paginato hanno 2^PAGE_SHIFT righe (la scacchiera di una pagina pesa 81 MiB: abbastanza piccola da non frammentare lo heap). */
	public static final int PAGE_SHIFT = 20;
	public static final int MAX_WHITE = 8, MAX_BLACK = 16;

	/** Shift "infinito": una sola pagina che contiene tutto ({@code i >>> 31 == 0} per ogni indice valido). */
	private static final int FLAT_SHIFT = 31;

	public final int n;
	/**
	 * Le colonne come array unici: SOLO se il dataset non e' paginato ({@code n <= FLAT_MAX}); altrimenti {@code null}. Per codice che deve
	 * funzionare con qualsiasi dimensione usare gli accessori {@link #role(int)}, {@link #boardArray(int)}, ...
	 */
	public final byte[] boards, role, outcome, outValid, valValid, source;
	public final float[] value;
	public final int[] game;
	public final short[] ply, toEnd;

	private final Store s;

	/**
	 * @throws IllegalArgumentException se le dimensioni sono incoerenti o un campione viola le invarianti (il messaggio dice quale e perche')
	 */
	public NnueDataset(byte[] boards, byte[] role, byte[] outcome, byte[] outValid, float[] value, byte[] valValid, int[] game, short[] ply, short[] toEnd,
			byte[] source) {
		this(checked(wrap(boards, role, outcome, outValid, value, valValid, game, ply, toEnd, source)));
	}

	/** Costruttore interno: i dati di {@code s} sono GIA' validati. */
	private NnueDataset(Store s) {
		this.s = s;
		this.n = s.n;
		final boolean flat = s.shift == FLAT_SHIFT;
		this.boards = flat ? s.boards[0] : null;
		this.role = flat ? s.role[0] : null;
		this.outcome = flat ? s.outcome[0] : null;
		this.outValid = flat ? s.outValid[0] : null;
		this.valValid = flat ? s.valValid[0] : null;
		this.source = flat ? s.source[0] : null;
		this.value = flat ? s.value[0] : null;
		this.game = flat ? s.game[0] : null;
		this.ply = flat ? s.ply[0] : null;
		this.toEnd = flat ? s.toEnd[0] : null;
	}

	// ====================================================================== memoria a pagine

	/** Le colonne, ciascuna come array di pagine: pagina {@code c} = righe [c << shift, (c + 1) << shift) (l'ultima piu' corta). */
	private static final class Store {
		final int n, shift, mask, chunks;
		final byte[][] boards, role, outcome, outValid, valValid, source;
		final float[][] value;
		final int[][] game;
		final short[][] ply, toEnd;

		Store(int n, int shift) {
			this.n = n;
			this.shift = shift;
			this.mask = shift >= FLAT_SHIFT ? Integer.MAX_VALUE : (1 << shift) - 1;
			this.chunks = n == 0 ? 1 : (int) (((long) n + mask) >>> shift);
			boards = new byte[chunks][];
			role = new byte[chunks][];
			outcome = new byte[chunks][];
			outValid = new byte[chunks][];
			valValid = new byte[chunks][];
			source = new byte[chunks][];
			value = new float[chunks][];
			game = new int[chunks][];
			ply = new short[chunks][];
			toEnd = new short[chunks][];
		}

		/** Numero di righe della pagina {@code c}. */
		int rows(int c) {
			return (int) Math.min(1L << shift, (long) n - ((long) c << shift));
		}

		/** Alloca (azzerate) tutte le colonne della pagina {@code c}. */
		void allocAll(int c) {
			final int r = rows(c);
			boards[c] = new byte[81 * r];
			role[c] = new byte[r];
			outcome[c] = new byte[r];
			outValid[c] = new byte[r];
			valValid[c] = new byte[r];
			source[c] = new byte[r];
			value[c] = new float[r];
			game[c] = new int[r];
			ply[c] = new short[r];
			toEnd[c] = new short[r];
		}
	}

	/** Pagine di un dataset di {@code n} righe: una sola (blocco unico) se {@code n <= flatMax}. */
	private static int shiftFor(long n, int flatMax, int pageShift) {
		return n <= flatMax ? FLAT_SHIFT : pageShift;
	}

	/** Avvolge colonne intere (blocco unico) controllando le dimensioni. */
	private static Store wrap(byte[] boards, byte[] role, byte[] outcome, byte[] outValid, float[] value, byte[] valValid, int[] game, short[] ply, short[] toEnd,
			byte[] source) {
		if (role == null || role.length > FLAT_MAX) {
			throw new IllegalArgumentException("numero di campioni non valido (massimo " + FLAT_MAX + " per un dataset a blocco unico: per di piu' usare Builder, merge o load)");
		}
		final int n = role.length;
		if (boards == null || boards.length != 81 * n || outcome == null || outcome.length != n || outValid == null || outValid.length != n || value == null
				|| value.length != n || valValid == null || valValid.length != n || game == null || game.length != n || ply == null || ply.length != n
				|| toEnd == null || toEnd.length != n || source == null || source.length != n) {
			throw new IllegalArgumentException("colonne di lunghezza incoerente (n = " + n + ")");
		}
		final Store t = new Store(n, FLAT_SHIFT);
		t.boards[0] = boards;
		t.role[0] = role;
		t.outcome[0] = outcome;
		t.outValid[0] = outValid;
		t.valValid[0] = valValid;
		t.source[0] = source;
		t.value[0] = value;
		t.game[0] = game;
		t.ply[0] = ply;
		t.toEnd[0] = toEnd;
		return t;
	}

	/** Valida tutti i campioni di {@code t} e lo restituisce. */
	private static Store checked(Store t) {
		final String bad = validateRange(t, 0, t.n);
		if (bad != null) {
			throw new IllegalArgumentException(bad);
		}
		return t;
	}

	/**
	 * Controlla le righe [from, from + len) di {@code t} (in parallelo fra pagine).
	 *
	 * @return {@code null} se sono tutte valide, altrimenti "campione i: motivo" per la PRIMA non valida, con {@code i} relativo a {@code from}
	 */
	private static String validateRange(Store t, int from, int len) {
		if (len <= 0) {
			return null;
		}
		final int first = from >>> t.shift, last = (from + len - 1) >>> t.shift;
		final String[] err = new String[last - first + 1];
		IntStream st = IntStream.rangeClosed(first, last);
		if (last > first) {
			st = st.parallel();
		}
		st.forEach(c -> {
			final int jFrom = c == first ? from & t.mask : 0;
			final int jTo = c == last ? ((from + len - 1) & t.mask) + 1 : t.rows(c);
			final byte[] b = t.boards[c], ro = t.role[c], oc = t.outcome[c], ov = t.outValid[c], vv = t.valValid[c], so = t.source[c];
			final float[] va = t.value[c];
			final int[] ga = t.game[c];
			final short[] pl = t.ply[c], te = t.toEnd[c];
			for (int j = jFrom; j < jTo; j++) {
				final String bad = checkSample(b, j, ro[j], oc[j], ov[j], va[j], vv[j], ga[j], pl[j], te[j], so[j]);
				if (bad != null) {
					err[c - first] = "campione " + (((long) c << t.shift) + j - from) + ": " + bad;
					return;
				}
			}
		});
		for (String e : err) {
			if (e != null) {
				return e;
			}
		}
		return null;
	}

	// ====================================================================== accesso

	/** Vero se il dataset e' paginato: gli array pubblici ({@link #boards}, {@link #role}, ...) sono {@code null}. */
	public boolean isPaged() {
		return s.shift != FLAT_SHIFT;
	}

	/** Pagina della scacchiera del campione {@code i}; la scacchiera sta nelle 81 celle da {@link #boardOffset(int)}. */
	public byte[] boardArray(int i) {
		return s.boards[i >>> s.shift];
	}

	/** Posizione, in {@link #boardArray(int)}, della prima cella della scacchiera del campione {@code i}. */
	public int boardOffset(int i) {
		return 81 * (i & s.mask);
	}

	/** Copia le 81 celle della scacchiera del campione {@code i} in {@code dst} a partire da {@code dstOff}. */
	public void copyBoard(int i, byte[] dst, int dstOff) {
		System.arraycopy(s.boards[i >>> s.shift], 81 * (i & s.mask), dst, dstOff, 81);
	}

	public byte role(int i) {
		return s.role[i >>> s.shift][i & s.mask];
	}

	public byte outcome(int i) {
		return s.outcome[i >>> s.shift][i & s.mask];
	}

	public byte outValid(int i) {
		return s.outValid[i >>> s.shift][i & s.mask];
	}

	public byte valValid(int i) {
		return s.valValid[i >>> s.shift][i & s.mask];
	}

	public byte source(int i) {
		return s.source[i >>> s.shift][i & s.mask];
	}

	public float value(int i) {
		return s.value[i >>> s.shift][i & s.mask];
	}

	public int game(int i) {
		return s.game[i >>> s.shift][i & s.mask];
	}

	public short ply(int i) {
		return s.ply[i >>> s.shift][i & s.mask];
	}

	public short toEnd(int i) {
		return s.toEnd[i >>> s.shift][i & s.mask];
	}

	/** Controlla un campione; {@code null} se e' valido, altrimenti il motivo. */
	static String checkSample(byte[] boards, int i, byte role, byte outcome, byte outValid, float value, byte valValid, int game, short ply, short toEnd,
			byte source) {
		final String pos = checkPosition(boards, 81 * i);
		if (pos != null) {
			return pos;
		}
		if (role != 0 && role != 1) {
			return "lato al tratto " + role + " (atteso 0 o 1)";
		}
		if (outValid != 0 && outValid != 1) {
			return "outValid " + outValid + " (atteso 0 o 1)";
		}
		if (valValid != 0 && valValid != 1) {
			return "valValid " + valValid + " (atteso 0 o 1)";
		}
		if (outcome < -1 || outcome > 1) {
			return "esito " + outcome + " (atteso -1, 0 o 1)";
		}
		if (outValid == 0 && outcome != 0) {
			return "esito " + outcome + " con outValid = 0 (la forma canonica vuole 0)";
		}
		if (Float.isNaN(value) || value < -1f || value > 1f) {
			return "valore " + value + " fuori da [-1, 1]";
		}
		if (valValid == 0 && value != 0f) {
			return "valore " + value + " con valValid = 0 (la forma canonica vuole 0)";
		}
		if (game < 0) {
			return "chiave di partita negativa " + game;
		}
		if (ply < -1) {
			return "ply " + ply + " (atteso >= -1)";
		}
		if (toEnd < -1) {
			return "toEnd " + toEnd + " (atteso >= -1)";
		}
		if (toEnd >= 0 && outValid == 0) {
			return "toEnd " + toEnd + " senza un esito valido";
		}
		if (source < 0 || source > 15) {
			return "sorgente " + source + " (attesa 0..15)";
		}
		return null;
	}

	/**
	 * Verifica che 81 codici a partire da {@code off} siano una posizione NON terminale ammessa.
	 *
	 * @return {@code null} se lo e', altrimenti il motivo
	 */
	public static String checkPosition(byte[] b, int off) {
		int kings = 0, white = 0, black = 0, kingSq = -1;
		for (int sq = 0; sq < 81; sq++) {
			final int p = b[off + sq];
			switch (p) {
			case BoardConstants.EMPTY:
				break;
			case BoardConstants.WHITE:
				white++;
				if (sq == BoardConstants.THRONE) {
					return "pedina bianca sul trono";
				}
				break;
			case BoardConstants.BLACK:
				black++;
				if (sq == BoardConstants.THRONE) {
					return "pedina nera sul trono";
				}
				break;
			case BoardConstants.KING:
				kings++;
				kingSq = sq;
				break;
			default:
				return "codice pedina " + p + " nella cella " + sq + " (ammessi 0, 1, 2, 4: il Re e' 4)";
			}
		}
		if (kings == 0) {
			return "Re assente (partita finita)";
		}
		if (kings > 1) {
			return "piu' di un Re";
		}
		if (white > MAX_WHITE || black > MAX_BLACK) {
			return "troppe pedine (" + white + " bianche, " + black + " nere)";
		}
		final int r = kingSq / 9, c = kingSq % 9;
		if (r == 0 || r == 8 || c == 0 || c == 8) {
			return "Re sul bordo (posizione terminale)";
		}
		return null;
	}

	// ====================================================================== costruzione

	/**
	 * Accumulatore di campioni con array primitivi (centinaia di milioni di righe senza boxing). Internamente a pagine: la prima cresce per
	 * raddoppio fino a {@code 2^PAGE_SHIFT} righe, le altre hanno quella dimensione fissa, cosi' non si copia mai tutto e non si supera mai il
	 * limite degli array. {@link #build} restituisce un dataset a blocco unico se i campioni sono al piu' {@link NnueDataset#FLAT_MAX}, altrimenti
	 * paginato (condividendo le pagine piene: nessuna copia).
	 */
	public static final class Builder {
		private final int flatMax, pageShift, pageRows, pageMask;
		private int n, pages, lastCap;
		private byte[][] cBoards = new byte[8][], cRole = new byte[8][], cOutcome = new byte[8][], cOutValid = new byte[8][], cValValid = new byte[8][],
				cSource = new byte[8][];
		private float[][] cValue = new float[8][];
		private int[][] cGame = new int[8][];
		private short[][] cPly = new short[8][], cToEnd = new short[8][];

		public Builder() {
			this(FLAT_MAX, PAGE_SHIFT);
		}

		/** Per i test: soglie piccole per esercitare il caso paginato senza milioni di campioni. */
		Builder(int flatMax, int pageShift) {
			if (flatMax < 0 || pageShift < 1 || pageShift > 30) {
				throw new IllegalArgumentException("soglie non valide");
			}
			this.flatMax = flatMax;
			this.pageShift = pageShift;
			this.pageRows = 1 << pageShift;
			this.pageMask = pageRows - 1;
		}

		public int size() {
			return n;
		}

		private void newPage() {
			if (pages == cBoards.length) {
				final int m = pages * 2;
				cBoards = Arrays.copyOf(cBoards, m);
				cRole = Arrays.copyOf(cRole, m);
				cOutcome = Arrays.copyOf(cOutcome, m);
				cOutValid = Arrays.copyOf(cOutValid, m);
				cValValid = Arrays.copyOf(cValValid, m);
				cSource = Arrays.copyOf(cSource, m);
				cValue = Arrays.copyOf(cValue, m);
				cGame = Arrays.copyOf(cGame, m);
				cPly = Arrays.copyOf(cPly, m);
				cToEnd = Arrays.copyOf(cToEnd, m);
			}
			final int cap = pages == 0 ? Math.min(256, pageRows) : pageRows;
			cBoards[pages] = new byte[81 * cap];
			cRole[pages] = new byte[cap];
			cOutcome[pages] = new byte[cap];
			cOutValid[pages] = new byte[cap];
			cValValid[pages] = new byte[cap];
			cSource[pages] = new byte[cap];
			cValue[pages] = new float[cap];
			cGame[pages] = new int[cap];
			cPly[pages] = new short[cap];
			cToEnd[pages] = new short[cap];
			lastCap = cap;
			pages++;
		}

		/** Raddoppia la capacita' della prima pagina (le altre nascono gia' piene di capacita'). */
		private void growLast() {
			final int p = pages - 1, m = Math.min(lastCap * 2, pageRows);
			cBoards[p] = Arrays.copyOf(cBoards[p], 81 * m);
			cRole[p] = Arrays.copyOf(cRole[p], m);
			cOutcome[p] = Arrays.copyOf(cOutcome[p], m);
			cOutValid[p] = Arrays.copyOf(cOutValid[p], m);
			cValValid[p] = Arrays.copyOf(cValValid[p], m);
			cSource[p] = Arrays.copyOf(cSource[p], m);
			cValue[p] = Arrays.copyOf(cValue[p], m);
			cGame[p] = Arrays.copyOf(cGame[p], m);
			cPly[p] = Arrays.copyOf(cPly[p], m);
			cToEnd[p] = Arrays.copyOf(cToEnd[p], m);
			lastCap = m;
		}

		/** Aggiunge un campione (senza controlli: li fa {@link #build}). */
		public void add(byte[] board81, int boardOff, int role, int outcome, boolean outValid, double value, boolean valValid, int game, int ply, int toEnd,
				int source) {
			if (n >= MAX_SAMPLES) {
				throw new IllegalStateException("troppi campioni (massimo " + MAX_SAMPLES + ")");
			}
			final int page = n >>> pageShift, j = n & pageMask;
			if (page == pages) {
				newPage();
			} else if (j == lastCap) {
				growLast();
			}
			System.arraycopy(board81, boardOff, cBoards[page], 81 * j, 81);
			cRole[page][j] = (byte) role;
			cOutValid[page][j] = (byte) (outValid ? 1 : 0);
			cOutcome[page][j] = (byte) (outValid ? outcome : 0);
			cValValid[page][j] = (byte) (valValid ? 1 : 0);
			cValue[page][j] = valValid ? (float) value : 0f;
			cGame[page][j] = game;
			cPly[page][j] = (short) Math.max(-1, Math.min(Short.MAX_VALUE, ply));
			cToEnd[page][j] = (short) (outValid ? Math.max(-1, Math.min(Short.MAX_VALUE, toEnd)) : -1);
			cSource[page][j] = (byte) source;
			n++;
		}

		/** @throws IllegalArgumentException se un campione viola le invarianti */
		public NnueDataset build() {
			if (n <= flatMax) { // blocco unico: array esatti, come sempre
				final Store t = new Store(n, FLAT_SHIFT);
				t.allocAll(0);
				for (int p = 0; p < pages; p++) {
					final int from = p << pageShift, len = Math.min(pageRows, n - from);
					System.arraycopy(cBoards[p], 0, t.boards[0], 81 * from, 81 * len);
					System.arraycopy(cRole[p], 0, t.role[0], from, len);
					System.arraycopy(cOutcome[p], 0, t.outcome[0], from, len);
					System.arraycopy(cOutValid[p], 0, t.outValid[0], from, len);
					System.arraycopy(cValValid[p], 0, t.valValid[0], from, len);
					System.arraycopy(cSource[p], 0, t.source[0], from, len);
					System.arraycopy(cValue[p], 0, t.value[0], from, len);
					System.arraycopy(cGame[p], 0, t.game[0], from, len);
					System.arraycopy(cPly[p], 0, t.ply[0], from, len);
					System.arraycopy(cToEnd[p], 0, t.toEnd[0], from, len);
				}
				return new NnueDataset(checked(t));
			}
			// paginato: le pagine piene sono condivise (il builder non le riscrive mai), solo l'ultima si copia accorciata
			final Store t = new Store(n, pageShift);
			for (int p = 0; p < pages - 1; p++) {
				t.boards[p] = cBoards[p];
				t.role[p] = cRole[p];
				t.outcome[p] = cOutcome[p];
				t.outValid[p] = cOutValid[p];
				t.valValid[p] = cValValid[p];
				t.source[p] = cSource[p];
				t.value[p] = cValue[p];
				t.game[p] = cGame[p];
				t.ply[p] = cPly[p];
				t.toEnd[p] = cToEnd[p];
			}
			final int p = pages - 1, rows = n - (p << pageShift);
			t.boards[p] = Arrays.copyOf(cBoards[p], 81 * rows);
			t.role[p] = Arrays.copyOf(cRole[p], rows);
			t.outcome[p] = Arrays.copyOf(cOutcome[p], rows);
			t.outValid[p] = Arrays.copyOf(cOutValid[p], rows);
			t.valValid[p] = Arrays.copyOf(cValValid[p], rows);
			t.source[p] = Arrays.copyOf(cSource[p], rows);
			t.value[p] = Arrays.copyOf(cValue[p], rows);
			t.game[p] = Arrays.copyOf(cGame[p], rows);
			t.ply[p] = Arrays.copyOf(cPly[p], rows);
			t.toEnd[p] = Arrays.copyOf(cToEnd[p], rows);
			return new NnueDataset(checked(t));
		}
	}

	/** Il sottoinsieme dei campioni {@code idx} (nell'ordine dato). */
	public NnueDataset subset(int[] idx) {
		return subset(idx, FLAT_MAX, PAGE_SHIFT);
	}

	NnueDataset subset(int[] idx, int flatMax, int pageShift) {
		final int m = idx.length;
		final Store t = new Store(m, shiftFor(m, flatMax, pageShift));
		for (int c = 0; c < t.chunks; c++) {
			t.allocAll(c);
			final int rows = t.rows(c), base = c << (t.shift == FLAT_SHIFT ? 0 : t.shift);
			final byte[] b = t.boards[c], ro = t.role[c], oc = t.outcome[c], ov = t.outValid[c], vv = t.valValid[c], so = t.source[c];
			final float[] va = t.value[c];
			final int[] ga = t.game[c];
			final short[] pl = t.ply[c], te = t.toEnd[c];
			for (int j = 0; j < rows; j++) {
				final int i = idx[base + j];
				copyBoard(i, b, 81 * j);
				ro[j] = role(i);
				oc[j] = outcome(i);
				ov[j] = outValid(i);
				vv[j] = valValid(i);
				so[j] = source(i);
				va[j] = value(i);
				ga[j] = game(i);
				pl[j] = ply(i);
				te[j] = toEnd(i);
			}
		}
		return new NnueDataset(t); // i campioni vengono da un dataset gia' validato
	}

	/**
	 * Concatena dataset. Le chiavi di partita sono RINUMERATE (densamente, per ordine di comparsa, distinguendo il dataset di provenienza):
	 * due file che usano gli stessi numeri non fondono partite diverse.
	 * <p>
	 * Attenzione: i dati sono COPIATI, quindi durante l'unione servono sia le parti sia il risultato. Per file grandi usare {@link #loadMerged}.
	 */
	public static NnueDataset merge(List<NnueDataset> parts) {
		return merge(parts, FLAT_MAX, PAGE_SHIFT);
	}

	static NnueDataset merge(List<NnueDataset> parts, int flatMax, int pageShift) {
		long total = 0;
		for (NnueDataset d : parts) {
			total += d.n;
		}
		if (total > MAX_SAMPLES) {
			throw new IllegalArgumentException("troppi campioni nell'unione: " + total + " (massimo " + MAX_SAMPLES + ")");
		}
		final Store t = new Store((int) total, shiftFor(total, flatMax, pageShift));
		final LongIntMap ids = new LongIntMap();
		int k = 0;
		for (int part = 0; part < parts.size(); part++) {
			final NnueDataset d = parts.get(part);
			final Store q = d.s;
			for (int pos = 0; pos < d.n;) {
				final int sc = pos >>> q.shift, sj = pos & q.mask, dk = k + pos, dc = dk >>> t.shift, dj = dk & t.mask;
				final int len = Math.min(d.n - pos, Math.min(q.rows(sc) - sj, t.rows(dc) - dj));
				if (t.role[dc] == null) {
					t.allocAll(dc);
				}
				System.arraycopy(q.boards[sc], 81 * sj, t.boards[dc], 81 * dj, 81 * len);
				System.arraycopy(q.role[sc], sj, t.role[dc], dj, len);
				System.arraycopy(q.outcome[sc], sj, t.outcome[dc], dj, len);
				System.arraycopy(q.outValid[sc], sj, t.outValid[dc], dj, len);
				System.arraycopy(q.valValid[sc], sj, t.valValid[dc], dj, len);
				System.arraycopy(q.source[sc], sj, t.source[dc], dj, len);
				System.arraycopy(q.value[sc], sj, t.value[dc], dj, len);
				System.arraycopy(q.ply[sc], sj, t.ply[dc], dj, len);
				System.arraycopy(q.toEnd[sc], sj, t.toEnd[dc], dj, len);
				final int[] src = q.game[sc], dst = t.game[dc];
				for (int i = 0; i < len; i++) {
					dst[dj + i] = ids.idFor(((long) part << 32) | (src[sj + i] & 0xFFFFFFFFL));
				}
				pos += len;
			}
			k += d.n;
		}
		if (t.n == 0) {
			t.allocAll(0); // unione di nessuno / di dataset vuoti: colonne vuote, mai null
		}
		return new NnueDataset(checked(t));
	}

	/**
	 * Copia di questo dataset con {@code value} e {@code valValid} SOSTITUITI (es. rietichettatura). Le altre colonne sono CONDIVISE, non copiate
	 * (gli array di un dataset non si modificano mai dopo la costruzione): costa solo le due colonne nuove. Se il dataset non e' paginato i due array
	 * passati diventano le colonne della copia (come con il costruttore): il chiamante non deve piu' modificarli. I dati sono validati.
	 *
	 * @throws IllegalArgumentException se le lunghezze non sono {@code n} o un campione viola le invarianti
	 */
	public NnueDataset withValues(float[] newValue, byte[] newValValid) {
		if (newValue == null || newValValid == null || newValue.length != n || newValValid.length != n) {
			throw new IllegalArgumentException("colonne di lunghezza incoerente (n = " + n + ")");
		}
		final Store t = new Store(n, s.shift);
		for (int c = 0; c < s.chunks; c++) {
			t.boards[c] = s.boards[c];
			t.role[c] = s.role[c];
			t.outcome[c] = s.outcome[c];
			t.outValid[c] = s.outValid[c];
			t.source[c] = s.source[c];
			t.game[c] = s.game[c];
			t.ply[c] = s.ply[c];
			t.toEnd[c] = s.toEnd[c];
			if (s.shift == FLAT_SHIFT) {
				t.value[c] = newValue;
				t.valValid[c] = newValValid;
			} else {
				final int from = c << s.shift, to = from + s.rows(c);
				t.value[c] = Arrays.copyOfRange(newValue, from, to);
				t.valValid[c] = Arrays.copyOfRange(newValValid, from, to);
			}
		}
		return new NnueDataset(checked(t));
	}

	// ====================================================================== split per partita

	/** Vero se la partita {@code game} appartiene al held-out ({@code pct} per cento delle partite), in modo deterministico da {@code (game, salt)}. */
	public static boolean isHeldOut(int game, long salt, int pct) {
		long h = (game & 0xFFFFFFFFL) * 0x9E3779B97F4A7C15L + salt * 0xBF58476D1CE4E5B9L;
		h = (h ^ (h >>> 30)) * 0xBF58476D1CE4E5B9L;
		h = (h ^ (h >>> 27)) * 0x94D049BB133111EBL;
		h ^= h >>> 31;
		return Math.floorMod(h, 100L) < pct;
	}

	/** Indici dei campioni di train (parte 0) e di held-out (parte 1): tutte le posizioni di una partita stanno dalla stessa parte. */
	public int[][] splitByGame(long salt, int pct) {
		int h = 0;
		for (int i = 0; i < n; i++) { // prima si conta, per allocare array esatti (con 10^9 indici una copia in piu' pesa gigabyte)
			if (isHeldOut(game(i), salt, pct)) {
				h++;
			}
		}
		final int[] tr = new int[n - h], he = new int[h];
		int a = 0, b = 0;
		for (int i = 0; i < n; i++) {
			if (isHeldOut(game(i), salt, pct)) {
				he[b++] = i;
			} else {
				tr[a++] = i;
			}
		}
		return new int[][] { tr, he };
	}

	// ====================================================================== riepilogo

	public int numGames() {
		final BitSet seen = new BitSet(); // le chiavi sono int >= 0: al piu' 256 MiB, e niente boxing
		for (int c = 0; c < s.chunks; c++) {
			for (int g : s.game[c]) {
				seen.set(g);
			}
		}
		return seen.cardinality();
	}

	/** Composizione, per i log. */
	public String describe() {
		long w = 0, b = 0, d = 0, none = 0, val = 0, wt = 0;
		final long[] src = new long[16];
		for (int i = 0; i < n; i++) {
			if (outValid(i) == 0) {
				none++;
			} else if (outcome(i) > 0) {
				w++;
			} else if (outcome(i) < 0) {
				b++;
			} else {
				d++;
			}
			val += valValid(i);
			wt += role(i) == 0 ? 1 : 0;
			src[source(i)]++;
		}
		final StringBuilder sb = new StringBuilder();
		sb.append(n).append(" posizioni in ").append(numGames()).append(" partite; muove il Bianco in ").append(wt).append(", il Nero in ").append(n - wt)
				.append("; esito (vista Bianco): +1 ").append(w).append(", -1 ").append(b).append(", 0 ").append(d).append(", senza esito ").append(none)
				.append("; con valore di ricerca ").append(val).append("; sorgenti:");
		for (int k = 0; k < 16; k++) {
			if (src[k] > 0) {
				sb.append(' ').append(k).append('=').append(src[k]);
			}
		}
		return sb.toString();
	}

	// ====================================================================== file

	public void save(String path) throws IOException {
		try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path)), 1 << 16))) {
			o.writeUTF(MAGIC);
			o.writeInt(n);
			for (byte[] a : s.boards) {
				o.write(a);
			}
			for (byte[] a : s.role) {
				o.write(a);
			}
			for (byte[] a : s.outcome) {
				o.write(a);
			}
			for (byte[] a : s.outValid) {
				o.write(a);
			}
			for (byte[] a : s.valValid) {
				o.write(a);
			}
			for (byte[] a : s.source) {
				o.write(a);
			}
			final byte[] buf = new byte[1 << 16];
			for (int[] a : s.game) {
				for (int off = 0; off < a.length;) {
					final int m = Math.min(a.length - off, buf.length / 4);
					ByteBuffer.wrap(buf, 0, 4 * m).asIntBuffer().put(a, off, m);
					o.write(buf, 0, 4 * m);
					off += m;
				}
			}
			for (short[] a : s.ply) {
				writeShorts(o, a, buf);
			}
			for (short[] a : s.toEnd) {
				writeShorts(o, a, buf);
			}
			for (float[] a : s.value) {
				for (int off = 0; off < a.length;) {
					final int m = Math.min(a.length - off, buf.length / 4);
					ByteBuffer.wrap(buf, 0, 4 * m).asFloatBuffer().put(a, off, m);
					o.write(buf, 0, 4 * m);
					off += m;
				}
			}
		}
	}

	private static void writeShorts(DataOutputStream o, short[] a, byte[] buf) throws IOException {
		for (int off = 0; off < a.length;) {
			final int m = Math.min(a.length - off, buf.length / 2);
			ByteBuffer.wrap(buf, 0, 2 * m).asShortBuffer().put(a, off, m);
			o.write(buf, 0, 2 * m);
			off += m;
		}
	}

	/**
	 * Carica un file. Le chiavi di partita restano quelle del file. Con piu' di {@link #FLAT_MAX} campioni il dataset e' paginato.
	 *
	 * @throws IOException file assente/illeggibile, non gzip, di un altro formato o versione, troncato, con byte in piu', corrotto (CRC di
	 *                     gzip) o con campioni che violano le invarianti
	 */
	public static NnueDataset load(String path) throws IOException {
		return loadFiles(Collections.singletonList(path), false, FLAT_MAX, PAGE_SHIFT);
	}

	/**
	 * Carica e UNISCE piu' file in un colpo solo, equivalente a {@code merge(load(p1), load(p2), ...)} (chiavi di partita rinumerate, dataset
	 * identico) ma senza mai tenere due copie dei dati: si leggono le intestazioni per sapere il totale, poi ogni file e' letto direttamente nelle
	 * pagine del risultato. E' il modo di caricare dataset enormi.
	 *
	 * @throws IllegalArgumentException se il totale supera {@link #MAX_SAMPLES}
	 * @throws IOException              come {@link #load}
	 */
	public static NnueDataset loadMerged(List<String> paths) throws IOException {
		return loadFiles(paths, true, FLAT_MAX, PAGE_SHIFT);
	}

	static NnueDataset loadFiles(List<String> paths, boolean renumber, int flatMax, int pageShift) throws IOException {
		final int[] ns = new int[paths.size()];
		long total = 0;
		for (int k = 0; k < ns.length; k++) {
			ns[k] = readCount(paths.get(k));
			total += ns[k];
		}
		if (total > MAX_SAMPLES) {
			throw new IllegalArgumentException("troppi campioni nell'unione: " + total + " (massimo " + MAX_SAMPLES + ")");
		}
		final Store t = new Store((int) total, shiftFor(total, flatMax, pageShift));
		final LongIntMap ids = renumber ? new LongIntMap() : null;
		int off = 0;
		for (int k = 0; k < ns.length; k++) {
			readInto(paths.get(k), t, off, ns[k], k, ids);
			off += ns[k];
		}
		if (t.n == 0) {
			t.allocAll(0); // dataset vuoto: colonne vuote, mai null
		}
		return new NnueDataset(t);
	}

	private static DataInputStream open(String path) throws IOException {
		return new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16));
	}

	private static IOException mapIo(IOException e, String path) {
		if (e instanceof EOFException) {
			return new IOException("file dati troncato: " + path, e);
		}
		if (e instanceof ZipException) {
			return new IOException("file dati non valido o corrotto (" + e.getMessage() + "): " + path, e);
		}
		return e;
	}

	/** Legge e controlla l'intestazione; restituisce il numero di campioni. */
	private static int readHeader(DataInputStream in, String path) throws IOException {
		final String magic;
		try {
			magic = in.readUTF();
		} catch (java.io.UTFDataFormatException e) {
			throw new IOException("non e' un dataset NNUE (" + MAGIC + "): " + path, e);
		}
		if (!magic.startsWith("TABLUT_NNUE_DATA")) {
			throw new IOException("non e' un dataset NNUE (" + MAGIC + "): " + path);
		}
		if (!magic.equals(MAGIC)) {
			throw new IOException("versione del formato dati non supportata ('" + magic + "', questo codice legge '" + MAGIC + "'): " + path);
		}
		final int n = in.readInt();
		if (n < 0 || n > MAX_SAMPLES) {
			throw new IOException("numero di campioni non plausibile: " + n + " (" + path + ")");
		}
		return n;
	}

	private static int readCount(String path) throws IOException {
		try (DataInputStream in = open(path)) {
			return readHeader(in, path);
		} catch (IOException e) {
			throw mapIo(e, path);
		}
	}

	/** Legge il file {@code path} (di {@code cnt} campioni) nelle righe [off, off + cnt) di {@code t}. */
	private static void readInto(String path, Store t, int off, int cnt, int part, LongIntMap ids) throws IOException {
		try (DataInputStream in = open(path)) {
			if (readHeader(in, path) != cnt) {
				throw new IOException("il file e' cambiato durante la lettura: " + path);
			}
			readBytes(in, t.boards, 81, t, off, cnt);
			readBytes(in, t.role, 1, t, off, cnt);
			readBytes(in, t.outcome, 1, t, off, cnt);
			readBytes(in, t.outValid, 1, t, off, cnt);
			readBytes(in, t.valValid, 1, t, off, cnt);
			readBytes(in, t.source, 1, t, off, cnt);
			final byte[] buf = new byte[1 << 16];
			for (int pos = 0; pos < cnt;) {
				final int k = off + pos, c = k >>> t.shift, j = k & t.mask, len = Math.min(cnt - pos, t.rows(c) - j);
				final int[] a = t.game[c] != null ? t.game[c] : (t.game[c] = new int[t.rows(c)]);
				for (int d = 0; d < len;) {
					final int m = Math.min(len - d, buf.length / 4);
					in.readFully(buf, 0, 4 * m);
					ByteBuffer.wrap(buf, 0, 4 * m).asIntBuffer().get(a, j + d, m);
					d += m;
				}
				pos += len;
			}
			readShorts(in, t.ply, t, off, cnt, buf);
			readShorts(in, t.toEnd, t, off, cnt, buf);
			for (int pos = 0; pos < cnt;) {
				final int k = off + pos, c = k >>> t.shift, j = k & t.mask, len = Math.min(cnt - pos, t.rows(c) - j);
				final float[] a = t.value[c] != null ? t.value[c] : (t.value[c] = new float[t.rows(c)]);
				for (int d = 0; d < len;) {
					final int m = Math.min(len - d, buf.length / 4);
					in.readFully(buf, 0, 4 * m);
					ByteBuffer.wrap(buf, 0, 4 * m).asFloatBuffer().get(a, j + d, m);
					d += m;
				}
				pos += len;
			}
			if (in.read() != -1) { // leggere fino alla fine fa verificare il CRC di gzip
				throw new IOException("byte in piu' dopo i dati (" + path + ")");
			}
			final String bad = validateRange(t, off, cnt);
			if (bad != null) {
				throw new IOException("dati non validi in " + path + ": " + bad);
			}
			if (ids != null) { // rinumerazione DOPO la validazione (le chiavi negative sono un errore, non un indice)
				for (int pos = 0; pos < cnt;) {
					final int k = off + pos, c = k >>> t.shift, j = k & t.mask, len = Math.min(cnt - pos, t.rows(c) - j);
					final int[] a = t.game[c];
					for (int i = 0; i < len; i++) {
						a[j + i] = ids.idFor(((long) part << 32) | (a[j + i] & 0xFFFFFFFFL));
					}
					pos += len;
				}
			}
		} catch (IOException e) {
			throw mapIo(e, path);
		}
	}

	/** Legge una colonna di byte ({@code per} byte per riga) nelle righe [off, off + cnt), allocando ogni pagina solo quando i dati arrivano. */
	private static void readBytes(DataInputStream in, byte[][] col, int per, Store t, int off, int cnt) throws IOException {
		for (int pos = 0; pos < cnt;) {
			final int k = off + pos, c = k >>> t.shift, j = k & t.mask, len = Math.min(cnt - pos, t.rows(c) - j);
			final byte[] a = col[c] != null ? col[c] : (col[c] = new byte[per * t.rows(c)]);
			in.readFully(a, per * j, per * len);
			pos += len;
		}
	}

	private static void readShorts(DataInputStream in, short[][] col, Store t, int off, int cnt, byte[] buf) throws IOException {
		for (int pos = 0; pos < cnt;) {
			final int k = off + pos, c = k >>> t.shift, j = k & t.mask, len = Math.min(cnt - pos, t.rows(c) - j);
			final short[] a = col[c] != null ? col[c] : (col[c] = new short[t.rows(c)]);
			for (int d = 0; d < len;) {
				final int m = Math.min(len - d, buf.length / 2);
				in.readFully(buf, 0, 2 * m);
				ByteBuffer.wrap(buf, 0, 2 * m).asShortBuffer().get(a, j + d, m);
				d += m;
			}
			pos += len;
		}
	}

	// ====================================================================== mappa long -> int

	/** Mappa a indirizzamento aperto da chiave long a id denso (0, 1, 2, ... per ordine di comparsa): niente boxing, ~12 byte per chiave. */
	private static final class LongIntMap {
		private long[] keys = new long[1 << 12];
		private int[] vals = new int[1 << 12];
		private int size;

		LongIntMap() {
			Arrays.fill(vals, -1);
		}

		private static int slot(long key, int mask) {
			final long h = key * 0x9E3779B97F4A7C15L;
			return (int) (h >>> 32) & mask;
		}

		int idFor(long key) {
			int mask = keys.length - 1;
			int h = slot(key, mask);
			while (vals[h] >= 0) {
				if (keys[h] == key) {
					return vals[h];
				}
				h = (h + 1) & mask;
			}
			if ((size + 1) * 2L > keys.length) { // fattore di carico massimo 1/2
				grow();
				mask = keys.length - 1;
				h = slot(key, mask);
				while (vals[h] >= 0) {
					h = (h + 1) & mask;
				}
			}
			keys[h] = key;
			vals[h] = size;
			return size++;
		}

		private void grow() {
			if (keys.length >= (1 << 30)) {
				throw new IllegalStateException("troppe partite distinte");
			}
			final long[] ok = keys;
			final int[] ov = vals;
			keys = new long[ok.length * 2];
			vals = new int[ok.length * 2];
			Arrays.fill(vals, -1);
			final int mask = keys.length - 1;
			for (int i = 0; i < ok.length; i++) {
				if (ov[i] >= 0) {
					int h = slot(ok[i], mask);
					while (vals[h] >= 0) {
						h = (h + 1) & mask;
					}
					keys[h] = ok[i];
					vals[h] = ov[i];
				}
			}
		}
	}

	// ====================================================================== uguaglianza

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof NnueDataset)) {
			return false;
		}
		final NnueDataset d = (NnueDataset) o;
		if (n != d.n) {
			return false;
		}
		if (s.shift == d.s.shift) { // stessa disposizione (dipende solo da n): confronto pagina per pagina
			for (int c = 0; c < s.chunks; c++) {
				if (!Arrays.equals(s.boards[c], d.s.boards[c]) || !Arrays.equals(s.role[c], d.s.role[c]) || !Arrays.equals(s.outcome[c], d.s.outcome[c])
						|| !Arrays.equals(s.outValid[c], d.s.outValid[c]) || !Arrays.equals(s.valValid[c], d.s.valValid[c])
						|| !Arrays.equals(s.source[c], d.s.source[c]) || !Arrays.equals(s.game[c], d.s.game[c]) || !Arrays.equals(s.ply[c], d.s.ply[c])
						|| !Arrays.equals(s.toEnd[c], d.s.toEnd[c]) || !Arrays.equals(s.value[c], d.s.value[c])) {
					return false;
				}
			}
			return true;
		}
		for (int i = 0; i < n; i++) { // disposizioni diverse (solo con soglie di test): campione per campione
			if (role(i) != d.role(i) || outcome(i) != d.outcome(i) || outValid(i) != d.outValid(i) || valValid(i) != d.valValid(i) || source(i) != d.source(i)
					|| game(i) != d.game(i) || ply(i) != d.ply(i) || toEnd(i) != d.toEnd(i) || Float.floatToIntBits(value(i)) != Float.floatToIntBits(d.value(i))) {
				return false;
			}
			final byte[] a = boardArray(i), b = d.boardArray(i);
			final int oa = boardOffset(i), ob = d.boardOffset(i);
			for (int k = 0; k < 81; k++) {
				if (a[oa + k] != b[ob + k]) {
					return false;
				}
			}
		}
		return true;
	}

	/** Stesso valore di prima per i dataset a blocco unico ({@code n * 31 + Arrays.hashCode(game) + Arrays.hashCode(value)}); definito allo stesso modo sulle pagine. */
	@Override
	public int hashCode() {
		int hg = 1, hv = 1;
		for (int c = 0; c < s.chunks; c++) {
			for (int g : s.game[c]) {
				hg = 31 * hg + g;
			}
			for (float v : s.value[c]) {
				hv = 31 * hv + Float.floatToIntBits(v);
			}
		}
		return n * 31 + hg + hv;
	}
}
