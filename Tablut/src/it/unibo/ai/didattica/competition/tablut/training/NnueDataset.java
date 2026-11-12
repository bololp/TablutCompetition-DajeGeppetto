package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

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
 * <h3>File</h3>
 * Flusso gzip: {@code TABLUT_NNUE_DATA v1}, n, poi per colonne: boards, role, outcome, outValid, valValid, source (byte), game (int), ply,
 * toEnd (short), value (float). Il caricamento esige di arrivare alla fine del flusso (cosi' il CRC di gzip viene verificato) e che non
 * restino byte in piu'; ogni anomalia e' una {@link IOException}.
 */
public final class NnueDataset {

	public static final String MAGIC = "TABLUT_NNUE_DATA v1";
	public static final byte SRC_LOG = 0, SRC_SELFPLAY = 1, SRC_IMPORT = 2;
	/** 81 * MAX_SAMPLES deve stare in un array di byte. */
	public static final int MAX_SAMPLES = 20_000_000;
	public static final int MAX_WHITE = 8, MAX_BLACK = 16;

	public final int n;
	public final byte[] boards, role, outcome, outValid, valValid, source;
	public final float[] value;
	public final int[] game;
	public final short[] ply, toEnd;

	/**
	 * @throws IllegalArgumentException se le dimensioni sono incoerenti o un campione viola le invarianti (il messaggio dice quale e perche')
	 */
	public NnueDataset(byte[] boards, byte[] role, byte[] outcome, byte[] outValid, float[] value, byte[] valValid, int[] game, short[] ply, short[] toEnd,
			byte[] source) {
		if (role == null || role.length > MAX_SAMPLES) {
			throw new IllegalArgumentException("numero di campioni non valido");
		}
		final int n = role.length;
		if (boards == null || boards.length != 81 * n || outcome == null || outcome.length != n || outValid == null || outValid.length != n
				|| value == null || value.length != n || valValid == null || valValid.length != n || game == null || game.length != n || ply == null
				|| ply.length != n || toEnd == null || toEnd.length != n || source == null || source.length != n) {
			throw new IllegalArgumentException("colonne di lunghezza incoerente (n = " + n + ")");
		}
		for (int i = 0; i < n; i++) {
			final String bad = checkSample(boards, i, role[i], outcome[i], outValid[i], value[i], valValid[i], game[i], ply[i], toEnd[i], source[i]);
			if (bad != null) {
				throw new IllegalArgumentException("campione " + i + ": " + bad);
			}
		}
		this.n = n;
		this.boards = boards;
		this.role = role;
		this.outcome = outcome;
		this.outValid = outValid;
		this.value = value;
		this.valValid = valValid;
		this.game = game;
		this.ply = ply;
		this.toEnd = toEnd;
		this.source = source;
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

	/** Accumulatore di campioni con array primitivi (centinaia di migliaia di righe senza boxing). */
	public static final class Builder {
		private int n;
		private byte[] boards = new byte[81 * 256], role = new byte[256], outcome = new byte[256], outValid = new byte[256], valValid = new byte[256],
				source = new byte[256];
		private float[] value = new float[256];
		private int[] game = new int[256];
		private short[] ply = new short[256], toEnd = new short[256];

		public int size() {
			return n;
		}

		/** Aggiunge un campione (senza controlli: li fa il costruttore di {@link NnueDataset} in {@link #build}). */
		public void add(byte[] board81, int boardOff, int role, int outcome, boolean outValid, double value, boolean valValid, int game, int ply, int toEnd,
				int source) {
			if (n == this.role.length) {
				final int m = n * 2;
				boards = Arrays.copyOf(boards, 81 * m);
				this.role = Arrays.copyOf(this.role, m);
				this.outcome = Arrays.copyOf(this.outcome, m);
				this.outValid = Arrays.copyOf(this.outValid, m);
				this.valValid = Arrays.copyOf(this.valValid, m);
				this.source = Arrays.copyOf(this.source, m);
				this.value = Arrays.copyOf(this.value, m);
				this.game = Arrays.copyOf(this.game, m);
				this.ply = Arrays.copyOf(this.ply, m);
				this.toEnd = Arrays.copyOf(this.toEnd, m);
			}
			System.arraycopy(board81, boardOff, boards, 81 * n, 81);
			this.role[n] = (byte) role;
			this.outValid[n] = (byte) (outValid ? 1 : 0);
			this.outcome[n] = (byte) (outValid ? outcome : 0);
			this.valValid[n] = (byte) (valValid ? 1 : 0);
			this.value[n] = valValid ? (float) value : 0f;
			this.game[n] = game;
			this.ply[n] = (short) Math.max(-1, Math.min(Short.MAX_VALUE, ply));
			this.toEnd[n] = (short) (outValid ? Math.max(-1, Math.min(Short.MAX_VALUE, toEnd)) : -1);
			this.source[n] = (byte) source;
			n++;
		}

		public NnueDataset build() {
			return new NnueDataset(Arrays.copyOf(boards, 81 * n), Arrays.copyOf(role, n), Arrays.copyOf(outcome, n), Arrays.copyOf(outValid, n),
					Arrays.copyOf(value, n), Arrays.copyOf(valValid, n), Arrays.copyOf(game, n), Arrays.copyOf(ply, n), Arrays.copyOf(toEnd, n),
					Arrays.copyOf(source, n));
		}
	}

	/** Il sottoinsieme dei campioni {@code idx} (nell'ordine dato). */
	public NnueDataset subset(int[] idx) {
		final int m = idx.length;
		final byte[] b = new byte[81 * m];
		final byte[] r = new byte[m], oc = new byte[m], ov = new byte[m], vv = new byte[m], so = new byte[m];
		final float[] v = new float[m];
		final int[] g = new int[m];
		final short[] p = new short[m], te = new short[m];
		for (int k = 0; k < m; k++) {
			final int i = idx[k];
			System.arraycopy(boards, 81 * i, b, 81 * k, 81);
			r[k] = role[i];
			oc[k] = outcome[i];
			ov[k] = outValid[i];
			vv[k] = valValid[i];
			so[k] = source[i];
			v[k] = value[i];
			g[k] = game[i];
			p[k] = ply[i];
			te[k] = toEnd[i];
		}
		return new NnueDataset(b, r, oc, ov, v, vv, g, p, te, so);
	}

	/**
	 * Concatena dataset. Le chiavi di partita sono RINUMERATE (densamente, per ordine di comparsa, distinguendo il dataset di provenienza):
	 * due file che usano gli stessi numeri non fondono partite diverse.
	 */
	public static NnueDataset merge(List<NnueDataset> parts) {
		int total = 0;
		for (NnueDataset d : parts) {
			total += d.n;
		}
		if (total > MAX_SAMPLES) {
			throw new IllegalArgumentException("troppi campioni nell'unione: " + total + " (massimo " + MAX_SAMPLES + ")");
		}
		final byte[] b = new byte[81 * total];
		final byte[] r = new byte[total], oc = new byte[total], ov = new byte[total], vv = new byte[total], so = new byte[total];
		final float[] v = new float[total];
		final int[] g = new int[total];
		final short[] p = new short[total], te = new short[total];
		final Map<Long, Integer> ids = new HashMap<>();
		int k = 0;
		for (int part = 0; part < parts.size(); part++) {
			final NnueDataset d = parts.get(part);
			System.arraycopy(d.boards, 0, b, 81 * k, 81 * d.n);
			System.arraycopy(d.role, 0, r, k, d.n);
			System.arraycopy(d.outcome, 0, oc, k, d.n);
			System.arraycopy(d.outValid, 0, ov, k, d.n);
			System.arraycopy(d.valValid, 0, vv, k, d.n);
			System.arraycopy(d.source, 0, so, k, d.n);
			System.arraycopy(d.value, 0, v, k, d.n);
			System.arraycopy(d.ply, 0, p, k, d.n);
			System.arraycopy(d.toEnd, 0, te, k, d.n);
			for (int i = 0; i < d.n; i++) {
				final long key = ((long) part << 32) | (d.game[i] & 0xFFFFFFFFL);
				Integer id = ids.get(key);
				if (id == null) {
					id = ids.size();
					ids.put(key, id);
				}
				g[k + i] = id;
			}
			k += d.n;
		}
		return new NnueDataset(b, r, oc, ov, v, vv, g, p, te, so);
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
		final int[] tr = new int[n], he = new int[n];
		int a = 0, b = 0;
		for (int i = 0; i < n; i++) {
			if (isHeldOut(game[i], salt, pct)) {
				he[b++] = i;
			} else {
				tr[a++] = i;
			}
		}
		return new int[][] { Arrays.copyOf(tr, a), Arrays.copyOf(he, b) };
	}

	// ====================================================================== riepilogo

	public int numGames() {
		final java.util.Set<Integer> s = new java.util.HashSet<>();
		for (int g : game) {
			s.add(g);
		}
		return s.size();
	}

	/** Composizione, per i log. */
	public String describe() {
		int w = 0, b = 0, d = 0, none = 0, val = 0, wt = 0;
		final int[] src = new int[16];
		for (int i = 0; i < n; i++) {
			if (outValid[i] == 0) {
				none++;
			} else if (outcome[i] > 0) {
				w++;
			} else if (outcome[i] < 0) {
				b++;
			} else {
				d++;
			}
			val += valValid[i];
			wt += role[i] == 0 ? 1 : 0;
			src[source[i]]++;
		}
		final StringBuilder sb = new StringBuilder();
		sb.append(n).append(" posizioni in ").append(numGames()).append(" partite; muove il Bianco in ").append(wt).append(", il Nero in ").append(n - wt)
				.append("; esito (vista Bianco): +1 ").append(w).append(", -1 ").append(b).append(", 0 ").append(d).append(", senza esito ").append(none)
				.append("; con valore di ricerca ").append(val).append("; sorgenti:");
		for (int s = 0; s < 16; s++) {
			if (src[s] > 0) {
				sb.append(' ').append(s).append('=').append(src[s]);
			}
		}
		return sb.toString();
	}

	// ====================================================================== file

	public void save(String path) throws IOException {
		try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path)), 1 << 16))) {
			o.writeUTF(MAGIC);
			o.writeInt(n);
			o.write(boards);
			o.write(role);
			o.write(outcome);
			o.write(outValid);
			o.write(valValid);
			o.write(source);
			for (int g : game) {
				o.writeInt(g);
			}
			for (short p : ply) {
				o.writeShort(p);
			}
			for (short t : toEnd) {
				o.writeShort(t);
			}
			for (float v : value) {
				o.writeFloat(v);
			}
		}
	}

	/**
	 * @throws IOException file assente/illeggibile, non gzip, di un altro formato o versione, troncato, con byte in piu', corrotto (CRC di
	 *                     gzip) o con campioni che violano le invarianti
	 */
	public static NnueDataset load(String path) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16))) {
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
			final byte[] boards = new byte[81 * n], role = new byte[n], outcome = new byte[n], outValid = new byte[n], valValid = new byte[n], source = new byte[n];
			in.readFully(boards);
			in.readFully(role);
			in.readFully(outcome);
			in.readFully(outValid);
			in.readFully(valValid);
			in.readFully(source);
			final int[] game = new int[n];
			final short[] ply = new short[n], toEnd = new short[n];
			final float[] value = new float[n];
			for (int i = 0; i < n; i++) {
				game[i] = in.readInt();
			}
			for (int i = 0; i < n; i++) {
				ply[i] = in.readShort();
			}
			for (int i = 0; i < n; i++) {
				toEnd[i] = in.readShort();
			}
			for (int i = 0; i < n; i++) {
				value[i] = in.readFloat();
			}
			if (in.read() != -1) { // leggere fino alla fine fa verificare il CRC di gzip
				throw new IOException("byte in piu' dopo i dati (" + path + ")");
			}
			return new NnueDataset(boards, role, outcome, outValid, value, valValid, game, ply, toEnd, source);
		} catch (EOFException e) {
			throw new IOException("file dati troncato: " + path, e);
		} catch (java.util.zip.ZipException e) {
			throw new IOException("file dati non valido o corrotto (" + e.getMessage() + "): " + path, e);
		} catch (IllegalArgumentException e) {
			throw new IOException("dati non validi in " + path + ": " + e.getMessage(), e);
		}
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof NnueDataset)) {
			return false;
		}
		final NnueDataset d = (NnueDataset) o;
		return n == d.n && Arrays.equals(boards, d.boards) && Arrays.equals(role, d.role) && Arrays.equals(outcome, d.outcome) && Arrays.equals(outValid, d.outValid)
				&& Arrays.equals(valValid, d.valValid) && Arrays.equals(source, d.source) && Arrays.equals(game, d.game) && Arrays.equals(ply, d.ply)
				&& Arrays.equals(toEnd, d.toEnd) && Arrays.equals(value, d.value);
	}

	@Override
	public int hashCode() {
		return n * 31 + Arrays.hashCode(game) + Arrays.hashCode(value);
	}
}
