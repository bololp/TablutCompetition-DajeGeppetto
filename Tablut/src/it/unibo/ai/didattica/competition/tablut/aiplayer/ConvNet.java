package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Rete convoluzionale per valutare una posizione di Tablut, progettata per la CPU (nessuna GPU, Java puro).
 *
 * <h3>Architettura</h3>
 * <pre>
 *  scacchiera 9x9, 3 piani one-hot (BIANCO, NERO, RE)
 *    |  conv 3x3 "sparsa" 3 -> C1 + mappa di bias posizionale (C1 x 81)     [livello 0]
 *    |  leaky-ReLU
 *    |  (conv 3x3 densa C(l-1) -> C(l), leaky-ReLU) x (L-1)                  [livelli 1..L-1]
 *    |  (conv 1x1 C(L-1) -> R, leaky-ReLU)   se {@code reduce} > 0
 *    |  flatten  ++  feature a mano (opzionali, {@code nFeat})
 *    |  denso condiviso (leaky-ReLU) x len(dense)
 *    +-> head del RUOLO (chi deve muovere: 0 = Bianco, 1 = Nero)
 *           denso (leaky-ReLU) x len(head)  ->  1 neurone, tanh   =>  valore in (-1,1), +1 = vince il Bianco
 * </pre>
 * <ul>
 * <li><b>Primo livello sparso, senza piani statici.</b> Le pedine sono ~25 su 81 celle: invece di una convoluzione
 * densa sui piani di ingresso si "spalma" il filtro 3x3 di ciascuna pedina (costo ~ pedine x 9 x C1, non 81 x 9 x 3 x C1).
 * Cittadelle, trono e vie di fuga non sono dati come piani costanti: sono una <b>mappa di bias per posizione</b>
 * (C1 x 81 parametri appresi), che include come caso particolare qualunque convoluzione su piani statici e costa zero MAC.</li>
 * <li><b>Ruolo.</b> Come nella {@link MLP} "B": il tronco (convoluzioni + denso condiviso) e' comune, la head e' scelta da
 * chi deve muovere. Un campione aggiorna il tronco e SOLO la head del suo ruolo.</li>
 * <li><b>Ibrido.</b> Con {@code nFeat > 0} il vettore di feature a mano ({@link TablutFeatures}: vie di fuga, minacce,
 * mobilita'...) e' concatenato al flatten. Le feature costano ~2,4 us e portano in dote informazione esatta e a lungo
 * raggio (le righe/colonne di fuga attraversano tutta la scacchiera: un 3x3 le vede solo a pezzi); il CNN deve solo imparare il
 * residuo spaziale. {@code nFeat = 0} e' il CNN puro.</li>
 * </ul>
 *
 * <h3>Perche' e' fatta cosi' (misurato)</h3>
 * In Java puro HotSpot non vettorizza i loop corti (97 elementi per piano): il kernel migliore trovato fa ~0,4 ns/MAC
 * (scalare). Una conv 3x3 16->16 su 9x9 costa quindi ~75 us, ~25 volte l'intera valutazione dell'MLP: le dimensioni vanno
 * scelte con {@code CnnBench}, non "il piu' grande che sta in memoria". Per questo: primo livello sparso, kernel 3x3 a 9 tap
 * fusi con somme ad albero, backward-input che riusa lo stesso kernel con pesi trasposti, conv 1x1 di riduzione prima del flatten.
 *
 * <h3>Layout in memoria</h3>
 * Ogni piano e' una scacchiera 11x11 con bordo di zeri (indice {@code (r+1)*11 + (c+1)}), canali uno dopo l'altro (NCHW). Un
 * kernel calcola l'intervallo contiguo [12,108] (97 celle, comprese 16 celle di bordo "spazzatura" a meta' riga); poi il
 * bordo viene azzerato. Le letture "oltre il bordo" cadono sempre in celle di bordo (zero): cosi' la convoluzione e'
 * esattamente "same" con zero-padding, senza controlli di indici.
 *
 * <h3>Parametri</h3>
 * Un unico array piatto {@code float[]} (si veda {@link #params()}); la disposizione e' pubblica ({@link #convW} ecc.) perche'
 * i test la usano per un oracolo indipendente in {@code double}. Backward/gradiente in {@code float}, accumulati in un array
 * piatto della stessa forma (comodo per eligibility trace e Adam).
 *
 * <h3>Thread-safety</h3>
 * La rete e' in sola lettura durante l'inferenza e tutto lo stato temporaneo sta in un {@link Workspace} (uno per thread).
 * I pesi si modificano solo dal thread che addestra, mentre gli altri thread usano copie ({@link #copy()}).
 */
public final class ConvNet {

	public static final int N = AshtonRules.SIZE; // 9
	public static final int PW = N + 2; // 11: lato del piano con bordo
	public static final int PS = PW * PW; // 121
	public static final int LO = PW + 1; // 12: prima cella interna
	public static final int HI = N * PW + N; // 108: ultima cella interna
	public static final int PIECE_TYPES = 3; // BIANCO, NERO, RE
	static final float LEAK = 0.01f;
	/** Scostamenti dei 9 tap 3x3 nel piano con bordo: (dr,dc) = (-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1). */
	static final int[] OFF = { -PW - 1, -PW, -PW + 1, -1, 0, 1, PW - 1, PW, PW + 1 };
	/** Le 40 celle di bordo del piano 11x11. */
	static final int[] BORDER = new int[PS - N * N];

	static {
		int k = 0;
		for (int r = 0; r < PW; r++) {
			for (int c = 0; c < PW; c++) {
				if (r == 0 || r == PW - 1 || c == 0 || c == PW - 1) {
					BORDER[k++] = r * PW + c;
				}
			}
		}
		if (k != BORDER.length) {
			throw new IllegalStateException("bordo");
		}
	}

	// ====================================================================== configurazione

	/** Forma della rete. Immutabile. */
	public static final class Config {
		/** Canali dei livelli conv 3x3 (almeno uno). {@code conv[0]} e' il livello sparso 3 -> conv[0]. */
		public final int[] conv;
		/** Canali del livello 1x1 prima del flatten; 0 = nessuna riduzione (flatten diretto dell'ultimo livello). */
		public final int reduce;
		/** Numero di feature a mano concatenate al flatten (0 = CNN puro). */
		public final int nFeat;
		/** Strati densi condivisi (possono mancare). */
		public final int[] dense;
		/** Strati nascosti di ciascuna head di ruolo (possono mancare: allora l'uscita e' lineare sul denso condiviso). */
		public final int[] head;

		public Config(int[] conv, int reduce, int nFeat, int[] dense, int[] head) {
			if (conv == null || conv.length == 0) {
				throw new IllegalArgumentException("serve almeno un livello conv");
			}
			for (int c : conv) {
				if (c < 1) {
					throw new IllegalArgumentException("canali conv non validi");
				}
			}
			if (reduce < 0 || nFeat < 0) {
				throw new IllegalArgumentException("reduce/nFeat negativi");
			}
			for (int d : dense) {
				if (d < 1) {
					throw new IllegalArgumentException("strato denso non valido");
				}
			}
			for (int d : head) {
				if (d < 1) {
					throw new IllegalArgumentException("strato head non valido");
				}
			}
			this.conv = conv.clone();
			this.reduce = reduce;
			this.nFeat = nFeat;
			this.dense = dense.clone();
			this.head = head.clone();
		}

		/** Canali dell'ultimo piano prima del flatten. */
		public int flatChannels() {
			return reduce > 0 ? reduce : conv[conv.length - 1];
		}

		/** Lunghezza del vettore flatten (senza feature). */
		public int flatSize() {
			return flatChannels() * N * N;
		}

		/** Ingresso del primo strato denso: flatten + feature a mano. */
		public int denseIn() {
			return flatSize() + nFeat;
		}

		@Override
		public String toString() {
			return "CNN conv=" + join(conv) + " reduce=" + reduce + " feat=" + nFeat + " dense=" + join(dense) + " head=" + join(head);
		}

		@Override
		public boolean equals(Object o) {
			if (!(o instanceof Config)) {
				return false;
			}
			Config c = (Config) o;
			return reduce == c.reduce && nFeat == c.nFeat && Arrays.equals(conv, c.conv) && Arrays.equals(dense, c.dense)
					&& Arrays.equals(head, c.head);
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(conv) * 31 + reduce * 17 + nFeat * 13 + Arrays.hashCode(dense) * 7 + Arrays.hashCode(head);
		}
	}

	// ====================================================================== stato

	private final Config cfg;
	private final float[] P; // parametri
	private final int numParams;

	// --- disposizione dei parametri (pubblica: la usa il test con l'oracolo in double) ---
	/** Inizio dei pesi del livello conv l: livello 0 = [C0][3][9]; livelli l>=1 = [Cl][C(l-1)][9]. */
	public final int[] convW;
	/** Bias del livello l: livello 0 = mappa di bias posizionale [C0][81]; livelli l>=1 = [Cl]. */
	public final int[] convB;
	/** Pesi della conv 1x1 [R][C(L-1)] e suoi bias [R] (solo se reduce > 0, altrimenti -1). */
	public final int redW, redB;
	/** Strati densi condivisi: pesi [out][in] riga per riga, bias [out]. */
	public final int[] denseW, denseB;
	/** Strati della head del ruolo r: {@code headW[r][j]} pesi [out][in], {@code headB[r][j]} bias. L'ultimo strato ha 1 uscita. */
	public final int[][] headW, headB;

	/** Nuova rete con pesi casuali (He, uscita piccola). */
	public ConvNet(Config cfg, long seed) {
		this(cfg, (float[]) null);
		initRandom(seed);
	}

	/** Rete con i parametri dati (copiati). */
	public ConvNet(Config cfg, float[] params) {
		this.cfg = cfg;
		int L = cfg.conv.length;
		convW = new int[L];
		convB = new int[L];
		int off = 0;
		convW[0] = off;
		off += cfg.conv[0] * PIECE_TYPES * 9;
		convB[0] = off;
		off += cfg.conv[0] * N * N;
		for (int l = 1; l < L; l++) {
			convW[l] = off;
			off += cfg.conv[l] * cfg.conv[l - 1] * 9;
			convB[l] = off;
			off += cfg.conv[l];
		}
		if (cfg.reduce > 0) {
			redW = off;
			off += cfg.reduce * cfg.conv[L - 1];
			redB = off;
			off += cfg.reduce;
		} else {
			redW = -1;
			redB = -1;
		}
		int m = cfg.dense.length;
		denseW = new int[m];
		denseB = new int[m];
		int in = cfg.denseIn();
		for (int i = 0; i < m; i++) {
			denseW[i] = off;
			off += in * cfg.dense[i];
			denseB[i] = off;
			off += cfg.dense[i];
			in = cfg.dense[i];
		}
		int headIn = in;
		int H = cfg.head.length + 1;
		headW = new int[2][H];
		headB = new int[2][H];
		for (int r = 0; r < 2; r++) {
			int hin = headIn;
			for (int j = 0; j < H; j++) {
				int hout = j < H - 1 ? cfg.head[j] : 1;
				headW[r][j] = off;
				off += hin * hout;
				headB[r][j] = off;
				off += hout;
				hin = hout;
			}
		}
		this.numParams = off;
		if (params != null) {
			if (params.length != numParams) {
				throw new IllegalArgumentException("attesi " + numParams + " parametri, ricevuti " + params.length);
			}
			this.P = params.clone();
		} else {
			this.P = new float[numParams];
		}
	}

	private void initRandom(long seed) {
		Random rnd = new Random(seed);
		int L = cfg.conv.length;
		// livello 0 (sparso): ogni cella riceve pochi contributi, fan-in efficace ~ 3*9
		double s0 = Math.sqrt(2.0 / (PIECE_TYPES * 9));
		for (int i = 0; i < cfg.conv[0] * PIECE_TYPES * 9; i++) {
			P[convW[0] + i] = (float) (rnd.nextGaussian() * s0);
		}
		// mappa di bias posizionale: parte da zero
		for (int l = 1; l < L; l++) {
			double s = Math.sqrt(2.0 / (cfg.conv[l - 1] * 9));
			for (int i = 0; i < cfg.conv[l] * cfg.conv[l - 1] * 9; i++) {
				P[convW[l] + i] = (float) (rnd.nextGaussian() * s);
			}
		}
		if (cfg.reduce > 0) {
			double s = Math.sqrt(2.0 / cfg.conv[L - 1]);
			for (int i = 0; i < cfg.reduce * cfg.conv[L - 1]; i++) {
				P[redW + i] = (float) (rnd.nextGaussian() * s);
			}
		}
		int in = cfg.denseIn();
		for (int i = 0; i < cfg.dense.length; i++) {
			double s = Math.sqrt(2.0 / in);
			for (int k = 0; k < in * cfg.dense[i]; k++) {
				P[denseW[i] + k] = (float) (rnd.nextGaussian() * s);
			}
			in = cfg.dense[i];
		}
		int H = cfg.head.length + 1;
		for (int r = 0; r < 2; r++) {
			int hin = in;
			for (int j = 0; j < H; j++) {
				int hout = j < H - 1 ? cfg.head[j] : 1;
				double s = Math.sqrt(2.0 / hin) * (j == H - 1 ? 0.2 : 1.0); // uscita piccola: si parte vicino a 0
				for (int k = 0; k < hin * hout; k++) {
					P[headW[r][j] + k] = (float) (rnd.nextGaussian() * s);
				}
				hin = hout;
			}
		}
	}

	public Config config() {
		return cfg;
	}

	/** Parametri VIVI (non una copia): chi li modifica deve farlo da un solo thread. */
	public float[] params() {
		return P;
	}

	public int numParams() {
		return numParams;
	}

	public ConvNet copy() {
		return new ConvNet(cfg, P);
	}

	/** Moltiplicazioni-addizioni per una valutazione (stima dei costi: celle 97 per piano, primo livello sparso a ~25 pedine). */
	public long macsPerEval() {
		int L = cfg.conv.length;
		long m = 25L * 9 * cfg.conv[0];
		for (int l = 1; l < L; l++) {
			m += (long) 97 * 9 * cfg.conv[l] * cfg.conv[l - 1];
		}
		if (cfg.reduce > 0) {
			m += (long) 97 * cfg.reduce * cfg.conv[L - 1];
		}
		int in = cfg.denseIn();
		for (int d : cfg.dense) {
			m += (long) in * d;
			in = d;
		}
		for (int d : cfg.head) {
			m += (long) in * d;
			in = d;
		}
		m += in;
		return m;
	}

	// ====================================================================== workspace

	/** Stato temporaneo di UN thread: pedine, attivazioni e (se si fa backward) gradienti. */
	public final class Workspace {
		final int[] pCell = new int[N * N];
		final int[] pType = new int[N * N];
		int nPieces;
		final float[][] a; // attivazioni dei livelli conv: a[l] = C_l * PS
		final float[] ar; // attivazioni della conv 1x1 (R * PS) o null
		final float[] flat; // ingresso del primo denso: flatten ++ feature
		final float[][] sh; // uscite dei densi condivisi
		final float[][][] hd; // uscite delle head: hd[ruolo][j]
		int role;
		float out;
		/** Logit pre-tanh dell'ultimo forward: {@code out == (float) Math.tanh(logit)} esattamente. */
		float logit;
		boolean valid; // true dopo un forward
		// gradienti (allocati alla prima backward)
		float[][] da;
		float[] dar, dflat, wt;
		float[][] dsh;
		float[][][] dhd;

		Workspace() {
			int L = cfg.conv.length;
			a = new float[L][];
			for (int l = 0; l < L; l++) {
				a[l] = new float[cfg.conv[l] * PS];
			}
			ar = cfg.reduce > 0 ? new float[cfg.reduce * PS] : null;
			flat = new float[cfg.denseIn()];
			sh = new float[cfg.dense.length][];
			for (int i = 0; i < sh.length; i++) {
				sh[i] = new float[cfg.dense[i]];
			}
			hd = new float[2][cfg.head.length + 1][];
			for (int r = 0; r < 2; r++) {
				for (int j = 0; j <= cfg.head.length; j++) {
					hd[r][j] = new float[j < cfg.head.length ? cfg.head[j] : 1];
				}
			}
		}

		/** Imposta le pedine dalla scacchiera di {@code s} (turno ignorato: il ruolo si passa a {@link ConvNet#forward}). Converte: alloca. */
		public void setBoard(State s) {
			setBoard(FastState.fromState(s));
		}

		/** Imposta le pedine dalla scacchiera compatta di {@code s} (turno ignorato: il ruolo si passa a {@link ConvNet#forward}). */
		public void setBoard(FastState s) {
			final byte[] b = s.board;
			int n = 0;
			for (int r = 0; r < N; r++) {
				for (int c = 0; c < N; c++) {
					final byte p = b[r * N + c];
					int t;
					if (p == BoardConstants.WHITE) {
						t = 0;
					} else if (p == BoardConstants.BLACK) {
						t = 1;
					} else if (p == BoardConstants.KING) {
						t = 2;
					} else {
						continue;
					}
					pCell[n] = (r + 1) * PW + c + 1;
					pType[n] = t;
					n++;
				}
			}
			nPieces = n;
		}

		/**
		 * Imposta le pedine da una scacchiera compatta di 81 codici (0 vuoto, 1 bianco, 2 nero, 3 re), applicando
		 * la simmetria {@code sym} ({@link BoardSymmetry}).
		 */
		public void setCodes(byte[] cells, int sym) {
			setCodes(cells, 0, sym);
		}

		/** Come {@link #setCodes(byte[], int)}, leggendo le 81 celle da {@code cells[offset .. offset+80]} (array piatto di campioni). */
		public void setCodes(byte[] cells, int offset, int sym) {
			int[] map = BoardSymmetry.MAP[sym];
			int n = 0;
			for (int i = 0; i < N * N; i++) {
				int code = cells[offset + i];
				if (code == 0) {
					continue;
				}
				int j = map[i];
				pCell[n] = (j / N + 1) * PW + (j % N) + 1;
				pType[n] = code - 1;
				n++;
			}
			nPieces = n;
		}

		/** Uscita dell'ultimo forward (in (-1,1)). */
		public float output() {
			return out;
		}

		/**
		 * Logit {@code z} dell'ultimo forward, cioe' l'ingresso della tanh finale ({@code output() == tanh(z)}). Serve alla loss nel
		 * dominio dei logit: ricavarlo con {@code atanh(output())} perderebbe tutta la precisione quando la tanh e' satura (in float
		 * {@code tanh(z)} vale 1 gia' per z ~ 9).
		 */
		public float logit() {
			return logit;
		}

		void ensureBackward() {
			if (da != null) {
				return;
			}
			int L = cfg.conv.length;
			da = new float[L][];
			int maxW = 0;
			for (int l = 0; l < L; l++) {
				da[l] = new float[cfg.conv[l] * PS];
				if (l >= 1) {
					maxW = Math.max(maxW, cfg.conv[l] * cfg.conv[l - 1] * 9);
				}
			}
			wt = new float[Math.max(1, maxW)];
			dar = cfg.reduce > 0 ? new float[cfg.reduce * PS] : null;
			dflat = new float[cfg.denseIn()];
			dsh = new float[cfg.dense.length][];
			for (int i = 0; i < dsh.length; i++) {
				dsh[i] = new float[cfg.dense[i]];
			}
			dhd = new float[2][cfg.head.length + 1][];
			for (int r = 0; r < 2; r++) {
				for (int j = 0; j <= cfg.head.length; j++) {
					dhd[r][j] = new float[j < cfg.head.length ? cfg.head[j] : 1];
				}
			}
		}
	}

	public Workspace newWorkspace() {
		return new Workspace();
	}

	// ====================================================================== forward

	/**
	 * Valuta la posizione impostata in {@code ws} (si veda {@link Workspace#setBoard}).
	 *
	 * @param feats feature a mano (lunghezza {@code nFeat}); ignorato se {@code nFeat == 0}
	 * @param role  0 = muove il Bianco, 1 = muove il Nero (sceglie la head)
	 * @return valore in (-1,1), dal punto di vista del Bianco
	 */
	public double forward(Workspace ws, float[] feats, int role) {
		final float[] p = P;
		final int L = cfg.conv.length;

		// ---- livello 0: bias posizionale + filtri 3x3 delle pedine (scatter) ----
		final int c0 = cfg.conv[0];
		final float[] a0 = ws.a[0];
		for (int o = 0; o < c0; o++) {
			final int zb = o * PS;
			Arrays.fill(a0, zb, zb + PS, 0f);
			final int bb = convB[0] + o * N * N;
			for (int r = 0; r < N; r++) {
				final int dst = zb + (r + 1) * PW + 1;
				final int src = bb + r * N;
				for (int c = 0; c < N; c++) {
					a0[dst + c] = p[src + c];
				}
			}
			// scatter: la pedina in q contribuisce alla cella q - OFF[k] con il peso k. Nove tap srotolati con scostamenti
			// costanti (OFF = {-12,-11,-10,-1,0,1,10,11,12}): niente lookup su OFF ne' cicli corti.
			final int nP = ws.nPieces;
			final int[] pc = ws.pCell, pt = ws.pType;
			for (int i = 0; i < nP; i++) {
				final int wb = convW[0] + (o * PIECE_TYPES + pt[i]) * 9;
				final int q = zb + pc[i];
				a0[q + PW + 1] += p[wb];
				a0[q + PW] += p[wb + 1];
				a0[q + PW - 1] += p[wb + 2];
				a0[q + 1] += p[wb + 3];
				a0[q] += p[wb + 4];
				a0[q - 1] += p[wb + 5];
				a0[q - PW + 1] += p[wb + 6];
				a0[q - PW] += p[wb + 7];
				a0[q - PW - 1] += p[wb + 8];
			}
			leakyRange(a0, zb);
		}
		zeroBorders(a0, c0);

		// ---- livelli 1..L-1: conv 3x3 densa ----
		for (int l = 1; l < L; l++) {
			final int cout = cfg.conv[l], cin = cfg.conv[l - 1];
			final float[] out = ws.a[l];
			for (int o = 0; o < cout; o++) {
				final float bias = p[convB[l] + o];
				final int ob = o * PS;
				for (int i = LO; i <= HI; i++) {
					out[ob + i] = bias;
				}
			}
			conv3Acc(ws.a[l - 1], cin, p, convW[l], out, cout);
			for (int o = 0; o < cout; o++) {
				leakyRange(out, o * PS);
			}
			zeroBorders(out, cout);
		}

		// ---- conv 1x1 di riduzione ----
		final float[] src;
		if (cfg.reduce > 0) {
			final int cin = cfg.conv[L - 1], R = cfg.reduce;
			final float[] in = ws.a[L - 1];
			for (int r = 0; r < R; r++) {
				final int rb = r * PS;
				final float bias = p[redB + r];
				for (int i = LO; i <= HI; i++) {
					ws.ar[rb + i] = bias;
				}
				for (int c = 0; c < cin; c++) {
					final float w = p[redW + r * cin + c];
					final int cb = c * PS;
					for (int i = LO; i <= HI; i++) {
						ws.ar[rb + i] += w * in[cb + i];
					}
				}
				leakyRange(ws.ar, rb);
			}
			zeroBorders(ws.ar, R);
			src = ws.ar;
		} else {
			src = ws.a[L - 1];
		}

		// ---- flatten (+ feature) ----
		final int nch = cfg.flatChannels();
		for (int ch = 0; ch < nch; ch++) {
			for (int r = 0; r < N; r++) {
				final int s = ch * PS + (r + 1) * PW + 1;
				final int d = ch * N * N + r * N;
				for (int c = 0; c < N; c++) {
					ws.flat[d + c] = src[s + c];
				}
			}
		}
		final int fs = cfg.flatSize();
		for (int i = 0; i < cfg.nFeat; i++) {
			ws.flat[fs + i] = feats[i];
		}

		// ---- densi condivisi ----
		float[] x = ws.flat;
		int in = cfg.denseIn();
		for (int i = 0; i < cfg.dense.length; i++) {
			dense(p, denseW[i], denseB[i], x, in, ws.sh[i], cfg.dense[i], false);
			x = ws.sh[i];
			in = cfg.dense[i];
		}

		// ---- head del ruolo ----
		final int H = cfg.head.length + 1;
		for (int j = 0; j < H; j++) {
			final int hout = j < H - 1 ? cfg.head[j] : 1;
			if (j == H - 1) {
				// ultimo strato (1 uscita): stesso calcolo di dense(...) ma conservando il logit z prima della tanh
				final float z = outPre(p, headW[role][j], headB[role][j], x, in);
				ws.logit = z;
				ws.hd[role][j][0] = (float) Math.tanh(z);
			} else {
				dense(p, headW[role][j], headB[role][j], x, in, ws.hd[role][j], hout, false);
			}
			x = ws.hd[role][j];
			in = hout;
		}
		ws.role = role;
		ws.out = x[0];
		ws.valid = true;
		return x[0];
	}

	// ====================================================================== backward

	/**
	 * Gradiente di (outGrad * uscita) rispetto a TUTTI i parametri, ACCUMULATO in {@code grad} (stessa forma di
	 * {@link #params()}). Va chiamata subito dopo il {@link #forward} della stessa posizione (usa le attivazioni
	 * salvate in {@code ws}). Il gradiente non tocca la head dell'altro ruolo.
	 */
	public void backward(Workspace ws, double outGrad, float[] grad) {
		backwardImpl(ws, outGrad, false, grad);
	}

	/**
	 * Come {@link #backward} ma {@code logitGrad} e' il gradiente rispetto al LOGIT {@code z} (ingresso della tanh finale), non
	 * rispetto all'uscita: lo strato finale non moltiplica per {@code 1 - y^2}. Per una loss {@code L(z)} nel dominio dei logit
	 * (es. {@code (z - t)^2}) si passa {@code dL/dz}; resta esatto anche quando la tanh e' satura (dove {@code 1 - y^2} = 0).
	 */
	public void backwardLogit(Workspace ws, double logitGrad, float[] grad) {
		backwardImpl(ws, logitGrad, true, grad);
	}

	private void backwardImpl(Workspace ws, double outGrad, boolean logitMode, float[] grad) {
		if (!ws.valid) {
			throw new IllegalStateException("backward senza forward");
		}
		ws.ensureBackward();
		final float[] p = P;
		final int L = cfg.conv.length;
		final int role = ws.role;

		// ---- head del ruolo ----
		final int H = cfg.head.length + 1;
		final int m = cfg.dense.length;
		final float[] headIn = m > 0 ? ws.sh[m - 1] : ws.flat;
		final float[] dHeadIn = m > 0 ? ws.dsh[m - 1] : ws.dflat;
		ws.dhd[role][H - 1][0] = (float) outGrad;
		for (int j = H - 1; j >= 0; j--) {
			final float[] x = j == 0 ? headIn : ws.hd[role][j - 1];
			final float[] dx = j == 0 ? dHeadIn : ws.dhd[role][j - 1];
			final int hin = j == 0 ? (m > 0 ? cfg.dense[m - 1] : cfg.denseIn()) : cfg.head[j - 1];
			final int hout = j < H - 1 ? cfg.head[j] : 1;
			if (logitMode && j == H - 1) {
				denseBackPre(p, headW[role][j], headB[role][j], x, hin, ws.dhd[role][j][0], grad, dx);
			} else {
				denseBack(p, headW[role][j], headB[role][j], x, hin, ws.hd[role][j], ws.dhd[role][j], hout, j == H - 1, grad, dx);
			}
		}

		// ---- densi condivisi ----
		for (int i = m - 1; i >= 0; i--) {
			final float[] x = i == 0 ? ws.flat : ws.sh[i - 1];
			final float[] dx = i == 0 ? ws.dflat : ws.dsh[i - 1];
			final int in = i == 0 ? cfg.denseIn() : cfg.dense[i - 1];
			denseBack(p, denseW[i], denseB[i], x, in, ws.sh[i], ws.dsh[i], cfg.dense[i], false, grad, dx);
		}

		// ---- unflatten: dflat[0..flatSize) -> piani ----
		final int nch = cfg.flatChannels();
		final float[] dsrc = cfg.reduce > 0 ? ws.dar : ws.da[L - 1];
		Arrays.fill(dsrc, 0, nch * PS, 0f);
		for (int ch = 0; ch < nch; ch++) {
			for (int r = 0; r < N; r++) {
				final int d = ch * PS + (r + 1) * PW + 1;
				final int s = ch * N * N + r * N;
				for (int c = 0; c < N; c++) {
					dsrc[d + c] = ws.dflat[s + c];
				}
			}
		}

		// ---- conv 1x1 ----
		if (cfg.reduce > 0) {
			final int cin = cfg.conv[L - 1], R = cfg.reduce;
			final float[] in = ws.a[L - 1];
			final float[] dIn = ws.da[L - 1];
			Arrays.fill(dIn, 0, cin * PS, 0f);
			for (int r = 0; r < R; r++) {
				final int rb = r * PS;
				// dz = dar * leaky'(ar), in place; il bordo resta 0
				float db = 0;
				for (int i = LO; i <= HI; i++) {
					final float dz = ws.dar[rb + i] * (ws.ar[rb + i] > 0 ? 1f : LEAK);
					ws.dar[rb + i] = dz;
					db += dz;
				}
				grad[redB + r] += db;
				for (int c = 0; c < cin; c++) {
					final int cb = c * PS;
					final float w = p[redW + r * cin + c];
					float s = 0;
					for (int i = LO; i <= HI; i++) {
						final float dz = ws.dar[rb + i];
						s += dz * in[cb + i];
						dIn[cb + i] += w * dz;
					}
					grad[redW + r * cin + c] += s;
				}
			}
			zeroBorders(dIn, cin);
		}

		// ---- livelli conv L-1 .. 1 ----
		for (int l = L - 1; l >= 1; l--) {
			final int cout = cfg.conv[l], cin = cfg.conv[l - 1];
			final float[] dz = ws.da[l];
			final float[] act = ws.a[l];
			// dz = da * leaky'(a) in place (il bordo ha da = 0)
			for (int o = 0; o < cout; o++) {
				final int ob = o * PS;
				float db = 0;
				for (int i = LO; i <= HI; i++) {
					final float v = dz[ob + i] * (act[ob + i] > 0 ? 1f : LEAK);
					dz[ob + i] = v;
					db += v;
				}
				grad[convB[l] + o] += db;
			}
			zeroBorders(dz, cout);
			convGradW(dz, cout, ws.a[l - 1], cin, grad, convW[l]);
			// dA(l-1) = conv(dz, W trasposto): lo stesso kernel del forward
			final float[] wt = ws.wt;
			final int wo = convW[l];
			for (int o = 0; o < cout; o++) {
				for (int c = 0; c < cin; c++) {
					final int srcw = wo + (o * cin + c) * 9;
					final int dstw = (c * cout + o) * 9;
					for (int k = 0; k < 9; k++) {
						wt[dstw + 8 - k] = p[srcw + k];
					}
				}
			}
			final float[] dPrev = ws.da[l - 1];
			Arrays.fill(dPrev, 0, cin * PS, 0f);
			conv3Acc(dz, cout, wt, 0, dPrev, cin);
			zeroBorders(dPrev, cin);
		}

		// ---- livello 0 ----
		final int c0 = cfg.conv[0];
		final float[] dz0 = ws.da[0];
		final float[] act0 = ws.a[0];
		for (int o = 0; o < c0; o++) {
			final int ob = o * PS;
			for (int i = LO; i <= HI; i++) {
				dz0[ob + i] *= act0[ob + i] > 0 ? 1f : LEAK;
			}
		}
		zeroBorders(dz0, c0);
		for (int o = 0; o < c0; o++) {
			final int ob = o * PS;
			final int bb = convB[0] + o * N * N;
			for (int r = 0; r < N; r++) {
				final int s = ob + (r + 1) * PW + 1;
				for (int c = 0; c < N; c++) {
					grad[bb + r * N + c] += dz0[s + c];
				}
			}
			for (int i = 0; i < ws.nPieces; i++) {
				final int wb = convW[0] + (o * PIECE_TYPES + ws.pType[i]) * 9;
				final int q = ob + ws.pCell[i];
				for (int k = 0; k < 9; k++) {
					grad[wb + k] += dz0[q - OFF[k]];
				}
			}
		}
	}

	/** {@code w[i] += scale * g[i]} per tutti i parametri (aggiornamento semi-gradiente TD). */
	public void addScaled(float[] g, double scale) {
		final float s = (float) scale;
		for (int i = 0; i < P.length; i++) {
			P[i] += s * g[i];
		}
	}

	// ====================================================================== kernel

	/** Leaky-ReLU sul tratto contiguo [LO,HI] del piano che inizia in {@code base}. */
	private static void leakyRange(float[] z, int base) {
		for (int i = base + LO; i <= base + HI; i++) {
			final float v = z[i];
			z[i] = v > 0 ? v : LEAK * v;
		}
	}

	/** Azzera le 40 celle di bordo di ciascuno dei {@code planes} piani. */
	private static void zeroBorders(float[] z, int planes) {
		for (int ch = 0; ch < planes; ch++) {
			final int b = ch * PS;
			for (int k = 0; k < BORDER.length; k++) {
				z[b + BORDER[k]] = 0f;
			}
		}
	}

	/**
	 * {@code out[o][p] += somma_c somma_k w[o][c][k] * in[c][p + OFF[k]]} per p in [LO,HI]. Due canali di uscita per volta
	 * (condividono i 9 load dell'ingresso) e somme ad albero (la catena seriale di 9 addizioni era il collo di bottiglia).
	 */
	static void conv3Acc(float[] in, int cin, float[] w, int wOff, float[] out, int cout) {
		int o = 0;
		for (; o + 1 < cout; o += 2) {
			final int ob0 = o * PS, ob1 = ob0 + PS;
			for (int c = 0; c < cin; c++) {
				final int ib = c * PS;
				final int wa = wOff + (o * cin + c) * 9, wb = wOff + ((o + 1) * cin + c) * 9;
				final float a0 = w[wa], a1 = w[wa + 1], a2 = w[wa + 2], a3 = w[wa + 3], a4 = w[wa + 4], a5 = w[wa + 5],
						a6 = w[wa + 6], a7 = w[wa + 7], a8 = w[wa + 8];
				final float b0 = w[wb], b1 = w[wb + 1], b2 = w[wb + 2], b3 = w[wb + 3], b4 = w[wb + 4], b5 = w[wb + 5],
						b6 = w[wb + 6], b7 = w[wb + 7], b8 = w[wb + 8];
				for (int p = LO; p <= HI; p++) {
					final int i = ib + p;
					final float x0 = in[i - PW - 1], x1 = in[i - PW], x2 = in[i - PW + 1], x3 = in[i - 1], x4 = in[i], x5 = in[i + 1],
							x6 = in[i + PW - 1], x7 = in[i + PW], x8 = in[i + PW + 1];
					out[ob0 + p] += (((a0 * x0 + a1 * x1) + (a2 * x2 + a3 * x3)) + ((a4 * x4 + a5 * x5) + (a6 * x6 + a7 * x7))) + a8 * x8;
					out[ob1 + p] += (((b0 * x0 + b1 * x1) + (b2 * x2 + b3 * x3)) + ((b4 * x4 + b5 * x5) + (b6 * x6 + b7 * x7))) + b8 * x8;
				}
			}
		}
		if (o < cout) {
			final int ob0 = o * PS;
			for (int c = 0; c < cin; c++) {
				final int ib = c * PS;
				final int wa = wOff + (o * cin + c) * 9;
				final float a0 = w[wa], a1 = w[wa + 1], a2 = w[wa + 2], a3 = w[wa + 3], a4 = w[wa + 4], a5 = w[wa + 5],
						a6 = w[wa + 6], a7 = w[wa + 7], a8 = w[wa + 8];
				for (int p = LO; p <= HI; p++) {
					final int i = ib + p;
					out[ob0 + p] += (((a0 * in[i - PW - 1] + a1 * in[i - PW]) + (a2 * in[i - PW + 1] + a3 * in[i - 1]))
							+ ((a4 * in[i] + a5 * in[i + 1]) + (a6 * in[i + PW - 1] + a7 * in[i + PW]))) + a8 * in[i + PW + 1];
				}
			}
		}
	}

	/** {@code grad[wOff + (o*cin+c)*9 + k] += somma_p dz[o][p] * in[c][p + OFF[k]]}: 9 accumulatori indipendenti per coppia (o,c). */
	private static void convGradW(float[] dz, int cout, float[] in, int cin, float[] grad, int wOff) {
		for (int o = 0; o < cout; o++) {
			final int ob = o * PS;
			for (int c = 0; c < cin; c++) {
				final int ib = c * PS;
				float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0;
				for (int p = LO; p <= HI; p++) {
					final float d = dz[ob + p];
					final int i = ib + p;
					s0 += d * in[i - PW - 1];
					s1 += d * in[i - PW];
					s2 += d * in[i - PW + 1];
					s3 += d * in[i - 1];
					s4 += d * in[i];
					s5 += d * in[i + 1];
					s6 += d * in[i + PW - 1];
					s7 += d * in[i + PW];
					s8 += d * in[i + PW + 1];
				}
				final int g = wOff + (o * cin + c) * 9;
				grad[g] += s0;
				grad[g + 1] += s1;
				grad[g + 2] += s2;
				grad[g + 3] += s3;
				grad[g + 4] += s4;
				grad[g + 5] += s5;
				grad[g + 6] += s6;
				grad[g + 7] += s7;
				grad[g + 8] += s8;
			}
		}
	}

	/**
	 * {@code y[o] = act(b[o] + somma_i w[o][i] x[i])}; act = leaky-ReLU, oppure tanh se {@code tanh}.
	 * Quattro neuroni di uscita per volta (ogni x[i] e' caricato una volta ogni quattro MAC) e due ingressi per iterazione: 8
	 * accumulatori indipendenti. Misurato ~2x piu' veloce del prodotto scalare a 4 accumulatori su un solo neurone.
	 */
	private static void dense(float[] p, int wOff, int bOff, float[] x, int in, float[] y, int out, boolean tanh) {
		int o = 0;
		for (; o + 3 < out; o += 4) {
			final int w0 = wOff + o * in, w1 = w0 + in, w2 = w1 + in, w3 = w2 + in;
			float a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
			int i = 0;
			for (; i + 1 < in; i += 2) {
				final float xa = x[i], xb = x[i + 1];
				a0 += p[w0 + i] * xa;
				b0 += p[w0 + i + 1] * xb;
				a1 += p[w1 + i] * xa;
				b1 += p[w1 + i + 1] * xb;
				a2 += p[w2 + i] * xa;
				b2 += p[w2 + i + 1] * xb;
				a3 += p[w3 + i] * xa;
				b3 += p[w3 + i + 1] * xb;
			}
			if (i < in) {
				final float xa = x[i];
				a0 += p[w0 + i] * xa;
				a1 += p[w1 + i] * xa;
				a2 += p[w2 + i] * xa;
				a3 += p[w3 + i] * xa;
			}
			y[o] = act(p[bOff + o] + (a0 + b0), tanh);
			y[o + 1] = act(p[bOff + o + 1] + (a1 + b1), tanh);
			y[o + 2] = act(p[bOff + o + 2] + (a2 + b2), tanh);
			y[o + 3] = act(p[bOff + o + 3] + (a3 + b3), tanh);
		}
		for (; o < out; o++) {
			final int wb = wOff + o * in;
			float s0 = 0, s1 = 0;
			int i = 0;
			for (; i + 1 < in; i += 2) {
				s0 += p[wb + i] * x[i];
				s1 += p[wb + i + 1] * x[i + 1];
			}
			if (i < in) {
				s0 += p[wb + i] * x[i];
			}
			y[o] = act(p[bOff + o] + (s0 + s1), tanh);
		}
	}

	/**
	 * Pre-attivazione (logit) di uno strato a UNA uscita. Replica ESATTAMENTE l'ordine delle somme del ramo scalare di
	 * {@link #dense}, cosi' {@code (float) Math.tanh(outPre(...))} e' bit-identico a cio' che {@code dense(..., tanh=true)} produceva.
	 */
	private static float outPre(float[] p, int wOff, int bOff, float[] x, int in) {
		float s0 = 0, s1 = 0;
		int i = 0;
		for (; i + 1 < in; i += 2) {
			s0 += p[wOff + i] * x[i];
			s1 += p[wOff + i + 1] * x[i + 1];
		}
		if (i < in) {
			s0 += p[wOff + i] * x[i];
		}
		return p[bOff] + (s0 + s1);
	}

	/** Backward di uno strato a UNA uscita dato il gradiente {@code dz} rispetto alla sua PRE-attivazione; SOVRASCRIVE {@code dx}. */
	private static void denseBackPre(float[] p, int wOff, int bOff, float[] x, int in, float dz, float[] grad, float[] dx) {
		Arrays.fill(dx, 0, in, 0f);
		grad[bOff] += dz;
		for (int i = 0; i < in; i++) {
			grad[wOff + i] += dz * x[i];
			dx[i] += p[wOff + i] * dz;
		}
	}

	private static float act(float z, boolean tanh) {
		return tanh ? (float) Math.tanh(z) : (z > 0 ? z : LEAK * z);
	}

	/** Backward di uno strato denso. {@code dy} = gradiente rispetto all'uscita (post-attivazione); SOVRASCRIVE {@code dx}. */
	private static void denseBack(float[] p, int wOff, int bOff, float[] x, int in, float[] y, float[] dy, int out, boolean tanh,
			float[] grad, float[] dx) {
		Arrays.fill(dx, 0, in, 0f);
		for (int o = 0; o < out; o++) {
			final float dz = tanh ? dy[o] * (1f - y[o] * y[o]) : dy[o] * (y[o] > 0 ? 1f : LEAK);
			grad[bOff + o] += dz;
			final int wb = wOff + o * in;
			for (int i = 0; i < in; i++) {
				grad[wb + i] += dz * x[i];
				dx[i] += p[wb + i] * dz;
			}
		}
	}

	// ====================================================================== salvataggio

	public static final String MAGIC = "TABLUT_CNN v1";

	/** Risultato del caricamento: rete + versione/gruppi di feature a mano con cui e' stata allenata (0/0 se CNN puro). */
	public static final class Loaded {
		public final ConvNet net;
		public final int featureVersion;
		public final int featureGroups;

		Loaded(ConvNet net, int featureVersion, int featureGroups) {
			this.net = net;
			this.featureVersion = featureVersion;
			this.featureGroups = featureGroups;
		}
	}

	/** Vero se il file inizia con l'intestazione dei pesi CNN (serve a scegliere fra {@link ConvNet} e {@link MLP}). */
	public static boolean isCnnFile(String path) {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			String first = r.readLine();
			return first != null && first.equals(MAGIC);
		} catch (IOException e) {
			return false;
		}
	}

	public void save(String path, int featureVersion, int featureGroups) throws IOException {
		try (BufferedWriter w = new BufferedWriter(new FileWriter(path))) {
			w.write(MAGIC);
			w.newLine();
			w.write("featureVersion " + featureVersion);
			w.newLine();
			w.write("featureGroups " + featureGroups);
			w.newLine();
			w.write("conv " + join(cfg.conv));
			w.newLine();
			w.write("reduce " + cfg.reduce);
			w.newLine();
			w.write("nFeat " + cfg.nFeat);
			w.newLine();
			w.write("dense " + join(cfg.dense));
			w.newLine();
			w.write("head " + join(cfg.head));
			w.newLine();
			w.write("params " + P.length);
			w.newLine();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < P.length; i++) {
				sb.append(Float.toString(P[i])); // round-trip esatto con Float.parseFloat
				sb.append(i % 8 == 7 ? '\n' : ' ');
			}
			w.write(sb.toString());
			w.newLine();
		}
	}

	public static Loaded load(String path) throws IOException {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			String first = r.readLine();
			if (first == null || !first.equals(MAGIC)) {
				throw new IOException("Formato file pesi non valido (atteso '" + MAGIC + "'): " + path);
			}
			try {
				int fv = Integer.parseInt(expect(r, "featureVersion").trim());
				int fg = Integer.parseInt(expect(r, "featureGroups").trim());
				int[] conv = parseList(expect(r, "conv").trim());
				int reduce = Integer.parseInt(expect(r, "reduce").trim());
				int nFeat = Integer.parseInt(expect(r, "nFeat").trim());
				int[] dense = parseList(expect(r, "dense").trim());
				int[] head = parseList(expect(r, "head").trim());
				int np = Integer.parseInt(expect(r, "params").trim());
				Config cfg = new Config(conv, reduce, nFeat, dense, head);
				float[] p = new float[np];
				int k = 0;
				String line;
				while (k < np && (line = r.readLine()) != null) {
					StringTokenizer tk = new StringTokenizer(line);
					while (tk.hasMoreTokens() && k < np) {
						p[k++] = Float.parseFloat(tk.nextToken());
					}
				}
				if (k != np) {
					throw new IOException("File pesi troncato: letti " + k + " parametri su " + np + " (" + path + ")");
				}
				return new Loaded(new ConvNet(cfg, p), fv, fg); // il costruttore verifica che np coincida con la forma
			} catch (IllegalArgumentException e) { // include NumberFormatException e la verifica di forma del costruttore
				throw new IOException("File pesi CNN non valido (" + path + "): " + e.getMessage(), e);
			}
		}
	}

	private static String expect(BufferedReader r, String key) throws IOException {
		String line = r.readLine();
		if (line == null || !line.startsWith(key + " ")) {
			throw new IOException("Atteso '" + key + "', trovato '" + line + "'");
		}
		return line.substring(key.length() + 1);
	}

	static String join(int[] a) {
		if (a.length == 0) {
			return "-";
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < a.length; i++) {
			if (i > 0) {
				sb.append(',');
			}
			sb.append(a[i]);
		}
		return sb.toString();
	}

	/** "8,8" -> {8,8}; "-" o "none" o "" -> {}. */
	public static int[] parseList(String s) {
		s = s.trim();
		if (s.isEmpty() || s.equals("-") || s.equalsIgnoreCase("none")) {
			return new int[0];
		}
		String[] t = s.split(",");
		int[] r = new int[t.length];
		for (int i = 0; i < t.length; i++) {
			r[i] = Integer.parseInt(t[i].trim());
		}
		return r;
	}
}
