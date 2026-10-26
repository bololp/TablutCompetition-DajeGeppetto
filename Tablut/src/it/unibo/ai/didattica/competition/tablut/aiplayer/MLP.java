package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

/**
 * Rete feedforward per la valutazione di posizioni di Tablut, riprogettata per
 * massimizzare la <b>forza per secondo di ricerca</b> (non l'accuratezza in
 * assoluto): ogni foglia del Negamax ne paga un forward pass.
 *
 * <h3>Struttura: tronco condiviso + head per ruolo</h3>
 * <pre>
 *   feature --[ tronco: T strati nascosti ]--+--[ head 0 (Bianco di turno) : H strati nascosti -> 1 ]
 *                                            +--[ head 1 (Nero di turno)   : H strati nascosti -> 1 ]
 * </pre>
 * Il "ruolo" che seleziona la head e' <b>chi deve muovere nella posizione
 * valutata</b> (una funzione della sola posizione, quindi coerente tra
 * giocatore, avversario e self-play). Un solo codice realizza le tre
 * architetture confrontate nel progetto:
 * <ul>
 * <li><b>A</b>: 1 head, ruolo come input ({@code heads=1, roleInput=true});</li>
 * <li><b>B</b>: tronco condiviso + 2 head ({@code trunk non vuoto, heads=2});</li>
 * <li><b>C</b>: due reti completamente separate ({@code trunk vuoto, heads=2}:
 * ogni head ha i propri strati e legge direttamente le feature).</li>
 * </ul>
 * Il tronco impara rappresentazioni utili a ENTRAMBI i lati (catene di Nero, vie di
 * fuga, materiale...) usando tutti i dati; le head imparano come combinarle in
 * modo diverso a seconda di chi muove (la stessa via di fuga aperta vale una
 * vittoria se muove il Bianco, e una minaccia bloccabile se muove il Nero).
 * Il gradiente di ogni posizione tocca il tronco e SOLO la head del suo ruolo:
 * le head sono addestrate separatamente su "attacco" e "difesa".
 *
 * <h3>Perche' e' veloce</h3>
 * <ul>
 * <li>Nessuna allocazione nel forward: tutto lo stato temporaneo vive in uno
 * {@link Scratch} per thread (la rete stessa e' immutabile durante l'inferenza,
 * quindi condivisibile tra thread).</li>
 * <li>Pesi {@code double[]} piatti, riga per riga; kernel "register-blocked" a 4
 * neuroni di uscita per volta (4 accumulatori indipendenti: ~2x piu' veloce del
 * ciclo naive, misurato; il JIT di HotSpot non vettorizza questi loop corti).</li>
 * <li>Attivazioni economiche (ReLU, hard-tanh, leaky ReLU): un {@code tanh} per
 * valutazione invece di uno per neurone ({@code Math.tanh} costa ~20 ns: la rete
 * a 36 neuroni della versione precedente ne spendeva ~700 solo li').</li>
 * <li>Solo l'uscita e' un {@code tanh} (output in (-1,1) = esito atteso, +1 vince il
 * Bianco, -1 il Nero), come richiede il TD(lambda) del trainer.</li>
 * </ul>
 *
 * <h3>Convenzioni</h3>
 * Tutti i parametri stanno in un unico {@code double[]}: per ogni strato prima la
 * matrice dei pesi {@code [nOut][nIn]} riga per riga, poi i bias. Cosi' i
 * gradienti, le eligibility trace del TD e le copie sono semplici vettori della
 * stessa lunghezza.
 */
public final class MLP {

	/** Attivazione degli strati nascosti. */
	public enum Act {
		RELU, HARDTANH, LEAKY, TANH
	}

	private static final double LEAK = 0.05;

	/** Descrizione dell'architettura (immutabile una volta costruita la rete). */
	public static final class Config {
		public final int nIn;
		/** Dimensioni degli strati nascosti del tronco condiviso (puo' essere vuoto). */
		public final int[] trunk;
		/** Dimensioni degli strati nascosti di OGNI head (puo' essere vuoto: l'head e' solo il neurone di uscita). */
		public final int[] head;
		/** 1 oppure 2 head. Con 2, la head e' scelta dal ruolo (0 = Bianco di turno, 1 = Nero di turno). */
		public final int heads;
		/** Se vero, il ruolo (+1/-1) viene aggiunto come feature in ingresso (architettura A). */
		public final boolean roleInput;
		public final Act act;

		public Config(int nIn, int[] trunk, int[] head, int heads, boolean roleInput, Act act) {
			if (heads != 1 && heads != 2) {
				throw new IllegalArgumentException("heads deve valere 1 o 2");
			}
			this.nIn = nIn;
			this.trunk = trunk.clone();
			this.head = head.clone();
			this.heads = heads;
			this.roleInput = roleInput;
			this.act = act;
		}

		/** Architettura B: tronco condiviso + 2 head. */
		public static Config sharedTrunk(int nIn, int[] trunk, int[] head, Act act) {
			return new Config(nIn, trunk, head, 2, false, act);
		}

		/** Architettura A: una sola head, ruolo come input. */
		public static Config singleHeadRoleInput(int nIn, int[] trunk, int[] head, Act act) {
			return new Config(nIn, trunk, head, 1, true, act);
		}

		/** Architettura C: due reti separate (nessun tronco condiviso). */
		public static Config separateNets(int nIn, int[] hidden, Act act) {
			return new Config(nIn, new int[0], hidden, 2, false, act);
		}

		public int effectiveInputs() {
			return nIn + (roleInput ? 1 : 0);
		}

		@Override
		public String toString() {
			String arch = heads == 1 ? "A(1 head+ruolo in input)" : (trunk.length == 0 ? "C(2 reti separate)" : "B(tronco+2 head)");
			return arch + " in=" + effectiveInputs() + " tronco=" + Arrays.toString(trunk) + " head="
					+ Arrays.toString(head) + " act=" + act;
		}
	}

	private final Config cfg;
	// strati: prima quelli del tronco, poi quelli di ogni head (hidden..., uscita)
	private final int[] lIn, lOut, lW, lB;
	/** path[h] = sequenza degli indici di strato percorsa da una posizione con head h. */
	private final int[][] path;
	private final double[] params;

	/** Stato temporaneo di un forward/backward. Un thread = uno Scratch. */
	public static final class Scratch {
		final double[][] a; // a[0] = input, a[i+1] = uscita del i-esimo strato del path
		final double[] delta1, delta2;
		int pathLen;
		int head;
		double out;

		Scratch(int[] widths, int maxWidth) {
			a = new double[widths.length][];
			for (int i = 0; i < widths.length; i++) {
				a[i] = new double[widths[i]];
			}
			delta1 = new double[maxWidth];
			delta2 = new double[maxWidth];
		}
	}

	public MLP(Config cfg, long seed) {
		this.cfg = cfg;
		int nTrunk = cfg.trunk.length;
		int nHeadLayers = cfg.head.length + 1;
		int total = nTrunk + cfg.heads * nHeadLayers;
		lIn = new int[total];
		lOut = new int[total];
		lW = new int[total];
		lB = new int[total];
		int offset = 0;
		int in = cfg.effectiveInputs();
		int id = 0;
		for (int i = 0; i < nTrunk; i++, id++) {
			offset = defineLayer(id, in, cfg.trunk[i], offset);
			in = cfg.trunk[i];
		}
		int trunkOut = in;
		path = new int[cfg.heads][nTrunk + nHeadLayers];
		for (int h = 0; h < cfg.heads; h++) {
			int hin = trunkOut;
			int p = 0;
			for (int i = 0; i < nTrunk; i++) {
				path[h][p++] = i;
			}
			for (int i = 0; i < cfg.head.length; i++, id++) {
				offset = defineLayer(id, hin, cfg.head[i], offset);
				hin = cfg.head[i];
				path[h][p++] = id;
			}
			offset = defineLayer(id, hin, 1, offset);
			path[h][p++] = id;
			id++;
		}
		params = new double[offset];
		initWeights(seed);
	}

	private MLP(Config cfg, double[] params) {
		this(cfg, 0L, params);
	}

	private MLP(Config cfg, long seed, double[] givenParams) {
		this.cfg = cfg;
		MLP shape = new MLP(cfg, seed);
		this.lIn = shape.lIn;
		this.lOut = shape.lOut;
		this.lW = shape.lW;
		this.lB = shape.lB;
		this.path = shape.path;
		if (givenParams.length != shape.params.length) {
			throw new IllegalArgumentException("numero di parametri incoerente: " + givenParams.length + " vs " + shape.params.length);
		}
		this.params = givenParams;
	}

	private int defineLayer(int id, int nIn, int nOut, int offset) {
		lIn[id] = nIn;
		lOut[id] = nOut;
		lW[id] = offset;
		lB[id] = offset + nIn * nOut;
		return lB[id] + nOut;
	}

	private void initWeights(long seed) {
		Random rnd = new Random(seed);
		for (int l = 0; l < lIn.length; l++) {
			double bound = Math.sqrt(6.0 / (lIn[l] + lOut[l]));
			if (lOut[l] == 1) {
				bound *= 0.5; // uscita piccola all'inizio: valori iniziali vicini a 0 = "non so ancora"
			}
			for (int k = 0; k < lIn[l] * lOut[l]; k++) {
				params[lW[l] + k] = (rnd.nextDouble() * 2 - 1) * bound;
			}
			// bias a 0
		}
	}

	public Config config() {
		return cfg;
	}

	/** Accesso diretto ai parametri (per ottimizzatori esterni e test). */
	public double[] params() {
		return params;
	}

	public int numParams() {
		return params.length;
	}

	/** Parametri effettivamente usati da UNA posizione (tronco + una head): e' il costo di inferenza. */
	public int paramsOnPath() {
		int n = 0;
		for (int l : path[0]) {
			n += lIn[l] * lOut[l] + lOut[l];
		}
		return n;
	}

	/**
	 * Tabella degli strati per {@link WeightConverter}: una riga {@code {offsetPesi, offsetBias, nIn, nOut, leggeIngresso}} per
	 * strato (pesi [out][in] row-major). {@code leggeIngresso} = 1 per gli strati che leggono direttamente il vettore di
	 * ingresso (il primo del tronco, oppure il primo di ciascuna head se non c'e' tronco).
	 */
	int[][] layerTable() {
		boolean[] first = new boolean[lIn.length];
		for (int[] p : path) {
			first[p[0]] = true;
		}
		int[][] t = new int[lIn.length][5];
		for (int l = 0; l < lIn.length; l++) {
			t[l][0] = lW[l];
			t[l][1] = lB[l];
			t[l][2] = lIn[l];
			t[l][3] = lOut[l];
			t[l][4] = first[l] ? 1 : 0;
		}
		return t;
	}

	/** Copia profonda indipendente (per i worker del trainer: pesi congelati). */
	public MLP copy() {
		return new MLP(cfg, params.clone());
	}

	public Scratch newScratch() {
		int pl = path[0].length;
		int[] widths = new int[pl + 1];
		widths[0] = cfg.effectiveInputs();
		int maxW = widths[0];
		for (int i = 0; i < pl; i++) {
			// larghezze massime tra le head (le head hanno la stessa forma)
			widths[i + 1] = lOut[path[0][i]];
			maxW = Math.max(maxW, widths[i + 1]);
		}
		return new Scratch(widths, maxW);
	}

	// ------------------------------------------------------------------ forward

	/**
	 * Valuta {@code x} con la head del ruolo {@code role} (0 = Bianco di turno,
	 * 1 = Nero di turno; ignorato se la rete ha una sola head e non usa il ruolo
	 * come input). Ritorna l'uscita in (-1,1). Thread-safe se ogni thread usa il
	 * proprio {@link Scratch}.
	 *
	 * @param x vettore di {@link #config()}.nIn feature (non viene modificato)
	 */
	public double forward(double[] x, int role, Scratch s) {
		final int h = cfg.heads == 2 ? role : 0;
		final int[] pth = path[h];
		s.head = h;
		s.pathLen = pth.length;
		double[] in = s.a[0];
		System.arraycopy(x, 0, in, 0, cfg.nIn);
		if (cfg.roleInput) {
			in[cfg.nIn] = role == 0 ? 1.0 : -1.0;
		}
		final int last = pth.length - 1;
		for (int p = 0; p < last; p++) {
			final int l = pth[p];
			dense(s.a[p], lIn[l], lOut[l], lW[l], lB[l], s.a[p + 1]);
			activate(s.a[p + 1], lOut[l]);
		}
		final int l = pth[last];
		final double[] prev = s.a[last];
		final double[] w = params;
		double z = w[lB[l]];
		final int wo = lW[l];
		for (int j = 0; j < lIn[l]; j++) {
			z += w[wo + j] * prev[j];
		}
		double v = Math.tanh(z);
		s.a[last + 1][0] = v;
		s.out = v;
		return v;
	}

	/** y = W x + b, 4 neuroni di uscita per volta (4 catene di addizioni indipendenti). */
	private void dense(double[] x, int nIn, int nOut, int wOff, int bOff, double[] y) {
		final double[] w = params;
		int i = 0;
		for (; i + 3 < nOut; i += 4) {
			double s0 = w[bOff + i], s1 = w[bOff + i + 1], s2 = w[bOff + i + 2], s3 = w[bOff + i + 3];
			final int o0 = wOff + i * nIn, o1 = o0 + nIn, o2 = o1 + nIn, o3 = o2 + nIn;
			for (int j = 0; j < nIn; j++) {
				final double xj = x[j];
				s0 += w[o0 + j] * xj;
				s1 += w[o1 + j] * xj;
				s2 += w[o2 + j] * xj;
				s3 += w[o3 + j] * xj;
			}
			y[i] = s0;
			y[i + 1] = s1;
			y[i + 2] = s2;
			y[i + 3] = s3;
		}
		for (; i < nOut; i++) {
			double s = w[bOff + i];
			final int o = wOff + i * nIn;
			for (int j = 0; j < nIn; j++) {
				s += w[o + j] * x[j];
			}
			y[i] = s;
		}
	}

	private void activate(double[] y, int n) {
		switch (cfg.act) {
		case RELU:
			for (int i = 0; i < n; i++) {
				if (y[i] < 0) {
					y[i] = 0;
				}
			}
			break;
		case HARDTANH:
			for (int i = 0; i < n; i++) {
				double v = y[i];
				y[i] = v < -1 ? -1 : (v > 1 ? 1 : v);
			}
			break;
		case LEAKY:
			for (int i = 0; i < n; i++) {
				if (y[i] < 0) {
					y[i] *= LEAK;
				}
			}
			break;
		default:
			for (int i = 0; i < n; i++) {
				y[i] = Math.tanh(y[i]);
			}
		}
	}

	/** Derivata dell'attivazione espressa tramite l'ATTIVAZIONE stessa (niente pre-attivazioni da memorizzare). */
	private double actDerivFromOutput(double a) {
		switch (cfg.act) {
		case RELU:
			return a > 0 ? 1.0 : 0.0;
		case HARDTANH:
			return (a > -1.0 && a < 1.0) ? 1.0 : 0.0;
		case LEAKY:
			return a > 0 ? 1.0 : LEAK;
		default:
			return 1.0 - a * a;
		}
	}

	// ----------------------------------------------------------------- backward

	/**
	 * Backpropagation dell'ultima {@link #forward} eseguita con questo
	 * {@code Scratch}: ACCUMULA in {@code grad} (stessa lunghezza di
	 * {@link #params()}) {@code outGrad * dV/dw} per i soli parametri su cui e'
	 * passata la posizione (tronco + head del suo ruolo).
	 * <p>
	 * Per il TD(lambda) si passa {@code outGrad = 1}: si vuole il gradiente del
	 * VALORE (dV/dw), che va aggiunto alla eligibility trace (gia' moltiplicata
	 * per lambda dal chiamante). Per un training supervisionato con perdita
	 * quadratica si passa {@code outGrad = V - target}.
	 */
	public void backward(Scratch s, double outGrad, double[] grad) {
		final int[] pth = path[s.head];
		final int last = pth.length - 1;
		double[] delta = s.delta1, next = s.delta2;
		delta[0] = outGrad * (1.0 - s.out * s.out); // tanh'(z) = 1 - v^2
		for (int p = last; p >= 0; p--) {
			final int l = pth[p];
			final int nIn = lIn[l], nOut = lOut[l];
			final double[] prev = s.a[p];
			final int wo = lW[l], bo = lB[l];
			for (int i = 0; i < nOut; i++) {
				final double d = delta[i];
				grad[bo + i] += d;
				final int o = wo + i * nIn;
				for (int j = 0; j < nIn; j++) {
					grad[o + j] += d * prev[j];
				}
			}
			if (p > 0) {
				for (int j = 0; j < nIn; j++) {
					double sum = 0;
					for (int i = 0; i < nOut; i++) {
						sum += params[wo + i * nIn + j] * delta[i];
					}
					next[j] = sum * actDerivFromOutput(prev[j]);
				}
				double[] tmp = delta;
				delta = next;
				next = tmp;
			}
		}
	}

	/** params += scale * g (aggiornamento TD: g = eligibility trace, scale = alpha * delta). */
	public void applyGradients(double[] g, double scale) {
		for (int i = 0; i < params.length; i++) {
			params[i] += scale * g[i];
		}
	}

	// ------------------------------------------------------------ salva / carica

	private static final String MAGIC = "TABLUT_MLP v2";

	/**
	 * Salva rete + metadati (versione delle feature): se in futuro cambiano le
	 * feature, {@link NeuralEvaluator#fromFile} rifiuta i vecchi pesi invece di
	 * leggerli silenziosamente con un altro significato.
	 */
	public void save(String path, int featureVersion, int featureGroups) throws IOException {
		try (BufferedWriter w = new BufferedWriter(new FileWriter(path))) {
			w.write(MAGIC);
			w.newLine();
			w.write("featureVersion " + featureVersion);
			w.newLine();
			w.write("featureGroups " + featureGroups);
			w.newLine();
			w.write("nIn " + cfg.nIn + " heads " + cfg.heads + " roleInput " + (cfg.roleInput ? 1 : 0) + " act " + cfg.act);
			w.newLine();
			w.write("trunk " + join(cfg.trunk));
			w.newLine();
			w.write("head " + join(cfg.head));
			w.newLine();
			w.write("params " + params.length);
			w.newLine();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < params.length; i++) {
				sb.append(Double.toString(params[i]));
				sb.append(i % 8 == 7 ? '\n' : ' ');
			}
			w.write(sb.toString());
			w.newLine();
		}
	}

	private static String join(int[] a) {
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

	private static int[] parseList(String s) {
		if (s.equals("-")) {
			return new int[0];
		}
		String[] t = s.split(",");
		int[] r = new int[t.length];
		for (int i = 0; i < t.length; i++) {
			r[i] = Integer.parseInt(t[i].trim());
		}
		return r;
	}

	/** Risultato del caricamento: rete + versione delle feature con cui e' stata allenata. */
	public static final class Loaded {
		public final MLP net;
		public final int featureVersion;
		public final int featureGroups;

		Loaded(MLP net, int featureVersion, int featureGroups) {
			this.net = net;
			this.featureVersion = featureVersion;
			this.featureGroups = featureGroups;
		}
	}

	public static Loaded load(String path) throws IOException {
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			String first = r.readLine();
			if (first != null && first.startsWith("sizes:")) {
				throw new IOException("file di pesi nel formato LEGACY (MLP 16-24-12-1 con feature v1): incompatibile con la "
						+ "rete attuale, va riallenata: " + path);
			}
			if (first == null || !first.equals(MAGIC)) {
				throw new IOException("Formato file pesi non valido (atteso '" + MAGIC + "'): " + path);
			}
			int featureVersion = Integer.parseInt(expect(r, "featureVersion").trim());
			int featureGroups = Integer.parseInt(expect(r, "featureGroups").trim());
			StringTokenizer st = new StringTokenizer(expect(r, "nIn"));
			int nIn = Integer.parseInt(st.nextToken());
			st.nextToken(); // heads
			int heads = Integer.parseInt(st.nextToken());
			st.nextToken(); // roleInput
			boolean roleInput = Integer.parseInt(st.nextToken()) == 1;
			st.nextToken(); // act
			Act act = Act.valueOf(st.nextToken());
			int[] trunk = parseList(expect(r, "trunk").trim());
			int[] head = parseList(expect(r, "head").trim());
			int np = Integer.parseInt(expect(r, "params").trim());
			Config cfg = new Config(nIn, trunk, head, heads, roleInput, act);
			double[] p = new double[np];
			int k = 0;
			String line;
			while ((line = r.readLine()) != null && k < np) {
				StringTokenizer tk = new StringTokenizer(line);
				while (tk.hasMoreTokens() && k < np) {
					p[k++] = Double.parseDouble(tk.nextToken());
				}
			}
			if (k != np) {
				throw new IOException("File pesi troncato: letti " + k + " parametri su " + np);
			}
			return new Loaded(new MLP(cfg, p), featureVersion, featureGroups);
		}
	}

	/** Legge una riga del tipo "chiave valori..." e ritorna la parte dopo la chiave. */
	private static String expect(BufferedReader r, String key) throws IOException {
		String line = r.readLine();
		if (line == null || !line.startsWith(key + " ")) {
			throw new IOException("Atteso '" + key + "', trovato '" + line + "'");
		}
		return line.substring(key.length() + 1);
	}
}
