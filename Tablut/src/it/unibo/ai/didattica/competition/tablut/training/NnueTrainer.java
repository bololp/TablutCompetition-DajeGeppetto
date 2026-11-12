package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Addestra una NNUE ({@link NnueNetwork}) su DATI DI GIOCO ({@link NnueDataset}): partite vere (log del server, {@code dati.gz}) e partite di
 * auto-gioco ({@link NnueSelfPlay}), con ESITO e/o VALORE DI RICERCA come etichette. La struttura della rete, la passata avanti/indietro, Adam e la
 * quantizzazione sono quelli di {@link NnueDistiller}.
 *
 * <h3>Bersaglio</h3>
 * Per ogni posizione, in [-1, 1] dal punto di vista del Bianco:
 * <pre>
 *   entrambe le etichette:  t = lambda * valore + (1 - lambda) * esito'
 *   solo il valore:         t = valore
 *   solo l'esito:           t = esito'
 *   nessuna:                la posizione non si usa
 *   esito' = esito * gamma^(semi-mosse alla fine)      (gamma = 1: nessuno sconto)
 * </pre>
 * Il valore di ricerca e' una stima locale, regolare, ma con l'errore del motore che l'ha prodotto; l'esito e' la verita' di quella partita ma e'
 * rumoroso (una partita decisa da un errore al ventesimo ply etichetta male le prime posizioni). Il mix e' il compromesso classico; con
 * {@code gamma < 1} le posizioni lontane dalla fine si fidano meno dell'esito. Perdita: errore quadratico fra {@code tanh(z)} e il bersaglio, pesato
 * per sorgente.
 *
 * <h3>Dati, validazione, simmetrie</h3>
 * I file di {@code data=} sono uniti (le chiavi di partita rinumerate) e divisi PER PARTITA in train e held-out ({@code valPct}). I file di
 * {@code val=} sono ESCLUSI dall'addestramento e usati solo per misurare: e' il modo di valutare su partite vere una rete addestrata altrove.
 * Con {@code augment} ogni campione e' visto a ogni epoca con una delle 8 simmetrie del quadrato scelta a caso (le regole sono invarianti, la rete no),
 * permutando gli indici delle feature. L'epoca che minimizza la perdita di validazione e' quella restituita (arresto anticipato dopo
 * {@code patience} epoche senza miglioramento). I pesi sono saturati ogni passo all'intervallo rappresentabile in {@code short} dopo la quantizzazione.
 * <p>
 * Con {@code init=} si riparte dai pesi di una rete esistente (stessa struttura): addestramento di rifinitura, tipicamente con {@code lr} piu' basso.
 * Tutto e' deterministico dato il seme.
 * <p>
 * Uso: {@code NnueTrainer data=A.gz[,B.gz] out=FILE.nnue [val=V.gz[,..]] [init=PESI.nnue] [hidden=64] [l2=16] [epochs=30] [batch=256] [lr=0.001]
 * [lrend=0.0001] [lambda=0.5] [extlambda=-1] [gamma=1] [augment=true] [valpct=5] [valsalt=1] [patience=8] [seed=1] [weights=0:1,1:1,2:1] [maxsamples=0]
 * [compare=X.nnue[,..]]}.
 */
public final class NnueTrainer {

	private NnueTrainer() {
	}

	public static final class Config {
		public int hidden = 64, l2 = 16;
		public int epochs = 30, batch = 256, patience = 8;
		public double lr = 1e-3, lrEnd = 1e-4;
		public double lambda = 0.5, gamma = 1.0;
		/**
		 * {@code lambda} con cui si giudica la validazione ESTERNA ({@code val=}); -1 = lo stesso dell'addestramento. Con 0 le partite vere si giudicano
		 * sul solo esito reale, qualunque sia il mix usato per addestrare (la verita' di riferimento, non una stima di ricerca).
		 */
		public double extLambda = -1;
		public boolean augment = true;
		public int valPct = 5;
		public long valSalt = 1, seed = 1;
		/** Peso di ciascuna sorgente (indice = {@link NnueDataset#source}); 0 esclude i campioni di quella sorgente dall'addestramento. */
		public double[] sourceWeight = defaultWeights();
		/** Se > 0, si addestra su un sottoinsieme casuale di questa dimensione. */
		public int maxSamples = 0;

		Config withLambda(double l) {
			final Config c = new Config();
			c.hidden = hidden;
			c.l2 = l2;
			c.epochs = epochs;
			c.batch = batch;
			c.patience = patience;
			c.lr = lr;
			c.lrEnd = lrEnd;
			c.lambda = l;
			c.extLambda = extLambda;
			c.gamma = gamma;
			c.augment = augment;
			c.valPct = valPct;
			c.valSalt = valSalt;
			c.seed = seed;
			c.sourceWeight = sourceWeight.clone();
			c.maxSamples = maxSamples;
			return c;
		}

		static double[] defaultWeights() {
			final double[] w = new double[16];
			Arrays.fill(w, 1.0);
			return w;
		}

		void validate() {
			if (hidden < 1 || hidden > NnueNetwork.MAX_HIDDEN || l2 < 1 || l2 > NnueNetwork.MAX_L2 || epochs < 1 || batch < 1 || patience < 1 || lr <= 0 || lrEnd <= 0
					|| lambda < 0 || lambda > 1 || extLambda > 1 || (extLambda < 0 && extLambda != -1) || gamma <= 0 || gamma > 1 || valPct < 0 || valPct > 90 || maxSamples < 0 || sourceWeight == null || sourceWeight.length != 16) {
				throw new IllegalArgumentException("configurazione non valida: hidden=" + hidden + " l2=" + l2 + " epochs=" + epochs + " batch=" + batch + " lr=" + lr + " lambda="
						+ lambda + " gamma=" + gamma + " valpct=" + valPct);
			}
			for (double w : sourceWeight) {
				if (w < 0 || Double.isNaN(w)) {
					throw new IllegalArgumentException("peso di sorgente non valido: " + w);
				}
			}
		}
	}

	// ====================================================================== bersaglio

	/** Bersaglio del campione {@code i} (vedi la documentazione della classe), oppure {@code NaN} se non ha alcuna etichetta. */
	public static double target(Config cfg, NnueDataset d, int i) {
		final boolean hv = d.valValid[i] == 1, ho = d.outValid[i] == 1;
		double o = 0;
		if (ho) {
			o = d.outcome[i];
			if (cfg.gamma < 1.0 && d.toEnd[i] >= 0) {
				o *= Math.pow(cfg.gamma, d.toEnd[i]);
			}
		}
		if (hv && ho) {
			return cfg.lambda * d.value[i] + (1.0 - cfg.lambda) * o;
		}
		if (hv) {
			return d.value[i];
		}
		return ho ? o : Double.NaN;
	}

	/** Indici dei campioni con un bersaglio e peso di sorgente > 0. */
	static int[] usable(Config cfg, NnueDataset d, int[] idx, boolean applyWeights) {
		final int[] out = new int[idx.length];
		int k = 0;
		for (int i : idx) {
			if (!Double.isNaN(target(cfg, d, i)) && (!applyWeights || cfg.sourceWeight[d.source[i]] > 0)) {
				out[k++] = i;
			}
		}
		return Arrays.copyOf(out, k);
	}

	// ====================================================================== addestramento

	public static final class Result {
		public final NnueDistiller.FloatNet net;
		/** Epoca (da 1) della rete restituita; 0 = nessuna epoca ha migliorato la validazione e la rete e' quella di partenza. */
		public final int bestEpoch, epochsRun;
		/** Campioni effettivamente usati per addestrare (con etichetta, peso di sorgente > 0, dopo {@code maxSamples}). */
		public final int trainSamples;
		public final double initialVal, bestVal, lastTrain;

		Result(NnueDistiller.FloatNet net, int bestEpoch, int epochsRun, int trainSamples, double initialVal, double bestVal, double lastTrain) {
			this.net = net;
			this.trainSamples = trainSamples;
			this.bestEpoch = bestEpoch;
			this.epochsRun = epochsRun;
			this.initialVal = initialVal;
			this.bestVal = bestVal;
			this.lastTrain = lastTrain;
		}
	}

	/** Limiti dopo i quali un peso non sta piu' in {@code short} una volta quantizzato; i bias interi sono limitati a un valore enorme rispetto a quelli sensati (O(1)). */
	private static final double FT_LIMIT = 32767.0 / NnueNetwork.QA - 0.01, W_LIMIT = 32767.0 / NnueNetwork.QW - 0.01, BIAS_LIMIT = 1000.0;

	/** Satura i pesi entro l'intervallo rappresentabile: cosi' la quantizzazione non perde nulla per overflow. */
	static void clampToQuantRange(NnueDistiller.FloatNet net) {
		final int H = net.hidden, L = net.l2;
		final int ftEnd = NnueNetwork.NUM_FEATURES * H + H;
		for (int i = 0; i < ftEnd; i++) {
			net.p[i] = Math.max(-FT_LIMIT, Math.min(FT_LIMIT, net.p[i]));
		}
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			final int w1 = net.offL1W(r), w2 = net.offL2W(r);
			for (int i = 0; i < L * H; i++) {
				net.p[w1 + i] = Math.max(-W_LIMIT, Math.min(W_LIMIT, net.p[w1 + i]));
			}
			for (int i = 0; i < L; i++) {
				net.p[w2 + i] = Math.max(-W_LIMIT, Math.min(W_LIMIT, net.p[w2 + i]));
				final int b1 = net.offL1B(r) + i;
				net.p[b1] = Math.max(-BIAS_LIMIT, Math.min(BIAS_LIMIT, net.p[b1]));
			}
			final int b2 = net.offL2B(r);
			net.p[b2] = Math.max(-BIAS_LIMIT, Math.min(BIAS_LIMIT, net.p[b2]));
		}
	}

	/** Un passo di Adam seguito dalla saturazione dei pesi all'intervallo quantizzabile (un solo punto: i test ne verificano l'effetto). */
	static void optimizerStep(NnueDistiller.AdamD adam, NnueDistiller.FloatNet net, double[] grad, double lr, double gradScale) {
		adam.step(net.p, grad, lr, gradScale);
		clampToQuantRange(net);
	}

	/** Perdita media (errore quadratico verso il bersaglio, senza pesi ne' simmetrie) di {@code net} sui campioni {@code idx} (gia' filtrati). */
	public static double meanLoss(NnueDistiller.FloatNet net, Config cfg, NnueDataset d, int[] idx) {
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(net.hidden, net.l2);
		double s = 0;
		for (int i : idx) {
			final double y = Math.tanh(NnueDistiller.forward(net, d.boards, 81 * i, d.role[i], null, ws));
			final double e = y - target(cfg, d, i);
			s += e * e;
		}
		return idx.length == 0 ? 0 : s / idx.length;
	}

	/**
	 * Addestra.
	 *
	 * @param start rete di partenza (si modifica), o {@code null} per un'inizializzazione casuale con {@code cfg.seed}
	 * @param trainIdx campioni di addestramento (anche senza etichetta: si filtrano)
	 * @param valIdx campioni di validazione di {@code d}, e/o {@code extraVal}/{@code extraIdx} (altro dataset, mai usato per addestrare)
	 */
	public static Result train(NnueDataset d, int[] trainIdx, int[] valIdx, NnueDataset extraVal, int[] extraIdx, Config cfg, NnueDistiller.FloatNet start, PrintStream log) {
		cfg.validate();
		final Random rnd = new Random(cfg.seed);
		final NnueDistiller.FloatNet net;
		if (start != null) {
			if (start.hidden != cfg.hidden || start.l2 != cfg.l2) {
				throw new IllegalArgumentException("la rete di partenza e' " + start.hidden + "/" + start.l2 + ", la configurazione " + cfg.hidden + "/" + cfg.l2);
			}
			net = start;
		} else {
			net = new NnueDistiller.FloatNet(cfg.hidden, cfg.l2);
			net.init(rnd);
		}
		int[] order = usable(cfg, d, trainIdx, true);
		if (cfg.maxSamples > 0 && order.length > cfg.maxSamples) {
			for (int i = order.length - 1; i > 0; i--) {
				final int j = rnd.nextInt(i + 1);
				final int t = order[i];
				order[i] = order[j];
				order[j] = t;
			}
			order = Arrays.copyOf(order, cfg.maxSamples);
		}
		final int[] val = usable(cfg, d, valIdx, false);
		final int[] xval = extraVal == null ? new int[0] : usable(cfg, extraVal, extraIdx, false);
		if (order.length == 0) {
			throw new IllegalArgumentException("nessun campione di addestramento con un'etichetta");
		}
		if (val.length == 0 && xval.length == 0) {
			throw new IllegalArgumentException("nessun campione di validazione: servono valpct > 0 oppure val=");
		}
		final boolean useExtra = xval.length > 0; // la validazione su dati esterni (partite vere) ha la precedenza per scegliere l'epoca
		final Config xcfg = cfg.extLambda >= 0 ? cfg.withLambda(cfg.extLambda) : cfg;
		final NnueDistiller.Ws ws = new NnueDistiller.Ws(cfg.hidden, cfg.l2);
		final double[] g = new double[net.numParams()];
		final NnueDistiller.AdamD adam = new NnueDistiller.AdamD(net.numParams());
		final double initial = useExtra ? meanLoss(net, xcfg, extraVal, xval) : meanLoss(net, cfg, d, val);
		// la rete di PARTENZA e' la prima candidata (epoca 0): se nessuna epoca migliora la validazione si restituisce invariata, mai una peggiore
		if (log != null) {
			log.printf("avvio: %d campioni di addestramento, %d di held-out%s, %d parametri; perdita di validazione iniziale %.5f%n", order.length, val.length,
					useExtra ? " + " + xval.length + " esterni (scelgono l'epoca)" : "", net.numParams(), initial);
		}
		final long totalSteps = (long) cfg.epochs * ((order.length + cfg.batch - 1) / cfg.batch);
		long step = 0;
		double bestVal = initial, lastTrain = 0;
		int bestEpoch = 0, run = 0;
		double[] best = net.p.clone();
		for (int e = 0; e < cfg.epochs; e++) {
			for (int i = order.length - 1; i > 0; i--) {
				final int j = rnd.nextInt(i + 1);
				final int t = order[i];
				order[i] = order[j];
				order[j] = t;
			}
			double sum = 0, wsum = 0;
			for (int b = 0; b < order.length; b += cfg.batch) {
				final int end = Math.min(order.length, b + cfg.batch);
				Arrays.fill(g, 0.0);
				for (int k = b; k < end; k++) {
					final int i = order[k];
					final double w = cfg.sourceWeight[d.source[i]];
					final int[] map = BoardSymmetry.MAP[cfg.augment ? rnd.nextInt(BoardSymmetry.COUNT) : 0];
					final double y = Math.tanh(NnueDistiller.forward(net, d.boards, 81 * i, d.role[i], map, ws));
					final double err = y - target(cfg, d, i);
					sum += w * err * err;
					wsum += w;
					NnueDistiller.backward(net, d.role[i], 2 * w * err * (1 - y * y), ws, g);
				}
				final double frac = totalSteps <= 1 ? 1.0 : (double) step / (totalSteps - 1);
				optimizerStep(adam, net, g, cfg.lr + (cfg.lrEnd - cfg.lr) * frac, 1.0 / (end - b));
				step++;
			}
			lastTrain = sum / Math.max(1e-12, wsum);
			final double v = useExtra ? meanLoss(net, xcfg, extraVal, xval) : meanLoss(net, cfg, d, val);
			run = e + 1;
			final boolean improved = v < bestVal;
			if (improved) {
				bestVal = v;
				bestEpoch = e + 1;
				best = net.p.clone();
			}
			if (log != null && (e == 0 || (e + 1) % Math.max(1, cfg.epochs / 10) == 0 || e == cfg.epochs - 1 || improved && cfg.epochs <= 12)) {
				final double hv = val.length > 0 ? meanLoss(net, cfg, d, val) : 0;
				log.printf("epoca %2d/%d: addestramento %.5f, held-out %.5f%s%s%n", e + 1, cfg.epochs, lastTrain, hv, useExtra ? String.format(", esterno %.5f", v) : "",
						improved ? " *" : "");
			}
			if (e + 1 - bestEpoch >= cfg.patience) {
				if (log != null) {
					log.println("arresto anticipato: nessun miglioramento da " + cfg.patience + " epoche (migliore: " + bestEpoch + ")");
				}
				break;
			}
		}
		System.arraycopy(best, 0, net.p, 0, best.length);
		return new Result(net, bestEpoch, run, order.length, initial, bestVal, lastTrain);
	}

	// ====================================================================== metriche sulla rete QUANTIZZATA

	public static final class Metrics {
		public int nTarget, nOutcome, nDecided, nValue;
		public double mseTarget, mseOutcome, zeroMseOutcome, outcomeAccuracy, outcomeCorr, mseValue, valueCorr;

		@Override
		public String toString() {
			return String.format("bersaglio: MSE %.4f (n=%d) | esito: MSE %.4f (predire 0: %.4f), segno giusto %.1f%% su %d decisi, correlazione %.3f (n=%d) | valore di ricerca: MSE %.4f, correlazione %.3f (n=%d)",
					mseTarget, nTarget, mseOutcome, zeroMseOutcome, 100 * outcomeAccuracy, nDecided, outcomeCorr, nOutcome, mseValue, valueCorr, nValue);
		}
	}

	private static final class Corr {
		double sx, sy, sxx, syy, sxy;
		int n;

		void add(double x, double y) {
			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
			n++;
		}

		double value() {
			if (n < 2) {
				return 0;
			}
			final double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n);
			return vx > 1e-12 && vy > 1e-12 ? (sxy / n - (sx / n) * (sy / n)) / Math.sqrt(vx * vy) : 0;
		}
	}

	/** Misura la rete quantizzata {@code q} sui campioni {@code idx} di {@code d} (tutti, anche senza etichetta: ciascuna misura usa solo quelli che la hanno). */
	public static Metrics evaluate(NnueNetwork q, Config cfg, NnueDataset d, int[] idx) {
		final NnueEvaluator ev = new NnueEvaluator(q);
		ev.outputScale = 1.0;
		final FastState fs = new FastState(); // l'NnueEvaluator legge solo board e turn
		final Metrics m = new Metrics();
		double seT = 0, seO = 0, se0 = 0, seV = 0;
		int accSign = 0;
		final Corr co = new Corr(), cv = new Corr();
		for (int i : idx) {
			System.arraycopy(d.boards, 81 * i, fs.board, 0, 81);
			fs.turn = d.role[i] == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
			final double y = ev.evaluate(fs);
			final double t = target(cfg, d, i);
			if (!Double.isNaN(t)) {
				seT += (y - t) * (y - t);
				m.nTarget++;
			}
			if (d.outValid[i] == 1) {
				final double o = d.outcome[i];
				seO += (y - o) * (y - o);
				se0 += o * o;
				m.nOutcome++;
				co.add(y, o);
				if (o != 0) {
					m.nDecided++;
					if ((y > 0) == (o > 0)) {
						accSign++;
					}
				}
			}
			if (d.valValid[i] == 1) {
				seV += (y - d.value[i]) * (y - d.value[i]);
				m.nValue++;
				cv.add(y, d.value[i]);
			}
		}
		m.mseTarget = m.nTarget == 0 ? 0 : seT / m.nTarget;
		m.mseOutcome = m.nOutcome == 0 ? 0 : seO / m.nOutcome;
		m.zeroMseOutcome = m.nOutcome == 0 ? 0 : se0 / m.nOutcome;
		m.outcomeAccuracy = m.nDecided == 0 ? 0 : accSign / (double) m.nDecided;
		m.outcomeCorr = co.value();
		m.mseValue = m.nValue == 0 ? 0 : seV / m.nValue;
		m.valueCorr = cv.value();
		return m;
	}

	// ====================================================================== main

	private static List<NnueDataset> loadAll(String csv) throws IOException {
		final List<NnueDataset> l = new ArrayList<>();
		for (String p : csv.split(",")) {
			if (!p.trim().isEmpty()) {
				l.add(NnueDataset.load(p.trim()));
			}
		}
		return l;
	}

	private static int[] range(int n) {
		final int[] a = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = i;
		}
		return a;
	}

	public static void main(String[] args) throws IOException {
		final Config cfg = new Config();
		String data = null, val = null, out = null, init = null, compare = null;
		boolean hiddenGiven = false;
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + a);
				System.exit(2);
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "data": data = v; break;
				case "val": val = v; break;
				case "out": out = v; break;
				case "init": init = v; break;
				case "compare": compare = v; break;
				case "hidden": cfg.hidden = Integer.parseInt(v); hiddenGiven = true; break;
				case "l2": cfg.l2 = Integer.parseInt(v); hiddenGiven = true; break;
				case "epochs": cfg.epochs = Integer.parseInt(v); break;
				case "batch": cfg.batch = Integer.parseInt(v); break;
				case "lr": cfg.lr = Double.parseDouble(v); break;
				case "lrend": cfg.lrEnd = Double.parseDouble(v); break;
				case "lambda": cfg.lambda = Double.parseDouble(v); break;
				case "gamma": cfg.gamma = Double.parseDouble(v); break;
				case "extlambda": cfg.extLambda = Double.parseDouble(v); break;
				case "augment": cfg.augment = Boolean.parseBoolean(v); break;
				case "valpct": cfg.valPct = Integer.parseInt(v); break;
				case "valsalt": cfg.valSalt = Long.parseLong(v); break;
				case "patience": cfg.patience = Integer.parseInt(v); break;
				case "seed": cfg.seed = Long.parseLong(v); break;
				case "maxsamples": cfg.maxSamples = Integer.parseInt(v); break;
				case "weights":
					for (String kv : v.split(",")) {
						final String[] p = kv.split(":");
						final int s = Integer.parseInt(p[0].trim());
						if (p.length != 2 || s < 0 || s > 15) {
							throw new NumberFormatException(kv);
						}
						cfg.sourceWeight[s] = Double.parseDouble(p[1].trim());
					}
					break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (NumberFormatException e) {
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (data == null || out == null) {
			System.out.println("Uso: NnueTrainer data=A.gz[,B.gz] out=FILE.nnue [val=V.gz[,..]] [init=PESI.nnue] [hidden=64] [l2=16] [epochs=30] [batch=256] [lr=0.001] [lrend=0.0001]"
					+ " [lambda=0.5] [extlambda=-1] [gamma=1] [augment=true] [valpct=5] [valsalt=1] [patience=8] [seed=1] [weights=0:1,1:1,2:1] [maxsamples=0] [compare=X.nnue[,..]]");
			System.exit(2);
		}
		try {
			NnueDistiller.FloatNet start = null;
			if (init != null) {
				final NnueNetwork q = NnueNetwork.load(init);
				if (hiddenGiven && (cfg.hidden != q.hidden() || cfg.l2 != q.l2())) {
					throw new IllegalArgumentException("init=" + init + " e' " + q.hidden() + "/" + q.l2() + ", ma hidden/l2 chiedono " + cfg.hidden + "/" + cfg.l2);
				}
				cfg.hidden = q.hidden();
				cfg.l2 = q.l2();
				start = NnueDistiller.FloatNet.fromQuantized(q);
			}
			cfg.validate();
			final NnueDataset d = NnueDataset.merge(loadAll(data));
			System.out.println("addestramento: " + d.describe());
			NnueDataset ext = null;
			if (val != null) {
				ext = NnueDataset.merge(loadAll(val));
				System.out.println("validazione esterna: " + ext.describe());
			}
			final int[][] sp = d.splitByGame(cfg.valSalt, cfg.valPct);
			final long t0 = System.currentTimeMillis();
			final Result r = train(d, sp[0], sp[1], ext, ext == null ? null : range(ext.n), cfg, start, System.out);
			System.out.printf("addestramento: %.1f s, %s (%d epoche eseguite), perdita di validazione %.5f -> %.5f%n", (System.currentTimeMillis() - t0) / 1000.0,
					r.bestEpoch == 0 ? "NESSUN MIGLIORAMENTO: rete invariata" : "migliore epoca " + r.bestEpoch, r.epochsRun, r.initialVal, r.bestVal);
			final NnueDistiller.Quantized q = NnueDistiller.quantize(r.net);
			if (q.clipped > 0) {
				System.out.println("ATTENZIONE: " + q.clipped + " pesi saturati nella quantizzazione: la rete scritta differisce da quella addestrata.");
			}
			new NnueEvaluator(q.net).save(out);
			System.out.println("scritto " + out + " (" + q.net.describe() + ")");
			report("rete addestrata", q.net, cfg, d, sp[1], ext);
			if (compare != null) {
				for (String c : compare.split(",")) {
					if (!c.trim().isEmpty()) {
						report("confronto " + c.trim(), NnueNetwork.load(c.trim()), cfg, d, sp[1], ext);
					}
				}
			}
		} catch (IllegalArgumentException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(2);
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}

	private static void report(String name, NnueNetwork net, Config cfg, NnueDataset d, int[] heldOut, NnueDataset ext) {
		if (heldOut.length > 0) {
			System.out.println(name + " | held-out (" + heldOut.length + " posizioni): " + evaluate(net, cfg, d, heldOut));
		}
		if (ext != null) {
			System.out.println(name + " | esterno (" + ext.n + " posizioni): " + evaluate(net, cfg.extLambda >= 0 ? cfg.withLambda(cfg.extLambda) : cfg, ext, range(ext.n)));
		}
	}
}
