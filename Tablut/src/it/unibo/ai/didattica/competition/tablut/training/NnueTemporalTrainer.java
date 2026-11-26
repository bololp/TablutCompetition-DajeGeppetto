package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalHistory;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalNnueNetwork;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;

/**
 * Addestra una {@link TemporalNnueNetwork}: una NNUE che oltre alla posizione vede le ultime {@code K} mosse ({@link TemporalHistory}), con
 * bersagli che possono guardare in avanti lungo la partita ({@link TemporalData#lambdaReturn}). Riusa il formato dati, la validazione per partita, Adam, la
 * saturazione per la quantizzazione e la garanzia di non regressione di {@link NnueTrainer}.
 *
 * <h3>Due leve temporali, indipendenti (si possono accendere una alla volta, quindi misurare una alla volta)</h3>
 * <ol>
 * <li><b>Storia in ingresso</b> ({@code hist=K}, K &gt;= 1): la rete vede le ultime K mosse. Utile per cio' che la sola posizione esprime male con
 * una rete piccola (quale pedina ha appena mosso, quindi cosa minaccia; iniziativa; catture appena avvenute). NON porta informazione nuova sul valore
 * della posizione in un gioco a informazione perfetta (Tablut e' markoviano, salvo le regole di ripetizione): e' un aiuto di capacita', e va misurato.</li>
 * <li><b>Bersagli multi-mossa</b> ({@code tdlambda=mu}, 0..1): il bersaglio di una posizione include i valori di ricerca delle posizioni SUCCESSIVE della
 * partita ({@link TemporalData#lambdaReturn}). Questa e' l'informazione davvero nuova: una tattica che si compie in 3 mosse e che la ricerca a
 * profondita' D non vedeva dalla posizione di partenza e' visibile alle ricerche delle posizioni successive. {@code tdlambda=1} (default) = comportamento di
 * {@link NnueTrainer} (esito della partita), {@code 0} = un solo passo.</li>
 * </ol>
 *
 * <h3>History dropout</h3>
 * Con probabilita' {@code histdrop} (default 0.5) la finestra di un campione e' TRONCATA ai soli {@code j} ultimi record, {@code j} uniforme in {@code 0..avail}.
 * E' il caso reale: dentro la ricerca i nodi vicini alla radice hanno storia (percorso + partita), ma MCTS, la valutazione stateless e le arene senza storia la
 * hanno vuota; senza dropout la rete imparerebbe a fidarsi di una storia che spesso non c'e'. Inoltre, in un'auto-partita la storia contiene mosse scelte dalla ricerca (buone)
 * mentre i foglie della ricerca hanno storie arbitrarie: il dropout riduce il rischio che la rete usi "e' stata appena giocata questa mossa" come prova di qualita'.
 * La validazione misura SEMPRE due condizioni: finestra completa e finestra vuota; l'epoca migliore e' quella con la media piu' bassa.
 *
 * <h3>Partenza</h3>
 * {@code init=} puo' essere una NNUE di sola posizione ({@code .nnue}: il blocco temporale parte a ZERO, quindi l'epoca 0 e' esattamente quella rete e, per la
 * garanzia di non regressione, si restituisce invariata se nessuna epoca la migliora), una rete temporale ({@code TABLUT_TNNUE}), oppure nulla (pesi casuali).
 * {@code trainbase=false} congela la parte di posizione e addestra solo il blocco temporale ("cosa aggiunge la storia" in forma pura); {@code usehist=false} addestra la STESSA rete
 * senza mai mostrarle la storia (confronto "solo posizione" a parita' di dati, bersagli e iperparametri). {@code val=} (dati esterni, per es. etichette di ricerca piu' profonde) di
 * default misura soltanto; con {@code extselect=true} sceglie anche l'epoca.
 * <p>
 * Uso: {@code NnueTemporalTrainer data=A.gz[,B.gz] out=FILE.tnnue [init=PESI.nnue|.tnnue] [val=V.gz] [hist=4] [hidden=64] [l2=16] [epochs=10] [batch=256] [lr=0.001]
 * [lrend=0.0001] [lambda=0.5] [tdlambda=1] [gamma=1] [histdrop=0.5] [augment=true] [valpct=5] [valsalt=1] [patience=4] [seed=1] [maxgames=0] [trainbase=true]
 * [weights=0:1,1:1] [extlambda=-1] [compare=A.nnue,B.nnue]}.
 */
public final class NnueTemporalTrainer {

	private NnueTemporalTrainer() {
	}

	public static final class Config {
		public int hist = 4, hidden = 64, l2 = 16;
		public int epochs = 10, batch = 256, patience = 4;
		public double lr = 1e-3, lrEnd = 1e-4;
		public double lambda = 0.5, gamma = 1.0, tdLambda = 1.0, extLambda = -1;
		public double histDrop = 0.5;
		public boolean augment = true, trainBase = true;
		/** Falso: la rete NON vede mai la storia (finestra sempre vuota): e' il confronto "solo posizione" con gli stessi dati, bersagli e iperparametri. */
		public boolean useHist = true;
		/** Vero: il dataset esterno {@code val=} sceglie anche l'epoca (come {@link NnueTrainer}); falso (default): serve solo a misurare, l'epoca la sceglie l'held-out interno. */
		public boolean extSelect = false;
		public int valPct = 5, maxGames = 0;
		public long valSalt = 1, seed = 1;
		public double[] sourceWeight = NnueTrainer.Config.defaultWeights();

		void validate() {
			if (hist < 1 || hist > TemporalHistory.MAX_AGES || hidden < 1 || hidden > NnueNetwork.MAX_HIDDEN || l2 < 1 || l2 > NnueNetwork.MAX_L2 || epochs < 1 || batch < 1
					|| patience < 1 || lr <= 0 || lrEnd <= 0 || lambda < 0 || lambda > 1 || gamma <= 0 || gamma > 1 || tdLambda < 0 || tdLambda > 1 || histDrop < 0 || histDrop > 1
					|| extLambda > 1 || (extLambda < 0 && extLambda != -1) || valPct < 0 || valPct > 90 || maxGames < 0 || sourceWeight == null || sourceWeight.length != 16) {
				throw new IllegalArgumentException("configurazione non valida: hist=" + hist + " hidden=" + hidden + " l2=" + l2 + " epochs=" + epochs + " batch=" + batch + " lr=" + lr
						+ " lambda=" + lambda + " tdlambda=" + tdLambda + " histdrop=" + histDrop + " valpct=" + valPct);
			}
			for (double w : sourceWeight) {
				if (w < 0 || Double.isNaN(w)) {
					throw new IllegalArgumentException("peso di sorgente non valido: " + w);
				}
			}
		}

		/** Configurazione di {@link NnueTrainer} equivalente per il solo bersaglio (lambda, gamma, extLambda). */
		NnueTrainer.Config targetConfig(boolean external) {
			final NnueTrainer.Config c = new NnueTrainer.Config();
			c.lambda = external && extLambda >= 0 ? extLambda : lambda;
			c.gamma = gamma;
			return c;
		}
	}

	// ====================================================================== bersagli

	/** Bersaglio di ogni campione di {@code d} (NaN = nessuna etichetta). Con {@code tdLambda >= 1} e' ESATTAMENTE quello di {@link NnueTrainer#target}. */
	public static float[] targets(Config cfg, NnueDataset d, boolean external) {
		if (external || cfg.tdLambda >= 1.0) {
			final NnueTrainer.Config tc = cfg.targetConfig(external);
			final float[] t = new float[d.n];
			for (int i = 0; i < d.n; i++) {
				t[i] = (float) NnueTrainer.target(tc, d, i);
			}
			return t;
		}
		return TemporalData.lambdaReturn(d, cfg.lambda, cfg.tdLambda);
	}

	/**
	 * Bersaglio "classico" ({@code lambda * valore + (1 - lambda) * esito}, come {@link NnueTrainer}) usato SEMPRE nei rapporti, qualunque bersaglio si sia usato per
	 * addestrare: e' il metro comune per confrontare configurazioni diverse. (Le colonne "esito" e "valore di ricerca" dei rapporti non dipendono dal bersaglio.)
	 */
	public static float[] classicTargets(Config cfg, NnueDataset d) {
		final NnueTrainer.Config c = new NnueTrainer.Config();
		c.lambda = cfg.lambda;
		c.gamma = cfg.gamma;
		final float[] t = new float[d.n];
		for (int i = 0; i < d.n; i++) {
			t[i] = (float) NnueTrainer.target(c, d, i);
		}
		return t;
	}

	static int[] usable(Config cfg, NnueDataset d, float[] tgt, int[] idx, boolean applyWeights) {
		int cnt = 0;
		for (int i : idx) {
			if (!Float.isNaN(tgt[i]) && (!applyWeights || cfg.sourceWeight[d.source(i)] > 0)) {
				cnt++;
			}
		}
		final int[] out = new int[cnt];
		int k = 0;
		for (int i : idx) {
			if (!Float.isNaN(tgt[i]) && (!applyWeights || cfg.sourceWeight[d.source(i)] > 0)) {
				out[k++] = i;
			}
		}
		return out;
	}

	// ====================================================================== addestramento

	public static final class Result {
		public final TemporalFloatNet net;
		public final int bestEpoch, epochsRun, trainSamples;
		/** Perdita di validazione (media di finestra completa e vuota) all'inizio e alla migliore epoca. */
		public final double initialVal, bestVal, lastTrain;

		Result(TemporalFloatNet net, int bestEpoch, int epochsRun, int trainSamples, double initialVal, double bestVal, double lastTrain) {
			this.net = net;
			this.bestEpoch = bestEpoch;
			this.epochsRun = epochsRun;
			this.trainSamples = trainSamples;
			this.initialVal = initialVal;
			this.bestVal = bestVal;
			this.lastTrain = lastTrain;
		}
	}

	/** Perdita media (errore quadratico di tanh(z) dal bersaglio) di {@code net} su {@code idx}; {@code useHist}: finestra completa o vuota. */
	public static double meanLoss(TemporalFloatNet net, TemporalData td, float[] tgt, int[] idx, boolean useHist) {
		final TemporalFloatNet.Ws ws = new TemporalFloatNet.Ws(net);
		final NnueDataset d = td.d;
		double s = 0;
		for (int i : idx) {
			final double y = Math.tanh(TemporalFloatNet.forward(net, d.boardArray(i), d.boardOffset(i), d.role(i), null, td.rec, i * td.k, useHist ? td.avail[i] : 0, ws));
			final double e = y - tgt[i];
			s += e * e;
		}
		return idx.length == 0 ? 0 : s / idx.length;
	}

	private static double score(TemporalFloatNet net, TemporalData td, float[] tgt, int[] idx, boolean allowHist) {
		return 0.5 * (meanLoss(net, td, tgt, idx, allowHist) + meanLoss(net, td, tgt, idx, false));
	}

	/**
	 * Addestra {@code net} (modificata sul posto, e' la rete di partenza: epoca 0).
	 *
	 * @param tgt    bersagli di {@code td.d} ({@link #targets})
	 * @param extra  campioni di validazione ESTERNI (altro dataset, mai usato per addestrare), o {@code null}; se presenti scelgono l'epoca
	 */
	public static Result train(TemporalData td, float[] tgt, int[] trainIdx, int[] valIdx, TemporalData extra, float[] extraTgt, int[] extraIdx, Config cfg, TemporalFloatNet net,
			PrintStream log) {
		cfg.validate();
		if (net.hist != td.k || net.hist != cfg.hist || net.hidden != cfg.hidden || net.l2 != cfg.l2) {
			throw new IllegalArgumentException("rete " + net.hidden + "/" + net.l2 + "/K" + net.hist + " incoerente con i dati K" + td.k + " e la configurazione " + cfg.hidden + "/" + cfg.l2 + "/K" + cfg.hist);
		}
		final NnueDataset d = td.d;
		final Random rnd = new Random(cfg.seed);
		int[] order = usable(cfg, d, tgt, trainIdx, true);
		final int[] val = usable(cfg, d, tgt, valIdx, false);
		final int[] xval = extra == null ? new int[0] : usable(cfg, extra.d, extraTgt, extraIdx, false);
		if (order.length == 0) {
			throw new IllegalArgumentException("nessun campione di addestramento con un'etichetta");
		}
		if (val.length == 0 && !(xval.length > 0 && cfg.extSelect)) {
			throw new IllegalArgumentException("nessun campione di validazione: servono valpct > 0 oppure val=");
		}
		final boolean useExtra = xval.length > 0 && cfg.extSelect;
		final TemporalFloatNet.Ws ws = new TemporalFloatNet.Ws(net);
		final double[] g = new double[net.numParams()];
		final NnueDistiller.AdamD adam = new NnueDistiller.AdamD(net.numParams());
		final int tStart = net.temporalStart(), tEnd = net.temporalEnd();
		final double initial = useExtra ? score(net, extra, extraTgt, xval, cfg.useHist) : score(net, td, tgt, val, cfg.useHist);
		if (log != null) {
			log.printf("avvio: %d campioni di addestramento, %d di held-out%s, K=%d, %d parametri (%d temporali); perdita di validazione iniziale %.5f%n", order.length, val.length,
					useExtra ? " + " + xval.length + " esterni (scelgono l'epoca)" : "", net.hist, net.numParams(), tEnd - tStart, initial);
		}
		final long totalSteps = (long) cfg.epochs * (((long) order.length + cfg.batch - 1) / cfg.batch);
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
			for (int b = 0, end; b < order.length; b = end) {
				end = (int) Math.min((long) order.length, (long) b + cfg.batch);
				Arrays.fill(g, 0.0);
				for (int k = b; k < end; k++) {
					final int i = order[k];
					final byte role = d.role(i);
					final double w = cfg.sourceWeight[d.source(i)];
					final int[] map = BoardSymmetry.MAP[cfg.augment ? rnd.nextInt(BoardSymmetry.COUNT) : 0];
					final int avail = cfg.useHist ? td.avail[i] : 0;
					final int len = rnd.nextDouble() < cfg.histDrop ? rnd.nextInt(avail + 1) : avail;
					final double y = Math.tanh(TemporalFloatNet.forward(net, d.boardArray(i), d.boardOffset(i), role, map, td.rec, i * td.k, len, ws));
					final double err = y - tgt[i];
					sum += w * err * err;
					wsum += w;
					TemporalFloatNet.backward(net, role, 2 * w * err * (1 - y * y), ws, g);
				}
				if (!cfg.trainBase) {
					Arrays.fill(g, 0, tStart, 0.0);
					Arrays.fill(g, tEnd, g.length, 0.0);
				}
				final double frac = totalSteps <= 1 ? 1.0 : (double) step / (totalSteps - 1);
				adam.step(net.p, g, cfg.lr + (cfg.lrEnd - cfg.lr) * frac, 1.0 / (end - b));
				net.clampToQuantRange();
				step++;
			}
			lastTrain = sum / Math.max(1e-12, wsum);
			final double v = useExtra ? score(net, extra, extraTgt, xval, cfg.useHist) : score(net, td, tgt, val, cfg.useHist);
			run = e + 1;
			final boolean improved = v < bestVal;
			if (improved) {
				bestVal = v;
				bestEpoch = e + 1;
				best = net.p.clone();
			}
			if (log != null) {
				log.printf("epoca %2d/%d: addestramento %.5f, validazione %.5f%s%n", e + 1, cfg.epochs, lastTrain, v, improved ? " *" : "");
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

	// ====================================================================== metriche sulla rete QUANTIZZATA (quella che giochera')

	public static final class Metrics {
		public int n, nOutcome, nDecided, nValue;
		public double mseTarget, mseOutcome, accuracy, mseValue;

		@Override
		public String toString() {
			return String.format("bersaglio MSE %.4f (n=%d) | esito: MSE %.4f, segno giusto %.1f%% su %d decisi (n=%d) | valore di ricerca: MSE %.4f (n=%d)", mseTarget, n, mseOutcome,
					100 * accuracy, nDecided, nOutcome, mseValue, nValue);
		}
	}

	/** Misura {@code q} su {@code idx} di {@code td.d}; {@code useHist}: con la storia disponibile o con finestra vuota. {@code tgt} = bersaglio di riferimento. */
	public static Metrics evaluate(TemporalNnueNetwork q, TemporalData td, float[] tgt, int[] idx, boolean useHist) {
		final NnueDataset d = td.d;
		final TemporalNnueNetwork.Scratch sc = new TemporalNnueNetwork.Scratch(q);
		final byte[] board = new byte[BoardConstants.NUM_SQUARES];
		final Metrics m = new Metrics();
		double seT = 0, seO = 0, seV = 0;
		int ok = 0;
		for (int i : idx) {
			d.copyBoard(i, board, 0);
			final double y = q.value(board, d.role(i), td.rec, i * td.k, useHist ? td.avail[i] : 0, sc);
			if (!Float.isNaN(tgt[i])) {
				seT += (y - tgt[i]) * (y - tgt[i]);
				m.n++;
			}
			if (d.outValid(i) == 1) {
				final double o = d.outcome(i);
				seO += (y - o) * (y - o);
				m.nOutcome++;
				if (o != 0) {
					m.nDecided++;
					if ((y > 0) == (o > 0)) {
						ok++;
					}
				}
			}
			if (d.valValid(i) == 1) {
				seV += (y - d.value(i)) * (y - d.value(i));
				m.nValue++;
			}
		}
		m.mseTarget = m.n == 0 ? 0 : seT / m.n;
		m.mseOutcome = m.nOutcome == 0 ? 0 : seO / m.nOutcome;
		m.accuracy = m.nDecided == 0 ? 0 : ok / (double) m.nDecided;
		m.mseValue = m.nValue == 0 ? 0 : seV / m.nValue;
		return m;
	}

	// ====================================================================== main

	private static List<String> paths(String csv) {
		final List<String> l = new ArrayList<>();
		for (String p : csv.split(",")) {
			if (!p.trim().isEmpty()) {
				l.add(p.trim());
			}
		}
		return l;
	}

	static int[] range(int n) {
		final int[] a = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = i;
		}
		return a;
	}

	/** I primi {@code maxGames} giochi del dataset (il file e' ordinato per partita e le chiavi sono rinumerate per ordine di comparsa: e' un prefisso). */
	static NnueDataset firstGames(NnueDataset d, int maxGames) {
		if (maxGames <= 0) {
			return d;
		}
		int m = 0;
		while (m < d.n && d.game(m) < maxGames) {
			m++;
		}
		return m == d.n ? d : d.subset(range(m));
	}

	public static void main(String[] args) throws IOException {
		System.exit(mainImpl(args, System.out));
	}

	/** Come {@code main} ma restituisce il codice di uscita (per i test). */
	public static int mainImpl(String[] args, PrintStream out) throws IOException {
		final Config cfg = new Config();
		String data = null, val = null, outFile = null, init = null, compare = null, evalOnly = null;
		boolean hiddenGiven = false;
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				out.println("argomento non valido (atteso chiave=valore): " + a);
				return 2;
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "data": data = v; break;
				case "val": val = v; break;
				case "out": outFile = v; break;
				case "init": init = v; break;
				case "compare": compare = v; break;
				case "eval": evalOnly = v; break;
				case "hist": cfg.hist = Integer.parseInt(v); break;
				case "hidden": cfg.hidden = Integer.parseInt(v); hiddenGiven = true; break;
				case "l2": cfg.l2 = Integer.parseInt(v); hiddenGiven = true; break;
				case "epochs": cfg.epochs = Integer.parseInt(v); break;
				case "batch": cfg.batch = Integer.parseInt(v); break;
				case "lr": cfg.lr = Double.parseDouble(v); break;
				case "lrend": cfg.lrEnd = Double.parseDouble(v); break;
				case "lambda": cfg.lambda = Double.parseDouble(v); break;
				case "tdlambda": cfg.tdLambda = Double.parseDouble(v); break;
				case "gamma": cfg.gamma = Double.parseDouble(v); break;
				case "extlambda": cfg.extLambda = Double.parseDouble(v); break;
				case "histdrop": cfg.histDrop = Double.parseDouble(v); break;
				case "augment": cfg.augment = Boolean.parseBoolean(v); break;
				case "trainbase": cfg.trainBase = Boolean.parseBoolean(v); break;
				case "usehist": cfg.useHist = Boolean.parseBoolean(v); break;
				case "extselect": cfg.extSelect = Boolean.parseBoolean(v); break;
				case "valpct": cfg.valPct = Integer.parseInt(v); break;
				case "valsalt": cfg.valSalt = Long.parseLong(v); break;
				case "patience": cfg.patience = Integer.parseInt(v); break;
				case "seed": cfg.seed = Long.parseLong(v); break;
				case "maxgames": cfg.maxGames = Integer.parseInt(v); break;
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
					out.println("opzione sconosciuta: " + k);
					return 2;
				}
			} catch (NumberFormatException e) {
				out.println("valore non valido per " + k + ": " + v);
				return 2;
			}
		}
		if (data == null || (outFile == null && evalOnly == null)) {
			out.println("Uso: NnueTemporalTrainer data=A.gz[,B.gz] out=FILE.tnnue [init=PESI.nnue|.tnnue] [val=V.gz] [hist=4] [hidden=64] [l2=16] [epochs=10] [batch=256] [lr=0.001] [lrend=0.0001]"
					+ " [lambda=0.5] [tdlambda=1] [gamma=1] [histdrop=0.5] [augment=true] [valpct=5] [valsalt=1] [patience=4] [seed=1] [maxgames=0] [trainbase=true] [usehist=true] [extselect=false] [weights=0:1,1:1]"
					+ " [extlambda=-1] [compare=A.nnue,B.nnue]");
			return 2;
		}
		if (evalOnly != null) { // solo misura: nessun addestramento. Le reti possono essere .tnnue o .nnue (di sola posizione)
			try {
				return evalFiles(cfg, data, val, evalOnly, out);
			} catch (IllegalArgumentException e) {
				out.println("ERRORE: " + e.getMessage());
				return 2;
			} catch (IOException e) {
				out.println("ERRORE: " + e.getMessage());
				return 1;
			}
		}
		try {
			TemporalFloatNet start = null;
			NnueNetwork baseline = null; // la rete di sola posizione da cui si parte: e' il confronto naturale
			if (init != null) {
				if (TemporalNnueNetwork.isTemporalFile(init)) {
					final TemporalNnueNetwork q = TemporalNnueNetwork.load(init);
					if (hiddenGiven && (cfg.hidden != q.hidden() || cfg.l2 != q.l2())) {
						throw new IllegalArgumentException("init=" + init + " e' " + q.hidden() + "/" + q.l2() + ", ma hidden/l2 chiedono " + cfg.hidden + "/" + cfg.l2);
					}
					cfg.hidden = q.hidden();
					cfg.l2 = q.l2();
					cfg.hist = q.hist();
					start = TemporalFloatNet.fromQuantized(q);
					baseline = q.base();
				} else {
					final NnueNetwork q = NnueNetwork.load(init);
					if (hiddenGiven && (cfg.hidden != q.hidden() || cfg.l2 != q.l2())) {
						throw new IllegalArgumentException("init=" + init + " e' " + q.hidden() + "/" + q.l2() + ", ma hidden/l2 chiedono " + cfg.hidden + "/" + cfg.l2);
					}
					cfg.hidden = q.hidden();
					cfg.l2 = q.l2();
					start = TemporalFloatNet.fromStatic(NnueDistiller.FloatNet.fromQuantized(q), cfg.hist);
					baseline = q;
				}
			}
			cfg.validate();
			NnueDataset d = NnueDataset.loadMerged(paths(data));
			d = firstGames(d, cfg.maxGames);
			out.println("addestramento: " + d.describe());
			NnueDataset ext = null;
			if (val != null) {
				ext = firstGames(NnueDataset.loadMerged(paths(val)), cfg.maxGames);
				out.println("validazione esterna: " + ext.describe());
			}
			final long t0 = System.currentTimeMillis();
			final TemporalData td = TemporalData.build(d, cfg.hist);
			long withHist = 0;
			for (int i = 0; i < d.n; i++) {
				withHist += td.avail[i] == cfg.hist ? 1 : 0;
			}
			out.printf("storie ricostruite: %.1f%% dei campioni hanno la finestra completa di %d mosse (%.1f s)%n", 100.0 * withHist / Math.max(1, d.n), cfg.hist, (System.currentTimeMillis() - t0) / 1000.0);
			final float[] tgt = targets(cfg, d, false);
			final float[] refTgt = cfg.tdLambda >= 1.0 ? tgt : classicTargets(cfg, d); // per i confronti: sempre il bersaglio "classico"
			TemporalData xtd = null;
			float[] xtgt = null;
			if (ext != null) {
				xtd = TemporalData.build(ext, cfg.hist);
				xtgt = targets(cfg, ext, true);
			}
			final int[][] sp = d.splitByGame(cfg.valSalt, cfg.valPct);
			if (start == null) {
				start = new TemporalFloatNet(cfg.hidden, cfg.l2, cfg.hist);
				start.init(new Random(cfg.seed));
			}
			final long t1 = System.currentTimeMillis();
			final Result r = train(td, tgt, sp[0], sp[1], xtd, xtgt, xtd == null ? null : range(ext.n), cfg, start, out);
			out.printf("addestramento: %.1f s, %s (%d epoche eseguite), validazione %.5f -> %.5f%n", (System.currentTimeMillis() - t1) / 1000.0,
					r.bestEpoch == 0 ? "NESSUN MIGLIORAMENTO: rete invariata" : "migliore epoca " + r.bestEpoch, r.epochsRun, r.initialVal, r.bestVal);
			final TemporalFloatNet.Quantized q = r.net.quantize();
			if (q.clipped > 0) {
				out.println("ATTENZIONE: " + q.clipped + " pesi saturati nella quantizzazione: la rete scritta differisce da quella addestrata.");
			}
			q.net.save(outFile);
			out.println("scritto " + outFile + " (" + q.net.describe() + ")");
			final int[] held = usable(cfg, d, refTgt, sp[1], false);
			report(out, "rete temporale", q.net, td, refTgt, held, xtd, xtgt);
			if (baseline != null) {
				report(out, "partenza (sola posizione)", TemporalNnueNetwork.fromBase(baseline, cfg.hist), td, refTgt, held, xtd, xtgt);
			}
			if (compare != null) {
				for (String c : compare.split(",")) {
					if (!c.trim().isEmpty()) {
						report(out, "confronto " + c.trim(), TemporalNnueNetwork.fromBase(NnueNetwork.load(c.trim()), cfg.hist), td, refTgt, held, xtd, xtgt);
					}
				}
			}
			return 0;
		} catch (IllegalArgumentException e) {
			out.println("ERRORE: " + e.getMessage());
			return 2;
		} catch (IOException e) {
			out.println("ERRORE: " + e.getMessage());
			return 1;
		}
	}

	private static TemporalNnueNetwork loadAny(String path, int hist) throws IOException {
		return TemporalNnueNetwork.isTemporalFile(path) ? TemporalNnueNetwork.load(path) : TemporalNnueNetwork.fromBase(NnueNetwork.load(path), hist);
	}

	/** Misura (senza addestrare) le reti dei file {@code nets} sullo stesso held-out per partita e sugli stessi dati esterni di un addestramento con la stessa configurazione. */
	static int evalFiles(Config cfg, String data, String val, String nets, PrintStream out) throws IOException {
		final String[] files = nets.split(",");
		int hist = cfg.hist;
		for (String f : files) {
			if (TemporalNnueNetwork.isTemporalFile(f.trim())) {
				hist = TemporalNnueNetwork.load(f.trim()).hist();
				break;
			}
		}
		cfg.hist = hist;
		cfg.validate();
		final NnueDataset d = firstGames(NnueDataset.loadMerged(paths(data)), cfg.maxGames);
		NnueDataset ext = null;
		if (val != null) {
			ext = firstGames(NnueDataset.loadMerged(paths(val)), cfg.maxGames);
		}
		final TemporalData td = TemporalData.build(d, hist);
		final float[] ref = classicTargets(cfg, d);
		TemporalData xtd = null;
		float[] xtgt = null;
		if (ext != null) {
			xtd = TemporalData.build(ext, hist);
			xtgt = targets(cfg, ext, true);
		}
		final int[] held = usable(cfg, d, ref, d.splitByGame(cfg.valSalt, cfg.valPct)[1], false);
		for (String f : files) {
			if (!f.trim().isEmpty()) {
				report(out, f.trim(), loadAny(f.trim(), hist), td, ref, held, xtd, xtgt);
			}
		}
		return 0;
	}

	private static void report(PrintStream out, String name, TemporalNnueNetwork net, TemporalData td, float[] refTgt, int[] held, TemporalData xtd, float[] xtgt) {
		if (held.length > 0) {
			out.println(name + " | held-out, con storia  (" + held.length + " posizioni): " + evaluate(net, td, refTgt, held, true));
			out.println(name + " | held-out, senza storia (" + held.length + " posizioni): " + evaluate(net, td, refTgt, held, false));
		}
		if (xtd != null) {
			final int[] all = usable(new Config(), xtd.d, xtgt, range(xtd.d.n), false);
			out.println(name + " | esterno, con storia  (" + all.length + " posizioni): " + evaluate(net, xtd, xtgt, all, true));
			out.println(name + " | esterno, senza storia: " + evaluate(net, xtd, xtgt, all, false));
		}
	}
}
