package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;

/**
 * Fine-tuning del CNN verso il valore di RICERCA (distillazione della ricerca / idea TD-leaf): avvicina il valore statico
 * {@code V(s) = tanh(z)} al valore che darebbe {@code Negamax(ev, D)} con la STESSA rete ({@link SearchLabeler}). A tempo di gioco
 * non cambia nulla: stessa rete, stessa dimensione, stessa inferenza (bit-identica: si veda {@link ConvNet.Workspace#logit()}).
 *
 * <h3>Loss</h3>
 * Per un campione con logit {@code z}, bersaglio di ricerca {@code y} (dal Bianco, in [-0.98, 0.98]) ed esito {@code o} (+1/-1/0):
 * <pre>
 *   L = (1 - lambda) * L_ricerca + lambda * L_esito          (lambda = 0.2 di default)
 *   L_ricerca = (z - atanh(y))^2                             [loss=logit, default]
 *             = (tanh(z) - y)^2                              [loss=tanh, ablation]
 *   L_esito   = (tanh(z) - o)^2                              (0 se l'esito non e' valido: partita troncata)
 * </pre>
 * Perche' nel dominio dei logit: il gradiente di un MSE su tanh e' moltiplicato per {@code 1 - tanh^2(z)}, che si annulla proprio nelle
 * posizioni "decise" (|V| vicino a 1), cioe' dove la ricerca ha piu' da insegnare sulla dimensione del vantaggio. Nel dominio dei logit
 * il gradiente e' {@code 2(z - t)} ovunque. Costo di questa scelta: un errore di logit pesa uguale a qualunque valore di {@code y},
 * quindi i bersagli estremi (clip a 0.98, atanh = 2.30) hanno un peso in tanh-unita' maggiore; per questo il clip e' fisso e
 * l'ablation {@code loss=tanh} e' disponibile. NON e' provato che il dominio logit sia migliore: e' una scelta di robustezza.
 * Il gradiente rispetto ai pesi usa {@link ConvNet#backwardLogit} (esatto anche a tanh satura).
 *
 * <h3>Anti-deriva (scelta documentata)</h3>
 * La ricerca usa la rete stessa: se la rete cambia, cambiano anche i bersagli (rischio autoreferenziale: la rete puo' imparare a
 * "assomigliare a se stessa", ad esempio schiacciarsi verso valori che minimizzano la distanza dalla propria ricerca, o amplificare
 * errori sistematici a ogni iterazione). Si e' scelto l'<b>ancoraggio L2 ai pesi</b>: {@code + mu * ||w - w_rif||^2} nella loss
 * di minibatch, con {@code w_rif} = pesi a inizio iterazione ({@code anchorTo=prev}, default) oppure i pesi ORIGINALI
 * ({@code anchorTo=init}, limita la deriva cumulata). Motivi: e' deterministico, entra nel gradiente (verificato con differenze
 * finite) e non richiede un secondo insieme da tenere pulito, a differenza di un set di sonda con arresto. In piu', per non lasciare la
 * deriva senza misura: l'iterazione e' SCARTATA (e il ciclo si ferma) se l'errore held-out contro i suoi bersagli non migliora, e si
 * riportano la distanza relativa dei pesi dall'originale e lo scostamento medio del valore statico dall'originale sull'held-out.
 *
 * <h3>Iterazioni</h3>
 * {@code iters} (3-5 consigliate) cicli etichetta -> addestra; D e' fissa DENTRO ogni iterazione ({@code depths}, una per iterazione).
 * La rete che etichetta l'iterazione k e' quella uscita dalla k-1. Il risultato NON e' adottato qui: si scrive solo il candidato; l'adozione e'
 * decisa da {@link DistillAcceptance} (tutte le porte).
 */
public final class CnnDistiller {

	private static final int NF = TablutFeatures.NUM_FEATURES;

	private CnnDistiller() {
	}

	public enum SearchLoss {
		LOGIT, TANH
	}

	// ====================================================================== loss per campione (pura, testata)

	/** Bersaglio nel dominio dei logit: {@code atanh(y)}, con {@code y} gia' in [-0.98, 0.98]. */
	public static double logitTarget(double y) {
		return 0.5 * Math.log((1 + y) / (1 - y));
	}

	/**
	 * Valore della loss di un campione. {@code outValid == false}: il termine d'esito e' omesso (peso 0, non rinormalizzato).
	 *
	 * @param y bersaglio di ricerca in tanh-dominio, dal Bianco
	 * @param o esito della partita dal Bianco
	 */
	public static double sampleLoss(double z, double y, double o, boolean outValid, SearchLoss kind, double lambda) {
		final double th = Math.tanh(z);
		final double ls = kind == SearchLoss.LOGIT ? sq(z - logitTarget(y)) : sq(th - y);
		final double lo = outValid ? sq(th - o) : 0.0;
		return (1 - lambda) * ls + lambda * lo;
	}

	/** {@code dL/dz} del campione (stessa convenzione di {@link #sampleLoss}). */
	public static double sampleDz(double z, double y, double o, boolean outValid, SearchLoss kind, double lambda) {
		final double th = Math.tanh(z);
		final double dth = 1 - th * th;
		final double ds = kind == SearchLoss.LOGIT ? 2 * (z - logitTarget(y)) : 2 * (th - y) * dth;
		final double dout = outValid ? 2 * (th - o) * dth : 0.0;
		return (1 - lambda) * ds + lambda * dout;
	}

	private static double sq(double x) {
		return x * x;
	}

	/** Penalita' di ancoraggio {@code mu * ||w - ref||^2}. */
	public static double anchorPenalty(float[] w, float[] ref, double mu) {
		double s = 0;
		for (int i = 0; i < w.length; i++) {
			double d = w[i] - ref[i];
			s += d * d;
		}
		return mu * s;
	}

	/**
	 * Aggiunge a {@code g} (gradiente SOMMA su un minibatch di {@code batchSize} campioni, che Adam scala per {@code 1/batchSize})
	 * il gradiente dell'ancoraggio: {@code batchSize * 2 mu (w - ref)}. Cosi' dopo lo scaling la loss e' (media dei campioni) + mu*||w-ref||^2.
	 */
	public static void addAnchorGradient(float[] w, float[] ref, double mu, int batchSize, float[] g) {
		final double c = 2.0 * mu * batchSize;
		for (int i = 0; i < w.length; i++) {
			g[i] += (float) (c * (w[i] - ref[i]));
		}
	}

	// ====================================================================== minibatch

	/** Stato di UN thread durante un minibatch. */
	static final class Worker {
		final ConvNet net;
		final ConvNet.Workspace ws;
		final float[] grad;
		final float[] fbuf;
		double loss;

		Worker(ConvNet net) {
			this.net = net;
			this.ws = net.newWorkspace();
			this.grad = new float[net.numParams()];
			this.fbuf = new float[net.config().nFeat];
		}

		void run(DistillSet d, int[] cols, double[] y, int[] idx, int from, int to, int epoch, long seed, boolean augment, SearchLoss kind,
				double lambda) {
			Arrays.fill(grad, 0f);
			loss = 0;
			for (int k = from; k < to; k++) {
				final int i = idx[k];
				ws.setCodes(d.cells, i * 81, augment ? CnnPretrainer.symOf(seed, epoch, i) : 0);
				for (int j = 0; j < cols.length; j++) {
					fbuf[j] = d.feats[i * NF + cols[j]];
				}
				net.forward(ws, fbuf, d.role[i]);
				final double z = ws.logit();
				final boolean ov = d.outValid[i] == 1;
				loss += sampleLoss(z, y[i], d.outcome[i], ov, kind, lambda);
				net.backwardLogit(ws, sampleDz(z, y[i], d.outcome[i], ov, kind, lambda), grad);
			}
		}
	}

	/** Gradiente SOMMA su {@code idx[from..to)} con {@code workers.length} thread; ritorna la somma delle loss dei campioni (senza ancoraggio). */
	static double batchGradient(ExecutorService pool, Worker[] workers, DistillSet d, int[] cols, double[] y, int[] idx, int from, int to, int epoch,
			long seed, boolean augment, SearchLoss kind, double lambda, float[] out) throws Exception {
		final int T = workers.length;
		final int len = to - from;
		List<Callable<Void>> tasks = new ArrayList<>();
		for (int t = 0; t < T; t++) {
			final int a = from + (int) ((long) len * t / T), b = from + (int) ((long) len * (t + 1) / T);
			final Worker w = workers[t];
			final int ep = epoch;
			tasks.add(new Callable<Void>() {
				@Override
				public Void call() {
					w.run(d, cols, y, idx, a, b, ep, seed, augment, kind, lambda);
					return null;
				}
			});
		}
		if (T == 1) {
			tasks.get(0).call();
		} else {
			for (Future<Void> f : pool.invokeAll(tasks)) {
				f.get();
			}
		}
		Arrays.fill(out, 0f);
		double loss = 0;
		for (Worker w : workers) {
			for (int k = 0; k < out.length; k++) {
				out[k] += w.grad[k];
			}
			loss += w.loss;
		}
		return loss;
	}

	// ====================================================================== misure

	/** Errori di una rete su un insieme di posizioni, senza augmentation. */
	public static final class Errors {
		/** MSE tra tanh(z) e il bersaglio y (dominio tanh). */
		public double tanhMse;
		/** MSE tra z e atanh(y) (dominio logit). */
		public double logitMse;
		/** MSE tra tanh(z) e l'esito, sui soli campioni con esito valido. */
		public double outcomeMse;
		public int n;
	}

	/** Valore statico {@code tanh(z)} per ogni posizione di {@code idx} (senza augmentation). */
	public static double[] staticValues(ConvNet net, DistillSet d, int[] cols, int[] idx) {
		ConvNet.Workspace ws = net.newWorkspace();
		float[] fbuf = new float[net.config().nFeat];
		double[] v = new double[idx.length];
		for (int k = 0; k < idx.length; k++) {
			final int i = idx[k];
			ws.setCodes(d.cells, i * 81, 0);
			for (int j = 0; j < cols.length; j++) {
				fbuf[j] = d.feats[i * NF + cols[j]];
			}
			net.forward(ws, fbuf, d.role[i]);
			v[k] = Math.tanh(ws.logit());
		}
		return v;
	}

	/** Errori contro i bersagli {@code y[idx[k]]} (array indicizzato come il set). */
	public static Errors errors(ConvNet net, DistillSet d, int[] cols, int[] idx, double[] y) {
		ConvNet.Workspace ws = net.newWorkspace();
		float[] fbuf = new float[net.config().nFeat];
		Errors e = new Errors();
		double st = 0, sl = 0, so = 0;
		int no = 0;
		for (int i : idx) {
			ws.setCodes(d.cells, i * 81, 0);
			for (int j = 0; j < cols.length; j++) {
				fbuf[j] = d.feats[i * NF + cols[j]];
			}
			net.forward(ws, fbuf, d.role[i]);
			final double z = ws.logit(), th = Math.tanh(z);
			st += sq(th - y[i]);
			sl += sq(z - logitTarget(y[i]));
			if (d.outValid[i] == 1) {
				so += sq(th - d.outcome[i]);
				no++;
			}
		}
		e.n = idx.length;
		e.tanhMse = idx.length == 0 ? 0 : st / idx.length;
		e.logitMse = idx.length == 0 ? 0 : sl / idx.length;
		e.outcomeMse = no == 0 ? 0 : so / no;
		return e;
	}

	/** Errore quadratico (dominio tanh) di OGNI posizione di {@code idx}: serve ai confronti appaiati (porta i). */
	public static double[] sqErrors(ConvNet net, DistillSet d, int[] cols, int[] idx, double[] y) {
		double[] v = staticValues(net, d, cols, idx);
		double[] r = new double[idx.length];
		for (int k = 0; k < idx.length; k++) {
			r[k] = sq(v[k] - y[idx[k]]);
		}
		return r;
	}

	/** Distanza relativa {@code ||w - w0|| / ||w0||}. */
	public static double relativeDrift(float[] w, float[] w0) {
		double a = 0, b = 0;
		for (int i = 0; i < w.length; i++) {
			a += sq(w[i] - w0[i]);
			b += sq(w0[i]);
		}
		return Math.sqrt(a / Math.max(1e-30, b));
	}

	// ====================================================================== addestramento di UNA iterazione

	/**
	 * Addestra {@code net} (in place) sui campioni {@code trainIdx} con la loss mista; Adam con decadimento a coseno del passo.
	 * {@code anchorRef} (null = nessun ancoraggio) sono i pesi di riferimento.
	 *
	 * @return loss media per campione dell'ultima epoca (senza il termine di ancoraggio)
	 */
	public static double trainIteration(ConvNet net, DistillSet d, int[] cols, double[] y, int[] trainIdx, int[] heldIdx, int epochs, double lr0,
			int batch, int threads, boolean augment, long seed, SearchLoss kind, double lambda, float[] anchorRef, double anchorMu, PrintStream log)
			throws Exception {
		if (cols.length != net.config().nFeat) {
			throw new IllegalArgumentException("colonne di feature (" + cols.length + ") != nFeat della rete (" + net.config().nFeat + ")");
		}
		ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
		try {
			Worker[] workers = new Worker[Math.max(1, threads)];
			for (int i = 0; i < workers.length; i++) {
				workers[i] = new Worker(net);
			}
			Adam adam = new Adam(net.numParams());
			float[] g = new float[net.numParams()];
			Random rnd = new Random(seed);
			int[] idx = trainIdx.clone();
			double trainLoss = 0;
			for (int ep = 0; ep < epochs; ep++) {
				for (int i = idx.length - 1; i > 0; i--) {
					int j = rnd.nextInt(i + 1);
					int t = idx[i];
					idx[i] = idx[j];
					idx[j] = t;
				}
				double lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs));
				double loss = 0;
				long cnt = 0;
				long t0 = System.nanoTime();
				for (int b = 0; b + batch <= idx.length; b += batch) {
					loss += batchGradient(pool, workers, d, cols, y, idx, b, b + batch, ep, seed, augment, kind, lambda, g);
					if (anchorRef != null && anchorMu > 0) {
						addAnchorGradient(net.params(), anchorRef, anchorMu, batch, g);
					}
					adam.step(net.params(), g, lr, 1.0 / batch);
					cnt += batch;
				}
				trainLoss = loss / Math.max(1, cnt);
				if (log != null) {
					Errors h = errors(net, d, cols, heldIdx, y);
					log.printf("    epoca %d | loss train=%.4f | held-out: MSE tanh=%.5f logit=%.4f esito=%.4f | lr=%.6f | %.1fs%n", ep + 1, trainLoss,
							h.tanhMse, h.logitMse, h.outcomeMse, lr, (System.nanoTime() - t0) / 1e9);
				}
			}
			return trainLoss;
		} finally {
			pool.shutdown();
		}
	}

	// ====================================================================== configurazione e ciclo di iterazioni

	/** Parametri del ciclo. I default seguono la specifica; NON sono stati tarati. */
	public static final class Params {
		public int iters = 3;
		/** Profondita' di ricerca per iterazione (se piu' corta di {@code iters}, l'ultima si ripete). */
		public int[] depths = { 3 };
		public int epochs = 3;
		public double lr = 2e-4;
		public int batch = 128;
		public double lambda = 0.2;
		public SearchLoss loss = SearchLoss.LOGIT;
		public double anchor = 0.01;
		/** true = ancora ai pesi ORIGINALI; false = ai pesi a inizio iterazione. */
		public boolean anchorToInit = false;
		public boolean augment = true;
		public int threads = Runtime.getRuntime().availableProcessors();
		public long seed = 1;
		public int ttLog2 = 16;
		/** Percorso base dei checkpoint di iterazione (null = nessuno). */
		public String iterOutPrefix = null;

		int depthOf(int iter) {
			return depths[Math.min(iter, depths.length - 1)];
		}
	}

	/** Statistiche di una iterazione. */
	public static final class IterStats {
		public int iter, depth;
		public Errors startHeld, endHeld;
		public double forcedShare, drift, valueDrift;
		public boolean kept;
		public double labelSeconds, trainSeconds;
	}

	public static final class Result {
		/** Rete candidata (ultima iterazione ACCETTATA); uguale alla partenza se nessuna e' stata accettata. */
		public ConvNet net;
		/** Bersagli dell'iterazione 1 (ricerca della rete di PARTENZA) per tutto il set: riferimento fisso della porta (i). */
		public double[] baselineTargets;
		public int baselineDepth;
		public final List<IterStats> stats = new ArrayList<>();
		public int keptIterations;
	}

	/** Colonne di feature per una rete valutata con i gruppi {@code groups}. */
	public static int[] colsFor(ConvNet net, int groups) {
		return net.config().nFeat > 0 ? TablutFeatures.columnsFor(groups) : new int[0];
	}

	/**
	 * Esegue le iterazioni etichetta -> addestra a partire da {@code baseline} (non modificata). Etichetta TUTTE le posizioni del set
	 * (train e held-out) a ogni iterazione con la rete corrente e profondita' fissa di quella iterazione.
	 */
	public static Result run(Params P, CnnEvaluator baseline, DistillSet set, PrintStream log) throws Exception {
		if (P.iters < 1 || P.depths.length < 1 || P.epochs < 1 || P.batch < 1 || P.lambda < 0 || P.lambda > 1) {
			throw new IllegalArgumentException("parametri non validi");
		}
		if (log != null && (P.iters < 3 || P.iters > 5)) {
			log.println("ATTENZIONE: iters=" + P.iters + " fuori dall'intervallo consigliato 3-5");
		}
		if (log != null && (P.epochs < 2 || P.epochs > 4 || P.lr < 1e-4 || P.lr > 3e-4)) {
			log.println("ATTENZIONE: epochs/lr fuori dagli intervalli consigliati (2-4 epoche, lr 1e-4..3e-4)");
		}
		final int groups = baseline.featureGroups;
		final int[] cols = colsFor(baseline.net, groups);
		final int[] train = set.indices(false), held = set.indices(true);
		if (train.length < P.batch || held.length == 0) {
			throw new IllegalArgumentException("serve almeno un minibatch di train e una posizione di held-out (train=" + train.length + ", held=" + held.length + ")");
		}
		final int[] all = new int[set.n];
		for (int i = 0; i < all.length; i++) {
			all[i] = i;
		}
		final float[] w0 = baseline.net.params().clone();
		ConvNet cur = baseline.net.copy();
		double[] v0 = staticValues(baseline.net, set, cols, held);
		Result res = new Result();
		res.net = cur;
		for (int it = 0; it < P.iters; it++) {
			final int depth = P.depthOf(it);
			IterStats st = new IterStats();
			st.iter = it + 1;
			st.depth = depth;
			if (log != null) {
				log.printf("== iterazione %d/%d | D=%d | loss=%s lambda=%.2f ancoraggio mu=%.4g (%s) | lr=%.1e epoche=%d batch=%d ==%n", it + 1, P.iters,
						depth, P.loss, P.lambda, P.anchor, P.anchorToInit ? "pesi originali" : "pesi di inizio iterazione", P.lr, P.epochs, P.batch);
			}
			// 1. etichette con la rete corrente
			CnnEvaluator ev = new CnnEvaluator(cur.copy(), groups);
			SearchLabeler.Labels lab = SearchLabeler.labelAll(ev, set.cells, set.role, all, depth, P.threads, P.ttLog2, log);
			double[] y = lab.y; // indicizzato come il set (idx = tutti)
			st.labelSeconds = lab.seconds;
			st.forcedShare = (double) lab.forced / set.n;
			if (it == 0) {
				res.baselineTargets = y.clone();
				res.baselineDepth = depth;
			}
			st.startHeld = errors(cur, set, cols, held, y);
			if (log != null) {
				log.printf("  etichette: %.1f s (%.1f ms/posizione/thread), vittorie/sconfitte forzate %.1f%% | rete di inizio: held-out MSE tanh=%.5f logit=%.4f%n",
						lab.seconds, 1000 * lab.seconds * P.threads / set.n, 100 * st.forcedShare, st.startHeld.tanhMse, st.startHeld.logitMse);
			}
			// 2. addestramento da una copia
			ConvNet cand = cur.copy();
			float[] ref = P.anchorToInit ? w0 : cur.params().clone();
			long t0 = System.nanoTime();
			trainIteration(cand, set, cols, y, train, held, P.epochs, P.lr, P.batch, P.threads, P.augment, P.seed + 1000L * it, P.loss, P.lambda, ref,
					P.anchor, log);
			st.trainSeconds = (System.nanoTime() - t0) / 1e9;
			st.endHeld = errors(cand, set, cols, held, y);
			st.drift = relativeDrift(cand.params(), w0);
			double[] v = staticValues(cand, set, cols, held);
			double vd = 0;
			for (int k = 0; k < v.length; k++) {
				vd += Math.abs(v[k] - v0[k]);
			}
			st.valueDrift = 1000.0 * vd / v.length;
			// 3. arresto: l'iterazione e' tenuta solo se l'errore held-out contro i SUOI bersagli scende
			st.kept = st.endHeld.tanhMse < st.startHeld.tanhMse;
			if (log != null) {
				log.printf("  held-out MSE tanh: %.5f -> %.5f | logit: %.4f -> %.4f | esito: %.4f -> %.4f | deriva pesi dall'originale %.3f%%, |V-V0| medio %.1f (scala 1000)%n",
						st.startHeld.tanhMse, st.endHeld.tanhMse, st.startHeld.logitMse, st.endHeld.logitMse, st.startHeld.outcomeMse, st.endHeld.outcomeMse,
						100 * st.drift, st.valueDrift);
				log.println(st.kept ? "  iterazione TENUTA" : "  iterazione SCARTATA (l'errore held-out contro i suoi bersagli non scende): ciclo interrotto");
			}
			res.stats.add(st);
			if (!st.kept) {
				break;
			}
			cur = cand;
			res.net = cur;
			res.keptIterations++;
			if (P.iterOutPrefix != null) {
				new CnnEvaluator(cur, groups).save(P.iterOutPrefix + ".iter" + (it + 1) + ".txt");
			}
		}
		return res;
	}

	// ====================================================================== porta (i)

	/** Esito del confronto appaiato dell'errore held-out contro bersagli di ricerca FISSI. */
	public static final class HeldOutGate {
		public double baseMse, candMse, meanGain, se;
		public int n;
		public boolean pass;

		@Override
		public String toString() {
			return String.format("MSE tanh held-out vs ricerca: partenza %.5f, candidata %.5f (guadagno %.5f +- %.5f ES, n=%d) -> %s", baseMse, candMse, meanGain,
					se, n, pass ? "SUPERATA" : "NON superata");
		}
	}

	/**
	 * Porta (i): l'errore held-out della candidata rispetto ai bersagli di ricerca deve essere inferiore a quello della rete di partenza.
	 * I bersagli {@code y} sono FISSI e uguali per le due reti (qui: la ricerca della rete di PARTENZA, {@link Result#baselineTargets}):
	 * confrontare ciascuna rete con la PROPRIA ricerca premierebbe una rete costante (errore zero). Passa se il guadagno medio e' > 0.
	 */
	public static HeldOutGate heldOutGate(ConvNet baseline, ConvNet cand, DistillSet d, int[] cols, int[] heldIdx, double[] y) {
		double[] eb = sqErrors(baseline, d, cols, heldIdx, y), ec = sqErrors(cand, d, cols, heldIdx, y);
		HeldOutGate g = new HeldOutGate();
		g.n = eb.length;
		double sb = 0, sc = 0, sd = 0, sd2 = 0;
		for (int k = 0; k < eb.length; k++) {
			sb += eb[k];
			sc += ec[k];
			double diff = eb[k] - ec[k];
			sd += diff;
			sd2 += diff * diff;
		}
		g.baseMse = sb / Math.max(1, g.n);
		g.candMse = sc / Math.max(1, g.n);
		g.meanGain = sd / Math.max(1, g.n);
		double var = g.n > 1 ? (sd2 - sd * sd / g.n) / (g.n - 1) : 0;
		g.se = Math.sqrt(Math.max(0, var) / Math.max(1, g.n));
		g.pass = g.n > 0 && g.candMse < g.baseMse;
		return g;
	}

	// ====================================================================== riga di comando

	static Map<String, String> parse(String[] args) {
		Map<String, String> p = new HashMap<>();
		for (String a : args) {
			int eq = a.indexOf('=');
			if (eq > 0) {
				p.put(a.substring(0, eq), a.substring(eq + 1));
			}
		}
		return p;
	}

	static int[] parseInts(String s) {
		String[] t = s.split(",");
		int[] r = new int[t.length];
		for (int i = 0; i < t.length; i++) {
			r[i] = Integer.parseInt(t[i].trim());
		}
		return r;
	}

	/**
	 * <pre>
	 * CnnDistiller init=pesi_v4.txt data=dati.gz [engineGames=200 engineMs=100 engineSet=eng.gz] n=20000 nHeld=2000 iters=3 depths=3
	 *              epochs=3 lr=2e-4 batch=128 lambda=0.2 loss=logit|tanh anchor=0.01 anchorTo=prev|init augment=1 threads=4 seed=1
	 *              out=cnn_distilled.txt setOut=distill_set.gz
	 * </pre>
	 * Scrive la candidata in {@code out} (NON adotta nulla: si decide con {@link DistillAcceptance}) e il set in {@code setOut}.
	 */
	public static void main(String[] args) throws Exception {
		int code = mainImpl(args);
		if (code != 0) {
			System.exit(code);
		}
	}

	/** Come {@link #main} ma ritorna il codice di uscita (0 ok, 2 uso errato, 3 nessuna iterazione tenuta) invece di chiamare {@code System.exit}. */
	static int mainImpl(String[] args) throws Exception {
		Map<String, String> p = parse(args);
		if (!p.containsKey("init")) {
			System.out.println("uso: CnnDistiller init=pesi.txt [data=dati.gz] [engineGames=N engineMs=ms engineSet=file] [n=20000] [nHeld=2000] [iters=3]"
					+ " [depths=3] [epochs=3] [lr=2e-4] [batch=128] [lambda=0.2] [loss=logit|tanh] [anchor=0.01] [anchorTo=prev|init] [augment=1]"
					+ " [threads=N] [seed=1] [out=cnn_distilled.txt] [setOut=distill_set.gz]");
			return 2;
		}
		Params P = new Params();
		P.iters = Integer.parseInt(p.getOrDefault("iters", "3"));
		P.depths = parseInts(p.getOrDefault("depths", "3"));
		P.epochs = Integer.parseInt(p.getOrDefault("epochs", "3"));
		P.lr = Double.parseDouble(p.getOrDefault("lr", "2e-4"));
		P.batch = Integer.parseInt(p.getOrDefault("batch", "128"));
		P.lambda = Double.parseDouble(p.getOrDefault("lambda", "0.2"));
		P.loss = p.getOrDefault("loss", "logit").equalsIgnoreCase("tanh") ? SearchLoss.TANH : SearchLoss.LOGIT;
		P.anchor = Double.parseDouble(p.getOrDefault("anchor", "0.01"));
		P.anchorToInit = p.getOrDefault("anchorTo", "prev").equalsIgnoreCase("init");
		P.augment = !p.getOrDefault("augment", "1").equals("0");
		P.threads = Integer.parseInt(p.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
		P.seed = Long.parseLong(p.getOrDefault("seed", "1"));
		P.ttLog2 = Integer.parseInt(p.getOrDefault("tt", "16"));
		String out = p.getOrDefault("out", "cnn_distilled.txt");
		P.iterOutPrefix = out.replaceAll("\\.txt$", "");
		int n = Integer.parseInt(p.getOrDefault("n", "20000")), nHeld = Integer.parseInt(p.getOrDefault("nHeld", "2000"));
		int heldPct = Integer.parseInt(p.getOrDefault("heldPct", "10"));
		CnnEvaluator baseline = CnnEvaluator.fromFile(p.get("init"));
		System.out.println("=== CnnDistiller === partenza: " + p.get("init") + " (" + baseline.net.config() + ", " + baseline.net.numParams() + " parametri)");

		DistillSet a = null, b = null;
		if (p.containsKey("data")) {
			a = DistillSet.fromDataset(DistillSet.loadDataset(p.get("data")), P.seed, heldPct);
			System.out.println("dataset " + p.get("data") + ": " + a.describe());
		}
		if (p.containsKey("engineSet")) {
			b = DistillSet.load(p.get("engineSet"));
			System.out.println("partite engine da " + p.get("engineSet") + ": " + b.describe());
		} else if (p.containsKey("engineGames")) {
			int g = Integer.parseInt(p.get("engineGames"));
			long ms = Long.parseLong(p.getOrDefault("engineMs", "100"));
			double eps = Double.parseDouble(p.getOrDefault("engineEps", "0.03"));
			System.out.println("genero " + g + " partite dell'engine (" + ms + " ms/mossa reali, 1 thread per lato)...");
			b = DistillSet.generateEngineGames(baseline, g, ms, eps, P.seed, P.threads, P.seed, heldPct, System.out);
			System.out.println("partite engine: " + b.describe());
			if (p.containsKey("engineOut")) {
				b.save(p.get("engineOut"));
			}
		}
		if (a == null && b == null) {
			System.out.println("serve almeno una sorgente di posizioni: data=... e/o engineGames=.../engineSet=...");
			return 2;
		}
		String[] rep = new String[1];
		DistillSet set = DistillSet.sample(a, b, n, nHeld, P.seed, rep);
		System.out.println("campione: " + rep[0]);
		if (a == null || b == null) {
			System.out.println("ATTENZIONE: una sola sorgente disponibile: non e' il mix meta' dataset / meta' engine richiesto");
		}
		if (!set.splitIsDisjointByGame()) {
			throw new IllegalStateException("train e held-out condividono partite");
		}
		set.save(p.getOrDefault("setOut", "distill_set.gz"));

		Result r = run(P, baseline, set, System.out);
		System.out.println("--- riepilogo: " + r.keptIterations + "/" + r.stats.size() + " iterazioni tenute ---");
		if (r.keptIterations == 0) {
			System.out.println("nessuna iterazione tenuta: nessun candidato scritto. La rete di partenza resta quella in uso.");
			return 3;
		}
		new CnnEvaluator(r.net, baseline.featureGroups).save(out);
		int[] cols = colsFor(baseline.net, baseline.featureGroups);
		HeldOutGate g = heldOutGate(baseline.net, r.net, set, cols, set.indices(true), r.baselineTargets);
		System.out.println("Porta (i) [bersagli = ricerca della rete di partenza a D=" + r.baselineDepth + "]: " + g);
		System.out.println("Candidata scritta in " + new File(out).getPath() + ". NON adottata: servono anche le porte (ii) HangingDiag e (iii) EqualTimeArena:");
		System.out.println("  java ... training.DistillAcceptance baseline=" + p.get("init") + " candidate=" + out + " set=" + p.getOrDefault("setOut", "distill_set.gz")
				+ " depth=" + r.baselineDepth);
		return 0;
	}

	/** Solo per i test: carica il candidato da file. */
	static CnnEvaluator load(String path) throws IOException {
		return CnnEvaluator.fromFile(path);
	}
}
