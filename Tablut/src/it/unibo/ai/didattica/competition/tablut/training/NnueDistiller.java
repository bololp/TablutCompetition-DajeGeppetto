package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Addestra una rete NNUE ({@link NnueNetwork}) a imitare un altro valutatore (la CNN, l'MLP o un'altra NNUE): <b>distillazione</b>.
 *
 * <h3>Perche' si distilla e non si converte</h3>
 * I pesi della CNN NON si possono convertire: l'ibrido ({@code CnnEvaluator}) somma ai filtri convoluzionali 44 feature a mano GLOBALI
 * (vie di fuga, minacce, mobilita') che dipendono da molte celle insieme e non sono funzioni sparse della posizione dei singoli pezzi;
 * l'NNUE vede invece solo (tipo di pedina, cella). Per avere comunque una rete NNUE giocabile e confrontabile si distilla: si campionano
 * posizioni, si etichettano con il valutatore "maestro" e si addestra l'NNUE a riprodurne il valore. Questo NON e' un allenamento
 * dai dati di gioco: l'NNUE impara cio' che il maestro sa ESPRIMERE con (tipo, cella), con il suo stesso errore.
 *
 * <h3>Dati</h3>
 * Partite generate qui, miste: ogni mossa e' casuale con probabilita' {@code 1 - greedy}, altrimenti e' la migliore secondo il maestro a un
 * ply (con una breve apertura casuale per la varieta'). Si registra una posizione NON terminale con probabilita' {@code keep}; con
 * {@code augment} se ne registrano le 8 simmetrie del quadrato ({@link BoardSymmetry}), perche' le regole sono invarianti e l'NNUE non ha
 * simmetrie incorporate. Bersaglio = valore del maestro sulla posizione (trasformata), diviso per la sua scala e limitato a [-1, 1].
 * Il held-out e' separato PER POSIZIONE DI PARTENZA ({@link Data#group}), cosi' le simmetrie della stessa posizione non finiscono da
 * entrambe le parti.
 *
 * <h3>Rete e perdita</h3>
 * Stessa struttura di {@link NnueNetwork} in virgola mobile ({@code double}): acc = ftB + somma delle colonne attive, {@code a = clamp(acc, 0, 1)},
 * poi per ciascun lato al tratto uno strato denso con clipped ReLU e un'uscita {@code z}; valore {@code tanh(z)}. Perdita {@code (tanh(z) - t)^2}.
 * Gradiente scritto a mano (verificato con differenze finite da {@code NnueTrainingCheck}), ottimizzatore Adam. Inizializzazione e
 * mescolamento hanno un seme: stesso seme = stessi pesi, bit per bit.
 *
 * <h3>Quantizzazione</h3>
 * {@code ftW,ftB * QA}, {@code l1W,l2W * QW}, {@code l1B,l2B * QA*QW}, arrotondati; un peso che non sta in {@code short} viene saturato e
 * CONTATO ({@link Quantized#clipped}): il chiamante decide cosa farne, il main avvisa. Le metriche sul held-out sono calcolate sulla rete
 * QUANTIZZATA, cioe' quella che giochera'.
 * <p>
 * Uso: {@code NnueDistiller teacher=PESI out=FILE.nnue [hidden=64] [l2=16] [games=600] [epochs=20] [batch=256] [lr=0.002] [seed=1]
 * [greedy=0.5] [keep=0.5] [augment=true] [teacherscale=1000]}.
 */
public final class NnueDistiller {

	private NnueDistiller() {
	}

	// ====================================================================== configurazione e dati

	public static final class Config {
		public int hidden = 64;
		public int l2 = 16;
		public int games = 600;
		public double keep = 0.5;
		public double greedy = 0.5;
		public boolean augment = true;
		public int maxPlies = 160;
		public int epochs = 20;
		public int batch = 256;
		public double lr = 2e-3;
		public double lrEnd = 1e-4;
		public double valFraction = 0.05;
		public double teacherScale = 1000.0;
		public long seed = 1;
	}

	/** Campioni: scacchiera (81 codici di {@link BoardConstants}), lato al tratto (0 Bianco, 1 Nero), bersaglio in [-1,1], posizione di partenza. */
	public static final class Data {
		public final int n;
		public final byte[] boards;
		public final byte[] role;
		public final double[] target;
		public final int[] group;

		public Data(byte[] boards, byte[] role, double[] target, int[] group) {
			this.n = role.length;
			if (boards.length != 81 * n || target.length != n || group.length != n) {
				throw new IllegalArgumentException("dimensioni incoerenti dei dati");
			}
			this.boards = boards;
			this.role = role;
			this.target = target;
			this.group = group;
		}
	}

	/** Una posizione trasformata da una delle 8 simmetrie del quadrato ({@code sym = 0} = identita'). */
	public static FastState transformed(FastState fs, int sym) {
		if (sym == 0) {
			return FastState.copyOf(fs);
		}
		final State t = BoardSymmetry.transform(fs.toState(), sym);
		return FastState.fromState(t);
	}

	/**
	 * Genera i campioni con {@code teacher} come maestro. Registra solo posizioni NON terminali (la ricerca non valuta mai le terminali).
	 */
	public static Data generate(Evaluator teacher, Config cfg, PrintStream log) {
		final Random rnd = new Random(cfg.seed * 7919 + 1);
		final List<byte[]> boards = new ArrayList<>();
		final List<Byte> roles = new ArrayList<>();
		final List<Double> targets = new ArrayList<>();
		final List<Integer> groups = new ArrayList<>();
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		int base = 0;
		for (int g = 0; g < cfg.games; g++) {
			final FastState fs = new FastState();
			final int openRandom = 2 + rnd.nextInt(6);
			for (int ply = 0; ply < cfg.maxPlies && fs.winner == FastState.WINNER_NONE; ply++) {
				final int n = fs.generateMoves(buf);
				if (n == 0) {
					break;
				}
				if (rnd.nextDouble() < cfg.keep) {
					final int syms = cfg.augment ? BoardSymmetry.COUNT : 1;
					for (int s = 0; s < syms; s++) {
						final FastState t = transformed(fs, s);
						final double v = teacher.evaluate(t) / cfg.teacherScale;
						boards.add(t.board.clone());
						roles.add((byte) (t.turn == BoardConstants.BLACK_TURN ? 1 : 0));
						targets.add(Math.max(-1.0, Math.min(1.0, v)));
						groups.add(base);
					}
					base++;
				}
				int m = buf[rnd.nextInt(n)];
				if (ply >= openRandom && rnd.nextDouble() < cfg.greedy) {
					m = bestMoveByTeacher(fs, buf, n, teacher, u, rnd);
				}
				fs.makeMove(m, u);
			}
		}
		final int n = roles.size();
		final byte[] b = new byte[81 * n];
		final byte[] r = new byte[n];
		final double[] t = new double[n];
		final int[] gr = new int[n];
		for (int i = 0; i < n; i++) {
			System.arraycopy(boards.get(i), 0, b, 81 * i, 81);
			r[i] = roles.get(i);
			t[i] = targets.get(i);
			gr[i] = groups.get(i);
		}
		if (log != null) {
			log.println("dati: " + base + " posizioni di partenza, " + n + " campioni" + (cfg.augment ? " (8 simmetrie ciascuna)" : ""));
		}
		return new Data(b, r, t, gr);
	}

	private static int bestMoveByTeacher(FastState fs, int[] buf, int n, Evaluator teacher, UndoInfo u, Random rnd) {
		final double sign = fs.turn == BoardConstants.WHITE_TURN ? 1.0 : -1.0;
		int best = buf[0];
		double bestV = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < n; i++) {
			fs.makeMove(buf[i], u);
			final double v;
			if (fs.winner == FastState.WINNER_WHITE) {
				v = sign * 1e9;
			} else if (fs.winner == FastState.WINNER_BLACK) {
				v = -sign * 1e9;
			} else if (fs.winner != FastState.WINNER_NONE) {
				v = 0;
			} else {
				v = sign * teacher.evaluate(fs);
			}
			fs.unmakeMove(buf[i], u);
			final double noisy = v + rnd.nextDouble() * 1e-6;
			if (noisy > bestV) {
				bestV = noisy;
				best = buf[i];
			}
		}
		return best;
	}

	// ====================================================================== rete in virgola mobile

	/** Stessa struttura di {@link NnueNetwork}, in {@code double}: parametri in un unico array (per l'ottimizzatore e il controllo del gradiente). */
	public static final class FloatNet {
		public final int hidden, l2;
		/** Layout: ftW[243*H], ftB[H], poi per ciascun lato: l1W[L2*H], l1B[L2], l2W[L2], l2B[1]. */
		public final double[] p;
		private final int oFtB, oRole, roleSize;

		public FloatNet(int hidden, int l2) {
			this.hidden = hidden;
			this.l2 = l2;
			this.oFtB = NnueNetwork.NUM_FEATURES * hidden;
			this.oRole = oFtB + hidden;
			this.roleSize = l2 * hidden + l2 + l2 + 1;
			this.p = new double[oRole + NnueNetwork.ROLES * roleSize];
		}

		public int numParams() {
			return p.length;
		}

		int offL1W(int r) {
			return oRole + r * roleSize;
		}

		int offL1B(int r) {
			return offL1W(r) + l2 * hidden;
		}

		int offL2W(int r) {
			return offL1B(r) + l2;
		}

		int offL2B(int r) {
			return offL2W(r) + l2;
		}

		public void init(Random rnd) {
			for (int i = 0; i < oFtB; i++) {
				p[i] = rnd.nextGaussian() * 0.15;
			}
			for (int j = 0; j < hidden; j++) {
				p[oFtB + j] = 0.2;
			}
			for (int r = 0; r < NnueNetwork.ROLES; r++) {
				final double s1 = 1.0 / Math.sqrt(hidden), s2 = 1.0 / Math.sqrt(l2);
				for (int i = 0; i < l2 * hidden; i++) {
					p[offL1W(r) + i] = rnd.nextGaussian() * s1;
				}
				for (int i = 0; i < l2; i++) {
					p[offL1B(r) + i] = 0.1;
					p[offL2W(r) + i] = rnd.nextGaussian() * s2;
				}
				p[offL2B(r)] = 0.0;
			}
		}
	}

	/** Buffer di lavoro di una passata (un thread). */
	public static final class Ws {
		final double[] h, a, u, c, da;
		final int[] act;
		int nAct;

		public Ws(int hidden, int l2) {
			h = new double[hidden];
			a = new double[hidden];
			da = new double[hidden];
			u = new double[l2];
			c = new double[l2];
			act = new int[81];
		}
	}

	private static double clamp01(double x) {
		return x < 0 ? 0 : (x > 1 ? 1 : x);
	}

	/** Passata avanti: restituisce il logit {@code z} (il valore e' {@code tanh(z)}). */
	public static double forward(FloatNet n, byte[] boards, int off, int role, Ws ws) {
		final int H = n.hidden, L = n.l2;
		ws.nAct = 0;
		for (int sq = 0; sq < 81; sq++) {
			final int piece = boards[off + sq];
			if (piece != BoardConstants.EMPTY) {
				ws.act[ws.nAct++] = NnueNetwork.featureIndex(piece, sq);
			}
		}
		final double[] p = n.p;
		for (int j = 0; j < H; j++) {
			ws.h[j] = p[n.oFtB + j];
		}
		for (int k = 0; k < ws.nAct; k++) {
			final int o = ws.act[k] * H;
			for (int j = 0; j < H; j++) {
				ws.h[j] += p[o + j];
			}
		}
		for (int j = 0; j < H; j++) {
			ws.a[j] = clamp01(ws.h[j]);
		}
		double z = p[n.offL2B(role)];
		for (int i = 0; i < L; i++) {
			double s = p[n.offL1B(role) + i];
			final int w = n.offL1W(role) + i * H;
			for (int j = 0; j < H; j++) {
				s += p[w + j] * ws.a[j];
			}
			ws.u[i] = s;
			ws.c[i] = clamp01(s);
			z += p[n.offL2W(role) + i] * ws.c[i];
		}
		return z;
	}

	/** Passata indietro dopo {@link #forward}: somma in {@code g} il gradiente della perdita data {@code dz = dL/dz}. */
	public static void backward(FloatNet n, int role, double dz, Ws ws, double[] g) {
		final int H = n.hidden, L = n.l2;
		final double[] p = n.p;
		g[n.offL2B(role)] += dz;
		java.util.Arrays.fill(ws.da, 0, H, 0.0);
		for (int i = 0; i < L; i++) {
			g[n.offL2W(role) + i] += dz * ws.c[i];
			final double dc = dz * p[n.offL2W(role) + i];
			if (ws.u[i] > 0 && ws.u[i] < 1) {
				final double du = dc;
				g[n.offL1B(role) + i] += du;
				final int w = n.offL1W(role) + i * H;
				for (int j = 0; j < H; j++) {
					g[w + j] += du * ws.a[j];
					ws.da[j] += du * p[w + j];
				}
			}
		}
		for (int j = 0; j < H; j++) {
			if (ws.h[j] > 0 && ws.h[j] < 1) {
				final double dh = ws.da[j];
				g[n.oFtB + j] += dh;
				for (int k = 0; k < ws.nAct; k++) {
					g[ws.act[k] * H + j] += dh;
				}
			}
		}
	}

	/** Perdita di un campione e, se {@code g != null}, ne accumula il gradiente. */
	public static double sampleLoss(FloatNet n, Data d, int i, Ws ws, double[] g) {
		final double z = forward(n, d.boards, 81 * i, d.role[i], ws);
		final double y = Math.tanh(z);
		final double err = y - d.target[i];
		if (g != null) {
			backward(n, d.role[i], 2 * err * (1 - y * y), ws, g);
		}
		return err * err;
	}

	// ====================================================================== ottimizzatore

	/** Adam in {@code double} (stessa formula di {@link Adam}, che lavora in {@code float}). */
	static final class AdamD {
		private final double[] m, v;
		private long t;

		AdamD(int n) {
			m = new double[n];
			v = new double[n];
		}

		void step(double[] w, double[] g, double lr, double gradScale) {
			t++;
			final double c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t);
			for (int i = 0; i < w.length; i++) {
				final double gi = g[i] * gradScale;
				m[i] = 0.9 * m[i] + 0.1 * gi;
				v[i] = 0.999 * v[i] + 0.001 * gi * gi;
				w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
			}
		}
	}

	// ====================================================================== addestramento

	public static final class Result {
		public final FloatNet net;
		public final double trainLoss, valLoss, initialValLoss;

		Result(FloatNet net, double trainLoss, double valLoss, double initialValLoss) {
			this.net = net;
			this.trainLoss = trainLoss;
			this.valLoss = valLoss;
			this.initialValLoss = initialValLoss;
		}
	}

	/** Indici dei campioni di train e di held-out (per gruppo: una posizione di partenza sta tutta da una parte). */
	public static int[][] split(Data d, double valFraction, long seed) {
		final List<Integer> tr = new ArrayList<>(), va = new ArrayList<>();
		for (int i = 0; i < d.n; i++) {
			final long h = (d.group[i] * 0x9E3779B97F4A7C15L) ^ (seed * 0xBF58476D1CE4E5B9L);
			final double x = ((h ^ (h >>> 31)) & 0xFFFFFFL) / (double) 0x1000000L;
			(x < valFraction ? va : tr).add(i);
		}
		return new int[][] { toArray(tr), toArray(va) };
	}

	private static int[] toArray(List<Integer> l) {
		final int[] a = new int[l.size()];
		for (int i = 0; i < a.length; i++) {
			a[i] = l.get(i);
		}
		return a;
	}

	public static double meanLoss(FloatNet n, Data d, int[] idx) {
		final Ws ws = new Ws(n.hidden, n.l2);
		double s = 0;
		for (int i : idx) {
			s += sampleLoss(n, d, i, ws, null);
		}
		return idx.length == 0 ? 0 : s / idx.length;
	}

	public static Result train(Data d, Config cfg, PrintStream log) {
		final int[][] sp = split(d, cfg.valFraction, cfg.seed);
		final int[] tr = sp[0], va = sp[1];
		if (tr.length == 0) {
			throw new IllegalArgumentException("nessun campione di addestramento");
		}
		final Random rnd = new Random(cfg.seed);
		final FloatNet net = new FloatNet(cfg.hidden, cfg.l2);
		net.init(rnd);
		final Ws ws = new Ws(cfg.hidden, cfg.l2);
		final double[] g = new double[net.numParams()];
		final AdamD adam = new AdamD(net.numParams());
		final double initial = meanLoss(net, d, va.length > 0 ? va : tr);
		if (log != null) {
			log.printf("avvio: %d campioni di train, %d di held-out, %d parametri; perdita iniziale (held-out) %.5f%n", tr.length, va.length, net.numParams(), initial);
		}
		final int[] order = tr.clone();
		final long totalSteps = (long) cfg.epochs * ((tr.length + cfg.batch - 1) / cfg.batch);
		long step = 0;
		double lastTrain = 0;
		for (int e = 0; e < cfg.epochs; e++) {
			for (int i = order.length - 1; i > 0; i--) { // Fisher-Yates con il seme
				final int j = rnd.nextInt(i + 1);
				final int t = order[i];
				order[i] = order[j];
				order[j] = t;
			}
			double sum = 0;
			for (int b = 0; b < order.length; b += cfg.batch) {
				final int end = Math.min(order.length, b + cfg.batch);
				java.util.Arrays.fill(g, 0.0);
				for (int k = b; k < end; k++) {
					sum += sampleLoss(net, d, order[k], ws, g);
				}
				final double frac = totalSteps <= 1 ? 1.0 : (double) step / (totalSteps - 1);
				adam.step(net.p, g, cfg.lr + (cfg.lrEnd - cfg.lr) * frac, 1.0 / (end - b));
				step++;
			}
			lastTrain = sum / order.length;
			if (log != null && (e == 0 || (e + 1) % Math.max(1, cfg.epochs / 10) == 0 || e == cfg.epochs - 1)) {
				log.printf("epoca %2d/%d: perdita train %.5f, held-out %.5f%n", e + 1, cfg.epochs, lastTrain, meanLoss(net, d, va.length > 0 ? va : tr));
			}
		}
		return new Result(net, lastTrain, meanLoss(net, d, va.length > 0 ? va : tr), initial);
	}

	// ====================================================================== quantizzazione e metriche

	public static final class Quantized {
		public final NnueNetwork net;
		/** Quanti pesi non stavano nell'intervallo {@code short} e sono stati saturati (0 = nessuna perdita). */
		public final int clipped;

		Quantized(NnueNetwork net, int clipped) {
			this.net = net;
			this.clipped = clipped;
		}
	}

	public static Quantized quantize(FloatNet f) {
		final int H = f.hidden, L = f.l2;
		final int[] clipped = new int[1];
		final short[] ftW = new short[NnueNetwork.NUM_FEATURES * H];
		for (int i = 0; i < ftW.length; i++) {
			ftW[i] = toShort(f.p[i] * NnueNetwork.QA, clipped);
		}
		final short[] ftB = new short[H];
		for (int j = 0; j < H; j++) {
			ftB[j] = toShort(f.p[f.oFtB + j] * NnueNetwork.QA, clipped);
		}
		final short[][] l1W = new short[NnueNetwork.ROLES][L * H];
		final int[][] l1B = new int[NnueNetwork.ROLES][L];
		final short[][] l2W = new short[NnueNetwork.ROLES][L];
		final int[] l2B = new int[NnueNetwork.ROLES];
		final double sb = (double) NnueNetwork.QA * NnueNetwork.QW;
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			for (int i = 0; i < L * H; i++) {
				l1W[r][i] = toShort(f.p[f.offL1W(r) + i] * NnueNetwork.QW, clipped);
			}
			for (int i = 0; i < L; i++) {
				l1B[r][i] = toInt(f.p[f.offL1B(r) + i] * sb, clipped);
				l2W[r][i] = toShort(f.p[f.offL2W(r) + i] * NnueNetwork.QW, clipped);
			}
			l2B[r] = toInt(f.p[f.offL2B(r)] * sb, clipped);
		}
		return new Quantized(new NnueNetwork(H, L, ftW, ftB, l1W, l1B, l2W, l2B), clipped[0]);
	}

	private static short toShort(double v, int[] clipped) {
		final long r = Math.round(v);
		if (r < Short.MIN_VALUE || r > Short.MAX_VALUE) {
			clipped[0]++;
			return (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, r));
		}
		return (short) r;
	}

	private static int toInt(double v, int[] clipped) {
		final long r = Math.round(v);
		if (r < Integer.MIN_VALUE / 2 || r > Integer.MAX_VALUE / 2) {
			clipped[0]++;
			return (int) Math.max(Integer.MIN_VALUE / 2, Math.min(Integer.MAX_VALUE / 2, r));
		}
		return (int) r;
	}

	/** Metriche della rete quantizzata contro i bersagli (valori in (-1,1)). */
	public static final class Metrics {
		public double mse, correlation, signAgreement, maxAbsErr;
		public int n, signN;

		@Override
		public String toString() {
			return String.format("MSE %.5f, correlazione %.4f, segno concorde %.1f%% (su %d con |bersaglio|>0.1), errore massimo %.3f, %d campioni", mse, correlation,
					100 * signAgreement, signN, maxAbsErr, n);
		}
	}

	public static Metrics evaluate(NnueNetwork q, Data d, int[] idx) {
		final NnueEvaluator ev = new NnueEvaluator(q);
		ev.outputScale = 1.0;
		final FastState fs = new FastState(); // l'NnueEvaluator legge solo board e turn
		final Metrics m = new Metrics();
		double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, se = 0;
		int agree = 0, signN = 0;
		for (int i : idx) {
			System.arraycopy(d.boards, 81 * i, fs.board, 0, 81);
			fs.turn = d.role[i] == 1 ? BoardConstants.BLACK_TURN : BoardConstants.WHITE_TURN;
			final double y = ev.evaluate(fs), t = d.target[i];
			sx += y;
			sy += t;
			sxx += y * y;
			syy += t * t;
			sxy += y * t;
			se += (y - t) * (y - t);
			m.maxAbsErr = Math.max(m.maxAbsErr, Math.abs(y - t));
			if (Math.abs(t) > 0.1) {
				signN++;
				if ((y > 0) == (t > 0)) {
					agree++;
				}
			}
		}
		final int n = idx.length;
		m.n = n;
		m.signN = signN;
		if (n > 0) {
			m.mse = se / n;
			final double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n);
			m.correlation = vx > 1e-12 && vy > 1e-12 ? (sxy / n - (sx / n) * (sy / n)) / Math.sqrt(vx * vy) : 0;
			m.signAgreement = signN == 0 ? 1.0 : (double) agree / signN;
		}
		return m;
	}

	// ====================================================================== main

	public static void main(String[] args) throws IOException {
		final Config cfg = new Config();
		String teacherPath = null, out = null;
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + a);
				System.exit(2);
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "teacher": teacherPath = v; break;
				case "out": out = v; break;
				case "hidden": cfg.hidden = Integer.parseInt(v); break;
				case "l2": cfg.l2 = Integer.parseInt(v); break;
				case "games": cfg.games = Integer.parseInt(v); break;
				case "epochs": cfg.epochs = Integer.parseInt(v); break;
				case "batch": cfg.batch = Integer.parseInt(v); break;
				case "lr": cfg.lr = Double.parseDouble(v); break;
				case "seed": cfg.seed = Long.parseLong(v); break;
				case "greedy": cfg.greedy = Double.parseDouble(v); break;
				case "keep": cfg.keep = Double.parseDouble(v); break;
				case "augment": cfg.augment = Boolean.parseBoolean(v); break;
				case "teacherscale": cfg.teacherScale = Double.parseDouble(v); break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (NumberFormatException e) {
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (teacherPath == null || out == null) {
			System.out.println("Uso: NnueDistiller teacher=PESI out=FILE.nnue [hidden=64] [l2=16] [games=600] [epochs=20] [batch=256] [lr=0.002] [seed=1]"
					+ " [greedy=0.5] [keep=0.5] [augment=true] [teacherscale=1000]");
			System.exit(2);
		}
		if (cfg.hidden < 1 || cfg.hidden > NnueNetwork.MAX_HIDDEN || cfg.l2 < 1 || cfg.l2 > NnueNetwork.MAX_L2 || cfg.batch < 1 || cfg.epochs < 1 || cfg.games < 1) {
			System.out.println("parametri fuori intervallo (hidden 1.." + NnueNetwork.MAX_HIDDEN + ", l2 1.." + NnueNetwork.MAX_L2 + ", batch/epochs/games >= 1)");
			System.exit(2);
		}
		final Evaluator teacher = EvaluatorLoader.load(teacherPath);
		System.out.println("maestro: " + EvaluatorLoader.describe(teacher) + " (" + teacherPath + ")");
		final long t0 = System.currentTimeMillis();
		final Data d = generate(teacher, cfg, System.out);
		System.out.printf("generazione: %.1f s%n", (System.currentTimeMillis() - t0) / 1000.0);
		final long t1 = System.currentTimeMillis();
		final Result r = train(d, cfg, System.out);
		System.out.printf("addestramento: %.1f s%n", (System.currentTimeMillis() - t1) / 1000.0);
		final Quantized q = quantize(r.net);
		if (q.clipped > 0) {
			System.out.println("ATTENZIONE: " + q.clipped + " pesi fuori dall'intervallo e saturati nella quantizzazione: la rete scritta differisce da quella addestrata.");
		}
		final int[][] sp = split(d, cfg.valFraction, cfg.seed);
		System.out.println("held-out (rete quantizzata): " + evaluate(q.net, d, sp[1].length > 0 ? sp[1] : sp[0]));
		System.out.println("train    (rete quantizzata): " + evaluate(q.net, d, sp[0]));
		new NnueEvaluator(q.net).save(out);
		System.out.println("scritto " + out + " (" + q.net.describe() + ")");
	}
}
