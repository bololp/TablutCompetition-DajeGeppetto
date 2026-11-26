package it.unibo.ai.didattica.competition.tablut.training;

import java.util.Arrays;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalHistory;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalNnueNetwork;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;

/**
 * Versione in {@code double} di {@link TemporalNnueNetwork}, con passata avanti/indietro scritte a mano (verificate con differenze finite da
 * {@code TemporalTrainerCheck}). Stessa struttura di {@link NnueDistiller.FloatNet}, con piu' righe nel trasformatore di feature: le 243 di
 * posizione, poi quelle temporali ({@link TemporalHistory}), tutte nello STESSO spazio di righe, quindi l'accumulatore e il gradiente non
 * distinguono i due tipi:
 * <pre>
 *   h[j] = ftB[j] + somma_{f in posizione} W[f][j] + somma_{g in storia} W[243 + g][j];   a = clamp(h, 0, 1);  poi due strati densi (testa per lato al tratto)
 * </pre>
 * Layout di {@code p}: {@code W[(243 + K*488) * H]}, {@code ftB[H]}, poi per ciascun lato {@code l1W[L2*H], l1B[L2], l2W[L2], l2B[1]} (come FloatNet).
 */
public final class TemporalFloatNet {

	public final int hidden, l2, hist, rows;
	public final double[] p;
	private final int oFtB, oRole, roleSize;

	public TemporalFloatNet(int hidden, int l2, int hist) {
		if (hist < 1 || hist > TemporalHistory.MAX_AGES) {
			throw new IllegalArgumentException("hist=" + hist);
		}
		this.hidden = hidden;
		this.l2 = l2;
		this.hist = hist;
		this.rows = NnueNetwork.NUM_FEATURES + TemporalHistory.numFeatures(hist);
		this.oFtB = rows * hidden;
		this.oRole = oFtB + hidden;
		this.roleSize = l2 * hidden + l2 + l2 + 1;
		this.p = new double[oRole + NnueNetwork.ROLES * roleSize];
	}

	public int numParams() {
		return p.length;
	}

	/** Prima riga temporale, in {@code p}. */
	public int temporalStart() {
		return NnueNetwork.NUM_FEATURES * hidden;
	}

	/** Fine delle righe temporali in {@code p} (= inizio di {@code ftB}). */
	public int temporalEnd() {
		return oFtB;
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
		final int staticEnd = NnueNetwork.NUM_FEATURES * hidden;
		for (int i = 0; i < staticEnd; i++) {
			p[i] = rnd.nextGaussian() * 0.15;
		}
		// le righe temporali partono a zero: all'inizio la rete e' la rete di sola posizione, la storia entra solo se riduce l'errore
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

	// ====================================================================== conversioni

	/** Rete in virgola mobile equivalente a una di sola posizione, con il blocco temporale a ZERO: valuta come {@code f} per qualunque storia. */
	public static TemporalFloatNet fromStatic(NnueDistiller.FloatNet f, int hist) {
		final TemporalFloatNet t = new TemporalFloatNet(f.hidden, f.l2, hist);
		final int H = f.hidden, staticEnd = NnueNetwork.NUM_FEATURES * H;
		System.arraycopy(f.p, 0, t.p, 0, staticEnd);
		System.arraycopy(f.p, staticEnd, t.p, t.oFtB, H);
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			System.arraycopy(f.p, f.offL1W(r), t.p, t.offL1W(r), t.roleSize);
		}
		return t;
	}

	/** La parte di sola posizione (le righe temporali sono scartate). */
	public NnueDistiller.FloatNet toStatic() {
		final NnueDistiller.FloatNet f = new NnueDistiller.FloatNet(hidden, l2);
		final int staticEnd = NnueNetwork.NUM_FEATURES * hidden;
		System.arraycopy(p, 0, f.p, 0, staticEnd);
		System.arraycopy(p, oFtB, f.p, staticEnd, hidden);
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			System.arraycopy(p, offL1W(r), f.p, f.offL1W(r), roleSize);
		}
		return f;
	}

	/** La rete in virgola mobile equivalente a una quantizzata ({@code quantize} di questa restituisce ESATTAMENTE {@code q}). */
	public static TemporalFloatNet fromQuantized(TemporalNnueNetwork q) {
		final TemporalFloatNet t = fromStatic(NnueDistiller.FloatNet.fromQuantized(q.base()), q.hist());
		final short[] w = q.temporalWeights();
		final int start = t.temporalStart();
		for (int i = 0; i < w.length; i++) {
			t.p[start + i] = (double) w[i] / NnueNetwork.QA;
		}
		return t;
	}

	public static final class Quantized {
		public final TemporalNnueNetwork net;
		/** Pesi non rappresentabili in {@code short} e saturati (0 = nessuna perdita). */
		public final int clipped;

		Quantized(TemporalNnueNetwork net, int clipped) {
			this.net = net;
			this.clipped = clipped;
		}
	}

	public Quantized quantize() {
		final NnueDistiller.Quantized base = NnueDistiller.quantize(toStatic());
		final short[] tw = new short[TemporalHistory.numFeatures(hist) * hidden];
		int clipped = base.clipped;
		final int start = temporalStart();
		for (int i = 0; i < tw.length; i++) {
			final double v = Math.rint(p[start + i] * NnueNetwork.QA);
			if (v > Short.MAX_VALUE || v < Short.MIN_VALUE) {
				clipped++;
			}
			tw[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, (long) v));
		}
		return new Quantized(new TemporalNnueNetwork(base.net, hist, tw), clipped);
	}

	// ====================================================================== passata avanti / indietro

	/** Buffer di lavoro di una passata (un thread). */
	public static final class Ws {
		final double[] h, a, u, c, da;
		final int[] act;
		int nAct;

		public Ws(TemporalFloatNet n) {
			h = new double[n.hidden];
			a = new double[n.hidden];
			da = new double[n.hidden];
			u = new double[n.l2];
			c = new double[n.l2];
			act = new int[81 + 3 * n.hist];
		}
	}

	private static double clamp01(double x) {
		return x < 0 ? 0 : (x > 1 ? 1 : x);
	}

	/**
	 * Logit {@code z} (il valore e' {@code tanh(z)}) di una posizione con la sua finestra di storia.
	 *
	 * @param map  simmetria applicata sia alla scacchiera sia alle mosse della finestra ({@code BoardSymmetry.MAP[s]}), o {@code null}
	 * @param recs record della finestra, il piu' recente per primo, da {@code roff}; {@code rlen} = quanti se ne usano
	 */
	public static double forward(TemporalFloatNet n, byte[] boards, int off, int role, int[] map, int[] recs, int roff, int rlen, Ws ws) {
		final int H = n.hidden, L = n.l2;
		ws.nAct = 0;
		for (int sq = 0; sq < BoardConstants.NUM_SQUARES; sq++) {
			final int piece = boards[off + sq];
			if (piece != BoardConstants.EMPTY) {
				ws.act[ws.nAct++] = NnueNetwork.featureIndex(piece, map == null ? sq : map[sq]);
			}
		}
		if (rlen > 0) {
			final int k = TemporalHistory.activeFeatures(recs, roff, rlen, n.hist, map, ws.act, ws.nAct);
			for (int i = 0; i < k; i++) {
				ws.act[ws.nAct + i] += NnueNetwork.NUM_FEATURES;
			}
			ws.nAct += k;
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
	public static void backward(TemporalFloatNet n, int role, double dz, Ws ws, double[] g) {
		final int H = n.hidden, L = n.l2;
		final double[] p = n.p;
		g[n.offL2B(role)] += dz;
		Arrays.fill(ws.da, 0, H, 0.0);
		for (int i = 0; i < L; i++) {
			g[n.offL2W(role) + i] += dz * ws.c[i];
			final double dc = dz * p[n.offL2W(role) + i];
			if (ws.u[i] > 0 && ws.u[i] < 1) {
				g[n.offL1B(role) + i] += dc;
				final int w = n.offL1W(role) + i * H;
				for (int j = 0; j < H; j++) {
					g[w + j] += dc * ws.a[j];
					ws.da[j] += dc * p[w + j];
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

	// ====================================================================== saturazione all'intervallo quantizzabile

	private static final double FT_LIMIT = 32767.0 / NnueNetwork.QA - 0.01, W_LIMIT = 32767.0 / NnueNetwork.QW - 0.01, BIAS_LIMIT = 1000.0;

	/** Satura i pesi all'intervallo rappresentabile in {@code short} dopo la quantizzazione (come {@code NnueTrainer.clampToQuantRange}). */
	public void clampToQuantRange() {
		final int ftEnd = oFtB + hidden;
		for (int i = 0; i < ftEnd; i++) {
			p[i] = Math.max(-FT_LIMIT, Math.min(FT_LIMIT, p[i]));
		}
		for (int r = 0; r < NnueNetwork.ROLES; r++) {
			final int w1 = offL1W(r), w2 = offL2W(r);
			for (int i = 0; i < l2 * hidden; i++) {
				p[w1 + i] = Math.max(-W_LIMIT, Math.min(W_LIMIT, p[w1 + i]));
			}
			for (int i = 0; i < l2; i++) {
				p[w2 + i] = Math.max(-W_LIMIT, Math.min(W_LIMIT, p[w2 + i]));
				final int b1 = offL1B(r) + i;
				p[b1] = Math.max(-BIAS_LIMIT, Math.min(BIAS_LIMIT, p[b1]));
			}
			final int b2 = offL2B(r);
			p[b2] = Math.max(-BIAS_LIMIT, Math.min(BIAS_LIMIT, p[b2]));
		}
	}
}
