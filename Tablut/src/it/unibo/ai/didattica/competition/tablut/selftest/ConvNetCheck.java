package it.unibo.ai.didattica.competition.tablut.selftest;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Collaudo di {@link ConvNet}, nello stile degli altri test del progetto: ogni verifica confronta il codice di produzione con un
 * <b>oracolo indipendente</b> scritto qui in modo ingenuo.
 * <ul>
 * <li><b>Oracolo forward</b>: la stessa rete calcolata in {@code double}, con cicli espliciti e controllo dei limiti
 * (nessun piano con bordo, nessun kernel fuso, nessun primo livello "sparso"). Legge i parametri dall'array piatto.</li>
 * <li><b>Gradiente</b>: differenze finite centrali dell'oracolo, su OGNI parametro, per entrambi i ruoli. Un parametro
 * a cavallo di un "angolo" della leaky-ReLU darebbe una differenza finita sbagliata: si rileva (due passi diversi non concordano)
 * e lo si scarta, contandolo (soglia: &lt; 1%).</li>
 * <li>Layout dei parametri: ogni parametro appartiene a una e una sola regione, con le dimensioni attese da una formula scritta qui.</li>
 * <li>Ruolo: il gradiente non tocca la head dell'altro ruolo.</li>
 * <li>Stato residuo: valutare A, poi B (molto diversa), poi A deve dare lo stesso valore di A su un workspace nuovo.</li>
 * <li>Simmetrie: {@link BoardSymmetry} e' un gruppo; le REGOLE ({@link AshtonRules}) sono equivarianti (condizione per usare
 * l'augmentation); {@code setCodes(sym)} coincide con {@code setBoard(transform)}.</li>
 * <li>Salvataggio (esatto al bit, rifiuto di file corrotti), copia indipendente, 4 thread su una rete condivisa.</li>
 * </ul>
 * Uso: {@code ConvNetCheck [seed]}. Esce con codice 1 se qualcosa fallisce.
 */
public class ConvNetCheck {

	static int failures = 0;
	static int checks = 0;

	static void check(boolean ok, String msg) {
		checks++;
		if (ok) {
			System.out.println("OK   - " + msg);
		} else {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	// ================================================================== posizioni di prova

	/** Posizioni NON terminali da partite casuali (turno incluso). */
	public static List<State> positions(int n, long seed) {
		Random rnd = new Random(seed);
		List<State> out = new ArrayList<>();
		while (out.size() < n) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			int steps = rnd.nextInt(80);
			for (int i = 0; i < steps; i++) {
				List<Action> mv = AshtonRules.generateMoves(s);
				if (mv.isEmpty() || AshtonRules.isTerminal(s)) {
					break;
				}
				s = AshtonRules.applyMove(s, mv.get(rnd.nextInt(mv.size())));
			}
			if (!AshtonRules.isTerminal(s) && !AshtonRules.generateMoves(s).isEmpty()) {
				out.add(s);
			}
		}
		return out;
	}

	public static byte[] codes(State s) {
		byte[] c = new byte[81];
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int k = 0; k < 9; k++) {
				Pawn p = b[r][k];
				c[r * 9 + k] = (byte) (p == Pawn.WHITE ? 1 : p == Pawn.BLACK ? 2 : p == Pawn.KING ? 3 : 0);
			}
		}
		return c;
	}

	public static float[] randomFeats(int n, Random rnd) {
		float[] f = new float[n];
		for (int i = 0; i < n; i++) {
			f[i] = rnd.nextFloat();
		}
		return f;
	}

	// ================================================================== oracolo

	static double leaky(double z) {
		return z > 0 ? z : 0.01 * z;
	}

	/** Forward ingenuo in double: stessa funzione di {@link ConvNet#forward}, nessuno dei suoi trucchi di implementazione. */
	public static double oracle(ConvNet net, double[] P, byte[] codes, double[] feats, int role) {
		final ConvNet.Config cfg = net.config();
		final int N = 9, L = cfg.conv.length;
		double[][][] in = new double[3][N][N];
		for (int i = 0; i < 81; i++) {
			if (codes[i] > 0) {
				in[codes[i] - 1][i / N][i % N] = 1.0;
			}
		}
		// livello 0
		int c0 = cfg.conv[0];
		double[][][] a = new double[c0][N][N];
		for (int o = 0; o < c0; o++) {
			for (int r = 0; r < N; r++) {
				for (int c = 0; c < N; c++) {
					double z = P[net.convB[0] + o * 81 + r * N + c];
					for (int t = 0; t < 3; t++) {
						for (int dr = -1; dr <= 1; dr++) {
							for (int dc = -1; dc <= 1; dc++) {
								int rr = r + dr, cc = c + dc;
								if (rr >= 0 && rr < N && cc >= 0 && cc < N) {
									z += P[net.convW[0] + (o * 3 + t) * 9 + (dr + 1) * 3 + (dc + 1)] * in[t][rr][cc];
								}
							}
						}
					}
					a[o][r][c] = leaky(z);
				}
			}
		}
		for (int l = 1; l < L; l++) {
			int cout = cfg.conv[l], cin = cfg.conv[l - 1];
			double[][][] b = new double[cout][N][N];
			for (int o = 0; o < cout; o++) {
				for (int r = 0; r < N; r++) {
					for (int c = 0; c < N; c++) {
						double z = P[net.convB[l] + o];
						for (int ci = 0; ci < cin; ci++) {
							for (int dr = -1; dr <= 1; dr++) {
								for (int dc = -1; dc <= 1; dc++) {
									int rr = r + dr, cc = c + dc;
									if (rr >= 0 && rr < N && cc >= 0 && cc < N) {
										z += P[net.convW[l] + (o * cin + ci) * 9 + (dr + 1) * 3 + (dc + 1)] * a[ci][rr][cc];
									}
								}
							}
						}
						b[o][r][c] = leaky(z);
					}
				}
			}
			a = b;
		}
		int nch = a.length;
		if (cfg.reduce > 0) {
			double[][][] b = new double[cfg.reduce][N][N];
			for (int r = 0; r < cfg.reduce; r++) {
				for (int y = 0; y < N; y++) {
					for (int x = 0; x < N; x++) {
						double z = P[net.redB + r];
						for (int c = 0; c < nch; c++) {
							z += P[net.redW + r * nch + c] * a[c][y][x];
						}
						b[r][y][x] = leaky(z);
					}
				}
			}
			a = b;
			nch = cfg.reduce;
		}
		double[] x = new double[nch * 81 + cfg.nFeat];
		for (int ch = 0; ch < nch; ch++) {
			for (int y = 0; y < N; y++) {
				for (int xx = 0; xx < N; xx++) {
					x[ch * 81 + y * N + xx] = a[ch][y][xx];
				}
			}
		}
		for (int i = 0; i < cfg.nFeat; i++) {
			x[nch * 81 + i] = feats[i];
		}
		for (int i = 0; i < cfg.dense.length; i++) {
			x = oracleDense(P, net.denseW[i], net.denseB[i], x, cfg.dense[i], false);
		}
		for (int j = 0; j <= cfg.head.length; j++) {
			boolean last = j == cfg.head.length;
			x = oracleDense(P, net.headW[role][j], net.headB[role][j], x, last ? 1 : cfg.head[j], last);
		}
		return x[0];
	}

	static double[] oracleDense(double[] P, int wOff, int bOff, double[] x, int out, boolean tanh) {
		double[] y = new double[out];
		for (int o = 0; o < out; o++) {
			double z = P[bOff + o];
			for (int i = 0; i < x.length; i++) {
				z += P[wOff + o * x.length + i] * x[i];
			}
			y[o] = tanh ? Math.tanh(z) : leaky(z);
		}
		return y;
	}

	public static double[] toDouble(float[] p) {
		double[] d = new double[p.length];
		for (int i = 0; i < p.length; i++) {
			d[i] = p[i];
		}
		return d;
	}

	static double[] toDouble(float[] p, int n) {
		double[] d = new double[n];
		for (int i = 0; i < n; i++) {
			d[i] = p[i];
		}
		return d;
	}

	// ================================================================== rete di prova

	/** Rete con parametri tutti NON nulli (bias e mappa posizionale compresi) e uscita lontana dalla saturazione. */
	public static ConvNet randomNet(ConvNet.Config cfg, long seed) {
		ConvNet base = new ConvNet(cfg, seed);
		float[] p = base.params();
		Random rnd = new Random(seed * 7919 + 13);
		for (int i = 0; i < p.length; i++) {
			p[i] += (float) (rnd.nextGaussian() * 0.15);
		}
		return base;
	}

	static ConvNet.Config[] configs(int nf) {
		return new ConvNet.Config[] {
				// ibrido completo
				new ConvNet.Config(new int[] { 4, 4 }, 2, nf, new int[] { 8 }, new int[] { 4 }),
				// canali dispari (percorso "coda" del kernel), nessuna riduzione, nessun denso, nessuna head nascosta
				new ConvNet.Config(new int[] { 3, 5, 3 }, 0, 0, new int[] {}, new int[] {}),
				// un solo livello conv, denso a due strati, head a due strati
				new ConvNet.Config(new int[] { 2 }, 3, nf, new int[] { 5, 3 }, new int[] { 4, 2 }),
				// due livelli, riduzione a 1 canale, head senza denso condiviso
				new ConvNet.Config(new int[] { 5, 2 }, 1, 0, new int[] {}, new int[] { 3 }) };
	}

	static long expectedParams(ConvNet.Config c) {
		int L = c.conv.length;
		long n = (long) c.conv[0] * 3 * 9 + (long) c.conv[0] * 81;
		for (int l = 1; l < L; l++) {
			n += (long) c.conv[l] * c.conv[l - 1] * 9 + c.conv[l];
		}
		if (c.reduce > 0) {
			n += (long) c.reduce * c.conv[L - 1] + c.reduce;
		}
		int in = (c.reduce > 0 ? c.reduce : c.conv[L - 1]) * 81 + c.nFeat;
		for (int d : c.dense) {
			n += (long) in * d + d;
			in = d;
		}
		for (int r = 0; r < 2; r++) {
			int hin = in;
			for (int d : c.head) {
				n += (long) hin * d + d;
				hin = d;
			}
			n += hin + 1;
		}
		return n;
	}

	// ================================================================== test

	public static void main(String[] args) throws Exception {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 12345;
		System.out.println("=== ConvNetCheck seed=" + seed + " ===");
		checkLayout();
		checkForwardVsOracle(seed);
		checkGradientVsFiniteDifferences(seed);
		checkRoleIsolationAndAccumulation(seed);
		checkNoResidualState(seed);
		checkSymmetryGroup();
		checkRulesEquivariance(seed);
		checkSetCodesVsSetBoard(seed);
		checkSaveLoad(seed);
		checkCopyIndependence(seed);
		checkConcurrency(seed);
		System.out.println("=== " + (failures == 0 ? "CONVNET CHECK OK" : "CONVNET CHECK FALLITO (" + failures + " errori)") + " (" + checks
				+ " verifiche) ===");
		if (failures > 0) {
			System.exit(1);
		}
	}

	// ------------------------------------------------------------------ layout

	static void checkLayout() {
		for (ConvNet.Config c : configs(6)) {
			ConvNet net = new ConvNet(c, 1);
			check(net.numParams() == expectedParams(c), "numParams = formula indipendente (" + c + "): " + net.numParams());
			int[] cover = new int[net.numParams()];
			int L = c.conv.length;
			mark(cover, net.convW[0], c.conv[0] * 27);
			mark(cover, net.convB[0], c.conv[0] * 81);
			for (int l = 1; l < L; l++) {
				mark(cover, net.convW[l], c.conv[l] * c.conv[l - 1] * 9);
				mark(cover, net.convB[l], c.conv[l]);
			}
			if (c.reduce > 0) {
				mark(cover, net.redW, c.reduce * c.conv[L - 1]);
				mark(cover, net.redB, c.reduce);
			}
			int in = c.denseIn();
			for (int i = 0; i < c.dense.length; i++) {
				mark(cover, net.denseW[i], in * c.dense[i]);
				mark(cover, net.denseB[i], c.dense[i]);
				in = c.dense[i];
			}
			for (int r = 0; r < 2; r++) {
				int hin = in;
				for (int j = 0; j <= c.head.length; j++) {
					int hout = j < c.head.length ? c.head[j] : 1;
					mark(cover, net.headW[r][j], hin * hout);
					mark(cover, net.headB[r][j], hout);
					hin = hout;
				}
			}
			boolean exact = true;
			for (int v : cover) {
				exact &= v == 1;
			}
			check(exact, "ogni parametro appartiene a una e una sola regione (" + c + ")");
		}
	}

	static void mark(int[] cover, int off, int size) {
		for (int i = 0; i < size; i++) {
			cover[off + i]++;
		}
	}

	// ------------------------------------------------------------------ forward

	static void checkForwardVsOracle(long seed) {
		Random rnd = new Random(seed);
		List<State> pos = positions(10, seed);
		double worst = 0;
		int n = 0;
		for (ConvNet.Config c : configs(6)) {
			ConvNet net = randomNet(c, seed + 5);
			ConvNet.Workspace ws = net.newWorkspace();
			double[] Pd = toDouble(net.params());
			double minAbs = 1, maxAbs = 0;
			for (State s : pos) {
				byte[] cd = codes(s);
				float[] f = randomFeats(c.nFeat, rnd);
				for (int role = 0; role < 2; role++) {
					ws.setBoard(s);
					double got = net.forward(ws, f, role);
					double want = oracle(net, Pd, cd, toDouble(f), role);
					worst = Math.max(worst, Math.abs(got - want));
					minAbs = Math.min(minAbs, Math.abs(want));
					maxAbs = Math.max(maxAbs, Math.abs(want));
					n++;
				}
			}
			check(maxAbs < 0.999, "uscite non saturate (|v| tra " + String.format("%.3f", minAbs) + " e " + String.format("%.3f", maxAbs)
					+ ") per " + c);
		}
		check(worst < 1e-4, "forward == oracolo double su " + n + " valutazioni (errore max " + String.format("%.2e", worst) + ")");
	}

	// ------------------------------------------------------------------ gradiente

	static void checkGradientVsFiniteDifferences(long seed) {
		Random rnd = new Random(seed + 1);
		List<State> pos = positions(2, seed + 2);
		double globalMaxRel = 0;
		long total = 0, skipped = 0, nonZero = 0;
		for (ConvNet.Config c : configs(4)) {
			ConvNet net = randomNet(c, seed + 11);
			ConvNet.Workspace ws = net.newWorkspace();
			double[] Pd = toDouble(net.params());
			int n = net.numParams();
			double cfgMaxRel = 0;
			long cfgSkipped = 0, cfgChecked = 0;
			for (State s : pos) {
				byte[] cd = codes(s);
				float[] f = randomFeats(c.nFeat, rnd);
				double[] fd = toDouble(f);
				for (int role = 0; role < 2; role++) {
					ws.setBoard(s);
					net.forward(ws, f, role);
					float[] g = new float[n];
					net.backward(ws, 1.0, g);
					for (int i = 0; i < n; i++) {
						double ana = g[i];
						double[] fdv = finiteDifference(net, Pd, i, cd, fd, role);
						total++;
						if (Math.abs(ana) > 0) {
							nonZero++;
						}
						if (fdv == null) {
							skipped++;
							cfgSkipped++;
							continue;
						}
						double num = fdv[0];
						double err = Math.abs(ana - num);
						double rel = err / Math.max(1e-3, Math.max(Math.abs(ana), Math.abs(num)));
						cfgMaxRel = Math.max(cfgMaxRel, Math.min(rel, err / 1e-3));
						cfgChecked++;
						if (err > 2e-4 + 2e-3 * Math.abs(num)) {
							check(false, "gradiente param " + i + " (role " + role + ") " + c + ": analitico " + ana + " vs differenze finite " + num);
							return;
						}
					}
				}
			}
			globalMaxRel = Math.max(globalMaxRel, cfgMaxRel);
			check(cfgSkipped * 100 < cfgChecked + cfgSkipped, "gradiente == differenze finite su " + cfgChecked
					+ " parametri-posizioni (scartati per angolo leaky: " + cfgSkipped + ") per " + c);
		}
		check(skipped * 100 < total, "angoli della leaky-ReLU: scartati " + skipped + " su " + total + " (< 1%)");
		check(nonZero > total / 3, "il test non e' vuoto: " + nonZero + " gradienti analitici non nulli su " + total);
	}

	/**
	 * Differenza finita centrale con due passi. Ritorna null se i due passi non concordano (angolo della leaky-ReLU
	 * dentro l'intervallo): la derivata non e' definita li' e il confronto non avrebbe senso.
	 */
	static double[] finiteDifference(ConvNet net, double[] Pd, int i, byte[] cd, double[] f, int role) {
		double orig = Pd[i];
		double[] hs = { 1e-5, 1e-7 };
		double[] est = new double[hs.length];
		for (int k = 0; k < hs.length; k++) {
			double h = hs[k];
			Pd[i] = orig + h;
			double up = oracle(net, Pd, cd, f, role);
			Pd[i] = orig - h;
			double dn = oracle(net, Pd, cd, f, role);
			Pd[i] = orig;
			est[k] = (up - dn) / (2 * h);
		}
		if (Math.abs(est[0] - est[1]) > 1e-5 + 1e-4 * Math.abs(est[1])) {
			return null;
		}
		return new double[] { est[1] };
	}

	static void checkRoleIsolationAndAccumulation(long seed) {
		ConvNet.Config c = configs(5)[0];
		ConvNet net = randomNet(c, seed + 3);
		ConvNet.Workspace ws = net.newWorkspace();
		Random rnd = new Random(seed);
		State s = positions(1, seed + 9).get(0);
		float[] f = randomFeats(c.nFeat, rnd);
		for (int role = 0; role < 2; role++) {
			ws.setBoard(s);
			net.forward(ws, f, role);
			float[] g = new float[net.numParams()];
			net.backward(ws, 1.0, g);
			int other = 1 - role;
			boolean otherZero = true, ownNonZero = false;
			for (int j = 0; j <= c.head.length; j++) {
				int hin = j == 0 ? c.dense[c.dense.length - 1] : c.head[j - 1];
				int hout = j < c.head.length ? c.head[j] : 1;
				for (int k = 0; k < hin * hout; k++) {
					otherZero &= g[net.headW[other][j] + k] == 0f;
					ownNonZero |= g[net.headW[role][j] + k] != 0f;
				}
				for (int k = 0; k < hout; k++) {
					otherZero &= g[net.headB[other][j] + k] == 0f;
				}
			}
			check(otherZero && ownNonZero, "ruolo " + role + ": il gradiente tocca la propria head e MAI quella dell'altro ruolo");

			// accumulo: due backward = il doppio; outGrad scala linearmente
			float[] g2 = new float[net.numParams()];
			ws.setBoard(s);
			net.forward(ws, f, role);
			net.backward(ws, 1.0, g2);
			net.backward(ws, 1.0, g2);
			float[] g3 = new float[net.numParams()];
			net.backward(ws, 0.5, g3);
			double e2 = 0, e3 = 0, mag = 0;
			for (int i = 0; i < g.length; i++) {
				e2 = Math.max(e2, Math.abs(g2[i] - 2 * g[i]));
				e3 = Math.max(e3, Math.abs(g3[i] - 0.5 * g[i]));
				mag = Math.max(mag, Math.abs(g[i]));
			}
			check(e2 < 1e-5 * Math.max(1, mag) && e3 < 1e-5 * Math.max(1, mag),
					"backward accumula (x2) e scala con outGrad (errore " + String.format("%.1e / %.1e", e2, e3) + ")");
		}
		boolean threw = false;
		try {
			net.backward(net.newWorkspace(), 1.0, new float[net.numParams()]);
		} catch (IllegalStateException e) {
			threw = true;
		}
		check(threw, "backward senza forward -> IllegalStateException");
	}

	// ------------------------------------------------------------------ stato residuo

	static void checkNoResidualState(long seed) {
		Random rnd = new Random(seed);
		List<State> pos = positions(40, seed + 4);
		boolean same = true;
		for (ConvNet.Config c : configs(7)) {
			ConvNet net = randomNet(c, seed + 8);
			ConvNet.Workspace reused = net.newWorkspace();
			float[] f = randomFeats(c.nFeat, rnd);
			for (int i = 0; i + 1 < pos.size(); i++) {
				State a = pos.get(i), b = pos.get((i * 7 + 3) % pos.size());
				ConvNet.Workspace fresh = net.newWorkspace();
				fresh.setBoard(a);
				double want = net.forward(fresh, f, 0);
				reused.setBoard(b); // una posizione diversa "sporca" il workspace
				net.forward(reused, f, 1);
				reused.setBoard(a);
				double got = net.forward(reused, f, 0);
				same &= got == want;
			}
		}
		check(same, "nessuno stato residuo: A, B, A sullo stesso workspace == A su workspace nuovo (uguaglianza esatta)");
	}

	// ------------------------------------------------------------------ simmetrie

	static void checkSymmetryGroup() {
		boolean perm = true;
		for (int s = 0; s < 8; s++) {
			boolean[] seen = new boolean[81];
			for (int i = 0; i < 81; i++) {
				seen[BoardSymmetry.MAP[s][i]] = true;
			}
			for (boolean b : seen) {
				perm &= b;
			}
		}
		check(perm, "ogni simmetria e' una permutazione delle 81 celle");
		Set<String> maps = new HashSet<>();
		for (int s = 0; s < 8; s++) {
			maps.add(Arrays.toString(BoardSymmetry.MAP[s]));
		}
		check(maps.size() == 8, "le 8 simmetrie sono distinte");
		boolean closed = true;
		for (int a = 0; a < 8; a++) {
			for (int b = 0; b < 8; b++) {
				int[] comp = new int[81];
				for (int i = 0; i < 81; i++) {
					comp[i] = BoardSymmetry.MAP[a][BoardSymmetry.MAP[b][i]];
				}
				boolean found = false;
				for (int s = 0; s < 8; s++) {
					found |= Arrays.equals(comp, BoardSymmetry.MAP[s]);
				}
				closed &= found;
			}
		}
		check(closed, "la composizione di due simmetrie e' una simmetria (gruppo chiuso, 64 coppie)");
		boolean fixCenter = true;
		for (int s = 0; s < 8; s++) {
			fixCenter &= BoardSymmetry.MAP[s][40] == 40;
		}
		check(fixCenter, "il trono (cella centrale) e' fisso per tutte le simmetrie");
	}

	static String key(State s) {
		StringBuilder sb = new StringBuilder();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				Pawn p = b[r][c];
				sb.append(p == Pawn.THRONE ? Pawn.EMPTY.toString().charAt(0) : p.toString().charAt(0)); // trono vuoto == vuoto
			}
		}
		sb.append(s.getTurn());
		return sb.toString();
	}

	/**
	 * Le regole sono equivarianti: l'immagine di una posizione dopo una mossa e' una delle posizioni raggiungibili dall'immagine.
	 * E' la condizione che rende VALIDA l'augmentation (altrimenti si insegnerebbe alla rete valori sbagliati).
	 */
	static void checkRulesEquivariance(long seed) {
		List<State> pos = positions(60, seed + 6);
		int cases = 0, bad = 0, moves = 0;
		for (State s : pos) {
			for (int sym = 1; sym < 8; sym++) {
				State t = BoardSymmetry.transform(s, sym);
				Set<String> imageOfChildren = new HashSet<>();
				List<Action> mv = AshtonRules.generateMoves(s);
				for (Action m : mv) {
					imageOfChildren.add(key(BoardSymmetry.transform(AshtonRules.applyMove(s, m), sym)));
				}
				Set<String> childrenOfImage = new HashSet<>();
				List<Action> mvT = AshtonRules.generateMoves(t);
				for (Action m : mvT) {
					childrenOfImage.add(key(AshtonRules.applyMove(t, m)));
				}
				moves += mv.size();
				cases++;
				if (mv.size() != mvT.size() || !imageOfChildren.equals(childrenOfImage)) {
					bad++;
				}
			}
		}
		check(bad == 0, "le regole Ashton sono equivarianti per le 8 simmetrie: " + cases + " (posizione, simmetria), " + moves
				+ " mosse, " + bad + " discordanze");
	}

	static void checkSetCodesVsSetBoard(long seed) {
		Random rnd = new Random(seed);
		ConvNet.Config c = configs(5)[0];
		ConvNet net = randomNet(c, seed + 2);
		ConvNet.Workspace w1 = net.newWorkspace(), w2 = net.newWorkspace();
		double worst = 0;
		for (State s : positions(20, seed + 12)) {
			float[] f = randomFeats(c.nFeat, rnd);
			for (int sym = 0; sym < 8; sym++) {
				w1.setCodes(codes(s), sym);
				w2.setBoard(BoardSymmetry.transform(s, sym));
				worst = Math.max(worst, Math.abs(net.forward(w1, f, 1) - net.forward(w2, f, 1)));
			}
		}
		check(worst < 1e-5, "setCodes(sym) == setBoard(transform(sym)) (errore " + String.format("%.1e", worst) + ")");
	}

	// ------------------------------------------------------------------ I/O

	static void checkSaveLoad(long seed) throws IOException {
		Random rnd = new Random(seed);
		File dir = new File(System.getProperty("java.io.tmpdir"));
		boolean exact = true, sameOut = true, meta = true;
		for (ConvNet.Config c : configs(9)) {
			ConvNet net = randomNet(c, seed + 21);
			File f = File.createTempFile("cnn_check", ".txt", dir);
			f.deleteOnExit();
			net.save(f.getPath(), 3, 11);
			check(ConvNet.isCnnFile(f.getPath()), "isCnnFile riconosce un file CNN");
			ConvNet.Loaded l = ConvNet.load(f.getPath());
			exact &= l.net.config().equals(c) && l.net.numParams() == net.numParams();
			for (int i = 0; i < net.numParams(); i++) {
				exact &= Float.floatToIntBits(l.net.params()[i]) == Float.floatToIntBits(net.params()[i]);
			}
			meta &= l.featureVersion == 3 && l.featureGroups == 11;
			ConvNet.Workspace a = net.newWorkspace(), b = l.net.newWorkspace();
			for (State s : positions(5, seed + 3)) {
				float[] ft = randomFeats(c.nFeat, rnd);
				a.setBoard(s);
				b.setBoard(s);
				sameOut &= net.forward(a, ft, 0) == l.net.forward(b, ft, 0);
			}
		}
		check(exact, "salva/carica: configurazione e parametri identici al bit");
		check(meta, "salva/carica: versione e gruppi di feature preservati");
		check(sameOut, "salva/carica: stesse uscite (uguaglianza esatta)");

		// file corrotti / di altro tipo
		ConvNet net = randomNet(configs(4)[0], 1);
		File good = File.createTempFile("cnn_good", ".txt", dir);
		good.deleteOnExit();
		net.save(good.getPath(), 3, 11);
		String text = new String(java.nio.file.Files.readAllBytes(good.toPath()), "UTF-8");

		check(rejects(text.substring(0, text.length() / 2), dir), "file troncato -> IOException");
		check(rejects(text.replace("TABLUT_CNN v1", "TABLUT_MLP v2"), dir), "intestazione sbagliata -> IOException");
		check(rejects(text.replaceFirst("params \\d+", "params 7"), dir), "numero di parametri incoerente con la forma -> IOException");
		check(rejects(text.replaceFirst("conv [0-9,]+", "conv x,y"), dir), "campo non numerico -> IOException");
		check(rejects("", dir), "file vuoto -> IOException");

		File mlp = File.createTempFile("mlp_check", ".txt", dir);
		mlp.deleteOnExit();
		NeuralEvaluator.randomInit(1).save(mlp.getPath());
		check(!ConvNet.isCnnFile(mlp.getPath()), "isCnnFile e' falso per un file MLP");
		check(!ConvNet.isCnnFile(new File(dir, "non_esiste_" + System.nanoTime()).getPath()), "isCnnFile e' falso per un file inesistente");
	}

	static boolean rejects(String content, File dir) throws IOException {
		File f = File.createTempFile("cnn_bad", ".txt", dir);
		f.deleteOnExit();
		try (FileWriter w = new FileWriter(f)) {
			w.write(content);
		}
		try {
			ConvNet.load(f.getPath());
			return false;
		} catch (IOException e) {
			return true;
		}
	}

	static void checkCopyIndependence(long seed) {
		ConvNet net = randomNet(configs(5)[0], seed);
		ConvNet cp = net.copy();
		float before = cp.params()[10];
		net.params()[10] += 1f;
		check(cp.params()[10] == before && cp.params() != net.params(), "copy() e' indipendente dall'originale");
	}

	// ------------------------------------------------------------------ concorrenza

	static void checkConcurrency(long seed) throws Exception {
		final ConvNet.Config c = configs(6)[0];
		final ConvNet net = randomNet(c, seed + 31);
		final List<State> pos = positions(60, seed + 32);
		final float[] f = randomFeats(c.nFeat, new Random(seed));
		final double[] ref = new double[pos.size()];
		ConvNet.Workspace w0 = net.newWorkspace();
		for (int i = 0; i < ref.length; i++) {
			w0.setBoard(pos.get(i));
			ref[i] = net.forward(w0, f, i % 2);
		}
		ExecutorService pool = Executors.newFixedThreadPool(4);
		List<Future<Boolean>> fs = new ArrayList<>();
		for (int t = 0; t < 4; t++) {
			final int tt = t;
			fs.add(pool.submit(new Callable<Boolean>() {
				@Override
				public Boolean call() {
					ConvNet.Workspace w = net.newWorkspace();
					Random r = new Random(tt);
					for (int it = 0; it < 3000; it++) {
						int i = r.nextInt(ref.length);
						w.setBoard(pos.get(i));
						if (net.forward(w, f, i % 2) != ref[i]) {
							return false;
						}
					}
					return true;
				}
			}));
		}
		boolean ok = true;
		for (Future<Boolean> fu : fs) {
			ok &= fu.get();
		}
		pool.shutdown();
		check(ok, "4 thread su una rete condivisa (12000 valutazioni): risultati identici al thread singolo");
	}
}
