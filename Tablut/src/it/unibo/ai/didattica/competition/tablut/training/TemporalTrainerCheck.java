package it.unibo.ai.didattica.competition.tablut.training;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PrintStream;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalHistory;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalNnueNetwork;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Test dell'addestramento temporale: storie ricavate dal dataset contro la verita', bersagli TD(lambda), gradiente contro differenze finite,
 * conversioni float/quantizzato, un esperimento controllato (etichetta che SOLO la storia puo' predire) e il CLI.
 * <p>
 * Uso: {@code TemporalTrainerCheck [seed]}.
 */
public class TemporalTrainerCheck {

	private static int failures = 0;
	private static long checks = 0;

	private static void check(boolean cond, String msg) {
		checks++;
		if (!cond) {
			failures++;
			if (failures <= 30) {
				System.out.println("FAIL - " + msg);
			}
		}
	}

	private static final int[] BUF = new int[FastState.MAX_MOVES];

	/** Etichettatore: dato (partita, indice nella partita, posizione, ultima mossa) restituisce {valore, valido}. */
	interface Labeler {
		double value(FastState pos, int lastFrom, int lastTo, int lastCaps, int lastType);
	}

	/** Dataset di {@code games} partite casuali con la verita' delle mosse. Le partite finite hanno outcome vero; {@code gap}: ogni tanto un campione e' saltato. */
	static final class Synth {
		NnueDataset d;
		int[] trueRec; // record della mossa che ha portato al campione (0 = primo della partita)
		int[] trueRecPrev;
	}

	static Synth synth(Random rnd, int games, Labeler lab, boolean outcomes, int gapEvery) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final int[] recs = new int[games * 200];
		int n = 0;
		final UndoInfo u = new UndoInfo();
		for (int g = 0; g < games; g++) {
			final FastState fs = new FastState();
			final int start = rnd.nextInt(3); // le prime mosse non sono registrate (come nei dati di auto-gioco)
			int lastRec = 0, lastFrom = 0, lastTo = 0, lastCaps = 0, lastType = 0;
			final int plies = 10 + rnd.nextInt(40);
			final java.util.List<Integer> idx = new java.util.ArrayList<>();
			int ply = 0;
			int winner = FastState.WINNER_NONE;
			int sample = 0;
			for (; ply < plies; ply++) {
				final int m0 = fs.generateMoves(BUF);
				if (m0 == 0) {
					break;
				}
				final int m = BUF[rnd.nextInt(m0)];
				fs.makeMove(m, u);
				lastFrom = Move.getFrom(m);
				lastTo = Move.getTo(m);
				lastCaps = u.capturedCount();
				lastType = fs.board[lastTo] == BoardConstants.WHITE ? 0 : fs.board[lastTo] == BoardConstants.BLACK ? 1 : 2;
				lastRec = TemporalHistory.record(fs.board[lastTo], lastFrom, lastTo, lastCaps);
				if (fs.winner != FastState.WINNER_NONE) {
					winner = fs.winner;
					ply++;
					break;
				}
				if (ply + 1 >= start) {
					sample++;
					if (gapEvery > 0 && sample % gapEvery == 0) {
						continue; // campione mancante: la catena si interrompe
					}
					idx.add(n);
					recs[n++] = lastRec;
					final int role = fs.turn == BoardConstants.BLACK_TURN ? 1 : 0;
					final double v = lab == null ? 0 : lab.value(fs, lastFrom, lastTo, lastCaps, lastType);
					final boolean outValid = outcomes && winner == FastState.WINNER_NONE && false; // si decide dopo: vedi sotto
					b.add(fs.board, 0, role, 0, outValid, v, lab != null, g, ply + 1, -1, NnueDataset.SRC_SELFPLAY);
				}
			}
		}
		final Synth s = new Synth();
		s.d = b.build();
		s.trueRec = java.util.Arrays.copyOf(recs, n);
		return s;
	}

	/** Dataset con partite complete etichettate: tre ply prima della fine e' noto l'esito. Serve a provare i bersagli TD. */
	static NnueDataset finishedGames(Random rnd, int games) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final UndoInfo u = new UndoInfo();
		int made = 0;
		while (made < games) {
			final FastState fs = new FastState();
			final java.util.List<byte[]> boards = new java.util.ArrayList<>();
			final java.util.List<Integer> roles = new java.util.ArrayList<>();
			int winner = FastState.WINNER_NONE;
			for (int ply = 0; ply < 300; ply++) {
				final int n = fs.generateMoves(BUF);
				if (n == 0) {
					break;
				}
				fs.makeMove(BUF[rnd.nextInt(n)], u);
				if (fs.winner != FastState.WINNER_NONE) {
					winner = fs.winner;
					break;
				}
				boards.add(fs.board.clone());
				roles.add(fs.turn == BoardConstants.BLACK_TURN ? 1 : 0);
			}
			if (winner != FastState.WINNER_WHITE && winner != FastState.WINNER_BLACK || boards.size() < 4) {
				continue;
			}
			final int out = winner == FastState.WINNER_WHITE ? 1 : -1;
			for (int i = 0; i < boards.size(); i++) {
				final double v = Math.max(-1, Math.min(1, rnd.nextGaussian() * 0.4));
				b.add(boards.get(i), 0, roles.get(i), out, true, v, true, made, i + 1, boards.size() - i, NnueDataset.SRC_SELFPLAY);
			}
			made++;
		}
		return b.build();
	}

	// ====================================================================== storie dal dataset

	private static void testHistories(Random rnd) {
		// partite vere registrate per intero (gapEvery=0): la finestra di ogni campione e' la verita'
		final Synth s = synth(rnd, 60, null, false, 0);
		final int K = 5;
		final TemporalData td = TemporalData.build(s.d, K);
		int full = 0, partial = 0;
		for (int i = 0; i < s.d.n; i++) {
			int expectAvail = 0;
			// la verita': i record precedenti nella stessa partita, in ordine inverso, finche' ci sono campioni consecutivi
			for (int a = 0; a < K; a++) {
				final int j = i - a;
				if (j < 1 || s.d.game(j) != s.d.game(i) || s.d.ply(j) != s.d.ply(j - 1) + 1 || s.d.game(j - 1) != s.d.game(j)) {
					break;
				}
				check(td.rec[i * K + a] == s.trueRec[j], "campione " + i + " eta' " + a + ": " + td.rec[i * K + a] + " vs vero " + s.trueRec[j]);
				expectAvail++;
			}
			check(td.avail[i] == expectAvail, "avail del campione " + i + ": " + td.avail[i] + " vs " + expectAvail);
			for (int a = expectAvail; a < K; a++) {
				check(td.rec[i * K + a] == 0, "oltre la finestra i record sono 0");
			}
			if (td.avail[i] == K) {
				full++;
			} else {
				partial++;
			}
		}
		check(full > 100 && partial > 0, "ci devono essere finestre complete (" + full + ") e parziali (" + partial + ")");
		// con campioni mancanti la catena si interrompe: nessuna finestra attraversa un buco
		final Synth gap = synth(rnd, 40, null, false, 4);
		final TemporalData tg = TemporalData.build(gap.d, 4);
		for (int i = 0; i < gap.d.n; i++) {
			for (int a = 0; a < tg.avail[i]; a++) {
				final int j = i - a;
				check(j >= 1 && gap.d.game(j - 1) == gap.d.game(i) && gap.d.ply(j) == gap.d.ply(j - 1) + 1, "la finestra non attraversa un buco (campione " + i + ")");
			}
		}
		boolean threw = false;
		try {
			TemporalData.build(s.d, 0);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "K=0 rifiutato");
	}

	// ====================================================================== bersagli TD(lambda)

	private static void testReturns(Random rnd) {
		final NnueDataset d = finishedGames(rnd, 12);
		final NnueTrainer.Config tc = new NnueTrainer.Config();
		for (double lam : new double[] { 0.0, 0.3, 0.5, 1.0 }) {
			tc.lambda = lam;
			final float[] mc = TemporalData.lambdaReturn(d, lam, 1.0);
			for (int i = 0; i < d.n; i++) {
				check(Math.abs(mc[i] - (float) NnueTrainer.target(tc, d, i)) < 1e-6, "mu=1 coincide con NnueTrainer.target (lambda " + lam + ", campione " + i + ")");
			}
		}
		// mu = 0, lambda = 0: il bersaglio e' il valore del successore; l'ultimo e' l'esito
		final float[] one = TemporalData.lambdaReturn(d, 0.0, 0.0);
		for (int i = 0; i < d.n; i++) {
			final float exp = TemporalData.consecutive(d, i) ? d.value(i + 1) : d.outcome(i);
			check(Math.abs(one[i] - exp) < 1e-6, "mu=0 lambda=0: successore (" + i + ")");
		}
		// calcolo a mano su una catena di 4 posizioni: V = 0.2, -0.4, 0.6, 0.0; esito +1; mu = 0.5, lambda = 0.25
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final FastState fs = new FastState();
		final double[] vs = { 0.2, -0.4, 0.6, 0.0 };
		final UndoInfo u = new UndoInfo();
		for (int i = 0; i < 4; i++) {
			b.add(fs.board, 0, fs.turn == BoardConstants.BLACK_TURN ? 1 : 0, 1, true, vs[i], true, 0, i, 4 - i, NnueDataset.SRC_SELFPLAY);
			final int n = fs.generateMoves(BUF);
			fs.makeMove(BUF[0], u);
		}
		final float[] h = TemporalData.lambdaReturn(b.build(), 0.25, 0.5);
		final double g3 = 1.0, g2 = 0.5 * 0.0 + 0.5 * g3, g1 = 0.5 * 0.6 + 0.5 * g2, g0 = 0.5 * -0.4 + 0.5 * g1;
		final double[] exp = { 0.25 * 0.2 + 0.75 * g0, 0.25 * -0.4 + 0.75 * g1, 0.25 * 0.6 + 0.75 * g2, 0.25 * 0.0 + 0.75 * g3 };
		for (int i = 0; i < 4; i++) {
			check(Math.abs(h[i] - exp[i]) < 1e-6, "TD(lambda) a mano, campione " + i + ": " + h[i] + " vs " + exp[i]);
		}
		// senza esito: l'ultimo si ancora al proprio valore di ricerca
		final NnueDataset.Builder b2 = new NnueDataset.Builder();
		final FastState f2 = new FastState();
		for (int i = 0; i < 2; i++) {
			b2.add(f2.board, 0, f2.turn == BoardConstants.BLACK_TURN ? 1 : 0, 0, false, i == 0 ? 0.5 : -0.5, true, 0, i, -1, NnueDataset.SRC_SELFPLAY);
			f2.generateMoves(BUF);
			f2.makeMove(BUF[0], u);
		}
		final float[] w = TemporalData.lambdaReturn(b2.build(), 0.5, 0.0);
		check(Math.abs(w[1] + 0.5f) < 1e-6 && Math.abs(w[0] - (0.5 * 0.5 + 0.5 * -0.5)) < 1e-6, "senza esito: bootstrap sul valore di ricerca");
		boolean threw = false;
		try {
			TemporalData.lambdaReturn(d, 1.5, 0);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "lambda fuori range rifiutato");
		// targets(): tdLambda=1 delega a NnueTrainer.target; < 1 usa il ritorno
		final NnueTemporalTrainer.Config cfg = new NnueTemporalTrainer.Config();
		cfg.lambda = 0.4;
		cfg.tdLambda = 1.0;
		tc.lambda = 0.4;
		final float[] t1 = NnueTemporalTrainer.targets(cfg, d, false);
		for (int i = 0; i < d.n; i++) {
			check(t1[i] == (float) NnueTrainer.target(tc, d, i), "targets con tdlambda=1 == NnueTrainer.target");
		}
		cfg.tdLambda = 0.0;
		final float[] t0 = NnueTemporalTrainer.targets(cfg, d, false);
		final float[] ref0 = TemporalData.lambdaReturn(d, 0.4, 0.0);
		check(java.util.Arrays.equals(t0, ref0), "targets con tdlambda=0 == lambdaReturn");
	}

	// ====================================================================== modello float

	private static TemporalFloatNet randomNet(Random rnd, int H, int L, int K) {
		final TemporalFloatNet n = new TemporalFloatNet(H, L, K);
		n.init(rnd);
		for (int i = n.temporalStart(); i < n.temporalEnd(); i++) {
			n.p[i] = rnd.nextGaussian() * 0.1;
		}
		return n;
	}

	private static void testGradient(Random rnd) {
		final int H = 12, L = 6, K = 3;
		final NnueDataset d = finishedGames(rnd, 3);
		final TemporalData td = TemporalData.build(d, K);
		final TemporalFloatNet net = randomNet(rnd, H, L, K);
		final TemporalFloatNet.Ws ws = new TemporalFloatNet.Ws(net);
		int tested = 0, worst = 0;
		double maxRel = 0;
		for (int t = 0; t < 40; t++) {
			final int i = rnd.nextInt(d.n);
			final int role = d.role(i);
			final int[] map = BoardSymmetry.MAP[rnd.nextInt(BoardSymmetry.COUNT)];
			final int len = rnd.nextInt(td.avail[i] + 1);
			final double target = rnd.nextDouble() * 2 - 1;
			final double[] g = new double[net.numParams()];
			final double z = TemporalFloatNet.forward(net, d.boardArray(i), d.boardOffset(i), role, map, td.rec, i * K, len, ws);
			final double y = Math.tanh(z);
			TemporalFloatNet.backward(net, role, 2 * (y - target) * (1 - y * y), ws, g);
			// differenze finite su un campione di parametri (tutti quelli con gradiente != 0 piu' qualche zero)
			for (int probe = 0; probe < 60; probe++) {
				int k;
				if (probe < 45) { // parametri con gradiente diverso da zero
					int tries = 0;
					do {
						k = rnd.nextInt(g.length);
					} while (g[k] == 0 && ++tries < 5000);
				} else {
					k = rnd.nextInt(g.length);
				}
				final double eps = 1e-6, old = net.p[k];
				net.p[k] = old + eps;
				final double yp = Math.tanh(TemporalFloatNet.forward(net, d.boardArray(i), d.boardOffset(i), role, map, td.rec, i * K, len, ws));
				net.p[k] = old - eps;
				final double ym = Math.tanh(TemporalFloatNet.forward(net, d.boardArray(i), d.boardOffset(i), role, map, td.rec, i * K, len, ws));
				net.p[k] = old;
				final double num = ((yp - target) * (yp - target) - (ym - target) * (ym - target)) / (2 * eps);
				final double err = Math.abs(num - g[k]) / Math.max(1e-6, Math.abs(num) + Math.abs(g[k]));
				// accanto a un ginocchio della clipped ReLU la derivata numerica non e' definita: si ignorano gli scarti isolati e si conta
				if (err > 1e-4) {
					worst++;
				}
				maxRel = Math.max(maxRel, Math.min(err, 1));
				tested++;
			}
		}
		check(worst <= tested / 100, "gradiente vs differenze finite: " + worst + "/" + tested + " scarti (errore relativo massimo " + maxRel + ")");
		// il gradiente tocca le righe temporali
		final int i0 = 0;
		final double[] g = new double[net.numParams()];
		TemporalFloatNet.forward(net, d.boardArray(i0), d.boardOffset(i0), d.role(i0), null, td.rec, 0, 0, ws);
		TemporalFloatNet.backward(net, d.role(i0), 1.0, ws, g);
		double sumT = 0;
		for (int k = net.temporalStart(); k < net.temporalEnd(); k++) {
			sumT += Math.abs(g[k]);
		}
		check(sumT == 0, "finestra vuota: nessun gradiente sulle righe temporali");
		int k2 = 0;
		for (int i = 0; i < d.n; i++) {
			if (td.avail[i] >= 2) {
				k2 = i;
				break;
			}
		}
		java.util.Arrays.fill(g, 0);
		TemporalFloatNet.forward(net, d.boardArray(k2), d.boardOffset(k2), d.role(k2), null, td.rec, k2 * K, td.avail[k2], ws);
		TemporalFloatNet.backward(net, d.role(k2), 1.0, ws, g);
		sumT = 0;
		for (int k = net.temporalStart(); k < net.temporalEnd(); k++) {
			sumT += Math.abs(g[k]);
		}
		check(sumT > 0, "finestra piena: il gradiente arriva alle righe temporali");
	}

	private static void testConversions(Random rnd) {
		final int H = 16, L = 8, K = 4;
		final NnueDataset d = finishedGames(rnd, 6);
		final TemporalData td = TemporalData.build(d, K);
		// una FloatNet di sola posizione e la sua estensione temporale a zero valutano allo stesso modo con qualunque storia
		final NnueDistiller.FloatNet f = new NnueDistiller.FloatNet(H, L);
		f.init(rnd);
		final TemporalFloatNet t = TemporalFloatNet.fromStatic(f, K);
		final NnueDistiller.Ws ws0 = new NnueDistiller.Ws(H, L);
		final TemporalFloatNet.Ws ws1 = new TemporalFloatNet.Ws(t);
		for (int i = 0; i < d.n; i++) {
			final double a = NnueDistiller.forward(f, d.boardArray(i), d.boardOffset(i), d.role(i), null, ws0);
			final double b = TemporalFloatNet.forward(t, d.boardArray(i), d.boardOffset(i), d.role(i), null, td.rec, i * K, td.avail[i], ws1);
			check(Math.abs(a - b) < 1e-12, "fromStatic: stessa uscita della rete di sola posizione");
		}
		// quantizzazione: float -> quantizzata -> float -> quantizzata e' un punto fisso; la rete di base e' quella quantizzata dalla rete statica
		final TemporalFloatNet r = randomNet(rnd, H, L, K);
		final TemporalFloatNet.Quantized q = r.quantize();
		check(q.clipped == 0, "nessun peso saturato");
		final TemporalFloatNet back = TemporalFloatNet.fromQuantized(q.net);
		check(back.quantize().net.equals(q.net), "quantizza -> float -> quantizza e' un punto fisso");
		check(q.net.base().equals(NnueDistiller.quantize(r.toStatic()).net), "la parte di base e' la quantizzazione della parte statica");
		// float vs quantizzata: vicine
		final TemporalNnueNetwork.Scratch sc = new TemporalNnueNetwork.Scratch(q.net);
		final byte[] board = new byte[81];
		double worst = 0;
		final TemporalFloatNet.Ws ws = new TemporalFloatNet.Ws(r);
		for (int i = 0; i < d.n; i++) {
			d.copyBoard(i, board, 0);
			final double yq = q.net.value(board, d.role(i), td.rec, i * K, td.avail[i], sc);
			final double yf = Math.tanh(TemporalFloatNet.forward(r, d.boardArray(i), d.boardOffset(i), d.role(i), null, td.rec, i * K, td.avail[i], ws));
			worst = Math.max(worst, Math.abs(yq - yf));
		}
		check(worst < 0.08, "float vs quantizzata: scarto massimo " + worst);
		// partenza da una NNUE di sola posizione: la rete temporale ha base IDENTICA e blocco a zero
		final NnueNetwork base = NnueNetwork.random(H, L, 4);
		final TemporalFloatNet z = TemporalFloatNet.fromStatic(NnueDistiller.FloatNet.fromQuantized(base), K);
		final TemporalNnueNetwork zq = z.quantize().net;
		check(zq.base().equals(base), "fromStatic(fromQuantized(base)).quantize() ha la stessa base");
		boolean allZero = true;
		for (short v : zq.temporalWeights()) {
			allZero &= v == 0;
		}
		check(allZero, "il blocco temporale parte a zero");
	}

	// ====================================================================== esperimento controllato

	/** Etichetta che dipende SOLO dalla destinazione dell'ultima mossa (meta' alta/bassa): la posizione da sola non la determina. */
	private static final Labeler LAST_TO_HALF = (pos, f, t, c, ty) -> (t / 9) <= 3 ? 0.7 : (t / 9) >= 5 ? -0.7 : 0.0;

	private static void testLearnsHistory(Random rnd) {
		final Synth s = synth(rnd, 700, LAST_TO_HALF, false, 0);
		final int K = 2;
		final TemporalData td = TemporalData.build(s.d, K);
		final NnueTemporalTrainer.Config cfg = new NnueTemporalTrainer.Config();
		cfg.hist = K;
		cfg.hidden = 16;
		cfg.l2 = 8;
		cfg.epochs = 6;
		cfg.lr = 5e-3;
		cfg.lrEnd = 5e-4;
		cfg.lambda = 1.0;
		cfg.patience = 6;
		cfg.histDrop = 0.0;
		cfg.augment = false;
		final float[] tgt = NnueTemporalTrainer.targets(cfg, s.d, false);
		final int[][] sp = s.d.splitByGame(1, 15);
		// con storia
		final TemporalFloatNet net = new TemporalFloatNet(16, 8, K);
		net.init(new Random(3));
		final NnueTemporalTrainer.Result r = NnueTemporalTrainer.train(td, tgt, sp[0], sp[1], null, null, null, cfg, net, null);
		final int[] val = NnueTemporalTrainer.usable(cfg, s.d, tgt, sp[1], false);
		final double withHist = NnueTemporalTrainer.meanLoss(r.net, td, tgt, val, true);
		final double noHist = NnueTemporalTrainer.meanLoss(r.net, td, tgt, val, false);
		// riferimento: la stessa rete addestrata senza poter vedere la storia (K=1 ma finestre sempre vuote: histDrop=1 e avail troncato non e' sufficiente, quindi si azzerano i record)
		final TemporalData blind = TemporalData.build(s.d, K);
		java.util.Arrays.fill(blind.avail, (byte) 0);
		final TemporalFloatNet net2 = new TemporalFloatNet(16, 8, K);
		net2.init(new Random(3));
		final NnueTemporalTrainer.Result r2 = NnueTemporalTrainer.train(blind, tgt, sp[0], sp[1], null, null, null, cfg, net2, null);
		final double blindLoss = NnueTemporalTrainer.meanLoss(r2.net, blind, tgt, val, false);
		System.out.printf("  controllato: con storia %.4f, senza storia (stessa rete, finestra vuota) %.4f, rete addestrata alla cieca %.4f%n", withHist, noHist, blindLoss);
		check(withHist < 0.05, "la rete temporale deve imparare l'etichetta che solo la storia determina (perdita " + withHist + ")");
		check(blindLoss > 4 * withHist, "senza storia l'etichetta non e' predicibile: " + blindLoss + " vs " + withHist);
		// quantizzata: stessa conclusione sulla rete che giochera'
		final TemporalNnueNetwork q = r.net.quantize().net;
		final NnueTemporalTrainer.Metrics mh = NnueTemporalTrainer.evaluate(q, td, tgt, val, true);
		final NnueTemporalTrainer.Metrics mn = NnueTemporalTrainer.evaluate(q, td, tgt, val, false);
		check(mh.mseTarget < 0.06 && mn.mseTarget > 3 * mh.mseTarget, "rete quantizzata: con storia " + mh.mseTarget + ", senza " + mn.mseTarget);
		// garanzia di non regressione: partendo da una rete gia' ottima non si peggiora, e con trainbase=false la parte di posizione non cambia
		final NnueTemporalTrainer.Config frozen = new NnueTemporalTrainer.Config();
		frozen.hist = K;
		frozen.hidden = 16;
		frozen.l2 = 8;
		frozen.epochs = 2;
		frozen.lambda = 1.0;
		frozen.trainBase = false;
		frozen.augment = false;
		frozen.histDrop = 0.0;
		final TemporalFloatNet fr = new TemporalFloatNet(16, 8, K);
		fr.init(new Random(9));
		final double[] before = fr.p.clone();
		NnueTemporalTrainer.train(td, tgt, sp[0], sp[1], null, null, null, frozen, fr, null);
		boolean baseSame = true, tempChanged = false;
		for (int i = 0; i < before.length; i++) {
			final boolean inT = i >= fr.temporalStart() && i < fr.temporalEnd();
			if (!inT && Math.abs(fr.p[i] - before[i]) > 1e-12) {
				baseSame = false;
			}
			if (inT && Math.abs(fr.p[i] - before[i]) > 1e-12) {
				tempChanged = true;
			}
		}
		check(baseSame, "trainbase=false: la parte di posizione resta identica");
		check(tempChanged || fr.p.length == 0, "trainbase=false: le righe temporali si allenano");
		// determinismo
		final TemporalFloatNet a = new TemporalFloatNet(16, 8, K), b = new TemporalFloatNet(16, 8, K);
		a.init(new Random(5));
		b.init(new Random(5));
		cfg.epochs = 2;
		NnueTemporalTrainer.train(td, tgt, sp[0], sp[1], null, null, null, cfg, a, null);
		NnueTemporalTrainer.train(td, tgt, sp[0], sp[1], null, null, null, cfg, b, null);
		check(java.util.Arrays.equals(a.p, b.p), "stesso seme -> stessi pesi, bit per bit");
	}

	private static void testNoRegression(Random rnd) {
		// bersaglio costante e rete di partenza che lo riproduce quasi esattamente: nessuna epoca puo' migliorare molto, la rete restituita non e' mai peggiore
		final Synth s = synth(rnd, 80, (p, f, t, c, ty) -> 0.0, false, 0);
		final TemporalData td = TemporalData.build(s.d, 2);
		final NnueTemporalTrainer.Config cfg = new NnueTemporalTrainer.Config();
		cfg.hist = 2;
		cfg.hidden = 8;
		cfg.l2 = 4;
		cfg.epochs = 3;
		cfg.lr = 0.5; // volutamente distruttivo
		cfg.lrEnd = 0.5;
		cfg.lambda = 1.0;
		final float[] tgt = NnueTemporalTrainer.targets(cfg, s.d, false);
		final TemporalFloatNet net = new TemporalFloatNet(8, 4, 2);
		net.init(new Random(1));
		java.util.Arrays.fill(net.p, net.temporalEnd() + 8, net.p.length, 0.0); // teste dense a zero: uscita 0 = bersaglio
		final double[] start = net.p.clone();
		final int[][] sp = s.d.splitByGame(1, 20);
		final NnueTemporalTrainer.Result r = NnueTemporalTrainer.train(td, tgt, sp[0], sp[1], null, null, null, cfg, net, null);
		final int[] val = NnueTemporalTrainer.usable(cfg, s.d, tgt, sp[1], false);
		final double after = 0.5 * (NnueTemporalTrainer.meanLoss(r.net, td, tgt, val, true) + NnueTemporalTrainer.meanLoss(r.net, td, tgt, val, false));
		check(Math.abs(after - r.bestVal) < 1e-9 && r.bestVal <= r.initialVal + 1e-12, "la rete restituita ha la perdita di validazione migliore, mai peggiore di quella di partenza");
		if (r.bestEpoch == 0) {
			check(java.util.Arrays.equals(net.p, start), "nessun miglioramento: rete di partenza invariata, bit per bit");
		}
	}

	// ====================================================================== CLI

	private static void testCli(Random rnd) throws Exception {
		final Synth s = synth(rnd, 120, LAST_TO_HALF, true, 0);
		final File data = File.createTempFile("tdata", ".gz"), out = File.createTempFile("tout", ".tnnue"), init = File.createTempFile("tinit", ".nnue");
		try {
			s.d.save(data.getPath());
			NnueNetwork.random(8, 4, 2).save(init.getPath());
			final ByteArrayOutputStream bo = new ByteArrayOutputStream();
			final PrintStream ps = new PrintStream(bo);
			final int rc = NnueTemporalTrainer.mainImpl(new String[] { "data=" + data.getPath(), "out=" + out.getPath(), "init=" + init.getPath(), "hist=3", "epochs=2", "valpct=20", "batch=64", "tdlambda=0.5", "lambda=1" }, ps);
			final String log = bo.toString();
			check(rc == 0, "CLI: codice di uscita " + rc + "\n" + log);
			if (rc == 0) {
				final TemporalNnueNetwork q = TemporalNnueNetwork.load(out.getPath());
				check(q.hist() == 3 && q.hidden() == 8 && q.l2() == 4, "CLI: la rete scritta ha le dimensioni attese");
				check(log.contains("rete temporale | held-out, con storia") && log.contains("partenza (sola posizione)"), "CLI: riporta rete e partenza");
			}
			check(NnueTemporalTrainer.mainImpl(new String[] { "data=" + data.getPath() }, new PrintStream(new ByteArrayOutputStream())) == 2, "CLI: argomenti mancanti -> 2");
			check(NnueTemporalTrainer.mainImpl(new String[] { "pippo=1" }, new PrintStream(new ByteArrayOutputStream())) == 2, "CLI: opzione sconosciuta -> 2");
			check(NnueTemporalTrainer.mainImpl(new String[] { "data=" + data.getPath(), "out=" + out.getPath(), "hist=99" }, new PrintStream(new ByteArrayOutputStream())) == 2, "CLI: hist=99 -> 2");
			check(NnueTemporalTrainer.mainImpl(new String[] { "data=/non/esiste.gz", "out=" + out.getPath() }, new PrintStream(new ByteArrayOutputStream())) == 1, "CLI: file mancante -> 1");
		} finally {
			data.delete();
			out.delete();
			init.delete();
		}
	}

	public static void main(String[] args) throws Exception {
		final long seed = args.length > 0 ? Long.parseLong(args[0]) : 1L;
		final Random rnd = new Random(seed);
		final long t0 = System.currentTimeMillis();
		testHistories(rnd);
		System.out.println("storie: ok (" + checks + " verifiche)");
		testReturns(rnd);
		System.out.println("bersagli TD: ok (" + checks + " verifiche)");
		testGradient(rnd);
		testConversions(rnd);
		System.out.println("gradiente e conversioni: ok (" + checks + " verifiche)");
		testLearnsHistory(rnd);
		testNoRegression(rnd);
		testCli(rnd);
		System.out.printf("TemporalTrainerCheck: %d verifiche, %d errori, %.1f s%n", checks, failures, (System.currentTimeMillis() - t0) / 1000.0);
		System.exit(failures == 0 ? 0 : 1);
	}
}
