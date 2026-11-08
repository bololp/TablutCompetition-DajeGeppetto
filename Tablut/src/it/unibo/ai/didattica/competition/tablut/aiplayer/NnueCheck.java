package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.zip.CRC32;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Test della rete NNUE ({@link NnueNetwork}, {@link NnueEvaluator}, {@link NnueSession}): confronta, per le stesse posizioni, la
 * valutazione RICOSTRUITA DA ZERO e quella con accumulatore AGGIORNATO PER DIFFERENZE, e le confronta con un ORACOLO indipendente.
 * <p>
 * Aritmetica a interi: il confronto e' di UGUAGLIANZA ESATTA (stesso {@code double} bit per bit), non a tolleranza. Le tolleranze compaiono
 * solo dove si confronta con un'altra funzione (la {@code tanh} esatta).
 * <p>
 * <b>Indipendenza dell'oracolo.</b> {@link Oracle} rifa' il calcolo dai pesi grezzi in aritmetica {@code long}, senza usare
 * {@code refresh/logit/valueOf} di {@link NnueNetwork}, e legge le posizioni da {@link State} giocati con il motore STORICO
 * ({@link AshtonRules}), non da {@link FastState}: se l'aggiornamento incrementale e il motore veloce sbagliassero insieme, l'oracolo lo vedrebbe.
 * <p>
 * Copre: codifica delle feature, posizione iniziale e posizioni dense, ogni mossa legale di ogni posizione (ordinaria, cattura singola e
 * multipla, Re, Re che lascia il trono, da/verso cittadelle, cattura del Re, fuga del Re), passeggiate lunghe con valutazioni rade (catene
 * lazy lunghe e crescita della pila), DFS con rami fratelli e ritorno con unmake, lato al tratto e posizioni terminali, disallineamenti
 * deliberati (la ricostruzione di ripiego), caricamento dei pesi e ogni errore di file, thread-safety dell'evaluator condiviso.
 * <p>
 * Uso: {@code NnueCheck [posizioniDense] [seed]}.
 */
public class NnueCheck {

	private static int failures = 0;
	private static long checks = 0;

	private static void fail(String msg) {
		failures++;
		if (failures <= 30) {
			System.out.println("FAIL - " + msg);
		}
	}

	private static void check(boolean cond, String msg) {
		checks++;
		if (!cond) {
			fail(msg);
		}
	}

	private static boolean same(double a, double b) {
		return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
	}

	private static int role(FastState s) {
		return s.turn == BoardConstants.BLACK_TURN ? 1 : 0;
	}

	// ====================================================================== oracolo

	/** Riscrittura indipendente della rete, in aritmetica {@code long}, dai pesi grezzi e da uno {@link State}. */
	static final class Oracle {
		final int h, l2;
		final short[] ftW, ftB;
		final short[][] l1W, l2W;
		final int[][] l1B;
		final int[] l2B;

		Oracle(NnueNetwork n) {
			h = n.hidden();
			l2 = n.l2();
			ftW = n.ftWeights();
			ftB = n.ftBiases();
			l1W = new short[][] { n.l1Weights(0), n.l1Weights(1) };
			l1B = new int[][] { n.l1Biases(0), n.l1Biases(1) };
			l2W = new short[][] { n.l2Weights(0), n.l2Weights(1) };
			l2B = new int[] { n.l2Bias(0), n.l2Bias(1) };
		}

		long logit(State st, int role) {
			final long[] acc = new long[h];
			for (int j = 0; j < h; j++) {
				acc[j] = ftB[j];
			}
			final Pawn[][] b = st.getBoard();
			for (int r = 0; r < 9; r++) {
				for (int c = 0; c < 9; c++) {
					final Pawn p = b[r][c];
					final int type = p == Pawn.WHITE ? 0 : p == Pawn.BLACK ? 1 : p == Pawn.KING ? 2 : -1;
					if (type < 0) {
						continue;
					}
					final int f = type * 81 + r * 9 + c;
					for (int j = 0; j < h; j++) {
						acc[j] += ftW[f * h + j];
					}
				}
			}
			final long[] a = new long[h];
			for (int j = 0; j < h; j++) {
				a[j] = Math.min(256L, Math.max(0L, acc[j]));
			}
			long z = l2B[role];
			for (int i = 0; i < l2; i++) {
				long u = l1B[role][i];
				for (int j = 0; j < h; j++) {
					u += (long) l1W[role][i * h + j] * a[j];
				}
				final long c = Math.min(256L, Math.max(0L, Math.floorDiv(u + 32, 64L)));
				z += l2W[role][i] * c;
			}
			return z;
		}

		static double value(long z) {
			final long zc = Math.max(-131072L, Math.min(131071L, z));
			final long t = zc + 131072L;
			final int idx = (int) (t / 1024);
			final long frac = t % 1024;
			final double lo = StrictMath.tanh((idx - 128) / 16.0);
			final double hi = StrictMath.tanh((idx + 1 - 128) / 16.0);
			return lo + (hi - lo) * (frac * (1.0 / 1024));
		}

		double eval(State st, int role, double scale) {
			return value(logit(st, role)) * scale;
		}
	}

	// ====================================================================== statistiche di copertura

	private static final class Cov {
		long plain, cap1, cap2, cap3plus, king, kingFromThrone, fromCitadel, toCitadel, kingCaptured, kingEscape, blackMoves, whiteMoves;

		@Override
		public String toString() {
			return "ordinarie " + plain + ", catture 1/2/3+ = " + cap1 + "/" + cap2 + "/" + cap3plus + ", mosse del Re " + king + " (dal trono " + kingFromThrone
					+ "), da cittadella " + fromCitadel + ", verso cittadella " + toCitadel + ", Re catturato " + kingCaptured + ", fuga del Re " + kingEscape
					+ ", muove il Bianco " + whiteMoves + " / il Nero " + blackMoves;
		}
	}

	// ====================================================================== main

	public static void main(String[] args) throws Exception {
		final int nDense = args.length > 0 ? Integer.parseInt(args[0]) : 150;
		final long seed = args.length > 1 ? Long.parseLong(args[1]) : 4242L;
		final Random rnd = new Random(seed);

		// piu' forme di rete: la principale, una piccola, e due degeneri (H=1, L2=1)
		final NnueNetwork[] nets = { NnueNetwork.random(64, 16, 11), NnueNetwork.random(8, 4, 12), NnueNetwork.random(1, 1, 13), NnueNetwork.random(3, 2, 14) };

		testFeatureIndex();
		testNumerics();
		testRegimes(nets[0], new Random(seed));
		testRegimes(nets[1], new Random(seed + 1));
		final Cov cov = new Cov();
		for (int k = 0; k < nets.length; k++) {
			final NnueEvaluator ev = new NnueEvaluator(nets[k]);
			final Oracle or = new Oracle(nets[k]);
			final int n = k == 0 ? nDense : Math.max(30, nDense / 4);
			testInitialPosition(ev, or);
			testAllMoves(ev, or, new Random(seed * 31 + k), n, cov, k == 0);
			testRandomWalks(ev, or, new Random(seed * 37 + k), k == 0 ? 12 : 4);
			testDfsSiblings(ev, new Random(seed * 41 + k), k == 0 ? 40 : 10);
			testSideToMoveAndTerminal(ev, or, new Random(seed * 43 + k));
		}
		System.out.println("Copertura delle mosse (rete principale): " + cov);
		check(cov.plain > 0 && cov.cap1 > 0 && cov.cap2 > 0, "copertura: servono mosse ordinarie, catture singole e doppie");
		check(cov.king > 0 && cov.kingFromThrone > 0, "copertura: servono mosse del Re, anche dal trono");
		check(cov.fromCitadel > 0 && cov.toCitadel > 0, "copertura: servono mosse da e verso le cittadelle");
		check(cov.kingCaptured > 0 && cov.kingEscape > 0, "copertura: servono la cattura e la fuga del Re");
		check(cov.whiteMoves > 0 && cov.blackMoves > 0, "copertura: devono muovere entrambi i lati");
		testSpecialSquareCaptures(new NnueEvaluator(nets[0]), new Oracle(nets[0]));
		testDesyncFallback(new NnueEvaluator(nets[0]), new Random(seed + 5));
		testSessionProtocolErrors(new NnueEvaluator(nets[0]));
		testWeightsIO(nets[0]);
		testNetworkValidation();
		testEvaluatorContract(nets[0], new Random(seed + 9));
		testThreadSafety(nets[0], new Random(seed + 11));

		if (failures == 0) {
			System.out.println("=== NNUE CHECK OK: " + checks + " verifiche ===");
		} else {
			System.out.println("=== NNUE CHECK: " + failures + " FALLIMENTI su " + checks + " verifiche ===");
			System.exit(1);
		}
	}

	// ====================================================================== feature e numerica

	private static void testFeatureIndex() {
		final int[] pieces = { BoardConstants.WHITE, BoardConstants.BLACK, BoardConstants.KING };
		final boolean[] seen = new boolean[NnueNetwork.NUM_FEATURES];
		for (int p : pieces) {
			for (int sq = 0; sq < 81; sq++) {
				final int f = NnueNetwork.featureIndex(p, sq);
				check(f >= 0 && f < NnueNetwork.NUM_FEATURES && !seen[f], "featureIndex(" + p + "," + sq + ") fuori intervallo o duplicato: " + f);
				if (f >= 0 && f < seen.length) {
					seen[f] = true;
				}
			}
		}
		for (boolean b : seen) {
			check(b, "esiste una feature non raggiunta: la codifica deve essere una biiezione su 3x81");
		}
		check(NnueNetwork.featureIndex(BoardConstants.WHITE, 0) == 0 && NnueNetwork.featureIndex(BoardConstants.BLACK, 0) == 81
				&& NnueNetwork.featureIndex(BoardConstants.KING, 80) == 242, "base delle feature: BIANCO 0, NERO 81, RE 162");
		final int[] bad = { BoardConstants.EMPTY, BoardConstants.THRONE_MARK, 5, -1, 127 };
		for (int p : bad) {
			boolean threw = false;
			try {
				NnueNetwork.featureIndex(p, 3);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "featureIndex con il codice pedina " + p + " dovrebbe lanciare");
		}
		for (int sq : new int[] { -1, 81, 1000 }) {
			boolean threw = false;
			try {
				NnueNetwork.featureIndex(BoardConstants.WHITE, sq);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "featureIndex con la cella " + sq + " dovrebbe lanciare");
		}
	}

	/** La tanh campionata: monotona, simmetrica, vicina a tanh, saturata; e il logit coincide con l'oracolo. */
	private static void testNumerics() {
		double prev = -2;
		double maxErr = 0;
		for (int z = -200000; z <= 200000; z += 37) {
			final double v = NnueNetwork.valueOf(z);
			check(v >= prev, "valueOf non monotona a z=" + z);
			check(same(v, Oracle.value(z)), "valueOf(" + z + ") diverso dall'oracolo");
			check(v > -1.0 && v < 1.0, "valueOf fuori da (-1,1) a z=" + z);
			prev = v;
			if (z >= -131072 && z < 131072) {
				maxErr = Math.max(maxErr, Math.abs(v - Math.tanh(z / 16384.0)));
			}
		}
		check(maxErr < 4e-4, "errore della tanh campionata troppo grande: " + maxErr);
		check(Math.abs(NnueNetwork.valueOf(0)) < 1e-12, "valueOf(0) deve essere 0");
		check(Math.abs(NnueNetwork.valueOf(100000) + NnueNetwork.valueOf(-100000)) < 1e-3, "la tanh deve essere (quasi) dispari");
		check(NnueNetwork.valueOf(Integer.MAX_VALUE) == NnueNetwork.valueOf(131071) && NnueNetwork.valueOf(Integer.MIN_VALUE) == NnueNetwork.valueOf(-131072),
				"saturazione agli estremi");
		System.out.println("tanh campionata: errore massimo " + maxErr);
	}

	/**
	 * Contro i test vuoti: la rete casuale deve far attraversare agli accumulatori TUTTI e tre i regimi della clipped ReLU e dare valori
	 * diversi a posizioni diverse; altrimenti un aggiornamento sbagliato potrebbe passare inosservato.
	 */
	private static void testRegimes(NnueNetwork net, Random rnd) {
		final NnueEvaluator ev = new NnueEvaluator(net);
		final int[] acc = new int[net.hidden()];
		long below = 0, mid = 0, above = 0;
		final List<Double> vals = new ArrayList<>();
		for (int i = 0; i < 300; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			net.refresh(fs.board, acc);
			for (int v : acc) {
				if (v < 0) {
					below++;
				} else if (v > NnueNetwork.QA) {
					above++;
				} else {
					mid++;
				}
			}
			vals.add(ev.evaluate(fs));
		}
		final double tot = below + mid + above;
		check(below / tot > 0.03 && mid / tot > 0.03 && above / tot > 0.03,
				"la rete casuale " + net.describe() + " non esercita i tre regimi della clipped ReLU: sotto " + below + ", dentro " + mid + ", sopra " + above);
		final double mean = vals.stream().mapToDouble(Double::doubleValue).average().orElse(0);
		double var = 0;
		for (double v : vals) {
			var += (v - mean) * (v - mean);
		}
		final double sd = Math.sqrt(var / vals.size());
		check(sd > 5.0, "i valori della rete casuale sono quasi costanti (sd " + sd + "): i test sarebbero poco sensibili");
		System.out.printf("regimi %s: sotto %.0f%% / dentro %.0f%% / sopra %.0f%%, sd dei valori %.1f%n", net.describe(), 100 * below / tot, 100 * mid / tot, 100 * above / tot, sd);
	}

	// ====================================================================== posizione iniziale

	private static void testInitialPosition(NnueEvaluator ev, Oracle or) {
		final State init = TestPositions.initial();
		final FastState fs = new FastState();
		final double full = ev.evaluate(fs);
		check(same(full, or.eval(init, 0, ev.outputScale)), "posizione iniziale: valutazione completa diversa dall'oracolo");
		check(same(full, ev.evaluate(init)), "evaluate(State) deve coincidere con evaluate(FastState)");
		final NnueSession ses = (NnueSession) ev.newSession(4);
		ses.reset(fs);
		check(same(ses.evaluate(0, fs), full), "posizione iniziale: la radice della sessione deve valere la valutazione completa");
		check(ses.fallbackRebuilds() == 0, "nessuna ricostruzione di ripiego alla radice");
		// la posizione iniziale e' simmetrica per le 4 rotazioni: la rete NON lo e' (pesi per cella), ma la ricostruzione e' deterministica
		check(same(ev.evaluate(new FastState()), full), "la valutazione completa deve essere deterministica");
	}

	// ====================================================================== ogni mossa legale di posizioni dense

	private static void testAllMoves(NnueEvaluator ev, Oracle or, Random rnd, int nPos, Cov cov, boolean count) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		long moves = 0;
		for (int p = 0; p < nPos; p++) {
			final State s = TestPositions.randomPosition(rnd, p % 5 == 0);
			final FastState fs = FastState.fromState(s);
			final FastState keep = FastState.copyOf(fs);
			final NnueSession ses = (NnueSession) ev.newSession(4);
			ses.reset(fs);
			final double v0 = ses.evaluate(0, fs);
			check(same(v0, ev.evaluate(fs)) && same(v0, or.eval(s, role(fs), ev.outputScale)), "posizione " + p + ": radice incoerente");
			final int n = fs.generateMoves(buf);
			for (int i = 0; i < n; i++) {
				final int m = buf[i];
				final int from = Move.getFrom(m), to = Move.getTo(m);
				final byte piece = fs.board[from];
				final boolean white = fs.turn == BoardConstants.WHITE_TURN;
				fs.makeMove(m, u);
				ses.push(1, m, fs, u);
				final State child = AshtonRules.applyMove(s, Move.toAction(m, s.getTurn()));
				final double inc = ses.evaluate(1, fs);
				final double full = ev.evaluate(fs);
				final double orc = or.eval(child, role(fs), ev.outputScale);
				check(same(inc, full), "posizione " + p + " mossa " + Move.toString(m) + ": incrementale " + inc + " != completa " + full);
				check(same(inc, orc), "posizione " + p + " mossa " + Move.toString(m) + ": incrementale " + inc + " != oracolo storico " + orc);
				moves++;
				if (count) {
					final int caps = u.capturedCount();
					if (caps == 0) {
						cov.plain++;
					} else if (caps == 1) {
						cov.cap1++;
					} else if (caps == 2) {
						cov.cap2++;
					} else {
						cov.cap3plus++;
					}
					if (piece == BoardConstants.KING) {
						cov.king++;
						if (from == BoardConstants.THRONE) {
							cov.kingFromThrone++;
						}
					}
					if (BoardConstants.IS_CITADEL[from]) {
						cov.fromCitadel++;
					}
					if (BoardConstants.IS_CITADEL[to]) {
						cov.toCitadel++;
					}
					if (fs.winner == FastState.WINNER_BLACK) {
						cov.kingCaptured++;
					}
					if (fs.winner == FastState.WINNER_WHITE) {
						cov.kingEscape++;
					}
					if (white) {
						cov.whiteMoves++;
					} else {
						cov.blackMoves++;
					}
				}
				fs.unmakeMove(m, u);
				// dopo l'unmake il frame del genitore e' ancora valido (i figli non lo toccano) e la posizione e' ripristinata
				check(same(ses.evaluate(0, fs), v0), "posizione " + p + " mossa " + Move.toString(m) + ": dopo unmake la valutazione del genitore e' cambiata");
				check(fs.samePosition(keep) && fs.zobristHash == keep.zobristHash, "posizione " + p + ": unmake non ha ripristinato lo stato");
			}
			check(ses.fallbackRebuilds() == 0, "posizione " + p + ": ricostruzioni di ripiego inattese: " + ses.fallbackRebuilds());
		}
		check(moves > 0, "nessuna mossa verificata");
	}

	// ====================================================================== passeggiate lunghe, valutazioni rade

	/**
	 * Partite casuali "avide" (preferiscono le catture) lunghe fino a 140 ply con pila iniziale di 2 frame (la pila cresce da sola).
	 * La valutazione e' richiesta ogni {@code stride} ply: con stride grande l'accumulatore si calcola in catene lazy lunghe (fino a
	 * tutta la partita). Ogni risultato e' confrontato con la valutazione completa e con l'oracolo storico.
	 */
	private static void testRandomWalks(NnueEvaluator ev, Oracle or, Random rnd, int games) {
		final int[] strides = { 1000, 1, 3, 7, 2, 5 }; // la prima partita (sempre eseguita) valuta una sola volta alla fine: catena lazy di 140 ply
		final int[] buf = new int[FastState.MAX_MOVES];
		long evals = 0, maxChain = 0;
		for (int g = 0; g < games; g++) {
			final int stride = strides[g % strides.length];
			final boolean quiet = g % 2 == 0; // le partite "quiete" scartano le mosse che terminano la partita: arrivano a 140 ply
			FastState fs = new FastState();
			State st = TestPositions.initial();
			final UndoInfo[] us = new UndoInfo[200];
			for (int i = 0; i < us.length; i++) {
				us[i] = new UndoInfo();
			}
			final NnueSession ses = (NnueSession) ev.newSession(2);
			ses.reset(fs);
			int ply = 0;
			int lastEval = 0;
			while (ply < 140 && fs.winner == FastState.WINNER_NONE) {
				final int n = fs.generateMoves(buf);
				if (n == 0) {
					break;
				}
				int m = buf[rnd.nextInt(n)];
				for (int t = 0; t < 6 && fs.tacticalValue(m) <= 0; t++) {
					m = buf[rnd.nextInt(n)];
				}
				for (int t = 0; quiet && t < 30; t++) {
					fs.makeMove(m, us[ply]);
					final boolean ends = fs.winner != FastState.WINNER_NONE;
					fs.unmakeMove(m, us[ply]);
					if (!ends) {
						break;
					}
					m = buf[rnd.nextInt(n)];
				}
				st = AshtonRules.applyMove(st, Move.toAction(m, st.getTurn()));
				fs.makeMove(m, us[ply]);
				ses.push(ply + 1, m, fs, us[ply]);
				ply++;
				final boolean terminal = fs.winner != FastState.WINNER_NONE;
				if (ply % stride == 0 || terminal || ply == 139) {
					final double inc = ses.evaluate(ply, fs);
					final double full = ev.evaluate(fs);
					check(same(inc, full), "passeggiata " + g + " (stride " + stride + ") ply " + ply + ": incrementale " + inc + " != completa " + full);
					check(same(inc, or.eval(st, role(fs), ev.outputScale)), "passeggiata " + g + " ply " + ply + ": diverso dall'oracolo storico");
					evals++;
					maxChain = Math.max(maxChain, ply - lastEval);
					lastEval = ply;
				}
			}
			check(ses.capacity() >= ply + 1, "passeggiata " + g + ": la pila non e' cresciuta (capacita' " + ses.capacity() + ", ply " + ply + ")");
			check(ses.fallbackRebuilds() == 0, "passeggiata " + g + ": ricostruzioni di ripiego inattese: " + ses.fallbackRebuilds());
		}
		check(evals > 0 && maxChain >= 100, "copertura: serve almeno una catena lazy lunga (>=100 ply), trovata " + maxChain);
		System.out.println("passeggiate: " + evals + " valutazioni, catena lazy piu' lunga " + maxChain + " ply");
	}

	// ====================================================================== DFS con rami fratelli

	private static void testDfsSiblings(NnueEvaluator ev, Random rnd, int roots) {
		final int[][] bufs = new int[6][FastState.MAX_MOVES];
		final UndoInfo[] us = new UndoInfo[6];
		for (int i = 0; i < us.length; i++) {
			us[i] = new UndoInfo();
		}
		long nodes = 0;
		for (int r = 0; r < roots; r++) {
			final FastState fs = FastState.fromState(r % 3 == 0 ? TestPositions.randomPosition(rnd) : somePlayedPosition(rnd));
			final FastState keep = FastState.copyOf(fs);
			final NnueSession ses = (NnueSession) ev.newSession(8);
			ses.reset(fs);
			final double v0 = ses.evaluate(0, fs);
			for (int mode = 0; mode < 3; mode++) { // 0 = ogni nodo, 1 = solo le foglie, 2 = nodi a caso
				nodes += dfs(fs, ses, ev, 0, 3, rnd, mode, us, bufs);
				check(fs.samePosition(keep) && fs.zobristHash == keep.zobristHash, "DFS " + r + " modo " + mode + ": posizione non ripristinata");
				check(same(ses.evaluate(0, fs), v0), "DFS " + r + " modo " + mode + ": la radice valutata dopo l'esplorazione e' cambiata");
			}
			check(ses.fallbackRebuilds() == 0, "DFS " + r + ": ricostruzioni di ripiego inattese: " + ses.fallbackRebuilds());
		}
		check(nodes > 1000, "DFS: troppo pochi nodi (" + nodes + ")");
	}

	/** Una posizione di una partita avida (non quella iniziale, se la partita e' abbastanza lunga). */
	private static State somePlayedPosition(Random rnd) {
		final List<State> ps = TestPositions.playGame(rnd, true, 60).positions;
		return ps.get(ps.size() > 4 ? 4 + rnd.nextInt(ps.size() - 4) : 0);
	}

	private static long dfs(FastState fs, NnueSession ses, NnueEvaluator ev, int ply, int depth, Random rnd, int mode, UndoInfo[] us, int[][] bufs) {
		long nodes = 1;
		final boolean evalHere = mode == 0 || (mode == 1 && depth == 0) || (mode == 2 && rnd.nextInt(3) == 0);
		double before = Double.NaN;
		if (evalHere) {
			before = ses.evaluate(ply, fs);
			check(same(before, ev.evaluate(fs)), "DFS ply " + ply + " (modo " + mode + "): incrementale " + before + " != completa " + ev.evaluate(fs));
		}
		if (depth == 0 || fs.winner != FastState.WINNER_NONE) {
			return nodes;
		}
		final int n = fs.generateMoves(bufs[ply]);
		final int limit = Math.min(n, 5);
		for (int k = 0; k < limit; k++) {
			final int m = bufs[ply][n <= 5 ? k : rnd.nextInt(n)];
			fs.makeMove(m, us[ply]);
			ses.push(ply + 1, m, fs, us[ply]);
			nodes += dfs(fs, ses, ev, ply + 1, depth - 1, rnd, mode, us, bufs);
			fs.unmakeMove(m, us[ply]);
		}
		// tornati dai figli (fratelli compresi) il frame del genitore deve ancora essere quello giusto
		final double after = ses.evaluate(ply, fs);
		check(same(after, ev.evaluate(fs)), "DFS ply " + ply + " (modo " + mode + "): dopo i figli l'incrementale non e' la valutazione completa");
		if (evalHere) {
			check(same(after, before), "DFS ply " + ply + " (modo " + mode + "): il valore del genitore e' cambiato dopo l'esplorazione dei figli");
		}
		return nodes;
	}

	// ====================================================================== lato al tratto, posizioni terminali

	private static void testSideToMoveAndTerminal(NnueEvaluator ev, Oracle or, Random rnd) {
		int differ = 0, total = 0;
		for (int i = 0; i < 80; i++) {
			final State s = TestPositions.randomPosition(rnd);
			final FastState w = FastState.fromState(s);
			final FastState b = FastState.copyOf(w);
			w.turn = BoardConstants.WHITE_TURN;
			b.turn = BoardConstants.BLACK_TURN;
			w.zobristHash = w.computeHash();
			b.zobristHash = b.computeHash();
			w.computeKeys();
			b.computeKeys();
			final double vw = ev.evaluate(w), vb = ev.evaluate(b);
			check(same(vw, or.eval(s, 0, ev.outputScale)), "lato al tratto Bianco: diverso dall'oracolo (testa 0)");
			check(same(vb, or.eval(s, 1, ev.outputScale)), "lato al tratto Nero: diverso dall'oracolo (testa 1)");
			// la sessione deve scegliere la testa giusta anche con la radice a Nero e dopo mosse che alternano il lato
			for (FastState fs : new FastState[] { w, b }) {
				final NnueSession ses = (NnueSession) ev.newSession(4);
				ses.reset(fs);
				check(same(ses.evaluate(0, fs), ev.evaluate(fs)), "radice con " + (fs == w ? "Bianco" : "Nero") + " al tratto: sessione diversa dalla completa");
				final int[] buf = new int[FastState.MAX_MOVES];
				final UndoInfo u = new UndoInfo();
				final int n = fs.generateMoves(buf);
				if (n > 0) {
					final int m = buf[rnd.nextInt(n)];
					fs.makeMove(m, u);
					ses.push(1, m, fs, u);
					check(same(ses.evaluate(1, fs), ev.evaluate(fs)), "dopo una mossa dalla radice con " + (fs == w ? "Bianco" : "Nero") + " al tratto: incoerente");
					fs.unmakeMove(m, u);
				}
			}
			total++;
			if (vw != vb) {
				differ++;
			}
		}
		check(differ > total * 0.9, "le due teste dovrebbero dare valori diversi sulla stessa scacchiera (differiscono " + differ + "/" + total + ")");

		// posizioni terminali: l'evaluator valuta la scacchiera cosi' com'e' (la ricerca decide da sola i terminali, ma i due percorsi
		// devono comunque concordare, perche' una mossa verso un terminale viene registrata con push come tutte le altre)
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		int terminals = 0;
		for (int p = 0; p < 200 && terminals < 25; p++) {
			final State s = TestPositions.randomPosition(rnd, true);
			final FastState fs = FastState.fromState(s);
			final NnueSession ses = (NnueSession) ev.newSession(3);
			ses.reset(fs);
			final int n = fs.generateMoves(buf);
			for (int i = 0; i < n; i++) {
				fs.makeMove(buf[i], u);
				ses.push(1, buf[i], fs, u);
				if (fs.winner != FastState.WINNER_NONE) {
					terminals++;
					final State child = AshtonRules.applyMove(s, Move.toAction(buf[i], s.getTurn()));
					final double inc = ses.evaluate(1, fs);
					check(same(inc, ev.evaluate(fs)), "posizione terminale (vince " + fs.winner + "): incrementale != completa");
					check(same(inc, or.eval(child, role(fs), ev.outputScale)), "posizione terminale (vince " + fs.winner + "): diverso dall'oracolo");
					if (fs.winner == FastState.WINNER_BLACK) {
						check(fs.kingSquare < 0, "re catturato: il motore veloce deve averlo rimosso (kingSquare = -1)");
					}
				}
				fs.unmakeMove(buf[i], u);
			}
			check(ses.fallbackRebuilds() == 0, "terminali: ricostruzioni di ripiego inattese");
		}
		check(terminals >= 5, "copertura: troppe poche posizioni terminali (" + terminals + ")");
	}

	// ====================================================================== catture contro trono e cittadelle (costruite a mano)

	/**
	 * Posizioni costruite a mano per le caselle speciali: la cattura di una pedina bianca contro il trono VUOTO e contro una cittadella,
	 * e la cattura di una pedina nera contro il trono OCCUPATO dal Re. Le regole le decide l'engine storico; qui si verifica che la
	 * cattura avvenga davvero (la posizione e' quella voluta) e che l'accumulatore tolga la pedina giusta.
	 */
	private static void testSpecialSquareCaptures(NnueEvaluator ev, Oracle or) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		int verified = 0;
		// (1) trono VUOTO: re lontano, bianco in (4,3) accanto al trono, un nero entra in (4,2) -> il bianco e' preso tra nero e trono
		State s = TestPositions.blank();
		s.getBoard()[4][4] = Pawn.THRONE;
		s.getBoard()[2][6] = Pawn.KING;
		s.getBoard()[4][3] = Pawn.WHITE;
		s.getBoard()[6][2] = Pawn.BLACK;
		s.getBoard()[7][7] = Pawn.BLACK;
		s.setTurn(Turn.BLACK);
		verified += checkCaptureMove(ev, or, s, Move.of(6 * 9 + 2, 4 * 9 + 2), 1, "bianco contro il trono vuoto", buf, u);
		// (2) cittadella: bianco in (0,2) accanto alla cittadella (0,3); un nero entra in (0,1) -> preso tra nero e cittadella
		s = TestPositions.blank();
		s.getBoard()[4][4] = Pawn.THRONE;
		s.getBoard()[6][6] = Pawn.KING;
		s.getBoard()[0][2] = Pawn.WHITE;
		s.getBoard()[3][1] = Pawn.BLACK;
		s.getBoard()[7][3] = Pawn.BLACK;
		s.setTurn(Turn.BLACK);
		verified += checkCaptureMove(ev, or, s, Move.of(3 * 9 + 1, 0 * 9 + 1), 1, "bianco contro una cittadella", buf, u);
		// (3) doppia cattura: bianchi in (3,3) e (5,3), neri in (2,3) e (6,3); un nero esce dal campo sinistro in (4,3) e li prende entrambi
		s = TestPositions.blank();
		s.getBoard()[4][4] = Pawn.THRONE;
		s.getBoard()[7][7] = Pawn.KING;
		s.getBoard()[3][3] = Pawn.WHITE;
		s.getBoard()[5][3] = Pawn.WHITE;
		s.getBoard()[2][3] = Pawn.BLACK;
		s.getBoard()[6][3] = Pawn.BLACK;
		s.getBoard()[4][0] = Pawn.BLACK; // (4,0) e' una cittadella: il nero sta nel suo campo e puo' uscire lungo la riga 4
		s.setTurn(Turn.BLACK);
		verified += checkCaptureMove(ev, or, s, Move.of(4 * 9 + 0, 4 * 9 + 3), 2, "doppia cattura", buf, u);
		// (4) cattura di un NERO contro il trono occupato dal Re: nero in (4,3) accanto al Re sul trono, un bianco entra in (4,2)
		s = TestPositions.blank();
		s.getBoard()[4][4] = Pawn.KING;
		s.getBoard()[4][3] = Pawn.BLACK;
		s.getBoard()[6][2] = Pawn.WHITE;
		s.getBoard()[1][7] = Pawn.BLACK;
		s.setTurn(Turn.WHITE);
		verified += checkCaptureMove(ev, or, s, Move.of(6 * 9 + 2, 4 * 9 + 2), 1, "nero contro il Re sul trono", buf, u);
		// (5) TRIPLA cattura: bianchi in (2,5), (4,5), (3,4); neri in (1,5), (5,5), (3,3); un nero da (3,7) entra in (3,5) e li prende tutti e tre
		s = TestPositions.blank();
		s.getBoard()[4][4] = Pawn.THRONE;
		s.getBoard()[7][1] = Pawn.KING;
		s.getBoard()[2][5] = Pawn.WHITE;
		s.getBoard()[4][5] = Pawn.WHITE;
		s.getBoard()[3][4] = Pawn.WHITE;
		s.getBoard()[1][5] = Pawn.BLACK;
		s.getBoard()[5][5] = Pawn.BLACK;
		s.getBoard()[3][3] = Pawn.BLACK;
		s.getBoard()[3][7] = Pawn.BLACK;
		s.setTurn(Turn.BLACK);
		verified += checkCaptureMove(ev, or, s, Move.of(3 * 9 + 7, 3 * 9 + 5), 3, "tripla cattura", buf, u);
		check(verified >= 5, "catture a mano: verificate solo " + verified + " posizioni su 5");
	}

	private static int checkCaptureMove(NnueEvaluator ev, Oracle or, State s, int move, int expectedCaptures, String label, int[] buf, UndoInfo u) {
		final FastState fs = FastState.fromState(s);
		final int n = fs.generateMoves(buf);
		boolean legal = false;
		for (int i = 0; i < n; i++) {
			legal |= buf[i] == move;
		}
		check(legal, label + ": la mossa " + Move.toString(move) + " non e' legale (posizione costruita male)");
		if (!legal) {
			return 0;
		}
		final NnueSession ses = (NnueSession) ev.newSession(3);
		ses.reset(fs);
		final int caps0 = fs.whiteCount + fs.blackCount;
		fs.makeMove(move, u);
		ses.push(1, move, fs, u);
		check(u.capturedCount() == expectedCaptures, label + ": catture " + u.capturedCount() + " invece di " + expectedCaptures);
		check(fs.whiteCount + fs.blackCount == caps0 - expectedCaptures, label + ": numero di pedine dopo la cattura incoerente");
		final State child = AshtonRules.applyMove(s, Move.toAction(move, s.getTurn()));
		final double inc = ses.evaluate(1, fs);
		check(same(inc, ev.evaluate(fs)), label + ": incrementale != completa");
		check(same(inc, or.eval(child, role(fs), ev.outputScale)), label + ": diverso dall'oracolo storico");
		check(ses.fallbackRebuilds() == 0, label + ": ricostruzione di ripiego inattesa");
		fs.unmakeMove(move, u);
		check(same(ses.evaluate(0, fs), ev.evaluate(fs)), label + ": dopo unmake il genitore e' incoerente");
		return 1;
	}

	// ====================================================================== disallineamenti deliberati

	/**
	 * La sessione non deve mai dare un valore sbagliato, nemmeno quando l'integrazione lo e': firma incoerente, push mancato, push con dati
	 * inutilizzabili, radice non installata. Il valore resta quello della valutazione completa e l'evento e' contato.
	 */
	private static void testDesyncFallback(NnueEvaluator ev, Random rnd) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		int tested = 0;
		for (int p = 0; p < 100 && tested < 12; p++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			final int n = fs.generateMoves(buf);
			int capMove = -1;
			for (int i = 0; i < n && capMove < 0; i++) {
				if (fs.tacticalValue(buf[i]) > 0) {
					fs.makeMove(buf[i], u);
					if (u.wasCapture() && fs.winner == FastState.WINNER_NONE) {
						capMove = buf[i];
					}
					fs.unmakeMove(buf[i], u);
				}
			}
			if (capMove < 0) {
				continue;
			}
			tested++;
			// (a) push mancato: la sessione ignora la mossa
			NnueSession ses = (NnueSession) ev.newSession(4);
			ses.reset(fs);
			fs.makeMove(capMove, u);
			check(same(ses.evaluate(1, fs), ev.evaluate(fs)), "push mancato: valore sbagliato");
			check(ses.fallbackRebuilds() == 1, "push mancato: attesa 1 ricostruzione di ripiego, trovate " + ses.fallbackRebuilds());
			// (b) push con una differenza INCOMPLETA (catture dimenticate): la firma deve tradirla
			ses = (NnueSession) ev.newSession(4);
			fs.unmakeMove(capMove, u);
			ses.reset(fs);
			fs.makeMove(capMove, u);
			ses.push(1, capMove, fs, new UndoInfo());
			check(same(ses.evaluate(1, fs), ev.evaluate(fs)), "differenza incompleta: valore sbagliato");
			check(ses.fallbackRebuilds() == 1, "differenza incompleta: la firma non l'ha rilevata (ricostruzioni " + ses.fallbackRebuilds() + ")");
			// (c) push con mossa spazzatura: differenza inutilizzabile
			ses = (NnueSession) ev.newSession(4);
			fs.unmakeMove(capMove, u);
			ses.reset(fs);
			fs.makeMove(capMove, u);
			ses.push(1, Move.NONE, fs, u);
			check(same(ses.evaluate(1, fs), ev.evaluate(fs)), "mossa spazzatura: valore sbagliato");
			check(ses.fallbackRebuilds() == 1, "mossa spazzatura: attesa 1 ricostruzione di ripiego");
			// (d) radice stantia: la posizione e' cambiata senza reset
			fs.unmakeMove(capMove, u);
			ses = (NnueSession) ev.newSession(4);
			ses.reset(new FastState());
			check(same(ses.evaluate(0, fs), ev.evaluate(fs)), "radice stantia: valore sbagliato");
			check(ses.fallbackRebuilds() == 1, "radice stantia: la firma non l'ha rilevata");
			// (e) mai installata: nessun reset
			ses = (NnueSession) ev.newSession(4);
			check(same(ses.evaluate(0, fs), ev.evaluate(fs)), "radice mai installata: valore sbagliato");
			check(ses.fallbackRebuilds() == 1, "radice mai installata: attesa 1 ricostruzione di ripiego");
			// (f) dopo un ripiego la sessione torna a lavorare per differenze, senza altri ripiegamenti
			ses = (NnueSession) ev.newSession(4);
			ses.reset(fs);
			fs.makeMove(capMove, u);
			ses.push(1, capMove, fs, new UndoInfo()); // sbagliata
			final double v = ses.evaluate(1, fs);
			check(same(v, ev.evaluate(fs)), "ripiego: valore sbagliato");
			final long before = ses.fallbackRebuilds();
			final int m2n = fs.generateMoves(buf);
			if (m2n > 0) {
				final UndoInfo u2 = new UndoInfo();
				fs.makeMove(buf[0], u2);
				ses.push(2, buf[0], fs, u2);
				check(same(ses.evaluate(2, fs), ev.evaluate(fs)), "dopo un ripiego: il passo successivo e' sbagliato");
				check(ses.fallbackRebuilds() == before, "dopo un ripiego il passo successivo non deve ripiegare di nuovo");
				fs.unmakeMove(buf[0], u2);
			}
			fs.unmakeMove(capMove, u);
		}
		check(tested >= 8, "disallineamenti: posizioni con cattura trovate solo " + tested);
	}

	private static void testSessionProtocolErrors(NnueEvaluator ev) {
		final NnueSession ses = (NnueSession) ev.newSession(2);
		final FastState fs = new FastState();
		ses.reset(fs);
		boolean threw = false;
		try {
			ses.push(0, 0, fs, new UndoInfo());
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "push(0, ...) dovrebbe lanciare: il ply 0 e' la radice");
		// una scacchiera con un codice pedina senza feature non puo' essere valutata ne' installata: errore esplicito, non un valore
		final FastState bad = new FastState();
		bad.board[10] = BoardConstants.THRONE_MARK;
		threw = false;
		try {
			ev.evaluate(bad);
		} catch (IllegalStateException e) {
			threw = true;
		}
		check(threw, "evaluate con un codice pedina non valido dovrebbe lanciare IllegalStateException");
		threw = false;
		try {
			ses.reset(bad);
		} catch (IllegalStateException e) {
			threw = true;
		}
		check(threw, "reset con un codice pedina non valido dovrebbe lanciare IllegalStateException");
		// una pila di capacita' ridicola cresce senza errori
		final NnueSession tiny = (NnueSession) ev.newSession(0);
		check(tiny.capacity() >= 2, "capacita' minima");
		tiny.reset(fs);
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo[] us = new UndoInfo[30];
		for (int i = 0; i < us.length; i++) {
			us[i] = new UndoInfo();
		}
		int pushed = 0;
		for (int ply = 0; ply < 25; ply++) {
			final int n = fs.generateMoves(buf);
			if (n == 0 || fs.winner != FastState.WINNER_NONE) {
				break;
			}
			fs.makeMove(buf[0], us[ply]);
			tiny.push(ply + 1, buf[0], fs, us[ply]);
			pushed = ply + 1;
		}
		check(pushed >= 10 && tiny.capacity() >= pushed + 1, "pila cresciuta: capacita' " + tiny.capacity() + " per " + pushed + " ply");
		check(same(tiny.evaluate(pushed, fs), ev.evaluate(fs)), "pila cresciuta: valore sbagliato");
		check(tiny.fallbackRebuilds() == 0, "pila cresciuta: ricostruzioni di ripiego inattese");
	}

	// ====================================================================== file dei pesi

	private static File tmp(String name) throws IOException {
		final File f = File.createTempFile("nnuecheck_" + name + "_", ".txt");
		f.deleteOnExit();
		return f;
	}

	private static void write(File f, String content) throws IOException {
		try (FileWriter w = new FileWriter(f)) {
			w.write(content);
		}
	}

	private static String readAll(File f) throws IOException {
		return new String(java.nio.file.Files.readAllBytes(f.toPath()), "US-ASCII");
	}

	private static long crcOf(int[] all) {
		final CRC32 crc = new CRC32();
		for (int v : all) {
			crc.update(new byte[] { (byte) (v >>> 24), (byte) (v >>> 16), (byte) (v >>> 8), (byte) v });
		}
		return crc.getValue();
	}

	/** Scrive un file NNUE completo e valido a partire dai valori; {@code header} permette di sbagliare una riga alla volta. */
	private static void writeNnue(File f, String magic, String features, int hidden, int l2, int qa, int qw, int paramsDeclared, int[] values, String checksum) throws IOException {
		final StringBuilder sb = new StringBuilder();
		sb.append(magic).append('\n');
		sb.append("features ").append(features).append('\n');
		sb.append("hidden ").append(hidden).append('\n');
		sb.append("l2 ").append(l2).append('\n');
		sb.append("qa ").append(qa).append('\n');
		sb.append("qw ").append(qw).append('\n');
		sb.append("params ").append(paramsDeclared).append('\n');
		for (int i = 0; i < values.length; i++) {
			sb.append(values[i]).append(i % 16 == 15 || i == values.length - 1 ? '\n' : ' ');
		}
		sb.append("checksum ").append(checksum == null ? String.valueOf(crcOf(values)) : checksum).append('\n');
		write(f, sb.toString());
	}

	private static int[] valuesOf(NnueNetwork net) throws IOException {
		final File f = tmp("vals");
		net.save(f.getPath());
		final String[] lines = readAll(f).split("\n");
		final List<Integer> vals = new ArrayList<>();
		for (int i = 7; i < lines.length - 1; i++) {
			for (String t : lines[i].trim().split("\\s+")) {
				vals.add(Integer.parseInt(t));
			}
		}
		final int[] r = new int[vals.size()];
		for (int i = 0; i < r.length; i++) {
			r[i] = vals.get(i);
		}
		return r;
	}

	/** Il file deve essere rifiutato con una IOException il cui messaggio contiene {@code keyword}. */
	private static void expectIoError(File f, String keyword, String label) {
		try {
			NnueNetwork.load(f.getPath());
			fail(label + ": il file errato e' stato accettato");
		} catch (IOException e) {
			checks++;
			if (keyword != null && !String.valueOf(e.getMessage()).contains(keyword)) {
				fail(label + ": messaggio '" + e.getMessage() + "' non contiene '" + keyword + "'");
			}
		} catch (RuntimeException e) {
			fail(label + ": eccezione non gestita " + e);
		}
	}

	private static void testWeightsIO(NnueNetwork net) throws IOException {
		// ---- file valido: andata e ritorno esatti, riconoscimento, evaluator
		final File ok = tmp("ok");
		net.save(ok.getPath());
		final NnueNetwork back = NnueNetwork.load(ok.getPath());
		check(back.equals(net), "andata e ritorno: la rete letta e' diversa da quella scritta");
		check(back.numParams() == NnueNetwork.paramCount(64, 16), "numParams incoerente con paramCount");
		check(NnueNetwork.isNnueFile(ok.getPath()), "isNnueFile deve riconoscere un file valido");
		final Evaluator viaLoader;
		try {
			viaLoader = EvaluatorLoader.load(ok.getPath());
			check(viaLoader instanceof NnueEvaluator, "EvaluatorLoader deve restituire un NnueEvaluator");
			final FastState fs = new FastState();
			check(same(viaLoader.evaluate(fs), new NnueEvaluator(net).evaluate(fs)), "valutatore caricato da file diverso da quello in memoria");
			check(EvaluatorLoader.describe(viaLoader).startsWith("NnueEvaluator"), "describe del loader");
		} catch (IOException e) {
			fail("EvaluatorLoader.load di un file NNUE valido ha lanciato " + e);
		}
		final File ok2 = tmp("ok2");
		NnueEvaluator.fromFile(ok.getPath()).save(ok2.getPath());
		check(readAll(ok).equals(readAll(ok2)), "la serializzazione deve essere deterministica e stabile (file identici)");

		final int[] vals = valuesOf(net);
		final int h = 64, l2 = 16, np = NnueNetwork.paramCount(h, l2);
		check(vals.length == np, "valori letti " + vals.length + " != params " + np);
		final String M = NnueNetwork.MAGIC, F = NnueNetwork.FEATURE_SET;

		// ---- file mancante / vuoto / di un altro formato
		expectIoError(new File("/nonexistent_dir_xyz/none.nnue"), null, "file inesistente");
		final File dir = new File(System.getProperty("java.io.tmpdir"));
		expectIoError(dir, null, "percorso che e' una directory");
		check(!NnueNetwork.isNnueFile("/nonexistent_dir_xyz/none.nnue"), "isNnueFile su file inesistente deve dare false");
		File f = tmp("empty");
		write(f, "");
		expectIoError(f, "Formato", "file vuoto");
		check(!NnueNetwork.isNnueFile(f.getPath()), "isNnueFile su file vuoto");
		f = tmp("cnn");
		write(f, "TABLUT_CNN v1\nconfig 1 2 3\n");
		expectIoError(f, "Formato", "intestazione di un'altra rete (CNN)");
		check(!NnueNetwork.isNnueFile(f.getPath()), "isNnueFile non deve riconoscere un file CNN");
		f = tmp("mlp");
		write(f, "0.1 0.2 0.3\n");
		expectIoError(f, "Formato", "file senza intestazione (MLP)");

		// ---- versione, insieme di feature, scale, dimensioni
		f = tmp("v2");
		writeNnue(f, "TABLUT_NNUE v2", F, h, l2, 256, 64, np, vals, null);
		expectIoError(f, "Versione", "versione sconosciuta");
		check(NnueNetwork.isNnueFile(f.getPath()), "isNnueFile riconosce l'intestazione di qualunque versione (la versione la controlla load)");
		f = tmp("feat");
		writeNnue(f, M, "halfkp", h, l2, 256, 64, np, vals, null);
		expectIoError(f, "feature", "insieme di feature diverso");
		f = tmp("qa");
		writeNnue(f, M, F, h, l2, 255, 64, np, vals, null);
		expectIoError(f, "Scale", "qa diverso");
		f = tmp("qw");
		writeNnue(f, M, F, h, l2, 256, 32, np, vals, null);
		expectIoError(f, "Scale", "qw diverso");
		f = tmp("h0");
		writeNnue(f, M, F, 0, l2, 256, 64, np, vals, null);
		expectIoError(f, "Dimensioni", "hidden = 0");
		f = tmp("hbig");
		writeNnue(f, M, F, NnueNetwork.MAX_HIDDEN + 1, l2, 256, 64, np, vals, null);
		expectIoError(f, "Dimensioni", "hidden oltre il massimo");
		f = tmp("l2big");
		writeNnue(f, M, F, h, NnueNetwork.MAX_L2 + 1, 256, 64, np, vals, null);
		expectIoError(f, "Dimensioni", "l2 oltre il massimo");
		f = tmp("params");
		writeNnue(f, M, F, h, l2, 256, 64, np + 1, vals, null);
		expectIoError(f, "params", "params incoerente con le dimensioni");
		f = tmp("dims");
		writeNnue(f, M, F, 32, l2, 256, 64, np, vals, null);
		expectIoError(f, "params", "dimensioni incoerenti con i valori");

		// ---- corpo: troncato, in eccesso, non numerico, fuori intervallo, checksum, coda
		f = tmp("trunc");
		writeNnue(f, M, F, h, l2, 256, 64, np, Arrays.copyOf(vals, np - 7), "0");
		expectIoError(f, "troncato", "valori mancanti");
		f = tmp("extra");
		final int[] more = Arrays.copyOf(vals, np + 3);
		writeNnue(f, M, F, h, l2, 256, 64, np, more, null);
		expectIoError(f, null, "valori in eccesso");
		f = tmp("nan");
		String body = readAll(ok);
		body = body.replaceFirst("(?m)^(-?\\d+ )", "x$1");
		write(f, body);
		expectIoError(f, "non valido", "valore non numerico");
		f = tmp("crc");
		writeNnue(f, M, F, h, l2, 256, 64, np, vals, "12345");
		expectIoError(f, "Checksum", "checksum sbagliato");
		f = tmp("nocrc");
		body = readAll(ok);
		write(f, body.substring(0, body.lastIndexOf("checksum")));
		expectIoError(f, "checksum", "checksum mancante");
		f = tmp("tail");
		write(f, readAll(ok) + "spazzatura\n");
		expectIoError(f, "inatteso", "contenuto dopo il checksum");
		f = tmp("short");
		final int[] vs = vals.clone();
		vs[5] = 40000; // fuori dall'intervallo short, checksum coerente: deve fermarlo il controllo di intervallo
		writeNnue(f, M, F, h, l2, 256, 64, np, vs, null);
		expectIoError(f, "short", "peso fuori dall'intervallo short");
		f = tmp("ovf");
		final int[] vo = vals.clone();
		vo[NnueNetwork.NUM_FEATURES * h + h + h * l2] = Integer.MAX_VALUE; // l1B[0][0]
		writeNnue(f, M, F, h, l2, 256, 64, np, vo, null);
		expectIoError(f, "overflow", "pesi che andrebbero in overflow");
		// una sola cifra cambiata: il checksum se ne accorge
		f = tmp("flip");
		body = readAll(ok);
		final String[] lines = body.split("\n", -1);
		final char[] row = lines[8].toCharArray();
		for (int i = 0; i < row.length; i++) {
			if (Character.isDigit(row[i])) {
				row[i] = row[i] == '7' ? '3' : '7'; // una sola cifra cambiata
				break;
			}
		}
		lines[8] = new String(row);
		write(f, String.join("\n", lines));
		expectIoError(f, null, "una cifra alterata");
		// versione corretta ma righe fuori ordine
		f = tmp("order");
		write(f, M + "\nhidden 64\nfeatures " + F + "\n");
		expectIoError(f, "Atteso", "righe di intestazione fuori ordine");
	}

	private static void testNetworkValidation() {
		final short[] ftW = new short[NnueNetwork.NUM_FEATURES * 2];
		final short[] ftB = new short[2];
		final short[][] l1W = { new short[2 * 3], new short[2 * 3] };
		final int[][] l1B = { new int[3], new int[3] };
		final short[][] l2W = { new short[3], new short[3] };
		final int[] l2B = new int[2];
		// la forma valida passa
		new NnueNetwork(2, 3, ftW, ftB, l1W, l1B, l2W, l2B);
		checks++;
		final Runnable[] bad = {
				() -> new NnueNetwork(0, 3, ftW, ftB, l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(NnueNetwork.MAX_HIDDEN + 1, 3, ftW, ftB, l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 0, ftW, ftB, l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(2, NnueNetwork.MAX_L2 + 1, ftW, ftB, l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 3, new short[5], ftB, l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 3, ftW, new short[3], l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 3, ftW, ftB, new short[][] { new short[6] }, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 3, ftW, ftB, new short[][] { new short[6], new short[5] }, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 3, ftW, ftB, l1W, new int[][] { new int[3], new int[2] }, l2W, l2B),
				() -> new NnueNetwork(2, 3, ftW, ftB, l1W, l1B, new short[][] { new short[3], null }, l2B),
				() -> new NnueNetwork(2, 3, ftW, ftB, l1W, l1B, l2W, new int[3]),
				() -> new NnueNetwork(2, 3, null, ftB, l1W, l1B, l2W, l2B),
				() -> new NnueNetwork(2, 3, ftW, ftB, null, l1B, l2W, l2B),
				() -> new NnueEvaluator(null),
		};
		for (int i = 0; i < bad.length; i++) {
			boolean threw = false;
			try {
				bad[i].run();
			} catch (IllegalArgumentException e) {
				threw = true;
			} catch (RuntimeException e) {
				fail("costruzione non valida #" + i + ": eccezione inattesa " + e);
				threw = true;
			}
			check(threw, "costruzione non valida #" + i + " accettata");
		}
		// overflow: un bias enorme nel secondo strato o nel primo
		final int[] hugeB = { Integer.MAX_VALUE, 0 };
		boolean threw = false;
		try {
			new NnueNetwork(2, 3, ftW, ftB, l1W, l1B, l2W, hugeB);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "l2B enorme dovrebbe essere rifiutato (overflow)");
		// i pesi sono copiati: modificare gli array passati dopo la costruzione non cambia la rete
		final NnueNetwork n = new NnueNetwork(2, 3, ftW, ftB, l1W, l1B, l2W, l2B);
		final short before = n.ftWeights()[0];
		ftW[0] = 1234;
		check(n.ftWeights()[0] == before, "la rete deve copiare i pesi (immutabilita')");
		final short[] copy = n.ftWeights();
		copy[1] = 77;
		check(n.ftWeights()[1] == 0, "gli accessori devono restituire copie, non gli array interni");
	}

	// ====================================================================== contratto dell'evaluator, thread-safety

	private static void testEvaluatorContract(NnueNetwork net, Random rnd) {
		final NnueEvaluator ev = new NnueEvaluator(net);
		boolean bounded = true, scaled = true;
		for (int i = 0; i < 200; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			ev.outputScale = 1000.0;
			final double v = ev.evaluate(fs);
			ev.outputScale = 1.0;
			final double v1 = ev.evaluate(fs);
			bounded &= Math.abs(v) < 1000.0 && Math.abs(v1) < 1.0;
			scaled &= Math.abs(v - v1 * 1000.0) < 1e-9;
			final FastState copy = FastState.copyOf(fs);
			ev.outputScale = 1000.0;
			check(same(ev.evaluate(fs), ev.evaluate(copy)), "la valutazione deve dipendere solo dalla posizione (copia identica)");
			// la valutazione non deve modificare lo stato
			final FastState keep = FastState.copyOf(fs);
			ev.evaluate(fs);
			check(fs.samePosition(keep) && fs.zobristHash == keep.zobristHash, "evaluate ha modificato la posizione");
		}
		check(bounded, "il valore deve stare in (-outputScale, +outputScale)");
		check(scaled, "outputScale deve scalare linearmente il valore");
		check(ev.describe().contains("243 -> 64 -> 16 -> 1"), "describe: " + ev.describe());
	}

	/**
	 * Un evaluator condiviso fra thread: ciascuno ha la sua sessione e usa il percorso completo; i risultati devono coincidere con quelli
	 * calcolati da un solo thread (nessuno stato mutabile condiviso, i buffer di lavoro sono per-thread).
	 */
	private static void testThreadSafety(NnueNetwork net, Random rnd) throws Exception {
		final NnueEvaluator ev = new NnueEvaluator(net);
		final int n = 400;
		final FastState[] pos = new FastState[n];
		final double[] ref = new double[n];
		for (int i = 0; i < n; i++) {
			pos[i] = FastState.fromState(TestPositions.randomPosition(rnd));
			ref[i] = ev.evaluate(pos[i]);
		}
		final int T = 4;
		final Thread[] ts = new Thread[T];
		final boolean[] ok = new boolean[T];
		final long[] fb = new long[T];
		for (int t = 0; t < T; t++) {
			final int tid = t;
			ts[t] = new Thread(() -> {
				boolean good = true;
				final NnueSession ses = (NnueSession) ev.newSession(4);
				final Random r = new Random(tid);
				for (int rep = 0; rep < 600; rep++) {
					final int i = r.nextInt(n);
					good &= same(ev.evaluate(pos[i]), ref[i]);
					final FastState fs = FastState.copyOf(pos[i]);
					ses.reset(fs);
					good &= same(ses.evaluate(0, fs), ref[i]);
				}
				ok[tid] = good;
				fb[tid] = ses.fallbackRebuilds();
			});
			ts[t].start();
		}
		for (Thread t : ts) {
			t.join();
		}
		for (int t = 0; t < T; t++) {
			check(ok[t], "thread " + t + ": valori diversi da quelli del singolo thread");
			check(fb[t] == 0, "thread " + t + ": ricostruzioni di ripiego inattese");
		}
	}
}
