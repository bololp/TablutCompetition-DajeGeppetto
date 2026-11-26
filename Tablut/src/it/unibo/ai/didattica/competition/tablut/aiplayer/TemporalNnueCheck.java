package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Test della NNUE con feature temporali (aiplayer): schema delle feature, derivazione della mossa da due scacchiere, rete e file, sessione incrementale
 * contro la valutazione completa (con storia prima della radice, catture, mosse del Re, fratelli, mosse nulle), tracker della partita reale e integrazione
 * con {@link Negamax}. Ogni confronto usa un oracolo scritto indipendentemente dal codice sotto test.
 * <p>
 * Uso: {@code TemporalNnueCheck [partite] [seed]}.
 */
public class TemporalNnueCheck {

	private static int failures = 0;
	private static long checks = 0;
	private static long captures = 0, kingMoves = 0, evals = 0, siblings = 0, nullMoves = 0;

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

	/** Una partita casuale: restituisce i record veri di ogni mossa e le posizioni prima di ciascuna. */
	private static final class Game {
		final List<Integer> recs = new ArrayList<>();
		final List<FastState> before = new ArrayList<>(); // before.get(i) = posizione prima della mossa i
		final List<Integer> moves = new ArrayList<>();
		FastState end;
	}

	private static Game randomGame(Random rnd, int plies) {
		final Game g = new Game();
		final FastState fs = new FastState();
		final UndoInfo u = new UndoInfo();
		for (int p = 0; p < plies && fs.winner == FastState.WINNER_NONE; p++) {
			final int n = fs.generateMoves(BUF);
			if (n == 0) {
				break;
			}
			final int m = BUF[rnd.nextInt(n)];
			g.before.add(FastState.copyOf(fs));
			g.moves.add(m);
			fs.makeMove(m, u);
			final int to = Move.getTo(m);
			g.recs.add(TemporalHistory.record(fs.board[to], Move.getFrom(m), to, u.capturedCount()));
			if (u.capturedCount() > 0) {
				captures++;
			}
			if (fs.board[to] == BoardConstants.KING) {
				kingMoves++;
			}
		}
		g.end = FastState.copyOf(fs);
		return g;
	}

	// ====================================================================== A: schema delle feature

	private static void testSchema(Random rnd, int games) {
		for (int t = 0; t < 2000; t++) {
			final int type = rnd.nextInt(3), f = rnd.nextInt(81), to = rnd.nextInt(81), c = rnd.nextInt(6);
			final int r = TemporalHistory.pack(type, f, to, c);
			check(TemporalHistory.isValid(r) && r != 0, "pack: record valido e non nullo");
			check(TemporalHistory.type(r) == type && TemporalHistory.from(r) == f && TemporalHistory.to(r) == to && TemporalHistory.caps(r) == Math.min(c, 3), "pack/unpack");
		}
		check(!TemporalHistory.isValid(0), "0 non e' valido");
		check(TemporalHistory.record(BoardConstants.EMPTY, 1, 2, 0) == 0 && TemporalHistory.record(BoardConstants.THRONE_MARK, 1, 2, 0) == 0, "record: pedine senza feature -> 0");
		check(TemporalHistory.record(BoardConstants.KING, -1, 2, 0) == 0 && TemporalHistory.record(BoardConstants.KING, 1, 81, 0) == 0, "record: celle fuori scacchiera -> 0");
		boolean threw = false;
		try {
			TemporalHistory.pack(3, 0, 0, 0);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "pack: tipo 3 rifiutato");

		// indici: nel range, distinti, formula scritta a mano
		for (int t = 0; t < 500; t++) {
			final int K = 1 + rnd.nextInt(TemporalHistory.MAX_AGES);
			final int len = rnd.nextInt(K + 3);
			final int[] recs = new int[K + 3];
			final int[] oracle = new int[3 * (K + 3)];
			int expected = 0;
			boolean stopped = false;
			final int cut = rnd.nextInt(K + 4); // da qui in poi record nulli
			for (int a = 0; a < recs.length; a++) {
				recs[a] = a >= cut ? 0 : TemporalHistory.pack(rnd.nextInt(3), rnd.nextInt(81), rnd.nextInt(81), rnd.nextInt(4));
			}
			for (int a = 0; a < Math.min(len, K) && !stopped; a++) {
				if (recs[a] == 0) {
					stopped = true;
					break;
				}
				final int ty = TemporalHistory.type(recs[a]), fr = TemporalHistory.from(recs[a]), to = TemporalHistory.to(recs[a]), cp = TemporalHistory.caps(recs[a]);
				oracle[expected++] = a * 488 + ty * 81 + fr;
				oracle[expected++] = a * 488 + 243 + ty * 81 + to;
				if (cp == 1) {
					oracle[expected++] = a * 488 + 486;
				} else if (cp >= 2) {
					oracle[expected++] = a * 488 + 487;
				}
			}
			final int[] out = new int[3 * K];
			final int n = TemporalHistory.activeFeatures(recs, 0, len, K, null, out, 0);
			check(n == expected && Arrays.equals(Arrays.copyOf(out, n), Arrays.copyOf(oracle, expected)), "activeFeatures: K=" + K + " len=" + len + " cut=" + cut);
			final Set<Integer> seen = new HashSet<>();
			boolean ok = true;
			for (int i = 0; i < n; i++) {
				ok &= out[i] >= 0 && out[i] < TemporalHistory.numFeatures(K) && seen.add(out[i]);
			}
			check(ok, "activeFeatures: indici nel range e distinti");
			// con una simmetria: stesse feature della mossa trasformata
			final int[] map = BoardSymmetry.MAP[rnd.nextInt(BoardSymmetry.COUNT)];
			final int[] mapped = new int[recs.length];
			for (int a = 0; a < recs.length; a++) {
				mapped[a] = recs[a] == 0 ? 0 : TemporalHistory.pack(TemporalHistory.type(recs[a]), map[TemporalHistory.from(recs[a])], map[TemporalHistory.to(recs[a])], TemporalHistory.caps(recs[a]));
			}
			final int[] o2 = new int[3 * K], o3 = new int[3 * K];
			final int n2 = TemporalHistory.activeFeatures(recs, 0, len, K, map, o2, 0);
			final int n3 = TemporalHistory.activeFeatures(mapped, 0, len, K, null, o3, 0);
			check(n2 == n3 && Arrays.equals(Arrays.copyOf(o2, n2), Arrays.copyOf(o3, n3)), "activeFeatures con simmetria == feature della mossa trasformata");
		}

		// derivazione della mossa dal confronto di due scacchiere, contro la verita'
		int derived = 0;
		for (int g = 0; g < games; g++) {
			final Game gm = randomGame(rnd, 120);
			final FastState cur = FastState.copyOf(new FastState());
			final UndoInfo u = new UndoInfo();
			for (int i = 0; i < gm.moves.size(); i++) {
				final FastState b = gm.before.get(i);
				cur.copyFrom(b);
				cur.makeMove(gm.moves.get(i), u);
				final int role = b.turn == BoardConstants.BLACK_TURN ? 1 : 0;
				final int r = TemporalHistory.deriveRecord(b.board, 0, cur.board, 0, role);
				check(r == gm.recs.get(i), "deriveRecord g" + g + " ply " + i + ": " + r + " vs vero " + gm.recs.get(i));
				derived++;
				check(TemporalHistory.deriveRecord(b.board, 0, b.board, 0, role) == 0, "deriveRecord: scacchiera invariata -> 0");
			}
		}
		check(captures > 0 && kingMoves > 0, "la copertura deve includere catture (" + captures + ") e mosse del Re (" + kingMoves + ")");
		// scacchiere non collegate da una mossa
		final FastState a = new FastState();
		final FastState b = new FastState();
		final int n = a.generateMoves(BUF);
		final UndoInfo u = new UndoInfo();
		b.makeMove(BUF[0], u);
		final int ok = TemporalHistory.deriveRecord(a.board, 0, b.board, 0, 0);
		check(ok != 0 || BUF.length == 0, "una mossa vera si deriva");
		final int n2 = b.generateMoves(BUF);
		b.makeMove(BUF[n2 / 2], u);
		check(TemporalHistory.deriveRecord(a.board, 0, b.board, 0, 0) == 0, "due mosse di fila: non e' una mossa -> 0");
		final byte[] swapped = a.board.clone();
		int w1 = -1, w2 = -1;
		for (int sq = 0; sq < 81 && w2 < 0; sq++) {
			if (swapped[sq] == BoardConstants.WHITE) {
				if (w1 < 0) {
					w1 = sq;
				} else {
					w2 = sq;
				}
			}
		}
		swapped[w1] = BoardConstants.EMPTY;
		swapped[w2] = BoardConstants.BLACK; // una pedina bianca diventa nera: non e' una mossa
		check(TemporalHistory.deriveRecord(a.board, 0, swapped, 0, 0) == 0, "cambio di tipo -> 0");
		check(n > 0 && derived > 1000, "abbastanza mosse derivate: " + derived);
	}

	// ====================================================================== B: rete, oracolo, file

	private static short[] randomTemporal(Random rnd, int hidden, int hist, double scale) {
		final short[] w = new short[TemporalHistory.numFeatures(hist) * hidden];
		for (int i = 0; i < w.length; i++) {
			w[i] = (short) Math.round(rnd.nextGaussian() * scale * NnueNetwork.QA);
		}
		return w;
	}

	/** Oracolo indipendente: accumulatore sommando i pesi pubblici e le righe temporali scelte con la formula scritta qui, poi la testa della rete di base. */
	private static double oracleValue(TemporalNnueNetwork net, FastState s, int[] recs, int len) {
		final NnueNetwork base = net.base();
		final int H = base.hidden();
		final short[] ftW = base.ftWeights(), ftB = base.ftBiases(), tW = net.temporalWeights();
		final int[] acc = new int[H];
		for (int j = 0; j < H; j++) {
			acc[j] = ftB[j];
		}
		for (int sq = 0; sq < 81; sq++) {
			final int pc = s.board[sq];
			if (pc == BoardConstants.EMPTY) {
				continue;
			}
			final int type = pc == BoardConstants.WHITE ? 0 : pc == BoardConstants.BLACK ? 1 : 2;
			for (int j = 0; j < H; j++) {
				acc[j] += ftW[(type * 81 + sq) * H + j];
			}
		}
		for (int a = 0; a < Math.min(len, net.hist()); a++) {
			final int r = recs[a];
			if (r == 0) {
				break;
			}
			final int ty = (r >>> 14) & 3, fr = r & 127, to = (r >>> 7) & 127, cp = (r >>> 16) & 3;
			final int[] rows = cp == 0 ? new int[] { a * 488 + ty * 81 + fr, a * 488 + 243 + ty * 81 + to } : new int[] { a * 488 + ty * 81 + fr, a * 488 + 243 + ty * 81 + to, a * 488 + 486 + (cp >= 2 ? 1 : 0) };
			for (int row : rows) {
				for (int j = 0; j < H; j++) {
					acc[j] += tW[row * H + j];
				}
			}
		}
		final int role = s.turn == BoardConstants.BLACK_TURN ? 1 : 0;
		return NnueNetwork.valueOf(base.logit(acc, new int[base.scratchSize()], role)) * 1000.0;
	}

	private static void testNetwork(Random rnd, int games) throws IOException {
		final NnueNetwork base = NnueNetwork.random(16, 8, 11);
		final TemporalNnueNetwork zero = TemporalNnueNetwork.fromBase(base, 4);
		final TemporalNnueEvaluator zev = new TemporalNnueEvaluator(zero);
		final NnueEvaluator bev = new NnueEvaluator(base);
		final TemporalNnueNetwork rndNet = new TemporalNnueNetwork(base, 4, randomTemporal(rnd, 16, 4, 0.08));
		final TemporalNnueEvaluator rev = new TemporalNnueEvaluator(rndNet);
		for (int g = 0; g < games; g++) {
			final Game gm = randomGame(rnd, 80);
			for (int i = 0; i < gm.before.size(); i++) {
				final FastState s = gm.before.get(i);
				if (s.winner != FastState.WINNER_NONE) {
					continue;
				}
				// finestra vera (mosse piu' recenti per prime) di lunghezza casuale
				final int avail = Math.min(i, 6), len = rnd.nextInt(avail + 1);
				final int[] recs = new int[Math.max(1, len)];
				for (int a = 0; a < len; a++) {
					recs[a] = gm.recs.get(i - 1 - a);
				}
				check(zev.evaluate(s) == bev.evaluate(s), "tW=0: stateless uguale alla rete di base");
				check(zev.evaluate(s, recs, len) == bev.evaluate(s), "tW=0: con qualunque finestra uguale alla rete di base");
				check(rev.evaluate(s, recs, len) == oracleValue(rndNet, s, recs, len), "valutazione completa == oracolo (len " + len + ")");
				check(rev.evaluate(s) == oracleValue(rndNet, s, recs, 0), "stateless == oracolo a finestra vuota");
				evals++;
			}
		}
		// il blocco temporale cambia il valore (altrimenti i test sopra sarebbero vuoti)
		final FastState s0 = new FastState();
		final FastState s1 = FastState.copyOf(s0);
		final int n = s1.generateMoves(BUF);
		final int m = BUF[0];
		final UndoInfo u = new UndoInfo();
		s1.makeMove(m, u);
		final int rec = TemporalHistory.record(s1.board[Move.getTo(m)], Move.getFrom(m), Move.getTo(m), 0);
		check(n > 0 && rev.evaluate(s1, new int[] { rec }, 1) != rev.evaluate(s1), "la storia deve poter cambiare il valore");

		// file
		final File tmp = File.createTempFile("tnnue", ".txt");
		try {
			rndNet.save(tmp.getPath());
			check(TemporalNnueNetwork.isTemporalFile(tmp.getPath()) && !NnueNetwork.isNnueFile(tmp.getPath()), "intestazione riconosciuta e distinta da TABLUT_NNUE");
			check(TemporalNnueNetwork.load(tmp.getPath()).equals(rndNet), "salva/carica: stessa rete");
			final File nn = File.createTempFile("nnue", ".txt");
			base.save(nn.getPath());
			check(!TemporalNnueNetwork.isTemporalFile(nn.getPath()) && NnueNetwork.isNnueFile(nn.getPath()), "una NNUE di base non e' un file temporale");
			expectIo(nn.getPath(), "NNUE di base");
			nn.delete();
			final List<String> lines = Files.readAllLines(tmp.toPath());
			// checksum alterato
			final List<String> bad = new ArrayList<>(lines);
			bad.set(bad.size() - 1, "checksum 12345");
			write(tmp, bad);
			expectIo(tmp.getPath(), "checksum errato");
			// un valore alterato
			final List<String> bad2 = new ArrayList<>(lines);
			bad2.set(10, bad2.get(10).replaceFirst("-?\\d+", "7777"));
			write(tmp, bad2);
			expectIo(tmp.getPath(), "valore alterato");
			// troncato
			write(tmp, lines.subList(0, lines.size() - 3));
			expectIo(tmp.getPath(), "troncato");
			// hist incoerente con le feature
			final List<String> bad3 = new ArrayList<>(lines);
			bad3.set(1, "features psq3x81+hist3");
			write(tmp, bad3);
			expectIo(tmp.getPath(), "feature incoerenti con hist");
			// versione
			final List<String> bad4 = new ArrayList<>(lines);
			bad4.set(0, "TABLUT_TNNUE v2");
			write(tmp, bad4);
			expectIo(tmp.getPath(), "versione");
			// contenuto dopo il checksum
			final List<String> bad5 = new ArrayList<>(lines);
			bad5.add("extra");
			write(tmp, bad5);
			expectIo(tmp.getPath(), "contenuto in eccesso");
		} finally {
			tmp.delete();
		}
		// pesi che andrebbero in overflow
		boolean threw = false;
		try {
			final short[] huge = new short[TemporalHistory.numFeatures(8) * 16];
			Arrays.fill(huge, Short.MAX_VALUE);
			new TemporalNnueNetwork(NnueNetwork.random(16, 8, 3), 8, huge).hashCode();
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw || true, "overflow: ammesso solo se la somma resta in un int"); // 8*488 righe * 32767 < 2^31: nessun overflow reale con K <= 8
		threw = false;
		try {
			new TemporalNnueNetwork(base, 0, new short[0]);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "hist=0 rifiutato");
		threw = false;
		try {
			new TemporalNnueNetwork(base, 4, new short[5]);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "tW di lunghezza sbagliata rifiutato");
	}

	private static void write(File f, List<String> lines) throws IOException {
		Files.write(f.toPath(), lines);
	}

	private static void expectIo(String path, String what) {
		boolean threw = false;
		try {
			TemporalNnueNetwork.load(path);
		} catch (IOException e) {
			threw = true;
		}
		check(threw, "load deve rifiutare: " + what);
	}

	// ====================================================================== C: sessione incrementale == valutazione completa

	private static void testSession(Random rnd, int games) {
		final NnueNetwork base = NnueNetwork.random(24, 8, 21);
		final TemporalNnueNetwork net = new TemporalNnueNetwork(base, 5, randomTemporal(rnd, 24, 5, 0.06));
		final TemporalNnueEvaluator ev = new TemporalNnueEvaluator(net);
		for (int g = 0; g < games; g++) {
			final Game gm = randomGame(rnd, 20 + rnd.nextInt(60));
			if (gm.before.size() < 3) {
				continue;
			}
			final int rootIdx = rnd.nextInt(gm.before.size());
			final FastState root = FastState.copyOf(gm.before.get(rootIdx));
			if (root.winner != FastState.WINNER_NONE) {
				continue;
			}
			// storia reale prima della radice: suffisso casuale della vera (puo' essere vuoto)
			final int priorLen = rnd.nextInt(Math.min(rootIdx, 7) + 1);
			final int[] prior = new int[priorLen];
			for (int a = 0; a < priorLen; a++) {
				prior[a] = gm.recs.get(rootIdx - 1 - a);
			}
			ev.setPriorMoves(prior);
			check(Arrays.equals(ev.priorMoves(), prior), "setPriorMoves/priorMoves");
			final IncrementalEvaluator.Session ses = ev.newSession(4);
			ses.reset(root);
			final FastState st = FastState.copyOf(root);
			final UndoInfo[] undo = new UndoInfo[40];
			for (int i = 0; i < undo.length; i++) {
				undo[i] = new UndoInfo();
			}
			final int[] pathRec = new int[40]; // pathRec[p] = record del frame p (0 = mossa nulla)
			final boolean[] wasNull = new boolean[40];
			int sp = 0;
			for (int step = 0; step < 220; step++) {
				final boolean canDown = sp < 30 && st.winner == FastState.WINNER_NONE;
				if (canDown && (sp == 0 || rnd.nextInt(100) < 62)) {
					final int n = st.generateMoves(BUF);
					if (n == 0) {
						continue;
					}
					if (rnd.nextInt(100) < 7 && sp >= 1) {
						st.makeNullMove(undo[sp]);
						check(ses.pushNull(sp + 1, st), "pushNull supportata");
						pathRec[sp + 1] = 0;
						wasNull[sp + 1] = true;
						nullMoves++;
					} else {
						final int m = BUF[rnd.nextInt(n)];
						st.makeMove(m, undo[sp]);
						ses.push(sp + 1, m, st, undo[sp]);
						final int to = Move.getTo(m);
						pathRec[sp + 1] = TemporalHistory.record(st.board[to], Move.getFrom(m), to, undo[sp].capturedCount());
						wasNull[sp + 1] = false;
					}
					sp++;
				} else if (sp > 0) {
					if (wasNull[sp]) {
						st.unmakeNullMove(undo[sp - 1]);
					} else {
						// annulla l'ultima mossa: serve la mossa, la ricaviamo dal record
						st.unmakeMove(moveOf(pathRec[sp]), undo[sp - 1]);
					}
					sp--;
					siblings++;
				} else {
					continue;
				}
				if (st.winner != FastState.WINNER_NONE) {
					continue;
				}
				if (rnd.nextInt(100) < 70) {
					// oracolo: finestra = record del percorso (dal piu' recente) poi la storia reale prima della radice; si ferma al primo record nullo
					final int[] win = new int[sp + prior.length + 1];
					int len = 0;
					for (int p = sp; p >= 1; p--) {
						win[len++] = pathRec[p];
					}
					for (int a = 0; a < prior.length; a++) {
						win[len++] = prior[a];
					}
					final double expected = ev.evaluate(st, win, len);
					final double got = ses.evaluate(sp, st);
					check(Double.doubleToLongBits(expected) == Double.doubleToLongBits(got), "g" + g + " ply " + sp + ": sessione " + got + " vs completa " + expected);
					evals++;
				}
			}
			check(ses.fallbackRebuilds() == 0, "g" + g + ": la sessione ha ricostruito per ripiego " + ses.fallbackRebuilds() + " volte");
		}
	}

	/** Ricostruisce la mossa compatta da un record (da, a). */
	private static int moveOf(int rec) {
		return Move.of(TemporalHistory.from(rec), TemporalHistory.to(rec));
	}

	// ====================================================================== D: tracker della partita reale

	private static void testTracker(Random rnd, int games) {
		for (int g = 0; g < games; g++) {
			final boolean weAreWhite = g % 2 == 0;
			final GameMoveTracker tr = new GameMoveTracker();
			final FastState fs = new FastState();
			final UndoInfo u = new UndoInfo();
			final List<Integer> truth = new ArrayList<>(); // piu' recente in fondo
			for (int ply = 0; ply < 90 && fs.winner == FastState.WINNER_NONE; ply++) {
				final boolean ourTurn = (fs.turn == BoardConstants.WHITE_TURN) == weAreWhite;
				final int n = fs.generateMoves(BUF);
				if (n == 0) {
					break;
				}
				final int m = BUF[rnd.nextInt(n)];
				if (ourTurn) {
					tr.observe(fs);
					tr.observe(fs); // idempotente
					final int[] exp = lastN(truth, TemporalHistory.MAX_AGES);
					check(Arrays.equals(tr.recent(), exp), "g" + g + " ply " + ply + ": storia del tracker " + Arrays.toString(tr.recent()) + " vs vera " + Arrays.toString(exp));
					final FastState before = FastState.copyOf(fs);
					fs.makeMove(m, u);
					tr.own(before, m);
				} else {
					fs.makeMove(m, u);
				}
				final int to = Move.getTo(m);
				truth.add(TemporalHistory.record(fs.board[to], Move.getFrom(m), to, u.capturedCount()));
			}
		}
		// fuori sincronia: una posizione che non e' una mossa dopo quella ricordata azzera la storia
		final GameMoveTracker tr = new GameMoveTracker();
		final FastState a = new FastState();
		final UndoInfo u = new UndoInfo();
		final int n = a.generateMoves(BUF);
		a.makeMove(BUF[0], u);
		tr.observe(a);
		check(tr.recent().length == 1, "una mossa dall'iniziale -> storia di 1");
		final FastState far = FastState.copyOf(a);
		for (int i = 0; i < 4; i++) {
			final int k = far.generateMoves(BUF);
			far.makeMove(BUF[k / 2], u);
		}
		tr.observe(far);
		check(n > 0 && tr.recent().length == 0, "salto di piu' mosse -> storia azzerata (finestra vuota, mai sbagliata)");
		tr.reset();
		check(tr.recent().length == 0, "reset");
	}

	private static int[] lastN(List<Integer> truth, int k) {
		final int m = Math.min(k, truth.size());
		final int[] r = new int[m];
		for (int a = 0; a < m; a++) {
			r[a] = truth.get(truth.size() - 1 - a);
		}
		return r;
	}

	// ====================================================================== E: integrazione con Negamax

	private static final long BIG = 100_000_000L;

	private static void testNegamax(Random rnd, int positions) {
		final NnueNetwork base = NnueNetwork.random(32, 8, 5);
		final TemporalNnueEvaluator zero = new TemporalNnueEvaluator(TemporalNnueNetwork.fromBase(base, 4));
		final NnueEvaluator plain = new NnueEvaluator(base);
		final TemporalNnueEvaluator live = new TemporalNnueEvaluator(new TemporalNnueNetwork(base, 4, randomTemporal(rnd, 32, 4, 0.25)));
		int changed = 0;
		for (int i = 0; i < positions; i++) {
			final FastState root = FastState.fromState(TestPositions.randomPosition(rnd));
			if (root.winner != FastState.WINNER_NONE) {
				continue;
			}
			final PositionHistory h = PositionHistory.empty();
			// blocco temporale a zero: stessa ricerca della NNUE di base, bit per bit
			final Negamax a = new Negamax(zero, 3, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
			final Negamax b = new Negamax(plain, 3, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
			check(a.incrementalSessionForTest() != null, "Negamax crea la sessione temporale");
			final int ma = a.findBestMove(FastState.copyOf(root), BIG, h);
			final int mb = b.findBestMove(FastState.copyOf(root), BIG, h);
			check(ma == mb && Double.doubleToLongBits(a.lastScore) == Double.doubleToLongBits(b.lastScore) && a.lastNodeCount == b.lastNodeCount && a.lastDepthReached == b.lastDepthReached,
					"pos " + i + ": tW=0 -> ricerca identica alla NNUE di base (" + Move.toString(ma) + "/" + Move.toString(mb) + ", " + a.lastScore + "/" + b.lastScore + ", nodi " + a.lastNodeCount + "/" + b.lastNodeCount + ")");
			check(a.incrementalFallbacksForTest() == 0, "pos " + i + ": nessuna ricostruzione di ripiego (tW=0)");
			// pesi temporali veri: nessun ripiego, mossa legale, e la storia della radice cambia (almeno a volte) il punteggio
			final Negamax c = new Negamax(live, 2, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
			live.setPriorMoves(null);
			final int mc = c.findBestMove(FastState.copyOf(root), BIG, h);
			final double scoreNoHist = c.lastScore;
			check(c.incrementalFallbacksForTest() == 0 && mc != Move.NONE, "pos " + i + ": temporale senza storia: nessun ripiego");
			final int n = root.generateMoves(BUF);
			if (n > 0 && root.winner == FastState.WINNER_NONE) {
				final Negamax d = new Negamax(live, 2, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
				final int fakeFrom = rnd.nextInt(81), fakeTo = rnd.nextInt(81);
				live.setPriorMoves(new int[] { TemporalHistory.pack(rnd.nextInt(3), fakeFrom, fakeTo, rnd.nextInt(3)), TemporalHistory.pack(rnd.nextInt(3), rnd.nextInt(81), rnd.nextInt(81), 0) });
				d.findBestMove(FastState.copyOf(root), BIG, h);
				check(d.incrementalFallbacksForTest() == 0, "pos " + i + ": temporale con storia: nessun ripiego");
				if (Double.doubleToLongBits(d.lastScore) != Double.doubleToLongBits(scoreNoHist)) {
					changed++;
				}
			}
			live.setPriorMoves(null);
		}
		check(changed > positions / 4, "la storia della radice deve influire sul punteggio (cambiato in " + changed + "/" + positions + ")");
		// Lazy SMP: piu' thread, ognuno con la propria sessione, tutti con la stessa storia
		final Negamax smp = new Negamax(live, 3, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 2).setThreads(2);
		live.setPriorMoves(new int[] { TemporalHistory.pack(0, 10, 11, 0) });
		final FastState root = FastState.fromState(TestPositions.randomPosition(rnd));
		final int ms = smp.findBestMove(FastState.copyOf(root), 150, PositionHistory.empty());
		check(ms != Move.NONE || root.generateMoves(BUF) == 0, "Lazy SMP con la rete temporale: mossa restituita");
		check(smp.incrementalFallbacksForTest() == 0, "Lazy SMP: nessun ripiego in nessuna sessione");
		check(smp.incrementalSessionsForTest() >= 1, "Lazy SMP: sessioni distinte per worker");
	}

	// ====================================================================== main

	public static void main(String[] args) throws Exception {
		final int games = args.length > 0 ? Integer.parseInt(args[0]) : 40;
		final long seed = args.length > 1 ? Long.parseLong(args[1]) : 1L;
		final Random rnd = new Random(seed);
		final long t0 = System.currentTimeMillis();
		testSchema(rnd, games);
		System.out.println("schema: ok (" + checks + " verifiche finora, " + captures + " catture, " + kingMoves + " mosse del Re)");
		testNetwork(rnd, Math.max(5, games / 2));
		System.out.println("rete e file: ok (" + checks + " verifiche, " + evals + " valutazioni)");
		testSession(rnd, games);
		System.out.println("sessione: ok (" + checks + " verifiche, " + evals + " valutazioni, " + siblings + " annullamenti, " + nullMoves + " mosse nulle)");
		testTracker(rnd, Math.max(4, games / 4));
		testNegamax(rnd, Math.max(6, games / 4));
		System.out.printf("TemporalNnueCheck: %d verifiche, %d errori, %.1f s%n", checks, failures, (System.currentTimeMillis() - t0) / 1000.0);
		System.exit(failures == 0 ? 0 : 1);
	}
}
