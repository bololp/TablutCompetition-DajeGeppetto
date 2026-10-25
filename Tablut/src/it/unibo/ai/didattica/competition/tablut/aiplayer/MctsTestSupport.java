package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Strumenti dei test dell'{@link Mcts}: oracoli di forza bruta e un "walker" che verifica un albero in modo indipendente dal codice che
 * l'ha costruito.
 * <p>
 * Il walker ripercorre l'albero RIGIOCANDO le mosse su una posizione propria e ricalcola da zero, senza fidarsi dei campi dell'albero:
 * le mosse legali (contro {@code generateMoves}); i terminali (vittoria dopo la mossa); le ripetizioni (con chiavi TESTUALI
 * scacchiera+turno e un conteggio dei pezzi per le catture, non con Zobrist ne' {@code wasCapture}: sono i due meccanismi che il codice
 * sotto test usa); i limiti dimostrati (propagati dalle foglie con la regola negamax); il valore minimax implicito; e le MEDIE {@code Q}
 * di ogni arco, ricostruite dalla struttura dell'albero (ogni simulazione termina in un evento ben definito: l'espansione di un nodo o una
 * visita statica di un arco, e il suo valore e' calcolabile con le valutazioni dell'evaluator).
 */
final class MctsTestSupport {

	private MctsTestSupport() {
	}

	// ------------------------------------------------------------------------------------------ utilita'

	static String key(FastState s) {
		final char[] c = new char[BoardConstants.NUM_SQUARES + 1];
		for (int i = 0; i < BoardConstants.NUM_SQUARES; i++) {
			c[i] = (char) ('0' + s.board[i]);
		}
		c[BoardConstants.NUM_SQUARES] = (char) ('0' + s.turn);
		return new String(c);
	}

	static int pieces(FastState s) {
		int n = 0;
		for (int i = 0; i < BoardConstants.NUM_SQUARES; i++) {
			if (s.board[i] != BoardConstants.EMPTY) {
				n++;
			}
		}
		return n;
	}

	static int[] legal(FastState s) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final int k = s.generateMoves(buf);
		return Arrays.copyOf(buf, k);
	}

	static boolean contains(int[] a, int v) {
		for (int x : a) {
			if (x == v) {
				return true;
			}
		}
		return false;
	}

	static int indexOfMove(Mcts.Node n, int move) {
		if (n == null) {
			return -1;
		}
		for (int i = 0; i < n.nMoves; i++) {
			if (n.moves[i] == (short) move) {
				return i;
			}
		}
		return -1;
	}

	static boolean sameFast(FastState a, FastState b) {
		return a.samePosition(b) && a.zobristHash == b.zobristHash && a.winner == b.winner && Arrays.equals(a.board, b.board);
	}

	/** Storia (hash Zobrist) con la sola radice: quella che il client passerebbe a inizio partita. */
	static long[] rootOnly(FastState s) {
		return new long[] { s.zobristHash };
	}

	static String[] keysOf(FastState s) {
		return new String[] { key(s) };
	}

	static List<FastState> gamePositions(long seed, int games, int stride) {
		final Random rnd = new Random(seed);
		final List<FastState> out = new ArrayList<>();
		for (int g = 0; g < games; g++) {
			final TestPositions.Game game = TestPositions.playGame(rnd, g % 2 == 0, 200);
			// passo sempre DISPARI: con un passo pari si campionerebbero solo gli indici dello stesso colore (il Bianco muove agli indici pari)
			final int step = stride % 2 == 0 ? stride + 1 : stride;
			for (int i = 3; i < game.positions.size(); i += step) {
				out.add(FastState.fromState(game.positions.get(i)));
			}
		}
		return out;
	}

	static List<FastState> densePositions(long seed, int n) {
		final Random rnd = new Random(seed);
		final List<FastState> out = new ArrayList<>();
		while (out.size() < n) {
			final State s = TestPositions.randomPosition(rnd);
			final FastState f = FastState.fromState(s);
			if (f.winner == FastState.WINNER_NONE) {
				out.add(f);
			}
		}
		return out;
	}

	// ------------------------------------------------------------------------------------------ oracoli di forza bruta

	/** Mosse con cui chi muove vince subito. */
	static int[] winNowMoves(FastState s) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		final int k = s.generateMoves(buf);
		final int[] out = new int[k];
		int n = 0;
		for (int i = 0; i < k; i++) {
			s.makeMove(buf[i], u);
			if (s.winner != FastState.WINNER_NONE) {
				out[n++] = buf[i];
			}
			s.unmakeMove(buf[i], u);
		}
		return Arrays.copyOf(out, n);
	}

	/**
	 * Come {@link #hasWinNow} ma molto piu' economico, per il walker: genera le mosse e chiede a {@code tacticalValue} (esatta, verificata dal progetto
	 * contro {@code makeMove}) se una mossa del Re vince subito. E' indipendente dallo scanner di {@link Mcts}, che ha un'altra implementazione, ed e'
	 * confrontata con la forza bruta completa da {@code MctsSearchCheck.testEscapeScan}.
	 */
	static boolean hasWinNowFast(FastState s) {
		if (s.winner != FastState.WINNER_NONE || s.kingSquare < 0) {
			return false;
		}
		final int[] buf = new int[FastState.MAX_MOVES];
		final int k = s.generateMoves(buf);
		for (int i = 0; i < k; i++) {
			if ((buf[i] & 127) == s.kingSquare && s.tacticalValue(buf[i]) == FastState.TACT_WIN) {
				return true;
			}
		}
		return false;
	}

	static boolean hasWinNow(FastState s) {
		return s.winner == FastState.WINNER_NONE && winNowMoves(s).length > 0;
	}

	/** Mosse dopo le quali l'avversario NON ha una vittoria immediata (le mosse che vincono subito contano come sicure). */
	static int[] safeMoves(FastState s) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		final int k = s.generateMoves(buf);
		final int[] out = new int[k];
		int n = 0;
		for (int i = 0; i < k; i++) {
			s.makeMove(buf[i], u);
			final boolean safe = s.winner != FastState.WINNER_NONE || !hasWinNow(s);
			s.unmakeMove(buf[i], u);
			if (safe) {
				out[n++] = buf[i];
			}
		}
		return Arrays.copyOf(out, n);
	}

	/**
	 * Mosse che vincono con la forza entro 3 semi-mosse (la nostra, QUALUNQUE risposta, la nostra): la vittoria immediata, oppure una mossa
	 * dopo la quale l'avversario non ha mosse (perde) o ogni sua risposta lascia una vittoria immediata. Entro 3 semi-mosse nessuna
	 * posizione puo' ripetersi: la regola di ripetizione non c'entra.
	 */
	static int[] win3Moves(FastState s) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		final int k = s.generateMoves(buf);
		final int[] out = new int[k];
		int n = 0;
		for (int i = 0; i < k; i++) {
			s.makeMove(buf[i], u);
			boolean win;
			if (s.winner != FastState.WINNER_NONE) {
				win = true;
			} else {
				final int[] rb = new int[FastState.MAX_MOVES];
				final UndoInfo u2 = new UndoInfo();
				final int kr = s.generateMoves(rb);
				if (kr == 0) {
					win = true; // l'avversario non ha mosse: perde
				} else {
					win = true;
					for (int j = 0; j < kr && win; j++) {
						s.makeMove(rb[j], u2);
						if (s.winner != FastState.WINNER_NONE || !hasWinNow(s)) {
							win = false;
						}
						s.unmakeMove(rb[j], u2);
					}
				}
			}
			s.unmakeMove(buf[i], u);
			if (win) {
				out[n++] = buf[i];
			}
		}
		return Arrays.copyOf(out, n);
	}

	/** Chi muove perde entro 2 semi-mosse: ha almeno una mossa e ognuna lascia all'avversario una vittoria immediata. */
	static boolean lossIn2(FastState s) {
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		final int k = s.generateMoves(buf);
		if (k == 0) {
			return false;
		}
		for (int i = 0; i < k; i++) {
			s.makeMove(buf[i], u);
			final boolean opponentWins = s.winner == FastState.WINNER_NONE && hasWinNow(s);
			s.unmakeMove(buf[i], u);
			if (!opponentWins) {
				return false;
			}
		}
		return true;
	}

	// ------------------------------------------------------------------------------------------ walker

	/** Risultato del ricalcolo indipendente di un sottoalbero. */
	static final class Res {
		long size;
		/** Somma, dal punto di vista di chi muove nel nodo, dei valori degli eventi terminali sotto i suoi archi (si veda il commento di classe). */
		double u;
		int lo, hi;
		float mm;
		/** Valore iniziale del nodo (massimo dei valori statici dei figli) = valore restituito dalla simulazione che l'ha espanso. */
		float leaf;
		/** Nessun arco esatto visitato ne' nodo risolto nel sottoalbero: le medie sono ricostruibili. */
		boolean clean;
	}

	static final class Walk {
		final Mcts mcts;
		final Evaluator ev;
		final boolean singleThread;
		final int expandAfter;
		String error;
		long nodes, termEdges, repEdges, escEdges, exactVisited, meansChecked, meansSkipped, maxDepth, edgesWithChild;
		private final FastState pos = new FastState();
		private final ArrayList<String> hist = new ArrayList<>();
		private int base;

		Walk(Mcts m, Evaluator ev, boolean singleThread, int expandAfter) {
			this.mcts = m;
			this.ev = ev;
			this.singleThread = singleThread;
			this.expandAfter = expandAfter;
		}

		/** @return null se tutto coincide, altrimenti la descrizione del primo disaccordo */
		String run(Mcts.Node root, FastState rootPos, String[] rootHistKeys) {
			error = null;
			pos.copyFrom(rootPos);
			hist.clear();
			hist.addAll(Arrays.asList(rootHistKeys));
			base = 0;
			visit(root, 0, "radice");
			return error;
		}

		private void fail(String where, String what) {
			if (error == null) {
				error = where + ": " + what;
			}
		}

		private boolean histContains(String k) {
			for (int i = base; i < hist.size(); i++) {
				if (hist.get(i).equals(k)) {
					return true;
				}
			}
			return false;
		}

		private Res visit(Mcts.Node n, int depth, String where) {
			final Res res = new Res();
			nodes++;
			maxDepth = Math.max(maxDepth, depth);
			if (n == Mcts.EXPANDING) {
				fail(where, "sentinella EXPANDING rimasta nell'albero");
				res.clean = false;
				return res;
			}
			final int[] legal = legal(pos);
			if (pos.winner != FastState.WINNER_NONE) {
				fail(where, "nodo su posizione terminale");
			}
			if (n.nMoves != legal.length) {
				fail(where, "numero di mosse " + n.nMoves + " invece di " + legal.length);
				res.clean = false;
				return res;
			}
			if (n.totalVloss != 0) {
				fail(where, "perdita virtuale residua totale " + n.totalVloss);
			}
			if (n.nMoves == 0) {
				if (n.nodeLo != -1 || n.nodeHi != -1 || n.nodeIm != -1f) {
					fail(where, "nodo senza mosse non e' una sconfitta esatta");
				}
				res.size = 1;
				res.lo = -1;
				res.hi = -1;
				res.mm = -1f;
				res.leaf = -1f;
				res.u = 0;
				res.clean = false;
				return res;
			}
			final boolean whiteToMove = pos.turn == BoardConstants.WHITE_TURN;
			final UndoInfo u = new UndoInfo();
			int nLo = -1, nHi = -1;
			float nMm = -Float.MAX_VALUE;
			float leaf = -Float.MAX_VALUE;
			long size = 1;
			double usum = 0;
			boolean clean = n.nodeLo != n.nodeHi;
			long sumVisits = 0;
			int maxPriorEdge = 0;
			float maxInit = -Float.MAX_VALUE;
			double priorSum = 0;
			final float[] inits = new float[n.nMoves];
			for (int i = 0; i < n.nMoves; i++) {
				final String w = where + "/" + i;
				final int m = legal[i];
				if (n.moves[i] != (short) m) {
					fail(w, "mossa " + n.moves[i] + " invece di " + m);
				}
				final int before = pieces(pos);
				pos.makeMove(m, u);
				final boolean capture = pieces(pos) != before;
				final byte winner = pos.winner;
				final String k = key(pos);
				final boolean rep = winner == FastState.WINNER_NONE && !capture && histContains(k);
				float init;
				int elo, ehi;
				boolean normal = false;
				if (winner != FastState.WINNER_NONE) {
					final boolean moverWins = (winner == FastState.WINNER_WHITE) == whiteToMove;
					init = moverWins ? 1f : -1f;
					elo = ehi = moverWins ? 1 : -1;
					termEdges++;
				} else if (rep) {
					init = 0f;
					elo = ehi = 0;
					repEdges++;
				} else if (mcts.config().escapeScan && pos.turn == BoardConstants.WHITE_TURN && hasWinNowFast(pos)) {
					// il Bianco e' di turno con una vittoria immediata (ricalcolata a forza bruta, non con lo scanner): chi ha mosso ha perso
					init = -1f;
					elo = ehi = -1;
					escEdges++;
				} else {
					normal = true;
					final float white = mcts.squash(ev.evaluate(pos));
					init = whiteToMove ? white : -white;
					elo = -1;
					ehi = 1;
				}
				inits[i] = init;
				if (init > leaf) {
					leaf = init;
				}
				final Mcts.Node child = n.child[i];
				if (n.vloss[i] != 0) {
					fail(w, "perdita virtuale residua " + n.vloss[i]);
				}
				sumVisits += n.visits[i];
				if (n.visits[i] < 0) {
					fail(w, "visite negative");
				}
				if (!(n.q[i] >= -1f && n.q[i] <= 1f)) {
					fail(w, "Q fuori da [-1,1]: " + n.q[i]);
				}
				float edgeIm = init;
				if (!normal) {
					if (child != null) {
						fail(w, "arco terminale/ripetizione con un figlio");
					}
					if (n.lo[i] != elo || n.hi[i] != ehi || n.q[i] != init || n.im[i] != init) {
						fail(w, "arco esatto (" + elo + ") con campi lo=" + n.lo[i] + " hi=" + n.hi[i] + " q=" + n.q[i] + " im=" + n.im[i]);
					}
					if (n.visits[i] > 0) {
						clean = false;
						exactVisited++;
					}
				} else if (child == null) {
					if (n.lo[i] != -1 || n.hi[i] != 1) {
						fail(w, "arco senza figlio con limiti (" + n.lo[i] + "," + n.hi[i] + ")");
					}
					if (n.q[i] != init || n.im[i] != init) {
						fail(w, "arco senza figlio: q=" + n.q[i] + " im=" + n.im[i] + " invece di " + init);
					}
					usum += n.visits[i] * (double) init; // visite statiche: ognuna vale il valore statico del figlio
				} else {
					edgesWithChild++;
					final int savedBase = base;
					final int savedSize = hist.size();
					if (capture) {
						base = savedSize; // una cattura azzera la memoria delle posizioni: il figlio e' la prima voce del nuovo segmento
					}
					hist.add(k);
					final Res r = visit(child, depth + 1, w);
					while (hist.size() > savedSize) {
						hist.remove(hist.size() - 1);
					}
					base = savedBase;
					elo = -r.hi;
					ehi = -r.lo;
					edgeIm = -r.mm;
					if (n.lo[i] != elo || n.hi[i] != ehi) {
						fail(w, "limiti dell'arco (" + n.lo[i] + "," + n.hi[i] + ") invece di (" + elo + "," + ehi + ") ricalcolati dal figlio");
					}
					if (n.im[i] != edgeIm) {
						fail(w, "minimax implicito dell'arco " + n.im[i] + " invece di " + edgeIm);
					}
					size += r.size;
					if (!r.clean || n.lo[i] == n.hi[i]) {
						clean = false;
					}
					final long staticVisits = (long) n.visits[i] - 1 - child.totalVisits;
					if (staticVisits < 0) {
						fail(w, "visite dell'arco " + n.visits[i] + " < 1 + visite del figlio " + child.totalVisits);
					}
					if (r.clean && n.lo[i] != n.hi[i]) {
						if (singleThread && staticVisits != expandAfter) {
							fail(w, "visite statiche prima dell'espansione " + staticVisits + " invece di " + expandAfter);
						}
						if (!singleThread && staticVisits < expandAfter) {
							fail(w, "visite statiche prima dell'espansione " + staticVisits + " < " + expandAfter);
						}
						final double expected = (staticVisits * (double) init + (-(double) r.leaf - r.u)) / n.visits[i];
						meansChecked++;
						if (Math.abs(expected - n.q[i]) > 1e-4) {
							fail(w, "media Q " + n.q[i] + " invece di " + expected + " ricostruita dalla struttura (visite=" + n.visits[i] + ", statiche="
									+ staticVisits + ")");
						}
						usum += staticVisits * (double) init + (-(double) r.leaf - r.u);
					} else {
						meansSkipped++;
					}
				}
				pos.unmakeMove(m, u);
				if (elo > nLo) {
					nLo = elo;
				}
				if (ehi > nHi) {
					nHi = ehi;
				}
				if (edgeIm > nMm) {
					nMm = edgeIm;
				}
				if (init > maxInit) {
					maxInit = init;
					maxPriorEdge = i;
				}
				priorSum += n.prior[i];
			}
			if (n.nodeLo != nLo || n.nodeHi != nHi) {
				fail(where, "limiti del nodo (" + n.nodeLo + "," + n.nodeHi + ") invece di (" + nLo + "," + nHi + ")");
			}
			if (n.nodeLo > n.nodeHi) {
				fail(where, "limiti incoerenti lo>hi");
			}
			if (n.nodeIm != nMm) {
				fail(where, "minimax implicito del nodo " + n.nodeIm + " invece di " + nMm);
			}
			if (n.totalVisits != sumVisits) {
				fail(where, "visite totali " + n.totalVisits + " invece della somma " + sumVisits);
			}
			// prior: sommano ~1 (quantizzazione a 16 bit) e il massimo spetta al valore statico piu' alto
			if (Math.abs(priorSum / 65535.0 - 1.0) > (n.nMoves + 1) / 65535.0) {
				fail(where, "prior con somma " + priorSum / 65535.0);
			}
			int maxPrior = 0;
			for (int i = 0; i < n.nMoves; i++) {
				maxPrior = Math.max(maxPrior, n.prior[i]);
			}
			if (n.prior[maxPriorEdge] != maxPrior) {
				// a parita' di valore statico piu' archi possono avere il prior massimo: basta che l'arco a valore massimo lo raggiunga
				fail(where, "l'arco col valore statico massimo non ha il prior massimo");
			}
			res.size = size;
			res.lo = nLo;
			res.hi = nHi;
			res.mm = nMm;
			res.leaf = leaf;
			res.u = usum;
			res.clean = clean;
			return res;
		}
	}
}
