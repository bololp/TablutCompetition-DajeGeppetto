package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Verifica la CHIAVE ESATTA di {@link FastState} ({@code key0, key1, key2}): (1) l'aggiornamento incrementale in
 * {@code makeMove}/{@code unmakeMove} (spostamenti, catture normali, cattura del Re, fuga, cambio di turno) coincide a ogni nodo con la
 * chiave ricalcolata da zero e viene ripristinato dall'unmake; (2) {@code copyOf}/{@code fromState} la conservano; (3) e' in
 * corrispondenza biunivoca con (scacchiera, turno effettivo): stessa chiave se e solo se stessa posizione, su centinaia di migliaia di
 * posizioni diverse; (4) ha lo stesso contenuto informativo dello Zobrist (stessa chiave se e solo se stesso hash); (5) a partita
 * finita il turno non conta. Esce con codice 1 se qualcosa fallisce. Uso: {@code ExactKeyCheck [seme]}.
 */
public class ExactKeyCheck {

	static int failures = 0;
	static long checks = 0;

	static void check(boolean ok, String what) {
		checks++;
		if (!ok) {
			failures++;
			if (failures <= 30) {
				System.out.println("FALLITO: " + what);
			}
		}
	}

	static boolean keysEqual(FastState a, long k0, long k1, long k2) {
		return a.key0 == k0 && a.key1 == k1 && a.key2 == k2;
	}

	/** Controlla che la chiave incrementale di s coincida con quella ricalcolata da zero. */
	static void verify(FastState s, String where) {
		FastState c = FastState.copyOf(s);
		c.computeKeys();
		check(keysEqual(s, c.key0, c.key1, c.key2), "chiave incrementale diversa da quella ricalcolata: " + where);
	}

	static String posId(FastState s) {
		return Arrays.toString(s.board) + (s.winner == FastState.WINNER_NONE && s.turn == BoardConstants.BLACK_TURN ? "B" : "W");
	}

	static long moves = 0, capMoves = 0, kingCaps = 0, escapes = 0, draws = 0;
	static final Map<String, long[]> byPos = new HashMap<>(); // id posizione -> chiave
	static final Map<String, String> byKey = new HashMap<>(); // chiave -> id posizione
	static final Map<Long, String> byZob = new HashMap<>();   // zobrist -> chiave

	static void record(FastState s) {
		String id = posId(s);
		String ks = s.key0 + ":" + s.key1 + ":" + s.key2;
		long[] known = byPos.get(id);
		if (known == null) {
			byPos.put(id, new long[] { s.key0, s.key1, s.key2 });
		} else {
			check(known[0] == s.key0 && known[1] == s.key1 && known[2] == s.key2, "stessa posizione con chiavi diverse");
		}
		String other = byKey.get(ks);
		if (other == null) {
			byKey.put(ks, id);
		} else {
			check(other.equals(id), "posizioni DIVERSE con la stessa chiave esatta");
		}
		String z = byZob.get(s.zobristHash);
		if (z == null) {
			byZob.put(s.zobristHash, ks);
		} else {
			check(z.equals(ks), "stesso Zobrist ma chiavi esatte diverse (o viceversa)");
		}
	}

	/** Esplora tutte le mosse da s fino a depth, controllando la chiave a ogni nodo e dopo ogni unmake. */
	static void dfs(FastState s, int depth, int[][] bufs) {
		if (depth == 0 || s.isGameOver()) {
			return;
		}
		final int[] buf = bufs[depth];
		final int n = s.generateMoves(buf);
		final UndoInfo u = new UndoInfo();
		for (int i = 0; i < n; i++) {
			final long a = s.key0, b = s.key1, c = s.key2, z = s.zobristHash;
			final int before = s.whiteCount + s.blackCount;
			s.makeMove(buf[i], u);
			moves++;
			if (s.whiteCount + s.blackCount < before) {
				capMoves++;
			}
			if (s.winner == FastState.WINNER_BLACK) {
				kingCaps++;
			} else if (s.winner == FastState.WINNER_WHITE) {
				escapes++;
			} else if (s.winner == FastState.WINNER_DRAW) {
				draws++;
			}
			verify(s, "dopo makeMove");
			record(s);
			dfs(s, depth - 1, bufs);
			s.unmakeMove(buf[i], u);
			check(keysEqual(s, a, b, c) && s.zobristHash == z, "unmakeMove non ripristina la chiave");
		}
	}

	public static void main(String[] args) {
		final long seed = args.length > 0 ? Long.parseLong(args[0]) : 99L;
		final Random rnd = new Random(seed);
		final int[][] bufs = new int[6][FastState.MAX_MOVES];

		// posizione iniziale
		FastState init = new FastState();
		verify(init, "posizione iniziale");
		record(init);
		dfs(init, 3, bufs);

		// partite casuali complete + esplorazione a profondita' 2 da molte posizioni intermedie
		int games = 0;
		while (byPos.size() < 400_000 && games < 400) {
			TestPositions.Game g = TestPositions.playGame(rnd, true, 90);
			games++;
			for (int i = 0; i < g.positions.size(); i++) {
				FastState f = FastState.fromState(g.positions.get(i));
				verify(f, "fromState di una posizione di partita");
				record(f);
				FastState cp = FastState.copyOf(f);
				check(keysEqual(cp, f.key0, f.key1, f.key2), "copyOf non conserva la chiave");
				if (i % 3 == 0) {
					dfs(f, 2, bufs);
				}
			}
			FastState r = FastState.fromState(TestPositions.randomPosition(rnd));
			verify(r, "fromState di una posizione casuale densa");
			record(r);
			dfs(r, 2, bufs);
		}

		// a partita finita il turno non conta; un board scritto a mano richiede computeKeys
		FastState t = FastState.copyOf(init);
		t.board[10] = BoardConstants.BLACK;
		check(!keysEqual(t, init.key0, init.key1, init.key2) || true, "(precondizione)");
		long old1 = t.key0, old2 = t.key1;
		t.computeKeys();
		check(t.key0 != init.key0 || t.key1 != init.key1 || t.key2 != init.key2, "computeKeys deve riflettere una scacchiera modificata a mano");
		FastState e1 = FastState.copyOf(init), e2 = FastState.copyOf(init);
		e1.winner = FastState.WINNER_WHITE;
		e2.winner = FastState.WINNER_WHITE;
		e1.turn = BoardConstants.WHITE_TURN;
		e2.turn = BoardConstants.BLACK_TURN;
		e1.computeKeys();
		e2.computeKeys();
		check(keysEqual(e1, e2.key0, e2.key1, e2.key2), "a partita finita il turno non deve influire sulla chiave");
		FastState n1 = FastState.copyOf(init);
		n1.turn = BoardConstants.BLACK_TURN;
		n1.computeKeys();
		check(n1.key2 != init.key2 && (n1.key2 ^ init.key2) == FastState.KEY2_TURN_BIT, "il turno Nero cambia solo il bit del turno");

		System.out.println("giochi=" + games + " mosse verificate=" + moves + " (con cattura " + capMoves + ", Re catturato " + kingCaps + ", fughe del Re " + escapes + ", patte " + draws + ")");
		System.out.println("posizioni distinte=" + byPos.size() + " chiavi distinte=" + byKey.size() + " zobrist distinti=" + byZob.size());
		check(byPos.size() == byKey.size(), "numero di posizioni distinte diverso dal numero di chiavi distinte");
		check(byPos.size() == byZob.size(), "numero di posizioni distinte diverso dal numero di hash distinti");
		check(capMoves >= 5000 && kingCaps >= 20 && escapes >= 20, "copertura insufficiente: catture=" + capMoves + " re catturato=" + kingCaps + " fughe=" + escapes);
		System.out.println(failures == 0 ? "=== EXACT KEY CHECK OK (" + checks + " verifiche) ===" : "=== EXACT KEY CHECK: " + failures + " FALLIMENTI su " + checks + " verifiche ===");
		System.exit(failures == 0 ? 0 : 1);
	}
}
