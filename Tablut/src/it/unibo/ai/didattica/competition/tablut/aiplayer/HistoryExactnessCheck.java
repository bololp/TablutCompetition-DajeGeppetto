package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.lang.reflect.Method;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Verifica che la storia degli stati sia rappresentata con chiavi ESATTE e che due posizioni diverse non possano mai essere scambiate:
 * (1) {@link PositionHistory}: contratto dell'API; (2) la chiave incrementale di {@link FastState} coincide con quella calcolata in modo
 * INDEPENDENTE da {@link TestPositions#exactKey(State)} su migliaia di coppie posizione/mossa (terminali compresi); (3) la storia costruita
 * come fa il client (aggiungi, svuota a ogni cattura) coincide con quella dell'oracolo indipendente lungo intere partite; (4) collisioni di
 * hash FORZATE: posizioni diverse con lo stesso Zobrist non entrano mai nella storia come la stessa posizione; (5) per riflessione: nessuna
 * API di ricerca o di tracciamento accetta piu' una storia come {@code long[]} o {@code List<Long>} di hash. Il rilevamento di ripetizioni
 * di {@code Negamax} e {@code Mcts} con collisioni forzate e' in {@code RepetitionAndStalemateCheck} e {@code MctsSearchCheck}.
 * Esce con codice 1 se qualcosa fallisce. Uso: {@code HistoryExactnessCheck [seme]}.
 */
public class HistoryExactnessCheck {

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

	static boolean sameKey(FastState f, long[] k) {
		return f.key0 == k[0] && f.key1 == k[1] && f.key2 == k[2];
	}

	public static void main(String[] args) {
		final long seed = args.length > 0 ? Long.parseLong(args[0]) : 1357L;
		positionHistoryApi();
		incrementalVersusIndependent(seed);
		clientStyleHistories(seed + 1);
		forcedCollisions(seed + 2);
		noHashHistoryInApis();
		negamaxPathStack(seed + 3);
		callerHistoryNotModified(seed + 4);
		mctsTreeReuseRespectsHistory(seed + 5);
		System.out.println(failures == 0 ? "=== HISTORY EXACTNESS CHECK OK (" + checks + " verifiche) ===" : "=== HISTORY EXACTNESS CHECK: " + failures + " FALLIMENTI su " + checks + " verifiche ===");
		System.exit(failures == 0 ? 0 : 1);
	}

	// ---- (1) contratto di PositionHistory ------------------------------------------------------------------------------------------------
	static void positionHistoryApi() {
		final FastState a = new FastState();
		final int[] buf = new int[FastState.MAX_MOVES];
		final FastState b = FastState.copyOf(a);
		b.makeMove(buf[0] = firstMove(a, buf), new UndoInfo());
		final FastState c = FastState.copyOf(b);
		c.makeMove(firstMove(b, buf), new UndoInfo());

		PositionHistory h = new PositionHistory();
		check(h.isEmpty() && h.size() == 0 && !h.contains(a), "storia nuova: vuota");
		h.add(a);
		h.add(b);
		check(h.size() == 2 && h.contains(a) && h.contains(b) && !h.contains(c), "add/contains");
		check(h.endsWith(b) && !h.endsWith(a), "endsWith guarda solo l'ultima posizione");
		check(h.key0(0) == a.key0 && h.key1(0) == a.key1 && h.key2(0) == a.key2 && h.key2(1) == b.key2, "chiavi per indice");
		final PositionHistory copy = h.copy();
		h.add(c);
		check(copy.size() == 2 && h.size() == 3 && !copy.contains(c), "copy e' indipendente dall'originale");
		check(copy.equals(PositionHistory.of(a, b)) && !copy.equals(h) && copy.hashCode() == PositionHistory.of(a, b).hashCode(), "equals/hashCode per contenuto");
		h.removeLast();
		check(h.equals(copy), "removeLast");
		h.clear();
		check(h.isEmpty() && !h.contains(a), "clear");
		boolean thrown = false;
		try {
			h.removeLast();
		} catch (IllegalStateException e) {
			thrown = true;
		}
		check(thrown, "removeLast su storia vuota lancia");
		thrown = false;
		try {
			h.key0(0);
		} catch (IndexOutOfBoundsException e) {
			thrown = true;
		}
		check(thrown, "indice fuori dai limiti lancia");
		// crescita oltre la capacita' iniziale
		final PositionHistory big = new PositionHistory(2);
		for (int i = 0; i < 1000; i++) {
			big.add(i, i * 3L, i * 7L);
		}
		check(big.size() == 1000 && big.contains(999, 2997, 6993) && !big.contains(999, 2997, 6994) && !big.contains(1000, 3000, 7000), "crescita e contains su 1000 voci");
		// copyKeysTo: stesso formato, con spostamento, senza toccare il resto
		final long[] dst = new long[3 * 6];
		java.util.Arrays.fill(dst, -1L);
		final PositionHistory two = PositionHistory.of(a, b);
		check(two.copyKeysTo(dst, 1) == 2 && dst[0] == -1 && dst[3] == a.key0 && dst[5] == a.key2 && dst[6] == b.key0 && dst[9] == -1, "copyKeysTo");
		// una posizione si distingue da un'altra anche se differisce in UNA sola parola della chiave
		final PositionHistory one = new PositionHistory();
		one.add(1L, 2L, 3L);
		check(one.contains(1L, 2L, 3L) && !one.contains(9L, 2L, 3L) && !one.contains(1L, 9L, 3L) && !one.contains(1L, 2L, 9L), "contains confronta tutte e tre le parole");
		// State e FastState danno la stessa chiave
		final State s = TestPositions.initial();
		check(PositionHistory.of(s).equals(PositionHistory.of(new FastState())) && PositionHistory.of(s).contains(s), "add(State) == add(FastState)");
		final PositionHistory withAll = new PositionHistory();
		withAll.addAll(two);
		withAll.addAll(two);
		check(withAll.size() == 4 && withAll.key0(2) == a.key0, "addAll");
		System.out.println("(1) PositionHistory: contratto dell'API (add/contains/copy/equals/removeLast/clear/capienza/copyKeysTo/State==FastState)");
	}

	static int firstMove(FastState s, int[] buf) {
		final int n = s.generateMoves(buf);
		check(n > 0, "almeno una mossa legale");
		return buf[0];
	}

	// ---- (2) chiave incrementale contro oracolo indipendente ----------------------------------------------------------------------------------------
	static void incrementalVersusIndependent(long seed) {
		final Random rnd = new Random(seed);
		final int[] buf = new int[FastState.MAX_MOVES];
		final UndoInfo u = new UndoInfo();
		long pairs = 0, terminals = 0, captures = 0;
		for (int g = 0; g < 14; g++) {
			final TestPositions.Game game = TestPositions.playGame(rnd, g % 2 == 0, 90);
			for (int i = 0; i < game.positions.size(); i += 2) {
				final State st = game.positions.get(i);
				final FastState root = FastState.fromState(st);
				check(sameKey(root, TestPositions.exactKey(st)), "fromState: chiave diversa dall'oracolo indipendente");
				final int n = root.generateMoves(buf);
				final int[] moves = java.util.Arrays.copyOf(buf, n);
				for (int mv : moves) {
					final int before = root.whiteCount + root.blackCount;
					root.makeMove(mv, u);
					pairs++;
					if (root.whiteCount + root.blackCount < before) {
						captures++;
					}
					if (root.isGameOver()) {
						terminals++;
					}
					final State child = root.toState();
					check(sameKey(root, TestPositions.exactKey(child)), "chiave incrementale diversa dall'oracolo indipendente dopo una mossa (" + (root.isGameOver() ? "terminale" : "normale") + ")");
					root.unmakeMove(mv, u);
					check(sameKey(root, TestPositions.exactKey(st)), "unmake: chiave diversa dall'oracolo");
				}
			}
		}
		for (int i = 0; i < 300; i++) {
			final State st = TestPositions.randomPosition(rnd);
			check(sameKey(FastState.fromState(st), TestPositions.exactKey(st)), "posizione densa: chiave diversa dall'oracolo indipendente");
		}
		check(pairs > 5000 && captures > 500 && terminals > 20, "copertura: coppie=" + pairs + " catture=" + captures + " terminali=" + terminals);
		System.out.println("(2) chiave incrementale == oracolo indipendente su " + pairs + " coppie posizione/mossa (" + captures + " catture, " + terminals + " terminali) e 300 posizioni dense");
	}

	// ---- (3) la storia costruita come fa il client == quella dell'oracolo ---------------------------------------------------------------------------------
	static void clientStyleHistories(long seed) {
		final Random rnd = new Random(seed);
		long positions = 0, resets = 0;
		for (int g = 0; g < 25; g++) {
			final TestPositions.Game game = TestPositions.playGame(rnd, g % 2 == 0, 120);
			final PositionHistory h = new PositionHistory();
			int prevPieces = -1;
			for (int i = 0; i < game.positions.size(); i++) {
				final FastState f = FastState.fromState(game.positions.get(i));
				final int pieces = f.whiteCount + f.blackCount + (f.kingSquare >= 0 ? 1 : 0);
				if (prevPieces >= 0 && pieces < prevPieces) {
					h.clear(); // una cattura azzera la memoria: come fa TablutNegamaxClient.updateRealGameHistory
					resets++;
				}
				prevPieces = pieces;
				h.add(f);
				positions++;
				check(h.equals(game.histories.get(i)), "storia costruita come il client diversa da quella dell'oracolo (partita " + g + ", posizione " + i + ")");
				check(h.endsWith(f) && h.contains(f), "la posizione corrente e' l'ultima della storia");
			}
		}
		check(resets > 10, "copertura: poche catture (" + resets + ")");
		System.out.println("(3) storia costruita come il client == oracolo indipendente su " + positions + " posizioni di 25 partite (" + resets + " azzeramenti per cattura)");
	}

	// ---- (4) collisioni di hash forzate -------------------------------------------------------------------------------------------------------------------
	static void forcedCollisions(long seed) {
		final Random rnd = new Random(seed);
		long variants = 0;
		for (int i = 0; i < 200; i++) {
			final FastState p = FastState.fromState(TestPositions.randomPosition(rnd));
			final FastState[] vs = MctsTestSupport.collidingVariants(p);
			for (FastState v : vs) {
				variants++;
				check(v.zobristHash == p.zobristHash, "precondizione: stesso hash forzato");
				check(!(v.key0 == p.key0 && v.key1 == p.key1 && v.key2 == p.key2), "precondizione: chiavi diverse");
				final PositionHistory h = new PositionHistory();
				h.add(v);
				check(!h.contains(p) && h.contains(v), "una posizione con lo stesso hash di p NON e' p");
				check(!h.endsWith(p), "endsWith non scambia la collisione per la radice");
				final PositionHistory hp = new PositionHistory();
				hp.add(p);
				check(!hp.equals(h), "storie diverse non sono uguali");
				// una storia a hash le avrebbe confuse
				final java.util.List<Long> hashHistory = new java.util.ArrayList<>();
				hashHistory.add(v.zobristHash);
				check(hashHistory.contains(p.zobristHash), "(documentazione) una storia a hash le confonderebbe");
			}
		}
		System.out.println("(4) " + variants + " collisioni di hash forzate: la storia a chiave esatta non le scambia mai (una a hash si')");
	}

	// ---- (6) la pila di storia del cammino di Negamax (le voci aggiunte DURANTE la ricerca) confronta tutte e tre le parole ---------------------------------
	static void negamaxPathStack(long seed) {
		try {
			final Method push = Negamax.class.getDeclaredMethod("pushHistory", long.class, long.class, long.class);
			final Method contains = Negamax.class.getDeclaredMethod("historyContains", long.class, long.class, long.class);
			push.setAccessible(true);
			contains.setAccessible(true);
			final FastState root = new FastState();
			final Negamax n = new Negamax(new SimpleEvaluator(), 10).setThreads(1);
			n.installForTest(root, PositionHistory.of(root), 8);
			check((Boolean) contains.invoke(n, root.key0, root.key1, root.key2), "la radice installata e' nella storia");
			final Random rnd = new Random(seed);
			final long[][] pushed = new long[6][3];
			for (int i = 0; i < pushed.length; i++) {
				// parole tutte diverse da zero: una pila che non salvasse una parola non ritroverebbe la voce
				pushed[i][0] = rnd.nextLong() | 1L;
				pushed[i][1] = rnd.nextLong() | 1L;
				pushed[i][2] = (rnd.nextLong() & ((1L << 35) - 1)) | 1L;
				push.invoke(n, pushed[i][0], pushed[i][1], pushed[i][2]);
			}
			for (int i = 0; i < pushed.length; i++) {
				final long a = pushed[i][0], b = pushed[i][1], c = pushed[i][2];
				check((Boolean) contains.invoke(n, a, b, c), "voce pushata " + i + " ritrovata");
				check(!(Boolean) contains.invoke(n, a ^ 1L, b, c), "voce " + i + ": differisce solo in k0 -> non e' la stessa");
				check(!(Boolean) contains.invoke(n, a, b ^ 1L, c), "voce " + i + ": differisce solo in k1 -> non e' la stessa");
				check(!(Boolean) contains.invoke(n, a, b, c ^ 1L), "voce " + i + ": differisce solo in k2 (celle 64-80/turno) -> non e' la stessa");
				check(!(Boolean) contains.invoke(n, a, b, c ^ FastState.KEY2_TURN_BIT), "voce " + i + ": differisce solo nel turno -> non e' la stessa");
			}
			System.out.println("(6) pila di storia del cammino di Negamax: " + pushed.length + " voci pushate, ritrovate solo con tutte e tre le parole uguali");
		} catch (Exception e) {
			check(false, "pila di storia di Negamax (riflessione): " + e);
		}
	}

	// ---- (7) la storia del chiamante non viene modificata (ricerca con e senza la radice in coda) ------------------------------------------------------------
	static void callerHistoryNotModified(long seed) {
		final Random rnd = new Random(seed);
		final FastState root = FastState.fromState(TestPositions.randomPosition(rnd));
		final FastState other = FastState.fromState(TestPositions.randomPosition(rnd));
		final PositionHistory withoutRoot = PositionHistory.of(other); // NON finisce con la radice: Mcts la aggiunge a una copia
		final PositionHistory withRoot = PositionHistory.of(other, root);
		final PositionHistory a0 = withoutRoot.copy(), b0 = withRoot.copy();
		final Mcts m = MctsSearchCheck.mk(MctsSearchCheck.constant(0), "expand=0,mm=0,cache=0", 1);
		m.setMaxSims(30);
		m.findBestMove(root, 100000, withoutRoot);
		m.findBestMove(root, 100000, withRoot);
		check(withoutRoot.equals(a0) && withoutRoot.size() == 1, "Mcts: la storia del chiamante senza la radice NON deve essere modificata (la radice si aggiunge a una copia)");
		check(withRoot.equals(b0), "Mcts: la storia del chiamante con la radice non e' modificata");
		final Negamax n = new Negamax(new SimpleEvaluator(), 3, true, 1 << 12, true, Negamax.ORDER_DEFAULT, 1);
		n.findBestMove(root, 1_000_000_000L, withoutRoot);
		n.findBestMove(root, 1_000_000_000L, withRoot);
		check(withoutRoot.equals(a0) && withRoot.equals(b0), "Negamax: la storia del chiamante non e' modificata");
		final Negamax smp = new Negamax(new SimpleEvaluator(), 40, true, 1 << 12, true, Negamax.ORDER_DEFAULT, 3).setThreads(3);
		smp.findBestMove(root, 150, withRoot);
		check(withRoot.equals(b0), "Negamax Lazy SMP: la storia del chiamante non e' modificata dagli helper");
		System.out.println("(7) la storia passata dal chiamante resta invariata (Mcts con e senza la radice in coda, Negamax, Negamax Lazy SMP)");
	}

	// ---- (8) il riuso dell'albero MCTS rispetta la storia ---------------------------------------------------------------------------------------------------------
	static void mctsTreeReuseRespectsHistory(long seed) {
		final Random rnd = new Random(seed);
		final FastState root = FastState.fromState(TestPositions.randomPosition(rnd));
		final FastState earlier = FastState.fromState(TestPositions.randomPosition(rnd));
		final PositionHistory h1 = PositionHistory.of(root);
		final PositionHistory h2 = PositionHistory.of(earlier, root); // stessa posizione, storia DIVERSA
		final Mcts m = MctsSearchCheck.mk(MctsSearchCheck.constant(0), "expand=0,mm=0,cache=0,reuse=true", 1);
		m.setMaxSims(40);
		m.findBestMove(root, 100000, h1);
		check(!m.lastReused, "prima ricerca: nessun albero da riusare");
		m.findBestMove(root, 100000, h1.copy());
		check(m.lastReused, "stessa posizione e stessa storia: l'albero si riusa");
		m.findBestMove(root, 100000, h2);
		check(!m.lastReused, "stessa posizione ma storia DIVERSA: l'albero NON si riusa (le ripetizioni dipendono dalla storia)");
		System.out.println("(8) riuso dell'albero MCTS: si riusa solo con stessa posizione E stessa storia");
	}

	// ---- (5) nessuna API di ricerca/tracciamento accetta storie a hash ------------------------------------------------------------------------------
	static void noHashHistoryInApis() {
		final Class<?>[] classes = { MoveSearcher.class, NegamaxSearcher.class, Negamax.class, Mcts.class, LegacyNegamax.class,
			it.unibo.ai.didattica.competition.tablut.training.GameTracker.class };
		int inspected = 0;
		for (Class<?> c : classes) {
			for (Method m : c.getDeclaredMethods()) {
				for (Class<?> t : m.getParameterTypes()) {
					check(!(t == java.util.List.class && m.getName().startsWith("findBestMove")), c.getSimpleName() + "." + m.getName() + " accetta ancora una List come storia");
				}
				if (m.getName().startsWith("findBestMove") || m.getName().equals("history")) {
					inspected++;
					final Class<?>[] pt = m.getParameterTypes();
					for (int i = 0; i + 1 < pt.length; i++) {
						check(!(pt[i] == long[].class && pt[i + 1] == int.class), c.getSimpleName() + "." + m.getName() + " accetta ancora (long[] hash, int len) come storia");
					}
					if (m.getName().equals("history")) {
						check(m.getReturnType() == PositionHistory.class, c.getSimpleName() + ".history() deve restituire PositionHistory, non " + m.getReturnType().getSimpleName());
					}
				}
			}
		}
		check(inspected >= 8, "copertura: troppo pochi metodi ispezionati (" + inspected + ")");
		System.out.println("(5) per riflessione: nessuno dei " + inspected + " metodi findBestMove/history ispezionati usa piu' storie a hash (List<Long> o long[] + lunghezza)");
	}
}
