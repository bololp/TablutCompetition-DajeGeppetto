package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.LockSupport;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Test del Lazy SMP e della transposition table lock-free (white-box: stesso package di {@link Negamax}).
 * <p>
 * Con piu' thread la ricerca non e' deterministica, quindi i test non confrontano "la stessa mossa di prima" ma
 * verificano PROPRIETA' che devono valere sempre:
 * <ol>
 * <li><b>TT</b>: uguale a un modello di riferimento indipendente a thread singolo (stessa politica di sostituzione, punteggio
 * esatto al bit); una voce "strappata" (campi di due scritture diverse, costruita a mano in tutte le combinazioni) deve dare
 * SEMPRE un miss; con 4 thread che martellano una tabella minuscola ogni hit deve avere un payload coerente con la chiave.</li>
 * <li><b>Logica</b>: schema di salto delle profondita' degli helper e regola di adozione del risultato.</li>
 * <li><b>Ricerca</b>: vittorie forzate trovate; mossa sempre legale; a profondita' fissa il valore coincide con la ricerca esatta
 * a thread singolo senza TT; vincoli di tempo; gli helper si fermano davvero; molte ricerche consecutive sulla stessa istanza;
 * un helper in errore non fa fallire la mossa; storia di ripetizione; adozione effettiva di un risultato piu' profondo;
 * configurazione (proprieta' di sistema, clamp, TT spenta = thread singolo).</li>
 * </ol>
 * NOTA: su una macchina con un solo core i thread si alternano per time-slicing: i test verificano correttezza e sicurezza,
 * NON lo speedup (che richiede core reali).
 */
public class LazySmpCheck {

	static int failures = 0;

	public static void main(String[] args) throws Exception {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 31337L;
		Random rnd = new Random(seed);

		checkTTAgainstModel(rnd);
		checkTTTornEntries();
		checkTTConcurrentStress();
		checkSkipPattern();
		checkChooseWorker();
		checkConfiguration();

		List<State> positions = SearchOptimizationsCheck.collectPositions(rnd, 24);
		Evaluator simple = new SimpleEvaluator();
		checkForcedWins(simple, rnd);
		checkLegalAndStats(simple, positions);
		checkFixedDepthAgreement(simple, positions);
		checkHelpersStop(simple, positions.get(5));
		checkRepeatedSearches(simple, positions);
		checkHelperFailureIsolated(simple, positions);
		checkRepetitionHistoryMT(simple);
		checkAdoption(simple, positions);

		if (failures == 0) {
			System.out.println("=== LAZY SMP CHECK OK ===");
		} else {
			System.out.println("=== LAZY SMP CHECK FALLITO: " + failures + " discrepanze ===");
			System.exit(1);
		}
	}

	static void fail(String m) {
		failures++;
		System.out.println("FAIL - " + m);
	}

	static boolean isHelperThread() {
		return Thread.currentThread().getName().startsWith("negamax-helper");
	}

	static boolean legal(State s, Action m) {
		if (m == null) {
			return false;
		}
		for (Action a : AshtonRules.generateMoves(s)) {
			if (SearchOptimizationsCheck.sameMove(a, m)) {
				return true;
			}
		}
		return false;
	}

	// ------------------------------------------------------------------ TT: modello di riferimento

	/** Riferimento a thread singolo, scritto direttamente dalla definizione (array paralleli come la TT originale). */
	static final class RefTT {
		final int mask;
		final long[] key;
		final int[] depth; // -1 = vuoto
		final double[] score;
		final byte[] flag;
		final int[] move;

		RefTT(int size) {
			mask = size - 1;
			key = new long[size];
			depth = new int[size];
			score = new double[size];
			flag = new byte[size];
			move = new int[size];
			java.util.Arrays.fill(depth, -1);
		}

		void store(long h, int d, double sc, byte f, int mv) {
			int i = (int) (h & mask);
			if (depth[i] == -1 || key[i] != h || d >= depth[i]) {
				key[i] = h;
				depth[i] = d;
				score[i] = sc;
				flag[i] = f;
				move[i] = mv;
			}
		}
	}

	static void checkTTAgainstModel(Random rnd) {
		int before = failures, ops = 0, hits = 0;
		for (int size : new int[] { 16, 64, 1024 }) {
			TranspositionTable tt = new TranspositionTable(size);
			RefTT ref = new RefTT(size);
			long[] pool = new long[300];
			for (int i = 0; i < pool.length; i++) {
				pool[i] = rnd.nextLong();
			}
			pool[0] = 0L;
			pool[1] = -1L;
			pool[2] = Long.MIN_VALUE;
			double[] special = { 0.0, -0.0, 1e-300, Double.MIN_VALUE, 1_000_000.0 + 7, -(1_000_000.0 + 3), 421.00939785026833, -1e15 };
			for (int op = 0; op < 70_000; op++) {
				long h = pool[rnd.nextInt(pool.length)];
				if (rnd.nextBoolean()) {
					int d = rnd.nextInt(31);
					double sc = rnd.nextInt(5) == 0 ? special[rnd.nextInt(special.length)] : rnd.nextGaussian() * 1000;
					byte f = (byte) rnd.nextInt(3);
					int mv = rnd.nextInt(3) == 0 ? -1 : (rnd.nextInt(9) << 12 | rnd.nextInt(9) << 8 | rnd.nextInt(9) << 4 | rnd.nextInt(9));
					tt.store(h, d, sc, f, mv);
					ref.store(h, d, sc, f, mv);
				} else {
					int need = rnd.nextInt(31);
					TranspositionTable.ProbeResult r = tt.probe(h, need);
					int i = (int) (h & ref.mask);
					boolean refHit = ref.depth[i] != -1 && ref.key[i] == h;
					ops++;
					if (r.hit != refHit) {
						fail("TT vs modello: hit=" + r.hit + " atteso " + refHit);
						return;
					}
					if (refHit) {
						hits++;
						boolean ok = Double.doubleToRawLongBits(r.score) == Double.doubleToRawLongBits(ref.score[i]) && r.flag == ref.flag[i]
								&& r.packedMove == ref.move[i] && r.depthSufficient == (ref.depth[i] >= need);
						if (!ok) {
							fail("TT vs modello: voce diversa (score " + r.score + " vs " + ref.score[i] + ", flag " + r.flag + " vs " + ref.flag[i]
									+ ", mossa " + r.packedMove + " vs " + ref.move[i] + ")");
							return;
						}
					}
				}
			}
		}
		if (hits < 1000) {
			fail("modello TT: troppi pochi hit (" + hits + "), test vacuo");
		}
		System.out.printf("%s %-28s %d interrogazioni (%d hit) su 3 dimensioni: identica al modello, punteggio esatto al bit%n",
				failures == before ? "OK  " : "FAIL", "[TT vs modello]", ops, hits);
	}

	// ------------------------------------------------------------------ TT: voci strappate

	static void checkTTTornEntries() {
		int before = failures, combos = 0;
		TranspositionTable t = new TranspositionTable(16);
		long hA = 0x1234_5678_9ABC_DEF0L;
		long hB = hA ^ (1L << 40); // stesso slot (stessi 4 bit bassi), chiave diversa
		if (t.slotOf(hA) != t.slotOf(hB)) {
			fail("setup: hA e hB non cadono nello stesso slot");
			return;
		}
		if (t.probe(0L, 0).hit) {
			fail("uno slot VUOTO risponde hit per la chiave 0");
		}
		t.store(hA, 5, 12.5, TranspositionTable.EXACT, 0x1234);
		long[] a = t.readRaw(hA);
		t.store(hB, 9, -7.25, TranspositionTable.LOWER_BOUND, 0x2345);
		long[] b = t.readRaw(hB);
		for (int mask = 1; mask < 7; mask++) { // 3 bit: 0=A, 1=B per (check, meta, score); si escludono AAA (0) e BBB (7)
			long c = ((mask & 1) == 0) ? a[0] : b[0];
			long m = ((mask & 2) == 0) ? a[1] : b[1];
			long sb = ((mask & 4) == 0) ? a[2] : b[2];
			t.writeRaw(hA, c, m, sb);
			combos++;
			if (t.probe(hA, 0).hit || t.probe(hB, 0).hit) {
				fail("voce strappata (combinazione " + mask + ") accettata come valida: dato falso usabile per tagli alpha-beta");
			}
		}
		t.writeRaw(hA, a[0], a[1], a[2]);
		TranspositionTable.ProbeResult ra = t.probe(hA, 5);
		if (!ra.hit || ra.score != 12.5 || ra.flag != TranspositionTable.EXACT || ra.packedMove != 0x1234 || !ra.depthSufficient || t.probe(hB, 0).hit) {
			fail("la voce A consistente non viene letta correttamente");
		}
		t.writeRaw(hB, b[0], b[1], b[2]);
		TranspositionTable.ProbeResult rb = t.probe(hB, 10);
		if (!rb.hit || rb.score != -7.25 || rb.flag != TranspositionTable.LOWER_BOUND || rb.packedMove != 0x2345 || rb.depthSufficient) {
			fail("la voce B consistente non viene letta correttamente");
		}
		System.out.printf("%s %-28s %d combinazioni di campi di scritture diverse: sempre miss; voci consistenti lette bene%n",
				failures == before ? "OK  " : "FAIL", "[TT voci strappate]", combos);
	}

	// ------------------------------------------------------------------ TT: stress concorrente

	// payload derivato dalla chiave: qualunque hit deve coincidere, altrimenti due scritture si sono mescolate
	static int dOf(long h) {
		return (int) ((h >>> 7) & 31);
	}

	static byte fOf(long h) {
		return (byte) (((h >>> 13) & 0x7FFFFFFF) % 3);
	}

	static int mOf(long h) {
		if (((h >>> 36) & 1) == 0) {
			return -1;
		}
		return ((int) ((h >>> 20) & 7) << 12) | ((int) ((h >>> 24) & 7) << 8) | ((int) ((h >>> 28) & 7) << 4) | (int) ((h >>> 32) & 7);
	}

	static double sOf(long h) {
		return ((h * 0x9E3779B97F4A7C15L) >> 11) / 1e3;
	}

	static void checkTTConcurrentStress() throws Exception {
		final TranspositionTable tt = new TranspositionTable(64); // tabella minuscola: collisioni continue
		final long[] pool = new long[4096];
		Random r0 = new Random(99);
		for (int i = 0; i < pool.length; i++) {
			pool[i] = r0.nextLong();
		}
		final int threads = 4, opsPerThread = 400_000;
		ExecutorService ex = Executors.newFixedThreadPool(threads);
		List<Future<long[]>> fs = new ArrayList<>();
		for (int t = 0; t < threads; t++) {
			final long sd = 1000 + t;
			fs.add(ex.submit(new Callable<long[]>() {
				@Override
				public long[] call() {
					Random r = new Random(sd);
					long hits = 0, bad = 0;
					for (int i = 0; i < opsPerThread; i++) {
						long h = pool[r.nextInt(pool.length)];
						if ((i & 1) == 0) {
							tt.store(h, dOf(h), sOf(h), fOf(h), mOf(h));
						} else {
							TranspositionTable.ProbeResult p = tt.probe(h, 0);
							if (p.hit) {
								hits++;
								if (p.score != sOf(h) || p.flag != fOf(h) || p.packedMove != mOf(h) || !p.depthSufficient) {
									bad++;
								}
							}
						}
					}
					return new long[] { hits, bad };
				}
			}));
		}
		long hits = 0, bad = 0;
		for (Future<long[]> f : fs) {
			long[] r = f.get();
			hits += r[0];
			bad += r[1];
		}
		ex.shutdown();
		if (bad > 0) {
			fail("TT concorrente: " + bad + " hit con payload incoerente con la chiave (voci mescolate)");
		}
		if (hits == 0) {
			fail("TT concorrente: nessun hit, test vacuo");
		}
		System.out.printf("%s %-28s 4 thread x %d operazioni su 64 slot: %d hit, %d incoerenti%n", failures == 0 || bad == 0 ? "OK  " : "FAIL",
				"[TT stress concorrente]", opsPerThread, hits, bad);
	}

	// ------------------------------------------------------------------ logica

	static void checkSkipPattern() {
		int before = failures;
		// helper 1: cerca le profondita' pari; helper 2: le dispari; helper 3: 1, 4-5, 8-9...
		boolean[][] searched = { { false, true, false, true, false, true, false, true }, // helper 1, profondita' 1..8
				{ true, false, true, false, true, false, true, false }, // helper 2
				{ true, false, false, true, true, false, false, true } }; // helper 3
		for (int w = 1; w <= 3; w++) {
			for (int d = 1; d <= 8; d++) {
				if (Negamax.skipDepth(w, d) == searched[w - 1][d - 1]) {
					fail("skipDepth(helper " + w + ", profondita' " + d + ") = " + Negamax.skipDepth(w, d) + ", atteso " + !searched[w - 1][d - 1]);
				}
			}
		}
		for (int d = 1; d <= 12; d++) {
			boolean covered = false;
			for (int w = 1; w <= 3; w++) {
				covered |= !Negamax.skipDepth(w, d);
			}
			if (!covered) {
				fail("la profondita' " + d + " non e' cercata da nessuno dei 3 helper");
			}
		}
		for (int w = 1; w <= 25; w++) { // anche oltre la lunghezza della tabella: nessuna eccezione
			Negamax.skipDepth(w, 5);
		}
		System.out.printf("%s %-28s helper 1-3 su profondita' 1..8 come da schema; ogni profondita' 1..12 coperta%n",
				failures == before ? "OK  " : "FAIL", "[schema di salto]");
	}

	static void checkChooseWorker() {
		int before = failures;
		double W = Negamax.WIN_SCORE;
		expectChoice("helper piu' profondo", 4, 10, new int[] { 5, 6, 5 }, new boolean[] { true, true, true }, 1);
		expectChoice("helper non utilizzabile ignorato", 4, 10, new int[] { 7, 6 }, new boolean[] { false, true }, 1);
		expectChoice("nessuno piu' profondo", 4, 10, new int[] { 3, 4, 2 }, new boolean[] { true, true, true }, -1);
		expectChoice("stessa profondita': resta il principale", 4, 10, new int[] { 4 }, new boolean[] { true }, -1);
		expectChoice("vittoria forzata del principale non si scavalca", 3, W + 3, new int[] { 9 }, new boolean[] { true }, -1);
		expectChoice("sconfitta forzata del principale non si scavalca", 3, -(W + 2), new int[] { 9 }, new boolean[] { true }, -1);
		expectChoice("parita' tra helper: il primo", 2, 0, new int[] { 6, 6 }, new boolean[] { true, true }, 0);
		expectChoice("nessun helper", 2, 0, new int[0], new boolean[0], -1);
		System.out.printf("%s %-28s 8 casi della regola di adozione del risultato%n", failures == before ? "OK  " : "FAIL", "[regola di adozione]");
	}

	static void expectChoice(String what, int mainDepth, double mainScore, int[] hd, boolean[] usable, int expected) {
		int got = Negamax.chooseWorker(mainDepth, mainScore, hd, usable);
		if (got != expected) {
			fail("chooseWorker (" + what + "): " + got + " invece di " + expected);
		}
	}

	static void checkConfiguration() {
		int before = failures;
		String old = System.getProperty("tablut.threads");
		try {
			System.clearProperty("tablut.threads");
			int cores = Runtime.getRuntime().availableProcessors();
			int expected = Math.max(1, Math.min(4, cores));
			if (Negamax.DEFAULT_THREADS != 4) {
				fail("DEFAULT_THREADS dovrebbe essere 4");
			}
			if (Negamax.defaultThreads() != expected) {
				fail("defaultThreads() = " + Negamax.defaultThreads() + ", atteso min(4, " + cores + ") = " + expected);
			}
			if (new Negamax(new SimpleEvaluator()).getThreads() != expected) {
				fail("il costruttore di default non usa defaultThreads()");
			}
			System.setProperty("tablut.threads", "3");
			if (Negamax.defaultThreads() != 3 || new Negamax(new SimpleEvaluator(), 4, true, 1 << 10).getThreads() != 3) {
				fail("la proprieta' tablut.threads=3 non viene rispettata");
			}
			System.setProperty("tablut.threads", "100");
			if (Negamax.defaultThreads() != Negamax.MAX_THREADS) {
				fail("tablut.threads=100 non viene limitato a " + Negamax.MAX_THREADS);
			}
			for (String bad : new String[] { "0", "-2", "abc", "" }) {
				System.setProperty("tablut.threads", bad);
				if (Negamax.defaultThreads() != expected) {
					fail("tablut.threads='" + bad + "' dovrebbe essere ignorata");
				}
			}
		} finally {
			if (old == null) {
				System.clearProperty("tablut.threads");
			} else {
				System.setProperty("tablut.threads", old);
			}
		}
		Negamax n = new Negamax(new SimpleEvaluator(), 4, true, 1 << 10, true, Negamax.ORDER_DEFAULT, 4);
		if (n.setThreads(0).getThreads() != 1 || n.setThreads(999).getThreads() != Negamax.MAX_THREADS) {
			fail("setThreads non limita i valori a [1, " + Negamax.MAX_THREADS + "]");
		}
		// TT disattivata = niente condivisione = thread singolo; thread=1 = comportamento storico
		State s = new it.unibo.ai.didattica.competition.tablut.domain.StateTablut();
		s.setTurn(Turn.WHITE);
		Negamax noTT = new Negamax(new SimpleEvaluator(), 3, false, 0, true, Negamax.ORDER_DEFAULT, 4);
		noTT.findBestMove(s, 600_000L);
		if (noTT.lastThreadsUsed != 1) {
			fail("senza TT la ricerca dovrebbe essere a thread singolo (usati " + noTT.lastThreadsUsed + ")");
		}
		Negamax one = new Negamax(new SimpleEvaluator(), 3, true, 1 << 12, true, Negamax.ORDER_DEFAULT, 1);
		one.findBestMove(s, 600_000L);
		if (one.lastThreadsUsed != 1 || one.lastTotalNodeCount != one.lastNodeCount || !one.helpersIdle()) {
			fail("threads=1: stato multi-thread inatteso");
		}
		System.out.printf("%s %-28s default 4 (limitato ai core), tablut.threads, clamp, TT spenta e threads=1 = thread singolo%n",
				failures == before ? "OK  " : "FAIL", "[configurazione]");
	}

	// ------------------------------------------------------------------ ricerca multi-thread

	static List<Action> winningMoves(State s) {
		List<Action> w = new ArrayList<>();
		Turn t = s.getTurn();
		for (Action a : AshtonRules.generateMoves(s)) {
			Turn after = AshtonRules.applyMove(s, a).getTurn();
			if ((t == Turn.WHITE && after == Turn.WHITEWIN) || (t == Turn.BLACK && after == Turn.BLACKWIN)) {
				w.add(a);
			}
		}
		return w;
	}

	static void checkForcedWins(Evaluator ev, Random rnd) {
		int before = failures;
		List<State> forced = new ArrayList<>();
		List<State> pool = SearchOptimizationsCheck.collectPositions(rnd, 3000);
		for (State s : pool) {
			for (Turn t : new Turn[] { Turn.WHITE, Turn.BLACK }) {
				State c = s.clone();
				c.setTurn(t);
				if (forced.size() < 14 && !AshtonRules.isTerminal(c) && !winningMoves(c).isEmpty()) {
					forced.add(c);
				}
			}
		}
		if (forced.size() < 6) {
			fail("trovate solo " + forced.size() + " posizioni con vittoria in una mossa: test vacuo");
			return;
		}
		Negamax mt = new Negamax(ev, 8, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		int found = 0;
		for (State s : forced) {
			Action m = mt.findBestMove(s, 1500);
			boolean wins = false;
			for (Action w : winningMoves(s)) {
				wins |= SearchOptimizationsCheck.sameMove(w, m);
			}
			if (!wins || mt.lastScore < Negamax.WIN_SCORE) {
				fail("vittoria in 1 NON trovata con 4 thread (mossa " + (m == null ? null : m.getFrom() + "->" + m.getTo()) + ", punteggio " + mt.lastScore + ")");
			} else {
				found++;
			}
		}
		mt.shutdown();
		System.out.printf("%s %-28s %d posizioni con vittoria in 1 mossa (Bianco e Nero): trovata sempre con 4 thread%n",
				failures == before ? "OK  " : "FAIL", "[vittorie forzate]", found);
	}

	static void checkLegalAndStats(Evaluator ev, List<State> positions) {
		int before = failures, searches = 0, withHelpers = 0, substantial = 0, helperWorked = 0;
		Negamax mt = new Negamax(ev, 40, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		for (State s : positions) {
			Action m = mt.findBestMove(s, 650);
			searches++;
			if (!legal(s, m)) {
				fail("mossa non legale con 4 thread");
			}
			if (mt.lastDepthReached < 1 || Double.isNaN(mt.lastScore)) {
				fail("risultato non valido: profondita' " + mt.lastDepthReached + ", punteggio " + mt.lastScore);
			}
			if (mt.lastTotalNodeCount < mt.lastNodeCount) {
				fail("nodi totali < nodi del principale");
			}
			if (mt.lastThreadsUsed > 1) {
				withHelpers++;
				// Se il principale finisce quasi subito (es. vittoria forzata a profondita' 1 in 15 nodi) gli helper vengono fermati
				// prima di valutare qualcosa: e' corretto. Solo se il principale ha lavorato abbastanza da far girare gli helper
				// (anche con un solo core, per time-slicing) i nodi totali devono di norma superare i suoi.
				if (mt.lastNodeCount >= 20_000) {
					substantial++;
					if (mt.lastTotalNodeCount > mt.lastNodeCount) {
						helperWorked++;
					}
				}
			}
		}
		mt.shutdown();
		if (withHelpers < searches / 2) {
			fail("gli helper sono partiti in meno della meta' delle ricerche (" + withHelpers + "/" + searches + ")");
		}
		if (substantial < 5) {
			fail("solo " + substantial + " ricerche abbastanza lunghe da verificare il lavoro degli helper: test vacuo");
		} else if (helperWorked < 0.8 * substantial) {
			fail("gli helper hanno contribuito nodi solo in " + helperWorked + "/" + substantial + " ricerche lunghe");
		}
		System.out.printf("%s %-28s %d ricerche da 650 ms con 4 thread: mosse legali, statistiche coerenti (helper partiti in %d, con lavoro osservato in %d/%d lunghe)%n",
				failures == before ? "OK  " : "FAIL", "[ricerca multi-thread]", searches, withHelpers, helperWorked, substantial);
	}

	/**
	 * A profondita' fissa (nessun limite di tempo, tutti i worker con lo stesso tetto di profondita') il valore del
	 * principale deve coincidere con la ricerca esatta a thread singolo senza TT. Gli helper possono aggiungere alla
	 * TT solo informazione di profondita' non superiore a quella necessaria.
	 */
	static void checkFixedDepthAgreement(Evaluator ev, List<State> positions) {
		int before = failures, compared = 0, mismatches = 0, forced = 0, earlyWins = 0;
		for (int p = 0; p < Math.min(12, positions.size()); p++) {
			State s = positions.get(p);
			for (int d = 1; d <= 3; d++) {
				Negamax exact = new Negamax(ev, d, false, 0, false, Negamax.ORDER_NONE, 1);
				exact.findBestMove(s, 600_000L);
				Negamax mt = new Negamax(ev, d, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
				Action m = mt.findBestMove(s, 600_000L);
				mt.shutdown();
				compared++;
				if (!legal(s, m)) {
					fail("mossa non legale a profondita' fissa");
				}
				boolean exactWin = exact.lastScore >= Negamax.WIN_SCORE, mtWin = mt.lastScore >= Negamax.WIN_SCORE;
				if (exactWin || mtWin) {
					// Vittoria forzata. Con la TT condivisa un helper (che sta cercando a profondita' maggiore) puo' aver gia' messo in TT
					// la vittoria: il principale la vede a un'iterazione PIU' BASSA e si ferma subito (profondita' minore, e punteggio
					// "WIN+r" con r diverso). E' il comportamento voluto, quindi non si richiede la stessa profondita' ne' lo stesso
					// punteggio: si richiede che la vittoria ci sia per ENTRAMBI e che la mossa restituita VINCA davvero entro d
					// semi-mosse secondo l'oracolo indipendente (misurato su 233 posizioni: 0 vittorie mancate, 0 mosse sbagliate).
					if (exactWin != mtWin) {
						mismatches++;
						System.out.println("  (info) pos " + p + " d=" + d + ": vittoria forzata vista solo da uno dei due (4 thread=" + mt.lastScore
								+ ", esatto=" + exact.lastScore + ")");
					} else {
						double v = SearchOptimizationsCheck.exactValueOfMove(ev, s, m, d, new ArrayList<Long>());
						if (v < Negamax.WIN_SCORE) {
							mismatches++;
							System.out.println("  (info) pos " + p + " d=" + d + ": la mossa dei 4 thread NON vince secondo l'oracolo (valore " + v + ")");
						}
						forced++;
						if (mt.lastDepthReached < exact.lastDepthReached) {
							earlyWins++;
						}
					}
				} else if (mt.lastDepthReached != exact.lastDepthReached || Math.abs(mt.lastScore - exact.lastScore) > 1e-9) {
					mismatches++;
					System.out.println("  (info) pos " + p + " d=" + d + ": 4 thread=" + mt.lastScore + " (prof. " + mt.lastDepthReached + ") esatto="
							+ exact.lastScore + " (prof. " + exact.lastDepthReached + ")");
				}
			}
		}
		if (mismatches > 0) {
			fail("a profondita' fissa " + mismatches + "/" + compared + " valori diversi dalla ricerca esatta");
		}
		System.out.printf("%s %-28s %d confronti (profondita' 1..3): 4 thread == ricerca esatta (valore e profondita'; per le %d vittorie forzate: mossa vincente verificata, %d viste in anticipo), %d divergenze%n",
				failures == before ? "OK  " : "FAIL", "[profondita' fissa]", compared, forced, earlyWins, mismatches);
	}

	/** Dopo che il principale ha finito, gli helper (resi LENTI apposta) devono fermarsi entro poche centinaia di ms, non a fine budget. */
	static void checkHelpersStop(final Evaluator base, State s) {
		int before = failures;
		Evaluator slowHelpers = new Evaluator() {
			@Override
			public double evaluate(State st) {
				if (isHelperThread()) {
					LockSupport.parkNanos(100_000); // ~100 us per valutazione: un helper da solo impiegherebbe secondi
				}
				return base.evaluate(st);
			}
		};
		Negamax n = new Negamax(slowHelpers, 4, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		long t0 = System.currentTimeMillis();
		Action m = n.findBestMove(s, 60_000L); // budget enorme: il principale termina per tetto di profondita'
		long elapsed = System.currentTimeMillis() - t0;
		if (!legal(s, m)) {
			fail("mossa non legale");
		}
		long limit = System.currentTimeMillis() + 900;
		while (!n.helpersIdle() && System.currentTimeMillis() < limit) {
			sleep(10);
		}
		boolean idle = n.helpersIdle();
		n.shutdown();
		if (!idle) {
			fail("gli helper sono ancora in esecuzione 900 ms dopo la fine della ricerca: il flag di stop non funziona");
		}
		if (elapsed > 3000) {
			fail("findBestMove ha impiegato " + elapsed + " ms invece di terminare col principale");
		}
		System.out.printf("%s %-28s principale finito in %d ms (budget 60 s): helper lenti fermati entro 900 ms%n",
				failures == before ? "OK  " : "FAIL", "[stop degli helper]", elapsed);
	}

	static void sleep(long ms) {
		try {
			Thread.sleep(ms);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	static void checkRepeatedSearches(Evaluator ev, List<State> positions) {
		int before = failures, runs = 0, errors = 0;
		Random rnd = new Random(5);
		long[] budgets = { 1, 20, 60, 120, 300 };
		Negamax mt = new Negamax(ev, 40, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		for (int i = 0; i < 40; i++) {
			State s = positions.get(rnd.nextInt(positions.size()));
			Action m = mt.findBestMove(s, budgets[rnd.nextInt(budgets.length)]);
			runs++;
			errors += mt.lastHelperErrors;
			if (!legal(s, m)) {
				fail("ricerca ripetuta #" + i + ": mossa non legale");
			}
		}
		long limit = System.currentTimeMillis() + 1500;
		while (!mt.helpersIdle() && System.currentTimeMillis() < limit) {
			sleep(10);
		}
		if (!mt.helpersIdle()) {
			fail("dopo 40 ricerche consecutive gli helper non tornano inattivi");
		}
		if (errors > 0) {
			fail(errors + " errori negli helper durante le ricerche ripetute");
		}
		mt.shutdown();
		System.out.printf("%s %-28s %d ricerche consecutive (budget 1-300 ms) sulla stessa istanza: tutte legali, 0 errori, helper inattivi alla fine%n",
				failures == before ? "OK  " : "FAIL", "[ricerche ripetute]", runs);
	}

	static void checkHelperFailureIsolated(final Evaluator base, List<State> positions) {
		int before = failures, withErrors = 0, runs = 0;
		Evaluator failing = new Evaluator() {
			@Override
			public double evaluate(State st) {
				if (isHelperThread()) {
					throw new IllegalStateException("errore simulato in un helper");
				}
				return base.evaluate(st);
			}
		};
		Negamax mt = new Negamax(failing, 40, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		for (int i = 0; i < 6; i++) {
			State s = positions.get(i);
			Action m = mt.findBestMove(s, 600);
			runs++;
			if (!legal(s, m) || mt.lastDepthReached < 1) {
				fail("con helper in errore la mossa non e' valida");
			}
			if (mt.lastHelperErrors > 0) {
				withErrors++;
			}
		}
		mt.shutdown();
		if (withErrors == 0) {
			fail("gli errori degli helper non sono mai stati rilevati: test vacuo");
		}
		System.out.printf("%s %-28s %d ricerche con helper che lanciano eccezioni: mossa sempre valida, errori rilevati in %d%n",
				failures == before ? "OK  " : "FAIL", "[helper in errore]", runs, withErrors);
	}

	static void checkRepetitionHistoryMT(Evaluator ev) {
		int before = failures;
		Object[] cyc = SearchOptimizationsCheck.buildCycle();
		State p0 = (State) cyc[0], p1 = (State) cyc[1], p2 = (State) cyc[2], p3 = (State) cyc[3];
		List<Long> history = new ArrayList<>();
		for (State p : new State[] { p0, p1, p2, p3 }) {
			history.add(ZobristHasher.fullHash(p));
		}
		Negamax mt = new Negamax(ev, 40, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		int runs = 0;
		for (int i = 0; i < 5; i++) {
			Action m = mt.findBestMove(p3, 600, history);
			runs++;
			if (!legal(p3, m) || Double.isNaN(mt.lastScore)) {
				fail("storia di ripetizione con 4 thread: risultato non valido");
			}
		}
		if (history.size() != 4) {
			fail("la lista di storia del chiamante e' stata modificata");
		}
		mt.shutdown();
		System.out.printf("%s %-28s %d ricerche con storia reale (ciclo di 4 posizioni) e 4 thread: legali, storia del chiamante intatta%n",
				failures == before ? "OK  " : "FAIL", "[ripetizione]", runs);
	}

	/**
	 * Il principale e' reso lento apposta (gli helper no): gli helper arrivano a profondita' maggiori e il loro risultato
	 * deve essere ADOTTATO (profondita' finale > profondita' del principale), restando una mossa legale.
	 */
	static void checkAdoption(final Evaluator base, List<State> positions) {
		int before = failures, adopted = 0, trials = 0;
		Evaluator slowMain = new Evaluator() {
			@Override
			public double evaluate(State st) {
				if (!isHelperThread()) {
					LockSupport.parkNanos(3_000_000); // 3 ms: misurato, adozione in 12 prove su 12 (con 150 us solo in 6/12)
				}
				return base.evaluate(st);
			}
		};
		Negamax mt = new Negamax(slowMain, 40, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 4);
		for (int i = 6; i < 14 && i < positions.size(); i++) {
			State s = positions.get(i);
			Action m = mt.findBestMove(s, 900);
			trials++;
			if (!legal(s, m)) {
				fail("mossa non legale dopo adozione");
			}
			if (mt.lastAdoptedWorker >= 1) {
				adopted++;
				if (mt.lastDepthReached <= mt.lastMainDepth) {
					fail("adottato un helper ma la profondita' non e' maggiore di quella del principale (" + mt.lastDepthReached + " <= " + mt.lastMainDepth + ")");
				}
			}
		}
		mt.shutdown();
		if (adopted < 2) { // misurato 4-6 su 8 con 1 core; la REGOLA e' coperta dai casi unitari, qui si esercita il percorso reale
			fail("risultato di un helper adottato solo in " + adopted + "/" + trials + " prove con principale rallentato");
		}
		System.out.printf("%s %-28s %d prove con principale lento: risultato di un helper piu' profondo adottato in %d, sempre legale%n",
				failures == before ? "OK  " : "FAIL", "[adozione del risultato]", trials, adopted);
	}
}
