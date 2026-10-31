package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Verifica la catena a chiave esatta: {@link ExactTranspositionTable} e {@link CachingEvaluator} con {@code stateKey=true}.
 * (1) PARITA': con le stesse operazioni (store/probe/newSearch, a slot singolo e a due slot, tabelle minuscole che sostituiscono di
 * continuo) restituisce gli stessi risultati di {@link TranspositionTable}; (2) ESATTEZZA: una collisione di hash FORZATA (due posizioni
 * diverse con lo stesso Zobrist) inganna la tabella a hash ma non quella esatta, e ogni cella e il turno contano; (3) protocollo
 * seqlock deterministico (scrittore fermo, chiavi "miste"); (4) 8 thread su tabelle minuscole, mai dati di un'altra posizione;
 * (5) RICERCA con la catena esatta e con quella a hash: stessa mossa, punteggio, nodi e profondita', ordinamento base e rec, slot singolo
 * e doppio, Simple/MLP/CNN; (6) cache con chiave dallo stato == cache che ricodifica; (7) nessuna allocazione; (8) Lazy SMP a 4 thread;
 * (9) memoria. Esce con codice 1 se qualcosa fallisce. Uso: {@code ExactKeyStackCheck [seme]}.
 */
public class ExactKeyStackCheck {

	static int failures = 0;

	static void check(boolean ok, String what) {
		if (!ok) {
			failures++;
			if (failures <= 40) {
				System.out.println("FALLITO: " + what);
			}
		}
	}

	static boolean same(double a, double b) {
		return Double.doubleToRawLongBits(a) == Double.doubleToRawLongBits(b);
	}

	/** Pool di posizioni con Zobrist e chiave coerenti, e con molte varianti a una cella di distanza. */
	static List<FastState> pool(long seed, int base, int variants) {
		List<FastState> out = new ArrayList<>(CachingEvaluatorCheck.positions(seed, base));
		Random rnd = new Random(seed + 1);
		byte[] codes = { 0, 1, 2, 4 };
		for (int i = 0; i < variants; i++) {
			FastState v = FastState.copyOf(out.get(rnd.nextInt(Math.min(60, base))));
			v.board[rnd.nextInt(81)] = codes[rnd.nextInt(4)];
			v.zobristHash = v.computeHash();
			v.computeKeys();
			out.add(v);
		}
		return out;
	}

	public static void main(String[] args) throws Exception {
		final long seed = args.length > 0 ? Long.parseLong(args[0]) : 2468L;
		final List<FastState> pool = pool(seed, 400, 400);
		parity(pool);
		exactness(pool);
		seqlock(pool);
		concurrency(pool);
		keyedCache(pool);
		searchEquivalence(pool);
		noAllocation(pool);
		lazySmp(pool);
		memory();
		System.out.println(failures == 0 ? "=== EXACT KEY STACK CHECK OK ===" : "=== FALLITI: " + failures + " ===");
		System.exit(failures == 0 ? 0 : 1);
	}

	// ---- (1) parita' con la TT a hash ----------------------------------------------------------------------------------------------------
	static void parity(List<FastState> pool) {
		long ops = 0, probes = 0, hits = 0;
		for (boolean two : new boolean[] { false, true }) {
			for (int bits : new int[] { 4, 6, 10 }) {
				TranspositionTable h = new TranspositionTable(1 << bits, two);
				ExactTranspositionTable x = new ExactTranspositionTable(1 << bits, two);
				check(h.capacity() == x.capacity() && h.isTwoSlot() == x.isTwoSlot(), "capacita'/two-slot diversi");
				Random r = new Random(77 + bits + (two ? 1000 : 0));
				TranspositionTable.ProbeResult ph = new TranspositionTable.ProbeResult(), px = new TranspositionTable.ProbeResult();
				for (int k = 0; k < 250_000; k++) {
					FastState s = pool.get(r.nextInt(pool.size()));
					int op = r.nextInt(12);
					ops++;
					if (op <= 6) {
						int depth = r.nextInt(14);
						double score = r.nextInt(5) == 0 ? (r.nextBoolean() ? 1e6 + depth : -1e6 - depth) : r.nextDouble() * 2000 - 1000;
						byte flag = (byte) r.nextInt(3);
						int mv = r.nextInt(3) == 0 ? -1 : r.nextInt(0xFFFF);
						h.store(s.zobristHash, depth, score, flag, mv);
						x.store(s, depth, score, flag, mv);
					} else if (op == 7) {
						h.newSearch();
						x.newSearch();
					} else {
						int need = r.nextInt(14);
						h.probeInto(s.zobristHash, need, ph);
						x.probeInto(s, need, px);
						probes++;
						if (ph.hit) {
							hits++;
						}
						boolean eq = ph.hit == px.hit && ph.depthSufficient == px.depthSufficient && same(ph.score, px.score) && ph.flag == px.flag && ph.packedMove == px.packedMove;
						check(eq, "parita': risultato diverso dalla TT a hash (" + (two ? "due slot" : "slot singolo") + ", 2^" + bits + ")");
					}
				}
				check(h.occupancy(1 << bits) == x.occupancy(1 << bits), "occupazione diversa a fine sequenza");
			}
		}
		check(hits > 10_000, "copertura: pochi hit nel test di parita' (" + hits + ")");
		System.out.println("(1) parita' con la TT a hash: " + ops + " operazioni, " + probes + " probe (" + hits + " hit), slot singolo e due slot, 16/64/1024 voci: risultati identici");
	}

	// ---- (2) esattezza: collisione di hash forzata ---------------------------------------------------------------------------------------------
	static void exactness(List<FastState> pool) {
		FastState a = FastState.copyOf(pool.get(7));
		FastState b = FastState.copyOf(a);
		b.board[0] = b.board[0] == BoardConstants.EMPTY ? BoardConstants.BLACK : BoardConstants.EMPTY;
		b.computeKeys();
		b.zobristHash = a.zobristHash; // collisione FORZATA: due posizioni diverse con lo stesso hash
		check(a.key0 != b.key0 || a.key1 != b.key1 || a.key2 != b.key2, "precondizione: chiavi diverse");
		TranspositionTable.ProbeResult r = new TranspositionTable.ProbeResult();
		for (boolean two : new boolean[] { false, true }) {
			TranspositionTable h = new TranspositionTable(1 << 10, two);
			h.store(a.zobristHash, 5, 123.5, TranspositionTable.EXACT, 42);
			h.probeInto(b.zobristHash, 0, r);
			check(r.hit && same(r.score, 123.5), "dimostrazione: la TT a hash viene ingannata dalla collisione forzata (e' il rischio che la chiave esatta elimina)");
			ExactTranspositionTable x = new ExactTranspositionTable(1 << 10, two);
			x.store(a, 5, 123.5, TranspositionTable.EXACT, 42);
			x.probeInto(b, 0, r);
			check(!r.hit, "la TT esatta NON deve servire la voce di un'altra posizione con lo stesso hash");
			x.probeInto(a, 0, r);
			check(r.hit && same(r.score, 123.5) && r.packedMove == 42, "la TT esatta serve la posizione giusta");
			x.store(b, 3, -7.25, TranspositionTable.LOWER_BOUND, 9); // meno profonda di A (5): con due slot va nello slot 1 e non sostituisce A
			x.probeInto(b, 0, r);
			check(r.hit && same(r.score, -7.25) && r.flag == TranspositionTable.LOWER_BOUND, "dopo lo store di B: B si legge");
			x.probeInto(a, 0, r);
			if (!two) {
				check(!r.hit, "slot singolo: B ha sostituito A (stesso slot), A non deve risultare un hit con i dati di B");
			} else {
				check(r.hit && same(r.score, 123.5), "due slot: A e B convivono nel bucket, ognuno con i propri dati");
			}
		}
		// ogni cella e il turno contano, anche con lo stesso Zobrist (tutte le varianti nello stesso slot)
		FastState base = FastState.copyOf(pool.get(11));
		ExactTranspositionTable x = new ExactTranspositionTable(1 << 12, false);
		x.store(base, 4, 1.0, TranspositionTable.EXACT, -1);
		int variants = 0;
		final byte[] codes = { 0, 1, 2, 4 };
		for (int sq = 0; sq < 81; sq++) {
			for (byte c : codes) {
				if (c == base.board[sq]) {
					continue;
				}
				FastState v = FastState.copyOf(base);
				v.board[sq] = c;
				v.computeKeys();
				x.probeInto(v, 0, r);
				check(!r.hit, "variante alla cella " + sq + " servita come hit");
				variants++;
			}
		}
		FastState t = FastState.copyOf(base);
		t.turn = (byte) (t.turn ^ 1);
		t.computeKeys();
		x.probeInto(t, 0, r);
		check(!r.hit, "il turno diverso non deve dare un hit");
		x.probeInto(base, 0, r);
		check(r.hit, "la posizione originale e' ancora li'");
		System.out.println("(2) collisione di hash forzata: la TT a hash viene ingannata, quella esatta no; " + (variants + 1) + " varianti (ogni cella/turno) tutte miss con lo stesso Zobrist");
	}

	// ---- (3) seqlock deterministico -------------------------------------------------------------------------------------------------------------
	static AtomicLongArray rawSlots(ExactTranspositionTable t) throws Exception {
		Field f = ExactTranspositionTable.class.getDeclaredField("slots");
		f.setAccessible(true);
		return (AtomicLongArray) f.get(t);
	}

	static void seqlock(List<FastState> pool) throws Exception {
		final long VER = 1L << 32;
		FastState a = pool.get(3);
		TranspositionTable.ProbeResult r = new TranspositionTable.ProbeResult();
		ExactTranspositionTable t = new ExactTranspositionTable(16, false);
		AtomicLongArray raw = rawSlots(t);
		t.store(a, 6, 55.5, TranspositionTable.EXACT, 3);
		int base = -1;
		for (int b = 0; b + 4 < raw.length(); b += 5) {
			if ((raw.get(b + 4) & 0xFF) != 0) {
				check(base < 0, "piu' di una voce occupata");
				base = b;
			}
		}
		check(base >= 0, "nessuna voce occupata dopo lo store");
		long ctl = raw.get(base + 4);
		check((ctl & VER) == 0, "dopo una scrittura completa la versione e' pari");
		t.probeInto(a, 0, r);
		check(r.hit, "voce coerente: hit");
		raw.set(base + 4, ctl + VER); // scrittore "congelato" dopo il CAS
		t.probeInto(a, 0, r);
		check(!r.hit, "versione dispari: miss");
		long[] before = new long[5];
		for (int w = 0; w < 5; w++) {
			before[w] = raw.get(base + w);
		}
		FastState other = FastState.copyOf(a);
		other.board[2] = other.board[2] == BoardConstants.EMPTY ? BoardConstants.BLACK : BoardConstants.EMPTY;
		other.computeKeys();
		other.zobristHash = a.zobristHash; // stesso slot
		t.store(a, 9, 1.0, TranspositionTable.EXACT, 1);
		t.store(other, 9, 2.0, TranspositionTable.EXACT, 2);
		boolean untouched = true;
		for (int w = 0; w < 5; w++) {
			untouched &= raw.get(base + w) == before[w];
		}
		check(untouched, "versione dispari: la scrittura concorrente dev'essere saltata e NESSUNA delle 5 parole dello slot deve cambiare");
		// la protezione dentro write() (guardia + CAS) conta solo in una corsa vera fra inspect() e write(): la esercito invocando write() direttamente
		Method wm = ExactTranspositionTable.class.getDeclaredMethod("write", int.class, long.class, long.class, long.class, long.class, double.class);
		wm.setAccessible(true);
		wm.invoke(t, base, 111L, 222L, 333L, TranspositionTable.packMeta(7, TranspositionTable.EXACT, 5), 9.5);
		boolean untouchedDirect = true;
		for (int i = 0; i < 5; i++) {
			untouchedDirect &= raw.get(base + i) == before[i];
		}
		check(untouchedDirect, "write() su uno slot con versione dispari non deve scrivere nulla (guardia + CAS)");
		raw.set(base + 4, ctl);
		t.probeInto(a, 0, r);
		check(r.hit && same(r.score, 55.5), "ripristino: hit");
		long k0 = raw.get(base), k1 = raw.get(base + 1), tl = raw.get(base + 2);
		long[][] cor = { { base, k0 ^ 1L }, { base, k0 ^ (1L << 63) }, { base + 1, k1 ^ 1L }, { base + 1, k1 ^ (1L << 62) }, { base + 2, tl ^ 1L }, { base + 2, tl ^ (1L << 34) } };
		for (long[] c : cor) {
			long orig = raw.get((int) c[0]);
			raw.set((int) c[0], c[1]);
			t.probeInto(a, 0, r);
			check(!r.hit, "parola " + (c[0] - base) + " alterata: dev'essere un miss (chiave completa)");
			raw.set((int) c[0], orig);
		}
		t.probeInto(a, 0, r);
		check(r.hit, "dopo il ripristino delle parole: hit");
		// slot in scrittura e a due slot: lo store rinuncia, non sovrascrive
		ExactTranspositionTable t2 = new ExactTranspositionTable(16, true);
		AtomicLongArray raw2 = rawSlots(t2);
		FastState c1 = pool.get(5);
		t2.store(c1, 3, 9.0, TranspositionTable.EXACT, -1);
		int occ = -1;
		for (int b = 0; b + 4 < raw2.length(); b += 5) {
			if ((raw2.get(b + 4) & 0xFF) != 0) {
				occ = b;
			}
		}
		long c0 = raw2.get(occ + 4);
		raw2.set(occ + 4, c0 + VER);
		long[] before2 = new long[5];
		for (int w = 0; w < 5; w++) {
			before2[w] = raw2.get(occ + w);
		}
		FastState d = FastState.copyOf(c1);
		d.board[1] = d.board[1] == BoardConstants.EMPTY ? BoardConstants.BLACK : BoardConstants.EMPTY;
		d.computeKeys();
		d.zobristHash = c1.zobristHash;
		t2.store(d, 8, 2.0, TranspositionTable.EXACT, -1);
		boolean untouched2 = true;
		for (int w = 0; w < 5; w++) {
			untouched2 &= raw2.get(occ + w) == before2[w];
		}
		check(untouched2, "due slot: con uno slot in scrittura lo store rinuncia e lo slot non cambia (5 parole)");
		System.out.println("(3) seqlock deterministico: versione dispari = miss e scrittura saltata; k0/k1/coda alterate = miss; ripristino = hit");
	}

	// ---- (4) concorrenza -------------------------------------------------------------------------------------------------------------------------------
	static double fScore(FastState s) {
		long h = s.key0 * 0x9E3779B97F4A7C15L ^ s.key1 * 0xC2B2AE3D27D4EB4FL ^ s.key2 * 0x165667B19E3779F9L;
		h ^= h >>> 29;
		return (h & 0xFFFFFFL) / 1000.0 - 8000.0;
	}

	static void concurrency(final List<FastState> pool) throws Exception {
		for (final boolean two : new boolean[] { false, true }) {
			for (final int bits : new int[] { 4, 8 }) {
				final ExactTranspositionTable t = new ExactTranspositionTable(1 << bits, two);
				final AtomicInteger bad = new AtomicInteger(), hitsN = new AtomicInteger();
				Thread[] th = new Thread[8];
				for (int i = 0; i < th.length; i++) {
					final int id = i;
					th[i] = new Thread(new Runnable() {
						@Override
						public void run() {
							Random r = new Random(500 + id);
							TranspositionTable.ProbeResult pr = new TranspositionTable.ProbeResult();
							for (int k = 0; k < 300_000; k++) {
								FastState s = pool.get(r.nextInt(pool.size()));
								double sc = fScore(s);
								int depth = (int) (Math.abs(sc) % 11);
								if (r.nextInt(3) == 0) {
									t.store(s, depth, sc, (byte) (depth % 3), depth * 7);
								} else {
									t.probeInto(s, 0, pr);
									if (pr.hit) {
										hitsN.incrementAndGet();
										if (!same(pr.score, sc) || pr.flag != (byte) (depth % 3) || pr.packedMove != depth * 7) {
											bad.incrementAndGet();
										}
									}
								}
								if ((k & 1023) == 0) {
									Thread.yield();
								}
							}
						}
					});
					th[i].start();
				}
				for (Thread x : th) {
					x.join();
				}
				check(bad.get() == 0, (two ? "due slot" : "slot singolo") + " 2^" + bits + ": " + bad.get() + " hit con dati di un'altra posizione");
				check(hitsN.get() > 0, "nessun hit nel test di concorrenza (" + (two ? "due slot" : "slot singolo") + " 2^" + bits + ")");
				System.out.println("(4) 8 thread x 300000 operazioni, TT esatta " + (two ? "a due slot" : "a slot singolo") + " da " + (1 << bits) + " voci: hit=" + hitsN.get() + ", dati di un'altra posizione = " + bad.get());
			}
		}
	}

	// ---- (6) cache con chiave dallo stato -------------------------------------------------------------------------------------------------------------
	static void keyedCache(List<FastState> pool) {
		NeuralEvaluator mlp = NeuralEvaluator.randomInit(21);
		CnnEvaluator cnn = CnnEvaluator.randomInit(CnnEvaluator.defaultConfig(TablutFeatures.G_DEFAULT), TablutFeatures.G_DEFAULT, 22);
		Evaluator[] nets = { mlp, cnn };
		String[] names = { "MLP", "CNN" };
		for (int e = 0; e < 2; e++) {
			CachingEvaluator keyed = new CachingEvaluator(nets[e], 16, true, true), enc = new CachingEvaluator(nets[e], 16, true, false);
			check(keyed.isStateKey() && !enc.isStateKey(), "flag stateKey");
			Random r = new Random(9 + e);
			int bad = 0;
			for (int k = 0; k < 30_000; k++) {
				FastState s = pool.get(r.nextInt(pool.size()));
				double direct = nets[e].evaluate(s);
				if (!same(keyed.evaluate(s), direct) || !same(enc.evaluate(s), direct)) {
					bad++;
				}
			}
			check(bad == 0, names[e] + ": " + bad + " valori diversi dal diretto (chiave dallo stato / ricodificata)");
			check(keyed.hits() == enc.hits() && keyed.calls() == enc.calls(), names[e] + ": le due modalita' di chiave devono dare gli stessi hit (" + keyed.hits() + " vs " + enc.hits() + ")");
			check(keyed.bypasses() == 0, names[e] + ": nessun salto della cache con la chiave dallo stato");
			System.out.println("(6) " + names[e] + ": cache con chiave dallo stato == cache che ricodifica (stesso valore al bit e stessi hit: " + keyed.hits() + "/" + keyed.calls() + ")");
		}
		// con la chiave dallo stato ogni cella e il turno contano (richiede computeKeys dopo una scrittura a mano)
		PureCounter inner = new PureCounter();
		CachingEvaluator c = new CachingEvaluator(inner, 20, true, true);
		FastState base = FastState.copyOf(pool.get(5));
		c.evaluate(base);
		int before = inner.calls;
		int n = 0;
		for (int sq = 0; sq < 81; sq++) {
			for (byte code : new byte[] { 0, 1, 2, 4 }) {
				if (code == base.board[sq]) {
					continue;
				}
				FastState v = FastState.copyOf(base);
				v.board[sq] = code;
				v.computeKeys();
				check(same(c.evaluate(v), PureCounter.value(v)), "chiave dallo stato: valore di un'altra posizione");
				n++;
			}
		}
		check(inner.calls - before == n, "chiave dallo stato: ogni variante dev'essere un miss (" + (inner.calls - before) + " su " + n + ")");
		System.out.println("(6) chiave dallo stato: " + n + " varianti a una cella, tutte miss con valore corretto");
	}

	static final class PureCounter implements Evaluator {
		int calls;

		static double value(FastState s) {
			return CachingEvaluatorCheck.Fake.value(s);
		}

		@Override
		public double evaluate(FastState state) {
			calls++;
			return value(state);
		}
	}

	// ---- (5) ricerca: stessa mossa/punteggio/nodi con la catena esatta ------------------------------------------------------------------------------
	static void searchEquivalence(List<FastState> pool) {
		Evaluator simple = new SimpleEvaluator();
		NeuralEvaluator mlp = NeuralEvaluator.randomInit(31);
		CnnEvaluator cnn = CnnEvaluator.randomInit(CnnEvaluator.defaultConfig(TablutFeatures.G_DEFAULT), TablutFeatures.G_DEFAULT, 32);
		Object[][] cfg = { { "Simple", simple, 4 }, { "MLP", mlp, 4 }, { "CNN", cnn, 3 } };
		final int[] orders = { Negamax.ORDER_DEFAULT, OrderingSpec.parse("rec") };
		final String[] oname = { "base", "rec" };
		for (Object[] c : cfg) {
			final String name = (String) c[0];
			final Evaluator ev = (Evaluator) c[1];
			final int depth = (Integer) c[2];
			for (int o = 0; o < 2; o++) {
				for (boolean two : new boolean[] { false, true }) {
					Negamax plain = new Negamax(ev, depth, true, 1 << 15, true, orders[o], 1, two);
					Evaluator exEv = CachingEvaluator.isCacheable(ev) ? new CachingEvaluator(ev, 18, true, true) : ev;
					Negamax exact = new Negamax(exEv, depth, true, 1 << 15, true, orders[o], 1, two, new ExactTranspositionTable(1 << 15, two));
					check(exact.transpositionTable() instanceof ExactTranspositionTable, "il Negamax deve usare la TT esatta fornita");
					int n = 0;
					for (int i = 0; i < 400 && n < 6; i += 53, n++) {
						FastState root = pool.get(i);
						int m1 = plain.findBestMove(root, 1_000_000_000L, new long[0], 0);
						int m2 = exact.findBestMove(root, 1_000_000_000L, new long[0], 0);
						String w = name + "/" + oname[o] + "/" + (two ? "2slot" : "1slot") + " pos " + i;
						check(m1 == m2, w + ": mossa diversa");
						check(same(plain.lastScore, exact.lastScore), w + ": punteggio diverso (" + plain.lastScore + " vs " + exact.lastScore + ")");
						check(plain.lastNodeCount == exact.lastNodeCount, w + ": nodi diversi (" + plain.lastNodeCount + " vs " + exact.lastNodeCount + ")");
						check(plain.lastDepthReached == exact.lastDepthReached, w + ": profondita' diversa");
					}
					if (exEv instanceof CachingEvaluator) {
						check(((CachingEvaluator) exEv).hits() > 0, name + ": la cache dovrebbe avere hit");
					}
				}
			}
			System.out.println("(5) " + name + " d=" + depth + ": catena esatta (TT esatta + cache dallo stato) == catena a hash: stessa mossa/punteggio/nodi/profondita' (base e rec, slot singolo e doppio, TT da 32768 voci con sostituzioni)");
		}
	}

	// ---- (7) allocazioni -------------------------------------------------------------------------------------------------------------------------------------
	static void noAllocation(List<FastState> pool) {
		Method m;
		Object bean = ManagementFactory.getThreadMXBean();
		try {
			m = Class.forName("com.sun.management.ThreadMXBean").getMethod("getThreadAllocatedBytes", long.class);
			m.setAccessible(true);
			m.invoke(bean, Thread.currentThread().getId());
		} catch (Throwable t) {
			System.out.println("(7) SALTATO: getThreadAllocatedBytes non disponibile (" + t + ")");
			return;
		}
		try {
			final long id = Thread.currentThread().getId();
			for (boolean two : new boolean[] { false, true }) {
				Negamax nm = new Negamax(new SimpleEvaluator(), 4, true, 1 << 16, true, OrderingSpec.parse("rec"), 1, two, new ExactTranspositionTable(1 << 16, two));
				for (int i = 0; i < 4; i++) {
					nm.findBestMove(pool.get(i * 7), 1_000_000_000L, new long[0], 0); // scaldamento
				}
				long a0 = (Long) m.invoke(bean, id);
				long nodes = 0;
				for (int i = 4; i < 8; i++) {
					nm.findBestMove(pool.get(i * 7), 1_000_000_000L, new long[0], 0);
					nodes += nm.lastNodeCount;
				}
				long a1 = (Long) m.invoke(bean, id);
				double per = (a1 - a0) / (double) nodes;
				check(per < 1.0, "allocazioni per nodo con la TT esatta (" + (two ? "due slot" : "slot singolo") + "): " + per + " B/nodo");
				System.out.printf("(7) TT esatta %s: %.4f B/nodo allocati su %d nodi (soglia 1)%n", two ? "a due slot" : "a slot singolo", per, nodes);
			}
		} catch (Throwable t) {
			check(false, "misura allocazioni fallita: " + t);
		}
	}

	// ---- (8) Lazy SMP -----------------------------------------------------------------------------------------------------------------------------------------
	static void lazySmp(List<FastState> pool) {
		NeuralEvaluator mlp = NeuralEvaluator.randomInit(41);
		CachingEvaluator ce = new CachingEvaluator(mlp, 16, true, true);
		Negamax smp = new Negamax(ce, 40, true, 1 << 16, true, OrderingSpec.parse("rec"), 4, false, new ExactTranspositionTable(1 << 16, false)).setThreads(4);
		int[] buf = new int[FastState.MAX_MOVES];
		for (int i = 0; i < 3; i++) {
			FastState root = pool.get(10 + i * 31);
			int n = root.generateMoves(buf);
			int mv = smp.findBestMove(root, 400, new long[0], 0);
			boolean legal = false;
			for (int k = 0; k < n; k++) {
				legal |= buf[k] == mv;
			}
			check(legal, "Lazy SMP con TT esatta: mossa non legale alla posizione " + i);
		}
		System.out.printf("(8) Lazy SMP a 4 thread con TT esatta e cache dallo stato: mosse legali, hit cache %.1f%% su %d lookup%n", 100 * ce.hitRate(), ce.calls());
	}

	// ---- (9) memoria e API ------------------------------------------------------------------------------------------------------------------------------------
	static void memory() {
		check(ExactTranspositionTable.ENTRY_BYTES == 40, "40 byte per voce");
		check(ExactTranspositionTable.bytesFor(1 << 22) == 160L << 20, "2^22 voci = 160 MiB");
		ExactTranspositionTable t = new ExactTranspositionTable(1000, true);
		check(t.capacity() == 1024 && t.isTwoSlot(), "capacita' arrotondata a potenza di 2");
		boolean a = false, b = false;
		try {
			t.probeInto(123L, 0, new TranspositionTable.ProbeResult());
		} catch (UnsupportedOperationException e) {
			a = true;
		}
		try {
			t.store(123L, 0, 0.0, (byte) 0, -1);
		} catch (UnsupportedOperationException e) {
			b = true;
		}
		check(a && b, "le API a solo hash devono lanciare UnsupportedOperationException");
		t.clear();
		check(t.occupancy(1024) == 0.0, "dopo clear la tabella e' vuota");
		System.out.println("(9) memoria: 40 byte per voce (2^22 voci = 160 MiB, contro 96 MiB della TT a hash); API a solo hash rifiutate");
	}
}
