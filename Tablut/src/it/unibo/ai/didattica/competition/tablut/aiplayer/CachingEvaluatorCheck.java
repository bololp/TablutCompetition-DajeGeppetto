package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Verifica {@link CachingEvaluator}: (1) valore identico bit per bit a quello dell'evaluator originale (MLP e CNN); (2) identita'
 * ESATTA della chiave: nessuno dei 163 bit di posizione puo' essere ignorato, e con una cache minuscola (collisioni d'indice
 * forzate) non si serve mai il valore di un'altra posizione, nemmeno se lo Zobrist coincide; (3) ricerca con cache attiva e
 * disattiva: stessa mossa, stesso punteggio, stessi nodi e stessa profondita' (ordinamento base e rec); (4) protocollo seqlock
 * verificato in modo DETERMINISTICO (scrittore fermo a meta', chiavi "miste") piu' prove a molti thread su cache minuscola;
 * (5) nessuna allocazione nel percorso di lookup/scrittura; (6) ciclo di vita ({@code outputScale}, {@code clear}); (7) formule di
 * memoria. Esce con codice 1 se qualcosa fallisce. Uso: {@code CachingEvaluatorCheck [seme]}.
 */
public class CachingEvaluatorCheck {

	static int failures = 0;

	static void check(boolean ok, String what) {
		if (!ok) {
			failures++;
			System.out.println("FALLITO: " + what);
		}
	}

	static boolean same(double a, double b) {
		return Double.doubleToRawLongBits(a) == Double.doubleToRawLongBits(b);
	}

	/** Funzione pura e rapida di (scacchiera, turno): sensibile a OGNI cella e al turno. Conta le chiamate (solo thread singolo). */
	static final class Fake implements Evaluator {
		int calls;

		static double value(FastState s) {
			long h = 0xCBF29CE484222325L ^ (s.turn * 0x9E3779B97F4A7C15L);
			for (int i = 0; i < 81; i++) {
				h = (h ^ (s.board[i] + 1)) * 0x100000001B3L;
				h ^= h >>> 29;
			}
			if (((h >>> 52) & 0x7FF) == 0x7FF) {
				h &= ~(1L << 62); // niente NaN/Inf: il confronto e' sui bit grezzi
			}
			return Double.longBitsToDouble(h);
		}

		@Override
		public double evaluate(FastState state) {
			calls++;
			return value(state);
		}
	}

	/** Come Fake ma stateless (usabile da piu' thread). */
	static final class PureFake implements Evaluator {
		@Override
		public double evaluate(FastState state) {
			return Fake.value(state);
		}
	}

	static List<FastState> positions(long seed, int n) {
		Random rnd = new Random(seed);
		List<FastState> out = new ArrayList<>();
		while (out.size() < n) {
			TestPositions.Game g = TestPositions.playGame(rnd, true, 70);
			for (int i = 2; i < g.positions.size() && out.size() < n; i += 2) {
				FastState f = FastState.fromState(g.positions.get(i));
				if (!f.isGameOver()) {
					out.add(f);
				}
			}
			if (out.size() < n) {
				FastState f = FastState.fromState(TestPositions.randomPosition(rnd));
				if (!f.isGameOver()) {
					out.add(f);
				}
			}
		}
		return out;
	}

	public static void main(String[] args) throws Exception {
		final long seed = args.length > 0 ? Long.parseLong(args[0]) : 4242L;
		final List<FastState> pos = positions(seed, 400);

		final NeuralEvaluator mlp = NeuralEvaluator.randomInit(11);
		final CnnEvaluator cnn = CnnEvaluator.randomInit(CnnEvaluator.defaultConfig(TablutFeatures.G_DEFAULT), TablutFeatures.G_DEFAULT, 13);

		bitExact("MLP", mlp, pos);
		bitExact("CNN", cnn, pos);
		exactKey(pos);
		specialValues();
		forcedCollisions(pos);
		outOfDomain(pos.get(0));
		searchEquivalence("MLP", mlp, 4, pos);
		searchEquivalence("CNN", cnn, 3, pos);
		seqlockProtocol(pos);
		concurrency(pos, mlp, cnn);
		noAllocation(pos);
		lifecycle(mlp, pos);
		memoryFormulas(mlp);

		System.out.println(failures == 0 ? "=== CACHING EVALUATOR CHECK OK ===" : "=== FALLITI: " + failures + " ===");
		System.exit(failures == 0 ? 0 : 1);
	}

	// ---- (1) uguaglianza al bit con le reti vere -----------------------------------------------------------------------------
	static void bitExact(String name, Evaluator net, List<FastState> pos) {
		CachingEvaluator ce = new CachingEvaluator(net, 18, true);
		int bad = 0;
		for (int pass = 0; pass < 3; pass++) {
			for (FastState s : pos) {
				if (!same(ce.evaluate(s), net.evaluate(s))) {
					bad++;
				}
			}
		}
		check(bad == 0, name + ": " + bad + " valori diversi dall'evaluator diretto");
		check(ce.hits() >= 2L * pos.size() * 9 / 10, name + ": secondo e terzo passaggio dovrebbero essere quasi tutti hit, hit=" + ce.hits());
		check(ce.bypasses() == 0, name + ": nessuna posizione reale deve saltare la cache (celle/turno sempre nel dominio), salti=" + ce.bypasses());
		System.out.println("(1) " + name + ": valore identico al bit su " + pos.size() + " posizioni x 3 passaggi; lookup=" + ce.calls() + " hit=" + ce.hits() + " salti=" + ce.bypasses());
	}

	// ---- (2a) ogni bit della chiave conta ------------------------------------------------------------------------------------
	static void exactKey(List<FastState> pos) {
		Fake inner = new Fake();
		CachingEvaluator ce = new CachingEvaluator(inner, 20, true);
		FastState base = FastState.copyOf(pos.get(5));
		ce.evaluate(base);
		check(inner.calls == 1, "prima valutazione: una chiamata all'evaluator interno");
		ce.evaluate(base);
		check(inner.calls == 1 && ce.hits() == 1, "seconda valutazione della stessa posizione: hit");
		final byte[] codes = { BoardConstants.EMPTY, BoardConstants.WHITE, BoardConstants.BLACK, BoardConstants.KING };
		List<FastState> variants = new ArrayList<>();
		for (int sq = 0; sq < 81; sq++) {
			for (byte c : codes) {
				if (c == base.board[sq]) {
					continue;
				}
				FastState v = FastState.copyOf(base);
				v.board[sq] = c; // quasi sempre una posizione illegale, ma e' una chiave valida: l'evaluator e' puro sulla scacchiera
				variants.add(v);
			}
		}
		FastState t = FastState.copyOf(base);
		t.turn = (byte) (t.turn ^ 1);
		variants.add(t);
		int before = inner.calls;
		for (FastState v : variants) {
			double got = ce.evaluate(v);
			check(same(got, Fake.value(v)), "variante a una cella/turno: valore di un'altra posizione");
		}
		check(inner.calls - before == variants.size(), "ognuna delle " + variants.size() + " varianti (81 celle x 3 valori + turno) deve essere un miss, chiamate=" + (inner.calls - before));
		try {
			AtomicLongArray raw = rawSlots(ce);
			int occ = 0;
			for (int b = 0; b + 3 < raw.length(); b += 4) {
				if ((raw.get(b + 3) & (1L << 35)) != 0) {
					occ++;
				}
			}
			// 245 chiavi quasi uguali in 2^20 slot: con un buon indice le collisioni attese sono ~0,03; una funzione debole ne da' decine
			check(occ >= variants.size() + 1 - 1, "dispersione dell'indice: " + occ + " slot occupati per " + (variants.size() + 1) + " chiavi (attesi quasi tutti distinti)");
			System.out.println("(2a) dispersione: " + occ + " slot occupati per " + (variants.size() + 1) + " chiavi vicine in 2^20 slot");
		} catch (Exception e) {
			check(false, "reflection sullo storage: " + e);
		}
		// e lo stesso identico Zobrist non deve confonderle: le varianti sono copie con lo stesso zobristHash della base
		boolean sameHash = true;
		for (FastState v : variants) {
			sameHash &= v.zobristHash == base.zobristHash || v == t; // la copia con turno diverso mantiene comunque l'hash copiato
		}
		check(sameHash, "precondizione: le varianti condividono lo Zobrist della base (la chiave esatta non deve dipenderne)");
		int h0 = (int) ce.hits();
		for (FastState v : variants) {
			check(same(ce.evaluate(v), Fake.value(v)), "secondo passaggio sulle varianti");
		}
		check(ce.hits() - h0 >= variants.size() * 9 / 10, "secondo passaggio sulle varianti: quasi tutti hit");
		System.out.println("(2a) " + variants.size() + " varianti (ogni cella/turno) tutte miss al primo passaggio, valore sempre corretto, Zobrist identico ignorato");
	}

	// ---- valori speciali conservati al bit ---------------------------------------------------------------------------------------
	static void specialValues() {
		final double[] vals = { 0.0, -0.0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -Double.MAX_VALUE, 1e6 + 7 };
		FastState s = new FastState();
		for (final double v : vals) {
			CachingEvaluator ce = new CachingEvaluator(new Evaluator() {
				int n;

				@Override
				public double evaluate(FastState st) {
					check(++n == 1, "valore speciale " + v + ": l'evaluator interno e' stato richiamato (hit mancato)");
					return v;
				}
			}, 8, true);
			check(same(ce.evaluate(s), v) && same(ce.evaluate(s), v), "valore speciale " + v + " non conservato al bit");
			check(ce.hits() == 1, "valore speciale " + v + ": la seconda lettura dev'essere un hit (anche +0.0)");
		}
		System.out.println("(2b) +0.0, -0.0, +-Inf, MIN_VALUE, -MAX_VALUE, 1e6+7 conservati al bit e serviti come hit");
	}

	// ---- (2c) collisioni d'indice forzate con 16 slot ---------------------------------------------------------------------------
	static void forcedCollisions(List<FastState> pos) {
		Fake inner = new Fake();
		CachingEvaluator ce = new CachingEvaluator(inner, 4, true); // 16 voci per 400 posizioni: collisioni ovunque
		Random rnd = new Random(7);
		int bad = 0;
		for (int k = 0; k < 200_000; k++) {
			FastState s = pos.get(rnd.nextInt(pos.size()));
			if (!same(ce.evaluate(s), Fake.value(s))) {
				bad++;
			}
		}
		check(bad == 0, "cache da 16 voci: " + bad + " valori di un'altra posizione");
		check(ce.hits() > 0, "cache da 16 voci: ci si aspetta comunque qualche hit");
		System.out.println("(2c) cache da 16 voci, 200000 lookup su " + pos.size() + " posizioni: valori errati = " + bad + ", hit = " + ce.hits());
	}

	// ---- posizioni fuori dominio: si salta la cache, mai valori sbagliati -------------------------------------------------------
	static void outOfDomain(FastState src) {
		Fake inner = new Fake();
		CachingEvaluator ce = new CachingEvaluator(inner, 10, true);
		FastState a = FastState.copyOf(src);
		a.board[10] = BoardConstants.THRONE_MARK; // marcatore che non deve mai stare in board
		FastState b = FastState.copyOf(src);
		b.board[10] = 7;
		FastState c = FastState.copyOf(src);
		c.turn = 2;
		for (int rep = 0; rep < 2; rep++) {
			check(same(ce.evaluate(a), Fake.value(a)) && same(ce.evaluate(b), Fake.value(b)) && same(ce.evaluate(c), Fake.value(c)), "fuori dominio: valore diverso dal diretto");
		}
		check(ce.bypasses() == 6 && ce.hits() == 0 && inner.calls == 6, "fuori dominio: sempre saltata (6 salti, 0 hit), salti=" + ce.bypasses() + " hit=" + ce.hits());
		System.out.println("(2d) cella fuori dominio (3 e 7) e turno 2: cache saltata, valore sempre quello diretto");
	}

	// ---- (3) ricerca: cache attiva e disattiva ---------------------------------------------------------------------------------
	static void searchEquivalence(String name, Evaluator net, int depth, List<FastState> pos) {
		final int[] orders = { Negamax.ORDER_DEFAULT, OrderingSpec.parse("rec") };
		final String[] oname = { "base", "rec" };
		for (int o = 0; o < orders.length; o++) {
			Negamax plain = new Negamax(net, depth, true, 1 << 18, true, orders[o], 1);
			CachingEvaluator ce = new CachingEvaluator(net, 18, true);
			Negamax cached = new Negamax(ce, depth, true, 1 << 18, true, orders[o], 1);
			int n = 0;
			for (int i = 0; i < 400 && n < 8; i += 37, n++) {
				FastState root = pos.get(i);
				int m1 = plain.findBestMove(root, 1_000_000_000L, new long[0], 0);
				int m2 = cached.findBestMove(root, 1_000_000_000L, new long[0], 0);
				check(m1 == m2, name + "/" + oname[o] + ": mossa diversa alla posizione " + i);
				check(same(plain.lastScore, cached.lastScore), name + "/" + oname[o] + ": punteggio diverso alla posizione " + i + " (" + plain.lastScore + " vs " + cached.lastScore + ")");
				check(plain.lastNodeCount == cached.lastNodeCount, name + "/" + oname[o] + ": nodi diversi alla posizione " + i);
				check(plain.lastDepthReached == cached.lastDepthReached, name + "/" + oname[o] + ": profondita' diversa alla posizione " + i);
			}
			System.out.printf("(3) %s d=%d order=%s: %d ricerche (istanza riusata, TT e cache persistenti): stessa mossa/punteggio/nodi/profondita'; hit %.1f%% su %d lookup%n",
				name, depth, oname[o], n, 100 * ce.hitRate(), ce.calls());
			check(ce.hits() > 0, name + "/" + oname[o] + ": nella ricerca ci si aspetta almeno un hit");
		}
	}

	// ---- (4a) protocollo seqlock: casi deterministici via reflection sullo storage grezzo ---------------------------------------
	static AtomicLongArray rawSlots(CachingEvaluator ce) throws Exception {
		Field f = CachingEvaluator.class.getDeclaredField("slots");
		f.setAccessible(true);
		return (AtomicLongArray) f.get(ce);
	}

	/** Indice base dell'unica voce occupata (cache fresca con una sola posizione dentro). */
	static int onlyOccupied(AtomicLongArray raw) {
		int found = -1;
		for (int b = 0; b + 3 < raw.length(); b += 4) {
			if ((raw.get(b + 3) & (1L << 35)) != 0) {
				check(found < 0, "piu' di una voce occupata");
				found = b;
			}
		}
		check(found >= 0, "nessuna voce occupata dopo una valutazione");
		return found;
	}

	static void seqlockProtocol(List<FastState> pos) throws Exception {
		final long SEQ_ONE = 1L << 36;
		FastState a = pos.get(3);

		// scrittore fermo a meta' (versione dispari): lettura = miss, scrittura saltata, poi ripristino = hit
		Fake inner = new Fake();
		CachingEvaluator ce = new CachingEvaluator(inner, 4, true);
		AtomicLongArray raw = rawSlots(ce);
		ce.evaluate(a);
		int base = onlyOccupied(raw);
		long tailEven = raw.get(base + 3);
		check((tailEven & SEQ_ONE) == 0, "dopo una scrittura completa la versione e' pari");
		raw.set(base + 3, tailEven + SEQ_ONE); // scrittore "congelato" dopo il CAS
		int c0 = inner.calls;
		check(same(ce.evaluate(a), Fake.value(a)) && inner.calls == c0 + 1, "versione dispari: dev'essere un miss (valore ricalcolato)");
		check(raw.get(base + 3) == tailEven + SEQ_ONE, "versione dispari: la scrittura concorrente dev'essere saltata (slot intatto)");
		raw.set(base + 3, tailEven); // lo scrittore finisce senza cambiare nulla
		c0 = inner.calls;
		ce.evaluate(a);
		check(inner.calls == c0, "versione di nuovo pari: hit");

		// chiave "mista": coda e valore di A ma un'altra k0 / k1 / bit di coda -> mai servire A, mai servire la posizione che ha quell'altra chiave
		long k0 = raw.get(base), k1 = raw.get(base + 1);
		long[][] corruptions = { { base, k0 ^ 1L }, { base, k0 ^ (1L << 63) }, { base + 1, k1 ^ 1L }, { base + 1, k1 ^ (1L << 62) }, { base + 3, tailEven ^ 1L }, { base + 3, tailEven ^ (1L << 34) } };
		for (long[] cr : corruptions) {
			long orig = raw.get((int) cr[0]);
			raw.set((int) cr[0], cr[1]);
			int before = inner.calls;
			check(same(ce.evaluate(a), Fake.value(a)) && inner.calls == before + 1, "parola " + (cr[0] - base) + " alterata: la posizione A dev'essere un miss (chiave completa)");
			// la valutazione sopra ha riscritto lo slot con A: ripristino un nuovo stato coerente per il caso successivo
			raw.set((int) cr[0], orig);
			raw.set(base, k0);
			raw.set(base + 1, k1);
			raw.set(base + 3, tailEven);
			raw.set(base + 2, Double.doubleToRawLongBits(Fake.value(a)));
		}
		System.out.println("(4a) seqlock deterministico: scrittore fermo = miss e scrittura saltata; k0/k1/coda alterate = miss; ripristino = hit");
	}

	// ---- (4b) molti thread su cache minuscola ---------------------------------------------------------------------------------------
	static void hammer(final Evaluator direct, final CachingEvaluator ce, final List<FastState> pos, final int threads, final int iters, final String label) throws Exception {
		final double[] expected = new double[pos.size()];
		for (int i = 0; i < expected.length; i++) {
			expected[i] = direct.evaluate(pos.get(i));
		}
		final AtomicInteger bad = new AtomicInteger();
		Thread[] th = new Thread[threads];
		for (int t = 0; t < threads; t++) {
			final int id = t;
			th[t] = new Thread(new Runnable() {
				@Override
				public void run() {
					Random r = new Random(1000 + id);
					for (int k = 0; k < iters; k++) {
						int i = r.nextInt(pos.size());
						if (!same(ce.evaluate(pos.get(i)), expected[i])) {
							bad.incrementAndGet();
						}
						if ((k & 1023) == 0) {
							Thread.yield(); // piu' interleaving anche su una sola CPU
						}
					}
				}
			});
			th[t].start();
		}
		for (Thread t : th) {
			t.join();
		}
		check(bad.get() == 0, label + ": " + bad.get() + " valori diversi dal diretto");
		System.out.printf("(4b) %s: %d thread x %d lookup, valori errati = %d, hit %.1f%%%n", label, threads, iters, bad.get(), 100 * ce.hitRate());
	}

	static void concurrency(List<FastState> pos, NeuralEvaluator mlp, CnnEvaluator cnn) throws Exception {
		// posizioni quasi uguali fra loro (stesso Zobrist copiato, una cella diversa) per stressare i confronti di chiave
		List<FastState> near = new ArrayList<>(pos.subList(0, 200));
		Random rnd = new Random(5);
		for (int i = 0; i < 400; i++) {
			FastState v = FastState.copyOf(pos.get(rnd.nextInt(50)));
			int sq = rnd.nextInt(81);
			v.board[sq] = new byte[] { 0, 1, 2, 4 }[rnd.nextInt(4)];
			near.add(v);
		}
		final PureFake pf = new PureFake();
		hammer(pf, new CachingEvaluator(pf, 4, true), near, 8, 400_000, "Fake, cache da 16 voci");
		hammer(pf, new CachingEvaluator(pf, 8, true), near, 8, 400_000, "Fake, cache da 256 voci");
		hammer(pf, new CachingEvaluator(pf, 20, true), near, 8, 400_000, "Fake, cache da 1 M voci");
		hammer(mlp, new CachingEvaluator(mlp, 5, true), near.subList(0, 300), 4, 20_000, "MLP, cache da 32 voci");
		hammer(cnn, new CachingEvaluator(cnn, 5, true), near.subList(0, 300), 4, 6_000, "CNN, cache da 32 voci");

		// ricerca Lazy SMP a 4 thread con la cache: una mossa legale, senza eccezioni
		CachingEvaluator ce = new CachingEvaluator(mlp, 16, true);
		Negamax smp = new Negamax(ce, 40, true, 1 << 18, true, OrderingSpec.parse("rec"), 4).setThreads(4);
		int[] buf = new int[FastState.MAX_MOVES];
		for (int i = 0; i < 3; i++) {
			FastState root = pos.get(10 + i * 31);
			int n = root.generateMoves(buf);
			int mv = smp.findBestMove(root, 400, new long[0], 0);
			boolean legal = false;
			for (int k = 0; k < n; k++) {
				legal |= buf[k] == mv;
			}
			check(legal, "Lazy SMP a 4 thread con cache: mossa non legale alla posizione " + i);
		}
		System.out.printf("(4b) Lazy SMP 4 thread con cache: mosse legali, hit %.1f%% su %d lookup%n", 100 * ce.hitRate(), ce.calls());
	}

	// ---- (5) nessuna allocazione ------------------------------------------------------------------------------------------------------
	static void noAllocation(List<FastState> pos) {
		Object bean = ManagementFactory.getThreadMXBean();
		Method m;
		try {
			m = Class.forName("com.sun.management.ThreadMXBean").getMethod("getThreadAllocatedBytes", long.class);
			m.setAccessible(true);
			m.invoke(bean, Thread.currentThread().getId());
		} catch (Throwable t) {
			System.out.println("(5) SALTATO: la JVM non espone getThreadAllocatedBytes (" + t + ")");
			return;
		}
		try {
			final long id = Thread.currentThread().getId();
			final PureFake pf = new PureFake();
			final CachingEvaluator ce = new CachingEvaluator(pf, 12, false); // statistiche spente: LongAdder potrebbe allocare celle
			final int n = 100;
			double sink = 0;
			for (int w = 0; w < 4000; w++) { // scaldamento (JIT)
				sink += ce.evaluate(pos.get(w % n));
			}
			long a0 = (Long) m.invoke(bean, id);
			final int iters = 300_000;
			for (int k = 0; k < iters; k++) {
				sink += ce.evaluate(pos.get(k % n)); // quasi tutti hit
			}
			long a1 = (Long) m.invoke(bean, id);
			for (int k = 0; k < iters; k++) {
				sink += ce.evaluate(pos.get(100 + (k % 300))); // molti miss (12 bit = 4096 voci: i primi giri sono miss e scrivono)
			}
			long a2 = (Long) m.invoke(bean, id);
			double perHit = (a1 - a0) / (double) iters, perMix = (a2 - a1) / (double) iters;
			check(perHit < 0.5 && perMix < 0.5, "allocazioni nel percorso di lookup: hit " + perHit + " B/chiamata, misto " + perMix + " B/chiamata");
			System.out.printf("(5) allocazioni per chiamata: percorso hit %.4f B, percorso misto hit/miss+scrittura %.4f B (soglia 0,5) [%s]%n", perHit, perMix, sink == 12345.678 ? "" : "ok");
		} catch (Throwable t) {
			check(false, "misura allocazioni fallita: " + t);
		}
	}

	// ---- (6) ciclo di vita ------------------------------------------------------------------------------------------------------------
	static void lifecycle(NeuralEvaluator net, List<FastState> pos) {
		NeuralEvaluator local = NeuralEvaluator.randomInit(77);
		CachingEvaluator ce = new CachingEvaluator(local, 14, true);
		FastState s = pos.get(9);
		double v1 = ce.evaluate(s);
		check(same(ce.evaluate(s), v1) && ce.hits() == 1, "prima del cambio di scala: hit");
		local.outputScale = 250.0;
		double direct = local.evaluate(s);
		check(!same(direct, v1), "precondizione: cambiare outputScale cambia il valore diretto");
		check(same(ce.evaluate(s), direct), "outputScale cambiato: la cache NON deve servire il valore obsoleto");
		check(ce.bypasses() >= 1, "outputScale cambiato: la cache viene saltata");
		local.outputScale = 1000.0;
		check(same(ce.evaluate(s), v1), "outputScale ripristinato: il valore originale torna valido");
		ce.clear();
		long h0 = ce.hits();
		check(same(ce.evaluate(s), v1) && ce.hits() == h0, "dopo clear(): miss, stesso valore");
		check(same(ce.evaluate(s), v1) && ce.hits() == h0 + 1, "dopo clear() e una valutazione: di nuovo hit");
		System.out.println("(6) outputScale cambiato = cache saltata (mai valori obsoleti), ripristinato = valida; clear() svuota");
	}

	// ---- (7) memoria e fabbriche ---------------------------------------------------------------------------------------------------------
	static void memoryFormulas(NeuralEvaluator net) {
		check(CachingEvaluator.bytesFor(22) == 128L << 20, "2^22 voci = 128 MiB");
		check(CachingEvaluator.bytesFor(4) == 512, "2^4 voci = 512 byte");
		final long GiB = 1L << 30;
		check(CachingEvaluator.fitToHeap(22, 2 * GiB) == 22, "2^22 (128 MiB) sta in 2 GiB (limite 256 MiB)");
		check(CachingEvaluator.fitToHeap(22, GiB) == 22, "2^22 (128 MiB) sta in 1 GiB (limite 128 MiB, uguale)");
		check(CachingEvaluator.fitToHeap(22, GiB / 2) == 21, "in 512 MiB il limite e' 64 MiB: 2^21");
		check(CachingEvaluator.fitToHeap(22, 1L << 20) == 12, "heap minuscolo: pavimento 2^12");
		for (int l = 12; l <= 26; l++) {
			int f = CachingEvaluator.fitToHeap(l, 2 * GiB);
			check(f <= l && (f == 12 || CachingEvaluator.bytesFor(f) <= 2 * GiB / 8) && (f == l || CachingEvaluator.bytesFor(f + 1) > 2 * GiB / 8), "fit 2^" + l + " -> 2^" + f);
		}
		SimpleEvaluator simple = new SimpleEvaluator();
		check(!CachingEvaluator.isCacheable(simple) && CachingEvaluator.wrapIfCacheable(simple, 22) == simple, "SimpleEvaluator non viene avvolto");
		check(CachingEvaluator.isCacheable(net) && CachingEvaluator.wrapIfCacheable(net, 0) == net, "log2=0 disattiva la cache");
		check(CachingEvaluator.wrapIfCacheable(net, 12) instanceof CachingEvaluator, "una rete viene avvolta");
		boolean t1 = false, t2 = false, t3 = false;
		try {
			new CachingEvaluator(null, 10);
		} catch (IllegalArgumentException e) {
			t1 = true;
		}
		try {
			new CachingEvaluator(net, 3);
		} catch (IllegalArgumentException e) {
			t2 = true;
		}
		try {
			new CachingEvaluator(net, 27);
		} catch (IllegalArgumentException e) {
			t3 = true;
		}
		check(t1 && t2 && t3, "argomenti non validi rifiutati");
		System.out.println("(7) memoria: 32 B/voce (2^22 voci = 128 MiB), tetto = 1/8 dell'heap, SimpleEvaluator escluso");
	}
}
