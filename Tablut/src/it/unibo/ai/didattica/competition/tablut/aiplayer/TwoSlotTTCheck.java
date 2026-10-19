package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import it.unibo.ai.didattica.competition.tablut.aiplayer.TranspositionTable.ProbeResult;

/**
 * Verifica la transposition table a DUE SLOT per bucket ({@link TranspositionTable#TranspositionTable(int, boolean)}).
 * <ol>
 * <li><b>Modello di riferimento</b> indipendente (due slot per bucket scritti a mano con la regola dichiarata nella specifica,
 * senza array di long ne' XOR): centinaia di migliaia di operazioni casuali su poche chiavi in pochi bucket (molte collisioni),
 * con avanzamenti di generazione ({@code newSearch}); ogni probe deve coincidere (hit, profondita' sufficiente, punteggio al bit,
 * flag, mossa);</li>
 * <li><b>scopo</b>: una voce profonda sopravvive a un diluvio di posizioni superficiali che cadono nello stesso bucket (a slot
 * singolo no: lo si verifica anche), e dopo {@code newSearch} (aging) puo' essere sostituita;</li>
 * <li><b>voci strappate</b>: campi di scritture diverse combinati a mano in OGNI slot danno sempre miss;</li>
 * <li><b>concorrenza</b>: 4 thread su 64 slot con payload derivato dalla chiave: ogni hit e' coerente (mai un dato mescolato);</li>
 * <li>{@code occupancy}: 0 sulla tabella vuota, circa il valore atteso dopo N inserimenti distinti, mai oltre 1.</li>
 * </ol>
 * Uso: {@code TwoSlotTTCheck [seme]}.
 */
public class TwoSlotTTCheck {

	static int failures = 0;

	static void fail(String m) {
		failures++;
		if (failures <= 30) {
			System.out.println("FAIL - " + m);
		}
	}

	/** Voce del modello: stessa informazione di una voce vera. */
	static final class E {
		long key;
		int depth, flag, move, gen;
		double score;
	}

	/** Modello: bucket = due voci (null = vuota); regola di sostituzione scritta qui in modo indipendente dal codice. */
	static final class Model {
		final E[][] b;
		int gen = 0;

		Model(int buckets) {
			b = new E[buckets][2];
		}

		void newSearch() {
			gen = (gen + 1) & 63;
		}

		E find(long key, int bucket) {
			if (b[bucket][0] != null && b[bucket][0].key == key) {
				return b[bucket][0];
			}
			if (b[bucket][1] != null && b[bucket][1].key == key) {
				return b[bucket][1];
			}
			return null;
		}

		void store(long key, int bucket, int depth, double score, int flag, int move) {
			E p = b[bucket][0], a = b[bucket][1];
			E n = new E();
			n.key = key;
			n.depth = depth;
			n.score = score;
			n.flag = flag;
			n.move = move;
			n.gen = gen;
			if (p != null && p.key == key) {
				if (depth >= p.depth) {
					b[bucket][0] = n;
				}
				return;
			}
			if (a != null && a.key == key) {
				if (depth >= a.depth) {
					b[bucket][1] = n;
				}
				return;
			}
			if (p == null || depth >= p.depth || p.gen != gen) {
				b[bucket][0] = n;
			} else {
				b[bucket][1] = n;
			}
		}
	}

	public static void main(String[] args) throws Exception {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 31337L;
		checkModel(seed, 8, 40);
		checkModel(seed + 1, 64, 12);
		checkModel(seed + 2, 1, 6); // un solo bucket: tutto collide
		checkDeepSurvives();
		checkTorn();
		checkOccupancy(seed);
		checkConcurrent();
		if (failures == 0) {
			System.out.println("=== TWO-SLOT TT CHECK OK ===");
		} else {
			System.out.println("=== TWO-SLOT TT CHECK FALLITO: " + failures + " discrepanze ===");
			System.exit(1);
		}
	}

	/** Chiave a 64 bit che cade nel bucket {@code bucket} (i bit bassi scelgono il bucket) con parte alta casuale tra {@code variants}. */
	static long keyFor(int bucket, int variant) {
		return ((long) (variant + 1) * 0x9E3779B97F4A7C15L << 16) | bucket;
	}

	static void checkModel(long seed, int buckets, int variants) {
		final int before = failures;
		Random rnd = new Random(seed);
		TranspositionTable tt = new TranspositionTable(buckets * 2, true);
		if (tt.capacity() != buckets * 2) {
			fail("capacity " + tt.capacity() + " attesa " + buckets * 2);
		}
		Model m = new Model(buckets);
		ProbeResult r = new ProbeResult();
		long ops = 0, hits = 0, deepHits = 0;
		for (int i = 0; i < 300_000; i++) {
			if (rnd.nextInt(400) == 0) {
				tt.newSearch();
				m.newSearch();
			}
			final int bucket = rnd.nextInt(buckets);
			final long key = keyFor(bucket, rnd.nextInt(variants));
			final int depth = rnd.nextInt(11);
			if (rnd.nextBoolean()) {
				final double score = rnd.nextInt(7) == 0 ? -0.0 : rnd.nextGaussian() * 1000;
				final int flag = rnd.nextInt(3);
				final int move = rnd.nextInt(5) == 0 ? -1 : rnd.nextInt(6561);
				tt.store(key, depth, score, (byte) flag, move);
				m.store(key, bucket, depth, score, flag, move);
			} else {
				tt.probeInto(key, depth, r);
				E e = m.find(key, bucket);
				ops++;
				if (r.hit != (e != null)) {
					fail("buckets=" + buckets + " op " + i + ": hit=" + r.hit + " atteso " + (e != null));
					return;
				}
				if (e != null) {
					hits++;
					if (e.depth >= 6) {
						deepHits++;
					}
					if (r.depthSufficient != (e.depth >= depth) || Double.doubleToRawLongBits(r.score) != Double.doubleToRawLongBits(e.score)
							|| r.flag != e.flag || r.packedMove != e.move) {
						fail("buckets=" + buckets + " op " + i + ": voce diversa dal modello (profondita' " + r.depthSufficient + "/" + (e.depth >= depth)
								+ ", punteggio " + r.score + "/" + e.score + ", flag " + r.flag + "/" + e.flag + ", mossa " + r.packedMove + "/" + e.move + ")");
						return;
					}
				}
			}
		}
		if (hits < 5000 || deepHits < 500) {
			fail("test vacuo: hit " + hits + ", hit profondi " + deepHits);
		}
		System.out.printf("%s %-24s %d bucket: %d probe identici al modello (%d hit, di cui %d con profondita' >= 6)%n",
				failures == before ? "OK  " : "FAIL", "[modello]", buckets, ops, hits, deepHits);
	}

	static void checkDeepSurvives() {
		final int before = failures;
		for (int mode = 0; mode < 2; mode++) {
			final boolean two = mode == 1;
			TranspositionTable tt = new TranspositionTable(64, two);
			final int bucket = 5;
			final long deep = keyFor(bucket, 0);
			tt.newSearch();
			tt.store(deep, 8, 123.5, TranspositionTable.EXACT, 77);
			for (int v = 1; v <= 2000; v++) { // diluvio di posizioni superficiali che collidono con la profonda
				tt.store(keyFor(bucket, v), v % 3, v, TranspositionTable.LOWER_BOUND, v);
			}
			ProbeResult r = tt.probe(deep, 8);
			final boolean alive = r.hit && r.depthSufficient && r.score == 123.5 && r.packedMove == 77;
			if (two && !alive) {
				fail("a due slot la voce profonda e' stata cancellata da posizioni superficiali");
			}
			if (!two && alive) {
				fail("(il test e' vacuo) a slot singolo la voce profonda e' sopravvissuta al diluvio: la differenza tra le politiche non e' esercitata");
			}
			if (two) {
				// aging: dopo newSearch la voce vecchia puo' essere sostituita da una nuova posizione con profondita' minore
				tt.newSearch();
				tt.store(keyFor(bucket, 5000), 1, 9.0, TranspositionTable.EXACT, 1);
				ProbeResult g = tt.probe(deep, 0);
				ProbeResult f = tt.probe(keyFor(bucket, 5000), 1);
				if (g.hit) {
					fail("aging: la voce della ricerca precedente non e' stata sostituita dalla nuova posizione");
				}
				if (!f.hit || f.score != 9.0) {
					fail("aging: la nuova voce non e' stata scritta");
				}
			}
		}
		System.out.printf("%s %-24s la voce profonda sopravvive a 2000 collisioni superficiali (a slot singolo no); aging dopo newSearch%n",
				failures == before ? "OK  " : "FAIL", "[protezione]");
	}

	static void checkTorn() {
		final int before = failures;
		final long h = keyFor(3, 7);
		for (int slot = 0; slot < 2; slot++) {
			TranspositionTable tt = new TranspositionTable(16, true);
			final long m1 = TranspositionTable.packMeta(5, TranspositionTable.EXACT, 100, 0);
			final long s1 = Double.doubleToRawLongBits(10.0);
			final long m2 = TranspositionTable.packMeta(6, TranspositionTable.LOWER_BOUND, 200, 0);
			final long s2 = Double.doubleToRawLongBits(-30.0);
			final long c1 = h ^ m1 ^ s1, c2 = (h + 99) ^ m2 ^ s2; // c2: scrittura di un'ALTRA posizione
			final long[][] combos = { { c1, m2, s1 }, { c1, m1, s2 }, { c2, m1, s1 }, { c2, m2, s1 }, { c2, m1, s2 }, { c1, m2, s2 } };
			for (long[] c : combos) {
				tt.writeRawSlot(h, slot, c[0], c[1], c[2]);
				if (tt.probe(h, 0).hit) {
					fail("slot " + slot + ": una voce strappata e' stata accettata");
				}
			}
			tt.writeRawSlot(h, slot, c1, m1, s1); // voce integra: hit
			ProbeResult ok = tt.probe(h, 5);
			if (!ok.hit || ok.score != 10.0 || ok.packedMove != 100) {
				fail("slot " + slot + ": la voce integra non e' letta");
			}
		}
		System.out.printf("%s %-24s 6 combinazioni di campi di scritture diverse x 2 slot: sempre miss; la voce integra e' letta%n",
				failures == before ? "OK  " : "FAIL", "[voci strappate]");
	}

	static void checkOccupancy(long seed) {
		final int before = failures;
		for (boolean two : new boolean[] { false, true }) {
			TranspositionTable tt = new TranspositionTable(1 << 12, two);
			if (tt.occupancy(1 << 12) != 0.0) {
				fail("occupancy della tabella vuota != 0");
			}
			Random rnd = new Random(seed);
			final int n = 1 << 11; // meta' della capacita'
			for (int i = 0; i < n; i++) {
				tt.store(rnd.nextLong(), 3, 0, TranspositionTable.EXACT, 1);
			}
			double occ = tt.occupancy(1 << 12);
			// n inserimenti distinti in C slot: occupazione attesa circa 1-e^{-n/C} (a slot singolo e a bucket con sostituzione, un poco di piu' con due slot)
			if (occ < 0.30 || occ > 0.50) {
				fail((two ? "due slot" : "slot singolo") + ": occupancy " + occ + " fuori da [0.30, 0.50] dopo n = capacita'/2 inserimenti");
			}
			for (int i = 0; i < 20 * (1 << 12); i++) {
				tt.store(rnd.nextLong(), rnd.nextInt(8), 0, TranspositionTable.EXACT, 1);
			}
			double full = tt.occupancy(1 << 12);
			if (full < 0.98 || full > 1.0) {
				fail((two ? "due slot" : "slot singolo") + ": occupancy " + full + " dopo 20 x capacita' inserimenti");
			}
		}
		System.out.printf("%s %-24s vuota=0; ~40%% dopo capacita'/2 inserimenti; ~100%% dopo 20 x capacita' (entrambe le politiche)%n",
				failures == before ? "OK  " : "FAIL", "[occupancy]");
	}

	static void checkConcurrent() throws Exception {
		final int before = failures;
		final TranspositionTable tt = new TranspositionTable(64, true);
		final AtomicInteger bad = new AtomicInteger(), hitsTotal = new AtomicInteger();
		final int OPS = 400_000;
		Thread[] ts = new Thread[4];
		for (int t = 0; t < ts.length; t++) {
			final long seed = 900 + t;
			ts[t] = new Thread(() -> {
				Random rnd = new Random(seed);
				ProbeResult r = new ProbeResult();
				for (int i = 0; i < OPS; i++) {
					final long key = keyFor(rnd.nextInt(32), rnd.nextInt(8));
					if (rnd.nextInt(5) == 0) {
						tt.newSearch();
					}
					if (rnd.nextBoolean()) {
						// payload derivato SOLO dalla chiave: punteggio e mossa; profondita' e flag liberi
						tt.store(key, rnd.nextInt(10), (double) (key >>> 5), (byte) rnd.nextInt(3), (int) (key & 0x1FFF));
					} else {
						tt.probeInto(key, rnd.nextInt(10), r);
						if (r.hit) {
							hitsTotal.incrementAndGet();
							if (r.score != (double) (key >>> 5) || r.packedMove != (int) (key & 0x1FFF)) {
								bad.incrementAndGet();
							}
						}
					}
				}
			});
			ts[t].start();
		}
		for (Thread t : ts) {
			t.join();
		}
		if (bad.get() != 0) {
			fail("stress a 4 thread: " + bad.get() + " hit incoerenti con la chiave su " + hitsTotal.get());
		}
		if (hitsTotal.get() < 10000) {
			fail("stress a 4 thread vacuo: solo " + hitsTotal.get() + " hit");
		}
		System.out.printf("%s %-24s 4 thread x %d operazioni su 64 slot: %d hit, tutti coerenti con la chiave%n",
				failures == before ? "OK  " : "FAIL", "[concorrenza]", OPS, hitsTotal.get());
	}
}
