package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * VERSIONE ORIGINALE (solo hash) della cache delle valutazioni, consegnata prima di quella a chiave esatta ({@link CachingEvaluator}).
 * Rinominata in HashCachingEvaluator solo per poter convivere con la versione finale nella stessa build e confrontarle.
 * <p>
 * Identita' della posizione = hash Zobrist a 64 bit ({@code state.zobristHash}). Voce = 2 long {@code [hash ^ bits, bits]} (trucco XOR:
 * una voce strappata da scritture concorrenti non supera la verifica, con probabilita' altissima ma non certa). 16 byte per voce,
 * 64 MiB con 2^22 voci. Un valore con bit tutti a zero (+0.0) non viene mai servito dalla cache (si ricalcola).
 * <p>
 * Limiti rispetto alla versione a chiave esatta: due posizioni diverse con lo stesso Zobrist si scambiano il valore (probabilita'
 * dell'ordine di 2^-64 per lookup, ipotesi di chiavi indipendenti); si fida che {@code zobristHash} sia coerente con la scacchiera;
 * nessuna guardia su {@code outputScale}; non e' collegata al client. Compatibile Java 8.
 */
public final class HashCachingEvaluator implements Evaluator {

	private final Evaluator inner;
	private final AtomicLongArray slots;
	private final int mask;
	private final LongAdder calls, hits; // null se le statistiche sono disattivate

	public HashCachingEvaluator(Evaluator inner, int log2Entries) {
		this(inner, log2Entries, false);
	}

	public HashCachingEvaluator(Evaluator inner, int log2Entries, boolean stats) {
		if (inner == null) {
			throw new IllegalArgumentException("evaluator nullo");
		}
		if (log2Entries < 4 || log2Entries > 26) {
			throw new IllegalArgumentException("log2Entries fuori da [4,26]: " + log2Entries);
		}
		this.inner = inner;
		this.slots = new AtomicLongArray(2 << log2Entries);
		this.mask = (1 << log2Entries) - 1;
		this.calls = stats ? new LongAdder() : null;
		this.hits = stats ? new LongAdder() : null;
	}

	private int index(long hash) {
		return 2 * (((int) (hash ^ (hash >>> 29))) & mask);
	}

	@Override
	public double evaluate(FastState state) {
		final long h = state.zobristHash;
		final int i = index(h);
		final long data = slots.get(i + 1);
		final long check = slots.get(i);
		if (calls != null) {
			calls.increment();
		}
		if (data != 0L && (check ^ data) == h) {
			if (hits != null) {
				hits.increment();
			}
			return Double.longBitsToDouble(data);
		}
		final double v = inner.evaluate(state);
		final long bits = Double.doubleToRawLongBits(v);
		slots.lazySet(i + 1, bits);
		slots.lazySet(i, h ^ bits);
		return v;
	}

	/** Svuota la cache (NON thread-safe rispetto a una ricerca in corso). */
	public void clear() {
		for (int i = 0; i < slots.length(); i++) {
			slots.lazySet(i, 0L);
		}
	}

	public Evaluator inner() {
		return inner;
	}

	/** Byte occupati: 16 per voce. */
	public static long bytesFor(int log2Entries) {
		return 16L << log2Entries;
	}

	public long calls() {
		return calls == null ? -1 : calls.sum();
	}

	public long hits() {
		return hits == null ? -1 : hits.sum();
	}

	public double hitRate() {
		return calls == null || calls.sum() == 0 ? Double.NaN : (double) hits.sum() / calls.sum();
	}
}
