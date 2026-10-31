package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Decoratore di {@link Evaluator} che memorizza le valutazioni statiche per POSIZIONE ESATTA (cache lossless di {@code double}).
 * <p>
 * <b>Identita' esatta, non hash.</b> La voce e' identificata dall'intera posizione da cui dipende la valutazione di
 * {@link NeuralEvaluator} e {@link CnnEvaluator}: i 81 valori della scacchiera ({@code FastState.board}, 2 bit per cella: i valori
 * ammessi sono EMPTY, WHITE, BLACK, KING) piu' il turno (1 bit) = 163 bit, confrontati per intero a ogni lookup. L'hash Zobrist NON
 * e' usato: l'indice di tabella deriva da un mescolamento della chiave esatta, e anche una collisione d'indice (due posizioni nello
 * stesso slot) produce solo un miss, mai il valore sbagliato. Se la posizione contiene un valore fuori dominio (cella diversa da
 * 0/1/2/4, turno diverso da 0/1, scacchiera non di 81 celle) la cache viene saltata e si valuta direttamente.
 * <p>
 * <b>Valore identico.</b> Si memorizzano i 64 bit di {@code Double.doubleToRawLongBits}: un hit restituisce esattamente il
 * {@code double} che l'evaluator interno ha restituito la prima volta (nessun arrotondamento, a differenza di {@link EvalCache}).
 * La valutazione e' una funzione pura della posizione (verificato: le due reti leggono solo {@code board} e {@code turn}), quindi
 * il riuso e' valido per qualunque nodo, iterazione, mossa di partita e thread.
 * <p>
 * <b>Concorrenza (seqlock per slot).</b> Voce = 4 {@code long}: {@code [k0, k1, valore, coda]}. {@code k0,k1} sono le celle 0..63,
 * la {@code coda} contiene celle 64..80 + turno (35 bit), un bit "occupato" (bit 35) e un contatore di versione (bit 36..63) il cui
 * bit meno significativo e' 1 mentre qualcuno scrive. Lettura: si legge la coda; se e' occupata, a versione pari e con la chiave di
 * coda uguale si leggono {@code k0,k1,valore} e si rilegge la coda: se e' cambiata, miss. Solo uno snapshot coerente viene
 * confrontato con la chiave completa. Scrittura: CAS della coda da versione pari a dispari (se fallisce, o e' gia' dispari, la
 * scrittura viene saltata: un altro thread sta scrivendo lo stesso slot), poi dati, poi coda con versione pari+2. Tutto il resto
 * (sovrascritture, scritture saltate, thread concorrenti, cache minuscola) puo' ridurre gli hit ma non restituire mai un valore
 * che non appartenga alla posizione richiesta. Unica ipotesi: un lettore non resta sospeso a meta' lookup mentre altre 2^27
 * scritture colpiscono lo STESSO slot (il contatore e' a 28 bit).
 * <p>
 * <b>Nessuna allocazione</b> nel percorso di lookup e di scrittura (chiave in variabili locali). <b>Memoria:</b>
 * {@link #bytesFor(int)} = 32 byte per voce, cioe' 128 MiB con 2^22 voci.
 * <p>
 * <b>Ciclo di vita.</b> La cache vale per UNA rete con pesi e {@code outputScale} fissi. Il client la crea subito dopo il
 * caricamento dei pesi e passa a Negamax solo il decoratore: nessun altro riferimento muta la rete. Difesa: se
 * {@code outputScale} dell'evaluator interno cambia dopo la costruzione, la cache viene saltata finche' non torna al valore
 * originale (mai valori obsoleti). Se i PESI vengono modificati (training), occorre {@link #clear()} a ricerca ferma, oppure non
 * usare la cache: non e' pensata per i trainer.
 * <p>
 * Solo con valutatori da qualche microsecondo (MLP/CNN) conviene: con {@link SimpleEvaluator} il guadagno misurato e' marginale,
 * per questo {@link #isCacheable(Evaluator)} e' vero solo per le due reti. Compatibile Java 8.
 */
public final class CachingEvaluator implements Evaluator {

	/** Bit 35 della coda: voce occupata (una voce vuota e' tutta a zero e non coincide con nessuna chiave). */
	private static final long OCCUPIED = 1L << 35;
	/** Bit 0..34 della coda: celle 64..80 (2 bit ciascuna) e turno (bit 34). */
	private static final long TAIL_KEY_MASK = (1L << 35) - 1;
	private static final int SEQ_SHIFT = 36;
	/** Il bit meno significativo del contatore di versione: 1 = scrittura in corso. */
	private static final long SEQ_ONE = 1L << SEQ_SHIFT;
	private static final long SEQ_MASK = -1L << SEQ_SHIFT;

	/** Codice a 2 bit per i soli valori ammessi in {@code FastState.board}; -1 = fuori dominio (si salta la cache). */
	private static final byte[] CODE = new byte[256];
	static {
		java.util.Arrays.fill(CODE, (byte) -1);
		CODE[BoardConstants.EMPTY] = 0;
		CODE[BoardConstants.WHITE] = 1;
		CODE[BoardConstants.BLACK] = 2;
		CODE[BoardConstants.KING] = 3;
	}

	private final Evaluator inner;
	private final AtomicLongArray slots;
	private final int mask;
	private final LongAdder calls, hits, bypasses; // null se le statistiche sono disattivate
	private final NeuralEvaluator nn;
	private final CnnEvaluator cnn;
	private final double scale0;
	private final boolean stateKey;

	/** @param log2Entries la cache ha {@code 2^log2Entries} voci di 32 byte, con {@code log2Entries} in [4, 26] */
	public CachingEvaluator(Evaluator inner, int log2Entries) {
		this(inner, log2Entries, false);
	}

	/** @param stats vero per contare chiamate, hit e salti (LongAdder: contesa trascurabile ma non nulla; spento di default) */
	public CachingEvaluator(Evaluator inner, int log2Entries, boolean stats) {
		this(inner, log2Entries, stats, false);
	}

	/**
	 * @param stateKey vero per usare la chiave esatta gia' mantenuta dallo stato ({@link FastState#key0}..{@link FastState#key2}) invece di
	 *                 ricodificare 81 celle a ogni lookup: stesso formato di chiave, stessa identita', meno costo per chiamata. Richiede che la
	 *                 chiave dello stato sia aggiornata (vale per ogni stato prodotto da makeMove/unmakeMove/caricamento; chi scrive la scacchiera a
	 *                 mano deve chiamare {@link FastState#computeKeys()}). Falso = si ricodifica dalla scacchiera, indipendente dalla chiave dello stato.
	 */
	public CachingEvaluator(Evaluator inner, int log2Entries, boolean stats, boolean stateKey) {
		if (inner == null) {
			throw new IllegalArgumentException("evaluator nullo");
		}
		if (log2Entries < 4 || log2Entries > 26) {
			throw new IllegalArgumentException("log2Entries fuori da [4,26]: " + log2Entries);
		}
		this.inner = inner;
		this.slots = new AtomicLongArray(4 << log2Entries);
		this.mask = (1 << log2Entries) - 1;
		this.calls = stats ? new LongAdder() : null;
		this.hits = stats ? new LongAdder() : null;
		this.bypasses = stats ? new LongAdder() : null;
		this.nn = inner instanceof NeuralEvaluator ? (NeuralEvaluator) inner : null;
		this.cnn = inner instanceof CnnEvaluator ? (CnnEvaluator) inner : null;
		this.scale0 = currentScale();
		this.stateKey = stateKey;
	}

	/** Vero se la cache usa la chiave esatta mantenuta dallo stato ({@code FastState.key0..key2}), falso se la ricodifica dalla scacchiera. */
	public boolean isStateKey() {
		return stateKey;
	}

	/** Vero per gli evaluator di cui e' stato verificato che dipendono solo da {@code board} e {@code turn}: MLP e CNN. */
	public static boolean isCacheable(Evaluator e) {
		return e instanceof NeuralEvaluator || e instanceof CnnEvaluator;
	}

	/** {@code e} avvolto in una cache di 2^log2Entries voci se e' MLP/CNN e {@code log2Entries > 0}, altrimenti {@code e} stesso. */
	public static Evaluator wrapIfCacheable(Evaluator e, int log2Entries) {
		return log2Entries > 0 && isCacheable(e) ? new CachingEvaluator(e, log2Entries) : e;
	}

	/** Come {@link #wrapIfCacheable(Evaluator, int)} scegliendo se la chiave viene dallo stato ({@code stateKey}) o ricodificata dalla scacchiera. */
	public static Evaluator wrapIfCacheable(Evaluator e, int log2Entries, boolean stateKey) {
		return log2Entries > 0 && isCacheable(e) ? new CachingEvaluator(e, log2Entries, false, stateKey) : e;
	}

	/** Byte occupati da una cache di 2^log2Entries voci (32 per voce). */
	public static long bytesFor(int log2Entries) {
		return 32L << log2Entries;
	}

	/** Riduce {@code log2Entries} (minimo 12) finche' la cache occupa al piu' un ottavo dell'heap {@code maxMemoryBytes}. */
	public static int fitToHeap(int log2Entries, long maxMemoryBytes) {
		int l = log2Entries;
		while (l > 12 && bytesFor(l) > maxMemoryBytes / 8) {
			l--;
		}
		return l;
	}

	private double currentScale() {
		return nn != null ? nn.outputScale : cnn != null ? cnn.outputScale : 0.0;
	}

	@Override
	public double evaluate(FastState state) {
		if ((nn != null || cnn != null) && currentScale() != scale0) {
			return bypass(state);
		}
		long k0 = 0, k1 = 0, tail = 0;
		if (stateKey) {
			// chiave esatta gia' mantenuta dallo stato: nessuna codifica, nessun controllo di dominio (la chiave e' coerente per costruzione)
			k0 = state.key0;
			k1 = state.key1;
			tail = state.key2 & TAIL_KEY_MASK;
		} else {
			final byte[] b = state.board;
			final int turn = state.turn;
			if (b.length != 81 || (turn & ~1) != 0) {
				return bypass(state);
			}
			final byte[] code = CODE;
			int bad = 0;
			for (int i = 0; i < 32; i++) {
				final int c = code[b[i] & 0xFF];
				bad |= c;
				k0 |= (long) c << (2 * i);
			}
			for (int i = 0; i < 32; i++) {
				final int c = code[b[32 + i] & 0xFF];
				bad |= c;
				k1 |= (long) c << (2 * i);
			}
			for (int i = 0; i < 17; i++) {
				final int c = code[b[64 + i] & 0xFF];
				bad |= c;
				tail |= (long) c << (2 * i);
			}
			if (bad < 0) { // almeno una cella fuori dominio (codice -1 = tutti i bit a 1)
				return bypass(state);
			}
			tail |= (long) turn << 34;
		}

		// indice dalla chiave ESATTA (non dallo Zobrist: un hash incoerente darebbe solo piu' miss)
		final int base = ((int) mix(k0 ^ mix(k1 ^ mix(tail))) & mask) << 2;

		if (calls != null) {
			calls.increment();
		}
		final long t1 = slots.get(base + 3);
		if ((t1 & OCCUPIED) != 0 && (t1 & SEQ_ONE) == 0 && (t1 & TAIL_KEY_MASK) == tail) {
			final long a = slots.get(base);
			final long c = slots.get(base + 1);
			final long v = slots.get(base + 2);
			if (slots.get(base + 3) == t1 && a == k0 && c == k1) { // snapshot coerente E chiave completa uguale
				if (hits != null) {
					hits.increment();
				}
				return Double.longBitsToDouble(v);
			}
		}
		final double value = inner.evaluate(state);
		store(base, k0, k1, tail, Double.doubleToRawLongBits(value));
		return value;
	}

	/** Finalizzatore a 64 bit di MurmurHash3 (fmix64): ogni bit di ingresso influenza tutti i bit d'uscita, anche per chiavi quasi uguali. */
	private static long mix(long z) {
		z = (z ^ (z >>> 33)) * 0xFF51AFD7ED558CCDL;
		z = (z ^ (z >>> 33)) * 0xC4CEB9FE1A85EC53L;
		return z ^ (z >>> 33);
	}

	private double bypass(FastState state) {
		if (bypasses != null) {
			bypasses.increment();
		}
		return inner.evaluate(state);
	}

	private void store(int base, long k0, long k1, long tail, long valueBits) {
		final long t = slots.get(base + 3);
		if ((t & SEQ_ONE) != 0) {
			return; // un altro thread sta scrivendo questo slot: si rinuncia (solo un hit in meno)
		}
		final long writing = t + SEQ_ONE; // versione dispari
		if (!slots.compareAndSet(base + 3, t, writing)) {
			return;
		}
		slots.lazySet(base, k0);
		slots.lazySet(base + 1, k1);
		slots.lazySet(base + 2, valueBits);
		slots.lazySet(base + 3, ((writing + SEQ_ONE) & SEQ_MASK) | OCCUPIED | tail);
	}

	/** Svuota la cache. NON e' thread-safe rispetto a una ricerca in corso: da chiamare a ricerca ferma (per esempio se cambiano i pesi). */
	public void clear() {
		for (int i = 0; i < slots.length(); i++) {
			slots.set(i, 0L);
		}
	}

	public Evaluator inner() {
		return inner;
	}

	/** Bit della dimensione: la cache ha 2^log2Entries voci. */
	public int log2Entries() {
		return Integer.numberOfTrailingZeros(mask + 1);
	}

	/** Chiamate che hanno consultato la cache (solo con statistiche attive), altrimenti -1. */
	public long calls() {
		return calls == null ? -1 : calls.sum();
	}

	public long hits() {
		return hits == null ? -1 : hits.sum();
	}

	/** Chiamate che hanno saltato la cache (posizione fuori dominio o outputScale cambiato), -1 senza statistiche. */
	public long bypasses() {
		return bypasses == null ? -1 : bypasses.sum();
	}

	/** Frazione di lookup serviti dalla cache (solo con statistiche attive), altrimenti NaN. */
	public double hitRate() {
		return calls == null || calls.sum() == 0 ? Double.NaN : (double) hits.sum() / calls.sum();
	}
}
