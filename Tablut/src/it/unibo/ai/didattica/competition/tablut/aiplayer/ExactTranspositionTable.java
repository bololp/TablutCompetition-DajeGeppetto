package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.concurrent.atomic.AtomicLongArray;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Transposition table che identifica la posizione con la CHIAVE ESATTA di {@link FastState} ({@code key0, key1, key2}: la
 * scacchiera e il turno, 163 bit), non con l'hash Zobrist. Stessa interfaccia, stesso indice (basso bit dello Zobrist: serve solo a
 * scegliere lo slot) e STESSA politica di sostituzione di {@link TranspositionTable}, a slot singolo o a due slot: con le stesse
 * operazioni restituisce gli stessi risultati, tranne che una collisione di hash non puo' mai restituire la voce di un'altra posizione.
 * <p>
 * Voce = 5 {@code long}: {@code [k0, k1, coda, punteggio, ctl]}. {@code coda} = {@code key2} (35 bit). {@code ctl} = versione (bit
 * 32..63, bit meno significativo = 1 mentre qualcuno scrive) e il campo "meta" di {@link TranspositionTable} (profondita', tipo,
 * eta', mossa: 32 bit; profondita' 0 = slot vuoto). Lettura: si legge {@code ctl}; se lo slot e' vuoto o in scrittura, miss; si leggono i
 * dati e si rilegge {@code ctl}: se e' cambiato, miss; solo uno snapshot coerente viene confrontato con la chiave completa.
 * Scrittura: CAS di {@code ctl} a versione dispari (se fallisce o e' gia' dispari si RINUNCIA: un solo thread scrive uno slot), poi dati,
 * poi {@code ctl} a versione pari+2. Le scritture saltate o le sovrascritture riducono solo gli hit, mai la correttezza. Unica
 * ipotesi: un lettore non resta sospeso a meta' lookup mentre altre 2^31 scritture colpiscono lo STESSO slot.
 * <p>
 * Costo: 40 byte per voce contro 24 della tabella a hash, cioe' a parita' di memoria il 40% di voci in meno
 * ({@link #bytesFor(int)}). Nessuna allocazione nel percorso di probe/store. Si usa solo tramite {@link #probeInto(FastState, int,
 * ProbeResult)} e {@link #store(FastState, int, double, byte, int)}: le versioni con il solo hash lanciano
 * {@link UnsupportedOperationException} (non esiste una chiave esatta a partire da un hash). Vale solo con stati la cui chiave e'
 * aggiornata (si veda {@link FastState#key0}). Compatibile Java 8.
 */
public final class ExactTranspositionTable extends TranspositionTable {

	private static final int ENTRY_LONGS = 5;
	/** Byte per voce. */
	public static final int ENTRY_BYTES = ENTRY_LONGS * 8;
	private static final int NO_MOVE_FIELD = 0xFFFF;
	private static final long TAIL_MASK = (1L << 35) - 1;
	private static final long VERSION_ONE = 1L << 32;
	private static final long META_MASK = 0xFFFFFFFFL;
	private static final long SAME = 1L << 40;

	private final int size;
	private final int mask;
	private final boolean twoSlot;
	private final AtomicLongArray slots;
	private volatile int generation;

	/** @param sizeHint voci desiderate (arrotondate alla potenza di 2 superiore) */
	public ExactTranspositionTable(int sizeHint) {
		this(sizeHint, false);
	}

	/** @param twoSlot vero per i bucket a due slot, come in {@link TranspositionTable#TranspositionTable(int, boolean)} */
	public ExactTranspositionTable(int sizeHint, boolean twoSlot) {
		super(1, false); // la tabella di base resta minuscola (3 long): non la usiamo
		int n = 1;
		while (n < sizeHint) {
			n <<= 1;
		}
		if (twoSlot && n < 2) {
			n = 2;
		}
		this.size = n;
		this.twoSlot = twoSlot;
		this.mask = twoSlot ? (n >> 1) - 1 : n - 1;
		this.slots = new AtomicLongArray(n * ENTRY_LONGS);
	}

	/** Byte occupati da una tabella di {@code entries} voci (gia' potenza di 2). */
	public static long bytesFor(long entries) {
		return entries * ENTRY_BYTES;
	}

	private int index(long zobrist) {
		return (twoSlot ? (int) (zobrist & mask) * 2 : (int) (zobrist & mask)) * ENTRY_LONGS;
	}

	@Override
	public boolean isTwoSlot() {
		return twoSlot;
	}

	@Override
	public int capacity() {
		return size;
	}

	@Override
	public void newSearch() {
		generation = (generation + 1) & 63;
	}

	@Override
	public double occupancy(int samples) {
		final int step = Math.max(1, size / Math.max(1, samples));
		int seen = 0, used = 0;
		for (int i = 0; i < size; i += step) {
			seen++;
			if ((slots.get(i * ENTRY_LONGS + 4) & 0xFF) != 0) {
				used++;
			}
		}
		return seen == 0 ? 0.0 : (double) used / seen;
	}

	@Override
	public void clear() {
		for (int i = 0; i < slots.length(); i++) {
			slots.set(i, 0L);
		}
	}

	@Override
	public void probeInto(long hash, int neededDepth, ProbeResult r) {
		throw new UnsupportedOperationException("ExactTranspositionTable si interroga con lo stato: probeInto(FastState, int, ProbeResult)");
	}

	@Override
	public void store(long hash, int depth, double score, byte flag, int packedMove) {
		throw new UnsupportedOperationException("ExactTranspositionTable si scrive con lo stato: store(FastState, int, double, byte, int)");
	}

	// ------------------------------------------------------------------------------------------------------------------ lettura

	@Override
	public void probeInto(FastState s, int neededDepth, ProbeResult r) {
		r.hit = false;
		r.depthSufficient = false;
		r.score = 0.0;
		r.flag = 0;
		r.packedMove = -1;
		final long k0 = s.key0, k1 = s.key1, tail = s.key2 & TAIL_MASK;
		final int base = index(s.zobristHash);
		if (!probeSlot(base, k0, k1, tail, neededDepth, r) && twoSlot) {
			probeSlot(base + ENTRY_LONGS, k0, k1, tail, neededDepth, r);
		}
	}

	private boolean probeSlot(int base, long k0, long k1, long tail, int neededDepth, ProbeResult r) {
		final long c1 = slots.get(base + 4);
		if ((c1 & 0xFF) == 0 || (c1 & VERSION_ONE) != 0) {
			return false; // vuoto o in scrittura
		}
		final long a = slots.get(base), b = slots.get(base + 1), t = slots.get(base + 2), sc = slots.get(base + 3);
		if (slots.get(base + 4) != c1) {
			return false; // strappata: qualcuno ha scritto fra le due letture
		}
		if (a != k0 || b != k1 || (t & TAIL_MASK) != tail) {
			return false; // posizione diversa
		}
		final long meta = c1 & META_MASK;
		r.hit = true;
		final int mv = (int) ((meta >>> 16) & 0xFFFF);
		r.packedMove = (mv == NO_MOVE_FIELD) ? -1 : mv;
		r.score = Double.longBitsToDouble(sc);
		r.flag = (byte) ((meta >>> 8) & 3);
		r.depthSufficient = ((int) (meta & 0xFF)) - 1 >= neededDepth;
		return true;
	}

	/** -1 se lo slot non e' leggibile (scrittura in corso); altrimenti il suo campo meta, piu' il bit SAME se contiene la posizione richiesta. */
	private long inspect(int base, long k0, long k1, long tail) {
		final long c1 = slots.get(base + 4);
		if ((c1 & VERSION_ONE) != 0) {
			return -1L;
		}
		if ((c1 & 0xFF) == 0) {
			return 0L; // vuoto
		}
		final long a = slots.get(base), b = slots.get(base + 1), t = slots.get(base + 2);
		if (slots.get(base + 4) != c1) {
			return -1L;
		}
		return (c1 & META_MASK) | ((a == k0 && b == k1 && (t & TAIL_MASK) == tail) ? SAME : 0L);
	}

	// ----------------------------------------------------------------------------------------------------------------- scrittura

	@Override
	public void store(FastState s, int depth, double score, byte flag, int packedMove) {
		final long k0 = s.key0, k1 = s.key1, tail = s.key2 & TAIL_MASK;
		final int b0 = index(s.zobristHash);
		if (!twoSlot) {
			final long x = inspect(b0, k0, k1, tail);
			if (x < 0) {
				return;
			}
			final boolean empty = (x & 0xFF) == 0;
			final boolean same = (x & SAME) != 0;
			final int oldDepth = ((int) (x & 0xFF)) - 1;
			if (empty || !same || depth >= oldDepth) {
				write(b0, k0, k1, tail, packMeta(depth, flag, packedMove), score);
			}
			return;
		}
		final int b1 = b0 + ENTRY_LONGS;
		final long p = inspect(b0, k0, k1, tail);
		final long a = inspect(b1, k0, k1, tail);
		if (p < 0 || a < 0) {
			return; // un altro thread sta scrivendo uno dei due slot: si rinuncia
		}
		final boolean pEmpty = (p & 0xFF) == 0;
		final boolean pSame = !pEmpty && (p & SAME) != 0;
		final int pDepth = ((int) (p & 0xFF)) - 1;
		if (pSame) {
			if (depth >= pDepth) {
				write(b0, k0, k1, tail, packMeta(depth, flag, packedMove, generation), score);
			}
			return;
		}
		final boolean aEmpty = (a & 0xFF) == 0;
		final boolean aSame = !aEmpty && (a & SAME) != 0;
		if (aSame) {
			if (depth >= ((int) (a & 0xFF)) - 1) {
				write(b1, k0, k1, tail, packMeta(depth, flag, packedMove, generation), score);
			}
			return;
		}
		final boolean pStale = ((int) ((p >>> 10) & 63)) != generation;
		if (pEmpty || depth >= pDepth || pStale) {
			write(b0, k0, k1, tail, packMeta(depth, flag, packedMove, generation), score);
		} else {
			write(b1, k0, k1, tail, packMeta(depth, flag, packedMove, generation), score);
		}
	}

	private void write(int base, long k0, long k1, long tail, long meta, double score) {
		final long c = slots.get(base + 4);
		if ((c & VERSION_ONE) != 0) {
			return;
		}
		final long writing = c + VERSION_ONE; // versione dispari
		if (!slots.compareAndSet(base + 4, c, writing)) {
			return;
		}
		slots.lazySet(base, k0);
		slots.lazySet(base + 1, k1);
		slots.lazySet(base + 2, tail);
		slots.lazySet(base + 3, Double.doubleToRawLongBits(score));
		slots.lazySet(base + 4, ((writing + VERSION_ONE) & ~META_MASK) | meta);
	}
}
