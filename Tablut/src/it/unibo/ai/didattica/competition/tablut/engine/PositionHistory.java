package it.unibo.ai.didattica.competition.tablut.engine;

import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.domain.State;

/**
 * Storia di posizioni rappresentata con la CHIAVE ESATTA di {@link FastState} ({@code key0, key1, key2}: la scacchiera e il turno,
 * 163 bit), non con l'hash Zobrist: due posizioni diverse hanno sempre chiavi diverse, quindi non possono mai essere scambiate per la
 * stessa (nemmeno per un falso rilevamento di ripetizione). Ogni posizione occupa {@value #LONGS_PER_POSITION} {@code long}, senza oggetti
 * per posizione. Ordine: la piu' vecchia per prima. Nel significato usato dalla ricerca: le posizioni REALMENTE occorse in partita
 * dall'ultima cattura, radice inclusa.
 * <p>
 * Non e' thread-safe: una ricerca ne usa una copia propria (i metodi {@code findBestMove} non la modificano e non la conservano).
 * Una chiave vale solo se lo stato che la fornisce ha la chiave aggiornata (si veda {@link FastState#key0}); per uno {@link State}
 * del dominio la conversione passa da {@link FastState#fromState(State)}, che la calcola.
 */
public final class PositionHistory {

	/** Quanti {@code long} occupa una posizione. */
	public static final int LONGS_PER_POSITION = 3;

	private long[] keys;
	private int size;

	public PositionHistory() {
		this(16);
	}

	/** @param capacity numero di posizioni per cui riservare spazio (cresce da solo) */
	public PositionHistory(int capacity) {
		this.keys = new long[LONGS_PER_POSITION * Math.max(1, capacity)];
	}

	/** Copia indipendente di {@code other}. */
	public PositionHistory(PositionHistory other) {
		this.keys = Arrays.copyOf(other.keys, Math.max(LONGS_PER_POSITION, LONGS_PER_POSITION * other.size));
		this.size = other.size;
	}

	/** Una storia vuota (nuova a ogni chiamata: non e' un singleton, perche' e' modificabile). */
	public static PositionHistory empty() {
		return new PositionHistory();
	}

	/** Storia con le posizioni date, la piu' vecchia per prima. */
	public static PositionHistory of(FastState... states) {
		final PositionHistory h = new PositionHistory(states.length);
		for (FastState s : states) {
			h.add(s);
		}
		return h;
	}

	/** Storia con gli stati del dominio dati, il piu' vecchio per primo. */
	public static PositionHistory of(State... states) {
		final PositionHistory h = new PositionHistory(states.length);
		for (State s : states) {
			h.add(s);
		}
		return h;
	}

	public int size() {
		return size;
	}

	public boolean isEmpty() {
		return size == 0;
	}

	public void clear() {
		size = 0;
	}

	/** Accoda la posizione di {@code s} (la sua chiave esatta corrente). */
	public void add(FastState s) {
		add(s.key0, s.key1, s.key2);
	}

	/** Accoda la posizione di uno {@link State} del dominio (lo converte in {@link FastState}: O(81)). */
	public void add(State s) {
		add(FastState.fromState(s));
	}

	public void add(long k0, long k1, long k2) {
		if (LONGS_PER_POSITION * (size + 1) > keys.length) {
			keys = Arrays.copyOf(keys, Math.max(LONGS_PER_POSITION * (size + 1), keys.length * 2));
		}
		final int o = LONGS_PER_POSITION * size++;
		keys[o] = k0;
		keys[o + 1] = k1;
		keys[o + 2] = k2;
	}

	/** Accoda tutte le posizioni di {@code other}. */
	public void addAll(PositionHistory other) {
		for (int i = 0; i < other.size; i++) {
			add(other.key0(i), other.key1(i), other.key2(i));
		}
	}

	/** Toglie l'ultima posizione. */
	public void removeLast() {
		if (size == 0) {
			throw new IllegalStateException("storia vuota");
		}
		size--;
	}

	public long key0(int i) {
		return keys[LONGS_PER_POSITION * checkIndex(i)];
	}

	public long key1(int i) {
		return keys[LONGS_PER_POSITION * checkIndex(i) + 1];
	}

	public long key2(int i) {
		return keys[LONGS_PER_POSITION * checkIndex(i) + 2];
	}

	private int checkIndex(int i) {
		if (i < 0 || i >= size) {
			throw new IndexOutOfBoundsException("indice " + i + ", dimensione " + size);
		}
		return i;
	}

	/** Vero se la storia contiene ESATTAMENTE la posizione di {@code s} (stessa scacchiera e stesso turno). */
	public boolean contains(FastState s) {
		return contains(s.key0, s.key1, s.key2);
	}

	public boolean contains(State s) {
		return contains(FastState.fromState(s));
	}

	public boolean contains(long k0, long k1, long k2) {
		final long[] a = keys;
		for (int i = 0, o = 0; i < size; i++, o += LONGS_PER_POSITION) {
			if (a[o] == k0 && a[o + 1] == k1 && a[o + 2] == k2) {
				return true;
			}
		}
		return false;
	}

	/** Vero se l'ultima posizione e' quella di {@code s}. */
	public boolean endsWith(FastState s) {
		if (size == 0) {
			return false;
		}
		final int o = LONGS_PER_POSITION * (size - 1);
		return keys[o] == s.key0 && keys[o + 1] == s.key1 && keys[o + 2] == s.key2;
	}

	/**
	 * Copia le chiavi in {@code dst} a partire dalla posizione {@code dstPosition} (in posizioni, non in {@code long}), nello stesso
	 * formato interno: {@value #LONGS_PER_POSITION} {@code long} per posizione. {@code dst} deve essere abbastanza grande.
	 *
	 * @return il numero di posizioni copiate
	 */
	public int copyKeysTo(long[] dst, int dstPosition) {
		System.arraycopy(keys, 0, dst, LONGS_PER_POSITION * dstPosition, LONGS_PER_POSITION * size);
		return size;
	}

	/** Copia indipendente. */
	public PositionHistory copy() {
		return new PositionHistory(this);
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof PositionHistory)) {
			return false;
		}
		final PositionHistory p = (PositionHistory) o;
		if (p.size != size) {
			return false;
		}
		for (int i = 0; i < LONGS_PER_POSITION * size; i++) {
			if (keys[i] != p.keys[i]) {
				return false;
			}
		}
		return true;
	}

	@Override
	public int hashCode() {
		int h = size;
		for (int i = 0; i < LONGS_PER_POSITION * size; i++) {
			h = 31 * h + Long.hashCode(keys[i]);
		}
		return h;
	}

	@Override
	public String toString() {
		return "PositionHistory[" + size + " posizioni]";
	}
}
