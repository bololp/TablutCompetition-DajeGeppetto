package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Cache di valutazioni statiche indicizzata con l'hash Zobrist della posizione, condivisa tra i thread dell'{@link Mcts}.
 * <p>
 * <b>Perche' e' sicura.</b> Una valutazione statica e' una funzione della sola posizione (scacchiera + turno, quello che l'hash
 * identifica), NON del cammino che ci ha portato li': a differenza dei valori della ricerca (che dipendono dalla storia di ripetizione,
 * il problema GHI di {@code Negamax}) puo' essere riusata da qualunque nodo e da qualunque thread. Serve perche' in Tablut la stessa
 * posizione e' raggiunta spesso con ordini di mosse diversi (mosse di pezzi diversi che commutano) e l'MCTS valuta i figli di ogni
 * nodo espanso: senza cache la stessa posizione viene valutata piu' volte.
 * <p>
 * <b>Struttura.</b> Due {@code long} per voce in un {@link AtomicLongArray} (nessun lock): {@code [chiave ^ dato, dato]} (il "trucco
 * XOR" di Hyatt, lo stesso di {@link TranspositionTable}). Chi legge ricostruisce la chiave e usa la voce solo se coincide: una voce
 * "strappata" da due scritture concorrenti diventa un semplice miss, mai un valore sbagliato. Politica di sostituzione: sempre.
 */
public final class EvalCache {

	private final AtomicLongArray slots;
	private final int mask;
	private final int size;

	/** @param log2Entries la cache ha 2^log2Entries voci (16 byte ciascuna), con log2Entries in [4, 26] */
	public EvalCache(int log2Entries) {
		if (log2Entries < 4 || log2Entries > 26) {
			throw new IllegalArgumentException("log2Entries fuori da [4,26]: " + log2Entries);
		}
		this.size = 1 << log2Entries;
		this.mask = size - 1;
		this.slots = new AtomicLongArray(2 * size);
	}

	public int capacity() {
		return size;
	}

	/** Indice della voce: si mescolano i bit alti nei bassi perche' l'hash Zobrist e' gia' uniforme ma non lo si assume. */
	private int index(long hash) {
		return (int) (hash ^ (hash >>> 29)) & mask;
	}

	/** La valutazione salvata per {@code hash}, oppure {@link Float#NaN} se assente (o se la voce e' stata sovrascritta/strappata). */
	public float get(long hash) {
		final int i = 2 * index(hash);
		final long data = slots.get(i + 1);
		final long check = slots.get(i);
		if ((check ^ data) == hash && (data >>> 32) == 0x1L) { // il bit 32 marca la voce "occupata": una voce vuota (0,0) non coincide mai
			return Float.intBitsToFloat((int) data);
		}
		return Float.NaN;
	}

	/** Salva la valutazione {@code value} (non NaN) per {@code hash}, sostituendo quella precedente. */
	public void put(long hash, float value) {
		final int i = 2 * index(hash);
		final long data = (1L << 32) | (Float.floatToRawIntBits(value) & 0xFFFFFFFFL);
		slots.set(i + 1, data);
		slots.set(i, hash ^ data);
	}

	/** Voci occupate (O(capacita'): solo per statistiche e test). */
	public int occupancy() {
		int used = 0;
		for (int e = 0; e < size; e++) {
			if ((slots.get(2 * e + 1) >>> 32) == 0x1L) {
				used++;
			}
		}
		return used;
	}
}
