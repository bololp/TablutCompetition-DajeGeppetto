package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.concurrent.atomic.AtomicLongArray;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Transposition table per Negamax con potatura alpha-beta, <b>condivisibile tra thread senza lock</b>
 * (Lazy SMP: tutti i thread di ricerca leggono e scrivono la stessa tabella).
 * <p>
 * <b>Rappresentazione.</b> Ogni voce occupa 3 {@code long} consecutivi in un {@link AtomicLongArray}
 * (compatibile Java 8; {@code get} e {@code lazySet} costano come una load/store normale su x86):
 * <pre>
 *   [0] check = chiave ^ meta ^ scoreBits      (verifica di integrita', si veda sotto)
 *   [1] meta  = (profondita'+1) | flag &lt;&lt; 8 | mossa &lt;&lt; 16   (profondita'+1 = 0 significa "slot vuoto")
 *   [2] scoreBits = Double.doubleToRawLongBits(punteggio)  (punteggio ESATTO, nessun arrotondamento)
 * </pre>
 * Le mosse sono "impacchettate" in un intero a 16 bit invece che oggetti {@link Action}; 0xFFFF = nessuna mossa. La
 * ricerca ({@link Negamax}) vi scrive direttamente la mossa compatta di {@link it.unibo.ai.didattica.competition.tablut.engine.Move}
 * (14 bit); i vecchi {@link #packMove(Action)} e {@code packedFromRow...} (4 bit per coordinata) restano per i test e per
 * il motore storico congelato.
 * <p>
 * <b>Perche' non serve un lock: il trucco "XOR" di Hyatt.</b> Una voce e' scritta con tre store separati.
 * Se due thread scrivono contemporaneamente nello stesso slot, un lettore potrebbe vedere i campi di una
 * scrittura mescolati con quelli dell'altra (per esempio il punteggio di una posizione con la chiave di
 * un'altra): usarli produrrebbe tagli alpha-beta SBAGLIATI, senza alcun errore visibile. Il campo
 * {@code check} lo impedisce: il lettore ricostruisce {@code chiave = check ^ meta ^ scoreBits} e usa la
 * voce solo se coincide con la chiave cercata. Una voce "strappata" non supera quasi mai la verifica (servirebbe
 * una coincidenza a 64 bit) e viene trattata come un semplice <i>miss</i>, che e' sempre sicuro. Quindi la
 * concorrenza puo' al massimo far perdere informazione, mai produrne di falsa.
 * <p>
 * <b>Semantica EXACT / LOWER_BOUND / UPPER_BOUND</b> (standard per alpha-beta + TT): il valore trovato e'
 * "esatto" solo se e' strettamente tra alpha e beta; se e' &gt;= beta (taglio) e' un limite INFERIORE del vero
 * valore; se nessuna mossa ha migliorato alpha e' un limite SUPERIORE. Riutilizzare un bound senza rispettare
 * questa distinzione produrrebbe risultati scorretti, non solo subottimali.
 * <p>
 * <b>Sostituzione "depth-preferred"</b> (invariata): si sovrascrive uno slot se e' vuoto, se contiene un'altra
 * posizione, o se la nuova ricerca e' andata almeno tanto in profondita' di quella memorizzata.
 */
public class TranspositionTable {

	public static final byte EXACT = 0;
	public static final byte LOWER_BOUND = 1;
	public static final byte UPPER_BOUND = 2;

	private static final int NO_MOVE_FIELD = 0xFFFF;
	private static final int ENTRY_LONGS = 3;

	private final int mask;
	private final AtomicLongArray slots;
	/** Numero di voci (slot) totali, potenza di 2. */
	private final int size;
	/**
	 * Politica a due slot per bucket (opzionale, spenta di default): lo slot 0 del bucket e' "depth-preferred con aging", lo slot 1
	 * "sempre-sostituisci". Con una tabella satura (una ricerca lunga con piu' thread) la politica a slot singolo lascia che
	 * QUALUNQUE posizione diversa nello stesso slot cancelli una voce profonda; qui la voce profonda sopravvive nello slot 0 finche'
	 * una piu' profonda (o della stessa ricerca, si veda {@link #newSearch}) non la sostituisce.
	 */
	private final boolean twoSlot;
	/** Generazione corrente (6 bit, bit 10..15 del campo meta): incrementata da {@link #newSearch}. */
	private volatile int generation;

	/** @param sizeHint dimensione desiderata (arrotondata alla potenza di 2 superiore) */
	public TranspositionTable(int sizeHint) {
		this(sizeHint, false);
	}

	/**
	 * @param sizeHint dimensione desiderata in voci (arrotondata alla potenza di 2 superiore; a parita' di memoria)
	 * @param twoSlot  vero per i bucket a due slot (depth-preferred con aging + sempre-sostituisci)
	 */
	public TranspositionTable(int sizeHint, boolean twoSlot) {
		int n = 1;
		while (n < sizeHint) {
			n <<= 1;
		}
		if (twoSlot && n < 2) {
			n = 2;
		}
		this.size = n;
		this.twoSlot = twoSlot;
		this.mask = twoSlot ? (n >> 1) - 1 : n - 1; // con due slot l'indice e' quello del BUCKET
		this.slots = new AtomicLongArray(n * ENTRY_LONGS); // tutto zero = tutti gli slot vuoti
	}

	/** Primo long della voce (o del bucket) in cui cade {@code hash}. */
	private int index(long hash) {
		return twoSlot ? (int) (hash & mask) * 2 : (int) (hash & mask);
	}

	public boolean isTwoSlot() {
		return twoSlot;
	}

	/** Numero di voci totali (slot). */
	public int capacity() {
		return size;
	}

	/** Segna l'inizio di una nuova ricerca (nuova mossa di partita): le voci scritte prima diventano "vecchie" e lo slot 0 puo' sostituirle. */
	public void newSearch() {
		generation = (generation + 1) & 63;
	}

	/** Frazione di slot non vuoti, stimata su al piu' {@code samples} slot equidistanti (0 se la tabella e' vuota). */
	public double occupancy(int samples) {
		final int step = Math.max(1, size / Math.max(1, samples));
		int seen = 0, used = 0;
		for (int i = 0; i < size; i += step) {
			seen++;
			if ((slots.get(i * ENTRY_LONGS + 1) & 0xFF) != 0) {
				used++;
			}
		}
		return seen == 0 ? 0.0 : (double) used / seen;
	}

	public static int packMove(Action a) {
		return (a.getRowFrom() << 12) | (a.getColumnFrom() << 8) | (a.getRowTo() << 4) | a.getColumnTo();
	}

	public static int packedFromRow(int packed) {
		return (packed >> 12) & 0xF;
	}

	public static int packedFromCol(int packed) {
		return (packed >> 8) & 0xF;
	}

	public static int packedToRow(int packed) {
		return (packed >> 4) & 0xF;
	}

	public static int packedToCol(int packed) {
		return packed & 0xF;
	}

	/** Risultato di una interrogazione della tabella. */
	public static final class ProbeResult {
		public boolean hit;
		public boolean depthSufficient;
		public double score;
		public byte flag;
		public int packedMove = -1; // utile per l'ordinamento anche quando depthSufficient e' false
	}

	/** Campo "meta" di una voce (package-private: usato anche dai test sulle voci strappate). */
	static long packMeta(int depth, byte flag, int packedMove) {
		return packMeta(depth, flag, packedMove, 0);
	}

	static long packMeta(int depth, byte flag, int packedMove, int gen) {
		long d = Math.max(0, Math.min(254, depth)) + 1L;
		long m = (packedMove < 0) ? NO_MOVE_FIELD : (packedMove & 0xFFFF);
		return d | ((long) (flag & 3) << 8) | ((long) (gen & 63) << 10) | (m << 16);
	}

	public ProbeResult probe(long hash, int neededDepth) {
		ProbeResult r = new ProbeResult();
		probeInto(hash, neededDepth, r);
		return r;
	}

	/**
	 * Come {@link #probe}, ma scrive il risultato in {@code r} (che viene azzerato) invece di allocarne uno nuovo: e' la
	 * forma usata dalla ricerca, che interroga la tabella a ogni nodo. {@code r} va riletto subito, prima di ricorrere
	 * (un'istanza per thread di ricerca basta).
	 */
	public void probeInto(long hash, int neededDepth, ProbeResult r) {
		r.hit = false;
		r.depthSufficient = false;
		r.score = 0.0;
		r.flag = 0;
		r.packedMove = -1;
		int base = index(hash) * ENTRY_LONGS;
		long check = slots.get(base);
		long meta = slots.get(base + 1);
		long scoreBits = slots.get(base + 2);
		if ((meta & 0xFF) == 0 || (check ^ meta ^ scoreBits) != hash) {
			if (twoSlot) {
				// secondo slot del bucket (sempre-sostituisci)
				base += ENTRY_LONGS;
				check = slots.get(base);
				meta = slots.get(base + 1);
				scoreBits = slots.get(base + 2);
				if ((meta & 0xFF) != 0 && (check ^ meta ^ scoreBits) == hash) {
					fillProbe(r, meta, scoreBits, neededDepth);
				}
			}
			return; // slot vuoto, posizione diversa o voce "strappata" da scritture concorrenti: miss
		}
		fillProbe(r, meta, scoreBits, neededDepth);
	}

	/**
	 * Come {@link #probeInto(long, int, ProbeResult)} ma identifica la posizione dallo STATO. Qui usa lo Zobrist ({@code s.zobristHash}), cioe'
	 * si comporta esattamente come prima; {@link ExactTranspositionTable} lo ridefinisce con la chiave esatta di {@link FastState}.
	 */
	public void probeInto(FastState s, int neededDepth, ProbeResult r) {
		probeInto(s.zobristHash, neededDepth, r);
	}

	/** Come {@link #store(long, int, double, byte, int)} ma identifica la posizione dallo stato (si veda {@link #probeInto(FastState, int, ProbeResult)}). */
	public void store(FastState s, int depth, double score, byte flag, int packedMove) {
		store(s.zobristHash, depth, score, flag, packedMove);
	}

	private static void fillProbe(ProbeResult r, long meta, long scoreBits, int neededDepth) {
		r.hit = true;
		int mv = (int) ((meta >>> 16) & 0xFFFF);
		r.packedMove = (mv == NO_MOVE_FIELD) ? -1 : mv;
		r.score = Double.longBitsToDouble(scoreBits);
		r.flag = (byte) ((meta >>> 8) & 3);
		r.depthSufficient = ((int) (meta & 0xFF)) - 1 >= neededDepth;
	}

	/**
	 * Politica di sostituzione "depth-preferred": si sovrascrive uno slot se e' vuoto, se rappresenta una posizione
	 * diversa (o e' illeggibile), oppure se la nuova ricerca e' andata almeno tanto in profondita' quanto quella gia'
	 * memorizzata. Altrimenti si mantiene l'entry piu' profonda (costata piu' lavoro).
	 */
	public void store(long hash, int depth, double score, byte flag, int packedMove) {
		if (twoSlot) {
			storeTwoSlot(hash, depth, score, flag, packedMove);
			return;
		}
		int base = index(hash) * ENTRY_LONGS;
		long oldCheck = slots.get(base);
		long oldMeta = slots.get(base + 1);
		long oldScoreBits = slots.get(base + 2);
		boolean empty = (oldMeta & 0xFF) == 0;
		boolean sameKey = !empty && (oldCheck ^ oldMeta ^ oldScoreBits) == hash;
		int oldDepth = ((int) (oldMeta & 0xFF)) - 1;
		if (empty || !sameKey || depth >= oldDepth) {
			long meta = packMeta(depth, flag, packedMove);
			long scoreBits = Double.doubleToRawLongBits(score);
			slots.lazySet(base + 1, meta);
			slots.lazySet(base + 2, scoreBits);
			slots.lazySet(base, hash ^ meta ^ scoreBits);
		}
	}

	private void writeAt(int base, long hash, int depth, double score, byte flag, int packedMove) {
		long meta = packMeta(depth, flag, packedMove, generation);
		long scoreBits = Double.doubleToRawLongBits(score);
		slots.lazySet(base + 1, meta);
		slots.lazySet(base + 2, scoreBits);
		slots.lazySet(base, hash ^ meta ^ scoreBits);
	}

	/**
	 * Sostituzione a due slot. Stessa posizione gia' presente in uno slot: si aggiorna solo se la nuova ricerca e' almeno altrettanto
	 * profonda (come a slot singolo). Posizione NUOVA: va nello slot 0 se questo e' vuoto, se non e' piu' profondo della nuova, oppure se
	 * e' di una ricerca precedente (aging); altrimenti nello slot 1, che si sovrascrive sempre.
	 */
	private void storeTwoSlot(long hash, int depth, double score, byte flag, int packedMove) {
		final int b0 = index(hash) * ENTRY_LONGS;
		final int b1 = b0 + ENTRY_LONGS;
		final long pCheck = slots.get(b0), pMeta = slots.get(b0 + 1), pScore = slots.get(b0 + 2);
		final boolean pEmpty = (pMeta & 0xFF) == 0;
		final boolean pSame = !pEmpty && (pCheck ^ pMeta ^ pScore) == hash;
		final int pDepth = ((int) (pMeta & 0xFF)) - 1;
		if (pSame) {
			if (depth >= pDepth) {
				writeAt(b0, hash, depth, score, flag, packedMove);
			}
			return;
		}
		final long aCheck = slots.get(b1), aMeta = slots.get(b1 + 1), aScore = slots.get(b1 + 2);
		final boolean aEmpty = (aMeta & 0xFF) == 0;
		final boolean aSame = !aEmpty && (aCheck ^ aMeta ^ aScore) == hash;
		if (aSame) {
			if (depth >= ((int) (aMeta & 0xFF)) - 1) {
				writeAt(b1, hash, depth, score, flag, packedMove);
			}
			return;
		}
		final boolean pStale = ((int) ((pMeta >>> 10) & 63)) != generation;
		if (pEmpty || depth >= pDepth || pStale) {
			writeAt(b0, hash, depth, score, flag, packedMove);
		} else {
			writeAt(b1, hash, depth, score, flag, packedMove);
		}
	}

	/** Svuota la tabella. NON thread-safe: da chiamare solo quando nessuna ricerca e' in corso. */
	public void clear() {
		for (int i = 0; i < slots.length(); i++) {
			slots.lazySet(i, 0L);
		}
	}

	// ---- hook per i test (package-private): permettono di costruire a mano voci "strappate" ----

	int slotOf(long hash) {
		return index(hash);
	}

	long[] readRaw(long hash) {
		int base = index(hash) * ENTRY_LONGS;
		return new long[] { slots.get(base), slots.get(base + 1), slots.get(base + 2) };
	}

	/** Come {@link #readRaw} ma per lo slot {@code slot} (0 o 1) del bucket (a due slot) di {@code hash}. */
	long[] readRawSlot(long hash, int slot) {
		int base = (index(hash) + slot) * ENTRY_LONGS;
		return new long[] { slots.get(base), slots.get(base + 1), slots.get(base + 2) };
	}

	void writeRawSlot(long hash, int slot, long check, long meta, long scoreBits) {
		int base = (index(hash) + slot) * ENTRY_LONGS;
		slots.lazySet(base, check);
		slots.lazySet(base + 1, meta);
		slots.lazySet(base + 2, scoreBits);
	}

	/** Generazione corrente (per i test). */
	int generationForTest() {
		return generation;
	}

	void writeRaw(long hash, long check, long meta, long scoreBits) {
		int base = index(hash) * ENTRY_LONGS;
		slots.lazySet(base, check);
		slots.lazySet(base + 1, meta);
		slots.lazySet(base + 2, scoreBits);
	}
}
