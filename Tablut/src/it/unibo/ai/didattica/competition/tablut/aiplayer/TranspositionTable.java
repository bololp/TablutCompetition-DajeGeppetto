package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.concurrent.atomic.AtomicLongArray;

import it.unibo.ai.didattica.competition.tablut.domain.Action;

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
 * Le mosse sono "impacchettate" (4 bit per ciascuna delle 4 coordinate riga/colonna) invece che oggetti
 * {@link Action}; 0xFFFF = nessuna mossa.
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

	/** @param sizeHint dimensione desiderata (arrotondata alla potenza di 2 superiore) */
	public TranspositionTable(int sizeHint) {
		int size = 1;
		while (size < sizeHint) {
			size <<= 1;
		}
		this.mask = size - 1;
		this.slots = new AtomicLongArray(size * ENTRY_LONGS); // tutto zero = tutti gli slot vuoti
	}

	private int index(long hash) {
		return (int) (hash & mask);
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
		long d = Math.max(0, Math.min(254, depth)) + 1L;
		long m = (packedMove < 0) ? NO_MOVE_FIELD : (packedMove & 0xFFFF);
		return d | ((long) (flag & 3) << 8) | (m << 16);
	}

	public ProbeResult probe(long hash, int neededDepth) {
		ProbeResult r = new ProbeResult();
		int base = index(hash) * ENTRY_LONGS;
		long check = slots.get(base);
		long meta = slots.get(base + 1);
		long scoreBits = slots.get(base + 2);
		if ((meta & 0xFF) == 0 || (check ^ meta ^ scoreBits) != hash) {
			return r; // slot vuoto, posizione diversa o voce "strappata" da scritture concorrenti: miss
		}
		r.hit = true;
		int mv = (int) ((meta >>> 16) & 0xFFFF);
		r.packedMove = (mv == NO_MOVE_FIELD) ? -1 : mv;
		r.score = Double.longBitsToDouble(scoreBits);
		r.flag = (byte) ((meta >>> 8) & 3);
		r.depthSufficient = ((int) (meta & 0xFF)) - 1 >= neededDepth;
		return r;
	}

	/**
	 * Politica di sostituzione "depth-preferred": si sovrascrive uno slot se e' vuoto, se rappresenta una posizione
	 * diversa (o e' illeggibile), oppure se la nuova ricerca e' andata almeno tanto in profondita' quanto quella gia'
	 * memorizzata. Altrimenti si mantiene l'entry piu' profonda (costata piu' lavoro).
	 */
	public void store(long hash, int depth, double score, byte flag, int packedMove) {
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

	void writeRaw(long hash, long check, long meta, long scoreBits) {
		int base = index(hash) * ENTRY_LONGS;
		slots.lazySet(base, check);
		slots.lazySet(base + 1, meta);
		slots.lazySet(base + 2, scoreBits);
	}
}
