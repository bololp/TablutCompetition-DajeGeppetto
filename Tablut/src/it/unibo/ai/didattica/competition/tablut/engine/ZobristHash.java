package it.unibo.ai.didattica.competition.tablut.engine;

import java.util.Random;

/**
 * Chiavi Zobrist per {@link FastState}: una chiave a 64 bit per ogni coppia (cella, tipo di pezzo) piu' una per
 * il turno del Nero.
 * <p>
 * Le chiavi sono GENERATE ESATTAMENTE come in {@code aiplayer.ZobristHasher} (stesso seme, stesso ordine di
 * estrazione): per ogni posizione non terminale l'hash incrementale di {@code FastState} coincide con
 * {@code ZobristHasher.fullHash(state)} calcolato sullo {@code State} equivalente. Lo verifica
 * {@code FastStateCheck}. Cosi' le storie di ripetizione gia' esistenti (client, {@code GameTracker}) restano
 * valide e il vecchio hasher serve da oracolo indipendente.
 * <p>
 * Convenzione (identica a quella storica): una cella vuota non contribuisce; il trono VUOTO contribuisce con la
 * chiave del "tipo THRONE" (distingue "trono vuoto" da "cella 40 mai considerata"); contribuisce con la chiave del
 * turno solo il Nero di turno in una posizione NON terminale (Bianco di turno e stati terminali: 0).
 */
public final class ZobristHash {

	/** {@code KEY[cella][codice pedina]}: indici 1 (bianco), 2 (nero), 3 (marcatore trono vuoto), 4 (re); 0 inutilizzato. */
	public static final long[][] KEY = new long[BoardConstants.NUM_SQUARES][5];
	public static final long TURN_KEY;
	/** Chiave del trono vuoto (cella 40, tipo THRONE). */
	public static final long THRONE_EMPTY_KEY;

	static {
		// Stesso seme e stesso ordine di ZobristHasher: per r, per c, per tipo (WHITE, BLACK, KING, THRONE), poi il turno.
		Random rnd = new Random(0xC0FFEEL);
		for (int r = 0; r < BoardConstants.SIZE; r++) {
			for (int c = 0; c < BoardConstants.SIZE; c++) {
				int sq = BoardConstants.coord(r, c);
				KEY[sq][BoardConstants.WHITE] = rnd.nextLong();
				KEY[sq][BoardConstants.BLACK] = rnd.nextLong();
				KEY[sq][BoardConstants.KING] = rnd.nextLong();
				KEY[sq][BoardConstants.THRONE_MARK] = rnd.nextLong();
			}
		}
		TURN_KEY = rnd.nextLong();
		THRONE_EMPTY_KEY = KEY[BoardConstants.THRONE][BoardConstants.THRONE_MARK];
	}

	private ZobristHash() {
	}

	/** Hash completo O(81) di una scacchiera compatta + turno. Usato solo per inizializzare/verificare: la ricerca usa l'aggiornamento incrementale. */
	public static long compute(byte[] board, byte turn, byte winner) {
		long h = 0L;
		for (int sq = 0; sq < BoardConstants.NUM_SQUARES; sq++) {
			byte p = board[sq];
			if (p != BoardConstants.EMPTY) {
				h ^= KEY[sq][p];
			}
		}
		if (board[BoardConstants.THRONE] == BoardConstants.EMPTY) {
			h ^= THRONE_EMPTY_KEY;
		}
		if (winner == FastState.WINNER_NONE && turn == BoardConstants.BLACK_TURN) {
			h ^= TURN_KEY;
		}
		return h;
	}
}
