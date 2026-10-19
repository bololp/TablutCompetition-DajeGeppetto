package it.unibo.ai.didattica.competition.tablut.engine;

/**
 * Costanti e tabelle precalcolate della scacchiera 9x9 di Ashton Tablut, per la rappresentazione leggera
 * {@link FastState}.
 *
 * <h3>Indicizzazione</h3>
 * Una cella e' un intero {@code sq = riga*9 + colonna} (0..80), con la STESSA convenzione di
 * {@code State.getBoard()[riga][colonna]} e di {@code Action} (riga 0 = traversa "1", colonna 0 = linea "a"):
 * {@code a1 = 0}, {@code e5 = 40} (il trono), {@code i9 = 80}.
 *
 * <h3>Codici delle pedine</h3>
 * Coincidono con {@code State.Pawn.ordinal()} (EMPTY=0, WHITE=1, BLACK=2, THRONE=3, KING=4): la conversione da/verso
 * {@code State} e' quindi una copia diretta. L'unica differenza e' il trono: nella scacchiera compatta il trono
 * VUOTO e' {@link #EMPTY} (e' implicito in "cella 40 vuota"), il codice {@link #THRONE_MARK} non compare mai in
 * {@code FastState.board}: serve solo come "tipo di pezzo" fittizio per la chiave Zobrist del trono vuoto e per le
 * copie con bordo sentinella degli estrattori di feature.
 *
 * <h3>Direzioni</h3>
 * Stesso ordine di {@code AshtonRules}: riga-1, riga+1, colonna-1, colonna+1 (indici 0..3). Mantenerlo uguale fa
 * generare le mosse nello STESSO ordine del motore storico: i confronti mossa per mossa (test di equivalenza) e il
 * tie-break alla radice della ricerca restano identici.
 */
public final class BoardConstants {

	public static final int SIZE = 9;
	public static final int NUM_SQUARES = SIZE * SIZE;

	/** Cella del trono (e5). */
	public static final int THRONE = 40;

	// ---- codici pedina (= Pawn.ordinal()) ----
	public static final byte EMPTY = 0;
	public static final byte WHITE = 1;
	public static final byte BLACK = 2;
	/** Solo marcatore (chiave Zobrist del trono vuoto, copie padded delle feature): MAI dentro {@code FastState.board}. */
	public static final byte THRONE_MARK = 3;
	public static final byte KING = 4;

	// ---- turno ----
	public static final byte WHITE_TURN = 0;
	public static final byte BLACK_TURN = 1;

	/** Numero di direzioni ortogonali. */
	public static final int NUM_DIRS = 4;
	/** Scostamento di indice per direzione: riga-1, riga+1, colonna-1, colonna+1. */
	public static final int[] DIR_OFFSET = { -SIZE, SIZE, -1, 1 };

	/** Riga e colonna di ogni cella. */
	public static final byte[] ROW = new byte[NUM_SQUARES];
	public static final byte[] COL = new byte[NUM_SQUARES];

	/**
	 * {@code STEPS[d][sq]} = quante celle restano dalla cella {@code sq} al bordo muovendosi nella direzione {@code d}
	 * (cioe' il massimo numero di passi che non esce dalla scacchiera). Sostituisce tutti i controlli di limiti.
	 */
	public static final byte[][] STEPS = new byte[NUM_DIRS][NUM_SQUARES];

	/** Le 16 citadel (accampamenti neri), come in {@code GameAshtonTablut}. */
	public static final boolean[] IS_CITADEL = new boolean[NUM_SQUARES];
	/** Le 4 punte delle citadel (a5, e1, e9, i5): non contano come muro ostile quando il BIANCO cattura. */
	public static final boolean[] IS_TIP = new boolean[NUM_SQUARES];
	/** Celle sul bordo (se il Re vi arriva, il Bianco vince). */
	public static final boolean[] IS_EDGE = new boolean[NUM_SQUARES];
	/** Le 4 celle ortogonalmente adiacenti al trono: d5, f5, e4, e6 (nell'ordine di {@code AshtonRules}). */
	public static final int[] THRONE_NEIGHBORS = { coord(3, 4), coord(5, 4), coord(4, 3), coord(4, 5) };
	public static final boolean[] IS_THRONE_NEIGHBOR = new boolean[NUM_SQUARES];

	static {
		for (int r = 0; r < SIZE; r++) {
			for (int c = 0; c < SIZE; c++) {
				int sq = coord(r, c);
				ROW[sq] = (byte) r;
				COL[sq] = (byte) c;
				STEPS[0][sq] = (byte) r; // riga-1
				STEPS[1][sq] = (byte) (SIZE - 1 - r); // riga+1
				STEPS[2][sq] = (byte) c; // colonna-1
				STEPS[3][sq] = (byte) (SIZE - 1 - c); // colonna+1
				IS_EDGE[sq] = r == 0 || r == SIZE - 1 || c == 0 || c == SIZE - 1;
			}
		}
		int[][] citadels = { { 3, 0 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, // a4,a5,a6,b5
				{ 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, // d1,e1,f1,e2
				{ 3, 8 }, { 4, 8 }, { 5, 8 }, { 4, 7 }, // i4,i5,i6,h5
				{ 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 } }; // d9,e9,f9,e8
		for (int[] cell : citadels) {
			IS_CITADEL[coord(cell[0], cell[1])] = true;
		}
		int[][] tips = { { 4, 0 }, { 0, 4 }, { 8, 4 }, { 4, 8 } };
		for (int[] cell : tips) {
			IS_TIP[coord(cell[0], cell[1])] = true;
		}
		for (int n : THRONE_NEIGHBORS) {
			IS_THRONE_NEIGHBOR[n] = true;
		}
	}

	private BoardConstants() {
	}

	public static int coord(int row, int col) {
		return row * SIZE + col;
	}

	public static int row(int sq) {
		return ROW[sq];
	}

	public static int col(int sq) {
		return COL[sq];
	}

	/** Nome algebrico della cella ({@code 0 -> "a1"}, {@code 40 -> "e5"}), lo stesso formato di {@code State.getBox}. */
	public static String name(int sq) {
		return "" + (char) ('a' + COL[sq]) + (ROW[sq] + 1);
	}
}
