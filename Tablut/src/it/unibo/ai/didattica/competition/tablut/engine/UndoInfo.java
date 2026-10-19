package it.unibo.ai.didattica.competition.tablut.engine;

/**
 * Tutto cio' che serve a {@link FastState#unmakeMove} per annullare una mossa in O(1): i campi scalari precedenti
 * e le (al massimo 4) pedine catturate. Si riusa senza allocare: la ricerca ne tiene uno per ogni ply.
 * <p>
 * Una mossa elimina al massimo 4 pezzi (una per ciascuna delle 4 celle adiacenti alla destinazione; quella verso
 * la cella di partenza e' sempre vuota, quindi in pratica al piu' 3), compreso il Re quando viene catturato.
 */
public final class UndoInfo {

	static final int MAX_CAPTURES = 4;

	byte oldTurn;
	byte oldKingSquare;
	long oldZobristHash;
	int capturedCount;
	final byte[] capturedSquares = new byte[MAX_CAPTURES];
	final byte[] capturedPieces = new byte[MAX_CAPTURES];

	void reset() {
		capturedCount = 0;
	}

	void addCapture(int square, byte piece) {
		capturedSquares[capturedCount] = (byte) square;
		capturedPieces[capturedCount] = piece;
		capturedCount++;
	}

	/** Vero se la mossa annullata da questo oggetto ha catturato almeno un pezzo (il Re compreso): equivale a {@code AshtonRules.wasCapture}. */
	public boolean wasCapture() {
		return capturedCount > 0;
	}

	public int capturedCount() {
		return capturedCount;
	}

	/** Cella dell'i-esima pedina catturata. */
	public int capturedSquare(int i) {
		return capturedSquares[i];
	}

	/** Codice ({@link BoardConstants}) dell'i-esima pedina catturata. */
	public byte capturedPiece(int i) {
		return capturedPieces[i];
	}
}
