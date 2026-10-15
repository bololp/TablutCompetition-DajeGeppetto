package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;

/**
 * Le 8 simmetrie della scacchiera 9x9 (gruppo diedrale D4: 4 rotazioni x eventuale riflessione).
 * <p>
 * Le regole Ashton sono invarianti per tutte e 8 (citadel, trono, bordo, catture): una posizione e la sua
 * immagine speculare hanno lo STESSO valore. Un CNN, a differenza delle feature a mano, e' solo equivariante per
 * traslazione, non per rotazione/riflessione: usare le 8 simmetrie come <i>data augmentation</i> in
 * addestramento gli insegna l'invarianza e moltiplica per 8 i dati effettivi, senza costare nulla a runtime.
 * <p>
 * L'invarianza delle regole NON e' data per scontata: {@code ConvNetCheck} la verifica sul motore
 * ({@link AshtonRules}) su posizioni di partita, mossa per mossa.
 */
public final class BoardSymmetry {

	public static final int COUNT = 8;
	private static final int N = AshtonRules.SIZE;

	/** {@code MAP[s][i]} = indice (riga*9+colonna) in cui finisce la cella {@code i} applicando la simmetria {@code s}. */
	public static final int[][] MAP = new int[COUNT][N * N];

	static {
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				int i = r * N + c;
				MAP[0][i] = r * N + c; // identita'
				MAP[1][i] = c * N + (N - 1 - r); // rotazione 90
				MAP[2][i] = (N - 1 - r) * N + (N - 1 - c); // rotazione 180
				MAP[3][i] = (N - 1 - c) * N + r; // rotazione 270
				MAP[4][i] = r * N + (N - 1 - c); // riflessione orizzontale
				MAP[5][i] = (N - 1 - r) * N + c; // riflessione verticale
				MAP[6][i] = c * N + r; // trasposta
				MAP[7][i] = (N - 1 - c) * N + (N - 1 - r); // anti-trasposta
			}
		}
	}

	private BoardSymmetry() {
	}

	/** Copia di {@code s} con la scacchiera trasformata dalla simmetria {@code sym} (il turno resta invariato). */
	public static State transform(State s, int sym) {
		State t = s.clone();
		Pawn[][] src = s.getBoard();
		Pawn[][] dst = t.getBoard();
		for (int r = 0; r < N; r++) {
			for (int c = 0; c < N; c++) {
				int j = MAP[sym][r * N + c];
				dst[j / N][j % N] = src[r][c];
			}
		}
		return t;
	}
}
