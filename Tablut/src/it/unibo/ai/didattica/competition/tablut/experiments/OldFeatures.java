package it.unibo.ai.didattica.competition.tablut.experiments;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Trasforma uno {@link State} in un vettore di feature numeriche in input
 * alla rete neurale ({@link MLP} / {@link NeuralEvaluator}).
 * <p>
 * <b>Importante</b>: questa e' l'UNICA classe che sa come si costruisce il
 * vettore di feature. Sia {@link NeuralEvaluator} (usato durante il gioco)
 * sia {@code SelfPlayTrainer} (usato durante il training) chiamano
 * {@link #extract(State)}: se in futuro cambi/aggiungi feature, lo fai in un
 * solo posto e training e inferenza restano automaticamente coerenti.
 * <p>
 * Le feature sono tutte normalizzate approssimativamente in [-1,1], una
 * buona pratica per reti piccole allenate con discesa del gradiente (evita
 * che una feature con scala molto piu' grande delle altre domini
 * l'apprendimento).
 * <p>
 * Questo e' volutamente un set di feature "fatte a mano" (non la board
 * grezza): con poche centinaia/migliaia di partite di self-play generabili
 * in un progetto di corso, imparare da zero pattern spaziali sulla board
 * grezza richiederebbe molti piu' dati. Partire da feature gia' informative
 * (le stesse idee di {@link SimpleEvaluator}, piu' qualche dettaglio in
 * piu') permette alla rete di imparare "come combinarle al meglio" con dati
 * limitati. Estendere questa classe con piani della board grezza (una
 * matrice 9x9 per ogni tipo di pedina) e' il naturale passo successivo se in
 * futuro si genereranno molte piu' partite di training.
 */
public class OldFeatures {

	public static final int NUM_FEATURES = 16;

	private static final int[][] DIRS = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

	private static boolean inBounds(int r, int c) {
		return r >= 0 && r < AshtonRules.SIZE && c >= 0 && c < AshtonRules.SIZE;
	}

	public static double[] extract(State state) {
		Pawn[][] b = state.getBoard();
		int kr = -1, kc = -1;
		int white = 0, black = 0;

		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				Pawn p = b[r][c];
				if (p == Pawn.WHITE) {
					white++;
				} else if (p == Pawn.BLACK) {
					black++;
				} else if (p == Pawn.KING) {
					kr = r;
					kc = c;
				}
			}
		}

		double[] f = new double[NUM_FEATURES];
		int idx = 0;

		// --- materiale (3 feature) ---
		f[idx++] = (white - black) / 16.0;
		f[idx++] = white / 8.0;
		f[idx++] = black / 16.0;

		if (kr == -1) {
			// Stato non dovrebbe mai arrivare qui senza Re (vedi nota in SimpleEvaluator):
			// ritorniamo un vettore "innocuo" invece di lanciare un'eccezione dentro
			// un ciclo di training che potrebbe girare per ore.
			return f;
		}

		// --- posizione del Re (3 feature) ---
		f[idx++] = (kr - 4) / 4.0;
		f[idx++] = (kc - 4) / 4.0;
		int distEdge = Math.min(Math.min(kr, 8 - kr), Math.min(kc, 8 - kc));
		f[idx++] = distEdge / 4.0;

		// --- liberta' di fuga del Re: aggregata + per direzione (6 feature) ---
		int openLanes = 0;
		int freedom = 0;
		double[] laneOpen = new double[4];
		for (int d = 0; d < 4; d++) {
			int step = 1;
			while (true) {
				int nr = kr + DIRS[d][0] * step;
				int nc = kc + DIRS[d][1] * step;
				if (!inBounds(nr, nc) || b[nr][nc] != Pawn.EMPTY) {
					break;
				}
				freedom++;
				if (nr == 0 || nr == AshtonRules.SIZE - 1 || nc == 0 || nc == AshtonRules.SIZE - 1) {
					laneOpen[d] = 1.0;
					openLanes++;
					break;
				}
				step++;
			}
		}
		f[idx++] = openLanes / 4.0;
		f[idx++] = freedom / 32.0;
		f[idx++] = laneOpen[0];
		f[idx++] = laneOpen[1];
		f[idx++] = laneOpen[2];
		f[idx++] = laneOpen[3];

		// --- pericolo/difesa immediata intorno al Re (2 feature) ---
		int danger = 0, defenders = 0;
		for (int[] d : DIRS) {
			int nr = kr + d[0], nc = kc + d[1];
			if (inBounds(nr, nc)) {
				if (b[nr][nc] == Pawn.BLACK) {
					danger++;
				} else if (b[nr][nc] == Pawn.WHITE) {
					defenders++;
				}
			}
		}
		f[idx++] = danger / 4.0;
		f[idx++] = defenders / 4.0;

		// --- controllo centrale del Nero (1 feature) ---
		double central = 0;
		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				if (b[r][c] == Pawn.BLACK) {
					double dist = Math.abs(r - 4) + Math.abs(c - 4);
					central += (8 - dist) / 8.0;
				}
			}
		}
		f[idx++] = central / 16.0;

		// --- chi deve muovere (1 feature: tempo/iniziativa) ---
		f[idx++] = state.getTurn() == Turn.WHITE ? 1.0 : -1.0;

		return f;
	}
}
