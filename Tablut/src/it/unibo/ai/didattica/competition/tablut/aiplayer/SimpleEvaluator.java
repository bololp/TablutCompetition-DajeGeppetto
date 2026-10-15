package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;

/**
 * Euristica "classica" scritta a mano. Non e' pensata per essere perfetta, ma
 * per essere un punto di partenza ragionevole e soprattutto FACILE DA TARARE:
 * i pesi sono campi pubblici non-final, modificabili liberamente (per provare
 * varianti, per un piccolo tuning automatico, ecc.).
 * <p>
 * L'idea chiave in Tablut e' che il materiale conta relativamente poco
 * rispetto alla "liberta' di fuga" del Re: una sola linea (riga o colonna)
 * completamente libera tra il Re e il bordo e' quasi sempre vittoria per il
 * Bianco (il Nero puo' bloccarne una per turno, quindi due linee libere
 * contemporaneamente sono spesso gia' decisive). Per questo
 * {@code kingOpenLaneWeight} ha un peso molto piu' alto di
 * {@code materialWeight}.
 */
public class SimpleEvaluator implements Evaluator {

	private static final int[][] DIRS = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

	/** Valore di una pedina di differenza (bianche - nere). */
	public double materialWeight = 10.0;
	/** Valore di ogni "corsia" completamente libera dal Re fino al bordo. */
	public double kingOpenLaneWeight = 45.0;
	/** Valore di ogni cella libera raggiungibile dal Re in linea retta (mobilita'). */
	public double kingFreedomWeight = 3.0;
	/** Penalita' per ogni pedina nera ortogonalmente adiacente al Re. */
	public double kingDangerWeight = 15.0;
	/** Piccolo bonus posizionale per il Nero che controlla il centro (fase iniziale). */
	public double centralControlWeight = 1.0;

	private static boolean inBounds(int r, int c) {
		return r >= 0 && r < AshtonRules.SIZE && c >= 0 && c < AshtonRules.SIZE;
	}

	@Override
	public double evaluate(State state) {
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

		if (kr == -1) {
			// Non dovrebbe mai capitare in uno stato non terminale: la cattura del
			// Re e' gia' gestita come BLACKWIN direttamente da AshtonRules/Negamax.
			// Per sicurezza, se succedesse comunque, trattiamolo come pessimo per il Bianco.
			return -100000;
		}

		double score = materialWeight * (white - black);

		int openLanes = 0;
		int freedom = 0;
		for (int[] d : DIRS) {
			int step = 1;
			while (true) {
				int nr = kr + d[0] * step;
				int nc = kc + d[1] * step;
				if (!inBounds(nr, nc)) {
					break;
				}
				if (b[nr][nc] != Pawn.EMPTY) {
					break;
				}
				freedom++;
				if (nr == 0 || nr == AshtonRules.SIZE - 1 || nc == 0 || nc == AshtonRules.SIZE - 1) {
					openLanes++;
					break;
				}
				step++;
			}
		}
		score += kingOpenLaneWeight * openLanes;
		score += kingFreedomWeight * freedom;

		int danger = 0;
		for (int[] d : DIRS) {
			int nr = kr + d[0], nc = kc + d[1];
			if (inBounds(nr, nc) && b[nr][nc] == Pawn.BLACK) {
				danger++;
			}
		}
		score -= kingDangerWeight * danger;

		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				if (b[r][c] == Pawn.BLACK) {
					double dist = Math.abs(r - 4) + Math.abs(c - 4);
					score -= centralControlWeight * (8 - dist) / 8.0;
				}
			}
		}

		return score;
	}
}
