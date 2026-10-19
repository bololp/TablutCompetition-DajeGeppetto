package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

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
 * <p>
 * Lavora su {@link FastState}: il materiale viene dai conteggi gia' mantenuti da
 * {@code makeMove} e la cella del Re e' un campo, quindi non c'e' nessuna scansione per
 * contare le pedine. La semantica e il risultato numerico (anche l'ordine delle somme in
 * virgola mobile) sono IDENTICI alla versione storica su {@code State}, comprese le sue
 * scelte note: una corsia e' "aperta" anche se termina su una citadel (si veda
 * MLP_REDESIGN.md), e il trono vuoto ferma la corsia del Re.
 */
public class SimpleEvaluator implements Evaluator {

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

	@Override
	public double evaluate(FastState state) {
		final byte[] b = state.board;
		final int ksq = state.kingSquare;
		if (ksq < 0) {
			// Non dovrebbe mai capitare in uno stato non terminale: la cattura del
			// Re e' gia' gestita come BLACKWIN direttamente da FastState/Negamax.
			// Per sicurezza, se succedesse comunque, trattiamolo come pessimo per il Bianco.
			return -100000;
		}

		double score = materialWeight * (state.whiteCount - state.blackCount);

		int openLanes = 0;
		int freedom = 0;
		for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
			final int steps = BoardConstants.STEPS[d][ksq];
			final int off = BoardConstants.DIR_OFFSET[d];
			int sq = ksq;
			for (int step = 1; step <= steps; step++) {
				sq += off;
				if (b[sq] != BoardConstants.EMPTY || sq == BoardConstants.THRONE) {
					break; // pedina, oppure trono vuoto
				}
				freedom++;
				if (BoardConstants.IS_EDGE[sq]) {
					openLanes++;
					break;
				}
			}
		}
		score += kingOpenLaneWeight * openLanes;
		score += kingFreedomWeight * freedom;

		int danger = 0;
		for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
			if (BoardConstants.STEPS[d][ksq] >= 1 && b[ksq + BoardConstants.DIR_OFFSET[d]] == BoardConstants.BLACK) {
				danger++;
			}
		}
		score -= kingDangerWeight * danger;

		// stesso ordine di somma (cella per cella) della versione storica: il risultato e' identico al bit
		for (int sq = 0; sq < BoardConstants.NUM_SQUARES; sq++) {
			if (b[sq] == BoardConstants.BLACK) {
				double dist = Math.abs(BoardConstants.ROW[sq] - 4) + Math.abs(BoardConstants.COL[sq] - 4);
				score -= centralControlWeight * (8 - dist) / 8.0;
			}
		}

		return score;
	}
}
