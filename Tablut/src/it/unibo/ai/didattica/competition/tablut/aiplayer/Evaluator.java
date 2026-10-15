package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.domain.State;

/**
 * Funzione di valutazione statica di una posizione NON terminale.
 * <p>
 * Convenzione: il punteggio e' sempre dal punto di vista del BIANCO (valori
 * positivi = posizione favorevole al Bianco, negativi = favorevole al Nero).
 * E' {@link Negamax} a occuparsi di girare il segno in base a chi deve
 * muovere, cosi' questa classe puo' restare semplice e "stateless".
 * <p>
 * Questa interfaccia e' pensata per essere intercambiabile: oggi
 * {@link SimpleEvaluator} implementa un'euristica scritta a mano; domani si
 * potra' scrivere un {@code NeuralEvaluator} che carica una rete allenata
 * (per esempio un value network alla AlphaZero) e la usa qui, senza
 * modificare una sola riga di {@link Negamax}.
 */
public interface Evaluator {
	/**
	 * @param state uno stato NON terminale (il chiamante gestisce già i casi
	 *              WHITEWIN / BLACKWIN / DRAW prima di arrivare qui)
	 * @return un punteggio, positivo se favorevole al Bianco
	 */
	double evaluate(State state);
}
