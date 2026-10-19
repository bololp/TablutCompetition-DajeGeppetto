package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Funzione di valutazione statica di una posizione NON terminale.
 * <p>
 * Convenzione: il punteggio e' sempre dal punto di vista del BIANCO (valori
 * positivi = posizione favorevole al Bianco, negativi = favorevole al Nero).
 * E' {@link Negamax} a occuparsi di girare il segno in base a chi deve
 * muovere, cosi' questa classe puo' restare semplice e "stateless".
 * <p>
 * Questa interfaccia e' pensata per essere intercambiabile: oggi
 * {@link SimpleEvaluator} implementa un'euristica scritta a mano;
 * {@link NeuralEvaluator} e {@link CnnEvaluator} usano una rete allenata, senza
 * modificare una sola riga di {@link Negamax}.
 * <p>
 * <b>Rappresentazione.</b> Il metodo da implementare lavora sulla rappresentazione leggera
 * {@link FastState} (scacchiera {@code byte[81]}, conteggi e cella del Re gia' pronti): e' quella che la ricerca
 * possiede e muta con make/unmake, quindi la valutazione non copia ne' converte nulla. Per chi ha ancora uno
 * {@link State} del dominio (test, strumenti, dati del server) c'e' la forma di comodo {@link #evaluate(State)}, che
 * converte (alloca: non usarla nel ciclo caldo) e delega.
 */
public interface Evaluator {
	/**
	 * @param state una posizione NON terminale (il chiamante gestisce gia' i casi di vittoria / pareggio prima di
	 *              arrivare qui). Non va modificata.
	 * @return un punteggio, positivo se favorevole al Bianco
	 */
	double evaluate(FastState state);

	/** Forma di comodo per uno {@link State} del dominio: converte in {@link FastState} e delega. */
	default double evaluate(State state) {
		return evaluate(FastState.fromState(state));
	}
}
