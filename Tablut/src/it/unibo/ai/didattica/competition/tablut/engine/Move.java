package it.unibo.ai.didattica.competition.tablut.engine;

import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Una mossa impacchettata in un {@code int}: {@code da | (a << 7)} con {@code da, a} in 0..80 (celle {@code riga*9+colonna},
 * si veda {@link BoardConstants}). Occupa 14 bit: entra nel campo mossa a 16 bit della transposition table e si
 * confronta/ordina come un intero. Sostituisce {@code Action} (due stringhe, tre {@code parseInt} per leggerne le
 * coordinate) in tutto il ciclo caldo della ricerca; {@link #toAction} la riconverte solo ai bordi (server, log).
 */
public final class Move {

	/** Nessuna mossa. */
	public static final int NONE = -1;

	/** Quanti valori distinti di {@link #index} esistono (da*81 + a): dimensione delle tabelle killer/history. */
	public static final int INDEX_RANGE = BoardConstants.NUM_SQUARES * BoardConstants.NUM_SQUARES;

	private Move() {
	}

	public static int of(int from, int to) {
		return from | (to << 7);
	}

	public static int getFrom(int move) {
		return move & 127;
	}

	public static int getTo(int move) {
		return move >>> 7;
	}

	/** Indice denso {@code da*81 + a} in [0, 6561): lo stesso schema delle tabelle killer e history del motore storico. */
	public static int index(int move) {
		return (move & 127) * BoardConstants.NUM_SQUARES + (move >>> 7);
	}

	/** Inversa di {@link #index}. */
	public static int fromIndex(int index) {
		return Move.of(index / BoardConstants.NUM_SQUARES, index % BoardConstants.NUM_SQUARES);
	}

	/** Converte una {@link Action} del dominio (coordinate algebriche, es. "e2" -> "e4") nella mossa compatta. */
	public static int fromAction(Action a) {
		return of(BoardConstants.coord(a.getRowFrom(), a.getColumnFrom()), BoardConstants.coord(a.getRowTo(), a.getColumnTo()));
	}

	/**
	 * Converte la mossa in una {@link Action} (alloca: usare solo ai bordi, mai nella ricerca).
	 *
	 * @param turn il giocatore che muove (finisce nel campo {@code turn} dell'azione, come in {@code AshtonRules})
	 */
	public static Action toAction(int move, Turn turn) {
		try {
			return new Action(BoardConstants.name(getFrom(move)), BoardConstants.name(getTo(move)), turn);
		} catch (IOException e) {
			throw new IllegalStateException(e); // non puo' accadere: i nomi hanno sempre lunghezza 2
		}
	}

	/** Forma leggibile, es. "e2-e4". */
	public static String toString(int move) {
		if (move < 0) {
			return "none";
		}
		return BoardConstants.name(getFrom(move)) + "-" + BoardConstants.name(getTo(move));
	}
}
