package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;

/**
 * Un algoritmo di ricerca della mossa, intercambiabile: {@link NegamaxSearcher} (alpha-beta + Lazy SMP, l'algoritmo storico)
 * oppure {@link Mcts} (Monte Carlo Tree Search). E' l'unica cosa che il client e le arene devono conoscere per poter scegliere
 * l'algoritmo con un'opzione, senza che {@code Negamax} (verificato contro il motore storico) debba cambiare una riga.
 * <p>
 * Contratto comune (lo stesso di {@code Negamax.findBestMove(FastState, long, PositionHistory)}):
 * <ul>
 * <li>{@code root} non viene modificata ne' conservata;</li>
 * <li>{@code history} sono le posizioni dall'ULTIMA CATTURA, la radice compresa come ultima voce, piu' vecchia per prima, come
 * chiavi ESATTE (non hash: due posizioni diverse non possono mai essere scambiate per la stessa); e' la storia che serve alla regola
 * "stessa posizione due volte = pareggio"; e' copiata, mai modificata;</li>
 * <li>il budget {@code timeBudgetMillis} e' il timeout della mossa imposto dal server: l'implementazione ne toglie da se' un margine
 * di sicurezza;</li>
 * <li>ritorna la mossa ({@link it.unibo.ai.didattica.competition.tablut.engine.Move}) oppure {@code Move.NONE} se chi deve muovere
 * non ha mosse legali.</li>
 * </ul>
 * Un'istanza non va usata da piu' thread contemporaneamente (una sola ricerca alla volta); puo' tenere stato tra una ricerca e la
 * successiva (tabella di trasposizione, sottoalbero riutilizzabile).
 */
public interface MoveSearcher {

	/** Trova la mossa migliore per chi deve muovere in {@code root}; si veda il contratto nel commento di classe. */
	int findBestMove(FastState root, long timeBudgetMillis, PositionHistory history);

	/** Nome breve dell'algoritmo, per i messaggi di avvio. */
	String name();

	/** Statistiche dell'ultima ricerca in una riga, per i log del client. */
	String describeLastSearch();

	/** Lavoro svolto dall'ultima ricerca, per confronti: nodi (Negamax) oppure valutazioni statiche (MCTS). */
	long lastWork();

	/** Profondita' dell'ultima ricerca: quella completata (Negamax) oppure la profondita' massima raggiunta dall'albero (MCTS). */
	double lastDepth();

	/** Rilascia i thread di lavoro (sono comunque "daemon"). */
	void shutdown();
}
