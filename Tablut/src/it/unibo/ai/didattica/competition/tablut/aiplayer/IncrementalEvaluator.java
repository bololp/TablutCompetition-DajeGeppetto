package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Un {@link Evaluator} che sa valutare IN MODO INCREMENTALE lungo il percorso di una ricerca make/unmake (stile NNUE).
 * <p>
 * Il contratto di {@link Evaluator} non cambia: {@link #evaluate(FastState)} resta la valutazione COMPLETA di una posizione, stateless e
 * thread-safe (serve a MCTS, trainer, test e a chi non ha un percorso). In piu' un evaluator incrementale sa creare una {@link Session}: lo
 * stato PRIVATO di UN worker di ricerca, che ne segue il percorso ply per ply e ricava la valutazione di un nodo dalle sole differenze
 * rispetto al genitore.
 * <p>
 * <b>Perche' una sessione e non lo stato dentro l'evaluator.</b> L'evaluator e' condiviso fra i thread di {@link Negamax} (Lazy SMP) e
 * fra i rami di una stessa ricerca; lo stato incrementale e' invece una funzione del percorso corrente, quindi appartiene al worker (uno
 * per istanza di {@code Negamax}, helper compresi) ed e' indicizzato dallo stesso ply dei suoi stack ({@code UndoInfo}, mosse). Nessuno
 * stato mutabile e' condiviso tra thread.
 * <p>
 * <b>Protocollo</b> (lo rispetta {@link Negamax}): {@link Session#reset} alla radice; dopo OGNI {@code FastState.makeMove} verso un
 * figlio, {@link Session#push}; {@link Session#evaluate} per valutare un nodo. L'annullamento di una mossa NON richiede nulla: i
 * frame sono uno stack e il frame del genitore non e' mai modificato dai figli, quindi tornare indietro (unmake) e passare al fratello
 * successivo (nuovo {@code push} sullo stesso ply) e' gratuito e corretto.
 */
public interface IncrementalEvaluator extends Evaluator {

	/**
	 * Crea la sessione di un worker di ricerca.
	 *
	 * @param capacityPlies numero di ply (radice compresa) che la sessione deve poter contenere senza crescere; una sessione cresce
	 *                      comunque da sola se serve di piu' (l'unico caso in cui alloca dopo la costruzione)
	 */
	Session newSession(int capacityPlies);

	/**
	 * Stato incrementale di UN worker. NON e' thread-safe: una sessione appartiene a un solo thread alla volta.
	 */
	interface Session {

		/** Installa {@code root} come ply 0 con una ricostruzione completa (inizializzazione). */
		void reset(FastState root);

		/**
		 * Registra la mossa appena eseguita: da chiamare DOPO {@code after.makeMove(move, undo)} e prima di valutare o di scendere.
		 *
		 * @param ply   ply del FIGLIO (la radice e' 0, quindi il primo figlio e' 1)
		 * @param move  la mossa ({@code Move}), la stessa passata a {@code makeMove}
		 * @param after la posizione DOPO la mossa
		 * @param undo  l'{@code UndoInfo} riempito da {@code makeMove} (catture comprese)
		 */
		void push(int ply, int move, FastState after, UndoInfo undo);

		/**
		 * Valuta la posizione del ply {@code ply}, con lo stesso significato di {@link Evaluator#evaluate(FastState)} (prospettiva del
		 * Bianco, posizione NON terminale). {@code state} e' la posizione di lavoro corrente della ricerca: l'implementazione puo'
		 * usarla per verificare la coerenza dello stato incrementale e, se serve, ricostruirlo da zero.
		 */
		double evaluate(int ply, FastState state);

		/**
		 * Quante volte la sessione ha dovuto RICOSTRUIRE da zero un frame perche' non riusciva ad applicare in sicurezza le differenze
		 * (stato incoerente con la posizione, differenza inutilizzabile). Le ricostruzioni attese (una per {@link #reset}) non contano.
		 * In una ricerca che rispetta il protocollo vale sempre 0: e' la spia di un'integrazione sbagliata, non un errore di valutazione
		 * (la valutazione resta corretta).
		 */
		long fallbackRebuilds();
	}
}
