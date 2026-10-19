package it.unibo.ai.didattica.competition.tablut.training;

import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Regole di FINE PARTITA per il self-play, coerenti con la ricerca ({@code Negamax}) e con la regola ufficiale. Prima di
 * questa classe il trainer le ignorava in due punti, con il risultato che l'esito con cui la rete veniva addestrata non
 * coincideva con quello che la ricerca (e il server) attribuiscono alla stessa situazione:
 * <ol>
 * <li><b>Nessuna mossa legale = SCONFITTA</b> per chi deve muovere (come in {@code Negamax}: punteggio WIN_SCORE+profondita'
 * per l'avversario). Il trainer usciva dal ciclo senza stato terminale ed etichettava la partita come pareggio (0).</li>
 * <li><b>Ripetizione = PAREGGIO</b>. Una posizione (stessa scacchiera e stesso turno) che si ripresenta dopo l'ultima cattura
 * chiude la partita in parita' (regola ufficiale con {@code repeatedState=0}). Il trainer non teneva alcuna storia: la ricerca
 * non conosceva la regola e le partite che giravano in tondo proseguivano fino al tetto di semi-mosse, oppure finivano
 * "vinte" molto dopo aver ripassato per una posizione che in una partita vera avrebbe gia' chiuso la partita in parita'.</li>
 * </ol>
 * La storia ha lo stesso contenuto che il client passa alla ricerca ({@code TablutNegamaxClient.updateRealGameHistory}): gli
 * hash Zobrist di tutte le posizioni dall'ULTIMA CATTURA in poi, compresa quella corrente, azzerata a ogni cattura (dopo una
 * cattura nessuna posizione precedente puo' ripresentarsi). Una cattura e' rilevata con {@code AshtonRules.wasCapture}.
 */
public final class GameTracker {

	/** Etichette di fine partita. Le prime tre coincidono con {@code Turn.toString()} (compatibilita' con i log esistenti). */
	public static final String LABEL_WHITE_WIN = "WW";
	public static final String LABEL_BLACK_WIN = "BW";
	public static final String LABEL_DRAW = "D";
	/** Ripetizione di posizione: pareggio. */
	public static final String LABEL_REPETITION = "REPETITION";
	/** Chi doveva muovere non aveva mosse legali: ha perso. L'etichetta indica il VINCITORE. */
	public static final String LABEL_NOMOVES_WHITE_WIN = "NOMOVES_WW";
	public static final String LABEL_NOMOVES_BLACK_WIN = "NOMOVES_BW";
	/** Partita troncata al tetto di semi-mosse: trattata come pareggio. */
	public static final String LABEL_MAXPLIES = "MAXPLIES";

	/** Esito di una partita dal punto di vista del Bianco: +1 vince il Bianco, -1 vince il Nero, 0 pareggio/troncata. */
	public static final class Result {
		public final double outcome;
		public final String label;

		Result(double outcome, String label) {
			this.outcome = outcome;
			this.label = label;
		}

		@Override
		public String toString() {
			return label + "(" + outcome + ")";
		}
	}

	private final List<Long> history = new ArrayList<>();
	private boolean repetitionDraw;

	/** @param initial posizione di partenza (viene inserita nella storia) */
	public GameTracker(State initial) {
		history.add(FastState.hashOf(initial));
	}

	/**
	 * Hash delle posizioni dall'ultima cattura, compresa la corrente: va passata a
	 * {@code Negamax.findBestMove(stato, tempo, storia)}. E' la lista VIVA del tracker (la ricerca ne fa una copia).
	 */
	public List<Long> history() {
		return history;
	}

	/** Vero se l'ultima mossa applicata ha ricreato una posizione gia' vista dall'ultima cattura: la partita e' patta. */
	public boolean isRepetitionDraw() {
		return repetitionDraw;
	}

	/**
	 * Applica {@code move} e aggiorna la storia: una cattura la azzera (resta solo la nuova posizione); altrimenti, se la
	 * nuova posizione e' gia' nella storia la partita e' patta per ripetizione, se no viene aggiunta. Uno stato terminale
	 * (vittoria) non tocca la storia: la partita e' finita.
	 */
	public State apply(State state, Action move) {
		State next = AshtonRules.applyMove(state, move);
		if (AshtonRules.isTerminal(next)) {
			return next;
		}
		long h = FastState.hashOf(next);
		if (AshtonRules.wasCapture(state, next, move)) {
			history.clear();
			history.add(h);
		} else if (history.contains(h)) {
			repetitionDraw = true;
		} else {
			history.add(h);
		}
		return next;
	}

	/** Esito della partita nello stato {@code finalState} (l'ultimo raggiunto), tenendo conto di un'eventuale ripetizione. */
	public Result result(State finalState) {
		return resultOf(finalState, repetitionDraw);
	}

	/**
	 * Esito in uno stato finale. Ordine: ripetizione (pareggio) > stato terminale (vittoria/pareggio del motore) > stallo
	 * (chi deve muovere non ha mosse: sconfitta) > troncata (pareggio).
	 */
	public static Result resultOf(State finalState, boolean repetitionDraw) {
		if (repetitionDraw) {
			return new Result(0.0, LABEL_REPETITION);
		}
		Turn t = finalState.getTurn();
		if (t == Turn.WHITEWIN) {
			return new Result(1.0, LABEL_WHITE_WIN);
		}
		if (t == Turn.BLACKWIN) {
			return new Result(-1.0, LABEL_BLACK_WIN);
		}
		if (t == Turn.DRAW) {
			return new Result(0.0, LABEL_DRAW);
		}
		if ((t == Turn.WHITE || t == Turn.BLACK) && AshtonRules.generateMoves(finalState).isEmpty()) {
			// stesso criterio di Negamax: chi non puo' muovere perde
			return t == Turn.WHITE ? new Result(-1.0, LABEL_NOMOVES_BLACK_WIN) : new Result(1.0, LABEL_NOMOVES_WHITE_WIN);
		}
		return new Result(0.0, LABEL_MAXPLIES);
	}
}
