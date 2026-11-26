package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Ricorda le ultime mosse REALI di una partita (record di {@link TemporalHistory}, la piu' recente per prima), per dare alla radice della ricerca
 * la storia che precede la posizione corrente.
 * <p>
 * Il client vede lo stato solo quando tocca a lui: fra due osservazioni passano DUE mosse (la propria e quella dell'avversario) e dal solo
 * confronto delle scacchiere non si distinguerebbero. Per questo il protocollo e':
 * <ol>
 * <li>{@link #observe(FastState)} con la posizione in cui si deve muovere: deriva la mossa dell'avversario dal confronto con la posizione che il
 * tracker ricorda (quella dopo la propria ultima mossa, oppure la posizione iniziale alla prima chiamata);</li>
 * <li>si cerca la mossa;</li>
 * <li>{@link #own(FastState, int)} con la mossa scelta: il tracker la applica a una copia e la registra.</li>
 * </ol>
 * Se il confronto non e' una singola mossa (partita ripresa a meta', stato fuori sincronia) la storia viene AZZERATA: una finestra vuota e' sempre
 * un input valido per {@link TemporalNnueNetwork}, mai un input sbagliato. Thread-safe (metodi sincronizzati).
 */
public final class GameMoveTracker {

	private FastState last;
	private final int[] recent = new int[TemporalHistory.MAX_AGES];
	private int n;

	public synchronized void reset() {
		last = null;
		n = 0;
	}

	private void push(int rec) {
		if (rec == 0) {
			n = 0;
			return;
		}
		System.arraycopy(recent, 0, recent, 1, recent.length - 1);
		recent[0] = rec;
		n = Math.min(n + 1, recent.length);
	}

	/** Registra la mossa dell'avversario che ha portato a {@code pos} (posizione in cui si deve muovere). */
	public synchronized void observe(FastState pos) {
		if (last == null) {
			last = new FastState(); // posizione iniziale: se la partita e' appena cominciata la prima mossa si ricava da qui
		}
		if (Arrays.equals(last.board, pos.board) && last.turn == pos.turn) {
			return; // gia' osservata
		}
		final int role = last.turn == BoardConstants.BLACK_TURN ? 1 : 0;
		final int rec = pos.turn == last.turn ? 0 : TemporalHistory.deriveRecord(last.board, 0, pos.board, 0, role);
		push(rec);
		last = FastState.copyOf(pos);
	}

	/** Registra la PROPRIA mossa {@code move}, giocata dalla posizione {@code before}. */
	public synchronized void own(FastState before, int move) {
		final FastState after = FastState.copyOf(before);
		final UndoInfo u = new UndoInfo();
		after.makeMove(move, u);
		final int to = move >>> 7;
		push(TemporalHistory.record(after.board[to], move & 127, to, u.capturedCount()));
		last = after;
	}

	/** Copia delle mosse note, la piu' recente per prima (puo' essere vuota). */
	public synchronized int[] recent() {
		return Arrays.copyOf(recent, n);
	}
}
