package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Adattatore che fa di {@link Negamax} un {@link MoveSearcher}. Non aggiunge logica: inoltra le chiamate e legge i campi
 * {@code last*}. Esiste perche' {@code Negamax} resta intatto (e' verificato nodo per nodo contro il motore storico).
 */
public final class NegamaxSearcher implements MoveSearcher {

	private final Negamax negamax;

	public NegamaxSearcher(Negamax negamax) {
		if (negamax == null) {
			throw new IllegalArgumentException("negamax nullo");
		}
		this.negamax = negamax;
	}

	/** La ricerca sottostante (per chi deve configurarla ulteriormente o leggerne le statistiche complete). */
	public Negamax negamax() {
		return negamax;
	}

	@Override
	public int findBestMove(FastState root, long timeBudgetMillis, long[] history, int historyLen) {
		return negamax.findBestMove(root, timeBudgetMillis, history, historyLen);
	}

	@Override
	public String name() {
		return "negamax";
	}

	@Override
	public String describeLastSearch() {
		return "profondita' raggiunta=" + negamax.lastDepthReached + " nodi=" + negamax.lastNodeCount + " valore="
				+ String.format("%.1f", negamax.lastScore);
	}

	@Override
	public long lastWork() {
		return negamax.lastTotalNodeCount;
	}

	@Override
	public double lastDepth() {
		return negamax.lastDepthReached;
	}

	@Override
	public void shutdown() {
		negamax.shutdown();
	}
}
