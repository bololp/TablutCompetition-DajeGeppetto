package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Valutatore con feature temporali ({@link TemporalNnueNetwork}). Come {@link NnueEvaluator} offre due percorsi:
 * <ul>
 * <li>{@link #evaluate(FastState)}: stateless e thread-safe, SENZA storia (finestra vuota = la rete di base). E' quello di MCTS, dei test e di chi non
 * ha un percorso di ricerca;</li>
 * <li>{@link #newSession(int)}: sessione incrementale per un worker di {@link Negamax}. Le mosse lungo il percorso di ricerca entrano da sole
 * nella finestra (le vede da {@code push}); le mosse REALI prima della radice si passano con {@link #setPriorMoves} o, piu' comodamente, con il
 * protocollo di {@link GameMoveTracker} ({@link #beforeSearch}, {@link #afterOwnMove}). Senza di esse la rete funziona lo stesso (le prime posizioni
 * del percorso hanno finestre piu' corte, caso su cui il trainer l'ha addestrata).</li>
 * </ul>
 * La storia prima della radice e' UNO stato condiviso dei worker, ma immutabile per ricerca ({@code volatile} su un array mai modificato): ogni sessione
 * ne prende una copia al {@code reset}. Stesse convenzioni di {@link NnueEvaluator}: prospettiva del Bianco, solo posizioni non terminali, {@link #outputScale}.
 * Non va avvolto in {@link CachingEvaluator} (non e' cacheable: il valore dipende anche dalla storia).
 */
public final class TemporalNnueEvaluator implements IncrementalEvaluator {

	public final TemporalNnueNetwork net;
	/** Porta il valore in (-1,1) sulla scala di {@link SimpleEvaluator}, come gli altri valutatori neurali. */
	public double outputScale = 1000.0;

	private static final int[] NO_MOVES = new int[0];
	private volatile int[] prior = NO_MOVES;
	private final GameMoveTracker tracker = new GameMoveTracker();
	private final ThreadLocal<TemporalNnueNetwork.Scratch> scratch;
	private final NnueEvaluator baseEvaluator;

	public TemporalNnueEvaluator(TemporalNnueNetwork net) {
		if (net == null) {
			throw new IllegalArgumentException("rete temporale nulla");
		}
		this.net = net;
		this.baseEvaluator = new NnueEvaluator(net.base());
		this.scratch = ThreadLocal.withInitial(() -> new TemporalNnueNetwork.Scratch(net));
	}

	public static TemporalNnueEvaluator fromFile(String path) throws IOException {
		return new TemporalNnueEvaluator(TemporalNnueNetwork.load(path));
	}

	public void save(String path) throws IOException {
		net.save(path);
	}

	public String describe() {
		return net.describe();
	}

	/** Storia prima della radice delle PROSSIME ricerche, la mossa piu' recente per prima (record di {@link TemporalHistory}); vuota = nessuna. */
	public void setPriorMoves(int[] mostRecentFirst) {
		prior = mostRecentFirst == null || mostRecentFirst.length == 0 ? NO_MOVES : mostRecentFirst.clone();
	}

	/** Storia prima della radice attualmente impostata (copia). */
	public int[] priorMoves() {
		return prior.clone();
	}

	/** Da chiamare con la posizione in cui si deve muovere, PRIMA di cercare: aggiorna il tracker e la storia della radice. */
	public void beforeSearch(FastState root) {
		tracker.observe(root);
		setPriorMoves(tracker.recent());
	}

	/** Da chiamare DOPO aver scelto la mossa, con la posizione da cui e' stata scelta. */
	public void afterOwnMove(FastState before, int move) {
		tracker.own(before, move);
	}

	/** Inizio di una nuova partita. */
	public void newGame() {
		tracker.reset();
		prior = NO_MOVES;
	}

	@Override
	public double evaluate(FastState s) {
		final TemporalNnueNetwork.Scratch c = scratch.get();
		return NnueNetwork.valueOf(net.logit(s.board, s.turn == BoardConstants.BLACK_TURN ? 1 : 0, NO_MOVES, 0, 0, c)) * outputScale;
	}

	/** Valutazione completa di una posizione con la sua storia ({@code recs}, la mossa piu' recente per prima): il percorso di riferimento dei test. */
	public double evaluate(FastState s, int[] recs, int len) {
		final TemporalNnueNetwork.Scratch c = scratch.get();
		return NnueNetwork.valueOf(net.logit(s.board, s.turn == BoardConstants.BLACK_TURN ? 1 : 0, recs, 0, len, c)) * outputScale;
	}

	@Override
	public Session newSession(int capacityPlies) {
		return new TemporalSession(this, new NnueSession(baseEvaluator, capacityPlies), capacityPlies);
	}

	int[] currentPrior() {
		return prior;
	}
}
