package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Sessione incrementale di UN worker per {@link TemporalNnueEvaluator}. L'accumulatore delle feature di POSIZIONE e' gestito da una
 * {@link NnueSession} (differenze, valutazione pigra, firma di coerenza: stesso codice e stessi test dell'NNUE di base); qui si aggiunge solo la
 * parte temporale, che NON si aggiorna per differenza: a ogni valutazione si somma, in una copia dell'accumulatore, la riga di ogni feature
 * temporale attiva (al piu' 3 per mossa, {@code K} mosse). Una finestra si sposta di un ply a ogni mossa e cambierebbe TUTTE le sue feature
 * (l'eta' e' parte dell'indice): ricalcolarla costa meno che aggiornarla.
 * <p>
 * Finestra al ply {@code p}: i record dei frame {@code p, p-1, ...} (le mosse del percorso di ricerca, registrate da {@link #push}), poi, esauriti i
 * frame fino alla radice, la storia reale prima della radice (copiata da {@link TemporalNnueEvaluator} al {@link #reset}). Un record mancante
 * (mossa nulla, storia reale piu' corta) TERMINA la finestra: e' sempre un suffisso della storia vera. Una sessione appartiene a un solo worker.
 */
final class TemporalSession implements IncrementalEvaluator.Session {

	private final TemporalNnueEvaluator ev;
	private final TemporalNnueNetwork net;
	private final NnueSession inner;
	private final int hist;
	private int[] rec;
	private int[] prior = new int[0];
	private final int[] acc, work, feat, win;

	TemporalSession(TemporalNnueEvaluator ev, NnueSession inner, int capacityPlies) {
		this.ev = ev;
		this.net = ev.net;
		this.inner = inner;
		this.hist = net.hist();
		this.rec = new int[Math.max(2, capacityPlies)];
		this.acc = new int[net.hidden()];
		this.work = new int[net.base().scratchSize()];
		this.feat = new int[3 * hist];
		this.win = new int[hist];
	}

	@Override
	public void reset(FastState root) {
		inner.reset(root);
		prior = ev.currentPrior(); // array immutabile: basta il riferimento
	}

	private void ensure(int ply) {
		if (ply >= rec.length) {
			rec = Arrays.copyOf(rec, Math.max(ply + 1, rec.length * 2));
		}
	}

	@Override
	public void push(int ply, int move, FastState after, UndoInfo undo) {
		inner.push(ply, move, after, undo);
		ensure(ply);
		final int to = move >>> 7;
		rec[ply] = to < BoardConstants.NUM_SQUARES ? TemporalHistory.record(after.board[to], move & 127, to, undo.capturedCount()) : 0;
	}

	@Override
	public boolean pushNull(int ply, FastState after) {
		if (!inner.pushNull(ply, after)) {
			return false;
		}
		ensure(ply);
		rec[ply] = 0; // dopo una mossa nulla la storia precedente non e' piu' una successione di mosse vere: la finestra si interrompe
		return true;
	}

	@Override
	public double evaluate(int ply, FastState s) {
		final int[] a = inner.accumulator(ply, s);
		System.arraycopy(a, 0, acc, 0, acc.length);
		int len = 0;
		for (int age = 0; age < hist; age++) {
			final int f = ply - age;
			final int r = f >= 1 ? rec[f] : (-f < prior.length ? prior[-f] : 0);
			if (r == 0) {
				break;
			}
			win[len++] = r;
		}
		net.addTemporal(acc, win, 0, len, feat);
		final int role = s.turn == BoardConstants.BLACK_TURN ? 1 : 0;
		return NnueNetwork.valueOf(net.base().logit(acc, work, role)) * ev.outputScale;
	}

	@Override
	public long fallbackRebuilds() {
		return inner.fallbackRebuilds();
	}
}
