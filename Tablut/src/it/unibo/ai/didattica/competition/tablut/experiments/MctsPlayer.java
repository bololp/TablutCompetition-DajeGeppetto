package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Monte Carlo Tree Search (PUCT) SPERIMENTALE, scritto solo per rispondere a "conviene passare a un MCTS?" a parita' di evaluator
 * e di tempo con {@code Negamax} (si veda {@link MctsArena}). NON e' pensato per la gara.
 *
 * <h3>Scelte (e perche')</h3>
 * <ul>
 * <li><b>Nessun rollout casuale.</b> Le partite casuali a Tablut sono quasi prive di informazione (il Re scappa o viene catturato per
 * caso) e darebbero all'MCTS un handicap gratuito. Le foglie sono valutate dallo stesso {@link Evaluator} di Negamax.</li>
 * <li><b>Espansione con valutazione di tutti i figli.</b> Alla prima visita di un nodo si valutano staticamente tutte le mosse
 * (come fa un'alpha-beta alla frontiera): il valore statico di ogni figlio e' il suo Q iniziale ("first play urgency" informata) e,
 * attraverso un softmax, il suo prior. Cosi' non servono una rete di policy ne' rollout, e il costo per valutazione e' lo stesso
 * di Negamax. Il valore restituito a monte dalla foglia e' il migliore dei suoi figli (un minimax a 1 piano).</li>
 * <li><b>Valori</b> in [-1,1] dal punto di vista del Bianco: {@code evaluate()/valueScale} troncato (per gli evaluator neurali,
 * che escono in (-1,1)*1000, {@code valueScale=1000}). Terminali esatti: re sul bordo +1, re catturato -1, chi non ha mosse perde.</li>
 * <li><b>Mossa finale</b>: il figlio piu' visitato del radice (a parita', il Q migliore).</li>
 * </ul>
 * <h3>Limiti dichiarati</h3>
 * Nessuna ripetizione di posizione (come {@code EqualTimeArena}, che pareggia a 160 semi-mosse), nessun solver di posizioni risolte,
 * nessuna parallelizzazione, iperparametri ({@code cPuct}, {@code tau}) poco tarati: un MCTS ben ottimizzato potrebbe fare meglio
 * di questo; Negamax invece e' quello di gara, con TT, PVS, killer e history.
 */
public class MctsPlayer {

	private final Evaluator evaluator;
	private final double valueScale, cPuct, tau;

	/** Statistiche dell'ultima ricerca. */
	public long lastSims, lastEvals;
	public int lastMaxDepth;
	/** Valore medio del radice dal punto di vista del Bianco (-1..+1). */
	public double lastRootValue;

	public MctsPlayer(Evaluator evaluator, double valueScale, double cPuct, double tau) {
		this.evaluator = evaluator;
		this.valueScale = valueScale;
		this.cPuct = cPuct;
		this.tau = tau;
	}

	private static final class Node {
		final State state;
		final boolean terminal;
		final double termValue; // dal punto di vista del Bianco, valido se terminal
		Action[] moves;
		double[] stat; // valore statico di ciascun figlio (Bianco)
		double[] prior;
		Node[] kids;
		boolean expanded;
		int n;
		double sum; // somma dei valori (Bianco)

		Node(State s, boolean terminal, double termValue) {
			this.state = s;
			this.terminal = terminal;
			this.termValue = termValue;
		}
	}

	private static Node makeNode(State s) {
		Turn t = s.getTurn();
		if (t == Turn.WHITEWIN) {
			return new Node(s, true, 1.0);
		}
		if (t == Turn.BLACKWIN) {
			return new Node(s, true, -1.0);
		}
		if (t == Turn.DRAW) {
			return new Node(s, true, 0.0);
		}
		return new Node(s, false, 0.0);
	}

	private double staticValue(State s) {
		lastEvals++;
		double v = evaluator.evaluate(s) / valueScale;
		return v > 1 ? 1 : (v < -1 ? -1 : v);
	}

	/** Genera le mosse e valuta tutti i figli. Ritorna il valore del nodo (Bianco): il migliore dei figli per chi muove. */
	private double expand(Node nd) {
		List<Action> mv = AshtonRules.generateMoves(nd.state);
		nd.expanded = true;
		boolean whiteMoves = nd.state.getTurn() == Turn.WHITE;
		if (mv.isEmpty()) { // chi deve muovere e non puo' perde
			nd.moves = new Action[0];
			nd.stat = new double[0];
			nd.prior = new double[0];
			nd.kids = new Node[0];
			return whiteMoves ? -1.0 : 1.0;
		}
		int k = mv.size();
		nd.moves = mv.toArray(new Action[k]);
		nd.stat = new double[k];
		nd.prior = new double[k];
		nd.kids = new Node[k];
		double sign = whiteMoves ? 1 : -1;
		double best = -2, norm = 0;
		for (int i = 0; i < k; i++) {
			State c = AshtonRules.applyMove(nd.state, nd.moves[i]);
			Turn t = c.getTurn();
			double v = t == Turn.WHITEWIN ? 1.0 : t == Turn.BLACKWIN ? -1.0 : t == Turn.DRAW ? 0.0 : staticValue(c);
			nd.stat[i] = v;
			best = Math.max(best, sign * v);
		}
		for (int i = 0; i < k; i++) {
			nd.prior[i] = Math.exp((sign * nd.stat[i] - best) / tau); // softmax stabile
			norm += nd.prior[i];
		}
		for (int i = 0; i < k; i++) {
			nd.prior[i] /= norm;
		}
		return sign * best;
	}

	private int select(Node nd) {
		double sign = nd.state.getTurn() == Turn.WHITE ? 1 : -1;
		double sq = Math.sqrt(Math.max(1, nd.n));
		int bi = 0;
		double bs = -1e18;
		for (int i = 0; i < nd.moves.length; i++) {
			Node kid = nd.kids[i];
			double q = (kid != null && kid.n > 0) ? kid.sum / kid.n : nd.stat[i];
			int ni = kid == null ? 0 : kid.n;
			double s = sign * q + cPuct * nd.prior[i] * sq / (1 + ni);
			if (s > bs) {
				bs = s;
				bi = i;
			}
		}
		return bi;
	}

	/** @return la mossa scelta, o {@code null} se non ce ne sono (posizione terminale o senza mosse) */
	public Action findBestMove(State rootState, long budgetMillis) {
		long deadline = System.nanoTime() + budgetMillis * 1_000_000L;
		lastSims = 0;
		lastEvals = 0;
		lastMaxDepth = 0;
		Node root = makeNode(rootState);
		if (root.terminal) {
			return null;
		}
		expand(root);
		if (root.moves.length == 0) {
			return null;
		}
		Node[] path = new Node[256];
		while (System.nanoTime() < deadline || lastSims == 0) {
			for (int rep = 0; rep < 8; rep++) { // si controlla l'orologio ogni 8 simulazioni
				int d = 0;
				Node nd = root;
				path[d++] = nd;
				while (nd.expanded && !nd.terminal && nd.moves.length > 0 && d < path.length - 1) {
					int i = select(nd);
					Node kid = nd.kids[i];
					if (kid == null) {
						kid = makeNode(AshtonRules.applyMove(nd.state, nd.moves[i]));
						nd.kids[i] = kid;
					}
					nd = kid;
					path[d++] = nd;
				}
				double v;
				if (nd.terminal) {
					v = nd.termValue;
				} else if (!nd.expanded) {
					v = expand(nd);
				} else { // espanso senza mosse (o percorso troppo lungo)
					v = nd.state.getTurn() == Turn.WHITE ? -1.0 : 1.0;
				}
				for (int j = 0; j < d; j++) {
					path[j].n++;
					path[j].sum += v;
				}
				lastSims++;
				lastMaxDepth = Math.max(lastMaxDepth, d - 1);
			}
		}
		int bi = 0, bn = -1;
		double bq = -1e18, sign = root.state.getTurn() == Turn.WHITE ? 1 : -1;
		for (int i = 0; i < root.moves.length; i++) {
			Node kid = root.kids[i];
			int n = kid == null ? 0 : kid.n;
			double q = sign * (kid != null && kid.n > 0 ? kid.sum / kid.n : root.stat[i]);
			if (n > bn || (n == bn && q > bq)) {
				bn = n;
				bq = q;
				bi = i;
			}
		}
		lastRootValue = root.n > 0 ? root.sum / root.n : 0;
		return root.moves[bi];
	}
}
