package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Valutatore NNUE (si veda {@link NnueNetwork} per feature, rete e quantizzazione).
 * <p>
 * Due percorsi che danno lo STESSO numero, bit per bit, per la stessa posizione:
 * <ul>
 * <li>{@link #evaluate(FastState)}: valutazione COMPLETA, stateless e thread-safe (ricostruisce l'accumulatore dalla scacchiera). E' il
 * percorso di inizializzazione, di test e di chi non ha un percorso di ricerca (MCTS, trainer, arene).</li>
 * <li>{@link #newSession(int)}: sessione INCREMENTALE per un worker di {@link Negamax} ({@link NnueSession}).</li>
 * </ul>
 * Convenzioni identiche a {@link CnnEvaluator}/{@link NeuralEvaluator}: valore dal punto di vista del Bianco (positivo = meglio per i
 * difensori), solo per posizioni NON terminali (la ricerca decide da sola le posizioni terminali; su una posizione terminale questa
 * classe valuta comunque la scacchiera cosi' com'e'), il lato al tratto sceglie la testa della rete. I pesi sono immutabili; l'unico
 * stato per-thread di {@code evaluate(FastState)} e' un buffer di lavoro in un {@link ThreadLocal}.
 * <p>
 * Non va avvolto in {@link CachingEvaluator}: il costo di una valutazione e' gia' inferiore a quello di una consultazione della
 * cache, e l'involucro nasconderebbe il percorso incrementale a {@link Negamax} ({@code CachingEvaluator.isCacheable} lo esclude).
 */
public final class NnueEvaluator implements IncrementalEvaluator {

	public final NnueNetwork net;

	/** Porta il valore in (-1,1) sulla scala di {@link SimpleEvaluator}; come negli altri valutatori neurali. */
	public double outputScale = 1000.0;

	private static final class Ctx {
		final int[] acc;
		final int[] act;

		Ctx(int hidden, int work) {
			acc = new int[hidden];
			act = new int[work];
		}
	}

	private final ThreadLocal<Ctx> ctx;

	public NnueEvaluator(NnueNetwork net) {
		if (net == null) {
			throw new IllegalArgumentException("rete NNUE nulla");
		}
		this.net = net;
		final int hidden = net.hidden();
		final int work = net.scratchSize();
		this.ctx = ThreadLocal.withInitial(() -> new Ctx(hidden, work));
	}

	/**
	 * @throws IOException se il file manca o non e' un file di pesi NNUE valido (si veda {@link NnueNetwork#load})
	 */
	public static NnueEvaluator fromFile(String path) throws IOException {
		return new NnueEvaluator(NnueNetwork.load(path));
	}

	public void save(String path) throws IOException {
		net.save(path);
	}

	@Override
	public double evaluate(FastState s) {
		final Ctx c = ctx.get();
		net.refresh(s.board, c.acc);
		return NnueNetwork.valueOf(net.logit(c.acc, c.act, s.turn == BoardConstants.BLACK_TURN ? 1 : 0)) * outputScale;
	}

	@Override
	public Session newSession(int capacityPlies) {
		return new NnueSession(this, capacityPlies);
	}

	public String describe() {
		return net.describe();
	}
}
