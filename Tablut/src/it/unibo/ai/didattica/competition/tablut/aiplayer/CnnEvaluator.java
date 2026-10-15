package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.domain.State;

/**
 * {@link Evaluator} basato su {@link ConvNet}. Sostituisce {@link NeuralEvaluator} (o lo affianca, per confronto) in
 * {@link Negamax} senza modificare una riga della ricerca.
 * <p>
 * <b>Ibrido o puro</b>: se la rete ha {@code nFeat > 0} le feature a mano di {@link TablutFeatures} (gruppi
 * {@link #featureGroups}) sono calcolate e passate alla rete insieme alla scacchiera; se {@code nFeat == 0} (CNN puro) la
 * rete vede solo le pedine e non si calcola alcuna feature.
 * <p>
 * <b>Ruolo</b>: come in {@link NeuralEvaluator}, la head e' scelta da chi deve muovere nella posizione valutata
 * ({@link NeuralEvaluator#roleOf}), cosi' lo stesso evaluator serve client Bianco, client Nero e self-play.
 * <p>
 * <b>Thread-safety</b>: la rete non e' modificata durante l'inferenza e lo stato temporaneo (workspace, feature) e'
 * per-thread ({@link ThreadLocal}); una istanza puo' essere condivisa dai thread di ricerca (Lazy SMP). I trainer modificano i
 * pesi solo dal thread principale, mentre i worker usano copie ({@link ConvNet#copy()}).
 * <p>
 * L'uscita della rete e' in (-1,1); {@link #outputScale} la porta sulla scala di {@link SimpleEvaluator} cosi' le costanti di
 * {@link Negamax} restano valide.
 */
public class CnnEvaluator implements Evaluator {

	public final ConvNet net;
	/** Gruppi di feature a mano (si veda {@link TablutFeatures}); ignorati se la rete e' pura ({@code nFeat == 0}). */
	public final int featureGroups;
	private final int[] cols;

	/** Scala l'uscita (-1,1) in un punteggio "stile SimpleEvaluator". */
	public double outputScale = 1000.0;

	private static final class Ctx {
		final ConvNet.Workspace ws;
		final TablutFeatures.Workspace fws = TablutFeatures.newWorkspace();
		final double[] full = new double[TablutFeatures.NUM_FEATURES];
		final float[] feats;

		Ctx(ConvNet net, int nFeat) {
			ws = net.newWorkspace();
			feats = new float[nFeat];
		}
	}

	private final ThreadLocal<Ctx> ctx;

	public CnnEvaluator(ConvNet net, int featureGroups) {
		this.net = net;
		this.featureGroups = featureGroups;
		final int nFeat = net.config().nFeat;
		this.cols = nFeat > 0 ? TablutFeatures.columnsFor(featureGroups) : new int[0];
		if (nFeat > 0 && cols.length != nFeat) {
			throw new IllegalArgumentException("la rete si aspetta " + nFeat + " feature, i gruppi " + featureGroups + " ne producono "
					+ cols.length);
		}
		this.ctx = new ThreadLocal<Ctx>() {
			@Override
			protected Ctx initialValue() {
				return new Ctx(net, nFeat);
			}
		};
	}

	/** True se la rete usa le feature a mano. */
	public boolean isHybrid() {
		return net.config().nFeat > 0;
	}

	/**
	 * Configurazione di default: ibrido leggero. Un livello conv da 4 canali (sparso) + riduzione a 2 canali + denso 32 + head 8
	 * (~7500 parametri, ~11 us per valutazione: ~3 volte l'MLP). Misurato su 165.000 posizioni: stessa qualita' di previsione
	 * dell'analogo da 8 canali (MSE held-out 0,636 vs 0,634) al 70% del costo; un secondo livello conv NON migliora
	 * (si veda README_CNN.md). E' un punto di partenza economico, non un ottimo dimostrato: i costi si leggono con {@code CnnBench}.
	 */
	public static ConvNet.Config defaultConfig(int featureGroups) {
		return new ConvNet.Config(new int[] { 4 }, 2, TablutFeatures.columnsFor(featureGroups).length, new int[] { 32 },
				new int[] { 8 });
	}

	public static CnnEvaluator randomInit(ConvNet.Config cfg, int featureGroups, long seed) {
		return new CnnEvaluator(new ConvNet(cfg, seed), featureGroups);
	}

	/**
	 * Carica i pesi. Rifiuta (IOException con messaggio chiaro) i file di una versione delle feature diversa dall'attuale (solo per
	 * le reti ibride: il CNN puro non dipende dalle feature): il client li intercetta e ripiega su {@link SimpleEvaluator}.
	 */
	public static CnnEvaluator fromFile(String path) throws IOException {
		ConvNet.Loaded l = ConvNet.load(path);
		int nFeat = l.net.config().nFeat;
		if (nFeat > 0) {
			if (l.featureVersion != TablutFeatures.VERSION) {
				throw new IOException("pesi allenati con feature v" + l.featureVersion + ", il codice usa la v" + TablutFeatures.VERSION
						+ ": vanno riallenati (" + path + ")");
			}
			int expected = TablutFeatures.columnsFor(l.featureGroups).length;
			if (expected != nFeat) {
				throw new IOException("pesi con " + nFeat + " feature, ma i gruppi " + l.featureGroups + " ne richiedono " + expected
						+ ": file incoerente (" + path + ")");
			}
		}
		return new CnnEvaluator(l.net, l.featureGroups);
	}

	public void save(String path) throws IOException {
		if (isHybrid()) {
			net.save(path, TablutFeatures.VERSION, featureGroups);
		} else {
			net.save(path, 0, 0);
		}
	}

	/** Calcola in {@code out} (lunghezza = {@code nFeat}) le feature a mano compattate, nell'ordine atteso dalla rete. */
	public void extractFeatures(State state, TablutFeatures.Workspace fws, double[] fullScratch, float[] out) {
		TablutFeatures.extract(state, fws, fullScratch, featureGroups);
		for (int i = 0; i < cols.length; i++) {
			out[i] = (float) fullScratch[cols[i]];
		}
	}

	@Override
	public double evaluate(State state) {
		Ctx c = ctx.get();
		c.ws.setBoard(state);
		if (cols.length > 0) {
			extractFeatures(state, c.fws, c.full, c.feats);
		}
		return net.forward(c.ws, c.feats, NeuralEvaluator.roleOf(state)) * outputScale;
	}
}
