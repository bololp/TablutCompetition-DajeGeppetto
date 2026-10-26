package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * {@link Evaluator} basato su {@link MLP} + {@link TablutFeatures}. Sostituisce
 * (o affianca, per confronto) {@link SimpleEvaluator} in {@link Negamax} senza
 * modificare una riga della ricerca.
 * <p>
 * <b>Ruolo</b>: la head della rete e' scelta da chi deve muovere nella posizione
 * valutata ({@code state.getTurn()}), si veda {@link MLP}. E' una funzione della
 * sola posizione, quindi lo stesso evaluator serve il client Bianco e il client Nero.
 * <p>
 * <b>Tier di feature</b>: {@link #featureGroups} sceglie quali gruppi di
 * {@link TablutFeatures} calcolare. Le colonne dei gruppi esclusi non vengono
 * ne' calcolate ne' passate alla rete (l'ingresso e' compattato con
 * {@link TablutFeatures#columnsFor}): un tier leggero e' piu' economico sia nelle
 * feature sia nel primo strato. La maschera e' salvata insieme ai pesi.
 * <p>
 * <b>Thread-safety</b>: la rete non viene mai modificata durante l'inferenza e
 * lo stato temporaneo (feature, attivazioni) e' per-thread ({@link ThreadLocal}):
 * una stessa istanza puo' essere usata da piu' thread, per esempio da una futura
 * ricerca parallela. (Il TD del trainer modifica i pesi: lo fa solo dal thread
 * principale, mentre i worker usano copie congelate, si veda
 * {@code SelfPlayTrainer}.)
 * <p>
 * L'uscita della rete e' in (-1,1) (+1 = vittoria bianca certa, -1 = nera, 0 =
 * equilibrio); {@link #outputScale} la porta sulla scala di {@link SimpleEvaluator}
 * cosi' le costanti di {@link Negamax} (es. WIN_SCORE) restano valide.
 */
public class NeuralEvaluator implements Evaluator {

	public final MLP net;
	/** Gruppi di feature calcolati (si veda {@link TablutFeatures}); salvati insieme ai pesi. */
	public final int featureGroups;
	private final int[] cols;

	/** Scala l'uscita (-1,1) in un punteggio "stile SimpleEvaluator". */
	public double outputScale = 1000.0;

	private static final class Ctx {
		final TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
		final double[] full = new double[TablutFeatures.NUM_FEATURES];
		final double[] x;
		final MLP.Scratch scratch;

		Ctx(MLP net, int nCols) {
			x = new double[nCols];
			scratch = net.newScratch();
		}
	}

	private final ThreadLocal<Ctx> ctx;

	public NeuralEvaluator(MLP net) {
		this(net, TablutFeatures.G_DEFAULT);
	}

	public NeuralEvaluator(MLP net, int featureGroups) {
		this.cols = TablutFeatures.columnsFor(featureGroups);
		if (net.config().nIn != cols.length) {
			throw new IllegalArgumentException("la rete si aspetta " + net.config().nIn + " feature, i gruppi " + featureGroups
					+ " ne producono " + cols.length);
		}
		this.net = net;
		this.featureGroups = featureGroups;
		this.ctx = ThreadLocal.withInitial(() -> new Ctx(net, cols.length));
	}

	/**
	 * Architettura di default: tronco condiviso (24) + 2 head (8 + uscita), attivazione leaky-ReLU.
	 * Scelta in base agli esperimenti di MLP_REDESIGN.md: qualita' sul plateau (piu' neuroni non migliorano
	 * l'errore di previsione) a ~0,5 us per valutazione, contro i ~1,4 us con tanh.
	 */
	public static MLP.Config defaultConfig(int featureGroups) {
		return MLP.Config.sharedTrunk(TablutFeatures.columnsFor(featureGroups).length, new int[] { 24 }, new int[] { 8 },
				MLP.Act.LEAKY);
	}

	public static MLP.Config defaultConfig() {
		return defaultConfig(TablutFeatures.G_DEFAULT);
	}

	public static NeuralEvaluator randomInit(long seed) {
		return new NeuralEvaluator(new MLP(defaultConfig(), seed));
	}

	/**
	 * Carica i pesi da file. Rifiuta (IOException, con messaggio chiaro) i file
	 * del formato precedente e quelli allenati con un'altra versione delle
	 * feature: il client li intercetta e ripiega su {@link SimpleEvaluator}.
	 */
	public static NeuralEvaluator fromFile(String path) throws IOException {
		MLP.Loaded l = MLP.load(path);
		if (l.featureVersion != TablutFeatures.VERSION) {
			throw new IOException("pesi allenati con feature v" + l.featureVersion + ", il codice usa la v"
					+ TablutFeatures.VERSION + ": convertirli con WeightConverter (esatto) o riallenarli (" + path + ")");
		}
		int expected = TablutFeatures.columnsFor(l.featureGroups).length;
		if (l.net.config().nIn != expected) {
			throw new IOException("pesi con " + l.net.config().nIn + " ingressi, ma i gruppi di feature " + l.featureGroups
					+ " ne richiedono " + expected + ": file incoerente (" + path + ")");
		}
		return new NeuralEvaluator(l.net, l.featureGroups);
	}

	public void save(String path) throws IOException {
		net.save(path, TablutFeatures.VERSION, featureGroups);
	}

	/** Head da usare per una posizione: 0 = Bianco di turno, 1 = Nero di turno. */
	public static int roleOf(State state) {
		return state.getTurn() == Turn.BLACK ? 1 : 0;
	}

	/** Head da usare per una posizione {@link FastState}: 0 = Bianco di turno, 1 = Nero di turno. */
	public static int roleOf(FastState state) {
		return state.turn == BoardConstants.BLACK_TURN ? 1 : 0;
	}

	/** Estrae in {@code x} (lunghezza = numero di ingressi della rete) le feature della posizione, nell'ordine atteso dalla rete. */
	public void extractInputs(FastState state, TablutFeatures.Workspace ws, double[] fullScratch, double[] x) {
		TablutFeatures.extract(state, ws, fullScratch, featureGroups);
		for (int i = 0; i < cols.length; i++) {
			x[i] = fullScratch[cols[i]];
		}
	}

	/** Come {@link #extractInputs(FastState, TablutFeatures.Workspace, double[], double[])} per uno {@link State} del dominio (converte: alloca). */
	public void extractInputs(State state, TablutFeatures.Workspace ws, double[] fullScratch, double[] x) {
		extractInputs(FastState.fromState(state), ws, fullScratch, x);
	}

	/** Numero di ingressi della rete (= numero di feature attive). */
	public int numInputs() {
		return cols.length;
	}

	@Override
	public double evaluate(FastState state) {
		Ctx c = ctx.get();
		extractInputs(state, c.ws, c.full, c.x);
		return net.forward(c.x, roleOf(state), c.scratch) * outputScale;
	}
}
