package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;

/**
 * Carica un {@link Evaluator} da un file di pesi riconoscendone il tipo dall'intestazione: {@code TABLUT_NNUE} -> {@link NnueEvaluator},
 * {@code TABLUT_CNN} -> {@link CnnEvaluator}, altrimenti {@link NeuralEvaluator} (MLP). Cosi' client e arena accettano tutti i formati
 * con lo stesso argomento.
 */
public final class EvaluatorLoader {

	private EvaluatorLoader() {
	}

	/** @throws IOException file illeggibile, formato non valido, o allenato con un'altra versione delle feature */
	public static Evaluator load(String path) throws IOException {
		if (NnueNetwork.isNnueFile(path)) {
			return NnueEvaluator.fromFile(path);
		}
		if (ConvNet.isCnnFile(path)) {
			return CnnEvaluator.fromFile(path);
		}
		return NeuralEvaluator.fromFile(path);
	}

	/** Nome breve del tipo di evaluator, per i messaggi di avvio. */
	public static String describe(Evaluator e) {
		if (e instanceof CnnEvaluator) {
			CnnEvaluator c = (CnnEvaluator) e;
			return "CnnEvaluator (" + c.net.config() + ", " + c.net.numParams() + " parametri)";
		}
		if (e instanceof NnueEvaluator) {
			return "NnueEvaluator (" + ((NnueEvaluator) e).describe() + ")";
		}
		return e.getClass().getSimpleName();
	}
}
