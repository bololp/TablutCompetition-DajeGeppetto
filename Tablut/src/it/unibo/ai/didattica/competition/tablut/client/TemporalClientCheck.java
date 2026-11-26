package it.unibo.ai.didattica.competition.tablut.client;

import java.io.File;
import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalNnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalNnueNetwork;

/** Selezione di {@code eval=tnnue} e riconoscimento automatico del file {@code TABLUT_TNNUE} nel client. Uso: {@code TemporalClientCheck}. */
public class TemporalClientCheck {

	private static int failures = 0, checks = 0;

	private static void check(boolean c, String m) {
		checks++;
		if (!c) {
			failures++;
			System.out.println("FAIL - " + m);
		}
	}

	private static boolean throwsIo(String backend, String file) {
		try {
			TablutNegamaxClient.selectEvaluator(backend, file);
			return false;
		} catch (IOException e) {
			return true;
		}
	}

	public static void main(String[] args) throws Exception {
		final File t = File.createTempFile("tcl", ".tnnue"), n = File.createTempFile("tcl", ".nnue");
		try {
			TemporalNnueNetwork.fromBase(NnueNetwork.random(8, 4, 1), 3).save(t.getPath());
			NnueNetwork.random(8, 4, 2).save(n.getPath());
			check(TablutNegamaxClient.isKnownEvalBackend("tnnue"), "tnnue e' un backend noto");
			check(TablutNegamaxClient.selectEvaluator("tnnue", t.getPath()) instanceof TemporalNnueEvaluator, "eval=tnnue carica la rete temporale");
			check(TablutNegamaxClient.selectEvaluator("auto", t.getPath()) instanceof TemporalNnueEvaluator, "eval=auto riconosce il file temporale");
			check(EvaluatorLoader.load(t.getPath()) instanceof TemporalNnueEvaluator && EvaluatorLoader.load(n.getPath()).getClass().getSimpleName().equals("NnueEvaluator"), "il loader distingue i due formati");
			check(EvaluatorLoader.describe(EvaluatorLoader.load(t.getPath())).startsWith("TemporalNnueEvaluator"), "describe");
			check(throwsIo("nnue", t.getPath()), "eval=nnue con un file temporale e' un errore");
			check(throwsIo("tnnue", n.getPath()), "eval=tnnue con una NNUE di sola posizione e' un errore");
			check(throwsIo("tnnue", null), "eval=tnnue senza pesi e' un errore");
			final Evaluator e = TablutNegamaxClient.selectEvaluator("nnue", n.getPath());
			check(!(e instanceof TemporalNnueEvaluator), "eval=nnue invariato");
		} finally {
			t.delete();
			n.delete();
		}
		System.out.println(failures == 0 ? "=== TEMPORAL CLIENT CHECK OK: " + checks + " verifiche ===" : "=== FALLITO: " + failures + " ===");
		System.exit(failures == 0 ? 0 : 1);
	}
}
