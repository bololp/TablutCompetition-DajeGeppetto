package it.unibo.ai.didattica.competition.tablut.client;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CachingEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;

/**
 * Selezione del backend di valutazione del client ({@code eval=auto|simple|mlp|cnn|nnue}).
 * <ul>
 * <li>{@link TablutNegamaxClient#selectEvaluator}: {@code auto} riconosce NNUE, CNN e MLP dall'intestazione e, come prima, ripiega su
 * {@link SimpleEvaluator} senza lanciare se i pesi mancano o non si caricano; le modalita' ESPLICITE sono severe (pesi mancanti, file non
 * valido o di un altro tipo = IOException con il nome del backend), cosi' non si gioca mai con un valutatore diverso da quello chiesto;</li>
 * <li>l'NNUE non e' mai avvolto in {@link CachingEvaluator} (la cache nasconderebbe a Negamax il percorso incrementale);</li>
 * <li>il {@code main} reale, lanciato in un sottoprocesso come fa {@link ClientDefaultsCheck}: scelta esplicita sbagliata = errore e uscita
 * PRIMA di connettersi; scelta corretta = banner con il valutatore e senza cache.</li>
 * </ul>
 * I pesi sono file temporanei di reti casuali. Richiede che la porta 5800 di localhost non abbia un server in ascolto.
 */
public class NnueClientCheck {

	private static int failures = 0;
	private static int checks = 0;

	private static void check(boolean ok, String msg) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	private static File tmp(String suffix) throws IOException {
		final File f = File.createTempFile("nnueclient_", suffix);
		f.deleteOnExit();
		return f;
	}

	/** Esito di {@code selectEvaluator}: l'evaluator, oppure il messaggio dell'IOException. */
	private static final class Res {
		Evaluator ev;
		String error;
	}

	private static Res select(String backend, String weights) {
		final Res r = new Res();
		try {
			r.ev = TablutNegamaxClient.selectEvaluator(backend, weights);
		} catch (IOException e) {
			r.error = String.valueOf(e.getMessage());
		}
		return r;
	}

	public static void main(String[] args) throws Exception {
		final File nnue = tmp(".nnue");
		NnueNetwork.random(16, 4, 3).save(nnue.getPath());
		final File cnn = tmp(".cnn");
		CnnEvaluator.randomInit(CnnEvaluator.defaultConfig(TablutFeatures.G_DEFAULT), TablutFeatures.G_DEFAULT, 5).save(cnn.getPath());
		final File mlp = tmp(".mlp");
		NeuralEvaluator.randomInit(3).save(mlp.getPath());
		final File junk = tmp(".junk");
		try (FileWriter w = new FileWriter(junk)) {
			w.write("TABLUT_NNUE v1\nfeatures psq3x81\nhidden 16\n");
		}
		final String missing = "/nonexistent_dir_xyz/none.nnue";

		// ---- nomi
		for (String ok : new String[] { "auto", "simple", "mlp", "cnn", "nnue" }) {
			check(TablutNegamaxClient.isKnownEvalBackend(ok), "'" + ok + "' deve essere un backend noto");
		}
		for (String bad : new String[] { "", "nnu", "nnue2", "NNUE", "foo", "cnn ", "auto,nnue" }) {
			check(!TablutNegamaxClient.isKnownEvalBackend(bad), "'" + bad + "' non deve essere un backend noto (il main porta in minuscolo prima di chiedere)");
		}

		// ---- auto: riconoscimento dall'intestazione e ripiego avvisato come prima
		Res r = select("auto", null);
		check(r.ev instanceof SimpleEvaluator && r.error == null, "auto senza pesi: SimpleEvaluator");
		r = select("auto", nnue.getPath());
		check(r.ev instanceof NnueEvaluator, "auto con un file NNUE: NnueEvaluator");
		r = select("auto", cnn.getPath());
		check(r.ev instanceof CnnEvaluator, "auto con un file CNN: CnnEvaluator (la CNN resta selezionabile)");
		r = select("auto", mlp.getPath());
		check(r.ev instanceof NeuralEvaluator, "auto con un file MLP: NeuralEvaluator");
		r = select("auto", junk.getPath());
		check(r.ev instanceof SimpleEvaluator && r.error == null, "auto con un file NNUE corrotto: ripiego su SimpleEvaluator senza lanciare (comportamento storico)");
		r = select("auto", missing);
		check(r.ev instanceof SimpleEvaluator && r.error == null, "auto con un file inesistente: ripiego su SimpleEvaluator senza lanciare (comportamento storico)");

		// ---- simple
		r = select("simple", nnue.getPath());
		check(r.ev instanceof SimpleEvaluator, "simple ignora i pesi");
		r = select("simple", null);
		check(r.ev instanceof SimpleEvaluator, "simple senza pesi");

		// ---- esplicito: la scelta giusta funziona
		r = select("nnue", nnue.getPath());
		check(r.ev instanceof NnueEvaluator && r.error == null, "nnue con un file NNUE");
		r = select("cnn", cnn.getPath());
		check(r.ev instanceof CnnEvaluator && r.error == null, "cnn con un file CNN");
		r = select("mlp", mlp.getPath());
		check(r.ev instanceof NeuralEvaluator && r.error == null, "mlp con un file MLP");

		// ---- esplicito: ogni errore e' un errore, mai un ripiego silenzioso
		for (String be : new String[] { "nnue", "cnn", "mlp" }) {
			r = select(be, null);
			check(r.ev == null && r.error != null && r.error.contains("eval=" + be) && r.error.contains("richiede"), be + " senza pesi: errore esplicito. Messaggio: " + r.error);
			r = select(be, missing);
			check(r.ev == null && r.error != null && r.error.contains("eval=" + be), be + " con file inesistente: errore esplicito. Messaggio: " + r.error);
		}
		r = select("nnue", junk.getPath());
		check(r.ev == null && r.error != null && r.error.contains("eval=nnue") && r.error.contains("Atteso"),
				"nnue con un file corrotto: errore esplicito con la causa. Messaggio: " + r.error);
		r = select("nnue", cnn.getPath());
		check(r.ev == null && r.error != null && r.error.contains("eval=nnue"), "nnue con un file CNN: rifiutato. Messaggio: " + r.error);
		r = select("nnue", mlp.getPath());
		check(r.ev == null && r.error != null && r.error.contains("eval=nnue"), "nnue con un file MLP: rifiutato. Messaggio: " + r.error);
		r = select("cnn", nnue.getPath());
		check(r.ev == null && r.error != null && r.error.contains("altro tipo"), "cnn con un file NNUE: rifiutato come altro tipo. Messaggio: " + r.error);
		r = select("mlp", nnue.getPath());
		check(r.ev == null && r.error != null && r.error.contains("altro tipo"), "mlp con un file NNUE: rifiutato come altro tipo. Messaggio: " + r.error);
		r = select("mlp", cnn.getPath());
		check(r.ev == null && r.error != null && r.error.contains("altro tipo"), "mlp con un file CNN: rifiutato come altro tipo. Messaggio: " + r.error);

		// ---- la cache non avvolge mai l'NNUE (avvolgerebbe via il percorso incrementale), e continua ad avvolgere CNN e MLP come prima
		final Evaluator nn = select("nnue", nnue.getPath()).ev;
		check(!CachingEvaluator.isCacheable(nn), "l'NNUE non e' cacheable");
		check(TablutNegamaxClient.withEvalCache("negamax", nn, 22, 1L << 33) == nn, "withEvalCache non deve avvolgere l'NNUE");
		check(TablutNegamaxClient.withEvalCache("negamax", select("cnn", cnn.getPath()).ev, 22, 1L << 33) instanceof CachingEvaluator, "la CNN resta avvolta dalla cache (nessuna regressione)");
		check(TablutNegamaxClient.withEvalCache("negamax", select("mlp", mlp.getPath()).ev, 22, 1L << 33) instanceof CachingEvaluator, "l'MLP resta avvolto dalla cache (nessuna regressione)");

		// ---- il main reale, in un sottoprocesso
		final List<String> heap = ClientDefaultsCheck.HEAP2G;
		String out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=nnue", nnue.getPath(), "threads=1");
		check(out != null && out.contains("Valutazione: NnueEvaluator (NNUE psq3x81 (243 -> 16 -> 4 -> 1"), "main eval=nnue: banner del valutatore. Output: " + out);
		check(out != null && ClientDefaultsCheck.banner(out).contains("cache valutazioni: disattivata"), "main eval=nnue: nessuna cache. Banner: " + (out == null ? "-" : ClientDefaultsCheck.banner(out)));
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", nnue.getPath(), "threads=1");
		check(out != null && out.contains("Valutazione: NnueEvaluator") && ClientDefaultsCheck.banner(out).contains("cache valutazioni: disattivata"),
				"main senza eval= (auto) con un file NNUE: usa l'NNUE. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "EVAL=NNUE", nnue.getPath(), "threads=1");
		check(out != null && out.contains("Valutazione: NnueEvaluator"), "main EVAL=NNUE (maiuscolo): il valore e' normalizzato come gli altri. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=nnue", missing, "threads=1");
		check(out != null && out.contains("ERRORE: eval=nnue") && ClientDefaultsCheck.banner(out).isEmpty(), "main eval=nnue con file inesistente: errore e uscita prima del banner. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=nnue", "threads=1");
		check(out != null && out.contains("ERRORE: eval=nnue") && out.contains("richiede") && ClientDefaultsCheck.banner(out).isEmpty(), "main eval=nnue senza pesi: errore. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=nnue", cnn.getPath(), "threads=1");
		check(out != null && out.contains("ERRORE: eval=nnue") && ClientDefaultsCheck.banner(out).isEmpty(), "main eval=nnue con un file CNN: errore. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=nnu", nnue.getPath(), "threads=1");
		check(out != null && out.contains("ERRORE: eval='nnu' sconosciuto") && ClientDefaultsCheck.banner(out).isEmpty(), "main eval= sconosciuto: errore, non si ignora. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=simple", nnue.getPath(), "threads=1");
		check(out != null && out.contains("ATTENZIONE: eval=simple") && !out.contains("Valutazione: NnueEvaluator"), "main eval=simple con pesi: pesi ignorati con avviso. Output: " + out);
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", "eval=cnn", cnn.getPath(), "threads=1");
		check(out != null && out.contains("Valutazione: CnnEvaluator") && ClientDefaultsCheck.banner(out).contains("cache valutazioni: 2^22"), "main eval=cnn: la CNN resta selezionabile e cacheata. Output: " + out);
		// retrocompatibilita': senza eval= e con file non valido il main ripiega con un avviso, come prima
		out = ClientDefaultsCheck.run(heap, "WHITE", "5", "localhost", junk.getPath(), "threads=1");
		check(out != null && out.contains("ATTENZIONE: impossibile caricare i pesi") && out.contains("Uso SimpleEvaluator") && !ClientDefaultsCheck.banner(out).isEmpty(),
				"main auto con un file non valido: ripiego avvisato (comportamento storico). Output: " + out);

		System.out.println(failures == 0 ? "=== NNUE CLIENT CHECK OK: " + checks + " verifiche ===" : "=== NNUE CLIENT CHECK: " + failures + " FALLIMENTI su " + checks + " ===");
		if (failures != 0) {
			System.exit(1);
		}
	}
}
