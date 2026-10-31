package it.unibo.ai.didattica.competition.tablut.client;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;

/**
 * Verifica il comportamento del {@code main} REALE di {@link TablutNegamaxClient} senza configurazione esplicita e con opzioni
 * esplicite: lancia il client in un sottoprocesso (contro una porta senza server: il banner viene stampato PRIMA della connessione,
 * che poi fallisce e termina il processo) e legge il banner. Verifica: ordinamento di default {@code rec}; ordinamento esplicito
 * ({@code order=base}, {@code order=wins+counter}) rispettato; valore non valido ignorato con avviso; cache delle valutazioni attiva
 * per MLP e CNN (2^22 voci = 128 MB), disattivata con {@code SimpleEvaluator}, con {@code evcache=0} e con {@code algo=mcts}; riduzione
 * della cache in un heap piccolo. I pesi sono file temporanei di reti casuali: nessuna dipendenza dai file del repository.
 * Richiede che la porta 5800 (WHITE) di localhost non abbia un server in ascolto.
 */
public class ClientDefaultsCheck {

	static int failures = 0;

	static void check(boolean ok, String msg) {
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	/** Esegue il client e restituisce il suo output (stdout+stderr), o null se non termina entro 60 s. */
	static String run(List<String> jvmOpts, String... clientArgs) throws Exception {
		List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.addAll(jvmOpts);
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.client.TablutNegamaxClient");
		for (String a : clientArgs) {
			cmd.add(a);
		}
		ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.redirectErrorStream(true);
		Process p = pb.start();
		StringBuilder sb = new StringBuilder();
		try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), "UTF-8"))) {
			String line;
			while ((line = r.readLine()) != null) {
				sb.append(line).append('\n');
			}
		}
		if (!p.waitFor(60, TimeUnit.SECONDS)) {
			p.destroyForcibly();
			return null;
		}
		return sb.toString();
	}

	static String banner(String out) {
		for (String l : out.split("\n")) {
			if (l.startsWith("Negamax client") || l.startsWith("MCTS client")) {
				return l;
			}
		}
		return "";
	}

	static final List<String> HEAP2G = java.util.Arrays.asList("-Xmx2g");

	/** Il pezzo di banner "ordinamento: NOME (modo N)": N e' il valore numerico effettivamente passato alla ricerca. */
	static String orderLabel(String name) {
		return "ordinamento: " + name + " (modo " + OrderingSpec.parse(name) + ")";
	}

	public static void main(String[] args) throws Exception {
		File dir = java.nio.file.Files.createTempDirectory("clientdefaults").toFile();
		File mlp = new File(dir, "mlp.txt");
		File cnn = new File(dir, "cnn.txt");
		NeuralEvaluator.randomInit(5).save(mlp.getPath());
		CnnEvaluator.randomInit(CnnEvaluator.defaultConfig(TablutFeatures.G_DEFAULT), TablutFeatures.G_DEFAULT, 6).save(cnn.getPath());
		final String M = mlp.getPath(), C = cnn.getPath();
		try {
			// 1) nessuna configurazione esplicita, senza pesi: SimpleEvaluator, ordinamento rec, nessuna cache
			String o = run(HEAP2G, "WHITE", "60", "127.0.0.1");
			check(o != null, "senza opzioni: il processo non termina (c'e' un server su 5800?)");
			if (o != null) {
				String b = banner(o);
				check(b.contains(orderLabel("rec")), "senza opzioni: ordinamento di default rec. Banner: " + b);
				check(b.contains("cache valutazioni: disattivata"), "senza pesi (SimpleEvaluator): nessuna cache. Banner: " + b);
				check(o.contains("Nessun file di pesi indicato: uso SimpleEvaluator"), "senza pesi: SimpleEvaluator");
			}
			// 2) MLP e CNN: rec + cache di default (2^22 voci = 128 MB)
			for (String w : new String[] { M, C }) {
				o = run(HEAP2G, "WHITE", "60", "127.0.0.1", w);
				check(o != null, "pesi " + (w.equals(M) ? "MLP" : "CNN") + ": il processo non termina");
				if (o != null) {
					String b = banner(o);
					String name = w.equals(M) ? "MLP" : "CNN";
					check(b.contains(orderLabel("rec")), name + ": ordinamento di default rec. Banner: " + b);
					check(b.contains("cache valutazioni: 2^22 voci (128 MB)"), name + ": cache di default 2^22 voci (128 MB). Banner: " + b);
				}
			}
			// 3) ordinamento esplicito diverso dal default
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "order=base");
			check(o != null && banner(o).contains(orderLabel("base")), "order=base viene rispettato. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "order=wins+counter");
			check(o != null && banner(o).contains(orderLabel("wins+counter")), "order=wins+counter viene rispettato. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "order=rec");
			check(o != null && banner(o).contains(orderLabel("rec")), "order=rec esplicito. Banner: " + (o == null ? "-" : banner(o)));
			// 4) valore non valido: avviso e si resta sul default
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "order=zzz");
			check(o != null && o.contains("ATTENZIONE: 'order=zzz' non valido") && banner(o).contains(orderLabel("rec")), "order non valido: avviso e default rec. Banner: " + (o == null ? "-" : banner(o)));
			// 5) cache: disattivabile, dimensione scelta, mai con Simple ne' con MCTS
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "evcache=0");
			check(o != null && banner(o).contains("cache valutazioni: disattivata"), "evcache=0 disattiva la cache. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "evcache=20");
			check(o != null && banner(o).contains("cache valutazioni: 2^20 voci (32 MB)"), "evcache=20 -> 2^20 voci (32 MB). Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", "evcache=22");
			check(o != null && banner(o).contains("cache valutazioni: disattivata"), "con SimpleEvaluator la cache resta disattivata anche con evcache=22. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "algo=mcts");
			check(o != null && banner(o).startsWith("MCTS client") && !o.contains("cache valutazioni"), "con algo=mcts il client e' MCTS e non avvolge la cache. Banner: " + (o == null ? "-" : banner(o)));
			// 5b) chiave esatta: spenta di default; accesa, TT e cache lo dichiarano nel banner (dedotto dagli oggetti costruiti)
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M);
			check(o != null && banner(o).contains("slot singolo, hash)") && !banner(o).contains("chiave esatta"), "senza exactkey: TT a hash e cache che ricodifica. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "exactkey=false");
			check(o != null && banner(o).contains("slot singolo, hash)") && !banner(o).contains("chiave esatta"), "exactkey=false esplicito. Banner: " + (o == null ? "-" : banner(o)));
			for (String w : new String[] { M, C }) {
				o = run(HEAP2G, "WHITE", "60", "127.0.0.1", w, "exactkey=true");
				String b = o == null ? "-" : banner(o);
				check(o != null && b.contains("TT: 2^24 voci (640 MB, slot singolo, chiave esatta)") && b.contains("cache valutazioni: 2^22 voci (128 MB, chiave esatta dallo stato)") && b.contains(orderLabel("rec")),
					"exactkey=true (" + (w.equals(M) ? "MLP" : "CNN") + "): TT esatta (2^24 voci = 640 MB in 2 GiB di heap) e cache dallo stato. Banner: " + b);
			}
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "exactkey=true", "tt2=true");
			check(o != null && banner(o).contains("due slot, chiave esatta"), "exactkey=true con tt2=true. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", "exactkey=true");
			check(o != null && banner(o).contains("chiave esatta)") && banner(o).contains("cache valutazioni: disattivata"), "exactkey=true con SimpleEvaluator: TT esatta, cache sempre disattivata. Banner: " + (o == null ? "-" : banner(o)));
			o = run(HEAP2G, "WHITE", "60", "127.0.0.1", M, "algo=mcts", "exactkey=true");
			check(o != null && o.contains("'exactkey=' ignorato") && banner(o).startsWith("MCTS client"), "algo=mcts: exactkey ignorato con avviso. Output: " + (o == null ? "-" : banner(o)));
			// 6) heap piccolo: la cache scende (al piu' un ottavo dell'heap)
			o = run(java.util.Arrays.asList("-Xmx256m"), "WHITE", "60", "127.0.0.1", M);
			check(o != null && o.contains("supera un ottavo dell'heap") && !banner(o).contains("cache valutazioni: 2^22"), "heap di 256 MB: la cache di 128 MB viene ridotta. Output: " + (o == null ? "-" : banner(o)));
		} finally {
			mlp.delete();
			cnn.delete();
			dir.delete();
		}
		if (failures == 0) {
			System.out.println("=== CLIENT DEFAULTS CHECK OK ===");
		} else {
			System.out.println("=== FALLITO: " + failures + " ===");
			System.exit(1);
		}
	}
}
