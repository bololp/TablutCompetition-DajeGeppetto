package it.unibo.ai.didattica.competition.tablut.client;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.training.GaGenome;

/**
 * Client di gioco del giocatore "genetico". NON contiene ne' esegue alcun algoritmo genetico: la GA ({@code training.GaTrainer}) gira OFFLINE,
 * molto prima del torneo, e produce un genoma ({@link GaGenome}) che qui viene soltanto APPLICATO una volta, all'avvio, alla rete di base
 * ({@code pesi_nnue/trained_v6.nnue} se non indicata). Poi il gioco e' ESATTAMENTE quello di {@link TablutNegamaxClient}: stesso protocollo, stessa ricerca
 * ({@code Negamax}: iterative deepening, TT, ordinamento, Lazy SMP), stessa gestione del tempo (budget = timeout - margine di sicurezza) e nessun
 * lavoro durante il turno avversario (il client resta bloccato nella lettura del socket).
 *
 * <h3>Come</h3>
 * L'eseguibile costruisce la rete evoluta, la scrive in un file temporaneo (cancellato all'uscita) e richiama {@link TablutNegamaxClient#main} con quel
 * file come pesi e {@code eval=nnue}: nessun parametro del riferimento viene duplicato, quindi tutte le sue opzioni ({@code threads=}, {@code tt=}, {@code margin=}...)
 * funzionano allo stesso modo e il confronto con il riferimento e' per costruzione a parita' di configurazione. Con il genoma neutro la rete e' identica
 * bit per bit alla base (e il giocatore coincide col riferimento).
 *
 * <h3>Uso</h3>
 * {@code TablutGeneticClient <WHITE|BLACK> [timeout=60] [ip=localhost] [genome=FILE.genome] [base=FILE.nnue] [strict=true|false] [opzioni di TablutNegamaxClient]}.
 * I primi tre argomenti sono posizionali, come nel riferimento. {@code strict=false} (default): se il genoma manca, e' corrotto o e' stato evoluto per un'altra base, si
 * stampa un AVVISO e si gioca con la BASE (il riferimento: un avvio che fallisce in torneo e' peggio); {@code strict=true}: errore e uscita con codice -1. Se la base stessa
 * non e' leggibile si esce sempre con -1 (non esiste un giocatore sensato da avviare).
 *
 * <p>
 * Nome dichiarato al server: quello di {@link TablutNegamaxClient} ({@code negamax}); i log del server non distinguono quindi questo client dal riferimento: per
 * riconoscerli in un torneo si usi il colore/ordine di avvio o il banner stampato all'avvio da questo programma.
 */
public final class TablutGeneticClient {

	static final String DEFAULT_BASE = "pesi_nnue/trained_v6.nnue";

	private TablutGeneticClient() {
	}

	/** La rete di gioco risolta e come ci si e' arrivati. */
	static final class Resolved {
		final NnueNetwork network;
		/** Vero se la rete e' la base con un genoma applicato (anche neutro). */
		final boolean genomeApplied;
		final List<String> warnings = new ArrayList<>();
		String description;

		Resolved(NnueNetwork network, boolean genomeApplied) {
			this.network = network;
			this.genomeApplied = genomeApplied;
		}
	}

	/**
	 * Risolve la rete di gioco.
	 *
	 * @param genomePath {@code null} = nessun genoma (si gioca con la base)
	 * @param strict     vero: ogni problema col genoma e' una {@link IOException}; falso: avviso e base
	 * @throws IOException base illeggibile; oppure, solo con {@code strict}, genoma mancante / non valido / per un'altra base
	 */
	static Resolved resolve(String basePath, String genomePath, boolean strict) throws IOException {
		final NnueNetwork base = NnueNetwork.load(basePath);
		final String fp = String.format("%08x", GaGenome.fingerprint(base));
		if (genomePath == null) {
			if (strict) {
				throw new IOException("strict=true richiede genome=FILE");
			}
			final Resolved r = new Resolved(base, false);
			r.warnings.add("nessun genoma indicato: gioco con la rete di base " + basePath);
			r.description = "base " + basePath + " (impronta " + fp + "), nessun genoma";
			return r;
		}
		try {
			final GaGenome g = GaGenome.load(genomePath, base);
			final Resolved r = new Resolved(g.decode(), true);
			r.description = "base " + basePath + " (impronta " + fp + ") + genoma " + genomePath + " (" + g.size() + " geni, gruppi " + g.layout().groupsText()
					+ (g.isNeutral() ? ", NEUTRO: identico alla base" : "") + ")";
			return r;
		} catch (IOException e) {
			if (strict) {
				throw new IOException("genoma " + genomePath + " non utilizzabile: " + e.getMessage(), e);
			}
			final Resolved r = new Resolved(base, false);
			r.warnings.add("genoma " + genomePath + " non utilizzabile (" + e.getMessage() + "): gioco con la rete di BASE " + basePath);
			r.description = "base " + basePath + " (impronta " + fp + "), genoma scartato";
			return r;
		}
	}

	/** Gli argomenti per {@link TablutNegamaxClient#main}: gli stessi di questo client meno {@code genome=, base=, strict=}, con il file dei pesi e {@code eval=nnue}. */
	static String[] delegateArgs(String[] args, String weightsPath) {
		final List<String> out = new ArrayList<>();
		out.add(args.length > 0 ? args[0] : "WHITE");
		out.add(args.length > 1 ? args[1] : "60");
		out.add(args.length > 2 ? args[2] : "localhost");
		out.add(weightsPath);
		boolean hasEval = false;
		final List<String> rest = new ArrayList<>();
		for (int i = 3; i < args.length; i++) {
			final String a = args[i];
			final String l = a.toLowerCase();
			if (l.startsWith("genome=") || l.startsWith("base=") || l.startsWith("strict=")) {
				continue;
			}
			if (l.startsWith("eval=")) {
				hasEval = true;
			}
			rest.add(a);
		}
		if (!hasEval) {
			out.add("eval=nnue");
		}
		out.addAll(rest);
		return out.toArray(new String[0]);
	}

	/** Il valore di un'opzione {@code chiave=valore} fra gli argomenti dal quarto in poi, o {@code dflt}. */
	static String option(String[] args, String key, String dflt) {
		for (int i = 3; i < args.length; i++) {
			if (args[i].toLowerCase().startsWith(key + "=")) {
				return args[i].substring(key.length() + 1);
			}
		}
		return dflt;
	}

	public static void main(String[] args) throws Exception {
		final String basePath = option(args, "base", DEFAULT_BASE);
		final String genomePath = option(args, "genome", null);
		final boolean strict = "true".equalsIgnoreCase(option(args, "strict", "false"));
		final Resolved r;
		try {
			r = resolve(basePath, genomePath, strict);
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(-1);
			return;
		}
		for (String w : r.warnings) {
			System.out.println("ATTENZIONE: " + w);
		}
		final File tmp;
		try {
			tmp = File.createTempFile("tablut_genetic_", ".nnue");
			tmp.deleteOnExit();
			r.network.save(tmp.getPath());
		} catch (IOException e) {
			System.out.println("ERRORE: impossibile scrivere la rete di gioco temporanea: " + e.getMessage());
			System.exit(-1);
			return;
		}
		System.out.println("Genetic client - " + r.description + (r.genomeApplied ? "" : " [FALLBACK/BASE]"));
		TablutNegamaxClient.main(delegateArgs(args, tmp.getPath()));
	}
}
