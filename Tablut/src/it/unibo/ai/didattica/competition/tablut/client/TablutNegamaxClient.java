package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CachingEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Mcts;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MctsConfig;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NegamaxSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Client che gioca usando una ricerca invece di mosse casuali: Negamax + alpha-beta (default) oppure Monte Carlo Tree Search
 * ({@code algo=mcts}, si veda {@link Mcts}).
 * Funziona indifferentemente come BIANCO o come NERO: non c'e' bisogno di due
 * classi separate (a differenza di {@code TablutRandomClient}, che duplica la
 * logica), perche' {@link Negamax} e {@link FastState} lavorano in modo simmetrico sul colore di chi
 * deve muovere. Il client parla {@code State}/{@code Action} col server e converte UNA volta per mossa (O(81)) nella
 * rappresentazione leggera {@link FastState} su cui lavora la ricerca.
 *
 * <p>Uso a riga di comando (stesso ordine richiesto dalla competizione):
 * <pre>
 *   java ... TablutNegamaxClient &lt;WHITE|BLACK&gt; [timeoutSeconds] [serverIp]
 * </pre>
 */
public class TablutNegamaxClient extends TablutClient {

	/** Ordinamento delle mosse di Negamax usato da {@link #main} se non c'e' {@code order=} (nomi di {@link OrderingSpec}). */
	static final String DEFAULT_ORDER_NAME = "rec";

	/** Bit della dimensione della cache delle valutazioni (MLP/CNN) di {@link #main} se non c'e' {@code evcache=}: 2^22 voci = 128 MiB. */
	static final int DEFAULT_EVAL_CACHE_LOG2 = 22;

	/** Timeout (in secondi) dichiarato dal server/arbitro per ogni mossa. */
	private final int timeoutSeconds;

	/** L'algoritmo di ricerca scelto: {@link NegamaxSearcher} (default) oppure {@link Mcts}. */
	private final MoveSearcher searcher;

	// --- storia REALE della partita, per il pareggio per ripetizione ---
	// (si veda Negamax: "la stessa posizione raggiunta due volte dall'ultima
	// cattura e' pareggio"). Aggiornata ad ogni stato osservato dal server,
	// azzerata ad ogni cattura (nessuna posizione precedente puo' piu'
	// ripresentarsi una volta sparita una pedina).
	private final List<Long> historySinceLastCapture = new ArrayList<>();
	private int lastObservedPieceCount = -1;

	/** Costruttore "storico": nessun file di pesi -> usa SimpleEvaluator. */
	public TablutNegamaxClient(String player, String name, int timeout, String ipAddress)
			throws UnknownHostException, IOException {
		this(player, name, timeout, ipAddress, new SimpleEvaluator());
	}

	public TablutNegamaxClient(String player, String name, int timeout, String ipAddress, Evaluator evaluator)
			throws UnknownHostException, IOException {
		this(player, name, timeout, ipAddress, evaluator, Negamax.defaultThreads());
	}

	/** @param threads thread di ricerca (Lazy SMP): si veda {@link Negamax#defaultThreads()} per il default. */
	public TablutNegamaxClient(String player, String name, int timeout, String ipAddress, Evaluator evaluator,
			int threads) throws UnknownHostException, IOException {
		this(player, name, timeout, ipAddress, new Negamax(evaluator).setThreads(threads));
	}

	/** Con una ricerca gia' configurata (ordinamento, TT, margine di sicurezza, thread): si veda {@link #main}. */
	public TablutNegamaxClient(String player, String name, int timeout, String ipAddress, Negamax negamax)
			throws UnknownHostException, IOException {
		this(player, name, timeout, ipAddress, (MoveSearcher) new NegamaxSearcher(negamax));
	}

	/** Con un algoritmo di ricerca qualunque ({@link NegamaxSearcher} o {@link Mcts}), gia' configurato. */
	public TablutNegamaxClient(String player, String name, int timeout, String ipAddress, MoveSearcher searcher)
			throws UnknownHostException, IOException {
		super(player, name, timeout, ipAddress);
		if (searcher == null) {
			throw new IllegalArgumentException("searcher nullo");
		}
		this.timeoutSeconds = timeout;
		this.searcher = searcher;
	}

	/** Normalizza il nome dell'algoritmo ({@code negamax} o {@code mcts}, senza badare alle maiuscole). */
	static String normalizeAlgo(String algo) {
		final String a = algo == null ? "" : algo.trim().toLowerCase();
		if (a.equals("negamax") || a.equals("mcts")) {
			return a;
		}
		throw new IllegalArgumentException("algoritmo sconosciuto '" + algo + "' (ammessi: negamax, mcts)");
	}

	/**
	 * Costruisce l'algoritmo di ricerca scelto. Per {@code mcts} i parametri sono quelli di {@link MctsConfig#parse(String)} e il tetto ai nodi
	 * (se non e' dato) si ricava dall'heap; {@code order}, {@code ttLog2} e {@code twoSlot} valgono solo per Negamax.
	 *
	 * @throws IllegalArgumentException algoritmo o parametri MCTS non validi
	 */
	static MoveSearcher buildSearcher(String algo, String mctsSpec, Evaluator evaluator, int threads, long margin, int order, int ttLog2, boolean twoSlot) {
		if (normalizeAlgo(algo).equals("mcts")) {
			final MctsConfig cfg = MctsConfig.parse(mctsSpec);
			if (cfg.maxNodes == 0) {
				cfg.maxNodes = Mcts.autoMaxNodes(Runtime.getRuntime().maxMemory());
			}
			return new Mcts(evaluator, cfg, threads).setSafetyMarginMillis(margin);
		}
		return new NegamaxSearcher(new Negamax(evaluator, 40, true, 1 << ttLog2, true, order, threads, twoSlot).setThreads(threads)
				.setSafetyMarginMillis(margin));
	}

	/**
	 * Avvolge {@code evaluator} in una {@link CachingEvaluator} di 2^log2 voci (ridotta se supera un ottavo dell'heap) SOLO se l'algoritmo e'
	 * Negamax (l'MCTS ha la sua {@code EvalCache}), {@code log2 > 0} e l'evaluator e' una rete (MLP/CNN: con {@link SimpleEvaluator} la cache non
	 * conviene); altrimenti restituisce lo stesso {@code evaluator}.
	 */
	static Evaluator withEvalCache(String algo, Evaluator evaluator, int log2, long maxMemoryBytes) {
		if (!algo.equals("negamax") || log2 <= 0 || !CachingEvaluator.isCacheable(evaluator)) {
			return evaluator;
		}
		final int fitted = CachingEvaluator.fitToHeap(log2, maxMemoryBytes);
		if (fitted != log2) {
			System.out.println("ATTENZIONE: la cache delle valutazioni di 2^" + log2 + " voci (" + CachingEvaluator.bytesFor(log2) / (1 << 20)
					+ " MB) supera un ottavo dell'heap (" + maxMemoryBytes / (1 << 20) + " MB): uso 2^" + fitted + ".");
		}
		return new CachingEvaluator(evaluator, fitted);
	}

	/** Dimensione in byte di una TT di 2^log2 voci (3 long ciascuna). */
	static long ttBytes(int log2) {
		return (1L << log2) * 3 * 8;
	}

	/**
	 * Riduce {@code log2} finche' la TT sta con ampio margine nell'heap della JVM ({@code maxMemory}): un OutOfMemoryError nel mezzo
	 * di una partita farebbe perdere per timeout. Lascia almeno meta' dell'heap al resto (alberi di ricerca, rete, GC).
	 */
	static int fitTtToHeap(int log2, long maxMemoryBytes) {
		int l = log2;
		while (l > 12 && ttBytes(l) > maxMemoryBytes / 2) {
			l--;
		}
		return l;
	}

	/** Margine di sicurezza di default per un timeout di mossa dato: 1,5 s con i tempi di gara, quello storico (300 ms) con timeout brevi (prove). */
	static long defaultMarginMillis(int timeoutSeconds) {
		return timeoutSeconds >= 20 ? 1500 : Negamax.defaultSafetyMarginMillis();
	}

	/**
	 * Margine di sicurezza AGGIUNTIVO di default per l'MCTS (solo se l'utente non da' {@code margin=}). La sua memoria viva e' un albero grande, quindi le
	 * pause del garbage collector sono piu' lunghe e piu' frequenti che per Negamax (misurato su un core con collector seriale: pause fino a 573 ms, e una
	 * partita persa per timeout con un timeout di 2 s); 500 ms in piu' costano meno dell'1% del tempo per mossa a 60 s. Con timeout piccoli (< 20 s, usati
	 * solo per prove) non si aggiunge nulla.
	 */
	static long mctsExtraMarginMillis(int timeoutSeconds) {
		return timeoutSeconds >= 20 ? 500 : 0;
	}

	/**
	 * Argomenti, nell'ordine richiesto dalla competizione piu' due opzionali:
	 * {@code <WHITE|BLACK> [timeoutSeconds] [serverIp] [pesiRete.txt] [threads=N] [order=SPEC] [tt=LOG2] [tt2=true|false] [margin=MS]
	 * [algo=negamax|mcts] [mcts=chiave=valore,...]}.
	 * {@code algo=} sceglie l'algoritmo di ricerca ({@code negamax}, default, oppure {@code mcts}: Monte Carlo Tree Search, si veda {@link Mcts});
	 * {@code mcts=} ne passa i parametri ({@link MctsConfig}, es. {@code mcts=cpuct=1.5,tau=0.2,expand=1}); con {@code algo=mcts} le opzioni
	 * {@code order}, {@code tt} e {@code tt2} non hanno effetto, {@code threads} e {@code margin} valgono come per Negamax. Si puo' scegliere
	 * l'algoritmo anche con la proprieta' di sistema {@code -Dtablut.algo=mcts} (usata da {@link TablutMctsClient}).
	 * {@code order=} sceglie l'ordinamento delle mosse ({@link OrderingSpec}: {@code rec} = wins+caps+rootsc, default;
	 * {@code base} = quello storico, killer+history; le misure sui nodi lo danno in vantaggio ma NON e' ancora confermato in arena a 4 core);
	 * {@code evcache=LOG2} sono i bit della dimensione della cache delle valutazioni di MLP/CNN ({@link CachingEvaluator}: default 22 =
	 * 4,2 M voci = 128 MiB, al piu' un ottavo dell'heap; {@code evcache=0} la disattiva; mai usata con SimpleEvaluator ne' con MCTS,
	 * che ha la sua); {@code tt=} sono i bit della
	 * dimensione della TT (default 22 = 4,2 M voci = 100 MB; si riduce da sola se non sta nell'heap); {@code tt2=true} usa i bucket a due
	 * slot; {@code margin=} e' il margine di sicurezza in ms tolto al timeout (default 1500 con timeout >= 20 s, altrimenti 300).
	 * Gli argomenti facoltativi (la competizione li permette, si veda
	 * slide 20: "Your player can have additional optional parameters") possono comparire in
	 * qualunque ordine dopo l'IP: un file di pesi e/o {@code threads=N}. Senza pesi, o se il file non si
	 * carica correttamente, si usa {@link SimpleEvaluator}. Senza {@code threads=N} i thread di ricerca
	 * (Lazy SMP) sono {@link Negamax#defaultThreads()}: 4 su una macchina a 4 core (o la proprieta' di
	 * sistema {@code -Dtablut.threads=N}); {@code threads=1} = ricerca a thread singolo.
	 */
	public static void main(String[] args) throws UnknownHostException, IOException, ClassNotFoundException {
		String role;
		int timeout = 60;
		String ipAddress = "localhost";

		if (args.length < 1) {
			System.out.println("Devi specificare il ruolo: WHITE oppure BLACK");
			System.exit(-1);
			return;
		}
		role = args[0];

		if (args.length >= 2) {
			timeout = Integer.parseInt(args[1]);
		}
		if (args.length >= 3) {
			ipAddress = args[2];
		}

		int threads = Negamax.defaultThreads();
		int order = OrderingSpec.parse(DEFAULT_ORDER_NAME);
		String orderName = DEFAULT_ORDER_NAME;
		int evCacheLog2 = DEFAULT_EVAL_CACHE_LOG2;
		int ttLog2 = 26;
		boolean twoSlot = false;
		long margin = defaultMarginMillis(timeout);
		boolean marginGiven = false;
		String weightsFile = null;
		String algo = System.getProperty("tablut.algo", "negamax");
		String mctsSpec = "";
		for (int i = 3; i < args.length; i++) {
			final String low = args[i].toLowerCase();
			try {
				if (low.startsWith("threads=")) {
					threads = Math.max(1, Integer.parseInt(args[i].substring("threads=".length()).trim()));
				} else if (low.startsWith("order=")) {
					final String name = args[i].substring("order=".length()).trim();
					order = OrderingSpec.parse(name); // lancia se il nome e' sconosciuto: allora order e orderName restano quelli di prima
					orderName = name;
				} else if (low.startsWith("evcache=")) {
					evCacheLog2 = Math.max(0, Math.min(26, Integer.parseInt(args[i].substring("evcache=".length()).trim())));
				} else if (low.startsWith("tt=")) {
					ttLog2 = Math.max(12, Math.min(28, Integer.parseInt(args[i].substring("tt=".length()).trim())));
				} else if (low.startsWith("tt2=")) {
					twoSlot = Boolean.parseBoolean(args[i].substring("tt2=".length()).trim());
				} else if (low.startsWith("margin=")) {
					margin = Math.max(0, Long.parseLong(args[i].substring("margin=".length()).trim()));
					marginGiven = true;
				} else if (low.startsWith("algo=")) {
					algo = normalizeAlgo(args[i].substring("algo=".length())); // lancia se sconosciuto: l'algoritmo resta quello di prima
				} else if (low.startsWith("mcts=")) {
					final String spec = args[i].substring("mcts=".length()).trim();
					MctsConfig.parse(spec); // solo per validarla: lancia se non valida (allora mctsSpec resta quella di prima)
					mctsSpec = spec;
				} else if (weightsFile == null) {
					weightsFile = args[i];
				}
			} catch (IllegalArgumentException e) { // NumberFormatException e nome di ordinamento sconosciuto
				System.out.println("ATTENZIONE: '" + args[i] + "' non valido (" + e.getMessage() + "), lo ignoro.");
			}
		}
		try {
			algo = normalizeAlgo(algo); // anche la proprieta' di sistema va validata
		} catch (IllegalArgumentException e) {
			System.out.println("ATTENZIONE: " + e.getMessage() + ": uso negamax.");
			algo = "negamax";
		}
		if (algo.equals("mcts") && !marginGiven) {
			margin += mctsExtraMarginMillis(timeout);
		}
		if (!mctsSpec.isEmpty() && !algo.equals("mcts")) {
			System.out.println("ATTENZIONE: 'mcts=' ignorato perche' l'algoritmo e' " + algo + " (aggiungi algo=mcts).");
		}
		final int fitted = fitTtToHeap(ttLog2, Runtime.getRuntime().maxMemory());
		if (fitted != ttLog2 && algo.equals("negamax")) {
			System.out.println("ATTENZIONE: la TT di 2^" + ttLog2 + " voci (" + ttBytes(ttLog2) / (1 << 20) + " MB) non sta nell'heap ("
					+ Runtime.getRuntime().maxMemory() / (1 << 20) + " MB): uso 2^" + fitted + ". Per una TT piu' grande lancia la JVM con -Xmx piu' alto.");
			ttLog2 = fitted;
		}

		Evaluator evaluator = new SimpleEvaluator();
		if (weightsFile != null) {
			try {
				evaluator = EvaluatorLoader.load(weightsFile); // MLP o CNN, riconosciuto dall'intestazione del file
				System.out.println("Valutazione: " + EvaluatorLoader.describe(evaluator) + ", pesi caricati da " + weightsFile);
			} catch (IOException e) {
				System.out.println("ATTENZIONE: impossibile caricare i pesi da '" + weightsFile + "' ("
						+ e.getMessage() + "). Uso SimpleEvaluator.");
			}
		} else {
			System.out.println("Nessun file di pesi indicato: uso SimpleEvaluator.");
		}

		// cache delle valutazioni: solo Negamax con una rete (si veda withEvalCache). Il decoratore e' creato QUI, subito dopo il
		// caricamento, e solo lui e' passato alla ricerca: i pesi non cambiano piu'. Il banner descrive l'oggetto effettivo.
		evaluator = withEvalCache(algo, evaluator, evCacheLog2, Runtime.getRuntime().maxMemory());
		final String evCacheDesc = evaluator instanceof CachingEvaluator
				? "2^" + ((CachingEvaluator) evaluator).log2Entries() + " voci (" + CachingEvaluator.bytesFor(((CachingEvaluator) evaluator).log2Entries()) / (1 << 20) + " MB)"
				: "disattivata";

		if (algo.equals("mcts")) {
			final MctsConfig shown = MctsConfig.parse(mctsSpec);
			final long cap = shown.maxNodes > 0 ? shown.maxNodes : Mcts.autoMaxNodes(Runtime.getRuntime().maxMemory());
			System.out.println("MCTS client - ruolo: " + role + " - timeout: " + timeout + "s - server: " + ipAddress + " - thread di ricerca: " + threads
					+ " - parametri: " + shown + " - tetto nodi: " + cap + " (heap " + Runtime.getRuntime().maxMemory() / (1 << 20)
					+ " MB; per un albero piu' grande lancia la JVM con -Xmx piu' alto) - margine: " + margin + " ms");
		} else {
			System.out.println("Negamax client - ruolo: " + role + " - timeout: " + timeout + "s - server: " + ipAddress
					+ " - thread di ricerca: " + threads + " - ordinamento: " + orderName + " (modo " + order + ") - TT: 2^" + ttLog2 + " voci ("
					+ ttBytes(ttLog2) / (1 << 20) + " MB, " + (twoSlot ? "due slot" : "slot singolo") + ") - cache valutazioni: " + evCacheDesc
					+ " - margine: " + margin + " ms");
		}

		MoveSearcher searcher = buildSearcher(algo, mctsSpec, evaluator, threads, margin, order, ttLog2, twoSlot);
		TablutNegamaxClient client = new TablutNegamaxClient(role, algo, timeout, ipAddress, searcher);
		client.run();
	}

	@Override
	public void run() {
		try {
			this.declareName();
		} catch (Exception e) {
			e.printStackTrace();
		}

		System.out.println("Sono il giocatore " + this.getPlayer().toString() + "!");

		while (true) {
			try {
				this.read();
			} catch (ClassNotFoundException | IOException e) {
				e.printStackTrace();
				System.exit(1);
			}

			State state = this.getCurrentState();
			Turn turn = state.getTurn();
			FastState position = FastState.fromState(state); // una sola conversione per mossa: storia e ricerca la condividono
			updateRealGameHistory(position);

			if (turn == this.getPlayer()) {
				System.out.println("Pensando...");
				// tocca a me: cerco la mossa migliore entro il tempo a disposizione
				long budgetMillis = this.timeoutSeconds * 1000L;
				long start = System.currentTimeMillis();

				final int histLen = this.historySinceLastCapture.size();
				final long[] history = new long[histLen];
				for (int i = 0; i < histLen; i++) {
					history[i] = this.historySinceLastCapture.get(i);
				}
				int move = this.searcher.findBestMove(position, budgetMillis, history, histLen);
				Action chosen = move == Move.NONE ? null : Move.toAction(move, turn);

				long elapsed = System.currentTimeMillis() - start;
				System.out.println("Mossa scelta: " + chosen + " | " + this.searcher.describeLastSearch() + " tempo=" + elapsed + "ms");

				if (chosen == null) {
					// Non dovrebbe mai succedere: se non ho mosse legali, la partita
					// dovrebbe essere gia' terminata (turno WHITEWIN/BLACKWIN) prima
					// che il server mi richieda una mossa. Per sicurezza usciamo con
					// un messaggio chiaro invece di mandare in crash silenzioso il client.
					System.out.println("ATTENZIONE: nessuna mossa legale disponibile!");
					System.exit(1);
				}

				try {
					this.write(chosen);
				} catch (ClassNotFoundException | IOException e) {
					e.printStackTrace();
				}

			} else if (turn == Turn.WHITEWIN) {
				System.out.println(this.getPlayer() == Turn.WHITE ? "HO VINTO!" : "HO PERSO!");
				System.exit(0);
			} else if (turn == Turn.BLACKWIN) {
				System.out.println(this.getPlayer() == Turn.BLACK ? "HO VINTO!" : "HO PERSO!");
				System.exit(0);
			} else if (turn == Turn.DRAW) {
				System.out.println("PAREGGIO!");
				System.exit(0);
			} else {
				System.out.println("In attesa della mossa dell'avversario...");
			}
		}
	}

	/**
	 * Aggiorna la storia reale della partita con lo stato appena osservato dal
	 * server (sia dopo una mia mossa sia dopo quella dell'avversario: il ciclo
	 * di lettura del client li vede entrambi, esattamente come dovrebbe). Se il
	 * numero di pedine e' diminuito rispetto all'ultimo stato osservato, una
	 * cattura e' avvenuta: si azzera la memoria, dato che nessuna posizione
	 * precedente a una cattura puo' piu' ripresentarsi (regola ufficiale).
	 */
	private void updateRealGameHistory(FastState position) {
		if (position.isGameOver()) {
			return; // partita finita: non serve piu' tracciare nulla
		}
		int pieceCount = position.whiteCount + position.blackCount + (position.kingSquare >= 0 ? 1 : 0);
		if (lastObservedPieceCount >= 0 && pieceCount < lastObservedPieceCount) {
			historySinceLastCapture.clear();
		}
		historySinceLastCapture.add(position.zobristHash);
		lastObservedPieceCount = pieceCount;
	}
}
