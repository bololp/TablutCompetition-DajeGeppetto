package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;

/**
 * Client che gioca usando Negamax + alpha-beta invece di mosse casuali.
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

	/** Timeout (in secondi) dichiarato dal server/arbitro per ogni mossa. */
	private final int timeoutSeconds;

	private final Negamax negamax;

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
		super(player, name, timeout, ipAddress);
		this.timeoutSeconds = timeout;
		this.negamax = negamax;
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
	 * Argomenti, nell'ordine richiesto dalla competizione piu' due opzionali:
	 * {@code <WHITE|BLACK> [timeoutSeconds] [serverIp] [pesiRete.txt] [threads=N] [order=SPEC] [tt=LOG2] [tt2=true|false] [margin=MS]}.
	 * {@code order=} sceglie l'ordinamento delle mosse ({@link OrderingSpec}: {@code base} = quello storico, default; {@code rec} =
	 * wins+caps+rootsc, consigliato dalle misure sui nodi ma NON ancora confermato in arena a 4 core); {@code tt=} sono i bit della
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
		int order = Negamax.ORDER_DEFAULT;
		String orderName = "base";
		int ttLog2 = 26;
		boolean twoSlot = false;
		long margin = defaultMarginMillis(timeout);
		String weightsFile = null;
		for (int i = 3; i < args.length; i++) {
			final String low = args[i].toLowerCase();
			try {
				if (low.startsWith("threads=")) {
					threads = Math.max(1, Integer.parseInt(args[i].substring("threads=".length()).trim()));
				} else if (low.startsWith("order=")) {
					final String name = args[i].substring("order=".length()).trim();
					order = OrderingSpec.parse(name); // lancia se il nome e' sconosciuto: allora order e orderName restano quelli di prima
					orderName = name;
				} else if (low.startsWith("tt=")) {
					ttLog2 = Math.max(12, Math.min(28, Integer.parseInt(args[i].substring("tt=".length()).trim())));
				} else if (low.startsWith("tt2=")) {
					twoSlot = Boolean.parseBoolean(args[i].substring("tt2=".length()).trim());
				} else if (low.startsWith("margin=")) {
					margin = Math.max(0, Long.parseLong(args[i].substring("margin=".length()).trim()));
				} else if (weightsFile == null) {
					weightsFile = args[i];
				}
			} catch (IllegalArgumentException e) { // NumberFormatException e nome di ordinamento sconosciuto
				System.out.println("ATTENZIONE: '" + args[i] + "' non valido (" + e.getMessage() + "), lo ignoro.");
			}
		}
		final int fitted = fitTtToHeap(ttLog2, Runtime.getRuntime().maxMemory());
		if (fitted != ttLog2) {
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

		System.out.println("Negamax client - ruolo: " + role + " - timeout: " + timeout + "s - server: " + ipAddress
				+ " - thread di ricerca: " + threads + " - ordinamento: " + orderName + " - TT: 2^" + ttLog2 + " voci ("
				+ ttBytes(ttLog2) / (1 << 20) + " MB, " + (twoSlot ? "due slot" : "slot singolo") + ") - margine: " + margin + " ms");

		Negamax negamax = new Negamax(evaluator, 40, true, 1 << ttLog2, true, order, threads, twoSlot).setThreads(threads)
				.setSafetyMarginMillis(margin);
		TablutNegamaxClient client = new TablutNegamaxClient(role, "negamax", timeout, ipAddress, negamax);
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

				int move = this.negamax.findBestMove(position, budgetMillis, this.historySinceLastCapture);
				Action chosen = move == Move.NONE ? null : Move.toAction(move, turn);

				long elapsed = System.currentTimeMillis() - start;
				System.out.println("Mossa scelta: " + chosen + " | profondita' raggiunta="
						+ this.negamax.lastDepthReached + " nodi=" + this.negamax.lastNodeCount + " valore="
						+ String.format("%.1f", this.negamax.lastScore) + " tempo=" + elapsed + "ms");

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
