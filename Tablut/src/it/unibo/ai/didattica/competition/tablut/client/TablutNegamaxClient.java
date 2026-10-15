package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ZobristHasher;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Client che gioca usando Negamax + alpha-beta invece di mosse casuali.
 * Funziona indifferentemente come BIANCO o come NERO: non c'e' bisogno di due
 * classi separate (a differenza di {@code TablutRandomClient}, che duplica la
 * logica), perche' {@link Negamax} e {@link it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules}
 * lavorano in modo simmetrico sul colore memorizzato in {@code state.getTurn()}.
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
		super(player, name, timeout, ipAddress);
		this.timeoutSeconds = timeout;
		this.negamax = new Negamax(evaluator).setThreads(threads);
	}

	/**
	 * Argomenti, nell'ordine richiesto dalla competizione piu' due opzionali:
	 * {@code <WHITE|BLACK> [timeoutSeconds] [serverIp] [pesiRete.txt] [threads=N]}.
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
		String weightsFile = null;
		for (int i = 3; i < args.length; i++) {
			if (args[i].toLowerCase().startsWith("threads=")) {
				try {
					threads = Math.max(1, Integer.parseInt(args[i].substring("threads=".length()).trim()));
				} catch (NumberFormatException e) {
					System.out.println("ATTENZIONE: '" + args[i] + "' non valido, uso " + threads + " thread.");
				}
			} else if (weightsFile == null) {
				weightsFile = args[i];
			}
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
				+ " - thread di ricerca: " + threads);

		TablutNegamaxClient client = new TablutNegamaxClient(role, "negamax", timeout, ipAddress, evaluator, threads);
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
			updateRealGameHistory(state);

			if (turn == this.getPlayer()) {
				System.out.println("Pensando...");
				// tocca a me: cerco la mossa migliore entro il tempo a disposizione
				long budgetMillis = this.timeoutSeconds * 1000L;
				long start = System.currentTimeMillis();

				Action chosen = this.negamax.findBestMove(state, budgetMillis, this.historySinceLastCapture);

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
	private void updateRealGameHistory(State state) {
		Turn t = state.getTurn();
		if (t != Turn.WHITE && t != Turn.BLACK) {
			return; // partita finita: non serve piu' tracciare nulla
		}
		int pieceCount = countPieces(state);
		if (lastObservedPieceCount >= 0 && pieceCount < lastObservedPieceCount) {
			historySinceLastCapture.clear();
		}
		historySinceLastCapture.add(ZobristHasher.fullHash(state));
		lastObservedPieceCount = pieceCount;
	}

	private static int countPieces(State state) {
		int n = 0;
		for (Pawn[] row : state.getBoard()) {
			for (Pawn p : row) {
				if (p == Pawn.WHITE || p == Pawn.BLACK || p == Pawn.KING) {
					n++;
				}
			}
		}
		return n;
	}
}
