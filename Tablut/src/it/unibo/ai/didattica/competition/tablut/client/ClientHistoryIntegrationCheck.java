package it.unibo.ai.didattica.competition.tablut.client;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

import com.google.gson.Gson;

import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
import it.unibo.ai.didattica.competition.tablut.util.Configuration;
import it.unibo.ai.didattica.competition.tablut.util.StreamUtils;

/**
 * Integrazione del CLIENT REALE ({@link TablutNegamaxClient}, ruolo Bianco) con la storia a chiavi esatte: un finto server su localhost
 * (porta {@link Configuration#whitePort}, che deve essere libera) invia al client, con il protocollo vero (stringhe JSON di {@code State}), le
 * posizioni NON terminali di una partita casuale con catture, comprese quelle dopo la mossa dell'avversario. Il client gira nel suo ciclo
 * {@code run()} con un {@link MoveSearcher} che registra una copia della storia che riceve a ogni suo turno e risponde con una mossa legale.
 * Si verifica che, a ogni turno del client, la storia ricevuta dal motore coincida con quella dell'oracolo indipendente
 * ({@link TestPositions#exactKey(State)}): le posizioni dall'ultima cattura, quella corrente inclusa, con l'azzeramento a ogni cattura.
 * Il test termina con {@code System.exit}: non chiude mai il socket prima (altrimenti il client terminerebbe la JVM).
 */
public class ClientHistoryIntegrationCheck {

	static int failures = 0;

	static void check(boolean ok, String what) {
		if (!ok) {
			failures++;
			System.out.println("FALLITO: " + what);
		}
	}

	/** Searcher finto: registra la storia ricevuta e gioca la prima mossa legale. */
	static final class Recorder implements MoveSearcher {
		final List<PositionHistory> seen = new ArrayList<>();
		final List<long[]> roots = new ArrayList<>();

		@Override
		public int findBestMove(FastState root, long timeBudgetMillis, PositionHistory history) {
			seen.add(history.copy()); // copia: il client continua a modificare la sua storia
			roots.add(new long[] { root.key0, root.key1, root.key2 });
			final int[] buf = new int[FastState.MAX_MOVES];
			root.generateMoves(buf);
			return buf[0];
		}

		@Override
		public String name() {
			return "recorder";
		}

		@Override
		public String describeLastSearch() {
			return "recorder";
		}

		@Override
		public long lastWork() {
			return 0;
		}

		@Override
		public double lastDepth() {
			return 0;
		}

		@Override
		public void shutdown() {
		}
	}

	public static void main(String[] args) throws Exception {
		// una partita con abbastanza catture da esercitare l'azzeramento della storia
		TestPositions.Game game = null;
		int captures = 0;
		for (long seed = 1; seed < 200 && game == null; seed++) {
			final TestPositions.Game g = TestPositions.playGame(new Random(seed), true, 100);
			int caps = 0;
			for (int i = 1; i < g.histories.size(); i++) {
				if (g.histories.get(i).size() == 1) {
					caps++;
				}
			}
			if (g.positions.size() >= 30 && caps >= 3) {
				game = g;
				captures = caps;
			}
		}
		check(game != null, "non trovata una partita di prova con abbastanza catture");
		final List<State> positions = game.positions;

		final ServerSocket ss = new ServerSocket(Configuration.whitePort);
		final AtomicReference<Throwable> serverError = new AtomicReference<>();
		final List<Integer> clientTurns = new ArrayList<>();
		final Thread server = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					final Socket s = ss.accept();
					final DataInputStream in = new DataInputStream(s.getInputStream());
					final DataOutputStream out = new DataOutputStream(s.getOutputStream());
					final String name = StreamUtils.readString(in); // il client dichiara il nome
					check(name != null && name.contains("histtest"), "il client dichiara il nome");
					final Gson gson = new Gson();
					for (int i = 0; i < positions.size(); i++) {
						final State st = positions.get(i);
						StreamUtils.writeString(out, gson.toJson(st));
						if (st.getTurn() == Turn.WHITE) { // tocca al client: risponde con la sua mossa
							clientTurns.add(i);
							final String action = StreamUtils.readString(in);
							check(action != null && !action.isEmpty(), "il client risponde con una mossa");
						}
					}
				} catch (Throwable t) {
					serverError.set(t);
				}
			}
		});
		server.setDaemon(true);
		server.start();

		final Recorder rec = new Recorder();
		final TablutNegamaxClient client = new TablutNegamaxClient("white", "histtest", 60, "localhost", rec);
		final Thread clientThread = new Thread(new Runnable() {
			@Override
			public void run() {
				client.run();
			}
		});
		clientThread.setDaemon(true);
		clientThread.start();

		server.join(120_000);
		check(!server.isAlive(), "il finto server non ha terminato entro 120 s (il client e' bloccato?)");
		check(serverError.get() == null, "errore nel finto server: " + serverError.get());
		// il client ha gia' risposto all'ultimo turno suo quando il server ha finito: attendo solo che la registrazione sia completa
		for (int i = 0; i < 100 && rec.seen.size() < clientTurns.size(); i++) {
			Thread.sleep(50);
		}

		check(rec.seen.size() == clientTurns.size() && !clientTurns.isEmpty(), "il client ha cercato a ogni suo turno: " + rec.seen.size() + " ricerche per " + clientTurns.size() + " turni");
		int compared = 0;
		for (int k = 0; k < Math.min(rec.seen.size(), clientTurns.size()); k++) {
			final int i = clientTurns.get(k);
			final PositionHistory expected = game.histories.get(i);
			check(rec.seen.get(k).equals(expected), "turno " + k + " (posizione " + i + "): la storia ricevuta dal motore e' diversa da quella dell'oracolo indipendente ("
				+ rec.seen.get(k).size() + " vs " + expected.size() + " posizioni)");
			final long[] root = TestPositions.exactKey(positions.get(i));
			check(rec.roots.get(k)[0] == root[0] && rec.roots.get(k)[1] == root[1] && rec.roots.get(k)[2] == root[2], "turno " + k + ": la radice non e' la posizione inviata dal server");
			check(rec.seen.get(k).size() >= 1 && rec.seen.get(k).endsWith(FastState.fromState(positions.get(i))), "turno " + k + ": la radice e' l'ultima voce della storia");
			compared++;
		}
		int resets = 0;
		for (int k = 0; k < clientTurns.size(); k++) {
			if (game.histories.get(clientTurns.get(k)).size() == 1 && clientTurns.get(k) > 0) {
				resets++;
			}
		}
		System.out.println("posizioni inviate: " + positions.size() + " (" + captures + " catture nella partita); turni del client confrontati: " + compared + ", di cui " + resets + " subito dopo un azzeramento");
		System.out.println(failures == 0 ? "=== CLIENT HISTORY INTEGRATION CHECK OK ===" : "=== FALLITI: " + failures + " ===");
		System.exit(failures == 0 ? 0 : 1); // NON si chiude il socket: il client terminerebbe la JVM con exit(1)
	}
}
