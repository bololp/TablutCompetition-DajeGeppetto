package it.unibo.ai.didattica.competition.tablut.client;

import java.io.IOException;
import java.net.UnknownHostException;

/**
 * Punto d'ingresso comodo per giocare con Monte Carlo Tree Search: equivale a {@link TablutNegamaxClient} con {@code algo=mcts}. Stessi argomenti
 * (nell'ordine richiesto dalla competizione piu' gli opzionali):
 * <pre>
 *   java ... TablutMctsClient &lt;WHITE|BLACK&gt; [timeoutSeconds] [serverIp] [pesiRete.txt] [threads=N] [margin=MS] [mcts=chiave=valore,...]
 * </pre>
 */
public final class TablutMctsClient {

	private TablutMctsClient() {
	}

	public static void main(String[] args) throws UnknownHostException, IOException, ClassNotFoundException {
		if (System.getProperty("tablut.algo") == null) {
			System.setProperty("tablut.algo", "mcts");
		}
		TablutNegamaxClient.main(args);
	}
}
