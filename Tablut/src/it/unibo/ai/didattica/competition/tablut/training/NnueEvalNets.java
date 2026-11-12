package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;

/**
 * Misura una o piu' reti NNUE su un {@link NnueDataset} (tipicamente partite VERE tenute da parte come test finale): errore verso il bersaglio, verso
 * l'esito reale (con la linea di base "predire 0"), precisione sul vincitore e correlazioni, come in {@link NnueTrainer#evaluate}. Non addestra e non
 * sceglie nulla: serve a confrontare reti su dati che nessuna ha usato per decidere.
 * <p>
 * Uso: {@code NnueEvalNets nets=A.nnue[,B.nnue] data=T.gz [lambda=0.5] [gamma=1]}.
 */
public final class NnueEvalNets {

	private NnueEvalNets() {
	}

	public static void main(String[] args) throws IOException {
		String nets = null, data = null;
		final NnueTrainer.Config cfg = new NnueTrainer.Config();
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + a);
				System.exit(2);
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "nets": nets = v; break;
				case "data": data = v; break;
				case "lambda": cfg.lambda = Double.parseDouble(v); break;
				case "gamma": cfg.gamma = Double.parseDouble(v); break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (NumberFormatException e) {
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (nets == null || data == null || cfg.lambda < 0 || cfg.lambda > 1 || cfg.gamma <= 0 || cfg.gamma > 1) {
			System.out.println("Uso: NnueEvalNets nets=A.nnue[,B.nnue] data=T.gz [lambda=0.5] [gamma=1]");
			System.exit(2);
		}
		try {
			final NnueDataset d = NnueDataset.load(data);
			System.out.println("dati " + data + ": " + d.describe());
			final int[] all = new int[d.n];
			for (int i = 0; i < d.n; i++) {
				all[i] = i;
			}
			for (String p : nets.split(",")) {
				if (!p.trim().isEmpty()) {
					System.out.println(p.trim() + " | " + NnueTrainer.evaluate(NnueNetwork.load(p.trim()), cfg, d, all));
				}
			}
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}
}
