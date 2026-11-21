package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;

/**
 * Aggiunge a un {@link NnueDataset} il VALORE DI RICERCA di un motore: ogni posizione e' cercata a profondita' fissa e il punteggio della radice,
 * dal punto di vista del Bianco, diventa l'etichetta {@code value} ({@link NnueSelfPlay#scoreToValue}). Serve soprattutto per le posizioni di
 * partite VERE (log del server, {@code dati.gz}), che hanno l'esito ma nessun valore: dopo la rietichettatura ciascuna ha sia l'esito reale sia una
 * valutazione di ricerca, e {@link NnueTrainer} puo' mescolarli.
 * <p>
 * La storia delle posizioni (ripetizioni) non e' nel dataset: si cerca con storia vuota. A profondita' fissa il punteggio e' deterministico
 * (un'unica istanza di {@link Negamax} serve tutte le posizioni, in ordine). Gli esiti, la partita, i ply e la sorgente non cambiano.
 * <p>
 * Uso: {@code NnueRelabel in=FILE.gz out=FILE.gz eval=PESI|simple [depth=4] [scale=1000] [onlymissing=false]}.
 */
public final class NnueRelabel {

	private NnueRelabel() {
	}

	/**
	 * @param onlyMissing se vero, tiene le etichette gia' presenti e cerca solo le posizioni senza
	 * @return una copia di {@code d} con {@code value/valValid} aggiornati
	 */
	public static NnueDataset relabel(NnueDataset d, Evaluator ev, int depth, double scale, boolean onlyMissing, PrintStream log) {
		if (depth < 1 || depth > 40 || scale <= 0) {
			throw new IllegalArgumentException("profondita' o scala non valide: " + depth + ", " + scale);
		}
		final Negamax nm = new Negamax(ev, depth, true, 1 << 18, true, Negamax.ORDER_RECOMMENDED, 1);
		nm.setSafetyMarginMillis(0);
		final float[] value = new float[d.n]; // copie di lavoro delle due sole colonne che cambiano (con le pagine non esiste un array unico da clonare)
		final byte[] valValid = new byte[d.n];
		for (int i = 0; i < d.n; i++) {
			value[i] = d.value(i);
			valValid[i] = d.valValid(i);
		}
		final long t0 = System.currentTimeMillis();
		int done = 0, failed = 0;
		final byte[] board = new byte[81];
		for (int i = 0; i < d.n; i++) {
			if (onlyMissing && d.valValid(i) == 1) {
				continue;
			}
			d.copyBoard(i, board, 0);
			final byte role = d.role(i);
			final State s = NnueDataImport.toState(board, role);
			final Action move = nm.findBestMove(s, 1_000_000_000L, PositionHistory.empty());
			if (move == null) {
				failed++; // nessuna mossa: la posizione non e' valutabile con la ricerca
				value[i] = 0f;
				valValid[i] = 0;
				continue;
			}
			value[i] = (float) NnueSelfPlay.scoreToValue(nm.lastScore, role == 0, scale);
			valValid[i] = 1;
			done++;
			if (log != null && done % 500 == 0) {
				log.printf("  rietichettate %d/%d (%.0f s)%n", done, d.n, (System.currentTimeMillis() - t0) / 1000.0);
			}
		}
		if (log != null) {
			log.println("rietichettate " + done + " posizioni a profondita' " + depth + (failed > 0 ? ", " + failed + " senza mosse (lasciate senza valore)" : ""));
		}
		return d.withValues(value, valValid); // le altre colonne sono condivise con d, non copiate
	}

	public static void main(String[] args) throws IOException {
		String in = null, out = null, eval = null;
		int depth = 4;
		double scale = 1000.0;
		boolean onlyMissing = false;
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + a);
				System.exit(2);
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "in": in = v; break;
				case "out": out = v; break;
				case "eval": eval = v; break;
				case "depth": depth = Integer.parseInt(v); break;
				case "scale": scale = Double.parseDouble(v); break;
				case "onlymissing": onlyMissing = Boolean.parseBoolean(v); break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (NumberFormatException e) {
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (in == null || out == null || eval == null || depth < 1 || depth > 40 || scale <= 0) {
			System.out.println("Uso: NnueRelabel in=FILE.gz out=FILE.gz eval=PESI|simple [depth=4] [scale=1000] [onlymissing=false]");
			System.exit(2);
		}
		try {
			final NnueDataset d = NnueDataset.load(in);
			final Evaluator ev = "simple".equals(eval) ? new SimpleEvaluator() : EvaluatorLoader.load(eval);
			System.out.println("in: " + d.describe());
			final NnueDataset r = relabel(d, ev, depth, scale, onlyMissing, System.out);
			r.save(out);
			System.out.println("scritto " + out + ": " + r.describe());
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}
}
