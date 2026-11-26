package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.PruningOptions;
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
 * <h3>Versione veloce ({@link #relabelFast})</h3>
 * Per rietichettare tante posizioni a profondita' alta: le posizioni si dividono in blocchi consecutivi di {@code chunk} (predefinito 256), ogni blocco ha una
 * {@link Negamax} propria (TT di {@code 2^tt} voci) che le cerca in ordine, e i blocchi girano su {@code threads} thread. Il risultato dipende dal blocco e NON dal
 * numero di thread: a parita' di {@code chunk} lo stesso dataset con 1 o con 16 thread. {@code prune=} abilita null-move/LMR/aspirazione (~5x a profondita' 7);
 * {@code every=K} etichetta una posizione ogni K (le altre non cambiano, con {@code keep=false} si scrivono solo le etichettate); {@code skipsat=X} non spreca la ricerca profonda sulle posizioni
 * con valore GIA' presente {@code |v| >= X}.
 * <p>
 * Uso: {@code NnueRelabel in=FILE.gz out=FILE.gz eval=PESI|simple [depth=4] [scale=1000] [onlymissing=false]} e, per la versione veloce, anche
 * {@code [threads=1] [prune=off] [tt=18] [chunk=256] [every=1] [keep=true] [skipsat=0]} (se compare una qualunque di queste si usa {@link #relabelFast}).
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

	/** Opzioni di {@link #relabelFast}; i valori predefiniti riproducono {@link #relabel} bloccando le posizioni a gruppi di {@code chunk}. */
	public static final class Options {
		public int threads = 1, ttLog2 = 18, chunk = 256, every = 1;
		public PruningOptions pruning = PruningOptions.OFF;
		public boolean keepUnlabelled = true;
		public double skipSat = 0;
		public boolean onlyMissing = false;
	}

	/** Rietichetta {@code d} a profondita' {@code depth} con blocchi indipendenti su piu' thread; si veda la documentazione della classe. */
	public static NnueDataset relabelFast(final NnueDataset d, final Evaluator ev, final int depth, final double scale, final Options o, final PrintStream log) throws Exception {
		if (depth < 1 || depth > 40 || scale <= 0 || o == null || o.threads < 1 || o.ttLog2 < 10 || o.ttLog2 > 28 || o.chunk < 1 || o.every < 1 || o.skipSat < 0 || o.skipSat > 1
				|| o.pruning == null) {
			throw new IllegalArgumentException("opzioni non valide di rietichettatura");
		}
		final float[] value = new float[d.n];
		final byte[] valValid = new byte[d.n];
		final byte[] selected = new byte[d.n]; // 1 = posizione scelta da etichettare (o gia' etichettata con keep)
		for (int i = 0; i < d.n; i++) {
			value[i] = d.value(i);
			valValid[i] = d.valValid(i);
			final boolean pick = i % o.every == 0 && !(o.onlyMissing && d.valValid(i) == 1) && !(o.skipSat > 0 && d.valValid(i) == 1 && Math.abs(d.value(i)) >= o.skipSat);
			selected[i] = (byte) (pick ? 1 : 0);
		}
		final int nChunks = (d.n + o.chunk - 1) / o.chunk;
		final AtomicInteger done = new AtomicInteger(), failed = new AtomicInteger();
		final long t0 = System.currentTimeMillis();
		final ExecutorService pool = Executors.newFixedThreadPool(o.threads);
		try {
			final List<Future<?>> fs = new ArrayList<>();
			for (int c = 0; c < nChunks; c++) {
				final int lo = c * o.chunk, hi = Math.min(d.n, lo + o.chunk);
				fs.add(pool.submit(new Runnable() {
					@Override
					public void run() {
						final Negamax nm = new Negamax(ev, depth, true, 1 << o.ttLog2, true, Negamax.ORDER_RECOMMENDED, 1);
						nm.setSafetyMarginMillis(0);
						if (o.pruning != PruningOptions.OFF) {
							nm.setPruning(o.pruning);
						}
						final byte[] board = new byte[81];
						for (int i = lo; i < hi; i++) {
							if (selected[i] == 0) {
								continue;
							}
							d.copyBoard(i, board, 0);
							final byte role = d.role(i);
							final State s = NnueDataImport.toState(board, role);
							final Action move = nm.findBestMove(s, 1_000_000_000L, PositionHistory.empty());
							if (move == null) {
								failed.incrementAndGet();
								value[i] = 0f;
								valValid[i] = 0;
								continue;
							}
							value[i] = (float) NnueSelfPlay.scoreToValue(nm.lastScore, role == 0, scale);
							valValid[i] = 1;
							final int n = done.incrementAndGet();
							if (log != null && n % 2000 == 0) {
								log.printf("  rietichettate %d (%.0f s)%n", n, (System.currentTimeMillis() - t0) / 1000.0);
							}
						}
					}
				}));
			}
			for (Future<?> f : fs) {
				f.get();
			}
		} finally {
			pool.shutdown();
		}
		if (log != null) {
			log.println("rietichettate " + done.get() + " posizioni a profondita' " + depth + (failed.get() > 0 ? ", " + failed.get() + " senza mosse (lasciate senza valore)" : "") + ", "
					+ (System.currentTimeMillis() - t0) / 1000.0 + " s");
		}
		NnueDataset r = d.withValues(value, valValid);
		if (!o.keepUnlabelled) {
			int k = 0;
			for (int i = 0; i < d.n; i++) {
				if (selected[i] == 1 && valValid[i] == 1) {
					k++;
				}
			}
			final int[] idx = new int[k];
			k = 0;
			for (int i = 0; i < d.n; i++) {
				if (selected[i] == 1 && valValid[i] == 1) {
					idx[k++] = i;
				}
			}
			r = r.subset(idx);
		}
		return r;
	}

	public static void main(String[] args) throws IOException {
		String in = null, out = null, eval = null;
		int depth = 4;
		double scale = 1000.0;
		boolean onlyMissing = false, fast = false;
		final Options opt = new Options();
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
				case "threads": opt.threads = Integer.parseInt(v); fast = true; break;
				case "prune": opt.pruning = PruningOptions.parse(v); fast = true; break;
				case "tt": opt.ttLog2 = Integer.parseInt(v); fast = true; break;
				case "chunk": opt.chunk = Integer.parseInt(v); fast = true; break;
				case "every": opt.every = Integer.parseInt(v); fast = true; break;
				case "keep": opt.keepUnlabelled = Boolean.parseBoolean(v); fast = true; break;
				case "skipsat": opt.skipSat = Double.parseDouble(v); fast = true; break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (IllegalArgumentException e) { // include NumberFormatException e PruningOptions.parse
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
			opt.onlyMissing = onlyMissing;
			final NnueDataset r;
			if (fast) {
				r = relabelFast(d, ev, depth, scale, opt, System.out);
			} else {
				r = relabel(d, ev, depth, scale, onlyMissing, System.out);
			}
			r.save(out);
			System.out.println("scritto " + out + ": " + r.describe());
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		} catch (Exception e) {
			System.out.println("ERRORE: " + e);
			System.exit(1);
		}
	}
}
