package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;

/**
 * Confronto a COPPIE di due valutatori NNUE (A contro B) con le regole vere ({@link GameTracker}: patta per ripetizione, nessuna mossa =
 * sconfitta, Re catturato / in fuga) e la stessa ricerca ({@code Negamax}, un thread, stessa configurazione per i due lati: cambia SOLO la
 * rete). Serve sia come FITNESS dell'algoritmo genetico ({@link GaTrainer}) sia come strumento di misura finale contro il riferimento.
 *
 * <h3>Coppia</h3>
 * Una coppia e' la stessa apertura casuale ({@code open} semi-mosse, determinata dal seme della coppia) giocata DUE volte a colori invertiti:
 * A Bianco contro B Nero, poi B Bianco contro A Nero. Il vantaggio del Nero (nell'auto-gioco del progetto il Nero vince circa il 78%)
 * e l'apertura si cancellano. Punteggio di A in una partita: 1 vittoria, 0,5 patta (o partita troncata al tetto di semi-mosse), 0 sconfitta; punteggio
 * della coppia = media delle due partite, in {0, 0,25, 0,5, 0,75, 1}. Due reti IDENTICHE giocano due partite identiche (motori
 * deterministici a profondita' fissa) e fanno quindi esattamente 0,5.
 *
 * <h3>Statistica</h3>
 * Le due partite di una coppia NON sono indipendenti (stessa apertura); le coppie si'. L'intervallo di confidenza e' percio' calcolato sulle
 * medie delle coppie (normale, {@code z} = 1,96 per il 95%): {@code media +- z * s / sqrt(n)} con {@code s} la deviazione standard campionaria. Con poche coppie
 * l'approssimazione normale e' ottimistica: {@link #verdict} rifiuta di concludere sotto {@code minPairs}.
 *
 * <h3>Determinismo</h3>
 * A profondita' fissa ({@code msPerMove = 0}) il risultato dipende solo dai semi, non dal numero di thread dell'arena ne' dal carico della macchina.
 * Con {@code msPerMove > 0} (tempo reale per mossa, margine di sicurezza 0) NO: due esecuzioni possono differire.
 *
 * <p>
 * Uso: {@code GaMatch a=SPEC b=SPEC [pairs=50] [depth=4] [ms=0] [open=6] [maxplies=200] [seed=1] [threads=1] [z=1.96] [margin=0.02] [minpairs=30] [require=superiore|noninferiore]}
 * dove SPEC e' {@code FILE.nnue} oppure {@code FILE.nnue+FILE.genome} (la rete della base con il genoma applicato).
 */
public final class GaMatch {

	private GaMatch() {
	}

	public static final String SUPERIORE = "SUPERIORE";
	public static final String NON_INFERIORE = "NON_INFERIORE";
	public static final String NON_DIMOSTRATO = "NON_DIMOSTRATO";
	public static final String INFERIORE = "INFERIORE";
	public static final String INSUFFICIENTE = "INSUFFICIENTE";

	public static final class Config {
		public int depth = 4;
		public long msPerMove = 0;
		public int openRandom = 6;
		public int maxPlies = 200;

		void validate() {
			if (depth < 1 || depth > 40 || msPerMove < 0 || openRandom < 0 || openRandom > 40 || maxPlies < 10 || maxPlies > 30000) {
				throw new IllegalArgumentException("configurazione non valida: depth=" + depth + " ms=" + msPerMove + " open=" + openRandom + " maxplies=" + maxPlies);
			}
		}

		NnueSelfPlay.Config toSelfPlay() {
			final NnueSelfPlay.Config c = new NnueSelfPlay.Config();
			c.depth = depth;
			c.msPerMove = msPerMove;
			c.eps = 0.0; // nessuna mossa casuale oltre all'apertura: le due partite di una coppia sono confrontabili
			c.openRandom = openRandom;
			c.maxPlies = maxPlies;
			c.games = 1;
			c.threads = 1;
			return c;
		}
	}

	/** Esito di una coppia dal punto di vista di A. */
	public static final class Pair {
		/** Punteggio di A con il Bianco e con il Nero, ciascuno in {0, 0.5, 1}. */
		public final double asWhite, asBlack;

		Pair(double asWhite, double asBlack) {
			this.asWhite = asWhite;
			this.asBlack = asBlack;
		}

		public double mean() {
			return 0.5 * (asWhite + asBlack);
		}
	}

	/** Gioca una coppia: stessa apertura (seme {@code seed}), A Bianco e poi A Nero. */
	public static Pair playPair(Evaluator a, Evaluator b, Config cfg, long seed) {
		cfg.validate();
		final NnueSelfPlay.Config sp = cfg.toSelfPlay();
		final NnueSelfPlay.Game g1 = NnueSelfPlay.playGame(a, b, sp, seed); // A Bianco
		final NnueSelfPlay.Game g2 = NnueSelfPlay.playGame(b, a, sp, seed); // A Nero
		return new Pair(0.5 * (g1.result.outcome + 1.0), 0.5 * (1.0 - g2.result.outcome));
	}

	/** Riepilogo di una serie di coppie. */
	public static final class Summary {
		public final int pairs;
		/** Partite (non coppie) vinte, patte, perse da A. */
		public final int wins, draws, losses;
		public final double mean, sd, se, lo, hi;

		Summary(int pairs, int wins, int draws, int losses, double mean, double sd, double se, double lo, double hi) {
			this.pairs = pairs;
			this.wins = wins;
			this.draws = draws;
			this.losses = losses;
			this.mean = mean;
			this.sd = sd;
			this.se = se;
			this.lo = lo;
			this.hi = hi;
		}

		@Override
		public String toString() {
			return String.format("%d coppie (%d partite): A +%d =%d -%d, punteggio %.1f%% [IC %.1f%% - %.1f%%], sd delle coppie %.3f", pairs, 2 * pairs, wins, draws, losses, 100 * mean, 100 * lo, 100 * hi, sd);
		}
	}

	/** Riepiloga {@code pairs} con intervallo {@code mean +- z*se} sulle medie delle coppie (con 0 o 1 coppia: sd = 0, intervallo degenere). */
	public static Summary summarize(List<Pair> pairs, double z) {
		final int n = pairs.size();
		int w = 0, d = 0, l = 0;
		double sum = 0;
		for (Pair p : pairs) {
			sum += p.mean();
			for (double s : new double[] { p.asWhite, p.asBlack }) {
				if (s > 0.75) {
					w++;
				} else if (s < 0.25) {
					l++;
				} else {
					d++;
				}
			}
		}
		final double mean = n == 0 ? 0.5 : sum / n;
		double var = 0;
		if (n > 1) {
			for (Pair p : pairs) {
				var += (p.mean() - mean) * (p.mean() - mean);
			}
			var /= (n - 1);
		}
		final double sd = Math.sqrt(var);
		final double se = n == 0 ? 0 : sd / Math.sqrt(n);
		return new Summary(n, w, d, l, mean, sd, se, mean - z * se, mean + z * se);
	}

	/**
	 * Verdetto sul confronto con il riferimento (B), da usare con {@code z=1.96}:
	 * <ul>
	 * <li>{@link #INSUFFICIENTE}: meno di {@code minPairs} coppie, non si conclude;</li>
	 * <li>{@link #SUPERIORE}: il limite inferiore dell'IC e' STRETTAMENTE sopra 0,5;</li>
	 * <li>{@link #INFERIORE}: il limite superiore e' sotto 0,5;</li>
	 * <li>{@link #NON_INFERIORE}: il limite inferiore non e' sotto {@code 0,5 - margin} (ma non e' sopra 0,5);</li>
	 * <li>{@link #NON_DIMOSTRATO}: nessuno dei precedenti.</li>
	 * </ul>
	 * Con sd = 0 (tutte le coppie a 0,5, p.es. due reti equivalenti) l'IC e' un punto a 0,5: mai SUPERIORE.
	 */
	public static String verdict(Summary s, double margin, int minPairs) {
		if (s.pairs < minPairs) {
			return INSUFFICIENTE;
		}
		if (s.lo > 0.5) {
			return SUPERIORE;
		}
		if (s.hi < 0.5) {
			return INFERIORE;
		}
		if (s.lo >= 0.5 - margin) {
			return NON_INFERIORE;
		}
		return NON_DIMOSTRATO;
	}

	private static final class DaemonFactory implements ThreadFactory {
		@Override
		public Thread newThread(Runnable r) {
			final Thread t = new Thread(r, "ga-match");
			t.setDaemon(true);
			return t;
		}
	}

	/**
	 * Gioca una coppia per ogni seme, su {@code threads} thread (una partita = un thread di ricerca). L'ordine dei risultati e' quello dei semi: a
	 * profondita' fissa il risultato e' identico per qualunque {@code threads}.
	 */
	public static List<Pair> playAll(final Evaluator a, final Evaluator b, final Config cfg, long[] seeds, int threads) {
		cfg.validate();
		final ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads), new DaemonFactory());
		try {
			final List<Future<Pair>> fs = new ArrayList<>();
			for (final long s : seeds) {
				fs.add(pool.submit(new Callable<Pair>() {
					@Override
					public Pair call() {
						return playPair(a, b, cfg, s);
					}
				}));
			}
			final List<Pair> out = new ArrayList<>();
			for (Future<Pair> f : fs) {
				try {
					out.add(f.get());
				} catch (Exception e) {
					throw new IllegalStateException("partita fallita: " + e, e);
				}
			}
			return out;
		} finally {
			pool.shutdownNow();
		}
	}

	/** Un valutatore da una SPEC: {@code FILE.nnue} oppure {@code FILE.nnue+FILE.genome}. */
	static NnueEvaluator evaluatorOf(String spec) throws IOException {
		final int plus = spec.indexOf('+');
		if (plus < 0) {
			return NnueEvaluator.fromFile(spec);
		}
		final NnueNetwork base = NnueNetwork.load(spec.substring(0, plus));
		return new NnueEvaluator(GaGenome.load(spec.substring(plus + 1), base).decode());
	}

	public static void main(String[] args) throws Exception {
		String a = null, b = null, require = null;
		int pairs = 50, threads = 1, minPairs = 30;
		long seed = 1;
		double z = 1.96, margin = 0.02;
		final Config cfg = new Config();
		for (String arg : args) {
			final int eq = arg.indexOf('=');
			if (eq < 0) {
				usage("argomento non valido (atteso chiave=valore): " + arg);
				return;
			}
			final String k = arg.substring(0, eq).toLowerCase(), v = arg.substring(eq + 1);
			try {
				switch (k) {
				case "a": a = v; break;
				case "b": b = v; break;
				case "pairs": pairs = Integer.parseInt(v); break;
				case "depth": cfg.depth = Integer.parseInt(v); break;
				case "ms": cfg.msPerMove = Long.parseLong(v); break;
				case "open": cfg.openRandom = Integer.parseInt(v); break;
				case "maxplies": cfg.maxPlies = Integer.parseInt(v); break;
				case "seed": seed = Long.parseLong(v); break;
				case "threads": threads = Integer.parseInt(v); break;
				case "z": z = Double.parseDouble(v); break;
				case "margin": margin = Double.parseDouble(v); break;
				case "minpairs": minPairs = Integer.parseInt(v); break;
				case "require": require = v.toLowerCase(); break;
				default:
					usage("opzione sconosciuta: " + k);
					return;
				}
			} catch (NumberFormatException e) {
				usage("valore non valido per " + k + ": " + v);
				return;
			}
		}
		if (a == null || b == null || pairs < 1 || threads < 1) {
			usage("servono a=, b=, pairs>=1, threads>=1");
			return;
		}
		try {
			cfg.validate();
		} catch (IllegalArgumentException e) {
			usage(e.getMessage());
			return;
		}
		final NnueEvaluator ea, eb;
		try {
			ea = evaluatorOf(a);
			eb = evaluatorOf(b);
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
			return;
		}
		final long[] seeds = new long[pairs];
		final java.util.Random r = new java.util.Random(seed * 1000003L + 11);
		for (int i = 0; i < pairs; i++) {
			seeds[i] = r.nextLong();
		}
		final long t0 = System.currentTimeMillis();
		final Summary s = summarize(playAll(ea, eb, cfg, seeds, threads), z);
		final String v = verdict(s, margin, minPairs);
		System.out.println("A=" + a + "  VS  B=" + b + "  (" + (cfg.msPerMove > 0 ? cfg.msPerMove + " ms/mossa reali, margine 0" : "profondita' fissa " + cfg.depth) + ", apertura casuale di " + cfg.openRandom
				+ " semi-mosse, seme " + seed + ", " + (System.currentTimeMillis() - t0) / 1000 + " s)");
		System.out.println(s);
		System.out.println("VERDETTO (z=" + z + ", margine di non inferiorita' " + margin + ", minimo " + minPairs + " coppie): " + v);
		if (require != null) {
			final boolean ok = require.equals("superiore") ? SUPERIORE.equals(v) : require.equals("noninferiore") ? (SUPERIORE.equals(v) || NON_INFERIORE.equals(v)) : false;
			if (!require.equals("superiore") && !require.equals("noninferiore")) {
				usage("require= vale superiore oppure noninferiore");
				return;
			}
			System.exit(ok ? 0 : 1);
		}
	}

	private static void usage(String msg) {
		System.out.println(msg);
		System.out.println("Uso: GaMatch a=SPEC b=SPEC [pairs=50] [depth=4] [ms=0] [open=6] [maxplies=200] [seed=1] [threads=1] [z=1.96] [margin=0.02] [minpairs=30] [require=superiore|noninferiore]");
		System.out.println("  SPEC = FILE.nnue | FILE.nnue+FILE.genome");
		System.exit(2);
	}
}
