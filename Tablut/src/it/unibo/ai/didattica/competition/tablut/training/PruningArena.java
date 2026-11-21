package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.PruningOptions;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Confronto a COPPIE fra due CONFIGURAZIONI DI RICERCA ({@link PruningOptions}) con la STESSA rete NNUE: misura se mossa nulla, LMR e aspirazione ({@code nmp},
 * {@code lmr}, {@code asp}) fanno giocare meglio o peggio della ricerca storica ({@code off}) a parita' di risorse. (Non confondere con {@link SearchArena}, preesistente,
 * che confronta algoritmi diversi, Negamax/MCTS.) Riusa {@link GaMatch#summarize} e {@link GaMatch#verdict}: stessa apertura casuale giocata a colori invertiti, regole vere
 * ({@link GameTracker}: ripetizione = patta, nessuna mossa = sconfitta), statistica sulle COPPIE, un thread di ricerca per lato.
 *
 * <h3>Parita' di risorse</h3>
 * Con {@code ms > 0} (consigliato: e' la condizione di gara) i due lati hanno lo stesso tempo REALE per mossa (margine di sicurezza 0): chi riduce meglio l'albero raggiunge
 * piu' profondita'. Con {@code ms = 0} si gioca a profondita' fissa (verifica solo che le tecniche non peggiorino a pari profondita': non misura il guadagno). I tempi reali
 * dipendono dal carico: partite concorrenti ({@code threads > 1}) al massimo una per core fisico.
 *
 * <h3>Accumulo fra esecuzioni</h3>
 * {@code out=FILE} aggiunge una riga {@code seme puntiBianco puntiNero} per coppia (punteggi di A); {@code combine=FILE} ricalcola il riepilogo dal file: una misura lunga si
 * spezza in sessioni brevi (usare semi diversi: {@code seed=}).
 *
 * <p>
 * Uso: {@code PruningArena net=FILE a=SPEC b=SPEC [pairs=20] [ms=200] [depth=4] [open=6] [maxplies=200] [seed=1] [threads=1] [out=FILE] [z=1.96] [margin=0.02] [minpairs=30]}
 * oppure {@code PruningArena combine=FILE [z=..] [margin=..] [minpairs=..]}. SPEC: {@code off}, {@code all}, {@code nmp,lmr}... ({@link PruningOptions#parse}).
 */
public final class PruningArena {

	private PruningArena() {
	}

	/** Partita: esito dal punto di vista del Bianco (+1, 0, -1; troncata al tetto di semi-mosse = 0). */
	static double playGame(Evaluator ev, PruningOptions white, PruningOptions black, int depth, long msPerMove, int openRandom, int maxPlies, long seed) {
		final Negamax nw = newSearcher(ev, white, depth, msPerMove), nb = newSearcher(ev, black, depth, msPerMove);
		final Random rnd = new Random(seed);
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		final GameTracker tracker = new GameTracker(s);
		final long budget = msPerMove > 0 ? msPerMove : 600000L;
		for (int ply = 0; ply < maxPlies && !AshtonRules.isTerminal(s) && !tracker.isRepetitionDraw(); ply++) {
			final List<Action> legal = AshtonRules.generateMoves(s);
			if (legal.isEmpty()) {
				break;
			}
			Action chosen;
			if (ply < openRandom) {
				chosen = legal.get(rnd.nextInt(legal.size()));
			} else {
				chosen = (s.getTurn() == Turn.WHITE ? nw : nb).findBestMove(s, budget, tracker.history());
				if (chosen == null) {
					chosen = legal.get(rnd.nextInt(legal.size()));
				}
			}
			s = tracker.apply(s, chosen);
		}
		final GameTracker.Result r = tracker.result(s);
		return GameTracker.LABEL_MAXPLIES.equals(r.label) ? 0.0 : r.outcome;
	}

	private static Negamax newSearcher(Evaluator ev, PruningOptions p, int depth, long msPerMove) {
		final Negamax n = new Negamax(ev, msPerMove > 0 ? 64 : depth, true, 1 << 20, true, Negamax.ORDER_RECOMMENDED, 1);
		n.setSafetyMarginMillis(0);
		n.setPruning(p);
		return n;
	}

	/** Una coppia: A Bianco contro B Nero e poi B Bianco contro A Nero, stessa apertura. */
	public static GaMatch.Pair playPair(Evaluator ev, PruningOptions a, PruningOptions b, int depth, long msPerMove, int openRandom, int maxPlies, long seed) {
		final double g1 = playGame(ev, a, b, depth, msPerMove, openRandom, maxPlies, seed); // A Bianco
		final double g2 = playGame(ev, b, a, depth, msPerMove, openRandom, maxPlies, seed); // A Nero
		return new GaMatch.Pair(0.5 * (g1 + 1.0), 0.5 * (1.0 - g2));
	}

	private static final class DaemonFactory implements ThreadFactory {
		@Override
		public Thread newThread(Runnable r) {
			final Thread t = new Thread(r, "pruning-arena");
			t.setDaemon(true);
			return t;
		}
	}

	static List<GaMatch.Pair> readPairs(String path) throws IOException {
		final List<GaMatch.Pair> out = new ArrayList<>();
		final File f = new File(path);
		if (!f.isFile()) {
			return out;
		}
		try (BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"))) {
			String line;
			while ((line = r.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty()) {
					continue;
				}
				final String[] t = line.split("\\s+");
				if (t.length != 3) {
					throw new IOException("riga non valida nel file delle coppie: '" + line + "'");
				}
				final double w, b;
				try {
					w = Double.parseDouble(t[1]);
					b = Double.parseDouble(t[2]);
				} catch (NumberFormatException e) {
					throw new IOException("numero non valido in '" + line + "'");
				}
				if (!(w == 0.0 || w == 0.5 || w == 1.0) || !(b == 0.0 || b == 0.5 || b == 1.0)) {
					throw new IOException("punteggio non valido (ammessi 0, 0.5, 1) in '" + line + "'");
				}
				out.add(new GaMatch.Pair(w, b));
			}
		}
		return out;
	}

	public static void main(String[] args) throws Exception {
		String net = null, a = "all", b = "off", out = null, combine = null;
		int pairs = 20, depth = 4, open = 6, maxPlies = 200, threads = 1, minPairs = 30;
		long ms = 200, seed = 1;
		double z = 1.96, margin = 0.02;
		for (String arg : args) {
			final int eq = arg.indexOf('=');
			if (eq < 0) {
				usage("argomento non valido (atteso chiave=valore): " + arg);
				return;
			}
			final String k = arg.substring(0, eq).toLowerCase(), v = arg.substring(eq + 1);
			try {
				switch (k) {
				case "net": net = v; break;
				case "a": a = v; break;
				case "b": b = v; break;
				case "pairs": pairs = Integer.parseInt(v); break;
				case "ms": ms = Long.parseLong(v); break;
				case "depth": depth = Integer.parseInt(v); break;
				case "open": open = Integer.parseInt(v); break;
				case "maxplies": maxPlies = Integer.parseInt(v); break;
				case "seed": seed = Long.parseLong(v); break;
				case "threads": threads = Integer.parseInt(v); break;
				case "out": out = v; break;
				case "combine": combine = v; break;
				case "z": z = Double.parseDouble(v); break;
				case "margin": margin = Double.parseDouble(v); break;
				case "minpairs": minPairs = Integer.parseInt(v); break;
				default:
					usage("opzione sconosciuta: " + k);
					return;
				}
			} catch (NumberFormatException e) {
				usage("valore non valido per " + k + ": " + v);
				return;
			}
		}
		if (combine != null) {
			final GaMatch.Summary s = GaMatch.summarize(readPairs(combine), z);
			System.out.println("file " + combine + ": " + s);
			System.out.println("VERDETTO (z=" + z + ", margine " + margin + ", minimo " + minPairs + " coppie): " + GaMatch.verdict(s, margin, minPairs));
			return;
		}
		if (net == null || pairs < 1 || threads < 1 || depth < 1 || ms < 0 || open < 0 || maxPlies < 10) {
			usage("serve net=, pairs>=1, threads>=1, depth>=1, ms>=0, open>=0, maxplies>=10");
			return;
		}
		final PruningOptions pa, pb;
		try {
			pa = PruningOptions.parse(a);
			pb = PruningOptions.parse(b);
		} catch (IllegalArgumentException e) {
			usage("specifica non valida: " + e.getMessage());
			return;
		}
		final NnueEvaluator ev = NnueEvaluator.fromFile(net);
		final Random r = new Random(seed * 1000003L + 11);
		final long[] seeds = new long[pairs];
		for (int i = 0; i < pairs; i++) {
			seeds[i] = r.nextLong();
		}
		final long t0 = System.currentTimeMillis();
		final ExecutorService pool = Executors.newFixedThreadPool(threads, new DaemonFactory());
		final List<GaMatch.Pair> res = new ArrayList<>();
		try {
			final List<Future<GaMatch.Pair>> fs = new ArrayList<>();
			final int dd = depth, oo = open, mp = maxPlies;
			final long mm = ms;
			for (final long sd : seeds) {
				fs.add(pool.submit(new Callable<GaMatch.Pair>() {
					@Override
					public GaMatch.Pair call() {
						return playPair(ev, pa, pb, dd, mm, oo, mp, sd);
					}
				}));
			}
			Writer w = null;
			try {
				if (out != null) {
					final File of = new File(out).getAbsoluteFile();
					if (of.getParentFile() != null) {
						of.getParentFile().mkdirs();
					}
					w = new OutputStreamWriter(new FileOutputStream(of, true), "UTF-8");
				}
				for (int i = 0; i < fs.size(); i++) {
					final GaMatch.Pair p = fs.get(i).get();
					res.add(p);
					if (w != null) {
						w.write(seeds[i] + " " + p.asWhite + " " + p.asBlack + "\n");
						w.flush();
					}
				}
			} finally {
				if (w != null) {
					w.close();
				}
			}
		} finally {
			pool.shutdownNow();
		}
		final GaMatch.Summary s = GaMatch.summarize(res, z);
		System.out.println("A=" + a + "  VS  B=" + b + "  (" + (ms > 0 ? ms + " ms reali/mossa, 1 thread per lato" : "profondita' fissa " + depth) + ", apertura casuale " + open + " semi-mosse, seme " + seed + ", "
				+ (System.currentTimeMillis() - t0) / 1000 + " s)");
		System.out.println(s);
		System.out.println("VERDETTO (z=" + z + ", margine " + margin + ", minimo " + minPairs + " coppie): " + GaMatch.verdict(s, margin, minPairs));
	}

	private static void usage(String msg) {
		System.out.println(msg);
		System.out.println("Uso: PruningArena net=FILE a=SPEC b=SPEC [pairs=20] [ms=200] [depth=4] [open=6] [maxplies=200] [seed=1] [threads=1] [out=FILE] [z=1.96] [margin=0.02] [minpairs=30]");
		System.out.println("     PruningArena combine=FILE [z=1.96] [margin=0.02] [minpairs=30]");
		System.out.println("  SPEC = off | all | nmp | lmr | asp | elenco con parametri (si veda PruningOptions)");
		System.exit(2);
	}
}
