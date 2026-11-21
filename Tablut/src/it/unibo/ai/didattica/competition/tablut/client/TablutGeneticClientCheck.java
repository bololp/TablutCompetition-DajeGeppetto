package it.unibo.ai.didattica.competition.tablut.client;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import com.google.gson.Gson;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MoveSearcher;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
import it.unibo.ai.didattica.competition.tablut.training.GaGenome;
import it.unibo.ai.didattica.competition.tablut.util.Configuration;
import it.unibo.ai.didattica.competition.tablut.util.StreamUtils;

/**
 * Verifica il giocatore genetico ({@link TablutGeneticClient}) rispetto ai vincoli del torneo:
 * <ol>
 * <li><b>Risoluzione del genoma</b>: genoma valido applicato; neutro = base; genoma mancante / corrotto / per un'altra base = AVVISO e rete di base (strict=false) oppure
 * errore (strict=true); base illeggibile = errore sempre.</li>
 * <li><b>Argomenti</b> passati al client di riferimento: posizionali intatti, {@code genome= base= strict=} tolti, {@code eval=nnue} aggiunto solo se manca.</li>
 * <li><b>Legalita'</b>: con reti evolute a caso la mossa scelta e' SEMPRE nell'insieme di {@code AshtonRules.generateMoves} (oracolo del progetto), con il turno giusto.</li>
 * <li><b>Regressione</b>: con il genoma neutro la ricerca da' lo stesso identico risultato (mossa, punteggio, nodi) di una ricerca con la rete di base.</li>
 * <li><b>Tempo</b>: con la STESSA costruzione di ricerca del client reale ({@code TablutNegamaxClient.buildSearcher}, 4 thread Lazy SMP) la mossa torna entro il timeout.
 * Con {@code -Dga.long=true} anche la prova completa a 60 s (dura ~1 minuto).</li>
 * <li><b>Nessun calcolo nel turno avversario</b>: dopo la mossa nessun thread Java consuma CPU (si misura la CPU di tutti i thread in una finestra di quiete; per
 * confronto la stessa misura DURANTE la ricerca deve vedere i thread helper al lavoro, cosi' la misura ha i denti).</li>
 * <li><b>Protocollo reale</b>: il client vero ({@code main}) in un sottoprocesso contro un finto server su localhost (porta 5800 per il Bianco, 5801 per il Nero, devono essere
 * libere) con lo scambio del server vero (lunghezza + UTF-8, stato JSON dopo OGNI mossa a entrambi): nome dichiarato, ogni risposta e' un'Action JSON legale, la risposta
 * arriva entro il timeout, il client termina con codice 0 allo stato terminale; con genoma corrotto gioca comunque legale dopo un avviso; con {@code strict=true} e genoma
 * assente esce con codice -1 senza connettersi.</li>
 * </ol>
 * Uso: {@code TablutGeneticClientCheck [pesi_nnue/trained_v6.nnue]} (se manca si usa una rete casuale 64x16 della stessa forma).
 */
public class TablutGeneticClientCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String what) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + what);
		}
	}

	static NnueNetwork base;
	static File baseFile, genomeFile, badGenomeFile, wrongBaseGenomeFile;
	static GaGenome evolvedGenome;

	static double[] randomGenes(int n, double scale, Random rnd) {
		final double[] g = new double[n];
		for (int i = 0; i < n; i++) {
			g[i] = Math.max(-1, Math.min(1, rnd.nextGaussian() * scale));
		}
		return g;
	}

	static boolean sameMove(Action a, Action b) {
		return a != null && b != null && a.getFrom().equalsIgnoreCase(b.getFrom()) && a.getTo().equalsIgnoreCase(b.getTo());
	}

	static boolean isLegal(State s, Action a) {
		for (Action m : AshtonRules.generateMoves(s)) {
			if (sameMove(m, a)) {
				return true;
			}
		}
		return false;
	}

	static void resolution() throws Exception {
		TablutGeneticClient.Resolved r = TablutGeneticClient.resolve(baseFile.getPath(), genomeFile.getPath(), false);
		check(r.genomeApplied && r.warnings.isEmpty() && GaGenome.fingerprint(r.network) == GaGenome.fingerprint(evolvedGenome.decode()) && GaGenome.fingerprint(r.network) != GaGenome.fingerprint(base),
				"genoma valido: rete evoluta (diversa dalla base), nessun avviso");
		final File neutral = File.createTempFile("neutral", ".genome");
		GaGenome.identity(GaGenome.Layout.of(base, "head,l1b")).save(neutral.getPath());
		r = TablutGeneticClient.resolve(baseFile.getPath(), neutral.getPath(), false);
		check(r.genomeApplied && GaGenome.fingerprint(r.network) == GaGenome.fingerprint(base) && r.description.contains("NEUTRO"), "genoma neutro: rete identica alla base, dichiarato nel banner");
		neutral.delete();
		r = TablutGeneticClient.resolve(baseFile.getPath(), null, false);
		check(!r.genomeApplied && r.warnings.size() == 1 && GaGenome.fingerprint(r.network) == GaGenome.fingerprint(base), "nessun genoma (non strict): avviso e base");
		r = TablutGeneticClient.resolve(baseFile.getPath(), "/nonexistent_dir_xyz/none.genome", false);
		check(!r.genomeApplied && r.warnings.size() == 1 && GaGenome.fingerprint(r.network) == GaGenome.fingerprint(base), "genoma inesistente (non strict): avviso e base");
		r = TablutGeneticClient.resolve(baseFile.getPath(), badGenomeFile.getPath(), false);
		check(!r.genomeApplied && r.warnings.size() == 1 && r.warnings.get(0).contains("checksum") && GaGenome.fingerprint(r.network) == GaGenome.fingerprint(base), "genoma corrotto (non strict): avviso (checksum) e base");
		r = TablutGeneticClient.resolve(baseFile.getPath(), wrongBaseGenomeFile.getPath(), false);
		check(!r.genomeApplied && r.warnings.size() == 1 && r.warnings.get(0).contains("un'altra rete di base"), "genoma di un'altra base (non strict): avviso e base");
		for (String g : new String[] { null, "/nonexistent_dir_xyz/none.genome", badGenomeFile.getPath(), wrongBaseGenomeFile.getPath() }) {
			try {
				TablutGeneticClient.resolve(baseFile.getPath(), g, true);
				check(false, "strict con genoma " + g + ": doveva fallire");
			} catch (IOException e) {
				check(true, "strict con genoma " + g + ": errore");
			}
		}
		try {
			TablutGeneticClient.resolve("/nonexistent_dir_xyz/base.nnue", genomeFile.getPath(), false);
			check(false, "base illeggibile: doveva fallire anche non strict");
		} catch (IOException e) {
			check(true, "base illeggibile: errore anche non strict");
		}
	}

	static void arguments() {
		String[] d = TablutGeneticClient.delegateArgs(new String[] { "BLACK", "30", "10.0.0.5", "genome=g.genome", "threads=2", "base=b.nnue", "strict=true", "margin=900" }, "/tmp/w.nnue");
		check(Arrays.equals(d, new String[] { "BLACK", "30", "10.0.0.5", "/tmp/w.nnue", "eval=nnue", "threads=2", "margin=900" }), "argomenti delegati: " + Arrays.toString(d));
		d = TablutGeneticClient.delegateArgs(new String[] { "WHITE" }, "/tmp/w.nnue");
		check(Arrays.equals(d, new String[] { "WHITE", "60", "localhost", "/tmp/w.nnue", "eval=nnue" }), "argomenti mancanti: timeout 60 e localhost di default (come il riferimento): " + Arrays.toString(d));
		d = TablutGeneticClient.delegateArgs(new String[] { "WHITE", "60", "localhost", "EVAL=nnue", "GENOME=x" }, "/tmp/w.nnue");
		check(Arrays.equals(d, new String[] { "WHITE", "60", "localhost", "/tmp/w.nnue", "EVAL=nnue" }), "eval= esplicito non duplicato, chiavi case-insensitive: " + Arrays.toString(d));
		check("x.g".equals(TablutGeneticClient.option(new String[] { "WHITE", "60", "h", "GENOME=x.g" }, "genome", null)) && "d".equals(TablutGeneticClient.option(new String[] { "WHITE" }, "base", "d")),
				"lettura delle opzioni");
	}

	static List<State> positions(long seed, int every, int max) {
		final TestPositions.Game g = TestPositions.playGame(new Random(seed), true, 90);
		final List<State> out = new ArrayList<>();
		for (int i = 0; i < g.positions.size() && out.size() < max; i += every) {
			out.add(g.positions.get(i));
		}
		return out;
	}

	static Negamax newNegamax(NnueNetwork net, int depth) {
		final Negamax n = new Negamax(new NnueEvaluator(net), depth, true, 1 << 16, true, Negamax.ORDER_RECOMMENDED, 1);
		n.setSafetyMarginMillis(0);
		return n;
	}

	static void legality() {
		final Random rnd = new Random(11);
		final GaGenome.Layout L = GaGenome.Layout.of(base, "head,l1b,ftb");
		int moves = 0, bothColours = 0;
		boolean white = false, black = false;
		for (int k = 0; k < 4; k++) {
			final NnueNetwork net = new GaGenome(L, randomGenes(L.size(), 0.6, rnd)).decode();
			for (State s : positions(100 + k, 3, 25)) {
				final Action a = newNegamax(net, 2).findBestMove(s, 600000);
				check(a != null && isLegal(s, a) && a.getTurn() == s.getTurn(), "mossa illegale o con turno sbagliato: " + a + " nel turno " + s.getTurn());
				moves++;
				white |= s.getTurn() == Turn.WHITE;
				black |= s.getTurn() == Turn.BLACK;
			}
		}
		bothColours = white && black ? 1 : 0;
		check(moves >= 60 && bothColours == 1, "legalita': " + moves + " mosse di reti evolute casuali, entrambi i colori");
	}

	static void regression() {
		final GaGenome neutral = GaGenome.identity(GaGenome.Layout.of(base, "head,l1b,ftb"));
		int n = 0;
		for (State s : positions(7, 3, 10)) {
			final Negamax a = newNegamax(neutral.decode(), 4), b = newNegamax(base, 4);
			final Action ma = a.findBestMove(s, 600000), mb = b.findBestMove(s, 600000);
			check(sameMove(ma, mb) && a.lastScore == b.lastScore && a.lastTotalNodeCount == b.lastTotalNodeCount && a.lastDepthReached == b.lastDepthReached,
					"neutro: stessa mossa/punteggio/nodi/profondita' della base (" + ma + " " + a.lastScore + " " + a.lastTotalNodeCount + " vs " + mb + " " + b.lastScore + " " + b.lastTotalNodeCount + ")");
			n++;
		}
		check(n >= 8, "regressione su " + n + " posizioni");
	}

	/** La costruzione della ricerca e' quella del client reale. */
	static MoveSearcher clientSearcher(NnueNetwork net, int threads, int timeout) {
		return TablutNegamaxClient.buildSearcher("negamax", null, new NnueEvaluator(net), threads, TablutNegamaxClient.defaultMarginMillis(timeout), Negamax.ORDER_RECOMMENDED, 18, false, true);
	}

	static long totalOtherThreadsCpuNanos(ThreadMXBean mx) {
		long sum = 0;
		final long me = Thread.currentThread().getId();
		for (long id : mx.getAllThreadIds()) {
			if (id != me) {
				final long t = mx.getThreadCpuTime(id);
				if (t > 0) {
					sum += t;
				}
			}
		}
		return sum;
	}

	static void timeAndQuiet(NnueNetwork net) throws Exception {
		final int timeout = 3;
		final MoveSearcher searcher = clientSearcher(net, 4, timeout);
		long worst = 0;
		int ok = 0;
		for (State s : positions(21, 9, 4)) {
			final FastState fs = FastState.fromState(s);
			final PositionHistory h = new PositionHistory();
			h.add(fs);
			final long t0 = System.nanoTime();
			final int mv = searcher.findBestMove(fs, timeout * 1000L, h);
			final long ms = (System.nanoTime() - t0) / 1_000_000;
			worst = Math.max(worst, ms);
			check(mv != Move.NONE && isLegal(s, Move.toAction(mv, s.getTurn())), "mossa legale entro il tempo");
			check(ms <= timeout * 1000L, "timeout " + timeout + " s con 4 thread: mossa in " + ms + " ms");
			ok++;
		}
		System.out.println("   tempo: " + ok + " mosse con timeout " + timeout + " s e 4 thread Lazy SMP, la piu' lenta " + worst + " ms (margine di sicurezza " + TablutNegamaxClient.defaultMarginMillis(timeout) + " ms)");

		// quiete: durante la ricerca gli helper consumano CPU, subito dopo no
		final ThreadMXBean mx = ManagementFactory.getThreadMXBean();
		if (!mx.isThreadCpuTimeSupported()) {
			System.out.println("   (CPU per thread non supportata da questa JVM: prova di quiete saltata)");
			return;
		}
		mx.setThreadCpuTimeEnabled(true);
		final State s = positions(33, 9, 2).get(1);
		final FastState fs = FastState.fromState(s);
		final PositionHistory h = new PositionHistory();
		h.add(fs);
		final long before = totalOtherThreadsCpuNanos(mx);
		searcher.findBestMove(fs, 1500, h);
		final long during = totalOtherThreadsCpuNanos(mx) - before;
		check(during > 100_000_000L, "la misura ha i denti: durante la ricerca gli helper consumano CPU (" + during / 1_000_000 + " ms)");
		Thread.sleep(300); // tempo di assestamento: gli helper si fermano entro ~2048 nodi
		final long q0 = totalOtherThreadsCpuNanos(mx);
		final long p0 = processCpuNanos();
		Thread.sleep(1500); // "turno dell'avversario"
		final long idle = totalOtherThreadsCpuNanos(mx) - q0;
		final long pidle = processCpuNanos() >= 0 ? processCpuNanos() - p0 : -1;
		check(idle < 50_000_000L, "turno avversario: i thread Java consumano " + idle / 1_000_000 + " ms di CPU in 1500 ms di quiete (soglia 50 ms)");
		if (pidle >= 0) {
			check(pidle < 450_000_000L, "turno avversario: l'intero processo consuma " + pidle / 1_000_000 + " ms di CPU in 1500 ms (soglia 450 ms: copre JIT/GC)");
		}
		int running = 0;
		for (Map.Entry<Thread, StackTraceElement[]> e : Thread.getAllStackTraces().entrySet()) {
			if (e.getKey().getState() == Thread.State.RUNNABLE) {
				for (StackTraceElement f : e.getValue()) {
					if (f.getClassName().endsWith("Negamax")) {
						running++;
					}
				}
			}
		}
		check(running == 0, "turno avversario: nessun thread esegue codice di Negamax (" + running + ")");
		System.out.println("   quiete: durante la ricerca " + during / 1_000_000 + " ms di CPU negli helper; nel turno avversario " + idle / 1_000_000 + " ms (thread Java), " + (pidle >= 0 ? pidle / 1_000_000 + " ms (processo)" : "processo n/d"));
		searcher.shutdown();
	}

	static long processCpuNanos() {
		try {
			final java.lang.management.OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
			// metodo dell'interfaccia pubblica com.sun.management.OperatingSystemMXBean (non e' nell'API Java 8 di --release: si chiama per riflessione)
			final java.lang.reflect.Method m = Class.forName("com.sun.management.OperatingSystemMXBean").getMethod("getProcessCpuTime");
			return ((Number) m.invoke(os)).longValue();
		} catch (Exception e) {
			return -1;
		}
	}

	/** Prova completa a 60 s (solo con -Dga.long=true): la mossa deve tornare entro 60 s con la configurazione di default del client. */
	static void sixtySeconds(NnueNetwork net) {
		final int timeout = 60;
		final MoveSearcher searcher = TablutNegamaxClient.buildSearcher("negamax", null, new NnueEvaluator(net), Negamax.defaultThreads(), TablutNegamaxClient.defaultMarginMillis(timeout),
				Negamax.ORDER_RECOMMENDED, 20, false, true);
		final State s = positions(33, 9, 2).get(1);
		final FastState fs = FastState.fromState(s);
		final PositionHistory h = new PositionHistory();
		h.add(fs);
		final long t0 = System.nanoTime();
		final int mv = searcher.findBestMove(fs, timeout * 1000L, h);
		final long ms = (System.nanoTime() - t0) / 1_000_000;
		System.out.println("   60 s: mossa " + Move.toString(mv) + " in " + ms + " ms con " + Negamax.defaultThreads() + " thread; " + searcher.describeLastSearch());
		check(mv != Move.NONE && ms < timeout * 1000L && ms > 40_000, "timeout 60 s: mossa in " + ms + " ms (attesi 40000..59999)");
		searcher.shutdown();
	}

	// ====================================================================== protocollo reale

	static final class Session {
		final List<Long> latencies = new ArrayList<>();
		final List<String> replies = new ArrayList<>();
		String name;
		int clientMoves, illegal;
		Integer exitCode;
		String output = "";
		Throwable serverError;
		boolean terminalSent;
	}

	static Session playSession(final Turn role, final int timeout, final int moves, final String... extra) throws Exception {
		final int port = role == Turn.WHITE ? Configuration.whitePort : Configuration.blackPort;
		final Session ses = new Session();
		final ServerSocket ss = new ServerSocket(port);
		final AtomicReference<Throwable> err = new AtomicReference<>();
		final Thread server = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					final Socket sock = ss.accept();
					final DataInputStream in = new DataInputStream(sock.getInputStream());
					final DataOutputStream out = new DataOutputStream(sock.getOutputStream());
					ses.name = StreamUtils.readString(in);
					final Gson gson = new Gson();
					final Random rnd = new Random(5);
					State s = TestPositions.initial();
					if (role == Turn.BLACK) { // il Bianco (avversario) apre: poi il server invia lo stato col turno del Nero
						final List<Action> w = AshtonRules.generateMoves(s);
						s = AshtonRules.applyMove(s, w.get(rnd.nextInt(w.size())));
					}
					StreamUtils.writeString(out, gson.toJson(s)); // come il server vero: lo stato iniziale/aggiornato va a entrambi
					while (ses.clientMoves < moves && !AshtonRules.isTerminal(s)) {
						final long t0 = System.nanoTime();
						final String reply = StreamUtils.readString(in);
						ses.latencies.add((System.nanoTime() - t0) / 1_000_000);
						ses.replies.add(reply);
						final Action a = gson.fromJson(reply, Action.class);
						a.setTurn(s.getTurn());
						ses.clientMoves++;
						if (!isLegal(s, a)) {
							ses.illegal++;
							break;
						}
						s = AshtonRules.applyMove(s, a);
						StreamUtils.writeString(out, gson.toJson(s)); // stato dopo la mossa del client (turno dell'avversario): il client resta in attesa
						if (AshtonRules.isTerminal(s)) {
							break;
						}
						final List<Action> opp = AshtonRules.generateMoves(s);
						if (opp.isEmpty()) {
							break;
						}
						Thread.sleep(150); // l'avversario "pensa": il client deve restare fermo nella read()
						s = AshtonRules.applyMove(s, opp.get(rnd.nextInt(opp.size())));
						StreamUtils.writeString(out, gson.toJson(s));
					}
					// chiusura come il server vero: stato terminale a entrambi
					final State end = TestPositions.initial();
					end.setTurn(Turn.DRAW);
					StreamUtils.writeString(out, gson.toJson(end));
					ses.terminalSent = true;
					Thread.sleep(3000); // il client deve uscire da solo (System.exit(0)) dopo aver letto lo stato terminale
				} catch (Throwable t) {
					err.set(t);
				}
			}
		});
		server.setDaemon(true);
		server.start();

		final List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.add("-Xmx512m");
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.client.TablutGeneticClient");
		cmd.add(role == Turn.WHITE ? "WHITE" : "BLACK");
		cmd.add(String.valueOf(timeout));
		cmd.add("localhost");
		cmd.addAll(Arrays.asList(extra));
		final ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.redirectErrorStream(true);
		final Process p = pb.start();
		final StringBuilder sb = new StringBuilder();
		final Thread reader = new Thread(new Runnable() {
			@Override
			public void run() {
				try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), "UTF-8"))) {
					String l;
					while ((l = r.readLine()) != null) {
						synchronized (sb) {
							sb.append(l).append('\n');
						}
					}
				} catch (IOException e) {
					// processo terminato
				}
			}
		});
		reader.setDaemon(true);
		reader.start();
		if (!p.waitFor(moves * (timeout + 3L) + 60, TimeUnit.SECONDS)) {
			p.destroyForcibly();
			ses.exitCode = null;
		} else {
			ses.exitCode = p.exitValue();
		}
		reader.join(2000);
		server.join(5000);
		ss.close();
		synchronized (sb) {
			ses.output = sb.toString();
		}
		ses.serverError = err.get();
		return ses;
	}

	static void protocol(Turn role, int timeout, int moves, boolean corruptedGenome) throws Exception {
		final String tag = role + (corruptedGenome ? " (genoma corrotto)" : "");
		final Session s = playSession(role, timeout, moves, "genome=" + (corruptedGenome ? badGenomeFile : genomeFile).getPath(), "base=" + baseFile.getPath(), "threads=1", "tt=16", "margin=1000");
		check(s.serverError == null, tag + ": errore nel finto server: " + s.serverError);
		check(s.name != null && !s.name.trim().isEmpty(), tag + ": il client dichiara un nome (" + s.name + ")");
		check(s.clientMoves == moves && s.illegal == 0, tag + ": " + s.clientMoves + " risposte, " + s.illegal + " illegali (attese " + moves + ", 0)");
		boolean json = true;
		for (String r : s.replies) {
			json &= r.contains("\"from\"") && r.contains("\"to\"");
		}
		check(json && !s.replies.isEmpty(), tag + ": ogni risposta e' un'Action JSON con from/to");
		long worst = 0;
		for (long l : s.latencies) {
			worst = Math.max(worst, l);
		}
		check(worst <= timeout * 1000L, tag + ": risposta piu' lenta " + worst + " ms > timeout " + timeout * 1000 + " ms");
		check(s.exitCode != null && s.exitCode == 0, tag + ": il client termina con codice 0 allo stato terminale (codice " + s.exitCode + ")");
		check(s.output.contains("Genetic client - ") && s.output.contains("PAREGGIO!"), tag + ": banner e messaggio di fine partita");
		if (corruptedGenome) {
			check(s.output.contains("ATTENZIONE: genoma") && s.output.contains("[FALLBACK/BASE]"), tag + ": avviso di fallback sulla base");
		} else {
			check(!s.output.contains("ATTENZIONE: genoma") && !s.output.contains("FALLBACK"), tag + ": nessun avviso con un genoma valido");
		}
		check(s.output.contains("In attesa della mossa dell'avversario"), tag + ": nel turno avversario il client resta in attesa (messaggio del riferimento)");
		System.out.println("   " + tag + ": " + s.clientMoves + " mosse legali, latenze (ms) " + s.latencies + ", uscita " + s.exitCode);
	}

	/** Costruzione del searcher come nel client reale: {@code prune=} spenta di default, accesa solo se richiesta; sessione del protocollo con opzione valida e non valida. */
	static void pruneOption(NnueNetwork net) throws Exception {
		final MoveSearcher def = TablutNegamaxClient.buildSearcher("negamax", null, new NnueEvaluator(net), 1, 300, Negamax.ORDER_RECOMMENDED, 16, false, true);
		check(def instanceof it.unibo.ai.didattica.competition.tablut.aiplayer.NegamaxSearcher, "buildSearcher di default: negamax");
		final MoveSearcher on = TablutNegamaxClient.buildSearcher("negamax", null, new NnueEvaluator(net), 1, 300, Negamax.ORDER_RECOMMENDED, 16, false, true,
				it.unibo.ai.didattica.competition.tablut.aiplayer.PruningOptions.parse("nmp,lmr"));
		check(on != null, "buildSearcher con potatura: costruito");
		// la ricerca con potatura gioca mosse legali dalle posizioni di prova, a profondita' fissa e con budget di tempo
		int ok = 0;
		for (State s : positions(61, 9, 3)) {
			final FastState fs = FastState.fromState(s);
			final PositionHistory h = new PositionHistory();
			h.add(fs);
			final int mv = on.findBestMove(fs, 700, h);
			ok += mv != Move.NONE && isLegal(s, Move.toAction(mv, s.getTurn())) ? 1 : 0;
		}
		check(ok == 3, "ricerca con prune=nmp,lmr dal builder del client: mosse legali (" + ok + "/3)");
	}

	static void protocolPrune(Turn role, int timeout, int moves, String pruneArg, boolean valid) throws Exception {
		final String tag = role + " prune=" + pruneArg;
		final Session s = playSession(role, timeout, moves, "genome=" + genomeFile.getPath(), "base=" + baseFile.getPath(), "threads=1", "tt=16", "margin=1000", "prune=" + pruneArg);
		check(s.serverError == null, tag + ": errore nel finto server: " + s.serverError);
		check(s.clientMoves == moves && s.illegal == 0, tag + ": " + s.clientMoves + " risposte, " + s.illegal + " illegali");
		long worst = 0;
		for (long l : s.latencies) {
			worst = Math.max(worst, l);
		}
		check(worst <= timeout * 1000L, tag + ": risposta piu' lenta " + worst + " ms > timeout " + timeout * 1000 + " ms");
		check(s.exitCode != null && s.exitCode == 0, tag + ": codice di uscita 0 (" + s.exitCode + ")");
		if (valid) {
			check(s.output.contains("potatura: nmp"), tag + ": il banner dichiara la potatura accesa");
		} else {
			check(!s.output.contains("potatura:"), tag + ": con un'opzione non valida la ricerca resta quella storica (nessuna potatura nel banner)");
			check(s.output.contains("prune") && s.output.contains("ATTENZIONE"), tag + ": avviso per l'opzione non valida");
		}
		System.out.println("   " + tag + ": " + s.clientMoves + " mosse legali, latenze (ms) " + s.latencies);
	}

	static void strictFailure() throws Exception {
		final List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.client.TablutGeneticClient");
		cmd.addAll(Arrays.asList("WHITE", "60", "localhost", "genome=/nonexistent_dir_xyz/none.genome", "base=" + baseFile.getPath(), "strict=true"));
		final ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.redirectErrorStream(true);
		final Process p = pb.start();
		final StringBuilder sb = new StringBuilder();
		try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), "UTF-8"))) {
			String l;
			while ((l = r.readLine()) != null) {
				sb.append(l).append('\n');
			}
		}
		final boolean done = p.waitFor(30, TimeUnit.SECONDS);
		check(done && p.exitValue() == 255 && sb.toString().contains("ERRORE") && !sb.toString().contains("Genetic client - "), "strict=true con genoma assente: esce con codice -1 e un ERRORE, senza avviare il client (" + (done ? p.exitValue() : -999) + ")");
	}

	public static void main(String[] args) throws Exception {
		final String v6 = args.length > 0 ? args[0] : "pesi_nnue/trained_v6.nnue";
		base = new File(v6).isFile() ? NnueNetwork.load(v6) : NnueNetwork.random(64, 16, 7);
		System.out.println("base di prova: " + (new File(v6).isFile() ? v6 : "rete casuale 64x16") + " (" + base.describe() + ")");
		final File dir = java.nio.file.Files.createTempDirectory("geneticclientcheck").toFile();
		try {
			baseFile = new File(dir, "base.nnue");
			base.save(baseFile.getPath());
			final GaGenome.Layout L = GaGenome.Layout.of(base, "head,l1b,ftb");
			evolvedGenome = new GaGenome(L, randomGenes(L.size(), 0.3, new Random(1)));
			genomeFile = new File(dir, "ok.genome");
			evolvedGenome.save(genomeFile.getPath());
			badGenomeFile = new File(dir, "bad.genome");
			final String t = evolvedGenome.toText();
			java.nio.file.Files.write(badGenomeFile.toPath(), t.replaceFirst("checksum [0-9a-f]{8}", "checksum 00000000").getBytes("UTF-8"));
			wrongBaseGenomeFile = new File(dir, "wrong.genome");
			final NnueNetwork other = NnueNetwork.random(base.hidden(), base.l2(), 4242);
			new GaGenome(GaGenome.Layout.of(other, "head,l1b,ftb"), randomGenes(L.size(), 0.3, new Random(2))).save(wrongBaseGenomeFile.getPath());

			resolution();
			arguments();
			legality();
			regression();
			timeAndQuiet(base);
			if (Boolean.getBoolean("ga.long")) {
				sixtySeconds(base);
			}
			pruneOption(base);
			strictFailure();
			protocolPrune(Turn.WHITE, 4, 3, "nmp,lmr,asp", true);
			protocolPrune(Turn.BLACK, 4, 2, "zzz", false);
			protocol(Turn.WHITE, 4, 4, false);
			protocol(Turn.BLACK, 4, 3, false);
			protocol(Turn.WHITE, 4, 2, true);
		} finally {
			final File[] fs = dir.listFiles();
			if (fs != null) {
				for (File f : fs) {
					f.delete();
				}
			}
			dir.delete();
		}
		if (failures == 0) {
			System.out.println("=== GENETIC CLIENT CHECK OK: " + checks + " verifiche ===");
		} else {
			System.out.println("=== FALLITI: " + failures + " su " + checks + " ===");
			System.exit(1);
		}
		System.exit(0); // pool di ricerca / thread dei test: nessuna attesa
	}
}
