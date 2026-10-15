package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Allena un {@link CnnEvaluator} con self-play e TD(lambda) semi-gradiente: LO STESSO schema di {@link SelfPlayTrainer} (TD-Gammon,
 * eligibility trace, nessun replay buffer, avversari {@link SimpleEvaluator} e checkpoint storici contro il forgetting, budget di
 * TEMPO casuale per mossa invece di profondita' fissa, generazione parallela con copie congelate dei pesi e aggiornamento
 * sequenziale), con {@link ConvNet} al posto dell'MLP. Si veda il javadoc di {@code SelfPlayTrainer} per le motivazioni (non
 * ripetute qui). Le regole di fine partita sono quelle di {@link GameTracker} (stallo = sconfitta, ripetizione = pareggio).
 *
 * <h3>Cosa e' diverso dal trainer dell'MLP</h3>
 * <ul>
 * <li>La traiettoria memorizza scacchiera (81 codici) e feature a mano, non un vettore di feature.</li>
 * <li>{@code alpha} di default e' <b>0,002</b> (0,02 per l'MLP): il CNN ha ~8000 parametri contro ~1400 e la traccia somma i gradienti
 * di molte posizioni; con 0,02 il primo TD rischia di rovinare i pesi pre-addestrati. E' un default prudente, non tarato: si
 * controlli {@code avgTD} nel log e si confronti con {@code EqualTimeArena} contro il punto di partenza.</li>
 * <li>Nessuna augmentation per simmetria nel TD (le posizioni arrivano gia' nell'ordine della partita e la traccia le lega): le
 * 8 simmetrie le ha gia' viste il pre-addestramento ({@link CnnPretrainer}).</li>
 * <li>Gli avversari storici possono essere CNN o MLP ({@link EvaluatorLoader}).</li>
 * </ul>
 *
 * <p>
 * Argomenti {@code chiave=valore}, tutti opzionali: episodes(3000), lambda(0.7), alpha(0.002), epsilonStart(0.1), epsilonEnd(0.02),
 * minMoveMs(100), maxMoveMs(1500), maxSearchDepthCap(8), opponentPoolProb(0.3), historicalOpponentProb(0.2),
 * opponentTimeMultiplier(1.0), checkpointEvery(500), checkpointDir(checkpoints_cnn), evalGames(10), evalDepths(2,3),
 * log(training_log_cnn.csv), out(cnn_trained.txt), resumeFrom(nessuno), startEpisode(0), totalPlanned(startEpisode+episodes),
 * threads(tutti i core), seed. Con rete nuova (senza resumeFrom): conv(4), reduce(2), dense(32), head(8), features(1), groups(G_DEFAULT).
 * <b>Consiglio</b>: partire sempre da {@code resumeFrom=<pesi di CnnPretrainer>}.
 */
public class CnnSelfPlayTrainer {

	private static final int MAX_PLIES_PER_EPISODE = 300;

	/** Una partita registrata: le posizioni in cui si doveva muovere, chi muoveva, l'esito. Dato immutabile, sicuro fra thread. */
	static final class Traj {
		final List<byte[]> cells;
		final List<float[]> feats;
		final int[] roles;
		final double outcome;
		final int plies;
		final String outcomeLabel;
		final Turn opponentControls;
		final String opponentType;
		final double avgDepthReached;
		State finalState; // stato finale della partita (per i test sull'esito)

		Traj(List<byte[]> cells, List<float[]> feats, int[] roles, double outcome, int plies, String outcomeLabel, Turn opponentControls,
				String opponentType, double avgDepthReached) {
			this.cells = cells;
			this.feats = feats;
			this.roles = roles;
			this.outcome = outcome;
			this.plies = plies;
			this.outcomeLabel = outcomeLabel;
			this.opponentControls = opponentControls;
			this.opponentType = opponentType;
			this.avgDepthReached = avgDepthReached;
		}
	}

	public static void main(String[] args) throws Exception {
		Map<String, String> p = new HashMap<>();
		for (String a : args) {
			int eq = a.indexOf('=');
			if (eq < 0) {
				throw new IllegalArgumentException("Argomento non valido (atteso chiave=valore): " + a);
			}
			p.put(a.substring(0, eq), a.substring(eq + 1));
		}
		int episodes = Integer.parseInt(p.getOrDefault("episodes", "3000"));
		double lambda = Double.parseDouble(p.getOrDefault("lambda", "0.7"));
		double alpha = Double.parseDouble(p.getOrDefault("alpha", "0.002"));
		double epsilonStart = Double.parseDouble(p.getOrDefault("epsilonStart", "0.1"));
		double epsilonEnd = Double.parseDouble(p.getOrDefault("epsilonEnd", "0.02"));
		long minMoveMs = Long.parseLong(p.getOrDefault("minMoveMs", "100"));
		long maxMoveMs = Long.parseLong(p.getOrDefault("maxMoveMs", "1500"));
		int maxSearchDepthCap = Integer.parseInt(p.getOrDefault("maxSearchDepthCap", "8"));
		double baselineProb = Double.parseDouble(p.getOrDefault("opponentPoolProb", "0.3"));
		double historicalProb = Double.parseDouble(p.getOrDefault("historicalOpponentProb", "0.2"));
		double oppTime = Double.parseDouble(p.getOrDefault("opponentTimeMultiplier", "1.0"));
		int checkpointEvery = Integer.parseInt(p.getOrDefault("checkpointEvery", "500"));
		String checkpointDir = p.getOrDefault("checkpointDir", "checkpoints_cnn");
		int evalGames = Integer.parseInt(p.getOrDefault("evalGames", "10"));
		int[] evalDepths = ConvNet.parseList(p.getOrDefault("evalDepths", "2,3"));
		String logFile = p.getOrDefault("log", "training_log_cnn.csv");
		String outFile = p.getOrDefault("out", "cnn_trained.txt");
		String resumeFrom = p.get("resumeFrom");
		int startEpisode = Integer.parseInt(p.getOrDefault("startEpisode", "0"));
		int totalPlanned = Integer.parseInt(p.getOrDefault("totalPlanned", String.valueOf(startEpisode + episodes)));
		int numThreads = Integer.parseInt(p.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
		long seed = Long.parseLong(p.getOrDefault("seed", String.valueOf(System.currentTimeMillis())));

		CnnEvaluator evaluator;
		if (resumeFrom != null) {
			evaluator = CnnEvaluator.fromFile(resumeFrom); // IOException chiara se il file e' di un altro tipo/versione
		} else {
			boolean hybrid = !p.getOrDefault("features", "1").equals("0");
			int groups = Integer.parseInt(p.getOrDefault("groups", String.valueOf(TablutFeatures.G_DEFAULT)));
			int nFeat = hybrid ? TablutFeatures.columnsFor(groups).length : 0;
			ConvNet.Config cfg = new ConvNet.Config(ConvNet.parseList(p.getOrDefault("conv", "4")),
					Integer.parseInt(p.getOrDefault("reduce", "2")), nFeat, ConvNet.parseList(p.getOrDefault("dense", "32")),
					ConvNet.parseList(p.getOrDefault("head", "8")));
			evaluator = new CnnEvaluator(new ConvNet(cfg, seed), hybrid ? groups : 0);
		}
		final ConvNet net = evaluator.net;
		final int groups = evaluator.featureGroups;
		System.out.println("=== CnnSelfPlayTrainer === episodes=" + episodes + " lambda=" + lambda + " alpha=" + alpha + " out=" + outFile
				+ " seed=" + seed + " opponentPoolProb=" + baselineProb + " historicalOpponentProb=" + historicalProb + " threads=" + numThreads
				+ (resumeFrom != null ? " | RESUME da " + resumeFrom + " (startEpisode=" + startEpisode + ", totalPlanned=" + totalPlanned + ")"
						: " | rete nuova (pesi casuali: sconsigliato, si parta da CnnPretrainer)"));
		System.out.println("Rete: " + net.config() + " | " + net.numParams() + " parametri, ~" + net.macsPerEval() + " MAC/valutazione");

		Random rnd = new Random(seed);
		SimpleEvaluator baseline = new SimpleEvaluator();
		File ckDir = new File(checkpointDir);
		ckDir.mkdirs();
		List<String> checkpointHistory = new ArrayList<>();
		File[] existing = ckDir.listFiles((dir, name) -> name.startsWith("ckpt_") && name.endsWith(".txt"));
		if (existing != null) {
			for (File f : existing) {
				try {
					EvaluatorLoader.load(f.getPath()); // solo i checkpoint leggibili con le feature attuali
					checkpointHistory.add(f.getPath());
				} catch (IOException e) {
					// ignorato: formato/versione non riutilizzabile
				}
			}
			checkpointHistory.sort(String::compareTo); // nomi zero-padded: ordine alfabetico = cronologico
			if (!checkpointHistory.isEmpty()) {
				System.out.println("Recuperati " + checkpointHistory.size() + " checkpoint storici da " + checkpointDir);
			}
		}

		boolean append = resumeFrom != null && new File(logFile).exists();
		final Map<String, Integer> endCounts = new TreeMap<>();
		ExecutorService pool = Executors.newFixedThreadPool(numThreads);
		try (PrintWriter log = new PrintWriter(new FileWriter(logFile, append))) {
			if (!append) {
				log.println("episode,avgAbsTdError,plies,outcome,epsilon,opponent,avgDepthReached,opponentType");
			}
			int done = 0;
			while (done < episodes) {
				int roundSize = Math.min(numThreads, episodes - done);
				// FASE 1: generazione in parallelo, ogni task con la PROPRIA copia congelata dei pesi attuali
				List<String> histSnapshot = new ArrayList<>(checkpointHistory);
				List<Callable<Traj>> tasks = new ArrayList<>();
				for (int i = 0; i < roundSize; i++) {
					int cumulativeEp = startEpisode + done + i + 1;
					double epsilon = epsilonSchedule(cumulativeEp, totalPlanned, epsilonStart, epsilonEnd);
					tasks.add(new GenerationTask(net.copy(), groups, baseline, histSnapshot, maxSearchDepthCap, epsilon, minMoveMs, maxMoveMs,
							oppTime, baselineProb, historicalProb, rnd.nextLong()));
				}
				List<Future<Traj>> futures = pool.invokeAll(tasks);

				// FASE 2: TD(lambda) sequenziale sui pesi VERI
				for (int i = 0; i < roundSize; i++) {
					int cumulativeEp = startEpisode + done + i + 1;
					Traj traj = futures.get(i).get();
					double avgTd = trainOnTrajectory(net, traj, lambda, alpha);
					endCounts.merge(traj.outcomeLabel, 1, Integer::sum);
					double epsLog = epsilonSchedule(cumulativeEp, totalPlanned, epsilonStart, epsilonEnd);
					log.println(cumulativeEp + "," + avgTd + "," + traj.plies + "," + traj.outcomeLabel + "," + epsLog + ","
							+ (traj.opponentControls == null ? "none" : traj.opponentControls) + "," + traj.avgDepthReached + "," + traj.opponentType);
					log.flush();
					if (cumulativeEp % 100 == 0) {
						evaluator.save(outFile);
						System.out.printf("episodio %6d (cumulativo) | profondita' media=%.1f | epsilon=%.3f | avgTD=%.4f | mosse=%3d | esito=%s%n",
								cumulativeEp, traj.avgDepthReached, epsLog, avgTd, traj.plies, traj.outcomeLabel);
						System.out.println("    esiti finora: " + endCounts);
					}
					if (cumulativeEp % checkpointEvery == 0) {
						String ck = String.format("%s/ckpt_%08d.txt", checkpointDir, cumulativeEp);
						evaluator.save(ck);
						checkpointHistory.add(ck);
						System.out.println("  -- valutazione a profondita' fisse (diagnostica) --");
						for (int d : evalDepths) {
							SelfPlayTrainer.evaluateAgainstBaseline(evaluator, baseline, evalGames, d, rnd);
						}
						if (checkpointHistory.size() > 1) {
							String oldest = checkpointHistory.get(0);
							try {
								Evaluator oldSelf = EvaluatorLoader.load(oldest);
								System.out.println("  -- confronto col proprio passato (rilevatore di forgetting) --");
								SelfPlayTrainer.evaluateAgainstOpponent(evaluator, oldSelf, "se stessa a " + new File(oldest).getName(),
										evalGames, evalDepths.length > 0 ? evalDepths[0] : 2, rnd);
							} catch (IOException e) {
								System.out.println("  [attenzione] checkpoint storico non leggibile: " + oldest);
							}
						}
					}
				}
				done += roundSize;
			}
			evaluator.save(outFile);
		} finally {
			pool.shutdown();
		}
		System.out.println("Esiti delle partite di questa esecuzione: " + endCounts);
		System.out.println("Training completato. Pesi salvati in " + outFile + " (log in " + logFile + "). Episodi totali visti da questa rete: "
				+ (startEpisode + episodes));
	}

	/** Epsilon decresce linearmente da {@code start} a {@code end} nel primo 70% di {@code totalPlanned}, poi resta fisso. */
	static double epsilonSchedule(int episode, int totalPlanned, double start, double end) {
		double frac = Math.min(1.0, episode / Math.max(1.0, totalPlanned * 0.7));
		return start + (end - start) * frac;
	}

	/**
	 * Fase 2: TD(lambda) sui pesi di {@code net}, una partita alla volta (chiamata solo dal thread principale). Per ogni posizione s_t:
	 * {@code trace = lambda*trace + dV(s_t)/dw}; bersaglio {@code V(s_{t+1})} (stessi pesi, aggiornati solo DOPO) oppure l'esito
	 * finale; {@code w += alpha * (bersaglio - V(s_t)) * trace}.
	 *
	 * @return media del valore assoluto dell'errore TD
	 */
	static double trainOnTrajectory(ConvNet net, Traj traj, double lambda, double alpha) {
		final float[] trace = new float[net.numParams()];
		final ConvNet.Workspace ws = net.newWorkspace();
		final float lam = (float) lambda;
		double sumAbsTd = 0;
		final int steps = traj.cells.size();
		for (int i = 0; i < steps; i++) {
			ws.setCodes(traj.cells.get(i), 0);
			final double v = net.forward(ws, traj.feats.get(i), traj.roles[i]);
			for (int k = 0; k < trace.length; k++) {
				trace[k] *= lam;
			}
			net.backward(ws, 1.0, trace); // ACCUMULA: subito dopo il forward di s_t e prima del forward del bersaglio
			final double target;
			if (i + 1 < steps) {
				ws.setCodes(traj.cells.get(i + 1), 0);
				target = net.forward(ws, traj.feats.get(i + 1), traj.roles[i + 1]);
			} else {
				target = traj.outcome;
			}
			final double delta = target - v;
			sumAbsTd += Math.abs(delta);
			net.addScaled(trace, alpha * delta);
		}
		return steps > 0 ? sumAbsTd / steps : 0;
	}

	/** Fase 1: genera UNA partita con una copia privata dei pesi; non tocca mai la rete condivisa. */
	static class GenerationTask implements Callable<Traj> {
		private final ConvNet myNet;
		private final int groups;
		private final SimpleEvaluator baseline;
		private final List<String> checkpoints;
		private final int depthCap;
		private final double epsilon;
		private final long minMoveMs, maxMoveMs;
		private final double oppTimeMultiplier, baselineProb, historicalProb;
		private final Random rnd;

		GenerationTask(ConvNet myNet, int groups, SimpleEvaluator baseline, List<String> checkpoints, int depthCap, double epsilon,
				long minMoveMs, long maxMoveMs, double oppTimeMultiplier, double baselineProb, double historicalProb, long seed) {
			this.myNet = myNet;
			this.groups = groups;
			this.baseline = baseline;
			this.checkpoints = checkpoints;
			this.depthCap = depthCap;
			this.epsilon = epsilon;
			this.minMoveMs = minMoveMs;
			this.maxMoveMs = maxMoveMs;
			this.oppTimeMultiplier = oppTimeMultiplier;
			this.baselineProb = baselineProb;
			this.historicalProb = historicalProb;
			this.rnd = new Random(seed);
		}

		@Override
		public Traj call() {
			final CnnEvaluator mine = new CnnEvaluator(myNet, groups);
			final TablutFeatures.Workspace fws = TablutFeatures.newWorkspace();
			final double[] full = new double[TablutFeatures.NUM_FEATURES];
			final int nFeat = myNet.config().nFeat;
			final Negamax mySearch = new Negamax(mine, depthCap, true, 1 << 20).setThreads(1);

			Turn oppControls = null;
			Negamax oppSearch = null;
			String oppType = "self";
			double r = rnd.nextDouble();
			if (r < baselineProb) {
				oppControls = rnd.nextBoolean() ? Turn.WHITE : Turn.BLACK;
				oppSearch = new Negamax(baseline, depthCap, true, 1 << 20).setThreads(1);
				oppType = "baseline";
			} else if (r < baselineProb + historicalProb && !checkpoints.isEmpty()) {
				String f = checkpoints.get(rnd.nextInt(checkpoints.size()));
				try {
					Evaluator hist = EvaluatorLoader.load(f);
					oppControls = rnd.nextBoolean() ? Turn.WHITE : Turn.BLACK;
					oppSearch = new Negamax(hist, depthCap, true, 1 << 20).setThreads(1);
					oppType = "historical";
				} catch (IOException e) {
					// checkpoint rimosso o corrotto: questo episodio diventa self-play puro
				}
			}

			State start = new StateTablut();
			start.setTurn(Turn.WHITE);
			final List<byte[]> cells = new ArrayList<>();
			final List<float[]> feats = new ArrayList<>();
			final List<Integer> roleList = new ArrayList<>();
			final long[] depthAcc = new long[2];
			final Turn oppTurn = oppControls;
			final Negamax oppS = oppSearch;

			SelfPlayTrainer.PositionListener sampler = new SelfPlayTrainer.PositionListener() {
				@Override
				public void onPosition(State s) {
					cells.add(CnnPretrainer.codesOf(s));
					float[] f = new float[nFeat];
					if (nFeat > 0) {
						mine.extractFeatures(s, fws, full, f);
					}
					feats.add(f);
					roleList.add(NeuralEvaluator.roleOf(s));
				}
			};
			SelfPlayTrainer.MoveChooser chooser = new SelfPlayTrainer.MoveChooser() {
				@Override
				public Action choose(State s, List<Long> history) {
					List<Action> legal = AshtonRules.generateMoves(s);
					long budget = minMoveMs + (maxMoveMs > minMoveMs ? Math.abs(rnd.nextLong() % (maxMoveMs - minMoveMs + 1)) : 0);
					Action move;
					if (rnd.nextDouble() < epsilon) {
						move = legal.get(rnd.nextInt(legal.size()));
					} else if (s.getTurn() == oppTurn) {
						move = oppS.findBestMove(s, (long) (budget * oppTimeMultiplier), history);
						depthAcc[0] += oppS.lastDepthReached;
						depthAcc[1]++;
					} else {
						move = mySearch.findBestMove(s, budget, history);
						depthAcc[0] += mySearch.lastDepthReached;
						depthAcc[1]++;
					}
					return move != null ? move : legal.get(rnd.nextInt(legal.size()));
				}
			};
			SelfPlayTrainer.GameOutcome game = SelfPlayTrainer.playGame(start, MAX_PLIES_PER_EPISODE, sampler, chooser, false);
			int[] roles = new int[roleList.size()];
			for (int i = 0; i < roles.length; i++) {
				roles[i] = roleList.get(i);
			}
			double avgDepth = depthAcc[1] > 0 ? depthAcc[0] / (double) depthAcc[1] : 0;
			Traj t = new Traj(cells, feats, roles, game.result.outcome, game.plies, game.result.label, oppControls, oppType, avgDepth);
			t.finalState = game.finalState;
			return t;
		}
	}
}
