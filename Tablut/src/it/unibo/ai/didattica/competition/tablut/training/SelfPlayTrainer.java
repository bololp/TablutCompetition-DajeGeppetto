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
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Allena {@link NeuralEvaluator} tramite self-play e TD(lambda) semi-gradient
 * (lo stesso schema usato da TD-Gammon: Sutton &amp; Barto, "Reinforcement
 * Learning: An Introduction", cap. 12 per le eligibility trace).
 * <p>
 * Idea generale: si gioca una partita della rete CONTRO SE STESSA (usando
 * {@link Negamax} a profondita' ridotta per restare veloci - il training
 * richiede migliaia di partite, non ha senso cercare a profondita' 5 come in
 * torneo). Ad ogni mossa t si osserva V(s_t) (predizione della rete) e la si
 * confronta con V(s_t+1) (bootstrap, la predizione della rete sulla mossa
 * successiva) oppure, se la partita e' finita, con l'esito reale
 * (+1 vittoria bianco, -1 vittoria nero, 0 pareggio). La differenza
 * (TD-error) e' il segnale di apprendimento: si aggiornano i pesi nella
 * direzione che avrebbe reso la vecchia predizione piu' vicina alla nuova,
 * usando le eligibility trace per propagare il segnale anche alle mosse di
 * qualche turno prima (non solo all'ultima).
 * <p>
 * <b>Nessun replay buffer</b>: a differenza del Deep Q-Learning (dove le
 * transizioni si accumulano in una memoria e si ricampionano piu' volte),
 * qui l'aggiornamento e' completamente ONLINE - ogni mossa viene osservata,
 * usata una sola volta per aggiornare i pesi (piu' l'eligibility trace, che
 * pero' vive solo per la durata dell'episodio corrente: viene azzerata ad
 * ogni nuova partita) e poi scartata. E' esattamente lo schema originale di
 * TD-Gammon, che precede il replay buffer di qualche decennio e non ne ha
 * mai avuto bisogno: TD(lambda) e' pensato per l'apprendimento online fin
 * dall'origine.
 * <p>
 * <b>Catastrophic forgetting</b>: il self-play e' un bersaglio mobile (la
 * rete genera le proprie partite, quindi la distribuzione delle posizioni
 * viste cambia man mano che migliora), il che e' terreno fertile per
 * dimenticare come gestire posizioni/stili tipici di versioni piu' deboli di
 * se stessa. Due contromisure, entrambe ispirate a come i sistemi di
 * self-play "seri" (AlphaZero e derivati) affrontano lo stesso problema:
 * <ul>
 * <li><b>pool di avversari storici</b> ({@code historicalOpponentProb}): con
 * una certa probabilita', l'avversario in un episodio non e' la rete attuale
 * ne' {@link SimpleEvaluator}, ma una VERSIONE PASSATA della rete stessa,
 * ricaricata da un checkpoint salvato in precedenza. Cosi' la rete deve
 * restare in grado di battere anche i propri "io" precedenti, non solo
 * quello attuale;</li>
 * <li><b>rilevatore di forgetting</b>: ogni {@code checkpointEvery} episodi,
 * oltre a valutare contro {@link SimpleEvaluator}, si rigioca anche contro il
 * checkpoint storico piu' vecchio disponibile. Se il winrate contro il
 * proprio passato scende nel tempo, e' un segnale diretto di forgetting (non
 * solo una supposizione).</li>
 * </ul>
 * Quello che NON c'e' (e sarebbe il passo successivo naturale, piu' invasivo
 * da implementare bene): un vero e proprio replay buffer con minibatch, o una
 * regolarizzazione esplicita verso i pesi di un checkpoint di riferimento
 * (stile Elastic Weight Consolidation).
 * <p>
 * <b>Profondita' variabile PER MOSSA, non per partita</b>: ogni singola mossa
 * (sia del lato in self-play sia dell'avversario) usa un budget di tempo
 * estratto a caso in [{@code minMoveMs}, {@code maxMoveMs}], e la profondita'
 * raggiunta da Negamax dipende da quel budget e dalla posizione (esattamente
 * come nel gioco vero, dove la profondita' raggiunta varia da mossa a mossa
 * in base al tempo a disposizione, non e' scelta a priori). Le versioni
 * precedenti fissavano UNA profondita' per l'intera partita: bastava una
 * manciata di partite per notare che la rete imparava a sfruttare i punti
 * ciechi tattici di un avversario a profondita' fissa, ripetendo lo stesso
 * schema mossa dopo mossa nella stessa partita - un comportamento che non si
 * ripresenta a profondita' diverse (vedi discussione sull'instabilita'
 * pari/dispari della ricerca game-tree). Variare il budget mossa per mossa
 * toglie quel bersaglio fisso da sfruttare.
 * <p>
 * <b>Parallelizzazione</b> ({@code threads}): la parte costosa (la ricerca
 * Negamax per scegliere le mosse) e' separata dalla parte che deve restare
 * rigorosamente sequenziale (l'aggiornamento dei pesi via TD(lambda)).
 * {@code threads} partite vengono GENERATE in parallelo, ciascuna da un
 * worker con la propria copia PRIVATA e CONGELATA dei pesi correnti (presa a
 * inizio "round", si veda {@link MLP#copy()}) - i worker non toccano mai i
 * pesi condivisi, quindi nessuna race condition, ma scelgono le mosse con
 * pesi leggermente meno recenti (di al massimo {@code threads} episodi) di
 * quelli piu' aggiornati. Una volta generate, le partite vengono passate una
 * alla volta al thread principale, che applica l'IDENTICO algoritmo TD(lambda)
 * sequenziale di sempre (si veda {@link #trainOnTrajectory}) sui pesi VERI e
 * condivisi. La staleness dei pesi usati per scegliere le mosse e' una
 * pratica standard nel RL parallelo (schema "actor-learner", come A3C/IMPALA)
 * e non altera in alcun modo la correttezza matematica dell'aggiornamento dei
 * pesi, che resta byte-per-byte lo stesso codice single-thread di prima -
 * cambia solo con quali pesi (di poco meno recenti) sono state scelte le
 * mosse della partita su cui si sta imparando.
 * <p>
 * Uso da riga di comando, argomenti in forma {@code chiave=valore} (ordine
 * libero, tutti opzionali):
 * <pre>
 *   java SelfPlayTrainer episodes=20000 minDepth=1 maxDepth=3 lambda=0.7 alpha=0.02 \
 *        out=trained_weights.txt seed=42 opponentPoolProb=0.3
 * </pre>
 * <b>Continuare un training precedente</b> invece di ripartire da pesi
 * casuali: aggiungere {@code resumeFrom=<file dei pesi gia' allenati>}.
 * Insieme a {@code resumeFrom}, {@code startEpisode} e {@code totalPlanned}
 * permettono di far proseguire con continuita' lo schema di decadimento di
 * epsilon (l'esplorazione) tra piu' lanci successivi: si veda
 * {@link #epsilonSchedule}.
 * <p>
 * <b>Rete (da rete nuova)</b>: {@code groups=<maschera>} (gruppi di feature, default
 * {@code TablutFeatures.G_DEFAULT}=11; 2 = tier leggero, 15 = tutte), {@code trunk=24},
 * {@code head=8} (liste separate da virgola, {@code none} = nessuno strato), {@code act=LEAKY}
 * (RELU | HARDTANH | LEAKY | TANH). Con {@code resumeFrom} questi argomenti sono ignorati:
 * architettura e gruppi di feature vengono dal file.
 * <p>
 * <b>Ruolo</b>: la rete ha un tronco condiviso e due head scelte da chi deve muovere nella
 * posizione ({@link MLP}). Nel TD ogni posizione aggiorna il tronco e SOLO la head del proprio
 * ruolo; il bersaglio V(s_{t+1}) usa la head del ruolo di s_{t+1}.
 * <p>
 * <b>Consiglio</b>: partire con {@code resumeFrom=<pesi di SupervisedPretrainer>}. Da pesi casuali la
 * rete impiega migliaia di episodi per smettere di essere (quasi) costante.
 */
public class SelfPlayTrainer {

	private static final int MAX_PLIES_PER_EPISODE = 300;
	private static final long SELF_PLAY_TIME_BUDGET_MS = 3000; // limite di sicurezza; a profondita' 2-3 non viene quasi mai raggiunto

	public static void main(String[] args) throws IOException {
		Map<String, String> p = parseArgs(args);

		int episodes = getInt(p, "episodes", 3000);
		// minDepth/maxDepth: usate SOLO per la valutazione diagnostica periodica
		// (profondita' fisse e controllate, apposta per misurare la generalizzazione).
		// La profondita' usata DURANTE il self-play di training e' invece decisa
		// mossa per mossa da un budget di tempo casuale, si vedano minMoveMs/maxMoveMs.
		int minDepth = getInt(p, "minDepth", 1);
		int maxDepth = getInt(p, "maxDepth", 4);
		long minMoveMs = getLong(p, "minMoveMs", 100);
		long maxMoveMs = getLong(p, "maxMoveMs", 1500);
		// Quanto e' "piu' forte" l'avversario (baseline/storico) rispetto al lato in
		// self-play, in termini di tempo di pensiero per mossa. 1.0 = stessa forza
		// (default consigliato: si veda la discussione nel javadoc della classe sul
		// perche' un avversario troppo piu' forte puo' PEGGIORARE il segnale di
		// apprendimento, non solo migliorarlo).
		double opponentTimeMultiplier = getDouble(p, "opponentTimeMultiplier", 1.0);
		int maxSearchDepthCap = getInt(p, "maxSearchDepthCap", 8); // tetto di sicurezza, raramente vincolante
		double lambda = getDouble(p, "lambda", 0.7);
		double alpha = getDouble(p, "alpha", 0.02);
		String outFile = p.getOrDefault("out", "trained_weights.txt");
		long seed = getLong(p, "seed", System.currentTimeMillis());
		double baselineOpponentProb = getDouble(p, "opponentPoolProb", 0.3);
		double historicalOpponentProb = getDouble(p, "historicalOpponentProb", 0.2);
		int checkpointEvery = getInt(p, "checkpointEvery", 500);
		String checkpointDir = p.getOrDefault("checkpointDir", "checkpoints");
		String logFile = p.getOrDefault("log", "training_log.csv");

		// --- parametri per CONTINUARE un training precedente ---
		String resumeFrom = p.get("resumeFrom"); // null se assente: si parte da pesi casuali
		int startEpisode = getInt(p, "startEpisode", 0); // quanti episodi ha gia' visto questa rete, cumulativamente
		int totalPlanned = getInt(p, "totalPlanned", startEpisode + episodes); // arco totale su cui far decadere epsilon
		double epsilonStart = getDouble(p, "epsilonStart", 0.3);
		double epsilonEnd = getDouble(p, "epsilonEnd", 0.02);
		int numThreads = getInt(p, "threads", Runtime.getRuntime().availableProcessors());
		// --- architettura della rete (ignorata con resumeFrom: la forma e i gruppi di feature vengono dal file) ---
		int groups = getInt(p, "groups", TablutFeatures.G_DEFAULT); // maschera di gruppi di feature (si veda TablutFeatures)
		int[] trunk = getIntList(p, "trunk", new int[] { 24 });
		int[] head = getIntList(p, "head", new int[] { 8 });
		MLP.Act act = MLP.Act.valueOf(p.getOrDefault("act", "LEAKY").toUpperCase());

		System.out.println("=== SelfPlayTrainer === episodes=" + episodes + " depth=[" + minDepth + "," + maxDepth
				+ "] lambda=" + lambda + " alpha=" + alpha + " out=" + outFile + " seed=" + seed
				+ " opponentPoolProb=" + baselineOpponentProb + " historicalOpponentProb=" + historicalOpponentProb
				+ " threads=" + numThreads
				+ (resumeFrom != null ? " | RESUME da " + resumeFrom + " (startEpisode=" + startEpisode
						+ ", totalPlanned=" + totalPlanned + ")" : " | rete nuova (pesi casuali)"));

		Random rnd = new Random(seed);
		NeuralEvaluator evaluator;
		if (resumeFrom != null) {
			evaluator = NeuralEvaluator.fromFile(resumeFrom); // IOException chiara se il file e' legacy / di un'altra versione di feature
			groups = evaluator.featureGroups;
		} else {
			MLP.Config cfg = MLP.Config.sharedTrunk(TablutFeatures.columnsFor(groups).length, trunk, head, act);
			evaluator = new NeuralEvaluator(new MLP(cfg, seed), groups);
		}
		final MLP net = evaluator.net;
		System.out.println("Rete: " + net.config() + " | gruppi di feature=" + groups + " | " + net.numParams() + " parametri ("
				+ net.paramsOnPath() + " per valutazione)");
		SimpleEvaluator baseline = new SimpleEvaluator();

		// Pool di avversari storici: checkpoint passati della rete stessa, usati sia
		// come possibili avversari nel self-play (contro il forgetting) sia come
		// pietra di paragone per rilevare il forgetting quando avviene. Se stiamo
		// riprendendo un training (resumeFrom), recuperiamo anche i checkpoint
		// salvati nei lanci precedenti, cosi' la "memoria storica" non si perde.
		File checkpointDirFile = new File(checkpointDir);
		checkpointDirFile.mkdirs();
		List<String> checkpointHistory = new ArrayList<>();
		File[] existingCkpts = checkpointDirFile.listFiles((dir, name) -> name.startsWith("ckpt_") && name.endsWith(".txt"));
		if (existingCkpts != null) {
			int skipped = 0;
			for (File f : existingCkpts) {
				try {
					NeuralEvaluator.fromFile(f.getPath()); // solo i checkpoint compatibili con le feature attuali
					checkpointHistory.add(f.getPath());
				} catch (IOException e) {
					skipped++;
				}
			}
			if (skipped > 0) {
				System.out.println("Ignorati " + skipped + " checkpoint in " + checkpointDir
						+ " (formato legacy o altra versione di feature: non riutilizzabili)");
			}
			checkpointHistory.sort(String::compareTo); // i nomi sono zero-padded: ordine alfabetico = ordine cronologico
			if (!checkpointHistory.isEmpty()) {
				System.out.println("Recuperati " + checkpointHistory.size() + " checkpoint storici da " + checkpointDir);
			}
		}

		// Se stiamo continuando un training precedente, accodiamo al log esistente
		// invece di sovrascriverlo, cosi' si puo' tracciare/graficare l'intero
		// arco (es. 0->50000 episodi) anche se e' stato fatto in piu' lanci.
		boolean append = resumeFrom != null && new File(logFile).exists();
		// quante partite sono finite in ciascun modo (WW, BW, D, REPETITION, NOMOVES_WW/BW, MAXPLIES): visibile nel log
		final Map<String, Integer> endCounts = new TreeMap<>();
		ExecutorService pool = Executors.newFixedThreadPool(numThreads);
		try (PrintWriter log = new PrintWriter(new FileWriter(logFile, append))) {
			if (!append) {
				log.println("episode,avgAbsTdError,plies,outcome,epsilon,opponent,avgDepthReached,opponentType");
			}

			int done = 0;
			while (done < episodes) {
				int roundSize = Math.min(numThreads, episodes - done);

				// --- FASE 1: GENERAZIONE, in parallelo ---
				// Ogni task riceve la PROPRIA copia indipendente dei pesi attuali (presa
				// ORA, prima che la fase di training qui sotto li modifichi): nessun
				// task tocca mai i pesi condivisi, quindi nessuna race condition,
				// indipendentemente da quanti thread girano insieme.
				List<String> checkpointHistorySnapshot = new ArrayList<>(checkpointHistory);
				List<GenerationTask> tasks = new ArrayList<>(roundSize);
				for (int i = 0; i < roundSize; i++) {
					int cumulativeEp = startEpisode + done + i + 1;
					double epsilon = epsilonSchedule(cumulativeEp, totalPlanned, epsilonStart, epsilonEnd);
					long taskSeed = rnd.nextLong();
					tasks.add(new GenerationTask(net.copy(), groups, baseline, checkpointHistorySnapshot, maxSearchDepthCap,
							epsilon, minMoveMs, maxMoveMs, opponentTimeMultiplier, baselineOpponentProb,
							historicalOpponentProb, taskSeed));
				}

				List<Future<Trajectory>> futures;
				try {
					futures = pool.invokeAll(tasks);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					throw new RuntimeException("Generazione interrotta", e);
				}

				// --- FASE 2: TRAINING, sequenziale, sui pesi VERI e condivisi ---
				// Da qui in poi e' l'IDENTICO codice single-thread di sempre: si
				// applicano gli aggiornamenti TD(lambda) una partita alla volta,
				// nell'ordine in cui sono state sottomesse (per riproducibilita' a
				// parita' di numero di thread).
				for (int i = 0; i < roundSize; i++) {
					int cumulativeEp = startEpisode + done + i + 1;
					Trajectory traj;
					try {
						traj = futures.get(i).get();
					} catch (Exception e) {
						throw new RuntimeException("Errore nella generazione dell'episodio " + cumulativeEp, e);
					}

					EpisodeResult res = trainOnTrajectory(net, traj, lambda, alpha);
					endCounts.merge(res.outcome, 1, Integer::sum);
					double epsilonForLog = epsilonSchedule(cumulativeEp, totalPlanned, epsilonStart, epsilonEnd);
					log.println(cumulativeEp + "," + res.avgAbsTdError + "," + res.plies + "," + res.outcome + ","
							+ epsilonForLog + "," + (traj.opponentControls == null ? "none" : traj.opponentControls)
							+ "," + res.avgDepthReached + "," + traj.opponentType);

					if (cumulativeEp % 100 == 0) {
						evaluator.save(outFile);
						System.out.printf(
								"episodio %6d (cumulativo) | profondita' media=%.1f | epsilon=%.3f | avgTD=%.4f | mosse=%3d | esito=%s%n",
								cumulativeEp, res.avgDepthReached, epsilonForLog, res.avgAbsTdError, res.plies,
								res.outcome);
						System.out.println("    esiti finora: " + endCounts);
					}
					if (cumulativeEp % checkpointEvery == 0) {
						String ckptPath = String.format("%s/ckpt_%08d.txt", checkpointDir, cumulativeEp);
						evaluator.save(ckptPath);
						checkpointHistory.add(ckptPath);

						System.out.println("  -- valutazione a piu' profondita' (generalizzazione) --");
						for (int d = minDepth; d <= maxDepth + 1; d++) {
							evaluateAgainstBaseline(evaluator, baseline, 12, d, rnd);
						}

						if (checkpointHistory.size() > 1) {
							// Rilevatore di forgetting: confronto diretto col proprio passato piu'
							// remoto disponibile. Se il winrate scende nel tempo, e' forgetting,
							// non solo una supposizione teorica.
							String oldest = checkpointHistory.get(0);
							try {
								NeuralEvaluator oldSelf = NeuralEvaluator.fromFile(oldest);
								System.out.println("  -- confronto col proprio passato (rilevatore di forgetting) --");
								evaluateAgainstOpponent(evaluator, oldSelf,
										"se stessa a " + new File(oldest).getName(), 12, minDepth, rnd);
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
		System.out.println("Training completato. Pesi salvati in " + outFile + " (log in " + logFile
				+ "). Episodi totali visti da questa rete: " + (startEpisode + episodes));
	}

	private static Map<String, String> parseArgs(String[] args) {
		Map<String, String> map = new HashMap<>();
		for (String a : args) {
			int eq = a.indexOf('=');
			if (eq < 0) {
				throw new IllegalArgumentException("Argomento non valido (atteso chiave=valore): " + a);
			}
			map.put(a.substring(0, eq), a.substring(eq + 1));
		}
		return map;
	}

	/** "24" -> {24}; "32,16" -> {32,16}; "none" -> {} */
	private static int[] getIntList(Map<String, String> p, String key, int[] def) {
		String v = p.get(key);
		if (v == null) {
			return def;
		}
		if (v.equalsIgnoreCase("none") || v.equals("-")) {
			return new int[0];
		}
		String[] t = v.split(",");
		int[] r = new int[t.length];
		for (int i = 0; i < t.length; i++) {
			r[i] = Integer.parseInt(t[i].trim());
		}
		return r;
	}

	private static int getInt(Map<String, String> p, String key, int def) {
		return p.containsKey(key) ? Integer.parseInt(p.get(key)) : def;
	}

	private static long getLong(Map<String, String> p, String key, long def) {
		return p.containsKey(key) ? Long.parseLong(p.get(key)) : def;
	}

	private static double getDouble(Map<String, String> p, String key, double def) {
		return p.containsKey(key) ? Double.parseDouble(p.get(key)) : def;
	}

	/**
	 * Epsilon decresce linearmente da {@code start} a {@code end} lungo il primo
	 * 70% di {@code totalPlanned} episodi, poi resta fisso a {@code end}.
	 * {@code episode} e {@code totalPlanned} sono pensati come CUMULATIVI
	 * attraverso piu' lanci (si veda {@code startEpisode}/{@code totalPlanned}
	 * in {@link #main}), cosi' che riprendere un training con {@code resumeFrom}
	 * continui il decadimento invece di farlo ripartire da capo.
	 */
	private static double epsilonSchedule(int episode, int totalPlanned, double start, double end) {
		double frac = Math.min(1.0, episode / Math.max(1.0, totalPlanned * 0.7));
		return start + (end - start) * frac;
	}

	private static class EpisodeResult {
		double avgAbsTdError;
		int plies;
		String outcome;
		double avgDepthReached;
	}

	/**
	 * Il risultato della fase di GENERAZIONE (parallelizzabile): la sequenza di
	 * posizioni attraversate in una partita, piu' l'esito finale. Non contiene
	 * alcun riferimento alla rete "vera" ne' alcuno stato che debba restare
	 * sincronizzato con altri thread - e' un semplice dato immutabile, sicuro
	 * da passare tra thread.
	 */
	static class Trajectory {
		final List<double[]> features; // features(s_0), features(s_1), ..., features(s_{T-1}) (gia' compattate per la rete)
		final int[] roles; // roles[t] = head che valuta s_t (0 = Bianco di turno, 1 = Nero di turno)
		final double outcome; // +1/-1/0: esito FINALE (si veda GameTracker: stallo = sconfitta, ripetizione = pareggio, troncata = 0)
		final int plies;
		final String outcomeLabel;
		final Turn opponentControls; // quale colore (se non null) era guidato dall'avversario, non dal self-play
		final String opponentType; // "self" | "baseline" | "historical", per il log
		final double avgDepthReached;
		/** Solo per i test (si veda {@link #recordStatesForTests}): tutti gli stati della partita, iniziale e finale compresi. */
		List<State> states;

		Trajectory(List<double[]> features, int[] roles, double outcome, int plies, String outcomeLabel, Turn opponentControls,
				String opponentType, double avgDepthReached) {
			this.features = features;
			this.roles = roles;
			this.outcome = outcome;
			this.plies = plies;
			this.outcomeLabel = outcomeLabel;
			this.opponentControls = opponentControls;
			this.opponentType = opponentType;
			this.avgDepthReached = avgDepthReached;
		}
	}

	/** Se vero, {@link GenerationTask} conserva tutti gli stati della partita in {@link Trajectory#states} (SOLO per i test). */
	static volatile boolean recordStatesForTests = false;

	/** Chiamata per ogni posizione in cui si deve muovere (anche quella senza mosse legali): serve a campionare le feature. */
	interface PositionListener {
		void onPosition(State state);
	}

	/** Sceglie la mossa. {@code historySinceLastCapture} va passata alla ricerca (si veda {@link GameTracker#history()}). */
	interface MoveChooser {
		Action choose(State state, List<Long> historySinceLastCapture);
	}

	/** Risultato di {@link #playGame}. */
	static final class GameOutcome {
		final State finalState;
		final int plies;
		final GameTracker.Result result;
		final List<State> states; // null se non richiesti

		GameOutcome(State finalState, int plies, GameTracker.Result result, List<State> states) {
			this.finalState = finalState;
			this.plies = plies;
			this.result = result;
			this.states = states;
		}
	}

	/**
	 * Gioca una partita da {@code start} applicando le regole di fine partita di {@link GameTracker}: vittoria, stallo (chi non
	 * puo' muovere PERDE), ripetizione (PAREGGIO alla prima ripetizione dall'ultima cattura) o tetto di semi-mosse (pareggio).
	 * Estratta da {@link GenerationTask} per poterla collaudare con scelte di mosse "scriptate" (e' l'unico modo di FORZARE
	 * ripetizioni e stalli, che nel self-play naturale sono rari).
	 */
	static GameOutcome playGame(State start, int maxPlies, PositionListener listener, MoveChooser chooser, boolean recordStates) {
		State state = start;
		GameTracker game = new GameTracker(state);
		List<State> states = recordStates ? new ArrayList<State>() : null;
		if (states != null) {
			states.add(state);
		}
		int ply = 0;
		for (; ply < maxPlies; ply++) {
			if (AshtonRules.isTerminal(state)) {
				break;
			}
			listener.onPosition(state);
			if (AshtonRules.generateMoves(state).isEmpty()) {
				break; // stallo: GameTracker.result lo trasforma in sconfitta di chi deve muovere
			}
			Action move = chooser.choose(state, game.history());
			if (move == null) {
				break;
			}
			state = game.apply(state, move);
			if (states != null) {
				states.add(state);
			}
			if (game.isRepetitionDraw()) {
				ply++; // la mossa che ha ricreato la posizione e' stata giocata
				break;
			}
		}
		return new GameOutcome(state, ply, game.result(state), states);
	}

	/**
	 * Genera UNA partita (fase 1, parallelizzabile). Non tocca MAI la rete
	 * condivisa: riceve gia' la propria copia privata ({@code myNet}, ottenuta
	 * con {@link MLP#copy()} prima di sottomettere il task) e la usa solo per
	 * scegliere le mosse (forward pass in sola lettura, mai backward/aggiornamento
	 * dei pesi). Puo' quindi essere eseguito in sicurezza da piu' thread
	 * contemporaneamente, ciascuno con la propria istanza di questa classe.
	 */
	static class GenerationTask implements Callable<Trajectory> {
		private final MLP myNet;
		private final int groups;
		private final SimpleEvaluator baseline; // condiviso: nessuno stato mutabile durante evaluate()
		private final List<String> checkpointHistorySnapshot;
		private final int maxSearchDepthCap;
		private final double epsilon;
		private final long minMoveMs, maxMoveMs;
		private final double opponentTimeMultiplier;
		private final double baselineOpponentProb, historicalOpponentProb;
		private final Random rnd; // privata per questo task: nessuna contesa, nessuna dipendenza dall'ordine di scheduling

		GenerationTask(MLP myNet, int groups, SimpleEvaluator baseline, List<String> checkpointHistorySnapshot,
				int maxSearchDepthCap, double epsilon, long minMoveMs, long maxMoveMs,
				double opponentTimeMultiplier, double baselineOpponentProb, double historicalOpponentProb,
				long taskSeed) {
			this.myNet = myNet;
			this.groups = groups;
			this.baseline = baseline;
			this.checkpointHistorySnapshot = checkpointHistorySnapshot;
			this.maxSearchDepthCap = maxSearchDepthCap;
			this.epsilon = epsilon;
			this.minMoveMs = minMoveMs;
			this.maxMoveMs = maxMoveMs;
			this.opponentTimeMultiplier = opponentTimeMultiplier;
			this.baselineOpponentProb = baselineOpponentProb;
			this.historicalOpponentProb = historicalOpponentProb;
			this.rnd = new Random(taskSeed);
		}

		@Override
		public Trajectory call() {
			NeuralEvaluator myEvaluator = new NeuralEvaluator(myNet, groups);
			TablutFeatures.Workspace featWs = TablutFeatures.newWorkspace();
			double[] featFull = new double[TablutFeatures.NUM_FEATURES];
			Negamax selfPlaySearch = new Negamax(myEvaluator, maxSearchDepthCap, true, 1 << 20).setThreads(1);
			Negamax baselineSearch = new Negamax(baseline, maxSearchDepthCap, true, 1 << 20).setThreads(1);

			Turn opponentControls = null;
			Negamax opponentSearch = null;
			String opponentType = "self";
			double r = rnd.nextDouble();
			if (r < baselineOpponentProb) {
				opponentControls = rnd.nextBoolean() ? Turn.WHITE : Turn.BLACK;
				opponentSearch = baselineSearch;
				opponentType = "baseline";
			} else if (r < baselineOpponentProb + historicalOpponentProb && !checkpointHistorySnapshot.isEmpty()) {
				String histFile = checkpointHistorySnapshot.get(rnd.nextInt(checkpointHistorySnapshot.size()));
				try {
					NeuralEvaluator histEval = NeuralEvaluator.fromFile(histFile);
					opponentControls = rnd.nextBoolean() ? Turn.WHITE : Turn.BLACK;
					opponentSearch = new Negamax(histEval, maxSearchDepthCap, true, 1 << 20).setThreads(1);
					opponentType = "historical";
				} catch (IOException e) {
					// checkpoint corrotto o rimosso: questo episodio diventa self-play puro
				}
			}

			State start = new StateTablut();
			start.setTurn(Turn.WHITE);
			final List<double[]> feats = new ArrayList<>();
			final List<Integer> roleList = new ArrayList<>();
			final long[] depthAcc = new long[2]; // [0] somma delle profondita' raggiunte, [1] numero di ricerche
			final Turn oppTurn = opponentControls;
			final Negamax oppSearch = opponentSearch;
			final Negamax mySearch = selfPlaySearch;
			final NeuralEvaluator evalForFeatures = myEvaluator;

			PositionListener sampler = new PositionListener() {
				@Override
				public void onPosition(State s) {
					double[] x = new double[evalForFeatures.numInputs()];
					evalForFeatures.extractInputs(s, featWs, featFull, x);
					feats.add(x);
					roleList.add(NeuralEvaluator.roleOf(s)); // quale head valuta questa posizione: chi deve muovere
				}
			};
			MoveChooser chooser = new MoveChooser() {
				@Override
				public Action choose(State s, List<Long> historySinceLastCapture) {
					List<Action> legalMoves = AshtonRules.generateMoves(s);
					long budget = minMoveMs + (maxMoveMs > minMoveMs ? rnd.nextLong() % (maxMoveMs - minMoveMs + 1) : 0);
					if (budget < 0) {
						budget = -budget;
					}
					Action move;
					if (rnd.nextDouble() < epsilon) {
						move = legalMoves.get(rnd.nextInt(legalMoves.size()));
					} else if (s.getTurn() == oppTurn) {
						// la STORIA di ripetizione va passata a ogni ricerca: cosi' sa che ripetere una posizione e' patta
						move = oppSearch.findBestMove(s, (long) (budget * opponentTimeMultiplier), historySinceLastCapture);
						depthAcc[0] += oppSearch.lastDepthReached;
						depthAcc[1]++;
					} else {
						move = mySearch.findBestMove(s, budget, historySinceLastCapture);
						depthAcc[0] += mySearch.lastDepthReached;
						depthAcc[1]++;
					}
					return move != null ? move : legalMoves.get(rnd.nextInt(legalMoves.size()));
				}
			};

			GameOutcome game = playGame(start, MAX_PLIES_PER_EPISODE, sampler, chooser, recordStatesForTests);
			double avgDepth = depthAcc[1] > 0 ? depthAcc[0] / (double) depthAcc[1] : 0;

			int[] roles = new int[roleList.size()];
			for (int i = 0; i < roles.length; i++) {
				roles[i] = roleList.get(i);
			}
			Trajectory trajectory = new Trajectory(feats, roles, game.result.outcome, game.plies, game.result.label, opponentControls,
					opponentType, avgDepth);
			trajectory.states = game.states;
			return trajectory;
		}
	}

	/**
	 * Fase 2, sequenziale: applica alla rete VERA e condivisa {@code net}
	 * l'IDENTICO algoritmo TD(lambda) di sempre, sulla sequenza di posizioni
	 * gia' generata (in parallelo o meno, non fa differenza da qui in poi).
	 * Nessuna concorrenza in questo metodo: chiamato sempre dal thread
	 * principale, un episodio alla volta.
	 */
	private static EpisodeResult trainOnTrajectory(MLP net, Trajectory traj, double lambda, double alpha) {
		// eligibility trace = vettore piatto della stessa lunghezza dei parametri; la head del ruolo di ogni
		// posizione e' quella che viene valutata e aggiornata (il tronco e' condiviso da entrambi i ruoli)
		final double[] trace = new double[net.numParams()];
		final MLP.Scratch sc = net.newScratch();
		double sumAbsTd = 0;
		int steps = traj.features.size();

		for (int i = 0; i < steps; i++) {
			double v = net.forward(traj.features.get(i), traj.roles[i], sc);

			// trace = lambda * trace + dV(s_t)/dw (il backward ACCUMULA: va fatto subito dopo il forward di s_t)
			for (int k = 0; k < trace.length; k++) {
				trace[k] *= lambda;
			}
			net.backward(sc, 1.0, trace);

			// bersaglio: V(s_{t+1}) valutato dalla head del ruolo di s_{t+1}, oppure l'esito finale
			double target = (i + 1 < steps) ? net.forward(traj.features.get(i + 1), traj.roles[i + 1], sc) : traj.outcome;
			double delta = target - v;
			sumAbsTd += Math.abs(delta);

			net.applyGradients(trace, alpha * delta);
		}

		EpisodeResult r = new EpisodeResult();
		r.avgAbsTdError = steps > 0 ? sumAbsTd / steps : 0;
		r.plies = traj.plies;
		r.outcome = traj.outcomeLabel;
		r.avgDepthReached = traj.avgDepthReached;
		return r;
	}

	/**
	 * Fa giocare {@code games} partite (meta' come bianco, meta' come nero) tra
	 * la rete in allenamento e {@link SimpleEvaluator}, stampando il bilancio.
	 * Utile per misurare oggettivamente se il training sta migliorando la
	 * forza di gioco, non solo se il TD-error scende.
	 */
	public static double evaluateAgainstBaseline(Evaluator neural, SimpleEvaluator baseline, int games,
			int depth, Random rnd) {
		return evaluateAgainstOpponent(neural, baseline, "SimpleEvaluator", games, depth, rnd);
	}

	/**
	 * Generalizzazione di {@link #evaluateAgainstBaseline}: l'avversario puo'
	 * essere un qualsiasi {@link Evaluator}, non solo {@link SimpleEvaluator} -
	 * in particolare, un {@link NeuralEvaluator} caricato da un checkpoint
	 * storico, per misurare direttamente il forgetting (la rete attuale gioca
	 * peggio contro una propria versione passata di quanto giocasse quella
	 * versione stessa a suo tempo?).
	 */
	public static double evaluateAgainstOpponent(Evaluator neural, Evaluator opponent, String opponentLabel,
			int games, int depth, Random rnd) {
		Negamax neuralSearch = new Negamax(neural, depth, true, 1 << 20).setThreads(1);
		Negamax opponentSearch = new Negamax(opponent, depth, true, 1 << 20).setThreads(1);
		int wins = 0, losses = 0, draws = 0;

		for (int g = 0; g < games; g++) {
			boolean neuralIsWhite = g % 2 == 0;
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			GameTracker game = new GameTracker(s);
			int ply = 0;
			while (!AshtonRules.isTerminal(s) && ply < MAX_PLIES_PER_EPISODE) {
				boolean whiteToMove = s.getTurn() == Turn.WHITE;
				Negamax mover = (whiteToMove == neuralIsWhite) ? neuralSearch : opponentSearch;
				Action a = mover.findBestMove(s, SELF_PLAY_TIME_BUDGET_MS, game.history());
				if (a == null) {
					break; // nessuna mossa legale: GameTracker.result lo conta come sconfitta di chi doveva muovere
				}
				s = game.apply(s, a);
				ply++;
				if (game.isRepetitionDraw()) {
					break; // ripetizione = pareggio, come nella partita vera
				}
			}
			GameTracker.Result end = game.result(s);
			if (end.outcome > 0) {
				if (neuralIsWhite) {
					wins++;
				} else {
					losses++;
				}
			} else if (end.outcome < 0) {
				if (!neuralIsWhite) {
					wins++;
				} else {
					losses++;
				}
			} else {
				draws++;
			}
		}
		double winRate = (wins + 0.5 * draws) / (double) games;
		System.out.printf("  [valutazione] rete attuale vs %s: %d V %d P %d S su %d partite (winrate=%.2f)%n",
				opponentLabel, wins, draws, losses, games, winRate);
		return winRate;
	}
}
