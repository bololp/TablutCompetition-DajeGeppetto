package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ConvNet;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Pre-addestramento SUPERVISIONATO del {@link ConvNet}: il "warm start" per {@link CnnSelfPlayTrainer} (come
 * {@link SupervisedPretrainer} per l'MLP). Da pesi casuali il TD impiega migliaia di episodi anche solo per smettere di essere
 * costante; un CNN, che ha piu' parametri e meno bias induttivi dell'MLP con feature a mano, ne ha ancora piu' bisogno.
 *
 * <h3>Cosa fa</h3>
 * <ol>
 * <li><b>Genera partite</b> con lo stesso mix di {@link SupervisedPretrainer} ({@link SimpleEvaluator} a profondita' 1-3 con quote di
 * mosse casuali; regole di fine partita di {@link GameTracker}; partite troncate scartate) e registra per ogni posizione: scacchiera
 * (81 codici), chi muove, TUTTE le 44 feature a mano ({@link TablutFeatures}) e l'esito. Salvare il dataset ({@code dataOut=})
 * permette di riaddestrare configurazioni diverse senza rigenerare: la generazione e' la parte lenta.</li>
 * <li><b>Addestra</b> a predire l'esito (errore quadratico, Adam, coseno decrescente), con la head del ruolo di chi muove. Con
 * {@code augment=1} (default) ogni campione e' presentato in una delle 8 simmetrie della scacchiera scelta a caso
 * ({@link it.unibo.ai.didattica.competition.tablut.aiplayer.BoardSymmetry}): il CNN non e' invariante per rotazione/riflessione
 * per costruzione, cosi' lo impara, e i dati effettivi sono 8 volte tanti. Le feature a mano sono gia' invarianti. Il minibatch
 * e' calcolato in parallelo su {@code threads} thread (stesso risultato, a meno dell'ordine delle somme in virgola mobile).</li>
 * <li><b>Valuta</b> su ~8% delle PARTITE tenute da parte (errore "held-out"), accanto all'errore del predittore costante.</li>
 * </ol>
 *
 * <h3>Limiti</h3>
 * Le etichette vengono da partite di giocatori deboli: il segnale e' "chi vince da qui, giocando cosi'". Serve a partire da pesi
 * sensati, NON a sostituire il TD self-play. Un errore held-out piu' basso dell'MLP sugli stessi dati e' necessario ma non
 * sufficiente: il CNN costa molto di piu' per valutazione (si veda {@code CnnBench}) e il confronto che decide e'
 * {@code EqualTimeArena} a tempo uguale.
 *
 * <p>
 * Argomenti {@code chiave=valore}: games(7000), epochs(10), conv(4), reduce(2), dense(32), head(8), features(1 = ibrido, 0 = CNN puro),
 * groups(G_DEFAULT), lr(0.002), batch(128), augment(1), seed(1), threads(tutti i core), out(cnn_pretrained.txt),
 * dataOut(nessuno), data(nessuno: se presente carica il dataset invece di generarlo), init(nessuno: se presente parte dai pesi di quel file
 * invece che da pesi casuali; la forma della rete e i gruppi di feature vengono dal file).
 */
public class CnnPretrainer {

	private static final int NF = TablutFeatures.NUM_FEATURES;
	private static final String DATA_MAGIC = "TABLUT_CNN_DATA v1";

	// ====================================================================== dataset

	/** Posizioni di partite, in array piatti (compatti in memoria: 250.000 posizioni ~ 66 MB). */
	public static final class Dataset {
		public final int n;
		/** Scacchiere: 81 codici per posizione (0 vuoto, 1 bianco, 2 nero, 3 re). */
		public final byte[] cells;
		/** Chi muove: 0 = Bianco, 1 = Nero. */
		public final byte[] role;
		/** Tutte le {@value TablutFeatures#NUM_FEATURES} feature, riga per posizione. */
		public final float[] feats;
		/** Esito della partita dal punto di vista del Bianco: +1, -1, 0. */
		public final byte[] outcome;
		/** Indice della partita di provenienza (serve a dividere train/test PER PARTITA). */
		public final int[] game;

		public Dataset(byte[] cells, byte[] role, float[] feats, byte[] outcome, int[] game) {
			this.n = role.length;
			if (cells.length != n * 81 || feats.length != n * NF || outcome.length != n || game.length != n) {
				throw new IllegalArgumentException("dimensioni incoerenti nel dataset");
			}
			this.cells = cells;
			this.role = role;
			this.feats = feats;
			this.outcome = outcome;
			this.game = game;
		}

		public void save(String path) throws IOException {
			try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path)), 1 << 16))) {
				o.writeUTF(DATA_MAGIC);
				o.writeInt(TablutFeatures.VERSION);
				o.writeInt(NF);
				o.writeInt(n);
				o.write(cells);
				o.write(role);
				o.write(outcome);
				for (int g : game) {
					o.writeInt(g);
				}
				for (float f : feats) {
					o.writeFloat(f);
				}
			}
		}

		public static Dataset load(String path) throws IOException {
			try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16))) {
				if (!DATA_MAGIC.equals(in.readUTF())) {
					throw new IOException("non e' un dataset CNN: " + path);
				}
				int fv = in.readInt(), nf = in.readInt(), n = in.readInt();
				if (fv != TablutFeatures.VERSION || nf != NF) {
					throw new IOException("dataset con feature v" + fv + "/" + nf + ", il codice usa v" + TablutFeatures.VERSION + "/" + NF
							+ ": rigenerarlo (" + path + ")");
				}
				if (n < 0 || n > 50_000_000) {
					throw new IOException("numero di posizioni non plausibile: " + n);
				}
				byte[] cells = new byte[n * 81], role = new byte[n], outcome = new byte[n];
				int[] game = new int[n];
				float[] feats = new float[n * NF];
				in.readFully(cells);
				in.readFully(role);
				in.readFully(outcome);
				for (int i = 0; i < n; i++) {
					game[i] = in.readInt();
				}
				for (int i = 0; i < feats.length; i++) {
					feats[i] = in.readFloat();
				}
				return new Dataset(cells, role, feats, outcome, game);
			}
		}
	}

	static final class GameRecord {
		State end; // stato finale della partita (serve ai test per verificare l'esito registrato)
		final List<byte[]> cells = new ArrayList<>();
		final List<float[]> feats = new ArrayList<>();
		final List<Byte> role = new ArrayList<>();
		int outcome;
		boolean truncated;
	}

	static byte[] codesOf(State s) {
		byte[] c = new byte[81];
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int k = 0; k < 9; k++) {
				Pawn p = b[r][k];
				c[r * 9 + k] = (byte) (p == Pawn.WHITE ? 1 : p == Pawn.BLACK ? 2 : p == Pawn.KING ? 3 : 0);
			}
		}
		return c;
	}

	/** Genera {@code games} partite (in parallelo) e le raccoglie in un {@link Dataset}; le troncate sono scartate. */
	public static Dataset generate(int games, long seed, int threads, PrintStream log) throws Exception {
		ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
		List<GameRecord> recs = new ArrayList<>();
		try {
			List<Future<GameRecord>> fs = new ArrayList<>();
			Random master = new Random(seed * 31 + 7);
			for (int g = 0; g < games; g++) {
				final long gameSeed = master.nextLong();
				fs.add(pool.submit(new Callable<GameRecord>() {
					@Override
					public GameRecord call() {
						return playOne(gameSeed);
					}
				}));
			}
			for (int i = 0; i < fs.size(); i++) {
				recs.add(fs.get(i).get());
				if (log != null && (i + 1) % 1000 == 0) {
					log.println("  generate " + (i + 1) + "/" + games + " partite");
				}
			}
		} finally {
			pool.shutdown();
		}
		int n = 0;
		for (GameRecord r : recs) {
			if (!r.truncated) {
				n += r.cells.size();
			}
		}
		byte[] cells = new byte[n * 81], role = new byte[n], outcome = new byte[n];
		float[] feats = new float[n * NF];
		int[] game = new int[n];
		int k = 0, gi = 0;
		for (GameRecord r : recs) {
			if (r.truncated) {
				continue;
			}
			for (int i = 0; i < r.cells.size(); i++, k++) {
				System.arraycopy(r.cells.get(i), 0, cells, k * 81, 81);
				System.arraycopy(r.feats.get(i), 0, feats, k * NF, NF);
				role[k] = r.role.get(i);
				outcome[k] = (byte) r.outcome;
				game[k] = gi;
			}
			gi++;
		}
		return new Dataset(cells, role, feats, outcome, game);
	}

	/** Una partita: stile scelto a caso (come {@link SupervisedPretrainer}), posizioni registrate prima di ogni mossa. */
	static GameRecord playOne(long gameSeed) {
		Random rnd = new Random(gameSeed);
		SimpleEvaluator simple = new SimpleEvaluator();
		int type = rnd.nextInt(10);
		int dW, dB;
		double epsW, epsB;
		if (type < 4) { // pari forza
			dW = 2;
			dB = 2;
			epsW = 0.10;
			epsB = 0.10;
		} else if (type < 7) { // profondita' 1 contro 3
			boolean sw = rnd.nextBoolean();
			dW = sw ? 1 : 3;
			dB = sw ? 3 : 1;
			epsW = 0.05;
			epsB = 0.05;
		} else { // un lato quasi casuale
			boolean sw = rnd.nextBoolean();
			dW = 2;
			dB = 2;
			epsW = sw ? 0.5 : 0.03;
			epsB = sw ? 0.03 : 0.5;
		}
		Negamax nW = new Negamax(simple, dW, true, 1 << 16).setThreads(1), nB = new Negamax(simple, dB, true, 1 << 16).setThreads(1);
		TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
		double[] full = new double[NF];
		GameRecord rec = new GameRecord();
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		GameTracker game = new GameTracker(s);
		for (int ply = 0; ply < 200 && !AshtonRules.isTerminal(s); ply++) {
			List<Action> mv = AshtonRules.generateMoves(s);
			if (mv.isEmpty()) {
				break;
			}
			TablutFeatures.extract(s, ws, full, TablutFeatures.G_ALL);
			float[] f = new float[NF];
			for (int i = 0; i < NF; i++) {
				f[i] = (float) full[i];
			}
			rec.cells.add(codesOf(s));
			rec.feats.add(f);
			rec.role.add((byte) NeuralEvaluator.roleOf(s));
			boolean white = s.getTurn() == Turn.WHITE;
			Action m;
			if (rnd.nextDouble() < (white ? epsW : epsB)) {
				m = mv.get(rnd.nextInt(mv.size()));
			} else {
				m = (white ? nW : nB).findBestMove(s, 5000, game.history());
			}
			if (m == null) {
				m = mv.get(rnd.nextInt(mv.size()));
			}
			s = game.apply(s, m);
			if (game.isRepetitionDraw()) {
				break;
			}
		}
		GameTracker.Result end = game.result(s);
		rec.outcome = (int) end.outcome;
		rec.truncated = GameTracker.LABEL_MAXPLIES.equals(end.label);
		rec.end = s;
		return rec;
	}

	// ====================================================================== addestramento

	/** Simmetria (0..7) del campione {@code i} all'epoca {@code epoch}: funzione deterministica, indipendente da come si divide il lavoro. */
	static int symOf(long seed, int epoch, int i) {
		long x = seed * 0x9E3779B97F4A7C15L + (epoch + 1) * 0xBF58476D1CE4E5B9L + (i + 1) * 0x94D049BB133111EBL;
		x ^= x >>> 30;
		x *= 0xBF58476D1CE4E5B9L;
		x ^= x >>> 27;
		x *= 0x94D049BB133111EBL;
		x ^= x >>> 31;
		return (int) (x & 7);
	}

	/** Stato di UN thread durante un minibatch: workspace privato, gradiente privato, ingressi di feature. */
	static final class Worker {
		final ConvNet net;
		final ConvNet.Workspace ws;
		final float[] grad;
		final float[] fbuf;
		double loss;

		Worker(ConvNet net) {
			this.net = net;
			this.ws = net.newWorkspace();
			this.grad = new float[net.numParams()];
			this.fbuf = new float[net.config().nFeat];
		}

		/** Accumula in {@link #grad} il gradiente di 1/2 (uscita - esito)^2 sui campioni {@code idx[from..to)}. */
		void run(Dataset d, int[] cols, int[] idx, int from, int to, int epoch, long seed, boolean augment) {
			Arrays.fill(grad, 0f);
			loss = 0;
			for (int k = from; k < to; k++) {
				final int i = idx[k];
				ws.setCodes(d.cells, i * 81, augment ? symOf(seed, epoch, i) : 0);
				for (int j = 0; j < cols.length; j++) {
					fbuf[j] = d.feats[i * NF + cols[j]];
				}
				final double err = net.forward(ws, fbuf, d.role[i]) - d.outcome[i];
				loss += err * err;
				net.backward(ws, err, grad);
			}
		}
	}

	/**
	 * Gradiente SOMMA su {@code idx[from..to)} calcolato da {@code workers.length} thread (la rete e' in sola lettura). Il risultato
	 * (in {@code out}, azzerato) e' lo stesso del calcolo a thread singolo a meno dell'ordine delle somme in virgola mobile.
	 *
	 * @return somma degli errori quadratici del minibatch
	 */
	static double batchGradient(ExecutorService pool, Worker[] workers, Dataset d, int[] cols, int[] idx, int from, int to, int epoch,
			long seed, boolean augment, float[] out) throws Exception {
		final int T = workers.length;
		final int len = to - from;
		List<Callable<Void>> tasks = new ArrayList<>();
		for (int t = 0; t < T; t++) {
			final int a = from + (int) ((long) len * t / T), b = from + (int) ((long) len * (t + 1) / T);
			final Worker w = workers[t];
			final int ep = epoch;
			tasks.add(new Callable<Void>() {
				@Override
				public Void call() {
					w.run(d, cols, idx, a, b, ep, seed, augment);
					return null;
				}
			});
		}
		if (T == 1) {
			tasks.get(0).call();
		} else {
			for (Future<Void> f : pool.invokeAll(tasks)) {
				f.get();
			}
		}
		Arrays.fill(out, 0f);
		double loss = 0;
		for (Worker w : workers) {
			for (int k = 0; k < out.length; k++) {
				out[k] += w.grad[k];
			}
			loss += w.loss;
		}
		return loss;
	}

	/** Errore quadratico medio (senza augmentation) su {@code idx}. */
	static double mse(ConvNet net, Dataset d, int[] cols, int[] idx) {
		Worker w = new Worker(net);
		double s = 0;
		for (int i : idx) {
			w.ws.setCodes(d.cells, i * 81, 0);
			for (int j = 0; j < cols.length; j++) {
				w.fbuf[j] = d.feats[i * NF + cols[j]];
			}
			double e = net.forward(w.ws, w.fbuf, d.role[i]) - d.outcome[i];
			s += e * e;
		}
		return idx.length == 0 ? 0 : s / idx.length;
	}

	/** Indici di train e di test (~8% delle partite, scelte per indice di partita). */
	static int[][] split(Dataset d) {
		int nt = 0;
		for (int i = 0; i < d.n; i++) {
			if (d.game[i] % 12 == 0) {
				nt++;
			}
		}
		int[] test = new int[nt], train = new int[d.n - nt];
		int a = 0, b = 0;
		for (int i = 0; i < d.n; i++) {
			if (d.game[i] % 12 == 0) {
				test[a++] = i;
			} else {
				train[b++] = i;
			}
		}
		return new int[][] { train, test };
	}

	/**
	 * Addestra {@code net} sul dataset. Ritorna {errore train dell'ultima epoca (sul dato aumentato), errore held-out}.
	 *
	 * @param cols colonne delle feature a mano da passare alla rete (lunghezza = nFeat; vuoto per il CNN puro)
	 */
	public static double[] train(ConvNet net, Dataset d, int[] cols, int[] trainIdx, int[] testIdx, int epochs, double lr0, int batch,
			int threads, boolean augment, long seed, PrintStream log) throws Exception {
		if (cols.length != net.config().nFeat) {
			throw new IllegalArgumentException("colonne di feature (" + cols.length + ") != nFeat della rete (" + net.config().nFeat + ")");
		}
		ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
		try {
			Worker[] workers = new Worker[Math.max(1, threads)];
			for (int i = 0; i < workers.length; i++) {
				workers[i] = new Worker(net);
			}
			Adam adam = new Adam(net.numParams());
			float[] g = new float[net.numParams()];
			Random rnd = new Random(seed);
			int[] idx = trainIdx.clone();
			double trainMse = 0, testMse = mse(net, d, cols, testIdx);
			if (log != null) {
				log.printf("  epoca  0 | MSE held-out=%.4f (rete non addestrata)%n", testMse);
			}
			for (int ep = 0; ep < epochs; ep++) {
				for (int i = idx.length - 1; i > 0; i--) {
					int j = rnd.nextInt(i + 1);
					int t = idx[i];
					idx[i] = idx[j];
					idx[j] = t;
				}
				double lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs));
				double loss = 0;
				long cnt = 0;
				long t0 = System.nanoTime();
				for (int b = 0; b + batch <= idx.length; b += batch) {
					loss += batchGradient(pool, workers, d, cols, idx, b, b + batch, ep, seed, augment, g);
					adam.step(net.params(), g, lr, 1.0 / batch);
					cnt += batch;
				}
				trainMse = loss / Math.max(1, cnt);
				testMse = mse(net, d, cols, testIdx);
				if (log != null) {
					log.printf("  epoca %2d | MSE train=%.4f | MSE held-out=%.4f | lr=%.5f | %.1fs%n", ep + 1, trainMse, testMse, lr,
							(System.nanoTime() - t0) / 1e9);
				}
			}
			return new double[] { trainMse, testMse };
		} finally {
			pool.shutdown();
		}
	}

	/** MSE del predittore costante (media dell'esito sul train) sul test: il riferimento sotto cui una rete "impara qualcosa". */
	static double constantBaseline(Dataset d, int[] trainIdx, int[] testIdx) {
		double mean = 0;
		for (int i : trainIdx) {
			mean += d.outcome[i];
		}
		mean /= Math.max(1, trainIdx.length);
		double s = 0;
		for (int i : testIdx) {
			double e = mean - d.outcome[i];
			s += e * e;
		}
		return s / Math.max(1, testIdx.length);
	}

	public static void main(String[] args) throws Exception {
		Map<String, String> p = new HashMap<>();
		for (String a : args) {
			int eq = a.indexOf('=');
			if (eq > 0) {
				p.put(a.substring(0, eq), a.substring(eq + 1));
			}
		}
		int games = Integer.parseInt(p.getOrDefault("games", "7000"));
		int epochs = Integer.parseInt(p.getOrDefault("epochs", "10"));
		int[] conv = ConvNet.parseList(p.getOrDefault("conv", "4"));
		int reduce = Integer.parseInt(p.getOrDefault("reduce", "2"));
		int[] dense = ConvNet.parseList(p.getOrDefault("dense", "32"));
		int[] head = ConvNet.parseList(p.getOrDefault("head", "8"));
		boolean hybrid = !p.getOrDefault("features", "1").equals("0");
		int groups = Integer.parseInt(p.getOrDefault("groups", String.valueOf(TablutFeatures.G_DEFAULT)));
		double lr = Double.parseDouble(p.getOrDefault("lr", "0.002"));
		int batch = Integer.parseInt(p.getOrDefault("batch", "128"));
		boolean augment = !p.getOrDefault("augment", "1").equals("0");
		long seed = Long.parseLong(p.getOrDefault("seed", "1"));
		int threads = Integer.parseInt(p.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
		String out = p.getOrDefault("out", "cnn_pretrained.txt");

		// init=<pesi>: riparte da pesi esistenti (forma, gruppi di feature e modalita' ibrida/pura vengono dal file; conv/reduce/dense/head/
		// features/groups sono ignorati). E' il punto di partenza dei fine-tuning (si veda CnnDistiller): con questo lr va abbassato (es. 2e-4).
		CnnEvaluator initEv = p.containsKey("init") ? CnnEvaluator.fromFile(p.get("init")) : null;
		if (initEv != null) {
			hybrid = initEv.isHybrid();
			groups = initEv.featureGroups;
		}
		int[] cols = hybrid ? TablutFeatures.columnsFor(groups) : new int[0];
		ConvNet.Config cfg = initEv != null ? initEv.net.config() : new ConvNet.Config(conv, reduce, cols.length, dense, head);
		ConvNet net = initEv != null ? initEv.net.copy() : new ConvNet(cfg, seed);
		System.out.println("=== CnnPretrainer === " + (initEv != null ? "init=" + p.get("init") + " | " : "") + cfg + " | " + net.numParams() + " parametri, ~" + net.macsPerEval() + " MAC/valutazione | gruppi="
				+ (hybrid ? groups : "nessuno (CNN puro)") + " epochs=" + epochs + " lr=" + lr + " batch=" + batch + " augment=" + augment
				+ " threads=" + threads);

		Dataset d;
		long t0 = System.currentTimeMillis();
		if (p.containsKey("data")) {
			d = Dataset.load(p.get("data"));
			System.out.println("Dataset caricato da " + p.get("data") + ": " + d.n + " posizioni");
		} else {
			d = generate(games, seed, threads, System.out);
			System.out.printf("Generate %d posizioni da %d partite in %.1fs%n", d.n, games, (System.currentTimeMillis() - t0) / 1000.0);
			if (p.containsKey("dataOut")) {
				d.save(p.get("dataOut"));
				System.out.println("Dataset salvato in " + p.get("dataOut") + " (si riusa con data=" + p.get("dataOut") + ")");
			}
		}
		int[][] sp = split(d);
		int white = 0, black = 0;
		for (int i = 0; i < d.n; i++) {
			white += d.outcome[i] > 0 ? 1 : 0;
			black += d.outcome[i] < 0 ? 1 : 0;
		}
		System.out.printf("Train %d / test %d posizioni | posizioni di partite vinte dal Bianco=%d dal Nero=%d | MSE del predittore costante sul test=%.4f%n",
				sp[0].length, sp[1].length, white, black, constantBaseline(d, sp[0], sp[1]));

		double[] r = train(net, d, cols, sp[0], sp[1], epochs, lr, batch, threads, augment, seed, System.out);
		new CnnEvaluator(net, hybrid ? groups : 0).save(out);
		System.out.printf("Salvato %s | MSE held-out finale=%.4f. Uso: CnnSelfPlayTrainer resumeFrom=%s%n", out, r[1], out);
	}
}
