package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
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

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
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
 * Pre-addestramento SUPERVISIONATO della rete: un "warm start" per il TD
 * self-play di {@link SelfPlayTrainer}, che da pesi casuali impiega migliaia di
 * episodi anche solo per smettere di essere costante (misurato: dopo 100
 * episodi da pesi casuali la rete valutava ogni posizione ~-0,77 con
 * deviazione standard 0,11).
 *
 * <h3>Cosa fa</h3>
 * <ol>
 * <li><b>Genera partite</b> con giocatori di stile e forza diversi
 * ({@link SimpleEvaluator} a profondita' 1/2/3, con una quota di mosse casuali per
 * varieta'; 40% pari-forza, 30% profondita' 1 contro 3, 30% contro un avversario
 * quasi casuale) e per ogni posizione registra le feature della rete, chi
 * muoveva e l'esito finale della partita (+1 Bianco, -1 Nero, 0 pareggio). Le regole di fine partita sono quelle di
 * {@link GameTracker}: stallo = sconfitta di chi non puo' muovere, ripetizione = pareggio (la ricerca riceve la storia).
 * Le partite troncate al tetto di semi-mosse vengono scartate.</li>
 * <li><b>Addestra</b> la rete a predire l'esito dalla posizione (errore
 * quadratico, Adam, coseno decrescente) con la head del ruolo di chi muove.
 * L'8% circa delle PARTITE e' tenuto da parte e solo valutato: l'errore
 * "held-out" mostra se la rete generalizza.</li>
 * <li><b>Salva</b> nel formato di {@link NeuralEvaluator#save}, pronto per
 * {@code SelfPlayTrainer resumeFrom=...} o per il client.</li>
 * </ol>
 *
 * <h3>Limiti (leggere prima di fidarsi dei pesi)</h3>
 * Le etichette vengono da partite di giocatori deboli e rumorose: il segnale e'
 * "chi vince da qui, giocando cosi'", non il valore teorico della posizione. Serve
 * a partire da pesi sensati (nei test del progetto batte {@code SimpleEvaluator}
 * a tempo uguale), NON a sostituire il TD self-play.
 *
 * <p>
 * Argomenti {@code chiave=valore}, tutti opzionali: games(7000), epochs(20),
 * groups(G_DEFAULT), trunk(24), head(8), act(LEAKY), lr(0.003), batch(128),
 * seed(1), threads(tutti i core, solo per la generazione), out(pretrained_weights.txt).
 */
public class SupervisedPretrainer {

	/** Una partita registrata: ingressi compatti per posizione, ruolo di chi muoveva, esito. */
	private static final class GameRecord {
		final List<float[]> x = new ArrayList<>();
		final List<Integer> role = new ArrayList<>();
		int outcome;
		boolean truncated;
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
		int epochs = Integer.parseInt(p.getOrDefault("epochs", "20"));
		int groups = Integer.parseInt(p.getOrDefault("groups", String.valueOf(TablutFeatures.G_DEFAULT)));
		int[] trunk = parseList(p.getOrDefault("trunk", "24"));
		int[] head = parseList(p.getOrDefault("head", "8"));
		MLP.Act act = MLP.Act.valueOf(p.getOrDefault("act", "LEAKY").toUpperCase());
		double lr0 = Double.parseDouble(p.getOrDefault("lr", "0.003"));
		int batch = Integer.parseInt(p.getOrDefault("batch", "128"));
		long seed = Long.parseLong(p.getOrDefault("seed", "1"));
		int threads = Integer.parseInt(p.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
		String out = p.getOrDefault("out", "pretrained_weights.txt");

		MLP.Config cfg = MLP.Config.sharedTrunk(TablutFeatures.columnsFor(groups).length, trunk, head, act);
		MLP net = new MLP(cfg, seed);
		NeuralEvaluator shape = new NeuralEvaluator(net, groups);
		System.out.println("=== SupervisedPretrainer === " + cfg + " gruppi=" + groups + " games=" + games + " epochs=" + epochs
				+ " threads=" + threads);

		// ---------------------------------------------------------------- generazione
		long t0 = System.currentTimeMillis();
		List<GameRecord> recs = generate(games, seed, groups, threads, shape);
		int nGames = 0, nPos = 0, white = 0, black = 0;
		for (GameRecord g : recs) {
			if (g.truncated) {
				continue;
			}
			nGames++;
			nPos += g.x.size();
			if (g.outcome > 0) {
				white++;
			} else if (g.outcome < 0) {
				black++;
			}
		}
		System.out.printf("Generate %d partite (non troncate) / %d posizioni in %.1fs | vittorie Bianco=%d Nero=%d%n", nGames, nPos,
				(System.currentTimeMillis() - t0) / 1000.0, white, black);

		// split per partita: ~8% held-out
		List<float[]> trX = new ArrayList<>(), teX = new ArrayList<>();
		List<Integer> trR = new ArrayList<>(), teR = new ArrayList<>();
		List<Float> trZ = new ArrayList<>(), teZ = new ArrayList<>();
		int gi = 0;
		for (GameRecord g : recs) {
			if (g.truncated) {
				continue;
			}
			boolean test = (gi++ % 12) == 0;
			for (int i = 0; i < g.x.size(); i++) {
				(test ? teX : trX).add(g.x.get(i));
				(test ? teR : trR).add(g.role.get(i));
				(test ? teZ : trZ).add((float) g.outcome);
			}
		}

		// ---------------------------------------------------------------- addestramento
		double[] w = net.params(), g = new double[w.length], m = new double[w.length], v = new double[w.length];
		MLP.Scratch sc = net.newScratch();
		Random rnd = new Random(seed);
		int n = trX.size();
		int[] idx = new int[n];
		for (int i = 0; i < n; i++) {
			idx[i] = i;
		}
		double[] xin = new double[cfg.nIn];
		long step = 0;
		for (int ep = 0; ep < epochs; ep++) {
			for (int i = n - 1; i > 0; i--) {
				int j = rnd.nextInt(i + 1);
				int t = idx[i];
				idx[i] = idx[j];
				idx[j] = t;
			}
			double lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * ep / epochs));
			double loss = 0;
			int cnt = 0;
			for (int b = 0; b + batch <= n; b += batch) {
				Arrays.fill(g, 0);
				for (int k = 0; k < batch; k++) {
					int i = idx[b + k];
					copy(trX.get(i), xin);
					double e = net.forward(xin, trR.get(i), sc) - trZ.get(i);
					loss += e * e;
					cnt++;
					net.backward(sc, e, g);
				}
				step++;
				double c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
				for (int k = 0; k < w.length; k++) {
					double gk = g[k] / batch;
					m[k] = 0.9 * m[k] + 0.1 * gk;
					v[k] = 0.999 * v[k] + 0.001 * gk * gk;
					w[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
				}
			}
			if (ep % 5 == 4 || ep == epochs - 1) {
				double te = 0;
				for (int i = 0; i < teX.size(); i++) {
					copy(teX.get(i), xin);
					double e = net.forward(xin, teR.get(i), sc) - teZ.get(i);
					te += e * e;
				}
				System.out.printf("  epoca %2d | MSE train=%.4f | MSE held-out=%.4f%n", ep + 1, loss / Math.max(1, cnt),
						te / Math.max(1, teX.size()));
			}
		}
		shape.save(out);
		System.out.println("Salvato " + out + " (" + net.numParams() + " parametri). Uso: SelfPlayTrainer resumeFrom=" + out);
	}

	private static void copy(float[] src, double[] dst) {
		for (int i = 0; i < src.length; i++) {
			dst[i] = src[i];
		}
	}

	private static int[] parseList(String s) {
		if (s.equalsIgnoreCase("none") || s.equals("-")) {
			return new int[0];
		}
		String[] t = s.split(",");
		int[] r = new int[t.length];
		for (int i = 0; i < t.length; i++) {
			r[i] = Integer.parseInt(t[i].trim());
		}
		return r;
	}

	// ---------------------------------------------------------------- generazione partite

	private static List<GameRecord> generate(int games, long seed, int groups, int threads, NeuralEvaluator shape)
			throws Exception {
		ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
		try {
			List<Future<GameRecord>> fs = new ArrayList<>();
			Random master = new Random(seed * 31 + 7);
			for (int gIdx = 0; gIdx < games; gIdx++) {
				final long gameSeed = master.nextLong();
				fs.add(pool.submit((Callable<GameRecord>) () -> playOne(gameSeed, groups, shape)));
			}
			List<GameRecord> out = new ArrayList<>();
			for (int i = 0; i < fs.size(); i++) {
				out.add(fs.get(i).get());
				if ((i + 1) % 1000 == 0) {
					System.out.println("  generate " + (i + 1) + "/" + games + " partite");
				}
			}
			return out;
		} finally {
			pool.shutdown();
		}
	}

	/** Una partita: stile scelto a caso, posizioni registrate prima di ogni mossa. */
	private static GameRecord playOne(long gameSeed, int groups, NeuralEvaluator shape) throws IOException {
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
		double[] full = new double[TablutFeatures.NUM_FEATURES];
		double[] x = new double[shape.numInputs()];

		GameRecord rec = new GameRecord();
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		GameTracker game = new GameTracker(s); // storia di ripetizione ed esito coerenti con la ricerca (si veda GameTracker)
		int ply = 0;
		for (; ply < 200 && !AshtonRules.isTerminal(s); ply++) {
			List<Action> mv = AshtonRules.generateMoves(s);
			if (mv.isEmpty()) {
				break;
			}
			shape.extractInputs(s, ws, full, x);
			float[] f = new float[x.length];
			for (int i = 0; i < f.length; i++) {
				f[i] = (float) x[i];
			}
			rec.x.add(f);
			rec.role.add(NeuralEvaluator.roleOf(s));
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
				break; // posizione ripetuta = pareggio, come nella partita vera
			}
		}
		// vittoria, STALLO (chi non puo' muovere perde), ripetizione (pareggio) o troncata: si veda GameTracker
		GameTracker.Result end = game.result(s);
		rec.outcome = (int) end.outcome;
		rec.truncated = GameTracker.LABEL_MAXPLIES.equals(end.label); // solo le partite troncate al tetto vengono scartate
		return rec;
	}
}
