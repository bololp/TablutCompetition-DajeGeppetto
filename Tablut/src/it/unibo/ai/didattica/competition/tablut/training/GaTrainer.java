package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.zip.CRC32;

import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;

/**
 * Algoritmo genetico che ottimizza, PER PARTITE contro il riferimento, un sottoinsieme di parametri della rete NNUE ({@link GaGenome}).
 * Si esegue OFFLINE, prima del torneo; in torneo non evolve nulla (si veda {@code TablutGeneticClient}).
 *
 * <h3>Individuo</h3>
 * Un {@link GaGenome}: geni reali in [-1,1] = scostamenti normalizzati dei parametri scelti (gruppi {@code head,l1b,ftb}) rispetto alla base.
 *
 * <h3>Popolazione</h3>
 * {@code pop} individui. Generazione 0: il genoma NEUTRO (= il riferimento, fitness 0,5 per costruzione) piu' individui casuali gaussiani attorno
 * al riferimento (ampiezza individuale in [0,25, 1] * {@code sigma}) piu', se dati, i genomi di {@code init=} (partenza a caldo).
 *
 * <h3>Fitness</h3>
 * Per ogni generazione si estraggono {@code pairs} semi di apertura, uguali per tutti gli individui (numeri casuali comuni: riduce la varianza
 * dei confronti); ogni individuo gioca contro il RIFERIMENTO una {@link GaMatch#playPair coppia} per seme (colori invertiti, profondita' fissa
 * {@code depth}, regole vere). {@code raw} = media dei punteggi delle coppie in [0,1] (0,5 = pari al riferimento). La fitness di selezione e'
 * {@code raw - lambda * meanSquare(geni)}: una piccola PRESSIONE DI PARSIMONIA che a parita' di forza preferisce l'individuo piu' vicino al
 * riferimento e frena la deriva casuale indotta dal rumore delle partite. I semi cambiano a ogni generazione (niente sovradattamento a poche
 * aperture); gli elitari sono RIVALUTATI sui nuovi semi (la loro fitness non e' mai quella, fortunata, di una generazione passata).
 *
 * <h3>Selezione, crossover, mutazione</h3>
 * <ul>
 * <li>Elitismo: i {@code elite} migliori passano invariati.</li>
 * <li>Genitori: torneo di {@code tourn} individui estratti a caso (vince la fitness migliore).</li>
 * <li>Crossover BLX-alfa con probabilita' {@code pc}: ogni gene del figlio e' uniforme in [min - alfa*d, max + alfa*d] (d = |a-b|), troncato a [-1,1];
 * altrimenti il figlio parte da una copia del primo genitore.</li>
 * <li>Mutazione: ogni gene con probabilita' {@code pm} (default max(0,05; 3/n)) riceve un rumore gaussiano di deviazione
 * {@code max(sigmaMin, sigma * sigmaDecay^generazione)}, troncato a [-1,1]; se il figlio sarebbe una copia del genitore si muta un gene a caso.</li>
 * <li>{@code immigrants} individui casuali per generazione (diversita').</li>
 * </ul>
 *
 * <h3>Arresto</h3>
 * (1) {@code gens} generazioni; (2) {@code budget} secondi (controllato fra generazioni, stimando la durata della successiva sull'ultima); (3) stagnazione:
 * il miglior valore "liscio" (media della fitness di un individuo su tutte le sue valutazioni, solo con almeno {@code minEvals} valutazioni) non migliora
 * di {@code minDelta} per {@code patience} generazioni.
 *
 * <h3>Dal risultato alla rete di gioco (cancello)</h3>
 * Il CAMPIONE e' scelto con le sole statistiche di addestramento (miglior media liscia). Viene poi giocato UNA volta contro il riferimento su semi
 * DISGIUNTI da quelli di addestramento, con {@code valPairs} coppie e profondita' {@code valDepth} (default depth+1: controlla che il vantaggio
 * regga a una profondita' diversa). Un solo candidato validato = nessun bias da selezione sul set di validazione. {@code final.nnue} e' il
 * campione SOLO se il verdetto e' {@link GaMatch#SUPERIORE} (con {@code acceptNonInferior=1} anche {@link GaMatch#NON_INFERIORE}); altrimenti e' una copia
 * bit per bit della base: la rete finale non e' mai peggiore del riferimento per costruzione, e un successo e' dichiarato solo con evidenza.
 * <b>Il cancello a profondita' fissa non e' la prova di forza a 60 s</b>: per quella si veda {@code README_GENETIC.md} (protocollo di torneo).
 *
 * <h3>Riproducibilita' e ripresa</h3>
 * Ogni operazione casuale di una generazione usa un {@link Random} derivato da (seme, generazione); le partite sono deterministiche a profondita' fissa:
 * stesso comando = stessi risultati, con qualunque {@code threads}. Dopo ogni generazione si scrive {@code OUT.ckpt}; con {@code resume=1} si riparte da li'
 * ottenendo lo stesso risultato di una esecuzione ininterrotta (anche estendendo {@code gens}).
 *
 * <p>
 * Uso: si veda {@link #main}.
 */
public final class GaTrainer {

	private GaTrainer() {
	}

	/** Parametri della GA (valori di default pensati per una macchina multi-core; si veda README_GENETIC.md). */
	public static final class Config {
		public String base = "pesi_nnue/trained_v6.nnue";
		public String out = "ga/run";
		public String groups = "head,l1b";
		public String init = null;
		public int pop = 24, gens = 30, pairs = 6, depth = 4, openRandom = 6, maxPlies = 200;
		public int elite = 2, tourn = 3, immigrants = 1;
		public double pc = 0.8, alpha = 0.3, sigma = 0.15, sigmaDecay = 0.95, sigmaMin = 0.02, pm = -1, lambda = 0.02;
		public int patience = 8, minEvals = 2;
		public double minDelta = 0.005;
		public long budgetSeconds = 0;
		public long seed = 1;
		public int threads = 1;
		public boolean resume = false;
		public int valPairs = 100, valDepth = -1, minPairs = 30;
		public double z = 1.96, margin = 0.02;
		public boolean acceptNonInferior = false;

		void validate() {
			if (pop < 4 || gens < 1 || pairs < 1 || depth < 1 || depth > 40 || openRandom < 0 || elite < 1 || elite + immigrants >= pop || tourn < 1 || immigrants < 0
					|| pc < 0 || pc > 1 || alpha < 0 || alpha > 2 || sigma <= 0 || sigma > 1 || sigmaDecay <= 0 || sigmaDecay > 1 || sigmaMin < 0 || sigmaMin > sigma
					|| pm > 1 || lambda < 0 || lambda > 1 || patience < 1 || minEvals < 1 || minDelta < 0 || budgetSeconds < 0 || threads < 1 || valPairs < 1 || valDepth == 0
					|| minPairs < 1 || z <= 0 || margin < 0 || margin >= 0.5) {
				throw new IllegalArgumentException("configurazione non valida: " + describe());
			}
		}

		int validationDepth() {
			return valDepth > 0 ? valDepth : depth + 1;
		}

		String describe() {
			return "base=" + base + " out=" + out + " groups=" + groups + " pop=" + pop + " gens=" + gens + " pairs=" + pairs + " depth=" + depth + " open=" + openRandom
					+ " elite=" + elite + " tourn=" + tourn + " immigrants=" + immigrants + " pc=" + pc + " alpha=" + alpha + " sigma=" + sigma + " sigmadecay=" + sigmaDecay
					+ " sigmamin=" + sigmaMin + " pm=" + pm + " lambda=" + lambda + " patience=" + patience + " minevals=" + minEvals + " mindelta=" + minDelta + " budget=" + budgetSeconds
					+ " seed=" + seed + " threads=" + threads + " valpairs=" + valPairs + " valdepth=" + valDepth + " z=" + z + " margin=" + margin + " acceptni=" + acceptNonInferior;
		}

		/** Parametri che cambiano il COMPORTAMENTO della GA: devono coincidere per riprendere un checkpoint (gens, budget, patience, validazione e threads possono cambiare). */
		String resumeKey(long baseFingerprint) {
			return String.format("%08x|%s|%d|%d|%d|%d|%d|%d|%d|%d|%s|%s|%s|%s|%s|%s|%s|%d|%d", baseFingerprint, groups.toLowerCase(), pop, pairs, depth, openRandom, maxPlies, elite, tourn, immigrants,
					pc, alpha, sigma, sigmaDecay, sigmaMin, pm, lambda, seed, minEvals);
		}

		double effectivePm(int n) {
			return pm >= 0 ? pm : Math.min(1.0, Math.max(0.05, 3.0 / n));
		}
	}

	/** Valuta una generazione: punteggio grezzo in [0,1] di ciascun individuo contro il riferimento, sugli stessi semi di apertura. */
	public interface Fitness {
		double[] evaluate(List<GaGenome> individuals, long[] pairSeeds, int generation) throws Exception;
	}

	/** Le statistiche di una generazione (per il log e per i test). */
	public static final class GenStat {
		public int gen;
		public double sigma, bestRaw, meanRaw, bestFit, seconds;
		public int played;

		@Override
		public String toString() {
			return String.format("gen %3d  sigma %.3f  raw: migliore %.4f medio %.4f  fitness migliore %.4f  coppie giocate %d  %.1f s", gen, sigma, bestRaw, meanRaw, bestFit, played, seconds);
		}
	}

	/** Esito dell'intera esecuzione. */
	public static final class Result {
		public GaGenome champion;
		public double championFit;
		public int championEvals;
		public String stopReason;
		public int generations;
		public List<GenStat> history = new ArrayList<>();
		public GaMatch.Summary validation;
		public String verdict;
		public boolean accepted;
		/** Il genoma che dà la rete finale: il campione se accettato, altrimenti il neutro. */
		public GaGenome finalGenome;
	}

	// ====================================================================== stato

	private static final class Entry {
		final GaGenome genome;
		double sumRaw;
		int count;

		Entry(GaGenome g) {
			genome = g;
		}

		double meanRaw() {
			return sumRaw / count;
		}

		double meanFit(double lambda) {
			return meanRaw() - lambda * genome.meanSquare();
		}
	}

	private static final class State {
		int nextGen;
		List<GaGenome> pop = new ArrayList<>();
		final Map<String, Entry> hof = new LinkedHashMap<>();
		double bestSmooth = Double.NEGATIVE_INFINITY;
		int sinceImprove;
	}

	static long mix(long a, long b, long c) {
		long x = a * 0x9E3779B97F4A7C15L + b;
		x ^= x >>> 30;
		x *= 0xBF58476D1CE4E5B9L;
		x ^= x >>> 27;
		x = (x + c) * 0x94D049BB133111EBL;
		x ^= x >>> 31;
		return x;
	}

	private static final long TAG_TRAIN = 0x7472616eL, TAG_VALID = 0x76616c69L, TAG_GEN = 0x67656e65L, TAG_INIT = 0x696e6974L;

	/** I semi di apertura di addestramento di una generazione. */
	static long[] trainSeeds(long seed, int gen, int n) {
		final Random r = new Random(mix(seed, TAG_TRAIN, gen));
		final long[] s = new long[n];
		for (int i = 0; i < n; i++) {
			s[i] = r.nextLong();
		}
		return s;
	}

	/** I semi di validazione: stesso schema con un'etichetta diversa, quindi (con probabilita' 1 - 2^-60 per coppia) disgiunti da quelli di addestramento di qualunque generazione. */
	static long[] validationSeeds(long seed, int n) {
		final Random r = new Random(mix(seed, TAG_VALID, 0));
		final long[] s = new long[n];
		for (int i = 0; i < n; i++) {
			s[i] = r.nextLong();
		}
		return s;
	}

	// ====================================================================== operatori genetici (package-private: li esercitano i test)

	static double clip(double v) {
		return v < -1.0 ? -1.0 : v > 1.0 ? 1.0 : v;
	}

	/** Individuo casuale: geni gaussiani di deviazione {@code scale} * u, con u in [0,25, 1] per individuo. */
	static GaGenome randomIndividual(GaGenome.Layout layout, double scale, Random rnd) {
		final double s = scale * (0.25 + 0.75 * rnd.nextDouble());
		final double[] g = new double[layout.size()];
		for (int i = 0; i < g.length; i++) {
			g[i] = clip(rnd.nextGaussian() * s);
		}
		return new GaGenome(layout, g);
	}

	/** Crossover BLX-alfa: gene del figlio uniforme in [min - alfa*d, max + alfa*d], troncato a [-1,1]. */
	static double[] blx(double[] a, double[] b, double alpha, Random rnd) {
		final double[] c = new double[a.length];
		for (int i = 0; i < c.length; i++) {
			final double lo = Math.min(a[i], b[i]), hi = Math.max(a[i], b[i]), d = hi - lo;
			c[i] = clip(lo - alpha * d + rnd.nextDouble() * (d * (1 + 2 * alpha)));
		}
		return c;
	}

	/** Mutazione gaussiana per gene; restituisce quanti geni sono cambiati (se nessuno e {@code forceOne}, ne muta uno a caso). */
	static int mutate(double[] g, double pm, double sigma, boolean forceOne, Random rnd) {
		int changed = 0;
		for (int i = 0; i < g.length; i++) {
			if (rnd.nextDouble() < pm) {
				final double v = clip(g[i] + rnd.nextGaussian() * sigma);
				if (v != g[i]) {
					changed++;
				}
				g[i] = v;
			}
		}
		if (changed == 0 && forceOne) {
			final int i = rnd.nextInt(g.length);
			g[i] = clip(g[i] + (rnd.nextBoolean() ? 1 : -1) * Math.max(1e-3, Math.abs(rnd.nextGaussian() * sigma)));
			changed = 1;
		}
		return changed;
	}

	/** Torneo: {@code k} indici a caso (con ripetizione), vince la fitness migliore (a parita' l'indice minore). */
	static int tournament(double[] fit, int k, Random rnd) {
		int best = rnd.nextInt(fit.length);
		for (int t = 1; t < k; t++) {
			final int c = rnd.nextInt(fit.length);
			if (fit[c] > fit[best] || (fit[c] == fit[best] && c < best)) {
				best = c;
			}
		}
		return best;
	}

	static double sigmaAt(Config cfg, int gen) {
		return Math.max(cfg.sigmaMin, cfg.sigma * Math.pow(cfg.sigmaDecay, gen));
	}

	// ====================================================================== esecuzione

	/** La fitness vera: coppie di partite contro il riferimento ({@link GaMatch}). Gli individui neutri valgono 0,5 senza giocare (due reti identiche = 0,5 esatto). */
	static final class MatchFitness implements Fitness {
		private final NnueEvaluator reference;
		private final GaMatch.Config mc;
		private final int threads;

		MatchFitness(NnueNetwork base, Config cfg) {
			this.reference = new NnueEvaluator(base);
			this.mc = new GaMatch.Config();
			mc.depth = cfg.depth;
			mc.openRandom = cfg.openRandom;
			mc.maxPlies = cfg.maxPlies;
			this.threads = cfg.threads;
		}

		@Override
		public double[] evaluate(List<GaGenome> inds, final long[] seeds, int generation) throws Exception {
			final int n = inds.size();
			final NnueEvaluator[] ev = new NnueEvaluator[n];
			for (int i = 0; i < n; i++) {
				if (!inds.get(i).isNeutral()) {
					ev[i] = new NnueEvaluator(inds.get(i).decode());
				}
			}
			final ExecutorService pool = Executors.newFixedThreadPool(threads, new ThreadFactory() {
				@Override
				public Thread newThread(Runnable r) {
					final Thread t = new Thread(r, "ga-fitness");
					t.setDaemon(true);
					return t;
				}
			});
			try {
				final List<List<Future<GaMatch.Pair>>> fs = new ArrayList<>();
				for (int i = 0; i < n; i++) {
					final List<Future<GaMatch.Pair>> row = new ArrayList<>();
					if (ev[i] != null) {
						final NnueEvaluator cand = ev[i];
						for (final long s : seeds) {
							row.add(pool.submit(new Callable<GaMatch.Pair>() {
								@Override
								public GaMatch.Pair call() {
									return GaMatch.playPair(cand, reference, mc, s);
								}
							}));
						}
					}
					fs.add(row);
				}
				final double[] raw = new double[n];
				for (int i = 0; i < n; i++) {
					if (ev[i] == null) {
						raw[i] = 0.5;
						continue;
					}
					double sum = 0;
					for (Future<GaMatch.Pair> f : fs.get(i)) {
						sum += f.get().mean();
					}
					raw[i] = sum / seeds.length;
				}
				return raw;
			} finally {
				pool.shutdownNow();
			}
		}
	}

	/** Esegue l'intera GA con la fitness a partite e scrive i file di uscita ({@code OUT.genome, .champion.nnue, .final.nnue, .verdict.txt, .ckpt}). */
	public static Result run(Config cfg, PrintStream log) throws Exception {
		cfg.validate();
		final NnueNetwork base = NnueNetwork.load(cfg.base);
		return run(cfg, base, new MatchFitness(base, cfg), log, true);
	}

	/**
	 * Come {@link #run(Config, PrintStream)} con una fitness fornita (test sul paesaggio sintetico) e, se {@code validate} e' falso, senza il
	 * cancello finale a partite (il verdetto resta {@code null}).
	 */
	public static Result run(Config cfg, NnueNetwork base, Fitness fitness, PrintStream log, boolean validate) throws Exception {
		cfg.validate();
		final GaGenome.Layout layout = GaGenome.Layout.of(base, cfg.groups);
		final long fp = layout.baseFingerprint();
		final String key = cfg.resumeKey(fp);
		final File ckpt = new File(cfg.out + ".ckpt");
		final long t0 = System.currentTimeMillis();
		State st;
		if (cfg.resume && ckpt.isFile()) {
			st = loadCheckpoint(ckpt, layout, key);
			log.println("RIPRESA da " + ckpt + ": generazione " + st.nextGen + ", " + st.hof.size() + " individui valutati nella memoria");
		} else {
			st = initial(cfg, layout, log);
		}
		log.println("GA: " + layout.size() + " geni (gruppi " + layout.groupsText() + "), base impronta " + String.format("%08x", fp) + ", " + cfg.describe());

		final Result res = new Result();
		double lastGenSeconds = 0;
		String stop = "gens";
		final int startGen = st.nextGen; // st.nextGen avanza a ogni generazione: il confronto del budget va fatto con la generazione di PARTENZA
		for (int g = startGen; g < cfg.gens; g++) {
			if (cfg.budgetSeconds > 0 && g > startGen && (System.currentTimeMillis() - t0) / 1000.0 + lastGenSeconds > cfg.budgetSeconds) {
				stop = "budget";
				break;
			}
			final long tg = System.currentTimeMillis();
			final long[] seeds = trainSeeds(cfg.seed, g, cfg.pairs);
			final double[] raw = fitness.evaluate(st.pop, seeds, g);
			if (raw.length != st.pop.size()) {
				throw new IllegalStateException("la fitness ha restituito " + raw.length + " valori per " + st.pop.size() + " individui");
			}
			final double[] fit = new double[raw.length];
			final GenStat gs = new GenStat();
			gs.gen = g;
			gs.sigma = sigmaAt(cfg, g);
			gs.bestRaw = Double.NEGATIVE_INFINITY;
			gs.bestFit = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < raw.length; i++) {
				final GaGenome ind = st.pop.get(i);
				fit[i] = raw[i] - cfg.lambda * ind.meanSquare();
				gs.bestRaw = Math.max(gs.bestRaw, raw[i]);
				gs.bestFit = Math.max(gs.bestFit, fit[i]);
				gs.meanRaw += raw[i] / raw.length;
				if (!ind.isNeutral()) {
					gs.played += seeds.length;
				}
				Entry e = st.hof.get(ind.key());
				if (e == null) {
					e = new Entry(ind);
					st.hof.put(ind.key(), e);
				}
				e.sumRaw += raw[i];
				e.count++;
			}
			// progresso "liscio": miglior media su tutte le valutazioni di un individuo valutato almeno minEvals volte
			final Entry sm = champion(st, cfg, true);
			if (sm != null) {
				final double cur = sm.meanFit(cfg.lambda);
				if (cur > st.bestSmooth + cfg.minDelta) {
					st.bestSmooth = cur;
					st.sinceImprove = 0;
				} else {
					st.sinceImprove++;
				}
			}
			gs.seconds = (System.currentTimeMillis() - tg) / 1000.0;
			lastGenSeconds = gs.seconds;
			res.history.add(gs);
			log.println(gs);

			// prossima popolazione (anche all'ultima generazione: il checkpoint e' cosi' sempre estendibile)
			st.pop = nextPopulation(cfg, layout, st.pop, fit, g);
			st.nextGen = g + 1;
			saveCheckpoint(ckpt, st, key);
			res.generations = g + 1;
			if (sm != null && st.sinceImprove >= cfg.patience) {
				stop = "patience";
				break;
			}
		}
		res.stopReason = stop;
		if (res.generations == 0) {
			res.generations = st.nextGen;
		}
		log.println("arresto: " + stop + " dopo " + res.generations + " generazioni");

		// campione: solo statistiche di addestramento
		Entry champ = champion(st, cfg, true);
		if (champ == null) {
			champ = champion(st, cfg, false);
		}
		res.champion = champ == null ? GaGenome.identity(layout) : champ.genome;
		res.championFit = champ == null ? 0.5 : champ.meanFit(cfg.lambda);
		res.championEvals = champ == null ? 0 : champ.count;
		log.println("campione: " + res.champion + ", fitness media " + String.format("%.4f", res.championFit) + " su " + res.championEvals + " valutazioni");

		// cancello
		res.finalGenome = GaGenome.identity(layout);
		if (res.champion.isNeutral()) {
			res.verdict = "NEUTRO";
			log.println("il campione e' il riferimento: nessun cancello da eseguire");
		} else if (validate) {
			final NnueEvaluator cand = new NnueEvaluator(res.champion.decode());
			final NnueEvaluator ref = new NnueEvaluator(base);
			final GaMatch.Config vc = new GaMatch.Config();
			vc.depth = cfg.validationDepth();
			vc.openRandom = cfg.openRandom;
			vc.maxPlies = cfg.maxPlies;
			final long[] vseeds = validationSeeds(cfg.seed, cfg.valPairs);
			log.println("validazione: " + cfg.valPairs + " coppie a profondita' " + vc.depth + " su semi disgiunti dall'addestramento...");
			res.validation = GaMatch.summarize(GaMatch.playAll(cand, ref, vc, vseeds, cfg.threads), cfg.z);
			res.verdict = GaMatch.verdict(res.validation, cfg.margin, cfg.minPairs);
			res.accepted = isAccepted(res.verdict, cfg.acceptNonInferior);
			if (res.accepted) {
				res.finalGenome = res.champion;
			}
			log.println(res.validation);
			log.println("VERDETTO: " + res.verdict + (res.accepted ? " -> ACCETTATO" : " -> RIFIUTATO (la rete finale e' il riferimento, invariato)"));
		}
		writeOutputs(cfg, base, res);
		return res;
	}

	private static Entry champion(State st, Config cfg, boolean needMinEvals) {
		Entry best = null;
		for (Entry e : st.hof.values()) {
			if (needMinEvals && e.count < cfg.minEvals) {
				continue;
			}
			if (best == null || e.meanFit(cfg.lambda) > best.meanFit(cfg.lambda)) {
				best = e;
			}
		}
		return best;
	}

	private static State initial(Config cfg, GaGenome.Layout layout, PrintStream log) throws IOException {
		final State st = new State();
		final Random rnd = new Random(mix(cfg.seed, TAG_INIT, 0));
		st.pop.add(GaGenome.identity(layout));
		if (cfg.init != null && !cfg.init.isEmpty()) {
			for (String f : cfg.init.split(",")) {
				if (!f.trim().isEmpty() && st.pop.size() < cfg.pop) {
					final GaGenome g = GaGenome.load(f.trim(), layout.base());
					if (!g.layout().groupsText().equals(layout.groupsText())) {
						throw new IOException("init=" + f + ": gruppi " + g.layout().groupsText() + " diversi da quelli richiesti " + layout.groupsText());
					}
					st.pop.add(g);
					log.println("partenza a caldo: " + f);
				}
			}
		}
		while (st.pop.size() < cfg.pop) {
			st.pop.add(randomIndividual(layout, cfg.sigma, rnd));
		}
		return st;
	}

	private static List<GaGenome> nextPopulation(Config cfg, GaGenome.Layout layout, List<GaGenome> pop, double[] fit, int gen) {
		final Random rg = new Random(mix(cfg.seed, TAG_GEN, gen));
		final double sigma = sigmaAt(cfg, gen + 1);
		final double pm = cfg.effectivePm(layout.size());
		final Integer[] order = new Integer[pop.size()];
		for (int i = 0; i < order.length; i++) {
			order[i] = i;
		}
		Arrays.sort(order, new Comparator<Integer>() {
			@Override
			public int compare(Integer a, Integer b) {
				final int c = Double.compare(fit[b], fit[a]);
				return c != 0 ? c : Integer.compare(a, b);
			}
		});
		final List<GaGenome> next = new ArrayList<>();
		for (int e = 0; e < cfg.elite; e++) {
			next.add(pop.get(order[e]));
		}
		while (next.size() < cfg.pop - cfg.immigrants) {
			final int i1 = tournament(fit, cfg.tourn, rg);
			int i2 = tournament(fit, cfg.tourn, rg);
			for (int retry = 0; retry < 3 && i2 == i1; retry++) {
				i2 = tournament(fit, cfg.tourn, rg);
			}
			final boolean cross = rg.nextDouble() < cfg.pc;
			final double[] child = cross ? blx(pop.get(i1).genes(), pop.get(i2).genes(), cfg.alpha, rg) : pop.get(i1).genes();
			mutate(child, pm, sigma, !cross, rg);
			next.add(new GaGenome(layout, child));
		}
		while (next.size() < cfg.pop) {
			next.add(randomIndividual(layout, sigma, rg));
		}
		return next;
	}

	// ====================================================================== file di uscita

	/** Regola di accettazione del cancello: solo SUPERIORE (o anche NON_INFERIORE se richiesto). Ogni altro verdetto (INSUFFICIENTE, INFERIORE, NON_DIMOSTRATO, NEUTRO, null) rifiuta. */
	static boolean isAccepted(String verdict, boolean acceptNonInferior) {
		return GaMatch.SUPERIORE.equals(verdict) || (acceptNonInferior && GaMatch.NON_INFERIORE.equals(verdict));
	}

	static void writeOutputs(Config cfg, NnueNetwork base, Result res) throws IOException {
		final File outFile = new File(cfg.out).getAbsoluteFile();
		final File dir = outFile.getParentFile();
		if (dir != null && !dir.isDirectory() && !dir.mkdirs()) {
			throw new IOException("impossibile creare la cartella " + dir);
		}
		res.champion.save(cfg.out + ".genome");
		res.champion.decode().save(cfg.out + ".champion.nnue");
		if (res.accepted) {
			Files.copy(new File(cfg.out + ".champion.nnue").toPath(), new File(cfg.out + ".final.nnue").toPath(), StandardCopyOption.REPLACE_EXISTING);
		} else {
			Files.copy(new File(cfg.base).toPath(), new File(cfg.out + ".final.nnue").toPath(), StandardCopyOption.REPLACE_EXISTING);
		}
		final StringBuilder sb = new StringBuilder();
		sb.append("base: ").append(cfg.base).append(" (impronta ").append(String.format("%08x", GaGenome.fingerprint(base))).append(")\n");
		sb.append("campione: ").append(res.champion).append(", fitness media ").append(String.format("%.4f", res.championFit)).append(" su ").append(res.championEvals).append(" valutazioni\n");
		sb.append("arresto: ").append(res.stopReason).append(" dopo ").append(res.generations).append(" generazioni\n");
		if (res.validation != null) {
			sb.append("validazione (profondita' ").append(cfg.validationDepth()).append(", semi disgiunti): ").append(res.validation).append('\n');
		}
		sb.append("verdetto: ").append(res.verdict).append(res.accepted ? " -> ACCETTATO (final.nnue = campione)" : " -> final.nnue = base, invariata").append('\n');
		sb.append("ATTENZIONE: il verdetto e' a profondita' fissa e a pochi nodi: NON dimostra la forza a 60 s per mossa (si veda README_GENETIC.md).\n");
		try (Writer w = new OutputStreamWriter(new FileOutputStream(cfg.out + ".verdict.txt"), "UTF-8")) {
			w.write(sb.toString());
		}
	}

	// ====================================================================== checkpoint

	private static void saveCheckpoint(File f, State st, String key) throws IOException {
		final StringBuilder sb = new StringBuilder();
		sb.append("TABLUT_GA_CKPT v1\n");
		sb.append("key ").append(key).append('\n');
		sb.append("nextgen ").append(st.nextGen).append('\n');
		sb.append("progress ").append(Double.toString(st.bestSmooth)).append(' ').append(st.sinceImprove).append('\n');
		sb.append("pop ").append(st.pop.size()).append('\n');
		for (GaGenome g : st.pop) {
			sb.append(csv(g.genes())).append('\n');
		}
		sb.append("hof ").append(st.hof.size()).append('\n');
		for (Entry e : st.hof.values()) {
			sb.append(Double.toString(e.sumRaw)).append(' ').append(e.count).append(' ').append(csv(e.genome.genes())).append('\n');
		}
		final CRC32 crc = new CRC32();
		final byte[] body = sb.toString().getBytes("UTF-8");
		crc.update(body, 0, body.length);
		sb.append("checksum ").append(String.format("%08x", crc.getValue())).append('\n');
		final File dir = f.getAbsoluteFile().getParentFile();
		if (dir != null && !dir.isDirectory() && !dir.mkdirs()) {
			throw new IOException("impossibile creare la cartella " + dir);
		}
		final File tmp = new File(f.getPath() + ".tmp");
		try (Writer w = new OutputStreamWriter(new FileOutputStream(tmp), "UTF-8")) {
			w.write(sb.toString());
		}
		Files.move(tmp.toPath(), f.toPath(), StandardCopyOption.REPLACE_EXISTING);
	}

	private static String csv(double[] a) {
		final StringBuilder sb = new StringBuilder();
		for (int i = 0; i < a.length; i++) {
			sb.append(i == 0 ? "" : ",").append(Double.toString(a[i]));
		}
		return sb.toString();
	}

	private static double[] parseCsv(String s) throws IOException {
		final String[] t = s.split(",");
		final double[] d = new double[t.length];
		try {
			for (int i = 0; i < t.length; i++) {
				d[i] = Double.parseDouble(t[i]);
			}
		} catch (NumberFormatException e) {
			throw new IOException("checkpoint: numero non valido");
		}
		return d;
	}

	private static State loadCheckpoint(File f, GaGenome.Layout layout, String key) throws IOException {
		final List<String> lines = new ArrayList<>();
		try (BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"))) {
			String l;
			while ((l = r.readLine()) != null) {
				lines.add(l);
			}
		}
		if (lines.size() < 7 || !lines.get(0).equals("TABLUT_GA_CKPT v1")) {
			throw new IOException("checkpoint " + f + ": non e' un checkpoint TABLUT_GA_CKPT v1");
		}
		final String last = lines.get(lines.size() - 1);
		final StringBuilder body = new StringBuilder();
		for (int i = 0; i < lines.size() - 1; i++) {
			body.append(lines.get(i)).append('\n');
		}
		final CRC32 crc = new CRC32();
		final byte[] bb = body.toString().getBytes("UTF-8");
		crc.update(bb, 0, bb.length);
		if (!last.equals("checksum " + String.format("%08x", crc.getValue()))) {
			throw new IOException("checkpoint " + f + ": checksum errato (file corrotto o troncato)");
		}
		int li = 1;
		if (!lines.get(li++).equals("key " + key)) {
			throw new IOException("checkpoint " + f + ": prodotto con parametri diversi da quelli attuali (si ripartirebbe da un'altra GA)");
		}
		final State st = new State();
		try {
			st.nextGen = Integer.parseInt(lines.get(li++).substring("nextgen ".length()));
			final String[] pr = lines.get(li++).substring("progress ".length()).split(" ");
			st.bestSmooth = Double.parseDouble(pr[0]);
			st.sinceImprove = Integer.parseInt(pr[1]);
			final int np = Integer.parseInt(lines.get(li++).substring("pop ".length()));
			for (int i = 0; i < np; i++) {
				st.pop.add(new GaGenome(layout, parseCsv(lines.get(li++))));
			}
			final int nh = Integer.parseInt(lines.get(li++).substring("hof ".length()));
			for (int i = 0; i < nh; i++) {
				final String[] t = lines.get(li++).split(" ", 3);
				final Entry e = new Entry(new GaGenome(layout, parseCsv(t[2])));
				e.sumRaw = Double.parseDouble(t[0]);
				e.count = Integer.parseInt(t[1]);
				st.hof.put(e.genome.key(), e);
			}
		} catch (RuntimeException e) {
			throw new IOException("checkpoint " + f + ": formato non valido (" + e + ")");
		}
		return st;
	}

	// ====================================================================== riga di comando

	public static void main(String[] args) throws Exception {
		final Config cfg = new Config();
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				usage("argomento non valido (atteso chiave=valore): " + a);
				return;
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			try {
				switch (k) {
				case "base": cfg.base = v; break;
				case "out": cfg.out = v; break;
				case "groups": cfg.groups = v; break;
				case "init": cfg.init = v; break;
				case "pop": cfg.pop = Integer.parseInt(v); break;
				case "gens": cfg.gens = Integer.parseInt(v); break;
				case "pairs": cfg.pairs = Integer.parseInt(v); break;
				case "depth": cfg.depth = Integer.parseInt(v); break;
				case "open": cfg.openRandom = Integer.parseInt(v); break;
				case "maxplies": cfg.maxPlies = Integer.parseInt(v); break;
				case "elite": cfg.elite = Integer.parseInt(v); break;
				case "tourn": cfg.tourn = Integer.parseInt(v); break;
				case "immigrants": cfg.immigrants = Integer.parseInt(v); break;
				case "pc": cfg.pc = Double.parseDouble(v); break;
				case "alpha": cfg.alpha = Double.parseDouble(v); break;
				case "sigma": cfg.sigma = Double.parseDouble(v); break;
				case "sigmadecay": cfg.sigmaDecay = Double.parseDouble(v); break;
				case "sigmamin": cfg.sigmaMin = Double.parseDouble(v); break;
				case "pm": cfg.pm = Double.parseDouble(v); break;
				case "lambda": cfg.lambda = Double.parseDouble(v); break;
				case "patience": cfg.patience = Integer.parseInt(v); break;
				case "minevals": cfg.minEvals = Integer.parseInt(v); break;
				case "mindelta": cfg.minDelta = Double.parseDouble(v); break;
				case "budget": cfg.budgetSeconds = Long.parseLong(v); break;
				case "seed": cfg.seed = Long.parseLong(v); break;
				case "threads": cfg.threads = Integer.parseInt(v); break;
				case "resume": cfg.resume = v.equals("1") || v.equalsIgnoreCase("true"); break;
				case "valpairs": cfg.valPairs = Integer.parseInt(v); break;
				case "valdepth": cfg.valDepth = Integer.parseInt(v); break;
				case "minpairs": cfg.minPairs = Integer.parseInt(v); break;
				case "z": cfg.z = Double.parseDouble(v); break;
				case "margin": cfg.margin = Double.parseDouble(v); break;
				case "acceptni": cfg.acceptNonInferior = v.equals("1") || v.equalsIgnoreCase("true"); break;
				default:
					usage("opzione sconosciuta: " + k);
					return;
				}
			} catch (NumberFormatException e) {
				usage("valore non valido per " + k + ": " + v);
				return;
			}
		}
		try {
			cfg.validate();
		} catch (IllegalArgumentException e) {
			usage(e.getMessage());
			return;
		}
		final File outFile = new File(cfg.out).getAbsoluteFile();
		if (outFile.getParentFile() != null) {
			outFile.getParentFile().mkdirs();
		}
		final PrintStream file = new PrintStream(new FileOutputStream(cfg.out + ".log", cfg.resume), true, "UTF-8");
		final PrintStream tee = new PrintStream(new java.io.OutputStream() {
			@Override
			public void write(int b) {
				System.out.write(b);
				file.write(b);
			}

			@Override
			public void write(byte[] b, int off, int len) {
				System.out.write(b, off, len);
				file.write(b, off, len);
			}

			@Override
			public void flush() {
				System.out.flush();
				file.flush();
			}
		}, true, "UTF-8");
		try {
			final Result r = run(cfg, tee);
			tee.println(r.accepted ? "ESITO: ACCETTATO -> usa " + cfg.out + ".final.nnue (o il genoma " + cfg.out + ".genome con TablutGeneticClient)"
					: "ESITO: RIFIUTATO -> " + cfg.out + ".final.nnue e' il riferimento invariato");
		} catch (IOException e) {
			tee.println("ERRORE: " + e.getMessage());
			System.exit(1);
		} finally {
			file.close();
		}
	}

	private static void usage(String msg) {
		System.out.println(msg);
		System.out.println("Uso: GaTrainer [base=pesi_nnue/trained_v6.nnue] [out=ga/run] [groups=head,l1b[,ftb]] [init=G1.genome,G2.genome] [pop=24] [gens=30] [pairs=6] [depth=4]");
		System.out.println("       [open=6] [maxplies=200] [elite=2] [tourn=3] [immigrants=1] [pc=0.8] [alpha=0.3] [sigma=0.15] [sigmadecay=0.95] [sigmamin=0.02] [pm=auto]");
		System.out.println("       [lambda=0.02] [patience=8] [minevals=2] [mindelta=0.005] [budget=SECONDI] [seed=1] [threads=1] [resume=0|1]");
		System.out.println("       [valpairs=100] [valdepth=depth+1] [minpairs=30] [z=1.96] [margin=0.02] [acceptni=0|1]");
		System.exit(2);
	}
}
