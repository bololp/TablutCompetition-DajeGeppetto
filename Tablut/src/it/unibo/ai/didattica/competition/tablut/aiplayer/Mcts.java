package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Monte Carlo Tree Search (PUCT) per Ashton Tablut, alternativa a {@link Negamax} ({@link MoveSearcher}). Lavora su {@link FastState}
 * come {@code Negamax}, con le stesse regole, lo stesso evaluator e la stessa regola di ripetizione.
 *
 * <h3>L'algoritmo</h3>
 * Ogni simulazione scende dalla radice scegliendo a ogni nodo l'arco che massimizza {@code Q + cPuct * P * sqrt(N) / (1 + n)} (PUCT),
 * arriva a una foglia, ne ricava un valore e lo propaga fino alla radice. <b>Non ci sono rollout casuali</b>: a Tablut una partita
 * casuale ha quasi zero informazione (il Re scappa o viene catturato per caso); il valore di una foglia viene dall'{@link Evaluator}
 * (lo stesso di {@code Negamax}: {@code SimpleEvaluator}, MLP o CNN). Un nodo si espande valutando staticamente TUTTI i suoi figli:
 * quel valore e' il Q iniziale dell'arco ("first play urgency" informata) e, attraverso un softmax, il suo prior {@code P}. Il valore
 * restituito dalla foglia e' il migliore dei suoi figli (un minimax a 1 piano, come un'alpha-beta alla frontiera).
 *
 * <h3>Ottimizzazioni (ognuna e' un parametro di {@link MctsConfig}, e ha un test)</h3>
 * <ol>
 * <li><b>Solver a limiti.</b> Ogni arco/nodo ha un limite inferiore e uno superiore {@code {-1,0,+1}} del suo valore di teoria dei giochi
 * (perdita/pareggio/vittoria per chi muove). Le fughe e le catture del Re sono terminali esatti; i limiti si propagano alla radice con la
 * regola negamax (Cazenave-Saffidine, "score-bounded MCTS"). Un arco il cui limite superiore e' sotto il limite inferiore di un fratello e'
 * dimostrato inferiore e non viene piu' scelto; un nodo con limiti uguali e' risolto e non si esplora oltre; la ricerca si ferma quando la
 * radice e' risolta. Cosi' l'MCTS non gioca mai una mossa dimostrata perdente se ne esiste una migliore, cosa che l'MCTS "puro" non
 * garantisce affatto.</li>
 * <li><b>Pareggio per ripetizione esatto.</b> La storia reale della partita piu' il cammino dalla radice rileva "stessa posizione due volte
 * dall'ultima cattura = pareggio" (regola del server). In un albero (senza condivisione di nodi tra cammini) il valore di un nodo dipende
 * dal suo solo cammino, quindi il problema GHI di {@code Negamax} con la TT non esiste: i limiti dimostrati sono esatti.</li>
 * <li><b>Implicit minimax backups</b> ({@link MctsConfig#mmWeight}, Lanctot et al. 2014): ogni arco conserva anche il valore minimax delle
 * valutazioni statiche del suo sottoalbero; la stima usata nella selezione e' {@code (1-mm)*media + mm*minimax}. Evita che la media diluisca
 * il valore di una linea tattica forzata.</li>
 * <li><b>Espansione ritardata</b> ({@link MctsConfig#expandAfter}): un arco si espande alla sua {@code (expand+1)}-esima visita. Le visite
 * precedenti restituiscono il valore statico del figlio, che e' gia' noto: nessuna valutazione in piu' e nessun nodo in memoria per le mosse
 * visitate una sola volta (la maggior parte).</li>
 * <li><b>Parallelismo ad albero con virtual loss.</b> {@code threads} thread condividono lo stesso albero. Un nodo ha il proprio lock
 * ({@code synchronized}) che protegge la selezione e l'aggiornamento delle statistiche (sezioni di pochi decine di ns); la parte costosa,
 * l'espansione (valutare tutti i figli), avviene FUORI da ogni lock. Un thread che scende lungo un arco vi lascia una "perdita virtuale"
 * (conta come visita di valore -1) cosi' gli altri vanno altrove; un arco in espansione e' marcato con una sentinella e saltato.</li>
 * <li><b>Riuso del sottoalbero</b> ({@link MctsConfig#reuse}): dopo la mossa giocata e la risposta avversaria la ricerca successiva
 * riparte dal nodo corrispondente, con tutte le sue statistiche, invece che da zero. Il riuso e' ammesso SOLO se la storia di ripetizione
 * passata dal chiamante coincide con quella che il sottoalbero presuppone (altrimenti i suoi pareggi per ripetizione sarebbero sbagliati).</li>
 * <li><b>Cache di valutazioni</b> ({@link EvalCache}): la stessa posizione raggiunta con ordini di mosse diversi si valuta una volta sola.</li>
 * <li><b>Tetto di memoria</b> ({@link MctsConfig#maxNodes}): a differenza di una TT, un albero cresce senza limite; raggiunto il tetto la
 * ricerca si ferma (invece di andare in OutOfMemoryError e perdere la partita per timeout).</li>
 * </ol>
 *
 * <h3>Convenzioni</h3>
 * Ogni valore memorizzato in un nodo e' "dal punto di vista di chi muove in QUEL nodo", in [-1,+1] (convenzione negamax: salendo di un
 * livello si cambia segno). I valori euristici sono compressi in [-0.98, 0.98], cosi' le vittorie/sconfitte esatte (+-1) restano distinte.
 * Il Re catturato o fuggito e' sempre una vittoria di CHI HA APPENA MOSSO (solo chi muove puo' vincere con la propria mossa); chi non ha
 * mosse legali perde (come in {@code Negamax}).
 *
 * <h3>Thread-safety</h3>
 * Un'istanza serve una ricerca alla volta; durante la ricerca i suoi thread di lavoro condividono l'albero come descritto sopra. L'evaluator
 * e' condiviso (gli evaluator di questo progetto sono thread-safe).
 */
public class Mcts implements MoveSearcher {

	static final int MAX_THREADS = 64;
	/** Massimo valore assoluto di una valutazione euristica: lascia +-1 ai terminali esatti. */
	static final double HEURISTIC_LIMIT = 0.98;
	private static final long DEFAULT_SAFETY_MARGIN_MILLIS = 300;
	private static final long MIN_SEARCH_MILLIS = 10;
	/** Quanto attendere gli helper a ricerca finita: sono fermi entro una simulazione (al piu' qualche ms). */
	private static final long HELPER_JOIN_MILLIS = 2000;
	/** Profondita' massima (in semi-mosse) a cui cercare, nel sottoalbero della mossa giocata, la nuova radice per il riuso. */
	static final int REUSE_MAX_DEPTH = 4;
	/**
	 * Byte per nodo espanso, usati per il tetto automatico dei nodi. MISURATI con {@code experiments.MctsBench} (heap dopo gc, 60.000 nodi): 1670-1755 byte/nodo
	 * con Simple, MLP e CNN e con {@code expand} 0/1 e {@code mm} 0/0.5 (la struttura dei nodi non dipende dall'evaluator); 1800 lascia un po' di margine.
	 */
	public static final int BYTES_PER_NODE_ESTIMATE = 1800;

	// tipi di foglia raggiunta da una simulazione
	private static final int K_SOLVED = 0, K_EXACT = 1, K_STATIC = 2, K_DESCEND = 3, K_EXPAND = 4, K_ABORT = 5;

	// ------------------------------------------------------------------------------------------ nodo

	/**
	 * Un nodo espanso: tutte le mosse legali (nell'ordine di {@link FastState#generateMoves}) con, per ciascuna, prior, statistiche e limiti.
	 * Quello che e' scritto alla costruzione ({@code moves}, {@code prior}) non cambia piu'. Tutto il resto e' protetto da {@code synchronized(this)};
	 * {@code nodeLo/nodeHi/nodeIm} sono anche {@code volatile} perche' il GENITORE li rilegge dentro il proprio lock (mai due lock insieme).
	 */
	static final class Node {
		final int nMoves;
		final short[] moves;
		/** Prior quantizzato: probabilita' * 65535. */
		final char[] prior;
		final int[] visits;
		/** Media dei valori dell'arco (prospettiva di chi muove qui); per un arco mai visitato e' il suo valore statico iniziale; per un arco risolto e' il valore esatto. */
		final float[] q;
		/** Valore minimax delle valutazioni statiche del sottoalbero dell'arco (prospettiva di chi muove qui). */
		final float[] im;
		/** Thread che stanno attraversando l'arco (perdita virtuale). */
		final byte[] vloss;
		/** Limiti del valore dell'arco, per chi muove qui: {-1,0,+1}. */
		final byte[] lo;
		final byte[] hi;
		final Node[] child;
		int totalVisits;
		int totalVloss;
		volatile byte nodeLo = -1;
		volatile byte nodeHi = 1;
		volatile float nodeIm;

		Node(int k) {
			nMoves = k;
			moves = new short[k];
			prior = new char[k];
			visits = new int[k];
			q = new float[k];
			im = new float[k];
			vloss = new byte[k];
			lo = new byte[k];
			hi = new byte[k];
			child = new Node[k];
		}

		boolean solved() {
			return nodeLo == nodeHi;
		}

		/** Ricalcola nodeLo/nodeHi/nodeIm da zero sugli archi (sotto lock). */
		void recomputeSummary() {
			byte l = -1, h = -1;
			float m = -Float.MAX_VALUE;
			for (int i = 0; i < nMoves; i++) {
				if (lo[i] > l) {
					l = lo[i];
				}
				if (hi[i] > h) {
					h = hi[i];
				}
				if (im[i] > m) {
					m = im[i];
				}
			}
			if (nMoves == 0) {
				l = -1;
				h = -1;
				m = -1;
			}
			nodeLo = l;
			nodeHi = h;
			nodeIm = m;
		}
	}

	/** Sentinella: l'arco e' in espansione da parte di un thread. Mai confrontata per contenuto, solo per identita'. */
	static final Node EXPANDING = new Node(0);

	// ------------------------------------------------------------------------------------------ stato condiviso di una ricerca

	/** Contesto di UNA ricerca: nuovo a ogni {@code findBestMove}, cosi' un helper in ritardo non puo' mai agire su quella successiva. */
	static final class Control {
		final Node root;
		final FastState rootPos;
		final long[] rootHist;
		final int rootHistLen;
		final long deadlineNanos;
		final long maxSims;
		final long maxNodes;
		volatile boolean stop;
		volatile boolean memoryFull;
		volatile Throwable error;
		final AtomicLong simsStarted = new AtomicLong();
		final AtomicLong simsDone = new AtomicLong();

		Control(Node root, FastState rootPos, long[] rootHist, int rootHistLen, long deadlineNanos, long maxSims, long maxNodes) {
			this.root = root;
			this.rootPos = rootPos;
			this.rootHist = rootHist;
			this.rootHistLen = rootHistLen;
			this.deadlineNanos = deadlineNanos;
			this.maxSims = maxSims;
			this.maxNodes = maxNodes;
		}
	}

	/** Stato PRIVATO di un thread di ricerca: posizione di lavoro, percorso, storia di ripetizione, contatori. */
	final class Worker {
		final FastState st = new FastState();
		final int[] moveBuf = new int[FastState.MAX_MOVES];
		final UndoInfo expandUndo = new UndoInfo();
		UndoInfo[] undo;
		Node[] pNode;
		int[] pEdge;
		long[] hist = new long[64];
		int histLen;
		int histBase;
		long evals, cacheHits, sims, aborts;
		long depthSum;
		int maxDepthSeen;
		int tick;

		Worker(int pathCapacity) {
			undo = new UndoInfo[pathCapacity];
			pNode = new Node[pathCapacity];
			pEdge = new int[pathCapacity];
			for (int i = 0; i < pathCapacity; i++) {
				undo[i] = new UndoInfo();
			}
		}

		void ensurePath(int n) {
			if (n < pNode.length) {
				return;
			}
			final int cap = Math.max(n + 1, pNode.length * 2);
			final int old = pNode.length;
			pNode = Arrays.copyOf(pNode, cap);
			pEdge = Arrays.copyOf(pEdge, cap);
			undo = Arrays.copyOf(undo, cap);
			for (int i = old; i < cap; i++) {
				undo[i] = new UndoInfo();
			}
		}

		void ensureHist(int n) {
			if (n > hist.length) {
				hist = Arrays.copyOf(hist, Math.max(n, hist.length * 2));
			}
		}

		void resetCounters() {
			evals = 0;
			cacheHits = 0;
			sims = 0;
			aborts = 0;
			depthSum = 0;
			maxDepthSeen = 0;
			tick = 0;
		}
	}

	// ------------------------------------------------------------------------------------------ campi

	private final Evaluator evaluator;
	private final MctsConfig cfg;
	private final double scale;
	private final boolean tanhSquash;
	private final EvalCache cache;
	private final int initialPathCapacity;
	private int threads;
	private long safetyMarginMillis = DEFAULT_SAFETY_MARGIN_MILLIS;
	/** Tetto alle simulazioni di una ricerca (test: ricerca deterministica a numero di simulazioni fisso). */
	private long maxSims = Long.MAX_VALUE;

	private ExecutorService pool;
	private Worker[] workers;

	// albero riutilizzabile
	private Node root;
	private FastState rootPos;
	private long[] rootHist;
	private final AtomicLong nodeCount = new AtomicLong();

	// statistiche dell'ultima ricerca
	public long lastSims;
	public long lastEvals;
	public long lastCacheHits;
	/** Simulazioni annullate perche' nessun arco era selezionabile (tutti in espansione da altri thread): diagnostica del parallelismo. */
	public long lastAborts;
	public long lastNodes;
	/** Nodi del sottoalbero riusato all'inizio dell'ultima ricerca (0 = albero nuovo). */
	public long lastReusedNodes;
	/** Visite della radice riusata all'inizio dell'ultima ricerca. */
	public long lastReusedVisits;
	public int lastMaxDepth;
	public double lastAvgDepth;
	public int lastPvDepth;
	/** Valore della mossa scelta dal punto di vista del BIANCO, in [-1,+1] (+-1 solo se dimostrato). */
	public double lastRootValue;
	/** Limiti dimostrati del valore della radice per chi muove ({-1,0,+1}); (+1,+1) = vittoria forzata, (-1,-1) = sconfitta forzata. */
	public int lastRootLo = -1, lastRootHi = 1;
	public int lastThreadsUsed = 1;
	public boolean lastMemoryCapHit;
	public boolean lastReused;
	public long lastElapsedMillis;
	public int lastBestVisits;
	/** Eccezione trovata in un thread dell'ultima ricerca (la mossa e' comunque stata restituita), o null. */
	public Throwable lastError;
	private String lastPv = "";

	public Mcts(Evaluator evaluator) {
		this(evaluator, new MctsConfig(), defaultThreads());
	}

	public Mcts(Evaluator evaluator, MctsConfig config, int threads) {
		this(evaluator, config, threads, 64);
	}

	/** {@code pathCapacity}: capacita' iniziale degli array del percorso (package-private: i test la rimpiccioliscono per esercitare la crescita). */
	Mcts(Evaluator evaluator, MctsConfig config, int threads, int pathCapacity) {
		if (evaluator == null || config == null) {
			throw new IllegalArgumentException("evaluator/config nulli");
		}
		this.evaluator = evaluator;
		this.cfg = new MctsConfig(config);
		final boolean[] tanhOut = new boolean[1];
		final double auto = defaultValueScale(evaluator, tanhOut);
		this.scale = Double.isNaN(cfg.valueScale) ? auto : cfg.valueScale;
		this.tanhSquash = Double.isNaN(cfg.valueScale) ? tanhOut[0] : false;
		final int cacheLog2 = cfg.evalCacheLog2 < 0 ? autoCacheLog2(evaluator) : cfg.evalCacheLog2;
		this.cache = cacheLog2 > 0 ? new EvalCache(cacheLog2) : null;
		this.threads = Math.max(1, Math.min(MAX_THREADS, threads));
		this.initialPathCapacity = Math.max(2, pathCapacity);
	}

	// ------------------------------------------------------------------------------------------ configurazione

	/**
	 * Scala dei valori dell'evaluator e tipo di compressione in [-1,1]. Le reti escono gia' in (-1,1)*{@code outputScale} e saltano a +-1
	 * (nelle posizioni di partita il valore assoluto mediano e' 560-750 su 1000): basta dividere. {@link SimpleEvaluator} e' illimitato (|v|
	 * mediano 60, 99-esimo percentile 148 su posizioni di partita): {@code tanh(v/100)}. Evaluator sconosciuti (test): v/1000.
	 *
	 * @param tanhOut [0] viene impostato a vero se la compressione e' tanh
	 */
	public static double defaultValueScale(Evaluator e, boolean[] tanhOut) {
		if (tanhOut != null) {
			tanhOut[0] = false;
		}
		if (e instanceof NeuralEvaluator) {
			return ((NeuralEvaluator) e).outputScale;
		}
		if (e instanceof CnnEvaluator) {
			return ((CnnEvaluator) e).outputScale;
		}
		if (e instanceof SimpleEvaluator) {
			if (tanhOut != null) {
				tanhOut[0] = true;
			}
			return 100.0;
		}
		return 1000.0;
	}

	/**
	 * Cache di valutazioni automatica: 2^20 voci per la CNN (~11 us a valutazione, ~10% di hit: la cache fa risparmiare il 5-14% del tempo), nessuna per gli
	 * altri evaluator (con Simple costa il 17-31%, con l'MLP e' neutra: {@code experiments.MctsBench --only=cache}).
	 */
	public static int autoCacheLog2(Evaluator e) {
		return e instanceof CnnEvaluator ? 20 : 0;
	}

	/** Thread di ricerca di default: quelli di {@link Negamax#defaultThreads()} (4 su una macchina a 4 core). */
	public static int defaultThreads() {
		return Negamax.defaultThreads();
	}

	/** Tetto automatico ai nodi espansi: il 30% dell'heap diviso la stima dei byte per nodo (il resto serve a JVM, evaluator, cache e GC). */
	public static long autoMaxNodes(long maxHeapBytes) {
		return Math.max(10_000L, (long) (maxHeapBytes * 0.30 / BYTES_PER_NODE_ESTIMATE));
	}

	public MctsConfig config() {
		return new MctsConfig(cfg);
	}

	public int getThreads() {
		return threads;
	}

	/** Imposta i thread di ricerca (1 = ricerca sul solo thread chiamante, deterministica). Va chiamato a ricerca ferma. */
	public Mcts setThreads(int n) {
		final int v = Math.max(1, Math.min(MAX_THREADS, n));
		if (v != threads) {
			shutdown();
			threads = v;
		}
		return this;
	}

	public long getSafetyMarginMillis() {
		return safetyMarginMillis;
	}

	/** Margine di sicurezza (ms, almeno 0) tolto al budget; come in {@link Negamax#setSafetyMarginMillis}. Ritorna {@code this}. */
	public Mcts setSafetyMarginMillis(long ms) {
		this.safetyMarginMillis = Math.max(0, ms);
		return this;
	}

	public static long defaultSafetyMarginMillis() {
		return DEFAULT_SAFETY_MARGIN_MILLIS;
	}

	/** La ricerca si ferma dopo {@code n} simulazioni COMPLETATE (esattamente, con qualunque numero di thread, salvo scadenza, tetto di memoria o radice risolta). */
	public Mcts setMaxSims(long n) {
		this.maxSims = n <= 0 ? Long.MAX_VALUE : n;
		return this;
	}

	public double valueScale() {
		return scale;
	}

	public EvalCache evalCache() {
		return cache;
	}

	/** Elimina l'albero conservato (la prossima ricerca riparte da zero). */
	public void clearTree() {
		root = null;
		rootPos = null;
		rootHist = null;
		nodeCount.set(0);
	}

	@Override
	public void shutdown() {
		if (pool != null) {
			pool.shutdownNow();
		}
		pool = null;
	}

	@Override
	public String name() {
		return "mcts";
	}

	// ------------------------------------------------------------------------------------------ API di confine

	/** Forma di comodo per uno {@link State} del dominio: converte la radice in {@link FastState} e la mossa in {@link Action}. */
	public Action findBestMove(State rootState, long timeBudgetMillis, List<Long> historySinceLastCapture) {
		final int n = historySinceLastCapture.size();
		final long[] h = new long[n];
		for (int i = 0; i < n; i++) {
			h[i] = historySinceLastCapture.get(i);
		}
		final int mv = findBestMove(FastState.fromState(rootState), timeBudgetMillis, h, n);
		return mv == Move.NONE ? null : Move.toAction(mv, rootState.getTurn());
	}

	public Action findBestMove(State rootState, long timeBudgetMillis) {
		return findBestMove(rootState, timeBudgetMillis, Collections.<Long>emptyList());
	}

	/**
	 * La ricerca. {@code rootState} non viene modificata. La storia e' quella del contratto di {@link MoveSearcher}; se e' vuota, o non finisce con
	 * l'hash della radice, la radice viene aggiunta (cosi' un cammino che torna alla radice e' riconosciuto come ripetizione).
	 *
	 * @return la mossa ({@link Move}), oppure {@link Move#NONE} se la partita e' finita o chi deve muovere non ha mosse legali
	 */
	@Override
	public int findBestMove(FastState rootState, long timeBudgetMillis, long[] history, int historyLen) {
		final long t0 = System.nanoTime();
		resetStats();
		if (rootState.winner != FastState.WINNER_NONE) {
			clearTree();
			return Move.NONE;
		}
		final long[] h = normalizeHistory(rootState, history, historyLen);
		final Worker w0 = worker(0);
		w0.resetCounters();
		Node r = null;
		try {
			if (cfg.reuse) {
				r = findReusable(rootState, h);
			}
			lastReused = r != null;
			if (r != null) {
				nodeCount.set(countNodes(r));
				lastReusedNodes = nodeCount.get();
				lastReusedVisits = r.totalVisits;
			} else {
				nodeCount.set(0);
				w0.st.copyFrom(rootState);
				loadHistory(w0, h);
				r = expand(w0);
				nodeCount.set(1);
			}
		} catch (Throwable t) {
			lastError = t;
			clearTree();
			return firstLegalMove(rootState);
		}
		if (r.nMoves == 0) { // chi deve muovere non ha mosse: la ricerca non ha nulla da scegliere (come Negamax)
			clearTree();
			return Move.NONE;
		}
		// Il vecchio albero non serve piu': di lui resta, al piu', il sottoalbero riusato (raggiungibile da r). Va reso irraggiungibile ORA e non a fine
		// ricerca: altrimenti durante tutta la ricerca la memoria viva e' la SOMMA del vecchio e del nuovo albero (e il tetto ai nodi non la vede), il
		// garbage collector lavora il doppio e le sue pause, che attraversano la scadenza, fanno perdere per timeout (misurato: pause fino a 483 ms).
		root = null;
		rootPos = null;
		rootHist = null;
		for (int i = 0; workers != null && i < workers.length; i++) {
			if (workers[i] != null) {
				Arrays.fill(workers[i].pNode, null); // il percorso dell'ultima simulazione precedente terrebbe vivi nodi (e sottoalberi) del vecchio albero
			}
		}

		final long budget = Math.max(MIN_SEARCH_MILLIS, timeBudgetMillis - safetyMarginMillis);
		final long maxNodes = cfg.maxNodes > 0 ? cfg.maxNodes : autoMaxNodes(Runtime.getRuntime().maxMemory());
		final Control c = new Control(r, rootState.duplicate(), h, h.length, t0 + budget * 1_000_000L, maxSims, maxNodes);

		if (r.nMoves > 1 && !r.solved()) {
			runSearch(c);
		}

		final int best = chooseEdge(r);
		finishStats(c, r, best, w0);
		lastElapsedMillis = (System.nanoTime() - t0) / 1_000_000L;
		if (lastError != null) {
			clearTree(); // un thread e' caduto a meta': le statistiche dell'albero non sono piu' affidabili, non si riusano
		} else {
			root = r;
			rootPos = c.rootPos;
			rootHist = h;
		}
		return r.moves[best];
	}

	private void resetStats() {
		lastSims = 0;
		lastEvals = 0;
		lastCacheHits = 0;
		lastAborts = 0;
		lastNodes = 0;
		lastReusedNodes = 0;
		lastReusedVisits = 0;
		lastMaxDepth = 0;
		lastAvgDepth = 0;
		lastPvDepth = 0;
		lastRootValue = 0;
		lastRootLo = -1;
		lastRootHi = 1;
		lastThreadsUsed = 1;
		lastMemoryCapHit = false;
		lastReused = false;
		lastElapsedMillis = 0;
		lastBestVisits = 0;
		lastError = null;
		lastPv = "";
	}

	private int firstLegalMove(FastState s) {
		final FastState tmp = s.duplicate();
		final int[] buf = new int[FastState.MAX_MOVES];
		return tmp.generateMoves(buf) > 0 ? buf[0] : Move.NONE;
	}

	/** Storia con la radice come ultima voce (la aggiunge se manca). */
	private static long[] normalizeHistory(FastState rootState, long[] history, int historyLen) {
		final int n = Math.max(0, Math.min(historyLen, history.length));
		if (n > 0 && history[n - 1] == rootState.zobristHash) {
			return Arrays.copyOf(history, n);
		}
		final long[] h = Arrays.copyOf(history, n + 1);
		h[n] = rootState.zobristHash;
		return h;
	}

	private void loadHistory(Worker w, long[] h) {
		w.ensureHist(h.length + 8);
		System.arraycopy(h, 0, w.hist, 0, h.length);
		w.histLen = h.length;
		w.histBase = 0;
	}

	Worker worker(int i) {
		if (workers == null || workers.length < threads) {
			final Worker[] ws = new Worker[threads];
			if (workers != null) {
				System.arraycopy(workers, 0, ws, 0, workers.length);
			}
			workers = ws;
		}
		if (workers[i] == null) {
			workers[i] = new Worker(initialPathCapacity);
		}
		return workers[i];
	}

	// ------------------------------------------------------------------------------------------ ricerca

	private void runSearch(final Control c) {
		final int nThreads = threads;
		lastThreadsUsed = nThreads;
		final Worker[] ws = new Worker[nThreads];
		for (int i = 0; i < nThreads; i++) {
			ws[i] = worker(i);
			if (i > 0) {
				ws[i].resetCounters();
			}
			loadHistory(ws[i], c.rootHist);
		}
		final List<Future<?>> futures = new ArrayList<>();
		if (nThreads > 1) {
			if (pool == null) {
				pool = Executors.newFixedThreadPool(nThreads - 1, new ThreadFactory() {
					@Override
					public Thread newThread(Runnable r) {
						final Thread t = new Thread(r, "mcts-worker");
						t.setDaemon(true);
						return t;
					}
				});
			}
			for (int i = 1; i < nThreads; i++) {
				final Worker w = ws[i];
				futures.add(pool.submit(new Runnable() {
					@Override
					public void run() {
						runWorker(w, c);
					}
				}));
			}
		}
		runWorker(ws[0], c);
		c.stop = true;
		boolean stuck = false;
		for (Future<?> f : futures) {
			try {
				f.get(HELPER_JOIN_MILLIS, TimeUnit.MILLISECONDS);
			} catch (TimeoutException e) {
				stuck = true;
			} catch (Exception e) {
				if (c.error == null) {
					c.error = e;
				}
			}
		}
		if (stuck) { // un helper non si e' fermato: potrebbe ancora toccare l'albero. Si abbandona tutto (nuovo pool, albero scartato)
			if (c.error == null) {
				c.error = new IllegalStateException("helper MCTS non terminato entro " + HELPER_JOIN_MILLIS + " ms");
			}
			shutdown();
		}
		if (c.error != null) {
			lastError = c.error;
		}
		lastMemoryCapHit = c.memoryFull;
	}

	void runWorker(Worker w, Control c) {
		try {
			while (!c.stop) {
				if (c.root.nodeLo == c.root.nodeHi) { // radice risolta: la risposta non puo' cambiare
					c.stop = true;
					break;
				}
				if (c.simsStarted.incrementAndGet() > c.maxSims) {
					// Slot oltre il tetto: si restituisce SENZA fermare gli altri thread. Una simulazione in volo potrebbe ancora annullarsi (nessun arco
					// selezionabile) e dovrebbe poterla ripetere: fermare tutti qui ne perderebbe una (si completerebbero N-1 invece di N). Si attende che
					// le simulazioni in volo finiscano; il tetto e' cosi' esatto con qualunque numero di thread (salvo scadenza, memoria o radice risolta).
					c.simsStarted.decrementAndGet(); // slot non assegnato
					if (c.simsDone.get() >= c.maxSims || System.nanoTime() >= c.deadlineNanos) {
						c.stop = true;
						break;
					}
					Thread.yield();
					continue;
				}
				if (simulate(w, c)) {
					w.sims++;
					c.simsDone.incrementAndGet();
				} else { // nessun arco selezionabile (tutti in espansione da altri thread): si cede e si riprova
					c.simsStarted.decrementAndGet(); // simulazione annullata: lo slot si restituisce
					w.aborts++;
					Thread.yield();
				}
				if ((++w.tick & 3) == 0 && System.nanoTime() >= c.deadlineNanos) {
					c.stop = true;
				}
			}
		} catch (Throwable t) {
			if (c.error == null) {
				c.error = t;
			}
			c.stop = true;
		}
	}

	/**
	 * UNA simulazione: scende dalla radice, raggiunge una foglia, ne ricava il valore e lo propaga.
	 *
	 * @return false se non c'era alcun arco selezionabile (la perdita virtuale lasciata lungo il cammino e' stata rimossa)
	 */
	boolean simulate(Worker w, Control c) {
		final FastState st = w.st;
		st.copyFrom(c.rootPos);
		w.histLen = c.rootHistLen;
		w.histBase = 0;
		Node node = c.root;
		int d = 0;
		double leafV;
		Node leafNode;
		for (;;) {
			w.ensurePath(d + 1);
			int kind;
			int edge = -1;
			Node next = null;
			float sv = 0;
			synchronized (node) {
				if (node.nodeLo == node.nodeHi) {
					kind = K_SOLVED;
					sv = node.nodeLo;
				} else {
					edge = selectEdge(node);
					if (edge < 0) {
						kind = K_ABORT;
					} else {
						node.vloss[edge]++;
						node.totalVloss++;
						if (node.lo[edge] == node.hi[edge]) {
							kind = K_EXACT;
							sv = node.lo[edge];
						} else {
							final Node ch = node.child[edge];
							if (ch != null) {
								kind = K_DESCEND;
								next = ch;
							} else if (d + 1 >= cfg.maxDepth || node.visits[edge] < cfg.expandAfter || c.memoryFull) {
								kind = K_STATIC;
								sv = node.q[edge];
							} else {
								node.child[edge] = EXPANDING;
								kind = K_EXPAND;
							}
						}
					}
				}
			}
			if (kind == K_ABORT) {
				release(w, d);
				return false;
			}
			if (kind == K_SOLVED) {
				leafV = sv;
				leafNode = node;
				break;
			}
			w.pNode[d] = node;
			w.pEdge[d] = edge;
			d++;
			if (kind == K_EXACT || kind == K_STATIC) {
				leafV = -sv;
				leafNode = null;
				break;
			}
			// K_DESCEND o K_EXPAND: si esegue la mossa sulla posizione di lavoro
			final UndoInfo u = w.undo[d - 1];
			st.makeMove(node.moves[edge], u);
			if (u.wasCapture()) {
				w.histBase = w.histLen; // una cattura azzera la memoria delle posizioni
			}
			w.ensureHist(w.histLen + 2);
			w.hist[w.histLen++] = st.zobristHash;
			if (kind == K_DESCEND) {
				node = next;
				continue;
			}
			Node fresh;
			try {
				fresh = expand(w);
			} catch (Throwable t) {
				synchronized (node) {
					node.child[edge] = null;
				}
				throw t;
			}
			synchronized (node) {
				node.child[edge] = fresh;
			}
			if (nodeCount.incrementAndGet() >= c.maxNodes) {
				c.memoryFull = true;
				c.stop = true;
			}
			leafV = fresh.nodeIm;
			leafNode = fresh;
			break;
		}
		w.depthSum += d;
		if (d > w.maxDepthSeen) {
			w.maxDepthSeen = d;
		}
		backprop(w, d, leafV, leafNode);
		return true;
	}

	/** Toglie la perdita virtuale lasciata dai primi {@code d} livelli del cammino (simulazione annullata). */
	private void release(Worker w, int d) {
		for (int j = d - 1; j >= 0; j--) {
			final Node n = w.pNode[j];
			synchronized (n) {
				n.vloss[w.pEdge[j]]--;
				n.totalVloss--;
			}
		}
	}

	/**
	 * Propaga il valore della foglia (dal punto di vista di chi vi muove) lungo il cammino, dal basso verso la radice: toglie la perdita
	 * virtuale, aggiorna visite e media di ogni arco e, se la foglia e' un nodo appena espanso (o risolto), i limiti dimostrati e il valore
	 * minimax. Un livello alla volta, MAI due lock insieme.
	 */
	void backprop(Worker w, int d, double leafV, Node leafNode) {
		double v = leafV;
		Node childNode = leafNode;
		for (int j = d - 1; j >= 0; j--) {
			final Node n = w.pNode[j];
			final int e = w.pEdge[j];
			v = -v;
			boolean changed = false;
			synchronized (n) {
				n.vloss[e]--;
				n.totalVloss--;
				final int nv = ++n.visits[e];
				n.totalVisits++;
				if (n.lo[e] != n.hi[e]) {
					n.q[e] += (float) ((v - n.q[e]) / nv);
				}
				if (childNode != null) {
					changed = absorbChild(n, e, childNode);
				}
			}
			childNode = changed ? n : null; // se la sintesi di n non e' cambiata, i livelli sopra non hanno nulla di nuovo da sapere
		}
	}

	/**
	 * Ricopia nell'arco {@code e} di {@code n} la sintesi (limiti e minimax) del figlio {@code ch}, con il cambio di segno negamax, e aggiorna
	 * la sintesi di {@code n}. Si chiama con il lock di {@code n}; legge i campi {@code volatile} del figlio, quindi vede sempre almeno l'ultimo
	 * valore scritto prima che il chiamante prendesse il lock: l'ultimo thread che cambia un figlio lascia quindi il genitore coerente.
	 *
	 * @return vero se la sintesi di {@code n} e' cambiata
	 */
	private boolean absorbChild(Node n, int e, Node ch) {
		final byte cLo = ch.nodeLo, cHi = ch.nodeHi;
		final float cIm = ch.nodeIm;
		final byte oldHi = n.hi[e];
		final float oldIm = n.im[e];
		if (-cHi > n.lo[e]) {
			n.lo[e] = (byte) -cHi;
		}
		if (-cLo < n.hi[e]) {
			n.hi[e] = (byte) -cLo;
		}
		n.im[e] = -cIm;
		if (n.lo[e] == n.hi[e]) {
			n.q[e] = n.lo[e];
		}
		boolean changed = false;
		if (n.lo[e] > n.nodeLo) {
			n.nodeLo = n.lo[e];
			changed = true;
		}
		if (n.hi[e] < oldHi && oldHi == n.nodeHi) { // l'arco era (anche) il massimizzatore del limite superiore: si ricalcola
			byte m = -1;
			for (int i = 0; i < n.nMoves; i++) {
				if (n.hi[i] > m) {
					m = n.hi[i];
				}
			}
			if (m != n.nodeHi) {
				n.nodeHi = m;
				changed = true;
			}
		}
		if (n.im[e] > n.nodeIm) {
			n.nodeIm = n.im[e];
			changed = true;
		} else if (n.im[e] < oldIm && oldIm == n.nodeIm) {
			float m = -Float.MAX_VALUE;
			for (int i = 0; i < n.nMoves; i++) {
				if (n.im[i] > m) {
					m = n.im[i];
				}
			}
			if (m != n.nodeIm) {
				n.nodeIm = m;
				changed = true;
			}
		}
		return changed;
	}

	/**
	 * PUCT con perdita virtuale, minimax implicito e potatura dei dominati. Si chiama con il lock di {@code n}.
	 *
	 * @return l'indice dell'arco, oppure -1 se nessuno e' selezionabile
	 */
	int selectEdge(Node n) {
		final int k = n.nMoves;
		final double cp = cfg.cPuct * Math.sqrt(Math.max(1, n.totalVisits + n.totalVloss)) * (1.0 / 65535.0);
		final double a = cfg.mmWeight;
		final double vw = cfg.vlossWeight;
		final int lo = n.nodeLo;
		int best = -1;
		double bestScore = -Double.MAX_VALUE;
		for (int i = 0; i < k; i++) {
			if (n.hi[i] < lo || n.child[i] == EXPANDING) {
				continue; // dimostrato inferiore a un fratello, oppure in espansione da un altro thread
			}
			final int nv = n.visits[i];
			final int vl = n.vloss[i];
			double qv = n.q[i];
			if (vl > 0 && n.lo[i] != n.hi[i]) {
				final double w = vw * vl;
				qv = (qv * nv - w) / (nv + w);
			}
			if (a > 0) {
				qv = (1 - a) * qv + a * n.im[i];
			}
			final double score = qv + cp * n.prior[i] / (1 + nv + vw * vl);
			if (score > bestScore) {
				bestScore = score;
				best = i;
			}
		}
		return best;
	}

	// ------------------------------------------------------------------------------------------ espansione

	/**
	 * Espande la posizione di lavoro di {@code w} (con la sua storia di ripetizione gia' aggiornata): genera le mosse e valuta ogni figlio.
	 * Chiamata FUORI da ogni lock. Un figlio terminale e' esatto (+1: chi muove vince); un figlio che ripete una posizione della storia e'
	 * un pareggio esatto (0); altrimenti vale la sua valutazione statica. Se non ci sono mosse, chi muove ha perso.
	 */
	private Node expand(Worker w) {
		final FastState st = w.st;
		final int k = st.generateMoves(w.moveBuf);
		if (k == 0) {
			final Node n = new Node(0);
			n.nodeLo = -1;
			n.nodeHi = -1;
			n.nodeIm = -1;
			return n;
		}
		final Node n = new Node(k);
		final boolean whiteToMove = st.turn == BoardConstants.WHITE_TURN;
		final UndoInfo u = w.expandUndo;
		float qmax = -Float.MAX_VALUE;
		for (int i = 0; i < k; i++) {
			final int m = w.moveBuf[i];
			n.moves[i] = (short) m;
			st.makeMove(m, u);
			float v;
			byte l = -1, h = 1;
			if (st.winner != FastState.WINNER_NONE) {
				if (st.winner == FastState.WINNER_DRAW) {
					v = 0;
					l = 0;
					h = 0;
				} else {
					final boolean moverWins = (st.winner == FastState.WINNER_WHITE) == whiteToMove;
					v = moverWins ? 1f : -1f;
					l = (byte) (moverWins ? 1 : -1);
					h = l;
				}
			} else if (!u.wasCapture() && historyContains(w, st.zobristHash)) {
				v = 0;
				l = 0;
				h = 0;
			} else if (cfg.escapeScan && kingCanEscapeNow(st)) {
				v = -1f; // il Bianco e' di turno e il Re ha una via libera al bordo: chi ha mosso (il Nero) ha perso
				l = -1;
				h = -1;
			} else {
				final float white = staticValueWhite(w, st);
				v = whiteToMove ? white : -white;
			}
			st.unmakeMove(m, u);
			n.q[i] = v;
			n.im[i] = v;
			n.lo[i] = l;
			n.hi[i] = h;
			if (v > qmax) {
				qmax = v;
			}
		}
		final double invTau = 1.0 / cfg.tau;
		double norm = 0;
		final double[] p = new double[k];
		for (int i = 0; i < k; i++) {
			p[i] = Math.exp((n.q[i] - qmax) * invTau);
			norm += p[i];
		}
		for (int i = 0; i < k; i++) {
			n.prior[i] = (char) Math.min(65535, Math.round(p[i] / norm * 65535.0));
		}
		n.recomputeSummary();
		return n;
	}

	/**
	 * Vero se in {@code st} il Bianco e' di turno e il Re puo' raggiungere il bordo con UNA mossa (cioe' il Bianco vince al prossimo tratto,
	 * qualunque cosa sia successa prima). Copia le regole di movimento del Re di {@link FastState#generateMoves}: si ferma davanti a una cella
	 * occupata o al trono, entra in una citadel solo partendo da una citadel e entro 5 passi; ogni cella di bordo raggiunta e' una fuga.
	 * Costa ~4 scansioni di raggio, senza generare ne' eseguire mosse: e' ESATTA ({@code MctsSearchCheck} la confronta con la forza bruta su
	 * decine di migliaia di posizioni) e permette all'MCTS di vedere una doppia via di fuga un piano prima di espandere il nodo.
	 */
	static boolean kingCanEscapeNow(FastState st) {
		if (st.turn != BoardConstants.WHITE_TURN || st.winner != FastState.WINNER_NONE) {
			return false;
		}
		final int ks = st.kingSquare;
		if (ks < 0) {
			return false;
		}
		final byte[] b = st.board;
		final boolean originCitadel = BoardConstants.IS_CITADEL[ks];
		for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
			final int steps = BoardConstants.STEPS[d][ks];
			final int off = BoardConstants.DIR_OFFSET[d];
			int to = ks;
			for (int step = 1; step <= steps; step++) {
				to += off;
				if (b[to] != BoardConstants.EMPTY || to == BoardConstants.THRONE) {
					break;
				}
				if (BoardConstants.IS_CITADEL[to] && (!originCitadel || step > 5)) {
					break;
				}
				if (BoardConstants.IS_EDGE[to]) {
					return true;
				}
			}
		}
		return false;
	}

	private boolean historyContains(Worker w, long key) {
		final long[] h = w.hist;
		for (int i = w.histBase; i < w.histLen; i++) {
			if (h[i] == key) {
				return true;
			}
		}
		return false;
	}

	/** Valutazione statica compressa in [-0.98, 0.98], dal punto di vista del BIANCO; con cache. */
	private float staticValueWhite(Worker w, FastState st) {
		final long hash = st.zobristHash;
		if (cache != null) {
			final float f = cache.get(hash);
			if (!Float.isNaN(f)) {
				w.cacheHits++;
				return f;
			}
		}
		final float v = squash(evaluator.evaluate(st));
		w.evals++;
		if (cache != null) {
			cache.put(hash, v);
		}
		return v;
	}

	float squash(double e) {
		double v = tanhSquash ? Math.tanh(e / scale) : e / scale;
		if (Double.isNaN(v)) {
			return 0f;
		}
		if (v > HEURISTIC_LIMIT) {
			v = HEURISTIC_LIMIT;
		} else if (v < -HEURISTIC_LIMIT) {
			v = -HEURISTIC_LIMIT;
		}
		return (float) v;
	}

	// ------------------------------------------------------------------------------------------ scelta della mossa

	/**
	 * La mossa da giocare alla radice: tra le vittorie dimostrate la PIU' BREVE (a parita' la piu' visitata); altrimenti, escluse le mosse
	 * dimostrate perdenti (se ne resta almeno una), la piu' visitata (a parita' la media migliore, poi il minimax migliore, poi la prima
	 * nell'ordine di generazione).
	 */
	int chooseEdge(Node r) {
		final int k = r.nMoves;
		int best = -1;
		int bestDist = Integer.MAX_VALUE;
		for (int i = 0; i < k; i++) {
			if (r.lo[i] == 1) {
				final int d = edgeWinDist(r, i);
				if (best < 0 || d < bestDist || (d == bestDist && r.visits[i] > r.visits[best])) {
					best = i;
					bestDist = d;
				}
			}
		}
		if (best >= 0) {
			return best;
		}
		boolean anyNotLost = false;
		for (int i = 0; i < k; i++) {
			if (r.hi[i] > -1) {
				anyNotLost = true;
				break;
			}
		}
		for (int i = 0; i < k; i++) {
			if (anyNotLost && r.hi[i] == -1) {
				continue;
			}
			if (best < 0 || better(r, i, best)) {
				best = i;
			}
		}
		return best;
	}

	/** Semi-mosse fino alla fine dell'arco {@code e} di {@code n}, dimostrato vincente per chi muove in {@code n}, con l'avversario che resiste il piu' a lungo possibile. */
	static int edgeWinDist(Node n, int e) {
		final Node ch = n.child[e];
		return ch == null ? 1 : 1 + lossDist(ch);
	}

	/** Per un nodo dimostrato PERSO per chi vi muove: semi-mosse alla fine con la resistenza piu' lunga (0 se non ha mosse). */
	private static int lossDist(Node y) {
		if (y.nMoves == 0) {
			return 0;
		}
		int worst = 0;
		for (int f = 0; f < y.nMoves; f++) {
			final Node c = y.child[f];
			// senza figlio, un arco perso e' una mossa dopo la quale il Re puo' fuggire: ancora una mossa e finisce
			final int d = c == null ? 2 : 1 + winDist(c);
			if (d > worst) {
				worst = d;
			}
		}
		return worst;
	}

	/** Per un nodo dimostrato VINTO per chi vi muove: semi-mosse alla vittoria piu' breve. */
	private static int winDist(Node x) {
		int best = Integer.MAX_VALUE;
		for (int e = 0; e < x.nMoves; e++) {
			if (x.lo[e] == 1) {
				best = Math.min(best, edgeWinDist(x, e));
			}
		}
		return best;
	}

	private static boolean better(Node r, int i, int j) {
		if (r.visits[i] != r.visits[j]) {
			return r.visits[i] > r.visits[j];
		}
		if (r.q[i] != r.q[j]) {
			return r.q[i] > r.q[j];
		}
		return r.im[i] > r.im[j];
	}

	private void finishStats(Control c, Node r, int best, Worker w0) {
		long evals = 0, hits = 0, sims = 0, depthSum = 0, aborts = 0;
		int maxD = 0;
		for (int i = 0; i < lastThreadsUsed && workers != null && i < workers.length; i++) {
			final Worker w = workers[i];
			if (w == null) {
				continue;
			}
			evals += w.evals;
			hits += w.cacheHits;
			sims += w.sims;
			aborts += w.aborts;
			depthSum += w.depthSum;
			maxD = Math.max(maxD, w.maxDepthSeen);
		}
		lastEvals = evals;
		lastCacheHits = hits;
		lastAborts = aborts;
		lastSims = sims;
		lastNodes = nodeCount.get();
		lastMaxDepth = maxD;
		lastAvgDepth = sims > 0 ? (double) depthSum / sims : 0;
		lastRootLo = r.nodeLo;
		lastRootHi = r.nodeHi;
		lastBestVisits = r.visits[best];
		final boolean whiteToMove = c.rootPos.turn == BoardConstants.WHITE_TURN;
		double qv = r.q[best];
		if (r.lo[best] == r.hi[best]) {
			qv = r.lo[best];
		}
		lastRootValue = whiteToMove ? qv : -qv;
		// variante principale: si segue l'arco piu' visitato finche' esiste il figlio
		final FastState pos = c.rootPos.duplicate();
		final UndoInfo u = new UndoInfo();
		final StringBuilder pv = new StringBuilder();
		Node n = r;
		int depth = 0;
		int e = best;
		while (n != null && n != EXPANDING && n.nMoves > 0 && depth < 40) {
			if (depth > 0) {
				e = chooseEdge(n);
			}
			final int mv = n.moves[e];
			if (depth < 6) {
				pv.append(depth > 0 ? " " : "").append(Move.toString(mv));
			}
			pos.makeMove(mv, u);
			depth++;
			n = n.child[e];
		}
		lastPvDepth = depth;
		lastPv = pv.toString();
	}

	@Override
	public String describeLastSearch() {
		final long total = lastEvals + lastCacheHits;
		return "simulazioni=" + lastSims + " valutazioni=" + lastEvals + " nodi=" + lastNodes + " profondita'(media/max/pv)="
				+ String.format("%.1f", lastAvgDepth) + "/" + lastMaxDepth + "/" + lastPvDepth + " valore="
				+ String.format("%+.3f", lastRootValue) + (lastRootLo == lastRootHi ? " (DIMOSTRATO " + (lastRootLo > 0 ? "vittoria" : lastRootLo < 0 ? "sconfitta" : "pareggio") + ")" : "")
				+ " visite=" + lastBestVisits + " cache=" + (total > 0 ? Math.round(100.0 * lastCacheHits / total) : 0) + "%"
				+ (lastReused ? " riuso=" + lastReusedVisits + " visite/" + lastReusedNodes + " nodi" : "")
				+ (lastMemoryCapHit ? " TETTO-MEMORIA" : "") + " thread=" + lastThreadsUsed + " pv=[" + lastPv + "]"
				+ (lastError != null ? " ERRORE=" + lastError : "");
	}

	@Override
	public long lastWork() {
		return lastEvals;
	}

	@Override
	public double lastDepth() {
		return lastPvDepth;
	}

	// ------------------------------------------------------------------------------------------ riuso del sottoalbero

	/**
	 * Cerca, nell'albero della ricerca precedente, il nodo che corrisponde alla nuova radice: la stessa radice (ricerca ripetuta) oppure un
	 * discendente fino a {@link #REUSE_MAX_DEPTH} semi-mosse (la nostra mossa e la risposta avversaria). Il nodo e' riusabile solo se
	 * coincidono sia la posizione sia la STORIA di ripetizione: i pareggi per ripetizione dei suoi archi sono stati calcolati con la storia
	 * del cammino che ha portato a quel nodo, che deve essere esattamente quella che ora ci viene passata.
	 */
	private Node findReusable(FastState newPos, long[] newHist) {
		if (root == null || rootPos == null || rootHist == null) {
			return null;
		}
		final FastState pos = rootPos.duplicate();
		if (pos.samePosition(newPos) && Arrays.equals(rootHist, newHist)) {
			return root;
		}
		return dfsReuse(root, pos, rootHist, 0, newPos, newHist, new UndoInfo());
	}

	private Node dfsReuse(Node n, FastState pos, long[] hist, int depth, FastState target, long[] targetHist, UndoInfo u) {
		if (depth >= REUSE_MAX_DEPTH) {
			return null;
		}
		for (int e = 0; e < n.nMoves; e++) {
			final Node ch = n.child[e];
			if (ch == null || ch == EXPANDING) {
				continue;
			}
			pos.makeMove(n.moves[e], u);
			final boolean capture = u.wasCapture();
			final long[] h2;
			if (capture) {
				h2 = new long[] { pos.zobristHash };
			} else {
				h2 = Arrays.copyOf(hist, hist.length + 1);
				h2[hist.length] = pos.zobristHash;
			}
			Node found = null;
			if (pos.samePosition(target) && Arrays.equals(h2, targetHist)) {
				found = ch;
			} else {
				found = dfsReuse(ch, pos, h2, depth + 1, target, targetHist, new UndoInfo());
			}
			pos.unmakeMove(n.moves[e], u);
			if (found != null) {
				return found;
			}
		}
		return null;
	}

	/** Numero di nodi espansi del sottoalbero (iterativo: la profondita' puo' essere di centinaia di livelli). */
	static long countNodes(Node r) {
		long count = 0;
		final ArrayDeque<Node> stack = new ArrayDeque<>();
		stack.push(r);
		while (!stack.isEmpty()) {
			final Node n = stack.pop();
			count++;
			for (int i = 0; i < n.nMoves; i++) {
				final Node ch = n.child[i];
				if (ch != null && ch != EXPANDING) {
					stack.push(ch);
				}
			}
		}
		return count;
	}

	// ------------------------------------------------------------------------------------------ accesso per i test

	Node rootNode() {
		return root;
	}

	FastState rootPosition() {
		return rootPos;
	}

	long[] rootHistory() {
		return rootHist;
	}

	long nodeCountForTest() {
		return nodeCount.get();
	}

	String lastPv() {
		return lastPv;
	}
}
