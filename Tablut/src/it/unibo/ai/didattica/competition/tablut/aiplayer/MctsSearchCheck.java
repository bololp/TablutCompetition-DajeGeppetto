package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import it.unibo.ai.didattica.competition.tablut.aiplayer.MctsTestSupport.Walk;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

import static it.unibo.ai.didattica.competition.tablut.aiplayer.MctsTestSupport.*;

/**
 * Test dell'{@link Mcts}: correttezza della ricerca (white-box, stesso package). Non c'e' un "risultato esatto" da confrontare, come per una
 * alpha-beta: si verifica quindi per vie indipendenti.
 * <ol>
 * <li>{@link MctsTestSupport.Walk}: dopo ogni ricerca ripercorre l'albero rigiocando le mosse e ricalcola mosse, terminali, ripetizioni, limiti
 * dimostrati, minimax implicito e le MEDIE di ogni arco dalla struttura dell'albero.</li>
 * <li>Oracoli di forza bruta per la tattica (vittoria in 1 e in 3 semi-mosse, sconfitta forzata, mosse sicure).</li>
 * <li>Confronti differenziali deterministici (un thread, simulazioni fisse): cache si/no, capacita' minuscole degli array, storia con o senza radice.</li>
 * <li>Scenari costruiti: ripetizione (ciclo di 4 semi-mosse), stallo.</li>
 * </ol>
 * La concorrenza, il riuso, i limiti e le partite complete sono in {@link MctsConcurrencyCheck}.
 * Uso: {@code MctsSearchCheck [seed]}.
 */
public class MctsSearchCheck {

	static int failures = 0;
	static int checks = 0;
	static long seed = 2024;
	static final Evaluator SIMPLE = new SimpleEvaluator();

	/** Con {@code -Dmcts.failfast=true} il primo fallimento termina subito il test (serve al mutation test, che cerca solo il primo rilevamento). */
	static final boolean FAILFAST = Boolean.getBoolean("mcts.failfast");

	static void check(String name, boolean ok) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("  FALLITO: " + name);
			if (FAILFAST) {
				System.out.println("=== MctsSearchCheck: FALLITO (failfast) ===");
				System.exit(1);
			}
		}
	}

	static void section(String s) {
		System.out.println("--- " + s + " ---");
	}

	/** Valutazione "rumore": dipende dalla posizione ma non ha alcun significato di gioco. La ricerca esatta (solver) non deve dipenderne. */
	static final Evaluator NOISE = new Evaluator() {
		@Override
		public double evaluate(FastState s) {
			long h = s.zobristHash * 0x9E3779B97F4A7C15L;
			h ^= h >>> 31;
			return ((h >>> 11) % 1401) - 700;
		}
	};

	static Evaluator constant(final double v) {
		return new Evaluator() {
			@Override
			public double evaluate(FastState s) {
				return v;
			}
		};
	}

	static Mcts mk(Evaluator ev, String cfg, int threads) {
		return new Mcts(ev, MctsConfig.parse(cfg), threads).setSafetyMarginMillis(0);
	}

	/** Ricerca deterministica a numero di simulazioni fisso (1 thread, radice con la sola storia di se stessa). */
	static int search(Mcts m, FastState s, long sims) {
		m.setMaxSims(sims);
		return m.findBestMove(s, 600000, rootOnly(s), 1);
	}

	public static void main(String[] args) throws Exception {
		if (args.length > 0) {
			seed = Long.parseLong(args[0]);
		}
		final long t0 = System.nanoTime();
		testEvalCache();
		testConfig();
		testScaleAndSquash();
		testEdgeCases();
		testSelectEdge();
		testBackpropByHand();
		testAbortReleasesVirtualLoss();
		testChooseEdgeByHand();
		testSlotAccounting();
		testStructure();
		testDeterminism();
		testOneSimulationExact();
		testEscapeScan();
		testTactics();
		testRepetition();
		testStalemateSolved();
		testNegamaxSearcher();
		System.out.println("verifiche: " + checks + ", tempo " + (System.nanoTime() - t0) / 1_000_000_000L + " s");
		if (failures == 0) {
			System.out.println("=== MctsSearchCheck: TUTTI I TEST OK ===");
		} else {
			System.out.println("=== MctsSearchCheck: " + failures + " VERIFICHE FALLITE ===");
			System.exit(1);
		}
	}

	// ------------------------------------------------------------------------------------------ EvalCache

	static void testEvalCache() throws Exception {
		section("EvalCache");
		boolean threw = false;
		try {
			new EvalCache(3);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check("log2 troppo piccolo rifiutato", threw);
		threw = false;
		try {
			new EvalCache(27);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check("log2 troppo grande rifiutato", threw);

		final EvalCache empty = new EvalCache(8);
		check("cache vuota: miss anche per l'hash 0", Float.isNaN(empty.get(0L)) && Float.isNaN(empty.get(12345L)) && Float.isNaN(empty.get(-1L)));

		final Random rnd = new Random(seed);
		final EvalCache big = new EvalCache(16);
		final long[] keys = new long[500];
		final float[] vals = new float[500];
		for (int i = 0; i < keys.length; i++) {
			keys[i] = rnd.nextLong();
			vals[i] = (float) (rnd.nextDouble() * 1.96 - 0.98);
			big.put(keys[i], vals[i]);
		}
		int exact = 0;
		for (int i = 0; i < keys.length; i++) {
			final float g = big.get(keys[i]);
			if (g == vals[i]) {
				exact++;
			} else if (!Float.isNaN(g)) {
				check("un hit restituisce un valore diverso da quello salvato", false);
			}
		}
		check("500 chiavi in 65536 voci: quasi tutte presenti (" + exact + ")", exact >= 495);
		big.put(7L, -0.0f);
		check("valore -0.0 salvato e riletto al bit", Float.floatToRawIntBits(big.get(7L)) == Float.floatToRawIntBits(-0.0f));
		big.put(0L, 0.98f);
		check("hash 0 salvabile", big.get(0L) == 0.98f);

		// modello: in una cache minuscola le voci si sovrascrivono; un hit deve sempre dare l'ULTIMO valore scritto per quella chiave
		final EvalCache tiny = new EvalCache(4);
		final long[] ks = new long[64];
		for (int i = 0; i < ks.length; i++) {
			ks[i] = rnd.nextLong();
		}
		final float[] last = new float[ks.length];
		Arrays.fill(last, Float.NaN);
		int hits = 0, misses = 0, wrong = 0;
		for (int op = 0; op < 40000; op++) {
			final int k = rnd.nextInt(ks.length);
			if (rnd.nextBoolean()) {
				final float v = (float) (rnd.nextDouble() - 0.5);
				tiny.put(ks[k], v);
				last[k] = v;
			} else {
				final float g = tiny.get(ks[k]);
				if (Float.isNaN(g)) {
					misses++;
				} else {
					hits++;
					if (g != last[k]) {
						wrong++;
					}
				}
			}
		}
		check("modello: hit sempre uguale all'ultima scrittura (" + hits + " hit, " + misses + " miss, " + wrong + " errati)", wrong == 0 && hits > 100 && misses > 100);
		check("occupazione nei limiti", tiny.occupancy() >= 1 && tiny.occupancy() <= tiny.capacity());

		// 4 thread che scrivono e leggono le stesse voci: il valore e' funzione della chiave, quindi ogni hit deve coincidere
		final EvalCache shared = new EvalCache(4);
		final long[] sk = new long[40];
		for (int i = 0; i < sk.length; i++) {
			sk[i] = rnd.nextLong();
		}
		final AtomicInteger bad = new AtomicInteger();
		final AtomicInteger good = new AtomicInteger();
		final Thread[] ts = new Thread[4];
		for (int t = 0; t < ts.length; t++) {
			final int tt = t;
			ts[t] = new Thread() {
				@Override
				public void run() {
					final Random r = new Random(77 + tt);
					for (int op = 0; op < 400000; op++) {
						final long k = sk[r.nextInt(sk.length)];
						final float expected = (float) (((k >>> 40) % 1000) / 1000.0 - 0.5);
						if (r.nextBoolean()) {
							shared.put(k, expected);
						} else {
							final float g = shared.get(k);
							if (!Float.isNaN(g)) {
								if (g == expected) {
									good.incrementAndGet();
								} else {
									bad.incrementAndGet();
								}
							}
						}
					}
				}
			};
			ts[t].start();
		}
		for (Thread t : ts) {
			t.join();
		}
		check("stress a 4 thread su 16 voci: " + good.get() + " hit coerenti, " + bad.get() + " incoerenti", bad.get() == 0 && good.get() > 1000);
	}

	// ------------------------------------------------------------------------------------------ MctsConfig

	static boolean rejects(String spec) {
		try {
			MctsConfig.parse(spec);
			return false;
		} catch (IllegalArgumentException e) {
			return true;
		}
	}

	static void testConfig() {
		section("MctsConfig");
		final MctsConfig d = new MctsConfig();
		check("parse(\"\") e parse(null) = default", MctsConfig.parse("").toString().equals(d.toString()) && MctsConfig.parse(null).toString().equals(d.toString()));
		final MctsConfig c = MctsConfig.parse("cpuct=2.5,tau=0.1,mm=0.5,expand=2,vloss=2,reuse=false,cache=0,maxnodes=1000,scale=500,maxdepth=100");
		check("tutti i campi letti", c.cPuct == 2.5 && c.tau == 0.1 && c.mmWeight == 0.5 && c.expandAfter == 2 && c.vlossWeight == 2 && !c.reuse
				&& c.evalCacheLog2 == 0 && c.maxNodes == 1000 && c.valueScale == 500 && c.maxDepth == 100);
		final MctsConfig c2 = MctsConfig.parse(" CPUCT = 1.5 , Reuse = 1 ,scale=auto");
		check("spazi e maiuscole tollerati, 1 = true, scale=auto", c2.cPuct == 1.5 && c2.reuse && Double.isNaN(c2.valueScale));
		check("toString e' rileggibile", MctsConfig.parse(c.toString()).toString().equals(c.toString()));
		check("la base non viene modificata", MctsConfig.parse("cpuct=3", d).cPuct == 3 && d.cPuct == 1.0);
		final String[] bad = { "foo=1", "cpuct", "cpuct=", "=3", "cpuct=abc", "cpuct=-1", "tau=0", "mm=1.5", "mm=-0.1", "expand=-1", "cache=3", "cache=27",
				"maxdepth=1", "reuse=maybe", "scale=0", "cpuct=NaN", "vloss=1000" };
		for (String b : bad) {
			check("rifiutata: '" + b + "'", rejects(b));
		}
		check("cache=0 e cache=4..26 accettate", !rejects("cache=0") && !rejects("cache=4") && !rejects("cache=26"));
		check("cache: il default e' auto (-1), 'auto' e' accettato e rileggibile", d.evalCacheLog2 == -1 && MctsConfig.parse("cache=AUTO").evalCacheLog2 == -1 && d.toString().contains("cache=auto")
				&& MctsConfig.parse(d.toString()).evalCacheLog2 == -1 && MctsConfig.parse("cache=12").toString().contains("cache=12"));
		check("cache: valori non validi rifiutati (cache=3, cache=-1, cache=x)", rejects("cache=3") && rejects("cache=-1") && rejects("cache=x"));
	}

	// ------------------------------------------------------------------------------------------ scala e compressione

	static void testScaleAndSquash() {
		section("scala dei valori");
		final Mcts s = mk(SIMPLE, "", 1);
		check("Simple: scala 100", s.valueScale() == 100.0);
		check("Simple: tanh(v/100)", s.squash(60) == (float) Math.tanh(0.6) && s.squash(-60) == -(float) Math.tanh(0.6) && s.squash(0) == 0f);
		check("Simple: compresso entro il limite euristico", s.squash(1e9) == (float) Mcts.HEURISTIC_LIMIT && s.squash(-1e9) == -(float) Mcts.HEURISTIC_LIMIT);
		check("NaN -> 0", s.squash(Double.NaN) == 0f);
		boolean mono = true;
		float prev = -2;
		for (double v = -500; v <= 500; v += 7) {
			final float q = s.squash(v);
			if (q < prev) {
				mono = false;
			}
			prev = q;
		}
		check("compressione monotona", mono);
		final Mcts k = mk(constant(0), "", 1);
		check("evaluator generico: lineare su 1000", k.valueScale() == 1000.0 && k.squash(500) == 0.5f && k.squash(5000) == (float) Mcts.HEURISTIC_LIMIT);
		check("cache automatica: spenta con Simple e con un evaluator generico, accesa (2^20) solo per la CNN", Mcts.autoCacheLog2(SIMPLE) == 0 && Mcts.autoCacheLog2(constant(0)) == 0
				&& mk(SIMPLE, "", 1).evalCache() == null && mk(SIMPLE, "cache=12", 1).evalCache() != null && mk(SIMPLE, "cache=12", 1).evalCache().capacity() == 4096
				&& mk(SIMPLE, "cache=0", 1).evalCache() == null);
		try {
			final Evaluator cnn = EvaluatorLoader.load("pretrained_weights/cnn_conv4_hybrid_pretrained.txt");
			if (cnn instanceof CnnEvaluator) {
				check("cache automatica con la CNN: 2^20 voci; cache=0 la spegne", Mcts.autoCacheLog2(cnn) == 20 && mk(cnn, "", 1).evalCache() != null
						&& mk(cnn, "", 1).evalCache().capacity() == (1 << 20) && mk(cnn, "cache=0", 1).evalCache() == null);
			}
		} catch (Throwable t) {
			System.out.println("  (pesi della CNN non trovati dalla cartella corrente: verifica saltata)");
		}
		final Mcts o = mk(SIMPLE, "scale=500", 1);
		check("scale esplicita: lineare", o.valueScale() == 500.0 && o.squash(250) == 0.5f);
		check("tetto automatico dei nodi: 30% dell'heap / byte per nodo, almeno 10000",
				Mcts.autoMaxNodes(2L << 30) == (long) ((2L << 30) * 0.30 / Mcts.BYTES_PER_NODE_ESTIMATE) && Mcts.autoMaxNodes(1000) == 10_000);
		try {
			final Evaluator nn = EvaluatorLoader.load("pretrained_weights/mlp_v3_pretrained.txt");
			if (nn instanceof NeuralEvaluator) {
				final Mcts m = mk(nn, "", 1);
				check("rete: scala = outputScale, lineare", m.valueScale() == ((NeuralEvaluator) nn).outputScale);
			}
		} catch (Throwable t) {
			System.out.println("  (pesi della rete non trovati dalla cartella corrente: verifica sulla rete saltata)");
		}
	}

	// ------------------------------------------------------------------------------------------ casi limite

	/** Posizione senza mosse legali per il Bianco (il Re e' circondato). */
	static FastState stalemateWhite() {
		final State s = new StateTablut();
		final Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE;
		b[2][2] = Pawn.KING;
		b[1][2] = Pawn.BLACK;
		b[3][2] = Pawn.BLACK;
		b[2][1] = Pawn.BLACK;
		b[2][3] = Pawn.BLACK;
		s.setTurn(Turn.WHITE);
		return FastState.fromState(s);
	}

	/** Posizione in cui il Bianco ha UNA sola mossa legale (il Re verso il basso di una cella). */
	static FastState singleMove() {
		final State s = new StateTablut();
		final Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE;
		b[2][2] = Pawn.KING;
		b[1][2] = Pawn.BLACK;
		b[2][1] = Pawn.BLACK;
		b[2][3] = Pawn.BLACK;
		b[4][2] = Pawn.BLACK;
		s.setTurn(Turn.WHITE);
		return FastState.fromState(s);
	}

	static void testEdgeCases() {
		section("casi limite");
		final Mcts m = mk(SIMPLE, "", 1);
		final FastState st = stalemateWhite();
		check("(costruzione) il Bianco non ha mosse", legal(st).length == 0 && st.winner == FastState.WINNER_NONE);
		check("nessuna mossa legale -> NONE, nessun errore", m.findBestMove(st, 1000, rootOnly(st), 1) == Move.NONE && m.lastError == null);
		check("... e nessun albero conservato", m.rootNode() == null);

		final FastState one = singleMove();
		final int[] l1 = legal(one);
		check("(costruzione) una sola mossa legale", l1.length == 1);
		final int mv = m.findBestMove(one, 1000, rootOnly(one), 1);
		check("una sola mossa: restituita subito senza simulazioni", mv == l1[0] && m.lastSims == 0 && m.lastElapsedMillis < 200);

		// terminale: dopo una mossa vincente
		FastState win = null;
		for (FastState p : gamePositions(seed, 40, 1)) {
			if (winNowMoves(p).length > 0) {
				win = p;
				break;
			}
		}
		check("(costruzione) trovata una posizione con vittoria in 1", win != null);
		if (win != null) {
			final FastState term = win.duplicate();
			term.makeMove(winNowMoves(win)[0], new UndoInfo());
			check("(costruzione) la posizione dopo la vittoria e' terminale", term.winner != FastState.WINNER_NONE);
			check("radice terminale -> NONE", m.findBestMove(term, 1000, rootOnly(term), 1) == Move.NONE && m.lastError == null);
		}

		// immutabilita' degli ingressi
		final FastState p = gamePositions(seed + 1, 6, 5).get(3);
		final FastState before = p.duplicate();
		final long[] hist = { 11L, 22L, 33L, p.zobristHash };
		final long[] histCopy = hist.clone();
		final Mcts m2 = mk(SIMPLE, "", 1);
		m2.setMaxSims(500);
		final int mv2 = m2.findBestMove(p, 100000, hist, hist.length);
		check("la radice non e' modificata", sameFast(p, before));
		check("la storia non e' modificata", Arrays.equals(hist, histCopy));
		check("la mossa e' legale", contains(legal(p), mv2));

		// API a Action
		final State dom = p.toState();
		final Mcts m3 = mk(SIMPLE, "", 1);
		m3.setMaxSims(300);
		final Action a = m3.findBestMove(dom, 100000);
		boolean legalForRules = false;
		if (a != null) {
			for (Action x : AshtonRules.generateMoves(dom)) {
				if (x.getFrom().equals(a.getFrom()) && x.getTo().equals(a.getTo())) {
					legalForRules = true;
				}
			}
		}
		check("API Action: mossa legale per il motore indipendente", legalForRules && contains(legal(p), Move.fromAction(a)));
		final Mcts m4 = mk(SIMPLE, "", 1);
		check("API Action: nessuna mossa -> null", m4.findBestMove(stalemateWhite().toState(), 1000) == null);
		check("nome e descrizione", m3.name().equals("mcts") && m3.describeLastSearch().contains("simulazioni="));
	}


	// ------------------------------------------------------------------------------------------ formula di selezione

	/** Un nodo costruito a mano: archi con (visite, q, im, prior in [0,65535]); limiti (-1,1) e nessun figlio. */
	static Mcts.Node handNode(int[] visits, float[] q, float[] im, int[] prior) {
		final int k = visits.length;
		final Mcts.Node n = new Mcts.Node(k);
		for (int i = 0; i < k; i++) {
			n.visits[i] = visits[i];
			n.totalVisits += visits[i];
			n.q[i] = q[i];
			n.im[i] = im == null ? q[i] : im[i];
			n.prior[i] = (char) prior[i];
			n.lo[i] = -1;
			n.hi[i] = 1;
		}
		n.nodeLo = -1;
		n.nodeHi = 1;
		float best = -Float.MAX_VALUE;
		for (int i = 0; i < k; i++) {
			best = Math.max(best, n.im[i]);
		}
		n.nodeIm = best;
		return n;
	}

	static void testSelectEdge() {
		section("formula di selezione PUCT, a esempi calcolati a mano");
		final Mcts m = mk(SIMPLE, "cpuct=1.0,mm=0,vloss=1.0", 1);
		// A: senza prior vince la Q piu' alta; con un prior grande il bonus sqrt(30)*1/(1+10) = 0.498 ribalta: 0.2 + 0.498 > 0.5
		Mcts.Node n = handNode(new int[] { 10, 10, 10 }, new float[] { 0.2f, 0.5f, 0.1f }, null, new int[] { 0, 0, 0 });
		check("A1: solo Q -> l'arco con Q piu' alta", m.selectEdge(n) == 1);
		n.prior[0] = 65535;
		check("A2: il prior grande ribalta la scelta (0.2 + 0.498 > 0.5)", m.selectEdge(n) == 0);
		// B: un arco non visitato vale il suo valore statico iniziale (q) piu' il bonus pieno sqrt(5)*0.5/(1+0)
		n = handNode(new int[] { 0, 5 }, new float[] { 0.3f, 0.4f }, null, new int[] { 0, 0 });
		check("B1: arco mai visitato senza prior: vale solo q (0.3 < 0.4)", m.selectEdge(n) == 1);
		n.prior[0] = 32768;
		check("B2: arco mai visitato con prior 0.5: bonus 1.118 -> scelto", m.selectEdge(n) == 0);
		// C: perdita virtuale: (0.5*5 - 2)/(5+2) = 0.0714 < 0.4; con peso 0 non conta
		n = handNode(new int[] { 5, 5 }, new float[] { 0.5f, 0.4f }, null, new int[] { 0, 0 });
		n.vloss[0] = 2;
		n.totalVloss = 2;
		check("C1: 2 thread in volo sull'arco 0 lo rendono meno attraente (0.071 < 0.4)", m.selectEdge(n) == 1);
		n.vloss[0] = 0;
		n.totalVloss = 0;
		check("C2: senza perdita virtuale vince l'arco 0 (0.5 > 0.4)", m.selectEdge(n) == 0);
		final Mcts noVl = mk(SIMPLE, "cpuct=1.0,mm=0,vloss=0", 1);
		n.vloss[0] = 2;
		n.totalVloss = 2;
		check("C3: peso della perdita virtuale 0: l'arco 0 resta il migliore", noVl.selectEdge(n) == 0);
		// D: miscela col minimax implicito (1-mm)*q + mm*im
		n = handNode(new int[] { 3, 3 }, new float[] { 0.5f, 0.1f }, new float[] { -0.9f, 0.2f }, new int[] { 0, 0 });
		check("D1: mm=0 usa solo la media (0.5 > 0.1)", mk(SIMPLE, "mm=0", 1).selectEdge(n) == 0);
		check("D2: mm=1 usa solo il minimax implicito (-0.9 < 0.2)", mk(SIMPLE, "mm=1", 1).selectEdge(n) == 1);
		check("D3: mm=0.25 pesa la media 0.75 (0.15 > 0.125)", mk(SIMPLE, "mm=0.25", 1).selectEdge(n) == 0);
		check("D4: mm=0.5 (-0.2 < 0.15)", mk(SIMPLE, "mm=0.5", 1).selectEdge(n) == 1);
		check("D5: mm=0.75 pesa il minimax 0.75 (-0.55 < 0.175)", mk(SIMPLE, "mm=0.75", 1).selectEdge(n) == 1);
		// E: potatura: un arco dimostrato perso non si sceglie, anche con Q altissima, se un fratello e' dimostrato almeno pari
		n = handNode(new int[] { 3, 3, 3 }, new float[] { 0.0f, 0.99f, -0.5f }, null, new int[] { 0, 0, 0 });
		n.lo[0] = 0;
		n.hi[0] = 0; // pareggio dimostrato
		n.lo[1] = -1;
		n.hi[1] = -1; // sconfitta dimostrata, ma con Q finta altissima
		n.nodeLo = 0;
		check("E1: l'arco dimostrato perso (hi < lo del nodo) e' potato", m.selectEdge(n) == 0);
		n.nodeLo = -1;
		check("E2: senza un fratello dimostrato migliore l'arco non e' potato", m.selectEdge(n) == 1);
		// F: archi in espansione da un altro thread
		n = handNode(new int[] { 3, 3, 3 }, new float[] { 0.1f, 0.9f, 0.5f }, null, new int[] { 0, 0, 0 });
		n.child[1] = Mcts.EXPANDING;
		check("F1: l'arco in espansione e' saltato (si prende il secondo migliore)", m.selectEdge(n) == 2);
		n.child[0] = Mcts.EXPANDING;
		n.child[2] = Mcts.EXPANDING;
		check("F2: se tutti sono in espansione non c'e' scelta (-1)", m.selectEdge(n) == -1);
		// G: parita' -> il primo arco nell'ordine di generazione (determinismo)
		n = handNode(new int[] { 2, 2, 2 }, new float[] { 0.3f, 0.3f, 0.3f }, null, new int[] { 100, 100, 100 });
		check("G: a parita' di punteggio vince il primo arco", m.selectEdge(n) == 0);
		// H: la radice del bonus e' il totale delle visite del nodo
		n = handNode(new int[] { 0, 100 }, new float[] { 0.0f, 0.5f }, null, new int[] { 65535, 0 });
		check("H: con 100 visite il bonus sqrt(100)*1 = 10 di un arco mai visitato supera ogni Q", m.selectEdge(n) == 0);
		// I: il bonus decresce con le visite dell'arco: (1+n) al denominatore
		n = handNode(new int[] { 50, 49 }, new float[] { 0.4f, 0.4f }, null, new int[] { 65535, 65535 });
		check("I: a parita' di Q e prior si preferisce l'arco meno visitato", m.selectEdge(n) == 1);
		// J: cpuct scala il bonus: con cpuct=0 conta solo Q
		n = handNode(new int[] { 10, 10 }, new float[] { 0.1f, 0.2f }, null, new int[] { 65535, 0 });
		check("J1: con cpuct=1 il prior ribalta (0.1 + 0.41 > 0.2)", m.selectEdge(n) == 0);
		check("J2: con cpuct=0 conta solo Q", mk(SIMPLE, "cpuct=0", 1).selectEdge(n) == 1);
		// K: il bonus e' diviso per (1 + visite + peso*perdita virtuale): con 1 thread in volo sull'arco 0 il bonus scende da 3.317/6 a 3.317/7
		n = handNode(new int[] { 5, 5 }, new float[] { 0.5f, 0.76f }, null, new int[] { 65535, 0 });
		n.vloss[0] = 1;
		n.totalVloss = 1;
		check("K: la perdita virtuale riduce anche il bonus di esplorazione (0.25 + 0.474 < 0.76; senza, 0.25 + 0.553 > 0.76)", m.selectEdge(n) == 1);
	}


	// ------------------------------------------------------------------------------------------ retropropagazione a mano

	static void testBackpropByHand() {
		section("retropropagazione su nodi costruiti a mano");
		final Mcts m = mk(SIMPLE, "", 1);
		final Mcts.Worker w = m.new Worker(4);

		// (a) arco normale con una simulazione in volo: v=0.8 al suo livello -> q = 0.5 + (0.8 - 0.5)/4 = 0.575, 4 visite, perdita virtuale rimossa
		Mcts.Node n = handNode(new int[] { 3, 0 }, new float[] { 0.5f, 0.1f }, null, new int[] { 0, 0 });
		n.vloss[0] = 1;
		n.totalVloss = 1;
		w.pNode[0] = n;
		w.pEdge[0] = 0;
		m.backprop(w, 1, -0.8, null); // il valore della foglia e' dal punto di vista di chi vi muove: all'arco arriva il suo opposto
		check("a: media q = 0.5 + (0.8-0.5)/4", Math.abs(n.q[0] - 0.575f) < 1e-6f && n.visits[0] == 4 && n.totalVisits == 4 && n.vloss[0] == 0 && n.totalVloss == 0);
		check("a: l'altro arco e' intatto", n.q[1] == 0.1f && n.visits[1] == 0);

		// (b) arrivo TARDIVO su un arco gia' risolto (una simulazione partita prima che un altro thread lo risolvesse): il valore esatto non si altera
		n = handNode(new int[] { 2, 0 }, new float[] { 1f, 0.1f }, null, new int[] { 0, 0 });
		n.lo[0] = 1;
		n.hi[0] = 1;
		n.nodeLo = 1;
		n.nodeHi = 1;
		n.vloss[0] = 1;
		n.totalVloss = 1;
		w.pNode[0] = n;
		w.pEdge[0] = 0;
		m.backprop(w, 1, 0.5, null); // all'arco arriva -0.5: non deve toccare q, che resta il valore esatto
		check("b: arrivo tardivo su un arco esatto: q resta 1.0, le visite salgono, la perdita virtuale scende", n.q[0] == 1f && n.visits[0] == 3 && n.vloss[0] == 0);

		// (c) due livelli: i segni si alternano. leaf=0.6 dal punto di vista di chi muove nella foglia -> nB: -0.6, nA: +0.6
		final Mcts.Node nA = handNode(new int[] { 1 }, new float[] { 0.4f }, null, new int[] { 0 });
		final Mcts.Node nB = handNode(new int[] { 0, 2 }, new float[] { 0.3f, 0.2f }, null, new int[] { 0, 0 });
		nA.vloss[0] = 1;
		nA.totalVloss = 1;
		nB.vloss[1] = 1;
		nB.totalVloss = 1;
		w.pNode[0] = nA;
		w.pEdge[0] = 0;
		w.pNode[1] = nB;
		w.pEdge[1] = 1;
		m.backprop(w, 2, 0.6, null);
		check("c: nB arco 1: q = 0.2 + (-0.6-0.2)/3", Math.abs(nB.q[1] - (0.2f + (-0.6f - 0.2f) / 3f)) < 1e-6f && nB.visits[1] == 3);
		check("c: nA arco 0: q = 0.4 + (0.6-0.4)/2 (segno opposto)", Math.abs(nA.q[0] - 0.5f) < 1e-6f && nA.visits[0] == 2 && nA.totalVloss == 0 && nB.totalVloss == 0);

		// (d) la foglia e' un nodo perso per chi vi muove: l'arco del genitore diventa una vittoria dimostrata, e il genitore (un solo arco) e' risolto
		final Mcts.Node loss = new Mcts.Node(0);
		loss.nodeLo = -1;
		loss.nodeHi = -1;
		loss.nodeIm = -1f;
		final Mcts.Node p = handNode(new int[] { 1 }, new float[] { 0.3f }, null, new int[] { 0 });
		p.vloss[0] = 1;
		p.totalVloss = 1;
		w.pNode[0] = p;
		w.pEdge[0] = 0;
		m.backprop(w, 1, -1.0, loss);
		check("d: arco = vittoria dimostrata (lo=hi=+1), q=+1, minimax +1", p.lo[0] == 1 && p.hi[0] == 1 && p.q[0] == 1f && p.im[0] == 1f);
		check("d: il nodo con un solo arco vinto e' risolto vinto (lo=hi=+1, im=+1)", p.nodeLo == 1 && p.nodeHi == 1 && p.nodeIm == 1f);

		// (e) la sintesi sale di un livello: nA ha un solo arco verso nB (vinto per chi muove in nB) -> nA perso (lo=hi=-1, im=-1)
		final Mcts.Node gp = handNode(new int[] { 1 }, new float[] { 0.3f }, null, new int[] { 0 });
		final Mcts.Node par = handNode(new int[] { 0, 0 }, new float[] { 0.1f, 0.2f }, null, new int[] { 0, 0 });
		gp.vloss[0] = 1;
		gp.totalVloss = 1;
		par.vloss[0] = 1;
		par.totalVloss = 1;
		w.pNode[0] = gp;
		w.pEdge[0] = 0;
		w.pNode[1] = par;
		w.pEdge[1] = 0;
		m.backprop(w, 2, -1.0, loss);
		check("e: par: arco 0 vinto, ma l'altro arco e' sconosciuto: limiti (1,1) e minimax +1", par.lo[0] == 1 && par.hi[0] == 1 && par.nodeLo == 1 && par.nodeHi == 1);
		check("e: gp: l'unico arco porta a un nodo vinto per l'avversario -> perso (lo=hi=-1, im=-1)", gp.lo[0] == -1 && gp.hi[0] == -1 && gp.nodeLo == -1 && gp.nodeHi == -1
				&& gp.nodeIm == -1f && gp.q[0] == -1f);

		// (f) se la sintesi del nodo non cambia, i livelli sopra non vengono toccati: foglia nuova con limiti (-1,1) e minimax uguale a quello gia' noto
		final Mcts.Node fresh = new Mcts.Node(0);
		fresh.nodeLo = -1;
		fresh.nodeHi = 1;
		fresh.nodeIm = -0.3f;
		final Mcts.Node top = handNode(new int[] { 1 }, new float[] { 0.3f }, null, new int[] { 0 });
		final Mcts.Node mid = handNode(new int[] { 0, 0 }, new float[] { 0.3f, 0.1f }, null, new int[] { 0, 0 });
		top.vloss[0] = 1;
		top.totalVloss = 1;
		mid.vloss[0] = 1;
		mid.totalVloss = 1;
		w.pNode[0] = top;
		w.pEdge[0] = 0;
		w.pNode[1] = mid;
		w.pEdge[1] = 0;
		final float topImBefore = top.im[0];
		m.backprop(w, 2, -0.3, fresh);
		check("f: il minimax dell'arco del genitore diventa -(minimax della foglia) = 0.3, invariato al massimo (0.3)", mid.im[0] == 0.3f && mid.nodeIm == 0.3f && top.im[0] == topImBefore && top.visits[0] == 2);
	}


	// ------------------------------------------------------------------------------------------ annullamento a meta' cammino

	static void testAbortReleasesVirtualLoss() {
		section("simulazione annullata a meta' cammino: la perdita virtuale lungo il cammino viene restituita");
		int tested = 0, bad = 0;
		for (FastState p : gamePositions(seed + 30, 10, 5)) {
			if (legal(p).length < 3) {
				continue;
			}
			final Mcts m = mk(SIMPLE, "expand=0,reuse=false,escape=false", 1);
			search(m, p, 300);
			final Mcts.Node r = m.rootNode();
			int e = -1;
			for (int i = 0; i < r.nMoves && e < 0; i++) {
				final Mcts.Node c = r.child[i];
				if (c != null && c.nMoves > 0 && !c.solved() && r.lo[i] == -1 && r.hi[i] == 1) {
					e = i;
				}
			}
			if (r.solved() || e < 0) {
				continue;
			}
			// stato costruito: alla radice e' selezionabile il solo arco e (gli altri sono dimostrati inferiori), e nel figlio tutti gli archi sono "in espansione"
			// da un altro thread: la simulazione scende di un livello e li trova tutti non selezionabili, quindi si annulla a profondita' 1
			r.lo[e] = 0;
			r.nodeLo = 0;
			for (int i = 0; i < r.nMoves; i++) {
				if (i != e) {
					r.lo[i] = -1;
					r.hi[i] = -1;
				}
			}
			final Mcts.Node child = r.child[e];
			for (int j = 0; j < child.nMoves; j++) {
				child.child[j] = Mcts.EXPANDING;
			}
			final Mcts.Worker w = m.new Worker(8);
			final Mcts.Control ctl = new Mcts.Control(r, p.duplicate(), new long[] { p.zobristHash }, 1, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE);
			final boolean done = m.simulate(w, ctl);
			tested++;
			if (done || r.vloss[e] != 0 || r.totalVloss != 0 || child.totalVloss != 0) {
				bad++;
				System.out.println("  simulate=" + done + " vloss radice=" + r.vloss[e] + " totale=" + r.totalVloss);
			}
			if (tested >= 6) {
				break;
			}
		}
		check("annullamento a profondita' 1: la simulazione restituisce false e nessuna perdita virtuale resta nei nodi (" + bad + " anomalie su " + tested + ")", tested >= 4 && bad == 0);
	}


	// ------------------------------------------------------------------------------------------ scelta della mossa alla radice

	static void testChooseEdgeByHand() {
		section("scelta della mossa alla radice, su alberi costruiti a mano");
		final Mcts m = mk(SIMPLE, "", 1);
		// Y: nodo perso per chi vi muove, con un solo arco che porta a Z; Z: nodo vinto con un arco terminale. Dalla radice l'arco verso Y vince in 3 semi-mosse
		final Mcts.Node z = new Mcts.Node(1);
		z.lo[0] = 1;
		z.hi[0] = 1;
		z.nodeLo = 1;
		z.nodeHi = 1;
		final Mcts.Node y = new Mcts.Node(1);
		y.lo[0] = -1;
		y.hi[0] = -1;
		y.child[0] = z;
		y.nodeLo = -1;
		y.nodeHi = -1;
		check("(costruzione) distanze: arco terminale = 1, arco verso Y = 3", Mcts.edgeWinDist(z, 0) == 1);

		// 0: vittoria terminale (1 semi-mossa, 1 visita); 1: vittoria in 3 (100 visite); 2: arco non dimostrato con moltissime visite
		Mcts.Node r = handNode(new int[] { 1, 100, 5000 }, new float[] { 1f, 1f, 0.9f }, null, new int[] { 0, 0, 0 });
		r.lo[0] = 1;
		r.hi[0] = 1;
		r.lo[1] = 1;
		r.hi[1] = 1;
		r.child[1] = y;
		check("(costruzione) distanza dell'arco 1 = 3", Mcts.edgeWinDist(r, 1) == 3 && Mcts.edgeWinDist(r, 0) == 1);
		check("a: tra due vittorie dimostrate si sceglie la PIU' BREVE anche se l'altra ha 100 volte le visite", m.chooseEdge(r) == 0);
		r.lo[0] = -1;
		r.hi[0] = 1;
		check("b: se la vittoria breve non e' dimostrata, si sceglie la sola dimostrata, non la piu' visitata", m.chooseEdge(r) == 1);
		// a parita' di distanza decidono le visite
		r = handNode(new int[] { 7, 30, 20 }, new float[] { 1f, 1f, 1f }, null, new int[] { 0, 0, 0 });
		for (int i = 0; i < 3; i++) {
			r.lo[i] = 1;
			r.hi[i] = 1;
		}
		check("c: a parita' di distanza vince l'arco piu' visitato", m.chooseEdge(r) == 1);
		// senza vittorie dimostrate: la piu' visitata, ma MAI un arco dimostrato perso se ne resta un altro
		r = handNode(new int[] { 500, 10, 3 }, new float[] { 0.9f, 0.2f, 0.1f }, null, new int[] { 0, 0, 0 });
		r.lo[0] = -1;
		r.hi[0] = -1;
		check("d: l'arco dimostrato perso e' escluso anche se e' il piu' visitato", m.chooseEdge(r) == 1);
		r.lo[1] = -1;
		r.hi[1] = -1;
		r.lo[2] = -1;
		r.hi[2] = -1;
		check("e: se tutti sono dimostrati persi si torna alla piu' visitata", m.chooseEdge(r) == 0);
		// parita' di visite: la media migliore, poi il minimax implicito migliore, poi il primo
		r = handNode(new int[] { 10, 10, 10 }, new float[] { 0.1f, 0.4f, 0.4f }, new float[] { 0f, 0.1f, 0.3f }, new int[] { 0, 0, 0 });
		check("f: a parita' di visite e di media vince il minimax implicito migliore (arco 2)", m.chooseEdge(r) == 2);
		r = handNode(new int[] { 10, 10, 10 }, new float[] { 0.1f, 0.4f, 0.3f }, new float[] { 0f, 0.1f, 0.3f }, new int[] { 0, 0, 0 });
		check("g: a parita' di visite vince la media migliore (arco 1)", m.chooseEdge(r) == 1);
		r = handNode(new int[] { 10, 10 }, new float[] { 0.4f, 0.4f }, null, new int[] { 0, 0 });
		check("h: a parita' totale vince il primo arco", m.chooseEdge(r) == 0);
	}


	// ------------------------------------------------------------------------------------------ contabilita' degli slot di simulazione

	static void testSlotAccounting() throws Exception {
		section("tetto di simulazioni: contabilita' degli slot (deterministica, senza corse tra thread)");
		final FastState p = gamePositions(seed + 50, 4, 5).get(2);
		final Mcts m = mk(SIMPLE, "expand=0,reuse=false", 1);
		final Mcts.Node root = handNode(new int[] { 0, 0, 0 }, new float[] { 0.1f, 0.2f, 0.3f }, null, new int[] { 100, 100, 100 });
		// tutti gli slot (5) sono stati assegnati, ma solo 3 simulazioni sono completate: 2 sono "in volo" e potrebbero ancora annullarsi e riprovare
		final Mcts.Control c = new Mcts.Control(root, p.duplicate(), new long[] { p.zobristHash }, 1, System.nanoTime() + 20_000_000_000L, 5, Long.MAX_VALUE);
		c.simsStarted.set(5);
		c.simsDone.set(3);
		final Mcts.Worker w = m.new Worker(8);
		final Thread t = new Thread() {
			@Override
			public void run() {
				m.runWorker(w, c);
			}
		};
		t.start();
		Thread.sleep(150);
		check("slot esauriti ma simulazioni in volo: il worker NON ferma gli altri e attende", !c.stop && t.isAlive());
		check("lo slot oltre il tetto e' stato restituito (nessun accumulo: " + c.simsStarted.get() + ")", c.simsStarted.get() == 5);
		c.simsDone.set(5); // le simulazioni in volo si completano
		t.join(3000);
		check("completate tutte le simulazioni previste: il worker ferma tutti ed esce", !t.isAlive() && c.stop);
		check("nessuna simulazione e' stata eseguita da questo worker (i slot erano di altri)", w.sims == 0 && c.simsStarted.get() == 5);

		// scadenza: con simulazioni ancora "in volo" ma il tempo finito, il worker non resta in attesa all'infinito
		final Mcts.Control late = new Mcts.Control(root, p.duplicate(), new long[] { p.zobristHash }, 1, System.nanoTime() - 1_000_000L, 5, Long.MAX_VALUE);
		late.simsStarted.set(5);
		late.simsDone.set(3);
		final Thread t2 = new Thread() {
			@Override
			public void run() {
				m.runWorker(m.new Worker(8), late);
			}
		};
		t2.start();
		t2.join(3000);
		check("scaduto il tempo il worker esce anche con simulazioni in volo", !t2.isAlive() && late.stop);
	}

	// ------------------------------------------------------------------------------------------ struttura dell'albero

	static final String[] STRUCT_CONFIGS = { "expand=0,mm=0,cache=0", "expand=1", "expand=2,mm=0.5", "expand=1,mm=1.0,cache=10", "expand=0,tau=1.0,cpuct=2.0,reuse=false" };

	static void testStructure() {
		section("struttura dell'albero (walker indipendente), 1 thread");
		final List<FastState> pos = new ArrayList<>(gamePositions(seed + 2, 12, 9));
		pos.addAll(densePositions(seed + 3, 25));
		long nodes = 0, means = 0, skipped = 0, term = 0, withChild = 0;
		int bad = 0, runs = 0;
		for (String cfg : STRUCT_CONFIGS) {
			final int expand = MctsConfig.parse(cfg).expandAfter;
			for (int i = 0; i < pos.size(); i++) {
				final FastState p = pos.get(i);
				final Mcts m = mk(SIMPLE, cfg, 1);
				final int mv = search(m, p, 400);
				runs++;
				if (mv == Move.NONE) {
					if (legal(p).length != 0) {
						check("NONE con mosse legali", false);
					}
					continue;
				}
				if (!contains(legal(p), mv)) {
					check("mossa illegale", false);
				}
				final Walk w = new Walk(m, SIMPLE, true, expand);
				final String err = w.run(m.rootNode(), p, keysOf(p));
				if (err != null) {
					bad++;
					if (bad <= 5) {
						System.out.println("  [" + cfg + " posizione " + i + "] " + err);
					}
				}
				nodes += w.nodes;
				means += w.meansChecked;
				skipped += w.meansSkipped;
				term += w.termEdges;
				withChild += w.edgesWithChild;
				if (m.rootNode() != null) {
					if (Mcts.countNodes(m.rootNode()) != m.nodeCountForTest() || m.nodeCountForTest() != m.lastNodes) {
						check("contatore dei nodi coerente con l'albero", false);
					}
					if (!m.rootNode().solved() && m.rootNode().totalVisits != m.lastSims) {
						check("visite della radice = simulazioni (" + m.rootNode().totalVisits + " vs " + m.lastSims + ")", false);
					}
				}
			}
		}
		check("walker: 0 discordanze su " + runs + " alberi (" + bad + ")", bad == 0);
		check("non vacuo: " + nodes + " nodi, " + withChild + " archi con figlio, " + term + " archi terminali, " + means + " medie ricostruite (" + skipped
				+ " saltate)", nodes > 10000 && means > 5000 && term > 100);
	}

	// ------------------------------------------------------------------------------------------ determinismo / differenziali

	static boolean sameRoot(Mcts a, Mcts b) {
		final Mcts.Node x = a.rootNode(), y = b.rootNode();
		if (x == null || y == null) {
			return x == y;
		}
		return x.nMoves == y.nMoves && Arrays.equals(x.visits, y.visits) && Arrays.equals(x.q, y.q) && Arrays.equals(x.im, y.im) && Arrays.equals(x.lo, y.lo)
				&& Arrays.equals(x.hi, y.hi) && Arrays.equals(x.moves, y.moves) && Arrays.equals(x.prior, y.prior);
	}

	static void testDeterminism() {
		section("determinismo e confronti differenziali (1 thread, simulazioni fisse)");
		final List<FastState> pos = new ArrayList<>(gamePositions(seed + 4, 8, 11));
		pos.addAll(densePositions(seed + 5, 8));
		int same = 0, capSame = 0, cacheSame = 0, histSame = 0, n = 0, evalsOk = 0, withHits = 0, eligible = 0, deepHist = 0;
		final Random rnd = new Random(seed);
		for (FastState p : pos) {
			if (legal(p).length < 2) {
				continue;
			}
			n++;
			final Mcts a = mk(SIMPLE, "expand=1,mm=0.3", 1);
			final Mcts b = mk(SIMPLE, "expand=1,mm=0.3", 1);
			final int ma = search(a, p, 1500), mb = search(b, p, 1500);
			if (ma == mb && sameRoot(a, b) && a.lastEvals == b.lastEvals) {
				same++;
			}
			// capacita' iniziale degli array del percorso minuscola: esercita la crescita (e la storia lunga quella della storia)
			final Mcts c = new Mcts(SIMPLE, MctsConfig.parse("expand=1,mm=0.3"), 1, 2).setSafetyMarginMillis(0);
			if (search(c, p, 1500) == ma && sameRoot(a, c) && c.lastError == null && a.lastError == null) {
				capSame++;
			}
			// cache si/no: stessi risultati; le valutazioni calcolate + quelle servite dalla cache = quelle del caso senza cache
			final Mcts on = mk(SIMPLE, "expand=1,cache=12", 1), off = mk(SIMPLE, "expand=1,cache=0", 1);
			final int mon = search(on, p, 1500), moff = search(off, p, 1500);
			if (mon == moff && sameRoot(on, off)) {
				cacheSame++;
			}
			if (on.lastEvals + on.lastCacheHits == off.lastEvals && off.lastCacheHits == 0) {
				evalsOk++;
			}
			if (on.lastSims >= 500) {
				eligible++;
				if (on.lastCacheHits > 0) {
					withHits++;
				}
			}
			// storia con la radice in fondo oppure senza: la ricerca la aggiunge da se'; e una storia LUNGA (oltre la capacita' iniziale) e' gestita
			final Mcts h1 = mk(SIMPLE, "expand=1", 1), h2 = mk(SIMPLE, "expand=1", 1);
			h1.setMaxSims(3000);
			h2.setMaxSims(3000);
			final long[] full = new long[120]; // 120 voci: la capacita' iniziale di 64 cresce a 128, restano 8 posti oltre la storia: un cammino piu' profondo li supera
			for (int i = 0; i < 119; i++) {
				full[i] = rnd.nextLong();
			}
			full[119] = p.zobristHash;
			final int r1 = h1.findBestMove(p, 100000, full, 120);
			final int r2 = h2.findBestMove(p, 100000, Arrays.copyOf(full, 119), 119);
			if (h1.lastMaxDepth > 8) {
				deepHist++;
			}
			if (r1 == r2 && sameRoot(h1, h2) && h1.lastError == null && h2.lastError == null) {
				histSame++;
			}
		}
		check("due ricerche identiche danno lo stesso albero (" + same + "/" + n + ")", same == n);
		check("capacita' iniziale 2 dell'array del percorso: stesso albero (" + capSame + "/" + n + ")", capSame == n);
		check("cache si/no: stesso albero (" + cacheSame + "/" + n + ")", cacheSame == n);
		check("cache: valutazioni + hit = valutazioni senza cache (" + evalsOk + "/" + n + ")", evalsOk == n);
		check("non vacuo: la cache serve valutazioni in tutte le ricerche con almeno 500 simulazioni (" + withHits + "/" + eligible + ")", eligible >= 30 && withHits == eligible);
		check("storia con o senza la radice in fondo: stesso albero (" + histSame + "/" + n + ")", histSame == n);
		check("non vacuo: con la storia di 120 voci molte ricerche scendono oltre 8 semi-mosse, cioe' oltre i posti liberi della storia (" + deepHist + "/" + n + "; basta che molte lo facciano, la soglia e' del 25%)", deepHist * 4 >= n);
	}

	// ------------------------------------------------------------------------------------------ una simulazione, valore esatto

	static void testOneSimulationExact() {
		section("una simulazione: valore dell'arco = minimax a 2 piani delle valutazioni statiche");
		final List<FastState> pos = new ArrayList<>(gamePositions(seed + 6, 40, 4));
		pos.addAll(densePositions(seed + 7, 300));
		int tested = 0, whiteTested = 0, blackTested = 0, wrong = 0;
		final Mcts probe = mk(SIMPLE, "", 1);
		for (FastState p : pos) {
			final Mcts m = mk(SIMPLE, "expand=0,mm=0,cache=0,escape=false", 1);
			if (legal(p).length < 2 || search(m, p, 1) == Move.NONE) {
				continue;
			}
			final Mcts.Node r = m.rootNode();
			int i = -1;
			for (int e = 0; e < r.nMoves; e++) {
				if (r.visits[e] == 1) {
					i = e;
				}
			}
			if (i < 0 || r.lo[i] == r.hi[i]) {
				continue; // arco terminale: valore esatto, non il minimax delle valutazioni
			}
			final FastState c = p.duplicate();
			c.makeMove(r.moves[i], new UndoInfo());
			final boolean childWhite = c.turn == BoardConstants.WHITE_TURN; // chi sceglie tra i nipoti
			final int[] replies = legal(c);
			if (replies.length == 0) {
				continue;
			}
			// valore di ogni nipote dal punto di vista di chi muove nel FIGLIO; il figlio sceglie il massimo; la radice vede il suo opposto
			double best = -2;
			for (int mv : replies) {
				final UndoInfo u2 = new UndoInfo();
				c.makeMove(mv, u2);
				double v;
				if (c.winner != FastState.WINNER_NONE) {
					v = 1;
				} else {
					final float white = probe.squash(SIMPLE.evaluate(c));
					v = childWhite ? white : -white;
				}
				c.unmakeMove(mv, u2);
				best = Math.max(best, v);
			}
			tested++;
			if (p.turn == BoardConstants.WHITE_TURN) {
				whiteTested++;
			} else {
				blackTested++;
			}
			if (Math.abs(r.q[i] - (-best)) > 1e-6) {
				wrong++;
				if (wrong < 4) {
					System.out.println("  q=" + r.q[i] + " atteso " + (-best) + " (radice " + (p.turn == BoardConstants.WHITE_TURN ? "bianca" : "nera") + ")");
				}
			}
		}
		check("valore dell'arco dopo 1 simulazione = -max(nipoti) su " + tested + " posizioni (" + wrong + " errori)", wrong == 0 && whiteTested >= 15 && blackTested >= 15);
		System.out.println("  posizioni: " + whiteTested + " con la radice al Bianco, " + blackTested + " al Nero");
	}

	// ------------------------------------------------------------------------------------------ rilevatore di fuga del Re

	static void testEscapeScan() {
		section("rilevatore di fuga del Re (kingCanEscapeNow) contro la forza bruta");
		final List<FastState> pool = new ArrayList<>(gamePositions(seed + 20, 120, 1));
		pool.addAll(densePositions(seed + 21, 20000));
		// posizioni "quasi di fuga": il Re vicino al bordo, per avere molti positivi e molti casi di bordo (citadel, trono)
		final Random rnd = new Random(seed + 22);
		for (int i = 0; i < 8000; i++) {
			final State st = TestPositions.randomPosition(rnd);
			final FastState f = FastState.fromState(st);
			if (f.winner == FastState.WINNER_NONE) {
				pool.add(f);
			}
		}
		int total = 0, positives = 0, whiteToMove = 0, mismatches = 0, blackFalse = 0;
		for (FastState p : pool) {
			total++;
			final boolean scan = Mcts.kingCanEscapeNow(p);
			if (p.turn == BoardConstants.WHITE_TURN) {
				whiteToMove++;
				final boolean oracle = winNowMoves(p).length > 0;
				if (scan) {
					positives++;
				}
				if (scan != oracle) {
					mismatches++;
					if (mismatches < 4) {
						System.out.println("  scanner=" + scan + " forza bruta=" + oracle + "\n" + p);
					}
				}
			} else if (scan) {
				blackFalse++;
			}
		}
		check("scanner == forza bruta su " + whiteToMove + " posizioni con il Bianco di turno (" + positives + " fughe, " + mismatches + " discordanze)", mismatches == 0
				&& positives >= 300 && whiteToMove - positives >= 300);
		check("con il Nero di turno non segnala mai nulla", blackFalse == 0);
		// Re che parte da una citadel: posizione non raggiungibile in partita (il Re non vi entra), ma generateMoves la tratta e lo scanner deve copiarla.
		// Senza queste posizioni il ramo "originCitadel" dello scanner non e' esercitato da nessun test.
		final int[][] camps = { { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, { 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 }, { 3, 0 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, { 3, 8 }, { 4, 8 }, { 5, 8 },
				{ 4, 7 } };
		int citTotal = 0, citPositives = 0, citMismatch = 0;
		for (int i = 0; i < 8000; i++) {
			final State st = TestPositions.randomPosition(rnd);
			final Pawn[][] b = st.getBoard();
			int kr = -1, kc = -1;
			for (int r = 0; r < 9; r++) {
				for (int c = 0; c < 9; c++) {
					if (b[r][c] == Pawn.KING) {
						kr = r;
						kc = c;
					}
				}
			}
			final int[] cm = camps[rnd.nextInt(camps.length)];
			if (kr < 0 || b[cm[0]][cm[1]] != Pawn.EMPTY) {
				continue;
			}
			b[kr][kc] = (kr == 4 && kc == 4) ? Pawn.THRONE : Pawn.EMPTY;
			b[cm[0]][cm[1]] = Pawn.KING;
			st.setTurn(Turn.WHITE);
			final FastState f = FastState.fromState(st);
			if (f.winner != FastState.WINNER_NONE) {
				continue;
			}
			citTotal++;
			final boolean scan = Mcts.kingCanEscapeNow(f);
			final boolean oracle = winNowMoves(f).length > 0;
			if (scan) {
				citPositives++;
			}
			if (scan != oracle) {
				citMismatch++;
				if (citMismatch < 3) {
					System.out.println("  Re su citadel: scanner=" + scan + " forza bruta=" + oracle + "\n" + f);
				}
			}
		}
		check("Re su una citadel: scanner == forza bruta su " + citTotal + " posizioni (" + citPositives + " fughe, " + citMismatch + " discordanze)", citMismatch == 0 && citTotal >= 800
				&& citPositives >= 50 && citTotal - citPositives >= 50);
		// costruite: (1) dalla citadel (1,4) il Re puo' attraversare la citadel (0,4), che e' anche un bordo -> fuga; (2) dalla citadel (3,0) con la colonna
		// bloccata, la riga 3 e' libera fino alla citadel (3,8) a 8 passi: oltre il limite di 5, non si entra -> nessuna fuga
		for (int variant = 0; variant < 2; variant++) {
			final State st = new StateTablut();
			final Pawn[][] b = st.getBoard();
			for (int r = 0; r < 9; r++) {
				for (int c = 0; c < 9; c++) {
					b[r][c] = Pawn.EMPTY;
				}
			}
			b[4][4] = Pawn.THRONE;
			if (variant == 0) {
				b[1][4] = Pawn.KING;
			} else {
				b[3][0] = Pawn.KING;
				b[2][0] = Pawn.WHITE;
				b[4][0] = Pawn.WHITE;
			}
			st.setTurn(Turn.WHITE);
			final FastState f = FastState.fromState(st);
			final boolean oracle = winNowMoves(f).length > 0;
			check("costruita " + (variant == 0 ? "(1,4)->(0,4)" : "(3,0) con la riga libera fino alla citadel a 8 passi") + ": forza bruta=" + oracle + ", scanner uguale e atteso "
					+ (variant == 0), Mcts.kingCanEscapeNow(f) == oracle && oracle == (variant == 0));
		}

		final FastState noKing = stalemateWhite();
		noKing.kingSquare = -1;
		check("Re catturato: nessuna fuga", !Mcts.kingCanEscapeNow(noKing));
	}

	// ------------------------------------------------------------------------------------------ tattica

	static final class Tac {
		final FastState s;
		final int[] win1, win3, safe;
		final boolean loss2;
		final int k;

		Tac(FastState s) {
			this.s = s;
			this.k = legal(s).length;
			this.win1 = winNowMoves(s);
			this.safe = safeMoves(s);
			this.win3 = win1.length > 0 ? win1 : win3Moves(s);
			this.loss2 = win1.length == 0 && lossIn2(s);
		}
	}

	static void testTactics() {
		section("tattica con oracoli di forza bruta");
		final long t0 = System.nanoTime();
		final List<FastState> pool = new ArrayList<>(gamePositions(seed + 8, 30, 3));
		pool.addAll(densePositions(seed + 9, 2500));
		final List<Tac> win1 = new ArrayList<>(), win3only = new ArrayList<>(), loss2 = new ArrayList<>(), mixedSafe = new ArrayList<>();
		for (FastState p : pool) {
			if (legal(p).length < 2) {
				continue;
			}
			final Tac t = new Tac(p);
			if (t.win1.length > 0) {
				win1.add(t);
			} else if (t.win3.length > 0) {
				win3only.add(t);
			} else if (t.loss2) {
				loss2.add(t);
			}
			if (t.win1.length == 0 && t.safe.length > 0 && t.safe.length < t.k && mixedSafe.size() < 60) {
				mixedSafe.add(t);
			}
		}
		System.out.println("  classificate in " + (System.nanoTime() - t0) / 1_000_000 + " ms: vittoria in 1: " + win1.size() + ", vittoria forzata in 3 (non in 1): "
				+ win3only.size() + ", sconfitta in 2: " + loss2.size() + ", mosse perdenti miste a sicure: " + mixedSafe.size());

		// vittoria in 1
		int ok = 0, white = 0, black = 0, n = 0;
		for (Tac t : win1) {
			final Mcts m = mk(SIMPLE, "expand=1", 1);
			final int mv = search(m, t.s, 100);
			n++;
			final boolean w = t.s.turn == BoardConstants.WHITE_TURN;
			if (contains(t.win1, mv) && m.lastRootLo == 1 && m.lastRootHi == 1 && m.lastRootValue == (w ? 1.0 : -1.0)) {
				ok++;
				if (w) {
					white++;
				} else {
					black++;
				}
			}
		}
		check("vittoria in 1: mossa vincente, radice dimostrata vinta, valore +-1 (" + ok + "/" + n + "; Bianco " + white + ", Nero " + black + ")",
				ok == n && white >= 5 && black >= 5);

		// sicurezza: mai una mossa che concede una vittoria immediata se ne esiste una che non la concede
		for (Evaluator ev : new Evaluator[] { SIMPLE, NOISE }) {
			ok = 0;
			n = 0;
			for (Tac t : mixedSafe) {
				final Mcts m = mk(ev, "expand=1", 1);
				final int mv = search(m, t.s, 2500);
				n++;
				if (contains(t.safe, mv)) {
					ok++;
				} else {
					System.out.println("  mossa non sicura scelta (" + (ev == SIMPLE ? "Simple" : "rumore") + "): " + Move.toString(mv) + ", sicure=" + t.safe.length + "/" + t.k);
				}
			}
			check("sicurezza (" + (ev == SIMPLE ? "Simple" : "rumore") + "): non concede una vittoria immediata (" + ok + "/" + n + ")", n >= 20 && ok == n);
		}

		// vittoria forzata in 3. Correttezza (rigida): ogni vittoria dichiarata ha un arco dimostrato, l'albero e' certificato dal walker (che ricalcola
		// i limiti dai soli terminali veri) e, se nell'albero era dimostrata una vittoria in 3, si sceglie una vittoria in 3 e non una piu' lunga.
		// Completezza (misurata, non pretesa al 100%): la ricerca e' guidata dall'evaluator, quindi una mossa vincente che la valutazione disprezza
		// (prior ~0) puo' non essere mai esplorata; il rilevatore di fuga copre le vittorie del Bianco, non le catture del Re del Nero.
		for (Evaluator ev : new Evaluator[] { SIMPLE, NOISE }) {
			final int[] tot = new int[2], prov = new int[2]; // 0 = Bianco di turno, 1 = Nero
			int longer = 0, notShortest = 0, certified = 0, claimed = 0, earlyStopMissed = 0;
			for (Tac t : win3only) {
				final Mcts m = mk(ev, "expand=1", 1);
				final int mv = search(m, t.s, 60000);
				final int col = t.s.turn == BoardConstants.WHITE_TURN ? 0 : 1;
				tot[col]++;
				final Mcts.Node r = m.rootNode();
				final int ci = indexOfMove(r, mv);
				if (m.lastRootLo == 1) {
					claimed++;
					if (m.lastSims >= 60000) {
						earlyStopMissed++; // radice dimostrata ma la ricerca e' proseguita fino al tetto: la fermata anticipata non funziona
					}
					if (m.lastRootHi == 1 && ci >= 0 && r.lo[ci] == 1) {
						prov[col]++;
					}
					final String err = new Walk(m, ev, true, 1).run(r, t.s, keysOf(t.s));
					if (err == null) {
						certified++;
					} else {
						System.out.println("  walker: " + err);
					}
					if (ci >= 0 && !contains(t.win3, mv)) {
						longer++;
						for (int e = 0; e < r.nMoves; e++) {
							if (r.lo[e] == 1 && contains(t.win3, r.moves[e])) {
								notShortest++; // una vittoria in 3 era dimostrata nell'albero ma e' stata scelta una piu' lunga
								break;
							}
						}
					}
				}
			}
			final String nm = ev == SIMPLE ? "Simple" : "rumore";
			System.out.println("  vittoria in 3, " + nm + ": dimostrate Bianco " + prov[0] + "/" + tot[0] + ", Nero " + prov[1] + "/" + tot[1] + " (scelte piu' lunghe della minima: "
					+ longer + ")");
			check("vittoria in 3 (" + nm + "), correttezza: ogni vittoria dichiarata ha un arco dimostrato e l'albero e' certificato (" + certified + "/" + claimed + ")",
					certified == claimed && prov[0] + prov[1] == claimed);
			check("vittoria in 3 (" + nm + "), la ricerca si ferma appena la radice e' dimostrata (" + earlyStopMissed + " ricerche proseguite fino a 60000 simulazioni)", earlyStopMissed == 0);
			check("vittoria in 3 (" + nm + "), brevita': tra le dimostrate si sceglie la piu' breve (" + notShortest + " non minime)", notShortest == 0);
			if (ev == SIMPLE) {
				check("vittoria in 3 (Simple), completezza: Bianco >= 95%, Nero >= 70% (" + prov[0] + "/" + tot[0] + ", " + prov[1] + "/" + tot[1] + ")",
						tot[0] >= 50 && prov[0] * 100 >= tot[0] * 95 && tot[1] >= 10 && prov[1] * 100 >= tot[1] * 70);
			} else {
				check("vittoria in 3 (rumore), completezza: almeno la meta' (" + (prov[0] + prov[1]) + "/" + (tot[0] + tot[1]) + ")", (prov[0] + prov[1]) * 2 >= tot[0] + tot[1]);
			}
		}

		// sconfitta forzata in 2: dimostrata, e comunque una mossa legale
		ok = 0;
		n = 0;
		for (Tac t : loss2) {
			final Mcts m = mk(SIMPLE, "expand=1", 1);
			final int mv = search(m, t.s, 20000);
			n++;
			final boolean w = t.s.turn == BoardConstants.WHITE_TURN;
			if (contains(legal(t.s), mv) && m.lastRootLo == -1 && m.lastRootHi == -1 && m.lastRootValue == (w ? -1.0 : 1.0)) {
				ok++;
			}
		}
		check("sconfitta forzata in 2: radice dimostrata persa, valore +-1, mossa legale (" + ok + "/" + n + ")", n >= 3 && ok == n);

		// nessuna contraddizione, su tutte le classi, con pochi colpi
		int contradictions = 0, total = 0;
		for (List<Tac> cls : Arrays.asList(win1, win3only, loss2, mixedSafe)) {
			for (Tac t : cls) {
				final Mcts m = mk(SIMPLE, "expand=1,mm=0.5", 1);
				search(m, t.s, 150);
				total++;
				final boolean isWin = t.win3.length > 0;
				if ((isWin && m.lastRootHi == -1) || (t.loss2 && m.lastRootLo == 1)) {
					contradictions++;
				}
				final Walk w = new Walk(m, SIMPLE, true, 1);
				final String err = m.rootNode() == null ? null : w.run(m.rootNode(), t.s, keysOf(t.s));
				if (err != null) {
					contradictions++;
					System.out.println("  walker: " + err);
				}
			}
		}
		check("nessuna contraddizione tra limiti dimostrati e oracolo, albero coerente (" + total + " posizioni)", contradictions == 0);
	}

	// ------------------------------------------------------------------------------------------ ripetizione

	/** Un ciclo di 4 semi-mosse senza catture a partire da una posizione con il Bianco di turno: {p0,p1,p2,p3} e le 4 mosse; null se non trovato. */
	static Object[] findCycle(FastState p0) {
		final UndoInfo u1 = new UndoInfo(), u2 = new UndoInfo(), u3 = new UndoInfo(), u4 = new UndoInfo();
		final FastState s = p0.duplicate();
		for (int m1 : legal(s)) {
			s.makeMove(m1, u1);
			if (!u1.wasCapture() && s.winner == FastState.WINNER_NONE) {
				final FastState p1 = s.duplicate();
				for (int m2 : legal(s)) {
					s.makeMove(m2, u2);
					if (!u2.wasCapture() && s.winner == FastState.WINNER_NONE) {
						final FastState p2 = s.duplicate();
						final int m3 = Move.of(Move.getTo(m1), Move.getFrom(m1));
						final int m4 = Move.of(Move.getTo(m2), Move.getFrom(m2));
						if (contains(legal(s), m3)) {
							s.makeMove(m3, u3);
							if (!u3.wasCapture() && s.winner == FastState.WINNER_NONE) {
								final FastState p3 = s.duplicate();
								if (contains(legal(s), m4)) {
									s.makeMove(m4, u4);
									final boolean back = s.samePosition(p0);
									s.unmakeMove(m4, u4);
									if (back) {
										s.unmakeMove(m3, u3);
										s.unmakeMove(m2, u2);
										s.unmakeMove(m1, u1);
										return new Object[] { p0, p1, p2, p3, m1, m2, m3, m4 };
									}
								}
							}
							s.unmakeMove(m3, u3);
						}
					}
					s.unmakeMove(m2, u2);
				}
			}
			s.unmakeMove(m1, u1);
		}
		return null;
	}

	static void testRepetition() {
		section("pareggio per ripetizione");
		Object[] cyc = null;
		for (FastState p : gamePositions(seed + 10, 30, 2)) {
			if (p.turn == BoardConstants.WHITE_TURN && winNowMoves(p).length == 0) {
				cyc = findCycle(p);
				if (cyc != null) {
					break;
				}
			}
		}
		check("(costruzione) ciclo di 4 semi-mosse senza catture trovato", cyc != null);
		if (cyc == null) {
			return;
		}
		final FastState p0 = (FastState) cyc[0], p1 = (FastState) cyc[1], p2 = (FastState) cyc[2], p3 = (FastState) cyc[3];
		final int m1 = (Integer) cyc[4], m2 = (Integer) cyc[5], m3 = (Integer) cyc[6], m4 = (Integer) cyc[7];
		System.out.println("  ciclo: " + Move.toString(m1) + " " + Move.toString(m2) + " " + Move.toString(m3) + " " + Move.toString(m4));

		// (a) radice P3 (Nero di turno) con la storia [P0,P1,P2,P3]: la mossa m4 ricrea P0 -> pareggio esatto alla radice
		final long[] hist = { p0.zobristHash, p1.zobristHash, p2.zobristHash, p3.zobristHash };
		final String[] keys = { key(p0), key(p1), key(p2), key(p3) };
		final Mcts a = mk(constant(0), "expand=0,mm=0,cache=0", 1);
		a.setMaxSims(50);
		a.findBestMove(p3, 100000, hist, 4);
		final Mcts.Node r = a.rootNode();
		final int idx = indexOfMove(r, m4);
		check("(a) la mossa che ricrea P0 e' nel nodo", idx >= 0);
		check("(a) arco di ripetizione: pareggio esatto (0,0), q=0, nessun figlio", idx >= 0 && r.lo[idx] == 0 && r.hi[idx] == 0 && r.q[idx] == 0f && r.child[idx] == null);
		final Mcts b = mk(constant(0), "expand=0,mm=0,cache=0", 1);
		b.setMaxSims(50);
		b.findBestMove(p3, 100000, new long[] { p3.zobristHash }, 1);
		final int idxB = indexOfMove(b.rootNode(), m4);
		check("(a') senza la storia la stessa mossa e' una mossa normale (-1,1)", idxB >= 0 && b.rootNode().lo[idxB] == -1 && b.rootNode().hi[idxB] == 1);

		// (b) scelta: chi perde (valutazione costante -> per il Nero +0.9 per il Bianco) preferisce il pareggio; chi vince lo evita
		// P3 ha il Nero di turno. Valutazione costante C (prospettiva bianca): il Nero vede -C. Con C=+900 il Nero sta perdendo: il pareggio (0) vale piu' di -0.9.
		final Mcts losing = mk(constant(900), "expand=1,mm=0,cache=0", 1);
		losing.setMaxSims(3000);
		final int chosenLosing = losing.findBestMove(p3, 100000, hist, 4);
		final Mcts winning = mk(constant(-900), "expand=1,mm=0,cache=0", 1);
		winning.setMaxSims(3000);
		final int chosenWinning = winning.findBestMove(p3, 100000, hist, 4);
		final boolean noTerminals = winNowMoves(p3).length == 0 && !lossIn2(p3);
		check("(b) (costruzione) niente terminali vicini a P3", noTerminals);
		check("(b) il Nero che sta perdendo sceglie il pareggio per ripetizione", chosenLosing == m4);
		check("(b) il Nero che sta vincendo evita il pareggio per ripetizione", chosenWinning != m4);

		// (c) ripetizione DENTRO l'albero: radice P2 (Bianco), storia [P0,P1,P2]: il ramo m3 porta a P3, dove la mossa m4 ricrea P0 (in storia)
		final long[] h2 = { p0.zobristHash, p1.zobristHash, p2.zobristHash };
		final Mcts c = mk(constant(0), "expand=0,mm=0,cache=0", 1);
		c.setMaxSims(4000);
		c.findBestMove(p2, 100000, h2, 3);
		final Mcts.Node rc = c.rootNode();
		final int e3 = indexOfMove(rc, m3);
		final Mcts.Node n3 = e3 >= 0 ? rc.child[e3] : null;
		check("(c) il ramo m3 e' stato espanso", n3 != null);
		if (n3 != null) {
			final int e4 = indexOfMove(n3, m4);
			check("(c) dentro l'albero la mossa m4 e' un pareggio esatto per ripetizione", e4 >= 0 && n3.lo[e4] == 0 && n3.hi[e4] == 0 && n3.child[e4] == null);
		}
		final Walk w = new Walk(c, constant(0), true, 0);
		final String err = w.run(rc, p2, new String[] { key(p0), key(p1), key(p2) });
		check("(c) il walker concorda sulle ripetizioni dell'intero albero (" + w.repEdges + " archi di ripetizione) " + (err == null ? "" : err), err == null && w.repEdges >= 1);

		// (d) senza la storia (solo la radice) il ramo non e' una ripetizione
		final Mcts d = mk(constant(0), "expand=0,mm=0,cache=0", 1);
		d.setMaxSims(4000);
		d.findBestMove(p2, 100000, new long[] { p2.zobristHash }, 1);
		final Mcts.Node rd = d.rootNode();
		final int d3 = indexOfMove(rd, m3);
		final Mcts.Node nd3 = d3 >= 0 ? rd.child[d3] : null;
		final int d4 = nd3 == null ? -1 : indexOfMove(nd3, m4);
		check("(d) senza storia, m4 dopo m3 e' una mossa normale", d4 >= 0 && nd3.lo[d4] == -1 && nd3.hi[d4] == 1);

		// (e) storia reale delle partite (azzerata dalle catture): il walker indipendente e Mcts concordano su cosa e' una ripetizione
		final Random rnd = new Random(seed + 11);
		int bad = 0, runs = 0;
		long rep = 0;
		for (int g = 0; g < 10; g++) {
			final TestPositions.Game game = TestPositions.playGame(rnd, g % 2 == 0, 200);
			for (int i = 5; i < game.positions.size(); i += 17) {
				final List<Long> hl = game.histories.get(i);
				final int L = hl.size();
				final long[] hh = new long[L];
				final String[] kk = new String[L];
				for (int j = 0; j < L; j++) {
					hh[j] = hl.get(j);
					kk[j] = key(FastState.fromState(game.positions.get(i - (L - 1) + j)));
				}
				final FastState rootPos = FastState.fromState(game.positions.get(i));
				final Mcts m = mk(SIMPLE, "expand=0", 1);
				m.setMaxSims(500);
				m.findBestMove(rootPos, 100000, hh, L);
				runs++;
				if (m.rootNode() == null) {
					continue;
				}
				final Walk ww = new Walk(m, SIMPLE, true, 0);
				final String e = ww.run(m.rootNode(), rootPos, kk);
				rep += ww.repEdges;
				if (e != null) {
					bad++;
					System.out.println("  " + e);
				}
			}
		}
		check("(e) storia reale con azzeramenti per cattura: walker e Mcts concordano su " + runs + " alberi (" + rep + " archi di ripetizione)", bad == 0);
	}

	// ------------------------------------------------------------------------------------------ stallo risolto dalla ricerca

	static void testStalemateSolved() {
		section("stallo dell'avversario dimostrato dalla ricerca");
		// Il Nero ha una sola pedina, in un angolo con la cella (1,0) occupata da un pezzo bianco; il Bianco puo' bloccare (0,1) e
		// lasciare il Nero senza mosse: il Nero perde. NON c'e' il Re: cosi' (a differenza di una posizione con il Re, dove spostare una
		// pedina apre una via di fuga forzata) l'unico modo di vincere e' lo stallo, e il meccanismo che si vuole provare e' isolato.
		final State s = new StateTablut();
		final Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE;
		b[3][4] = Pawn.WHITE;
		b[5][4] = Pawn.WHITE;
		b[4][3] = Pawn.WHITE;
		b[4][5] = Pawn.WHITE;
		b[0][0] = Pawn.BLACK;
		b[1][0] = Pawn.WHITE;
		b[0][2] = Pawn.WHITE; // si sposta di un passo in (0,1) e chiude il Nero nell'angolo
		s.setTurn(Turn.WHITE);
		final FastState p = FastState.fromState(s);
		final int block = Move.of(BoardConstants.coord(0, 2), BoardConstants.coord(0, 1));
		check("(costruzione) il blocco e' una mossa legale", contains(legal(p), block));
		final FastState after = p.duplicate();
		after.makeMove(block, new UndoInfo());
		check("(costruzione) dopo il blocco il Nero non ha mosse e la partita non e' gia' finita", legal(after).length == 0 && after.winner == FastState.WINNER_NONE);
		check("(costruzione) nessuna vittoria in 1 per il Bianco e il Re non c'e'", winNowMoves(p).length == 0 && p.kingSquare < 0);
		final Mcts m = mk(SIMPLE, "expand=1", 1);
		final int mv = search(m, p, 20000);
		check("scelto il blocco, radice dimostrata vinta", mv == block && m.lastRootLo == 1 && m.lastRootHi == 1);
		final Mcts.Node r = m.rootNode();
		final int e = indexOfMove(r, block);
		check("il figlio del blocco e' un nodo senza mosse, sconfitta esatta per il Nero", e >= 0 && r.child[e] != null && r.child[e].nMoves == 0 && r.child[e].nodeLo == -1
				&& r.child[e].nodeHi == -1);
		final Walk w = new Walk(m, SIMPLE, true, 1);
		final String err = w.run(r, p, keysOf(p));
		check("walker: albero coerente " + (err == null ? "" : err), err == null);
	}

	// ------------------------------------------------------------------------------------------ adattatore

	static void testNegamaxSearcher() {
		section("NegamaxSearcher");
		final FastState p = gamePositions(seed + 12, 6, 5).get(2);
		final Negamax n1 = new Negamax(SIMPLE, 4).setThreads(1);
		final Negamax n2 = new Negamax(SIMPLE, 4).setThreads(1);
		final NegamaxSearcher ad = new NegamaxSearcher(n2);
		final int direct = n1.findBestMove(p, 100000, rootOnly(p), 1);
		final int viaAdapter = ad.findBestMove(p, 100000, rootOnly(p), 1);
		check("stessa mossa dell'algoritmo sottostante (ricerca a profondita' fissa, 1 thread)", direct == viaAdapter || n1.lastScore == n2.lastScore);
		check("statistiche inoltrate", ad.lastWork() == n2.lastTotalNodeCount && ad.lastDepth() == n2.lastDepthReached && ad.describeLastSearch().contains("profondita'"));
		check("nome", ad.name().equals("negamax") && ad.negamax() == n2);
		boolean threw = false;
		try {
			new NegamaxSearcher(null);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check("negamax nullo rifiutato", threw);
		ad.shutdown();
	}
}
