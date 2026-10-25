package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import it.unibo.ai.didattica.competition.tablut.aiplayer.MctsTestSupport.Walk;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

import static it.unibo.ai.didattica.competition.tablut.aiplayer.MctsTestSupport.*;

/**
 * Test dell'{@link Mcts} che riguardano il PARALLELISMO e il CICLO DI VITA: ricerca a piu' thread (le invarianti dell'albero devono valere anche
 * dopo corse tra thread), alberi minuscoli (dove i thread si contendono gli stessi archi), il riuso del sottoalbero tra una mossa e la successiva
 * (anche con storia sbagliata, e a profondita' 1-4), i limiti (nodi, tempo, simulazioni), i guasti (un evaluator che lancia eccezioni, un helper che
 * non si ferma) e partite complete contro il motore di regole indipendente {@link AshtonRules}.
 * Uso: {@code MctsConcurrencyCheck [seed]}.
 */
public class MctsConcurrencyCheck {

	static int failures = 0;
	static int checks = 0;
	static long seed = 2025;
	static final Evaluator SIMPLE = MctsSearchCheck.SIMPLE;

	/** Con {@code -Dmcts.failfast=true} il primo fallimento termina subito il test (serve al mutation test, che cerca solo il primo rilevamento). */
	static final boolean FAILFAST = Boolean.getBoolean("mcts.failfast");

	static void check(String name, boolean ok) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("  FALLITO: " + name);
			if (FAILFAST) {
				System.out.println("=== MctsConcurrencyCheck: FALLITO (failfast) ===");
				System.exit(1);
			}
		}
	}

	static long sectionStart = System.nanoTime();
	static String sectionName = "";

	static void section(String s) {
		final long now = System.nanoTime();
		if (!sectionName.isEmpty()) {
			System.out.println("    (" + (now - sectionStart) / 1_000_000_000L + " s)");
		}
		sectionStart = now;
		sectionName = s;
		System.out.println("--- " + s + " ---");
	}

	static Mcts mk(String cfg, int threads) {
		return MctsSearchCheck.mk(SIMPLE, cfg, threads);
	}

	public static void main(String[] args) throws Exception {
		if (args.length > 0) {
			seed = Long.parseLong(args[0]);
		}
		final long t0 = System.nanoTime();
		testParallelInvariants();
		testNeuralEvaluators();
		testTinyTreeStress();
		testRepeatedSearchesAndThreads();
		testReuse();
		testDeepReuse();
		testOldTreeReleased();
		testLimits();
		testErrors();
		testGames();
		section("");
		System.out.println("verifiche: " + checks + ", tempo " + (System.nanoTime() - t0) / 1_000_000_000L + " s");
		if (failures == 0) {
			System.out.println("=== MctsConcurrencyCheck: TUTTI I TEST OK ===");
		} else {
			System.out.println("=== MctsConcurrencyCheck: " + failures + " VERIFICHE FALLITE ===");
			System.exit(1);
		}
	}

	static int workerThreads() {
		int n = 0;
		for (Thread t : Thread.getAllStackTraces().keySet()) {
			if (t.getName().equals("mcts-worker") && t.isAlive()) {
				n++;
			}
		}
		return n;
	}

	// ------------------------------------------------------------------------------------------ piu' thread

	static void testParallelInvariants() {
		section("ricerca a piu' thread: le invarianti dell'albero valgono dopo le corse");
		final List<FastState> pos = new ArrayList<>(gamePositions(seed, 3, 13).subList(0, 4));
		pos.addAll(densePositions(seed + 1, 2));
		final String[] cfgs = { "expand=1", "expand=0,mm=0.5", "expand=2,vloss=3,tau=0.5" };
		int runs = 0, bad = 0, illegal = 0, errors = 0, visitsBad = 0;
		long sims = 0, nodes = 0, means = 0;
		for (int threads : new int[] { 2, 4, 8 }) {
			for (String cfg : cfgs) {
				final int expand = MctsConfig.parse(cfg).expandAfter;
				for (FastState p : pos) {
					if (legal(p).length < 2) {
						continue;
					}
					final Mcts m = mk(cfg, threads);
					final int mv = m.findBestMove(p, 90, rootOnly(p), 1);
					runs++;
					sims += m.lastSims;
					if (m.lastError != null) {
						errors++;
						System.out.println("  errore: " + m.lastError);
					}
					if (!contains(legal(p), mv)) {
						illegal++;
					}
					final Mcts.Node r = m.rootNode();
					if (r == null) {
						bad++;
						continue;
					}
					final Walk w = new Walk(m, SIMPLE, false, expand);
					final String err = w.run(r, p, keysOf(p));
					nodes += w.nodes;
					means += w.meansChecked;
					if (err != null) {
						bad++;
						if (bad <= 5) {
							System.out.println("  [" + threads + " thread, " + cfg + "] " + err);
						}
					}
					if (!r.solved() && r.totalVisits != m.lastSims) {
						visitsBad++;
					}
					if (Mcts.countNodes(r) != m.nodeCountForTest()) {
						bad++;
					}
					m.shutdown();
				}
			}
		}
		check("walker: 0 discordanze su " + runs + " ricerche parallele (" + bad + ")", bad == 0);
		check("nessuna eccezione nei thread, tutte le mosse legali (" + errors + " errori, " + illegal + " illegali)", errors == 0 && illegal == 0);
		check("visite della radice = simulazioni completate, a radice non risolta (" + visitsBad + " discordanze)", visitsBad == 0);
		check("non vacuo: " + sims + " simulazioni, " + nodes + " nodi, " + means + " medie ricostruite", sims > 20000 && nodes > 5000 && means > 1000);
	}


	// ------------------------------------------------------------------------------------------ reti neurali condivise

	static void testNeuralEvaluators() throws Exception {
		section("evaluator neurali (MLP e CNN) condivisi tra i thread");
		final String[] files = { "pretrained_weights/mlp_v3_pretrained.txt", "pretrained_weights/cnn_conv4_hybrid_pretrained.txt" };
		final List<FastState> pos = new ArrayList<>(gamePositions(seed + 7, 6, 11));
		int tested = 0;
		for (final String f : files) {
			final Evaluator ev;
			try {
				ev = EvaluatorLoader.load(f);
			} catch (Throwable t) {
				System.out.println("  (" + f + " non trovato dalla cartella corrente: verifica saltata)");
				continue;
			}
			tested++;
			final String nm = f.substring(f.lastIndexOf('/') + 1);
			// 1) l'evaluator e' thread-safe: 4 thread danno gli stessi valori della chiamata sequenziale
			final int n = Math.min(40, pos.size());
			final double[] ref = new double[n];
			for (int i = 0; i < n; i++) {
				ref[i] = ev.evaluate(pos.get(i));
			}
			final AtomicInteger wrong = new AtomicInteger();
			final Thread[] ts = new Thread[4];
			for (int t = 0; t < ts.length; t++) {
				final int tt = t;
				ts[t] = new Thread() {
					@Override
					public void run() {
						final Random r = new Random(5 + tt);
						for (int k = 0; k < 4000; k++) {
							final int i = r.nextInt(n);
							if (ev.evaluate(pos.get(i)) != ref[i]) {
								wrong.incrementAndGet();
							}
						}
					}
				};
				ts[t].start();
			}
			for (Thread t : ts) {
				t.join();
			}
			check(nm + ": 16000 valutazioni concorrenti identiche a quelle sequenziali (" + wrong.get() + " diverse)", wrong.get() == 0);
			// 2) ricerca a 4 thread con la rete: nessun errore, mosse legali, albero coerente
			int bad = 0;
			long sims = 0;
			for (int i = 0; i < 3 && i < pos.size(); i++) {
				final FastState p = pos.get(i);
				if (legal(p).length < 2) {
					continue;
				}
				final Mcts m = new Mcts(ev, MctsConfig.parse("expand=1"), 4).setSafetyMarginMillis(0);
				final int mv = m.findBestMove(p, 250, rootOnly(p), 1);
				sims += m.lastSims;
				if (m.lastError != null || !contains(legal(p), mv)) {
					bad++;
				}
				final String err = m.rootNode() == null ? "radice assente" : new Walk(m, ev, false, 1).run(m.rootNode(), p, keysOf(p));
				if (err != null) {
					bad++;
					System.out.println("  " + err);
				}
				m.shutdown();
			}
			check(nm + ": ricerca a 4 thread con la rete, alberi coerenti (" + bad + " problemi, " + sims + " simulazioni)", bad == 0 && sims > 100);
		}
		check("almeno una rete verificata (i pesi sono in pretrained_weights/)", tested >= 1);
	}

	// ------------------------------------------------------------------------------------------ alberi minuscoli

	/** Evaluator lento (~150 us): le espansioni durano abbastanza perche' gli altri thread trovino l'arco marcato "in espansione". */
	static final Evaluator SLOW = new Evaluator() {
		@Override
		public double evaluate(FastState s) {
			java.util.concurrent.locks.LockSupport.parkNanos(150_000);
			return SIMPLE.evaluate(s);
		}
	};

	static void testTinyTreeStress() {
		section("alberi minuscoli con 16 thread (contesa sugli stessi archi)");
		final List<FastState> tiny = new ArrayList<>();
		for (FastState p : densePositions(seed + 2, 6000)) {
			final int k = legal(p).length;
			if (k >= 2 && k <= 4) {
				tiny.add(p);
			}
			if (tiny.size() >= 14) {
				break;
			}
		}
		check("(costruzione) trovate posizioni con 2-4 mosse legali (" + tiny.size() + ")", tiny.size() >= 8);
		long aborts = 0, sims = 0;
		int bad = 0, errors = 0, illegal = 0;
		for (FastState p : tiny) {
			final Mcts m = MctsSearchCheck.mk(SLOW, "expand=0", 16);
			final int mv = m.findBestMove(p, 250, rootOnly(p), 1);
			aborts += m.lastAborts;
			sims += m.lastSims;
			if (m.lastError != null) {
				errors++;
				System.out.println("  errore: " + m.lastError);
			}
			if (!contains(legal(p), mv)) {
				illegal++;
			}
			if (m.rootNode() != null) {
				final String err = new Walk(m, SIMPLE, false, 0).run(m.rootNode(), p, keysOf(p));
				if (err != null) {
					bad++;
					System.out.println("  " + err);
				}
			}
			m.shutdown();
		}
		check("invarianti dopo la contesa (" + bad + " discordanze, " + errors + " errori, " + illegal + " mosse illegali)", bad == 0 && errors == 0 && illegal == 0);
		// il tetto di simulazioni conta quelle COMPLETATE: le annullate (nessun arco selezionabile) non lo consumano
		int exact = 0, withAborts = 0, tried = 0;
		for (FastState p : densePositions(seed + 4, 3000)) {
			final int k = legal(p).length;
			if (k < 5 || k > 8) {
				continue;
			}
			final Mcts m = MctsSearchCheck.mk(SLOW, "expand=0,reuse=false,escape=false", 8);
			m.setMaxSims(40);
			m.findBestMove(p, 600000, rootOnly(p), 1);
			if (m.rootNode() == null || m.rootNode().solved()) {
				m.shutdown();
				continue;
			}
			tried++;
			if (m.lastSims == 40) {
				exact++;
			} else {
				System.out.println("  simulazioni completate " + m.lastSims + " invece di 40 (annullate " + m.lastAborts + ")");
			}
			if (m.lastAborts > 0) {
				withAborts++;
			}
			m.shutdown();
			if (tried >= 8) {
				break;
			}
		}
		check("maxSims con 8 thread: esattamente 40 simulazioni completate anche con annullamenti (" + exact + "/" + tried + ", " + withAborts + " con annullamenti)", tried >= 5 && exact == tried
				&& withAborts >= 2);
		check("non vacuo: il percorso 'nessun arco selezionabile' e' stato esercitato (" + aborts + " simulazioni annullate su " + sims + ")", aborts > 0 && sims > 200);
	}

	// ------------------------------------------------------------------------------------------ ricerche ripetute, thread

	static void testRepeatedSearchesAndThreads() throws Exception {
		section("ricerche ripetute sulla stessa istanza (4 thread), nessun thread di troppo");
		final Mcts m = mk("expand=1", 4);
		int bad = 0, plies = 0, reused = 0, maxWorkers = 0;
		for (int game = 0; game < 4; game++) {
			final Random rnd = new Random(seed + 3 + game);
			final GameTracker g = new GameTracker(TestPositions.initial());
			m.clearTree();
			for (int i = 0; i < 24 && !g.over(); i++) {
				final int mv = m.findBestMove(g.cur, 80, g.hist(), g.hist().length);
				plies++;
				if (m.lastReused) {
					reused++;
				}
				maxWorkers = Math.max(maxWorkers, workerThreads());
				if (m.lastError != null || !contains(legal(g.cur), mv)) {
					bad++;
				}
				final String err = m.rootNode() == null ? null : new Walk(m, SIMPLE, false, 1).run(m.rootNode(), g.cur, g.keys());
				if (err != null) {
					bad++;
					System.out.println("  " + err);
				}
				g.apply(mv);
				if (g.over()) {
					break;
				}
				final int[] replies = legal(g.cur);
				if (replies.length == 0) {
					break;
				}
				g.apply(replies[rnd.nextInt(replies.length)]);
			}
		}
		check("ricerche consecutive sulla stessa istanza: nessun errore, mosse legali, alberi coerenti (" + bad + " problemi, " + reused + " riusi su " + plies + ")", bad == 0 && plies >= 15
				&& reused >= 5);
		check("al piu' 3 thread di lavoro (4 thread totali, il chiamante e' uno) invece di " + maxWorkers, maxWorkers <= 3);
		m.shutdown();
		long end = System.currentTimeMillis() + 3000;
		while (workerThreads() > 0 && System.currentTimeMillis() < end) {
			Thread.sleep(20);
		}
		check("dopo shutdown() nessun thread di lavoro vivo (" + workerThreads() + ")", workerThreads() == 0);
	}

	// ------------------------------------------------------------------------------------------ riuso

	/** La partita come la vede un client: posizione corrente e storia dall'ultima cattura, con le regole scritte indipendentemente da Mcts (conteggio pezzi). */
	static final class GameTracker {
		FastState cur;
		final List<FastState> seg = new ArrayList<>();

		GameTracker(State s) {
			cur = FastState.fromState(s);
			seg.add(cur.duplicate());
		}

		GameTracker(FastState s) {
			cur = s.duplicate();
			seg.add(cur.duplicate());
		}

		/** @return vero se la mossa ha catturato (la storia di ripetizione si azzera) */
		boolean apply(int move) {
			final int before = pieces(cur);
			cur.makeMove(move, new UndoInfo());
			final boolean captured = pieces(cur) != before;
			if (captured) {
				seg.clear();
			}
			seg.add(cur.duplicate());
			return captured;
		}

		boolean over() {
			return cur.winner != FastState.WINNER_NONE;
		}

		long[] hist() {
			final long[] h = new long[seg.size()];
			for (int i = 0; i < h.length; i++) {
				h[i] = seg.get(i).zobristHash;
			}
			return h;
		}

		String[] keys() {
			final String[] k = new String[seg.size()];
			for (int i = 0; i < k.length; i++) {
				k[i] = key(seg.get(i));
			}
			return k;
		}
	}

	/**
	 * Gioca una partita (noi con Mcts a 1 thread e simulazioni fisse, l'avversario a caso) e a ogni mossa verifica la PREVISIONE di riuso: la nuova radice
	 * dovrebbe essere riusata se e solo se il sottoalbero della nostra mossa e della risposta avversaria era stato espanso.
	 *
	 * @param corruptOdd a un tentativo su tre (il terzo, il sesto...) passa una storia con una voce in piu' in testa: il riuso deve essere rifiutato; e poiche'
	 *            l'albero conservato dopo quella ricerca presuppone la storia alterata, anche la ricerca SUCCESSIVA (corretta) non puo' riusarlo
	 */
	static int[] reuseGame(long gameSeed, boolean reuseOn, boolean corruptOdd) {
		final Random rnd = new Random(gameSeed);
		final Mcts m = mk(reuseOn ? "expand=1" : "expand=1,reuse=false", 1);
		m.setMaxSims(500);
		final GameTracker g = new GameTracker(TestPositions.initial());
		Mcts.Node expected = null;
		long preVisits = 0, preNodes = 0;
		boolean prevCorrupt = false;
		boolean capturedSince = false;
		int searches = 0, predicted = 0, reusedCount = 0, mismatches = 0, walkerBad = 0, badIdentity = 0;
		for (int i = 0; i < 40 && !g.over(); i++) {
			final boolean corrupt = corruptOdd && (i % 3 == 2);
			long[] h = g.hist();
			String[] keys = g.keys();
			if (corrupt) {
				final long[] h2 = new long[h.length + 1];
				final String[] k2 = new String[h.length + 1];
				h2[0] = 0x1234567L;
				k2[0] = "bogus";
				System.arraycopy(h, 0, h2, 1, h.length);
				System.arraycopy(keys, 0, k2, 1, keys.length);
				h = h2;
				keys = k2;
			}
			final int mv = m.findBestMove(g.cur, 600000, h, h.length);
			searches++;
			final boolean prevCorruptBefore = prevCorrupt;
			// dopo una ricerca con storia alterata l'albero conservato presuppone quella storia: il riuso e' valido solo se nel frattempo c'e' stata una
			// cattura (la memoria delle posizioni si azzera, e il prefisso alterato sta PRIMA della cattura: non influenza piu' nulla)
			final boolean shouldReuse = reuseOn && expected != null && !corrupt && (!prevCorrupt || capturedSince);
			prevCorrupt = corrupt;
			if (m.lastReused != shouldReuse) {
				mismatches++;
				if (DEBUG) {
					System.out.println("  [riuso] partita " + gameSeed + " ricerca " + i + ": corrupt=" + corrupt + " prevCorrupt=" + prevCorruptBefore + " previsto=" + shouldReuse
							+ " riusato=" + m.lastReused + " atteso-nodo=" + (expected != null));
				}
			}
			if (shouldReuse && m.lastReused) {
				reusedCount++;
				if (m.lastReusedVisits != preVisits || m.lastReusedNodes != preNodes || m.rootNode() != expected) {
					badIdentity++;
				}
			}
			if (m.lastError != null || !contains(legal(g.cur), mv)) {
				mismatches++;
			}
			if (m.rootNode() != null) {
				final String err = new Walk(m, SIMPLE, true, 1).run(m.rootNode(), g.cur, keys);
				if (err != null) {
					walkerBad++;
					System.out.println("  " + err);
				}
			}
			// la mossa, poi la risposta avversaria; si calcola il nodo che dovrebbe essere riusato
			final Mcts.Node r0 = m.rootNode();
			final int e = indexOfMove(r0, mv);
			final Mcts.Node c = e >= 0 ? r0.child[e] : null;
			capturedSince = g.apply(mv);
			expected = null;
			if (g.over()) {
				break;
			}
			final int[] replies = legal(g.cur);
			if (replies.length == 0) {
				break;
			}
			final int reply = replies[rnd.nextInt(replies.length)];
			if (c != null) {
				final int j = indexOfMove(c, reply);
				if (j >= 0 && c.child[j] != null) {
					expected = c.child[j];
					preVisits = expected.totalVisits;
					preNodes = Mcts.countNodes(expected);
					predicted++;
				}
			}
			capturedSince |= g.apply(reply);
		}
		m.shutdown();
		return new int[] { searches, predicted, reusedCount, mismatches, walkerBad, badIdentity };
	}

	static void testReuse() {
		section("riuso del sottoalbero tra una mossa e la successiva");
		int searches = 0, predicted = 0, reused = 0, mism = 0, wbad = 0, ident = 0;
		for (int g = 0; g < 16; g++) {
			final int[] r = reuseGame(seed + 100 + g, true, false);
			searches += r[0];
			predicted += r[1];
			reused += r[2];
			mism += r[3];
			wbad += r[4];
			ident += r[5];
		}
		check("riuso: il riuso avviene se e solo se la risposta era nell'albero (" + mism + " previsioni sbagliate su " + searches + " ricerche)", mism == 0);
		check("riuso: la radice riusata e' proprio quel nodo, con le sue visite e i suoi nodi intatti (" + ident + " anomalie)", ident == 0);
		check("riuso: alberi coerenti anche dopo il riuso (" + wbad + " discordanze)", wbad == 0);
		check("non vacuo: " + reused + " riusi su " + predicted + " previsti", reused >= 25 && reused == predicted);

		int s2 = 0, p2 = 0, r2 = 0, m2 = 0, w2 = 0;
		for (int g = 0; g < 12; g++) {
			final int[] r = reuseGame(seed + 200 + g, true, true);
			s2 += r[0];
			p2 += r[1];
			r2 += r[2];
			m2 += r[3];
			w2 += r[4];
		}
		check("storia sbagliata: il riuso e' rifiutato con la storia alterata (e dopo), rispettato altrimenti (" + m2 + " previsioni sbagliate su " + s2 + ", " + r2 + " riusi)", m2 == 0
				&& r2 > 0);
		check("storia sbagliata: alberi coerenti con la storia effettivamente usata (" + w2 + ")", w2 == 0);

		int s3 = 0, rr3 = 0;
		for (int g = 0; g < 3; g++) {
			final int[] r = reuseGame(seed + 300 + g, false, false);
			s3 += r[0];
			rr3 += r[2];
			check("reuse=false: mai riuso (" + r[3] + " discordanze)", r[3] == 0);
		}
		check("reuse=false: nessun riuso su " + s3 + " ricerche", rr3 == 0);

		// la stessa posizione cercata due volte di fila: la seconda riparte dalla prima
		final FastState p = gamePositions(seed + 4, 6, 7).get(4);
		final Mcts m = mk("expand=1", 1);
		m.setMaxSims(300);
		m.findBestMove(p, 600000, rootOnly(p), 1);
		final long v1 = m.rootNode().totalVisits;
		final Mcts.Node first = m.rootNode();
		m.findBestMove(p, 600000, rootOnly(p), 1);
		check("stessa posizione due volte: riusata la radice (" + m.lastReusedVisits + " visite) e le visite si sommano", m.lastReused && m.rootNode() == first && m.lastReusedVisits == v1
				&& m.rootNode().totalVisits == v1 + m.lastSims);
		// stessa posizione ma storia DIVERSA (una voce in piu' in testa): i pareggi per ripetizione dell'albero sarebbero calcolati con un'altra storia
		m.findBestMove(p, 600000, new long[] { 0x7777L, p.zobristHash }, 2);
		check("stessa posizione con storia diversa: il riuso e' rifiutato", !m.lastReused && m.lastReusedNodes == 0);
		m.findBestMove(p, 600000, new long[] { 0x7777L, p.zobristHash }, 2);
		check("... e la volta dopo, con quella stessa storia, l'albero nuovo e' riusabile", m.lastReused);
		final FastState other = gamePositions(seed + 5, 6, 7).get(5);
		m.findBestMove(other, 600000, rootOnly(other), 1);
		check("posizione diversa non raggiungibile: albero nuovo", !m.lastReused && m.lastReusedNodes == 0);
		m.clearTree();
		check("clearTree: albero eliminato", m.rootNode() == null);
	}

	static void testDeepReuse() {
		section("riuso a profondita' 1, 2, 3 e 4 (con le catture che azzerano la storia)");
		int tested = 0, ok = 0;
		final List<FastState> pos = new ArrayList<>(gamePositions(seed + 6, 12, 9));
		int used = 0;
		for (FastState p : pos) {
			if (used >= 9) {
				break;
			}
			if (legal(p).length < 3) {
				continue;
			}
			used++;
			for (int depth = 1; depth <= 4; depth++) {
				final Mcts m = mk("expand=0", 1);
				m.setMaxSims(1500);
				final GameTracker g = new GameTracker(p);
				m.findBestMove(p, 600000, g.hist(), 1);
				// si segue nell'albero l'arco piu' visitato per `depth` semi-mosse
				Mcts.Node n = m.rootNode();
				final int[] path = new int[depth];
				Mcts.Node target = null;
				boolean possible = true;
				for (int d = 0; d < depth && possible; d++) {
					int best = -1;
					for (int e = 0; e < n.nMoves; e++) {
						if (n.child[e] != null && (best < 0 || n.visits[e] > n.visits[best])) {
							best = e;
						}
					}
					if (best < 0 || n.child[best] == null) {
						possible = false;
						break;
					}
					path[d] = n.moves[best];
					n = n.child[best];
					target = n;
				}
				if (!possible || target == null || g.over()) {
					continue;
				}
				final long pre = target.totalVisits;
				final long preNodes = Mcts.countNodes(target);
				for (int mv : path) {
					if (g.over()) {
						break;
					}
					g.apply(mv);
				}
				if (g.over() || legal(g.cur).length < 1) {
					continue;
				}
				tested++;
				m.setMaxSims(50);
				m.findBestMove(g.cur, 600000, g.hist(), g.hist().length);
				if (m.lastReused && m.rootNode() == target && m.lastReusedVisits == pre && m.lastReusedNodes == preNodes) {
					ok++;
				} else {
					System.out.println("  profondita' " + depth + ": riuso=" + m.lastReused + " identico=" + (m.rootNode() == target));
				}
				final String err = new Walk(m, SIMPLE, true, 0).run(m.rootNode(), g.cur, g.keys());
				if (err != null) {
					check("walker dopo il riuso profondo: " + err, false);
				}
			}
		}
		check("riuso a tutte le profondita' da 1 a 4: " + ok + "/" + tested, tested >= 20 && ok == tested);
	}

	// ------------------------------------------------------------------------------------------ limiti

	static void testLimits() {
		section("limiti: nodi, tempo, simulazioni, profondita'");
		final FastState start = FastState.fromState(TestPositions.initial());

		// tetto ai nodi
		final Mcts c1 = mk("expand=0,maxnodes=60,reuse=false", 1);
		final int mv1 = c1.findBestMove(start, 60000, rootOnly(start), 1);
		check("tetto 60 nodi (1 thread): raggiunto esattamente, ricerca fermata, mossa legale", c1.lastMemoryCapHit && c1.nodeCountForTest() == 60 && contains(legal(start), mv1));
		check("tetto: albero coerente", new Walk(c1, SIMPLE, false, 0).run(c1.rootNode(), start, keysOf(start)) == null);
		check("tetto: la ricerca e' terminata presto (" + c1.lastElapsedMillis + " ms)", c1.lastElapsedMillis < 3000);
		final Mcts c4 = mk("expand=0,maxnodes=200,reuse=false", 4);
		final int mv4 = c4.findBestMove(start, 60000, rootOnly(start), 1);
		check("tetto 200 nodi (4 thread): superato al piu' del numero dei thread (" + c4.nodeCountForTest() + ")", c4.lastMemoryCapHit && c4.nodeCountForTest() >= 200
				&& c4.nodeCountForTest() <= 200 + 4 && contains(legal(start), mv4));
		check("tetto (4 thread): albero coerente", new Walk(c4, SIMPLE, false, 0).run(c4.rootNode(), start, keysOf(start)) == null);
		c4.shutdown();

		// profondita' massima
		final Mcts d3 = mk("expand=0,maxdepth=3,reuse=false", 1);
		d3.setMaxSims(3000);
		d3.findBestMove(start, 600000, rootOnly(start), 1);
		final Walk wd = new Walk(d3, SIMPLE, false, 0);
		final String ed = wd.run(d3.rootNode(), start, keysOf(start));
		check("maxdepth=3: nessun nodo oltre la profondita' 2, albero coerente (" + wd.maxDepth + ") " + (ed == null ? "" : ed), ed == null && wd.maxDepth == 2);

		// numero esatto di simulazioni
		boolean exact = true;
		for (long n : new long[] { 1, 2, 17, 500 }) {
			final Mcts m = mk("expand=1,reuse=false", 1);
			m.setMaxSims(n);
			m.findBestMove(start, 600000, rootOnly(start), 1);
			if (m.lastSims != n || m.rootNode().totalVisits != n) {
				exact = false;
				System.out.println("  n=" + n + " lastSims=" + m.lastSims + " visite=" + m.rootNode().totalVisits);
			}
		}
		check("maxSims: esattamente N simulazioni con 1 thread", exact);

		// tempo
		// il caso {1000, 700} distingue un margine non sottratto (1000 ms invece di 300) anche con la tolleranza di +300 ms per il carico della macchina
		final long[][] cases = { { 300, 0 }, { 1000, 700 }, { 0, 0 } };
		for (int threads : new int[] { 1, 4 }) {
			for (long[] cs : cases) {
				final Mcts m = mk("expand=1,reuse=false", threads);
				m.setSafetyMarginMillis(cs[1]);
				final long t0 = System.nanoTime();
				final int mv = m.findBestMove(start, cs[0], rootOnly(start), 1);
				final long ms = (System.nanoTime() - t0) / 1_000_000L;
				final long want = Math.max(10, cs[0] - cs[1]);
				check("tempo: budget " + cs[0] + " margine " + cs[1] + " (" + threads + " thread): " + ms + " ms, atteso ~" + want, ms >= want - 5 && ms <= want + 300
						&& contains(legal(start), mv));
				m.shutdown();
			}
		}
	}


	// ------------------------------------------------------------------------------------------ memoria: il vecchio albero

	/**
	 * Evaluator-sonda per verificare la memoria. Quando e' armato: alla PRIMA chiamata (dentro la prima simulazione della ricerca) controlla che ogni nodo
	 * ancora presente nell'array del percorso del worker 0 appartenga al nuovo albero; alla N-esima forza il garbage collector e registra se un nodo (tenuto
	 * solo con un riferimento debole) e' stato raccolto.
	 */
	static final class GcProbe implements Evaluator {
		volatile boolean armed;
		volatile java.lang.ref.WeakReference<Object> target;
		volatile Mcts mcts;
		volatile Mcts.Node newRoot;
		volatile int triggerAt = 200;
		final AtomicInteger calls = new AtomicInteger();
		volatile Boolean collected;
		volatile int staleEntries = -1;

		@Override
		public double evaluate(FastState s) {
			if (armed) {
				final int n = calls.incrementAndGet();
				if (n == 1) {
					final java.util.IdentityHashMap<Mcts.Node, Boolean> inNew = new java.util.IdentityHashMap<>();
					final java.util.ArrayDeque<Mcts.Node> stack = new java.util.ArrayDeque<>();
					stack.push(newRoot);
					while (!stack.isEmpty()) {
						final Mcts.Node x = stack.pop();
						inNew.put(x, Boolean.TRUE);
						for (int i = 0; i < x.nMoves; i++) {
							final Mcts.Node ch = x.child[i];
							if (ch != null && ch != Mcts.EXPANDING) {
								stack.push(ch);
							}
						}
					}
					int stale = 0;
					for (Mcts.Node x : mcts.worker(0).pNode) {
						if (x != null && !inNew.containsKey(x)) {
							stale++;
						}
					}
					staleEntries = stale;
				}
				if (n == triggerAt && collected == null) {
					for (int i = 0; i < 5; i++) {
						System.gc();
					}
					collected = target.get() == null;
				}
			}
			return SIMPLE.evaluate(s);
		}
	}

	static void testOldTreeReleased() {
		section("memoria: il vecchio albero non resta raggiungibile durante la ricerca successiva");
		int tested = 0, released = 0, clean = 0, retainedSubtree = 0;
		for (FastState p : gamePositions(seed + 40, 20, 5)) {
			if (tested >= 5) {
				break;
			}
			if (legal(p).length < 3) {
				continue;
			}
			final GcProbe probe = new GcProbe();
			final Mcts m = new Mcts(probe, MctsConfig.parse("expand=0,cache=0"), 1).setSafetyMarginMillis(0);
			m.setMaxSims(1500);
			final GameTracker g = new GameTracker(p);
			m.findBestMove(p, 600000, g.hist(), 1);
			Mcts.Node oldRootNode = m.rootNode();
			// La nostra mossa e' scelta DIVERSA dal primo arco dell'ultimo percorso simulato (cosi' i nodi che l'array del percorso ricorda sono nell'albero che
			// si scarta, non nel sottoalbero riusato) e con un figlio espanso, la cui risposta piu' visitata e' anch'essa espansa.
			final int lastFirst = m.worker(0).pEdge[0];
			int e = -1;
			for (int i = 0; i < oldRootNode.nMoves; i++) {
				final Mcts.Node c = oldRootNode.child[i];
				if (i == lastFirst || c == null) {
					continue;
				}
				boolean grand = false;
				for (int j = 0; j < c.nMoves; j++) {
					if (c.child[j] != null) {
						grand = true;
					}
				}
				if (grand && (e < 0 || oldRootNode.visits[i] > oldRootNode.visits[e])) {
					e = i;
				}
			}
			if (e < 0) {
				continue;
			}
			Mcts.Node c = oldRootNode.child[e];
			int j = -1;
			for (int k = 0; k < c.nMoves; k++) {
				if (c.child[k] != null && (j < 0 || c.visits[k] > c.visits[j])) {
					j = k;
				}
			}
			final Mcts.Node target = c.child[j];
			final java.lang.ref.WeakReference<Object> weak = new java.lang.ref.WeakReference<Object>(oldRootNode);
			final int moveA = oldRootNode.moves[e], moveB = c.moves[j];
			// il test NON deve tenere riferimenti forti al vecchio albero durante la ricerca successiva (un riferimento locale vivo lo terrebbe in vita)
			oldRootNode = null;
			c = null;
			g.apply(moveA);
			if (g.over()) {
				continue;
			}
			g.apply(moveB);
			probe.target = weak;
			probe.mcts = m;
			probe.newRoot = target;
			probe.armed = true;
			m.setMaxSims(1500);
			m.findBestMove(g.cur, 600000, g.hist(), g.hist().length);
			tested++;
			if (m.lastReused && m.lastReusedVisits > 0) {
				retainedSubtree++;
			}
			if (Boolean.TRUE.equals(probe.collected)) {
				released++;
			} else {
				System.out.println("  il vecchio albero e' ancora raggiungibile durante la ricerca (collected=" + probe.collected + ", riuso=" + m.lastReused + ")");
			}
			if (probe.staleEntries == 0) {
				clean++;
			} else {
				System.out.println("  l'array del percorso del worker ricorda " + probe.staleEntries + " nodi fuori dal nuovo albero");
			}
		}
		check("il vecchio albero e' stato raccolto durante la ricerca successiva (" + released + "/" + tested + ", con riuso del sottoalbero " + retainedSubtree + ")", tested >= 4
				&& released == tested && retainedSubtree == tested);
		check("l'array del percorso del worker non ricorda nodi del vecchio albero a inizio ricerca (" + clean + "/" + tested + ")", clean == tested);
	}

	// ------------------------------------------------------------------------------------------ guasti

	/** Evaluator che, quando e' armato, lancia un'eccezione dopo un certo numero di chiamate (solo nei thread indicati). */
	static final class Faulty implements Evaluator {
		volatile boolean armed;
		volatile boolean onlyHelpers;
		volatile int afterCalls;
		final AtomicInteger calls = new AtomicInteger();
		volatile CountDownLatch blockHelpers;

		@Override
		public double evaluate(FastState s) {
			if (armed) {
				final boolean helper = Thread.currentThread().getName().equals("mcts-worker");
				if (helper && blockHelpers != null) {
					try {
						blockHelpers.await();
					} catch (InterruptedException e) {
						throw new RuntimeException("interrotto");
					}
				}
				if ((!onlyHelpers || helper) && calls.incrementAndGet() > afterCalls && blockHelpers == null) {
					throw new IllegalStateException("guasto simulato");
				}
			}
			return SIMPLE.evaluate(s);
		}
	}

	static void testErrors() throws Exception {
		section("guasti: evaluator che lancia eccezioni, helper che non si ferma");
		final FastState start = FastState.fromState(TestPositions.initial());

		// l'eccezione alla prima valutazione della radice: si risponde comunque con una mossa legale
		final Faulty f = new Faulty();
		final Mcts m = new Mcts(f, MctsConfig.parse("expand=1"), 1).setSafetyMarginMillis(0);
		f.armed = true;
		f.afterCalls = 0;
		final int mv = m.findBestMove(start, 1000, rootOnly(start), 1);
		check("guasto nell'espansione della radice: mossa legale restituita, errore registrato, nessun albero", contains(legal(start), mv) && m.lastError != null && m.rootNode() == null);

		// l'istanza non resta avvelenata
		f.armed = false;
		m.setMaxSims(200);
		final int mv2 = m.findBestMove(start, 1000, rootOnly(start), 1);
		check("dopo il guasto l'istanza funziona di nuovo", contains(legal(start), mv2) && m.lastError == null && m.rootNode() != null && m.lastSims == 200);

		// guasto durante la ricerca (1 thread)
		f.calls.set(0);
		f.armed = true;
		f.afterCalls = 3000;
		m.clearTree();
		m.setMaxSims(0);
		final int mv3 = m.findBestMove(start, 2000, rootOnly(start), 1);
		check("guasto a meta' ricerca (1 thread): mossa legale, errore registrato, albero scartato", contains(legal(start), mv3) && m.lastError != null && m.rootNode() == null);
		f.armed = false;
		m.setMaxSims(100);
		check("... e poi di nuovo utilizzabile", contains(legal(start), m.findBestMove(start, 1000, rootOnly(start), 1)) && m.lastError == null);
		m.shutdown();

		// guasto in un helper (4 thread): la ricerca del thread principale non fallisce
		final Faulty f4 = new Faulty();
		final Mcts m4 = new Mcts(f4, MctsConfig.parse("expand=1"), 4).setSafetyMarginMillis(0);
		f4.armed = true;
		f4.onlyHelpers = true;
		f4.afterCalls = 2000;
		final int mv4 = m4.findBestMove(start, 600, rootOnly(start), 1);
		check("guasto in un helper (4 thread): mossa legale, errore registrato, albero scartato", contains(legal(start), mv4) && m4.lastError != null && m4.rootNode() == null);
		f4.armed = false;
		m4.setMaxSims(0);
		final int mv5 = m4.findBestMove(start, 300, rootOnly(start), 1);
		check("... e il pool e' di nuovo utilizzabile (" + m4.lastSims + " simulazioni)", contains(legal(start), mv5) && m4.lastError == null && m4.lastSims > 100);
		m4.shutdown();

		// helper che non termina: dopo l'attesa massima si abbandona tutto, senza restare appesi
		final Faulty fb = new Faulty();
		final Mcts mb = new Mcts(fb, MctsConfig.parse("expand=1"), 2).setSafetyMarginMillis(0);
		fb.armed = true;
		fb.onlyHelpers = true;
		final CountDownLatch latch = new CountDownLatch(1);
		fb.blockHelpers = latch;
		final long t0 = System.nanoTime();
		final int mvb = mb.findBestMove(start, 200, rootOnly(start), 1);
		final long ms = (System.nanoTime() - t0) / 1_000_000L;
		check("helper bloccato: si ritorna dopo l'attesa massima (" + ms + " ms) con mossa legale, errore, albero scartato", contains(legal(start), mvb) && mb.lastError != null
				&& mb.rootNode() == null && ms >= 1500 && ms < 6000);
		latch.countDown();
		fb.armed = false;
		fb.blockHelpers = null;
		mb.setMaxSims(100);
		check("... e l'istanza e' di nuovo utilizzabile con un pool nuovo", contains(legal(start), mb.findBestMove(start, 1000, rootOnly(start), 1)) && mb.lastError == null);
		mb.shutdown();
	}

	// ------------------------------------------------------------------------------------------ partite complete

	static final int PLY_CAP = 200;
	static final boolean DEBUG = Boolean.getBoolean("mcts.debug");

	/**
	 * Una partita con l'arbitro INDIPENDENTE ({@link AshtonRules} su {@code State}): Mcts gioca il colore {@code mctsWhite}, l'altro lato e' a caso (o un
	 * secondo Mcts). Ogni mossa di Mcts deve essere legale per l'arbitro. Ripetizione = pareggio (come il server), tetto di semi-mosse = pareggio.
	 *
	 * @return {esito (+1 vince il primo Mcts, -1 vince l'altro, 0 pareggio), mosse illegali, vittorie dimostrate, vittorie dimostrate poi non convertite}
	 */
	static int[] playRefereed(long gameSeed, boolean mctsWhite, boolean opponentIsMcts) {
		final Random rnd = new Random(gameSeed);
		final Mcts a = mk("expand=1", 1);
		final Mcts b = opponentIsMcts ? mk("expand=1,cpuct=1.5,mm=0.5", 1) : null;
		a.setMaxSims(250);
		if (b != null) {
			b.setMaxSims(250);
		}
		State s = TestPositions.initial();
		final List<Long> hist = new ArrayList<>();
		hist.add(ZobristHasher.fullHash(s));
		int illegal = 0, provenWins = 0;
		boolean provenAndNotWon = false;
		int outcome = 0;
		for (int ply = 0; ply < PLY_CAP; ply++) {
			if (AshtonRules.isTerminal(s)) {
				break;
			}
			final List<Action> moves = AshtonRules.generateMoves(s);
			if (moves.isEmpty()) { // chi non puo' muovere perde
				outcome = (s.getTurn() == Turn.WHITE) == mctsWhite ? -1 : 1;
				break;
			}
			final boolean whiteToMove = s.getTurn() == Turn.WHITE;
			final boolean mctsMoves = whiteToMove == mctsWhite;
			Action chosen;
			if (mctsMoves || b != null) {
				final Mcts who = mctsMoves ? a : b;
				final long[] h = new long[hist.size()];
				for (int i = 0; i < h.length; i++) {
					h[i] = hist.get(i);
				}
				final int mv = who.findBestMove(FastState.fromState(s), 600000, h, h.length);
				if (mv == Move.NONE) {
					illegal++;
					break;
				}
				chosen = Move.toAction(mv, s.getTurn());
				boolean ok = false;
				for (Action x : moves) {
					if (x.getFrom().equals(chosen.getFrom()) && x.getTo().equals(chosen.getTo())) {
						ok = true;
					}
				}
				if (!ok) {
					illegal++;
					break;
				}
				if (who == a && who.lastRootLo == 1 && who.lastRootHi == 1) {
					provenWins++;
					provenAndNotWon = true; // si azzera se a fine partita ha vinto
				}
			} else {
				chosen = moves.get(rnd.nextInt(moves.size()));
			}
			final State next = AshtonRules.applyMove(s, chosen);
			if (AshtonRules.isTerminal(next)) {
				final boolean whiteWins = next.getTurn() == Turn.WHITEWIN;
				outcome = (whiteWins == mctsWhite) ? 1 : -1;
				s = next;
				break;
			}
			if (AshtonRules.wasCapture(s, next, chosen)) {
				hist.clear();
			}
			final long hn = ZobristHasher.fullHash(next);
			if (hist.contains(hn)) { // stessa posizione due volte dall'ultima cattura: pareggio
				outcome = 0;
				s = next;
				break;
			}
			hist.add(hn);
			s = next;
		}
		a.shutdown();
		if (b != null) {
			b.shutdown();
		}
		return new int[] { outcome, illegal, provenWins, provenAndNotWon && outcome != 1 ? 1 : 0 };
	}

	static void testGames() {
		section("partite complete contro l'arbitro indipendente (AshtonRules)");
		int games = 0, wins = 0, losses = 0, draws = 0, illegal = 0, proven = 0, notConverted = 0;
		for (int g = 0; g < 12; g++) {
			final int[] r = playRefereed(seed + 500 + g, g % 2 == 0, false);
			games++;
			if (r[0] > 0) {
				wins++;
			} else if (r[0] < 0) {
				losses++;
			} else {
				draws++;
			}
			illegal += r[1];
			proven += r[2];
			notConverted += r[3];
		}
		check("12 partite contro un avversario a caso: nessuna mossa rifiutata dall'arbitro (" + illegal + ")", illegal == 0);
		check("risultato: " + wins + " vittorie, " + draws + " pareggi, " + losses + " sconfitte (almeno 9 vittorie su 12)", wins >= 9);
		check("ogni vittoria dimostrata (" + proven + " volte) e' stata poi convertita in vittoria (" + notConverted + " non convertite)", notConverted == 0);
		int mi = 0, mg = 0;
		for (int g = 0; g < 4; g++) {
			final int[] r = playRefereed(seed + 600 + g, true, true);
			mg++;
			mi += r[1];
		}
		check("4 partite Mcts contro Mcts: tutte concluse senza mosse illegali (" + mi + ")", mi == 0 && mg == 4);
	}
}
