package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Equivalenza della ricerca dopo la migrazione a {@link FastState}: la nuova {@link Negamax} (make/unmake, mosse {@code int},
 * zero allocazioni per nodo) deve comportarsi ESATTAMENTE come la versione storica su {@code State}, congelata in
 * {@link LegacyNegamax} (che usa {@link LegacySimpleEvaluator}, {@link AshtonRules} e {@link ZobristHasher}).
 * <p>
 * "Esattamente" e' una pretesa forte e voluta: a thread singolo, profondita' fissa (tempo praticamente illimitato) e stessa
 * configurazione, i due motori devono dare lo STESSO punteggio, la STESSA mossa, la STESSA profondita', lo STESSO numero di
 * nodi, lo stesso numero di scout/ri-ricerche PVS e di ripetizioni incontrate. Il numero di nodi e' il rilevatore piu'
 * sensibile: basta una mossa generata in un ordine diverso, un hash che differisce, una storia di ripetizione tenuta male
 * o un taglio mancato per cambiarlo. Si confrontano:
 * <ul>
 * <li>4 configurazioni (produzione: TT + PVS + killer + history; alpha-beta puro; solo killer; solo history) su posizioni di
 * partite complete (con la storia REALE di ripetizione, come la passa il client) e su posizioni casuali dense;</li>
 * <li>due evaluator: {@link SimpleEvaluator} contro {@link LegacySimpleEvaluator} (confronta anche il valore numerico, al bit)
 * e una rete {@link NeuralEvaluator} con pesi casuali (valori continui);</li>
 * <li>una partita intera con la STESSA istanza di ricerca riusata mossa dopo mossa (TT, history, killer persistenti);</li>
 * <li>ricerche a TEMPO limitato: mossa sempre legale, e la posizione di lavoro e' sempre tornata identica alla radice
 * (ogni makeMove e' stata annullata, anche quando il tempo scade a meta' albero); la posizione del chiamante non e' toccata;</li>
 * <li>le due API (con {@code State}/{@code Action} e con {@code FastState}/{@code int}) danno la stessa mossa.</li>
 * </ul>
 * Uso: {@code SearchEquivalenceCheck [partite] [posizioniCasuali] [seed]}.
 */
public class SearchEquivalenceCheck {

	private static int failures = 0;
	private static long checks = 0;
	private static final long BIG_BUDGET = 100_000_000L;

	private static void fail(String msg) {
		failures++;
		if (failures <= 25) {
			System.out.println("FAIL - " + msg);
		}
	}

	private static void check(boolean cond, String msg) {
		checks++;
		if (!cond) {
			fail(msg);
		}
	}

	private static final class Cfg {
		final String name;
		final boolean tt, pvs;
		final int order;

		Cfg(String name, boolean tt, boolean pvs, int order) {
			this.name = name;
			this.tt = tt;
			this.pvs = pvs;
			this.order = order;
		}
	}

	private static final Cfg[] CFGS = { new Cfg("produzione (TT+PVS+killer+history)", true, true, Negamax.ORDER_DEFAULT),
			new Cfg("alpha-beta puro", false, false, Negamax.ORDER_NONE), new Cfg("TT+killer", true, false, Negamax.ORDER_KILLERS),
			new Cfg("PVS+history senza TT", false, true, Negamax.ORDER_HISTORY) };

	private static long nodesNew = 0, nodesLegacy = 0, comparisons = 0, repetitionsSeen = 0, wins = 0;

	public static void main(String[] args) {
		// Abilita le asserzioni (invarianti interni) di Negamax PRIMA che la classe venga inizializzata, senza richiedere -ea.
		ClassLoader.getSystemClassLoader().setClassAssertionStatus("it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax", true);
		int games = args.length > 0 ? Integer.parseInt(args[0]) : 12;
		int randomPositions = args.length > 1 ? Integer.parseInt(args[1]) : 40;
		long seed = args.length > 2 ? Long.parseLong(args[2]) : 2024L;
		Random rnd = new Random(seed);

		Evaluator simpleNew = new SimpleEvaluator();
		Evaluator simpleOld = new LegacySimpleEvaluator();
		Evaluator net = NeuralEvaluator.randomInit(7);

		// ---- 1) posizioni di partite complete, con storia reale; profondita' 1..4 con Simple
		for (int g = 0; g < games; g++) {
			TestPositions.Game game = TestPositions.playGame(rnd, rnd.nextInt(3) != 0, 120);
			int n = game.positions.size();
			if (n == 0) {
				continue;
			}
			for (int k = 0; k < 4; k++) {
				int i = rnd.nextInt(n);
				// preferisce le fasi centrali: la posizione iniziale e' sempre uguale
				if (k == 0) {
					i = Math.min(n - 1, 2 + rnd.nextInt(4));
				}
				State s = game.positions.get(i);
				List<Long> h = game.histories.get(i);
				for (Cfg cfg : CFGS) {
					int depth = cfg.tt ? 4 : 3;
					compare("partita " + g + " ply " + i + " [" + cfg.name + "] Simple d" + depth, s, h, simpleNew, simpleOld, cfg, depth);
				}
				compare("partita " + g + " ply " + i + " rete d3", s, h, net, net, CFGS[0], 3);
			}
		}
		System.out.println("Posizioni di partita: confronti " + comparisons + ", nodi nuovo " + nodesNew + " / storico " + nodesLegacy);

		// ---- 2) posizioni casuali dense (catture, vittorie in 1, mosse mancanti)
		long before = comparisons;
		for (int i = 0; i < randomPositions; i++) {
			State s = TestPositions.randomPosition(rnd);
			List<Long> h = new ArrayList<>();
			h.add(FastState.hashOf(s));
			for (Cfg cfg : CFGS) {
				compare("casuale " + i + " [" + cfg.name + "] d" + (cfg.tt ? 4 : 3), s, h, simpleNew, simpleOld, cfg, cfg.tt ? 4 : 3);
			}
			compare("casuale " + i + " rete d3", s, h, net, net, CFGS[0], 3);
		}
		System.out.println("Posizioni casuali: confronti " + (comparisons - before) + ", vittorie forzate viste " + wins);

		// ---- 3) istanza riusata per tutta una partita
		for (int g = 0; g < 3; g++) {
			reusedInstanceGame(rnd, simpleNew, simpleOld, g);
		}

		// ---- 4) stalli dentro l'albero, ricerche interrotte, tempo limitato, invarianti, API
		stalemateScenarios(simpleNew, simpleOld);
		timeoutPollution(rnd, simpleNew, net);
		timedSearches(rnd, simpleNew, net);
		apiConsistency(rnd, simpleNew);

		System.out.println("Ripetizioni incontrate dalla ricerca nei confronti: " + repetitionsSeen);
		check(comparisons > 200, "copertura: confronti troppo pochi (" + comparisons + ")");
		check(nodesNew > 1_000_000, "copertura: nodi troppo pochi (" + nodesNew + ")");
		check(wins > 0, "copertura: nessuna vittoria forzata vista");
		check(repetitionsSeen > 0, "copertura: nessuna ripetizione mai incontrata (la storia non e' esercitata)");

		if (failures == 0) {
			System.out.println("=== SearchEquivalenceCheck OK: " + checks + " verifiche (" + comparisons + " ricerche confrontate, " + nodesNew
					+ " nodi nuovo = " + nodesLegacy + " nodi storico) ===");
		} else {
			System.out.println("=== SearchEquivalenceCheck: " + failures + " FALLIMENTI su " + checks + " verifiche ===");
			System.exit(1);
		}
	}

	// ------------------------------------------------------------------------------------------ confronto di una ricerca

	private static boolean sameMove(Action a, Action b) {
		if (a == null || b == null) {
			return a == b;
		}
		return a.getFrom().equals(b.getFrom()) && a.getTo().equals(b.getTo()) && a.getTurn() == b.getTurn();
	}

	private static void compare(String label, State root, List<Long> hist, Evaluator evNew, Evaluator evOld, Cfg cfg, int depth) {
		LegacyNegamax legacy = new LegacyNegamax(evOld, depth, cfg.tt, 1 << 14, cfg.pvs, cfg.order);
		Negamax fresh = new Negamax(evNew, depth, cfg.tt, 1 << 14, cfg.pvs, cfg.order, 1);

		FastState before = FastState.fromState(root);
		Action expected = legacy.findBestMove(root, BIG_BUDGET, hist);
		Action got = fresh.findBestMove(root, BIG_BUDGET, hist);
		comparisons++;
		nodesNew += fresh.lastNodeCount;
		nodesLegacy += legacy.lastNodeCount;

		check(sameMove(expected, got), label + ": mossa " + got + " vs storico " + expected);
		if (expected == null || got == null) {
			return;
		}
		check(fresh.lastScore == legacy.lastScore, label + ": punteggio " + fresh.lastScore + " vs storico " + legacy.lastScore);
		check(fresh.lastDepthReached == legacy.lastDepthReached, label + ": profondita' " + fresh.lastDepthReached + " vs " + legacy.lastDepthReached);
		check(fresh.lastNodeCount == legacy.lastNodeCount, label + ": nodi " + fresh.lastNodeCount + " vs storico " + legacy.lastNodeCount);
		check(fresh.lastPvsScouts == legacy.lastPvsScouts && fresh.lastPvsReSearches == legacy.lastPvsReSearches,
				label + ": scout/ri-ricerche PVS " + fresh.lastPvsScouts + "/" + fresh.lastPvsReSearches + " vs " + legacy.lastPvsScouts + "/"
						+ legacy.lastPvsReSearches);
		check(fresh.repetitionEventCounter == legacy.repetitionEventCounter,
				label + ": ripetizioni incontrate " + fresh.repetitionEventCounter + " vs " + legacy.repetitionEventCounter);
		repetitionsSeen += fresh.repetitionEventCounter;
		if (Math.abs(fresh.lastScore) >= Negamax.WIN_SCORE) {
			wins++;
		}
		// la posizione di lavoro e' tornata alla radice (ogni makeMove annullata) e quella del chiamante non e' cambiata
		check(fresh.workingState().samePosition(before) && fresh.workingState().zobristHash == before.zobristHash,
				label + ": la posizione di lavoro NON e' tornata alla radice dopo la ricerca");
		check(FastState.fromState(root).samePosition(before), label + ": lo State del chiamante e' stato modificato");
		// la mossa e' legale
		int[] buf = new int[FastState.MAX_MOVES];
		int n = before.generateMoves(buf);
		int mv = Move.fromAction(got);
		boolean legal = false;
		for (int i = 0; i < n; i++) {
			legal |= buf[i] == mv;
		}
		check(legal, label + ": mossa non legale " + got);
	}

	// ------------------------------------------------------------------------------------------ istanza riusata

	/**
	 * Una partita intera giocata da due istanze (storica e nuova) RIUSATE per ogni mossa: la TT, la tabella history (con il suo
	 * invecchiamento) e il resto dello stato persistono tra una ricerca e la successiva, come nel client reale. Le due istanze
	 * devono scegliere sempre la stessa mossa con gli stessi numeri; la partita prosegue con la mossa comune.
	 */
	private static void reusedInstanceGame(Random rnd, Evaluator simpleNew, Evaluator simpleOld, int g) {
		LegacyNegamax legacy = new LegacyNegamax(simpleOld, 3, true, 1 << 13, true, Negamax.ORDER_DEFAULT);
		Negamax fresh = new Negamax(simpleNew, 3, true, 1 << 13, true, Negamax.ORDER_DEFAULT, 1);
		TestPositions.Game script = TestPositions.playGame(rnd, true, 40); // fornisce solo una linea di partenza realistica
		State s = TestPositions.initial();
		List<Long> history = new ArrayList<>();
		history.add(FastState.hashOf(s));
		int played = 0;
		for (int ply = 0; ply < 40; ply++) {
			Action expected = legacy.findBestMove(s, BIG_BUDGET, history);
			Action got = fresh.findBestMove(s, BIG_BUDGET, history);
			String label = "istanza riusata, partita " + g + " ply " + ply;
			check(sameMove(expected, got), label + ": mossa " + got + " vs " + expected);
			if (expected == null || got == null) {
				break;
			}
			check(fresh.lastScore == legacy.lastScore && fresh.lastNodeCount == legacy.lastNodeCount,
					label + ": punteggio/nodi " + fresh.lastScore + "/" + fresh.lastNodeCount + " vs " + legacy.lastScore + "/" + legacy.lastNodeCount);
			nodesNew += fresh.lastNodeCount;
			nodesLegacy += legacy.lastNodeCount;
			comparisons++;
			// la partita avanza con una mossa casuale ogni 3 (altrimenti il "gioco migliore" ripete le stesse mosse): si varia
			Action play = (ply % 3 == 2) ? randomMove(s, rnd) : expected;
			State next = AshtonRules.applyMove(s, play);
			if (AshtonRules.isTerminal(next)) {
				break;
			}
			if (AshtonRules.wasCapture(s, next, play)) {
				history.clear();
			}
			history.add(FastState.hashOf(next));
			s = next;
			played++;
		}
		check(played > 5 || script.positions.isEmpty(), "partita riusata " + g + ": troppo corta (" + played + ")");
	}

	private static Action randomMove(State s, Random rnd) {
		List<Action> mv = AshtonRules.generateMoves(s);
		return mv.get(rnd.nextInt(mv.size()));
	}

	// ------------------------------------------------------------------------------------------ stalli

	/**
	 * Posizioni costruite a mano in cui l'avversario resta SENZA MOSSE dentro l'albero (non alla radice): chi non puo' muovere
	 * perde. Il Nero ha una sola pedina in i9 (angolo, non catturabile) bloccata da due pedine bianche; il Bianco, muovendo
	 * una qualunque altra pedina, lascia il Nero in stallo. La ricerca a profondita' >= 2 deve vedere la vittoria forzata, con lo
	 * stesso punteggio (WIN_SCORE + profondita') e gli stessi nodi della versione storica, e con il segno giusto per entrambi i colori.
	 */
	private static void stalemateScenarios(Evaluator evNew, Evaluator evOld) {
		// Bianco di turno: il Nero in i9 e' bloccato da pedine bianche in i8 e h9
		State white = TestPositions.blank();
		it.unibo.ai.didattica.competition.tablut.domain.State.Pawn[][] b = white.getBoard();
		b[4][4] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.KING;
		b[3][4] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE;
		b[5][4] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE;
		b[4][3] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE;
		b[4][5] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE;
		b[7][8] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE;
		b[8][7] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE;
		b[8][8] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.BLACK;
		b[1][1] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.WHITE; // una pedina libera di muoversi senza sbloccare il Nero
		white.setTurn(Turn.WHITE);
		List<Long> h = new ArrayList<>();
		h.add(FastState.hashOf(white));
		for (Cfg cfg : CFGS) {
			compare("stallo del Nero (Bianco muove) [" + cfg.name + "] d3", white, h, evNew, evOld, cfg, 3);
			compare("stallo del Nero (Bianco muove) [" + cfg.name + "] d4", white, h, evNew, evOld, cfg, 4);
		}
		Negamax n = new Negamax(evNew, 3, true, 1 << 12, true, Negamax.ORDER_DEFAULT, 1);
		n.findBestMove(white, BIG_BUDGET, h);
		check(n.lastScore >= Negamax.WIN_SCORE, "stallo del Nero: il Bianco deve vedere la vittoria forzata (punteggio " + n.lastScore + ")");

		// Nero di turno: il Bianco (il solo Re, bloccato da due pedine nere) resta senza mosse se il Nero muove la pedina libera
		State black = TestPositions.blank();
		it.unibo.ai.didattica.competition.tablut.domain.State.Pawn[][] c = black.getBoard();
		c[8][8] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.KING; // angolo: posizione artificiale, il Re e' bloccato
		c[7][8] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.BLACK;
		c[8][7] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.BLACK;
		c[1][1] = it.unibo.ai.didattica.competition.tablut.domain.State.Pawn.BLACK; // pedina nera libera
		black.setTurn(Turn.BLACK);
		h = new ArrayList<>();
		h.add(FastState.hashOf(black));
		for (Cfg cfg : CFGS) {
			compare("stallo del Bianco (Nero muove) [" + cfg.name + "] d3", black, h, evNew, evOld, cfg, 3);
		}
		Negamax m = new Negamax(evNew, 3, true, 1 << 12, true, Negamax.ORDER_DEFAULT, 1);
		m.findBestMove(black, BIG_BUDGET, h);
		// lastScore e' dal punto di vista di CHI MUOVE alla radice (qui il Nero): la vittoria per stallo del Bianco e' +WIN_SCORE
		check(m.lastScore >= Negamax.WIN_SCORE, "stallo del Bianco: il Nero (che muove) vince, punteggio " + m.lastScore + " (atteso >= WIN_SCORE)");
	}

	// ------------------------------------------------------------------------------------------ interruzioni

	/**
	 * Una ricerca INTERROTTA a meta' (il tempo scade in un punto deterministico dell'albero) non deve lasciare nella TT valori
	 * parziali: dopo di essa, una ricerca completa sulla STESSA istanza deve dare lo stesso punteggio di una senza TT. (Se i nodi
	 * in corso di unwinding fossero salvati, i loro valori sarebbero inventati; oltre a questo confronto, {@code Negamax} verifica
	 * con un'asserzione che non si memorizzi mai a tempo scaduto: le asserzioni sono attivate da {@link #main}.)
	 * <p>
	 * Il confronto con la ricerca senza TT e' valido SOLO dove la storia-grafo (GHI) non puo' intervenire: con ripetizioni
	 * nell'albero, la TT puo' differire dall'alpha-beta puro anche senza alcuna interruzione (limite NOTO e identico nella
	 * versione storica: una voce calcolata su un percorso e riusata su un altro con una ripetizione diversa; la protezione
	 * copre solo i sottoalberi che hanno gia' incontrato una ripetizione). Quindi si confrontano i punteggi solo nei casi in cui
	 * NESSUNA delle ricerche ha incontrato ripetizioni (profondita' basse, dove sono rare), e si richiede che siano la maggioranza.
	 */
	private static void timeoutPollution(Random rnd, Evaluator simple, Evaluator net) {
		int cases = 0, interrupted = 0, strict = 0;
		for (int g = 0; g < 10; g++) {
			TestPositions.Game game = TestPositions.playGame(rnd, true, 80);
			for (int k = 0; k < 8 && !game.positions.isEmpty(); k++) {
				int i = rnd.nextInt(game.positions.size());
				State s = game.positions.get(i);
				List<Long> h = game.histories.get(i);
				boolean useNet = (k & 1) == 1;
				Evaluator ev = useNet ? net : simple;
				int depth = useNet ? 3 : 4;
				Negamax polluted = new Negamax(ev, depth, true, 1 << 16, true, Negamax.ORDER_DEFAULT, 1);
				polluted.forceTimeUpAfterNodes = 2048 * (1 + rnd.nextInt(3)); // 2048, 4096 o 6144 nodi: molto prima della fine
				polluted.findBestMove(s, BIG_BUDGET, h);
				if (polluted.lastNodeCount >= polluted.forceTimeUpAfterNodes) {
					interrupted++;
				}
				polluted.forceTimeUpAfterNodes = 0;
				Action a = polluted.findBestMove(s, BIG_BUDGET, h); // completa, stessa istanza (TT "sporcata" dall'interruzione)
				Negamax noTT = new Negamax(ev, depth, false, 0, false, Negamax.ORDER_DEFAULT, 1);
				Action b = noTT.findBestMove(s, BIG_BUDGET, h);
				cases++;
				check(a != null && b != null, "inquinamento: nessuna mossa");
				if (a == null || b == null) {
					continue;
				}
				if (noTT.repetitionEventCounter == 0 && polluted.repetitionEventCounter == 0) {
					strict++;
					check(Math.abs(polluted.lastScore - noTT.lastScore) < 1e-9, "inquinamento da timeout: punteggio " + polluted.lastScore
							+ " dopo un'interruzione vs " + noTT.lastScore + " senza TT (nessuna ripetizione nell'albero: la TT deve essere esatta)");
				}
			}
		}
		System.out.println("Interruzioni a meta' albero: " + cases + " casi (" + interrupted + " effettivamente interrotti, " + strict + " confrontabili senza GHI)");
		check(interrupted > cases * 2 / 3, "copertura: troppe poche ricerche effettivamente interrotte (" + interrupted + "/" + cases + ")");
		check(strict > cases / 2, "copertura: troppi pochi casi confrontabili (" + strict + "/" + cases + ")");
	}

	// ------------------------------------------------------------------------------------------ tempo limitato

	private static void timedSearches(Random rnd, Evaluator simple, Evaluator net) {
		int searches = 0;
		for (int g = 0; g < 6; g++) {
			TestPositions.Game game = TestPositions.playGame(rnd, true, 100);
			for (int k = 0; k < 6 && !game.positions.isEmpty(); k++) {
				int i = rnd.nextInt(game.positions.size());
				State s = game.positions.get(i);
				List<Long> h = game.histories.get(i);
				FastState before = FastState.fromState(s);
				boolean useNet = (k & 1) == 1;
				// budget da 1 a 120 ms: il tempo scade in punti qualsiasi dell'albero
				long budget = 300 + 1 + rnd.nextInt(120);
				Negamax n = new Negamax(useNet ? net : simple, 40, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
				Action a = n.findBestMove(s, budget, h);
				searches++;
				check(a != null, "tempo limitato: nessuna mossa");
				if (a == null) {
					continue;
				}
				int[] buf = new int[FastState.MAX_MOVES];
				int cnt = before.generateMoves(buf);
				int mv = Move.fromAction(a);
				boolean legal = false;
				for (int j = 0; j < cnt; j++) {
					legal |= buf[j] == mv;
				}
				check(legal, "tempo limitato: mossa non legale " + a);
				check(n.workingState().samePosition(before) && n.workingState().zobristHash == before.zobristHash,
						"tempo limitato (budget " + budget + " ms, " + n.lastNodeCount + " nodi): la posizione di lavoro non e' tornata alla radice");
				check(n.historyLengthForTest() == h.size(), "tempo limitato: la storia di ripetizione non e' tornata alla lunghezza iniziale ("
						+ n.historyLengthForTest() + " vs " + h.size() + ")");
				check(n.lastDepthReached >= 1, "tempo limitato: nessuna iterazione completata");
			}
		}
		System.out.println("Ricerche a tempo limitato verificate: " + searches);
	}

	// ------------------------------------------------------------------------------------------ API

	/** Le due API pubbliche (State/Action e FastState/int) devono dare la stessa mossa; la radice FastState non viene toccata. */
	private static void apiConsistency(Random rnd, Evaluator simple) {
		for (int g = 0; g < 4; g++) {
			TestPositions.Game game = TestPositions.playGame(rnd, true, 80);
			for (int k = 0; k < 4 && !game.positions.isEmpty(); k++) {
				int i = rnd.nextInt(game.positions.size());
				State s = game.positions.get(i);
				List<Long> h = game.histories.get(i);
				Negamax a = new Negamax(simple, 3, true, 1 << 13, true, Negamax.ORDER_DEFAULT, 1);
				Negamax b = new Negamax(simple, 3, true, 1 << 13, true, Negamax.ORDER_DEFAULT, 1);
				Negamax c = new Negamax(simple, 3, true, 1 << 13, true, Negamax.ORDER_DEFAULT, 1);
				Action viaState = a.findBestMove(s, BIG_BUDGET, h);
				FastState root = FastState.fromState(s);
				FastState copy = root.duplicate();
				int viaFastList = b.findBestMove(root, BIG_BUDGET, h);
				long[] arr = new long[h.size() + 3]; // capienza maggiore della lunghezza: conta historyLen, non arr.length
				for (int j = 0; j < h.size(); j++) {
					arr[j] = h.get(j);
				}
				Arrays.fill(arr, h.size(), arr.length, 0x5EEDL);
				int viaFastArray = c.findBestMove(root, BIG_BUDGET, arr, h.size());
				check(viaState != null && Move.fromAction(viaState) == viaFastList && viaFastList == viaFastArray,
						"API: State=" + viaState + " FastState(List)=" + Move.toString(viaFastList) + " FastState(long[])=" + Move.toString(viaFastArray));
				check(root.samePosition(copy) && root.zobristHash == copy.zobristHash, "API: la FastState passata e' stata modificata");
				check(viaState == null || viaState.getTurn() == s.getTurn(), "API: il turno dell'Action coincide con quello della radice");
				check(a.lastNodeCount == b.lastNodeCount && b.lastNodeCount == c.lastNodeCount, "API: nodi diversi tra le tre forme");
			}
		}
		// radice senza mosse / terminale
		State terminal = TestPositions.initial();
		terminal.setTurn(Turn.WHITEWIN);
		Negamax n = new Negamax(simple, 3, true, 1 << 10, true, Negamax.ORDER_DEFAULT, 1);
		check(n.findBestMove(terminal, 1000) == null, "radice terminale: deve ritornare null");
		check(n.findBestMove(FastState.fromState(terminal), 1000) == Move.NONE, "radice terminale (FastState): deve ritornare Move.NONE");
	}
}
