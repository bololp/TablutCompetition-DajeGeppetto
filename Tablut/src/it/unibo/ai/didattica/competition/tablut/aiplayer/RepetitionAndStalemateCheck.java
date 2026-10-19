package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Test "white-box" (stesso package di {@link Negamax}, per chiamare
 * direttamente {@code negamax()}/{@code exploreChild()} invece di doverne
 * dedurre il comportamento indirettamente da quale mossa sceglie una ricerca
 * completa). Copre due scenari specifici:
 * <ol>
 * <li><b>Ripetizione</b>: una posizione ricreata esattamente (stessa board,
 * stesso turno) viene riconosciuta come pareggio SOLO quando la storia lo
 * dice, mai per coincidenza; e il risultato resta identico con o senza
 * transposition table (sicurezza rispetto al problema Graph History
 * Interaction).</li>
 * <li><b>Nessuna mossa legale</b>: una posizione dove chi deve muovere non ha
 * alcuna mossa disponibile viene valutata come sconfitta immediata, il
 * risultato viene correttamente memorizzato in TT e riusato da cache quando
 * la stessa posizione si ripresenta.</li>
 * </ol>
 */
public class RepetitionAndStalemateCheck {

	private static int failures = 0;

	public static void main(String[] args) {
		Object[] scenario = buildRepetitionScenario();
		State p0 = (State) scenario[0];
		State p1 = (State) scenario[1];
		State p2 = (State) scenario[2];
		State p3 = (State) scenario[3];
		Action m4 = (Action) scenario[4];

		System.out.println("Scenario di ripetizione costruito: " + describeMove((Action) scenario[5]) + ", "
				+ describeMove((Action) scenario[6]) + ", " + describeMove((Action) scenario[7]) + ", poi "
				+ describeMove(m4) + " ricrea la posizione iniziale (turno " + p0.getTurn() + ")");

		testRepetitionDetectedWithHistory(p0, p1, p2, p3, m4);
		testNoRepetitionWithoutHistory(p3, m4);
		testRepetitionSafeWithAndWithoutTT(p0, p1, p2, p3);
		testNoLegalMoves();
		testNoLegalMovesCachedCorrectly();

		if (failures == 0) {
			System.out.println("=== TUTTI I TEST OK ===");
		} else {
			System.out.println("=== " + failures + " TEST FALLITI ===");
			System.exit(1);
		}
	}

	private static void check(String label, boolean condition) {
		if (condition) {
			System.out.println("OK   - " + label);
		} else {
			System.out.println("FAIL - " + label);
			failures++;
		}
	}

	private static String describeMove(Action a) {
		return a.getFrom() + "->" + a.getTo();
	}

	// ------------------------------------------------------------------
	// Costruzione dello scenario: cerca (esplorando le VERE mosse legali,
	// nessuna coordinata scritta a mano) una coppia mossa-e-ritorno per il
	// bianco e una per il nero tali che, applicate in sequenza, ricreino
	// ESATTAMENTE la posizione iniziale (stessa board, stesso turno).
	// ------------------------------------------------------------------

	private static State initial() {
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		return s;
	}

	/**
	 * Punto di partenza per la ricerca del ciclo: NON la posizione iniziale
	 * pura, ma una raggiunta dopo qualche mossa casuale. Motivo: TUTTE le 16
	 * pedine nere partono dentro le citadels, e una volta uscite non possono
	 * piu' rientrarvi (regola ufficiale) - quindi la primissima mossa del nero
	 * e' per definizione irreversibile, e un ciclo "esci e torna" non puo'
	 * esistere a partire dalla posizione di partenza pura. Dopo qualche mossa
	 * casuale, invece, e' normale trovare pedine (bianche e nere) gia' fuori
	 * dalle citadels e libere di muoversi avanti e indietro.
	 */
	private static State warmup(long seed, int plies) {
		State s = initial();
		Random rnd = new Random(seed);
		for (int i = 0; i < plies; i++) {
			List<Action> moves = AshtonRules.generateMoves(s);
			Action chosen = moves.get(rnd.nextInt(moves.size()));
			State next = AshtonRules.applyMove(s, chosen);
			if (AshtonRules.wasCapture(s, next, chosen) || AshtonRules.isTerminal(next)) {
				continue; // riprova con un'altra mossa allo stesso passo, teniamo la board "tranquilla"
			}
			s = next;
		}
		return s;
	}

	/** {p0, p1, p2, p3, mossaFinale, mossaBianca1, mossaNera1, mossaBianca2} */
	private static Object[] buildRepetitionScenario() {
		for (long seed = 1; seed < 50; seed++) {
			Object[] found = tryBuildFrom(warmup(seed, 6));
			if (found != null) {
				return found;
			}
		}
		throw new IllegalStateException("Nessun ciclo di ripetizione trovato dopo vari tentativi (inatteso)");
	}

	private static Object[] tryBuildFrom(State p0) {
		if (p0.getTurn() != Turn.WHITE) {
			return null; // per semplicita' vogliamo sempre iniziare col bianco di turno
		}
		for (Action m1 : AshtonRules.generateMoves(p0)) {
			State p1 = AshtonRules.applyMove(p0, m1);
			if (AshtonRules.wasCapture(p0, p1, m1)) {
				continue; // vogliamo un ciclo SENZA catture (altrimenti la storia si azzererebbe)
			}
			for (Action m2 : AshtonRules.generateMoves(p1)) {
				State p2 = AshtonRules.applyMove(p1, m2);
				if (AshtonRules.wasCapture(p1, p2, m2)) {
					continue;
				}
				Action m3 = findReverse(p2, m1);
				if (m3 == null) {
					continue;
				}
				State p3 = AshtonRules.applyMove(p2, m3);
				if (AshtonRules.wasCapture(p2, p3, m3)) {
					continue;
				}
				Action m4 = findReverse(p3, m2);
				if (m4 == null) {
					continue;
				}
				State p4 = AshtonRules.applyMove(p3, m4);
				if (sameBoardAndTurn(p4, p0)) {
					return new Object[] { p0, p1, p2, p3, m4, m1, m2, m3 };
				}
			}
		}
		return null;
	}

	/** Hash di FastState dopo aver giocato {@code m} su una COPIA di {@code root} (la radice non si tocca). */
	private static long hashAfter(FastState root, Action m) {
		FastState c = root.duplicate();
		c.makeMove(Move.fromAction(m), new UndoInfo());
		return c.zobristHash;
	}

	private static Action findReverse(State state, Action original) {
		for (Action a : AshtonRules.generateMoves(state)) {
			if (a.getFrom().equals(original.getTo()) && a.getTo().equals(original.getFrom())) {
				return a;
			}
		}
		return null;
	}

	private static boolean sameBoardAndTurn(State a, State b) {
		if (a.getTurn() != b.getTurn()) {
			return false;
		}
		Pawn[][] ba = a.getBoard(), bb = b.getBoard();
		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				if (ba[r][c] != bb[r][c]) {
					return false;
				}
			}
		}
		return true;
	}

	// ------------------------------------------------------------------
	// Test 1: ripetizione rilevata SOLO se la storia lo dice
	// ------------------------------------------------------------------

	private static void testRepetitionDetectedWithHistory(State p0, State p1, State p2, State p3, Action m4) {
		System.out.println("--- testRepetitionDetectedWithHistory ---");
		Negamax n = new Negamax(new SimpleEvaluator(), 10).setThreads(1);

		long h0 = ZobristHasher.fullHash(p0);
		long h1 = ZobristHasher.fullHash(p1);
		long h2 = ZobristHasher.fullHash(p2);
		long h3 = ZobristHasher.fullHash(p3);
		List<Long> history = new ArrayList<>();
		history.add(h0);
		history.add(h1);
		history.add(h2);
		history.add(h3);

		State p4 = AshtonRules.applyMove(p3, m4);
		long h4 = ZobristHasher.fullHash(p4);
		check("la mossa finale ricrea esattamente l'hash della posizione iniziale", h4 == h0);

		long counterBefore = n.repetitionEventCounter;
		FastState root = FastState.fromState(p3);
		n.installForTest(root, history, 3);
		check("l'hash incrementale di FastState dopo la mossa coincide con quello storico", hashAfter(root, m4) == h4);
		double score = n.exploreChild(Move.fromAction(m4), 3, -1e9, 1e9, 1, 0);
		check("con la storia corretta, la ripetizione e' un pareggio esatto (punteggio 0)", score == 0.0);
		check("il contatore di ripetizioni e' stato incrementato", n.repetitionEventCounter == counterBefore + 1);
		check("make/unmake: la posizione di lavoro e' tornata quella del genitore", n.workingState().samePosition(root));
		check("la storia e' tornata alla lunghezza iniziale", n.historyLengthForTest() == history.size());
	}

	private static void testNoRepetitionWithoutHistory(State p3, Action m4) {
		System.out.println("--- testNoRepetitionWithoutHistory ---");
		Negamax n = new Negamax(new SimpleEvaluator(), 10).setThreads(1);

		State p4 = AshtonRules.applyMove(p3, m4);
		long h4 = ZobristHasher.fullHash(p4);
		long counterBefore = n.repetitionEventCounter;

		// STESSA posizione, STESSA mossa, ma SENZA storia: non deve essere
		// affatto trattata come ripetizione (altrimenti scatterebbe per
		// coincidenza sulla board, non per la vera storia della partita).
		FastState root = FastState.fromState(p3);
		n.installForTest(root, new ArrayList<Long>(), 0);
		double score = n.exploreChild(Move.fromAction(m4), 0, -1e9, 1e9, 1, 0);
		check("senza storia, la stessa posizione NON e' trattata come pareggio", score != 0.0);
		check("il contatore di ripetizioni non e' cambiato", n.repetitionEventCounter == counterBefore);
		check("make/unmake: la posizione di lavoro e' tornata quella del genitore", n.workingState().samePosition(root));
	}

	// ------------------------------------------------------------------
	// Test 2: sicurezza GHI - stesso risultato con e senza TT quando la
	// ricerca incontra davvero una ripetizione da qualche parte al suo interno
	// ------------------------------------------------------------------

	private static void testRepetitionSafeWithAndWithoutTT(State p0, State p1, State p2, State p3) {
		System.out.println("--- testRepetitionSafeWithAndWithoutTT ---");
		long h0 = ZobristHasher.fullHash(p0);
		long h1 = ZobristHasher.fullHash(p1);
		long h2 = ZobristHasher.fullHash(p2);
		List<Long> history = new ArrayList<>();
		history.add(h0);
		history.add(h1);
		history.add(h2);
		long generousBudget = 200_000L;

		for (int depth = 2; depth <= 5; depth++) {
			Negamax withTT = new Negamax(new SimpleEvaluator(), depth, true, 1 << 14).setThreads(1);
			Negamax withoutTT = new Negamax(new SimpleEvaluator(), depth, false, 0).setThreads(1);

			Action moveWithTT = withTT.findBestMove(p3, generousBudget, history);
			Action moveWithoutTT = withoutTT.findBestMove(p3, generousBudget, history);

			boolean scoreMatches = Math.abs(withTT.lastScore - withoutTT.lastScore) < 1e-9;
			boolean moveMatches = sameMove(moveWithTT, moveWithoutTT);
			check("profondita' " + depth + ": punteggio identico con/senza TT (" + withTT.lastScore + " vs "
					+ withoutTT.lastScore + ")", scoreMatches);
			check("profondita' " + depth + ": mossa identica con/senza TT (" + moveWithTT + " vs " + moveWithoutTT
					+ ")", moveMatches);
		}
	}

	private static boolean sameMove(Action a, Action b) {
		if (a == null || b == null) {
			return a == b;
		}
		return a.getRowFrom() == b.getRowFrom() && a.getColumnFrom() == b.getColumnFrom()
				&& a.getRowTo() == b.getRowTo() && a.getColumnTo() == b.getColumnTo();
	}

	// ------------------------------------------------------------------
	// Test 3: nessuna mossa legale per chi deve muovere
	// ------------------------------------------------------------------

	/** Costruisce una posizione artificiale in cui il Re bianco (unica pedina
	 *  bianca rimasta sul tabellone) e' circondato su tutti e 4 i lati e non
	 *  ha quindi alcuna mossa legale. Nota: essendo costruita direttamente
	 *  (non raggiunta tramite una vera mossa), la "cattura attiva" non scatta
	 *  mai qui - e' una posizione valida di "nessuna mossa disponibile", non
	 *  una cattura implicita (la regola stessa lo prevede esplicitamente:
	 *  "the capture must be active"). */
	private static State buildNoLegalMovesPosition() {
		State s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE; // il Re ha lasciato il trono
		b[2][2] = Pawn.KING;
		b[1][2] = Pawn.BLACK;
		b[3][2] = Pawn.BLACK;
		b[2][1] = Pawn.BLACK;
		b[2][3] = Pawn.BLACK;
		s.setTurn(Turn.WHITE);
		return s;
	}

	private static void testNoLegalMoves() {
		System.out.println("--- testNoLegalMoves ---");
		State s = buildNoLegalMovesPosition();
		check("nessuna mossa legale generata per il bianco", AshtonRules.generateMoves(s).isEmpty());

		Negamax n = new Negamax(new SimpleEvaluator(), 5).setThreads(1);
		long hash = ZobristHasher.fullHash(s);
		int depth = 3;
		n.installForTest(FastState.fromState(s), new ArrayList<Long>(), depth);
		check("l'hash di FastState coincide con quello storico anche per questa posizione artificiale", n.workingState().zobristHash == hash);
		double score = n.negamax(depth, -1e9, 1e9, 1, 0);
		double expected = -(Negamax.WIN_SCORE + depth); // perspective=1 (bianco): sconfitta = punteggio molto negativo
		check("punteggio esatto per sconfitta immediata (atteso " + expected + ", ottenuto " + score + ")",
				score == expected);
	}

	private static void testNoLegalMovesCachedCorrectly() {
		System.out.println("--- testNoLegalMovesCachedCorrectly ---");
		State s = buildNoLegalMovesPosition();
		Negamax n = new Negamax(new SimpleEvaluator(), 5, true, 1 << 10).setThreads(1);
		long hash = ZobristHasher.fullHash(s);
		int depth = 3;

		n.installForTest(FastState.fromState(s), new ArrayList<Long>(), depth);
		double first = n.negamax(depth, -1e9, 1e9, 1, 0);

		TranspositionTable.ProbeResult probe = n.tt.probe(hash, depth);
		check("la posizione senza mosse legali e' stata memorizzata in TT", probe.hit);
		check("memorizzata con profondita' sufficiente", probe.depthSufficient);
		check("memorizzata come valore ESATTO (non un bound)", probe.flag == TranspositionTable.EXACT);
		check("il valore in cache coincide con quello calcolato", probe.score == first);

		double second = n.negamax(depth, -1e9, 1e9, 1, 0);
		check("la seconda chiamata (servita dalla cache) da' lo stesso risultato", second == first);
	}
}
