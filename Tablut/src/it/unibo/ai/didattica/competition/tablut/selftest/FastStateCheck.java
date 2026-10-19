package it.unibo.ai.didattica.competition.tablut.selftest;

import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ZobristHasher;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Test DIFFERENZIALE della nuova rappresentazione {@link FastState} contro il motore storico, che fa da oracolo
 * indipendente: {@link AshtonRules} (verificato a sua volta contro il motore ufficiale da {@code CrossCheck}) e
 * {@link ZobristHasher} (hash calcolato sullo {@code State}).
 * <p>
 * Per OGNI posizione visitata e per OGNI mossa legale si controlla che:
 * <ol>
 * <li>{@code FastState.generateMoves} produca la STESSA lista di {@code AshtonRules.generateMoves}, nello stesso ordine
 * (l'ordine conta: determina il tie-break alla radice e quindi l'identita' dei risultati di ricerca);</li>
 * <li>{@code makeMove} dia lo stesso stato di {@code AshtonRules.applyMove}: scacchiera, turno/terminalita', e
 * {@code UndoInfo.wasCapture} uguale a {@code AshtonRules.wasCapture};</li>
 * <li>l'hash incrementale coincida con l'hash ricalcolato da zero E con {@code ZobristHasher.fullHash} dello stato
 * equivalente;</li>
 * <li>conteggi delle pedine e cella del Re restino coerenti;</li>
 * <li>{@code unmakeMove} ripristini ESATTAMENTE lo stato precedente (tutti i campi, anche l'hash);</li>
 * <li>{@code fromState}/{@code toState} facciano un giro completo senza perdere nulla.</li>
 * </ol>
 * Due fonti di posizioni: (a) partite complete (distribuzione realistica, comprese vittorie e catture) e (b) posizioni
 * casuali DENSE costruite a mano, che stressano le catture rare (Re vicino al trono, muri-citadel, punte...) molto piu'
 * di quanto facciano le partite. Alla fine stampa la copertura: se un evento chiave non e' mai stato visto il test FALLISCE
 * (un test che non esercita un caso non lo verifica).
 *
 * Uso: {@code FastStateCheck [partite] [posizioniCasuali] [seed]}.
 */
public class FastStateCheck {

	private static int failures = 0;
	private static long checks = 0;

	// copertura
	private static long positions, movesCompared, withCapture, whiteEscapes, blackKingCapturesThrone, blackKingCapturesAdj,
			blackKingCapturesElse, whiteCaptures, blackCaptures, multiCaptures, terminalPositionsConverted, throneEmptyPositions, kingOnEdgePositions;
	// copertura di tacticalValue
	private static long tactWin, tactCap1, tactCap2plus, tactZero, tactEscapeWithCapture;

	/** Numero di pedine di tipo {@code p} sulla scacchiera di {@code s} (oracolo indipendente da FastState). */
	private static int countPawns(State s, Pawn p) {
		int c = 0;
		for (Pawn[] row : s.getBoard()) {
			for (Pawn x : row) {
				if (x == p) {
					c++;
				}
			}
		}
		return c;
	}

	/** Valore tattico atteso di una mossa, ricavato SOLO dal motore storico: vittoria immediata, altrimenti pedine avversarie perse (Re escluso). */
	private static int expectedTactical(State before, State after) {
		if (after.getTurn() == Turn.WHITEWIN || after.getTurn() == Turn.BLACKWIN) {
			return FastState.TACT_WIN;
		}
		final Pawn victim = before.getTurn() == Turn.WHITE ? Pawn.BLACK : Pawn.WHITE;
		return countPawns(before, victim) - countPawns(after, victim);
	}

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

	public static void main(String[] args) {
		int games = args.length > 0 ? Integer.parseInt(args[0]) : 300;
		int randomPositions = args.length > 1 ? Integer.parseInt(args[1]) : 20000;
		long seed = args.length > 2 ? Long.parseLong(args[2]) : 12345L;
		Random rnd = new Random(seed);

		testInitialPosition();
		testConstants();

		for (int g = 0; g < games; g++) {
			playGame(rnd);
		}
		System.out.println("Partite complete: " + games + " -> " + positions + " posizioni, " + movesCompared + " mosse confrontate");

		long before = positions;
		for (int i = 0; i < randomPositions; i++) {
			State s = TestPositions.randomPosition(rnd, rnd.nextInt(4) == 0); // 1 su 4 con il Re anche sul bordo
			compareAllMoves(s);
		}
		System.out.println("Posizioni casuali dense: " + randomPositions + " (" + (positions - before) + " effettivamente confrontate)");

		testHandBuiltCases();

		System.out.println("Copertura: mosse " + movesCompared + ", con cattura " + withCapture + ", catture bianche " + whiteCaptures
				+ ", catture nere " + blackCaptures + ", mosse con >=2 catture " + multiCaptures + ", fughe del Re " + whiteEscapes
				+ ", Re catturato sul trono " + blackKingCapturesThrone + " / adiacente al trono " + blackKingCapturesAdj + " / altrove "
				+ blackKingCapturesElse + ", posizioni a trono vuoto " + throneEmptyPositions + ", Re sul bordo (non terminali) " + kingOnEdgePositions + ", terminali convertiti "
				+ terminalPositionsConverted);
		check(withCapture > 1000, "copertura: troppo poche catture (" + withCapture + ")");
		check(whiteEscapes > 20, "copertura: troppe poche fughe del Re (" + whiteEscapes + ")");
		check(blackKingCapturesThrone > 5, "copertura: Re catturato sul trono mai/poco visto (" + blackKingCapturesThrone + ")");
		check(blackKingCapturesAdj > 5, "copertura: Re catturato adiacente al trono mai/poco visto (" + blackKingCapturesAdj + ")");
		check(blackKingCapturesElse > 5, "copertura: Re catturato altrove mai/poco visto (" + blackKingCapturesElse + ")");
		check(multiCaptures > 20, "copertura: catture multiple mai/poco viste (" + multiCaptures + ")");
		check(kingOnEdgePositions > randomPositions / 50, "copertura: posizioni con il Re sul bordo (catture senza cella oltre) troppo poche (" + kingOnEdgePositions + ")");
		check(throneEmptyPositions > 100, "copertura: posizioni con trono vuoto poco viste (" + throneEmptyPositions + ")");
		System.out.println("tacticalValue: vittorie immediate " + tactWin + " (fughe con cattura " + tactEscapeWithCapture + "), 0 catture " + tactZero
				+ ", 1 cattura " + tactCap1 + ", 2+ catture " + tactCap2plus);
		check(tactWin > 1000 && tactCap1 > 1000 && tactCap2plus > 20 && tactZero > 100000, "copertura: tacticalValue non ha visto tutte le classi");
		check(tactEscapeWithCapture > 0, "copertura: nessuna fuga del Re che cattura anche (la precedenza della vittoria non e' stata esercitata)");

		if (failures == 0) {
			System.out.println("=== FastStateCheck OK: " + checks + " verifiche, 0 discordanze ===");
		} else {
			System.out.println("=== FastStateCheck: " + failures + " FALLIMENTI su " + checks + " verifiche ===");
			System.exit(1);
		}
	}

	// ------------------------------------------------------------------------------------------ casi semplici

	private static void testInitialPosition() {
		State legacy = new StateTablut();
		legacy.setTurn(Turn.WHITE);
		FastState f = new FastState();
		FastState conv = FastState.fromState(legacy);
		check(f.samePosition(conv), "la posizione iniziale di FastState coincide con StateTablut (Bianco di turno)");
		check(f.zobristHash == conv.zobristHash, "hash della posizione iniziale coerente tra costruttore e fromState");
		check(f.zobristHash == ZobristHasher.fullHash(legacy), "hash iniziale == ZobristHasher.fullHash (stesse chiavi)");
		check(f.whiteCount == 8 && f.blackCount == 16 && f.kingSquare == 40, "conteggi iniziali 8/16, Re sul trono");
		int[] buf = new int[FastState.MAX_MOVES];
		int n = f.generateMoves(buf);
		check(n == AshtonRules.generateMoves(legacy).size(), "numero di mosse iniziali del Bianco coincide con AshtonRules (" + n + ")");
		FastState black = f.duplicate();
		black.turn = BoardConstants.BLACK_TURN;
		black.zobristHash = black.computeHash();
		State legacyBlack = legacy.clone();
		legacyBlack.setTurn(Turn.BLACK);
		check(black.zobristHash == ZobristHasher.fullHash(legacyBlack), "hash con il Nero di turno == ZobristHasher");
		check(black.generateMoves(buf) == AshtonRules.generateMoves(legacyBlack).size(), "numero di mosse iniziali del Nero coincide");
	}

	private static void testConstants() {
		// tabelle: 16 citadel, 4 punte, 32 celle di bordo, STEPS coerente con le coordinate
		int cit = 0, tips = 0, edge = 0;
		for (int sq = 0; sq < 81; sq++) {
			cit += BoardConstants.IS_CITADEL[sq] ? 1 : 0;
			tips += BoardConstants.IS_TIP[sq] ? 1 : 0;
			edge += BoardConstants.IS_EDGE[sq] ? 1 : 0;
			int r = sq / 9, c = sq % 9;
			check(BoardConstants.STEPS[0][sq] == r && BoardConstants.STEPS[1][sq] == 8 - r && BoardConstants.STEPS[2][sq] == c
					&& BoardConstants.STEPS[3][sq] == 8 - c, "STEPS della cella " + sq);
			check(BoardConstants.name(sq).equals("" + (char) ('a' + c) + (r + 1)), "nome della cella " + sq);
		}
		check(cit == 16 && tips == 4 && edge == 32, "16 citadel, 4 punte, 32 celle di bordo (" + cit + "," + tips + "," + edge + ")");
		State s = new StateTablut();
		for (int sq = 0; sq < 81; sq++) {
			check(s.getBox(sq / 9, sq % 9).equals(BoardConstants.name(sq)), "BoardConstants.name coincide con State.getBox per " + sq);
		}
		// Move: giri completi azione <-> intero
		for (int from = 0; from < 81; from += 7) {
			for (int to = 0; to < 81; to += 5) {
				int m = Move.of(from, to);
				check(Move.getFrom(m) == from && Move.getTo(m) == to, "Move.of/getFrom/getTo");
				check(Move.fromIndex(Move.index(m)) == m, "Move.index/fromIndex");
				Action a = Move.toAction(m, Turn.WHITE);
				check(Move.fromAction(a) == m, "Move.toAction/fromAction " + Move.toString(m));
				check(m >= 0 && m < 0xFFFF, "la mossa entra nel campo a 16 bit della TT");
			}
		}
	}

	// ------------------------------------------------------------------------------------------ partite

	private static void playGame(Random rnd) {
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		// aperture e stili diversi: puramente casuale, oppure "avido" (preferisce mosse che catturano/avvicinano il Re al bordo)
		boolean greedy = rnd.nextInt(3) != 0;
		for (int ply = 0; ply < 220 && !AshtonRules.isTerminal(s); ply++) {
			List<Action> moves = AshtonRules.generateMoves(s);
			if (moves.isEmpty()) {
				compareAllMoves(s);
				return;
			}
			compareAllMoves(s);
			Action chosen = moves.get(rnd.nextInt(moves.size()));
			if (greedy) {
				for (int tries = 0; tries < 6; tries++) {
					Action cand = moves.get(rnd.nextInt(moves.size()));
					State next = AshtonRules.applyMove(s, cand);
					if (AshtonRules.isTerminal(next) || AshtonRules.wasCapture(s, next, cand)) {
						chosen = cand;
						break;
					}
				}
			}
			s = AshtonRules.applyMove(s, chosen);
		}
		if (AshtonRules.isTerminal(s)) {
			compareAllMoves(s); // una posizione terminale: nessuna mossa, conversione corretta
		}
	}

	// ------------------------------------------------------------------------------------------ posizioni casuali

	// ------------------------------------------------------------------------------------------ confronto

	/** Confronta TUTTO su una posizione: conversione, lista mosse, e ogni mossa con make/unmake. */
	private static void compareAllMoves(State legacy) {
		positions++;
		FastState f = FastState.fromState(legacy);
		final String where = "\n" + legacy + "\n";

		// ---- conversione
		check(f.zobristHash == f.computeHash(), "fromState: hash incrementale == ricalcolato" + where);
		if (!AshtonRules.isTerminal(legacy)) {
			check(f.zobristHash == ZobristHasher.fullHash(legacy), "fromState: hash == ZobristHasher.fullHash" + where);
		}
		check(f.turnAsDomain() == legacy.getTurn(), "fromState: turno" + where);
		if (AshtonRules.isTerminal(legacy)) {
			terminalPositionsConverted++;
		}
		if (f.board[BoardConstants.THRONE] == BoardConstants.EMPTY) {
			throneEmptyPositions++;
		}
		if (f.kingSquare >= 0 && BoardConstants.IS_EDGE[f.kingSquare] && !AshtonRules.isTerminal(legacy)) {
			kingOnEdgePositions++;
		}
		State back = f.toState();
		check(sameBoard(back, legacy, false) && back.getTurn() == legacy.getTurn(), "toState(fromState(s)) == s" + where);

		// ---- lista mosse: stessa lista, stesso ordine
		List<Action> expected = AshtonRules.generateMoves(legacy);
		int[] buf = new int[FastState.MAX_MOVES];
		int n = f.generateMoves(buf);
		check(n == expected.size(), "numero di mosse " + n + " vs " + expected.size() + where);
		check(f.hasLegalMove() == !expected.isEmpty(), "hasLegalMove" + where);
		if (n != expected.size()) {
			return;
		}
		for (int i = 0; i < n; i++) {
			check(buf[i] == Move.fromAction(expected.get(i)), "mossa #" + i + ": " + Move.toString(buf[i]) + " vs " + expected.get(i) + where);
		}

		// ---- ogni mossa: make == applyMove, unmake ripristina
		FastState snapshot = f.duplicate();
		UndoInfo undo = new UndoInfo();
		final Turn mover = legacy.getTurn();
		final int kingBefore = f.kingSquare;
		for (int i = 0; i < n; i++) {
			movesCompared++;
			int mv = buf[i];
			Action a = expected.get(i);
			State next = AshtonRules.applyMove(legacy, a);
			boolean cap = AshtonRules.wasCapture(legacy, next, a);

			final int tactical = f.tacticalValue(mv); // PRIMA di eseguire la mossa
			f.makeMove(mv, undo);
			final String ctx = " mossa " + a + where;
			final int tacticalExpected = expectedTactical(legacy, next);
			check(tactical == tacticalExpected, "tacticalValue " + tactical + " vs atteso " + tacticalExpected + ctx);
			check(f.zobristHash == f.computeHash(), "hash dopo tacticalValue+makeMove" + ctx);
			if (tacticalExpected == FastState.TACT_WIN) {
				tactWin++;
				if (cap && next.getTurn() == Turn.WHITEWIN) {
					tactEscapeWithCapture++;
				}
			} else if (tacticalExpected == 0) {
				tactZero++;
			} else if (tacticalExpected == 1) {
				tactCap1++;
			} else {
				tactCap2plus++;
			}

			check(f.turnAsDomain() == next.getTurn(), "dopo la mossa: turno/terminalita' " + f.turnAsDomain() + " vs " + next.getTurn() + ctx);
			check(sameBoard(f.toState(), next, true), "dopo la mossa: scacchiera" + ctx + "atteso:\n" + next + "\nottenuto:\n" + f.toState());
			check(undo.wasCapture() == cap, "wasCapture " + undo.wasCapture() + " vs " + cap + ctx);
			check(f.zobristHash == f.computeHash(), "hash incrementale == ricalcolato" + ctx);
			boolean kingCapturedOnThrone = next.getTurn() == Turn.BLACKWIN && kingBefore == BoardConstants.THRONE;
			if (!kingCapturedOnThrone) {
				// unica differenza nota: il Re catturato SUL trono lascia Pawn.EMPTY nel motore storico (hash senza marcatore)
				check(f.zobristHash == ZobristHasher.fullHash(next), "hash incrementale == ZobristHasher.fullHash" + ctx);
			}
			int w = 0, bl = 0, king = -1;
			for (int sq = 0; sq < 81; sq++) {
				if (f.board[sq] == BoardConstants.WHITE) {
					w++;
				} else if (f.board[sq] == BoardConstants.BLACK) {
					bl++;
				} else if (f.board[sq] == BoardConstants.KING) {
					king = sq;
				}
			}
			check(f.whiteCount == w && f.blackCount == bl && f.kingSquare == king, "conteggi/Re incrementali" + ctx);
			FastState reloaded = FastState.fromState(next);
			check(f.samePosition(reloaded) || kingCapturedOnThrone, "makeMove == fromState(applyMove)" + ctx);

			// statistiche di copertura
			if (cap) {
				withCapture++;
				int removed = 0;
				for (int k = 0; k < undo.capturedCount(); k++) {
					if (undo.capturedPiece(k) == BoardConstants.BLACK) {
						whiteCaptures++;
					} else if (undo.capturedPiece(k) == BoardConstants.WHITE) {
						blackCaptures++;
					}
					removed++;
				}
				if (removed >= 2) {
					multiCaptures++;
				}
			}
			if (next.getTurn() == Turn.WHITEWIN) {
				whiteEscapes++;
			} else if (next.getTurn() == Turn.BLACKWIN) {
				if (kingBefore == BoardConstants.THRONE) {
					blackKingCapturesThrone++;
				} else if (BoardConstants.IS_THRONE_NEIGHBOR[kingBefore]) {
					blackKingCapturesAdj++;
				} else {
					blackKingCapturesElse++;
				}
			}
			if (mover != Turn.WHITE && mover != Turn.BLACK) {
				fail("posizione non terminale attesa" + ctx);
			}

			f.unmakeMove(mv, undo);
			check(f.samePosition(snapshot) && f.zobristHash == snapshot.zobristHash && f.whiteCount == snapshot.whiteCount
					&& f.blackCount == snapshot.blackCount && f.kingSquare == snapshot.kingSquare && f.winner == snapshot.winner,
					"unmakeMove ripristina lo stato" + ctx);
		}
	}

	/**
	 * Uguaglianza di scacchiera. Con {@code lenientThrone} si tollera la sola differenza nota: cella (4,4) vuota con
	 * {@code Pawn.THRONE} da una parte e {@code Pawn.EMPTY} dall'altra (Re catturato sul trono).
	 */
	private static boolean sameBoard(State a, State b, boolean lenientThrone) {
		Pawn[][] x = a.getBoard(), y = b.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				if (x[r][c] != y[r][c]) {
					boolean throneVsEmpty = r == 4 && c == 4 && (x[r][c] == Pawn.THRONE || x[r][c] == Pawn.EMPTY)
							&& (y[r][c] == Pawn.THRONE || y[r][c] == Pawn.EMPTY);
					if (!(lenientThrone && throneVsEmpty)) {
						return false;
					}
				}
			}
		}
		return true;
	}

	// ------------------------------------------------------------------------------------------ casi costruiti a mano

	private static State blank() {
		StateTablut s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE;
		return s;
	}

	/** Casi limite scritti a mano, ciascuno con l'esito atteso scritto A PRIORI dalle regole (non dedotto dal codice). */
	private static void testHandBuiltCases() {
		UndoInfo undo = new UndoInfo();
		int[] buf = new int[FastState.MAX_MOVES];

		// 1) il Nero cattura una pedina bianca contro il TRONO occupato dal Re (muro "per coordinata")
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[4][4] = Pawn.KING;
			b[4][3] = Pawn.WHITE; // d5, adiacente al trono a ovest
			b[4][1] = Pawn.BLACK; // b5? e' una citadel: nero in citadel, muove su c5 (4,2)
			s.setTurn(Turn.BLACK);
			FastState f = FastState.fromState(s);
			int mv = Move.of(BoardConstants.coord(4, 1), BoardConstants.coord(4, 2));
			boolean legal = contains(f, buf, mv);
			check(legal, "caso 1: la mossa b5-c5 e' legale");
			f.makeMove(mv, undo);
			check(f.board[BoardConstants.coord(4, 3)] == BoardConstants.EMPTY, "caso 1: la pedina bianca in d5 e' catturata contro il trono occupato dal Re");
			check(undo.wasCapture() && f.whiteCount == 0, "caso 1: una cattura, nessuna pedina bianca rimasta");
		}

		// 2) il Bianco NON cattura contro una punta di citadel (e5? no: a5/e1/e9/i5 non sono muri per il Bianco), SI' contro le altre citadel
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[4][8] = Pawn.EMPTY;
			b[4][7] = Pawn.BLACK; // h5 (citadel non-punta? h5 e' (4,7) una citadel non punta): nero adiacente alla punta i5 (4,8)
			b[4][5] = Pawn.WHITE; // f5
			b[0][0] = Pawn.KING;
			b[6][6] = Pawn.WHITE;
			s.setTurn(Turn.WHITE);
			FastState f = FastState.fromState(s);
			// mossa del bianco g7(6,6) -> g5 (4,6): il nero in h5 (4,7) ha oltre di se' i5 = PUNTA: NON catturato
			int mv = Move.of(BoardConstants.coord(6, 6), BoardConstants.coord(4, 6));
			check(contains(f, buf, mv), "caso 2: la mossa g7-g5 e' legale");
			f.makeMove(mv, undo);
			check(f.board[BoardConstants.coord(4, 7)] == BoardConstants.BLACK, "caso 2: contro la punta i5 la pedina nera NON viene catturata");
			check(!undo.wasCapture(), "caso 2: nessuna cattura");
		}

		// 3) il Re e' un muro valido per il Bianco e il bianco cattura contro la citadel non-punta
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[2][2] = Pawn.KING;
			b[2][3] = Pawn.BLACK;
			b[2][5] = Pawn.WHITE;
			b[6][6] = Pawn.WHITE;
			s.setTurn(Turn.WHITE);
			FastState f = FastState.fromState(s);
			// f3 (2,5) bianco sposta in e3? (2,4): il nero in d3 (2,3) ha dietro il Re in c3 (2,2): catturato
			int mv = Move.of(BoardConstants.coord(2, 5), BoardConstants.coord(2, 4));
			check(contains(f, buf, mv), "caso 3: f3-e3 legale");
			f.makeMove(mv, undo);
			check(f.board[BoardConstants.coord(2, 3)] == BoardConstants.EMPTY && f.blackCount == 0, "caso 3: nero catturato tra bianco e Re");
		}

		// 4) Re sul trono: servono i 4 lati neri
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[4][4] = Pawn.KING;
			b[3][4] = Pawn.BLACK;
			b[5][4] = Pawn.BLACK;
			b[4][3] = Pawn.BLACK;
			b[4][6] = Pawn.BLACK; // arrivera' in f5 (4,5)
			s.setTurn(Turn.BLACK);
			FastState f = FastState.fromState(s);
			int mv = Move.of(BoardConstants.coord(4, 6), BoardConstants.coord(4, 5));
			check(contains(f, buf, mv), "caso 4: g5-f5 legale");
			f.makeMove(mv, undo);
			check(f.winner == FastState.WINNER_BLACK && f.kingSquare == -1, "caso 4: Re catturato sul trono con 4 lati neri");
			check(f.turnAsDomain() == Turn.BLACKWIN, "caso 4: BLACKWIN");
			f.unmakeMove(mv, undo);
			check(f.winner == FastState.WINNER_NONE && f.kingSquare == BoardConstants.THRONE && f.board[BoardConstants.THRONE] == BoardConstants.KING,
					"caso 4: unmake riporta il Re sul trono");
		}

		// 5) con 3 lati neri il Re sul trono NON e' catturato
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[4][4] = Pawn.KING;
			b[3][4] = Pawn.BLACK;
			b[5][4] = Pawn.BLACK;
			b[4][6] = Pawn.BLACK;
			b[4][3] = Pawn.EMPTY;
			s.setTurn(Turn.BLACK);
			FastState f = FastState.fromState(s);
			int mv = Move.of(BoardConstants.coord(4, 6), BoardConstants.coord(4, 5));
			f.makeMove(mv, undo);
			check(f.winner == FastState.WINNER_NONE && f.kingSquare == BoardConstants.THRONE, "caso 5: con soli 3 lati neri il Re sul trono sopravvive");
		}

		// 6) Re adiacente al trono: bastano 3 lati (il trono conta come lato ostile)
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[3][4] = Pawn.KING; // e4, adiacente al trono (sotto)
			b[2][4] = Pawn.BLACK; // e3
			b[3][3] = Pawn.BLACK; // d4
			b[3][6] = Pawn.BLACK; // g4 -> f4
			s.setTurn(Turn.BLACK);
			FastState f = FastState.fromState(s);
			int mv = Move.of(BoardConstants.coord(3, 6), BoardConstants.coord(3, 5));
			check(contains(f, buf, mv), "caso 6: g4-f4 legale");
			f.makeMove(mv, undo);
			check(f.winner == FastState.WINNER_BLACK, "caso 6: Re adiacente al trono catturato con 3 lati neri + trono");
		}

		// 7) Re altrove: tenaglia tra nero e citadel (anche punta) senza pedina nera dall'altra parte
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[4][2] = Pawn.KING; // c5; a sinistra b5 (4,1) e' una citadel
			b[4][6] = Pawn.BLACK; // g5 -> d5 (4,3)
			s.setTurn(Turn.BLACK);
			FastState f = FastState.fromState(s);
			int mv = Move.of(BoardConstants.coord(4, 6), BoardConstants.coord(4, 3));
			boolean legal = contains(f, buf, mv); // il trono vuoto e' tra g5 e d5? no: g5(4,6)->d5(4,3) passa da f5,e5 (trono vuoto: blocca)
			check(!legal, "caso 7: il trono vuoto blocca il passaggio g5-d5 (nessuno lo attraversa)");
			b[4][6] = Pawn.EMPTY;
			b[2][3] = Pawn.BLACK; // d3 -> d5? d4 e d5: (2,3)->(4,3) passa da (3,3): vuoto
			f = FastState.fromState(s);
			mv = Move.of(BoardConstants.coord(2, 3), BoardConstants.coord(4, 3));
			check(contains(f, buf, mv), "caso 7: d3-d5 legale");
			f.makeMove(mv, undo);
			check(f.winner == FastState.WINNER_BLACK, "caso 7: Re in c5 catturato tra d5 (nero) e la citadel b5");
		}

		// 8) fuga del Re: ha la precedenza e le catture del Re nella stessa mossa restano
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[2][2] = Pawn.KING; // c3
			b[2][0] = Pawn.EMPTY;
			b[1][0] = Pawn.BLACK; // a2 (1,0) adiacente ad a3 (2,0)
			b[0][0] = Pawn.WHITE; // a1 e' il muro oltre a2 rispetto ad a3
			s.setTurn(Turn.WHITE);
			FastState f = FastState.fromState(s);
			int mv = Move.of(BoardConstants.coord(2, 2), BoardConstants.coord(2, 0)); // c3 -> a3: bordo
			check(contains(f, buf, mv), "caso 8: c3-a3 legale");
			f.makeMove(mv, undo);
			check(f.winner == FastState.WINNER_WHITE, "caso 8: il Re raggiunge il bordo, vince il Bianco");
			check(undo.wasCapture() && f.board[BoardConstants.coord(1, 0)] == BoardConstants.EMPTY, "caso 8: il Re arrivando cattura anche la pedina nera in a2 (muro bianco in a1)");
			check(f.zobristHash == f.computeHash(), "caso 8: hash coerente sullo stato terminale");
		}

		// 9) stallo: nessuna mossa -> generateMoves = 0, hasLegalMove = false
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[2][2] = Pawn.KING;
			b[1][2] = Pawn.BLACK;
			b[3][2] = Pawn.BLACK;
			b[2][1] = Pawn.BLACK;
			b[2][3] = Pawn.BLACK;
			s.setTurn(Turn.WHITE);
			FastState f = FastState.fromState(s);
			check(f.generateMoves(buf) == 0 && !f.hasLegalMove(), "caso 9: Re circondato, nessuna mossa");
		}

		// 10) citadel: un nero fuori dalle citadel non puo' rientrarvi, uno dentro puo' muoversi nelle citadel entro 5 passi
		{
			State s = blank();
			Pawn[][] b = s.getBoard();
			b[0][4] = Pawn.BLACK; // e1 (citadel punta)
			b[6][1] = Pawn.BLACK; // b7: fuori dalle citadel
			b[2][6] = Pawn.KING;
			s.setTurn(Turn.BLACK);
			FastState f = FastState.fromState(s);
			int n = f.generateMoves(buf);
			List<Action> exp = AshtonRules.generateMoves(s);
			check(n == exp.size(), "caso 10: numero di mosse nere coincide (" + n + ")");
			boolean reenters = false;
			for (int i = 0; i < n; i++) {
				if (Move.getFrom(buf[i]) == BoardConstants.coord(6, 1) && BoardConstants.IS_CITADEL[Move.getTo(buf[i])]) {
					reenters = true;
				}
			}
			check(!reenters, "caso 10: un nero fuori dalle citadel non puo' entrarvi");
		}
	}

	private static boolean contains(FastState f, int[] buf, int move) {
		int n = f.generateMoves(buf);
		for (int i = 0; i < n; i++) {
			if (buf[i] == move) {
				return true;
			}
		}
		return false;
	}
}
