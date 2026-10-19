package it.unibo.ai.didattica.competition.tablut.engine;

import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Rappresentazione leggera e veloce di una posizione di Ashton Tablut, pensata per il ciclo caldo della ricerca.
 * <p>
 * Ispirata al {@code FastState} dell'altro progetto, ma con le REGOLE di questo progetto (quelle di
 * {@code aiplayer.AshtonRules}, verificate contro il motore ufficiale {@code GameAshtonTablut}): per costruzione
 * {@link #generateMoves} e {@link #makeMove} producono le stesse mosse, nello stesso ordine, e lo stesso risultato
 * di {@code AshtonRules.generateMoves}/{@code applyMove} ({@code FastStateCheck} lo confronta su migliaia di
 * posizioni di partite complete).
 *
 * <h3>Cosa cambia rispetto a {@code State} + {@code AshtonRules}</h3>
 * <ul>
 * <li><b>Scacchiera piatta {@code byte[81]}</b> (cella = riga*9+colonna, si veda {@link BoardConstants}) al posto di
 * {@code Pawn[][]} (81 riferimenti a enum in 10 oggetti): sta in due righe di cache e si copia con un
 * {@code arraycopy}. Il trono vuoto e' semplicemente la cella 40 vuota.</li>
 * <li><b>Make/unmake invece di copia</b>: {@code AshtonRules.applyMove} clonava lo {@code State} per ogni figlio
 * (costruttore di {@code StateTablut} che riempie la scacchiera iniziale, poi la copia: ~0,7 us e vari oggetti per
 * nodo). Qui {@link #makeMove} modifica la posizione sul posto e {@link UndoInfo} permette di annullarla in O(1):
 * ZERO allocazioni per nodo.</li>
 * <li><b>Mosse come {@code int}</b> ({@link Move}) scritte in un buffer {@code int[]} fornito dal chiamante, al posto di
 * una {@code List<Action>} di oggetti con due stringhe (e {@code parseInt} per rileggerne le coordinate).</li>
 * <li><b>Campi incrementali</b>: turno, cella del Re, numero di pedine, vincitore e hash Zobrist sono sempre
 * aggiornati da {@link #makeMove}: nessun hash ricalcolato, nessuna scansione per contare le pedine.</li>
 * </ul>
 *
 * <h3>Stati terminali</h3>
 * Come in {@code AshtonRules}, la mossa che porta il Re al bordo da' {@link #WINNER_WHITE} (ha precedenza) e quella
 * che lo cattura {@link #WINNER_BLACK}; il Re catturato viene RIMOSSO dalla scacchiera ({@link #kingSquare} = -1).
 * {@link #turn} alterna comunque a ogni mossa; per una posizione con vincitore e' privo di significato (e il suo
 * contributo all'hash e' nullo, come nel motore storico). Non si devono giocare mosse da una posizione terminale.
 *
 * <h3>Thread-safety</h3>
 * Un'istanza e' mutabile e NON e' thread-safe: ogni thread di ricerca lavora sulla propria copia
 * ({@link #copyFrom}/{@link #duplicate}). Le tabelle statiche sono di sola lettura.
 */
public final class FastState {

	public static final byte WINNER_NONE = 0;
	public static final byte WINNER_WHITE = 1;
	public static final byte WINNER_BLACK = 2;
	public static final byte WINNER_DRAW = 3;

	/**
	 * Dimensione consigliata del buffer per {@link #generateMoves}. Il massimo teorico per una posizione legale e' 256
	 * (16 pedine nere x 16 celle raggiungibili al piu'): 512 lascia margine anche per posizioni costruite a mano.
	 */
	public static final int MAX_MOVES = 512;

	/** Scacchiera: {@code board[riga*9+colonna]} con i codici di {@link BoardConstants} (0 vuoto, 1 bianco, 2 nero, 4 re). */
	public final byte[] board = new byte[BoardConstants.NUM_SQUARES];
	/** Chi deve muovere: {@link BoardConstants#WHITE_TURN} o {@link BoardConstants#BLACK_TURN} (privo di significato se {@link #winner} != NONE). */
	public byte turn;
	/** Cella del Re, oppure -1 se il Re e' stato catturato. */
	public byte kingSquare;
	/** Pedine bianche (Re ESCLUSO) e nere sulla scacchiera. */
	public byte whiteCount;
	public byte blackCount;
	/** {@code WINNER_*}. */
	public byte winner;
	/** Hash Zobrist della posizione, sempre aggiornato (si veda {@link ZobristHash}). */
	public long zobristHash;

	/** Posizione iniziale di Ashton Tablut, Bianco di turno. */
	public FastState() {
		initInitialState();
	}

	/** Costruttore interno per le factory: scacchiera non inizializzata. */
	private FastState(boolean unused) {
	}

	// ------------------------------------------------------------------------------------------ costruzione

	/** Riporta questa istanza alla posizione iniziale, Bianco di turno. */
	public void initInitialState() {
		Arrays.fill(board, BoardConstants.EMPTY);
		board[BoardConstants.THRONE] = BoardConstants.KING;
		int[][] white = { { 2, 4 }, { 3, 4 }, { 5, 4 }, { 6, 4 }, { 4, 2 }, { 4, 3 }, { 4, 5 }, { 4, 6 } };
		for (int[] c : white) {
			board[BoardConstants.coord(c[0], c[1])] = BoardConstants.WHITE;
		}
		int[][] black = { { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, { 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 }, { 3, 0 }, { 4, 0 },
				{ 5, 0 }, { 4, 1 }, { 3, 8 }, { 4, 8 }, { 5, 8 }, { 4, 7 } };
		for (int[] c : black) {
			board[BoardConstants.coord(c[0], c[1])] = BoardConstants.BLACK;
		}
		turn = BoardConstants.WHITE_TURN;
		kingSquare = (byte) BoardConstants.THRONE;
		whiteCount = 8;
		blackCount = 16;
		winner = WINNER_NONE;
		zobristHash = computeHash();
	}

	/** Copia di {@code other} (stesso contenuto, nessuna condivisione). */
	public static FastState copyOf(FastState other) {
		FastState f = new FastState(false);
		f.copyFrom(other);
		return f;
	}

	public void copyFrom(FastState other) {
		System.arraycopy(other.board, 0, this.board, 0, BoardConstants.NUM_SQUARES);
		this.turn = other.turn;
		this.kingSquare = other.kingSquare;
		this.whiteCount = other.whiteCount;
		this.blackCount = other.blackCount;
		this.winner = other.winner;
		this.zobristHash = other.zobristHash;
	}

	public FastState duplicate() {
		return copyOf(this);
	}

	/** Converte uno {@link State} del dominio (es. quello ricevuto dal server) nella rappresentazione compatta. */
	public static FastState fromState(State s) {
		FastState f = new FastState(false);
		f.loadFrom(s);
		return f;
	}

	/** Come {@link #fromState} ma riusando questa istanza (nessuna allocazione). */
	public void loadFrom(State s) {
		final Pawn[][] bd = s.getBoard();
		int w = 0, bl = 0, king = -1;
		for (int r = 0; r < BoardConstants.SIZE; r++) {
			final Pawn[] row = bd[r];
			for (int c = 0; c < BoardConstants.SIZE; c++) {
				final Pawn p = row[c];
				final int sq = r * BoardConstants.SIZE + c;
				byte code;
				if (p == Pawn.WHITE) {
					code = BoardConstants.WHITE;
					w++;
				} else if (p == Pawn.BLACK) {
					code = BoardConstants.BLACK;
					bl++;
				} else if (p == Pawn.KING) {
					code = BoardConstants.KING;
					king = sq;
				} else {
					code = BoardConstants.EMPTY; // EMPTY, e THRONE (trono vuoto: nella forma compatta e' la cella 40 vuota)
				}
				board[sq] = code;
			}
		}
		whiteCount = (byte) w;
		blackCount = (byte) bl;
		kingSquare = (byte) king;
		final Turn t = s.getTurn();
		if (t == Turn.WHITE) {
			turn = BoardConstants.WHITE_TURN;
			winner = WINNER_NONE;
		} else if (t == Turn.BLACK) {
			turn = BoardConstants.BLACK_TURN;
			winner = WINNER_NONE;
		} else if (t == Turn.WHITEWIN) {
			turn = BoardConstants.BLACK_TURN; // lo stesso valore che avrebbe makeMove dopo la mossa vincente del Bianco
			winner = WINNER_WHITE;
		} else if (t == Turn.BLACKWIN) {
			turn = BoardConstants.WHITE_TURN;
			winner = WINNER_BLACK;
		} else {
			turn = BoardConstants.WHITE_TURN;
			winner = WINNER_DRAW;
		}
		zobristHash = computeHash();
	}

	/**
	 * Converte in uno {@link State} del dominio (alloca: solo per i bordi, es. log/test, mai nella ricerca). Il trono
	 * vuoto diventa {@code Pawn.THRONE}. Differenza (innocua) rispetto a {@code AshtonRules.applyMove}: se il Re e'
	 * catturato SUL trono, il motore storico lascia in quella cella {@code Pawn.EMPTY}, qui {@code Pawn.THRONE}.
	 */
	public State toState() {
		StateTablut st = new StateTablut();
		Pawn[][] b = st.getBoard();
		for (int sq = 0; sq < BoardConstants.NUM_SQUARES; sq++) {
			Pawn p;
			switch (board[sq]) {
			case BoardConstants.WHITE:
				p = Pawn.WHITE;
				break;
			case BoardConstants.BLACK:
				p = Pawn.BLACK;
				break;
			case BoardConstants.KING:
				p = Pawn.KING;
				break;
			default:
				p = (sq == BoardConstants.THRONE) ? Pawn.THRONE : Pawn.EMPTY;
				break;
			}
			b[sq / BoardConstants.SIZE][sq % BoardConstants.SIZE] = p;
		}
		st.setTurn(turnAsDomain());
		return st;
	}

	/** Il {@code State.Turn} equivalente (WHITE/BLACK se in corso, WHITEWIN/BLACKWIN/DRAW se terminale). */
	public Turn turnAsDomain() {
		if (winner == WINNER_WHITE) {
			return Turn.WHITEWIN;
		}
		if (winner == WINNER_BLACK) {
			return Turn.BLACKWIN;
		}
		if (winner == WINNER_DRAW) {
			return Turn.DRAW;
		}
		return turn == BoardConstants.WHITE_TURN ? Turn.WHITE : Turn.BLACK;
	}

	/** Hash Zobrist di uno {@code State}: la chiave con cui la ricerca e la TT identificano quella posizione. */
	public static long hashOf(State s) {
		return fromState(s).zobristHash;
	}

	/** Hash ricalcolato da zero (O(81)); coincide sempre con {@link #zobristHash} (lo verifica {@code FastStateCheck}). */
	public long computeHash() {
		return ZobristHash.compute(board, turn, winner);
	}

	// ------------------------------------------------------------------------------------------ interrogazioni

	public boolean isGameOver() {
		return winner != WINNER_NONE;
	}

	public boolean isWhiteToMove() {
		return turn == BoardConstants.WHITE_TURN;
	}

	/** Vero se i due oggetti rappresentano la stessa posizione (scacchiera, turno e vincitore: la stessa uguaglianza di {@code State.equals}). */
	public boolean samePosition(FastState o) {
		return turn == o.turn && winner == o.winner && Arrays.equals(board, o.board);
	}

	@Override
	public boolean equals(Object obj) {
		return obj instanceof FastState && samePosition((FastState) obj);
	}

	@Override
	public int hashCode() {
		return (int) (zobristHash ^ (zobristHash >>> 32));
	}

	// ------------------------------------------------------------------------------------------ generazione mosse

	/**
	 * Scrive in {@code out} tutte le mosse legali di chi deve muovere e ne ritorna il numero (0 se la partita e' finita o
	 * se non ci sono mosse: chi non puo' muovere perde, ma e' il chiamante a deciderlo). Stesse regole, e STESSO ORDINE,
	 * di {@code AshtonRules.generateMoves}: pezzi per cella crescente, direzioni riga-1, riga+1, colonna-1, colonna+1,
	 * celle a distanza crescente.
	 * <ul>
	 * <li>una pedina si ferma davanti a qualunque cella occupata, trono vuoto compreso (nessuno lo attraversa);</li>
	 * <li>una citadel si puo' attraversare o occupare solo partendo da una citadel, e non oltre 5 passi.</li>
	 * </ul>
	 *
	 * @param out buffer di almeno {@link #MAX_MOVES} elementi
	 */
	public int generateMoves(int[] out) {
		if (winner != WINNER_NONE) {
			return 0;
		}
		final byte[] b = board;
		final boolean white = turn == BoardConstants.WHITE_TURN;
		final boolean[] cit = BoardConstants.IS_CITADEL;
		final int[] offs = BoardConstants.DIR_OFFSET;
		int n = 0;
		for (int from = 0; from < BoardConstants.NUM_SQUARES; from++) {
			final byte p = b[from];
			if (white ? (p != BoardConstants.WHITE && p != BoardConstants.KING) : (p != BoardConstants.BLACK)) {
				continue;
			}
			final boolean originCitadel = cit[from];
			for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
				final int steps = BoardConstants.STEPS[d][from];
				final int off = offs[d];
				int to = from;
				for (int step = 1; step <= steps; step++) {
					to += off;
					if (b[to] != BoardConstants.EMPTY || to == BoardConstants.THRONE) {
						break; // pedina, oppure trono vuoto: blocca
					}
					if (cit[to] && (!originCitadel || step > 5)) {
						break; // citadel: solo da una citadel e entro 5 passi (la distanza cresce: oltre sarebbe comunque > 5)
					}
					out[n++] = from | (to << 7);
				}
			}
		}
		return n;
	}

	/** Vero se chi deve muovere ha almeno una mossa legale (esce alla prima trovata: piu' economico di {@link #generateMoves}). */
	public boolean hasLegalMove() {
		if (winner != WINNER_NONE) {
			return false;
		}
		final byte[] b = board;
		final boolean white = turn == BoardConstants.WHITE_TURN;
		final boolean[] cit = BoardConstants.IS_CITADEL;
		final int[] offs = BoardConstants.DIR_OFFSET;
		for (int from = 0; from < BoardConstants.NUM_SQUARES; from++) {
			final byte p = b[from];
			if (white ? (p != BoardConstants.WHITE && p != BoardConstants.KING) : (p != BoardConstants.BLACK)) {
				continue;
			}
			final boolean originCitadel = cit[from];
			for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
				if (BoardConstants.STEPS[d][from] < 1) {
					continue;
				}
				final int to = from + offs[d];
				if (b[to] != BoardConstants.EMPTY || to == BoardConstants.THRONE) {
					continue;
				}
				if (cit[to] && !originCitadel) {
					continue; // il primo passo (step = 1 <= 5) e' valido solo se si parte da una citadel
				}
				return true;
			}
		}
		return false;
	}

	// ------------------------------------------------------------------------------------------ make / unmake

	/**
	 * Esegue una mossa LEGALE (es. prodotta da {@link #generateMoves}) sul posto e registra in {@code undo} quanto serve
	 * per annullarla. Replica {@code AshtonRules.applyMove}: spostamento, catture (comprese le regole speciali del Re
	 * vicino al trono), fuga del Re (vittoria Bianca, ha precedenza) e cattura del Re (vittoria Nera). Aggiorna in modo
	 * incrementale hash, conteggi, cella del Re, vincitore e turno. Non va chiamata su una posizione terminale.
	 */
	public void makeMove(int move, UndoInfo undo) {
		final int from = move & 127;
		final int to = move >>> 7;
		final byte[] b = board;
		final byte piece = b[from];

		undo.capturedCount = 0;
		undo.oldTurn = turn;
		undo.oldKingSquare = kingSquare;
		undo.oldZobristHash = zobristHash;

		final long[][] key = ZobristHash.KEY;
		long h = zobristHash ^ key[from][piece] ^ key[to][piece];
		if (from == BoardConstants.THRONE) {
			h ^= ZobristHash.THRONE_EMPTY_KEY; // il Re lascia il trono: ora e' vuoto
		}
		zobristHash = h;
		b[from] = BoardConstants.EMPTY;
		b[to] = piece;

		boolean escaped = false;
		boolean kingCaptured = false;
		if (piece == BoardConstants.KING) {
			kingSquare = (byte) to;
		}
		if (turn == BoardConstants.WHITE_TURN) {
			escaped = piece == BoardConstants.KING && BoardConstants.IS_EDGE[to];
			captureByWhite(to, undo);
		} else {
			kingCaptured = captureByBlack(to, undo);
		}

		final long turnOld = turn == BoardConstants.BLACK_TURN ? ZobristHash.TURN_KEY : 0L;
		final byte newWinner = escaped ? WINNER_WHITE : (kingCaptured ? WINNER_BLACK : WINNER_NONE);
		winner = newWinner;
		turn = (byte) (turn ^ 1);
		final long turnNew = (newWinner == WINNER_NONE && turn == BoardConstants.BLACK_TURN) ? ZobristHash.TURN_KEY : 0L;
		zobristHash ^= turnOld ^ turnNew;
	}

	/** Annulla l'ultima {@link #makeMove} (stessa {@code move}, stesso {@code undo}), ripristinando ESATTAMENTE lo stato precedente. */
	public void unmakeMove(int move, UndoInfo undo) {
		final int from = move & 127;
		final int to = move >>> 7;
		final byte[] b = board;
		final byte piece = b[to];
		b[to] = BoardConstants.EMPTY;
		b[from] = piece;
		for (int i = 0; i < undo.capturedCount; i++) {
			final byte cp = undo.capturedPieces[i];
			b[undo.capturedSquares[i]] = cp;
			if (cp == BoardConstants.WHITE) {
				whiteCount++;
			} else if (cp == BoardConstants.BLACK) {
				blackCount++;
			}
		}
		turn = undo.oldTurn;
		kingSquare = undo.oldKingSquare;
		winner = WINNER_NONE;
		zobristHash = undo.oldZobristHash;
	}

	// ------------------------------------------------------------------------------------------ catture

	/**
	 * Catture del Bianco attorno alla cella {@code to} appena occupata: una pedina nera adiacente e' catturata se la
	 * cella oltre (il "muro") e' una pedina bianca, il Re, il trono (anche vuoto) oppure una citadel che non sia una
	 * delle 4 punte. Stesso ordine di direzioni di {@code AshtonRules.applyCaptures}.
	 */
	private void captureByWhite(int to, UndoInfo undo) {
		final byte[] b = board;
		for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
			if (BoardConstants.STEPS[d][to] < 2) {
				continue; // nessun vicino, oppure il muro oltre il vicino sarebbe fuori scacchiera
			}
			final int off = BoardConstants.DIR_OFFSET[d];
			final int adj = to + off;
			if (b[adj] != BoardConstants.BLACK) {
				continue;
			}
			final int wall = adj + off;
			final byte w = b[wall];
			if (w == BoardConstants.WHITE || w == BoardConstants.KING || (wall == BoardConstants.THRONE && w == BoardConstants.EMPTY)
					|| (BoardConstants.IS_CITADEL[wall] && !BoardConstants.IS_TIP[wall])) {
				b[adj] = BoardConstants.EMPTY;
				undo.addCapture(adj, BoardConstants.BLACK);
				blackCount--;
				zobristHash ^= ZobristHash.KEY[adj][BoardConstants.BLACK];
			}
		}
	}

	/**
	 * Catture del Nero attorno alla cella {@code to} appena occupata. Una pedina bianca adiacente e' catturata se il muro
	 * oltre e' una pedina nera, il trono (sempre ostile "per coordinata", anche se occupato dal Re) o una qualunque
	 * citadel. Il Re adiacente e' catturato secondo {@link #isKingCaptured}. Ritorna vero se il Re e' stato catturato.
	 */
	private boolean captureByBlack(int to, UndoInfo undo) {
		final byte[] b = board;
		boolean kingCaptured = false;
		for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
			final int steps = BoardConstants.STEPS[d][to];
			if (steps < 1) {
				continue;
			}
			final int off = BoardConstants.DIR_OFFSET[d];
			final int adj = to + off;
			final byte p = b[adj];
			if (p == BoardConstants.WHITE) {
				if (steps < 2) {
					continue;
				}
				final int wall = adj + off;
				if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {
					b[adj] = BoardConstants.EMPTY;
					undo.addCapture(adj, BoardConstants.WHITE);
					whiteCount--;
					zobristHash ^= ZobristHash.KEY[adj][BoardConstants.WHITE];
				}
			} else if (p == BoardConstants.KING) {
				if (isKingCaptured(adj, to, off, steps >= 2)) {
					b[adj] = BoardConstants.EMPTY;
					kingSquare = -1;
					kingCaptured = true;
					undo.addCapture(adj, BoardConstants.KING);
					zobristHash ^= ZobristHash.KEY[adj][BoardConstants.KING];
					if (adj == BoardConstants.THRONE) {
						zobristHash ^= ZobristHash.THRONE_EMPTY_KEY; // la cella 40 e' ora vuota
					}
				}
			}
		}
		return kingCaptured;
	}

	/**
	 * Il Re in {@code k}, adiacente alla pedina nera appena mossa in {@code m}, e' catturato? Tre casi:
	 * sul trono servono i 4 lati neri; adiacente al trono bastano i 3 lati liberi (il trono conta da solo come lato
	 * ostile); altrove cattura a tenaglia: la cella opposta a {@code m} (in direzione {@code off}) deve essere una pedina
	 * nera o una qualunque citadel. Il lato occupato dal pezzo appena mosso e' nero per definizione.
	 */
	private boolean isKingCaptured(int k, int m, int off, boolean hasCellBeyond) {
		final byte[] b = board;
		if (k == BoardConstants.THRONE) {
			for (int n : BoardConstants.THRONE_NEIGHBORS) {
				if (n != m && b[n] != BoardConstants.BLACK) {
					return false;
				}
			}
			return true;
		}
		if (BoardConstants.IS_THRONE_NEIGHBOR[k]) {
			for (int e = 0; e < BoardConstants.NUM_DIRS; e++) {
				if (BoardConstants.STEPS[e][k] < 1) {
					return false; // fuori scacchiera: non e' un lato nero (mai per una cella adiacente al trono)
				}
				final int n = k + BoardConstants.DIR_OFFSET[e];
				if (n == BoardConstants.THRONE || n == m) {
					continue;
				}
				if (b[n] != BoardConstants.BLACK) {
					return false;
				}
			}
			return true;
		}
		if (!hasCellBeyond) {
			return false;
		}
		final int f = k + off;
		return b[f] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[f];
	}

	// ------------------------------------------------------------------------------------------ debug

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(winner == WINNER_NONE ? (turn == BoardConstants.WHITE_TURN ? "Turno: BIANCO" : "Turno: NERO")
				: "Terminale: " + turnAsDomain()).append('\n');
		sb.append("  a b c d e f g h i\n");
		for (int r = BoardConstants.SIZE - 1; r >= 0; r--) {
			sb.append(r + 1).append(' ');
			for (int c = 0; c < BoardConstants.SIZE; c++) {
				final int sq = BoardConstants.coord(r, c);
				final byte p = board[sq];
				char ch = '.';
				if (p == BoardConstants.WHITE) {
					ch = 'W';
				} else if (p == BoardConstants.BLACK) {
					ch = 'B';
				} else if (p == BoardConstants.KING) {
					ch = 'K';
				} else if (sq == BoardConstants.THRONE) {
					ch = 'T';
				} else if (BoardConstants.IS_CITADEL[sq]) {
					ch = '#';
				}
				sb.append(ch).append(' ');
			}
			sb.append(r + 1).append('\n');
		}
		sb.append("  a b c d e f g h i\n");
		return sb.toString();
	}

	// ------------------------------------------------------------------------------------------ valore tattico

	/** Valore di {@link #tacticalValue} per una mossa che vince subito (fuga del Re o cattura del Re). */
	public static final int TACT_WIN = 100;

	/**
	 * Classifica una mossa LEGALE della posizione corrente SENZA eseguirla, solo per ordinare le mosse nella ricerca:
	 * {@link #TACT_WIN} se vince subito (il Re raggiunge il bordo, oppure la pedina nera cattura il Re), altrimenti il
	 * numero (0..3) di pedine avversarie (Re escluso) che {@link #makeMove} catturerebbe.
	 * <p>
	 * E' ESATTA (stessa risposta di {@code makeMove}: {@code FastStateCheck} la confronta con {@code UndoInfo} e
	 * {@code winner} su milioni di mosse) perche' le regole di cattura leggono solo le celle adiacenti alla destinazione e
	 * quelle subito oltre, e la cella di partenza non puo' mai essere tra queste: se lo fosse, il percorso della mossa
	 * attraverserebbe la pedina da catturare, che non sarebbe quindi un ostacolo vuoto. Non modifica la posizione. Costo: 4
	 * letture di celle per mossa se non c'e' nessun avversario adiacente (il caso comune).
	 */
	public int tacticalValue(int move) {
		final int from = move & 127;
		final int to = move >>> 7;
		final byte[] b = board;
		int count = 0;
		if (turn == BoardConstants.WHITE_TURN) {
			if (b[from] == BoardConstants.KING && BoardConstants.IS_EDGE[to]) {
				return TACT_WIN;
			}
			for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
				if (BoardConstants.STEPS[d][to] < 2) {
					continue;
				}
				final int off = BoardConstants.DIR_OFFSET[d];
				final int adj = to + off;
				if (b[adj] != BoardConstants.BLACK) {
					continue;
				}
				final int wall = adj + off;
				final byte w = b[wall];
				if (w == BoardConstants.WHITE || w == BoardConstants.KING || (wall == BoardConstants.THRONE && w == BoardConstants.EMPTY)
						|| (BoardConstants.IS_CITADEL[wall] && !BoardConstants.IS_TIP[wall])) {
					count++;
				}
			}
			return count;
		}
		for (int d = 0; d < BoardConstants.NUM_DIRS; d++) {
			final int steps = BoardConstants.STEPS[d][to];
			if (steps < 1) {
				continue;
			}
			final int off = BoardConstants.DIR_OFFSET[d];
			final int adj = to + off;
			final byte p = b[adj];
			if (p == BoardConstants.WHITE) {
				if (steps >= 2) {
					final int wall = adj + off;
					if (wall == BoardConstants.THRONE || b[wall] == BoardConstants.BLACK || BoardConstants.IS_CITADEL[wall]) {
						count++;
					}
				}
			} else if (p == BoardConstants.KING && isKingCaptured(adj, to, off, steps >= 2)) {
				return TACT_WIN;
			}
		}
		return count;
	}
}
