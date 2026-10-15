package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Reimplementazione "pura" delle regole Ashton Tablut (le stesse implementate in
 * {@code GameAshtonTablut}), pensata per essere usata come motore interno di un
 * algoritmo di ricerca (minimax/negamax).
 * <p>
 * PERCHE' QUESTA CLASSE ESISTE (e perche' NON si usa direttamente GameAshtonTablut
 * dentro la ricerca):
 * <ul>
 * <li>{@code GameAshtonTablut} scrive un file di log ad ogni singola mossa
 * (un {@link java.util.logging.FileHandler} viene aperto nel costruttore e
 * {@code loggGame.fine(...)} viene chiamato ripetutamente in {@code checkMove}).
 * In una ricerca negamax si generano facilmente decine/centinaia di migliaia di
 * nodi al secondo: usare quella classe nodo per nodo saturerebbe il disco in
 * pochi secondi (esattamente quello che le slide della competizione chiedono
 * di evitare).</li>
 * <li>{@code GameAshtonTablut} tiene un contatore interno (mosse senza catture)
 * e una cache di stati per il pareggio per ripetizione: sono stato mutabile
 * "di partita". Se lo stesso oggetto venisse riusato per migliaia di mosse
 * simulate nell'albero di ricerca, questo contatore verrebbe corrotto rispetto
 * alla partita reale.</li>
 * <li>{@code checkMove} lavora per eccezioni (try/catch): comodo per validare
 * UNA mossa proposta da un client esterno, molto costoso se lo si usa per
 * generare/validare migliaia di mosse candidate ad ogni nodo.</li>
 * </ul>
 * Questa classe replica quindi le stesse regole in modo puro (nessun I/O,
 * nessuna eccezione, nessuno stato condiviso) cosi' da poter essere chiamata
 * quanto necessario durante la ricerca.
 *
 * <p><b>Nota bene</b>: questa classe non tiene traccia della storia della
 * partita (deliberatamente: e' una libreria di funzioni pure su singole
 * posizioni), quindi non puo' da sola riconoscere un pareggio per ripetizione
 * di stato - quello dipende da quali posizioni sono gia' state visitate
 * PRIMA, un'informazione che vive fuori da questa classe. Il rilevamento
 * della ripetizione e' implementato a livello di ricerca in {@link Negamax}
 * (che tiene la storia del proprio percorso di ricerca) e nel client (che
 * tiene la storia della partita reale). Questa classe offre pero'
 * {@link #wasCapture} come mattone di supporto: la regola ufficiale
 * "un evento di cattura azzera la memoria delle posizioni ripetute" (nessuna
 * posizione precedente a una cattura puo' mai ripresentarsi, dato che nessun
 * pezzo catturato torna mai in gioco) richiede di sapere se una mossa ha
 * catturato qualcosa.
 */
public class AshtonRules {

	public static final int SIZE = 9;
	private static final int THRONE_R = 4;
	private static final int THRONE_C = 4;

	// N, S, W, E
	private static final int[][] DIRS = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

	// le 4 celle ortogonalmente adiacenti al trono (d5, f5, e4, e6)
	private static final int[][] THRONE_NEIGHBORS = { { 3, 4 }, { 5, 4 }, { 4, 3 }, { 4, 5 } };

	// citadels/camps, esattamente come definite in GameAshtonTablut
	private static final boolean[][] CITADEL = new boolean[SIZE][SIZE];
	// le 4 celle "punta" delle citadels (quelle che toccano il bordo): a5, e1, e9, i5
	// non contano come muro ostile quando il BIANCO cattura una pedina NERA
	private static final boolean[][] CITADEL_TIP = new boolean[SIZE][SIZE];

	static {
		int[][] citadelCells = { { 3, 0 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, // a4,a5,a6,b5
				{ 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, // d1,e1,f1,e2
				{ 3, 8 }, { 4, 8 }, { 5, 8 }, { 4, 7 }, // i4,i5,i6,h5
				{ 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 } // d9,e9,f9,e8
		};
		for (int[] cell : citadelCells) {
			CITADEL[cell[0]][cell[1]] = true;
		}
		int[][] tips = { { 4, 0 }, { 0, 4 }, { 8, 4 }, { 4, 8 } }; // a5, e1, e9, i5
		for (int[] cell : tips) {
			CITADEL_TIP[cell[0]][cell[1]] = true;
		}
	}

	private AshtonRules() {
	}

	private static boolean inBounds(int r, int c) {
		return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
	}

	public static boolean isTerminal(State state) {
		Turn t = state.getTurn();
		return t == Turn.WHITEWIN || t == Turn.BLACKWIN || t == Turn.DRAW;
	}

	private static Action newAction(State state, int rf, int cf, int rt, int ct, Turn turn) {
		try {
			return new Action(state.getBox(rf, cf), state.getBox(rt, ct), turn);
		} catch (IOException e) {
			// non puo' accadere: getBox produce sempre stringhe di lunghezza 2
			throw new IllegalStateException(e);
		}
	}

	/**
	 * Genera tutte le mosse legali per il giocatore che deve muovere in
	 * {@code state} ({@code state.getTurn()}). Se lo stato e' terminale (partita
	 * finita) ritorna una lista vuota.
	 */
	public static List<Action> generateMoves(State state) {
		List<Action> moves = new ArrayList<>();
		Turn turn = state.getTurn();
		if (turn != Turn.WHITE && turn != Turn.BLACK) {
			return moves;
		}
		Pawn[][] b = state.getBoard();

		for (int r = 0; r < SIZE; r++) {
			for (int c = 0; c < SIZE; c++) {
				Pawn p = b[r][c];
				boolean mine = (turn == Turn.WHITE) ? (p == Pawn.WHITE || p == Pawn.KING) : (p == Pawn.BLACK);
				if (!mine) {
					continue;
				}
				boolean originCitadel = CITADEL[r][c];

				for (int[] d : DIRS) {
					int step = 1;
					while (true) {
						int nr = r + d[0] * step;
						int nc = c + d[1] * step;
						if (!inBounds(nr, nc)) {
							break;
						}
						Pawn target = b[nr][nc];
						// qualsiasi cella non vuota blocca il movimento: sia un'altra pedina,
						// sia il trono (occupato dal Re oppure "vuoto" ma marcato THRONE)
						if (target != Pawn.EMPTY) {
							break;
						}
						boolean targetCitadel = CITADEL[nr][nc];
						if (targetCitadel) {
							if (!originCitadel) {
								// non si puo' ne' atterrare ne' attraversare una citadel se non si
								// e' partiti da una citadel (eccezione valida solo per il Nero
								// che non ha ancora lasciato il proprio accampamento)
								break;
							}
							int dist = (d[0] != 0) ? Math.abs(nr - r) : Math.abs(nc - c);
							if (dist <= 5) {
								moves.add(newAction(state, r, c, nr, nc, turn));
							} else {
								// la distanza cresce monotonamente lungo la direzione: da qui in poi
								// sarebbe comunque > 5
								break;
							}
						} else {
							moves.add(newAction(state, r, c, nr, nc, turn));
						}
						step++;
					}
				}
			}
		}
		return moves;
	}

	/**
	 * Applica una mossa (gia' assunta legale, tipicamente generata da
	 * {@link #generateMoves}) a una COPIA dello stato, gestendo spostamento,
	 * catture (comprese le regole speciali del Re vicino al trono) e condizioni
	 * di vittoria per fuga del Re o cattura del Re. Lo stato originale non viene
	 * modificato.
	 */
	public static State applyMove(State state, Action a) {
		State newState = state.clone();
		Pawn[][] b = newState.getBoard();
		Turn mover = newState.getTurn();

		int rf = a.getRowFrom(), cf = a.getColumnFrom();
		int rt = a.getRowTo(), ct = a.getColumnTo();

		Pawn moving = b[rf][cf];
		b[rf][cf] = (rf == THRONE_R && cf == THRONE_C) ? Pawn.THRONE : Pawn.EMPTY;
		b[rt][ct] = moving;

		boolean whiteEscapes = mover == Turn.WHITE && moving == Pawn.KING
				&& (rt == 0 || rt == SIZE - 1 || ct == 0 || ct == SIZE - 1);

		boolean kingCaptured = applyCaptures(b, rt, ct, mover);

		if (whiteEscapes) {
			newState.setTurn(Turn.WHITEWIN);
		} else if (kingCaptured) {
			newState.setTurn(Turn.BLACKWIN);
		} else {
			newState.setTurn(mover == Turn.WHITE ? Turn.BLACK : Turn.WHITE);
		}
		return newState;
	}

	/**
	 * Vero se la mossa {@code a} (applicata per ottenere {@code after} da
	 * {@code before}) ha catturato almeno una pedina. Guarda solo l'origine e
	 * i 4 vicini della destinazione (le uniche celle che una mossa puo'
	 * modificare, si veda {@link ZobristHasher} per la stessa osservazione
	 * applicata all'hashing): se una di quelle celle aveva una pedina prima e
	 * ora e' vuota, escludendo l'origine stessa (che diventa vuota semplicemente
	 * perche' il pezzo se n'e' andato, non per una cattura), allora c'e' stata
	 * una cattura.
	 */
	public static boolean wasCapture(State before, State after, Action a) {
		Pawn[][] bBefore = before.getBoard();
		Pawn[][] bAfter = after.getBoard();
		int rf = a.getRowFrom(), cf = a.getColumnFrom();
		int rt = a.getRowTo(), ct = a.getColumnTo();

		for (int[] d : DIRS) {
			int nr = rt + d[0], nc = ct + d[1];
			if (nr == rf && nc == cf) {
				continue; // e' l'origine: si svuota perche' il pezzo si e' spostato, non e' una cattura
			}
			if (inBounds(nr, nc) && bBefore[nr][nc] != Pawn.EMPTY && bAfter[nr][nc] == Pawn.EMPTY) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Controlla ed esegue le catture intorno alla cella (mr,mc) appena occupata
	 * dal pezzo mosso dal giocatore {@code mover}. Ritorna true se il Re e'
	 * stato catturato (evento possibile solo se {@code mover == BLACK}).
	 */
	private static boolean applyCaptures(Pawn[][] b, int mr, int mc, Turn mover) {
		boolean kingCaptured = false;
		for (int[] d : DIRS) {
			int ar = mr + d[0], ac = mc + d[1];
			if (!inBounds(ar, ac)) {
				continue;
			}
			Pawn adj = b[ar][ac];
			if (mover == Turn.WHITE) {
				if (adj == Pawn.BLACK) {
					int wr = mr + 2 * d[0], wc = mc + 2 * d[1];
					if (isWallForWhiteCapture(b, wr, wc)) {
						b[ar][ac] = Pawn.EMPTY;
					}
				}
			} else { // mover == BLACK
				if (adj == Pawn.WHITE) {
					int wr = mr + 2 * d[0], wc = mc + 2 * d[1];
					if (isWallForBlackCapture(b, wr, wc)) {
						b[ar][ac] = Pawn.EMPTY;
					}
				} else if (adj == Pawn.KING) {
					if (isKingCaptured(b, ar, ac, mr, mc)) {
						b[ar][ac] = Pawn.EMPTY;
						kingCaptured = true;
					}
				}
			}
		}
		return kingCaptured;
	}

	/** Muro valido per il BIANCO che cattura una pedina NERA: W, T, K, oppure citadel (tranne le 4 punte). */
	private static boolean isWallForWhiteCapture(Pawn[][] b, int r, int c) {
		if (!inBounds(r, c)) {
			return false;
		}
		Pawn w = b[r][c];
		if (w == Pawn.WHITE || w == Pawn.THRONE || w == Pawn.KING) {
			return true;
		}
		return CITADEL[r][c] && !CITADEL_TIP[r][c];
	}

	/**
	 * Muro valido per il NERO che cattura una pedina BIANCA (non Re): B, T,
	 * qualsiasi citadel, oppure la casella del trono stessa (anche quando e'
	 * occupata dal Re, cioe' contiene K e non T: nel motore ufficiale il trono
	 * e' sempre un muro ostile per il Bianco "per coordinata", non per
	 * contenuto - si veda il controllo esplicito su "e5" in
	 * GameAshtonTablut.checkCaptureBlackPawn*).
	 */
	private static boolean isWallForBlackCapture(Pawn[][] b, int r, int c) {
		if (!inBounds(r, c)) {
			return false;
		}
		if (r == THRONE_R && c == THRONE_C) {
			return true; // trono sempre ostile per il Bianco, occupato dal Re o no
		}
		Pawn w = b[r][c];
		if (w == Pawn.BLACK || w == Pawn.THRONE) {
			return true;
		}
		return CITADEL[r][c];
	}

	/**
	 * Determina se il Re, che si trova in (kr,kc) ed e' adiacente al pezzo nero
	 * appena mosso in (mr,mc), risulta catturato. Gestisce i tre casi: Re sul
	 * trono (serve l'accerchiamento su tutti e 4 i lati), Re adiacente al trono
	 * (bastano i 3 lati liberi, il trono conta come "muro" automaticamente),
	 * Re altrove (cattura a tenaglia classica, dove il muro opposto puo' essere
	 * una pedina nera o una qualsiasi citadel, senza eccezioni per le punte).
	 */
	private static boolean isKingCaptured(Pawn[][] b, int kr, int kc, int mr, int mc) {
		if (kr == THRONE_R && kc == THRONE_C) {
			for (int[] n : THRONE_NEIGHBORS) {
				if (n[0] == mr && n[1] == mc) {
					continue; // lato del pezzo che ha appena mosso: nero per definizione
				}
				if (b[n[0]][n[1]] != Pawn.BLACK) {
					return false;
				}
			}
			return true;
		}
		if (isThroneNeighbor(kr, kc)) {
			for (int[] d : DIRS) {
				int nr = kr + d[0], nc = kc + d[1];
				if (nr == THRONE_R && nc == THRONE_C) {
					continue; // il trono vuoto/occupato conta comunque come lato ostile
				}
				if (nr == mr && nc == mc) {
					continue; // il pezzo appena mosso: nero per definizione
				}
				if (!inBounds(nr, nc) || b[nr][nc] != Pawn.BLACK) {
					return false;
				}
			}
			return true;
		}
		// Re "altrove": tenaglia classica sui due lati opposti
		int fr = 2 * kr - mr, fc = 2 * kc - mc;
		if (!inBounds(fr, fc)) {
			return false;
		}
		return b[fr][fc] == Pawn.BLACK || CITADEL[fr][fc];
	}

	private static boolean isThroneNeighbor(int r, int c) {
		for (int[] n : THRONE_NEIGHBORS) {
			if (n[0] == r && n[1] == c) {
				return true;
			}
		}
		return false;
	}
}
