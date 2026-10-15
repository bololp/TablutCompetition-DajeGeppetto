package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * Zobrist hashing per Ashton Tablut: una chiave a 64 bit per ogni combinazione
 * (cella, tipo di pedina), XORate insieme per ottenere l'hash di una board.
 * Due tabelle diverse rappresentano la stessa posizione con lo stesso hash
 * (a meno di collisioni, statisticamente trascurabili con 64 bit).
 * <p>
 * <b>{@link #fullHash(State)}</b>: calcolo completo, O(81). Da usare UNA
 * volta alla radice di ogni ricerca (in {@link Negamax#findBestMove}), mai
 * dentro la ricorsione.
 * <p>
 * <b>{@link #updateHash(long, State, State, Action)}</b>: aggiornamento
 * incrementale, esamina solo le (al massimo) 6 celle che una mossa puo'
 * effettivamente modificare: l'origine, la destinazione, e i 4 vicini
 * ortogonali della destinazione. Questo non e' un limite arbitrario: guardando
 * {@link AshtonRules#applyMove}, le uniche scritture sulla board sono
 * esattamente su queste celle - le celle "muro" usate per determinare se una
 * cattura avviene (a volte piu' lontane, es. i vicini del trono per il Re)
 * vengono SOLO LETTE, mai scritte. Percio' non serve conoscere i dettagli
 * interni di {@link AshtonRules} per calcolare l'hash incrementale: bastano
 * l'{@link Action} e gli stati prima/dopo, entrambi gia' disponibili a chi
 * chiama {@link AshtonRules#applyMove}.
 */
public final class ZobristHasher {

	private static final int NUM_PAWN_TYPES = 4; // WHITE, BLACK, KING, THRONE (EMPTY non ha bisogno di una chiave: XOR con 0)
	private static final long[][][] PIECE_KEY = new long[AshtonRules.SIZE][AshtonRules.SIZE][NUM_PAWN_TYPES];
	private static final long TURN_KEY;
	private static final int[][] NEIGHBOR_DIRS = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

	static {
		// Seed fissa: la casualita' qui serve solo per una buona distribuzione
		// delle chiavi, non per sicurezza - un seed fissa rende l'hashing
		// riproducibile tra esecuzioni diverse, comodo per debug.
		Random rnd = new Random(0xC0FFEEL);
		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				for (int t = 0; t < NUM_PAWN_TYPES; t++) {
					PIECE_KEY[r][c][t] = rnd.nextLong();
				}
			}
		}
		TURN_KEY = rnd.nextLong();
	}

	private ZobristHasher() {
	}

	private static int pawnIndex(Pawn p) {
		if (p == Pawn.WHITE) {
			return 0;
		}
		if (p == Pawn.BLACK) {
			return 1;
		}
		if (p == Pawn.KING) {
			return 2;
		}
		if (p == Pawn.THRONE) {
			return 3;
		}
		return -1; // EMPTY
	}

	private static long keyFor(int r, int c, Pawn p) {
		int idx = pawnIndex(p);
		return idx < 0 ? 0L : PIECE_KEY[r][c][idx];
	}

	/** Calcolo completo dell'hash, O(81). Da usare una sola volta alla radice. */
	public static long fullHash(State state) {
		long h = 0L;
		Pawn[][] b = state.getBoard();
		for (int r = 0; r < AshtonRules.SIZE; r++) {
			for (int c = 0; c < AshtonRules.SIZE; c++) {
				h ^= keyFor(r, c, b[r][c]);
			}
		}
		if (state.getTurn() == Turn.BLACK) {
			h ^= TURN_KEY;
		}
		return h;
	}

	/**
	 * Aggiornamento incrementale dopo una mossa: esamina solo origine,
	 * destinazione e i 4 vicini della destinazione (si veda il javadoc della
	 * classe per la motivazione). {@code before} e {@code after} sono gli
	 * stati rispettivamente prima e dopo {@code a} (tipicamente
	 * {@code before} e il risultato di {@code AshtonRules.applyMove(before, a)});
	 * non vengono modificati, solo letti.
	 */
	public static long updateHash(long hashBefore, State before, State after, Action a) {
		long h = hashBefore;
		Pawn[][] bBefore = before.getBoard();
		Pawn[][] bAfter = after.getBoard();

		int rf = a.getRowFrom(), cf = a.getColumnFrom();
		int rt = a.getRowTo(), ct = a.getColumnTo();

		h = xorIfChanged(h, rf, cf, bBefore, bAfter);
		h = xorIfChanged(h, rt, ct, bBefore, bAfter);
		for (int[] d : NEIGHBOR_DIRS) {
			int nr = rt + d[0], nc = ct + d[1];
			if (nr == rf && nc == cf) {
				continue; // gia' processata come origine: su una mossa di una sola
							// casella l'origine COINCIDE con uno dei 4 vicini della
							// destinazione - ricontrollarla la XORerebbe due volte,
							// annullando l'aggiornamento (bug reale, trovato dal test)
			}
			if (nr >= 0 && nr < AshtonRules.SIZE && nc >= 0 && nc < AshtonRules.SIZE) {
				h = xorIfChanged(h, nr, nc, bBefore, bAfter);
			}
		}
		// NON un semplice "toggle": il turno non alterna sempre tra WHITE e BLACK,
		// puo' anche diventare WHITEWIN/BLACKWIN (partita finita con una mossa
		// vincente, es. il Re scappa). Bisogna XORare il contributo del turno
		// PRIMA (per toglierlo) e quello DOPO (per aggiungerlo), esattamente
		// come per una cella qualsiasi - un toggle incondizionato e' sbagliato
		// ogni volta che lo stato risultante e' terminale (trovato dal test).
		h ^= turnContribution(before.getTurn()) ^ turnContribution(after.getTurn());
		return h;
	}

	/** Stesso contributo del turno usato in {@link #fullHash}: solo BLACK ha una
	 *  chiave dedicata (WHITE, e qualunque stato terminale, contribuiscono 0). */
	private static long turnContribution(Turn t) {
		return t == Turn.BLACK ? TURN_KEY : 0L;
	}

	private static long xorIfChanged(long h, int r, int c, Pawn[][] before, Pawn[][] after) {
		if (before[r][c] != after[r][c]) {
			h ^= keyFor(r, c, before[r][c]);
			h ^= keyFor(r, c, after[r][c]);
		}
		return h;
	}
}
