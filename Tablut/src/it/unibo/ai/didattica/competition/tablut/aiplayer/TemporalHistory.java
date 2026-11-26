package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;

/**
 * Schema delle feature TEMPORALI di {@link TemporalNnueNetwork}: descrivono le ultime {@code K} semi-mosse (ply) che hanno portato alla posizione.
 * E' l'unico punto che definisce indici e impacchettamento, usato sia dal trainer ({@code NnueTemporalTrainer}) sia dalla ricerca
 * ({@link TemporalNnueEvaluator}): cosi' i due non possono divergere.
 *
 * <h3>Record di una mossa</h3>
 * Una mossa passata e' un {@code int} {@code rec} (0 = "ignota / non disponibile"):
 * <pre>
 *   bit  0..6   cella di partenza (0..80)
 *   bit  7..13  cella di arrivo   (0..80)
 *   bit 14..15  tipo di pedina mossa: 0 bianco, 1 nero, 2 re   (= {@code NnueNetwork.baseOf(pedina) / 81})
 *   bit 16..17  pedine catturate da QUELLA mossa (0..3, saturato)
 *   bit 20      valido
 * </pre>
 *
 * <h3>Feature (sparse, aggiunte allo STESSO accumulatore delle feature di posizione)</h3>
 * Per ogni eta' {@code a} (0 = l'ultima mossa giocata, 1 = quella prima, ...): {@link #PER_AGE} = 488 feature:
 * <pre>
 *   a*488 + tipo*81 + da            (0..242)    da dove e' partita la pedina
 *   a*488 + 243 + tipo*81 + a_cella (243..485)  dove e' arrivata
 *   a*488 + 486                     catturata esattamente 1 pedina
 *   a*488 + 487                     catturate 2 o piu' pedine
 * </pre>
 * Al piu' 3 feature attive per mossa. La finestra e' sempre una SUFFISSO della storia vera: se si conoscono solo le {@code j} mosse piu'
 * recenti (es. dentro la ricerca, dove il genitore del primo nodo e' noto ma la partita prima della radice no) si passano quelle e basta,
 * senza buchi: il trainer addestra proprio su finestre troncate in modo casuale ("history dropout") perche' la rete sia corretta in
 * entrambi i casi.
 */
public final class TemporalHistory {

	private TemporalHistory() {
	}

	/** Massimo numero di mosse nella finestra. */
	public static final int MAX_AGES = 8;
	/** Feature per eta': 243 partenza + 243 arrivo + 2 catture. */
	public static final int PER_AGE = 2 * NnueNetwork.NUM_FEATURES + 2;

	private static final int VALID = 1 << 20;

	/** Numero di feature temporali per una finestra di {@code k} mosse. */
	public static int numFeatures(int k) {
		return k * PER_AGE;
	}

	/**
	 * Impacchetta una mossa.
	 *
	 * @param type 0 bianco, 1 nero, 2 re
	 * @param caps pedine catturate (saturato a 3)
	 */
	public static int pack(int type, int from, int to, int caps) {
		if (type < 0 || type > 2 || from < 0 || from >= BoardConstants.NUM_SQUARES || to < 0 || to >= BoardConstants.NUM_SQUARES || caps < 0) {
			throw new IllegalArgumentException("record non valido: tipo " + type + " da " + from + " a " + to + " catture " + caps);
		}
		return VALID | (Math.min(caps, 3) << 16) | (type << 14) | (to << 7) | from;
	}

	/** Record di una mossa dato il CODICE della pedina mossa ({@link BoardConstants}); 0 se il codice non ha feature o le celle sono fuori scacchiera. */
	public static int record(int piece, int from, int to, int caps) {
		final int b = NnueNetwork.baseOf(piece);
		if (b < 0 || from < 0 || from >= BoardConstants.NUM_SQUARES || to < 0 || to >= BoardConstants.NUM_SQUARES || caps < 0) {
			return 0;
		}
		return pack(b / BoardConstants.NUM_SQUARES, from, to, caps);
	}

	public static boolean isValid(int rec) {
		return (rec & VALID) != 0;
	}

	public static int type(int rec) {
		return (rec >>> 14) & 3;
	}

	public static int from(int rec) {
		return rec & 127;
	}

	public static int to(int rec) {
		return (rec >>> 7) & 127;
	}

	public static int caps(int rec) {
		return (rec >>> 16) & 3;
	}

	/**
	 * Scrive in {@code out} gli indici delle feature temporali attive di una finestra.
	 *
	 * @param recs  record, il PIU' RECENTE per primo ({@code recs[off]} ha eta' 0)
	 * @param len   quanti record leggere da {@code recs[off..]}; se {@code > k} si usano solo i primi {@code k}; un record nullo TERMINA la finestra
	 * @param map   permutazione delle celle (simmetria, {@code BoardSymmetry.MAP[s]}) o {@code null} per l'identita'
	 * @param out   destinazione (almeno {@code 3 * k} posti da {@code outOff})
	 * @return numero di indici scritti
	 */
	public static int activeFeatures(int[] recs, int off, int len, int k, int[] map, int[] out, int outOff) {
		int n = 0;
		final int m = Math.min(len, k);
		for (int a = 0; a < m; a++) {
			final int r = recs[off + a];
			if (!isValid(r)) {
				break;
			}
			final int base = a * PER_AGE;
			final int t = type(r) * BoardConstants.NUM_SQUARES;
			final int f = from(r), to = to(r);
			out[outOff + n++] = base + t + (map == null ? f : map[f]);
			out[outOff + n++] = base + NnueNetwork.NUM_FEATURES + t + (map == null ? to : map[to]);
			final int c = caps(r);
			if (c >= 1) {
				out[outOff + n++] = base + 2 * NnueNetwork.NUM_FEATURES + (c >= 2 ? 1 : 0);
			}
		}
		return n;
	}

	/**
	 * Ricava il record della mossa che ha portato dalla scacchiera {@code prev} alla scacchiera {@code cur}, confrontandole (serve a ricostruire
	 * la storia da dati che hanno solo posizioni: {@code NnueDataset}, log del server).
	 *
	 * @param moverRole lato che ha mosso: 0 = Bianco (pedine bianche e Re), 1 = Nero
	 * @return il record, oppure 0 se le due scacchiere NON sono collegate da una sola mossa legale-per-forma (nessuna pedina del lato mossa o piu'
	 *         di una cambia cella, una pedina cambia tipo, compare una pedina avversaria, ...): il chiamante tratta la storia come interrotta
	 */
	public static int deriveRecord(byte[] prev, int prevOff, byte[] cur, int curOff, int moverRole) {
		int from = -1, to = -1, fromPiece = 0, toPiece = 0, caps = 0;
		for (int sq = 0; sq < BoardConstants.NUM_SQUARES; sq++) {
			final int p = prev[prevOff + sq], c = cur[curOff + sq];
			if (p == c) {
				continue;
			}
			if (p != BoardConstants.EMPTY && c == BoardConstants.EMPTY) {
				if (isMover(p, moverRole)) {
					if (from >= 0) {
						return 0;
					}
					from = sq;
					fromPiece = p;
				} else {
					caps++; // una pedina avversaria sparisce: catturata
				}
			} else if (p == BoardConstants.EMPTY && c != BoardConstants.EMPTY) {
				if (!isMover(c, moverRole) || to >= 0) {
					return 0;
				}
				to = sq;
				toPiece = c;
			} else {
				return 0; // una cella cambia da una pedina a un'altra: non e' una mossa
			}
		}
		if (from < 0 || to < 0 || fromPiece != toPiece) {
			return 0;
		}
		return record(fromPiece, from, to, caps);
	}

	private static boolean isMover(int piece, int role) {
		return role == 0 ? (piece == BoardConstants.WHITE || piece == BoardConstants.KING) : piece == BoardConstants.BLACK;
	}
}
