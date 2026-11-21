package it.unibo.ai.didattica.competition.tablut.aiplayer;

/**
 * Configurazione, IMMUTABILE, delle tecniche di riduzione dell'albero di {@link Negamax}: potatura con mossa nulla (null-move), riduzione delle
 * mosse tardive (LMR) e finestre di aspirazione alla radice. Sono tutte SPENTE nel default ({@link #OFF}): con {@code OFF} {@code Negamax} e'
 * identico, nodo per nodo, a quello senza queste tecniche ({@code SearchEquivalenceCheck}, {@code PVSCheck} ecc. lo verificano).
 *
 * <h3>Perche' sono tecniche "con rischio" e cosa si fa per contenerlo</h3>
 * L'alpha-beta e le sue ottimizzazioni gia' presenti (PVS, ordinamento, TT) non cambiano il valore della radice: queste SI'. Null-move e LMR
 * sono euristiche: possono scartare una mossa buona. Il rischio e' ridotto cosi' (dettagli e motivazioni nel Javadoc di {@link Negamax}):
 * <ul>
 * <li><b>mai su una vittoria provata</b>: nessuna riduzione/mossa nulla quando una finestra contiene punteggi di vittoria; ogni punteggio di vittoria che
 * entra in un nodo proviene da una ricerca a profondita' PIENA; dopo che una ricerca ha trovato una vittoria (o una sconfitta) forzata la ricerca
 * successiva e' interamente senza queste tecniche (cosi' resta valida la garanzia "la distanza dalla vittoria scende ad ogni mossa");</li>
 * <li><b>mai su posizioni "forzanti"</b> (il Re ha una via libera verso il bordo con il Nero al tratto; il Re e' accanto a una pedina nera con il
 * Bianco al tratto) ne' in finali con pochi pezzi (rischio di zugzwang);</li>
 * <li><b>mai ridotte</b> le mosse che catturano, che vincono, le mosse del Re, la mossa della TT, le killer/countermove;</li>
 * <li>un figlio ridotto ha sempre profondita' &gt;= 1, cosi' vede ancora le vittorie immediate dell'avversario (la ricerca non ha quiescenza);</li>
 * <li>i risultati che dipendono dalla mossa nulla NON entrano mai nella transposition table; la storia di ripetizione non viene alterata.</li>
 * </ul>
 *
 * <h3>Testo ({@link #parse})</h3>
 * Elenco separato da virgole: {@code off} | {@code all} | {@code nmp}, {@code lmr}, {@code asp} (accendono le tre tecniche) piu' parametri
 * {@code chiave=valore}: {@code nmp.r} (riduzione della mossa nulla, 2), {@code nmp.depth} (profondita' minima, 4), {@code nmp.pieces} (pezzi
 * minimi del lato al tratto, 5), {@code nmp.verify} (da questa profondita' la potatura e' verificata con una ricerca senza mossa nulla, 10; 0 = mai),
 * {@code lmr.depth} (3), {@code lmr.moves} (mosse piene prima di ridurre, 4), {@code asp.depth} (5), {@code asp.delta} (40).
 * Esempio: {@code "nmp,lmr,asp,nmp.r=3,asp.delta=60"}.
 */
public final class PruningOptions {

	/** Tutto spento: la ricerca storica. */
	public static final PruningOptions OFF = new PruningOptions(false, false, false, 2, 4, 5, 10, 3, 4, 5, 40);

	public final boolean nullMove, lmr, aspiration;
	/** Riduzione R della mossa nulla: il figlio nullo si cerca a {@code depth - 1 - R}. */
	public final int nullR;
	/** Profondita' minima del nodo per tentare la mossa nulla (almeno {@code nullR + 2}: il figlio nullo ha profondita' >= 1). */
	public final int nullMinDepth;
	/** Pezzi minimi del lato al tratto (Re escluso) per tentare la mossa nulla: sotto, il rischio di zugzwang e' alto. */
	public final int nullMinPieces;
	/** Da questa profondita' una potatura con mossa nulla e' VERIFICATA da una ricerca a {@code depth - R} senza mosse nulle; 0 = mai. */
	public final int nullVerifyDepth;
	/** Profondita' minima del nodo per ridurre una mossa tardiva (almeno 3: il figlio ridotto ha profondita' >= 1). */
	public final int lmrMinDepth;
	/** Le prime {@code lmrMinMoves} mosse di un nodo non si riducono mai. */
	public final int lmrMinMoves;
	/** Profondita' minima dell'iterazione per usare la finestra di aspirazione. */
	public final int aspMinDepth;
	/** Semi-ampiezza minima della finestra di aspirazione (unita' del punteggio, scala +/-1000 per la NNUE). */
	public final int aspDelta;

	public PruningOptions(boolean nullMove, boolean lmr, boolean aspiration, int nullR, int nullMinDepth, int nullMinPieces, int nullVerifyDepth, int lmrMinDepth, int lmrMinMoves,
			int aspMinDepth, int aspDelta) {
		if (nullR < 1 || nullR > 6) {
			throw new IllegalArgumentException("nmp.r deve essere in [1,6]: " + nullR);
		}
		if (nullMinDepth < nullR + 2 || nullMinDepth > 40) {
			throw new IllegalArgumentException("nmp.depth deve essere in [nmp.r+2, 40] (il figlio nullo deve avere profondita' >= 1): " + nullMinDepth);
		}
		if (nullMinPieces < 0 || nullMinPieces > 16) {
			throw new IllegalArgumentException("nmp.pieces deve essere in [0,16]: " + nullMinPieces);
		}
		if (nullVerifyDepth < 0 || (nullVerifyDepth > 0 && nullVerifyDepth < nullMinDepth)) {
			throw new IllegalArgumentException("nmp.verify deve essere 0 oppure >= nmp.depth: " + nullVerifyDepth);
		}
		if (lmrMinDepth < 3 || lmrMinDepth > 40) {
			throw new IllegalArgumentException("lmr.depth deve essere in [3,40] (il figlio ridotto deve avere profondita' >= 1): " + lmrMinDepth);
		}
		if (lmrMinMoves < 2 || lmrMinMoves > 64) {
			throw new IllegalArgumentException("lmr.moves deve essere in [2,64] (la prima mossa non si riduce mai): " + lmrMinMoves);
		}
		if (aspMinDepth < 2 || aspMinDepth > 40) {
			throw new IllegalArgumentException("asp.depth deve essere in [2,40]: " + aspMinDepth);
		}
		if (aspDelta < 1 || aspDelta > 100000) {
			throw new IllegalArgumentException("asp.delta deve essere in [1,100000]: " + aspDelta);
		}
		this.nullMove = nullMove;
		this.lmr = lmr;
		this.aspiration = aspiration;
		this.nullR = nullR;
		this.nullMinDepth = nullMinDepth;
		this.nullMinPieces = nullMinPieces;
		this.nullVerifyDepth = nullVerifyDepth;
		this.lmrMinDepth = lmrMinDepth;
		this.lmrMinMoves = lmrMinMoves;
		this.aspMinDepth = aspMinDepth;
		this.aspDelta = aspDelta;
	}

	/** I valori di default dei parametri con le tre tecniche accese o spente come indicato. */
	public static PruningOptions of(boolean nullMove, boolean lmr, boolean aspiration) {
		return new PruningOptions(nullMove, lmr, aspiration, OFF.nullR, OFF.nullMinDepth, OFF.nullMinPieces, OFF.nullVerifyDepth, OFF.lmrMinDepth, OFF.lmrMinMoves, OFF.aspMinDepth, OFF.aspDelta);
	}

	/** Vero se almeno una tecnica e' accesa. */
	public boolean any() {
		return nullMove || lmr || aspiration;
	}

	/**
	 * @throws IllegalArgumentException token sconosciuto, valore non numerico o fuori intervallo
	 */
	public static PruningOptions parse(String spec) {
		if (spec == null || spec.trim().isEmpty()) {
			throw new IllegalArgumentException("specifica di potatura vuota (usare 'off' per spegnere)");
		}
		boolean nmp = false, lmr = false, asp = false;
		int r = OFF.nullR, nd = OFF.nullMinDepth, np = OFF.nullMinPieces, nv = OFF.nullVerifyDepth, ld = OFF.lmrMinDepth, lm = OFF.lmrMinMoves, ad = OFF.aspMinDepth, adl = OFF.aspDelta;
		for (String raw : spec.split(",")) {
			final String tok = raw.trim().toLowerCase();
			if (tok.isEmpty()) {
				throw new IllegalArgumentException("elemento vuoto in '" + spec + "'");
			}
			final int eq = tok.indexOf('=');
			if (eq < 0) {
				switch (tok) {
				case "off":
					break;
				case "all":
					nmp = lmr = asp = true;
					break;
				case "nmp":
					nmp = true;
					break;
				case "lmr":
					lmr = true;
					break;
				case "asp":
					asp = true;
					break;
				default:
					throw new IllegalArgumentException("token di potatura sconosciuto '" + tok + "' (ammessi: off, all, nmp, lmr, asp, chiave=valore)");
				}
				continue;
			}
			final String k = tok.substring(0, eq);
			final int v;
			try {
				v = Integer.parseInt(tok.substring(eq + 1));
			} catch (NumberFormatException e) {
				throw new IllegalArgumentException("valore non numerico in '" + tok + "'");
			}
			switch (k) {
			case "nmp.r":
				r = v;
				break;
			case "nmp.depth":
				nd = v;
				break;
			case "nmp.pieces":
				np = v;
				break;
			case "nmp.verify":
				nv = v;
				break;
			case "lmr.depth":
				ld = v;
				break;
			case "lmr.moves":
				lm = v;
				break;
			case "asp.depth":
				ad = v;
				break;
			case "asp.delta":
				adl = v;
				break;
			default:
				throw new IllegalArgumentException("parametro di potatura sconosciuto '" + k + "'");
			}
		}
		return new PruningOptions(nmp, lmr, asp, r, nd, np, nv, ld, lm, ad, adl);
	}

	/** Descrizione in una riga, nello stesso formato di {@link #parse} (ricaricabile). */
	@Override
	public String toString() {
		if (!any()) {
			return "off";
		}
		final StringBuilder sb = new StringBuilder();
		if (nullMove) {
			sb.append("nmp,nmp.r=").append(nullR).append(",nmp.depth=").append(nullMinDepth).append(",nmp.pieces=").append(nullMinPieces).append(",nmp.verify=").append(nullVerifyDepth);
		}
		if (lmr) {
			sb.append(sb.length() > 0 ? "," : "").append("lmr,lmr.depth=").append(lmrMinDepth).append(",lmr.moves=").append(lmrMinMoves);
		}
		if (aspiration) {
			sb.append(sb.length() > 0 ? "," : "").append("asp,asp.depth=").append(aspMinDepth).append(",asp.delta=").append(aspDelta);
		}
		return sb.toString();
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof PruningOptions)) {
			return false;
		}
		final PruningOptions p = (PruningOptions) o;
		return nullMove == p.nullMove && lmr == p.lmr && aspiration == p.aspiration && nullR == p.nullR && nullMinDepth == p.nullMinDepth && nullMinPieces == p.nullMinPieces
				&& nullVerifyDepth == p.nullVerifyDepth && lmrMinDepth == p.lmrMinDepth && lmrMinMoves == p.lmrMinMoves && aspMinDepth == p.aspMinDepth && aspDelta == p.aspDelta;
	}

	@Override
	public int hashCode() {
		int h = (nullMove ? 1 : 0) | (lmr ? 2 : 0) | (aspiration ? 4 : 0);
		for (int v : new int[] { nullR, nullMinDepth, nullMinPieces, nullVerifyDepth, lmrMinDepth, lmrMinMoves, aspMinDepth, aspDelta }) {
			h = h * 31 + v;
		}
		return h;
	}
}
