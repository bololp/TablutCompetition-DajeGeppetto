package it.unibo.ai.didattica.competition.tablut.training;

import it.unibo.ai.didattica.competition.tablut.aiplayer.TemporalHistory;

/**
 * Le storie di un {@link NnueDataset}: per ogni campione le ultime {@code K} mosse che hanno portato alla sua posizione, ricavate DAL CONFRONTO
 * fra le scacchiere di campioni consecutivi della stessa partita (il dataset non registra le mosse). Due campioni {@code i-1, i} sono consecutivi se
 * hanno la stessa chiave di partita, {@code ply} successivi e lati al tratto opposti: solo allora il confronto da' una mossa
 * ({@link TemporalHistory#deriveRecord}); altrimenti la storia si interrompe (la finestra del campione ha meno di {@code K} mosse, mai buchi).
 * <p>
 * Contenuto: {@code rec[i*K + a]} = mossa di eta' {@code a} del campione {@code i} (0 = non disponibile), {@code avail[i]} = quante ne ha (le prime
 * {@code avail[i]} sono tutte valide). Costo: {@code 4*K} byte per campione.
 * <p>
 * Contiene anche i bersagli multi-mossa ({@link #lambdaReturn}), perche' si calcolano lungo le stesse catene di campioni consecutivi.
 */
public final class TemporalData {

	public final NnueDataset d;
	public final int k;
	public final int[] rec;
	public final byte[] avail;

	private TemporalData(NnueDataset d, int k, int[] rec, byte[] avail) {
		this.d = d;
		this.k = k;
		this.rec = rec;
		this.avail = avail;
	}

	/** Vero se il campione {@code i + 1} e' il successore di {@code i} nella stessa partita (una mossa dopo). */
	public static boolean consecutive(NnueDataset d, int i) {
		return i >= 0 && i + 1 < d.n && d.game(i) == d.game(i + 1) && d.ply(i) >= 0 && d.ply(i + 1) == d.ply(i) + 1 && d.role(i) != d.role(i + 1);
	}

	public static TemporalData build(NnueDataset d, int k) {
		if (k < 1 || k > TemporalHistory.MAX_AGES) {
			throw new IllegalArgumentException("k=" + k);
		}
		final int n = d.n;
		if ((long) n * k > Integer.MAX_VALUE - 8) {
			throw new IllegalArgumentException("dataset troppo grande per K=" + k + " (usare maxsamples o K minore)");
		}
		final int[] trans = new int[n]; // trans[j] = la mossa che porta da j-1 a j
		for (int j = 1; j < n; j++) {
			if (consecutive(d, j - 1)) {
				trans[j] = TemporalHistory.deriveRecord(d.boardArray(j - 1), d.boardOffset(j - 1), d.boardArray(j), d.boardOffset(j), d.role(j - 1));
			}
		}
		final int[] rec = new int[n * k];
		final byte[] avail = new byte[n];
		for (int i = 0; i < n; i++) {
			int a = 0;
			while (a < k && i - a >= 1 && trans[i - a] != 0) {
				rec[i * k + a] = trans[i - a];
				a++;
			}
			avail[i] = (byte) a;
		}
		return new TemporalData(d, k, rec, avail);
	}

	/**
	 * Bersagio multi-mossa (TD(lambda) in avanti) per ogni campione, in [-1, 1] dal punto di vista del Bianco, o {@code NaN} se non ne ha. Si calcola
	 * all'indietro lungo ogni catena di campioni consecutivi:
	 * <pre>
	 *   G_i = (1 - mu) * V_{i+1} + mu * G_{i+1}                  V = valore di ricerca (se manca: G del successore)
	 *   ultimo campione di una partita finita (toEnd = 1):  G = esito vero
	 *   catena interrotta con esito noto:  G = esito;  senza esito:  G = V_i
	 *   bersaglio_i = lambda * V_i + (1 - lambda) * G_i
	 * </pre>
	 * {@code mu = 1} e' il Monte Carlo puro (G = esito della partita, come {@code NnueTrainer} con {@code gamma = 1}); {@code mu = 0} e' un solo passo:
	 * il valore di ricerca della posizione DOPO la mossa giocata, cioe' cio' che una ricerca a profondita' D ha visto una mossa piu' avanti lungo la
	 * linea realmente giocata. Valori intermedi spargono all'indietro, di poco a ogni passo, cio' che le ricerche successive hanno scoperto: una tattica
	 * che si compie in 3 mosse arriva alla posizione di partenza anche se la ricerca a profondita' 2 di li' non la vedeva. Valori e esito sono sempre dal punto
	 * di vista del Bianco: nessun cambio di segno fra posizioni consecutive.
	 */
	public static float[] lambdaReturn(NnueDataset d, double lambda, double mu) {
		if (lambda < 0 || lambda > 1 || mu < 0 || mu > 1) {
			throw new IllegalArgumentException("lambda=" + lambda + " mu=" + mu);
		}
		final int n = d.n;
		final float[] t = new float[n];
		double gNext = Double.NaN;
		for (int i = n - 1; i >= 0; i--) {
			final boolean hasV = d.valValid(i) == 1, hasO = d.outValid(i) == 1;
			final double v = hasV ? d.value(i) : Double.NaN;
			final double g;
			if (consecutive(d, i)) {
				final double vNext = d.valValid(i + 1) == 1 ? d.value(i + 1) : gNext;
				g = Double.isNaN(gNext) ? Double.NaN : (1.0 - mu) * vNext + mu * gNext;
			} else if (hasO) {
				g = d.outcome(i); // ultimo della catena: esito vero (partita finita qui, oppure catena interrotta: ripiego Monte Carlo)
			} else {
				g = v; // senza esito e senza successore: l'unica stima e' il valore di ricerca (puo' essere NaN)
			}
			gNext = g;
			if (hasV && !Double.isNaN(g)) {
				t[i] = (float) (lambda * v + (1.0 - lambda) * g);
			} else if (hasV) {
				t[i] = (float) v;
			} else {
				t[i] = Double.isNaN(g) ? Float.NaN : (float) g;
			}
		}
		return t;
	}
}
