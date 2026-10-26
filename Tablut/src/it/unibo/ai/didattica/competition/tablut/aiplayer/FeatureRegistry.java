package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Storia delle feature a mano: per ogni versione di {@link TablutFeatures} l'elenco ORDINATO degli id stabili, con
 * gruppo e divisore di normalizzazione di ciascuna. E' cio' che permette a {@link WeightConverter} di portare i pesi
 * di una versione su un'altra: gli indici scorrono quando si aggiunge o si toglie una feature, gli id no.
 *
 * <h3>Regole del registro</h3>
 * <ul>
 * <li>La versione CORRENTE e' derivata direttamente da {@link TablutFeatures} ({@code IDS}, {@code FEATURE_GROUP},
 * {@code DIV}): e' impossibile che registro e codice divergano. Le versioni storiche sono tabelle congelate.</li>
 * <li>Un id ha lo stesso SIGNIFICATO in tutte le versioni in cui compare (il valore grezzo, cioe' prima della divisione
 * per DIV, e' la stessa grandezza). Se si cambia la definizione di una feature va dato un NUOVO id, mai riusato quello
 * vecchio: il convertitore non puo' accorgersi di un cambio di significato.</li>
 * <li>{@link #DERIVED}: identita' lineari ESATTE che esprimono feature storiche rimosse con quelle attuali (in unita'
 * grezze). Servono a conservare l'informazione dei pesi quando una colonna sparisce.</li>
 * <li>Quando si aggiunge una versione: incrementare {@code TablutFeatures.VERSION}, congelare qui la versione che si
 * sta lasciando (copiandone IDS/gruppi/DIV, o tenendo una copia dell'estrattore come {@code LegacyTablutFeaturesV3}) e,
 * se una feature viene rimossa, aggiungere la sua identita' lineare a {@code DERIVED} (se esiste).</li>
 * </ul>
 */
public final class FeatureRegistry {

	/** Descrizione di una versione delle feature. */
	public static final class Schema {
		public final int version;
		/** Id delle feature nell'ordine degli indici di quella versione. */
		public final String[] ids;
		/** Maschera del gruppo di ciascuna feature (0 = BASE). */
		public final int[] group;
		/** Divisore di normalizzazione (valore in ingresso alla rete = grezzo / div). */
		public final double[] div;

		/** Pubblico per poter descrivere schemi sintetici nei test. */
		public Schema(int version, String[] ids, int[] group, double[] div) {
			this.version = version;
			this.ids = ids.clone();
			this.group = group.clone();
			this.div = div.clone();
		}

		public int size() {
			return ids.length;
		}

		/** Indici (crescenti) delle feature attive con la maschera {@code groups}: come {@link TablutFeatures#columnsFor}. */
		public int[] columnsFor(int groups) {
			int n = 0;
			for (int i = 0; i < ids.length; i++) {
				if (group[i] == 0 || (groups & group[i]) != 0) {
					n++;
				}
			}
			int[] cols = new int[n];
			n = 0;
			for (int i = 0; i < ids.length; i++) {
				if (group[i] == 0 || (groups & group[i]) != 0) {
					cols[n++] = i;
				}
			}
			return cols;
		}

		public String[] columnIds(int groups) {
			int[] cols = columnsFor(groups);
			String[] r = new String[cols.length];
			for (int i = 0; i < cols.length; i++) {
				r[i] = ids[cols[i]];
			}
			return r;
		}

		/** Maschera di tutti i gruppi che esistono in questa versione. */
		public int allGroups() {
			int m = 0;
			for (int g : group) {
				m |= g;
			}
			return m;
		}
	}

	/** Id delle 44 feature della v3, nell'ordine storico degli indici. */
	private static final String[] V3_IDS = { "N_WHITE", "N_BLACK", "KING_RING", "KING_AXIS_MIN", "KING_AXIS_MAX",
			"KING_ON_THRONE", "KING_NEAR_THRONE", "KING_SUPPORT", "BLACK_NEAR_KING", "KING_ADJ_BLACK", "KING_ADJ_WHITE",
			"KING_CAPTURE_MISSING", "BLACK_CENTRAL", "WHITE_CENTRAL", "BLACK_RING1", "BLACK_RING2", "BLACK_RING3",
			"BLACK_RING4", "WHITE_RING1", "WHITE_RING2", "WHITE_RING3", "WHITE_RING4", "WHITE_ISOLATED",
			"BLACK_ISOLATED", "BLACK_DIAG_LINKS", "BLACK_ORTHO_LINKS", "FREE_LINES", "BLACK_LINES", "WHITE_MOVES",
			"BLACK_MOVES", "KING_MOVES", "WHITE_BLOCKED", "BLACK_BLOCKED", "BLACK_EDGE_REACH", "KING_LANES",
			"KING_LANES_UNBLOCKABLE", "KING_2STEP_CELLS", "KING_2STEP_FORKS", "FLOOD_REGION", "FLOOD_EDGE", "FLOOD_SIDES",
			"WHITE_THREATENED", "BLACK_THREATENED", "KING_THREAT" };

	/**
	 * Identita' lineari esatte, in UNITA' GREZZE (valore prima della divisione per DIV): {@code vecchio = somma coef * nuovo}.
	 * Verificate su 160.000 posizioni (rango delle colonne v3 = 38 su 41) e dal test di conversione contro l'estrattore
	 * congelato {@link LegacyTablutFeaturesV3}.
	 */
	public static final Map<String, Map<String, Double>> DERIVED;
	static {
		Map<String, Map<String, Double>> d = new LinkedHashMap<>();
		d.put("KING_AXIS_MAX", terms("KING_RING", 1)); // RING = max(|dr|,|dc|) = KING_AXIS_MAX
		d.put("BLACK_RING4", terms("N_BLACK", 1, "BLACK_RING1", -1, "BLACK_RING2", -1, "BLACK_RING3", -1));
		d.put("WHITE_RING4", terms("N_WHITE", 1, "WHITE_RING1", -1, "WHITE_RING2", -1, "WHITE_RING3", -1));
		DERIVED = Collections.unmodifiableMap(d);
	}

	private static Map<String, Double> terms(Object... kv) {
		Map<String, Double> m = new LinkedHashMap<>();
		for (int i = 0; i < kv.length; i += 2) {
			m.put((String) kv[i], ((Number) kv[i + 1]).doubleValue());
		}
		return Collections.unmodifiableMap(m);
	}

	private FeatureRegistry() {
	}

	/** La versione corrente, derivata dal codice. */
	public static Schema current() {
		return new Schema(TablutFeatures.VERSION, TablutFeatures.IDS, TablutFeatures.FEATURE_GROUP, TablutFeatures.DIV);
	}

	/**
	 * Lo schema di una versione. La v3 e' congelata (gruppi e DIV dall'estrattore {@link LegacyTablutFeaturesV3});
	 * la v2 e' la v3 senza {@code WHITE_RING4} (cosi' la descrive il README del progetto; non esiste piu' un estrattore v2
	 * contro cui verificarla, quindi e' l'unica parte del registro non provata da un test di conversione).
	 *
	 * @throws IllegalArgumentException per una versione sconosciuta
	 */
	public static Schema forVersion(int version) {
		if (version == TablutFeatures.VERSION) {
			return current();
		}
		if (version == 3) {
			return new Schema(3, V3_IDS, LegacyTablutFeaturesV3.FEATURE_GROUP, LegacyTablutFeaturesV3.DIV);
		}
		if (version == 2) {
			Schema v3 = forVersion(3);
			int skip = Arrays.asList(v3.ids).indexOf("WHITE_RING4");
			String[] ids = new String[v3.size() - 1];
			int[] group = new int[ids.length];
			double[] div = new double[ids.length];
			for (int i = 0, k = 0; i < v3.size(); i++) {
				if (i != skip) {
					ids[k] = v3.ids[i];
					group[k] = v3.group[i];
					div[k] = v3.div[i];
					k++;
				}
			}
			return new Schema(2, ids, group, div);
		}
		throw new IllegalArgumentException("versione delle feature sconosciuta: " + version
				+ " (il registro conosce la 2, la 3 e la corrente " + TablutFeatures.VERSION + ")");
	}
}
