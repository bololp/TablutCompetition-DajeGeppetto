package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Porta i pesi di una rete (CNN ibrida o MLP) allenata con una versione/maschera di feature precedente sulla versione
 * CORRENTE di {@link TablutFeatures}, senza riallenare e SENZA cambiare la funzione calcolata.
 *
 * <h3>Perche' si puo' fare esattamente</h3>
 * Le feature entrano nella rete SOLO attraverso le colonne del primo strato che le legge (la parte "feature" delle
 * righe del primo strato denso della CNN, oppure l'ingresso del tronco dell'MLP). Tutto il resto (convoluzioni, strati
 * successivi, head) non dipende da quali feature esistano. Quel primo strato calcola {@code sum_j w_j * x_j} dove
 * {@code x_j = grezzo_j / DIV_j}. Lavorando in <b>unita' grezze</b> ({@code u_j = w_j / DIV_j}, il peso per ogni unita'
 * della grandezza) la mappatura fra due versioni e' una trasformazione lineare dei pesi:
 * <ul>
 * <li>feature presente in entrambe (stesso id): {@code u} si copia, il peso si riscala per il nuovo DIV (1:1 se uguale:
 * copia bit a bit);</li>
 * <li>feature NUOVA: peso 0, quindi la funzione non cambia; il training la fara' emergere;</li>
 * <li>feature RIMOSSA ma derivabile dalle altre ({@link FeatureRegistry#DERIVED}, es. {@code BLACK_RING4 = N_BLACK -
 * RING1 - RING2 - RING3}): il suo peso viene "ripiegato" sulle feature da cui dipende, e la funzione resta identica;</li>
 * <li>feature rimossa e NON derivabile (o esclusa dalla nuova maschera di gruppi): informazione persa, la funzione CAMBIA;
 * il {@link Plan} lo dichiara in {@link Plan#lost} e {@link Plan#isExact()} e' falso.</li>
 * </ul>
 * L'equivalenza esatta e' verificata da {@code FeatureConversionCheck} confrontando le uscite della rete vecchia (con
 * l'estrattore congelato {@link LegacyTablutFeaturesV3}) e della rete convertita (con l'estrattore attuale).
 *
 * <p>Uso: {@code java ... aiplayer.WeightConverter in=cnn_vecchio.txt out=cnn_v4.txt [groups=N] [dry=1]}.
 * Senza {@code groups} la nuova maschera e' quella vecchia piu' i gruppi aggiunti dalla v4 (ESCAPE, SAFEFORK, GATES).
 */
public final class WeightConverter {

	/** Gruppi introdotti dalla v4: di default una conversione li accende (le colonne nuove partono da 0). */
	public static final int V4_NEW_GROUPS = TablutFeatures.G_ESCAPE | TablutFeatures.G_SAFEFORK | TablutFeatures.G_GATES;

	private WeightConverter() {
	}

	/** Il piano di conversione: quali colonne si copiano, si sommano, si azzerano o si perdono. */
	public static final class Plan {
		public final FeatureRegistry.Schema from, to;
		public final int fromGroups, toGroups;
		public final String[] oldIds, newIds;
		/** t[nuova][vecchia]: coefficiente (gia' comprensivo del rapporto fra i DIV) con cui il peso vecchio entra nel nuovo. */
		final double[][] t;
		/** Id nuovi (peso iniziale 0). */
		public final List<String> added = new ArrayList<>();
		/** Id rimossi il cui peso e' stato ripiegato su altri (conversione esatta). */
		public final List<String> folded = new ArrayList<>();
		/** Id vecchi il cui contributo NON puo' essere conservato (conversione NON esatta). */
		public final List<String> lost = new ArrayList<>();

		public Plan(int fromVersion, int fromGroups, int toGroups) {
			this(FeatureRegistry.forVersion(fromVersion), fromGroups, FeatureRegistry.current(), toGroups);
		}

		/** Piano fra due schemi qualunque (i test lo usano con schemi sintetici, per esempio con DIV diversi). */
		public Plan(FeatureRegistry.Schema from, int fromGroups, FeatureRegistry.Schema to, int toGroups) {
			this.from = from;
			this.to = to;
			this.fromGroups = fromGroups;
			this.toGroups = toGroups;
			this.oldIds = from.columnIds(fromGroups);
			this.newIds = to.columnIds(toGroups);
			int[] oldCols = from.columnsFor(fromGroups), newCols = to.columnsFor(toGroups);
			Map<String, Integer> newPos = new HashMap<>();
			for (int j = 0; j < newIds.length; j++) {
				newPos.put(newIds[j], j);
			}
			t = new double[newIds.length][oldIds.length];
			Map<String, Boolean> oldHas = new HashMap<>();
			for (String id : oldIds) {
				oldHas.put(id, true);
			}
			for (int j = 0; j < oldIds.length; j++) {
				String id = oldIds[j];
				double divOld = from.div[oldCols[j]];
				Integer jn = newPos.get(id);
				if (jn != null) {
					t[jn][j] += to.div[newCols[jn]] / divOld;
					continue;
				}
				Map<String, Double> der = FeatureRegistry.DERIVED.get(id);
				boolean ok = der != null;
				if (ok) {
					for (String dep : der.keySet()) {
						if (!newPos.containsKey(dep)) {
							ok = false;
						}
					}
				}
				if (!ok) {
					lost.add(id);
					continue;
				}
				for (Map.Entry<String, Double> e : der.entrySet()) {
					int jd = newPos.get(e.getKey());
					t[jd][j] += e.getValue() * to.div[newCols[jd]] / divOld;
				}
				folded.add(id);
			}
			for (String id : newIds) {
				if (!oldHas.containsKey(id)) {
					added.add(id);
				}
			}
		}

		public boolean isExact() {
			return lost.isEmpty();
		}

		/** Vero se non c'e' niente da convertire (stessa versione e stessa maschera). */
		public boolean isIdentity() {
			return from.version == to.version && fromGroups == toGroups;
		}

		/** Mappa la parte "feature" di una riga: {@code newW[newOff..)} da {@code oldW[oldOff..)}. */
		void mapRow(double[] oldW, int oldOff, double[] newW, int newOff) {
			for (int jn = 0; jn < newIds.length; jn++) {
				double acc = 0.0;
				final double[] row = t[jn];
				for (int j = 0; j < oldIds.length; j++) {
					if (row[j] != 0.0) {
						acc += row[j] * oldW[oldOff + j];
					}
				}
				newW[newOff + jn] = acc;
			}
		}

		public String describe() {
			StringBuilder sb = new StringBuilder();
			sb.append("feature v").append(from.version).append(" (gruppi ").append(fromGroups).append(", ").append(oldIds.length)
					.append(" colonne) -> v").append(to.version).append(" (gruppi ").append(toGroups).append(", ")
					.append(newIds.length).append(" colonne)\n");
			sb.append("  aggiunte (peso 0)          : ").append(added.isEmpty() ? "-" : added).append('\n');
			sb.append("  rimosse e ripiegate (exact): ").append(folded.isEmpty() ? "-" : folded).append('\n');
			sb.append("  PERSE (funzione cambia)    : ").append(lost.isEmpty() ? "-" : lost).append('\n');
			sb.append("  conversione ").append(isExact() ? "ESATTA: la funzione calcolata non cambia" : "NON ESATTA");
			return sb.toString();
		}
	}

	// ------------------------------------------------------------------ CNN

	/** Converte una CNN ibrida. La rete ritornata ha {@code nFeat = plan.newIds.length} e gli stessi altri parametri. */
	public static ConvNet convertCnn(ConvNet old, Plan plan) {
		ConvNet.Config c = old.config();
		if (c.nFeat != plan.oldIds.length) {
			throw new IllegalArgumentException("la rete ha " + c.nFeat + " feature ma lo schema v" + plan.from.version
					+ "/gruppi " + plan.fromGroups + " ne ha " + plan.oldIds.length);
		}
		if (c.dense.length == 0) {
			throw new IllegalArgumentException("CNN senza strato denso condiviso: conversione non supportata");
		}
		ConvNet.Config nc = new ConvNet.Config(c.conv, c.reduce, plan.newIds.length, c.dense, c.head);
		ConvNet fresh = new ConvNet(nc, 1L);
		if (fresh.denseW[0] != old.denseW[0]) {
			throw new IllegalStateException("layout dei parametri inatteso: il primo strato denso non inizia allo stesso offset");
		}
		final float[] op = old.params();
		final float[] np = new float[fresh.numParams()];
		final int w0 = old.denseW[0];
		System.arraycopy(op, 0, np, 0, w0); // conv + riduzione: invariati
		final int fs = c.flatSize(), oldIn = c.denseIn(), newIn = nc.denseIn(), out = c.dense[0];
		double[] oldRow = new double[plan.oldIds.length], newRow = new double[plan.newIds.length];
		for (int o = 0; o < out; o++) {
			System.arraycopy(op, w0 + o * oldIn, np, w0 + o * newIn, fs); // parte spaziale (flatten): invariata
			for (int j = 0; j < oldRow.length; j++) {
				oldRow[j] = op[w0 + o * oldIn + fs + j];
			}
			plan.mapRow(oldRow, 0, newRow, 0);
			for (int j = 0; j < newRow.length; j++) {
				np[w0 + o * newIn + fs + j] = (float) newRow[j];
			}
		}
		final int tailOld = w0 + oldIn * out, tailNew = w0 + newIn * out;
		if (op.length - tailOld != np.length - tailNew) {
			throw new IllegalStateException("coda dei parametri di lunghezza diversa");
		}
		System.arraycopy(op, tailOld, np, tailNew, op.length - tailOld); // bias e strati successivi: invariati
		return new ConvNet(nc, np);
	}

	// ------------------------------------------------------------------ MLP

	/** Converte un MLP: agisce sugli strati che leggono direttamente il vettore di ingresso (la colonna del "ruolo", se c'e', si sposta in coda). */
	public static MLP convertMlp(MLP old, Plan plan) {
		MLP.Config c = old.config();
		if (c.nIn != plan.oldIds.length) {
			throw new IllegalArgumentException("la rete ha " + c.nIn + " ingressi ma lo schema v" + plan.from.version
					+ "/gruppi " + plan.fromGroups + " ne ha " + plan.oldIds.length);
		}
		MLP.Config nc = new MLP.Config(plan.newIds.length, c.trunk, c.head, c.heads, c.roleInput, c.act);
		MLP fresh = new MLP(nc, 1L);
		final int[][] to = old.layerTable(), tn = fresh.layerTable();
		final double[] op = old.params(), np = fresh.params();
		double[] newRow = new double[plan.newIds.length];
		for (int l = 0; l < to.length; l++) {
			final int out = to[l][3];
			if (to[l][4] == 0) { // strato interno: invariato
				System.arraycopy(op, to[l][0], np, tn[l][0], to[l][2] * out);
			} else {
				for (int o = 0; o < out; o++) {
					plan.mapRow(op, to[l][0] + o * to[l][2], newRow, 0);
					System.arraycopy(newRow, 0, np, tn[l][0] + o * tn[l][2], newRow.length);
					if (c.roleInput) {
						np[tn[l][0] + o * tn[l][2] + plan.newIds.length] = op[to[l][0] + o * to[l][2] + plan.oldIds.length];
					}
				}
			}
			System.arraycopy(op, to[l][1], np, tn[l][1], out); // bias
		}
		return fresh;
	}

	// ------------------------------------------------------------------ file

	/**
	 * Converte un file di pesi (CNN o MLP, riconosciuto dall'intestazione) e lo scrive in {@code outPath}.
	 *
	 * @param toGroups maschera di gruppi della rete risultante, oppure -1 per "quella vecchia piu' i gruppi nuovi della v4"
	 * @return il piano (per riportarlo all'utente)
	 */
	public static Plan convertFile(String inPath, String outPath, int toGroups) throws IOException {
		if (ConvNet.isCnnFile(inPath)) {
			ConvNet.Loaded l = ConvNet.load(inPath);
			if (l.net.config().nFeat == 0) {
				throw new IOException("CNN puro (nessuna feature a mano): non dipende dalla versione delle feature, nessuna conversione necessaria");
			}
			Plan plan = planFor(l.featureVersion, l.featureGroups, l.net.config().nFeat, toGroups, inPath);
			if (plan.isIdentity()) {
				throw new IOException("i pesi sono gia' alla versione corrente con la stessa maschera di gruppi: " + inPath);
			}
			convertCnn(l.net, plan).save(outPath, TablutFeatures.VERSION, plan.toGroups);
			return plan;
		}
		MLP.Loaded l = MLP.load(inPath);
		Plan plan = planFor(l.featureVersion, l.featureGroups, l.net.config().nIn, toGroups, inPath);
		if (plan.isIdentity()) {
			throw new IOException("i pesi sono gia' alla versione corrente con la stessa maschera di gruppi: " + inPath);
		}
		convertMlp(l.net, plan).save(outPath, TablutFeatures.VERSION, plan.toGroups);
		return plan;
	}

	private static Plan planFor(int fromVersion, int fromGroups, int nFeat, int toGroups, String path) throws IOException {
		Plan plan;
		try {
			plan = new Plan(fromVersion, fromGroups, toGroups >= 0 ? toGroups : (fromGroups | V4_NEW_GROUPS) & TablutFeatures.G_ALL);
		} catch (IllegalArgumentException e) {
			throw new IOException(e.getMessage() + " (" + path + ")", e);
		}
		if (plan.oldIds.length != nFeat) {
			throw new IOException("file incoerente: " + nFeat + " feature, ma lo schema v" + fromVersion + " con gruppi " + fromGroups
					+ " ne prevede " + plan.oldIds.length + " (" + path + ")");
		}
		return plan;
	}

	public static void main(String[] args) throws Exception {
		Map<String, String> p = new HashMap<>();
		for (String a : args) {
			int i = a.indexOf('=');
			if (i > 0) {
				p.put(a.substring(0, i), a.substring(i + 1));
			}
		}
		if (!p.containsKey("in") || (!p.containsKey("out") && !"1".equals(p.get("dry")))) {
			System.out.println("uso: WeightConverter in=pesi_vecchi.txt out=pesi_nuovi.txt [groups=N] [dry=1]\n"
					+ "  groups: maschera di gruppi della rete risultante (default: vecchia | ESCAPE | SAFEFORK | GATES)\n"
					+ "  dry=1 : mostra solo il piano, non scrive nulla\n"
					+ "  gruppi: MOB=1 LANES=2 FLOOD=4 THREATS=8 ESCAPE=16 SAFEFORK=32 GATES=64; default attuale = " + TablutFeatures.G_DEFAULT);
			System.exit(2);
		}
		int groups = p.containsKey("groups") ? Integer.parseInt(p.get("groups")) : -1;
		String in = p.get("in");
		try {
			if ("1".equals(p.get("dry"))) {
				int fv, fg, nf;
				if (ConvNet.isCnnFile(in)) {
					ConvNet.Loaded l = ConvNet.load(in);
					fv = l.featureVersion;
					fg = l.featureGroups;
					nf = l.net.config().nFeat;
				} else {
					MLP.Loaded l = MLP.load(in);
					fv = l.featureVersion;
					fg = l.featureGroups;
					nf = l.net.config().nIn;
				}
				System.out.println(planFor(fv, fg, nf, groups, in).describe());
				return;
			}
			Plan plan = convertFile(in, p.get("out"), groups);
			System.out.println(plan.describe());
			System.out.println("scritto " + p.get("out"));
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}
}
