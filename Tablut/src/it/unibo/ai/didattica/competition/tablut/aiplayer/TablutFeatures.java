package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Trasforma una posizione ({@link FastState}, o uno {@link State} del dominio) nel vettore di feature in input alla rete
 * ({@link MLP}, {@link NeuralEvaluator}). E' l'UNICO posto che sa come si
 * costruisce quel vettore: training e gioco chiamano entrambi
 * {@link #extract}, quindi restano coerenti per costruzione.
 *
 * <h3>Versione 2 - cosa cambia rispetto alla v1 (16 feature)</h3>
 * <ol>
 * <li><b>Vie di fuga ESATTE.</b> La v1 (e {@code SimpleEvaluator}) contava come
 * "corsia aperta" ogni raggio libero Re-bordo. Ma il Re non puo' entrare nelle
 * citadel ({@code AshtonRules.generateMoves}): le corsie lungo le righe/colonne
 * 3-4-5 terminano su una citadel e NON sono vie di fuga. Su 3000 posizioni reali
 * la definizione vecchia contava 1298 corsie contro 244 vere (81% "fantasma").
 * Qui ogni feature di mobilita'/fuga deriva dalle stesse regole di
 * {@code generateMoves} (stessa semantica, citadel comprese).</li>
 * <li><b>Invarianza per simmetria.</b> Le regole sono invarianti per le 8
 * simmetrie della scacchiera (rotazioni/riflessioni D4). La v1 dava al Re
 * coordinate con segno e una feature per ciascuna direzione, costringendo la
 * rete a reimparare lo stesso schema fino a 8 volte. Qui tutte le feature sono
 * funzioni invarianti (conteggi, distanze, coppie non ordinate): verificato da
 * {@code FeaturesCheck} applicando le 8 simmetrie a posizioni casuali.</li>
 * <li><b>Costo.</b> Nessuna allocazione, nessun {@code Pawn[][]} nel ciclo
 * caldo: la scacchiera viene copiata UNA volta in un {@code byte[121]} con bordo
 * sentinella (11x11) e tutto il resto lavora su indici interi senza controlli
 * sui limiti. Mobilita', minacce di cattura e corsie bloccabili escono da
 * un'unica passata che costruisce le "mappe di raggiungibilita'" dei due
 * colori.</li>
 * </ol>
 *
 * <h3>Le 4 categorie richieste</h3>
 * <ul>
 * <li><b>Materiale</b>: pedine, posizione del Re (anello, assi min/max, trono).</li>
 * <li><b>Mobilita'</b>: mosse di entrambi i lati, mosse del Re, pezzi bloccati,
 * celle di bordo raggiungibili dal Nero.</li>
 * <li><b>Sicurezza/minacce del Re</b>: vie di fuga immediate (e quante il Nero
 * non puo' bloccare), fuga in 2 mosse e "forchette", accerchiamento (flood fill),
 * cattura imminente del Re, pedine catturabili da entrambi i lati.</li>
 * <li><b>Struttura</b>: controllo del centro, distribuzione ad anelli, pedine
 * isolate, catene/blocchi del Nero (legami ortogonali e diagonali), corridoi
 * liberi/presidiati, prossimita' attaccanti-Re.</li>
 * </ul>
 *
 * <p>
 * Tutte le feature sono scalate in modo che i valori tipici stiano in [0,1].
 * Thread-safety: i metodi statici sono puri; lo stato temporaneo vive in un
 * {@link Workspace} che ogni thread deve possedere in esclusiva.
 */
public final class TablutFeatures {

	/** Incrementare quando cambia QUALSIASI definizione/ordine di feature: i
	 *  file di pesi lo memorizzano e {@link NeuralEvaluator#fromFile} rifiuta
	 *  pesi allenati con una versione diversa (altrimenti la rete leggerebbe
	 *  input con un significato diverso, senza alcun errore visibile).
	 *  <ul>
	 *  <li>v2: 43 feature (vie di fuga esatte, invarianza per simmetria).</li>
	 *  <li>v3: 44 feature. Aggiunta {@code WHITE_RING4} (pedine bianche sul bordo): la distribuzione
	 *  ad anelli era asimmetrica, i neri avevano gli anelli 1-4 e i bianchi solo 1-3. Tutti gli indici
	 *  successivi a {@code F_WHITE_RING3} sono scorsi di uno.</li>
	 *  <li>v4: 47 feature (44 nel set di default). <b>Rimosse 3 feature linearmente ridondanti</b> (misurato: le 41 colonne
	 *  del v3 avevano rango 38): {@code KING_AXIS_MAX} (identica a {@code KING_RING}), {@code BLACK_RING4} e
	 *  {@code WHITE_RING4} (derivabili da {@code N_BLACK}/{@code N_WHITE} meno gli anelli 1-3). <b>Aggiunte 6 feature</b>:
	 *  {@link #G_ESCAPE} (distanza di fuga graduata del Re), {@link #G_SAFEFORK} (forchette sicure) e {@link #G_GATES}
	 *  (stato dei 16 "gate" di fuga). Tutti gli indici sono cambiati: la conversione dei pesi v2/v3 e'
	 *  esatta e si fa con {@link WeightConverter}.</li>
	 *  </ul> */
	public static final int VERSION = 4;

	// ---------------------------------------------------------------- indici
	// BASE: dipendono solo dal contenuto della scacchiera (sempre calcolate)
	public static final int F_N_WHITE = 0;
	public static final int F_N_BLACK = 1;
	public static final int F_KING_RING = 2;
	public static final int F_KING_AXIS_MIN = 3;
	public static final int F_KING_ON_THRONE = 4;
	public static final int F_KING_NEAR_THRONE = 5;
	public static final int F_KING_SUPPORT = 6;
	public static final int F_BLACK_NEAR_KING = 7;
	public static final int F_KING_ADJ_BLACK = 8;
	public static final int F_KING_ADJ_WHITE = 9;
	public static final int F_KING_CAPTURE_MISSING = 10;
	public static final int F_BLACK_CENTRAL = 11;
	public static final int F_WHITE_CENTRAL = 12;
	public static final int F_BLACK_RING1 = 13;
	public static final int F_BLACK_RING2 = 14;
	public static final int F_BLACK_RING3 = 15;
	public static final int F_WHITE_RING1 = 16;
	public static final int F_WHITE_RING2 = 17;
	public static final int F_WHITE_RING3 = 18;
	public static final int F_WHITE_ISOLATED = 19;
	public static final int F_BLACK_ISOLATED = 20;
	public static final int F_BLACK_DIAG_LINKS = 21;
	public static final int F_BLACK_ORTHO_LINKS = 22;
	public static final int F_FREE_LINES = 23;
	public static final int F_BLACK_LINES = 24;
	// MOB: mobilita' (richiede la passata sulle mosse)
	public static final int F_WHITE_MOVES = 25;
	public static final int F_BLACK_MOVES = 26;
	public static final int F_KING_MOVES = 27;
	public static final int F_WHITE_BLOCKED = 28;
	public static final int F_BLACK_BLOCKED = 29;
	public static final int F_BLACK_EDGE_REACH = 30;
	// LANES: vie di fuga del Re
	public static final int F_KING_LANES = 31;
	public static final int F_KING_LANES_UNBLOCKABLE = 32;
	public static final int F_KING_2STEP_CELLS = 33;
	public static final int F_KING_2STEP_FORKS = 34;
	// ESCAPE: distanza di fuga graduata (BFS delle mosse del Re su scacchiera statica, fino a 3 mosse)
	public static final int F_ESC_DIST = 35;
	public static final int F_ESC_3STEP_CELLS = 36;
	public static final int F_ESC_EDGE3 = 37;
	// SAFEFORK: forchette sicure
	public static final int F_SAFE_FORKS = 38;
	// GATES: i 16 corridoi di fuga (righe/colonne 1,2,6,7 viste da ciascun estremo)
	public static final int F_GATES_CLOSED = 39;
	public static final int F_GATE_RUN = 40;
	// FLOOD: accerchiamento
	public static final int F_FLOOD_REGION = 41;
	public static final int F_FLOOD_EDGE = 42;
	public static final int F_FLOOD_SIDES = 43;
	// THREATS: catture imminenti
	public static final int F_WHITE_THREATENED = 44;
	public static final int F_BLACK_THREATENED = 45;
	public static final int F_KING_THREAT = 46;

	public static final int NUM_FEATURES = 47;

	/**
	 * Identificatori STABILI delle feature, nell'ordine degli indici. A differenza degli indici (che scorrono quando si
	 * aggiunge o toglie una feature) gli id non cambiano mai di significato: {@link FeatureRegistry} e
	 * {@link WeightConverter} li usano per mappare i pesi di una versione sull'altra. {@code FeaturesCheck} verifica che
	 * {@code IDS[i]} sia il nome della costante {@code F_<id>} con valore i.
	 */
	public static final String[] IDS = { "N_WHITE", "N_BLACK", "KING_RING", "KING_AXIS_MIN", "KING_ON_THRONE",
			"KING_NEAR_THRONE", "KING_SUPPORT", "BLACK_NEAR_KING", "KING_ADJ_BLACK", "KING_ADJ_WHITE",
			"KING_CAPTURE_MISSING", "BLACK_CENTRAL", "WHITE_CENTRAL", "BLACK_RING1", "BLACK_RING2", "BLACK_RING3",
			"WHITE_RING1", "WHITE_RING2", "WHITE_RING3", "WHITE_ISOLATED", "BLACK_ISOLATED", "BLACK_DIAG_LINKS",
			"BLACK_ORTHO_LINKS", "FREE_LINES", "BLACK_LINES", "WHITE_MOVES", "BLACK_MOVES", "KING_MOVES",
			"WHITE_BLOCKED", "BLACK_BLOCKED", "BLACK_EDGE_REACH", "KING_LANES", "KING_LANES_UNBLOCKABLE",
			"KING_2STEP_CELLS", "KING_2STEP_FORKS", "ESC_DIST", "ESC_3STEP_CELLS", "ESC_EDGE3", "SAFE_FORKS",
			"GATES_CLOSED", "GATE_RUN", "FLOOD_REGION", "FLOOD_EDGE", "FLOOD_SIDES", "WHITE_THREATENED",
			"BLACK_THREATENED", "KING_THREAT" };

	/** Gruppi di feature (maschera di bit): per escluderne alcuni senza pagarne il costo (ablation, varianti "lite"). */
	public static final int G_MOB = 1;
	public static final int G_LANES = 2;
	public static final int G_FLOOD = 4;
	public static final int G_THREATS = 8;
	/** Distanza di fuga graduata del Re (v4). */
	public static final int G_ESCAPE = 16;
	/** Forchette sicure (v4). */
	public static final int G_SAFEFORK = 32;
	/** Stato dei 16 gate di fuga (v4). */
	public static final int G_GATES = 64;
	public static final int G_ALL = G_MOB | G_LANES | G_FLOOD | G_THREATS | G_ESCAPE | G_SAFEFORK | G_GATES;
	/** Set di default in produzione: tutto tranne FLOOD (misurato: nessun valore predittivo, ~0,4 us di costo). */
	public static final int G_DEFAULT = G_MOB | G_LANES | G_THREATS | G_ESCAPE | G_SAFEFORK | G_GATES;
	/** Tier "leggero": solo le vie di fuga del Re oltre alle feature BASE (la scansione dei pezzi bianchi viene saltata). */
	public static final int G_LEAN = G_LANES;

	/** Gruppo di appartenenza di ogni feature (0 = BASE, sempre calcolata). */
	public static final int[] FEATURE_GROUP = new int[NUM_FEATURES];
	static {
		for (int i = F_WHITE_MOVES; i <= F_BLACK_EDGE_REACH; i++) {
			FEATURE_GROUP[i] = G_MOB;
		}
		for (int i = F_KING_LANES; i <= F_KING_2STEP_FORKS; i++) {
			FEATURE_GROUP[i] = G_LANES;
		}
		for (int i = F_ESC_DIST; i <= F_ESC_EDGE3; i++) {
			FEATURE_GROUP[i] = G_ESCAPE;
		}
		FEATURE_GROUP[F_SAFE_FORKS] = G_SAFEFORK;
		for (int i = F_GATES_CLOSED; i <= F_GATE_RUN; i++) {
			FEATURE_GROUP[i] = G_GATES;
		}
		for (int i = F_FLOOD_REGION; i <= F_FLOOD_SIDES; i++) {
			FEATURE_GROUP[i] = G_FLOOD;
		}
		for (int i = F_WHITE_THREATENED; i <= F_KING_THREAT; i++) {
			FEATURE_GROUP[i] = G_THREATS;
		}
	}

	/**
	 * Indici (in ordine crescente) delle feature attive con la maschera
	 * {@code groups}: BASE piu' i gruppi richiesti. Una rete allenata con una
	 * certa maschera ha esattamente {@code columnsFor(groups).length} ingressi:
	 * le colonne dei gruppi esclusi non vengono ne' calcolate ne' moltiplicate.
	 */
	public static int[] columnsFor(int groups) {
		int n = 0;
		for (int i = 0; i < NUM_FEATURES; i++) {
			if (FEATURE_GROUP[i] == 0 || (groups & FEATURE_GROUP[i]) != 0) {
				n++;
			}
		}
		int[] cols = new int[n];
		n = 0;
		for (int i = 0; i < NUM_FEATURES; i++) {
			if (FEATURE_GROUP[i] == 0 || (groups & FEATURE_GROUP[i]) != 0) {
				cols[n++] = i;
			}
		}
		return cols;
	}

	/** Divisore di normalizzazione di ciascuna feature (valore grezzo / DIV = valore in input alla rete).
	 *  Unica fonte di verita': estrattore e test la usano entrambi. */
	public static final double[] DIV = new double[NUM_FEATURES];
	static {
		DIV[F_N_WHITE] = 8.0;
		DIV[F_N_BLACK] = 16.0;
		DIV[F_KING_RING] = 4.0;
		DIV[F_KING_AXIS_MIN] = 4.0;
		DIV[F_KING_ON_THRONE] = 1.0;
		DIV[F_KING_NEAR_THRONE] = 1.0;
		DIV[F_KING_SUPPORT] = 8.0;
		DIV[F_BLACK_NEAR_KING] = 8.0;
		DIV[F_KING_ADJ_BLACK] = 4.0;
		DIV[F_KING_ADJ_WHITE] = 4.0;
		DIV[F_KING_CAPTURE_MISSING] = 4.0;
		DIV[F_BLACK_CENTRAL] = 128.0;
		DIV[F_WHITE_CENTRAL] = 64.0;
		DIV[F_BLACK_RING1] = 8.0;
		DIV[F_BLACK_RING2] = 8.0;
		DIV[F_BLACK_RING3] = 8.0;
		DIV[F_WHITE_RING1] = 8.0;
		DIV[F_WHITE_RING2] = 8.0;
		DIV[F_WHITE_RING3] = 8.0;
		DIV[F_WHITE_ISOLATED] = 8.0;
		DIV[F_BLACK_ISOLATED] = 16.0;
		DIV[F_BLACK_DIAG_LINKS] = 16.0;
		DIV[F_BLACK_ORTHO_LINKS] = 16.0;
		DIV[F_FREE_LINES] = 8.0;
		DIV[F_BLACK_LINES] = 8.0;
		DIV[F_WHITE_MOVES] = 64.0;
		DIV[F_BLACK_MOVES] = 100.0;
		DIV[F_KING_MOVES] = 16.0;
		DIV[F_WHITE_BLOCKED] = 8.0;
		DIV[F_BLACK_BLOCKED] = 16.0;
		DIV[F_BLACK_EDGE_REACH] = 20.0;
		DIV[F_KING_LANES] = 4.0;
		DIV[F_KING_LANES_UNBLOCKABLE] = 4.0;
		DIV[F_KING_2STEP_CELLS] = 8.0;
		DIV[F_KING_2STEP_FORKS] = 8.0;
		DIV[F_ESC_DIST] = 1.0; // valore grezzo = 1/d (d = mosse minime del Re per fuggire, 1..3; 0 se oltre 3)
		DIV[F_ESC_3STEP_CELLS] = 20.0;
		DIV[F_ESC_EDGE3] = 12.0;
		DIV[F_SAFE_FORKS] = 8.0;
		DIV[F_GATES_CLOSED] = 16.0;
		DIV[F_GATE_RUN] = 100.0;
		DIV[F_FLOOD_REGION] = 64.0;
		DIV[F_FLOOD_EDGE] = 20.0;
		DIV[F_FLOOD_SIDES] = 4.0;
		DIV[F_WHITE_THREATENED] = 8.0;
		DIV[F_BLACK_THREATENED] = 8.0;
		DIV[F_KING_THREAT] = 1.0;
	}

	/** 1/DIV: nel ciclo caldo si moltiplica (le divisioni in virgola mobile costano ~4x). */
	private static final double[] INV = new double[NUM_FEATURES];
	static {
		for (int i = 0; i < NUM_FEATURES; i++) {
			INV[i] = 1.0 / DIV[i];
		}
	}

	/** Nomi leggibili, nello stesso ordine degli indici (debug/spiegabilita'). */
	public static final String[] NAMES = new String[NUM_FEATURES];
	static {
		String[] n = NAMES;
		n[F_N_WHITE] = "materiale: pedine bianche";
		n[F_N_BLACK] = "materiale: pedine nere";
		n[F_KING_RING] = "re: anello (distanza di Chebyshev dal trono)";
		n[F_KING_AXIS_MIN] = "re: min(|dr|,|dc|) dal trono";
		n[F_KING_ON_THRONE] = "re: sul trono";
		n[F_KING_NEAR_THRONE] = "re: adiacente al trono";
		n[F_KING_SUPPORT] = "re: pedine bianche entro 2 celle";
		n[F_BLACK_NEAR_KING] = "re: pedine nere entro 2 celle";
		n[F_KING_ADJ_BLACK] = "re: pedine nere adiacenti";
		n[F_KING_ADJ_WHITE] = "re: pedine bianche adiacenti";
		n[F_KING_CAPTURE_MISSING] = "re: ostili ancora mancanti per catturarlo";
		n[F_BLACK_CENTRAL] = "struttura: centralita' nera";
		n[F_WHITE_CENTRAL] = "struttura: centralita' bianca";
		n[F_BLACK_RING1] = "struttura: nere nell'anello 1";
		n[F_BLACK_RING2] = "struttura: nere nell'anello 2";
		n[F_BLACK_RING3] = "struttura: nere nell'anello 3";
		n[F_WHITE_RING1] = "struttura: bianche nell'anello 1";
		n[F_WHITE_RING2] = "struttura: bianche nell'anello 2";
		n[F_WHITE_RING3] = "struttura: bianche nell'anello 3";
		n[F_WHITE_ISOLATED] = "struttura: bianche isolate";
		n[F_BLACK_ISOLATED] = "struttura: nere isolate";
		n[F_BLACK_DIAG_LINKS] = "struttura: legami diagonali neri (blocchi)";
		n[F_BLACK_ORTHO_LINKS] = "struttura: legami ortogonali neri (blocchi)";
		n[F_FREE_LINES] = "corridoi: linee di fuga completamente libere";
		n[F_BLACK_LINES] = "corridoi: linee di fuga presidiate dal Nero";
		n[F_WHITE_MOVES] = "mobilita': mosse bianche";
		n[F_BLACK_MOVES] = "mobilita': mosse nere";
		n[F_KING_MOVES] = "mobilita': mosse del re";
		n[F_WHITE_BLOCKED] = "mobilita': bianche bloccate";
		n[F_BLACK_BLOCKED] = "mobilita': nere bloccate";
		n[F_BLACK_EDGE_REACH] = "mobilita': celle di fuga raggiungibili dal Nero";
		n[F_KING_LANES] = "fuga: vie di fuga immediate del re";
		n[F_KING_LANES_UNBLOCKABLE] = "fuga: vie immediate NON bloccabili dal Nero";
		n[F_KING_2STEP_CELLS] = "fuga: celle (1 mossa) da cui il re ha una via";
		n[F_KING_2STEP_FORKS] = "fuga: celle (1 mossa) da cui il re ha >=2 vie (forchetta)";
		n[F_ESC_DIST] = "fuga: 1/d, d = mosse minime del re per fuggire (solo d<=3)";
		n[F_ESC_3STEP_CELLS] = "fuga: celle a 2 mosse del re da cui ha una via (fuga in 3)";
		n[F_ESC_EDGE3] = "fuga: celle di fuga distinte raggiungibili in <=3 mosse del re";
		n[F_SAFE_FORKS] = "fuga: forchette sicure (re non catturabile all'arrivo)";
		n[F_GATES_CLOSED] = "gate: gate di fuga occupati (corsa libera = 0)";
		n[F_GATE_RUN] = "gate: somma delle corse libere dai 16 gate verso l'interno";
		n[F_FLOOD_REGION] = "accerchiamento: celle raggiungibili dal re";
		n[F_FLOOD_EDGE] = "accerchiamento: celle di fuga nella regione del re";
		n[F_FLOOD_SIDES] = "accerchiamento: lati toccati dalla regione del re";
		n[F_WHITE_THREATENED] = "minacce: bianche catturabili dal Nero";
		n[F_BLACK_THREATENED] = "minacce: nere catturabili dal Bianco";
		n[F_KING_THREAT] = "minacce: re catturabile subito";
	}

	// -------------------------------------------------------------- tabelle
	// Scacchiera 9x9 dentro una griglia 11x11 con bordo sentinella OUT: gli
	// indici sono p = (r+1)*11 + (c+1). I codici coincidono con Pawn.ordinal() e con quelli
	// di BoardConstants (EMPTY, WHITE, BLACK, THRONE, KING): lo verifica FeaturesCheck. In piu':
	// OUT = bordo sentinella e CEMPTY = citadel VUOTA. Codificare le citadel
	// vuote a parte rende il caso comune (pezzo che non parte da una citadel)
	// un loop stretto "while (b[q] == EMPTY)": una citadel lo ferma da sola.
	// isFree(v) e' vero per EMPTY e CEMPTY (cella senza pedine).
	static final byte EMPTY = 0, WHITE = 1, BLACK = 2, THRONE = 3, KING = 4, OUT = 5, CEMPTY = 8;
	static final int W = 11;
	static final int PAD = W * W;
	/** N, S, W, E in indici padded. */
	static final int[] DIR = { -W, W, -1, 1 };
	static final int THRONE_P = idx(4, 4);

	static final boolean[] CIT = new boolean[PAD];
	static final boolean[] TIP = new boolean[PAD];
	static final boolean[] EDGE = new boolean[PAD];
	/** Le 20 celle di bordo che NON sono citadel (32 di bordo - 12 citadel di bordo): le uniche dove il Re puo' davvero fuggire. */
	static final int[] FREE_EDGE;
	/** Chebyshev dal centro (0 = trono ... 4 = bordo). */
	static final byte[] RING = new byte[PAD];
	/** Manhattan dal centro. */
	static final byte[] MAN = new byte[PAD];
	static final byte[] ROW = new byte[PAD];
	static final byte[] COL = new byte[PAD];
	/** bit0 = riga 0, bit1 = riga 8, bit2 = colonna 0, bit3 = colonna 8. */
	static final byte[] SIDE = new byte[PAD];

	/** Le 16 citadel (indici padded). */
	static final int[] CIT_CELLS = new int[16];

	static int idx(int r, int c) {
		return (r + 1) * W + (c + 1);
	}

	static boolean isFree(byte v) {
		return (v & 7) == 0;
	}

	static {
		int[][] cit = { { 3, 0 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 4 }, { 3, 8 },
				{ 4, 8 }, { 5, 8 }, { 4, 7 }, { 8, 3 }, { 8, 4 }, { 8, 5 }, { 7, 4 } };
		for (int i = 0; i < cit.length; i++) {
			CIT[idx(cit[i][0], cit[i][1])] = true;
			CIT_CELLS[i] = idx(cit[i][0], cit[i][1]);
		}
		int[][] tips = { { 4, 0 }, { 0, 4 }, { 8, 4 }, { 4, 8 } };
		for (int[] c : tips) {
			TIP[idx(c[0], c[1])] = true;
		}
		int[] freeEdge = new int[32];
		int nfe = 0;
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				int p = idx(r, c);
				EDGE[p] = r == 0 || r == 8 || c == 0 || c == 8;
				RING[p] = (byte) Math.max(Math.abs(r - 4), Math.abs(c - 4));
				MAN[p] = (byte) (Math.abs(r - 4) + Math.abs(c - 4));
				ROW[p] = (byte) r;
				COL[p] = (byte) c;
				SIDE[p] = (byte) ((r == 0 ? 1 : 0) | (r == 8 ? 2 : 0) | (c == 0 ? 4 : 0) | (c == 8 ? 8 : 0));
				if (EDGE[p] && !CIT[p]) {
					freeEdge[nfe++] = p;
				}
			}
		}
		FREE_EDGE = Arrays.copyOf(freeEdge, nfe); // 20
	}

	private TablutFeatures() {
	}

	/** Stato temporaneo di una estrazione. Un thread = un Workspace. */
	public static final class Workspace {
		final byte[] b = new byte[PAD];
		final byte[] reachW = new byte[PAD];
		final byte[] reachB = new byte[PAD];
		final int[] wl = new int[24];
		final int[] bl = new int[24];
		final int[] kd = new int[32];
		final int[] seen = new int[PAD];
		final int[] stack = new int[96];
		final int[] bfsA = new int[96], bfsB = new int[96];
		final byte[] reachTmp = new byte[PAD];
		final int[] rowAll = new int[9], colAll = new int[9], rowBlack = new int[9], colBlack = new int[9];
		int stamp = 0;
		int nW, nB, kp, edgeDirs, wBlocked, bBlocked;
		boolean nearThrone;
		// risultati intermedi della passata sulle mosse: pubblici SOLO come hook per i test
		public int whiteMoves, blackMoves, kingMoves;

		public Workspace() {
			Arrays.fill(b, OUT);
		}

		/** Hook di test: quante mosse BIANCHE (Re compreso) finiscono sulla cella (r,c). Valido dopo extract con G_MOB/G_LANES/G_THREATS. */
		public int reachWhiteAt(int r, int c) {
			return reachW[idx(r, c)];
		}

		/** Hook di test: quante mosse NERE finiscono sulla cella (r,c). */
		public int reachBlackAt(int r, int c) {
			return reachB[idx(r, c)];
		}
	}

	public static Workspace newWorkspace() {
		return new Workspace();
	}

	/** Comoda per i test: alloca workspace e vettore ad ogni chiamata (NON usare nel ciclo caldo). */
	public static double[] extract(State state) {
		return extract(FastState.fromState(state));
	}

	/** Come {@link #extract(State)} per una posizione {@link FastState}. */
	public static double[] extract(FastState state) {
		double[] out = new double[NUM_FEATURES];
		extract(state, newWorkspace(), out, G_ALL);
		return out;
	}

	// --------------------------------------------------------------- estrazione

	/**
	 * Come {@link #extract(FastState, Workspace, double[], int)} per uno {@link State} del dominio: lo converte (alloca) e
	 * delega. E' l'unica implementazione delle feature: training e gioco restano coerenti per costruzione.
	 */
	public static void extract(State state, Workspace ws, double[] out, int groups) {
		extract(FastState.fromState(state), ws, out, groups);
	}

	/**
	 * Riempie {@code out[0..NUM_FEATURES)} con le feature di {@code state}. Nessuna allocazione: la scacchiera compatta
	 * viene copiata nel buffer con bordo sentinella del {@link Workspace} e tutto il resto lavora su quello.
	 *
	 * @param groups maschera di gruppi da calcolare ({@link #G_ALL} in produzione);
	 *               le feature dei gruppi esclusi restano a 0
	 */
	public static void extract(FastState state, Workspace ws, double[] out, int groups) {
		Arrays.fill(out, 0, NUM_FEATURES, 0.0);
		if (!loadBoard(state, ws)) {
			// Re assente: non dovrebbe mai essere valutato (BLACKWIN gestito dalla ricerca).
			// Vettore "neutro" invece di un'eccezione dentro un ciclo di training di ore.
			return;
		}
		fillKingAndWhite(ws, out);
		fillBlackAndLines(ws, out);
		fillKingSafetyBase(ws, out);
		if ((groups & (G_MOB | G_LANES | G_THREATS)) != 0) {
			scanMoves(ws, (groups & (G_MOB | G_THREATS)) != 0);
			if ((groups & G_MOB) != 0) {
				fillMobility(ws, out);
			}
			if ((groups & G_LANES) != 0) {
				fillLanes(ws, out);
			}
			if ((groups & G_THREATS) != 0) {
				fillThreats(ws, out);
			}
		}
		if ((groups & (G_ESCAPE | G_SAFEFORK)) != 0) {
			fillEscape(ws, out, groups);
		}
		if ((groups & G_GATES) != 0) {
			fillGates(ws, out);
		}
		if ((groups & G_FLOOD) != 0) {
			fillFlood(ws, out);
		}
	}

	/** Copia la scacchiera nel buffer padded e costruisce le liste dei pezzi. false se manca il Re. */
	private static boolean loadBoard(FastState state, Workspace ws) {
		final byte[] b = ws.b;
		final int[] wl = ws.wl, bl = ws.bl;
		final byte[] src = state.board;
		int nW = 0, nB = 0, kp = -1;
		for (int r = 0; r < 9; r++) {
			int base = (r + 1) * W + 1;
			int sbase = r * 9;
			for (int c = 0; c < 9; c++) {
				// i codici di FastState coincidono con quelli dell'estrattore (0 vuoto, 1 bianco, 2 nero, 4 re)
				int v = src[sbase + c];
				b[base + c] = (byte) v;
				if (v == WHITE) {
					wl[nW++] = base + c;
				} else if (v == BLACK) {
					bl[nB++] = base + c;
				} else if (v == KING) {
					kp = base + c;
				}
			}
		}
		if (b[THRONE_P] == EMPTY) {
			b[THRONE_P] = THRONE; // in FastState il trono vuoto e' la cella 40 vuota; qui ha il suo codice
		}
		for (int p : CIT_CELLS) {
			if (b[p] == EMPTY) {
				b[p] = CEMPTY;
			}
		}
		ws.nW = nW;
		ws.nB = nB;
		ws.kp = kp;
		ws.nearThrone = kp >= 0 && MAN[kp] == 1;
		return kp >= 0;
	}

	private static void fillKingAndWhite(Workspace ws, double[] out) {
		final byte[] b = ws.b;
		final int kp = ws.kp, nW = ws.nW;
		final int kr = ROW[kp], kc = COL[kp];
		final int dr = Math.abs(kr - 4), dc = Math.abs(kc - 4);
		out[F_N_WHITE] = nW * INV[F_N_WHITE];
		out[F_N_BLACK] = ws.nB * INV[F_N_BLACK];
		out[F_KING_RING] = RING[kp] * INV[F_KING_RING];
		out[F_KING_AXIS_MIN] = (Math.min(dr, dc)) * INV[F_KING_AXIS_MIN];
		out[F_KING_ON_THRONE] = kp == THRONE_P ? 1.0 : 0.0;
		out[F_KING_NEAR_THRONE] = ws.nearThrone ? 1.0 : 0.0;

		int whiteCentral = 0, support = 0, isoW = 0, wr1 = 0, wr2 = 0, wr3 = 0;
		Arrays.fill(ws.rowAll, 0);
		Arrays.fill(ws.colAll, 0);
		Arrays.fill(ws.rowBlack, 0);
		Arrays.fill(ws.colBlack, 0);
		ws.rowAll[kr]++;
		ws.colAll[kc]++;
		for (int i = 0; i < nW; i++) {
			int p = ws.wl[i];
			whiteCentral += 8 - MAN[p];
			int ring = RING[p];
			if (ring == 1) {
				wr1++;
			} else if (ring == 2) {
				wr2++;
			} else if (ring == 3) {
				wr3++;
			}
			if (Math.max(Math.abs(ROW[p] - kr), Math.abs(COL[p] - kc)) <= 2) {
				support++;
			}
			if (!friendNear(b, p, WHITE, KING)) {
				isoW++;
			}
			ws.rowAll[ROW[p]]++;
			ws.colAll[COL[p]]++;
		}
		out[F_WHITE_CENTRAL] = whiteCentral * INV[F_WHITE_CENTRAL];
		out[F_WHITE_RING1] = (wr1) * INV[F_WHITE_RING1];
		out[F_WHITE_RING2] = (wr2) * INV[F_WHITE_RING2];
		out[F_WHITE_RING3] = (wr3) * INV[F_WHITE_RING3];
		out[F_WHITE_ISOLATED] = isoW * INV[F_WHITE_ISOLATED];
		out[F_KING_SUPPORT] = support * INV[F_KING_SUPPORT];
	}

	private static void fillBlackAndLines(Workspace ws, double[] out) {
		final byte[] b = ws.b;
		final int kr = ROW[ws.kp], kc = COL[ws.kp];
		int blackCentral = 0, near = 0, isoB = 0, diag = 0, ortho = 0, br1 = 0, br2 = 0, br3 = 0;
		for (int i = 0; i < ws.nB; i++) {
			int p = ws.bl[i];
			blackCentral += 8 - MAN[p];
			int ring = RING[p];
			if (ring == 1) {
				br1++;
			} else if (ring == 2) {
				br2++;
			} else if (ring == 3) {
				br3++;
			}
			if (Math.max(Math.abs(ROW[p] - kr), Math.abs(COL[p] - kc)) <= 2) {
				near++;
			}
			if (!friendNear(b, p, BLACK, BLACK)) {
				isoB++;
			}
			if (b[p + 1] == BLACK) {
				ortho++;
			}
			if (b[p + W] == BLACK) {
				ortho++;
			}
			if (b[p + W + 1] == BLACK) {
				diag++;
			}
			if (b[p + W - 1] == BLACK) {
				diag++;
			}
			ws.rowAll[ROW[p]]++;
			ws.colAll[COL[p]]++;
			ws.rowBlack[ROW[p]]++;
			ws.colBlack[COL[p]]++;
		}
		out[F_BLACK_CENTRAL] = blackCentral * INV[F_BLACK_CENTRAL];
		out[F_BLACK_NEAR_KING] = near * INV[F_BLACK_NEAR_KING];
		out[F_BLACK_RING1] = (br1) * INV[F_BLACK_RING1];
		out[F_BLACK_RING2] = (br2) * INV[F_BLACK_RING2];
		out[F_BLACK_RING3] = (br3) * INV[F_BLACK_RING3];
		out[F_BLACK_ISOLATED] = isoB * INV[F_BLACK_ISOLATED];
		out[F_BLACK_DIAG_LINKS] = diag * INV[F_BLACK_DIAG_LINKS];
		out[F_BLACK_ORTHO_LINKS] = ortho * INV[F_BLACK_ORTHO_LINKS];

		// linee di fuga: righe/colonne 1,2,6,7 (le uniche i cui due estremi sono celle di fuga vere)
		int free = 0, guarded = 0;
		for (int k = 0; k < 4; k++) {
			int L = k < 2 ? k + 1 : k + 4; // 1,2,6,7
			if (ws.rowAll[L] == 0) {
				free++;
			}
			if (ws.colAll[L] == 0) {
				free++;
			}
			if (ws.rowBlack[L] > 0) {
				guarded++;
			}
			if (ws.colBlack[L] > 0) {
				guarded++;
			}
		}
		out[F_FREE_LINES] = free * INV[F_FREE_LINES];
		out[F_BLACK_LINES] = guarded * INV[F_BLACK_LINES];
	}

	/** Ostili adiacenti al Re e quanti ne mancano per catturarlo (solo contenuto della scacchiera). */
	private static void fillKingSafetyBase(Workspace ws, double[] out) {
		final byte[] b = ws.b;
		final int kp = ws.kp;
		int adjB = 0, adjW = 0;
		for (int k = 0; k < 4; k++) {
			byte v = b[kp + DIR[k]];
			if (v == BLACK) {
				adjB++;
			} else if (v == WHITE) {
				adjW++;
			}
		}
		out[F_KING_ADJ_BLACK] = adjB * INV[F_KING_ADJ_BLACK];
		out[F_KING_ADJ_WHITE] = adjW * INV[F_KING_ADJ_WHITE];
		int need, have;
		if (kp == THRONE_P) {
			need = 4;
			have = adjB;
		} else if (ws.nearThrone) {
			need = 3;
			have = adjB; // il trono non e' BLACK: i neri adiacenti sono tutti sugli altri 3 lati
		} else {
			need = 2;
			int h1 = (hostileForKing(b, kp - W) ? 1 : 0) + (hostileForKing(b, kp + W) ? 1 : 0);
			int h2 = (hostileForKing(b, kp - 1) ? 1 : 0) + (hostileForKing(b, kp + 1) ? 1 : 0);
			have = Math.max(h1, h2);
		}
		out[F_KING_CAPTURE_MISSING] = (Math.max(0, need - have)) * INV[F_KING_CAPTURE_MISSING];
	}

	/**
	 * Passata sulle mosse di tutti i pezzi: costruisce le mappe di
	 * raggiungibilita' dei due colori (quante mosse di quel colore finiscono su
	 * ogni cella) e conta mosse/pezzi bloccati. Per il Re registra anche le
	 * destinazioni e le direzioni che arrivano al bordo.
	 */
	private static void scanMoves(Workspace ws, boolean needWhite) {
		final byte[] b = ws.b, reachW = ws.reachW, reachB = ws.reachB;
		Arrays.fill(reachW, (byte) 0);
		Arrays.fill(reachB, (byte) 0);
		final int kp = ws.kp;
		int wMoves = 0, wBlocked = 0, bBlocked = 0;
		if (needWhite) { // i bianchi servono solo a MOB e THREATS; le vie di fuga (LANES) usano solo la raggiungibilita' del Nero
			for (int i = 0; i < ws.nW; i++) {
				int m = scan(b, reachW, ws.wl[i]);
				wMoves += m;
				if (m == 0) {
					wBlocked++;
				}
			}
		}
		int nk = 0, edgeDirs = 0;
		for (int k = 0; k < 4; k++) {
			final int d = DIR[k];
			int q = kp;
			while (true) {
				q += d;
				if (b[q] != EMPTY) { // il Re non parte mai da una citadel: una citadel vuota (CEMPTY) lo ferma
					break;
				}
				reachW[q]++;
				ws.kd[nk++] = q;
				if (EDGE[q]) {
					edgeDirs |= 1 << k;
					break;
				}
			}
		}
		ws.kingMoves = nk;
		ws.edgeDirs = edgeDirs;
		ws.whiteMoves = wMoves + nk;
		int bMoves = 0;
		for (int i = 0; i < ws.nB; i++) {
			int m = scan(b, reachB, ws.bl[i]);
			bMoves += m;
			if (m == 0) {
				bBlocked++;
			}
		}
		ws.blackMoves = bMoves;
		ws.wBlocked = wBlocked;
		ws.bBlocked = bBlocked;
	}

	private static void fillMobility(Workspace ws, double[] out) {
		int reachEdge = 0;
		for (int p : FREE_EDGE) {
			if (ws.reachB[p] > 0) {
				reachEdge++;
			}
		}
		out[F_WHITE_MOVES] = ws.whiteMoves * INV[F_WHITE_MOVES];
		out[F_BLACK_MOVES] = ws.blackMoves * INV[F_BLACK_MOVES];
		out[F_KING_MOVES] = ws.kingMoves * INV[F_KING_MOVES];
		out[F_WHITE_BLOCKED] = ws.wBlocked * INV[F_WHITE_BLOCKED];
		out[F_BLACK_BLOCKED] = ws.bBlocked * INV[F_BLACK_BLOCKED];
		out[F_BLACK_EDGE_REACH] = reachEdge * INV[F_BLACK_EDGE_REACH];
	}

	private static void fillLanes(Workspace ws, double[] out) {
		final byte[] b = ws.b, reachB = ws.reachB;
		final int kp = ws.kp;
		int lanes = 0, unblockable = 0;
		for (int k = 0; k < 4; k++) {
			if ((ws.edgeDirs & (1 << k)) == 0) {
				continue;
			}
			lanes++;
			boolean blockable = false;
			final int d = DIR[k];
			int q = kp;
			do {
				q += d;
				if (reachB[q] > 0) {
					blockable = true;
					break;
				}
			} while (!EDGE[q]);
			if (!blockable) {
				unblockable++;
			}
		}
		out[F_KING_LANES] = lanes * INV[F_KING_LANES];
		out[F_KING_LANES_UNBLOCKABLE] = unblockable * INV[F_KING_LANES_UNBLOCKABLE];

		// fuga in 2 mosse: il Re si sposta (senza catture: approssimazione dichiarata)
		// in una cella c, la sua vecchia casella si libera, e da c si contano le vie
		int cells = 0, forks = 0;
		byte saved = b[kp];
		b[kp] = kp == THRONE_P ? THRONE : EMPTY;
		for (int i = 0; i < ws.kingMoves; i++) {
			int c = ws.kd[i];
			if (EDGE[c]) {
				continue; // gia' contata come via immediata
			}
			int n = lanesFrom(b, c);
			if (n >= 1) {
				cells++;
			}
			if (n >= 2) {
				forks++;
			}
		}
		b[kp] = saved;
		out[F_KING_2STEP_CELLS] = cells * INV[F_KING_2STEP_CELLS];
		out[F_KING_2STEP_FORKS] = forks * INV[F_KING_2STEP_FORKS];
	}

	private static void fillThreats(Workspace ws, double[] out) {
		final byte[] b = ws.b;
		int whiteThr = 0, blackThr = 0;
		for (int i = 0; i < ws.nW; i++) {
			if (capturable(b, ws.reachB, ws.wl[i], true)) {
				whiteThr++;
			}
		}
		for (int i = 0; i < ws.nB; i++) {
			if (capturable(b, ws.reachW, ws.bl[i], false)) {
				blackThr++;
			}
		}
		out[F_WHITE_THREATENED] = whiteThr * INV[F_WHITE_THREATENED];
		out[F_BLACK_THREATENED] = blackThr * INV[F_BLACK_THREATENED];
		out[F_KING_THREAT] = kingCapturable(b, ws.reachB, ws.kp) ? 1.0 : 0.0;
	}

	/**
	 * ESCAPE e SAFEFORK. BFS delle mosse del Re su scacchiera STATICA (il Nero non muove, il Re non cattura spostandosi:
	 * stessa approssimazione dichiarata di {@code F_KING_2STEP_*}), con la cella di partenza liberata. Gli strati sono
	 * le celle raggiungibili in esattamente 1, 2, 3 mosse (ogni cella nel primo strato in cui compare). Un raggio si ferma
	 * alla prima cella non vuota (pedine, trono, citadel) o al bordo: arrivarci e' la fuga.
	 * <ul>
	 * <li>{@code ESC_DIST} = 1/d dove d e' lo strato in cui compare la prima cella di fuga (1..3), 0 se oltre 3 mosse.</li>
	 * <li>{@code ESC_3STEP_CELLS} = celle dello strato 2 da cui il Re ha almeno una via immediata (fuga in 3 mosse).</li>
	 * <li>{@code ESC_EDGE3} = celle di fuga DISTINTE raggiunte entro 3 mosse.</li>
	 * <li>{@code SAFE_FORKS} = celle dello strato 1 con almeno 2 vie immediate in cui, arrivato il Re, nessuna mossa
	 * del Nero lo cattura. Con il Bianco a muovere due vie di fuga e un Re non catturabile sono una vittoria forzata
	 * in 3 semi-mosse (il Nero puo' bloccare una sola via); {@code FeaturesCheck} lo verifica col motore.</li>
	 * </ul>
	 */
	private static void fillEscape(Workspace ws, double[] out, int groups) {
		final byte[] b = ws.b;
		final int kp = ws.kp;
		final boolean wantEsc = (groups & G_ESCAPE) != 0, wantSafe = (groups & G_SAFEFORK) != 0;
		final int maxDepth = wantEsc ? 3 : 1;
		final int[] seen = ws.seen;
		final int stamp = ++ws.stamp;
		int[] cur = ws.bfsA, nxt = ws.bfsB;
		int nCur = 1;
		cur[0] = kp;
		seen[kp] = stamp;
		final byte saved = b[kp];
		b[kp] = kp == THRONE_P ? THRONE : EMPTY;
		int dist = 0, edgeCells = 0, threeStep = 0, safeForks = 0;
		for (int depth = 1; depth <= maxDepth; depth++) {
			int nNxt = 0, edgesHere = 0;
			for (int i = 0; i < nCur; i++) {
				final int c = cur[i];
				for (int k = 0; k < 4; k++) {
					final int d = DIR[k];
					int q = c;
					while (true) {
						q += d;
						if (b[q] != EMPTY) {
							break;
						}
						if (seen[q] != stamp) {
							seen[q] = stamp;
							if (EDGE[q]) {
								edgesHere++;
							} else {
								nxt[nNxt++] = q;
							}
						}
						if (EDGE[q]) {
							break;
						}
					}
				}
			}
			edgeCells += edgesHere;
			if (dist == 0 && edgesHere > 0) {
				dist = depth;
			}
			if (depth == 1 && wantSafe) {
				safeForks = countSafeForks(ws, nxt, nNxt);
			}
			if (depth == 2) {
				for (int i = 0; i < nNxt; i++) {
					if (lanesFrom(b, nxt[i]) >= 1) {
						threeStep++;
					}
				}
			}
			int[] t = cur;
			cur = nxt;
			nxt = t;
			nCur = nNxt;
		}
		b[kp] = saved;
		if (wantEsc) {
			out[F_ESC_DIST] = dist == 0 ? 0.0 : (1.0 / dist) * INV[F_ESC_DIST];
			out[F_ESC_3STEP_CELLS] = threeStep * INV[F_ESC_3STEP_CELLS];
			out[F_ESC_EDGE3] = edgeCells * INV[F_ESC_EDGE3];
		}
		if (wantSafe) {
			out[F_SAFE_FORKS] = safeForks * INV[F_SAFE_FORKS];
		}
	}

	/**
	 * Forchette sicure fra le celle {@code cells[0..n)} (strato 1 della BFS, con la cella di partenza del Re GIA' liberata
	 * in {@code ws.b}): per ogni cella con >= 2 vie immediate si mette il Re li', si ricostruisce la raggiungibilita' del
	 * Nero su quella scacchiera (il Re blocca e la sua vecchia cella e' libera) e si controlla che non sia catturabile.
	 */
	private static int countSafeForks(Workspace ws, int[] cells, int n) {
		final byte[] b = ws.b, rt = ws.reachTmp;
		int safe = 0;
		for (int i = 0; i < n; i++) {
			final int c = cells[i];
			if (lanesFrom(b, c) < 2) {
				continue;
			}
			b[c] = KING;
			Arrays.fill(rt, (byte) 0);
			for (int j = 0; j < ws.nB; j++) {
				scan(b, rt, ws.bl[j]);
			}
			if (!kingCapturable(b, rt, c)) {
				safe++;
			}
			b[c] = EMPTY;
		}
		return safe;
	}

	/** Cella di bordo e passo verso l'interno dei 16 gate: righe/colonne 1,2,6,7 viste da entrambi gli estremi. */
	private static final int[] GATE_CELL = new int[16], GATE_STEP = new int[16];
	static {
		int n = 0;
		for (int l : new int[] { 1, 2, 6, 7 }) {
			GATE_CELL[n] = idx(l, 0);
			GATE_STEP[n++] = 1;
			GATE_CELL[n] = idx(l, 8);
			GATE_STEP[n++] = -1;
			GATE_CELL[n] = idx(0, l);
			GATE_STEP[n++] = W;
			GATE_CELL[n] = idx(8, l);
			GATE_STEP[n++] = -W;
		}
	}

	/**
	 * GATES: per ciascuno dei 16 gate (la cella di fuga all'estremo di una riga/colonna 1, 2, 6, 7) la "corsa libera" e'
	 * il numero di celle vuote consecutive dal gate verso l'interno (si ferma alla prima pedina, al Re, al trono o a una
	 * citadel). {@code GATES_CLOSED} = gate con corsa 0 (cella di fuga occupata); {@code GATE_RUN} = somma delle corse.
	 */
	private static void fillGates(Workspace ws, double[] out) {
		final byte[] b = ws.b;
		int closed = 0, runSum = 0;
		for (int g = 0; g < 16; g++) {
			final int st = GATE_STEP[g];
			int q = GATE_CELL[g], run = 0;
			while (b[q] == EMPTY) {
				run++;
				q += st;
			}
			if (run == 0) {
				closed++;
			}
			runSum += run;
		}
		out[F_GATES_CLOSED] = closed * INV[F_GATES_CLOSED];
		out[F_GATE_RUN] = runSum * INV[F_GATE_RUN];
	}

	/** Flood fill dal Re sulle celle in cui puo' camminare: misura quanto e' "accerchiato". */
	private static void fillFlood(Workspace ws, double[] out) {
		final byte[] b = ws.b;
		final int[] seen = ws.seen, st = ws.stack;
		final int stamp = ++ws.stamp;
		int sp = 0, region = 0, edgeCells = 0, sides = 0;
		seen[ws.kp] = stamp;
		st[sp++] = ws.kp;
		while (sp > 0) {
			int p = st[--sp];
			for (int k = 0; k < 4; k++) {
				int q = p + DIR[k];
				if (seen[q] == stamp || b[q] != EMPTY) {
					continue;
				}
				seen[q] = stamp;
				region++;
				if (EDGE[q]) {
					edgeCells++;
					sides |= SIDE[q];
				} else {
					st[sp++] = q; // dal bordo non si prosegue: arrivarci e' gia' la fuga
				}
			}
		}
		out[F_FLOOD_REGION] = region * INV[F_FLOOD_REGION];
		out[F_FLOOD_EDGE] = edgeCells * INV[F_FLOOD_EDGE];
		out[F_FLOOD_SIDES] = (Integer.bitCount(sides)) * INV[F_FLOOD_SIDES];
	}

	// ---------------------------------------------------------------- helper

	/** Esiste una pedina amica (tipo a o c) tra gli 8 vicini di p? */
	private static boolean friendNear(byte[] b, int p, byte a, byte c) {
		byte v;
		v = b[p - W - 1];
		if (v == a || v == c) {
			return true;
		}
		v = b[p - W];
		if (v == a || v == c) {
			return true;
		}
		v = b[p - W + 1];
		if (v == a || v == c) {
			return true;
		}
		v = b[p - 1];
		if (v == a || v == c) {
			return true;
		}
		v = b[p + 1];
		if (v == a || v == c) {
			return true;
		}
		v = b[p + W - 1];
		if (v == a || v == c) {
			return true;
		}
		v = b[p + W];
		if (v == a || v == c) {
			return true;
		}
		v = b[p + W + 1];
		return v == a || v == c;
	}

	/**
	 * Scansiona le mosse del pezzo in p con la stessa semantica di
	 * {@code AshtonRules.generateMoves}: qualsiasi cella non vuota ferma il
	 * raggio (pedine, trono, bordo); una citadel si puo' toccare solo partendo
	 * da una citadel e a distanza <= 5. Incrementa {@code reach} per ogni
	 * destinazione e ritorna il numero di mosse.
	 */
	private static int scan(byte[] b, byte[] reach, int p) {
		if (CIT[p]) {
			return scanFromCitadel(b, reach, p);
		}
		// caso comune: una citadel vuota (CEMPTY) ferma il raggio esattamente come una pedina
		int count = 0;
		for (int k = 0; k < 4; k++) {
			final int d = DIR[k];
			int q = p + d;
			while (b[q] == EMPTY) {
				reach[q]++;
				count++;
				q += d;
			}
		}
		return count;
	}

	/** Pezzo che parte da una citadel (pedine nere non ancora uscite dal campo): puo' attraversare/atterrare sulle citadel entro 5 celle. */
	private static int scanFromCitadel(byte[] b, byte[] reach, int p) {
		int count = 0;
		for (int k = 0; k < 4; k++) {
			final int d = DIR[k];
			int q = p;
			int step = 0;
			while (true) {
				q += d;
				step++;
				byte v = b[q];
				if (v != EMPTY && (v != CEMPTY || step > 5)) {
					break;
				}
				reach[q]++;
				count++;
			}
		}
		return count;
	}

	/** Vie di fuga immediate da c (il Re vi parte: non e' una citadel). */
	private static int lanesFrom(byte[] b, int c) {
		int n = 0;
		for (int k = 0; k < 4; k++) {
			final int d = DIR[k];
			int q = c;
			while (true) {
				q += d;
				if (b[q] != EMPTY) {
					break;
				}
				if (EDGE[q]) {
					n++;
					break;
				}
			}
		}
		return n;
	}

	/** Lato ostile per la cattura "a tenaglia" del Re lontano dal trono: pedina nera o citadel. */
	private static boolean hostileForKing(byte[] b, int p) {
		return b[p] == BLACK || CIT[p];
	}

	/** Muro valido per il Nero che cattura una pedina bianca (si veda AshtonRules.isWallForBlackCapture). */
	private static boolean wallForBlack(byte[] b, int p) {
		return p == THRONE_P || b[p] == BLACK || CIT[p];
	}

	/** Muro valido per il Bianco che cattura una pedina nera (AshtonRules.isWallForWhiteCapture). */
	private static boolean wallForWhite(byte[] b, int p) {
		byte v = b[p];
		return v == WHITE || v == THRONE || v == KING || (CIT[p] && !TIP[p]);
	}

	/**
	 * La pedina in p (bianca se {@code victimIsWhite}, altrimenti nera) puo'
	 * essere catturata con UNA mossa dell'avversario? Serve una cella adiacente
	 * vuota raggiungibile dall'attaccante ({@code reach}) con, dal lato opposto,
	 * un "muro" valido. Il Re e' trattato a parte ({@link #kingCapturable}).
	 */
	private static boolean capturable(byte[] b, byte[] reach, int p, boolean victimIsWhite) {
		for (int axis = 0; axis < 2; axis++) {
			int a = p + (axis == 0 ? -W : -1);
			int c = p + (axis == 0 ? W : 1);
			if (isFree(b[a]) && reach[a] > 0 && (victimIsWhite ? wallForBlack(b, c) : wallForWhite(b, c))) {
				return true;
			}
			if (isFree(b[c]) && reach[c] > 0 && (victimIsWhite ? wallForBlack(b, a) : wallForWhite(b, a))) {
				return true;
			}
		}
		return false;
	}

	/** Il Nero puo' catturare il Re con una mossa? (stesse tre regole di AshtonRules.isKingCaptured) */
	private static boolean kingCapturable(byte[] b, byte[] reachB, int kp) {
		if (kp == THRONE_P || MAN[kp] == 1) {
			// cattura a 4 lati (sul trono) o a 3 lati (accanto al trono, che conta come quarto lato):
			// tutti i lati tranne uno sono neri, e quell'unico lato e' una cella vuota raggiungibile dal Nero
			int missing = -1;
			for (int k = 0; k < 4; k++) {
				int s = kp + DIR[k];
				if (s == THRONE_P) {
					continue;
				}
				if (b[s] != BLACK) {
					if (missing >= 0) {
						return false;
					}
					missing = s;
				}
			}
			return missing >= 0 && isFree(b[missing]) && reachB[missing] > 0;
		}
		for (int axis = 0; axis < 2; axis++) {
			int a = kp + (axis == 0 ? -W : -1);
			int c = kp + (axis == 0 ? W : 1);
			if (isFree(b[a]) && reachB[a] > 0 && hostileForKing(b, c)) {
				return true;
			}
			if (isFree(b[c]) && reachB[c] > 0 && hostileForKing(b, a)) {
				return true;
			}
		}
		return false;
	}
}
