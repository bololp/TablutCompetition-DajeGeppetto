package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.engine.ZobristHash;

/**
 * Negamax con potatura alpha-beta e iterative deepening.
 * <p>
 * Iterative deepening = si cerca prima a profondita' 1, poi 2, poi 3, ... Non
 * e' uno spreco: ogni iterazione e' molto piu' rapida della successiva (la
 * dimensione dell'albero cresce esponenzialmente con la profondita', quindi
 * le iterazioni precedenti costano relativamente poco), e permette due cose
 * fondamentali per una competizione con un timeout fisso:
 * <ol>
 * <li>si puo' interrompere la ricerca in QUALSIASI momento e avere comunque
 * una mossa valida pronta (quella dell'ultima profondita' completata);</li>
 * <li>la mossa migliore trovata a profondita' N viene provata per prima a
 * profondita' N+1 (move ordering), il che rende il taglio alpha-beta molto
 * piu' efficace.</li>
 * </ol>
 * <p>
 * Questa classe dipende solo da {@link FastState} (posizione e regole) e da
 * {@link Evaluator} (la valutazione di una posizione): per passare a una rete
 * neurale basta passare al costruttore un {@code Evaluator} diverso, senza
 * toccare questa classe.
 * <p>
 * <b>Rappresentazione dello stato (make/unmake).</b> La ricerca possiede UNA {@link FastState} di lavoro per thread e la
 * muta sul posto: {@link FastState#makeMove} prima di scendere in un figlio, {@link FastState#unmakeMove} al ritorno
 * (un {@link UndoInfo} per ply). Le mosse sono {@code int} ({@link Move}) scritti in buffer {@code int[]} preallocati
 * per ply, l'hash Zobrist e' aggiornato incrementalmente dalla stessa {@code makeMove}, la storia di ripetizione e' uno
 * stack di {@code long} e la lettura della TT riusa un unico {@code ProbeResult}. Risultato: NESSUNA allocazione per
 * nodo (prima: uno {@code State} clonato + {@code Action} con due stringhe + boxing di {@code Long} per ogni figlio).
 * Le regole, l'ordine di generazione delle mosse, i punteggi e il tie-break sono IDENTICI a quelli della versione su
 * {@code State}/{@code AshtonRules} (congelata in {@code LegacyNegamax}): {@code SearchEquivalenceCheck} verifica che
 * a parita' di configurazione punteggio, mossa, profondita' e numero di nodi coincidano.
 * <p>
 * <b>Transposition table</b> ({@link TranspositionTable}, indicizzata con
 * {@link ZobristHash}): posizioni raggiunte da mosse diverse (frequenti in
 * Tablut, dove pedine come torri possono attraversare piu' caselle) vengono
 * riconosciute come identiche ed evitano di riesplorare lo stesso sottoalbero.
 * Fondamentale anche per l'iterative deepening: la mossa migliore trovata per
 * un nodo alla profondita' N viene riusata per ordinare le mosse quando lo
 * stesso nodo si ripresenta alla profondita' N+1, rendendo il taglio
 * alpha-beta molto piu' efficace. La tabella e' un campo di istanza: persiste
 * per tutta la vita di questo oggetto {@code Negamax} (tipicamente l'intera
 * partita), non solo per una singola chiamata a {@link #findBestMove}.
 * <p>
 * <b>Pareggio per ripetizione</b>: la regola ufficiale (slide 9; server con
 * {@code repeatedState=0} di default) e' che la STESSA posizione (board +
 * turno, la stessa uguaglianza usata da {@code State.equals}) raggiunta una
 * SECONDA volta dall'ultima cattura e' pareggio. Questo e' rilevato tenendo
 * traccia, lungo il percorso di ricerca corrente, delle posizioni gia'
 * visitate dall'ultima cattura (azzerate ad ogni cattura, dato che nessuna
 * posizione precedente puo' piu' ripresentarsi una volta sparita una
 * pedina). {@link #findBestMove(State, long, List)} accetta la storia REALE
 * della partita (tenuta dal chiamante, tipicamente {@code TablutNegamaxClient})
 * cosi' la ricerca puo' riconoscere non solo i cicli interni al proprio
 * albero, ma anche le mosse che ricreerebbero una posizione gia' occorsa
 * davvero in partita.
 * <p>
 * <b>Interazione con la transposition table (Graph History Interaction)</b>:
 * il valore "pareggio per ripetizione" di un nodo NON e' funzione della sola
 * board, dipende da quali posizioni sono gia' state visitate in QUESTO
 * specifico percorso - due percorsi diversi possono raggiungere la stessa
 * board con storie diverse, e quindi con esiti di ripetizione diversi. Una TT
 * indicizzata solo sulla board rischierebbe di riusare, per un percorso, un
 * valore calcolato per un percorso diverso con una storia di ripetizioni
 * incompatibile: un problema noto in letteratura come "Graph History
 * Interaction". La contromisura qui: si tiene un contatore globale di quante
 * volte una ripetizione e' stata rilevata durante l'intera ricerca; un nodo
 * viene memorizzato in TT solo se quel contatore non e' cambiato mentre si
 * esplorava il SUO sottoalbero (cioe' nessuna ripetizione e' mai stata
 * incontrata sotto di lui) - la stragrande maggioranza dei nodi, che non
 * incontra mai ripetizioni, continua a beneficiare pienamente della cache;
 * solo i (rari) nodi il cui valore potrebbe dipendere dalla storia vengono
 * esclusi dalla memorizzazione, in modo conservativo e verificabile.
 * <p>
 * <b>Principal Variation Search (PVS)</b>: in ogni nodo la PRIMA mossa (quella
 * che l'ordinamento ritiene la migliore: mossa della TT / dell'iterazione
 * precedente) viene cercata a finestra piena (-beta,-alpha); le successive con
 * una finestra NULLA (-alpha-eps,-alpha), molto piu' stretta e quindi molto
 * piu' potata, che si limita a rispondere "questa mossa e' meglio di quanto
 * ho gia' trovato, si' o no?". Solo se la risposta e' si' (valore tra alpha e
 * beta) la mossa viene ri-cercata a finestra piena per ottenerne il valore
 * esatto. Quando l'ordinamento e' buono (la prima mossa e' spesso la migliore)
 * le ri-ricerche sono rare e si risparmiano nodi; quando e' pessimo le
 * ri-ricerche possono costare piu' di quanto si risparmia: per questo il
 * guadagno va MISURATO, non assunto (si veda {@code PVSCheck} e il README).
 * <p>
 * <b>Perche' il risultato non cambia</b>: con una ricerca fail-soft un valore
 * di ritorno <= alpha e' un limite superiore del valore vero, uno >= beta un
 * limite inferiore, uno strettamente in mezzo e' esatto. Lo scout con finestra
 * (alpha,alpha+eps) restituisce quindi: un limite superiore <= alpha (la mossa non
 * migliora: si scarta, e' provato), oppure un valore > alpha (si ri-cerca a finestra
 * piena: valore esatto). Il valore alla radice e' percio' IDENTICO a quello
 * dell'alpha-beta puro (verificato da {@code PVSCheck}); cambia solo il numero di nodi.
 * {@code eps} e' piccolo ma > 0 perche' i punteggi sono {@code double}: con una
 * finestra di ampiezza 0 non si potrebbe distinguere "uguale" da "migliore".
 * L'interazione con la TT e' automatica: i flag EXACT/LOWER/UPPER sono gia'
 * calcolati rispetto alla finestra effettivamente usata.
 * <p>
 * <b>Ordinamento delle mosse: killer moves e history heuristic</b>. In un nodo interno
 * le mosse sono provate in quest'ordine: (1) la mossa della transposition table, (2) le
 * due "killer" di quel ply (mosse che hanno provocato un taglio beta in un nodo FRATELLO,
 * cioe' alla stessa distanza dalla radice), (3) le altre per punteggio "history"
 * (tabella [lato][casella di partenza][casella di arrivo] incrementata di profondita'^2
 * ogni volta che quella mossa provoca un taglio: impara quali mosse sono "buone in
 * generale" in questa ricerca). La selezione e' PIGRA (si estrae la migliore tra le rimaste
 * solo quando serve), perche' i tagli arrivano quasi sempre entro le prime 1-3 mosse.
 * E' SOLO ordinamento: non puo' cambiare il valore di nessun nodo, solo quanti nodi servono
 * per calcolarlo (verificato da {@code SearchOptimizationsCheck} con un alpha-beta
 * indipendente). Alla radice l'ordine NON cambia (miglior mossa precedente in testa, poi
 * ordine di generazione) cosi' il tie-break tra mosse di pari valore resta lo stesso di
 * prima. Killer: azzerate a ogni {@code findBestMove}; history: dimezzata a ogni
 * {@code findBestMove} ("invecchiamento") invece di azzerata, perche' tra una mossa e la
 * successiva la posizione cambia poco.
 * <p>
 * <b>Lazy SMP (ricerca parallela)</b>. Con {@code threads > 1} (default: 4, o meno se la macchina ha
 * meno core; si veda {@link #defaultThreads()}) {@link #findBestMove} avvia {@code threads-1} "helper": altre
 * istanze di questa stessa classe che cercano la STESSA posizione radice con iterative deepening e condividono
 * con il thread principale SOLO la transposition table (lock-free, si veda {@link TranspositionTable}) e
 * l'evaluator. Tutto il resto (killer, history, contatori, contatore di ripetizioni, rilevamento dei cicli,
 * orologio) e' privato di ciascun thread, quindi la sicurezza GHI descritta sopra vale per ognuno: un nodo entra
 * in TT solo se il SUO sottoalbero non ha incontrato ripetizioni, cioe' con un valore indipendente dal percorso, e
 * quindi riusabile da qualunque altro thread. Gli helper non coordinano il lavoro: si diversificano saltando
 * alcune profondita' (schema usato da Stockfish, {@link #skipDepth}) e si aiutano riempiendo la TT di bound e mosse
 * migliori che il principale trovera' gia' pronti.
 * <p>
 * Il thread chiamante resta AUTOREVOLE: il risultato e' il suo, tranne se un helper ha completato un'iterazione
 * PIU' PROFONDA e il principale non ha gia' provato una vittoria/sconfitta forzata ({@link #chooseWorker}).
 * Allo scadere del tempo (o quando il principale termina) un flag di stop, nuovo per ogni ricerca, ferma gli
 * helper entro ~2048 nodi; un helper ancora impegnato non viene riutilizzato dalla ricerca successiva e un
 * errore in un helper non puo' mai far fallire la mossa. Con {@code threads = 1} (o TT disattivata, che
 * renderebbe inutile la condivisione) il comportamento e' IDENTICO alla versione a thread singolo.
 * <p>
 * Un'istanza non va usata da piu' thread contemporaneamente (una sola {@code findBestMove} alla volta).
 * {@link #shutdown()} rilascia i thread degli helper (sono comunque "daemon").
 */
public class Negamax {

	/** Punteggio "vittoria": molto piu' alto di qualunque valore euristico plausibile. */
	// package-private (non private): permette ai test (stesso package) di verificare i punteggi esatti
	// restituiti per vittoria/sconfitta, invece di dover indovinare una soglia arbitraria.
	static final double WIN_SCORE = 1_000_000;

	/** Margine di sicurezza tolto al tempo a disposizione, per lasciare tempo
	 *  di risalire la ricorsione e inviare la mossa al server prima dello scadere
	 *  del timeout imposto dalla competizione. */
	private static final long SAFETY_MARGIN_MILLIS = 300;
	/** Margine di sicurezza di QUESTA istanza (default {@link #SAFETY_MARGIN_MILLIS}); si cambia con {@link #setSafetyMarginMillis}. */
	private long safetyMarginMillis = SAFETY_MARGIN_MILLIS;

	/** Modalita' di ordinamento delle mosse (maschera di bit): nessuno (solo mossa TT), killer, history. */
	public static final int ORDER_NONE = 0;
	public static final int ORDER_KILLERS = 1;
	public static final int ORDER_HISTORY = 2;
	public static final int ORDER_DEFAULT = ORDER_KILLERS | ORDER_HISTORY;
	/**
	 * Opzioni di ordinamento ESTESE (tutte spente nel default; ognuna si accende da sola, cosi' si misura una alla volta). Sono
	 * SOLO ordinamento: il valore di ogni nodo non cambia, cambia quanti nodi servono ({@code OrderingExtCheck} lo verifica
	 * contro l'alpha-beta puro).
	 * <ul>
	 * <li>{@link #ORDER_WINS}: le mosse che vincono subito (fuga del Re; cattura del Re) subito dopo la mossa TT;</li>
	 * <li>{@link #ORDER_CAPTURES}: le mosse che catturano, per numero di catture, dopo killer e countermove e prima delle quiete;</li>
	 * <li>{@link #ORDER_CAPTURES_FIRST}: come sopra ma PRIMA delle killer;</li>
	 * <li>{@link #ORDER_COUNTER}: countermove (la risposta che ha tagliato l'ultima volta dopo la stessa mossa avversaria);</li>
	 * <li>{@link #ORDER_KILLER_CARRY}: le killer passano da una mossa alla successiva (slittate di 2 ply) invece di essere azzerate;</li>
	 * <li>{@link #ORDER_ROOT_SCORES}: alla radice le mosse sono ordinate per punteggio dell'iterazione precedente (cambia il tie-break
	 * tra mosse di pari valore).</li>
	 * </ul>
	 */
	public static final int ORDER_WINS = 4;
	public static final int ORDER_CAPTURES = 8;
	public static final int ORDER_CAPTURES_FIRST = 16;
	public static final int ORDER_COUNTER = 32;
	public static final int ORDER_KILLER_CARRY = 64;
	public static final int ORDER_ROOT_SCORES = 128;
	/**
	 * Ordinamento consigliato dalle misure (si veda README_ORDERING): il default storico piu' vittorie immediate, catture dopo le
	 * killer e ordine della radice per punteggio. NON e' il default dei costruttori (che resta {@link #ORDER_DEFAULT}, cosi' i test
	 * di equivalenza con il motore storico restano validi): lo si passa esplicitamente (client, arena con {@code ord=rec:}).
	 */
	public static final int ORDER_RECOMMENDED = ORDER_DEFAULT | ORDER_WINS | ORDER_CAPTURES | ORDER_ROOT_SCORES;
	/** Le opzioni che richiedono il punteggio esteso ({@link #computeOrderScoresExt}) nei nodi interni. */
	static final int ORDER_EXT_MASK = ORDER_WINS | ORDER_CAPTURES | ORDER_CAPTURES_FIRST | ORDER_COUNTER;

	// Livelli del punteggio di ordinamento esteso (mai usati dal punteggio base): TT > vittoria immediata > [catture] > killer >
	// countermove > [catture] > quiete per history. Gli intervalli non si sovrappongono (history < 2^26 per costruzione).
	static final int SC_TT = Integer.MAX_VALUE;
	static final int SC_WIN = Integer.MAX_VALUE - 1;
	static final int SC_KILLER0_A = Integer.MAX_VALUE - 2, SC_KILLER1_A = Integer.MAX_VALUE - 3, SC_COUNTER_A = Integer.MAX_VALUE - 4;
	static final int SC_CAPTURE_BASE_A = 1 << 29; // catture dopo killer/countermove
	static final int SC_KILLER0_B = (1 << 29) + 2, SC_KILLER1_B = (1 << 29) + 1, SC_COUNTER_B = 1 << 29;
	static final int SC_CAPTURE_BASE_B = 1 << 30; // catture prima delle killer
	static final int SC_CAPTURE_STEP = 1 << 26;
	static final int SC_HISTORY_CAP = (1 << 26) - 1;

	/** Sopra questo valore la tabella history viene dimezzata (evita overflow e mantiene il peso dei tagli recenti). */
	private static final int HISTORY_LIMIT = 1 << 24;

	/** Ampiezza della finestra nulla dello scout PVS (i punteggi sono double: serve eps > 0). */
	static final double PVS_EPSILON = 1e-6;

	/** Ogni quanti nodi controllare l'orologio (System.currentTimeMillis ha un
	 *  costo non nullo se chiamato ad ogni singolo nodo). */
	private static final int TIME_CHECK_INTERVAL = 2048;

	/** Dimensione di default della transposition table (voci arrotondate alla
	 *  potenza di 2 superiore): 2^22 = ~4 milioni di voci, ~100 MB (3 long per voce).
	 *  Con piu' thread la tabella si riempie piu' in fretta, e in una ricerca da 60 s
	 *  una tabella da 1M di voci verrebbe sovrascritta continuamente. */
	public static final int DEFAULT_TT_SIZE = 1 << 22;

	private final Evaluator evaluator;
	private final int maxDepth;
	private final boolean useTT;
	private final boolean usePVS;
	private final int orderingMode;
	// ---- Lazy SMP ----
	private int threads;
	private final boolean isHelper;
	private SearchControl control; // flag di stop della ricerca corrente (null = thread singolo)
	private ExecutorService pool;
	private Negamax[] helpers;
	private Future<?>[] helperFutures;
	volatile Throwable helperError;
	/** Migliore mossa (compatta, {@link Move}) dell'ultima ricerca alla radice di QUESTO worker (usata per adottare il risultato di un helper). */
	int lastBestMove = Move.NONE;
	/** killers[ply][0..1]: indice mossa (da*81+a, si veda {@link Move#index}) oppure -1. */
	final int[][] killers;
	/** historyScore[lato][da*81+a]; lato 0 = Bianco, 1 = Nero. */
	final int[][] historyScore = new int[2][Move.INDEX_RANGE];
	/** countermove[lato che deve muovere][indice della mossa precedente] = indice della risposta che ha tagliato, oppure -1 (solo con ORDER_COUNTER). */
	final int[][] counterMove;
	/** Indice della mossa che ha portato al nodo di stack {@code sp} (da*81+a), oppure -1 alla radice. */
	int[] prevMoveAt; // package-private: lo legge OrderingExtCheck
	/** Punteggi dell'ultima iterazione per ogni mossa della radice (parallelo a moveStack[0]); solo con ORDER_ROOT_SCORES. */
	final double[] rootScores = new double[FastState.MAX_MOVES]; // package-private: lo legge OrderingExtCheck
	/** Osservabili per i test (non cambiano la ricerca): mosse alla radice dell'ultima ricerca di questo worker, chiamate del punteggio esteso, verifica della countermove. */
	int rootMoveCountForTest;
	long extScoreCalls;
	boolean debugVerifyCounter;
	int debugCounterViolations;
	int[] rootMovesForTest() {
		return moveStack[0];
	}
	/** Profondita' dell'iterazione corrente alla radice: ply = rootIterDepth - depth. */
	int rootIterDepth; // package-private: lo legge OrderingExtCheck
	// package-private (non private): permette ai test white-box di sondare
	// direttamente la tabella dopo una ricerca, per confermare che un'entry sia
	// stata davvero memorizzata (e non solo che il punteggio finale sia giusto).
	final TranspositionTable tt;

	// ---- stato di lavoro di QUESTO worker (privato: ogni thread di ricerca ha il suo) ----
	/** La posizione che la ricerca muta con make/unmake. Mai condivisa tra thread. */
	private final FastState st = new FastState();
	/** Un {@link UndoInfo} per ply (indice = ply del nodo che fa la mossa). */
	private UndoInfo[] undoStack;
	/** Buffer delle mosse generate per ply (indice = ply del nodo; 0 = radice). */
	private int[][] moveStack;
	/** Punteggi di ordinamento per ply (stessa indicizzazione di {@link #moveStack}). */
	private int[][] scoreStack;
	/** Storia di ripetizione: hash delle posizioni dall'ultima cattura, radice compresa, nel percorso corrente. */
	private long[] histKeys = new long[256];
	private int histLen;
	/** Inizio del segmento "dall'ultima cattura" dentro {@link #histKeys}: le posizioni prima di una cattura non possono ripresentarsi. */
	private int histBase;
	/** Risultato dell'ultima interrogazione della TT (riusato: si legge subito, prima di ricorrere). */
	private final TranspositionTable.ProbeResult probe = new TranspositionTable.ProbeResult();

	private long deadline;
	private boolean timeUp;
	private int nodeCounter;
	/** Incrementato ogni volta che una ripetizione viene rilevata durante la
	 *  ricerca corrente: usato per decidere quali nodi e' sicuro memorizzare
	 *  in transposition table (si veda il javadoc di classe, sezione GHI).
	 *  Package-private (non private): il test white-box lo legge per
	 *  confermare che il rilevamento sia davvero scattato. */
	long repetitionEventCounter;

	/** Statistiche dell'ultima chiamata a {@link #findBestMove}, utili per il debug/tuning. */
	public int lastDepthReached;
	public long lastNodeCount;
	public double lastScore;
	/** PVS: quante mosse sono state "scoutate" a finestra nulla e quante di queste hanno richiesto la ri-ricerca a finestra piena. */
	public long lastPvsScouts;
	public long lastPvsReSearches;
	/** Lazy SMP: nodi di TUTTI i thread, thread effettivamente usati, helper andati in errore, indice dell'helper adottato (-1 = nessuno). */
	public long lastTotalNodeCount;
	public int lastThreadsUsed = 1;
	public int lastHelperErrors;
	public int lastAdoptedWorker = -1;
	/** Profondita' completata dal solo thread principale (prima dell'eventuale adozione del risultato di un helper). */
	public int lastMainDepth;
	private long pvsScouts;
	private long pvsReSearches;

	public Negamax(Evaluator evaluator) {
		this(evaluator, 40);
	}

	public Negamax(Evaluator evaluator, int maxDepth) {
		this(evaluator, maxDepth, true, DEFAULT_TT_SIZE);
	}

	/**
	 * @param useTT false per disabilitare la transposition table (utile solo
	 *              per confronti/debug: con lo stesso identico alpha-beta,
	 *              deve dare sempre lo stesso identico punteggio/mossa di
	 *              {@code true}, solo piu' lentamente - se cosi' non fosse,
	 *              ci sarebbe un bug nella gestione dei bound EXACT/LOWER/UPPER)
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize) {
		this(evaluator, maxDepth, useTT, ttSize, true);
	}

	/**
	 * @param usePVS false per usare l'alpha-beta classico (utile solo per confronti/debug: il valore
	 *               alla radice deve essere identico, cambia solo il numero di nodi)
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS) {
		this(evaluator, maxDepth, useTT, ttSize, usePVS, ORDER_DEFAULT);
	}

	/**
	 * @param orderingMode {@link #ORDER_NONE} (solo mossa TT, come prima), {@link #ORDER_KILLERS}, {@link #ORDER_HISTORY}
	 *                     o la loro unione ({@link #ORDER_DEFAULT}); utile per confronti: il valore di ogni nodo e' lo stesso
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS, int orderingMode) {
		this(evaluator, maxDepth, useTT, ttSize, usePVS, orderingMode, defaultThreads());
	}

	/** Proprieta' di sistema {@code -Dtablut.tt2=true}: i costruttori che non lo specificano usano la TT a due slot (utile per rieseguire un'intera suite di test con l'altra politica). */
	static boolean defaultTwoSlotTT() {
		return Boolean.parseBoolean(System.getProperty("tablut.tt2", "false"));
	}

	/**
	 * @param threads numero di thread di ricerca (Lazy SMP), 1 = ricerca a thread singolo identica a quella storica.
	 *                Il default degli altri costruttori e' {@link #defaultThreads()}.
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS, int orderingMode,
			int threads) {
		this(evaluator, maxDepth, useTT, ttSize, usePVS, orderingMode, threads, defaultTwoSlotTT());
	}

	/**
	 * @param twoSlotTT vero per la transposition table a bucket di due slot (depth-preferred con aging + sempre-sostituisci, si
	 *                  veda {@link TranspositionTable}); {@code ttSize} resta il numero totale di voci (stessa memoria)
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS, int orderingMode,
			int threads, boolean twoSlotTT) {
		this(evaluator, maxDepth, useTT, ttSize, usePVS, orderingMode, threads, twoSlotTT, null);
	}

	/**
	 * @param table la transposition table da usare (condivisa dagli helper Lazy SMP), oppure {@code null} per crearne una
	 *              {@link TranspositionTable} di {@code ttSize} voci; serve a sostituirla, per esempio con {@link ExactTranspositionTable}.
	 *              Ignorata se {@code useTT} e' falso.
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS, int orderingMode,
			int threads, boolean twoSlotTT, TranspositionTable table) {
		this.evaluator = evaluator;
		this.maxDepth = maxDepth;
		this.useTT = useTT;
		this.usePVS = usePVS;
		this.orderingMode = orderingMode;
		this.killers = newKillers(maxDepth);
		this.counterMove = newCounterMove(orderingMode);
		this.tt = useTT ? (table != null ? table : new TranspositionTable(ttSize, twoSlotTT)) : null;
		this.threads = Math.max(1, Math.min(MAX_THREADS, threads));
		this.isHelper = false;
		allocStacks(maxDepth + 3);
	}

	/** Costruttore degli helper Lazy SMP: stessa configurazione e STESSA transposition table del principale, stato di ricerca proprio. */
	private Negamax(Negamax main) {
		this.evaluator = main.evaluator;
		this.maxDepth = main.maxDepth;
		this.useTT = main.useTT;
		this.usePVS = main.usePVS;
		this.orderingMode = main.orderingMode;
		this.killers = newKillers(main.maxDepth);
		this.counterMove = newCounterMove(main.orderingMode); // tabella PRIVATA dell'helper (non condivisa)
		this.tt = main.tt;
		this.threads = 1;
		this.isHelper = true;
		allocStacks(main.maxDepth + 3);
	}

	private static int[][] newCounterMove(int orderingMode) {
		if ((orderingMode & ORDER_COUNTER) == 0) {
			return null;
		}
		int[][] c = new int[2][Move.INDEX_RANGE];
		for (int[] side : c) {
			java.util.Arrays.fill(side, -1);
		}
		return c;
	}

	private static int[][] newKillers(int maxDepth) {
		int[][] k = new int[Math.max(2, maxDepth + 2)][2];
		for (int[] e : k) {
			e[0] = -1;
			e[1] = -1;
		}
		return k;
	}

	/** (Ri)alloca gli stack per ply con {@code n} livelli. Gli stack esistenti non si accorciano. */
	private void allocStacks(int n) {
		final int old = undoStack == null ? 0 : undoStack.length;
		if (n <= old) {
			return;
		}
		prevMoveAt = prevMoveAt == null ? new int[n] : java.util.Arrays.copyOf(prevMoveAt, n);
		UndoInfo[] u = new UndoInfo[n];
		int[][] mv = new int[n][];
		int[][] sc = new int[n][];
		for (int i = 0; i < n; i++) {
			if (i < old) {
				u[i] = undoStack[i];
				mv[i] = moveStack[i];
				sc[i] = scoreStack[i];
			} else {
				u[i] = new UndoInfo();
				mv[i] = new int[FastState.MAX_MOVES];
				sc[i] = new int[FastState.MAX_MOVES];
			}
		}
		undoStack = u;
		moveStack = mv;
		scoreStack = sc;
	}

	private void ensureHist(int capacity) {
		if (histKeys.length < capacity) {
			histKeys = java.util.Arrays.copyOf(histKeys, Math.max(capacity, histKeys.length * 2));
		}
	}

	// ------------------------------------------------------------------------------------------ API pubblica

	/**
	 * Come {@link #findBestMove(State, long, List)}, ma senza storia reale di
	 * partita nota: la ricerca rileva comunque i cicli interni al proprio
	 * stesso albero (tornare due volte sulla posizione radice, per esempio),
	 * ma non puo' sapere se una posizione ha gia' occorso PRIMA di questa
	 * chiamata (es. nelle partite di training di {@code SelfPlayTrainer},
	 * dove ogni episodio e' una partita a se' e non ha senso parlare di
	 * "storia reale" tra chiamate diverse).
	 */
	public Action findBestMove(State rootState, long timeBudgetMillis) {
		return findBestMove(rootState, timeBudgetMillis, Collections.<Long>emptyList());
	}

	/**
	 * Trova la migliore mossa per il giocatore di turno in {@code rootState}, usando al massimo {@code timeBudgetMillis}
	 * millisecondi. Forma "di confine" per chi ha uno {@link State} del dominio (client, trainer, test): converte la
	 * radice in {@link FastState} (O(81), una volta per mossa) e la mossa trovata in {@link Action}. La ricerca vera e'
	 * quella di {@link #findBestMove(FastState, long, long[], int)}.
	 *
	 * @param historySinceLastCapture hash Zobrist ({@link ZobristHash}, gli stessi di {@code ZobristHasher.fullHash})
	 *        delle posizioni REALMENTE occorse in partita dall'ultima cattura (radice inclusa), piu' vecchia per prima. Il
	 *        chiamante (tipicamente {@code TablutNegamaxClient}) la mantiene aggiornata; qui viene copiata, mai modificata.
	 * @return la miglior mossa trovata, oppure {@code null} se in {@code rootState} il giocatore di turno non ha mosse
	 *         legali (situazione che, di norma, il chiamante non dovrebbe nemmeno raggiungere: se non si hanno mosse
	 *         legali si e' gia' perso).
	 */
	public Action findBestMove(State rootState, long timeBudgetMillis, List<Long> historySinceLastCapture) {
		int mv = findBestMove(FastState.fromState(rootState), timeBudgetMillis, historySinceLastCapture);
		return mv == Move.NONE ? null : Move.toAction(mv, rootState.getTurn());
	}

	/** Come {@link #findBestMove(FastState, long, long[], int)} senza storia. Ritorna la mossa compatta, o {@link Move#NONE}. */
	public int findBestMove(FastState rootState, long timeBudgetMillis) {
		return findBestMove(rootState, timeBudgetMillis, NO_HISTORY, 0);
	}

	/** Come {@link #findBestMove(FastState, long, long[], int)} con la storia in una {@code List<Long>}. */
	public int findBestMove(FastState rootState, long timeBudgetMillis, List<Long> historySinceLastCapture) {
		final int n = historySinceLastCapture.size();
		final long[] h = new long[n];
		for (int i = 0; i < n; i++) {
			h[i] = historySinceLastCapture.get(i);
		}
		return findBestMove(rootState, timeBudgetMillis, h, n);
	}

	private static final long[] NO_HISTORY = new long[0];

	/**
	 * La ricerca vera: trova la migliore mossa per chi deve muovere in {@code rootState} entro {@code timeBudgetMillis}.
	 * {@code rootState} non viene modificata ne' conservata (ogni worker lavora su una propria copia).
	 *
	 * @param history       hash Zobrist delle posizioni dall'ultima cattura, radice inclusa, piu' vecchia per prima
	 *                      (prime {@code historyLen} voci; copiata, mai modificata)
	 * @return la mossa ({@link Move}), oppure {@link Move#NONE} se chi deve muovere non ha mosse legali
	 */
	public int findBestMove(FastState rootState, long timeBudgetMillis, long[] history, int historyLen) {
		final long dl = System.currentTimeMillis() + Math.max(50, timeBudgetMillis - safetyMarginMillis);
		if (tt != null) {
			tt.newSearch(); // nuova mossa di partita: le voci precedenti diventano "vecchie" (conta solo per la TT a due slot)
		}
		final int nThreads = effectiveThreads();
		if (nThreads <= 1) {
			this.control = null;
			this.deadline = dl;
			int r = searchRoot(rootState, history, historyLen, 0);
			this.lastTotalNodeCount = this.lastNodeCount;
			this.lastThreadsUsed = 1;
			this.lastHelperErrors = 0;
			this.lastAdoptedWorker = -1;
			this.lastMainDepth = this.lastDepthReached;
			return r;
		}
		return findBestMoveParallel(rootState, dl, history, historyLen, nThreads);
	}

	/** Solo per i test: la risposta registrata come countermove e' una mossa che il lato al tratto puo' fare davvero in {@code st} (pedina sua in partenza, arrivo vuoto, stessa riga/colonna). */
	private boolean counterReplyIsPlausible(int move) {
		final int from = Move.getFrom(move), to = Move.getTo(move);
		final byte p = st.board[from];
		final boolean mine = st.turn == it.unibo.ai.didattica.competition.tablut.engine.BoardConstants.WHITE_TURN
				? (p == it.unibo.ai.didattica.competition.tablut.engine.BoardConstants.WHITE || p == it.unibo.ai.didattica.competition.tablut.engine.BoardConstants.KING)
				: p == it.unibo.ai.didattica.competition.tablut.engine.BoardConstants.BLACK;
		return mine && st.board[to] == it.unibo.ai.didattica.competition.tablut.engine.BoardConstants.EMPTY && from != to
				&& (from / 9 == to / 9 || from % 9 == to % 9);
	}

	/**
	 * Prepara le killer per una nuova ricerca: azzerate (default) oppure, con {@link #ORDER_KILLER_CARRY}, slittate di 2 ply. La
	 * nuova radice e' 2 ply piu' avanti nella partita (la nostra mossa e la risposta): le killer del ply p+2 diventano quelle del
	 * ply p e gli ultimi due ply restano vuoti. Package-private per i test.
	 */
	void rollKillersForNewSearch() {
		if ((orderingMode & ORDER_KILLER_CARRY) != 0) {
			for (int p = 0; p < killers.length; p++) {
				final int q = p + 2;
				killers[p][0] = q < killers.length ? killers[q][0] : -1;
				killers[p][1] = q < killers.length ? killers[q][1] : -1;
			}
		} else {
			for (int[] k : killers) {
				k[0] = -1;
				k[1] = -1;
			}
		}
	}

	/**
	 * Iterative deepening alla radice di UN worker (il principale, workerId 0, o un helper). Usa {@link #deadline}
	 * e {@link #control} gia' impostati dal chiamante e lascia i risultati nei campi {@code last*}.
	 */
	int searchRoot(FastState rootState, long[] history, int historyLen, int workerId) {
		this.timeUp = false;
		this.nodeCounter = 0;
		this.repetitionEventCounter = 0;
		this.pvsScouts = 0;
		this.pvsReSearches = 0;
		this.rootIterDepth = 0;
		rollKillersForNewSearch();
		for (int[] h : historyScore) {
			for (int i = 0; i < h.length; i++) {
				h[i] >>= 1; // invecchiamento
			}
		}
		prevMoveAt[0] = -1; // alla radice la mossa precedente (dell'avversario) non e' nota alla ricerca

		final FastState s = st;
		s.copyFrom(rootState); // posizione di lavoro privata di questo worker: la ricerca la muta e la ripristina
		final int[] rootMoves = moveStack[0];
		final int nRoot = s.generateMoves(rootMoves);
		rootMoveCountForTest = nRoot;
		if (nRoot == 0) {
			return Move.NONE;
		}

		final int mySign = (s.turn == BoardConstants.WHITE_TURN) ? 1 : -1;
		// Storia: copia nello stack interno, che la ricerca estende/accorcia ma che non altera mai l'array del chiamante.
		ensureHist(historyLen + maxDepth + 8);
		System.arraycopy(history, 0, histKeys, 0, historyLen);
		histLen = historyLen;
		histBase = 0;

		final boolean rootByScores = (orderingMode & ORDER_ROOT_SCORES) != 0;
		int best = rootMoves[0];
		double bestScore = 0;
		int depthReached = 0;

		for (int depth = 1; depth <= maxDepth && !timeUp; depth++) {
			if (workerId > 0 && skipDepth(workerId, depth)) {
				continue; // diversificazione Lazy SMP: gli helper saltano alcune profondita'
			}
			this.rootIterDepth = depth;
			if (!rootByScores) {
				orderMovesInPlace(rootMoves, nRoot, best);
			}

			int iterationBest = Move.NONE;
			double iterationBestScore = Double.NEGATIVE_INFINITY;
			double alpha = -WIN_SCORE - maxDepth - 1;
			double beta = WIN_SCORE + maxDepth + 1;

			boolean firstRootMove = true;
			for (int mi = 0; mi < nRoot; mi++) {
				if (checkTimeUp()) {
					break;
				}
				final int a = rootMoves[mi];
				double score;
				if (usePVS && !firstRootMove && alpha + PVS_EPSILON < beta) {
					// PVS: scout a finestra nulla; ri-ricerca a finestra piena solo se la mossa sembra migliorare
					pvsScouts++;
					score = -exploreChild(a, depth - 1, -(alpha + PVS_EPSILON), -alpha, -mySign, 0);
					if (!timeUp && score > alpha && score < beta) {
						pvsReSearches++;
						score = -exploreChild(a, depth - 1, -beta, -alpha, -mySign, 0);
					}
				} else {
					score = -exploreChild(a, depth - 1, -beta, -alpha, -mySign, 0);
				}
				firstRootMove = false;
				rootScores[mi] = score;
				if (score > iterationBestScore) {
					iterationBestScore = score;
					iterationBest = a;
				}
				if (score > alpha) {
					alpha = score;
				}
			}

			if (!timeUp && iterationBest != Move.NONE) {
				best = iterationBest;
				bestScore = iterationBestScore;
				depthReached = depth;
				if (rootByScores) {
					sortRootByScores(rootMoves, rootScores, nRoot, best);
				}
				// se abbiamo gia' trovato una vittoria forzata, non serve cercare piu' a fondo
				if (bestScore >= WIN_SCORE) {
					break;
				}
			}
		}

		this.lastDepthReached = depthReached;
		this.lastNodeCount = nodeCounter;
		this.lastScore = bestScore;
		this.lastPvsScouts = pvsScouts;
		this.lastPvsReSearches = pvsReSearches;
		this.lastBestMove = best;
		return best;
	}

	// ------------------------------------------------------------------------------------------ ricorsione

	/**
	 * Gestisce la transizione verso UN figlio (root o interno): esegue la mossa sulla posizione di lavoro, controlla se e'
	 * una ripetizione (nel qual caso e' un pareggio immediato, senza nemmeno ricorrere), altrimenti aggiorna la storia del
	 * percorso (azzerandola se la mossa ha catturato, dato che nessuna posizione precedente puo' piu' ripresentarsi), ricorre,
	 * ripristina la storia per i fratelli successivi e ANNULLA la mossa. Centralizzato qui cosi' la radice
	 * ({@link #searchRoot}) e i nodi interni ({@link #negamax}) condividono ESATTAMENTE la stessa logica.
	 * <p>
	 * Pre-condizione: la posizione di lavoro e' quella del GENITORE; post-condizione: lo e' ancora (anche con tempo scaduto).
	 *
	 * @param move        mossa legale nella posizione di lavoro
	 * @param sp          indice di stack del nodo genitore (0 = radice): ne usa l'{@link UndoInfo}
	 * @param perspective come in {@link #negamax}: la prospettiva del FIGLIO
	 */
	// package-private (non private) apposta: permette a un test "white-box" (RepetitionAndStalemateCheck, stesso
	// package) di verificare con precisione chirurgica la gestione di ripetizione e "nessuna mossa legale", invece di
	// dover dedurle indirettamente da quale mossa sceglie una ricerca completa.
	double exploreChild(int move, int depth, double alpha, double beta, int perspective, int sp) {
		final FastState s = st;
		final UndoInfo u = undoStack[sp];
		prevMoveAt[sp + 1] = Move.index(move);
		s.makeMove(move, u);
		final long childHash = s.zobristHash;
		double score;
		if (u.wasCapture()) {
			// una cattura azzera la memoria: il segmento di storia riparte da questa sola posizione
			final int savedLen = histLen, savedBase = histBase;
			histBase = histLen;
			histKeys[histLen++] = childHash;
			score = negamax(depth, alpha, beta, perspective, sp + 1);
			histLen = savedLen;
			histBase = savedBase;
		} else if (historyContains(childHash)) {
			repetitionEventCounter++;
			score = 0; // seconda occorrenza della stessa posizione dall'ultima cattura: pareggio
		} else {
			histKeys[histLen++] = childHash;
			score = negamax(depth, alpha, beta, perspective, sp + 1);
			histLen--; // backtrack: non deve restare visibile ai fratelli successivi
		}
		s.unmakeMove(move, u);
		return score;
	}

	private boolean historyContains(long key) {
		final long[] h = histKeys;
		for (int i = histBase; i < histLen; i++) {
			if (h[i] == key) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Valuta la posizione di lavoro corrente.
	 *
	 * @param perspective +1 se il punteggio va restituito dal punto di vista del
	 *                    Bianco, -1 dal punto di vista del Nero (classica
	 *                    convenzione negamax "a colori").
	 * @param sp          indice di stack di QUESTO nodo (radice = 0, suoi figli = 1, ...): selezione dei buffer di
	 *                    mosse e di ordinamento. L'hash del nodo e' {@code st.zobristHash}, aggiornato incrementalmente.
	 *                    La storia ({@link #histKeys}) contiene il percorso dalla radice fino a questo nodo (compreso).
	 */
	// package-private per lo stesso motivo di exploreChild, si veda sopra.
	double negamax(int depth, double alpha, double beta, int perspective, int sp) {
		nodeCounter++;
		if ((nodeCounter & (TIME_CHECK_INTERVAL - 1)) == 0 && checkTimeUp()) {
			return 0; // valore ignorato dal chiamante: appena timeUp==true, la iterazione corrente viene scartata
		}

		final FastState s = st;
		final byte winner = s.winner;
		if (winner != FastState.WINNER_NONE) {
			if (winner == FastState.WINNER_WHITE) {
				return perspective * (WIN_SCORE + depth); // vince prima => punteggio piu' alto
			}
			if (winner == FastState.WINNER_BLACK) {
				return perspective * (-(WIN_SCORE + depth));
			}
			return 0; // pareggio
		}

		final long hash = s.zobristHash;
		double alphaOriginal = alpha;
		int ttMovePacked = -1;
		// "Firma" del contatore di ripetizioni a questo punto: se cambia mentre
		// esploriamo il sottoalbero di questo nodo, il suo valore potrebbe
		// dipendere dalla storia del percorso e NON verra' messo in TT (si veda
		// il javadoc di classe, sezione Graph History Interaction).
		final long repCounterAtEntry = repetitionEventCounter;

		if (useTT) {
			final TranspositionTable.ProbeResult pr = probe;
			tt.probeInto(s, depth, pr);
			if (pr.hit) {
				ttMovePacked = pr.packedMove; // utile per l'ordinamento anche se la profondita' non basta
				if (pr.depthSufficient) {
					if (pr.flag == TranspositionTable.EXACT) {
						return pr.score;
					} else if (pr.flag == TranspositionTable.LOWER_BOUND) {
						if (pr.score > alpha) {
							alpha = pr.score;
						}
					} else { // UPPER_BOUND
						if (pr.score < beta) {
							beta = pr.score;
						}
					}
					if (alpha >= beta) {
						return pr.score;
					}
				}
			}
		}

		if (depth == 0) {
			return perspective * evaluator.evaluate(s);
		}

		final int[] moves = moveStack[sp];
		final int nMoves = s.generateMoves(moves);
		if (nMoves == 0) {
			// Chi deve muovere non ha mosse legali: perde immediatamente. Questo
			// valore NON dipende in alcun modo dalla storia del percorso (e'
			// una proprieta' della sola board+turno), quindi e' sempre sicuro
			// memorizzarlo in TT, a differenza del caso generale sotto.
			double absolute = (s.turn == BoardConstants.WHITE_TURN) ? -(WIN_SCORE + depth) : (WIN_SCORE + depth);
			double result = perspective * absolute;
			if (useTT) {
				tt.store(s, depth, result, TranspositionTable.EXACT, -1);
			}
			return result;
		}

		final int[] orderScore;
		final int sideIdx = (s.turn == BoardConstants.WHITE_TURN) ? 0 : 1;
		final int ply = Math.max(0, Math.min(killers.length - 1, rootIterDepth - depth));
		if (orderingMode != ORDER_NONE) {
			orderScore = scoreStack[sp];
			if ((orderingMode & ORDER_EXT_MASK) != 0) {
				computeOrderScoresExt(moves, nMoves, sideIdx, ply, ttMovePacked, prevMoveAt[sp], orderScore);
			} else {
				computeOrderScores(moves, nMoves, sideIdx, ply, ttMovePacked, orderScore);
			}
		} else {
			orderScore = null;
			if (ttMovePacked >= 0) {
				orderMovesInPlace(moves, nMoves, ttMovePacked);
			}
		}

		double best = Double.NEGATIVE_INFINITY;
		int bestMoveFound = Move.NONE;
		boolean firstMove = true;
		for (int mi = 0; mi < nMoves; mi++) {
			if (orderScore != null) {
				selectBest(moves, orderScore, mi, nMoves); // selezione pigra: la migliore tra le rimaste va in posizione mi
			}
			final int a = moves[mi];
			double score;
			if (usePVS && !firstMove && alpha + PVS_EPSILON < beta) {
				// scout a finestra nulla (-alpha-eps,-alpha); se la finestra residua e' piu' stretta di eps si cerca normalmente
				pvsScouts++;
				score = -exploreChild(a, depth - 1, -(alpha + PVS_EPSILON), -alpha, -perspective, sp);
				if (!timeUp && score > alpha && score < beta) {
					// la mossa migliora il meglio trovato finora: ora serve il valore ESATTO
					pvsReSearches++;
					score = -exploreChild(a, depth - 1, -beta, -alpha, -perspective, sp);
				}
			} else {
				score = -exploreChild(a, depth - 1, -beta, -alpha, -perspective, sp);
			}
			firstMove = false;
			if (score > best) {
				best = score;
				bestMoveFound = a;
			}
			if (best > alpha) {
				alpha = best;
			}
			if (alpha >= beta) {
				if (orderScore != null && !timeUp) {
					recordCutoff(a, sideIdx, ply, depth); // killer + history: solo se il valore e' affidabile
					if (counterMove != null && prevMoveAt[sp] >= 0) {
						counterMove[sideIdx][prevMoveAt[sp]] = Move.index(a);
						if (debugVerifyCounter && (counterMove[sideIdx][prevMoveAt[sp]] != Move.index(a) || !counterReplyIsPlausible(a))) {
							debugCounterViolations++;
						}
					}
				}
				break; // taglio alpha-beta
			}
			if (timeUp) {
				break;
			}
		}

		if (useTT && !timeUp && repetitionEventCounter == repCounterAtEntry) {
			// Se il tempo e' scaduto A META' di questo nodo, "best" potrebbe non
			// riflettere una vera minimax (mosse rimanenti mai esplorate): non lo
			// memorizziamo. Idem se una ripetizione e' stata incontrata da qualche
			// parte sotto questo nodo: il suo valore potrebbe non essere lo stesso
			// se lo si raggiungesse con una storia diversa (Graph History Interaction).
			assert !timeUp : "TT: un nodo interrotto a meta' non va mai memorizzato"; // invariante verificato dai test (-ea / SearchEquivalenceCheck)
			byte flag;
			if (best <= alphaOriginal) {
				flag = TranspositionTable.UPPER_BOUND;
			} else if (best >= beta) {
				flag = TranspositionTable.LOWER_BOUND;
			} else {
				flag = TranspositionTable.EXACT;
			}
			tt.store(s, depth, best, flag, bestMoveFound == Move.NONE ? -1 : bestMoveFound);
		}

		return best;
	}

	// ------------------------------------------------------------------------------------------ ordinamento

	/** Indice mossa da*81+a (stesso schema per killer e history): si veda {@link Move#index}. */
	static int moveIndex(int move) {
		return Move.index(move);
	}

	/**
	 * Punteggio di ordinamento di ogni mossa, scritto in {@code sc[0..n)}: mossa TT > prima killer > seconda killer >
	 * punteggio history (0 se la history e' disattivata). Package-private per i test.
	 *
	 * @param ttPacked mossa suggerita dalla TT ({@link Move}) oppure -1
	 */
	void computeOrderScores(int[] moves, int n, int sideIdx, int ply, int ttPacked, int[] sc) {
		final int ttIdx = ttPacked >= 0 ? Move.index(ttPacked) : -1;
		int k0 = -1, k1 = -1;
		if ((orderingMode & ORDER_KILLERS) != 0) {
			k0 = killers[ply][0];
			k1 = killers[ply][1];
		}
		final boolean useHistory = (orderingMode & ORDER_HISTORY) != 0;
		final int[] hist = historyScore[sideIdx];
		for (int i = 0; i < n; i++) {
			final int idx = Move.index(moves[i]);
			if (idx == ttIdx) {
				sc[i] = Integer.MAX_VALUE;
			} else if (idx == k0) {
				sc[i] = Integer.MAX_VALUE - 1;
			} else if (idx == k1) {
				sc[i] = Integer.MAX_VALUE - 2;
			} else if (useHistory) {
				sc[i] = hist[idx];
			} else {
				sc[i] = 0;
			}
		}
	}

	/**
	 * Punteggio di ordinamento ESTESO (si usa quando e' accesa almeno una delle opzioni di {@link #ORDER_EXT_MASK}); valuta le
	 * mosse sulla posizione di lavoro corrente. Livelli, dal piu' alto: mossa TT; vittoria immediata ({@link #ORDER_WINS}); poi,
	 * per default, killer, countermove, catture (per numero di catture, a parita' per history), quiete per history; con
	 * {@link #ORDER_CAPTURES_FIRST} le catture passano prima delle killer. Senza {@code ORDER_CAPTURES*} le vittorie immediate
	 * non accese da {@code ORDER_WINS} sono mosse qualunque. Package-private per i test.
	 *
	 * @param prevIdx indice ({@link Move#index}) della mossa che ha portato a questo nodo, oppure -1 (serve al countermove)
	 */
	void computeOrderScoresExt(int[] moves, int n, int sideIdx, int ply, int ttPacked, int prevIdx, int[] sc) {
		extScoreCalls++;
		final int ttIdx = ttPacked >= 0 ? Move.index(ttPacked) : -1;
		final boolean capsFirst = (orderingMode & ORDER_CAPTURES_FIRST) != 0;
		final boolean wins = (orderingMode & ORDER_WINS) != 0;
		final boolean caps = (orderingMode & (ORDER_CAPTURES | ORDER_CAPTURES_FIRST)) != 0;
		final boolean tactical = wins || caps;
		int k0 = -1, k1 = -1, ctr = -1;
		if ((orderingMode & ORDER_KILLERS) != 0) {
			k0 = killers[ply][0];
			k1 = killers[ply][1];
		}
		if (counterMove != null && prevIdx >= 0) {
			ctr = counterMove[sideIdx][prevIdx];
		}
		final boolean useHistory = (orderingMode & ORDER_HISTORY) != 0;
		final int[] hist = historyScore[sideIdx];
		final int s0 = capsFirst ? SC_KILLER0_B : SC_KILLER0_A, s1 = capsFirst ? SC_KILLER1_B : SC_KILLER1_A;
		final int sCtr = capsFirst ? SC_COUNTER_B : SC_COUNTER_A;
		final int capBase = capsFirst ? SC_CAPTURE_BASE_B : SC_CAPTURE_BASE_A;
		final FastState s = st;
		for (int i = 0; i < n; i++) {
			final int mv = moves[i];
			final int idx = Move.index(mv);
			if (idx == ttIdx) {
				sc[i] = SC_TT;
				continue;
			}
			final int tv = tactical ? s.tacticalValue(mv) : 0;
			if (wins && tv == FastState.TACT_WIN) {
				sc[i] = SC_WIN;
			} else if (capsFirst && caps && tv > 0) {
				// catture PRIMA delle killer: una cattura che e' anche killer e' una cattura
				final int rank = tv >= FastState.TACT_WIN ? 4 : tv;
				sc[i] = capBase + rank * SC_CAPTURE_STEP + Math.min(useHistory ? hist[idx] : 0, SC_HISTORY_CAP);
			} else if (idx == k0) {
				sc[i] = s0;
			} else if (idx == k1) {
				sc[i] = s1;
			} else if (idx == ctr) {
				sc[i] = sCtr;
			} else {
				final int h = useHistory ? hist[idx] : 0;
				if (caps && tv > 0) {
					final int rank = tv >= FastState.TACT_WIN ? 4 : tv;
					sc[i] = capBase + rank * SC_CAPTURE_STEP + Math.min(h, SC_HISTORY_CAP);
				} else {
					sc[i] = h;
				}
			}
		}
	}

	/**
	 * Riordina le mosse della radice per punteggio decrescente dell'ultima iterazione completa (ordinamento STABILE: a pari
	 * punteggio resta l'ordine precedente) e porta {@code best} in testa. I punteggi delle mosse non migliori sono limiti
	 * superiori (scout PVS a finestra nulla): bastano come euristica, e la migliore ha sempre il punteggio piu' alto.
	 */
	static void sortRootByScores(int[] moves, double[] scores, int n, int best) {
		for (int i = 1; i < n; i++) {
			final int m = moves[i];
			final double v = scores[i];
			int j = i - 1;
			while (j >= 0 && scores[j] < v) {
				moves[j + 1] = moves[j];
				scores[j + 1] = scores[j];
				j--;
			}
			moves[j + 1] = m;
			scores[j + 1] = v;
		}
		if (moves[0] != best) {
			int at = -1;
			for (int i = 0; i < n; i++) {
				if (moves[i] == best) {
					at = i;
					break;
				}
			}
			if (at > 0) {
				final int mb = moves[at];
				final double vb = scores[at];
				System.arraycopy(moves, 0, moves, 1, at);
				System.arraycopy(scores, 0, scores, 1, at);
				moves[0] = mb;
				scores[0] = vb;
			}
		}
	}

	/** Porta in posizione i la mossa di punteggio massimo tra quelle da i in poi (scambiando mosse E punteggi). A parita' vince la prima. */
	static void selectBest(int[] moves, int[] sc, int i, int n) {
		int best = i, bv = sc[i];
		for (int j = i + 1; j < n; j++) {
			if (sc[j] > bv) {
				bv = sc[j];
				best = j;
			}
		}
		if (best != i) {
			final int tmp = moves[i];
			moves[i] = moves[best];
			moves[best] = tmp;
			sc[best] = sc[i];
			sc[i] = bv;
		}
	}

	/** La mossa {@code cutoff} ha provocato un taglio beta: aggiorna killer (due per ply) e history. */
	private void recordCutoff(int cutoff, int sideIdx, int ply, int depth) {
		final int idx = Move.index(cutoff);
		if ((orderingMode & ORDER_KILLERS) != 0 && killers[ply][0] != idx) {
			killers[ply][1] = killers[ply][0];
			killers[ply][0] = idx;
		}
		if ((orderingMode & ORDER_HISTORY) != 0) {
			final int[] hist = historyScore[sideIdx];
			hist[idx] += depth * depth;
			if (hist[idx] > HISTORY_LIMIT) {
				for (int[] h : historyScore) {
					for (int i = 0; i < h.length; i++) {
						h[i] >>= 1;
					}
				}
			}
		}
	}

	/**
	 * Porta in testa alla lista la mossa {@code preferred} (tipicamente la migliore trovata all'iterazione precedente, o
	 * quella suggerita dalla TT), scambiandola con quella in posizione 0. E' l'ottimizzazione col miglior rapporto
	 * costo/beneficio per l'alpha-beta pruning con iterative deepening. Se {@code preferred} non e' nella lista (o e'
	 * {@link Move#NONE}) non fa nulla.
	 */
	private static void orderMovesInPlace(int[] moves, int n, int preferred) {
		if (preferred == Move.NONE) {
			return;
		}
		for (int i = 0; i < n; i++) {
			if (moves[i] == preferred) {
				if (i > 0) {
					final int tmp = moves[0];
					moves[0] = moves[i];
					moves[i] = tmp;
				}
				return;
			}
		}
	}

	private boolean checkTimeUp() {
		if (!timeUp) {
			SearchControl c = control;
			if ((c != null && c.stop) || System.currentTimeMillis() >= deadline
					|| (forceTimeUpAfterNodes > 0 && nodeCounter >= forceTimeUpAfterNodes)) {
				timeUp = true;
			}
		}
		return timeUp;
	}

	/**
	 * Solo per i test: se > 0, la ricerca si considera "a tempo scaduto" appena ha visitato almeno questo numero di nodi
	 * (controllato ogni {@value #TIME_CHECK_INTERVAL} nodi, come l'orologio). Permette di interrompere una ricerca in un punto
	 * deterministico dell'albero, senza dipendere dal tempo reale. In produzione resta 0 (nessun effetto).
	 */
	int forceTimeUpAfterNodes = 0;

	// ------------------------------------------------------------------------------------------ hook per i test

	/**
	 * Hook di test (white-box): installa {@code position} (copiata) come posizione di lavoro e {@code history} come storia
	 * di ripetizione, e prepara gli stack per una discesa di {@code depth} ply. Dopo di questo si possono chiamare
	 * direttamente {@link #negamax} ed {@link #exploreChild} con {@code sp = 0}.
	 */
	void installForTest(FastState position, List<Long> history, int depth) {
		allocStacks(depth + 4);
		st.copyFrom(position);
		final int n = history.size();
		ensureHist(n + depth + 8);
		for (int i = 0; i < n; i++) {
			histKeys[i] = history.get(i);
		}
		histLen = n;
		histBase = 0;
		timeUp = false;
		deadline = Long.MAX_VALUE;
	}

	/** Hook di test: la posizione di lavoro corrente (deve essere uguale a quella installata dopo ogni exploreChild/negamax). */
	FastState workingState() {
		return st;
	}

	/** Hook di test: lunghezza corrente della storia di ripetizione (deve tornare al valore iniziale dopo ogni chiamata). */
	int historyLengthForTest() {
		return histLen;
	}

	// ------------------------------------------------------------------------------------------ Lazy SMP

	/** Obiettivo di thread di ricerca (la macchina di gara ha 4 core). */
	public static final int DEFAULT_THREADS = 4;
	static final int MAX_THREADS = 64;
	/** Quanto attendere gli helper dopo che il principale ha finito (di norma si fermano in pochi ms). */
	private static final long HELPER_JOIN_GRACE_MILLIS = 150;

	/** Flag di stop di UNA ricerca: nuovo per ogni findBestMove, cosi' un helper in ritardo non puo' mai "risvegliarsi" in una ricerca successiva. */
	static final class SearchControl {
		volatile boolean stop;
	}

	/**
	 * Thread di ricerca di default: {@code min(DEFAULT_THREADS, core disponibili)}, quindi 4 su una macchina a 4 core e
	 * meno su una piu' piccola (piu' thread dei core rallenterebbe anche il principale). Si puo' forzare con la proprieta'
	 * di sistema {@code -Dtablut.threads=N}.
	 */
	public static int defaultThreads() {
		String prop = System.getProperty("tablut.threads");
		if (prop != null) {
			try {
				int v = Integer.parseInt(prop.trim());
				if (v >= 1) {
					return Math.min(v, MAX_THREADS);
				}
			} catch (NumberFormatException e) {
				// proprieta' non valida: si ignora e si usa il default
			}
		}
		return Math.max(1, Math.min(DEFAULT_THREADS, Runtime.getRuntime().availableProcessors()));
	}

	public int getThreads() {
		return threads;
	}

	/** Margine di sicurezza (ms) tolto al budget di {@code findBestMove}: tempo per fermare la ricerca, attendere gli helper e spedire la mossa. */
	public long getSafetyMarginMillis() {
		return safetyMarginMillis;
	}

	/** Imposta il margine di sicurezza (ms, almeno 0). Il budget effettivo resta almeno 50 ms. Ritorna {@code this}. */
	public Negamax setSafetyMarginMillis(long ms) {
		this.safetyMarginMillis = Math.max(0, ms);
		return this;
	}

	/** Margine di default (300 ms). */
	public static long defaultSafetyMarginMillis() {
		return SAFETY_MARGIN_MILLIS;
	}

	/** La transposition table (null se disattivata): per misurarne il riempimento. */
	public TranspositionTable transpositionTable() {
		return tt;
	}

	/** Imposta il numero di thread di ricerca (1 = thread singolo). Va chiamato a ricerca ferma. Ritorna {@code this} per concatenare. */
	public Negamax setThreads(int n) {
		int v = Math.max(1, Math.min(MAX_THREADS, n));
		if (v != threads) {
			shutdown();
			threads = v;
		}
		return this;
	}

	/** Thread effettivi: la condivisione avviene solo tramite la TT, quindi senza TT (o per un helper) si cerca a thread singolo. */
	private int effectiveThreads() {
		return (useTT && !isHelper) ? threads : 1;
	}

	/** Rilascia i thread degli helper (opzionale: sono "daemon" e vengono ricreati alla prossima ricerca parallela). */
	public void shutdown() {
		if (pool != null) {
			pool.shutdownNow();
		}
		pool = null;
		helpers = null;
		helperFutures = null;
	}

	/** Vero se nessun helper e' in esecuzione (package-private: per i test). */
	boolean helpersIdle() {
		if (helperFutures == null) {
			return true;
		}
		for (Future<?> f : helperFutures) {
			if (f != null && !f.isDone()) {
				return false;
			}
		}
		return true;
	}

	private static final int[] SKIP_SIZE = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };
	private static final int[] SKIP_PHASE = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7 };

	/**
	 * Schema di diversificazione degli helper (workerId >= 1) alla profondita' {@code depth}: veri se l'helper deve
	 * SALTARE quella profondita'. Con 3 helper: il 1o cerca solo profondita' pari, il 2o solo dispari, il 3o a coppie
	 * (1, 4-5, 8-9...). Cosi' i thread non lavorano tutti sulla stessa iterazione e le profondita' vicine riempiono la TT.
	 */
	static boolean skipDepth(int workerId, int depth) {
		int k = (workerId - 1) % SKIP_SIZE.length;
		return ((depth + SKIP_PHASE[k]) / SKIP_SIZE[k]) % 2 != 0;
	}

	/**
	 * Quale risultato usare: il principale, tranne se un helper ha completato un'iterazione PIU' PROFONDA e il
	 * principale non ha gia' provato una vittoria/sconfitta forzata (quella e' esatta e non va scavalcata).
	 *
	 * @return indice (0-based) dell'helper da adottare, oppure -1 per tenere il risultato del principale
	 */
	static int chooseWorker(int mainDepth, double mainScore, int[] helperDepth, boolean[] helperUsable) {
		if (Math.abs(mainScore) >= WIN_SCORE) {
			return -1;
		}
		int best = -1, bestDepth = mainDepth;
		for (int i = 0; i < helperDepth.length; i++) {
			if (helperUsable[i] && helperDepth[i] > bestDepth) {
				bestDepth = helperDepth[i];
				best = i;
			}
		}
		return best;
	}

	private int findBestMoveParallel(FastState rootState, long dl, long[] history, int historyLen, int nThreads) {
		if (!rootState.hasLegalMove()) {
			this.lastThreadsUsed = 1;
			return Move.NONE;
		}
		ensureHelpers(nThreads);
		final SearchControl ctl = new SearchControl();
		this.control = ctl;
		this.deadline = dl;
		final int nHelpers = nThreads - 1;
		final boolean[] started = new boolean[nHelpers];
		int startedCount = 0;
		for (int i = 0; i < nHelpers; i++) {
			Future<?> previous = helperFutures[i];
			if (previous != null && !previous.isDone()) {
				continue; // ancora impegnato dalla ricerca precedente: mai due ricerche contemporanee sulla stessa istanza
			}
			final Negamax h = helpers[i];
			h.control = ctl;
			h.deadline = dl;
			h.helperError = null;
			final FastState rootCopy = rootState.duplicate();
			final long[] historyCopy = java.util.Arrays.copyOf(history, historyLen);
			final int workerId = i + 1;
			helperFutures[i] = pool.submit(new Runnable() {
				@Override
				public void run() {
					try {
						h.searchRoot(rootCopy, historyCopy, historyCopy.length, workerId);
					} catch (Throwable t) {
						h.helperError = t; // un helper in errore non deve mai far fallire la mossa
					}
				}
			});
			started[i] = true;
			startedCount++;
		}

		int best;
		try {
			best = searchRoot(rootState, history, historyLen, 0);
		} finally {
			ctl.stop = true; // il principale ha finito: gli helper si fermano al prossimo controllo (~2048 nodi)
		}

		// attesa breve degli helper; quelli che non finiscono in tempo vengono ignorati (e non riutilizzati finche' non finiscono)
		long grace = System.currentTimeMillis() + HELPER_JOIN_GRACE_MILLIS;
		long totalNodes = this.lastNodeCount;
		int errors = 0;
		int[] hDepth = new int[nHelpers];
		boolean[] usable = new boolean[nHelpers];
		for (int i = 0; i < nHelpers; i++) {
			if (!started[i]) {
				continue;
			}
			try {
				helperFutures[i].get(Math.max(1, grace - System.currentTimeMillis()), TimeUnit.MILLISECONDS);
				Negamax h = helpers[i];
				totalNodes += h.lastNodeCount;
				if (h.helperError != null) {
					errors++;
				} else if (h.lastBestMove != Move.NONE) {
					hDepth[i] = h.lastDepthReached;
					usable[i] = true;
				}
			} catch (TimeoutException e) {
				// ancora in esecuzione: si ignora il suo risultato
			} catch (ExecutionException e) {
				errors++;
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				break;
			}
		}

		int mainDepth = this.lastDepthReached;
		this.lastMainDepth = mainDepth;
		int adopt = chooseWorker(mainDepth, this.lastScore, hDepth, usable);
		this.lastAdoptedWorker = -1;
		if (adopt >= 0) {
			Negamax h = helpers[adopt];
			best = h.lastBestMove;
			this.lastDepthReached = h.lastDepthReached;
			this.lastScore = h.lastScore;
			this.lastAdoptedWorker = adopt + 1;
		}
		this.lastBestMove = best;
		this.lastTotalNodeCount = totalNodes;
		this.lastThreadsUsed = 1 + startedCount;
		this.lastHelperErrors = errors;
		return best;
	}

	private void ensureHelpers(int nThreads) {
		if (pool != null && helpers != null && helpers.length == nThreads - 1) {
			return;
		}
		shutdown();
		final AtomicInteger counter = new AtomicInteger();
		pool = Executors.newFixedThreadPool(nThreads - 1, new java.util.concurrent.ThreadFactory() {
			@Override
			public Thread newThread(Runnable r) {
				Thread t = new Thread(r, "negamax-helper-" + counter.incrementAndGet());
				t.setDaemon(true);
				return t;
			}
		});
		helpers = new Negamax[nThreads - 1];
		helperFutures = new Future<?>[nThreads - 1];
		for (int i = 0; i < helpers.length; i++) {
			helpers[i] = new Negamax(this);
		}
	}

}
