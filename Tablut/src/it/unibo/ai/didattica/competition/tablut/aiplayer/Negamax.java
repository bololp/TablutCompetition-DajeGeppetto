package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
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
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

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
 * Questa classe dipende solo da {@link AshtonRules} (le regole) e da
 * {@link Evaluator} (la valutazione di una posizione): per passare in futuro
 * a una rete neurale bastera' passare al costruttore un {@code Evaluator}
 * diverso, senza toccare questa classe.
 * <p>
 * <b>Transposition table</b> ({@link TranspositionTable}, indicizzata con
 * {@link ZobristHasher}): posizioni raggiunte da mosse diverse (frequenti in
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
	// package-private (non private): permette a RepetitionAndStalemateCheck
	// (stesso package, in testsrc) di verificare i punteggi esatti restituiti
	// per vittoria/sconfitta, invece di dover indovinare una soglia arbitraria.
	static final double WIN_SCORE = 1_000_000;

	/** Margine di sicurezza tolto al tempo a disposizione, per lasciare tempo
	 *  di risalire la ricorsione e inviare la mossa al server prima dello scadere
	 *  del timeout imposto dalla competizione. */
	private static final long SAFETY_MARGIN_MILLIS = 300;

	/** Modalita' di ordinamento delle mosse (maschera di bit): nessuno (solo mossa TT), killer, history. */
	public static final int ORDER_NONE = 0;
	public static final int ORDER_KILLERS = 1;
	public static final int ORDER_HISTORY = 2;
	public static final int ORDER_DEFAULT = ORDER_KILLERS | ORDER_HISTORY;

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
	private static final int DEFAULT_TT_SIZE = 1 << 22;

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
	/** Migliore mossa dell'ultima ricerca alla radice di QUESTO worker (usata per adottare il risultato di un helper). */
	Action lastBestMove;
	/** killers[ply][0..1]: indice mossa (da*81+a) oppure -1. */
	final int[][] killers;
	/** historyScore[lato][da*81+a]; lato 0 = Bianco, 1 = Nero. */
	final int[][] historyScore = new int[2][81 * 81];
	/** Profondita' dell'iterazione corrente alla radice: ply = rootIterDepth - depth. */
	private int rootIterDepth;
	// package-private (non private): permette al test white-box di sondare
	// direttamente la tabella dopo una ricerca, per confermare che un'entry sia
	// stata davvero memorizzata (e non solo che il punteggio finale sia giusto).
	final TranspositionTable tt;

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
	 *              {@code true}, solo più lentamente - se così non fosse,
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

	/**
	 * @param threads numero di thread di ricerca (Lazy SMP), 1 = ricerca a thread singolo identica a quella storica.
	 *                Il default degli altri costruttori e' {@link #defaultThreads()}.
	 */
	public Negamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS, int orderingMode,
			int threads) {
		this.evaluator = evaluator;
		this.maxDepth = maxDepth;
		this.useTT = useTT;
		this.usePVS = usePVS;
		this.orderingMode = orderingMode;
		this.killers = newKillers(maxDepth);
		this.tt = useTT ? new TranspositionTable(ttSize) : null;
		this.threads = Math.max(1, Math.min(MAX_THREADS, threads));
		this.isHelper = false;
	}

	/** Costruttore degli helper Lazy SMP: stessa configurazione e STESSA transposition table del principale, stato di ricerca proprio. */
	private Negamax(Negamax main) {
		this.evaluator = main.evaluator;
		this.maxDepth = main.maxDepth;
		this.useTT = main.useTT;
		this.usePVS = main.usePVS;
		this.orderingMode = main.orderingMode;
		this.killers = newKillers(main.maxDepth);
		this.tt = main.tt;
		this.threads = 1;
		this.isHelper = true;
	}

	private static int[][] newKillers(int maxDepth) {
		int[][] k = new int[Math.max(2, maxDepth + 2)][2];
		for (int[] e : k) {
			e[0] = -1;
			e[1] = -1;
		}
		return k;
	}

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
		return findBestMove(rootState, timeBudgetMillis, Collections.emptyList());
	}

	/**
	 * Trova la migliore mossa per il giocatore di turno in {@code rootState},
	 * usando al massimo {@code timeBudgetMillis} millisecondi.
	 *
	 * @param historySinceLastCapture hash Zobrist ({@link ZobristHasher}) delle
	 *        posizioni REALMENTE occorse in partita dall'ultima cattura (radice
	 *        inclusa), piu' vecchia per prima. Il chiamante (tipicamente
	 *        {@code TablutNegamaxClient}) la mantiene aggiornata; qui viene
	 *        copiata, mai modificata.
	 * @return la miglior mossa trovata, oppure {@code null} se in {@code rootState}
	 *         il giocatore di turno non ha mosse legali (situazione che, di norma,
	 *         il chiamante non dovrebbe nemmeno raggiungere: se non si hanno mosse
	 *         legali si e' gia' perso).
	 */
	public Action findBestMove(State rootState, long timeBudgetMillis, List<Long> historySinceLastCapture) {
		final long dl = System.currentTimeMillis() + Math.max(50, timeBudgetMillis - SAFETY_MARGIN_MILLIS);
		final int nThreads = effectiveThreads();
		if (nThreads <= 1) {
			this.control = null;
			this.deadline = dl;
			Action r = searchRoot(rootState, historySinceLastCapture, 0);
			this.lastTotalNodeCount = this.lastNodeCount;
			this.lastThreadsUsed = 1;
			this.lastHelperErrors = 0;
			this.lastAdoptedWorker = -1;
			this.lastMainDepth = this.lastDepthReached;
			return r;
		}
		return findBestMoveParallel(rootState, dl, historySinceLastCapture, nThreads);
	}

	/**
	 * Iterative deepening alla radice di UN worker (il principale, workerId 0, o un helper). Usa {@link #deadline}
	 * e {@link #control} gia' impostati dal chiamante e lascia i risultati nei campi {@code last*}.
	 */
	Action searchRoot(State rootState, List<Long> historySinceLastCapture, int workerId) {
		this.timeUp = false;
		this.nodeCounter = 0;
		this.repetitionEventCounter = 0;
		this.pvsScouts = 0;
		this.pvsReSearches = 0;
		this.rootIterDepth = 0;
		for (int[] k : killers) {
			k[0] = -1;
			k[1] = -1;
		}
		for (int[] h : historyScore) {
			for (int i = 0; i < h.length; i++) {
				h[i] >>= 1; // invecchiamento
			}
		}

		List<Action> rootMoves = AshtonRules.generateMoves(rootState);
		if (rootMoves.isEmpty()) {
			return null;
		}

		int mySign = (rootState.getTurn() == Turn.WHITE) ? 1 : -1;
		long rootHash = ZobristHasher.fullHash(rootState); // unico calcolo completo: O(81), una volta per mossa
		// Copia difensiva e mutabile: la ricerca vi aggiunge/rimuove le proprie
		// posizioni ipotetiche (avanti prima di ricorrere, indietro dopo il
		// ritorno), ma non deve mai alterare la lista del chiamante.
		List<Long> history = new ArrayList<>(historySinceLastCapture);

		Action best = rootMoves.get(0);
		double bestScore = 0;
		int depthReached = 0;

		for (int depth = 1; depth <= maxDepth && !timeUp; depth++) {
			if (workerId > 0 && skipDepth(workerId, depth)) {
				continue; // diversificazione Lazy SMP: gli helper saltano alcune profondita'
			}
			this.rootIterDepth = depth;
			orderMovesInPlace(rootMoves, best);

			Action iterationBest = null;
			double iterationBestScore = Double.NEGATIVE_INFINITY;
			double alpha = -WIN_SCORE - maxDepth - 1;
			double beta = WIN_SCORE + maxDepth + 1;

			boolean firstRootMove = true;
			for (Action a : rootMoves) {
				if (checkTimeUp()) {
					break;
				}
				State child = AshtonRules.applyMove(rootState, a);
				long childHash = ZobristHasher.updateHash(rootHash, rootState, child, a);
				double score;
				if (usePVS && !firstRootMove && alpha + PVS_EPSILON < beta) {
					// PVS: scout a finestra nulla; ri-ricerca a finestra piena solo se la mossa sembra migliorare
					pvsScouts++;
					score = -exploreChild(rootState, child, a, depth - 1, -(alpha + PVS_EPSILON), -alpha, -mySign,
							childHash, history);
					if (!timeUp && score > alpha && score < beta) {
						pvsReSearches++;
						score = -exploreChild(rootState, child, a, depth - 1, -beta, -alpha, -mySign, childHash,
								history);
					}
				} else {
					score = -exploreChild(rootState, child, a, depth - 1, -beta, -alpha, -mySign, childHash, history);
				}
				firstRootMove = false;
				if (score > iterationBestScore) {
					iterationBestScore = score;
					iterationBest = a;
				}
				if (score > alpha) {
					alpha = score;
				}
			}

			if (!timeUp && iterationBest != null) {
				best = iterationBest;
				bestScore = iterationBestScore;
				depthReached = depth;
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

	/**
	 * Gestisce la transizione verso UN figlio (root o interno): controlla se e'
	 * una ripetizione (nel qual caso e' un pareggio immediato, senza nemmeno
	 * ricorrere), altrimenti aggiorna la storia del percorso (azzerandola se la
	 * mossa ha catturato, dato che nessuna posizione precedente puo' piu'
	 * ripresentarsi), ricorre, e ripristina la storia per i fratelli successivi.
	 * Centralizzato qui cosi' la radice ({@link #findBestMove}) e i nodi interni
	 * ({@link #negamax}) condividono ESATTAMENTE la stessa logica.
	 */
	// package-private (non private) apposta: permette un test "white-box" in
	// RepetitionAndStalemateCheck (stesso package, in testsrc) di verificare
	// con precisione chirurgica la gestione di ripetizione e "nessuna mossa
	// legale", invece di dover dedurle indirettamente da quale mossa sceglie
	// una ricerca completa.
	double exploreChild(State parent, State child, Action a, int depth, double alpha, double beta,
			int perspective, long childHash, List<Long> history) {
		if (AshtonRules.wasCapture(parent, child, a)) {
			List<Long> freshHistory = new ArrayList<>();
			freshHistory.add(childHash);
			return negamax(child, depth, alpha, beta, perspective, childHash, freshHistory);
		}
		if (history.contains(childHash)) {
			repetitionEventCounter++;
			return 0; // seconda occorrenza della stessa posizione dall'ultima cattura: pareggio
		}
		history.add(childHash);
		double score = negamax(child, depth, alpha, beta, perspective, childHash, history);
		history.remove(history.size() - 1); // backtrack: non deve restare visibile ai fratelli successivi
		return score;
	}

	/**
	 * @param perspective +1 se il punteggio va restituito dal punto di vista del
	 *                    Bianco, -1 dal punto di vista del Nero (classica
	 *                    convenzione negamax "a colori").
	 * @param hash        hash Zobrist di {@code state}, calcolato incrementalmente
	 *                    dal chiamante (mai da zero qui dentro).
	 * @param history     posizioni occorse (reali + ipotetiche in questo
	 *                    percorso di ricerca) dall'ultima cattura, radice
	 *                    inclusa. Mutata e ripristinata da {@link #exploreChild}
	 *                    per ogni figlio: quando siamo QUI dentro, contiene
	 *                    esattamente il percorso dalla radice fino a questo nodo.
	 */
	// package-private per lo stesso motivo di exploreChild, si veda sopra.
	double negamax(State state, int depth, double alpha, double beta, int perspective, long hash,
			List<Long> history) {
		nodeCounter++;
		if ((nodeCounter & (TIME_CHECK_INTERVAL - 1)) == 0 && checkTimeUp()) {
			return 0; // valore ignorato dal chiamante: appena timeUp==true, la iterazione corrente viene scartata
		}

		Turn t = state.getTurn();
		if (t == Turn.WHITEWIN) {
			return perspective * (WIN_SCORE + depth); // vince prima => punteggio piu' alto
		}
		if (t == Turn.BLACKWIN) {
			return perspective * (-(WIN_SCORE + depth));
		}
		if (t == Turn.DRAW) {
			return 0;
		}

		double alphaOriginal = alpha;
		int ttMovePacked = -1;
		// "Firma" del contatore di ripetizioni a questo punto: se cambia mentre
		// esploriamo il sottoalbero di questo nodo, il suo valore potrebbe
		// dipendere dalla storia del percorso e NON verra' messo in TT (si veda
		// il javadoc di classe, sezione Graph History Interaction).
		long repCounterAtEntry = repetitionEventCounter;

		if (useTT) {
			TranspositionTable.ProbeResult probe = tt.probe(hash, depth);
			if (probe.hit) {
				ttMovePacked = probe.packedMove; // utile per l'ordinamento anche se la profondita' non basta
				if (probe.depthSufficient) {
					if (probe.flag == TranspositionTable.EXACT) {
						return probe.score;
					} else if (probe.flag == TranspositionTable.LOWER_BOUND) {
						if (probe.score > alpha) {
							alpha = probe.score;
						}
					} else { // UPPER_BOUND
						if (probe.score < beta) {
							beta = probe.score;
						}
					}
					if (alpha >= beta) {
						return probe.score;
					}
				}
			}
		}

		if (depth == 0) {
			return perspective * evaluator.evaluate(state);
		}

		List<Action> moves = AshtonRules.generateMoves(state);
		if (moves.isEmpty()) {
			// Chi deve muovere non ha mosse legali: perde immediatamente. Questo
			// valore NON dipende in alcun modo dalla storia del percorso (e'
			// una proprieta' della sola board+turno), quindi e' sempre sicuro
			// memorizzarlo in TT, a differenza del caso generale sotto.
			double absolute = (t == Turn.WHITE) ? -(WIN_SCORE + depth) : (WIN_SCORE + depth);
			double result = perspective * absolute;
			if (useTT) {
				tt.store(hash, depth, result, TranspositionTable.EXACT, -1);
			}
			return result;
		}

		final int nMoves = moves.size();
		final int[] orderScore;
		final int sideIdx = (t == Turn.WHITE) ? 0 : 1;
		final int ply = Math.max(0, Math.min(killers.length - 1, rootIterDepth - depth));
		if (orderingMode != ORDER_NONE) {
			orderScore = computeOrderScores(moves, sideIdx, ply, ttMovePacked);
		} else {
			orderScore = null;
			if (ttMovePacked >= 0) {
				orderMovesByPackedHint(moves, ttMovePacked);
			}
		}

		double best = Double.NEGATIVE_INFINITY;
		Action bestMoveFound = null;
		boolean firstMove = true;
		for (int mi = 0; mi < nMoves; mi++) {
			if (orderScore != null) {
				selectBest(moves, orderScore, mi); // selezione pigra: la migliore tra le rimaste va in posizione mi
			}
			Action a = moves.get(mi);
			State child = AshtonRules.applyMove(state, a);
			long childHash = ZobristHasher.updateHash(hash, state, child, a);
			double score;
			if (usePVS && !firstMove && alpha + PVS_EPSILON < beta) {
				// scout a finestra nulla (-alpha-eps,-alpha); se la finestra residua e' piu' stretta di eps si cerca normalmente
				pvsScouts++;
				score = -exploreChild(state, child, a, depth - 1, -(alpha + PVS_EPSILON), -alpha, -perspective,
						childHash, history);
				if (!timeUp && score > alpha && score < beta) {
					// la mossa migliora il meglio trovato finora: ora serve il valore ESATTO
					pvsReSearches++;
					score = -exploreChild(state, child, a, depth - 1, -beta, -alpha, -perspective, childHash,
							history);
				}
			} else {
				score = -exploreChild(state, child, a, depth - 1, -beta, -alpha, -perspective, childHash, history);
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
					recordCutoff(a, sideIdx, ply, depth, moves, mi); // killer + history: solo se il valore e' affidabile
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
			byte flag;
			if (best <= alphaOriginal) {
				flag = TranspositionTable.UPPER_BOUND;
			} else if (best >= beta) {
				flag = TranspositionTable.LOWER_BOUND;
			} else {
				flag = TranspositionTable.EXACT;
			}
			int packedMove = bestMoveFound != null ? TranspositionTable.packMove(bestMoveFound) : -1;
			tt.store(hash, depth, best, flag, packedMove);
		}

		return best;
	}

	/** Indice mossa da*81+a, dove da/a = riga*9+colonna (stesso schema per killer e history). Solo due charAt: niente parseInt. */
	static int moveIndex(Action a) {
		String f = a.getFrom(), t = a.getTo();
		return ((f.charAt(1) - '1') * 9 + ((f.charAt(0) | 0x20) - 'a')) * 81 + ((t.charAt(1) - '1') * 9 + ((t.charAt(0) | 0x20) - 'a'));
	}

	/**
	 * Punteggio di ordinamento di ogni mossa: mossa TT > prima killer > seconda killer > punteggio history
	 * (0 se la history e' disattivata). Package-private per i test.
	 */
	int[] computeOrderScores(List<Action> moves, int sideIdx, int ply, int ttPacked) {
		final int n = moves.size();
		final int[] sc = new int[n];
		int ttIdx = -1;
		if (ttPacked >= 0) {
			ttIdx = (TranspositionTable.packedFromRow(ttPacked) * 9 + TranspositionTable.packedFromCol(ttPacked)) * 81
					+ TranspositionTable.packedToRow(ttPacked) * 9 + TranspositionTable.packedToCol(ttPacked);
		}
		int k0 = -1, k1 = -1;
		if ((orderingMode & ORDER_KILLERS) != 0) {
			k0 = killers[ply][0];
			k1 = killers[ply][1];
		}
		final boolean useHistory = (orderingMode & ORDER_HISTORY) != 0;
		final int[] hist = historyScore[sideIdx];
		for (int i = 0; i < n; i++) {
			int idx = moveIndex(moves.get(i));
			if (idx == ttIdx) {
				sc[i] = Integer.MAX_VALUE;
			} else if (idx == k0) {
				sc[i] = Integer.MAX_VALUE - 1;
			} else if (idx == k1) {
				sc[i] = Integer.MAX_VALUE - 2;
			} else if (useHistory) {
				sc[i] = hist[idx];
			}
		}
		return sc;
	}

	/** Porta in posizione i la mossa di punteggio massimo tra quelle da i in poi (scambiando mosse E punteggi). A parita' vince la prima. */
	static void selectBest(List<Action> moves, int[] sc, int i) {
		final int n = sc.length;
		int best = i, bv = sc[i];
		for (int j = i + 1; j < n; j++) {
			if (sc[j] > bv) {
				bv = sc[j];
				best = j;
			}
		}
		if (best != i) {
			Action tmp = moves.get(i);
			moves.set(i, moves.get(best));
			moves.set(best, tmp);
			sc[best] = sc[i];
			sc[i] = bv;
		}
	}

	/** La mossa {@code cutoff} ha provocato un taglio beta: aggiorna killer (due per ply) e history. */
	private void recordCutoff(Action cutoff, int sideIdx, int ply, int depth, List<Action> moves, int upTo) {
		final int idx = moveIndex(cutoff);
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

	private boolean checkTimeUp() {
		if (!timeUp) {
			SearchControl c = control;
			if ((c != null && c.stop) || System.currentTimeMillis() >= deadline) {
				timeUp = true;
			}
		}
		return timeUp;
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

	private Action findBestMoveParallel(State rootState, long dl, List<Long> historySinceLastCapture, int nThreads) {
		if (AshtonRules.generateMoves(rootState).isEmpty()) {
			this.lastThreadsUsed = 1;
			return null;
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
			final State rootCopy = rootState.clone();
			final List<Long> historyCopy = new ArrayList<>(historySinceLastCapture);
			final int workerId = i + 1;
			helperFutures[i] = pool.submit(new Runnable() {
				@Override
				public void run() {
					try {
						h.searchRoot(rootCopy, historyCopy, workerId);
					} catch (Throwable t) {
						h.helperError = t; // un helper in errore non deve mai far fallire la mossa
					}
				}
			});
			started[i] = true;
			startedCount++;
		}

		Action best;
		try {
			best = searchRoot(rootState, historySinceLastCapture, 0);
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
				} else if (h.lastBestMove != null) {
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

	/** Porta in testa alla lista la mossa {@code preferred} (tipicamente la
	 *  migliore trovata all'iterazione precedente), lasciando le altre nel loro
	 *  ordine. E' l'ottimizzazione col miglior rapporto costo/beneficio per
	 *  l'alpha-beta pruning con iterative deepening. */
	private static void orderMovesInPlace(List<Action> moves, Action preferred) {
		if (preferred == null) {
			return;
		}
		int idx = moves.indexOf(preferred);
		if (idx > 0) {
			Action tmp = moves.get(0);
			moves.set(0, moves.get(idx));
			moves.set(idx, tmp);
		}
	}

	/**
	 * Come {@link #orderMovesInPlace}, ma confronta per CONTENUTO (coordinate
	 * riga/colonna) invece che per riferimento: la mossa suggerita dalla
	 * transposition table e' impacchettata in un intero, non l'oggetto
	 * {@link Action} originale (che {@link Action} non ridefinisce
	 * {@code equals()}, quindi {@code indexOf} non la troverebbe mai).
	 */
	private static void orderMovesByPackedHint(List<Action> moves, int packedMove) {
		int fr = TranspositionTable.packedFromRow(packedMove);
		int fc = TranspositionTable.packedFromCol(packedMove);
		int tr = TranspositionTable.packedToRow(packedMove);
		int tc = TranspositionTable.packedToCol(packedMove);
		for (int i = 0; i < moves.size(); i++) {
			Action a = moves.get(i);
			if (a.getRowFrom() == fr && a.getColumnFrom() == fc && a.getRowTo() == tr && a.getColumnTo() == tc) {
				if (i > 0) {
					Action tmp = moves.get(0);
					moves.set(0, a);
					moves.set(i, tmp);
				}
				break;
			}
		}
	}
}
