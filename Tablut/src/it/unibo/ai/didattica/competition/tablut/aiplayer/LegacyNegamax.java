package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;

/**
 * COPIA CONGELATA del motore di ricerca storico (prima della migrazione a {@code FastState}): Negamax alpha-beta con
 * iterative deepening, transposition table, pareggio per ripetizione con protezione GHI, PVS, killer e history. Lavora su
 * {@link State}/{@link Action} con {@link AshtonRules} e {@link ZobristHasher}, e ALLOCA uno {@code State} per ogni figlio.
 * <p>
 * Serve SOLO come riferimento dei test ({@code SearchEquivalenceCheck}): la ricerca attuale ({@link Negamax}) deve dare, a
 * profondita' fissa e a thread singolo, lo stesso punteggio, la stessa mossa e lo stesso numero di nodi di questa. Nessun
 * codice di gioco la usa. E' stata tagliata solo la parte Lazy SMP (non confrontabile: non deterministica); la logica di
 * ricerca e' quella originale, riga per riga. Non va "migliorata": perderebbe il suo valore di riferimento.
 */
public class LegacyNegamax {

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
	private long pvsScouts;
	private long pvsReSearches;

	public LegacyNegamax(Evaluator evaluator) {
		this(evaluator, 40);
	}

	public LegacyNegamax(Evaluator evaluator, int maxDepth) {
		this(evaluator, maxDepth, true, DEFAULT_TT_SIZE);
	}

	/**
	 * @param useTT false per disabilitare la transposition table (utile solo
	 *              per confronti/debug: con lo stesso identico alpha-beta,
	 *              deve dare sempre lo stesso identico punteggio/mossa di
	 *              {@code true}, solo piu' lentamente - se cosi' non fosse,
	 *              ci sarebbe un bug nella gestione dei bound EXACT/LOWER/UPPER)
	 */
	public LegacyNegamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize) {
		this(evaluator, maxDepth, useTT, ttSize, true);
	}

	/**
	 * @param usePVS false per usare l'alpha-beta classico (utile solo per confronti/debug: il valore
	 *               alla radice deve essere identico, cambia solo il numero di nodi)
	 */
	public LegacyNegamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS) {
		this(evaluator, maxDepth, useTT, ttSize, usePVS, ORDER_DEFAULT);
	}

	/**
	 * @param orderingMode {@link #ORDER_NONE} (solo mossa TT, come prima), {@link #ORDER_KILLERS}, {@link #ORDER_HISTORY}
	 *                     o la loro unione ({@link #ORDER_DEFAULT}); utile per confronti: il valore di ogni nodo e' lo stesso
	 */
	public LegacyNegamax(Evaluator evaluator, int maxDepth, boolean useTT, int ttSize, boolean usePVS, int orderingMode) {
		this.evaluator = evaluator;
		this.maxDepth = maxDepth;
		this.useTT = useTT;
		this.usePVS = usePVS;
		this.orderingMode = orderingMode;
		this.killers = newKillers(maxDepth);
		this.tt = useTT ? new TranspositionTable(ttSize) : null;
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
		this.deadline = System.currentTimeMillis() + Math.max(50, timeBudgetMillis - SAFETY_MARGIN_MILLIS);
		return searchRoot(rootState, historySinceLastCapture);
	}

	/**
	 * Iterative deepening alla radice. Usa {@link #deadline} gia' impostato dal chiamante e lascia i risultati nei campi {@code last*}.
	 */
	Action searchRoot(State rootState, List<Long> historySinceLastCapture) {
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
		if (!timeUp && System.currentTimeMillis() >= deadline) {
			timeUp = true;
		}
		return timeUp;
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
