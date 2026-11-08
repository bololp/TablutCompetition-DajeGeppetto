package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Arrays;

import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Stato incrementale NNUE di UN worker di ricerca: una pila di accumulatori indicizzata dallo stesso ply degli stack di {@link Negamax}
 * ({@code UndoInfo}, mosse). Il frame {@code p} descrive la posizione raggiunta dopo {@code p} mosse dalla radice (frame 0).
 *
 * <h3>Aggiornamento lazy</h3>
 * {@link #push} registra soltanto la DIFFERENZA di feature della mossa (pedina mossa, da, a, catture): costa pochi byte. L'accumulatore
 * di un frame si calcola solo quando serve ({@link #evaluate}), a partire dal piu' vicino antenato gia' calcolato, applicando in
 * avanti le differenze dei ply intermedi (e lasciandoli calcolati, cosi' i fratelli ripartono dal genitore). I nodi che non vengono
 * mai valutati (terminali, ripetizioni, tagli della TT a profondita' 0, nodi interni) non pagano l'aggiornamento.
 * <ul>
 * <li><b>Mossa ordinaria</b>: {@code acc' = acc - W[(tipo, da)] + W[(tipo, a)]}.</li>
 * <li><b>Cattura</b> (anche multipla, Re compreso): in piu' {@code - W[(tipo catturato, cella)]} per ogni pedina tolta, lette
 * dall'{@link UndoInfo} della mossa.</li>
 * <li><b>Mossa del Re</b>: come una mossa ordinaria con il tipo RE; nessuna ricostruzione (la prospettiva e' unica, si veda
 * {@link NnueNetwork}).</li>
 * <li><b>Unmake</b>: nessuna azione. I frame sono una pila e il frame del genitore non e' mai toccato dai figli: dopo l'annullamento
 * di una mossa il genitore e' ancora valido e il fratello successivo riscrive il frame del figlio con un nuovo {@code push}.</li>
 * </ul>
 *
 * <h3>Coerenza e ricostruzione completa</h3>
 * Ogni frame porta la FIRMA dell'insieme di feature attive (XOR delle chiavi Zobrist di ciascuna feature), aggiornata con le stesse
 * differenze. Prima di valutare la si confronta con la firma che {@link FastState} ricava dal proprio hash incrementale
 * ({@link NnueNetwork#boardSignature}), calcolato da codice indipendente da questo: se differiscono (qualcuno ha cambiato la posizione
 * senza avvisare la sessione, un {@code push} mancato, una radice non installata) il frame viene RICOSTRUITO da zero dalla scacchiera
 * e il valore resta corretto; l'evento e' contato ({@link #fallbackRebuilds()}) perche' in una ricerca normale non deve mai accadere.
 * Si ricostruisce da zero anche quando un {@code push} riceve dati inutilizzabili (cella fuori scacchiera, pedina senza feature) o
 * manca il frame di partenza. La ricostruzione e' sempre possibile solo per il ply CORRENTE (e' l'unico di cui si ha la scacchiera):
 * i frame intermedi restano da calcolare.
 * <p>
 * Nessuna allocazione in {@link #push}/{@link #evaluate} dopo la costruzione, salvo la crescita della pila oltre la capacita' iniziale.
 * Non e' thread-safe: una sessione appartiene a un solo worker.
 */
public final class NnueSession implements IncrementalEvaluator.Session {

	private static final byte PENDING = 0; // differenza registrata, accumulatore da calcolare
	private static final byte COMPUTED = 1; // accumulatore e firma validi per il percorso corrente
	private static final byte REBUILD = 2; // nessuna differenza utilizzabile: solo una ricostruzione dalla scacchiera puo' dare il frame

	private static final int MAX_CAP = 4; // pedine catturabili da una mossa ({@code UndoInfo})

	private final NnueEvaluator ev;
	private final NnueNetwork net;
	private final int[] act;

	private int cap;
	private int[][] acc;
	private long[] sig;
	private byte[] state;
	private byte[] mvPiece, mvFrom, mvTo, capN;
	private byte[] capSq, capPc; // MAX_CAP per frame

	private long deltaSteps;
	private long fullRebuilds;
	private long fallbackRebuilds;

	NnueSession(NnueEvaluator ev, int capacityPlies) {
		this.ev = ev;
		this.net = ev.net;
		this.act = new int[net.scratchSize()];
		allocate(Math.max(2, capacityPlies));
	}

	private void allocate(int n) {
		final int old = acc == null ? 0 : acc.length;
		if (n <= old) {
			return;
		}
		acc = acc == null ? new int[n][] : Arrays.copyOf(acc, n);
		for (int i = old; i < n; i++) {
			acc[i] = new int[net.hidden()];
		}
		sig = sig == null ? new long[n] : Arrays.copyOf(sig, n);
		state = state == null ? new byte[n] : Arrays.copyOf(state, n);
		for (int i = old; i < n; i++) {
			state[i] = REBUILD;
		}
		mvPiece = mvPiece == null ? new byte[n] : Arrays.copyOf(mvPiece, n);
		mvFrom = mvFrom == null ? new byte[n] : Arrays.copyOf(mvFrom, n);
		mvTo = mvTo == null ? new byte[n] : Arrays.copyOf(mvTo, n);
		capN = capN == null ? new byte[n] : Arrays.copyOf(capN, n);
		capSq = capSq == null ? new byte[n * MAX_CAP] : Arrays.copyOf(capSq, n * MAX_CAP);
		capPc = capPc == null ? new byte[n * MAX_CAP] : Arrays.copyOf(capPc, n * MAX_CAP);
		cap = n;
	}

	@Override
	public void reset(FastState root) {
		rebuild(0, root);
		fullRebuilds++;
	}

	@Override
	public void push(int ply, int move, FastState after, UndoInfo undo) {
		if (ply < 1) {
			throw new IllegalArgumentException("push: il ply del figlio e' >= 1 (0 e' la radice, si installa con reset)");
		}
		if (ply >= cap) {
			allocate(Math.max(ply + 1, cap * 2));
		}
		final int from = move & 127;
		final int to = move >>> 7;
		boolean usable = from < NnueNetwork.NUM_SQUARES && to < NnueNetwork.NUM_SQUARES;
		final byte piece = usable ? after.board[to] : 0;
		usable = usable && NnueNetwork.baseOf(piece) >= 0;
		final int n = undo.capturedCount();
		usable = usable && n <= MAX_CAP;
		if (usable) {
			final int o = ply * MAX_CAP;
			for (int i = 0; i < n; i++) {
				final int sq = undo.capturedSquare(i);
				final byte pc = undo.capturedPiece(i);
				if (sq < 0 || sq >= NnueNetwork.NUM_SQUARES || NnueNetwork.baseOf(pc) < 0) {
					usable = false;
					break;
				}
				capSq[o + i] = (byte) sq;
				capPc[o + i] = pc;
			}
		}
		if (!usable) {
			state[ply] = REBUILD;
			return;
		}
		mvPiece[ply] = piece;
		mvFrom[ply] = (byte) from;
		mvTo[ply] = (byte) to;
		capN[ply] = (byte) n;
		state[ply] = PENDING;
	}

	@Override
	public double evaluate(int ply, FastState s) {
		if (ply >= cap) {
			allocate(ply + 1);
		}
		if (state[ply] != COMPUTED) {
			materialize(ply, s);
		}
		if (sig[ply] != NnueNetwork.boardSignature(s)) {
			rebuild(ply, s);
			fallbackRebuilds++;
		}
		final int role = s.turn == BoardConstants.BLACK_TURN ? 1 : 0;
		return NnueNetwork.valueOf(net.logit(acc[ply], act, role)) * ev.outputScale;
	}

	/** Porta il frame {@code ply} allo stato COMPUTED applicando le differenze a partire dal piu' vicino antenato calcolato. */
	private void materialize(int ply, FastState s) {
		int k = ply;
		while (k > 0 && state[k] != COMPUTED) {
			k--;
		}
		boolean ok = state[k] == COMPUTED;
		for (int j = k + 1; ok && j <= ply; j++) {
			ok = state[j] == PENDING;
		}
		if (!ok) {
			rebuild(ply, s);
			fallbackRebuilds++;
			return;
		}
		for (int j = k + 1; j <= ply; j++) {
			applyDelta(j);
		}
	}

	private void applyDelta(int j) {
		final int base = NnueNetwork.baseOf(mvPiece[j]);
		final int fromF = base + mvFrom[j];
		final int toF = base + mvTo[j];
		final int[] dst = acc[j];
		net.applyMove(acc[j - 1], dst, fromF, toF);
		long g = sig[j - 1] ^ NnueNetwork.FEATURE_KEY[fromF] ^ NnueNetwork.FEATURE_KEY[toF];
		final int o = j * MAX_CAP;
		for (int i = 0, n = capN[j]; i < n; i++) {
			final int f = NnueNetwork.baseOf(capPc[o + i]) + capSq[o + i];
			net.subtract(dst, f);
			g ^= NnueNetwork.FEATURE_KEY[f];
		}
		sig[j] = g;
		state[j] = COMPUTED;
		deltaSteps++;
	}

	private void rebuild(int ply, FastState s) {
		if (ply >= cap) {
			allocate(ply + 1);
		}
		sig[ply] = net.refresh(s.board, acc[ply]);
		state[ply] = COMPUTED;
	}

	// ---- osservabili (test, benchmark): non influenzano la valutazione ----

	/** Frame calcolati applicando differenze (il lavoro incrementale vero). */
	public long deltaSteps() {
		return deltaSteps;
	}

	/** Ricostruzioni attese: una per {@link #reset}. */
	public long fullRebuilds() {
		return fullRebuilds;
	}

	/**
	 * Ricostruzioni di ripiego: firma incoerente con la scacchiera, differenza inutilizzabile o frame di partenza mancante. In una
	 * ricerca normale vale sempre 0.
	 */
	@Override
	public long fallbackRebuilds() {
		return fallbackRebuilds;
	}

	/** Capacita' corrente della pila (ply). */
	public int capacity() {
		return cap;
	}
}
