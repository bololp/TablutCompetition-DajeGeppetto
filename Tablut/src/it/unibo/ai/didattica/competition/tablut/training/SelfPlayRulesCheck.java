package it.unibo.ai.didattica.competition.tablut.training;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.MLP;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.aiplayer.ZobristHasher;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Collauda le regole di fine partita del self-play ({@link GameTracker}, {@link SelfPlayTrainer#playGame}):
 * <ol>
 * <li><b>Stallo = sconfitta</b> di chi deve muovere (come in {@code Negamax}), non pareggio.</li>
 * <li><b>Ripetizione = pareggio</b> alla prima ripetizione di una posizione (scacchiera + turno) dall'ultima cattura; la storia
 * passata alla ricerca e' quella che passa il client (hash delle posizioni dall'ultima cattura, corrente compresa, azzerata
 * dalle catture).</li>
 * </ol>
 * Gli oracoli sono INDIPENDENTI dal codice sotto test: chiavi testuali scacchiera+turno al posto degli hash Zobrist, conteggio
 * dei pezzi al posto di {@code AshtonRules.wasCapture}. Ripetizioni e stalli nel self-play naturale sono rari (0 su 6308 episodi
 * nei log esistenti), quindi si FORZANO con scelte di mosse scriptate; in piu' si verificano le invarianti su episodi veri di
 * {@code GenerationTask} (ricerca con la rete, avversario Simple, mosse casuali).
 */
public class SelfPlayRulesCheck {

	static int failures = 0;

	public static void main(String[] args) throws Exception {
		long seed = args.length > 0 ? Long.parseLong(args[0]) : 777L;
		Random rnd = new Random(seed);
		checkResultOf();
		checkTrackerAgainstOracle(rnd);
		checkPlayGameScripted(rnd);
		checkTerminalAndMaxPlies(rnd);
		checkRealEpisodes(seed);
		if (failures == 0) {
			System.out.println("=== SELF-PLAY RULES CHECK OK ===");
		} else {
			System.out.println("=== SELF-PLAY RULES CHECK FALLITO: " + failures + " discrepanze ===");
			System.exit(1);
		}
	}

	static void fail(String m) {
		failures++;
		System.out.println("FAIL - " + m);
	}

	// ------------------------------------------------------------------ oracolo indipendente

	static String keyOf(State s) {
		StringBuilder sb = new StringBuilder();
		for (Pawn[] row : s.getBoard()) {
			for (Pawn p : row) {
				sb.append(p.toString());
			}
		}
		return sb.append('|').append(s.getTurn()).toString();
	}

	static int pieces(State s) {
		int n = 0;
		for (Pawn[] row : s.getBoard()) {
			for (Pawn p : row) {
				if (p == Pawn.WHITE || p == Pawn.BLACK || p == Pawn.KING) {
					n++;
				}
			}
		}
		return n;
	}

	/**
	 * Applica l'oracolo a una sequenza di stati consecutivi. Ritorna l'indice del PRIMO stato che ripete una posizione dall'ultima
	 * cattura (-1 se nessuno). Gli stati terminali interrompono l'analisi (la partita e' finita).
	 */
	static int firstRepetitionIndex(List<State> states) {
		List<String> seen = new ArrayList<>();
		seen.add(keyOf(states.get(0)));
		for (int i = 1; i < states.size(); i++) {
			State prev = states.get(i - 1), cur = states.get(i);
			if (AshtonRules.isTerminal(cur)) {
				return -1;
			}
			if (pieces(cur) < pieces(prev)) { // cattura: nessuna posizione precedente puo' ripresentarsi
				seen.clear();
				seen.add(keyOf(cur));
			} else if (seen.contains(keyOf(cur))) {
				return i;
			} else {
				seen.add(keyOf(cur));
			}
		}
		return -1;
	}

	// ------------------------------------------------------------------ posizioni costruite a mano

	static State emptyBoard(Turn turn) {
		State s = new StateTablut();
		Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[4][4] = Pawn.THRONE;
		s.setTurn(turn);
		return s;
	}

	/** Bianco di turno senza mosse: il Re (unico pezzo bianco) e' circondato da 4 pedine nere. */
	static State whiteStalemate() {
		State s = emptyBoard(Turn.WHITE);
		Pawn[][] b = s.getBoard();
		b[2][2] = Pawn.KING;
		b[1][2] = Pawn.BLACK;
		b[3][2] = Pawn.BLACK;
		b[2][1] = Pawn.BLACK;
		b[2][3] = Pawn.BLACK;
		return s;
	}

	/** Nero di turno senza mosse: l'unica pedina nera e' in un angolo, bloccata da due pedine bianche. */
	static State blackStalemate() {
		State s = emptyBoard(Turn.BLACK);
		Pawn[][] b = s.getBoard();
		b[4][4] = Pawn.KING;
		b[0][0] = Pawn.BLACK;
		b[0][1] = Pawn.WHITE;
		b[1][0] = Pawn.WHITE;
		return s;
	}

	// ------------------------------------------------------------------ 1) esiti

	static void expectResult(String what, GameTracker.Result r, double outcome, String label) {
		if (r.outcome != outcome || !r.label.equals(label)) {
			fail(what + ": ottenuto " + r + ", atteso " + label + "(" + outcome + ")");
		}
	}

	static void checkResultOf() {
		int before = failures;
		State initial = new StateTablut();
		initial.setTurn(Turn.WHITE);
		expectResult("posizione in corso", GameTracker.resultOf(initial, false), 0.0, GameTracker.LABEL_MAXPLIES);
		expectResult("ripetizione su posizione in corso", GameTracker.resultOf(initial, true), 0.0, GameTracker.LABEL_REPETITION);
		State w = initial.clone();
		w.setTurn(Turn.WHITEWIN);
		State b = initial.clone();
		b.setTurn(Turn.BLACKWIN);
		State d = initial.clone();
		d.setTurn(Turn.DRAW);
		expectResult("vittoria bianca", GameTracker.resultOf(w, false), 1.0, "WW");
		expectResult("vittoria nera", GameTracker.resultOf(b, false), -1.0, "BW");
		expectResult("pareggio del motore", GameTracker.resultOf(d, false), 0.0, "D");
		// la ripetizione ha la precedenza (la partita e' stata chiusa li')
		expectResult("ripetizione ha la precedenza", GameTracker.resultOf(w, true), 0.0, GameTracker.LABEL_REPETITION);
		// STALLO: chi deve muovere senza mosse legali PERDE (come Negamax), non pareggia
		State ws = whiteStalemate(), bs = blackStalemate();
		if (!AshtonRules.generateMoves(ws).isEmpty() || !AshtonRules.generateMoves(bs).isEmpty()) {
			fail("setup: le posizioni di stallo costruite a mano hanno mosse legali");
		}
		expectResult("stallo del Bianco", GameTracker.resultOf(ws, false), -1.0, GameTracker.LABEL_NOMOVES_BLACK_WIN);
		expectResult("stallo del Nero", GameTracker.resultOf(bs, false), 1.0, GameTracker.LABEL_NOMOVES_WHITE_WIN);
		System.out.printf("%s %-30s esiti: vittorie, pareggio, ripetizione (precedenza), stallo = sconfitta per entrambi i lati%n",
				failures == before ? "OK  " : "FAIL", "[GameTracker.resultOf]");
	}

	// ------------------------------------------------------------------ 2) tracker vs oracolo

	/** Sceglie, con probabilita' 0.7, l'inversa della PROPRIA ultima mossa (se legale), altrimenti a caso: genera molte ripetizioni. */
	static Action shuffleMove(List<Action> legal, Action ownPrevious, Random rnd) {
		if (ownPrevious != null && rnd.nextDouble() < 0.7) {
			for (Action a : legal) {
				if (a.getFrom().equals(ownPrevious.getTo()) && a.getTo().equals(ownPrevious.getFrom())) {
					return a;
				}
			}
		}
		return legal.get(rnd.nextInt(legal.size()));
	}

	static void checkTrackerAgainstOracle(Random rnd) {
		int before = failures, games = 0, repetitions = 0, captureResets = 0;
		long moves = 0;
		for (int g = 0; g < 500; g++) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			GameTracker tracker = new GameTracker(s);
			List<State> states = new ArrayList<>();
			states.add(s);
			Map<Turn, Action> last = new HashMap<>();
			for (int ply = 0; ply < 120 && !AshtonRules.isTerminal(s); ply++) {
				List<Action> legal = AshtonRules.generateMoves(s);
				if (legal.isEmpty()) {
					break;
				}
				// contratto con la ricerca: la storia contiene gli hash dall'ultima cattura, posizione corrente compresa
				List<Long> expectedHashes = expectedHistory(states);
				if (!tracker.history().equals(expectedHashes)) {
					fail("storia del tracker diversa dall'attesa (gioco " + g + ", semi-mossa " + ply + "): " + tracker.history().size() + " vs "
							+ expectedHashes.size());
					return;
				}
				Action m = shuffleMove(legal, last.get(s.getTurn()), rnd);
				last.put(s.getTurn(), m);
				State prev = s;
				s = tracker.apply(s, m);
				states.add(s);
				moves++;
				if (pieces(s) < pieces(prev) && !AshtonRules.isTerminal(s)) {
					captureResets++;
				}
				boolean expectedRep = firstRepetitionIndex(states) == states.size() - 1;
				if (tracker.isRepetitionDraw() != expectedRep) {
					fail("ripetizione: tracker=" + tracker.isRepetitionDraw() + " oracolo=" + expectedRep + " (gioco " + g + ", semi-mossa " + ply + ")");
					return;
				}
				if (expectedRep) {
					repetitions++;
					break;
				}
			}
			games++;
		}
		if (repetitions < 20 || captureResets < 5) {
			fail("copertura insufficiente: " + repetitions + " ripetizioni e " + captureResets + " azzeramenti per cattura (test vacuo)");
		}
		System.out.printf("%s %-30s %d partite / %d mosse contro l'oracolo: %d ripetizioni, %d azzeramenti per cattura, storia sempre identica%n",
				failures == before ? "OK  " : "FAIL", "[tracker vs oracolo]", games, moves, repetitions, captureResets);
	}

	/** Hash attesi: le posizioni dall'ultima cattura (oracolo = conteggio pezzi), la corrente compresa. */
	static List<Long> expectedHistory(List<State> states) {
		int start = 0;
		for (int i = 1; i < states.size(); i++) {
			if (pieces(states.get(i)) < pieces(states.get(i - 1))) {
				start = i;
			}
		}
		List<Long> h = new ArrayList<>();
		for (int i = start; i < states.size(); i++) {
			h.add(ZobristHasher.fullHash(states.get(i)));
		}
		return h;
	}

	// ------------------------------------------------------------------ 3) playGame scriptato

	/** Scelta "a oscillazione" che verifica anche il contratto sulla storia ricevuta ad ogni chiamata. */
	static final class ShuffleChooser implements SelfPlayTrainer.MoveChooser {
		final Random rnd;
		final Map<Turn, Action> last = new HashMap<>();
		final List<State> seen = new ArrayList<>();
		boolean contractViolated;

		ShuffleChooser(Random rnd) {
			this.rnd = rnd;
		}

		@Override
		public Action choose(State state, List<Long> history) {
			seen.add(state);
			if (!history.equals(expectedHistoryOfCurrent(seen))) {
				contractViolated = true;
			}
			List<Action> legal = AshtonRules.generateMoves(state);
			Action m = shuffleMove(legal, last.get(state.getTurn()), rnd);
			last.put(state.getTurn(), m);
			return m;
		}

		/** Gli stati passati al chooser sono consecutivi: l'oracolo li concatena con la mossa applicata dal tracker. */
		private List<Long> expectedHistoryOfCurrent(List<State> states) {
			return expectedHistory(states);
		}
	}

	static void checkPlayGameScripted(Random rnd) {
		int before = failures, games = 0, rep = 0, other = 0, features = 0;
		for (int g = 0; g < 250; g++) {
			State start = new StateTablut();
			start.setTurn(Turn.WHITE);
			final int[] sampled = { 0 };
			SelfPlayTrainer.PositionListener listener = new SelfPlayTrainer.PositionListener() {
				@Override
				public void onPosition(State s) {
					sampled[0]++;
				}
			};
			ShuffleChooser chooser = new ShuffleChooser(rnd);
			SelfPlayTrainer.GameOutcome out = SelfPlayTrainer.playGame(start, 200, listener, chooser, true);
			games++;
			// la storia che il chooser riceve a ogni mossa e' quella dell'oracolo (stati gia' visti + corrente)
			// nota: il chooser vede solo gli stati in cui si muove: l'oracolo va applicato alla sequenza COMPLETA
			int firstRep = firstRepetitionIndex(out.states);
			String label = out.result.label;
			boolean repLabel = GameTracker.LABEL_REPETITION.equals(label);
			if (repLabel) {
				rep++;
				if (firstRep != out.states.size() - 1) {
					fail("etichetta REPETITION ma l'oracolo vede la prima ripetizione all'indice " + firstRep + " su " + (out.states.size() - 1));
				}
				if (out.result.outcome != 0.0) {
					fail("la ripetizione deve valere 0");
				}
			} else {
				other++;
				if (firstRep != -1) {
					fail("la partita contiene una ripetizione (indice " + firstRep + ") ma NON e' stata chiusa per ripetizione: etichetta " + label);
				}
			}
			if (out.plies != out.states.size() - 1) {
				fail("plies=" + out.plies + " ma gli stati sono " + out.states.size());
			}
			// si campiona una posizione per ogni mossa giocata, piu' quella finale SOLO se la partita si e' fermata per stallo
			boolean stalemateEnd = label.startsWith("NOMOVES");
			int expectedSampled = out.states.size() - 1 + (stalemateEnd ? 1 : 0);
			if (sampled[0] != expectedSampled) {
				fail("posizioni campionate " + sampled[0] + ", attese " + expectedSampled + " (etichetta " + label + ")");
			}
			if (!AshtonRules.isTerminal(out.finalState) && !repLabel && !stalemateEnd && !label.equals(GameTracker.LABEL_MAXPLIES)) {
				fail("etichetta inattesa " + label);
			}
			features += sampled[0];
			if (chooser.contractViolated) {
				fail("la storia passata al chooser non coincide con quella dell'oracolo (partita " + g + ")");
			}
		}
		if (rep < 20) {
			fail("solo " + rep + " partite chiuse per ripetizione su " + games + ": test vacuo");
		}
		System.out.printf("%s %-30s %d partite scriptate: %d chiuse per ripetizione (verificate dall'oracolo), %d altre senza alcuna ripetizione residua%n",
				failures == before ? "OK  " : "FAIL", "[playGame ripetizione]", games, rep, other);
	}

	static void checkTerminalAndMaxPlies(Random rnd) {
		int before = failures;
		// --- stallo come stato di PARTENZA: la partita finisce subito, la posizione viene campionata, l'esito e' una sconfitta ---
		for (int side = 0; side < 2; side++) {
			State start = side == 0 ? whiteStalemate() : blackStalemate();
			final int[] sampled = { 0 };
			SelfPlayTrainer.GameOutcome out = SelfPlayTrainer.playGame(start, 100, new SelfPlayTrainer.PositionListener() {
				@Override
				public void onPosition(State s) {
					sampled[0]++;
				}
			}, new SelfPlayTrainer.MoveChooser() {
				@Override
				public Action choose(State s, List<Long> h) {
					fail("il chooser non deve essere chiamato in una posizione senza mosse");
					return null;
				}
			}, true);
			double expected = side == 0 ? -1.0 : 1.0;
			String label = side == 0 ? GameTracker.LABEL_NOMOVES_BLACK_WIN : GameTracker.LABEL_NOMOVES_WHITE_WIN;
			expectResult("stallo di partenza (" + (side == 0 ? "Bianco" : "Nero") + ")", out.result, expected, label);
			if (out.plies != 0 || sampled[0] != 1 || out.states.size() != 1) {
				fail("stallo di partenza: plies=" + out.plies + ", campionate=" + sampled[0] + ", stati=" + out.states.size());
			}
		}
		// --- tetto di semi-mosse: prima mossa legale, 3 semi-mosse ---
		State start = new StateTablut();
		start.setTurn(Turn.WHITE);
		SelfPlayTrainer.GameOutcome max = SelfPlayTrainer.playGame(start, 3, new SelfPlayTrainer.PositionListener() {
			@Override
			public void onPosition(State s) {
			}
		}, new SelfPlayTrainer.MoveChooser() {
			@Override
			public Action choose(State s, List<Long> h) {
				return AshtonRules.generateMoves(s).get(0);
			}
		}, true);
		expectResult("tetto di semi-mosse", max.result, 0.0, GameTracker.LABEL_MAXPLIES);
		if (max.plies != 3) {
			fail("tetto: plies=" + max.plies);
		}
		// --- vittoria: si cerca una posizione con fuga del Re (Bianco) o cattura del Re (Nero) in una mossa ---
		int wins = 0;
		for (int attempt = 0; attempt < 6000 && wins < 6; attempt++) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			for (int i = 0; i < 6 + rnd.nextInt(40) && !AshtonRules.isTerminal(s); i++) {
				List<Action> mv = AshtonRules.generateMoves(s);
				if (mv.isEmpty()) {
					break;
				}
				s = AshtonRules.applyMove(s, mv.get(rnd.nextInt(mv.size())));
			}
			if (AshtonRules.isTerminal(s) || AshtonRules.generateMoves(s).isEmpty()) {
				continue;
			}
			Action winning = null;
			for (Action a : AshtonRules.generateMoves(s)) {
				Turn after = AshtonRules.applyMove(s, a).getTurn();
				if (after == (s.getTurn() == Turn.WHITE ? Turn.WHITEWIN : Turn.BLACKWIN)) {
					winning = a;
				}
			}
			if (winning == null) {
				continue;
			}
			final Action win = winning;
			final int[] sampled = { 0 };
			SelfPlayTrainer.GameOutcome out = SelfPlayTrainer.playGame(s, 50, new SelfPlayTrainer.PositionListener() {
				@Override
				public void onPosition(State st) {
					sampled[0]++;
				}
			}, new SelfPlayTrainer.MoveChooser() {
				@Override
				public Action choose(State st, List<Long> h) {
					return win;
				}
			}, true);
			boolean whiteWon = s.getTurn() == Turn.WHITE;
			expectResult("vittoria in 1 mossa", out.result, whiteWon ? 1.0 : -1.0, whiteWon ? "WW" : "BW");
			if (out.plies != 1 || sampled[0] != 1 || !AshtonRules.isTerminal(out.finalState)) {
				fail("vittoria in 1: plies=" + out.plies + ", campionate=" + sampled[0]);
			}
			wins++;
		}
		if (wins < 4) {
			fail("trovate solo " + wins + " posizioni con vittoria in 1 mossa: test vacuo");
		}
		System.out.printf("%s %-30s stallo di partenza (Bianco e Nero), tetto di semi-mosse, %d vittorie in 1 mossa (WW/BW)%n",
				failures == before ? "OK  " : "FAIL", "[playGame esiti]", wins);
	}

	// ------------------------------------------------------------------ 4) episodi veri

	static void checkRealEpisodes(long seed) {
		int before = failures;
		MLP net = new MLP(NeuralEvaluator.defaultConfig(), seed);
		SelfPlayTrainer.recordStatesForTests = true;
		Map<String, Integer> counts = new HashMap<>();
		int episodes = 0;
		try {
			for (int i = 0; i < 24; i++) {
				double eps = (i % 3 == 0) ? 0.0 : 0.25; // anche mosse casuali: aprono linee insolite
				SelfPlayTrainer.GenerationTask task = new SelfPlayTrainer.GenerationTask(net, TablutFeatures.G_DEFAULT, new SimpleEvaluator(),
						new ArrayList<String>(), 3, eps, 10, 20, 1.0, 0.4, 0.0, seed * 1000 + i);
				SelfPlayTrainer.Trajectory t = task.call();
				episodes++;
				counts.merge(t.outcomeLabel, 1, Integer::sum);
				verifyEpisode(t);
			}
		} finally {
			SelfPlayTrainer.recordStatesForTests = false;
		}
		System.out.printf("%s %-30s %d episodi veri (rete + avversario Simple + mosse casuali): invarianti rispettate; esiti %s%n",
				failures == before ? "OK  " : "FAIL", "[episodi reali]", episodes, counts);
	}

	/** Invarianti di un episodio, verificate con l'oracolo sugli stati registrati. */
	static void verifyEpisode(SelfPlayTrainer.Trajectory t) {
		List<State> st = t.states;
		if (st == null || st.isEmpty()) {
			fail("stati non registrati");
			return;
		}
		State last = st.get(st.size() - 1);
		int firstRep = firstRepetitionIndex(st);
		String label = t.outcomeLabel;
		boolean rep = GameTracker.LABEL_REPETITION.equals(label);
		if (t.plies != st.size() - 1) {
			fail("plies=" + t.plies + " e stati=" + st.size());
		}
		switch (label) {
		case "WW":
		case "BW":
			if (last.getTurn() != (label.equals("WW") ? Turn.WHITEWIN : Turn.BLACKWIN)) {
				fail("etichetta " + label + " ma lo stato finale ha turno " + last.getTurn());
			}
			if (t.outcome != (label.equals("WW") ? 1.0 : -1.0)) {
				fail("esito numerico incoerente con " + label);
			}
			break;
		case GameTracker.LABEL_REPETITION:
			if (firstRep != st.size() - 1 || t.outcome != 0.0) {
				fail("REPETITION non confermata dall'oracolo (prima ripetizione all'indice " + firstRep + " su " + (st.size() - 1) + ")");
			}
			break;
		case GameTracker.LABEL_NOMOVES_BLACK_WIN:
		case GameTracker.LABEL_NOMOVES_WHITE_WIN:
			boolean whiteStalemated = label.equals(GameTracker.LABEL_NOMOVES_BLACK_WIN);
			if (!AshtonRules.generateMoves(last).isEmpty() || last.getTurn() != (whiteStalemated ? Turn.WHITE : Turn.BLACK)
					|| t.outcome != (whiteStalemated ? -1.0 : 1.0)) {
				fail("stallo non confermato: " + label);
			}
			break;
		case GameTracker.LABEL_MAXPLIES:
			if (t.plies != 300 || t.outcome != 0.0) {
				fail("MAXPLIES con plies=" + t.plies);
			}
			break;
		default:
			fail("etichetta sconosciuta " + label);
		}
		if (!rep && firstRep != -1) {
			fail("l'episodio (" + label + ") contiene una ripetizione all'indice " + firstRep + " e non e' stato chiuso: il self-play ignora la regola");
		}
		// feature campionate: una per posizione in cui si doveva muovere (non per lo stato finale terminale / ripetuto / al tetto)
		int expected = st.size();
		if (AshtonRules.isTerminal(last) || rep || GameTracker.LABEL_MAXPLIES.equals(label)) {
			expected--;
		}
		if (t.features.size() != expected || t.roles.length != expected) {
			fail("feature campionate " + t.features.size() + ", attese " + expected + " (" + label + ")");
		}
	}
}
