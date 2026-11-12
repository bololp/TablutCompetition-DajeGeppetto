package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Test dell'auto-gioco ({@link NnueSelfPlay}) e della rietichettatura ({@link NnueRelabel}), e dei {@code main} degli strumenti di dati.
 * <ul>
 * <li><b>Conversione punteggio -> valore</b>: segno (il punteggio di ricerca e' del lato al tratto, il valore e' del Bianco), soglia delle vittorie
 * forzate, scala, limiti.</li>
 * <li><b>Ogni partita e' rigiocata in modo INDIPENDENTE</b> con le regole storiche ({@link AshtonRules}, {@link GameTracker}): le mosse registrate sono
 * legali, le posizioni salvate sono quelle della partita (con il lato giusto, dal ply di registrazione), l'esito coincide e vale solo se la partita
 * non e' troncata.</li>
 * <li><b>Determinismo</b>: stesso seme = stessa partita; lo stesso dataset con 1 o 3 thread del generatore, bit per bit.</li>
 * <li>Rietichettatura: vittorie forzate nei due versi (Bianco che fugge = +1, Nero che cattura = -1), {@code onlyMissing}, campi non toccati.</li>
 * <li>Configurazioni non valide rifiutate; {@code main} di import, auto-gioco e rietichettatura in un sottoprocesso.</li>
 * </ul>
 */
public class NnueSelfPlayCheck {

	private static int failures = 0;
	private static long checks = 0;

	private static void check(boolean ok, String msg) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	private static boolean same(double a, double b) {
		return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
	}

	public static void main(String[] args) throws Exception {
		testScoreToValue();
		testPlayGame();
		testGenerate();
		testConfigErrors();
		testRelabel();
		testCli();
		System.out.println(failures == 0 ? "=== NNUE SELFPLAY CHECK OK: " + checks + " verifiche ===" : "=== NNUE SELFPLAY CHECK: " + failures + " FALLIMENTI su " + checks + " ===");
		if (failures != 0) {
			System.exit(1);
		}
	}

	private static NnueSelfPlay.Config cfg(int games, int depth, double eps, int open, int maxPlies, long seed) {
		final NnueSelfPlay.Config c = new NnueSelfPlay.Config();
		c.games = games;
		c.depth = depth;
		c.eps = eps;
		c.openRandom = open;
		c.maxPlies = maxPlies;
		c.seed = seed;
		return c;
	}

	// ====================================================================== punteggio -> valore

	private static void testScoreToValue() {
		check(same(NnueSelfPlay.scoreToValue(500, true, 1000), 0.5), "Bianco al tratto, +500: valore +0.5");
		check(same(NnueSelfPlay.scoreToValue(500, false, 1000), -0.5), "Nero al tratto, +500 per il Nero: valore -0.5 per il Bianco");
		check(same(NnueSelfPlay.scoreToValue(-200, true, 1000), -0.2) && same(NnueSelfPlay.scoreToValue(-200, false, 1000), 0.2), "segno invertito dal lato al tratto");
		check(same(NnueSelfPlay.scoreToValue(5000, true, 1000), 1.0) && same(NnueSelfPlay.scoreToValue(-5000, true, 1000), -1.0), "limitato a [-1, 1]");
		check(same(NnueSelfPlay.scoreToValue(1_000_003, true, 1000), 1.0) && same(NnueSelfPlay.scoreToValue(-1_000_003, true, 1000), -1.0), "vittoria/sconfitta forzata del Bianco al tratto");
		check(same(NnueSelfPlay.scoreToValue(1_000_003, false, 1000), -1.0) && same(NnueSelfPlay.scoreToValue(-1_000_003, false, 1000), 1.0), "il Nero al tratto che vince = -1 per il Bianco");
		check(same(NnueSelfPlay.scoreToValue(50, true, 100), 0.5), "scala 100");
		// la soglia delle vittorie forzate vale anche con una scala enorme (dove la divisione le lascerebbe vicine a 0)
		check(same(NnueSelfPlay.scoreToValue(NnueSelfPlay.PROVEN_SCORE, true, 1e9), 1.0) && NnueSelfPlay.scoreToValue(NnueSelfPlay.PROVEN_SCORE - 1, true, 1e9) < 0.01,
				"la soglia di vittoria forzata e' indipendente dalla scala");
		check(same(NnueSelfPlay.scoreToValue(0, true, 1000), 0.0), "zero");
	}

	// ====================================================================== una partita

	/** Rigioca {@code g.moves} con le regole storiche e controlla tutto cio' che la partita dichiara. */
	private static void verifyGame(NnueSelfPlay.Game g, NnueSelfPlay.Config cfg, String label) {
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		final GameTracker tracker = new GameTracker(s);
		boolean legal = true, positionsOk = true, rolesOk = true;
		int recorded = 0;
		for (int i = 0; i < g.moves.size(); i++) {
			boolean found = false;
			for (Action a : AshtonRules.generateMoves(s)) {
				found |= a.getFrom().equals(g.moves.get(i).getFrom()) && a.getTo().equals(g.moves.get(i).getTo());
			}
			legal &= found;
			if (!found) {
				break;
			}
			if (i >= cfg.openRandom) {
				// la posizione di ply i e' registrata
				final int k = recorded++;
				positionsOk &= k < g.boards.size() && g.plies.get(k) == i && Arrays.equals(g.boards.get(k), FastState.fromState(s).board);
				rolesOk &= k < g.roles.size() && g.roles.get(k) == (s.getTurn() == Turn.WHITE ? 0 : 1) && (i % 2 == 0) == (g.roles.get(k) == 0);
			}
			s = tracker.apply(s, g.moves.get(i));
		}
		check(legal, label + ": tutte le mosse registrate devono essere legali");
		check(positionsOk && recorded == g.boards.size(), label + ": le posizioni registrate devono essere quelle della partita (registrate " + g.boards.size() + ", attese " + recorded + ")");
		check(rolesOk, label + ": il lato al tratto registrato deve essere quello della partita");
		check(g.applied == g.moves.size() && g.boards.size() == g.values.size() && g.boards.size() == g.plies.size() && g.boards.size() == g.roles.size(), label + ": liste coerenti");
		final GameTracker.Result res = tracker.result(s);
		check(res.label.equals(g.result.label) && same(res.outcome, g.result.outcome), label + ": l'esito dichiarato (" + g.result.label + ") deve essere quello della riproduzione (" + res.label + ")");
		check(g.outValid == !GameTracker.LABEL_MAXPLIES.equals(res.label), label + ": l'esito vale solo se la partita non e' troncata");
		if (GameTracker.LABEL_MAXPLIES.equals(res.label)) {
			check(g.applied >= cfg.maxPlies || g.moves.size() == cfg.maxPlies, label + ": troncata solo al limite di semi-mosse (" + g.applied + ")");
		}
		boolean range = true;
		for (double v : g.values) {
			range &= v >= -1.0 && v <= 1.0 && !Double.isNaN(v);
		}
		check(range, label + ": i valori devono stare in [-1, 1]");
	}

	private static void testPlayGame() {
		final Evaluator simple = new SimpleEvaluator();
		final Evaluator nnue = new NnueEvaluator(NnueNetwork.random(16, 4, 3));
		for (Evaluator ev : new Evaluator[] { simple, nnue }) {
			final String name = ev instanceof NnueEvaluator ? "NNUE" : "Simple";
			for (long seed = 1; seed <= 4; seed++) {
				final NnueSelfPlay.Config c = cfg(1, 2, 0.1, 3, 200, seed);
				final NnueSelfPlay.Game g = NnueSelfPlay.playGame(ev, ev, c, seed * 977);
				verifyGame(g, c, name + " seme " + seed);
				check(g.boards.size() > 5, name + " seme " + seed + ": partita troppo corta per essere significativa (" + g.boards.size() + ")");
				// determinismo
				final NnueSelfPlay.Game g2 = NnueSelfPlay.playGame(ev, ev, c, seed * 977);
				boolean eq = g.moves.size() == g2.moves.size() && g.boards.size() == g2.boards.size();
				for (int i = 0; eq && i < g.moves.size(); i++) {
					eq &= g.moves.get(i).toString().equals(g2.moves.get(i).toString());
				}
				for (int i = 0; eq && i < g.values.size(); i++) {
					eq &= same(g.values.get(i), g2.values.get(i)) && Arrays.equals(g.boards.get(i), g2.boards.get(i));
				}
				check(eq && g.outValid == g2.outValid && same(g.result.outcome, g2.result.outcome), name + " seme " + seed + ": lo stesso seme deve dare la stessa partita");
			}
		}
		// semi diversi, partite diverse
		final NnueSelfPlay.Config c = cfg(1, 2, 0.1, 3, 200, 1);
		final NnueSelfPlay.Game a = NnueSelfPlay.playGame(simple, simple, c, 1), b = NnueSelfPlay.playGame(simple, simple, c, 2);
		boolean differ = a.moves.size() != b.moves.size();
		for (int i = 0; !differ && i < a.moves.size(); i++) {
			differ = !a.moves.get(i).toString().equals(b.moves.get(i).toString());
		}
		check(differ, "semi diversi devono dare partite diverse");
		// esplorazione totale: ogni mossa dopo l'apertura e' casuale, ma le posizioni sono comunque etichettate dalla ricerca
		final NnueSelfPlay.Config all = cfg(1, 2, 1.0, 2, 80, 5);
		final NnueSelfPlay.Game ge = NnueSelfPlay.playGame(simple, simple, all, 5);
		verifyGame(ge, all, "eps = 1");
		check(ge.boards.size() > 5 && ge.values.size() == ge.boards.size(), "eps = 1: le posizioni sono comunque registrate ed etichettate");
		// eps cambia davvero le mosse: a parita' di seme, tutte casuali (eps = 1) e tutte cercate (eps = 0) danno partite diverse
		final NnueSelfPlay.Config e0 = cfg(1, 2, 0.0, 2, 60, 5), e1 = cfg(1, 2, 1.0, 2, 60, 5);
		final NnueSelfPlay.Game g0 = NnueSelfPlay.playGame(simple, simple, e0, 5), g1 = NnueSelfPlay.playGame(simple, simple, e1, 5);
		boolean diff = g0.moves.size() != g1.moves.size();
		for (int i = 0; !diff && i < g0.moves.size(); i++) {
			diff = !g0.moves.get(i).toString().equals(g1.moves.get(i).toString());
		}
		check(diff, "eps = 1 deve dare una partita diversa da eps = 0 (le mosse sono casuali)");
		// nessuna posizione registrata se l'apertura casuale copre tutta la partita; troncata, senza esito
		final NnueSelfPlay.Config op = cfg(1, 2, 0.0, 50, 12, 5);
		final NnueSelfPlay.Game go = NnueSelfPlay.playGame(simple, simple, op, 5);
		verifyGame(go, op, "apertura casuale lunga");
		check(go.boards.isEmpty() && !go.outValid && go.applied == 12, "apertura >= limite: nessuna posizione registrata, partita troncata senza esito (" + go.boards.size() + ", " + go.applied + ")");
		// troncamento
		final NnueSelfPlay.Config tr = cfg(1, 2, 0.0, 0, 6, 9);
		final NnueSelfPlay.Game gt = NnueSelfPlay.playGame(simple, simple, tr, 9);
		verifyGame(gt, tr, "troncata a 6");
		check(!gt.outValid && gt.applied == 6 && gt.boards.size() == 6, "troncata a 6 semi-mosse: nessun esito, 6 posizioni (" + gt.applied + ", " + gt.boards.size() + ")");
		// due valutatori diversi (Bianco Simple, Nero NNUE)
		final NnueSelfPlay.Config mix = cfg(1, 2, 0.05, 2, 150, 7);
		final NnueSelfPlay.Game gm = NnueSelfPlay.playGame(simple, nnue, mix, 7);
		verifyGame(gm, mix, "Simple contro NNUE");
		check(gm.boards.size() > 5, "Simple contro NNUE: partita giocata");
	}

	// ====================================================================== un dataset

	private static void testGenerate() throws Exception {
		final Evaluator ev = new SimpleEvaluator();
		final NnueSelfPlay.Config c = cfg(8, 2, 0.1, 3, 150, 21);
		c.threads = 1;
		final NnueDataset d1 = NnueSelfPlay.generate(ev, ev, c, null);
		c.threads = 3;
		final NnueDataset d3 = NnueSelfPlay.generate(ev, ev, c, null);
		check(d1.equals(d3), "lo stesso seme deve dare lo stesso dataset con 1 e con 3 thread (a profondita' fissa)");
		check(d1.n > 100 && d1.numGames() <= 8, "dataset: attese molte posizioni (" + d1.n + ") in al piu' 8 partite (" + d1.numGames() + ")");
		// le partite di un dataset sono DIVERSE fra loro (un seme per partita): le prime posizioni registrate non coincidono
		final java.util.Set<String> firstPositions = new java.util.HashSet<>();
		final boolean[] seen = new boolean[8];
		for (int i = 0; i < d1.n; i++) {
			if (!seen[d1.game[i]]) {
				seen[d1.game[i]] = true;
				firstPositions.add(Arrays.toString(Arrays.copyOfRange(d1.boards, 81 * i, 81 * i + 81)));
			}
		}
		check(firstPositions.size() >= 4, "dataset: le partite devono differire fra loro (prime posizioni distinte: " + firstPositions.size() + " su " + d1.numGames() + ")");
		boolean src = true, vals = true, plyOk = true, toEndOk = true, outConst = true;
		final int[] firstOut = new int[8], gameLen = new int[8];
		Arrays.fill(firstOut, Integer.MIN_VALUE);
		Arrays.fill(gameLen, -1);
		for (int i = 0; i < d1.n; i++) {
			src &= d1.source[i] == NnueDataset.SRC_SELFPLAY;
			vals &= d1.valValid[i] == 1;
			plyOk &= d1.ply[i] >= c.openRandom && d1.game[i] >= 0 && d1.game[i] < 8;
			final int g = d1.game[i];
			if (d1.outValid[i] == 1) {
				toEndOk &= d1.toEnd[i] >= 1;
				final int len = d1.ply[i] + d1.toEnd[i]; // lunghezza della partita: la stessa per tutte le sue posizioni
				if (gameLen[g] < 0) {
					gameLen[g] = len;
					firstOut[g] = d1.outcome[i];
				}
				toEndOk &= len == gameLen[g];
				outConst &= d1.outcome[i] == firstOut[g];
			} else {
				toEndOk &= d1.toEnd[i] == -1;
			}
		}
		check(src, "dataset: sorgente auto-gioco");
		check(vals, "dataset: ogni posizione ha il valore di ricerca");
		check(plyOk, "dataset: ply >= apertura casuale, chiavi di partita nell'intervallo");
		check(toEndOk, "dataset: toEnd coerente (ply + toEnd = lunghezza della partita)");
		check(outConst, "dataset: tutte le posizioni di una partita hanno lo stesso esito");
		// ogni posizione e' quella di una partita (si e' gia' verificato in verifyGame); qui il dataset non ha posizioni terminali
		boolean nonTerminal = true;
		for (int i = 0; i < d1.n; i++) {
			nonTerminal &= NnueDataset.checkPosition(d1.boards, 81 * i) == null;
		}
		check(nonTerminal, "dataset: nessuna posizione terminale");
		// seme diverso, dataset diverso
		c.seed = 22;
		check(!NnueSelfPlay.generate(ev, ev, c, null).equals(d1), "un altro seme deve dare un altro dataset");
		// salvataggio e ricarica
		final File f = File.createTempFile("selfplay_", ".gz");
		f.deleteOnExit();
		d1.save(f.getPath());
		check(NnueDataset.load(f.getPath()).equals(d1), "il dataset di auto-gioco si salva e si rilegge");
	}

	private static void testConfigErrors() throws Exception {
		final Evaluator ev = new SimpleEvaluator();
		final NnueSelfPlay.Config[] bad = { cfg(0, 2, 0.1, 3, 100, 1), cfg(1, 0, 0.1, 3, 100, 1), cfg(1, 41, 0.1, 3, 100, 1), cfg(1, 2, -0.1, 3, 100, 1), cfg(1, 2, 1.1, 3, 100, 1),
				cfg(1, 2, 0.1, -1, 100, 1), cfg(1, 2, 0.1, 3, 1, 1) };
		for (int i = 0; i < bad.length; i++) {
			boolean threw = false;
			try {
				NnueSelfPlay.generate(ev, ev, bad[i], null);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "configurazione non valida #" + i + " accettata da generate");
		}
		final NnueSelfPlay.Config c1 = cfg(1, 2, 0.1, 3, 100, 1);
		c1.valueScale = 0;
		final NnueSelfPlay.Config c2 = cfg(1, 2, 0.1, 3, 100, 1);
		c2.threads = 0;
		final NnueSelfPlay.Config c3 = cfg(1, 2, 0.1, 3, 100, 1);
		c3.msPerMove = -5;
		for (NnueSelfPlay.Config c : new NnueSelfPlay.Config[] { c1, c2, c3 }) {
			boolean threw = false;
			try {
				NnueSelfPlay.playGame(ev, ev, c, 1);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "scala/thread/ms non validi accettati da playGame");
		}
	}

	// ====================================================================== rietichettatura

	private static NnueDataset datasetOf(List<byte[]> boards, int[] roles, int[] outcomes, boolean[] outValid, double[] values, boolean[] valValid) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		for (int i = 0; i < boards.size(); i++) {
			b.add(boards.get(i), 0, roles[i], outcomes[i], outValid[i], values[i], valValid[i], i, 0, outValid[i] ? 1 : -1, NnueDataset.SRC_LOG);
		}
		return b.build();
	}

	private static State escapeStart() {
		final byte[] b = new byte[81];
		b[2 * 9 + 2] = BoardConstants.KING; // c3: fugge in c1
		b[6 * 9 + 6] = BoardConstants.BLACK;
		b[7 * 9 + 2] = BoardConstants.BLACK;
		b[5 * 9 + 5] = BoardConstants.WHITE;
		b[1 * 9 + 6] = BoardConstants.WHITE;
		return NnueDataImport.toState(b, 0);
	}

	private static State captureStart() {
		final byte[] b = new byte[81];
		b[2 * 9 + 2] = BoardConstants.KING; // c3: il Nero lo cattura con h3 -> d3
		b[2 * 9 + 1] = BoardConstants.BLACK;
		b[2 * 9 + 7] = BoardConstants.BLACK;
		b[6 * 9 + 6] = BoardConstants.WHITE;
		b[6 * 9 + 2] = BoardConstants.WHITE;
		return NnueDataImport.toState(b, 1);
	}

	private static void testRelabel() {
		final Evaluator ev = new SimpleEvaluator();
		final byte[] esc = FastState.fromState(escapeStart()).board.clone(), cap = FastState.fromState(captureStart()).board.clone(), init = new FastState().board.clone();
		final List<byte[]> boards = Arrays.asList(esc, cap, init, init);
		final NnueDataset d = datasetOf(boards, new int[] { 0, 1, 0, 1 }, new int[] { 1, -1, 1, 0 }, new boolean[] { true, true, false, true }, new double[] { 0, 0, 0.25, 0 },
				new boolean[] { false, false, true, false });
		final NnueDataset r = NnueRelabel.relabel(d, ev, 2, 1000, false, null);
		check(r.valValid[0] == 1 && same(r.value[0], 1.0f), "fuga del Re in una mossa (Bianco al tratto): valore +1 (" + r.value[0] + ")");
		check(r.valValid[1] == 1 && same(r.value[1], -1.0f), "cattura del Re in una mossa (Nero al tratto): valore -1 (" + r.value[1] + ")");
		check(r.valValid[2] == 1 && r.valValid[3] == 1 && Math.abs(r.value[2]) < 1.0f && Math.abs(r.value[3]) < 1.0f, "la posizione iniziale non e' decisa: ne' +1 ne' -1 (" + r.value[2] + ", " + r.value[3] + ")");
		// tutto il resto resta com'era
		boolean untouched = Arrays.equals(r.boards, d.boards) && Arrays.equals(r.role, d.role) && Arrays.equals(r.outcome, d.outcome) && Arrays.equals(r.outValid, d.outValid)
				&& Arrays.equals(r.game, d.game) && Arrays.equals(r.ply, d.ply) && Arrays.equals(r.toEnd, d.toEnd) && Arrays.equals(r.source, d.source);
		check(untouched, "la rietichettatura cambia solo value/valValid");
		check(r.equals(NnueRelabel.relabel(d, ev, 2, 1000, false, null)), "la rietichettatura e' deterministica");
		// con la stessa posizione e lato opposto la ricerca puo' dare un altro valore, ma la convenzione resta quella del Bianco: +1 per la fuga anche se la fuga e' del Bianco
		// onlyMissing: tiene l'etichetta esistente (0.25) e cerca solo le altre
		final NnueDataset m = NnueRelabel.relabel(d, ev, 2, 1000, true, null);
		check(same(m.value[2], 0.25f) && m.valValid[2] == 1, "onlyMissing: l'etichetta esistente resta (" + m.value[2] + ")");
		check(same(m.value[0], 1.0f) && same(m.value[1], -1.0f), "onlyMissing: le posizioni senza etichetta vengono cercate");
		check(!same(r.value[2], 0.25f), "senza onlyMissing l'etichetta esistente viene ricalcolata");
		// la scala cambia il valore delle posizioni non decise, non quello delle vittorie forzate
		final NnueDataset big = NnueRelabel.relabel(d, ev, 2, 100000, false, null);
		check(same(big.value[0], 1.0f) && same(big.value[1], -1.0f) && Math.abs(big.value[3]) <= Math.abs(r.value[3]) + 1e-6, "scala grande: le vittorie restano +-1, le altre si schiacciano verso 0");
		for (int[] bad : new int[][] { { 0 }, { 41 } }) {
			boolean threw = false;
			try {
				NnueRelabel.relabel(d, ev, bad[0], 1000, false, null);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "profondita' " + bad[0] + " non valida");
		}
		boolean threw = false;
		try {
			NnueRelabel.relabel(d, ev, 2, 0, false, null);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "scala 0 non valida");
	}

	// ====================================================================== main in un sottoprocesso

	private static int run(String cls, List<String> args, StringBuilder out) throws Exception {
		final List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.add("-Xmx1g");
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.training." + cls);
		cmd.addAll(args);
		final Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		final java.io.ByteArrayOutputStream bo = new java.io.ByteArrayOutputStream();
		final byte[] buf = new byte[8192];
		int n;
		while ((n = p.getInputStream().read(buf)) > 0) {
			bo.write(buf, 0, n);
		}
		out.append(new String(bo.toByteArray(), "UTF-8"));
		if (!p.waitFor(300, TimeUnit.SECONDS)) {
			p.destroyForcibly();
			return -999;
		}
		return p.exitValue();
	}

	private static void testCli() throws Exception {
		final File sp = File.createTempFile("cli_sp_", ".gz"), lab = File.createTempFile("cli_lab_", ".gz"), imp = File.createTempFile("cli_imp_", ".gz");
		sp.deleteOnExit();
		lab.deleteOnExit();
		imp.deleteOnExit();
		StringBuilder o = new StringBuilder();
		int rc = run("NnueSelfPlay", Arrays.asList("eval=simple", "out=" + sp.getPath(), "games=3", "depth=2", "seed=4"), o);
		check(rc == 0 && o.toString().contains("scritto "), "NnueSelfPlay: uscita " + rc + ". Output: " + o);
		final NnueDataset d = NnueDataset.load(sp.getPath());
		check(d.n > 20 && d.valValid[0] == 1, "NnueSelfPlay: il file scritto e' un dataset con valori");
		o = new StringBuilder();
		rc = run("NnueRelabel", Arrays.asList("in=" + sp.getPath(), "out=" + lab.getPath(), "eval=simple", "depth=1"), o);
		check(rc == 0 && NnueDataset.load(lab.getPath()).n == d.n, "NnueRelabel: uscita " + rc + ". Output: " + o);
		// import di una cartella di log sintetici
		final File dir = Files.createTempDirectory("cli_logs_").toFile();
		final File logf = new File(dir, "x_gameLog.txt");
		logf.deleteOnExit();
		dir.deleteOnExit();
		final State s = escapeStart();
		final String text = "BUONO: Stato:\n" + rows(s) + "-\nW\nBUONO: Turn: W Pawn from c3 to c1\nBUONO: Stato:\n" + rows(AshtonRules.applyMove(s, new Action("c3", "c1", Turn.WHITE))) + "-\nWW\n";
		Files.write(logf.toPath(), text.getBytes("ISO-8859-1"));
		o = new StringBuilder();
		rc = run("NnueDataImport", Arrays.asList("logs=" + dir.getPath(), "out=" + imp.getPath()), o);
		check(rc == 0 && NnueDataset.load(imp.getPath()).n == 1, "NnueDataImport: uscita " + rc + ". Output: " + o);
		check(o.toString().contains("1 con esito vero"), "NnueDataImport: stampa il rapporto. Output: " + o);

		// argomenti sbagliati
		final String[][] badSp = { {}, { "eval=simple" }, { "out=x.gz" }, { "eval=simple", "out=" + sp.getPath(), "games=0" }, { "eval=simple", "out=" + sp.getPath(), "depth=abc" },
				{ "eval=simple", "out=" + sp.getPath(), "boh=1" }, { "eval=simple", "out=" + sp.getPath(), "senzauguale" }, { "eval=simple", "out=" + sp.getPath(), "eps=2" } };
		for (String[] b : badSp) {
			o = new StringBuilder();
			rc = run("NnueSelfPlay", Arrays.asList(b), o);
			check(rc == 2, "NnueSelfPlay " + Arrays.toString(b) + ": uscita " + rc + " (attesa 2): " + o);
		}
		o = new StringBuilder();
		rc = run("NnueSelfPlay", Arrays.asList("eval=/nonexistent_dir_xyz/none.nnue", "out=" + sp.getPath(), "games=1"), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueSelfPlay con pesi inesistenti: uscita " + rc + " (attesa 1): " + o);
		final String[][] badRel = { {}, { "in=" + sp.getPath() }, { "in=" + sp.getPath(), "out=" + lab.getPath(), "eval=simple", "depth=0" }, { "in=" + sp.getPath(), "out=" + lab.getPath(), "eval=simple", "scale=0" } };
		for (String[] b : badRel) {
			o = new StringBuilder();
			rc = run("NnueRelabel", Arrays.asList(b), o);
			check(rc == 2, "NnueRelabel " + Arrays.toString(b) + ": uscita " + rc + " (attesa 2): " + o);
		}
		o = new StringBuilder();
		rc = run("NnueRelabel", Arrays.asList("in=/nonexistent_dir_xyz/x.gz", "out=" + lab.getPath(), "eval=simple"), o);
		check(rc == 1, "NnueRelabel con dataset inesistente: uscita " + rc);
		final String[][] badImp = { {}, { "out=" + imp.getPath() }, { "logs=" + dir.getPath() }, { "out=" + imp.getPath(), "logs=" + dir.getPath(), "boh=1" }, { "out=" + imp.getPath(), "logs=" + dir.getPath(), "x" } };
		for (String[] b : badImp) {
			o = new StringBuilder();
			rc = run("NnueDataImport", Arrays.asList(b), o);
			check(rc == 2, "NnueDataImport " + Arrays.toString(b) + ": uscita " + rc + " (attesa 2): " + o);
		}
		o = new StringBuilder();
		rc = run("NnueDataImport", Arrays.asList("out=" + imp.getPath(), "logs=/nonexistent_dir_xyz"), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "NnueDataImport con log inesistenti: uscita " + rc + " (attesa 1): " + o);
	}

	private static String rows(State s) {
		final StringBuilder sb = new StringBuilder();
		final byte[] codes = FastState.fromState(s).board;
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				final byte code = codes[r * 9 + c];
				sb.append(code == BoardConstants.WHITE ? 'W' : code == BoardConstants.BLACK ? 'B' : code == BoardConstants.KING ? 'K' : (r == 4 && c == 4 ? 'T' : 'O'));
			}
			sb.append('\n');
		}
		return sb.toString();
	}
}
