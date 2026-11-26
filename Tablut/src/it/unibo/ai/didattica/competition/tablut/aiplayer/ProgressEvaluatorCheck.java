package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/** Test di {@link ProgressEvaluator}: neutro fuori dalla saturazione, premia il materiale solo dentro, simmetrico, limitato, coerente con le sessioni. Uso: {@code ProgressEvaluatorCheck}. */
public class ProgressEvaluatorCheck {

	private static int failures = 0, checks = 0;

	private static void check(boolean c, String m) {
		checks++;
		if (!c) {
			failures++;
			System.out.println("FAIL - " + m);
		}
	}

	/** Valutatore finto: valore costante (prospettiva del Bianco). */
	private static final class Const implements Evaluator {
		final double v;

		Const(double v) {
			this.v = v;
		}

		@Override
		public double evaluate(FastState s) {
			return v;
		}
	}

	private static FastState withCounts(int white, int black) {
		final FastState s = new FastState();
		s.whiteCount = (byte) white;
		s.blackCount = (byte) black;
		return s;
	}

	public static void main(String[] args) {
		final FastState full = withCounts(8, 16);
		// fuori dalla saturazione: identico, qualunque sia il materiale
		for (double v : new double[] { 0, 100, -350, 699.9, -700, 700 }) {
			final ProgressEvaluator p = new ProgressEvaluator(new Const(v));
			check(p.evaluate(withCounts(3, 16)) == v && p.evaluate(withCounts(8, 5)) == v, "gate: " + v + " non deve cambiare");
		}
		// Nero in vantaggio (valore negativo per il Bianco): ogni pedina bianca mangiata aumenta il vantaggio del Nero
		final ProgressEvaluator pn = new ProgressEvaluator(new Const(-980));
		final double base = pn.evaluate(full), w7 = pn.evaluate(withCounts(7, 16)), w6 = pn.evaluate(withCounts(6, 16)), b15 = pn.evaluate(withCounts(8, 15));
		check(base == -980, "materiale pari: nessun termine");
		check(w7 < base && w6 < w7, "Nero in vantaggio: una pedina bianca in meno migliora il suo valore (" + base + " -> " + w7 + " -> " + w6 + ")");
		check(Math.abs((w7 - base) + 40 * 1.0) < 1e-9, "una pedina bianca = 40 punti a gate pieno");
		check(b15 > base, "Nero in vantaggio: perdere una pedina propria peggiora il suo valore");
		// simmetria: Bianco in vantaggio, mangia pedine nere
		final ProgressEvaluator pw = new ProgressEvaluator(new Const(980));
		check(pw.evaluate(withCounts(8, 15)) > 980 && pw.evaluate(withCounts(7, 16)) < 980, "Bianco in vantaggio: mangiare nere aiuta, perdere bianche no");
		check(Math.abs((pw.evaluate(withCounts(8, 15)) - 980) - 20) < 1e-9, "una pedina nera = 20 punti");
		// il gate e' graduale
		final ProgressEvaluator mid = new ProgressEvaluator(new Const(-800));
		check(Math.abs((mid.evaluate(withCounts(7, 16)) + 800) + 20) < 1e-9, "a meta' gate (800 su 700..900) il termine e' meta': -20");
		// limite: mai oltre EVAL_LIMIT, NaN lasciato a chi sta sopra
		final ProgressEvaluator big = new ProgressEvaluator(new Const(-(Negamax.EVAL_LIMIT - 1)), 0, 1, 1e9, 0);
		check(Math.abs(big.evaluate(withCounts(0, 16))) <= Negamax.EVAL_LIMIT, "limitato a EVAL_LIMIT");
		check(Double.isNaN(new ProgressEvaluator(new Const(Double.NaN)).evaluate(full)), "NaN passa invariato (lo tratta Negamax)");
		// parametri non validi e specifica
		boolean threw = false;
		try {
			new ProgressEvaluator(new Const(0), 900, 700, 1, 1);
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "gate invertito rifiutato");
		check(ProgressEvaluator.parse("off", new Const(0)) == null && ProgressEvaluator.parse("on", new Const(0)) != null, "parse off/on");
		final ProgressEvaluator q = ProgressEvaluator.parse("600:800:30:10", new Const(0));
		check(q.gateLow == 600 && q.gateHigh == 800 && q.whiteWeight == 30 && q.blackWeight == 10, "parse completo");
		threw = false;
		try {
			ProgressEvaluator.parse("1:2", new Const(0));
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "parse: specifica incompleta rifiutata");

		// sessione incrementale == valutazione stateless, con una NNUE vera sotto (rete casuale: valori bassi, quindi neutra) e con una scalata per saturare
		final Random rnd = new Random(3);
		final NnueEvaluator nn = new NnueEvaluator(NnueNetwork.random(16, 8, 7));
		nn.outputScale = 5000; // porta molte posizioni oltre il gate
		final ProgressEvaluator pe = new ProgressEvaluator(nn);
		final IncrementalEvaluator.Session ses = pe.newSession(8);
		final int[] buf = new int[FastState.MAX_MOVES];
		int gated = 0;
		for (int g = 0; g < 30; g++) {
			final FastState root = FastState.fromState(TestPositions.randomPosition(rnd));
			if (root.winner != FastState.WINNER_NONE) {
				continue;
			}
			ses.reset(root);
			final FastState st = FastState.copyOf(root);
			final UndoInfo u = new UndoInfo();
			for (int ply = 0; ply < 6 && st.winner == FastState.WINNER_NONE; ply++) {
				final int n = st.generateMoves(buf);
				if (n == 0) {
					break;
				}
				final int m = buf[rnd.nextInt(n)];
				st.makeMove(m, u);
				ses.push(ply + 1, m, st, u);
				if (st.winner != FastState.WINNER_NONE) {
					break;
				}
				final double a = ses.evaluate(ply + 1, st), b = pe.evaluate(st);
				check(Double.doubleToLongBits(a) == Double.doubleToLongBits(b), "sessione == stateless");
				if (Math.abs(nn.evaluate(st)) > 700) {
					gated++;
				}
			}
			check(ses.fallbackRebuilds() == 0, "nessuna ricostruzione di ripiego");
		}
		check(gated > 0, "il test deve toccare posizioni dentro il gate (" + gated + ")");
		check(EvaluatorLoader.describe(pe).startsWith("ProgressEvaluator (") && EvaluatorLoader.describe(pe).contains("NnueEvaluator"), "describe del wrapper: " + EvaluatorLoader.describe(pe));
		// Negamax con il wrapper: cerca, non ricostruisce e restituisce una mossa legale
		for (int i = 0; i < 6; i++) {
			final FastState root = FastState.fromState(TestPositions.randomPosition(rnd));
			if (root.winner != FastState.WINNER_NONE || root.generateMoves(buf) == 0) {
				continue;
			}
			final Negamax ng = new Negamax(pe, 3, true, 1 << 14, true, Negamax.ORDER_DEFAULT, 1);
			final int m = ng.findBestMove(FastState.copyOf(root), 100_000_000L, PositionHistory.empty());
			check(m != 0 && ng.incrementalFallbacksForTest() == 0, "Negamax con ProgressEvaluator");
		}
		System.out.println(failures == 0 ? "=== PROGRESS EVALUATOR CHECK OK: " + checks + " verifiche ===" : "=== FALLITO: " + failures + " ===");
		System.exit(failures == 0 ? 0 : 1);
	}
}
