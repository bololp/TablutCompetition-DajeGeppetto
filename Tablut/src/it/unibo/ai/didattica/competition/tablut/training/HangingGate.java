package it.unibo.ai.didattica.competition.tablut.training;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Porta (ii) dell'adozione: la diagnostica "il Bianco si appende" ({@code analysis_v4/HangingDiag.java}, README_FEATURES_V4 §7) come
 * strumento riusabile e confrontabile fra due reti. Un porting minimale: SOLO le due misure della porta, con le stesse definizioni.
 *
 * <h3>Cosa si misura (una rete gioca contro se stessa, entrambi i lati)</h3>
 * Si registra ogni mossa BIANCA di PEDINA (non del Re) dal ply 4 in poi (i primi 4 sono un'apertura casuale, identica per le due reti
 * a parita' di {@code seed}), che non termina la partita:
 * <ul>
 * <li><b>quota "appesa"</b> ({@link Stats#hangAnyShare}): dopo la mossa il Nero puo' catturare QUALUNQUE pedina bianca con una sua mossa
 * (e' la "quota di mosse che lasciano pezzi catturabili"). Si riporta anche la sottoquota in cui e' catturabile proprio la pedina
 * appena mossa ({@link Stats#hungMovedShare}, la definizione della tabella del §7).</li>
 * <li><b>materiale perso nei 3 ply successivi</b> ({@link Stats#whiteLoss3}): pedine bianche in meno fra l'inizio del ply della mossa e
 * 3 ply dopo (mossa bianca, risposta nera, mossa bianca), media per mossa di pedina. Si riporta anche quello nero, solo informativo.</li>
 * </ul>
 * Come in HangingDiag la cattura del Re non conta come "pezzo catturabile" (e' una sconfitta, non materiale).
 *
 * <h3>Regola di decisione ({@link #compare})</h3>
 * "Non deve peggiorare": la candidata passa se ENTRAMBE le stime puntuali (quota appesa e materiale perso) sono &lt;= quelle della rete di
 * partenza piu' una tolleranza {@code tol} (default 0). La verifica e' dichiarata NON ESEGUITA se le mosse registrate sono meno di
 * {@code minPawnMoves} per una delle due reti. Sono stime su partite diverse (dopo l'apertura le due reti giocano mosse diverse) e con
 * campioni piccoli il rumore e' grande: si stampano gli intervalli; un "passa" con intervalli che si sovrappongono NON dimostra un miglioramento,
 * solo l'assenza di un peggioramento rilevabile.
 */
public final class HangingGate {

	private HangingGate() {
	}

	/** Contatori di una serie di partite. */
	public static final class Stats {
		public int games, pawnMoves, hangAny, hungMoved, events3;
		public double whiteLost3, blackLost3;

		public double hangAnyShare() {
			return pawnMoves == 0 ? 0 : (double) hangAny / pawnMoves;
		}

		public double hungMovedShare() {
			return pawnMoves == 0 ? 0 : (double) hungMoved / pawnMoves;
		}

		public double whiteLoss3() {
			return events3 == 0 ? 0 : whiteLost3 / events3;
		}

		public double blackLoss3() {
			return events3 == 0 ? 0 : blackLost3 / events3;
		}

		void add(Stats o) {
			games += o.games;
			pawnMoves += o.pawnMoves;
			hangAny += o.hangAny;
			hungMoved += o.hungMoved;
			events3 += o.events3;
			whiteLost3 += o.whiteLost3;
			blackLost3 += o.blackLost3;
		}

		@Override
		public String toString() {
			double[] w = wilson(hangAny, pawnMoves);
			return String.format("%d partite, %d mosse di pedina | pezzi catturabili dopo la mossa: %.1f%% [IC95 %.1f-%.1f] (pedina mossa: %.1f%%) | pedine bianche perse nei 3 ply: %.3f (nere: %.3f, n=%d)",
					games, pawnMoves, 100 * hangAnyShare(), 100 * w[0], 100 * w[1], 100 * hungMovedShare(), whiteLoss3(), blackLoss3(), events3);
		}
	}

	/** Intervallo di Wilson al 95% per {@code k} successi su {@code n}. */
	public static double[] wilson(int k, int n) {
		if (n == 0) {
			return new double[] { 0, 1 };
		}
		double p = (double) k / n, z = 1.96, den = 1 + z * z / n;
		double centre = (p + z * z / (2 * n)) / den;
		double half = z * Math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / den;
		return new double[] { centre - half, centre + half };
	}

	// ====================================================================== primitive (le stesse di HangingDiag)

	static int count(State s, Pawn p) {
		int n = 0;
		for (Pawn[] r : s.getBoard()) {
			for (Pawn x : r) {
				if (x == p) {
					n++;
				}
			}
		}
		return n;
	}

	/**
	 * Il Nero (a muovere in {@code s1}) puo' catturare? Ritorna {pedine bianche perse al massimo con UNA mossa nera,
	 * 1 se la pedina appena mossa in (dr,dc) risulta catturata}. La cattura del Re e' ignorata.
	 */
	static int[] blackBest(State s1, int dr, int dc) {
		int whiteBefore = count(s1, Pawn.WHITE);
		int maxLoss = 0, movedLost = 0;
		for (Action m : AshtonRules.generateMoves(s1)) {
			State s2 = AshtonRules.applyMove(s1, m);
			if (s2.getTurn() == Turn.BLACKWIN) {
				continue;
			}
			int loss = whiteBefore - count(s2, Pawn.WHITE);
			if (loss > maxLoss) {
				maxLoss = loss;
			}
			if (loss > 0 && dr >= 0 && s2.getBoard()[dr][dc] == Pawn.EMPTY) {
				movedLost = 1;
			}
		}
		return new int[] { maxLoss, movedLost };
	}

	/** (riga, colonna) della casella di arrivo di {@code a}. */
	static int[] cellTo(Action a) {
		String t = a.getTo();
		return new int[] { t.charAt(1) - '1', t.charAt(0) - 'a' };
	}

	static int[] cellFrom(Action a) {
		String t = a.getFrom();
		return new int[] { t.charAt(1) - '1', t.charAt(0) - 'a' };
	}

	// ====================================================================== una partita

	/**
	 * Una partita di {@code ev} contro se stessa. {@code depth > 0}: ricerca a profondita' fissa (deterministica); altrimenti {@code ms} ms REALI
	 * per mossa (margine di sicurezza 0).
	 */
	static Stats playGame(Evaluator ev, long ms, int depth, long gameSeed) {
		Random rnd = new Random(gameSeed);
		int maxDepth = depth > 0 ? depth : 64;
		long budget = depth > 0 ? 600000L : ms;
		Negamax ng = new Negamax(ev, maxDepth, true, 1 << 18, true, Negamax.ORDER_DEFAULT, 1, false);
		ng.setSafetyMarginMillis(0);
		Stats st = new Stats();
		st.games = 1;
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		GameTracker game = new GameTracker(s);
		List<int[]> hist = new ArrayList<>(); // per ply: {bianche, nere} PRIMA della mossa
		List<Integer> events = new ArrayList<>();
		for (int ply = 0; ply < 140 && !AshtonRules.isTerminal(s); ply++) {
			hist.add(new int[] { count(s, Pawn.WHITE), count(s, Pawn.BLACK) });
			List<Action> legal = AshtonRules.generateMoves(s);
			if (legal.isEmpty()) {
				break;
			}
			Action mv;
			if (ply < 4) {
				mv = legal.get(rnd.nextInt(legal.size()));
			} else {
				mv = ng.findBestMove(s, budget, game.history());
				if (mv == null) {
					break;
				}
			}
			if (s.getTurn() == Turn.WHITE && ply >= 4) {
				int[] fr = cellFrom(mv);
				boolean isKing = s.getBoard()[fr[0]][fr[1]] == Pawn.KING;
				State s1 = AshtonRules.applyMove(s, mv);
				if (!isKing && !AshtonRules.isTerminal(s1)) {
					int[] to = cellTo(mv);
					int[] bb = blackBest(s1, to[0], to[1]);
					st.pawnMoves++;
					if (bb[0] > 0) {
						st.hangAny++;
					}
					if (bb[1] == 1) {
						st.hungMoved++;
					}
					events.add(ply);
				}
			}
			s = game.apply(s, mv);
			if (game.isRepetitionDraw()) {
				break;
			}
		}
		hist.add(new int[] { count(s, Pawn.WHITE), count(s, Pawn.BLACK) });
		for (int p : events) {
			if (p + 3 >= hist.size()) {
				continue; // la partita e' finita prima: i 3 ply non esistono
			}
			st.events3++;
			st.whiteLost3 += hist.get(p)[0] - hist.get(p + 3)[0];
			st.blackLost3 += hist.get(p)[1] - hist.get(p + 3)[1];
		}
		return st;
	}

	/** Gioca {@code games} partite di {@code ev} (aperture determinate da {@code seed}: stesse per ogni rete) e somma i contatori. */
	public static Stats measure(final Evaluator ev, int games, final long ms, final int depth, long seed, int threads, PrintStream log) throws Exception {
		ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
		Stats tot = new Stats();
		try {
			Random master = new Random(seed);
			List<Future<Stats>> fs = new ArrayList<>();
			for (int g = 0; g < games; g++) {
				final long gs = master.nextLong();
				fs.add(pool.submit(new Callable<Stats>() {
					@Override
					public Stats call() {
						return playGame(ev, ms, depth, gs);
					}
				}));
			}
			for (int g = 0; g < fs.size(); g++) {
				tot.add(fs.get(g).get());
				if (log != null && (g + 1) % 10 == 0) {
					log.println("  hanging: " + (g + 1) + "/" + games + " partite");
				}
			}
		} finally {
			pool.shutdown();
		}
		return tot;
	}

	// ====================================================================== decisione

	/** Esito della porta. */
	public static final class Verdict {
		/** false = non eseguibile (campione insufficiente): la porta NON e' superata. */
		public final boolean verified;
		public final boolean pass;
		public final String text;

		Verdict(boolean verified, boolean pass, String text) {
			this.verified = verified;
			this.pass = pass;
			this.text = text;
		}
	}

	/**
	 * La candidata non deve peggiorare ne' la quota di mosse che lasciano pezzi catturabili ne' il materiale bianco perso nei 3 ply.
	 *
	 * @param minPawnMoves mosse di pedina minime registrate per ciascuna rete, sotto le quali la porta e' NON VERIFICATA
	 * @param tol          tolleranza assoluta sulle due stime (quota in [0,1]; materiale in pedine): passa se cand &lt;= base + tol
	 */
	public static Verdict compare(Stats base, Stats cand, int minPawnMoves, double tol) {
		if (base.pawnMoves < minPawnMoves || cand.pawnMoves < minPawnMoves || base.events3 == 0 || cand.events3 == 0) {
			return new Verdict(false, false, "NON VERIFICATA: mosse di pedina registrate " + base.pawnMoves + " (partenza) / " + cand.pawnMoves
					+ " (candidata), minimo " + minPawnMoves);
		}
		boolean shareOk = cand.hangAnyShare() <= base.hangAnyShare() + tol;
		boolean lossOk = cand.whiteLoss3() <= base.whiteLoss3() + tol;
		String t = String.format("quota appesa %.1f%% -> %.1f%% (%s) | materiale bianco perso in 3 ply %.3f -> %.3f (%s)", 100 * base.hangAnyShare(),
				100 * cand.hangAnyShare(), shareOk ? "non peggiora" : "PEGGIORA", base.whiteLoss3(), cand.whiteLoss3(), lossOk ? "non peggiora" : "PEGGIORA");
		return new Verdict(true, shareOk && lossOk, t);
	}
}
