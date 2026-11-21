package it.unibo.ai.didattica.competition.tablut.aiplayer;

import java.io.File;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.engine.PositionHistory;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
import it.unibo.ai.didattica.competition.tablut.training.NnueSelfPlay;

/**
 * Verifica le tecniche di riduzione dell'albero di {@link Negamax} ({@link PruningOptions}: mossa nulla, LMR, finestre di aspirazione). Idea guida: queste
 * tecniche sono EURISTICHE (a differenza di PVS, ordinamento e TT, che non cambiano il valore), quindi non si puo' chiedere l'uguaglianza con la
 * ricerca piena; si chiedono invece (A) le <b>equivalenze</b> che DEVONO valere, (B) la <b>solidita'</b>: nessuna vittoria/sconfitta affermata che la ricerca
 * piena non confermi, (C) l'<b>integrita'</b> di stato, TT e cache, (D) i <b>vincoli</b> di tempo e di quiete.
 * <ol>
 * <li>{@link PruningOptions}: valori di default, parsing/andata e ritorno, rifiuto di OGNI parametro non valido.</li>
 * <li>Primitive della mossa nulla ({@code FastState.makeNullMove}) contro un ORACOLO indipendente (la stessa posizione col turno invertito costruita dal
 * {@code State} di dominio): ripristino esatto, chiavi e hash identici, valutazione identica; {@code NnueSession.pushNull} in passeggiate casuali con mosse
 * vere e nulle mescolate = valutazione completa, senza ricostruzioni di ripiego.</li>
 * <li>Equivalenze: ogni tecnica con parametri che la disattivano di fatto (profondita' minima irraggiungibile, nessuna mossa riducibile, finestra enorme) da' lo
 * STESSO risultato, nodi inclusi, della ricerca storica; l'aspirazione, anche con la finestra piu' stretta e con TT accesa o spenta, da' lo stesso valore e la
 * stessa mossa della finestra piena (e i test provano che fail-low e fail-high avvengono davvero).</li>
 * <li>Solidita' su posizioni di FINALE di partite decise: con ogni tecnica nessuna vittoria/sconfitta forzata affermata e' falsa o di segno opposto, e quelle
 * forzate trovate dalla ricerca piena sono trovate anche da quella ridotta.</li>
 * <li>Convergenza: dopo che una ricerca ridotta trova una vittoria le ricerche successive sono SENZA riduzioni (sospensione) e la partita termina con la
 * vittoria entro il numero di semi-mosse annunciato.</li>
 * <li>Integrita': la ricerca ridotta con la valutazione incrementale (con mosse nulle) e' IDENTICA a quella con la valutazione completa e con le cache di
 * valutazione; il taglio da mossa nulla NON scrive il nodo in TT (TT "registratore"), la storia di ripetizione non e' toccata, il figlio nullo non e' una posizione di storia.</li>
 * <li>Funzione "forzante" su posizioni costruite a mano; vincoli di tempo con 4 thread, propagazione della configurazione agli helper Lazy SMP e quiete subito dopo la mossa.</li>
 * </ol>
 * Uso: {@code PruningCheck [pesi_nnue/trained_v6.nnue] [sezioni]} con sezioni = elenco di options,primitives,session,equiv,sound,conv,integrity,tt,forcing,guards,alloc,smp,eff (default: tutte). Se il file manca si usa una rete NNUE casuale 64x16: restano valide le verifiche di solidita', non quelle sulla riduzione).
 */
public class PruningCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String what) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + what);
		}
	}

	static final double WIN = Negamax.WIN_SCORE;

	static Negamax mk(Evaluator ev, int depth, boolean tt, PruningOptions p) {
		final Negamax n = new Negamax(ev, depth, tt, tt ? (1 << 18) : 0, true, Negamax.ORDER_RECOMMENDED, 1);
		n.setSafetyMarginMillis(0);
		n.setPruning(p);
		return n;
	}

	static FastState fs(State s) {
		return FastState.fromState(s);
	}

	static boolean sameMove(Action a, Action b) {
		return a != null && b != null && a.getFrom().equalsIgnoreCase(b.getFrom()) && a.getTo().equalsIgnoreCase(b.getTo());
	}

	/** Un evaluatore che NON e' incrementale: ogni nodo si valuta da zero (serve a verificare la sessione incrementale). */
	static final class FullOnly implements Evaluator {
		final NnueEvaluator in;

		FullOnly(NnueEvaluator in) {
			this.in = in;
		}

		@Override
		public double evaluate(FastState s) {
			return in.evaluate(s);
		}
	}

	/** TT che registra gli hash di ogni {@code store}: prova quali nodi la ricerca scrive. */
	static final class RecordingTT extends TranspositionTable {
		final Set<Long> keys = new HashSet<>();

		RecordingTT(int n) {
			super(n);
		}

		@Override
		public void store(FastState s, int depth, double score, byte flag, int packedMove) {
			keys.add(s.zobristHash);
			super.store(s, depth, score, flag, packedMove);
		}

		@Override
		public void store(long hash, int depth, double score, byte flag, int packedMove) {
			keys.add(hash);
			super.store(hash, depth, score, flag, packedMove);
		}
	}

	// ====================================================================== fixture

	static NnueSelfPlay.Config spCfg() {
		final NnueSelfPlay.Config c = new NnueSelfPlay.Config();
		c.depth = 3;
		c.eps = 0.08;
		c.openRandom = 4;
		c.maxPlies = 120;
		c.games = 1;
		c.threads = 1;
		return c;
	}

	/** Posizioni di meta' partita da auto-gioco (non partite casuali: giocate con la rete). */
	static List<State> midgame(NnueEvaluator ev, int games, int every, long seed) {
		final List<State> out = new ArrayList<>();
		for (int g = 0; g < games; g++) {
			final NnueSelfPlay.Game game = NnueSelfPlay.playGame(ev, ev, spCfg(), seed + g);
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			int ply = 0;
			for (Action a : game.moves) {
				if (ply >= 8 && ply % every == 0) {
					out.add(s);
				}
				s = it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules.applyMove(s, a);
				if (AshtonRules.isTerminal(s)) {
					break;
				}
				ply++;
			}
		}
		return out;
	}

	/** Gli ultimi {@code lastK} semi-turni delle partite DECISE: abbondano vittorie e sconfitte forzate. */
	static List<State> endgames(NnueEvaluator ev, int games, int lastK, long seed) {
		final List<State> out = new ArrayList<>();
		for (int g = 0; g < games; g++) {
			final NnueSelfPlay.Game game = NnueSelfPlay.playGame(ev, ev, spCfg(), seed + g);
			if (game.result.outcome == 0) {
				continue;
			}
			final List<State> all = new ArrayList<>();
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			for (Action a : game.moves) {
				all.add(s);
				s = AshtonRules.applyMove(s, a);
				if (AshtonRules.isTerminal(s)) {
					break;
				}
			}
			for (int k = 1; k <= lastK && all.size() - k >= 0; k++) {
				out.add(all.get(all.size() - k));
			}
		}
		return out;
	}

	// ====================================================================== 1. PruningOptions

	static void options() {
		final PruningOptions off = PruningOptions.OFF;
		check(!off.any() && !off.nullMove && !off.lmr && !off.aspiration, "OFF: tutte le tecniche spente");
		check(PruningOptions.parse("off").equals(off) && "off".equals(off.toString()), "parse('off') = OFF");
		final PruningOptions all = PruningOptions.parse("all");
		check(all.nullMove && all.lmr && all.aspiration && all.any(), "parse('all'): tre tecniche accese");
		check(PruningOptions.parse("nmp").nullMove && !PruningOptions.parse("nmp").lmr && !PruningOptions.parse("nmp").aspiration, "parse('nmp'): solo mossa nulla");
		check(PruningOptions.parse("LMR").lmr && PruningOptions.parse(" asp ").aspiration, "i token sono case-insensitive e ripuliti dagli spazi");
		check(all.nullR == 2 && all.nullMinDepth == 4 && all.nullMinPieces == 5 && all.nullVerifyDepth == 10 && all.lmrMinDepth == 3 && all.lmrMinMoves == 4 && all.aspMinDepth == 5 && all.aspDelta == 40,
				"valori di default dei parametri");
		final PruningOptions custom = PruningOptions.parse("nmp,lmr,asp,nmp.r=3,nmp.depth=6,nmp.pieces=7,nmp.verify=12,lmr.depth=4,lmr.moves=6,asp.depth=6,asp.delta=70");
		check(custom.nullR == 3 && custom.nullMinDepth == 6 && custom.nullMinPieces == 7 && custom.nullVerifyDepth == 12 && custom.lmrMinDepth == 4 && custom.lmrMinMoves == 6 && custom.aspMinDepth == 6
				&& custom.aspDelta == 70, "parametri personalizzati");
		for (String spec : new String[] { "off", "all", "nmp", "lmr", "asp", "nmp,lmr", "nmp,lmr,asp,nmp.r=3,nmp.depth=6,nmp.pieces=7,nmp.verify=12,lmr.depth=4,lmr.moves=6,asp.depth=6,asp.delta=70", "nmp,nmp.verify=0" }) {
			final PruningOptions p = PruningOptions.parse(spec);
			check(PruningOptions.parse(p.toString()).equals(p) && PruningOptions.parse(p.toString()).hashCode() == p.hashCode(), "andata e ritorno di toString: " + spec + " -> " + p);
		}
		for (String bad : new String[] { "", "   ", "x", "nmp,,lmr", "nmp.r=0", "nmp.r=7", "nmp.r=3", "nmp.depth=3", "nmp.depth=41", "nmp.pieces=-1", "nmp.pieces=17", "nmp.verify=3", "nmp.verify=-1", "lmr.depth=2",
				"lmr.moves=1", "lmr.moves=65", "asp.depth=1", "asp.delta=0", "asp.delta=100001", "foo=3", "nmp.r=abc", "nmp.r=", "nmp.r=2.5" }) {
			try {
				PruningOptions.parse(bad);
				check(false, "parametro non valido accettato: '" + bad + "'");
			} catch (IllegalArgumentException e) {
				check(true, "rifiutato: '" + bad + "'");
			}
		}
		try {
			PruningOptions.parse(null);
			check(false, "null accettato");
		} catch (IllegalArgumentException e) {
			check(true, "null rifiutato");
		}
	}

	// ====================================================================== 2. mossa nulla: primitive

	static void nullMovePrimitives(NnueEvaluator ev, List<State> pool) {
		int done = 0, flipOk = 0;
		for (State st : pool) {
			final FastState f = fs(st);
			if (f.winner != FastState.WINNER_NONE) {
				continue;
			}
			final byte[] board0 = f.board.clone();
			final byte turn0 = f.turn, king0 = f.kingSquare, winner0 = f.winner;
			final int w0 = f.whiteCount, b0 = f.blackCount;
			final long h0 = f.zobristHash, k00 = f.key0, k10 = f.key1, k20 = f.key2;
			final UndoInfo u = new UndoInfo();
			f.makeNullMove(u);
			// oracolo: la STESSA posizione col turno invertito costruita dal modello di dominio
			final State flipped = st.clone();
			flipped.setTurn(turn0 == 0 ? Turn.BLACK : Turn.WHITE);
			final FastState o = fs(flipped);
			final boolean keys = f.zobristHash == o.zobristHash && f.key0 == o.key0 && f.key1 == o.key1 && f.key2 == o.key2 && f.turn == o.turn;
			final boolean same = Arrays.equals(f.board, board0) && f.kingSquare == king0 && f.winner == winner0 && f.whiteCount == w0 && f.blackCount == b0;
			check(keys, "mossa nulla: hash/chiavi/turno uguali a quelli della posizione col turno invertito (oracolo)");
			check(same, "mossa nulla: scacchiera, Re, conteggi e vincitore invariati");
			check(f.turn != turn0 && f.zobristHash != h0, "mossa nulla: il turno cambia e l'hash cambia");
			check(Double.doubleToLongBits(ev.evaluate(f)) == Double.doubleToLongBits(ev.evaluate(o)), "mossa nulla: valutazione identica a quella dell'oracolo");
			// doppia mossa nulla = identita'
			final UndoInfo u2 = new UndoInfo();
			f.makeNullMove(u2);
			check(f.zobristHash == h0 && f.key0 == k00 && f.key1 == k10 && f.key2 == k20 && f.turn == turn0, "due mosse nulle consecutive riportano alle chiavi di partenza");
			f.unmakeNullMove(u2);
			f.unmakeNullMove(u);
			check(Arrays.equals(f.board, board0) && f.turn == turn0 && f.kingSquare == king0 && f.winner == winner0 && f.whiteCount == w0 && f.blackCount == b0 && f.zobristHash == h0 && f.key0 == k00
					&& f.key1 == k10 && f.key2 == k20, "annullamento esatto dello stato completo");
			flipOk += (keys && same) ? 1 : 0;
			done++;
		}
		check(done >= 100 && flipOk == done, "mossa nulla verificata su " + done + " posizioni (" + flipOk + " perfette)");
	}

	static void sessionWalks(NnueEvaluator ev, List<State> pool) {
		final Random rnd = new Random(5);
		int evaluations = 0, nulls = 0;
		boolean allEqual = true;
		for (int round = 0; round < 60; round++) {
			final State st = pool.get(rnd.nextInt(pool.size()));
			final FastState cur = fs(st);
			if (cur.winner != FastState.WINNER_NONE) {
				continue;
			}
			final IncrementalEvaluator.Session ses = ev.newSession(32);
			ses.reset(cur);
			final UndoInfo[] us = new UndoInfo[16];
			final int[] mv = new int[16];
			final boolean[] isNull = new boolean[16];
			for (int i = 0; i < us.length; i++) {
				us[i] = new UndoInfo();
			}
			int ply = 0;
			final int[] buf = new int[300];
			for (int step = 0; step < 14; step++) {
				// a volte si risale di qualche ply (la sessione sovrascrive i frame piu' profondi)
				if (ply > 2 && rnd.nextInt(5) == 0) {
					final int up = 1 + rnd.nextInt(2);
					for (int k = 0; k < up && ply > 0; k++) {
						ply--;
						if (isNull[ply]) {
							cur.unmakeNullMove(us[ply]);
						} else {
							cur.unmakeMove(mv[ply], us[ply]);
						}
					}
				}
				if (ply >= us.length - 1 || cur.winner != FastState.WINNER_NONE) {
					break;
				}
				final boolean doNull = rnd.nextInt(3) == 0;
				if (doNull) {
					cur.makeNullMove(us[ply]);
					isNull[ply] = true;
					check(ses.pushNull(ply + 1, cur), "NnueSession.pushNull deve essere supportata");
					nulls++;
				} else {
					final int n = cur.generateMoves(buf);
					if (n == 0) {
						break;
					}
					mv[ply] = buf[rnd.nextInt(n)];
					cur.makeMove(mv[ply], us[ply]);
					isNull[ply] = false;
					ses.push(ply + 1, mv[ply], cur, us[ply]);
				}
				ply++;
				if (cur.winner == FastState.WINNER_NONE) {
					final double inc = ses.evaluate(ply, cur), full = ev.evaluate(cur);
					allEqual &= Double.doubleToLongBits(inc) == Double.doubleToLongBits(full);
					evaluations++;
				}
			}
			check(ses.fallbackRebuilds() == 0, "passeggiata " + round + ": nessuna ricostruzione di ripiego (" + ses.fallbackRebuilds() + ")");
		}
		check(allEqual, "valutazione incrementale (con mosse nulle e vere mescolate) = valutazione completa, su " + evaluations + " nodi");
		check(evaluations > 300 && nulls > 100, "copertura delle passeggiate: " + evaluations + " valutazioni, " + nulls + " mosse nulle");
		try {
			ev.newSession(8).pushNull(0, fs(pool.get(0)));
			check(false, "pushNull(0) doveva essere rifiutato (0 e' la radice)");
		} catch (IllegalArgumentException e) {
			check(true, "pushNull sulla radice rifiutato");
		}
	}

	// ====================================================================== 3-4. equivalenze

	static void equivalences(NnueEvaluator ev, List<State> mid) {
		final String[] degenerate = { "nmp,nmp.depth=40,nmp.verify=0", "lmr,lmr.depth=40", "asp,asp.depth=40", "asp,asp.delta=100000", "nmp,lmr,asp,nmp.depth=40,nmp.verify=0,lmr.depth=40,asp.depth=40" };
		int compared = 0;
		for (State st : mid.subList(0, Math.min(10, mid.size()))) {
			final Negamax plain = mk(ev, 5, true, PruningOptions.OFF);
			final int mp = plain.findBestMove(fs(st), 600000);
			for (String spec : degenerate) {
				final Negamax q = mk(ev, 5, true, PruningOptions.parse(spec));
				final int mq = q.findBestMove(fs(st), 600000);
				final boolean eq = mp == mq && plain.lastScore == q.lastScore && plain.lastNodeCount == q.lastNodeCount && plain.lastDepthReached == q.lastDepthReached;
				check(eq, "[" + spec + "] con parametri che la disattivano = ricerca storica, nodi inclusi (" + plain.lastNodeCount + " vs " + q.lastNodeCount + ", " + plain.lastScore + " vs " + q.lastScore + ")");
				compared++;
			}
		}
		check(compared >= 50, "equivalenze degeneri verificate: " + compared);

		// aspirazione: stessa mossa e stesso valore della finestra piena (anche con la finestra piu' stretta)
		long failLow = 0, failHigh = 0;
		int tested = 0;
		for (boolean tt : new boolean[] { false, true }) {
			for (State st : mid.subList(0, Math.min(12, mid.size()))) {
				final Negamax plain = mk(ev, 5, tt, PruningOptions.OFF), q = mk(ev, 5, tt, PruningOptions.parse("asp,asp.depth=2,asp.delta=1"));
				final int mp = plain.findBestMove(fs(st), 600000), mq = q.findBestMove(fs(st), 600000);
				check(mp == mq && plain.lastScore == q.lastScore, "aspirazione (TT " + (tt ? "accesa" : "spenta") + ", finestra minima) = finestra piena: " + Move.toString(mp) + " " + plain.lastScore + " vs " + Move.toString(mq) + " " + q.lastScore);
				failLow += q.lastAspFailLow;
				failHigh += q.lastAspFailHigh;
				tested++;
			}
		}
		check(tested >= 20 && failLow > 0 && failHigh > 0, "il test dell'aspirazione ha i denti: " + failLow + " fail-low e " + failHigh + " fail-high su " + tested + " ricerche");
		// l'aspirazione con la finestra di default non fallisce piu' di tanto e non cambia il risultato
		int same = 0;
		for (State st : mid.subList(0, Math.min(12, mid.size()))) {
			final Negamax plain = mk(ev, 6, true, PruningOptions.OFF), q = mk(ev, 6, true, PruningOptions.parse("asp"));
			same += plain.findBestMove(fs(st), 600000) == q.findBestMove(fs(st), 600000) && plain.lastScore == q.lastScore ? 1 : 0;
		}
		check(same == Math.min(12, mid.size()), "aspirazione di default a profondita' 6: stesso risultato della finestra piena (" + same + ")");
	}

	// ====================================================================== 5. solidita' tattica

	static void soundness(NnueEvaluator ev, List<State> end, boolean strictDetection) {
		final int n = end.size();
		final double[] truth = new double[n];
		final int[] tmove = new int[n];
		int forced = 0;
		for (int i = 0; i < n; i++) {
			final Negamax p = mk(ev, 5, true, PruningOptions.OFF);
			tmove[i] = p.findBestMove(fs(end.get(i)), 600000);
			truth[i] = p.lastScore;
			forced += Math.abs(truth[i]) >= WIN ? 1 : 0;
		}
		check(n >= 80 && forced >= 20, "suite di finale: " + n + " posizioni, " + forced + " con vittoria/sconfitta forzata a profondita' 5");
		for (String spec : new String[] { "asp", "nmp", "lmr", "nmp,lmr", "all" }) {
			int found = 0, missed = 0, wrongSign = 0, falseClaim = 0, sameMove = 0, illegal = 0, sameScore = 0;
			long reductions = 0, reSearches = 0;
			for (int i = 0; i < n; i++) {
				final Negamax q = mk(ev, 5, true, PruningOptions.parse(spec));
				final int m = q.findBestMove(fs(end.get(i)), 600000);
				reductions += q.lastLmrReductions;
				reSearches += q.lastLmrReSearches;
				final boolean tw = Math.abs(truth[i]) >= WIN, qw = Math.abs(q.lastScore) >= WIN;
				if (tw && qw) {
					if (Math.signum(truth[i]) == Math.signum(q.lastScore)) {
						found++;
						sameScore += q.lastScore == truth[i] ? 1 : 0;
					} else {
						wrongSign++;
					}
				} else if (tw) {
					missed++;
				} else if (qw) {
					falseClaim++;
				}
				sameMove += m == tmove[i] ? 1 : 0;
				illegal += isLegal(end.get(i), Move.toAction(m, end.get(i).getTurn())) ? 0 : 1;
			}
			System.out.println("   [" + spec + "] profondita' 5: forzate viste " + found + "/" + forced + ", perse " + missed + ", segno opposto " + wrongSign + ", affermazioni non confermate " + falseClaim
					+ ", stessa mossa della ricerca piena " + sameMove + "/" + n);
			check(wrongSign == 0 && falseClaim == 0, "[" + spec + "] SOLIDITA': nessuna vittoria/sconfitta affermata e' falsa o di segno opposto (" + falseClaim + " false, " + wrongSign + " opposte)");
			check(illegal == 0, "[" + spec + "] tutte le mosse scelte sono legali");
			if (strictDetection) {
				// il punteggio di vittoria/sconfitta codifica la DISTANZA (WIN_SCORE + profondita' residua): uguale a quello della ricerca piena = la vittoria
				// piu' breve (la sconfitta piu' lenta) e' preservata. Una ricerca ridotta con punteggi di vittoria incoerenti (profondita' residua ridotta) lo violerebbe.
				check(sameScore == found, "[" + spec + "] la DISTANZA dalla vittoria/sconfitta e' preservata: stesso punteggio della ricerca piena in " + sameScore + "/" + found);
			}
			if (spec.contains("lmr")) {
				check(reductions > 1000, "[" + spec + "] l'LMR riduce davvero mosse nei finali (" + reductions + ")");
				check(reSearches > 0 && reSearches < reductions, "[" + spec + "] l'LMR ripete a profondita' piena alcune mosse ridotte (" + reSearches + " su " + reductions + ")");
			}
			if (strictDetection) {
				check(missed == 0, "[" + spec + "] nessuna vittoria/sconfitta forzata della ricerca piena e' persa (" + missed + ")");
			}
			check(sameMove >= (int) (0.9 * n), "[" + spec + "] la mossa coincide con quella della ricerca piena in almeno il 90% dei casi (" + sameMove + "/" + n + ")");
			if (spec.equals("asp")) {
				check(sameMove == n && missed == 0, "[asp] l'aspirazione e' ESATTA: tutte le mosse uguali");
			}
		}
	}

	static boolean isLegal(State s, Action a) {
		for (Action m : AshtonRules.generateMoves(s)) {
			if (sameMove(m, a)) {
				return true;
			}
		}
		return false;
	}

	// ====================================================================== 6. convergenza

	static void convergence(NnueEvaluator ev, List<State> end) {
		int played = 0, converged = 0, suspendedChecks = 0, suspendedOk = 0;
		for (State st : end) {
			if (played >= 24) {
				break;
			}
			final Negamax probe = mk(ev, 5, true, PruningOptions.OFF);
			probe.findBestMove(fs(st), 600000);
			if (probe.lastScore < WIN) {
				continue; // serve una vittoria forzata per il lato al tratto
			}
			played++;
			final Turn winnerSide = st.getTurn();
			final Negamax winner = mk(ev, 9, true, PruningOptions.parse("all")), loser = mk(ev, 2, true, PruningOptions.OFF);
			State s = st;
			int plies = 0, claimed = -1;
			boolean winnerKeeps = true;
			while (!AshtonRules.isTerminal(s) && plies < 40) {
				final boolean mine = s.getTurn() == winnerSide;
				final Negamax n = mine ? winner : loser;
				final int m = n.findBestMove(fs(s), 600000);
				if (mine) {
					if (claimed < 0) {
						claimed = winner.lastDepthReached;
						check(winner.lastScore >= WIN, "la ricerca ridotta trova la vittoria forzata che la ricerca piena trova");
					} else {
						winnerKeeps &= winner.lastScore >= WIN;
						suspendedChecks++;
						suspendedOk += winner.lastPruningSuspended && winner.lastNullTries == 0 && winner.lastLmrReductions == 0 && winner.lastAspFailLow == 0 && winner.lastAspFailHigh == 0 ? 1 : 0;
					}
				}
				s = AshtonRules.applyMove(s, Move.toAction(m, s.getTurn()));
				plies++;
			}
			final boolean won = s.getTurn() == (winnerSide == Turn.WHITE ? Turn.WHITEWIN : Turn.BLACKWIN);
			if (won && plies <= claimed && winnerKeeps) {
				converged++;
			} else {
				System.out.println("   NON CONVERGE: vincitore " + winnerSide + ", annunciata " + claimed + " semi-mosse, giocate " + plies + ", esito " + s.getTurn() + ", vittoria mantenuta " + winnerKeeps);
			}
		}
		check(played >= 10, "partite di convergenza giocate: " + played);
		check(converged == played, "la vittoria annunciata arriva entro i semi-turni annunciati in tutte le partite (" + converged + "/" + played + ")");
		check(suspendedChecks > 0 && suspendedOk == suspendedChecks, "dopo la prima vittoria trovata le ricerche successive sono SENZA riduzioni ne' aspirazione (" + suspendedOk + "/" + suspendedChecks + ")");
	}

	// ====================================================================== 7. integrita': incrementale, cache

	static void integrity(NnueEvaluator ev, List<State> mid) {
		final List<State> sub = mid.subList(0, Math.min(8, mid.size()));
		for (String spec : new String[] { "nmp", "lmr", "nmp,lmr", "all" }) {
			int same = 0;
			long nullsSeen = 0, redSeen = 0;
			for (State st : sub) {
				final Negamax inc = mk(ev, 5, true, PruningOptions.parse(spec)), full = mk(new FullOnly(ev), 5, true, PruningOptions.parse(spec));
				final int mi = inc.findBestMove(fs(st), 600000), mf = full.findBestMove(fs(st), 600000);
				same += mi == mf && inc.lastScore == full.lastScore && inc.lastNodeCount == full.lastNodeCount ? 1 : 0;
				nullsSeen += inc.lastNullTries;
				redSeen += inc.lastLmrReductions;
				check(inc.incrementalFallbacksForTest() == 0, "[" + spec + "] la sessione incrementale non ha mai ricostruito di ripiego");
			}
			check(same == sub.size(), "[" + spec + "] ricerca con valutazione INCREMENTALE = con valutazione completa (move/punteggio/nodi), " + same + "/" + sub.size());
			if (spec.contains("nmp")) {
				check(nullsSeen > 0, "[" + spec + "] le mosse nulle sono state davvero tentate (" + nullsSeen + ")");
			}
			if (spec.contains("lmr")) {
				check(redSeen > 0, "[" + spec + "] le riduzioni sono state davvero applicate (" + redSeen + ")");
			}
		}
		// le cache di valutazione (chiave con il turno) sono trasparenti anche con la mossa nulla
		int same = 0;
		for (State st : sub) {
			final Negamax plain = mk(new FullOnly(ev), 5, true, PruningOptions.parse("all"));
			final Negamax cached = mk(new CachingEvaluator(new FullOnly(ev), 16, false, true), 5, true, PruningOptions.parse("all"));
			final Negamax hashed = mk(new HashCachingEvaluator(new FullOnly(ev), 16), 5, true, PruningOptions.parse("all"));
			final int a = plain.findBestMove(fs(st), 600000), b = cached.findBestMove(fs(st), 600000), c = hashed.findBestMove(fs(st), 600000);
			same += a == b && b == c && plain.lastScore == cached.lastScore && cached.lastScore == hashed.lastScore && plain.lastNodeCount == cached.lastNodeCount && cached.lastNodeCount == hashed.lastNodeCount ? 1 : 0;
		}
		check(same == sub.size(), "cache di valutazione (chiave esatta e hash) trasparenti con mossa nulla, LMR e aspirazione accese: " + same + "/" + sub.size());
	}

	// ====================================================================== 8. TT e storia con la mossa nulla

	static void ttAndHistory(NnueEvaluator ev, List<State> mid) {
		int cutsSeen = 0, controlSeen = 0, historyOk = 0, noNullKeyOk = 0;
		for (State st : mid) {
			if (cutsSeen >= 6) {
				break;
			}
			final FastState node = fs(st);
			if (node.winner != FastState.WINNER_NONE || Negamax.forcingForTest(node) || (node.turn == 0 ? node.whiteCount + 1 : node.blackCount) < 5) {
				continue;
			}
			final int persp = node.turn == 0 ? 1 : -1;
			final double beta = persp * ev.evaluate(node) - 300, alpha = beta - Negamax.PVS_EPSILON;
			if (Math.abs(beta) >= 900) {
				continue;
			}
			final State flipped = st.clone();
			flipped.setTurn(node.turn == 0 ? Turn.BLACK : Turn.WHITE);
			final FastState nullPos = fs(flipped);

			// con la mossa nulla: il nodo che taglia NON si scrive in TT; nemmeno il figlio nullo e' una posizione di storia
			final RecordingTT tt = new RecordingTT(1 << 16);
			final Negamax ng = new Negamax(ev, 7, true, 1 << 16, true, Negamax.ORDER_RECOMMENDED, 1, false, tt);
			ng.setSafetyMarginMillis(0);
			ng.setPruning(PruningOptions.parse("nmp"));
			final PositionHistory h = new PositionHistory();
			h.add(nullPos); // il figlio nullo e' GIA' nella storia: se fosse trattato come posizione vera darebbe una (falsa) ripetizione
			h.add(node);
			ng.installForTest(node, h, 7);
			ng.setPruningActiveForTest(true, false);
			final int histLen = ng.historyLengthForTest();
			final long[] keyBefore = { node.zobristHash, node.key0, node.key1, node.key2 };
			final double v = ng.negamax(7, alpha, beta, persp, 0);
			final long[] c = ng.pruningCountersForTest();
			final boolean restored = node.zobristHash == keyBefore[0] && node.key0 == keyBefore[1] && node.key1 == keyBefore[2] && node.key2 == keyBefore[3];
			check(restored, "dopo il nodo con mossa nulla lo stato e' ripristinato");
			if (c[1] == 1) { // taglio per mossa nulla
				cutsSeen++;
				check(v == beta, "il taglio da mossa nulla restituisce beta (un limite), non un valore di vittoria: " + v + " vs " + beta);
				check(!tt.keys.contains(node.zobristHash), "il nodo tagliato dalla mossa nulla NON e' stato scritto in TT");
				noNullKeyOk += !tt.keys.contains(node.zobristHash) ? 1 : 0;
				check(tt.keys.size() > 0, "ma i nodi REALI del sottoalbero nullo si memorizzano normalmente (" + tt.keys.size() + " voci)");
				historyOk += ng.historyLengthForTest() == histLen ? 1 : 0;
				check(ng.historyLengthForTest() == histLen, "la storia di ripetizione non e' stata alterata dalla mossa nulla");
				// controllo: senza mossa nulla lo stesso nodo SI scrive (la registrazione funziona)
				final RecordingTT tt2 = new RecordingTT(1 << 16);
				final Negamax ctl = new Negamax(ev, 7, true, 1 << 16, true, Negamax.ORDER_RECOMMENDED, 1, false, tt2);
				ctl.setSafetyMarginMillis(0);
				ctl.installForTest(node, new PositionHistory(), 7);
				ctl.negamax(5, alpha, beta, persp, 0);
				if (tt2.keys.contains(node.zobristHash)) {
					controlSeen++;
				}
			}
		}
		check(cutsSeen >= 3, "tagli da mossa nulla osservati nel test della TT: " + cutsSeen);
		check(controlSeen == cutsSeen, "controllo: senza mossa nulla gli stessi nodi vengono scritti in TT (" + controlSeen + "/" + cutsSeen + ")");

		// nessun taglio: il nodo si scrive e il sottoalbero nullo non lascia tracce nel valore
		int stored = 0, tried = 0;
		for (State st : mid) {
			if (tried >= 4) {
				break;
			}
			final FastState node = fs(st);
			if (node.winner != FastState.WINNER_NONE || Negamax.forcingForTest(node) || (node.turn == 0 ? node.whiteCount + 1 : node.blackCount) < 5) {
				continue;
			}
			final int persp = node.turn == 0 ? 1 : -1;
			final double beta = persp * ev.evaluate(node) + 1, alpha = beta - Negamax.PVS_EPSILON; // valutazione statica < beta: la mossa nulla non si tenta
			if (Math.abs(beta) >= 900) {
				continue;
			}
			final RecordingTT tt = new RecordingTT(1 << 16);
			final Negamax ng = new Negamax(ev, 6, true, 1 << 16, true, Negamax.ORDER_RECOMMENDED, 1, false, tt);
			ng.setSafetyMarginMillis(0);
			ng.setPruning(PruningOptions.parse("nmp"));
			ng.installForTest(node, new PositionHistory(), 6);
			ng.setPruningActiveForTest(true, false);
			ng.negamax(4, alpha, beta, persp, 0);
			tried++;
			check(ng.pruningCountersForTest()[0] == 0, "valutazione statica sotto beta: nessuna mossa nulla tentata");
			stored += tt.keys.contains(node.zobristHash) ? 1 : 0;
		}
		check(tried >= 3 && stored == tried, "nodo senza taglio nullo: si scrive in TT (" + stored + "/" + tried + ")");
	}

	// ====================================================================== 8b. protezione delle finestre di vittoria

	/**
	 * Con una finestra di vittoria/sconfitta (entrambi i limiti oltre {@code +-WIN_SCORE}, e quindi, ricorsivamente, tutte quelle dei discendenti) NON si ride' ne' si prova la mossa nulla
	 * (la profondita' residua di un punteggio di vittoria deve restare coerente); con una finestra normale si'. Chiamata diretta di {@link Negamax#negamax} su un nodo.
	 */
	static void windowGuards(NnueEvaluator ev, List<State> mid) {
		long normalRed = 0, normalNull = 0, winRed = 0, winNull = 0, lossRed = 0, lossNull = 0;
		int nodes = 0;
		for (State st : mid) {
			if (nodes >= 6) {
				break;
			}
			final FastState node = fs(st);
			if (node.winner != FastState.WINNER_NONE) {
				continue;
			}
			final int persp = node.turn == 0 ? 1 : -1;
			final double e = Negamax.PVS_EPSILON;
			// finestre REALISTICHE: i punteggi sono euristici (|s| <= 1000) o di vittoria (|s| >= WIN), quindi una finestra di vittoria ha ENTRAMBI i limiti oltre WIN e, ricorsivamente,
			// anche le finestre di tutti i discendenti (figli e scout) restano di vittoria
			final double[][] windows = { { -300, -300 + e }, { WIN + 5, WIN + 10 }, { -WIN - 10, -WIN - 5 } };
			for (int w = 0; w < windows.length; w++) {
				final Negamax ng = mk(ev, 6, true, PruningOptions.parse("nmp,lmr,nmp.verify=0"));
				ng.installForTest(node, new PositionHistory(), 6);
				ng.setPruningActiveForTest(true, true);
				ng.negamax(6, windows[w][0], windows[w][1], persp, 0);
				final long[] c = ng.pruningCountersForTest();
				if (w == 0) {
					normalNull += c[0];
					normalRed += c[4];
				} else if (w == 1) {
					winNull += c[0];
					winRed += c[4];
				} else {
					lossNull += c[0];
					lossRed += c[4];
				}
			}
			nodes++;
		}
		check(nodes >= 5, "nodi provati per le finestre di vittoria: " + nodes);
		check(normalRed > 0 && normalNull > 0, "controllo: con una finestra normale riduzioni e mosse nulle scattano (" + normalRed + " riduzioni, " + normalNull + " mosse nulle)");
		check(winRed == 0 && winNull == 0, "finestra di VITTORIA (alpha, beta > WIN): nessuna riduzione e nessuna mossa nulla (" + winRed + ", " + winNull + ")");
		check(lossRed == 0 && lossNull == 0, "finestra di SCONFITTA (alpha, beta < -WIN): nessuna riduzione e nessuna mossa nulla (" + lossRed + ", " + lossNull + ")");
	}

	// ====================================================================== 8c. nessuna allocazione nella ricerca (invariante del progetto, NoAllocationCheck)

	static long allocatedBytes() {
		try {
			final Class<?> c = Class.forName("com.sun.management.ThreadMXBean");
			final Object bean = ManagementFactory.getThreadMXBean();
			if (!c.isInstance(bean)) {
				return -1;
			}
			return ((Number) c.getMethod("getThreadAllocatedBytes", long.class).invoke(bean, Thread.currentThread().getId())).longValue();
		} catch (Exception e) {
			return -1;
		}
	}

	static void noAllocation(NnueEvaluator ev, List<State> mid) {
		if (allocatedBytes() < 0) {
			System.out.println("   (allocazioni per thread non misurabili su questa JVM: prova saltata)");
			return;
		}
		final State st = mid.get(3 % mid.size());
		for (String spec : new String[] { "off", "all" }) {
			final Negamax ng = mk(ev, 6, true, PruningOptions.parse(spec));
			ng.findBestMove(fs(st), 600000); // riscaldamento: classi, JIT, buffer
			final FastState root = fs(st);
			final long b0 = allocatedBytes();
			ng.findBestMove(root, 600000);
			final long bytes = allocatedBytes() - b0;
			final long nodes = ng.lastNodeCount;
			check(bytes < 16 * 1024, "[" + spec + "] la ricerca non alloca: " + bytes + " byte su " + nodes + " nodi (soglia 16 KB, cioe' < 0,01 byte/nodo)");
		}
	}

	// ====================================================================== 9. funzione forzante

	static State emptyBoardWithKing(int kr, int kc, Turn turn) {
		final State s = TestPositions.initial();
		final Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				b[r][c] = Pawn.EMPTY;
			}
		}
		b[kr][kc] = Pawn.KING;
		s.setTurn(turn);
		return s;
	}

	static void forcingFunction() {
		// Nero al tratto: il Re con una via libera verso il bordo e' una minaccia
		State s = emptyBoardWithKing(4, 4, Turn.BLACK);
		check(Negamax.forcingForTest(fs(s)), "Nero al tratto, Re con le quattro vie libere: forzante");
		s = emptyBoardWithKing(4, 4, Turn.BLACK);
		s.getBoard()[3][4] = Pawn.WHITE;
		s.getBoard()[5][4] = Pawn.WHITE;
		s.getBoard()[4][3] = Pawn.WHITE;
		check(Negamax.forcingForTest(fs(s)), "Nero al tratto, tre lati bloccati e uno libero: forzante");
		s.getBoard()[4][5] = Pawn.WHITE;
		check(!Negamax.forcingForTest(fs(s)), "Nero al tratto, Re circondato: NON forzante");
		s = emptyBoardWithKing(4, 4, Turn.BLACK);
		s.getBoard()[3][4] = Pawn.BLACK;
		s.getBoard()[5][4] = Pawn.BLACK;
		s.getBoard()[4][3] = Pawn.BLACK;
		s.getBoard()[4][5] = Pawn.BLACK;
		check(!Negamax.forcingForTest(fs(s)), "Nero al tratto, vie del Re bloccate da pedine nere: NON forzante");
		s = emptyBoardWithKing(4, 4, Turn.BLACK);
		s.getBoard()[1][4] = Pawn.BLACK;
		s.getBoard()[7][4] = Pawn.BLACK;
		s.getBoard()[4][1] = Pawn.BLACK;
		check(Negamax.forcingForTest(fs(s)), "Nero al tratto, una sola via libera lontana dal Re (4,5)->(4,8): forzante");
		// Bianco al tratto: una pedina nera adiacente al Re minaccia la cattura
		s = emptyBoardWithKing(4, 4, Turn.WHITE);
		check(!Negamax.forcingForTest(fs(s)), "Bianco al tratto, nessuna pedina nera vicina al Re: NON forzante (anche con le vie libere)");
		s = emptyBoardWithKing(4, 4, Turn.WHITE);
		s.getBoard()[4][5] = Pawn.BLACK;
		check(Negamax.forcingForTest(fs(s)), "Bianco al tratto, pedina nera adiacente al Re: forzante");
		s = emptyBoardWithKing(0, 4, Turn.WHITE); // Re sul bordo: le direzioni verso l'esterno non hanno celle
		s.getBoard()[0][5] = Pawn.BLACK;
		check(Negamax.forcingForTest(fs(s)), "Re sul bordo con pedina nera adiacente: forzante (nessun accesso fuori scacchiera)");
		s = emptyBoardWithKing(0, 0, Turn.BLACK);
		check(Negamax.forcingForTest(fs(s)), "Re in un angolo: ok, nessuna eccezione");
	}

	// ====================================================================== 10. tempo, Lazy SMP, quiete

	static long otherThreadsCpu(ThreadMXBean mx) {
		long sum = 0;
		final long me = Thread.currentThread().getId();
		for (long id : mx.getAllThreadIds()) {
			if (id != me) {
				final long t = mx.getThreadCpuTime(id);
				if (t > 0) {
					sum += t;
				}
			}
		}
		return sum;
	}

	static void timeAndSmp(NnueEvaluator ev, List<State> mid) throws Exception {
		final Negamax ng = new Negamax(ev, 40, true, 1 << 18, true, Negamax.ORDER_RECOMMENDED, 4);
		ng.setThreads(4);
		ng.setSafetyMarginMillis(300);
		// gli helper nascono con la ricerca: prima senza potatura, poi si accende e si controlla che la configurazione arrivi agli helper
		final State probeSt = mid.get(2);
		ng.findBestMove(fs(probeSt), 300);
		check(ng.helperPruningForTest(0) != null && !ng.helperPruningForTest(0).any(), "helper creati con la configurazione storica");
		ng.setPruning(PruningOptions.parse("all"));
		check(ng.helperPruningForTest(0).equals(PruningOptions.parse("all")), "setPruning propaga subito la configurazione agli helper esistenti");
		long worst = 0;
		for (int i = 0; i < 3; i++) {
			final State st = mid.get(i * 3 % mid.size());
			final long t0 = System.nanoTime();
			final int m = ng.findBestMove(fs(st), 1500);
			final long ms = (System.nanoTime() - t0) / 1_000_000;
			worst = Math.max(worst, ms);
			check(m != Move.NONE && isLegal(st, Move.toAction(m, st.getTurn())), "4 thread con potatura: mossa legale");
			check(ms <= 1500, "4 thread con potatura: mossa in " + ms + " ms <= 1500 ms di budget");
			check(ng.helperPruningForTest(0).equals(PruningOptions.parse("all")), "gli helper usano la configurazione accesa");
		}
		System.out.println("   tempo: 4 thread Lazy SMP con tutte le tecniche, budget 1500 ms (margine 300): la piu' lenta " + worst + " ms");
		ng.setPruning(PruningOptions.OFF);
		check(!ng.helperPruningForTest(0).any(), "setPruning(OFF) spegne anche gli helper");
		ng.setPruning(PruningOptions.parse("all"));

		final ThreadMXBean mx = ManagementFactory.getThreadMXBean();
		if (mx.isThreadCpuTimeSupported()) {
			mx.setThreadCpuTimeEnabled(true);
			final State st = mid.get(4 % mid.size());
			final long before = otherThreadsCpu(mx);
			ng.findBestMove(fs(st), 1200);
			final long during = otherThreadsCpu(mx) - before;
			Thread.sleep(300);
			final long q0 = otherThreadsCpu(mx);
			Thread.sleep(1200);
			final long idle = otherThreadsCpu(mx) - q0;
			check(during > 100_000_000L, "la misura ha i denti: durante la ricerca gli helper consumano CPU (" + during / 1_000_000 + " ms)");
			check(idle < 50_000_000L, "dopo la mossa, con la potatura accesa, nessun thread consuma CPU nel turno avversario (" + idle / 1_000_000 + " ms in 1200 ms)");
			System.out.println("   quiete: durante la ricerca " + during / 1_000_000 + " ms di CPU negli helper, a riposo " + idle / 1_000_000 + " ms");
		}
	}

	// ====================================================================== 11. efficienza (con prova)

	static void efficiency(NnueEvaluator ev, List<State> mid, boolean netIsReal) {
		long nPlain = 0, nAll = 0, nNmp = 0, nLmr = 0, nAsp = 0;
		final List<State> sub = mid.subList(0, Math.min(8, mid.size()));
		for (State st : sub) {
			final Negamax p = mk(ev, 6, true, PruningOptions.OFF), a = mk(ev, 6, true, PruningOptions.parse("all")), n = mk(ev, 6, true, PruningOptions.parse("nmp")),
					l = mk(ev, 6, true, PruningOptions.parse("lmr")), as = mk(ev, 6, true, PruningOptions.parse("asp"));
			p.findBestMove(fs(st), 600000);
			a.findBestMove(fs(st), 600000);
			n.findBestMove(fs(st), 600000);
			l.findBestMove(fs(st), 600000);
			as.findBestMove(fs(st), 600000);
			nPlain += p.lastNodeCount;
			nAll += a.lastNodeCount;
			nNmp += n.lastNodeCount;
			nLmr += l.lastNodeCount;
			nAsp += as.lastNodeCount;
		}
		System.out.printf("   nodi a profondita' 6 su %d posizioni (ricerca piena = 100%%): aspirazione %.1f%%, mossa nulla %.1f%%, LMR %.1f%%, tutto %.1f%%%n", sub.size(), 100.0 * nAsp / nPlain, 100.0 * nNmp / nPlain,
				100.0 * nLmr / nPlain, 100.0 * nAll / nPlain);
		if (netIsReal) {
			check(nAll <= 0.8 * nPlain, "con tutte le tecniche i nodi scendono di almeno il 20% (" + 100.0 * nAll / nPlain + "% del piano)");
			check(nLmr <= 0.9 * nPlain && nNmp <= 0.95 * nPlain, "LMR e mossa nulla riducono i nodi da sole");
		}
		check(nAsp <= nPlain, "l'aspirazione non aumenta i nodi");
	}

	public static void main(String[] args) throws Exception {
		final String v6 = args.length > 0 ? args[0] : "pesi_nnue/trained_v6.nnue";
		final boolean real = new File(v6).isFile();
		final NnueEvaluator ev = real ? NnueEvaluator.fromFile(v6) : new NnueEvaluator(NnueNetwork.random(64, 16, 7));
		System.out.println("rete: " + (real ? v6 : "casuale 64x16 (le verifiche sull'efficacia e sul rilevamento sono rilassate)"));
		final long t0 = System.currentTimeMillis();
		final List<State> mid = midgame(ev, 9, 4, 300);
		final List<State> end = endgames(ev, 10, 12, 700);
		final List<State> pool = new ArrayList<>(mid);
		pool.addAll(end);
		pool.addAll(TestPositions.playGame(new Random(3), true, 80).positions);
		System.out.println("fixture: " + mid.size() + " posizioni di meta' partita, " + end.size() + " di finale (" + (System.currentTimeMillis() - t0) / 1000 + " s)");

		final Set<String> only = new HashSet<>();
		if (args.length > 1) {
			only.addAll(Arrays.asList(args[1].split(",")));
		}
		if (only.isEmpty() || only.contains("options")) {
			options();
		}
		if (only.isEmpty() || only.contains("primitives")) {
			nullMovePrimitives(ev, pool);
		}
		if (only.isEmpty() || only.contains("session")) {
			sessionWalks(ev, pool);
		}
		if (only.isEmpty() || only.contains("equiv")) {
			equivalences(ev, mid);
		}
		if (only.isEmpty() || only.contains("sound")) {
			soundness(ev, end, real);
		}
		if (only.isEmpty() || only.contains("conv")) {
			convergence(ev, end);
		}
		if (only.isEmpty() || only.contains("integrity")) {
			integrity(ev, mid);
		}
		if (only.isEmpty() || only.contains("tt")) {
			ttAndHistory(ev, mid);
		}
		if (only.isEmpty() || only.contains("forcing")) {
			forcingFunction();
		}
		if (only.isEmpty() || only.contains("alloc")) {
			noAllocation(ev, mid);
		}
		if (only.isEmpty() || only.contains("guards")) {
			windowGuards(ev, mid);
		}
		if (only.isEmpty() || only.contains("smp")) {
			timeAndSmp(ev, mid);
		}
		if (only.isEmpty() || only.contains("eff")) {
			efficiency(ev, mid, real);
		}

		if (failures == 0) {
			System.out.println("=== PRUNING CHECK OK: " + checks + " verifiche ===");
			System.exit(0);
		} else {
			System.out.println("=== FALLITI: " + failures + " su " + checks + " ===");
			System.exit(1);
		}
	}
}
