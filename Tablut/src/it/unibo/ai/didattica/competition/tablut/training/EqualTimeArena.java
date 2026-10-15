package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Partite a TEMPO UGUALE per mossa tra due evaluator: l'unico confronto che
 * misura "forza per secondo di ricerca" (un evaluator piu' lento raggiunge meno
 * profondita' e deve compensarlo con la qualita'). Serve per decidere se un
 * nuovo file di pesi / un checkpoint / un tier di feature e' davvero migliore.
 *
 * <p>
 * Uso: {@code EqualTimeArena <A> <B> [coppie=30] [msPerMossa=100] [seed=1] [threads] [profonditaFissa]} dove A e B sono
 * {@code simple} oppure il percorso di un file di pesi, eventualmente preceduto da un prefisso che sceglie la RICERCA di quel
 * lato ({@code ab:}, {@code pvs:}, {@code nopvs:}: si veda {@link #usesPVS}) per misurare a tempo uguale il guadagno di PVS e
 * dell'ordinamento delle mosse. Un prefisso {@code tN:} (es. {@code t4:simple}, {@code t1:t4:...} non ha senso: basta uno)
 * fissa i THREAD di ricerca (Lazy SMP) di quel lato; l'argomento finale {@code threads} e' il valore comune di default
 * ({@link Negamax#defaultThreads()}). Attenzione: con piu' thread dei core fisici il confronto penalizza il lato parallelo.
 * Ogni "coppia" e' una apertura
 * casuale di 4 semi-mosse giocata DUE volte a colori invertiti (cosi' il vantaggio del
 * primo mossa e l'apertura si cancellano): 30 coppie = 60 partite. Stampa il bilancio
 * dal punto di vista di A con intervallo di confidenza di Wilson al 95% e, per ciascun lato,
 * profondita' media raggiunta e nodi/s.
 *
 * <p>
 * <b>Attenzione al tempo EFFETTIVO</b>: {@code Negamax.findBestMove} toglie al budget ricevuto un margine di sicurezza di 400 ms (pensato
 * per il server) e ha un minimo di 50 ms. Quindi {@code msPerMossa=100} significa 50 ms di ricerca reale, e per ottenere T ms effettivi
 * serve passare {@code T + 400}. Il confronto fra due lati Negamax resta simmetrico (entrambi subiscono lo stesso margine), ma il
 * confronto con un altro algoritmo (si veda {@code experiments.MctsArena}) va compensato.
 *
 * <p>
 * Avvertenze: le partite NON tracciano la ripetizione (tetto di 160 semi-mosse = pareggio) e
 * girano a poche centinaia di ms per mossa: le profondita' sono molto inferiori a quelle di
 * gara (60 s). Con 60 partite l'IC95 e' largo (~ +-12 punti): differenze piccole non sono
 * significative.
 */
public class EqualTimeArena {

	/**
	 * Prefisso opzionale che sceglie la ricerca di quel lato (stesso evaluator, stessa TT, stesso tempo):
	 * nessun prefisso = completa (PVS + killer + history); {@code pvs:} = PVS senza ordinamento (versione precedente);
	 * {@code nopvs:} = alpha-beta con killer + history; {@code ab:} = alpha-beta puro senza ordinamento (ricerca originale).
	 */
	static boolean usesPVS(String spec) {
		String s = spec.replaceFirst("^t\\d+:", "");
		return !(s.startsWith("nopvs:") || s.startsWith("ab:"));
	}

	static int orderingOf(String spec) {
		String s = spec.replaceFirst("^t\\d+:", "");
		return (s.startsWith("pvs:") || s.startsWith("ab:")) ? Negamax.ORDER_NONE : Negamax.ORDER_DEFAULT;
	}

	/** Prefisso opzionale {@code tN:} (N cifre): thread di ricerca di QUEL lato, al posto del valore comune. Si puo' combinare con gli altri prefissi (es. {@code t4:pvs:simple}). */
	static int threadsOf(String spec, int dflt) {
		java.util.regex.Matcher m = java.util.regex.Pattern.compile("^t(\\d+):").matcher(spec);
		return m.find() ? Math.max(1, Integer.parseInt(m.group(1))) : dflt;
	}

	static String stripPrefix(String spec) {
		boolean changed = true;
		while (changed) {
			changed = false;
			java.util.regex.Matcher m = java.util.regex.Pattern.compile("^t\\d+:").matcher(spec);
			if (m.find()) {
				spec = spec.substring(m.end());
				changed = true;
			}
			for (String pre : new String[] { "nopvs:", "pvs:", "ab:" }) {
				if (spec.startsWith(pre)) {
					spec = spec.substring(pre.length());
					changed = true;
				}
			}
		}
		return spec;
	}

	static Evaluator build(String spec) throws IOException {
		String s = stripPrefix(spec);
		return s.equals("simple") ? new SimpleEvaluator() : EvaluatorLoader.load(s); // MLP o CNN, dall'intestazione del file
	}

	private static final class Stats {
		double depthSum, nodeSum, timeSum;
		long moves;
	}

	public static void main(String[] a) throws IOException {
		if (a.length < 2) {
			System.out.println("Uso: EqualTimeArena <simple|pesiA.txt> <simple|pesiB.txt> [coppie=30] [msPerMossa=100] [seed=1] [threads]");
			return;
		}
		String specA = a[0], specB = a[1];
		int pairs = a.length > 2 ? Integer.parseInt(a[2]) : 30;
		long ms = a.length > 3 ? Long.parseLong(a[3]) : 100;
		long seed = a.length > 4 ? Long.parseLong(a[4]) : 1;
		final int threads = a.length > 5 ? Math.max(1, Integer.parseInt(a[5])) : Negamax.defaultThreads();
		// 7o argomento opzionale: profondita' FISSA (stessa per entrambi, tempo praticamente illimitato). Misura la qualita' pura
		// dell'evaluator a parita' di profondita', senza il costo per nodo: utile per capire se un evaluator piu' lento "merita" il tempo.
		final int fixedDepth = a.length > 6 ? Math.max(1, Integer.parseInt(a[6])) : 0;
		final int maxDepth = fixedDepth > 0 ? fixedDepth : 30;
		if (fixedDepth > 0) {
			ms = 600000;
		}
		Evaluator evA = build(specA), evB = build(specB);
		Random rnd = new Random(seed);
		Stats sa = new Stats(), sb = new Stats();
		int aw = 0, ad = 0, al = 0;

		for (int p = 0; p < pairs; p++) {
			State open = new StateTablut();
			open.setTurn(Turn.WHITE);
			for (int i = 0; i < 4; i++) {
				List<Action> mv = AshtonRules.generateMoves(open);
				open = AshtonRules.applyMove(open, mv.get(rnd.nextInt(mv.size())));
			}
			for (int swap = 0; swap < 2; swap++) {
				Evaluator white = swap == 0 ? evA : evB, black = swap == 0 ? evB : evA;
				Stats stW = swap == 0 ? sa : sb, stB = swap == 0 ? sb : sa;
				String sw = swap == 0 ? specA : specB, sbk = swap == 0 ? specB : specA;
				Negamax nw = new Negamax(white, maxDepth, true, 1 << 18, usesPVS(sw), orderingOf(sw)).setThreads(threadsOf(sw, threads));
				Negamax nb = new Negamax(black, maxDepth, true, 1 << 18, usesPVS(sbk), orderingOf(sbk)).setThreads(threadsOf(sbk, threads));
				State s = open;
				for (int ply = 4; ply < 160 && !AshtonRules.isTerminal(s); ply++) {
					boolean wt = s.getTurn() == Turn.WHITE;
					Negamax n = wt ? nw : nb;
					Stats st = wt ? stW : stB;
					long t0 = System.nanoTime();
					Action m = n.findBestMove(s, ms);
					st.timeSum += (System.nanoTime() - t0) / 1e9;
					st.depthSum += n.lastDepthReached;
					st.nodeSum += n.lastTotalNodeCount; // nodi di TUTTI i thread di ricerca
					st.moves++;
					if (m == null) {
						break;
					}
					s = AshtonRules.applyMove(s, m);
				}
				Turn t = s.getTurn();
				int res; // dal punto di vista di A
				if (t == Turn.WHITEWIN) {
					res = swap == 0 ? 1 : -1;
				} else if (t == Turn.BLACKWIN) {
					res = swap == 0 ? -1 : 1;
				} else if (!AshtonRules.isTerminal(s) && AshtonRules.generateMoves(s).isEmpty()) {
					res = (s.getTurn() == Turn.WHITE) == (swap == 0) ? -1 : 1; // chi non ha mosse perde
				} else {
					res = 0;
				}
				if (res > 0) {
					aw++;
				} else if (res < 0) {
					al++;
				} else {
					ad++;
				}
			}
			if ((p + 1) % 5 == 0) {
				System.out.printf("  [%d/%d coppie] A: +%d =%d -%d%n", p + 1, pairs, aw, ad, al);
			}
		}
		int n = aw + ad + al;
		double score = (aw + 0.5 * ad) / n, z = 1.96;
		double den = 1 + z * z / n;
		double centre = (score + z * z / (2 * n)) / den;
		double half = z * Math.sqrt(score * (1 - score) / n + z * z / (4.0 * n * n)) / den;
		System.out.printf("[thread di ricerca: A=%d B=%d] ", threadsOf(specA, threads), threadsOf(specB, threads));
		System.out.printf("A=%s  VS  B=%s  @%dms/mossa: %d partite  A +%d =%d -%d  punteggio A = %.1f%% [IC95 %.1f%% - %.1f%%]%n",
				specA, specB, ms, n, aw, ad, al, 100 * score, 100 * (centre - half), 100 * (centre + half));
		System.out.printf("   A: profondita' media %.2f, nodi/s %.0f | B: profondita' media %.2f, nodi/s %.0f%n",
				sa.depthSum / sa.moves, sa.nodeSum / sa.timeSum, sb.depthSum / sb.moves, sb.nodeSum / sb.timeSum);
	}
}
