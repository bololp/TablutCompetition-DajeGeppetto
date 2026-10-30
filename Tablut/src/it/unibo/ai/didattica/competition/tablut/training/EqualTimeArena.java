package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.EvaluatorLoader;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;
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
 * lato ({@code ab:}, {@code pvs:}, {@code nopvs:}, {@code ord=rec:}, {@code tt=20:}, {@code tt2:}, {@code m=0:}: si veda {@link #usesPVS}) per
 * misurare a tempo uguale il guadagno di PVS, dell'ordinamento delle mosse, della dimensione/politica della TT. Un prefisso {@code tN:} (es. {@code t4:simple}, {@code t1:t4:...} non ha senso: basta uno)
 * fissa i THREAD di ricerca (Lazy SMP) di quel lato; l'argomento finale {@code threads} e' il valore comune di default
 * ({@link Negamax#defaultThreads()}). Attenzione: con piu' thread dei core fisici il confronto penalizza il lato parallelo.
 * Ogni "coppia" e' una apertura
 * casuale di 4 semi-mosse giocata DUE volte a colori invertiti (cosi' il vantaggio del
 * primo mossa e l'apertura si cancellano): 30 coppie = 60 partite. Stampa il bilancio
 * dal punto di vista di A con intervallo di confidenza di Wilson al 95% e, per ciascun lato,
 * profondita' media raggiunta e nodi/s.
 *
 * <p>
 * <b>Attenzione al tempo EFFETTIVO</b>: {@code Negamax.findBestMove} toglie al budget ricevuto un margine di sicurezza (300 ms di
 * default, pensato per il server) e ha un minimo di 50 ms. Quindi {@code msPerMossa=100} significa 50 ms di ricerca reale, e per ottenere
 * T ms effettivi serve passare {@code T + 300}, oppure dare a ENTRAMBI i lati il prefisso {@code m=0:} (allora {@code msPerMossa} e' il tempo reale). Il confronto fra due lati Negamax resta simmetrico (entrambi subiscono lo stesso margine), ma il
 * confronto con un altro algoritmo (si veda {@code experiments.MctsArena}) va compensato.
 *
 * <p>
 * Avvertenze: le partite NON tracciano la ripetizione (tetto di 160 semi-mosse = pareggio) e
 * girano a poche centinaia di ms per mossa: le profondita' sono molto inferiori a quelle di
 * gara (60 s). Con 60 partite l'IC95 e' largo (~ +-12 punti): differenze piccole non sono
 * significative.
 */
public class EqualTimeArena {

	private static final java.util.regex.Pattern PREFIX = java.util.regex.Pattern
			.compile("^(t\\d+|ord=[a-z+]+|tt=\\d+|tt2|m=\\d+|nopvs|pvs|ab):");

	/** I prefissi di {@code spec}, nell'ordine in cui compaiono (senza i due punti). */
	static java.util.List<String> prefixes(String spec) {
		java.util.List<String> out = new java.util.ArrayList<>();
		java.util.regex.Matcher m = PREFIX.matcher(spec);
		while (m.find()) {
			out.add(m.group(1));
			spec = spec.substring(m.end());
			m = PREFIX.matcher(spec);
		}
		return out;
	}

	static String stripPrefix(String spec) {
		java.util.regex.Matcher m = PREFIX.matcher(spec);
		while (m.find()) {
			spec = spec.substring(m.end());
			m = PREFIX.matcher(spec);
		}
		return spec;
	}

	/**
	 * Prefissi opzionali (combinabili, in qualunque ordine) che scelgono la ricerca di QUEL lato, a parita' di evaluator e tempo:
	 * nessuno = completa (PVS + killer + history); {@code pvs:} = PVS senza ordinamento; {@code nopvs:} = alpha-beta con killer +
	 * history; {@code ab:} = alpha-beta puro senza ordinamento; {@code ord=<nomi>:} = ordinamento scelto con {@link OrderingSpec}
	 * (es. {@code ord=rec:}, {@code ord=wins+caps:}); {@code tt=<log2>:} = TT di 2^log2 voci (default 2^18); {@code tt2:} = TT a due
	 * slot; {@code m=<ms>:} = margine di sicurezza di Negamax (default 300; con {@code m=0:} su ENTRAMBI i lati {@code msPerMossa}
	 * e' il tempo di ricerca reale); {@code tN:} = N thread di ricerca.
	 */
	static boolean usesPVS(String spec) {
		java.util.List<String> p = prefixes(spec);
		return !(p.contains("nopvs") || p.contains("ab"));
	}

	static int orderingOf(String spec) {
		java.util.List<String> p = prefixes(spec);
		int mode = (p.contains("pvs") || p.contains("ab")) ? Negamax.ORDER_NONE : Negamax.ORDER_DEFAULT;
		for (String t : p) {
			if (t.startsWith("ord=")) {
				mode = OrderingSpec.parse(t.substring(4));
			}
		}
		return mode;
	}

	/** Prefisso opzionale {@code tN:} (N cifre): thread di ricerca di QUEL lato, al posto del valore comune. */
	static int threadsOf(String spec, int dflt) {
		for (String t : prefixes(spec)) {
			if (t.matches("t\\d+")) {
				return Math.max(1, Integer.parseInt(t.substring(1)));
			}
		}
		return dflt;
	}

	static int ttLog2Of(String spec, int dflt) {
		for (String t : prefixes(spec)) {
			if (t.startsWith("tt=")) {
				return Math.max(4, Math.min(26, Integer.parseInt(t.substring(3))));
			}
		}
		return dflt;
	}

	static boolean twoSlotOf(String spec) {
		return prefixes(spec).contains("tt2");
	}

	/** Margine di sicurezza richiesto con {@code m=<ms>:}, oppure -1 (lascia quello di Negamax). */
	static long marginOf(String spec) {
		for (String t : prefixes(spec)) {
			if (t.startsWith("m=")) {
				return Long.parseLong(t.substring(2));
			}
		}
		return -1;
	}

	/** La ricerca di un lato secondo i suoi prefissi. */
	static Negamax newSearcher(Evaluator ev, int maxDepth, String spec, int threads) {
		Negamax n = new Negamax(ev, maxDepth, true, 1 << ttLog2Of(spec, 18), usesPVS(spec), orderingOf(spec), threadsOf(spec, threads), twoSlotOf(spec))
				.setThreads(threadsOf(spec, threads));
		if (marginOf(spec) >= 0) {
			n.setSafetyMarginMillis(marginOf(spec));
		}
		return n;
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
		run(specA, specB, pairs, ms, seed, threads, fixedDepth);
	}

	/**
	 * Il corpo di {@link #main} come metodo richiamabile (usato da {@link DistillAcceptance}): stessa partita, stesso output.
	 *
	 * @param fixedDepth 0 = tempo uguale ({@code ms} per mossa); &gt; 0 = profondita' fissa uguale per i due lati
	 * @return {vittorie di A, patte, sconfitte di A}
	 */
	public static int[] run(String specA, String specB, int pairs, long ms, long seed, final int threads, final int fixedDepth) throws IOException {
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
				Negamax nw = newSearcher(white, maxDepth, sw, threads);
				Negamax nb = newSearcher(black, maxDepth, sbk, threads);
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
		return new int[] { aw, ad, al };
	}
}
