package it.unibo.ai.didattica.competition.tablut.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Map;

import it.unibo.ai.didattica.competition.tablut.aiplayer.CnnEvaluator;

/**
 * Porte di accettazione della rete distillata. La candidata puo' sostituire quella di partenza SOLO se le superano TUTTE;
 * una porta fallita oppure non verificata (non eseguita, campione insufficiente) vale come NON adottare.
 * <ol>
 * <li><b>(i) errore held-out vs ricerca</b>: MSE (dominio tanh) della candidata sui bersagli di ricerca &lt; quello della rete di partenza,
 * su posizioni di partite MAI viste in training ({@link DistillSet#heldOut}). I bersagli sono quelli della ricerca della rete di PARTENZA a
 * profondita' {@code depth}, uguali per le due reti ({@link CnnDistiller#heldOutGate}). Un guadagno offline NON implica un guadagno in arena.</li>
 * <li><b>(ii) HangingDiag</b> ({@link HangingGate}): la quota di mosse che lasciano pezzi catturabili e il materiale perso nei 3 ply
 * successivi non devono peggiorare.</li>
 * <li><b>(iii) {@link EqualTimeArena}</b>, candidata contro partenza, in DUE modi: tempo uguale con 4 thread per lato e profondita' fissa.
 * Criterio (non-inferiorita' con evidenza): punteggio della candidata &gt;= 50% e limite inferiore dell'IC95 di Wilson &gt;= 50% - {@code arenaMargin}
 * (default 5 punti), in ENTRAMBI i modi. Con 30 partite l'IC95 e' ~+-17 punti: il default e' 200 partite per modo (~+-7); per distinguere un
 * 55% da un 50% ne servono ~800. Il numero di partite e' configurabile; la porta resta NON VERIFICATA se non e' stata eseguita ({@code skipArena=1}).</li>
 * </ol>
 * L'unico effetto collaterale e' la sostituzione richiesta esplicitamente con {@code adoptTo=<percorso>} (con copia di sicurezza del file
 * preesistente in {@code <percorso>.bak}) e solo se tutte e tre le porte sono SUPERATE.
 */
public final class DistillAcceptance {

	private DistillAcceptance() {
	}

	static final String PASS = "SUPERATA", FAIL = "NON SUPERATA", UNVERIFIED = "NON VERIFICATA";

	/** Esito di un confronto in arena. */
	public static final class ArenaVerdict {
		public final boolean verified, pass;
		public final double score, lo, hi;
		public final int games;
		public final String text;

		ArenaVerdict(boolean verified, boolean pass, double score, double lo, double hi, int games, String text) {
			this.verified = verified;
			this.pass = pass;
			this.score = score;
			this.lo = lo;
			this.hi = hi;
			this.games = games;
			this.text = text;
		}
	}

	/**
	 * Criterio della porta (iii) per UN modo: {@code wdl} = {vittorie, patte, sconfitte} della candidata (lato A).
	 * Passa se punteggio &gt;= 0.5 e limite inferiore di Wilson &gt;= 0.5 - margin.
	 */
	public static ArenaVerdict arenaVerdict(int[] wdl, double margin) {
		int n = wdl[0] + wdl[1] + wdl[2];
		if (n == 0) {
			return new ArenaVerdict(false, false, 0, 0, 1, 0, "nessuna partita giocata");
		}
		double score = (wdl[0] + 0.5 * wdl[1]) / n, z = 1.96, den = 1 + z * z / n;
		double centre = (score + z * z / (2 * n)) / den;
		double half = z * Math.sqrt(score * (1 - score) / n + z * z / (4.0 * n * n)) / den;
		double lo = centre - half, hi = centre + half;
		boolean pass = score >= 0.5 && lo >= 0.5 - margin;
		return new ArenaVerdict(true, pass, score, lo, hi, n, String.format("%d partite +%d =%d -%d: punteggio candidata %.1f%% [IC95 %.1f-%.1f], soglia: >=50%% e limite inferiore >=%.1f%%",
				n, wdl[0], wdl[1], wdl[2], 100 * score, 100 * lo, 100 * hi, 100 * (0.5 - margin)));
	}

	/** Esito complessivo: vero solo se tutte le porte sono verificate E superate. */
	public static boolean adoptable(boolean gate1, boolean gate2Verified, boolean gate2Pass, boolean arenaTimeVerified, boolean arenaTimePass,
			boolean arenaDepthVerified, boolean arenaDepthPass) {
		return gate1 && gate2Verified && gate2Pass && arenaTimeVerified && arenaTimePass && arenaDepthVerified && arenaDepthPass;
	}

	/**
	 * <pre>
	 * DistillAcceptance baseline=pesi_v4.txt candidate=cnn_distilled.txt set=distill_set.gz depth=3
	 *   [hgGames=100 hgDepth=3 hgMs=0 hgSeed=1 hgMinMoves=200 hgTol=0 hgThreads=1]
	 *   [arenaPairs=100 arenaMs=5000 arenaThreads=4 arenaDepth=4 arenaMargin=0.05 arenaSeed=1 | skipArena=1]
	 *   [threads=4] [adoptTo=percorso]
	 * </pre>
	 * Codice di uscita: 0 = tutte le porte superate; 1 = almeno una porta NON superata; 4 = nessuna fallita ma almeno una NON VERIFICATA.
	 * {@code arenaMs} e' il tempo REALE per mossa (si usa il prefisso {@code m=0:}); {@code arenaPairs} sono coppie di partite a colori invertiti
	 * (200 partite per 100 coppie).
	 */
	public static void main(String[] args) throws Exception {
		System.exit(mainImpl(args));
	}

	/** Come {@link #main} ma ritorna il codice di uscita (0 / 1 / 4 / 2 = uso errato) invece di chiamare {@code System.exit}. */
	static int mainImpl(String[] args) throws Exception {
		Map<String, String> p = CnnDistiller.parse(args);
		if (!p.containsKey("baseline") || !p.containsKey("candidate") || !p.containsKey("set")) {
			System.out.println("uso: DistillAcceptance baseline=pesi.txt candidate=pesi_nuovi.txt set=distill_set.gz [depth=3] [hgGames=100 hgDepth=3 hgMs=0 hgSeed=1"
					+ " hgMinMoves=200 hgTol=0] [arenaPairs=100 arenaMs=5000 arenaThreads=4 arenaDepth=4 arenaMargin=0.05 | skipArena=1] [threads=N] [adoptTo=percorso]");
			return 2;
		}
		final String basePath = p.get("baseline"), candPath = p.get("candidate");
		int threads = Integer.parseInt(p.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
		int depth = Integer.parseInt(p.getOrDefault("depth", "3"));
		CnnEvaluator base = CnnEvaluator.fromFile(basePath), cand = CnnEvaluator.fromFile(candPath);
		System.out.println("=== DistillAcceptance ===\n  partenza:  " + basePath + "\n  candidata: " + candPath);
		boolean any_fail = false, any_unverified = false;

		// ---------------------------------------------------------------- (i)
		boolean g1;
		if (base.featureGroups != cand.featureGroups || !base.net.config().equals(cand.net.config())) {
			System.out.println("(i)   errore held-out vs ricerca: " + FAIL + " (le due reti hanno forma/gruppi di feature diversi: confronto non appaiabile)");
			g1 = false;
		} else {
			DistillSet set = DistillSet.load(p.get("set"));
			int[] held = set.indices(true);
			int[] cols = CnnDistiller.colsFor(base.net, base.featureGroups);
			System.out.println("(i)   etichetto " + held.length + " posizioni di held-out (partite non viste in training) con la rete di PARTENZA a D=" + depth + " ...");
			SearchLabeler.Labels lab = SearchLabeler.labelAll(base, set.cells, set.role, held, depth, threads, 16, System.out);
			double[] y = new double[set.n];
			for (int k = 0; k < held.length; k++) {
				y[held[k]] = lab.y[k];
			}
			CnnDistiller.HeldOutGate g = CnnDistiller.heldOutGate(base.net, cand.net, set, cols, held, y);
			System.out.println("(i)   " + g + (held.length < 500 ? "  [ATTENZIONE: held-out piccolo]" : ""));
			g1 = g.pass;
		}
		any_fail |= !g1;

		// ---------------------------------------------------------------- (ii)
		int hgGames = Integer.parseInt(p.getOrDefault("hgGames", "100"));
		int hgDepth = Integer.parseInt(p.getOrDefault("hgDepth", "3"));
		long hgMs = Long.parseLong(p.getOrDefault("hgMs", "0"));
		long hgSeed = Long.parseLong(p.getOrDefault("hgSeed", "1"));
		int hgMin = Integer.parseInt(p.getOrDefault("hgMinMoves", "200"));
		double hgTol = Double.parseDouble(p.getOrDefault("hgTol", "0"));
		int hgThreads = Integer.parseInt(p.getOrDefault("hgThreads", "1"));
		System.out.println("(ii)  HangingDiag: " + hgGames + " partite per rete, " + (hgMs > 0 && hgDepth == 0 ? hgMs + " ms/mossa" : "profondita' fissa " + hgDepth) + ", stesse aperture");
		HangingGate.Stats sb = HangingGate.measure(base, hgGames, hgMs, hgDepth, hgSeed, hgThreads, System.out);
		HangingGate.Stats sc = HangingGate.measure(cand, hgGames, hgMs, hgDepth, hgSeed, hgThreads, System.out);
		System.out.println("      partenza:  " + sb + "\n      candidata: " + sc);
		HangingGate.Verdict v2 = HangingGate.compare(sb, sc, hgMin, hgTol);
		System.out.println("(ii)  " + (v2.verified ? (v2.pass ? PASS : FAIL) : UNVERIFIED) + " - " + v2.text);
		any_fail |= v2.verified && !v2.pass;
		any_unverified |= !v2.verified;

		// ---------------------------------------------------------------- (iii)
		ArenaVerdict at, ad;
		if (p.getOrDefault("skipArena", "0").equals("1")) {
			System.out.println("(iii) EqualTimeArena: " + UNVERIFIED + " (skipArena=1: non eseguita). Comandi:\n" + arenaCommands(basePath, candPath, p));
			at = new ArenaVerdict(false, false, 0, 0, 1, 0, "non eseguita");
			ad = at;
		} else {
			int pairs = Integer.parseInt(p.getOrDefault("arenaPairs", "100"));
			long ms = Long.parseLong(p.getOrDefault("arenaMs", "5000"));
			int at4 = Integer.parseInt(p.getOrDefault("arenaThreads", "4"));
			int aDepth = Integer.parseInt(p.getOrDefault("arenaDepth", "4"));
			double margin = Double.parseDouble(p.getOrDefault("arenaMargin", "0.05"));
			long seed = Long.parseLong(p.getOrDefault("arenaSeed", "1"));
			if (Runtime.getRuntime().availableProcessors() < at4) {
				System.out.println("      ATTENZIONE: " + Runtime.getRuntime().availableProcessors() + " core disponibili < " + at4 + " thread per lato: il confronto a tempo"
						+ " uguale e' falsato (e lento). Eseguirlo su una macchina con almeno " + 2 * at4 + " core.");
			}
			System.out.println("(iii) arena a TEMPO UGUALE: " + pairs + " coppie (" + 2 * pairs + " partite), " + ms + " ms reali/mossa, " + at4 + " thread per lato");
			int[] wdlT = EqualTimeArena.run("t" + at4 + ":m=0:" + candPath, "t" + at4 + ":m=0:" + basePath, pairs, ms, seed, at4, 0);
			at = arenaVerdict(wdlT, margin);
			System.out.println("(iii) tempo uguale: " + (at.pass ? PASS : FAIL) + " - " + at.text);
			System.out.println("(iii) arena a PROFONDITA' FISSA " + aDepth + ": " + pairs + " coppie");
			int[] wdlD = EqualTimeArena.run("t1:m=0:" + candPath, "t1:m=0:" + basePath, pairs, 0, seed + 1, 1, aDepth);
			ad = arenaVerdict(wdlD, margin);
			System.out.println("(iii) profondita' fissa: " + (ad.pass ? PASS : FAIL) + " - " + ad.text);
			any_fail |= !at.pass || !ad.pass;
		}
		any_unverified |= !at.verified || !ad.verified;

		// ---------------------------------------------------------------- verdetto
		boolean ok = adoptable(g1, v2.verified, v2.pass, at.verified, at.pass, ad.verified, ad.pass);
		System.out.println("--------------------------------------------------");
		System.out.println("(i) " + (g1 ? PASS : FAIL) + " | (ii) " + (v2.verified ? (v2.pass ? PASS : FAIL) : UNVERIFIED) + " | (iii) tempo " + (at.verified ? (at.pass ? PASS : FAIL) : UNVERIFIED)
				+ ", profondita' " + (ad.verified ? (ad.pass ? PASS : FAIL) : UNVERIFIED));
		if (ok) {
			System.out.println("VERDETTO: tutte le porte superate -> la candidata e' ADOTTABILE.");
			if (p.containsKey("adoptTo")) {
				adopt(candPath, p.get("adoptTo"));
			} else {
				System.out.println("(nessuna sostituzione eseguita: passare adoptTo=<percorso> per copiare la candidata, con copia di sicurezza)");
			}
			return 0;
		}
		System.out.println("VERDETTO: NON ADOTTARE (" + (any_fail ? "almeno una porta non superata" : "almeno una porta non verificata") + "). La rete di partenza resta in uso.");
		return any_fail ? 1 : 4;
	}

	static String arenaCommands(String base, String cand, Map<String, String> p) {
		int pairs = Integer.parseInt(p.getOrDefault("arenaPairs", "100"));
		long ms = Long.parseLong(p.getOrDefault("arenaMs", "5000"));
		int d = Integer.parseInt(p.getOrDefault("arenaDepth", "4"));
		String cp = "-cp \"build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar\" it.unibo.ai.didattica.competition.tablut.training.EqualTimeArena";
		return "      java " + cp + " t4:m=0:" + cand + " t4:m=0:" + base + " " + pairs + " " + ms + " 1 4\n      java " + cp + " t1:m=0:" + cand + " t1:m=0:" + base + " " + pairs
				+ " 0 1 1 " + d;
	}

	/** Sostituisce {@code target} con {@code candidate}, salvando prima {@code target} in {@code target.bak} se esiste. */
	static void adopt(String candidate, String target) throws IOException {
		File t = new File(target);
		if (t.exists()) {
			Files.copy(t.toPath(), new File(target + ".bak").toPath(), StandardCopyOption.REPLACE_EXISTING);
			System.out.println("copia di sicurezza: " + target + ".bak");
		}
		Files.copy(new File(candidate).toPath(), t.toPath(), StandardCopyOption.REPLACE_EXISTING);
		System.out.println("ADOTTATA: " + candidate + " -> " + target);
	}
}
