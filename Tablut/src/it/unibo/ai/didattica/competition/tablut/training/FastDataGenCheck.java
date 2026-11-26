package it.unibo.ai.didattica.competition.tablut.training;

import java.util.HashMap;
import java.util.Map;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork;
import it.unibo.ai.didattica.competition.tablut.aiplayer.PruningOptions;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;

/**
 * Test delle opzioni "veloci" di generazione ({@link NnueSelfPlay}: prune, tt, movedepth, labelevery, skipsat, adj) e di {@link NnueRelabel#relabelFast}:
 * le opzioni spente non cambiano nulla, i dataset non dipendono dal numero di thread, ogni opzione fa quello che dichiara. Uso: {@code FastDataGenCheck}.
 */
public class FastDataGenCheck {

	private static int failures = 0, checks = 0;

	private static void check(boolean c, String m) {
		checks++;
		if (!c) {
			failures++;
			System.out.println("FAIL - " + m);
		}
	}

	private static NnueSelfPlay.Config cfg(int depth, int games) {
		final NnueSelfPlay.Config c = new NnueSelfPlay.Config();
		c.games = games;
		c.depth = depth;
		c.maxPlies = 40;
		c.seed = 5;
		c.eps = 0.1;
		return c;
	}

	private static Map<Integer, java.util.List<Integer>> pliesByGame(NnueDataset d) {
		final Map<Integer, java.util.List<Integer>> m = new HashMap<>();
		for (int i = 0; i < d.n; i++) {
			m.computeIfAbsent(d.game(i), k -> new java.util.ArrayList<>()).add((int) d.ply(i));
		}
		return m;
	}

	public static void main(String[] args) throws Exception {
		final Evaluator nn = new NnueEvaluator(NnueNetwork.random(16, 8, 3));
		final Evaluator simple = new SimpleEvaluator();

		// 1. opzioni spente == predefinito, e con qualunque numero di thread
		NnueSelfPlay.Config a = cfg(3, 6);
		final NnueDataset base = NnueSelfPlay.generate(nn, nn, a, null);
		NnueSelfPlay.Config b = cfg(3, 6);
		b.moveDepth = 0;
		b.labelEvery = 1;
		b.skipSat = 0;
		b.adjPlies = 0;
		b.ttLog2 = 18;
		b.pruning = PruningOptions.OFF;
		check(NnueSelfPlay.generate(nn, nn, b, null).equals(base), "opzioni esplicitamente spente: stesso dataset del predefinito");
		check(base.n > 50, "il dataset di riferimento ha posizioni (" + base.n + ")");

		// 2. potatura: dataset valido e indipendente dai thread
		NnueSelfPlay.Config p1 = cfg(4, 6), p3 = cfg(4, 6);
		p1.pruning = PruningOptions.parse("all");
		p3.pruning = PruningOptions.parse("all");
		p3.threads = 3;
		final NnueDataset dp1 = NnueSelfPlay.generate(nn, nn, p1, null), dp3 = NnueSelfPlay.generate(nn, nn, p3, null);
		check(dp1.equals(dp3) && dp1.n > 50, "prune=all: lo stesso dataset con 1 e con 3 thread");
		boolean range = true;
		for (int i = 0; i < dp1.n; i++) {
			range &= dp1.valValid(i) == 1 && Math.abs(dp1.value(i)) <= 1.0f;
		}
		check(range, "prune=all: valori in [-1,1] e validi");

		// 3. movedepth + labelevery: una posizione ogni K con fase casuale, indipendente dai thread
		NnueSelfPlay.Config s1 = cfg(3, 8), s3 = cfg(3, 8);
		for (NnueSelfPlay.Config c : new NnueSelfPlay.Config[] { s1, s3 }) {
			c.moveDepth = 1;
			c.labelEvery = 3;
		}
		s3.threads = 3;
		final NnueDataset ds1 = NnueSelfPlay.generate(nn, nn, s1, null), ds3 = NnueSelfPlay.generate(nn, nn, s3, null);
		check(ds1.equals(ds3), "movedepth/labelevery: stesso dataset con 1 e con 3 thread");
		boolean spaced = true;
		int multi = 0;
		for (java.util.List<Integer> pl : pliesByGame(ds1).values()) {
			for (int k = 1; k < pl.size(); k++) {
				spaced &= pl.get(k) - pl.get(k - 1) == 3;
			}
			if (pl.size() > 1) {
				multi++;
			}
		}
		check(spaced && multi > 3, "labelevery=3: le posizioni registrate distano 3 semi-mosse (partite con piu' posizioni: " + multi + ")");
		NnueSelfPlay.Config full = cfg(3, 8);
		final NnueDataset dfull = NnueSelfPlay.generate(nn, nn, full, null);
		check(ds1.n < dfull.n * 0.5 && ds1.n > dfull.n / 6, "una posizione ogni 3: " + ds1.n + " contro " + dfull.n + " registrando tutto");

		// l'etichetta e' quella della ricerca profonda: confronto con una ricerca nuova a depth sulle stesse posizioni
		final NnueDataset re = NnueRelabel.relabel(ds1, nn, 3, 1000, false, null);
		int same = 0, closeEnough = 0;
		for (int i = 0; i < ds1.n; i++) {
			if (Math.abs(re.value(i) - ds1.value(i)) < 1e-6) {
				same++;
			}
			if (Math.abs(re.value(i) - ds1.value(i)) < 0.1) {
				closeEnough++;
			}
		}
		check(same >= ds1.n * 0.7 && closeEnough >= ds1.n * 0.9, "etichette == ricerca a depth (uguali " + same + "/" + ds1.n + ", entro 0,1: " + closeEnough + ")");

		// 4. skipsat: nessuna posizione con |v| >= soglia
		NnueSelfPlay.Config k0 = cfg(2, 8), k1 = cfg(2, 8);
		k1.skipSat = 0.2;
		final NnueDataset dk0 = NnueSelfPlay.generate(simple, simple, k0, null), dk1 = NnueSelfPlay.generate(simple, simple, k1, null);
		boolean allBelow = true;
		for (int i = 0; i < dk1.n; i++) {
			allBelow &= Math.abs(dk1.value(i)) < 0.2f;
		}
		int sat0 = 0;
		for (int i = 0; i < dk0.n; i++) {
			sat0 += Math.abs(dk0.value(i)) >= 0.2f ? 1 : 0;
		}
		check(allBelow && sat0 > 0 && dk1.n == dk0.n - sat0, "skipsat: nessuna posizione satura (" + sat0 + " scartate su " + dk0.n + ")");

		// 5. aggiudicazione: partite chiuse presto, senza esito, con la striscia di valori saturi in fondo
		NnueSelfPlay.Config j = cfg(2, 10);
		j.maxPlies = 120;
		j.adjThreshold = 0.05;
		j.adjPlies = 3;
		j.eps = 0;
		int adjudicated = 0;
		for (int g = 0; g < 10; g++) {
			final NnueSelfPlay.Game game = NnueSelfPlay.playGame(simple, simple, j, 1000 + g);
			if (game.adjudicated) {
				adjudicated++;
				check(!game.outValid, "partita aggiudicata: nessun esito");
				final int n = game.values.size();
				boolean run = n >= 3;
				for (int t = n - 3; t < n && run; t++) {
					run = Math.abs(game.values.get(t)) >= 0.05 && Math.signum(game.values.get(t)) == Math.signum(game.values.get(n - 1));
				}
				check(run, "aggiudicata dopo 3 ricerche consecutive sature dello stesso segno");
			}
		}
		check(adjudicated > 0, "almeno una partita aggiudicata con Simple a 0,05:3 (" + adjudicated + " su 10)");
		final NnueSelfPlay.Config j0 = cfg(2, 10);
		j0.maxPlies = 120;
		j0.eps = 0;
		int plyAdj = 0, plyNo = 0;
		for (int g = 0; g < 10; g++) {
			plyAdj += NnueSelfPlay.playGame(simple, simple, j, 1000 + g).applied;
			plyNo += NnueSelfPlay.playGame(simple, simple, j0, 1000 + g).applied;
		}
		check(plyAdj < plyNo, "l'aggiudicazione accorcia le partite (" + plyAdj + " contro " + plyNo + " semi-mosse)");

		// 6. configurazioni non valide
		final NnueSelfPlay.Config[] bad = new NnueSelfPlay.Config[8];
		for (int i = 0; i < bad.length; i++) {
			bad[i] = cfg(3, 2);
		}
		bad[0].labelEvery = 0;
		bad[1].ttLog2 = 5;
		bad[2].skipSat = 1.5;
		bad[3].adjPlies = 3; // senza soglia
		bad[4].moveDepth = 2;
		bad[4].msPerMove = 100; // movedepth con tempo reale
		bad[5].moveDepth = -1;
		bad[6].adjPlies = -1;
		bad[7].pruning = null;
		for (int i = 0; i < bad.length; i++) {
			boolean threw = false;
			try {
				NnueSelfPlay.generate(nn, nn, bad[i], null);
			} catch (IllegalArgumentException e) {
				threw = true;
			}
			check(threw, "configurazione non valida #" + i + " rifiutata");
		}

		// 7. rietichettatura veloce
		final NnueDataset src = NnueSelfPlay.generate(nn, nn, cfg(2, 6), null);
		final NnueRelabel.Options one = new NnueRelabel.Options();
		one.chunk = src.n + 10; // un solo blocco: deve coincidere con la rietichettatura storica
		final NnueDataset legacy = NnueRelabel.relabel(src, nn, 3, 1000, false, null);
		check(NnueRelabel.relabelFast(src, nn, 3, 1000, one, null).equals(legacy), "relabelFast con un solo blocco == relabel storico");
		final NnueRelabel.Options o1 = new NnueRelabel.Options(), o4 = new NnueRelabel.Options();
		o1.chunk = 7;
		o4.chunk = 7;
		o4.threads = 4;
		check(NnueRelabel.relabelFast(src, nn, 3, 1000, o1, null).equals(NnueRelabel.relabelFast(src, nn, 3, 1000, o4, null)), "relabelFast: stesso risultato con 1 e con 4 thread (stesso chunk)");
		final NnueRelabel.Options op = new NnueRelabel.Options();
		op.pruning = PruningOptions.parse("all");
		op.chunk = 50;
		final NnueDataset pr = NnueRelabel.relabelFast(src, nn, 4, 1000, op, null), un = NnueRelabel.relabelFast(src, nn, 4, 1000, new NnueRelabel.Options(), null);
		double se = 0;
		for (int i = 0; i < src.n; i++) {
			se += (pr.value(i) - un.value(i)) * (pr.value(i) - un.value(i));
		}
		check(pr.n == src.n && Math.sqrt(se / src.n) < 0.2, "relabelFast con potatura: valori vicini a quelli senza (RMS " + Math.sqrt(se / src.n) + ")");
		final NnueRelabel.Options ev = new NnueRelabel.Options();
		ev.every = 2;
		ev.chunk = 9;
		final NnueDataset e2 = NnueRelabel.relabelFast(src, nn, 3, 1000, ev, null);
		boolean evOk = true;
		for (int i = 0; i < src.n; i++) {
			evOk &= i % 2 == 0 ? Math.abs(e2.value(i) - legacy.value(i)) < 0.25 : e2.value(i) == src.value(i);
		}
		check(evOk, "every=2: solo le posizioni pari cambiano");
		ev.keepUnlabelled = false;
		final NnueDataset e2k = NnueRelabel.relabelFast(src, nn, 3, 1000, ev, null);
		check(e2k.n == (src.n + 1) / 2, "keep=false: si scrivono solo le posizioni etichettate da questa esecuzione (" + e2k.n + " su " + src.n + ", attese " + (src.n + 1) / 2 + ")");
		final NnueRelabel.Options sk = new NnueRelabel.Options();
		sk.skipSat = 0.0000001; // tutto cio' che ha gia' un valore non nullo viene saltato
		sk.chunk = 11;
		final NnueDataset sks = NnueRelabel.relabelFast(src, nn, 3, 1000, sk, null);
		boolean skOk = true;
		for (int i = 0; i < src.n; i++) {
			skOk &= Math.abs(src.value(i)) >= 0.0000001 ? sks.value(i) == src.value(i) : true;
		}
		check(skOk, "skipsat: le posizioni gia' sature non vengono ricercate");
		boolean threw = false;
		try {
			final NnueRelabel.Options badO = new NnueRelabel.Options();
			badO.chunk = 0;
			NnueRelabel.relabelFast(src, nn, 3, 1000, badO, null);
		} catch (IllegalArgumentException x) {
			threw = true;
		}
		check(threw, "chunk=0 rifiutato");

		System.out.println(failures == 0 ? "=== FAST DATAGEN CHECK OK: " + checks + " verifiche ===" : "=== FALLITO: " + failures + " ===");
		System.exit(failures == 0 ? 0 : 1);
	}
}
