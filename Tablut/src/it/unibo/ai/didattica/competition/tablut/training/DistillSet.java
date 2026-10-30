package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Insieme di posizioni per la distillazione della ricerca ({@link CnnDistiller}): scacchiera, chi muove, tutte le feature a mano,
 * l'esito della partita di provenienza (per la loss mista) e la PARTITA di provenienza.
 *
 * <h3>Regole che il resto del flusso da' per garantite</h3>
 * <ul>
 * <li><b>Solo posizioni non terminali</b>: partita non finita e chi muove ha almeno una mossa legale ({@link SearchLabeler#isTerminal}).</li>
 * <li><b>Train e held-out disgiunti PER PARTITA</b>: {@link #isHeldOut} decide da {@code (chiave di partita, sale)}, quindi tutte le
 * posizioni di una partita stanno dalla stessa parte (altrimenti posizioni consecutive, quasi identiche, finirebbero in entrambi).</li>
 * <li><b>Due sorgenti</b> ({@link #SRC_DATASET} = partite deboli di {@code dati.gz}, {@link #SRC_ENGINE} = partite dell'engine con un budget
 * di tempo reale): {@link #sample} ne prende meta' e meta' per quanto i dati lo permettono e dichiara lo scarto.</li>
 * <li>L'esito vale per la loss solo se {@code outValid}: le partite dell'engine troncate al tetto di semi-mosse non hanno un
 * esito e non lo imitano (le partite di {@code dati.gz} troncate sono gia' scartate alla generazione).</li>
 * </ul>
 */
public final class DistillSet {

	public static final int NF = TablutFeatures.NUM_FEATURES;
	public static final byte SRC_DATASET = 0, SRC_ENGINE = 1;
	/** Le chiavi di partita dell'engine partono da qui, per non collidere con gli indici di partita di {@code dati.gz}. */
	static final int ENGINE_BASE = 1 << 30;
	private static final String MAGIC = "TABLUT_DISTILL_SET v1";

	public final int n;
	/** 81 codici per posizione (0 vuoto, 1 bianco, 2 nero, 3 re). */
	public final byte[] cells;
	/** 0 = muove il Bianco. */
	public final byte[] role;
	/** Esito dal punto di vista del Bianco: +1, -1, 0. */
	public final byte[] outcome;
	/** 1 se {@code outcome} e' un vero esito di partita. */
	public final byte[] outValid;
	/** Chiave di partita (unica tra le sorgenti). */
	public final int[] game;
	public final byte[] source;
	public final boolean[] heldOut;
	/** Tutte le {@value TablutFeatures#NUM_FEATURES} feature, riga per posizione. */
	public final float[] feats;

	public DistillSet(byte[] cells, byte[] role, byte[] outcome, byte[] outValid, int[] game, byte[] source, boolean[] heldOut, float[] feats) {
		this.n = role.length;
		if (cells.length != n * 81 || outcome.length != n || outValid.length != n || game.length != n || source.length != n || heldOut.length != n
				|| feats.length != n * NF) {
			throw new IllegalArgumentException("dimensioni incoerenti nel DistillSet");
		}
		this.cells = cells;
		this.role = role;
		this.outcome = outcome;
		this.outValid = outValid;
		this.game = game;
		this.source = source;
		this.heldOut = heldOut;
		this.feats = feats;
	}

	// ====================================================================== split per partita

	/** Hash deterministico (SplitMix64) di {@code (chiave di partita, sale)} in [0,100): la partita e' held-out se < {@code pct}. */
	public static boolean isHeldOut(int gameKey, long salt, int pct) {
		long x = (gameKey + 1L) * 0x9E3779B97F4A7C15L + salt * 0xBF58476D1CE4E5B9L;
		x ^= x >>> 30;
		x *= 0xBF58476D1CE4E5B9L;
		x ^= x >>> 27;
		x *= 0x94D049BB133111EBL;
		x ^= x >>> 31;
		return Long.remainderUnsigned(x, 100) < pct;
	}

	/** Indici delle posizioni di train ({@code held == false}) o di held-out. */
	public int[] indices(boolean held) {
		int c = 0;
		for (int i = 0; i < n; i++) {
			if (heldOut[i] == held) {
				c++;
			}
		}
		int[] r = new int[c];
		int k = 0;
		for (int i = 0; i < n; i++) {
			if (heldOut[i] == held) {
				r[k++] = i;
			}
		}
		return r;
	}

	/** Indici della sorgente {@code src} nella parte {@code held}. */
	public int[] indices(boolean held, byte src) {
		int c = 0;
		for (int i = 0; i < n; i++) {
			if (heldOut[i] == held && source[i] == src) {
				c++;
			}
		}
		int[] r = new int[c];
		int k = 0;
		for (int i = 0; i < n; i++) {
			if (heldOut[i] == held && source[i] == src) {
				r[k++] = i;
			}
		}
		return r;
	}

	/** Vero se nessuna partita compare sia nel train sia nell'held-out. */
	public boolean splitIsDisjointByGame() {
		java.util.Set<Integer> tr = new java.util.HashSet<>(), he = new java.util.HashSet<>();
		for (int i = 0; i < n; i++) {
			(heldOut[i] ? he : tr).add(game[i]);
		}
		tr.retainAll(he);
		return tr.isEmpty();
	}

	// ====================================================================== costruzione

	/** Raccoglitore incrementale di posizioni. */
	static final class Builder {
		final List<byte[]> cells = new ArrayList<>();
		final List<float[]> feats = new ArrayList<>();
		final List<int[]> meta = new ArrayList<>(); // {role, outcome, valid, game, source}
		/** Solo diagnostica (partite dell'engine): etichetta di fine partita di {@link GameTracker}. */
		String endLabel = "";

		void add(byte[] c, float[] f, int role, int outcome, boolean valid, int game, int source) {
			cells.add(c);
			feats.add(f);
			meta.add(new int[] { role, outcome, valid ? 1 : 0, game, source });
		}

		DistillSet build(long salt, int heldPct) {
			int n = cells.size();
			byte[] c = new byte[n * 81], ro = new byte[n], oc = new byte[n], ov = new byte[n], so = new byte[n];
			int[] g = new int[n];
			boolean[] h = new boolean[n];
			float[] f = new float[n * NF];
			for (int i = 0; i < n; i++) {
				System.arraycopy(cells.get(i), 0, c, i * 81, 81);
				System.arraycopy(feats.get(i), 0, f, i * NF, NF);
				int[] m = meta.get(i);
				ro[i] = (byte) m[0];
				oc[i] = (byte) m[1];
				ov[i] = (byte) m[2];
				g[i] = m[3];
				so[i] = (byte) m[4];
				h[i] = isHeldOut(m[3], salt, heldPct);
			}
			return new DistillSet(c, ro, oc, ov, g, so, h, f);
		}
	}

	/** Carica un dataset di {@link CnnPretrainer} ({@code dati.gz}); se e' di un'altra versione di feature lo aggiorna ({@link DatasetUpgrade}). */
	public static CnnPretrainer.Dataset loadDataset(String path) throws IOException {
		try {
			return CnnPretrainer.Dataset.load(path);
		} catch (IOException versionMismatch) {
			System.out.println("(" + versionMismatch.getMessage() + " -> ricalcolo le feature con DatasetUpgrade)");
			return DatasetUpgrade.loadAndRecompute(path);
		}
	}

	/** Tutte le posizioni NON terminali di un dataset di {@link CnnPretrainer}, sorgente {@link #SRC_DATASET}. */
	public static DistillSet fromDataset(CnnPretrainer.Dataset d, long salt, int heldPct) {
		Builder b = new Builder();
		for (int i = 0; i < d.n; i++) {
			if (SearchLabeler.isTerminal(SearchLabeler.fastStateFromCodes(d.cells, i * 81, d.role[i]))) {
				continue;
			}
			byte[] c = new byte[81];
			float[] f = new float[NF];
			System.arraycopy(d.cells, i * 81, c, 0, 81);
			System.arraycopy(d.feats, i * NF, f, 0, NF);
			b.add(c, f, d.role[i], d.outcome[i], true, d.game[i], SRC_DATASET);
		}
		return b.build(salt, heldPct);
	}

	// ====================================================================== partite dell'engine

	/**
	 * Gioca {@code games} partite dell'ENGINE (Negamax con {@code ev} da entrambi i lati, 1 thread per lato, {@code msPerMove} ms
	 * REALI per mossa: margine di sicurezza 0) e ne raccoglie le posizioni non terminali dal ply 4 in poi (i primi 4 sono un'apertura
	 * casuale per diversificare). Una mossa su {@code 1/eps} e' casuale. Le partite si giocano in parallelo (una per thread). Sorgente
	 * {@link #SRC_ENGINE}; chiavi di partita da {@link #ENGINE_BASE}.
	 */
	public static DistillSet generateEngineGames(final Evaluator ev, int games, final long msPerMove, final double eps, long seed, int threads, long salt,
			int heldPct, PrintStream log) throws Exception {
		ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
		Builder b = new Builder();
		try {
			List<Future<Builder>> fs = new ArrayList<>();
			Random master = new Random(seed * 131 + 17);
			for (int g = 0; g < games; g++) {
				final long gs = master.nextLong();
				final int key = ENGINE_BASE + g;
				fs.add(pool.submit(new Callable<Builder>() {
					@Override
					public Builder call() {
						return playEngineGame(ev, msPerMove, eps, gs, key);
					}
				}));
			}
			for (int g = 0; g < fs.size(); g++) {
				Builder r = fs.get(g).get();
				b.cells.addAll(r.cells);
				b.feats.addAll(r.feats);
				b.meta.addAll(r.meta);
				if (log != null && (g + 1) % 5 == 0) {
					log.println("  partite engine " + (g + 1) + "/" + games);
				}
			}
		} finally {
			pool.shutdown();
		}
		return b.build(salt, heldPct);
	}

	static Builder playEngineGame(Evaluator ev, long msPerMove, double eps, long gameSeed, int key) {
		Random rnd = new Random(gameSeed);
		Negamax nw = new Negamax(ev, 64, true, 1 << 18, true, Negamax.ORDER_DEFAULT, 1, false);
		Negamax nb = new Negamax(ev, 64, true, 1 << 18, true, Negamax.ORDER_DEFAULT, 1, false);
		nw.setSafetyMarginMillis(0);
		nb.setSafetyMarginMillis(0);
		TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
		double[] full = new double[NF];
		List<byte[]> cells = new ArrayList<>();
		List<float[]> feats = new ArrayList<>();
		List<Integer> roles = new ArrayList<>();
		State s = new StateTablut();
		s.setTurn(Turn.WHITE);
		GameTracker game = new GameTracker(s);
		for (int ply = 0; ply < 200 && !AshtonRules.isTerminal(s); ply++) {
			List<Action> mv = AshtonRules.generateMoves(s);
			if (mv.isEmpty()) {
				break;
			}
			if (ply >= 4) {
				TablutFeatures.extract(s, ws, full, TablutFeatures.G_ALL);
				float[] f = new float[NF];
				for (int i = 0; i < NF; i++) {
					f[i] = (float) full[i];
				}
				cells.add(CnnPretrainer.codesOf(s));
				feats.add(f);
				roles.add(NeuralEvaluator.roleOf(s));
			}
			Action m;
			if (ply < 4 || rnd.nextDouble() < eps) {
				m = mv.get(rnd.nextInt(mv.size()));
			} else {
				m = (s.getTurn() == Turn.WHITE ? nw : nb).findBestMove(s, msPerMove, game.history());
				if (m == null) {
					m = mv.get(rnd.nextInt(mv.size()));
				}
			}
			s = game.apply(s, m);
			if (game.isRepetitionDraw()) {
				break;
			}
		}
		GameTracker.Result end = game.result(s);
		boolean valid = !GameTracker.LABEL_MAXPLIES.equals(end.label);
		Builder b = new Builder();
		b.endLabel = end.label;
		for (int i = 0; i < cells.size(); i++) {
			b.add(cells.get(i), feats.get(i), roles.get(i), (int) end.outcome, valid, key, SRC_ENGINE);
		}
		return b;
	}

	// ====================================================================== campionamento meta' e meta'

	/**
	 * Estrae da {@code a} (dataset) e {@code b} (engine) un nuovo insieme di {@code nTrain} posizioni di train e {@code nHeld} di
	 * held-out, per quanto possibile meta' per sorgente (le sorgenti sono quelle di {@code a} e {@code b}; l'assegnazione train/held-out
	 * e' gia' per partita). Se una sorgente non basta, il resto e' preso dall'altra. Una sorgente nulla = si usa l'altra.
	 *
	 * @return l'insieme campionato; {@code report[0]} riceve una descrizione della composizione
	 */
	public static DistillSet sample(DistillSet a, DistillSet b, int nTrain, int nHeld, long seed, String[] report) {
		Random rnd = new Random(seed);
		List<Integer> fromA = new ArrayList<>(), fromB = new ArrayList<>();
		StringBuilder rep = new StringBuilder();
		for (int part = 0; part < 2; part++) {
			boolean held = part == 1;
			int want = held ? nHeld : nTrain;
			int[] ia = a == null ? new int[0] : shuffled(a.indices(held), rnd);
			int[] ib = b == null ? new int[0] : shuffled(b.indices(held), rnd);
			int takeA = Math.min(ia.length, want / 2), takeB = Math.min(ib.length, want - want / 2);
			// resto: se una sorgente e' scarsa, l'altra compensa (il totale resta, la composizione si dichiara)
			int missing = want - takeA - takeB;
			int extraA = Math.min(ia.length - takeA, Math.max(0, missing));
			takeA += extraA;
			missing -= extraA;
			int extraB = Math.min(ib.length - takeB, Math.max(0, missing));
			takeB += extraB;
			for (int k = 0; k < takeA; k++) {
				fromA.add(ia[k]);
			}
			for (int k = 0; k < takeB; k++) {
				fromB.add(ib[k]);
			}
			rep.append(held ? "held-out" : "train").append(": ").append(takeA).append(" da dataset + ").append(takeB).append(" da engine (richieste ")
					.append(want).append(")").append(held ? "" : " | ");
		}
		Builder out = new Builder();
		for (int i : fromA) {
			out.add(slice(a.cells, i * 81, 81), sliceF(a.feats, i * NF, NF), a.role[i], a.outcome[i], a.outValid[i] == 1, a.game[i], a.source[i]);
		}
		for (int i : fromB) {
			out.add(slice(b.cells, i * 81, 81), sliceF(b.feats, i * NF, NF), b.role[i], b.outcome[i], b.outValid[i] == 1, b.game[i], b.source[i]);
		}
		int n = out.cells.size();
		// il flag held-out va conservato dalla sorgente: si ricostruisce dagli indici scelti
		boolean[] held = new boolean[n];
		int k = 0;
		for (int i : fromA) {
			held[k++] = a.heldOut[i];
		}
		for (int i : fromB) {
			held[k++] = b.heldOut[i];
		}
		DistillSet built = out.build(0, 0);
		DistillSet res = new DistillSet(built.cells, built.role, built.outcome, built.outValid, built.game, built.source, held, built.feats);
		if (report != null && report.length > 0) {
			report[0] = rep.toString();
		}
		return res;
	}

	private static int[] shuffled(int[] a, Random rnd) {
		int[] r = a.clone();
		for (int i = r.length - 1; i > 0; i--) {
			int j = rnd.nextInt(i + 1);
			int t = r[i];
			r[i] = r[j];
			r[j] = t;
		}
		return r;
	}

	private static byte[] slice(byte[] a, int off, int len) {
		byte[] r = new byte[len];
		System.arraycopy(a, off, r, 0, len);
		return r;
	}

	private static float[] sliceF(float[] a, int off, int len) {
		float[] r = new float[len];
		System.arraycopy(a, off, r, 0, len);
		return r;
	}

	// ====================================================================== file

	public void save(String path) throws IOException {
		try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path)), 1 << 16))) {
			o.writeUTF(MAGIC);
			o.writeInt(TablutFeatures.VERSION);
			o.writeInt(NF);
			o.writeInt(n);
			o.write(cells);
			o.write(role);
			o.write(outcome);
			o.write(outValid);
			o.write(source);
			for (int i = 0; i < n; i++) {
				o.writeInt(game[i]);
				o.writeBoolean(heldOut[i]);
			}
			for (float f : feats) {
				o.writeFloat(f);
			}
		}
	}

	public static DistillSet load(String path) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16))) {
			if (!MAGIC.equals(in.readUTF())) {
				throw new IOException("non e' un DistillSet: " + path);
			}
			int fv = in.readInt(), nf = in.readInt(), n = in.readInt();
			if (fv != TablutFeatures.VERSION || nf != NF) {
				throw new IOException("DistillSet con feature v" + fv + "/" + nf + ", il codice usa v" + TablutFeatures.VERSION + "/" + NF + ": rigenerarlo");
			}
			if (n < 0 || n > 50_000_000) {
				throw new IOException("numero di posizioni non plausibile: " + n);
			}
			byte[] cells = new byte[n * 81], role = new byte[n], oc = new byte[n], ov = new byte[n], so = new byte[n];
			int[] g = new int[n];
			boolean[] h = new boolean[n];
			float[] f = new float[n * NF];
			in.readFully(cells);
			in.readFully(role);
			in.readFully(oc);
			in.readFully(ov);
			in.readFully(so);
			for (int i = 0; i < n; i++) {
				g[i] = in.readInt();
				h[i] = in.readBoolean();
			}
			for (int i = 0; i < f.length; i++) {
				f[i] = in.readFloat();
			}
			return new DistillSet(cells, role, oc, ov, g, so, h, f);
		}
	}

	/** Riepilogo di composizione, per i log. */
	public String describe() {
		int[][] c = new int[2][2];
		for (int i = 0; i < n; i++) {
			c[heldOut[i] ? 1 : 0][source[i]]++;
		}
		return n + " posizioni | train: " + c[0][0] + " dataset + " + c[0][1] + " engine | held-out: " + c[1][0] + " dataset + " + c[1][1] + " engine";
	}
}
