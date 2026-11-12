package it.unibo.ai.didattica.competition.tablut.training;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;

/**
 * Test del formato dati ({@link NnueDataset}) e degli importatori ({@link NnueDataImport}).
 * <ul>
 * <li><b>Formato</b>: ogni invariante rifiutata con il suo motivo; andata e ritorno esatti e deterministici; ogni forma di file corrotto (assente,
 * non gzip, altro formato/versione, troncato, byte alterati, byte in piu', campioni non validi, numero di campioni assurdo) e' una {@code IOException};
 * unione con rinumerazione delle partite; split per partita.</li>
 * <li><b>Log del server</b>: SINTETICI nel formato esatto del server (anche con fine riga CRLF, Re catturato lasciato sulla scacchiera, blocco di
 * {@code endGame} duplicato): vittoria Bianca e Nera, partita interrotta, forfeit con mossa nulla, forfeit con esito non terminale, mossa illegale
 * rifiutata e mossa illegale applicata, divergenza della scacchiera, patta del server, mosse rifiutate, blocco malformato, doppioni, cartelle. Le
 * posizioni importate sono confrontate con quelle della partita giocata dal motore storico.</li>
 * <li><b>Log VERI</b> (cartella {@code logs/}): nessuna divergenza dalle regole.</li>
 * <li><b>dati.gz e DistillSet</b>: scritti con i writer ORIGINALI del progetto e riletti dagli importatori (Re 3 -> 4, esiti non validi scartati).</li>
 * </ul>
 */
public class NnueDataCheck {

	private static int failures = 0;
	private static long checks = 0;

	private static void check(boolean ok, String msg) {
		checks++;
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	private static void expectIae(Runnable r, String keyword, String label) {
		try {
			r.run();
			check(false, label + ": accettato, doveva essere rifiutato");
		} catch (IllegalArgumentException e) {
			check(keyword == null || String.valueOf(e.getMessage()).contains(keyword), label + ": messaggio '" + e.getMessage() + "' non contiene '" + keyword + "'");
		}
	}

	private static File tmp(String suffix) throws IOException {
		final File f = File.createTempFile("nnudata_", suffix);
		f.deleteOnExit();
		return f;
	}

	public static void main(String[] args) throws Exception {
		testCheckPosition();
		testInvariants();
		testBuilder();
		testIo();
		testLoadErrors();
		testMergeSubsetSplit();
		testParserAndReplay();
		testDirectoryImport();
		testRealLogs();
		testLegacyImporters();
		testDataTool();
		System.out.println(failures == 0 ? "=== NNUE DATA CHECK OK: " + checks + " verifiche ===" : "=== NNUE DATA CHECK: " + failures + " FALLIMENTI su " + checks + " ===");
		if (failures != 0) {
			System.exit(1);
		}
	}

	// ====================================================================== dati di prova

	private static byte[] initial() {
		return new FastState().board.clone();
	}

	/** Un dataset valido di {@code n} posizioni casuali con tutte le combinazioni di etichette. */
	private static NnueDataset randomDataset(Random rnd, int n, int games) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		while (b.size() < n) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			if (NnueDataset.checkPosition(fs.board, 0) != null) {
				continue;
			}
			final int i = b.size();
			b.add(fs.board, 0, fs.turn == BoardConstants.BLACK_TURN ? 1 : 0, rnd.nextInt(3) - 1, i % 4 != 0, rnd.nextDouble() * 2 - 1, i % 3 != 0, i % games, i % 200, 1 + i % 50,
					i % 3);
		}
		return b.build();
	}

	// ====================================================================== posizioni

	private static void testCheckPosition() {
		check(NnueDataset.checkPosition(initial(), 0) == null, "la posizione iniziale deve essere valida");
		final Random rnd = new Random(1);
		int valid = 0;
		for (int i = 0; i < 300; i++) {
			final FastState fs = FastState.fromState(TestPositions.randomPosition(rnd));
			if (NnueDataset.checkPosition(fs.board, 0) == null) {
				valid++;
			}
		}
		check(valid > 200, "le posizioni casuali valide devono essere la maggior parte (" + valid + "/300)");

		byte[] b = initial();
		b[10] = 3;
		final String code3 = NnueDataset.checkPosition(b, 0);
		check(code3 != null && code3.contains("codice"), "il codice 3 (Re dei dati legacy) va rifiutato: " + code3);
		b = initial();
		b[10] = 5;
		check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("codice"), "codice 5 sconosciuto");
		b = initial();
		b[10] = -1;
		check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("codice"), "codice negativo");
		b = initial();
		b[BoardConstants.THRONE] = BoardConstants.EMPTY;
		check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("Re assente"), "senza Re (partita finita)");
		b = initial();
		b[0] = BoardConstants.KING;
		check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("piu' di un Re"), "due Re");
		for (int sq : new int[] { 0 * 9 + 2, 8 * 9 + 2, 2 * 9 + 0, 2 * 9 + 8 }) {
			b = initial();
			b[BoardConstants.THRONE] = BoardConstants.EMPTY;
			b[sq] = BoardConstants.KING;
			check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("bordo"), "Re sul bordo (cella " + sq + ") e' terminale");
		}
		for (byte piece : new byte[] { BoardConstants.WHITE, BoardConstants.BLACK }) {
			// il Re sta altrove (in d4): l'unica anomalia e' la pedina sul trono, e il motivo deve essere proprio quello
			b = initial();
			b[3 * 9 + 3] = BoardConstants.KING;
			b[BoardConstants.THRONE] = piece;
			final String why = NnueDataset.checkPosition(b, 0);
			check(why != null && why.contains("sul trono"), (piece == BoardConstants.WHITE ? "bianco" : "nero") + " sul trono: " + why);
		}
		b = initial();
		int added = 0;
		for (int sq = 0; sq < 81 && added < 2; sq++) {
			if (b[sq] == BoardConstants.EMPTY) {
				b[sq] = BoardConstants.WHITE;
				added++;
			}
		}
		check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("troppe pedine"), "10 bianchi");
		b = initial();
		added = 0;
		for (int sq = 0; sq < 81 && added < 2; sq++) {
			if (b[sq] == BoardConstants.EMPTY) {
				b[sq] = BoardConstants.BLACK;
				added++;
			}
		}
		check(String.valueOf(NnueDataset.checkPosition(b, 0)).contains("troppe pedine"), "18 neri");
		// l'offset: una posizione valida in coda a un buffer piu' grande
		final byte[] two = new byte[162];
		System.arraycopy(initial(), 0, two, 81, 81);
		check(NnueDataset.checkPosition(two, 81) == null && NnueDataset.checkPosition(two, 0) != null, "checkPosition rispetta l'offset");
	}

	// ====================================================================== invarianti del dataset

	private static NnueDataset one(byte[] board, byte role, byte outcome, byte outValid, float value, byte valValid, int game, short ply, short toEnd, byte source) {
		return new NnueDataset(board, new byte[] { role }, new byte[] { outcome }, new byte[] { outValid }, new float[] { value }, new byte[] { valValid }, new int[] { game },
				new short[] { ply }, new short[] { toEnd }, new byte[] { source });
	}

	private static void testInvariants() {
		final byte[] ok = initial();
		check(one(ok, (byte) 1, (byte) -1, (byte) 1, 0.5f, (byte) 1, 7, (short) 3, (short) 9, (byte) 2).n == 1, "un campione valido deve passare");
		check(one(ok, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 0, 0, (short) -1, (short) -1, (byte) 0).n == 1, "senza etichette, con ply ignoto: valido");
		expectIae(() -> one(ok, (byte) 2, (byte) 0, (byte) 0, 0f, (byte) 0, 0, (short) 0, (short) -1, (byte) 0), "lato", "ruolo 2");
		expectIae(() -> one(ok, (byte) 0, (byte) 2, (byte) 1, 0f, (byte) 0, 0, (short) 0, (short) 1, (byte) 0), "esito", "esito 2");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 2, 0f, (byte) 0, 0, (short) 0, (short) -1, (byte) 0), "outValid", "outValid 2");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 2, 0, (short) 0, (short) -1, (byte) 0), "valValid", "valValid 2");
		expectIae(() -> one(ok, (byte) 0, (byte) 1, (byte) 0, 0f, (byte) 0, 0, (short) 0, (short) -1, (byte) 0), "canonica", "esito non nullo con outValid 0");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 0.3f, (byte) 0, 0, (short) 0, (short) -1, (byte) 0), "canonica", "valore non nullo con valValid 0");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, Float.NaN, (byte) 1, 0, (short) 0, (short) -1, (byte) 0), "fuori", "valore NaN");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 1.5f, (byte) 1, 0, (short) 0, (short) -1, (byte) 0), "fuori", "valore 1.5");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, -1.01f, (byte) 1, 0, (short) 0, (short) -1, (byte) 0), "fuori", "valore -1.01");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 0, -1, (short) 0, (short) -1, (byte) 0), "partita", "chiave di partita negativa");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 0, 0, (short) -2, (short) -1, (byte) 0), "ply", "ply -2");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 1, 0f, (byte) 0, 0, (short) 0, (short) -2, (byte) 0), "toEnd", "toEnd -2");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 0, 0, (short) 0, (short) 5, (byte) 0), "toEnd", "toEnd senza esito valido");
		expectIae(() -> one(ok, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 0, 0, (short) 0, (short) -1, (byte) 16), "sorgente", "sorgente 16");
		final byte[] bad = initial();
		bad[10] = 3;
		expectIae(() -> one(bad, (byte) 0, (byte) 0, (byte) 0, 0f, (byte) 0, 0, (short) 0, (short) -1, (byte) 0), "codice", "posizione non valida dentro il dataset");
		// il numero del campione sbagliato e' nel messaggio
		try {
			new NnueDataset(concat(initial(), bad), new byte[2], new byte[2], new byte[2], new float[2], new byte[2], new int[2], new short[] { 0, 0 }, new short[] { -1, -1 }, new byte[2]);
			check(false, "il secondo campione non valido deve essere rifiutato");
		} catch (IllegalArgumentException e) {
			check(e.getMessage().contains("campione 1"), "il messaggio deve dire quale campione: " + e.getMessage());
		}
		expectIae(() -> new NnueDataset(initial(), new byte[2], new byte[2], new byte[2], new float[2], new byte[2], new int[2], new short[2], new short[2], new byte[2]), "lunghezza",
				"scacchiere di lunghezza sbagliata");
		expectIae(() -> new NnueDataset(initial(), new byte[1], new byte[1], new byte[1], new float[2], new byte[1], new int[1], new short[1], new short[1], new byte[1]), "lunghezza",
				"colonna di lunghezza sbagliata");
		expectIae(() -> new NnueDataset(null, new byte[1], new byte[1], new byte[1], new float[1], new byte[1], new int[1], new short[1], new short[1], new byte[1]), "lunghezza", "colonna nulla");
	}

	private static byte[] concat(byte[] a, byte[] b) {
		final byte[] c = new byte[a.length + b.length];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	private static void testBuilder() {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final byte[] pos = initial();
		for (int i = 0; i < 1000; i++) { // oltre la capacita' iniziale: la crescita non deve perdere nulla
			b.add(pos, 0, i % 2, 1, i % 5 != 0, 0.7, i % 7 != 0, i / 10, i, 1000 - i, NnueDataset.SRC_SELFPLAY);
		}
		final NnueDataset d = b.build();
		check(d.n == 1000 && b.size() == 1000, "il Builder deve contenere 1000 campioni");
		boolean ok = true;
		for (int i = 0; i < d.n; i++) {
			ok &= d.role[i] == i % 2 && d.game[i] == i / 10 && d.ply[i] == i && d.source[i] == NnueDataset.SRC_SELFPLAY;
			ok &= (d.outValid[i] == 1) == (i % 5 != 0) && d.outcome[i] == (i % 5 != 0 ? 1 : 0);
			ok &= (d.valValid[i] == 1) == (i % 7 != 0) && d.value[i] == (i % 7 != 0 ? 0.7f : 0f);
			ok &= d.toEnd[i] == (i % 5 != 0 ? 1000 - i : -1);
		}
		check(ok, "il Builder deve conservare i campioni e portare alla forma canonica le etichette non valide");
		// ply e toEnd fuori dall'intervallo short sono limitati, non avvolti
		final NnueDataset.Builder b2 = new NnueDataset.Builder();
		b2.add(pos, 0, 0, 1, true, 0, false, 0, 100000, 100000, 0);
		final NnueDataset d2 = b2.build();
		check(d2.ply[0] == Short.MAX_VALUE && d2.toEnd[0] == Short.MAX_VALUE, "ply/toEnd enormi vanno limitati a Short.MAX_VALUE");
		check(new NnueDataset.Builder().build().n == 0, "un Builder vuoto da' un dataset vuoto");
	}

	// ====================================================================== file

	private static void testIo() throws IOException {
		final NnueDataset d = randomDataset(new Random(2), 400, 37);
		final File f = tmp(".gz"), f2 = tmp(".gz");
		d.save(f.getPath());
		final NnueDataset back = NnueDataset.load(f.getPath());
		check(back.equals(d), "andata e ritorno: il dataset letto deve essere identico");
		check(back.describe().equals(d.describe()), "describe identico");
		back.save(f2.getPath());
		check(Arrays.equals(Files.readAllBytes(f.toPath()), Files.readAllBytes(f2.toPath())), "il salvataggio deve essere deterministico: stessi byte");
		final NnueDataset empty = new NnueDataset.Builder().build();
		empty.save(f2.getPath());
		check(NnueDataset.load(f2.getPath()).n == 0, "un dataset vuoto si salva e si rilegge");
		check(d.describe().contains("400 posizioni") && d.numGames() == 37, "describe/numGames: " + d.describe() + " / " + d.numGames());
	}

	/** Scrive un file nel formato, riproducendolo a mano: serve per costruire file con difetti mirati ma gzip valido. */
	private static void writeRaw(File f, String magic, int n, byte[] boards, byte[] role, byte[] outcome, byte[] outValid, byte[] valValid, byte[] source, int extraBytes) throws IOException {
		try (DataOutputStream o = new DataOutputStream(new GZIPOutputStream(Files.newOutputStream(f.toPath())))) {
			o.writeUTF(magic);
			o.writeInt(n);
			o.write(boards);
			o.write(role);
			o.write(outcome);
			o.write(outValid);
			o.write(valValid);
			o.write(source);
			for (int i = 0; i < n; i++) {
				o.writeInt(i);
			}
			for (int i = 0; i < n; i++) {
				o.writeShort(i);
			}
			for (int i = 0; i < n; i++) {
				o.writeShort(-1);
			}
			for (int i = 0; i < n; i++) {
				o.writeFloat(0f);
			}
			for (int i = 0; i < extraBytes; i++) {
				o.writeByte(7);
			}
		}
	}

	private static void expectIo(File f, String keyword, String label) {
		try {
			NnueDataset.load(f.getPath());
			check(false, label + ": il file errato e' stato accettato");
		} catch (IOException e) {
			check(keyword == null || String.valueOf(e.getMessage()).contains(keyword), label + ": messaggio '" + e.getMessage() + "' non contiene '" + keyword + "'");
		} catch (RuntimeException e) {
			check(false, label + ": eccezione non gestita " + e);
		}
	}

	private static void testLoadErrors() throws IOException {
		final NnueDataset d = randomDataset(new Random(3), 200, 20);
		final File good = tmp(".gz");
		d.save(good.getPath());
		final byte[] bytes = Files.readAllBytes(good.toPath());

		expectIo(new File("/nonexistent_dir_xyz/none.gz"), null, "file inesistente");
		expectIo(new File(System.getProperty("java.io.tmpdir")), null, "directory");
		File f = tmp(".gz");
		Files.write(f.toPath(), new byte[0]);
		expectIo(f, null, "file vuoto");
		Files.write(f.toPath(), "questo non e' gzip, e' testo".getBytes("UTF-8"));
		expectIo(f, "non valido", "file non gzip");
		// gzip valido con un'altra intestazione / versione
		final byte[] one = initial();
		f = tmp(".gz");
		writeRaw(f, "TABLUT_CNN_DATA v1", 1, one, new byte[1], new byte[1], new byte[1], new byte[1], new byte[1], 0);
		expectIo(f, "non e' un dataset NNUE", "intestazione di un altro formato");
		writeRaw(f, "TABLUT_NNUE_DATA v2", 1, one, new byte[1], new byte[1], new byte[1], new byte[1], new byte[1], 0);
		expectIo(f, "versione", "versione sconosciuta");
		// file valido scritto a mano: serve da controllo del writer di prova
		writeRaw(f, NnueDataset.MAGIC, 1, one, new byte[1], new byte[1], new byte[1], new byte[1], new byte[1], 0);
		try {
			check(NnueDataset.load(f.getPath()).n == 1, "il file scritto a mano (controllo) deve essere valido");
		} catch (IOException e) {
			check(false, "il file di controllo scritto a mano non si legge: " + e);
		}
		// byte in piu' dopo i dati
		writeRaw(f, NnueDataset.MAGIC, 1, one, new byte[1], new byte[1], new byte[1], new byte[1], new byte[1], 3);
		expectIo(f, "in piu'", "byte in piu' dopo i dati");
		// campione non valido dentro un file gzip valido
		final byte[] badBoard = initial();
		badBoard[10] = 3;
		writeRaw(f, NnueDataset.MAGIC, 1, badBoard, new byte[1], new byte[1], new byte[1], new byte[1], new byte[1], 0);
		expectIo(f, "non validi", "campione non valido");
		writeRaw(f, NnueDataset.MAGIC, 1, one, new byte[] { 5 }, new byte[1], new byte[1], new byte[1], new byte[1], 0);
		expectIo(f, "lato", "ruolo non valido nel file");
		// numero di campioni assurdo / negativo
		for (int n : new int[] { -1, NnueDataset.MAX_SAMPLES + 1, Integer.MAX_VALUE }) {
			f = tmp(".gz");
			try (DataOutputStream o = new DataOutputStream(new GZIPOutputStream(Files.newOutputStream(f.toPath())))) {
				o.writeUTF(NnueDataset.MAGIC);
				o.writeInt(n);
			}
			expectIo(f, "plausibile", "n = " + n);
		}
		// troncato: il flusso gzip finisce a meta'
		f = tmp(".gz");
		Files.write(f.toPath(), Arrays.copyOf(bytes, bytes.length / 2));
		expectIo(f, "troncato", "file troncato a meta'");
		Files.write(f.toPath(), Arrays.copyOf(bytes, bytes.length - 3));
		expectIo(f, null, "file troncato negli ultimi byte (CRC di gzip)");
		// alterazione di un byte del flusso compresso, in piu' punti: ogni alterazione deve essere rilevata (CRC o decodifica)
		int detected = 0, tried = 0;
		for (int pos = 30; pos < bytes.length - 8; pos += Math.max(1, bytes.length / 40)) {
			final byte[] c = bytes.clone();
			c[pos] ^= 0x5A;
			Files.write(f.toPath(), c);
			tried++;
			try {
				NnueDataset.load(f.getPath());
			} catch (IOException e) {
				detected++;
			}
		}
		check(tried >= 30 && detected == tried, "ogni byte alterato del flusso deve essere rilevato: " + detected + "/" + tried);
		// alterazione del trailer CRC
		final byte[] c = bytes.clone();
		c[c.length - 8] ^= 0x01;
		Files.write(f.toPath(), c);
		expectIo(f, null, "CRC di gzip alterato");
	}

	// ====================================================================== unione, sottoinsieme, split

	private static void testMergeSubsetSplit() {
		final NnueDataset a = randomDataset(new Random(4), 60, 6), b = randomDataset(new Random(5), 40, 5); // entrambi con chiavi 0..5 / 0..4: collidono
		final NnueDataset m = NnueDataset.merge(Arrays.asList(a, b));
		check(m.n == 100, "unione: 100 campioni");
		check(m.numGames() == a.numGames() + b.numGames(), "unione: le partite con la stessa chiave in file diversi restano distinte (" + m.numGames() + " vs " + (a.numGames() + b.numGames()) + ")");
		boolean contiguous = true, preserved = true;
		for (int i = 0; i < m.n; i++) {
			contiguous &= m.game[i] >= 0 && m.game[i] < m.numGames();
			final NnueDataset src = i < 60 ? a : b;
			final int j = i < 60 ? i : i - 60;
			preserved &= m.role[i] == src.role[j] && m.value[i] == src.value[j] && m.outcome[i] == src.outcome[j] && m.source[i] == src.source[j] && m.ply[i] == src.ply[j];
			for (int k = 0; k < 81; k++) {
				preserved &= m.boards[81 * i + k] == src.boards[81 * j + k];
			}
		}
		check(contiguous, "unione: le chiavi di partita sono rinumerate densamente");
		check(preserved, "unione: i campioni si conservano nell'ordine dato");
		check(NnueDataset.merge(Arrays.asList(a)).numGames() == a.numGames() && NnueDataset.merge(new ArrayList<NnueDataset>()).n == 0, "unione di uno / di nessuno");
		// partite coerenti: due campioni della stessa partita di a restano nella stessa partita
		for (int i = 0; i < a.n; i++) {
			for (int j = i + 1; j < a.n; j++) {
				if ((a.game[i] == a.game[j]) != (m.game[i] == m.game[j])) {
					check(false, "unione: l'appartenenza alla stessa partita non si conserva");
					i = a.n;
					break;
				}
			}
		}

		final NnueDataset s = m.subset(new int[] { 5, 3, 99 });
		check(s.n == 3 && s.role[0] == m.role[5] && s.role[1] == m.role[3] && s.role[2] == m.role[99] && s.game[2] == m.game[99], "subset nell'ordine dato");
		check(m.subset(new int[0]).n == 0, "subset vuoto");

		// split per partita
		final int[][] sp = m.splitByGame(7, 30);
		final java.util.Set<Integer> tg = new java.util.HashSet<>(), hg = new java.util.HashSet<>();
		for (int i : sp[0]) {
			tg.add(m.game[i]);
		}
		for (int i : sp[1]) {
			hg.add(m.game[i]);
		}
		final java.util.Set<Integer> inter = new java.util.HashSet<>(tg);
		inter.retainAll(hg);
		check(inter.isEmpty() && sp[0].length + sp[1].length == m.n, "split: nessuna partita da entrambe le parti, nessun campione perso");
		check(Arrays.equals(sp[1], m.splitByGame(7, 30)[1]) && !Arrays.equals(sp[1], m.splitByGame(8, 30)[1]), "split: deterministico, e il sale lo cambia");
		check(m.splitByGame(7, 0)[1].length == 0 && m.splitByGame(7, 100)[0].length == 0, "split: 0% = nessun held-out, 100% = tutto held-out");
		int held = 0, none = 0, everything = 0;
		for (int g = 0; g < 10000; g++) {
			held += NnueDataset.isHeldOut(g, 3, 10) ? 1 : 0;
			none += NnueDataset.isHeldOut(g, 3, 0) ? 1 : 0;
			everything += NnueDataset.isHeldOut(g, 3, 100) ? 1 : 0;
		}
		check(none == 0 && everything == 10000, "isHeldOut: 0% non seleziona nulla, 100% tutto (" + none + ", " + everything + " su 10000)");
		check(held > 800 && held < 1200, "isHeldOut: circa il 10% delle partite (" + held + "/10000)");
	}

	// ====================================================================== log sintetici nel formato del server

	private static Action act(String from, String to, Turn t) {
		try {
			return new Action(from, to, t);
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
	}

	/** Un'intera partita di prova: stato di partenza, mosse e come il server la racconta. */
	private static String stateBlock(State s, String token, byte[] kingCellIfKept) {
		final StringBuilder sb = new StringBuilder("BUONO: Stato:\n");
		final byte[] codes = FastState.fromState(s).board;
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				byte code = codes[r * 9 + c];
				if (kingCellIfKept != null && r * 9 + c == kingCellIfKept[0]) {
					code = BoardConstants.KING;
				}
				sb.append(code == BoardConstants.WHITE ? 'W' : code == BoardConstants.BLACK ? 'B' : code == BoardConstants.KING ? 'K' : (r == 4 && c == 4 ? 'T' : 'O'));
			}
			sb.append('\n');
		}
		return sb.append("-\n").append(token).append('\n').toString();
	}

	private static String token(State s) {
		final Turn t = s.getTurn();
		return t == Turn.WHITE ? "W" : t == Turn.BLACK ? "B" : t == Turn.WHITEWIN ? "WW" : t == Turn.BLACKWIN ? "BW" : "D";
	}

	private static final String NOISE = "ott 29, 2026 2:33:27 PM it.unibo.ai.didattica.competition.tablut.server.GameAshtonTablut checkMove\n";

	/**
	 * Racconta una partita come il server: stato iniziale, per ogni mossa la riga {@code Turn:}, il blocco di stato, e a fine partita il blocco di
	 * {@code endGame} duplicato. {@code serverKeepsKing}: alla cattura del Re il server lascia il Re sulla scacchiera (come i log veri).
	 */
	private static String renderLog(State start, List<Action> moves, boolean serverKeepsKing, boolean crlf, boolean duplicateFinal) {
		final StringBuilder sb = new StringBuilder();
		sb.append(NOISE).append("BUONO: Inizio partita\n").append(stateBlock(start, token(start), null));
		State s = start;
		for (Action a : moves) {
			sb.append(NOISE).append("BUONO: Turn: ").append(s.getTurn() == Turn.WHITE ? "W" : "B").append(" Pawn from ").append(a.getFrom()).append(" to ").append(a.getTo()).append('\n');
			sb.append(NOISE).append("BUONO: Movimento pedina\n");
			final byte kingBefore = (byte) FastState.fromState(s).kingSquare;
			s = AshtonRules.applyMove(s, a);
			sb.append(NOISE).append("BUONO: Current draw cache size: 1\n");
			sb.append(stateBlock(s, token(s), serverKeepsKing && s.getTurn() == Turn.BLACKWIN ? new byte[] { kingBefore } : null));
		}
		if (duplicateFinal && (s.getTurn() != Turn.WHITE && s.getTurn() != Turn.BLACK)) {
			sb.append(NOISE).append("BUONO: Stato:\n").append(stateBlock(s, token(s), null).substring("BUONO: Stato:\n".length()));
		}
		String text = sb.toString();
		if (crlf) {
			text = text.replace("\n", "\r\n");
		}
		return text;
	}

	private static byte[] codes(State s) {
		return FastState.fromState(s).board.clone();
	}

	/** Un finale di fuga del Re: Bianco muove, Re c3 -> c1 (bordo). */
	private static State escapeStart() {
		final byte[] b = new byte[81];
		b[2 * 9 + 2] = BoardConstants.KING; // c3
		b[6 * 9 + 6] = BoardConstants.BLACK;
		b[7 * 9 + 2] = BoardConstants.BLACK;
		b[5 * 9 + 5] = BoardConstants.WHITE;
		b[1 * 9 + 6] = BoardConstants.WHITE;
		return NnueDataImport.toState(b, 0);
	}

	/** Un finale di cattura del Re: Nero muove, h3 -> d3 prende il Re in c3 tra d3 e b3. */
	private static State captureStart() {
		final byte[] b = new byte[81];
		b[2 * 9 + 2] = BoardConstants.KING; // c3
		b[2 * 9 + 1] = BoardConstants.BLACK; // b3
		b[2 * 9 + 7] = BoardConstants.BLACK; // h3
		b[6 * 9 + 6] = BoardConstants.WHITE;
		b[6 * 9 + 2] = BoardConstants.WHITE;
		return NnueDataImport.toState(b, 1);
	}

	private static NnueDataImport.Replay replayOf(String text) {
		return NnueDataImport.replay(NnueDataImport.parseLog(text));
	}

	/** Una partita di {@code plies} semi-mosse casuali legali dalla posizione iniziale (nessun terminale se possibile). */
	private static List<Action> randomMoves(Random rnd, int plies) {
		State s = new it.unibo.ai.didattica.competition.tablut.domain.StateTablut();
		s.setTurn(Turn.WHITE);
		final List<Action> out = new ArrayList<>();
		for (int i = 0; i < plies; i++) {
			if (AshtonRules.isTerminal(s)) {
				break;
			}
			final List<Action> legal = AshtonRules.generateMoves(s);
			if (legal.isEmpty()) {
				break;
			}
			final Action a = legal.get(rnd.nextInt(legal.size()));
			out.add(a);
			s = AshtonRules.applyMove(s, a);
		}
		return out;
	}

	private static State initialState() {
		final State s = new it.unibo.ai.didattica.competition.tablut.domain.StateTablut();
		s.setTurn(Turn.WHITE);
		return s;
	}

	private static void testParserAndReplay() throws IOException {
		// ---- vittoria Bianca: il Re fugge
		final State es = escapeStart();
		final List<Action> escape = Arrays.asList(act("c3", "c1", Turn.WHITE));
		for (boolean crlf : new boolean[] { false, true }) {
			final String text = renderLog(es, escape, false, crlf, true);
			final NnueDataImport.ParsedLog p = NnueDataImport.parseLog(text);
			check(p.problem == null && p.boards.size() == 2 && p.moves.size() == 1 && "WW".equals(p.lastToken) && "W".equals(p.tokens.get(0)),
					"fuga (crlf=" + crlf + "): parse stati " + p.boards.size() + " mosse " + p.moves.size() + " token " + p.lastToken);
			final NnueDataImport.Replay r = NnueDataImport.replay(p);
			check("COMPLETE".equals(r.end) && r.outValid && r.outcome == 1 && r.applied == 1 && r.positions.size() == 1 && r.roles.get(0) == 0, "fuga (crlf=" + crlf + "): " + r.end + " esito " + r.outcome);
			check(Arrays.equals(r.positions.get(0), codes(es)), "fuga: la posizione registrata e' quella di partenza");
		}
		// ---- vittoria Nera, il server lascia il Re sulla scacchiera / lo toglie
		final State cs = captureStart();
		final List<Action> capture = Arrays.asList(act("h3", "d3", Turn.BLACK));
		for (boolean keeps : new boolean[] { true, false }) {
			final NnueDataImport.Replay r = replayOf(renderLog(cs, capture, keeps, false, true));
			check("COMPLETE".equals(r.end) && r.outValid && r.outcome == -1, "cattura del Re (server " + (keeps ? "lascia" : "toglie") + " il Re): " + r.end + " " + r.detail);
			check(r.roles.size() == 1 && r.roles.get(0) == 1, "cattura: muove il Nero");
		}
		// il Re del server in una cella diversa da quella di prima della mossa NON e' una cattura valida
		String bad = renderLog(cs, capture, true, false, false);
		final int idx = bad.lastIndexOf("OBKBOOOOO"); // riga 3 dopo la cattura: b3 nero, Re in c3 (lasciato dal server), d3 nero
		if (idx >= 0) {
			bad = bad.substring(0, idx) + "OBOBOKOOO" + bad.substring(idx + 9);
			check("DIVERGENCE".equals(replayOf(bad).end), "il Re del server spostato di cella alla cattura e' una divergenza");
		} else {
			check(false, "il log sintetico della cattura non contiene la riga attesa del Re (test mal costruito)");
		}

		// ---- partita interrotta: nessun token di fine
		final List<Action> walk = randomMoves(new Random(11), 24);
		final State init = initialState();
		NnueDataImport.Replay r = replayOf(renderLog(init, walk, true, false, false));
		check("INCOMPLETE".equals(r.end) && !r.outValid && r.applied == walk.size() && r.positions.size() == walk.size(), "partita interrotta: " + r.end + " mosse " + r.applied);
		// le posizioni importate sono quelle giocate dal motore storico
		State s = init;
		boolean same = true;
		for (int i = 0; i < walk.size() && i < r.positions.size(); i++) {
			same &= Arrays.equals(r.positions.get(i), codes(s)) && r.roles.get(i) == (s.getTurn() == Turn.WHITE ? 0 : 1);
			s = AshtonRules.applyMove(s, walk.get(i));
		}
		check(same, "le posizioni importate coincidono con quelle giocate dal motore storico, con il lato giusto");

		// ---- forfeit: mossa nulla, con vittoria assegnata dal server
		final String forfeit = NOISE + "BUONO: Inizio partita\n" + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from null to null\n" + stateBlock(init, "BW", null)
				+ stateBlock(init, "BW", null);
		r = replayOf(forfeit);
		check("FORFEIT_NULL".equals(r.end) && !r.outValid && r.applied == 0 && r.positions.size() == 1, "forfeit con mossa nulla: " + r.end);
		// ---- forfeit con esito non terminale: la partita finisce con WW ma la scacchiera non e' terminale
		final Action first = AshtonRules.generateMoves(init).get(0);
		final State after = AshtonRules.applyMove(init, first);
		final String forfeit2 = NOISE + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from " + first.getFrom() + " to " + first.getTo() + "\n" + stateBlock(after, "B", null)
				+ stateBlock(after, "WW", null); // il blocco di endGame dichiara la vittoria senza che nessuna mossa l'abbia prodotta (timeout)
		r = replayOf(forfeit2);
		check("FORFEIT_RESULT".equals(r.end) && !r.outValid && r.applied == 1, "vittoria assegnata su una scacchiera non terminale: " + r.end);
		// ---- mossa illegale RIFIUTATA dal server (scacchiera invariata): forfeit, non divergenza
		final String illegal = NOISE + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from a1 to a9\n" + stateBlock(init, "BW", null);
		r = replayOf(illegal);
		check("ILLEGAL".equals(r.end) && !r.outValid, "mossa illegale rifiutata dal server: " + r.end);
		// ---- mossa illegale APPLICATA dal server: le regole divergono
		final byte[] moved = codes(init);
		moved[0 * 9 + 0] = BoardConstants.BLACK;
		final String illegalApplied = NOISE + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from a1 to a9\n" + stateBlock(NnueDataImport.toState(moved, 1), "B", null);
		r = replayOf(illegalApplied);
		check("DIVERGENCE".equals(r.end) && !r.outValid && r.detail.contains("applicata"), "mossa illegale applicata dal server: " + r.end + " / " + r.detail);
		// ---- divergenza della scacchiera: una cella diversa dopo una mossa legale
		final byte[] wrong = codes(after);
		for (int k = 0; k < 81; k++) {
			if (wrong[k] == BoardConstants.EMPTY && k != BoardConstants.THRONE) {
				wrong[k] = BoardConstants.BLACK;
				break;
			}
		}
		final String diverge = NOISE + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from " + first.getFrom() + " to " + first.getTo() + "\n"
				+ stateBlock(NnueDataImport.toState(wrong, 1), "B", null);
		r = replayOf(diverge);
		check("DIVERGENCE".equals(r.end) && !r.outValid && r.detail.contains("scacchiera"), "scacchiera diversa da quella del motore: " + r.end + " / " + r.detail);
		// ---- patta dichiarata dal server (ripetizione) su una scacchiera non terminale
		final String draw = NOISE + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from " + first.getFrom() + " to " + first.getTo() + "\n" + stateBlock(after, "D", null);
		r = replayOf(draw);
		check("SERVER_DRAW".equals(r.end) && r.outValid && r.outcome == 0, "patta del server: " + r.end);

		// ---- mosse rifiutate dal server (riga di mossa senza stato): contate e ignorate
		final String rej = NOISE + stateBlock(init, "W", null) + NOISE + "BUONO: Turn: W Pawn from a1 to a2\n" + NOISE + "BUONO: Turn: W Pawn from " + first.getFrom() + " to " + first.getTo()
				+ "\n" + stateBlock(after, "B", null);
		final NnueDataImport.ParsedLog pr = NnueDataImport.parseLog(rej);
		check(pr.rejectedMoves == 1 && pr.moves.size() == 1 && pr.boards.size() == 2, "mosse rifiutate: " + pr.rejectedMoves + " rifiutate, " + pr.moves.size() + " accettate");
		// ---- blocco malformato
		final NnueDataImport.ParsedLog pm = NnueDataImport.parseLog("BUONO: Stato:\nOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\n-\nW\n");
		check(pm.problem != null && "BAD_LOG".equals(NnueDataImport.replay(pm).end), "riga di scacchiera di lunghezza sbagliata: log non interpretabile");
		final NnueDataImport.ParsedLog pc = NnueDataImport.parseLog("BUONO: Stato:\nOOOOXOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\nOOOOOOOOO\n-\nW\n");
		check(pc.problem != null, "carattere sconosciuto nella scacchiera: log non interpretabile");
		check("BAD_LOG".equals(NnueDataImport.replay(NnueDataImport.parseLog("niente di utile\n")).end), "testo senza stati: BAD_LOG");
		// blocco troncato in fondo: ignorato senza eccezioni
		final String cut = renderLog(init, walk.subList(0, 3), true, false, false);
		final NnueDataImport.ParsedLog pt = NnueDataImport.parseLog(cut.substring(0, cut.length() - 25));
		check(pt.problem == null && pt.boards.size() == 3 && pt.moves.size() == 2, "blocco finale troncato ignorato: stati " + pt.boards.size());
		// un primo stato in terminale: non e' una partita giocabile
		final String startsTerminal = stateBlock(es, "WW", null);
		check("BAD_LOG".equals(replayOf(startsTerminal).end), "primo stato con token di fine: BAD_LOG");

		// ---- coordinate: la lettera e' la colonna, il numero la riga + 1
		check(NnueDataImport.square("a1") == 0 && NnueDataImport.square("i9") == 80 && NnueDataImport.square("d5") == 4 * 9 + 3 && NnueDataImport.square("e1") == 4, "coordinate");
		check(NnueDataImport.square("null") == -1 && NnueDataImport.square("j1") == -1 && NnueDataImport.square("a0") == -1 && NnueDataImport.square("") == -1, "coordinate non valide");
	}

	// ====================================================================== cartelle, doppioni

	private static void testDirectoryImport() throws IOException {
		final File dir = Files.createTempDirectory("nnulogs_").toFile();
		dir.deleteOnExit();
		final State es = escapeStart(), cs = captureStart();
		final String win = renderLog(es, Arrays.asList(act("c3", "c1", Turn.WHITE)), false, false, true);
		final String lose = renderLog(cs, Arrays.asList(act("h3", "d3", Turn.BLACK)), true, true, true);
		final String walk = renderLog(initialState(), randomMoves(new Random(5), 14), true, false, false);
		write(new File(dir, "a_gameLog.txt"), win);
		write(new File(dir, "b_gameLog.txt"), win); // doppione di a
		write(new File(dir, "c_gameLog.txt"), lose);
		write(new File(dir, "d_gameLog.txt"), walk);
		write(new File(dir, "e_systemLog.txt"), win); // non e' un gameLog: ignorato
		write(new File(dir, "leggimi.md"), "x");
		new File(dir, "a_gameLog.txt").deleteOnExit();
		new File(dir, "b_gameLog.txt").deleteOnExit();
		new File(dir, "c_gameLog.txt").deleteOnExit();
		new File(dir, "d_gameLog.txt").deleteOnExit();
		new File(dir, "e_systemLog.txt").deleteOnExit();
		new File(dir, "leggimi.md").deleteOnExit();

		NnueDataImport.Report rep = new NnueDataImport.Report();
		NnueDataset d = NnueDataImport.fromServerLogs(dir.getPath(), true, false, rep);
		check(rep.files == 4 && rep.games == 4 && rep.duplicateGames == 1 && rep.completeGames == 2 && rep.incompleteGames == 1 && rep.divergentGames == 0, "cartella: " + rep);
		check(d.n == 2 && d.numGames() == 2, "cartella (dedupe): solo le due partite concluse, una posizione ciascuna: " + d.describe());
		check(d.outValid[0] == 1 && d.outValid[1] == 1 && d.toEnd[0] == 1 && d.toEnd[1] == 1 && d.ply[0] == 0 && d.source[0] == NnueDataset.SRC_LOG, "cartella: etichette dei campioni");
		check(d.outcome[0] == 1 && d.outcome[1] == -1, "cartella: l'ordine dei file e' per nome (a, c): Bianco poi Nero");
		check(rep.droppedUnlabeled == 14 && rep.kept == 2, "cartella: 14 posizioni della partita interrotta scartate senza etichetta: " + rep);
		rep = new NnueDataImport.Report();
		d = NnueDataImport.fromServerLogs(dir.getPath(), false, true, rep);
		check(rep.duplicateGames == 0 && d.n == 17, "cartella (senza dedupe, con senza-etichetta): 1 + 1 (doppione) + 1 + 14 (partita interrotta) posizioni: " + d.n + " / " + rep);
		boolean unl = false;
		for (int i = 0; i < d.n; i++) {
			unl |= d.outValid[i] == 0 && d.toEnd[i] == -1 && d.valValid[i] == 0;
		}
		check(unl, "keepUnlabeled: le posizioni senza esito sono tenute, con toEnd -1");
		// un file singolo
		rep = new NnueDataImport.Report();
		check(NnueDataImport.fromServerLogs(new File(dir, "c_gameLog.txt").getPath(), true, false, rep).n == 1, "un file singolo si importa");
		try {
			NnueDataImport.fromServerLogs("/nonexistent_dir_xyz", true, false, new NnueDataImport.Report());
			check(false, "una cartella inesistente deve dare IOException");
		} catch (IOException e) {
			check(true, "ok");
		}
	}

	private static void write(File f, String text) throws IOException {
		Files.write(f.toPath(), text.getBytes("ISO-8859-1"));
	}

	// ====================================================================== log veri

	private static void testRealLogs() throws IOException {
		final File dir = new File("logs");
		if (!dir.isDirectory()) {
			System.out.println("(log veri: cartella 'logs' assente, controllo saltato)");
			return;
		}
		final NnueDataImport.Report rep = new NnueDataImport.Report();
		final NnueDataset d = NnueDataImport.fromServerLogs(dir.getPath(), true, false, rep);
		System.out.println("log veri: " + rep.toString().split("\n")[0]);
		check(rep.games >= 50, "log veri: attese molte partite, trovate " + rep.games);
		check(rep.divergentGames == 0, "log veri: le regole del motore devono riprodurre OGNI partita del server (divergenze: " + rep.divergentGames + ", " + rep.notes + ")");
		check(rep.completeGames >= 30 && d.n >= 1000, "log veri: troppo poche partite concluse/posizioni (" + rep.completeGames + " / " + d.n + ")");
		check(rep.forfeitGames + rep.illegalGames + rep.incompleteGames + rep.completeGames + rep.divergentGames + rep.duplicateGames == rep.games, "log veri: ogni partita e' classificata una sola volta");
		boolean white = false, black = false;
		for (int i = 0; i < d.n; i++) {
			white |= d.outcome[i] > 0;
			black |= d.outcome[i] < 0;
		}
		check(white && black, "log veri: devono comparire vittorie di entrambi i lati (la cattura del Re lasciato sulla scacchiera e' gestita)");
	}

	// ====================================================================== NnueDataTool

	private static int runTool(List<String> args, StringBuilder out) throws Exception {
		final List<String> cmd = new ArrayList<>();
		cmd.add(System.getProperty("java.home") + File.separator + "bin" + File.separator + "java");
		cmd.add("-Xmx1g");
		cmd.add("-cp");
		cmd.add(System.getProperty("java.class.path"));
		cmd.add("it.unibo.ai.didattica.competition.tablut.training.NnueDataTool");
		cmd.addAll(args);
		final Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		final ByteArrayOutputStream bo = new ByteArrayOutputStream();
		final byte[] buf = new byte[8192];
		int n;
		while ((n = p.getInputStream().read(buf)) > 0) {
			bo.write(buf, 0, n);
		}
		out.append(new String(bo.toByteArray(), "UTF-8"));
		p.waitFor();
		return p.exitValue();
	}

	private static void testDataTool() throws Exception {
		final NnueDataset a = randomDataset(new Random(30), 200, 20), b = randomDataset(new Random(31), 100, 10);
		// filter: sorgenti, esito, valore
		final NnueDataset all = NnueDataset.merge(Arrays.asList(a, b));
		final boolean[] only1 = new boolean[16];
		only1[1] = true;
		final NnueDataset f1 = NnueDataTool.filter(all, only1, false, false);
		boolean ok = true;
		int expected = 0;
		for (int i = 0; i < all.n; i++) {
			expected += all.source[i] == 1 ? 1 : 0;
		}
		for (int i = 0; i < f1.n; i++) {
			ok &= f1.source[i] == 1;
		}
		check(ok && f1.n == expected && expected > 0, "filter per sorgente: " + f1.n + " campioni, attesi " + expected);
		final NnueDataset fo = NnueDataTool.filter(all, null, true, false), fv = NnueDataTool.filter(all, null, false, true), fb = NnueDataTool.filter(all, null, true, true);
		boolean ov = true, vv = true, bv = true;
		for (int i = 0; i < fo.n; i++) {
			ov &= fo.outValid[i] == 1;
		}
		for (int i = 0; i < fv.n; i++) {
			vv &= fv.valValid[i] == 1;
		}
		for (int i = 0; i < fb.n; i++) {
			bv &= fb.outValid[i] == 1 && fb.valValid[i] == 1;
		}
		check(ov && vv && bv && fo.n < all.n && fv.n < all.n && fb.n <= Math.min(fo.n, fv.n), "filter per esito/valore: " + fo.n + ", " + fv.n + ", " + fb.n + " su " + all.n);
		check(NnueDataTool.filter(all, null, false, false).equals(all), "filter senza condizioni = identita'");

		// CLI
		final File fa = tmp(".gz"), fb2 = tmp(".gz"), merged = tmp(".gz"), tr = tmp(".gz"), he = tmp(".gz"), fil = tmp(".gz");
		a.save(fa.getPath());
		b.save(fb2.getPath());
		StringBuilder o = new StringBuilder();
		int rc = runTool(Arrays.asList("info", "in=" + fa.getPath() + "," + fb2.getPath()), o);
		check(rc == 0 && o.toString().contains("200 posizioni") && o.toString().contains("100 posizioni"), "info: uscita " + rc + ". Output: " + o);
		o = new StringBuilder();
		rc = runTool(Arrays.asList("merge", "in=" + fa.getPath() + "," + fb2.getPath(), "out=" + merged.getPath()), o);
		final NnueDataset m = NnueDataset.load(merged.getPath());
		check(rc == 0 && m.n == 300 && m.numGames() == a.numGames() + b.numGames(), "merge: uscita " + rc + ", " + m.n + " campioni. Output: " + o);
		o = new StringBuilder();
		rc = runTool(Arrays.asList("split", "in=" + merged.getPath(), "train=" + tr.getPath(), "held=" + he.getPath(), "pct=40", "salt=3"), o);
		final NnueDataset t = NnueDataset.load(tr.getPath()), h = NnueDataset.load(he.getPath());
		check(rc == 0 && t.n + h.n == 300 && t.n > 0 && h.n > 0, "split: uscita " + rc + ", " + t.n + " + " + h.n + ". Output: " + o);
		final java.util.Set<Integer> tg = new java.util.HashSet<>(), hg = new java.util.HashSet<>();
		for (int g : t.game) {
			tg.add(g);
		}
		for (int g : h.game) {
			hg.add(g);
		}
		final java.util.Set<Integer> inter = new java.util.HashSet<>(tg);
		inter.retainAll(hg);
		check(inter.isEmpty(), "split: nessuna partita da entrambe le parti");
		// ripetibile e coerente con la libreria: stessa divisione di splitByGame(salt, pct) sul dataset unito
		final int[][] expect = m.splitByGame(3, 40);
		check(t.equals(m.subset(expect[0])) && h.equals(m.subset(expect[1])), "split: la divisione e' quella di splitByGame (stesso sale e percentuale)");
		o = new StringBuilder();
		rc = runTool(Arrays.asList("filter", "in=" + merged.getPath(), "out=" + fil.getPath(), "sources=1", "needoutcome=true"), o);
		check(rc == 0 && NnueDataset.load(fil.getPath()).equals(NnueDataTool.filter(m, only1, true, false)), "filter: uscita " + rc + ". Output: " + o);

		// errori
		final String[][] bad = { {}, { "info" }, { "boh", "in=" + fa.getPath() }, { "merge", "in=" + fa.getPath() }, { "split", "in=" + fa.getPath(), "train=x.gz" },
				{ "split", "in=" + fa.getPath(), "train=x.gz", "held=y.gz", "pct=101" }, { "filter", "in=" + fa.getPath() }, { "filter", "in=" + fa.getPath(), "out=" + fil.getPath(), "sources=99" },
				{ "info", "in=" + fa.getPath(), "boh=1" }, { "info", "in=" + fa.getPath(), "senzauguale" }, { "split", "in=" + fa.getPath(), "train=x.gz", "held=y.gz", "pct=abc" },
				{ "merge", "in=,", "out=" + fil.getPath() } };
		for (String[] x : bad) {
			o = new StringBuilder();
			rc = runTool(Arrays.asList(x), o);
			check(rc == 2, "NnueDataTool " + Arrays.toString(x) + ": uscita " + rc + " (attesa 2): " + o);
		}
		o = new StringBuilder();
		rc = runTool(Arrays.asList("info", "in=/nonexistent_dir_xyz/x.gz"), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "info su file inesistente: uscita " + rc + " (attesa 1): " + o);
		final File junk = tmp(".gz");
		Files.write(junk.toPath(), "non e' gzip".getBytes("UTF-8"));
		o = new StringBuilder();
		rc = runTool(Arrays.asList("merge", "in=" + fa.getPath() + "," + junk.getPath(), "out=" + fil.getPath()), o);
		check(rc == 1 && o.toString().contains("ERRORE"), "merge con un file corrotto: uscita " + rc + " (attesa 1): " + o);
	}

	// ====================================================================== dati.gz e DistillSet

	private static void testLegacyImporters() throws IOException {
		final int nf = TablutFeatures.NUM_FEATURES;
		// tre posizioni: iniziale (re = 3 nel formato legacy), una con il re altrove, una NON valida (re sul bordo)
		final int n = 4;
		final byte[] cells = new byte[81 * n];
		final byte[] role = { 0, 1, 0, 1 };
		final byte[] outcome = { 1, -1, 0, 1 };
		final int[] game = { 0, 0, 5, 5 };
		final FastState f0 = new FastState();
		for (int sq = 0; sq < 81; sq++) {
			cells[sq] = (byte) (f0.board[sq] == BoardConstants.KING ? 3 : f0.board[sq]);
		}
		final byte[] p1 = initial();
		p1[BoardConstants.THRONE] = BoardConstants.EMPTY;
		p1[3 * 9 + 3] = BoardConstants.KING;
		for (int sq = 0; sq < 81; sq++) {
			cells[81 + sq] = (byte) (p1[sq] == BoardConstants.KING ? 3 : p1[sq]);
		}
		final byte[] p2 = initial();
		p2[BoardConstants.THRONE] = BoardConstants.EMPTY;
		p2[2 * 9 + 2] = BoardConstants.KING;
		for (int sq = 0; sq < 81; sq++) {
			cells[162 + sq] = (byte) (p2[sq] == BoardConstants.KING ? 3 : p2[sq]);
		}
		final byte[] p3 = initial();
		p3[BoardConstants.THRONE] = BoardConstants.EMPTY;
		p3[0 * 9 + 4] = BoardConstants.KING; // sul bordo: terminale
		for (int sq = 0; sq < 81; sq++) {
			cells[243 + sq] = (byte) (p3[sq] == BoardConstants.KING ? 3 : p3[sq]);
		}
		final float[] feats = new float[n * nf];

		// ---- dati.gz scritto dal writer originale
		final File cnn = tmp(".gz");
		new CnnPretrainer.Dataset(cells, role, feats, outcome, game).save(cnn.getPath());
		NnueDataImport.Report rep = new NnueDataImport.Report();
		NnueDataset d = NnueDataImport.fromCnnData(cnn.getPath(), rep);
		check(d.n == 3 && rep.droppedInvalid == 1 && rep.kept == 3, "dati.gz: 3 valide, 1 non valida (Re sul bordo) scartata e contata: " + rep);
		check(Arrays.equals(Arrays.copyOfRange(d.boards, 0, 81), f0.board), "dati.gz: il Re passa da codice 3 a 4");
		check(d.outcome[0] == 1 && d.outcome[1] == -1 && d.outcome[2] == 0 && d.role[1] == 1 && d.outValid[2] == 1, "dati.gz: esiti e ruoli (la patta e' un esito valido)");
		check(d.game[0] == 0 && d.game[1] == 0 && d.game[2] == 5 && d.numGames() == 2, "dati.gz: chiavi di partita");
		check(d.valValid[0] == 0 && d.ply[0] == -1 && d.toEnd[0] == -1 && d.source[0] == NnueDataset.SRC_IMPORT, "dati.gz: niente valore, ply/toEnd ignoti, sorgente import");
		// errori
		File bad = tmp(".gz");
		try (DataOutputStream o = new DataOutputStream(new GZIPOutputStream(Files.newOutputStream(bad.toPath())))) {
			o.writeUTF("TABLUT_DISTILL_SET v1");
		}
		try {
			NnueDataImport.fromCnnData(bad.getPath(), new NnueDataImport.Report());
			check(false, "dati.gz: un DistillSet non e' un dataset CNN");
		} catch (IOException e) {
			check(e.getMessage().contains("TABLUT_CNN_DATA"), "dati.gz con intestazione sbagliata: " + e.getMessage());
		}
		final byte[] bytes = Files.readAllBytes(cnn.toPath());
		Files.write(bad.toPath(), Arrays.copyOf(bytes, bytes.length / 2));
		try {
			NnueDataImport.fromCnnData(bad.getPath(), new NnueDataImport.Report());
			check(false, "dati.gz troncato accettato");
		} catch (IOException e) {
			check(true, "ok");
		}
		try {
			NnueDataImport.fromCnnData("/nonexistent_dir_xyz/x.gz", new NnueDataImport.Report());
			check(false, "dati.gz inesistente accettato");
		} catch (IOException e) {
			check(true, "ok");
		}

		// ---- DistillSet scritto dal writer originale: usa il suo outValid, mappa le sorgenti
		final File ds = tmp(".gz");
		new DistillSet(cells, role, outcome, new byte[] { 1, 0, 1, 1 }, game, new byte[] { 0, 1, 1, 0 }, new boolean[n], feats).save(ds.getPath());
		rep = new NnueDataImport.Report();
		d = NnueDataImport.fromDistillSet(ds.getPath(), rep);
		check(d.n == 2 && rep.droppedInvalid == 1 && rep.droppedUnlabeled == 1, "DistillSet: 4 -> 2 (una senza esito, una non valida): " + rep);
		check(d.outcome[0] == 1 && d.source[0] == NnueDataset.SRC_IMPORT && d.source[1] == NnueDataset.SRC_SELFPLAY, "DistillSet: esiti e sorgenti (engine -> auto-gioco, dataset -> import)");
		check(Arrays.equals(Arrays.copyOfRange(d.boards, 0, 81), f0.board), "DistillSet: Re 3 -> 4");
		try {
			NnueDataImport.fromDistillSet(cnn.getPath(), new NnueDataImport.Report());
			check(false, "un dati.gz non e' un DistillSet");
		} catch (IOException e) {
			check(e.getMessage().contains("DistillSet"), "DistillSet con intestazione sbagliata: " + e.getMessage());
		}
		// le importazioni si uniscono e si salvano come un dataset normale
		final NnueDataset merged = NnueDataset.merge(Arrays.asList(NnueDataImport.fromCnnData(cnn.getPath(), new NnueDataImport.Report()), d));
		final File out = tmp(".gz");
		merged.save(out.getPath());
		check(NnueDataset.load(out.getPath()).equals(merged), "i dati importati si salvano e si rileggono");
		check(merged.numGames() == 4, "l'unione di due importazioni con chiavi uguali tiene distinte le partite: " + merged.numGames());
	}
}
