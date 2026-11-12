package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;
import it.unibo.ai.didattica.competition.tablut.engine.BoardConstants;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

/**
 * Importa in un {@link NnueDataset} i dati di gioco che il progetto gia' possiede:
 * <ul>
 * <li><b>log delle partite del server</b> ({@code *_gameLog.txt}): partite VERE fra motori, umani e altre squadre. Il testo non e' creduto:
 * ogni mossa e' riprodotta con le regole ({@link AshtonRules}, tramite {@link GameTracker}) e la scacchiera ottenuta e' confrontata con quella
 * registrata dal server; alla prima divergenza la partita e' marcata e non porta esito. Cosi' l'importazione e' anche una verifica delle
 * regole del motore contro quelle del server su migliaia di mosse vere;</li>
 * <li><b>{@code dati.gz}</b> ({@code TABLUT_CNN_DATA v1}) e i <b>{@code DistillSet}</b> ({@code TABLUT_DISTILL_SET v1}): si leggono le colonne
 * che non dipendono dalle feature a mano (scacchiera, ruolo, esito, partita) e si SALTANO le feature; il Re passa da codice 3 a 4.</li>
 * </ul>
 *
 * <h3>Esito di una partita dei log</h3>
 * Vero ({@code outValid = 1}) solo se la riproduzione termina in uno stato che le regole giudicano finale (Re catturato, Re sul bordo,
 * nessuna mossa legale, patta) oppure il server dichiara patta. NON e' un esito: il <b>forfeit</b> (mossa {@code null to null}, mossa illegale:
 * il server assegna la vittoria ma la scacchiera non e' terminale), la partita <b>interrotta</b> (nessun token di fine), la <b>divergenza</b>
 * fra regole e server. Le posizioni di queste partite non hanno etichette e vengono scartate (opzione {@code keepUnlabeled} per tenerle).
 * <p>
 * <b>Doppioni</b>: motori deterministici danno partite identiche; con {@code dedupe} (default) una partita con la stessa sequenza di mosse di
 * una gia' importata e' scartata, altrimenti peserebbe piu' volte nell'addestramento e fra train e held-out.
 * <p>
 * Uso: {@code NnueDataImport out=FILE.gz [logs=DIR|FILE] [cnndata=FILE] [distill=FILE] [dedupe=true] [keepunlabeled=false]}.
 */
public final class NnueDataImport {

	private NnueDataImport() {
	}

	// ====================================================================== rapporto

	/** Cosa e' successo durante l'importazione: nulla e' scartato in silenzio. */
	public static final class Report {
		public int files, games, completeGames, forfeitGames, illegalGames, incompleteGames, divergentGames, duplicateGames, serverDraws;
		public int positions, kept, droppedUnlabeled, droppedInvalid, rejectedMoves;
		public final Map<String, Integer> notes = new LinkedHashMap<>();

		void note(String what) {
			final Integer c = notes.get(what);
			notes.put(what, c == null ? 1 : c + 1);
		}

		@Override
		public String toString() {
			final StringBuilder sb = new StringBuilder();
			sb.append(files).append(" file, ").append(games).append(" partite: ").append(completeGames).append(" con esito vero (di cui ").append(serverDraws)
					.append(" patte dichiarate dal server), ").append(forfeitGames).append(" forfeit/timeout, ").append(illegalGames).append(" con mossa illegale, ")
					.append(incompleteGames).append(" interrotte, ").append(divergentGames).append(" con divergenza dalle regole, ").append(duplicateGames)
					.append(" doppioni scartati; posizioni viste ").append(positions).append(", tenute ").append(kept).append(", senza etichetta scartate ")
					.append(droppedUnlabeled).append(", non valide scartate ").append(droppedInvalid).append(", mosse rifiutate dal server ").append(rejectedMoves);
			for (Map.Entry<String, Integer> e : notes.entrySet()) {
				sb.append("\n  nota: ").append(e.getKey()).append(" x").append(e.getValue());
			}
			return sb.toString();
		}
	}

	// ====================================================================== conversioni

	/** Codici {@link BoardConstants} -> {@link State} (il trono vuoto e' {@link Pawn#THRONE}). */
	public static State toState(byte[] codes, int role) {
		final State s = new StateTablut();
		final Pawn[][] b = s.getBoard();
		for (int r = 0; r < 9; r++) {
			for (int c = 0; c < 9; c++) {
				final int code = codes[r * 9 + c];
				b[r][c] = code == BoardConstants.WHITE ? Pawn.WHITE : code == BoardConstants.BLACK ? Pawn.BLACK : code == BoardConstants.KING ? Pawn.KING
						: (r == 4 && c == 4 ? Pawn.THRONE : Pawn.EMPTY);
			}
		}
		s.setTurn(role == 0 ? Turn.WHITE : Turn.BLACK);
		return s;
	}

	// ====================================================================== log del server

	private static final Pattern MOVE = Pattern.compile("Turn: ([WB]) Pawn from (\\S+) to (\\S+)\\s*$");

	/** Il contenuto di un log, nell'ordine: stati (con il token del turno/fine) e mosse. {@code moves.get(k-1)} porta dallo stato k-1 al k. */
	public static final class ParsedLog {
		public final List<byte[]> boards = new ArrayList<>();
		public final List<String> tokens = new ArrayList<>();
		/** {rigaDa, colonnaDa, rigaA, colonnaA}, oppure {@code null} per una mossa "null to null". */
		public final List<int[]> moves = new ArrayList<>();
		public int rejectedMoves;
		/** Token dell'ultimo blocco di stato (anche duplicato di fine partita). */
		public String lastToken;
		public String problem;
	}

	/**
	 * Interpreta il testo di un log. Non applica le regole: raccoglie gli stati ("Stato:" + 9 righe + "-" + token) e le mosse
	 * ({@code Turn: W Pawn from d5 to d4}) e le accoppia: il PRIMO stato dopo una mossa e' lo stato risultante; un secondo stato senza mossa in mezzo
	 * (il blocco di {@code endGame}) e' un duplicato; una mossa non seguita da uno stato prima di un'altra mossa e' stata rifiutata dal server.
	 */
	public static ParsedLog parseLog(String text) {
		final ParsedLog p = new ParsedLog();
		final String[] lines = text.split("\n", -1);
		int[] pending = null;
		boolean pendingSet = false;
		for (int i = 0; i < lines.length; i++) {
			final String line = lines[i].replace("\r", "");
			final Matcher m = MOVE.matcher(line);
			if (m.find()) {
				if (pendingSet) {
					p.rejectedMoves++;
				}
				pending = square(m.group(2)) < 0 || square(m.group(3)) < 0 ? null
						: new int[] { square(m.group(2)) / 9, square(m.group(2)) % 9, square(m.group(3)) / 9, square(m.group(3)) % 9 };
				pendingSet = true;
				continue;
			}
			if (!line.endsWith("Stato:")) {
				continue;
			}
			if (i + 11 >= lines.length) {
				break; // blocco troncato in fondo al file
			}
			final byte[] b = new byte[81];
			boolean ok = true;
			for (int r = 0; r < 9 && ok; r++) {
				final String row = lines[i + 1 + r].replace("\r", "");
				if (row.length() != 9) {
					ok = false;
					break;
				}
				for (int c = 0; c < 9; c++) {
					final char ch = row.charAt(c);
					b[r * 9 + c] = (byte) (ch == 'W' ? BoardConstants.WHITE : ch == 'B' ? BoardConstants.BLACK : ch == 'K' ? BoardConstants.KING
							: ch == 'O' || ch == 'T' ? BoardConstants.EMPTY : -1);
					if (b[r * 9 + c] < 0) {
						ok = false;
						break;
					}
				}
			}
			final String sep = lines[i + 10].replace("\r", "").trim();
			final String token = lines[i + 11].replace("\r", "").trim();
			if (!ok || !sep.equals("-") || token.isEmpty()) {
				p.problem = "blocco di stato malformato alla riga " + (i + 1);
				break;
			}
			if (p.boards.isEmpty()) {
				p.boards.add(b);
				p.tokens.add(token);
			} else if (pendingSet) {
				p.boards.add(b);
				p.tokens.add(token);
				p.moves.add(pending);
				pendingSet = false;
			} else {
				p.lastToken = token; // duplicato di fine partita (endGame)
			}
			p.lastToken = token;
			i += 11;
		}
		if (pendingSet) {
			p.rejectedMoves++;
		}
		return p;
	}

	/** "d5" -> indice di cella (riga*9 + colonna), oppure -1 (anche per "null"). La lettera e' la colonna, il numero la riga + 1. */
	static int square(String s) {
		if (s.length() != 2) {
			return -1;
		}
		final int c = Character.toLowerCase(s.charAt(0)) - 'a', r = s.charAt(1) - '1';
		return c < 0 || c > 8 || r < 0 || r > 8 ? -1 : r * 9 + c;
	}

	/** Esito della riproduzione di un log con le regole. */
	public static final class Replay {
		/** Posizioni NON terminali prima di ciascuna mossa riprodotta (codici), con il lato al tratto. */
		public final List<byte[]> positions = new ArrayList<>();
		public final List<Integer> roles = new ArrayList<>();
		/** Mosse riprodotte con successo (= semi-mosse giocate). */
		public int applied;
		public boolean outValid;
		/** +1, -1, 0 vista Bianco; significativo solo con {@code outValid}. */
		public int outcome;
		/** COMPLETE, SERVER_DRAW, FORFEIT_NULL, FORFEIT_RESULT, ILLEGAL, INCOMPLETE, DIVERGENCE, BAD_LOG. */
		public String end;
		/** Dettaglio di una divergenza (passo e natura), per i rapporti. */
		public String detail = "";
	}

	/** Riproduce un log con le regole e ne ricava le posizioni e l'esito vero (se c'e'). */
	public static Replay replay(ParsedLog p) {
		final Replay out = new Replay();
		if (p.problem != null || p.boards.isEmpty()) {
			out.end = "BAD_LOG";
			return out;
		}
		final String t0 = p.tokens.get(0);
		if (!t0.equals("W") && !t0.equals("B")) {
			out.end = "BAD_LOG";
			return out;
		}
		if (NnueDataset.checkPosition(p.boards.get(0), 0) != null) {
			out.end = "BAD_LOG";
			return out;
		}
		State state = toState(p.boards.get(0), t0.equals("W") ? 0 : 1);
		final GameTracker tracker = new GameTracker(state);
		String end = null;
		for (int k = 1; k <= p.moves.size(); k++) {
			if (AshtonRules.isTerminal(state) || AshtonRules.generateMoves(state).isEmpty()) {
				break; // dati in piu' dopo la fine: ignorati
			}
			final byte[] codes = FastState.fromState(state).board.clone();
			out.positions.add(codes);
			out.roles.add(state.getTurn() == Turn.WHITE ? 0 : 1);
			final int[] mv = p.moves.get(k - 1);
			if (mv == null) {
				end = "FORFEIT_NULL";
				break;
			}
			Action chosen = null;
			for (Action a : AshtonRules.generateMoves(state)) {
				if (a.getRowFrom() == mv[0] && a.getColumnFrom() == mv[1] && a.getRowTo() == mv[2] && a.getColumnTo() == mv[3]) {
					chosen = a;
					break;
				}
			}
			if (chosen == null) {
				// una mossa che il motore giudica illegale: se il server l'ha RIFIUTATA (scacchiera invariata) e' un forfeit del giocatore; se l'ha
				// APPLICATA le regole del motore e del server divergono
				end = Arrays.equals(p.boards.get(k), p.boards.get(k - 1)) ? "ILLEGAL" : "DIVERGENCE";
				out.detail = "passo " + k + ": mossa " + Arrays.toString(mv) + (end.equals("ILLEGAL") ? " rifiutata dal server" : " applicata dal server ma illegale per il motore");
				break;
			}
			final int kingBefore = FastState.fromState(state).kingSquare;
			state = tracker.apply(state, chosen);
			final byte[] engineBoard = FastState.fromState(state).board;
			final boolean boardOk = state.getTurn() == Turn.BLACKWIN ? kingCapturedBoardMatches(engineBoard, p.boards.get(k), kingBefore)
					: Arrays.equals(engineBoard, p.boards.get(k));
			if (!boardOk || !turnMatches(state, p.tokens.get(k))) {
				end = "DIVERGENCE";
				out.detail = "passo " + k + ": " + (boardOk ? "turno/fine " + p.tokens.get(k) + " diverso da " + state.getTurn() : "scacchiera diversa da quella del server");
				break;
			}
			out.applied++;
			if (tracker.isRepetitionDraw()) {
				break;
			}
		}
		if (end != null) {
			out.end = end;
			return out;
		}
		final GameTracker.Result res = tracker.result(state);
		final String last = p.lastToken == null ? "" : p.lastToken;
		if (AshtonRules.isTerminal(state) || AshtonRules.generateMoves(state).isEmpty()) {
			out.outValid = true;
			out.outcome = (int) res.outcome;
			out.end = "COMPLETE";
		} else if (tracker.isRepetitionDraw() || last.equals("D")) {
			out.outValid = true;
			out.outcome = 0;
			out.end = "SERVER_DRAW";
		} else if (last.equals("WW") || last.equals("BW")) {
			out.end = "FORFEIT_RESULT"; // il server assegna la vittoria, ma la scacchiera non e' terminale
		} else {
			out.end = "INCOMPLETE";
		}
		return out;
	}

	/**
	 * Alla cattura del Re il server lascia il Re SULLA scacchiera (nella cella in cui e' stato preso), il motore lo rimuove: la scacchiera
	 * del server, tolto il Re, deve coincidere con quella del motore, e il Re del server stare dove stava prima della mossa. Se il server il Re
	 * lo toglie, le due scacchiere devono semplicemente coincidere.
	 */
	static boolean kingCapturedBoardMatches(byte[] engine, byte[] logged, int kingBefore) {
		final byte[] b = logged.clone();
		int kings = 0, at = -1;
		for (int i = 0; i < 81; i++) {
			if (b[i] == BoardConstants.KING) {
				kings++;
				at = i;
			}
		}
		if (kings == 0) {
			return Arrays.equals(engine, b); // un server che toglie il Re dalla scacchiera: stessa rappresentazione del motore
		}
		if (kings != 1 || at != kingBefore) {
			return false;
		}
		b[at] = BoardConstants.EMPTY;
		return Arrays.equals(engine, b);
	}

	private static boolean turnMatches(State s, String token) {
		final Turn t = s.getTurn();
		// "D": patta dichiarata dal server (regola delle ripetizioni): lo stato del motore e' ancora non terminale, la scacchiera deve pero' coincidere
		return (token.equals("W") && t == Turn.WHITE) || (token.equals("B") && t == Turn.BLACK) || (token.equals("WW") && t == Turn.WHITEWIN)
				|| (token.equals("BW") && t == Turn.BLACKWIN) || (token.equals("D") && (t == Turn.DRAW || t == Turn.WHITE || t == Turn.BLACK));
	}

	/**
	 * Importa un log o tutti i {@code *gameLog*.txt} di una cartella (in ordine di nome, per un risultato deterministico).
	 *
	 * @param keepUnlabeled tiene anche le posizioni senza etichetta (partite senza esito vero): per l'addestramento non servono
	 */
	public static NnueDataset fromServerLogs(String path, boolean dedupe, boolean keepUnlabeled, Report rep) throws IOException {
		final File f = new File(path);
		final List<File> files = new ArrayList<>();
		if (f.isDirectory()) {
			final File[] all = f.listFiles();
			if (all != null) {
				for (File x : all) {
					if (x.isFile() && x.getName().contains("gameLog") && x.getName().endsWith(".txt")) {
						files.add(x);
					}
				}
			}
			Collections.sort(files);
		} else if (f.isFile()) {
			files.add(f);
		} else {
			throw new IOException("log inesistente: " + path);
		}
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final Set<String> seen = new HashSet<>();
		int game = 0;
		for (File x : files) {
			rep.files++;
			final String text = new String(java.nio.file.Files.readAllBytes(x.toPath()), "ISO-8859-1");
			final ParsedLog p = parseLog(text);
			if (p.boards.isEmpty()) {
				rep.note("file senza partita: " + x.getName());
				continue;
			}
			rep.games++;
			rep.rejectedMoves += p.rejectedMoves;
			if (dedupe && !seen.add(moveKey(p))) {
				rep.duplicateGames++;
				continue;
			}
			final Replay r = replay(p);
			switch (r.end) {
			case "COMPLETE":
				rep.completeGames++;
				break;
			case "SERVER_DRAW":
				rep.completeGames++;
				rep.serverDraws++;
				break;
			case "FORFEIT_NULL":
			case "FORFEIT_RESULT":
				rep.forfeitGames++;
				break;
			case "ILLEGAL":
				rep.illegalGames++;
				break;
			case "DIVERGENCE":
				rep.divergentGames++;
				rep.note("divergenza dalle regole in " + x.getName() + " (" + r.detail + ")");
				break;
			case "BAD_LOG":
				rep.note("log non interpretabile: " + x.getName());
				break;
			default:
				rep.incompleteGames++;
			}
			final int key = game++;
			for (int i = 0; i < r.positions.size(); i++) {
				rep.positions++;
				final byte[] pos = r.positions.get(i);
				if (NnueDataset.checkPosition(pos, 0) != null) {
					rep.droppedInvalid++;
					continue;
				}
				if (!r.outValid && !keepUnlabeled) {
					rep.droppedUnlabeled++;
					continue;
				}
				b.add(pos, 0, r.roles.get(i), r.outcome, r.outValid, 0, false, key, i, r.outValid ? r.applied - i : -1, NnueDataset.SRC_LOG);
				rep.kept++;
			}
		}
		return b.build();
	}

	private static String moveKey(ParsedLog p) {
		final StringBuilder sb = new StringBuilder();
		sb.append(Arrays.toString(p.boards.get(0))).append('|').append(p.tokens.get(0));
		for (int[] m : p.moves) {
			sb.append(';').append(m == null ? "null" : Arrays.toString(m));
		}
		return sb.toString();
	}

	// ====================================================================== dati.gz e DistillSet

	private static void skipFully(DataInputStream in, long bytes) throws IOException {
		while (bytes > 0) {
			final long s = in.skip(bytes);
			if (s <= 0) {
				if (in.read() < 0) {
					throw new EOFException();
				}
				bytes--;
			} else {
				bytes -= s;
			}
		}
	}

	private static int codeOfLegacy(int code) {
		return code == 3 ? BoardConstants.KING : code;
	}

	/** Importa un {@code dati.gz} ({@code TABLUT_CNN_DATA v1}): esito sempre vero (le partite troncate sono scartate alla generazione), niente valore. */
	public static NnueDataset fromCnnData(String path, Report rep) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16))) {
			final String magic = in.readUTF();
			if (!"TABLUT_CNN_DATA v1".equals(magic)) {
				throw new IOException("non e' un dataset CNN (TABLUT_CNN_DATA v1): " + path);
			}
			in.readInt(); // versione delle feature: non serve
			final int nf = in.readInt(), n = in.readInt();
			if (n < 0 || n > NnueDataset.MAX_SAMPLES || nf < 1 || nf > 4096) {
				throw new IOException("intestazione non plausibile (n=" + n + ", nf=" + nf + "): " + path);
			}
			final byte[] cells = new byte[81 * n], role = new byte[n], outcome = new byte[n];
			final int[] game = new int[n];
			in.readFully(cells);
			in.readFully(role);
			in.readFully(outcome);
			for (int i = 0; i < n; i++) {
				game[i] = in.readInt();
			}
			skipFully(in, (long) n * nf * 4);
			if (in.read() != -1) {
				throw new IOException("byte in piu' dopo i dati (" + path + ")");
			}
			rep.files++;
			return legacyToDataset(n, cells, role, outcome, null, game, null, rep);
		} catch (EOFException e) {
			throw new IOException("file troncato: " + path, e);
		} catch (java.util.zip.ZipException e) {
			throw new IOException("file non valido o corrotto (" + e.getMessage() + "): " + path, e);
		}
	}

	/** Importa un {@code DistillSet} ({@code TABLUT_DISTILL_SET v1}): usa il suo {@code outValid}; il valore non c'e'. */
	public static NnueDataset fromDistillSet(String path, Report rep) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16))) {
			final String magic = in.readUTF();
			if (!"TABLUT_DISTILL_SET v1".equals(magic)) {
				throw new IOException("non e' un DistillSet (TABLUT_DISTILL_SET v1): " + path);
			}
			in.readInt();
			final int nf = in.readInt(), n = in.readInt();
			if (n < 0 || n > NnueDataset.MAX_SAMPLES || nf < 1 || nf > 4096) {
				throw new IOException("intestazione non plausibile (n=" + n + ", nf=" + nf + "): " + path);
			}
			final byte[] cells = new byte[81 * n], role = new byte[n], outcome = new byte[n], outValid = new byte[n], source = new byte[n];
			final int[] game = new int[n];
			in.readFully(cells);
			in.readFully(role);
			in.readFully(outcome);
			in.readFully(outValid);
			in.readFully(source);
			for (int i = 0; i < n; i++) {
				game[i] = in.readInt();
				in.readBoolean(); // heldOut: lo decide NnueDataset.isHeldOut
			}
			skipFully(in, (long) n * nf * 4);
			if (in.read() != -1) {
				throw new IOException("byte in piu' dopo i dati (" + path + ")");
			}
			rep.files++;
			return legacyToDataset(n, cells, role, outcome, outValid, game, source, rep);
		} catch (EOFException e) {
			throw new IOException("file troncato: " + path, e);
		} catch (java.util.zip.ZipException e) {
			throw new IOException("file non valido o corrotto (" + e.getMessage() + "): " + path, e);
		}
	}

	private static NnueDataset legacyToDataset(int n, byte[] cells, byte[] role, byte[] outcome, byte[] outValid, int[] game, byte[] source, Report rep) {
		final NnueDataset.Builder b = new NnueDataset.Builder();
		final byte[] pos = new byte[81];
		final Set<Integer> games = new HashSet<>();
		for (int i = 0; i < n; i++) {
			rep.positions++;
			for (int sq = 0; sq < 81; sq++) {
				pos[sq] = (byte) codeOfLegacy(cells[81 * i + sq]);
			}
			final boolean valid = outValid == null || outValid[i] == 1;
			if (NnueDataset.checkPosition(pos, 0) != null || (role[i] != 0 && role[i] != 1) || outcome[i] < -1 || outcome[i] > 1 || game[i] < 0) {
				rep.droppedInvalid++;
				continue;
			}
			if (!valid) {
				rep.droppedUnlabeled++;
				continue;
			}
			b.add(pos, 0, role[i], outcome[i], true, 0, false, game[i], -1, -1,
					source != null && source[i] == 1 ? NnueDataset.SRC_SELFPLAY : NnueDataset.SRC_IMPORT);
			rep.kept++;
			games.add(game[i]);
		}
		rep.games += games.size();
		rep.completeGames += games.size();
		return b.build();
	}

	// ====================================================================== main

	public static void main(String[] args) throws IOException {
		String out = null;
		final List<String> logs = new ArrayList<>(), cnn = new ArrayList<>(), distill = new ArrayList<>();
		boolean dedupe = true, keep = false;
		for (String a : args) {
			final int eq = a.indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + a);
				System.exit(2);
			}
			final String k = a.substring(0, eq).toLowerCase(), v = a.substring(eq + 1);
			switch (k) {
			case "out":
				out = v;
				break;
			case "logs":
				logs.addAll(Arrays.asList(v.split(",")));
				break;
			case "cnndata":
				cnn.addAll(Arrays.asList(v.split(",")));
				break;
			case "distill":
				distill.addAll(Arrays.asList(v.split(",")));
				break;
			case "dedupe":
				dedupe = Boolean.parseBoolean(v);
				break;
			case "keepunlabeled":
				keep = Boolean.parseBoolean(v);
				break;
			default:
				System.out.println("opzione sconosciuta: " + k);
				System.exit(2);
			}
		}
		if (out == null || (logs.isEmpty() && cnn.isEmpty() && distill.isEmpty())) {
			System.out.println("Uso: NnueDataImport out=FILE.gz [logs=DIR|FILE[,...]] [cnndata=FILE[,...]] [distill=FILE[,...]] [dedupe=true] [keepunlabeled=false]");
			System.exit(2);
		}
		final Report rep = new Report();
		final List<NnueDataset> parts = new ArrayList<>();
		try {
			for (String p : logs) {
				final NnueDataset d = fromServerLogs(p, dedupe, keep, rep);
				System.out.println("log " + p + ": " + d.describe());
				parts.add(d);
			}
			for (String p : cnn) {
				final NnueDataset d = fromCnnData(p, rep);
				System.out.println("dati CNN " + p + ": " + d.describe());
				parts.add(d);
			}
			for (String p : distill) {
				final NnueDataset d = fromDistillSet(p, rep);
				System.out.println("DistillSet " + p + ": " + d.describe());
				parts.add(d);
			}
			final NnueDataset all = NnueDataset.merge(parts);
			all.save(out);
			System.out.println(rep);
			System.out.println("scritto " + out + ": " + all.describe());
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}
}
