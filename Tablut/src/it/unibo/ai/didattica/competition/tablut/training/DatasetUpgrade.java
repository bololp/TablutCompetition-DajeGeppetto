package it.unibo.ai.didattica.competition.tablut.training;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

import it.unibo.ai.didattica.competition.tablut.aiplayer.TablutFeatures;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Aggiorna un dataset del {@link CnnPretrainer} (file {@code dati.gz}) a una versione precedente delle feature: legge scacchiere,
 * ruolo, esiti e partite (che NON dipendono dalle feature) e RICALCOLA le feature con l'estrattore attuale. Evita di rigenerare
 * le partite. Uso: {@code DatasetUpgrade in=vecchio.gz out=nuovo.gz}.
 */
public final class DatasetUpgrade {

	private DatasetUpgrade() {
	}

	/** Come {@link CnnPretrainer.Dataset#load}, ma accetta qualunque versione/numero di feature e le ricalcola. */
	public static CnnPretrainer.Dataset loadAndRecompute(String path) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(path)), 1 << 16))) {
			if (!"TABLUT_CNN_DATA v1".equals(in.readUTF())) {
				throw new IOException("non e' un dataset CNN: " + path);
			}
			int fv = in.readInt(), nf = in.readInt(), n = in.readInt();
			if (n < 0 || n > 50_000_000 || nf < 1 || nf > 4096) {
				throw new IOException("intestazione non plausibile (n=" + n + ", nf=" + nf + ")");
			}
			byte[] cells = new byte[n * 81], role = new byte[n], outcome = new byte[n];
			int[] game = new int[n];
			in.readFully(cells);
			in.readFully(role);
			in.readFully(outcome);
			for (int i = 0; i < n; i++) {
				game[i] = in.readInt();
			}
			long skip = (long) n * nf * 4;
			while (skip > 0) { // le vecchie feature si scartano
				long s = in.skip(skip);
				if (s <= 0) {
					if (in.read() < 0) {
						throw new IOException("file troncato");
					}
					s = 1;
				}
				skip -= s;
			}
			final int NF = TablutFeatures.NUM_FEATURES;
			float[] feats = new float[n * NF];
			TablutFeatures.Workspace ws = TablutFeatures.newWorkspace();
			double[] full = new double[NF];
			for (int i = 0; i < n; i++) {
				State s = new StateTablut();
				Pawn[][] b = s.getBoard();
				for (int r = 0; r < 9; r++) {
					for (int c = 0; c < 9; c++) {
						int code = cells[i * 81 + r * 9 + c];
						b[r][c] = code == 1 ? Pawn.WHITE : code == 2 ? Pawn.BLACK : code == 3 ? Pawn.KING
								: (r == 4 && c == 4 ? Pawn.THRONE : Pawn.EMPTY);
					}
				}
				s.setTurn(role[i] == 0 ? Turn.WHITE : Turn.BLACK);
				TablutFeatures.extract(s, ws, full, TablutFeatures.G_ALL);
				for (int f = 0; f < NF; f++) {
					feats[i * NF + f] = (float) full[f];
				}
			}
			System.out.println("dataset v" + fv + "/" + nf + " feature -> v" + TablutFeatures.VERSION + "/" + NF + ": " + n + " posizioni ricalcolate");
			return new CnnPretrainer.Dataset(cells, role, feats, outcome, game);
		}
	}

	public static void main(String[] args) throws Exception {
		String in = null, out = null;
		for (String a : args) {
			if (a.startsWith("in=")) {
				in = a.substring(3);
			} else if (a.startsWith("out=")) {
				out = a.substring(4);
			}
		}
		if (in == null || out == null) {
			System.out.println("uso: DatasetUpgrade in=vecchio.gz out=nuovo.gz");
			System.exit(2);
		}
		loadAndRecompute(in).save(out);
		System.out.println("scritto " + out);
	}
}
