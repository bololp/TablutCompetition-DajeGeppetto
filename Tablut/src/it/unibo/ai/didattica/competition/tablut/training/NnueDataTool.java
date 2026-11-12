package it.unibo.ai.didattica.competition.tablut.training;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Operazioni da riga di comando sui {@link NnueDataset}:
 * <pre>
 *   NnueDataTool info   in=A.gz[,B.gz]                         composizione dei dataset
 *   NnueDataTool merge  in=A.gz,B.gz,... out=C.gz              unione (chiavi di partita rinumerate)
 *   NnueDataTool split  in=A.gz train=T.gz held=H.gz pct=50 [salt=1]   divisione PER PARTITA: pct% delle partite in held
 *   NnueDataTool filter in=A.gz out=B.gz [sources=0,1] [needoutcome=true] [needvalue=true]
 * </pre>
 * {@code split} serve a tenere un insieme di partite VERE come test finale mai usato per decidere nulla (epoca, iperparametri), separato da quello di
 * validazione: stesso {@code salt} e {@code pct} danno sempre la stessa divisione. {@code filter} tiene i campioni delle sorgenti indicate e/o che hanno
 * l'esito e/o il valore.
 */
public final class NnueDataTool {

	private NnueDataTool() {
	}

	/** Campioni di {@code d} delle sorgenti {@code sources} (null = tutte) e con le etichette richieste. */
	public static NnueDataset filter(NnueDataset d, boolean[] sources, boolean needOutcome, boolean needValue) {
		final int[] keep = new int[d.n];
		int k = 0;
		for (int i = 0; i < d.n; i++) {
			if ((sources == null || sources[d.source[i]]) && (!needOutcome || d.outValid[i] == 1) && (!needValue || d.valValid[i] == 1)) {
				keep[k++] = i;
			}
		}
		return d.subset(java.util.Arrays.copyOf(keep, k));
	}

	private static List<NnueDataset> loadAll(String csv) throws IOException {
		final List<NnueDataset> l = new ArrayList<>();
		for (String p : csv.split(",")) {
			if (!p.trim().isEmpty()) {
				l.add(NnueDataset.load(p.trim()));
			}
		}
		if (l.isEmpty()) {
			throw new IllegalArgumentException("nessun file in input");
		}
		return l;
	}

	public static void main(String[] args) throws IOException {
		if (args.length < 1) {
			usage();
		}
		final String cmd = args[0].toLowerCase();
		String in = null, out = null, train = null, held = null, sources = null;
		int pct = 50;
		long salt = 1;
		boolean needOutcome = false, needValue = false;
		for (int i = 1; i < args.length; i++) {
			final int eq = args[i].indexOf('=');
			if (eq < 0) {
				System.out.println("argomento non valido (atteso chiave=valore): " + args[i]);
				System.exit(2);
			}
			final String k = args[i].substring(0, eq).toLowerCase(), v = args[i].substring(eq + 1);
			try {
				switch (k) {
				case "in": in = v; break;
				case "out": out = v; break;
				case "train": train = v; break;
				case "held": held = v; break;
				case "pct": pct = Integer.parseInt(v); break;
				case "salt": salt = Long.parseLong(v); break;
				case "sources": sources = v; break;
				case "needoutcome": needOutcome = Boolean.parseBoolean(v); break;
				case "needvalue": needValue = Boolean.parseBoolean(v); break;
				default:
					System.out.println("opzione sconosciuta: " + k);
					System.exit(2);
				}
			} catch (NumberFormatException e) {
				System.out.println("valore non valido per " + k + ": " + v);
				System.exit(2);
			}
		}
		if (in == null) {
			usage();
		}
		try {
			switch (cmd) {
			case "info":
				for (String p : in.split(",")) {
					System.out.println(p + ": " + NnueDataset.load(p.trim()).describe());
				}
				break;
			case "merge": {
				if (out == null) {
					usage();
				}
				final NnueDataset m = NnueDataset.merge(loadAll(in));
				m.save(out);
				System.out.println("scritto " + out + ": " + m.describe());
				break;
			}
			case "split": {
				if (train == null || held == null || pct < 0 || pct > 100) {
					usage();
				}
				final NnueDataset d = NnueDataset.merge(loadAll(in));
				final int[][] sp = d.splitByGame(salt, pct);
				final NnueDataset t = d.subset(sp[0]), h = d.subset(sp[1]);
				t.save(train);
				h.save(held);
				System.out.println("train " + train + ": " + t.describe());
				System.out.println("held  " + held + ": " + h.describe());
				break;
			}
			case "filter": {
				if (out == null) {
					usage();
				}
				boolean[] src = null;
				if (sources != null) {
					src = new boolean[16];
					for (String s : sources.split(",")) {
						final int id = Integer.parseInt(s.trim());
						if (id < 0 || id > 15) {
							throw new NumberFormatException(s);
						}
						src[id] = true;
					}
				}
				final NnueDataset f = filter(NnueDataset.merge(loadAll(in)), src, needOutcome, needValue);
				f.save(out);
				System.out.println("scritto " + out + ": " + f.describe());
				break;
			}
			default:
				usage();
			}
		} catch (IllegalArgumentException e) { // include NumberFormatException
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(2);
		} catch (IOException e) {
			System.out.println("ERRORE: " + e.getMessage());
			System.exit(1);
		}
	}

	private static void usage() {
		System.out.println("Uso: NnueDataTool info in=A.gz[,B.gz] | merge in=A.gz,B.gz out=C.gz | split in=A.gz train=T.gz held=H.gz [pct=50] [salt=1]"
				+ " | filter in=A.gz out=B.gz [sources=0,1] [needoutcome=true] [needvalue=true]");
		System.exit(2);
	}
}
