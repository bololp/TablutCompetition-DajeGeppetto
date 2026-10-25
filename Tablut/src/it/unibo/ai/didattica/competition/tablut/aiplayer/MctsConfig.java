package it.unibo.ai.didattica.competition.tablut.aiplayer;

/**
 * Parametri di {@link Mcts}. Tutti hanno un default sensato; si cambiano con i campi pubblici oppure con {@link #parse(String)}, che
 * legge una lista {@code chiave=valore} separata da virgole (la forma usata dall'opzione {@code mcts=...} del client e dall'arena):
 * <pre>
 *   cpuct=1.0,tau=0.2,mm=0.0,expand=1,vloss=1.0,reuse=true,cache=auto,maxnodes=0,scale=auto,maxdepth=512,escape=true
 * </pre>
 * <ul>
 * <li>{@code cpuct}: peso dell'esplorazione nella formula PUCT {@code Q + cpuct * P * sqrt(N) / (1 + n)}.</li>
 * <li>{@code tau}: temperatura del softmax che trasforma i valori statici dei figli nei prior {@code P}: piu' e' bassa, piu' la
 * ricerca si concentra sulle mosse che la valutazione statica preferisce.</li>
 * <li>{@code mm}: peso (0..1) dei "implicit minimax backups": la stima di un arco e' {@code (1-mm)*media + mm*valoreMinimax}. 0 = MCTS
 * classico (solo la media).</li>
 * <li>{@code expand}: una mossa viene espansa (generate le mosse del figlio e valutati i nipoti) alla sua {@code (expand+1)}-esima
 * visita; prima, la visita restituisce il valore statico del figlio, gia' noto, senza costo. 0 = espansione alla prima visita.</li>
 * <li>{@code vloss}: peso della "perdita virtuale" con cui un thread rende meno attraente il cammino che sta percorrendo (parallelismo
 * ad albero): conta come {@code vloss} visite di valore -1.</li>
 * <li>{@code reuse}: riusa il sottoalbero della mossa giocata e della risposta avversaria nella ricerca successiva.</li>
 * <li>{@code cache}: log2 delle voci della cache di valutazioni (0 = spenta, {@code auto} = default: attiva, 2^20 voci, solo per la CNN). Un A/B a parita' di
 * albero ({@code experiments.MctsBench --only=cache}) ha mostrato che la cache COSTA il 17-31% con {@code SimpleEvaluator} (una valutazione da 0,3 us costa
 * meno di un accesso casuale alla tabella), e' neutra con l'MLP (0,87-1,07x) e conviene solo con la CNN (0,86-0,95x): il suo valore dipende dal costo
 * dell'evaluator.</li>
 * <li>{@code maxnodes}: tetto ai nodi espansi (0 = automatico dalla memoria dell'heap, si veda {@link Mcts#autoMaxNodes}).</li>
 * <li>{@code scale}: scala dei valori dell'evaluator ({@code auto} = quella dell'evaluator, si veda {@link Mcts#defaultValueScale}).</li>
 * <li>{@code maxdepth}: profondita' massima dell'albero (oltre, il nodo si valuta staticamente).</li>
 * <li>{@code escape}: rilevatore esatto di "fuga del Re al prossimo tratto": se dopo una mossa del Nero il Bianco ha una via libera al
 * bordo, quella mossa e' una sconfitta dimostrata senza espandere nulla (si veda {@link Mcts#kingCanEscapeNow}).</li>
 * </ul>
 */
public final class MctsConfig {

	public double cPuct = 1.0;
	public double tau = 0.2;
	public double mmWeight = 0.0;
	public int expandAfter = 1;
	public double vlossWeight = 1.0;
	public boolean reuse = true;
	/** -1 = automatica (si veda {@link Mcts#autoCacheLog2}); 0 = spenta; 4..26 = 2^n voci. */
	public int evalCacheLog2 = -1;
	public long maxNodes = 0;
	/** NaN = automatica. */
	public double valueScale = Double.NaN;
	public int maxDepth = 512;
	public boolean escapeScan = true;

	public MctsConfig() {
	}

	/** Copia. */
	public MctsConfig(MctsConfig o) {
		this.cPuct = o.cPuct;
		this.tau = o.tau;
		this.mmWeight = o.mmWeight;
		this.expandAfter = o.expandAfter;
		this.vlossWeight = o.vlossWeight;
		this.reuse = o.reuse;
		this.evalCacheLog2 = o.evalCacheLog2;
		this.maxNodes = o.maxNodes;
		this.valueScale = o.valueScale;
		this.maxDepth = o.maxDepth;
		this.escapeScan = o.escapeScan;
	}

	/**
	 * Applica a {@code base} (o ai default se nullo) le coppie {@code chiave=valore} di {@code spec}; la stringa vuota o nulla non
	 * cambia nulla.
	 *
	 * @throws IllegalArgumentException chiave sconosciuta, valore non numerico o fuori dall'intervallo ammesso
	 */
	public static MctsConfig parse(String spec, MctsConfig base) {
		MctsConfig c = base == null ? new MctsConfig() : new MctsConfig(base);
		if (spec != null && !spec.trim().isEmpty()) {
			for (String part : spec.split(",")) {
				final String p = part.trim();
				if (p.isEmpty()) {
					continue;
				}
				final int eq = p.indexOf('=');
				if (eq <= 0 || eq == p.length() - 1) {
					throw new IllegalArgumentException("opzione MCTS non valida (serve chiave=valore): '" + p + "'");
				}
				final String key = p.substring(0, eq).trim().toLowerCase();
				final String val = p.substring(eq + 1).trim();
				try {
					switch (key) {
					case "cpuct":
						c.cPuct = num(val, 0, 100, key);
						break;
					case "tau":
						c.tau = num(val, 0.005, 100, key);
						break;
					case "mm":
						c.mmWeight = num(val, 0, 1, key);
						break;
					case "expand":
						c.expandAfter = (int) num(val, 0, 1000, key);
						break;
					case "vloss":
						c.vlossWeight = num(val, 0, 100, key);
						break;
					case "reuse":
						c.reuse = bool(val, key);
						break;
					case "cache":
						if (val.equalsIgnoreCase("auto")) {
							c.evalCacheLog2 = -1;
						} else {
							c.evalCacheLog2 = (int) num(val, 0, 26, key);
							if (c.evalCacheLog2 != 0 && c.evalCacheLog2 < 4) {
								throw new IllegalArgumentException("cache: auto, 0 (spenta) oppure da 4 a 26, non " + val);
							}
						}
						break;
					case "maxnodes":
						c.maxNodes = (long) num(val, 0, 1e10, key);
						break;
					case "scale":
						c.valueScale = val.equalsIgnoreCase("auto") ? Double.NaN : num(val, 1e-3, 1e6, key);
						break;
					case "maxdepth":
						c.maxDepth = (int) num(val, 2, 100000, key);
						break;
					case "escape":
						c.escapeScan = bool(val, key);
						break;
					default:
						throw new IllegalArgumentException("opzione MCTS sconosciuta: '" + key + "'");
					}
				} catch (NumberFormatException e) {
					throw new IllegalArgumentException("opzione MCTS '" + key + "': valore non numerico '" + val + "'");
				}
			}
		}
		return c;
	}

	public static MctsConfig parse(String spec) {
		return parse(spec, null);
	}

	private static double num(String v, double min, double max, String key) {
		final double d = Double.parseDouble(v);
		if (Double.isNaN(d) || d < min || d > max) {
			throw new IllegalArgumentException("opzione MCTS '" + key + "': " + v + " fuori dall'intervallo [" + min + ", " + max + "]");
		}
		return d;
	}

	private static boolean bool(String v, String key) {
		if (v.equalsIgnoreCase("true") || v.equals("1") || v.equalsIgnoreCase("on")) {
			return true;
		}
		if (v.equalsIgnoreCase("false") || v.equals("0") || v.equalsIgnoreCase("off")) {
			return false;
		}
		throw new IllegalArgumentException("opzione MCTS '" + key + "': atteso true/false, non '" + v + "'");
	}

	@Override
	public String toString() {
		return "cpuct=" + cPuct + ",tau=" + tau + ",mm=" + mmWeight + ",expand=" + expandAfter + ",vloss=" + vlossWeight + ",reuse=" + reuse
				+ ",cache=" + (evalCacheLog2 < 0 ? "auto" : String.valueOf(evalCacheLog2)) + ",maxnodes=" + maxNodes + ",scale=" + (Double.isNaN(valueScale) ? "auto" : String.valueOf(valueScale))
				+ ",maxdepth=" + maxDepth + ",escape=" + escapeScan;
	}
}
