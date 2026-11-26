package it.unibo.ai.didattica.competition.tablut.aiplayer;

import it.unibo.ai.didattica.competition.tablut.engine.FastState;
import it.unibo.ai.didattica.competition.tablut.engine.UndoInfo;

/**
 * Restituisce risoluzione al valore di una rete SATURA. Una NNUE addestrata su {@code tanh} e MSE da' a una posizione "vinta" un valore come +0,98 e li'
 * perde ogni sensibilita': togliere una pedina all'avversario sposta il valore di 2-4 punti su 1000 (contro 47 in apertura), cioe' la rete non
 * distingue "blocco con 8 difensori" da "blocco con 7". La ricerca trova allora tante mosse di attesa a pari valore e ne sceglie una a caso: la
 * partita non avanza. Questo wrapper aggiunge, SOLO dove il valore e' saturo, un termine di materiale non saturante a favore del lato che sta vincendo:
 * <pre>
 *   v' = v + L * g * (  pesoCatturato * (pedine nemiche perse)  -  pesoPerso * (pedine proprie perse)  )
 *   L = segno di v (il lato in vantaggio), g = clamp((|v| - gateLow) / (gateHigh - gateLow), 0, 1)
 * </pre>
 * Fuori dalla saturazione ({@code |v| < gateLow}) il valore e' ESATTAMENTE quello della rete: il gioco in apertura e a meta' partita non cambia.
 * Il termine dipende solo dalla posizione (conteggi di pedine), quindi e' compatibile con TT, sessioni incrementali e Lazy SMP.
 * Resta entro {@code +/-(WIN_SCORE - 1)}: non scavalca mai un matto dimostrato.
 */
public final class ProgressEvaluator implements IncrementalEvaluator {

	public final Evaluator inner;
	/** Valore (su scala 1000) sotto il quale il termine non agisce, e sopra {@code gateHigh} agisce al massimo. */
	public final double gateLow, gateHigh;
	/** Punti per una pedina BIANCA/NERA mangiata o persa (il Bianco e' 8 contro 16: ogni difensore pesa di piu'). */
	public final double whiteWeight, blackWeight;

	private static final int WHITE_START = 8, BLACK_START = 16;

	public ProgressEvaluator(Evaluator inner, double gateLow, double gateHigh, double whiteWeight, double blackWeight) {
		if (inner == null || !(gateHigh > gateLow) || gateLow < 0 || whiteWeight < 0 || blackWeight < 0) {
			throw new IllegalArgumentException("parametri non validi");
		}
		this.inner = inner;
		this.gateLow = gateLow;
		this.gateHigh = gateHigh;
		this.whiteWeight = whiteWeight;
		this.blackWeight = blackWeight;
	}

	public ProgressEvaluator(Evaluator inner) {
		this(inner, 700, 900, 40, 20);
	}

	/**
	 * Legge la specifica di {@code progress=}: {@code off} (nessun wrapper: restituisce {@code null}), {@code on} (valori predefiniti 700:900:40:20) oppure
	 * {@code basso:alto:pesoBianco:pesoNero}, per es. {@code 700:900:40:20}.
	 *
	 * @throws IllegalArgumentException specifica non valida
	 */
	public static ProgressEvaluator parse(String spec, Evaluator inner) {
		final String s = spec.trim().toLowerCase();
		if (s.equals("off") || s.equals("false") || s.isEmpty()) {
			return null;
		}
		if (s.equals("on") || s.equals("true")) {
			return new ProgressEvaluator(inner);
		}
		final String[] p = s.split(":");
		if (p.length != 4) {
			throw new IllegalArgumentException("atteso off, on oppure basso:alto:pesoBianco:pesoNero");
		}
		return new ProgressEvaluator(inner, Double.parseDouble(p[0]), Double.parseDouble(p[1]), Double.parseDouble(p[2]), Double.parseDouble(p[3]));
	}

	public String describe() {
		return "progresso su " + gateLow + ".." + gateHigh + ", pedina bianca " + whiteWeight + ", nera " + blackWeight;
	}

	/** Valore con il termine di progresso, a partire dal valore {@code v} della rete (prospettiva del Bianco) nella posizione {@code s}. */
	public double adjust(double v, FastState s) {
		final double a = Math.abs(v);
		if (!(a > gateLow)) { // include NaN: lo gestisce chi sta sopra
			return v;
		}
		final double g = Math.min(1.0, (a - gateLow) / (gateHigh - gateLow));
		final int whiteLost = WHITE_START - s.whiteCount, blackLost = BLACK_START - s.blackCount;
		// il lato in vantaggio e' quello con v > 0 (Bianco) o v < 0 (Nero)
		final double adv = v > 0 ? whiteWeight * 0 + blackWeight * blackLost - whiteWeight * whiteLost : whiteWeight * whiteLost - blackWeight * blackLost;
		final double r = v + Math.signum(v) * g * adv;
		return Math.max(-Negamax.EVAL_LIMIT, Math.min(Negamax.EVAL_LIMIT, r));
	}

	@Override
	public double evaluate(FastState s) {
		return adjust(inner.evaluate(s), s);
	}

	@Override
	public Session newSession(int capacityPlies) {
		final Session in = inner instanceof IncrementalEvaluator ? ((IncrementalEvaluator) inner).newSession(capacityPlies) : null;
		return new Session() {
			@Override
			public void reset(FastState root) {
				if (in != null) {
					in.reset(root);
				}
			}

			@Override
			public void push(int ply, int move, FastState after, UndoInfo undo) {
				if (in != null) {
					in.push(ply, move, after, undo);
				}
			}

			@Override
			public boolean pushNull(int ply, FastState after) {
				return in == null ? true : in.pushNull(ply, after);
			}

			@Override
			public double evaluate(int ply, FastState state) {
				return adjust(in != null ? in.evaluate(ply, state) : inner.evaluate(state), state);
			}

			@Override
			public long fallbackRebuilds() {
				return in == null ? 0 : in.fallbackRebuilds();
			}
		};
	}
}
