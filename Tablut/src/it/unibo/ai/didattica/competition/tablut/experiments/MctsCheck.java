package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import it.unibo.ai.didattica.competition.tablut.aiplayer.AshtonRules;
import it.unibo.ai.didattica.competition.tablut.aiplayer.NeuralEvaluator;
import it.unibo.ai.didattica.competition.tablut.aiplayer.SimpleEvaluator;
import it.unibo.ai.didattica.competition.tablut.domain.Action;
import it.unibo.ai.didattica.competition.tablut.domain.State;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.domain.StateTablut;

/**
 * Collaudo di {@link MctsPlayer}: mosse sempre legali; trova le vittorie in una mossa per entrambi i colori (posizioni trovate con
 * l'oracolo {@link AshtonRules}, non costruite a mano); segno del valore del radice; rispetta il tempo; casi limite (posizione terminale,
 * senza mosse).
 */
public class MctsCheck {

	static int failures = 0, checks = 0;

	static void check(boolean ok, String msg) {
		checks++;
		System.out.println((ok ? "OK   - " : "FAIL - ") + msg);
		if (!ok) {
			failures++;
		}
	}

	static boolean wins(State s, Action m) {
		Turn t = AshtonRules.applyMove(s, m).getTurn();
		return t == (s.getTurn() == Turn.WHITE ? Turn.WHITEWIN : Turn.BLACKWIN);
	}

	public static void main(String[] args) {
		System.out.println("=== MctsCheck ===");
		NeuralEvaluator ev = NeuralEvaluator.randomInit(3); // rete casuale: il test non deve dipendere dalla qualita' dell'evaluator
		MctsPlayer mcts = new MctsPlayer(ev, 1000.0, 1.0, 0.2);
		Random rnd = new Random(4);
		List<State> winW = new ArrayList<>(), winB = new ArrayList<>(), plain = new ArrayList<>();
		for (int g = 0; g < 4000 && (winW.size() < 8 || winB.size() < 8 || plain.size() < 40); g++) {
			State s = new StateTablut();
			s.setTurn(Turn.WHITE);
			int steps = 10 + rnd.nextInt(120);
			for (int i = 0; i < steps && !AshtonRules.isTerminal(s); i++) {
				List<Action> mv = AshtonRules.generateMoves(s);
				if (mv.isEmpty()) {
					break;
				}
				boolean win = false;
				for (Action a : mv) {
					win |= wins(s, a);
				}
				if (win) {
					(s.getTurn() == Turn.WHITE ? winW : winB).add(s);
					break;
				}
				if (plain.size() < 40 && rnd.nextInt(20) == 0) {
					plain.add(s);
				}
				s = AshtonRules.applyMove(s, mv.get(rnd.nextInt(mv.size())));
			}
		}
		System.out.println("(posizioni trovate: " + winW.size() + " con vittoria immediata del Bianco, " + winB.size() + " del Nero, " + plain.size() + " normali)");

		boolean legal = true;
		for (State s : plain) {
			Action m = mcts.findBestMove(s, 15);
			boolean ok = false;
			for (Action a : AshtonRules.generateMoves(s)) {
				ok |= m != null && a.toString().equals(m.toString());
			}
			legal &= ok;
		}
		check(legal && plain.size() >= 20, "mossa sempre legale su " + plain.size() + " posizioni");

		for (int side = 0; side < 2; side++) {
			List<State> ps = side == 0 ? winW : winB;
			int found = 0;
			double worstValue = side == 0 ? 1 : -1;
			for (State s : ps) {
				Action m = mcts.findBestMove(s, 30);
				found += m != null && wins(s, m) ? 1 : 0;
				worstValue = side == 0 ? Math.min(worstValue, mcts.lastRootValue) : Math.max(worstValue, mcts.lastRootValue);
			}
			check(ps.size() >= 5 && found == ps.size(), "trova la vittoria in una mossa del " + (side == 0 ? "Bianco" : "Nero") + ": " + found + "/" + ps.size());
			check(side == 0 ? worstValue > 0.5 : worstValue < -0.5, "valore del radice col segno giusto quando " + (side == 0 ? "vince il Bianco" : "vince il Nero")
					+ " (valore piu' sfavorevole osservato " + String.format("%.2f", worstValue) + ")");
		}

		// prospettiva del valore: con un evaluator costante il valore del radice e' lo stesso costante, per QUALUNQUE lato muova
		// (un segno sbagliato nella propagazione delle foglie non terminali lo ribalterebbe)
		for (double c : new double[] { 900.0, -900.0 }) {
			final double cc = c;
			MctsPlayer mc = new MctsPlayer(new it.unibo.ai.didattica.competition.tablut.aiplayer.Evaluator() {
				@Override
				public double evaluate(State st) {
					return cc;
				}
			}, 1000.0, 1.0, 0.2);
			int close = 0, whiteRoots = 0, blackRoots = 0;
			double worst = 0;
			for (State st : plain) {
				mc.findBestMove(st, 10);
				double dev = Math.abs(mc.lastRootValue - c / 1000.0);
				close += dev < 0.1 ? 1 : 0;
				worst = Math.max(worst, dev);
				whiteRoots += st.getTurn() == Turn.WHITE ? 1 : 0;
				blackRoots += st.getTurn() == Turn.BLACK ? 1 : 0;
			}
			// NON "tutti": in qualche posizione un esito terminale e' raggiungibile in 3 semi-mosse (verificato a forza bruta con le regole) e
			// l'MCTS lo scopre, contraddicendo giustamente l'evaluator costante. Un segno sbagliato, invece, ribalta TUTTI i radici.
			check(close >= 0.9 * plain.size() && whiteRoots > 5 && blackRoots > 5, "evaluator costante " + c + ": valore del radice = " + c / 1000.0 + " in " + close + "/"
					+ plain.size() + " radici (Bianco " + whiteRoots + ", Nero " + blackRoots + "); le eccezioni sono tattiche reali, non un errore di segno");
		}

		long t0 = System.nanoTime();
		mcts.findBestMove(plain.get(0), 100);
		double ms = (System.nanoTime() - t0) / 1e6;
		check(ms < 250 && mcts.lastSims > 20 && mcts.lastEvals > mcts.lastSims, "rispetta il tempo (100 ms richiesti, " + String.format("%.0f", ms) + " ms usati, " + mcts.lastSims
				+ " simulazioni, " + mcts.lastEvals + " valutazioni)");

		State term = new StateTablut();
		term.setTurn(Turn.WHITEWIN);
		check(mcts.findBestMove(term, 20) == null, "posizione terminale -> nessuna mossa (null), nessuna eccezione");
		check(mcts.findBestMove(plain.get(1), 0) != null, "con budget 0 esegue comunque almeno una simulazione e risponde");

		// contro un evaluator senza rete: stessi invarianti
		MctsPlayer m2 = new MctsPlayer(new SimpleEvaluator(), 500.0, 1.0, 0.2);
		boolean ok2 = true;
		for (State s : winW) {
			ok2 &= wins(s, m2.findBestMove(s, 30));
		}
		check(ok2, "anche con SimpleEvaluator trova le vittorie in una mossa del Bianco");
		System.out.println("=== " + (failures == 0 ? "MCTS CHECK OK" : "MCTS CHECK FALLITO (" + failures + ")") + " (" + checks + " verifiche) ===");
		if (failures > 0) {
			System.exit(1);
		}
	}
}
