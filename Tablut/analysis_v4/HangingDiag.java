import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.FastState;

public class HangingDiag {
  static int count(State s, Pawn p) { int n = 0; for (Pawn[] r : s.getBoard()) for (Pawn x : r) if (x == p) n++; return n; }
  static State withTurn(State s, Turn t) { State c = s.clone(); c.setTurn(t); return c; }
  /** Il Nero (a muovere in s1) puo' catturare con una mossa? Ritorna {pedineBianchePerse(max), pedinaMossaPersa(0/1), reCatturato(0/1)}. */
  static int[] blackBest(State s1, int dr, int dc) {
    int whiteBefore = count(s1, Pawn.WHITE); int maxLoss = 0, movedLost = 0, king = 0;
    for (Action m : AshtonRules.generateMoves(s1)) {
      State s2 = AshtonRules.applyMove(s1, m);
      if (s2.getTurn() == Turn.BLACKWIN) { king = 1; continue; }
      int loss = whiteBefore - count(s2, Pawn.WHITE);
      if (loss > maxLoss) maxLoss = loss;
      if (loss > 0 && dr >= 0 && s2.getBoard()[dr][dc] == Pawn.EMPTY) movedLost = 1;
    }
    return new int[] { maxLoss, movedLost, king };
  }
  static int[] cell(Action a) { String t = a.getTo(); return new int[] { t.charAt(1) - '1', t.charAt(0) - 'a' }; }
  static boolean citOrBlackAdj(State s, int r, int c) {
    int[][] d = { {-1,0},{1,0},{0,-1},{0,1} };
    for (int[] x : d) { int rr = r + x[0], cc = c + x[1]; if (rr < 0 || cc < 0 || rr > 8 || cc > 8) continue;
      if (s.getBoard()[rr][cc] == Pawn.BLACK) return true;
      int[][] cit = {{3,0},{4,0},{5,0},{4,1},{0,3},{0,4},{0,5},{1,4},{3,8},{4,8},{5,8},{4,7},{8,3},{8,4},{8,5},{7,4}};
      for (int[] q : cit) if (q[0] == rr && q[1] == cc) return true; }
    return false;
  }
  public static void main(String[] a) throws Exception {
    CnnEvaluator ev = CnnEvaluator.fromFile(a[0]);
    int games = Integer.parseInt(a[1]); long ms = Long.parseLong(a[2]); long seed = Long.parseLong(a[3]); double tmax = Double.parseDouble(a[4]);
    Random rnd = new Random(seed);
    Negamax ng = new Negamax(ev, 64, true, 1 << 20); ng.setThreads(1); ng.setSafetyMarginMillis(0);
    Negamax ng3 = new Negamax(ev, 3, true, 1 << 18); ng3.setThreads(1); ng3.setSafetyMarginMillis(0);
    long t0 = System.nanoTime();
    int wMoves = 0, pawnMoves = 0, hangMoved = 0, hangMovedPattern = 0, depthOk = 0; double depthSum = 0; int hangAny = 0;
    int evaluated = 0, chosenBest = 0, safeExists = 0, chosenBeatsSafe = 0; List<Double> margins = new ArrayList<>();
    int blackCaptured = 0; double dW = 0, dB = 0; int nEv = 0;                      // esiti reali dopo mossa "appesa"
    double dWbase = 0, dBbase = 0; int nBase = 0;                                    // riferimento: mosse bianche non appese
    int[] bPawn = new int[3], bHang = new int[3]; List<List<Double>> bMarg = new ArrayList<>(); for (int q = 0; q < 3; q++) bMarg.add(new ArrayList<>()); int gDone = 0; double sumAbsStep = 0; int nStep = 0;
    for (int g = 0; g < games; g++) {
      if ((System.nanoTime() - t0) / 1e9 > tmax) break;
      FastState f0 = new FastState(); f0.initInitialState(); State s = f0.toState();
      List<int[]> hist = new ArrayList<>(); // per ply: {whites, blacks}
      List<Integer> eventIdx = new ArrayList<>(); List<Boolean> eventHang = new ArrayList<>();
      double prevV = Double.NaN;
      for (int ply = 0; ply < 140 && !AshtonRules.isTerminal(s); ply++) {
        hist.add(new int[] { count(s, Pawn.WHITE), count(s, Pawn.BLACK) });
        Action mv;
        if (ply < 4) { List<Action> L = AshtonRules.generateMoves(s); mv = L.get(rnd.nextInt(L.size())); }
        else {
          mv = ng.findBestMove(s, ms);
          double v = ev.evaluate(FastState.fromState(s)); if (!Double.isNaN(prevV)) { sumAbsStep += Math.abs(v - prevV); nStep++; } prevV = v;
        }
        if (s.getTurn() == Turn.WHITE && ply >= 4) {
          wMoves++;
          int[] to = cell(mv); String fr = mv.getFrom(); int fr_r = fr.charAt(1) - '1', fr_c = fr.charAt(0) - 'a';
          boolean isKing = s.getBoard()[fr_r][fr_c] == Pawn.KING;
          int depth = ng.lastDepthReached;
          State s1 = AshtonRules.applyMove(s, mv);
          if (!isKing && !AshtonRules.isTerminal(s1)) {
            pawnMoves++;
            double v0r = Math.abs(ev.evaluate(FastState.fromState(s))); int bk = v0r < 300 ? 0 : v0r < 800 ? 1 : 2; bPawn[bk]++;
            int[] bb = blackBest(s1, to[0], to[1]);
            boolean hung = bb[1] == 1; if (hung) bHang[bk]++;
            if (bb[0] > 0) hangAny++;
            eventIdx.add(ply); eventHang.add(hung);
            if (hung) {
              hangMoved++; depthSum += depth; if (depth >= 2) depthOk++;
              boolean pat = citOrBlackAdj(s1, to[0], to[1]); if (pat) hangMovedPattern++;
              // alternative: valore a profondita' 3 di ogni mossa bianca legale
              double vCh = Double.NaN, bestSafe = -1e18, bestAll = -1e18; boolean safe = false;
              for (Action m2 : AshtonRules.generateMoves(s)) {
                State c = AshtonRules.applyMove(s, m2);
                double v;
                if (c.getTurn() == Turn.WHITEWIN) v = 1e6; else if (AshtonRules.isTerminal(c)) v = -1e6;
                else { ng3.findBestMove(c, 100000); v = -ng3.lastScore; }
                int[] t2 = cell(m2); String f2 = m2.getFrom();
                boolean k2 = s.getBoard()[f2.charAt(1)-'1'][f2.charAt(0)-'a'] == Pawn.KING;
                boolean h2 = !AshtonRules.isTerminal(c) && blackBest(c, k2 ? -1 : t2[0], t2[1])[1] == 1;
                if (m2.toString().equals(mv.toString())) vCh = v;
                if (v > bestAll) bestAll = v;
                if (!h2 && v > bestSafe) { bestSafe = v; safe = true; }
              }
              evaluated++; if (vCh >= bestAll - 1e-9) chosenBest++;
              if (safe) { safeExists++; double mg = vCh - bestSafe; margins.add(mg); bMarg.get(bk).add(mg); if (mg > 1e-9) chosenBeatsSafe++; }
            }
          }
        }
        s = AshtonRules.applyMove(s, mv);
      }
      hist.add(new int[] { count(s, Pawn.WHITE), count(s, Pawn.BLACK) });
      for (int k = 0; k < eventIdx.size(); k++) {
        int p = eventIdx.get(k); if (p + 3 >= hist.size()) continue;
        double dw = hist.get(p)[0] - hist.get(p + 3)[0], db = hist.get(p)[1] - hist.get(p + 3)[1];
        if (eventHang.get(k)) { dW += dw; dB += db; nEv++; } else { dWbase += dw; dBbase += db; nBase++; }
      }
      gDone++;
    }
    System.out.printf("partite %d (%.0f s) | mosse bianche (da ply 4) %d, di pedina %d%n", gDone, (System.nanoTime()-t0)/1e9, wMoves, pawnMoves);
    System.out.printf("mosse di pedina che lasciano la pedina appena mossa catturabile subito: %d (%.1f%% delle mosse di pedina); di cui accanto a citadel/pedina nera: %d%n", hangMoved, 100.0*hangMoved/Math.max(1,pawnMoves), hangMovedPattern);
    System.out.printf("   mosse di pedina dopo cui il Nero puo' catturare QUALUNQUE pedina bianca: %d (%.1f%%)%n", hangAny, 100.0*hangAny/Math.max(1,pawnMoves));
    System.out.printf("   profondita' di ricerca in quei casi: media %.2f, >=2 in %d/%d  (a >=2 il Negamax VEDE la cattura)%n", depthSum/Math.max(1,hangMoved), depthOk, hangMoved);
    System.out.printf("   analisi alternative (prof.3) su %d casi: la mossa scelta e' la migliore a prof.3 in %d; esiste un'alternativa sicura in %d; la scelta batte la migliore sicura in %d%n", evaluated, chosenBest, safeExists, chosenBeatsSafe);
    if (!margins.isEmpty()) { Collections.sort(margins); System.out.printf("   margine (scelta - miglior alternativa sicura), unita' 1000=vittoria: mediana %.1f, 25%% %.1f, 75%% %.1f%n", margins.get(margins.size()/2), margins.get(margins.size()/4), margins.get(3*margins.size()/4)); }
    System.out.printf("   nei 3 ply successivi: dopo mossa appesa  bianche perse %.2f, nere perse %.2f (n=%d) | dopo mossa non appesa bianche perse %.2f, nere perse %.2f (n=%d)%n", dW/Math.max(1,nEv), dB/Math.max(1,nEv), nEv, dWbase/Math.max(1,nBase), dBbase/Math.max(1,nBase), nBase);
    String[] bn = {"|V|<300 (equilibrata)", "300<=|V|<800", "|V|>=800 (decisa)"}; for (int q = 0; q < 3; q++) { List<Double> m = bMarg.get(q); Collections.sort(m); System.out.printf("   %s: mosse di pedina %d, appese %d (%.1f%%), margine mediano %s%n", bn[q], bPawn[q], bHang[q], 100.0*bHang[q]/Math.max(1,bPawn[q]), m.isEmpty() ? "-" : String.format("%.2f", m.get(m.size()/2))); }
    System.out.printf("variazione media |V(t+1)-V(t)| della valutazione statica nelle partite: %.1f (scala 1000)%n", sumAbsStep/Math.max(1,nStep));
  }
}
