package it.unibo.ai.didattica.competition.tablut.experiments;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import java.nio.file.*; import java.util.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.Pawn;
import it.unibo.ai.didattica.competition.tablut.domain.State.Turn;
import it.unibo.ai.didattica.competition.tablut.engine.*; import it.unibo.ai.didattica.competition.tablut.engine.Move;
import it.unibo.ai.didattica.competition.tablut.selftest.TestPositions;
/** Riparte da una posizione REALE del log (Nero al tratto) e gioca N mosse del Nero: Nero = rete (con o senza termine di progresso), Bianco = rete pura a profondita' piu' bassa. */
public class StallMini {
  static FastState make(String b, boolean blackTurn) {
    State st = TestPositions.blank(); Pawn[][] bd = st.getBoard();
    for (int i = 0; i < 81; i++) { char c = b.charAt(i); bd[i/9][i%9] = c=='W'?Pawn.WHITE : c=='B'?Pawn.BLACK : c=='K'?Pawn.KING : c=='T'?Pawn.THRONE : Pawn.EMPTY; }
    st.setTurn(blackTurn ? Turn.BLACK : Turn.WHITE); return FastState.fromState(st);
  }
  public static void main(String[] a) throws Exception {
    Evaluator net = NnueEvaluator.fromFile(a[0]);
    List<String> lines = Files.readAllLines(Paths.get(a[1]));
    int startPly = Integer.parseInt(a[2]), blackMoves = Integer.parseInt(a[3]), dB = Integer.parseInt(a[4]), dW = Integer.parseInt(a[5]);
    boolean progress = a[6].equals("progress");
    int lastCap = 0; // per la storia: dall'ultima cattura (il server azzera la cache delle ripetizioni)
    int[] cnt = new int[lines.size()];
    for (int k = 0; k < lines.size(); k++) { String[] p = lines.get(k).split(" "); int c = 0; for (char ch : p[2].toCharArray()) if (ch=='W'||ch=='B') c++; cnt[k] = c; if (k>0 && cnt[k]!=cnt[k-1]) lastCap = k; }
    PositionHistory hist = new PositionHistory();
    for (int k = lastCap; k < startPly; k++) { String[] p = lines.get(k).split(" "); hist.add(make(p[2], p[1].equals("B"))); }
    String[] p0 = lines.get(startPly).split(" ");
    FastState fs = make(p0[2], p0[1].equals("B"));
    hist.add(fs);
    Evaluator evB = progress ? new ProgressEvaluator(net) : net;
    int[] buf = new int[FastState.MAX_MOVES]; UndoInfo u = new UndoInfo();
    int startW = fs.whiteCount; String result = "in corso"; int ply = 0, firstCapPly = -1;
    for (int bm = 0; bm < blackMoves && result.equals("in corso"); bm++) {
      for (int side = 0; side < 2 && result.equals("in corso"); side++) {
        boolean blackToMove = fs.turn == it.unibo.ai.didattica.competition.tablut.engine.BoardConstants.BLACK_TURN;
        Negamax ng = new Negamax(blackToMove ? evB : net, blackToMove ? dB : dW, true, 1<<20, true, Negamax.ORDER_DEFAULT, 1);
        int m = ng.findBestMove(FastState.copyOf(fs), 100_000_000L, hist);
        if (m == Move.NONE) { int nm = fs.generateMoves(buf); result = "RICERCA SENZA MOSSA (lato "+(blackToMove?"Nero":"Bianco")+", mosse legali "+nm+", winner="+fs.winner+", depth "+ng.lastDepthReached+", score "+ng.lastScore+")"; break; }
        int before = fs.whiteCount;
        fs.makeMove(m, u); ply++;
        if (fs.whiteCount != before || u.capturedCount() > 0) { hist.clear(); if (firstCapPly < 0) firstCapPly = ply; }
        if (fs.winner == FastState.WINNER_BLACK) result = "VITTORIA NERO (Re catturato)";
        else if (fs.winner == FastState.WINNER_WHITE) result = "VITTORIA BIANCO (Re fuggito)";
        else if (hist.contains(fs)) result = "PAREGGIO per ripetizione";
        hist.add(fs);
      }
    }
    System.out.printf("%-9s da ply %d: dopo %d ply  bianchi %d -> %d (neri %d)  prima cattura al ply %s  esito: %s%n", progress ? "PROGRESS" : "BASE", startPly, ply, startW, fs.whiteCount, fs.blackCount, firstCapPly < 0 ? "-" : String.valueOf(firstCapPly), result);
  }
}
