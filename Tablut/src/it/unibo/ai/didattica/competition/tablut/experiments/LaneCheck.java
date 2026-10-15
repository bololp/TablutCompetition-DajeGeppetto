package it.unibo.ai.didattica.competition.tablut.experiments;

import java.util.*;
import it.unibo.ai.didattica.competition.tablut.aiplayer.*;
import it.unibo.ai.didattica.competition.tablut.domain.*;
import it.unibo.ai.didattica.competition.tablut.domain.State.*;

/** Le "corsie aperte" dell'euristica attuale coincidono con le vere vie di fuga (mosse legali del Re che finiscono sul bordo)? */
public class LaneCheck {
    public static void main(String[] a) {
        List<State> ps = Bench0.positions(3000, 5);
        int n = 0, mismatchPos = 0, fakeLanes = 0, trueLanes = 0, oldLanes = 0;
        for (State s : ps) {
            Pawn[][] b = s.getBoard(); int kr = -1, kc = -1;
            for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) if (b[r][c] == Pawn.KING) { kr = r; kc = c; }
            if (kr < 0) continue;
            // vecchia definizione (SimpleEvaluator / TablutFeatures): raggio libero fino al bordo, ignorando le citadel
            int[][] D = {{-1,0},{1,0},{0,-1},{0,1}}; int old = 0;
            for (int[] d : D) { int st = 1; while (true) { int nr = kr + d[0]*st, nc = kc + d[1]*st;
                if (nr<0||nr>8||nc<0||nc>8||b[nr][nc]!=Pawn.EMPTY) break;
                if (nr==0||nr==8||nc==0||nc==8) { old++; break; } st++; } }
            // definizione vera: mosse LEGALI del Re (motore) che atterrano sul bordo
            State w = s.clone(); w.setTurn(Turn.WHITE); int real = 0;
            for (Action m : AshtonRules.generateMoves(w)) if (m.getRowFrom()==kr && m.getColumnFrom()==kc) { int r=m.getRowTo(), c=m.getColumnTo(); if (r==0||r==8||c==0||c==8) real++; }
            n++; oldLanes += old; trueLanes += real; if (old != real) { mismatchPos++; fakeLanes += old - real; }
        }
        System.out.printf("posizioni=%d | corsie 'aperte' secondo l'euristica attuale=%d | vie di fuga reali (mosse legali)=%d%n", n, oldLanes, trueLanes);
        System.out.printf("posizioni in cui le due definizioni DIVERGONO: %d (%.1f%%) | eccesso netto di corsie fantasma=%d%n", mismatchPos, 100.0*mismatchPos/n, fakeLanes);
    }
}
