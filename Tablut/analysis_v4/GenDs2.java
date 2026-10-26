import it.unibo.ai.didattica.competition.tablut.training.CnnPretrainer;
import java.io.*;
public class GenDs2 {
  public static void main(String[] a) throws Exception {
    int games = Integer.parseInt(a[0]); long seed = Long.parseLong(a[1]);
    CnnPretrainer.Dataset d = CnnPretrainer.generate(games, seed, 1, System.out);
    try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(a[2])))) {
      o.writeInt(d.n); o.write(d.cells); o.write(d.role); o.write(d.outcome);
      for (int g : d.game) o.writeInt(g);
      for (float f : d.feats) o.writeFloat(f);
    }
    System.out.println("posizioni: " + d.n);
  }
}
