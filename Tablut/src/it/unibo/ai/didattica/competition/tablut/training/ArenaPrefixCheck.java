package it.unibo.ai.didattica.competition.tablut.training;

import it.unibo.ai.didattica.competition.tablut.aiplayer.Negamax;
import it.unibo.ai.didattica.competition.tablut.aiplayer.OrderingSpec;

/** Verifica l'interpretazione dei prefissi di {@link EqualTimeArena} e dei nomi di {@link OrderingSpec}. */
public class ArenaPrefixCheck {
	static int failures = 0;

	static void check(boolean ok, String msg) {
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	public static void main(String[] args) {
		// nessun prefisso: tutti i valori di default
		String plain = "pretrained_weights/mlp.txt";
		check(EqualTimeArena.prefixes(plain).isEmpty(), "nessun prefisso in un percorso semplice");
		check(EqualTimeArena.stripPrefix(plain).equals(plain), "stripPrefix lascia il percorso");
		check(EqualTimeArena.usesPVS(plain) && EqualTimeArena.orderingOf(plain) == Negamax.ORDER_DEFAULT, "default: PVS e ordinamento storico");
		check(EqualTimeArena.threadsOf(plain, 7) == 7 && EqualTimeArena.ttLog2Of(plain, 18) == 18 && !EqualTimeArena.twoSlotOf(plain)
				&& EqualTimeArena.marginOf(plain) == -1, "default: thread, TT, politica, margine");
		// tutti i prefissi combinati, in ordine sparso
		String all = "t4:ord=wins+caps:tt=20:tt2:m=0:pvs:simple";
		check(EqualTimeArena.stripPrefix(all).equals("simple"), "stripPrefix toglie tutti i prefissi: " + EqualTimeArena.stripPrefix(all));
		check(EqualTimeArena.threadsOf(all, 1) == 4, "tN");
		check(EqualTimeArena.ttLog2Of(all, 18) == 20 && EqualTimeArena.twoSlotOf(all) && EqualTimeArena.marginOf(all) == 0, "tt=, tt2, m=");
		check(EqualTimeArena.orderingOf(all) == (Negamax.ORDER_DEFAULT | Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES), "ord= vince su pvs:");
		check(EqualTimeArena.usesPVS(all), "pvs: usa PVS");
		check(EqualTimeArena.orderingOf("pvs:simple") == Negamax.ORDER_NONE, "pvs: = nessun ordinamento");
		check(!EqualTimeArena.usesPVS("ab:simple") && EqualTimeArena.orderingOf("ab:simple") == Negamax.ORDER_NONE, "ab: = alpha-beta puro");
		check(!EqualTimeArena.usesPVS("nopvs:simple") && EqualTimeArena.orderingOf("nopvs:simple") == Negamax.ORDER_DEFAULT, "nopvs: = alpha-beta con killer+history");
		check(EqualTimeArena.orderingOf("m=0:ord=rec:x.txt") == Negamax.ORDER_RECOMMENDED, "ord=rec");
		check(EqualTimeArena.ttLog2Of("tt=99:x", 18) == 26 && EqualTimeArena.ttLog2Of("tt=1:x", 18) == 4, "tt= limitato a [4, 26]");
		// un file il cui nome SEMBRA un prefisso ma non ha i due punti non va scambiato per tale
		check(EqualTimeArena.stripPrefix("t4_weights.txt").equals("t4_weights.txt") && EqualTimeArena.stripPrefix("tt2.txt").equals("tt2.txt"),
				"nomi di file simili ai prefissi");
		// nomi degli ordinamenti
		final int D = Negamax.ORDER_DEFAULT;
		check(OrderingSpec.parse("base") == D && OrderingSpec.parse("") == D, "base");
		check(OrderingSpec.parse("none") == Negamax.ORDER_NONE, "none");
		check(OrderingSpec.parse("wins") == (D | Negamax.ORDER_WINS), "wins");
		check(OrderingSpec.parse("caps") == (D | Negamax.ORDER_CAPTURES), "caps");
		check(OrderingSpec.parse("capsfirst") == (D | Negamax.ORDER_CAPTURES_FIRST), "capsfirst");
		check(OrderingSpec.parse("counter") == (D | Negamax.ORDER_COUNTER), "counter");
		check(OrderingSpec.parse("kcarry") == (D | Negamax.ORDER_KILLER_CARRY), "kcarry");
		check(OrderingSpec.parse("rootsc") == (D | Negamax.ORDER_ROOT_SCORES), "rootsc");
		check(OrderingSpec.parse("rec") == Negamax.ORDER_RECOMMENDED, "rec");
		check(OrderingSpec.parse("wins+caps+rootsc") == Negamax.ORDER_RECOMMENDED, "wins+caps+rootsc == rec");
		check(OrderingSpec.parse("all") == (D | Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES | Negamax.ORDER_COUNTER | Negamax.ORDER_KILLER_CARRY
				| Negamax.ORDER_ROOT_SCORES), "all");
		check(OrderingSpec.parse("allfirst") == (D | Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES_FIRST | Negamax.ORDER_COUNTER | Negamax.ORDER_KILLER_CARRY
				| Negamax.ORDER_ROOT_SCORES), "allfirst");
		boolean threw = false;
		try {
			OrderingSpec.parse("wins+inesistente");
		} catch (IllegalArgumentException e) {
			threw = true;
		}
		check(threw, "un nome sconosciuto e' rifiutato");
		if (failures == 0) {
			System.out.println("=== ARENA PREFIX CHECK OK ===");
		} else {
			System.out.println("=== ARENA PREFIX CHECK FALLITO: " + failures + " ===");
			System.exit(1);
		}
	}
}
