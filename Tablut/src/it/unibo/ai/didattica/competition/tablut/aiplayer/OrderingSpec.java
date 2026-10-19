package it.unibo.ai.didattica.competition.tablut.aiplayer;

/**
 * Nomi leggibili per le modalita' di ordinamento di {@link Negamax} (per arena e banchi di misura). Una specifica e' una lista
 * di nomi separati da {@code +}, applicati a {@link Negamax#ORDER_DEFAULT}: {@code base} (solo il default), {@code none} (nessun
 * ordinamento tranne la mossa TT), {@code wins}, {@code caps}, {@code capsfirst}, {@code counter}, {@code kcarry}, {@code rootsc},
 * {@code rec} ({@link Negamax#ORDER_RECOMMENDED}: wins+caps+rootsc), {@code all} (tutte con le catture dopo le killer), {@code allfirst} (tutte con le catture prima delle killer).
 * Esempio: {@code wins+counter}.
 */
public final class OrderingSpec {

	private OrderingSpec() {
	}

	public static int parse(String spec) {
		int mode = Negamax.ORDER_DEFAULT;
		for (String t : spec.split("\\+")) {
			switch (t.trim().toLowerCase()) {
			case "":
			case "base":
				break;
			case "none":
				mode = Negamax.ORDER_NONE;
				break;
			case "wins":
				mode |= Negamax.ORDER_WINS;
				break;
			case "caps":
				mode |= Negamax.ORDER_CAPTURES;
				break;
			case "capsfirst":
				mode |= Negamax.ORDER_CAPTURES_FIRST;
				break;
			case "counter":
				mode |= Negamax.ORDER_COUNTER;
				break;
			case "kcarry":
				mode |= Negamax.ORDER_KILLER_CARRY;
				break;
			case "rootsc":
				mode |= Negamax.ORDER_ROOT_SCORES;
				break;
			case "rec":
				mode |= Negamax.ORDER_RECOMMENDED;
				break;
			case "all":
				mode |= Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES | Negamax.ORDER_COUNTER | Negamax.ORDER_KILLER_CARRY
						| Negamax.ORDER_ROOT_SCORES;
				break;
			case "allfirst":
				mode |= Negamax.ORDER_WINS | Negamax.ORDER_CAPTURES_FIRST | Negamax.ORDER_COUNTER | Negamax.ORDER_KILLER_CARRY
						| Negamax.ORDER_ROOT_SCORES;
				break;
			default:
				throw new IllegalArgumentException("modalita' di ordinamento sconosciuta: " + t);
			}
		}
		return mode;
	}
}
