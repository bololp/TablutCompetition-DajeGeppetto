package it.unibo.ai.didattica.competition.tablut.client;

/** Verifica le funzioni di configurazione del client: margine di default, dimensione della TT, adattamento all'heap. */
public class NegamaxClientArgsCheck {
	static int failures = 0;

	static void check(boolean ok, String msg) {
		if (!ok) {
			failures++;
			System.out.println("FAIL - " + msg);
		}
	}

	public static void main(String[] args) {
		check(TablutNegamaxClient.defaultMarginMillis(60) == 1500, "60 s -> 1500 ms");
		check(TablutNegamaxClient.defaultMarginMillis(20) == 1500, "20 s -> 1500 ms");
		check(TablutNegamaxClient.defaultMarginMillis(19) == 300 && TablutNegamaxClient.defaultMarginMillis(2) == 300, "timeout brevi -> margine storico 300 ms");
		check(TablutNegamaxClient.ttBytes(22) == 100_663_296L, "2^22 voci = 96 MiB");
		check(TablutNegamaxClient.ttBytes(25) == 805_306_368L, "2^25 voci = 768 MiB");
		final long GiB = 1L << 30;
		check(TablutNegamaxClient.fitTtToHeap(22, 2 * GiB) == 22, "2^22 sta in 2 GiB");
		check(TablutNegamaxClient.fitTtToHeap(25, 2 * GiB) == 25, "2^25 (768 MiB) sta in 2 GiB: serve meta' dell'heap, 1 GiB");
		check(TablutNegamaxClient.fitTtToHeap(26, 2 * GiB) == 25, "2^26 (1,5 GiB) non sta in 2 GiB: scende a 2^25");
		check(TablutNegamaxClient.fitTtToHeap(25, GiB / 2) == 23, "in 512 MiB il limite e' 256 MiB: 2^23 (192 MiB) ci sta, 2^24 (384 MiB) no");
		check(TablutNegamaxClient.fitTtToHeap(22, 1L << 20) == 14, "heap di 1 MiB: il limite e' 512 KiB, 2^14 (384 KiB) ci sta e 2^15 (768 KiB) no");
		check(TablutNegamaxClient.fitTtToHeap(22, 1L << 16) == 12, "heap di 64 KiB: non scende sotto il pavimento 2^12");
		for (int log2 = 12; log2 <= 26; log2++) {
			final long heap = 1L << 31;
			final int f = TablutNegamaxClient.fitTtToHeap(log2, heap);
			check(f <= log2 && TablutNegamaxClient.ttBytes(f) <= heap / 2, "fit 2^" + log2 + " -> 2^" + f + " rispetta il limite");
			check(f == log2 || TablutNegamaxClient.ttBytes(f + 1) > heap / 2, "fit 2^" + log2 + " -> 2^" + f + " e' il massimo possibile");
		}
		if (failures == 0) {
			System.out.println("=== NEGAMAX CLIENT ARGS CHECK OK ===");
		} else {
			System.out.println("=== FALLITO: " + failures + " ===");
			System.exit(1);
		}
	}
}
