#!/bin/sh
# Compila il progetto (Java 8) ed esegue i test delle tecniche di riduzione dell'albero (null-move, LMR, aspirazione).
#
# Uso, dalla cartella Tablut:
#   sh tools/run_pruning_checks.sh [cartellaClassi]        (default: /tmp/tablut_prune_build)
#   RUN_EXISTING=1 sh tools/run_pruning_checks.sh           (esegue anche la batteria ESISTENTE sulla ricerca, come regressione con le tecniche spente)
#   SECTIONS=sound,tt sh tools/run_pruning_checks.sh        (solo alcune sezioni di PruningCheck)
#
# Richiede un JDK. Tempi su 1 core: PruningCheck ~100 s; la batteria esistente ~6 minuti in piu'.
# Esce con codice != 0 al primo test fallito.
set -e
OUT="${1:-/tmp/tablut_prune_build}"
LIBS="lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
T=it.unibo.ai.didattica.competition.tablut

rm -rf "$OUT"; mkdir -p "$OUT"
echo "== compilazione (javac --release 8) =="
javac --release 8 -nowarn -Xlint:-options -encoding iso-8859-1 -d "$OUT" -cp "$LIBS" $(find src -name "*.java")

# I test stampano '=== ... OK ... ===' come ultima riga: si controlla quella (e il codice di uscita).
check() {
	CLASS="$1"; shift
	echo "== $CLASS $* =="
	LAST=$(java -Xss4m -Xmx2g -cp "$OUT:$LIBS" "$T.$CLASS" "$@" | tee /dev/stderr | tail -1)
	case "$LAST" in *" OK"*) ;; *) echo "FALLITO: $CLASS"; exit 1 ;; esac
}

if [ -n "$SECTIONS" ]; then
	check aiplayer.PruningCheck pesi_nnue/trained_v6.nnue "$SECTIONS"
else
	check aiplayer.PruningCheck pesi_nnue/trained_v6.nnue
fi

if [ -n "$RUN_EXISTING" ]; then
	# con le tecniche SPENTE (default) la ricerca deve essere identica a quella storica: SearchEquivalenceCheck la confronta nodo per nodo con LegacyNegamax
	for C in aiplayer.SearchEquivalenceCheck aiplayer.PVSCheck aiplayer.MateSoundnessCheck aiplayer.HistoryExactnessCheck aiplayer.RepetitionAndStalemateCheck \
		aiplayer.ExactKeyStackCheck aiplayer.TwoSlotTTCheck aiplayer.LazySmpCheck aiplayer.NoAllocationCheck aiplayer.OrderingExtCheck aiplayer.SearchOptimizationsCheck \
		aiplayer.NnueSearchCheck aiplayer.NnueCheck aiplayer.CachingEvaluatorCheck selftest.ExactKeyCheck; do
		check "$C"
	done
fi
echo
echo "TUTTI I TEST DELLA POTATURA OK"
