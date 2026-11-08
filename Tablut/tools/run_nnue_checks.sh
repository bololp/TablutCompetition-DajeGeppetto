#!/bin/sh
# Compila il progetto (Java 8) ed esegue i test della migrazione NNUE.
#
# Uso, dalla cartella Tablut:
#   sh tools/run_nnue_checks.sh [cartellaClassi]        (default: /tmp/tablut_nnue_build)
#
# Richiede un JDK (javac). Esce con codice != 0 al primo test fallito.
# La porta 5800 di localhost deve essere libera (NnueClientCheck lancia il client reale in un sottoprocesso).
set -e
OUT="${1:-/tmp/tablut_nnue_build}"
LIBS="lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
T=it.unibo.ai.didattica.competition.tablut

rm -rf "$OUT"
mkdir -p "$OUT"
echo "== compilazione (javac --release 8) =="
javac --release 8 -nowarn -Xlint:-options -encoding iso-8859-1 -d "$OUT" -cp "$LIBS" $(find src -name "*.java")

# Ogni test stampa '=== ... OK ... ===' come ultima riga (e termina con codice 1 se fallisce): si controlla l'ultima riga.
check() {
	echo "== $1 =="
	LAST=$(java -Xss4m -Xmx2g -cp "$OUT:$LIBS" "$T.$1" | tee /dev/stderr | tail -1)
	case "$LAST" in
	*" OK"*) ;;
	*) echo "FALLITO: $1"; exit 1 ;;
	esac
}

check aiplayer.NnueCheck           # rete, evaluator, sessione incrementale, file dei pesi (oracolo indipendente)
check aiplayer.NnueSearchCheck     # integrazione con Negamax: incrementale = completa = motore storico, Lazy SMP, interruzioni
check training.NnueTrainingCheck   # distillatore: gradiente, quantizzazione, dati, determinismo
check client.NnueClientCheck       # selezione eval=auto|simple|mlp|cnn|nnue nel client reale
check aiplayer.NoAllocationCheck   # zero allocazioni per nodo (include l'NNUE)

echo
echo "TUTTI I TEST NNUE OK"
