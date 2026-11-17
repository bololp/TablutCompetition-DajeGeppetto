#!/bin/sh
# Compila il progetto (Java 8) ed esegue i test del giocatore genetico (GA).
#
# Uso, dalla cartella Tablut:
#   sh tools/run_genetic_checks.sh [cartellaClassi]        (default: /tmp/tablut_ga_build)
#   LONG=1 sh tools/run_genetic_checks.sh                   (aggiunge la prova a 60 s: dura ~1 minuto in piu')
#   RUN_EXISTING=1 sh tools/run_genetic_checks.sh           (esegue anche NnueCheck e NnueClientCheck esistenti, come regressione)
#
# Richiede un JDK (javac). Le porte 5800 e 5801 di localhost devono essere libere (il test del protocollo
# lancia il client reale in un sottoprocesso e gli fa da server). Esce con codice != 0 al primo test fallito.
# Tempi indicativi su 1 core: GaGenomeCheck ~3 s, GaTrainerCheck ~10 s, TablutGeneticClientCheck ~1 minuto.
set -e
OUT="${1:-/tmp/tablut_ga_build}"
LIBS="lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
T=it.unibo.ai.didattica.competition.tablut

rm -rf "$OUT"
mkdir -p "$OUT"
echo "== compilazione (javac --release 8) =="
javac --release 8 -nowarn -Xlint:-options -encoding iso-8859-1 -d "$OUT" -cp "$LIBS" $(find src -name "*.java")

# Ogni test stampa '=== ... OK ... ===' come ultima riga (e termina con codice 1 se fallisce): si controlla l'ultima riga.
check() {
	CLASS="$1"
	shift
	echo "== $CLASS =="
	LAST=$(java -Xss4m -Xmx2g "$@" -cp "$OUT:$LIBS" "$T.$CLASS" | tee /dev/stderr | tail -1)
	case "$LAST" in
	*" OK"*) ;;
	*) echo "FALLITO: $CLASS"; exit 1 ;;
	esac
}

check training.GaGenomeCheck            # genoma: identita' bit per bit, mappatura gene->parametro, formato file, compatibilita' NNUE, incrementale=completa
check training.GaTrainerCheck           # GA: operatori, convergenza, determinismo, arresto, ripresa, verdetto statistico, file di uscita, partite vere
if [ -n "$LONG" ]; then
	check client.TablutGeneticClientCheck -Dga.long=true   # + prova completa a 60 s
else
	check client.TablutGeneticClientCheck                  # fallback/strict, argomenti, legalita', regressione, tempo, quiete, protocollo reale (5800/5801)
fi

if [ -n "$RUN_EXISTING" ]; then
	check aiplayer.NnueCheck
	check client.NnueClientCheck
fi
echo
echo "TUTTI I TEST GENETICI OK"
