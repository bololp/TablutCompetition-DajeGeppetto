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
check training.NnueDataCheck       # formato dati, importatori (log veri del server, dati.gz, DistillSet), strumento dataset
check training.NnueSelfPlayCheck   # auto-gioco rigiocato con le regole storiche, rietichettatura, CLI dei dati
check training.NnueTrainerCheck    # addestramento da dati di gioco: apprende, non regredisce, i dati esterni non addestrano
check client.NnueClientCheck       # selezione eval=auto|simple|mlp|cnn|nnue nel client reale
check aiplayer.NoAllocationCheck   # zero allocazioni per nodo (include l'NNUE)

# ---- smoke test della pipeline di addestramento (tools/nnue_pipeline.sh): rifiuto, accettazione, errori
echo "== tools/nnue_pipeline.sh (smoke test)"
if [ ! -d logs ]; then
	echo "(cartella 'logs' con i log del server assente: smoke test della pipeline SALTATO)"
	echo
	echo "TUTTI I TEST NNUE OK (pipeline saltata)"
	exit 0
fi
W=$(mktemp -d)
INIT="$W/init.nnue"
cat > "$W/MakeNet.java" <<'EOF2'
public class MakeNet {
	public static void main(String[] a) throws Exception {
		it.unibo.ai.didattica.competition.tablut.aiplayer.NnueNetwork.random(16, 4, 3).save(a[0]);
	}
}
EOF2
javac -cp "$OUT:$LIBS" -d "$W" "$W/MakeNet.java"
java -cp "$W:$OUT:$LIBS" MakeNet "$INIT"
export CP="$OUT:$LIBS" ROUNDS=1 GAMES=6 DEPTH=2 EPOCHS=2 SEED=1
# rifiuto: margine impossibile -> la finale e' l'iniziale, invariata
MARGIN=100 sh tools/nnue_pipeline.sh "$W/rej" "$INIT" > "$W/rej.log"
grep -q "RIFIUTATA" "$W/rej.log" || { echo "FALLITO: pipeline, ramo di rifiuto"; exit 1; }
cmp -s "$W/rej/final.nnue" "$INIT" || { echo "FALLITO: pipeline, dopo un rifiuto la finale deve essere l'iniziale"; exit 1; }
# accettazione: margine molto negativo -> la finale e' la candidata
MARGIN=-100 sh tools/nnue_pipeline.sh "$W/acc" "$INIT" > "$W/acc.log"
grep -q "ACCETTATA" "$W/acc.log" || { echo "FALLITO: pipeline, ramo di accettazione"; exit 1; }
cmp -s "$W/acc/final.nnue" "$W/acc/cand_1.nnue" || { echo "FALLITO: pipeline, dopo un'accettazione la finale deve essere la candidata"; exit 1; }
# errori: senza argomenti (uscita 2) e con una rete iniziale inesistente (uscita 1)
if sh tools/nnue_pipeline.sh >/dev/null 2>&1; then echo "FALLITO: pipeline senza argomenti deve fallire"; exit 1; fi
if sh tools/nnue_pipeline.sh "$W/x" /nonexistent_dir_xyz/none.nnue >/dev/null 2>&1; then echo "FALLITO: pipeline con rete inesistente deve fallire"; exit 1; fi
echo "pipeline OK (rifiuto, accettazione, errori)"
rm -rf "$W"

echo
echo "TUTTI I TEST NNUE OK"
