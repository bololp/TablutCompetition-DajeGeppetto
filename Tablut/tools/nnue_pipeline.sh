#!/bin/sh
# Pipeline di addestramento NNUE da dati di gioco, con CANCELLO DI ACCETTAZIONE sulle partite vere.
#
#   import dei log del server -> rietichettatura con la ricerca -> divisione PER PARTITA in validazione e test
#   per ogni giro:  auto-gioco con la rete corrente -> addestramento (partenza dalla corrente) -> confronto sul TEST
#   la candidata diventa la corrente solo se batte la corrente sul TEST (partite vere mai usate per scegliere nulla)
#
# Uso, dalla cartella Tablut:
#   sh tools/nnue_pipeline.sh CARTELLA_LAVORO RETE_INIZIALE.nnue
#
# Variabili d'ambiente (tutte facoltative):
#   CP        classpath con le classi compilate e le librerie   (default: bin:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar)
#   LOGS      cartella dei log del server                       (default: logs)
#   ROUNDS    numero di giri                                    (default: 1)
#   GAMES     partite di auto-gioco per giro                    (default: 400)
#   DEPTH     profondita' di ricerca per le etichette           (default: 4)
#   EPS       probabilita' di mossa casuale dopo l'apertura     (default: 0.1)
#   EPOCHS    epoche di addestramento                           (default: 30)
#   LAMBDA    peso del valore di ricerca contro l'esito         (default: 0.5)
#   LR        learning rate iniziale                            (default: 0.001)
#   SEED      seme                                              (default: 1)
#   MARGIN    miglioramento minimo dell'MSE sull'esito reale del TEST per accettare   (default: 0.002)
#   JAVA_OPTS opzioni della JVM                                 (default: -Xmx2g)
#
# Il cancello esige una MARGINE di miglioramento perche' con poche partite vere (decine) il test e' rumoroso: una candidata che "non e'
# peggiore" non e' una candidata migliore. Se nessun giro accetta, la rete finale e' quella iniziale (copiata, invariata).
set -e

if [ $# -ne 2 ]; then
	echo "Uso: sh tools/nnue_pipeline.sh CARTELLA_LAVORO RETE_INIZIALE.nnue" >&2
	exit 2
fi
WORK="$1"
CUR="$2"
[ -f "$CUR" ] || { echo "ERRORE: rete iniziale inesistente: $CUR" >&2; exit 1; }

CP="${CP:-bin:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar}"
LOGS="${LOGS:-logs}"
ROUNDS="${ROUNDS:-1}"
GAMES="${GAMES:-400}"
DEPTH="${DEPTH:-4}"
EPS="${EPS:-0.1}"
EPOCHS="${EPOCHS:-30}"
LAMBDA="${LAMBDA:-0.5}"
LR="${LR:-0.001}"
SEED="${SEED:-1}"
MARGIN="${MARGIN:-0.002}"
JAVA_OPTS="${JAVA_OPTS:--Xmx2g}"
P=it.unibo.ai.didattica.competition.tablut.training

mkdir -p "$WORK"
j() { cls="$1"; shift; java $JAVA_OPTS -cp "$CP" "$P.$cls" "$@"; }

# MSE sull'esito reale di una rete su un dataset (riga "esito: MSE x" di NnueEvalNets)
mse() {
	j NnueEvalNets nets="$1" data="$2" lambda=0 | sed -n 's/.*esito: MSE \([0-9.]*\) .*/\1/p' | tail -1
}

echo "== 1. import dei log reali ($LOGS)"
j NnueDataImport logs="$LOGS" out="$WORK/logs.gz" | tail -2
echo "== 2. rietichettatura dei log con la ricerca a profondita' $DEPTH (rete $CUR)"
j NnueRelabel in="$WORK/logs.gz" out="$WORK/logs_lab.gz" eval="$CUR" depth="$DEPTH" | tail -1
echo "== 3. divisione per partita: validazione (sceglie l'epoca) e TEST (decide l'accettazione)"
j NnueDataTool split in="$WORK/logs_lab.gz" train="$WORK/logs_val.gz" held="$WORK/logs_test.gz" pct=50 salt=7

DATA=""
BEST="$CUR"
r=1
while [ "$r" -le "$ROUNDS" ]; do
	echo "== giro $r/$ROUNDS: auto-gioco con $BEST ($GAMES partite, profondita' $DEPTH)"
	j NnueSelfPlay eval="$BEST" out="$WORK/sp_$r.gz" games="$GAMES" depth="$DEPTH" eps="$EPS" seed=$((SEED * 100 + r)) | tail -1
	DATA="${DATA:+$DATA,}$WORK/sp_$r.gz"
	echo "== giro $r: addestramento (partenza da $BEST)"
	j NnueTrainer data="$DATA" val="$WORK/logs_val.gz" init="$BEST" out="$WORK/cand_$r.nnue" lambda="$LAMBDA" extlambda=0 epochs="$EPOCHS" lr="$LR" seed="$SEED" \
		| grep "addestramento:\|NESSUN"
	CUR_MSE=$(mse "$BEST" "$WORK/logs_test.gz")
	CAND_MSE=$(mse "$WORK/cand_$r.nnue" "$WORK/logs_test.gz")
	echo "   TEST (partite vere): MSE sull'esito: corrente $CUR_MSE, candidata $CAND_MSE (margine richiesto $MARGIN)"
	if [ -z "$CUR_MSE" ] || [ -z "$CAND_MSE" ]; then
		echo "ERRORE: impossibile leggere le metriche di test" >&2
		exit 1
	fi
	if awk -v c="$CUR_MSE" -v n="$CAND_MSE" -v m="$MARGIN" 'BEGIN { exit !(n <= c - m) }'; then
		echo "   ACCETTATA: la candidata batte la corrente sul test"
		BEST="$WORK/cand_$r.nnue"
	else
		echo "   RIFIUTATA: la corrente resta"
	fi
	r=$((r + 1))
done

cp "$BEST" "$WORK/final.nnue"
echo "== rete finale: $WORK/final.nnue (da $BEST)"
echo "   Per la forza di gioco confrontarla in arena (non fa parte del cancello):"
echo "   java -cp \"$CP\" $P.EqualTimeArena m=0:$CUR m=0:$WORK/final.nnue 15 200 1 1"
