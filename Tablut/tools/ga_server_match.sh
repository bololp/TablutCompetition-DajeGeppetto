#!/bin/sh
# Partite col SERVER VERO (regole Ashton, -r 4) fra il giocatore genetico e il riferimento, con i CLIENT VERI (protocollo, tempo, nessun calcolo
# nel turno avversario). I colori si alternano. NON e' lo strumento statistico (il server parte sempre dalla posizione iniziale e i motori sono
# quasi deterministici: le partite si ripetono): serve a confermare protocollo, legalita' e rispetto del timeout in condizioni reali.
#
# Uso, dalla cartella Tablut:
#   sh tools/ga_server_match.sh GAMES TIMEOUT GENOMA.genome [BASE.nnue] [RIF.nnue] [cartellaClassi]
#   (default: BASE = RIF = pesi_nnue/trained_v6.nnue, classi in /tmp/tablut_ga_build)
# Variabili: CLIENT_OPTS (opzioni aggiunte a entrambi i client, es. "threads=1"), JAVA_OPTS (default -Xmx2g). Porte 5800/5801 libere.
# Il server scrive i suoi log in una cartella temporanea (non in logs/ del progetto).
GAMES="$1"; TIMEOUT="$2"; GENOME="$3"
BASE="${4:-pesi_nnue/trained_v6.nnue}"; REF="${5:-$BASE}"; OUT="${6:-/tmp/tablut_ga_build}"
[ -n "$GAMES" ] && [ -n "$TIMEOUT" ] && [ -n "$GENOME" ] || { echo "uso: sh tools/ga_server_match.sh GAMES TIMEOUT GENOMA [BASE] [RIF] [classi]"; exit 2; }
ROOT="$(pwd)"
abs() { case "$1" in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }   # percorsi relativi alla cartella Tablut
GENOME="$(abs "$GENOME")"; BASE="$(abs "$BASE")"; REF="$(abs "$REF")"; OUT="$(abs "$OUT")"
LIBS="$ROOT/lib/gson-2.2.2.jar:$ROOT/lib/commons-cli-1.4.jar"
CP="$OUT:$LIBS"; P=it.unibo.ai.didattica.competition.tablut
JAVA_OPTS="${JAVA_OPTS:--Xmx2g}"
W="$(mktemp -d)"; cd "$W" || exit 1
cand_wins=0; ref_wins=0; draws=0; other=0
i=1
while [ "$i" -le "$GAMES" ]; do
	java $JAVA_OPTS -cp "$CP" $P.server.Server -t "$TIMEOUT" > "$W/server_$i.log" 2>&1 &
	SRV=$!
	sleep 2
	if [ $((i % 2)) -eq 1 ]; then CW=cand; else CW=ref; fi    # i dispari: il candidato e' Bianco
	for COLOR in WHITE BLACK; do
		if { [ "$COLOR" = WHITE ] && [ "$CW" = cand ]; } || { [ "$COLOR" = BLACK ] && [ "$CW" = ref ]; }; then
			java $JAVA_OPTS -cp "$CP" $P.client.TablutGeneticClient "$COLOR" "$TIMEOUT" localhost genome="$GENOME" base="$BASE" $CLIENT_OPTS > "$W/${COLOR}_$i.log" 2>&1 &
		else
			java $JAVA_OPTS -cp "$CP" $P.client.TablutNegamaxClient "$COLOR" "$TIMEOUT" localhost "$REF" eval=nnue $CLIENT_OPTS > "$W/${COLOR}_$i.log" 2>&1 &
		fi
		sleep 1
	done
	wait $SRV
	wait
	R=$(grep -E "RESULT:|Timeout|TOO MANY ERRORS" "$W/server_$i.log" | tail -1)
	case "$R" in
	*"WHITE WIN"*) if [ "$CW" = cand ]; then cand_wins=$((cand_wins+1)); else ref_wins=$((ref_wins+1)); fi ;;
	*"BLACK WIN"*) if [ "$CW" = cand ]; then ref_wins=$((ref_wins+1)); else cand_wins=$((cand_wins+1)); fi ;;
	*"DRAW"*) draws=$((draws+1)) ;;
	*) other=$((other+1)) ;;
	esac
	echo "partita $i (candidato $( [ "$CW" = cand ] && echo Bianco || echo Nero )): ${R:-nessun risultato (vedi $W/server_$i.log)}"
	i=$((i+1))
done
echo "RIEPILOGO: candidato +$cand_wins =$draws -$ref_wins, anomalie (timeout/errori/senza esito): $other   [log in $W]"
