#!/bin/sh
# Protocollo di MISURA del giocatore genetico contro il riferimento (TablutNegamaxClient + pesi_nnue/trained_v6.nnue).
# Non dichiara nulla da solo: stampa i risultati con intervallo di confidenza e il verdetto di GaMatch.
#
# Uso, dalla cartella Tablut:
#   sh tools/ga_tournament.sh CANDIDATO [RIFERIMENTO] [cartellaClassi]
#     CANDIDATO    = FILE.nnue  oppure  FILE.nnue+FILE.genome  (base + genoma, come SPEC di GaMatch)
#     RIFERIMENTO  = default pesi_nnue/trained_v6.nnue
#     cartellaClassi = default /tmp/tablut_ga_build (compilata da run_genetic_checks.sh)
# Variabili d'ambiente (default fra parentesi):
#   SEED_BASE (9001)  semi del TORNEO: scegliere valori DIVERSI da quelli usati nell'addestramento (seed=) e nel suo cancello
#   PAIRS_FIXED (200) coppie per ciascuna profondita' fissa      DEPTHS ("4 5 6")
#   PAIRS_TIMED (40)  coppie a tempo reale                        MS (2000) ms reali per mossa (margine di sicurezza 0 per entrambi)
#   THREADS (1)       partite in parallelo (1 core per partita: non superare i core fisici - e' una misura di tempo reale)
#   SKIP_TIMED (0)    1 = salta la fase a tempo reale
# Costo indicativo della fase a tempo: PAIRS_TIMED * 2 partite * ~100 mosse * MS / THREADS  (40 coppie a 2 s ~ 4,5 h con THREADS=1).
set -e
CAND="$1"
REF="${2:-pesi_nnue/trained_v6.nnue}"
OUT="${3:-/tmp/tablut_ga_build}"
[ -n "$CAND" ] || { echo "uso: sh tools/ga_tournament.sh CANDIDATO [RIFERIMENTO] [cartellaClassi]"; exit 2; }
[ -d "$OUT" ] || { echo "cartella classi '$OUT' assente: esegui prima sh tools/run_genetic_checks.sh"; exit 2; }
LIBS="lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"
SEED_BASE="${SEED_BASE:-9001}"; PAIRS_FIXED="${PAIRS_FIXED:-200}"; DEPTHS="${DEPTHS:-4 5 6}"
PAIRS_TIMED="${PAIRS_TIMED:-40}"; MS="${MS:-2000}"; THREADS="${THREADS:-1}"; SKIP_TIMED="${SKIP_TIMED:-0}"
M="java -Xss4m -Xmx2g -cp $OUT:$LIBS it.unibo.ai.didattica.competition.tablut.training.GaMatch"

echo "### Fase 1: profondita' fissa (deterministica, riproducibile con qualunque THREADS)"
for D in $DEPTHS; do
	$M a="$CAND" b="$REF" pairs="$PAIRS_FIXED" depth="$D" seed="$SEED_BASE" threads="$THREADS"
done
if [ "$SKIP_TIMED" != "1" ]; then
	echo "### Fase 2: tempo reale uguale per mossa ($MS ms), un thread di ricerca per lato (NON riproducibile bit per bit)"
	$M a="$CAND" b="$REF" pairs="$PAIRS_TIMED" ms="$MS" seed="$((SEED_BASE + 1))" threads="$THREADS"
fi
echo "### Fase 3 (a parte): partite col SERVER VERO e i client veri: sh tools/ga_server_match.sh"
