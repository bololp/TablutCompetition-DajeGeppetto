# NNUE temporale: storia delle mosse in ingresso e bersagli multi-mossa

Estende la NNUE esistente (243 feature, `NnueNetwork`) con due leve **indipendenti** per dare alla valutazione il senso del "come ci siamo arrivati".
Niente del formato `TABLUT_NNUE` e del percorso di ricerca di base e' cambiato: `eval=nnue` si comporta come prima.

## Cosa c'e'

| Leva | Dove | Cosa fa |
|---|---|---|
| **Storia in ingresso** (`hist=K`) | `TemporalHistory`, `TemporalNnueNetwork`, `TemporalNnueEvaluator`, `TemporalSession` | Alle 243 feature di posizione si aggiungono, nello STESSO accumulatore, le feature delle ultime K mosse (pedina, da, a, catture; 488 feature per eta'). Il blocco temporale parte a ZERO: con `tW = 0` o finestra vuota la rete e' esattamente la NNUE di base, bit per bit. |
| **Bersagli multi-mossa** (`tdlambda=mu`) | `TemporalData.lambdaReturn` | Il bersaglio di una posizione include i valori di ricerca delle posizioni SUCCESSIVE della stessa partita (TD(lambda) in avanti). `mu=1` = comportamento di `NnueTrainer` (esito della partita), `mu=0` = un solo passo. |

Addestramento: `training/NnueTemporalTrainer` (dati = i soliti `NnueDataset`; le storie si ricostruiscono dal confronto fra posizioni consecutive della stessa partita).
Gioco: file `TABLUT_TNNUE v1`, riconosciuto da `EvaluatorLoader` (quindi anche da `eval=auto`) e da `eval=tnnue`; il client passa alla rete la storia REALE prima della radice (`GameMoveTracker`).

## Uso

```sh
cd Tablut
P=it.unibo.ai.didattica.competition.tablut
# partire da una NNUE gia' addestrata (blocco temporale a zero) e addestrare con storia K=4 e bersagli TD(0.5); val= misura soltanto
java -cp "bin:lib/*" $P.training.NnueTemporalTrainer data=dati_nnue_d2.gz val=dati_nnue_d4.gz extlambda=1 \
     init=pesi_nnue/trained_v64.nnue out=temporale.tnnue hist=4 tdlambda=0.5 epochs=8 maxgames=30000
# stessa configurazione SENZA storia (controllo "solo posizione")
java -cp "bin:lib/*" $P.training.NnueTemporalTrainer ... usehist=false
# misurare reti gia' addestrate (.tnnue o .nnue) senza riaddestrare
java -cp "bin:lib/*" $P.training.NnueTemporalTrainer data=... val=... eval=a.tnnue,b.nnue
# giocare
java -cp "bin:lib/*" $P.client.TablutNegamaxClient WHITE 60 localhost eval=tnnue temporale.tnnue
```
Opzioni utili: `histdrop` (prob. di troncare la finestra, default 0.5), `trainbase=false` (addestra solo il blocco temporale), `extselect=true` (i dati `val=` scelgono anche l'epoca; default: solo misura), `maxgames`.

## Scelte da conoscere

* **Tablut e' markoviano** (a parte la ripetizione): la storia non aggiunge informazione sul valore della posizione. Puo' solo aiutare una rete piccola. I bersagli multi-mossa invece portano informazione vera (cio' che le ricerche delle posizioni successive hanno visto).
* **History dropout**: la finestra e' troncata a caso ai soli ultimi `j` record. E' il caso reale (MCTS e valutazione stateless hanno finestra vuota; nei foglie vicini alla radice mancano le mosse prima della radice) e riduce il rischio che la rete usi "questa mossa e' stata appena giocata" come prova di qualita': in auto-gioco le mosse in storia sono scelte da una ricerca, nei foglie sono arbitrarie.
* **Costo in gioco**: la parte temporale NON si aggiorna per differenza (la finestra scorre e cambia tutte le feature): a ogni valutazione si sommano le righe di al piu' 3K feature a una copia dell'accumulatore. Misurato: ~1,17 M nodi/s contro ~1,57 M della NNUE di base (profondita' 5, 1 thread, alberi diversi: indicativo).
* La transposition table non conosce la storia: due arrivi alla stessa posizione con storie diverse condividono la voce (come per ogni evaluator dipendente dal percorso).
* Dopo una mossa nulla la finestra si interrompe (mai una finestra sbagliata, al massimo piu' corta).

## Verifiche

`aiplayer.TemporalNnueCheck` (26-35k verifiche: schema, derivazione della mossa da due scacchiere su partite vere, rete/file con file corrotti, sessione incrementale == valutazione completa con storia prima della radice, catture, mosse del Re, fratelli e mosse nulle, tracker, e Negamax con `tW=0` identico alla NNUE di base per mossa/punteggio/nodi/profondita'), `training.TemporalTrainerCheck` (~20k: storie dal dataset, TD(lambda) a mano, gradiente contro differenze finite, conversioni float/quantizzato, esperimento controllato in cui l'etichetta dipende solo dalla storia, non regressione, determinismo, CLI), `client.TemporalClientCheck`.
Regressione sui test NNUE esistenti (`NnueCheck`, `NnueSearchCheck`, `NoAllocationCheck`, `NnueClientCheck`): verdi. `NegamaxClientArgsCheck` (1 errore) e `ClientDefaultsCheck` (4) falliscono IDENTICI sul codice originale (default di `exactkey`).
Test rapido di mutazione: 4 guasti introdotti di proposito (righe non offsettate, simmetria ignorata, finestra sfasata, mu invertito) sono stati tutti rilevati.

## Risultati misurati (sandbox 1 core, NESSUNA arena di gioco)

Partenza `pesi_nnue/trained_v64.nnue`; addestramento sulle prime 30 000 partite di `dati_nnue_d2.gz` (etichette di ricerca a profondita' 2, ~895k posizioni), 8 epoche, K=4, `lambda=0.5`; held-out per partita (5%) e metro ESTERNO = valore di ricerca a profondita' 4 su 1,02 M posizioni di `dati_nnue_d4.gz` (partite diverse). Errore quadratico medio, piu' basso e' meglio; "acc" = segno dell'esito vero predetto, su partite d4.

| rete | held-out d2: MSE valore | esterno d4: MSE valore | acc esito d4 |
|---|---|---|---|
| partenza (v64) | 0.0743 | 0.0793 | 77.7% |
| solo posizione, esito (`tdlambda=1`, mix 0.5) | 0.0564 | 0.0664 | 77.5% |
| **temporale**, stesso | 0.0566 | 0.0692 | 77.4% |
| solo posizione, solo valore (`lambda=1`) | 0.0367 / 0.0366 (2 semi) | 0.0554 / 0.0552 | 76.4% / 76.5% |
| **temporale**, stesso | 0.0356 / 0.0355 | 0.0570 / 0.0566 | 76.4% / 76.5% |
| solo posizione, TD(0.5) | 0.0432 / 0.0429 | 0.0534 / 0.0540 | 76.8% / 76.9% |
| **temporale**, TD(0.5) | 0.0431 / 0.0419 | 0.0538 / 0.0545 | 76.8% / 76.9% |
| solo posizione, TD a un passo | 0.0450 | 0.0553 | 76.8% |
| **temporale**, TD a un passo | 0.0449 | 0.0568 | 76.8% |

Lettura:
* **Storia in ingresso: nessun beneficio dimostrato.** In distribuzione migliora di poco (2-3% sul valore d2 con `lambda=1` e TD(0.5)), sul metro piu' profondo e' pari o PEGGIO (fino a +3%), e l'accuratezza sull'esito non cambia. Il rumore fra semi e' ~0.0002-0.0006.
* **Bersagli TD: effetto piccolo ma coerente.** TD(0.5) batte il solo valore sul metro d4 (0.0534-0.0540 contro 0.0552-0.0554) e recupera ~0.4 punti di accuratezza sull'esito; resta sotto il mix con l'esito (77.5%) in accuratezza, ma e' molto meglio in accordo con la ricerca profonda (0.0664). Un solo valore di mu (0.5) e un solo dataset: non e' uno sweep.
* **Non e' stata misurata la forza di gioco.** Le metriche sopra non la predicono (il lavoro precedente nel repository lo mostra: ottimo accordo col maestro e pari forza in arena). Per decidere servono partite a tempo reale su piu' core (`EqualTimeArena`, `PruningArena`) contro la NNUE di base.
