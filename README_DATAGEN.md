# Generare dataset a profondita' alta in meno tempo

Tutte le misure: sandbox a 1 core, rete `pesi_nnue/trained_v64.nnue` (64/16; la 128/64 dei vostri log e' piu' lenta per nodo), 1 thread, TT nuova per posizione.
Su una macchina diversa cambiano i secondi, non i rapporti.

## Dove andava il tempo
`NnueSelfPlay` cercava ogni posizione con la ricerca storica (**senza null-move/LMR/aspirazione**) e una TT di 2^18 voci, mentre in gara si gioca con `prune=all` e TT 2^26.
`NnueRelabel` era a **un solo thread** e senza potatura.

## Cosa ho aggiunto (tutto spento di default: con le opzioni spente il dataset e' identico byte per byte al codice precedente, verificato)
`NnueSelfPlay`: `prune=`, `tt=`, `movedepth=`, `labelevery=`, `skipsat=`, `adj=X:N`. `NnueRelabel`: `threads=`, `prune=`, `tt=`, `chunk=`, `every=`, `keep=`, `skipsat=` (versione parallela, il risultato dipende da `chunk` e NON dal numero di thread).
Test: `training.FastDataGenCheck` (44 verifiche) piu' i test esistenti di self-play e dati, tutti verdi.

## Misure
| prova | risultato |
|---|---|
| `prune=all`, profondita' 8, 4 posizioni | **8,25 s contro 37,6 s per posizione (4,5x)**, 55 contro 319 Mnodi; stessa mossa 3/4, RMS del valore 0,06 |
| `prune=all`, profondita' 7, 10 posizioni | 0,92-1,0 s contro 4,8 s (4,8x); stessa mossa 8/10, RMS 0,03 |
| auto-gioco completo a profondita' 6, 6 partite | 0,32 s per posizione registrata (base) contro 0,17 s (`prune=all tt=20`): 1,9x. Il vantaggio cresce con la profondita' |
| `movedepth=4 labelevery=3` (etichetta a 6) | 0,28 s per etichetta: **non** e' piu' veloce per etichetta di `prune=all` (0,17 s), perche' ogni etichetta e' una ricerca in piu'. Serve per avere piu' partite diverse, non per risparmiare |
| `adj=0.97:8` | a questa scala poco effetto (le partite finiscono in 40-50 semi-mosse); utile solo se le partite si bloccano |
| costo di una etichetta a profondita' 8 con `prune=all` | circa 4,5-8 s per core (10 + 4 posizioni diverse) |

## Cose che NON conviene fare (misurate)
* **Deduplicare**: solo lo 0,3% delle posizioni di `dati_nnue_d2/d4` e' duplicato.
* **Scegliere solo le posizioni "difficili"** (dove la rete statica e l'etichetta piu' economica divergono di piu'): a parita' di 100 000 etichette, ripartendo da v64 e valutando su 341 000 posizioni di partite mai viste, l'errore sul valore di test e' 0,0486 con un campione casuale e **0,0773 (invariato, nessuna epoca migliora) con le piu' difficili**. Un campione solo difficile e' troppo sbilanciato.
* **Scartare le posizioni sature dal dataset** (`skipsat` in auto-gioco): a parita' di dimensione, l'errore di test sale da 0,0491 a 0,0507 (le posizioni sature sono il 22% e la rete deve riconoscerle). `skipsat` ha senso solo in `NnueRelabel`, per non spendere una ricerca profonda sulle posizioni che gia' hanno etichetta satura (restano con la vecchia).
* Etichette a profondita' bassa al posto della 8: l'RMS rispetto alla profondita' 8 e' 0,23 (d4), 0,29 (d5), 0,16 (d6), 0,27 (d7) su 10 posizioni: le etichette poco profonde non sono una buona approssimazione (e le profondita' dispari si comportano diversamente dalle pari: restate su profondita' pari).

## Dimensionamento
Per mettere a punto una rete gia' addestrata bastano poche etichette: 100 000 posizioni casuali portano v64 da 0,0773 a 0,0486 su partite mai viste (5 s di addestramento).
Costo: `etichette x 5-8 s / core`. 100 000 etichette a profondita' 8 sono circa 140-220 ore-core (14-22 ore su 10 core) con `prune=all`; senza, 4,5 volte di piu'.
La scalabilita' sui core NON l'ho misurata (sandbox a 1 core): le partite e i blocchi di rietichettatura sono indipendenti, quindi il tempo dovrebbe scendere quasi linearmente fino al numero di core fisici, con `threads` = core fisici (non usate Lazy SMP per generare: perde efficienza).

## Ricette
```sh
P=it.unibo.ai.didattica.competition.tablut.training
# 1) partite direttamente a profondita' 8, ~4,5x piu' veloci di prima (stessa ricerca con cui si gioca)
java -Xmx12g -cp "bin:lib/*" $P.NnueSelfPlay eval=pesi_nnue/trained_v7.nnue out=d8.gz games=2000 depth=8 threads=8 prune=all tt=22 seed=1 adj=0.99:12
# 2) riusare partite gia' fatte (es. dati_nnue_d4.gz) e dare l'etichetta di profondita' 8 solo a un campione, in parallelo
java -Xmx12g -cp "bin:lib/*" $P.NnueRelabel in=dati_nnue_d4.gz out=d4_rel8.gz eval=pesi_nnue/trained_v7.nnue depth=8 threads=8 prune=all tt=22 every=30 keep=false skipsat=0.97
```
(`tt=22` e' ~100 MB per partita in corso/blocco: con 8 thread bastano 1-2 GB in piu'.)
Misura utile prima di lanciare tutto: `experiments.DataGenSpeed` (secondi per posizione, con e senza potatura, e quanto si discostano i valori) e `experiments.DataGenDepths` (quanto cambia l'etichetta tra profondita').
