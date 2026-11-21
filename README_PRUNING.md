# Riduzione dell'albero in `Negamax`: mossa nulla, LMR, finestre di aspirazione

Tre tecniche per raggiungere **profondita' maggiore a pari tempo**, aggiunte come **opt-in**: nel default sono tutte **spente** e la ricerca e' quella storica,
nodo per nodo. Si accendono con `prune=` (client) o `Negamax.setPruning(PruningOptions)`.

> **Stato dei risultati.** Le tecniche funzionano e sono verificate (sezione 5). Il guadagno misurato e' **moderato** (+0,5 / +0,7 semi-mosse a pari tempo) perche' le
> salvaguardie sono volutamente prudenti. **Non ho una prova di maggiore forza di gioco**: 28 coppie di partite a 200 ms/mossa hanno dato 51,8 % [42,7 – 60,8],
> cioe' nessuna differenza rilevabile (ne' in meglio ne' in peggio). Per questo **non le ho accese di default**.

## 1. Cosa e' stato verificato nel codice esistente (prima di toccare qualcosa)

`aiplayer/Negamax` (letto per intero) ha un contratto che queste tecniche mettono a rischio:

* i punteggi di vittoria valgono `±(WIN_SCORE + d)` con `d` = profondita' residua del nodo terminale e **non si leggono mai dalla TT**; la ricerca si ferma alla prima
  profondita' in cui compare una vittoria, e questo da' la vittoria piu' breve: il Javadoc dice esplicitamente "*senza pruning non sicuri*";
* GHI: un nodo si memorizza in TT solo se il suo sottoalbero non ha incontrato ripetizioni (`repetitionEventCounter`);
* **nessuna quiescenza**: a `depth == 0` si valuta staticamente, quindi un figlio ridotto a profondita' 0 non vede le vittorie immediate dell'avversario;
* la radice non taglia mai su `beta`; un'iterazione interrotta dal tempo viene scartata per intero; ogni worker Lazy SMP ha stato privato e condivide solo TT ed evaluator;
* le cache di valutazione si indicizzano con la chiave esatta o con `zobristHash`, entrambe comprensive del turno; `boardSignature` della sessione NNUE **toglie** il bit del turno.

Nessun test esistente e' stato modificato. Con le tecniche spente 17 classi di test esistenti sulla ricerca sono verdi, incluso `SearchEquivalenceCheck`, che confronta i nodi
con `LegacyNegamax` (584 ricerche, 5118 verifiche).

## 2. Uso

```sh
# client (TablutNegamaxClient e TablutGeneticClient, che inoltra le opzioni): spente di default
java ... TablutNegamaxClient WHITE 60 localhost pesi_nnue/trained_v6.nnue eval=nnue prune=all
java ... TablutNegamaxClient WHITE 60 localhost pesi_nnue/trained_v6.nnue eval=nnue prune=nmp,lmr,asp,asp.delta=60
```
`prune=` accetta `off`, `all`, `nmp`, `lmr`, `asp` e parametri `chiave=valore`: `nmp.r` (2), `nmp.depth` (4), `nmp.pieces` (5), `nmp.verify` (10; 0 = mai),
`lmr.depth` (3), `lmr.moves` (4), `asp.depth` (5), `asp.delta` (40). Valori incoerenti (p.es. `nmp.r=3` col `nmp.depth` di default) sono **rifiutati**; nel client un valore non valido
produce un `ATTENZIONE` e la ricerca resta quella storica. Il banner mostra `potatura: ...` solo se accesa. Vale solo per il negamax (con `algo=mcts` e' ignorata, con avviso).

## 3. Come funziona e perche' e' sicura (o quanto)

**Aspirazione (alla radice).** Da `asp.depth`, finestra `precedente ± delta`, con `delta = max(asp.delta, 0,75 × |variazione fra le due iterazioni precedenti|)` (il punteggio NNUE oscilla:
mediana ≈ 36, p90 ≈ 185–240 fra iterazioni successive, su scala ±1000); fail-low/high **ripetono** con finestra piu' larga (raddoppio; al terzo insuccesso, o se compare una vittoria, il lato si apre del tutto).
Un risultato strettamente dentro la finestra e' **esatto**: stesso valore e stessa mossa della finestra piena (verificato, anche con TT spenta e finestra minima). Il ciclo storico e' stato
isolato in `rootPass` con finestra piena = identico a prima. Alla radice si interrompe al primo `score >= beta` (altrimenti alpha ≥ beta). Un'iterazione interrotta resta scartata.

**Mossa nulla.** Solo in un nodo con: finestra nulla (non-PV) e **senza punteggi di vittoria**, `depth >= nmp.depth` (il figlio nullo ha profondita' ≥ 1: vede le vittorie immediate avversarie),
≥ 3 mosse legali (il caso "nessuna mossa = sconfitta" e' gia' gestito), ≥ `nmp.pieces` pezzi del lato al tratto (zugzwang), posizione **non forzante** (`forcing`: Nero al tratto con il Re
con via libera verso il bordo; Bianco al tratto con una pedina nera accanto al Re; sovrastima apposta), valutazione statica ≥ beta, e mai due mosse nulle di fila. Poi:
* `FastState.makeNullMove/unmakeNullMove` (nuovi, additivi) invertono turno, `zobristHash` e `key2` con la stessa regola di `makeMove`: il figlio e' la posizione "stessa scacchiera, altro turno" **per ogni cache**;
* `IncrementalEvaluator.Session.pushNull` (metodo di default `false`: nessun implementatore esistente cambia); `NnueSession` lo implementa con un frame "copia del genitore" lazy;
* **il taglio restituisce `beta` e NON scrive nulla in TT**; un risultato di vittoria/sconfitta dalla ricerca nulla non taglia mai (la mossa nulla non e' una mossa); da `nmp.verify` il taglio e' **verificato**
  da una ricerca a `depth-R` senza mosse nulle;
* la storia di ripetizione **non viene toccata**: il figlio nullo non si registra ne' si confronta (non e' una posizione di partita); le ripetizioni nel sottoalbero nullo bloccano la TT dei genitori
  solo se il taglio avviene (se non taglia, il contatore GHI e' ripristinato: il sottoalbero nullo non fa parte del valore del nodo).

**LMR.** Candidata ogni mossa non-prima con `depth >= lmr.depth`, indice ≥ `lmr.moves`, finestra senza vittorie, punteggio di ordinamento non "speciale" (TT, vittoria, killer, countermove, catture),
non cattura/vittoria, non mossa del Re, nodo non forzante. Riduzione 1 (2 da profondita' 6 e indice ≥ 2×`lmr.moves`), sempre con figlio a profondita' ≥ 1. Il figlio ridotto si **ripete a profondita'
piena** se sembra migliorare **o se il risultato e' una vittoria/sconfitta** (la sua profondita' residua ridotta non e' coerente): nessun punteggio di vittoria entra in un nodo da una ricerca ridotta.
Se il figlio ridotto risulta **forzante** (la mossa tranquilla ha creato una minaccia, p.es. ha aperto una linea al Re) la riduzione viene **annullata** (`pendingReduction`).
Le voci di TT dei figli ridotti hanno la profondita' con cui sono state cercate: una ricerca piu' profonda non le usa come fatti di profondita' piena.

**Sospensione dopo una vittoria.** Se una ricerca trova una vittoria o sconfitta **forzata**, la successiva e' **interamente senza tecniche** (l'algoritmo storico, che ritrova la vittoria piu' breve):
cosi' resta valida la garanzia "la distanza dalla vittoria scende ad ogni turno" (testata: ogni partita di convergenza finisce con la vittoria entro i semi-turni annunciati).

**Lazy SMP.** Gli helper ricevono la configurazione sia da `setPruning` (anche se gia' creati) sia a ogni ricerca; nessun nuovo thread, nessun calcolo dopo la mossa.

## 4. Difetti trovati lungo il percorso (e corretti)

* L'LMR nella mia prima versione era quasi inerte (−1,3 % dei nodi): strumentando il motore ho visto che lo applicavo solo al ramo "scout" dei nodi PV; i nodi non-PV (finestra gia' nulla), dove sta il grosso, no.
* Con tutto acceso comparve una **sconfitta forzata all'orizzonte non vista** (pos. #13 del campione di prova). Causa: riduzione di mosse tranquille che *creano* minacce. Rimedio: annullare la riduzione se il figlio e'
  forzante. Effetto: mosse uguali a quelle della ricerca piena 18/24 -> 23/24 e **0 discordanze di vittoria/sconfitta**.
* I contatori dell'ultima ricerca restavano quelli della precedente quando la ricerca usciva dalla scorciatoia "vittoria immediata" (trovato da un test): corretto.
* Un mio test usava `lmr.moves=64` credendolo disattivante: un nodo puo' avere piu' di 64 mosse. Un altro usava finestre di vittoria irrealistiche (`WIN-1`): corretti i test, non il motore.

## 5. Cosa ho misurato

Tutte le misure su `trained_v6.nnue`, **1 core**, posizioni **realistiche** da auto-gioco della rete (non partite casuali). Le cifre sono quelle ottenute, su campioni piccoli.

| Misura | Risultato |
|---|---|
| Nodi a profondita' 6, 24 posizioni di meta' partita (piena = 100 %) | aspirazione 64,8 %, mossa nulla 73,2 %, LMR 49,7 %, tutto 38,4 % |
| Fedelta' a profondita' 6 (stessa mossa della piena) | aspirazione 24/24 (**esatta**), LMR 23/24, tutto 23/24; 0 discordanze di vittoria/sconfitta |
| Suite di finale (ultimi 12 semi-turni di partite decise), profondita' 5, 168 posizioni, 61 forzate | tutte le configurazioni: forzate viste 61/61, perse 0, segno opposto 0, affermazioni non confermate 0; stessa mossa 168 (asp) / 166 (nmp) / 159 (lmr) / 155 (tutto) |
| Idem profondita' 6, 84 posizioni, 43 forzate | viste 43/43 per nmp, lmr e tutto; stessa mossa 84 / 83 / 82 |
| Distanza dalla vittoria (suite del test, 51 forzate, prof. 5) | punteggio **identico** a quello della ricerca piena in 51/51 per ogni configurazione (la vittoria piu' breve e' preservata) |
| Profondita' a pari tempo, 14 posizioni, 2 s/mossa, 1 thread | piena 6,57 -> tutto 7,07 (asp 6,57; nmp 6,79; lmr 6,79) |
| Idem 6 posizioni, 10 s/mossa | piena 7,00 -> tutto 7,67 |
| Partite `all` vs `off`, 200 ms/mossa, stesse risorse, coppie a colori invertiti | 28 coppie: **51,8 % [42,7 – 60,8]**, `INSUFFICIENTE`: nessuna differenza rilevabile |
| Variazione dei parametri (`lmr.moves` 2/3, `nmp.r=3`, `nmp.pieces=3`) | nodi 35-39 %: cambia poco; **ho tenuto i default prudenti** |
| Allocazioni nella ricerca | 0 byte con ricerca storica e con tutto acceso |

**Letture oneste.** (1) A pari profondita' nominale le tecniche cambiano la mossa in qualche caso (e' nella loro natura: sono euristiche); la domanda giusta e' la forza a pari *tempo*, che non ho potuto
misurare con potenza sufficiente. (2) Il guadagno di profondita' cresce con il tempo ma resta ben sotto i +2-3 semi-mosse tipici dei motori da scacchi: l'ho pagato scegliendo la sicurezza. (3) Nei finali le
salvaguardie (pochi pezzi, posizioni forzanti) spengono quasi tutto: nodi 71-91 %. (4) Le suite tattiche sono su finali di auto-gioco della stessa rete: un campione, non una prova.

## 6. Test (`aiplayer/PruningCheck`, ~100 s su 1 core, 1928+ verifiche) e loro qualita'

`sh tools/run_pruning_checks.sh` (opzioni: `RUN_EXISTING=1`, `SECTIONS=sound,tt,...`). Sezioni: `options`, `primitives` (mossa nulla contro un **oracolo indipendente**: la stessa posizione col turno invertito costruita dal `State` di dominio;
sessione NNUE in passeggiate casuali con mosse vere e nulle, = valutazione completa, 0 ricostruzioni), `session`, `equiv` (parametrizzazioni che disattivano di fatto una tecnica = ricerca storica, **nodi inclusi**; aspirazione esatta,
con prova che fail-low e fail-high avvengono), `sound` (finali: nessuna vittoria/sconfitta falsa, nessuna forzata persa, **distanza preservata**, LMR che riduce e ripete), `conv` (convergenza e sospensione), `integrity` (ricerca incrementale =
completa con potatura; cache di valutazione trasparenti), `tt` (TT "registratore": il taglio nullo **non scrive** il nodo; la storia non cambia; il figlio nullo non e' una posizione di storia), `forcing`, `guards` (finestre di
vittoria/sconfitta: nessuna riduzione ne' mossa nulla), `alloc`, `smp` (tempo con 4 thread, propagazione agli helper, quiete: 0 ms di CPU dopo la mossa), `eff`.
Nel client: `TablutGeneticClientCheck` (ora 154 verifiche) prova `prune=` col protocollo reale (valida e non valida).

**Test per mutazione** (22 varianti difettose introdotte a mano una alla volta, ognuna con le sezioni pertinenti): **16 uccise**, **3 equivalenti** (non osservabili per costruzione: ripristino di `key1`, che la mossa nulla non
modifica; riduzione della prima mossa, gia' esclusa da `lmr.moves >= 2`; riassegnazione ridondante della configurazione agli helper), **3 sopravvissute**, che sono 2 comportamenti: (a) togliere la ripetizione a profondita' piena dei
*punteggi di vittoria* provenienti da ricerche ridotte (varianti L5 e L2); (b) togliere la ri-ricerca per miglioramento nel solo ramo scout dei nodi PV (L3), comportamento poi coperto dalla variante che la toglie ovunque (L6, uccisa).
Il caso (a) non ha effetto osservabile sulle mie fixture: e' una protezione teorica di coerenza della distanza che ho mantenuto. Tra le uccise: rimuovere l'annullamento della riduzione sul figlio forzante fa **perdere una vittoria
forzata**, quindi quella salvaguardia e' necessaria. Limite: sono i difetti che ho saputo immaginare.

## 7. Limiti e informazioni mancanti

* **Forza di gioco non dimostrata**; `PruningArena` e' lo strumento: `PruningArena net=pesi_nnue/trained_v6.nnue a=all b=off pairs=40 ms=2000 seed=1 out=ris.txt` (poi altri semi, e `combine=ris.txt`).
  Per rilevare +5 % servono ~280 coppie (sd delle coppie ≈ 0,25-0,3 misurata qui da campioni piccoli). `PruningArena` non e' `SearchArena` (preesistente, confronta algoritmi diversi).
* Il rischio residuo e' intrinseco: sono euristiche; mosse buone possono essere scartate. Le salvaguardie lo riducono, non lo annullano.
* Hardware del torneo ignoto: `Negamax.defaultThreads()` vale `min(DEFAULT_THREADS, core)` e si forza con `-Dtablut.threads=N`; lancia riferimento e candidato con le stesse opzioni.
* Con valutatori non NNUE incrementali la mossa nulla funziona solo se `pushNull` e' implementato (altrimenti non viene tentata); con valutatori non incrementali usa la valutazione completa.
* Il GA (`README_GENETIC.md`) e' indipendente: puo' valutare candidati con o senza `prune=` passandolo al client; i suoi cancelli usano la ricerca storica.

## 8. File

```
Nuovi:      aiplayer/PruningOptions.java, aiplayer/PruningCheck.java, training/PruningArena.java, tools/run_pruning_checks.sh, README_PRUNING.md
Modificati (additivi, comportamento di default invariato):
            aiplayer/Negamax.java            setPruning, rootPass, null-move, LMR, forcing, sospensione, ganci di test package-private
            aiplayer/IncrementalEvaluator.java  Session.pushNull (metodo di default)
            aiplayer/NnueSession.java        pushNull / frame nullo
            engine/FastState.java            makeNullMove / unmakeNullMove
            client/TablutNegamaxClient.java  opzione prune= (+ overload di buildSearcher), banner solo se accesa
            client/TablutGeneticClientCheck.java  prove di prune= col protocollo reale
```
