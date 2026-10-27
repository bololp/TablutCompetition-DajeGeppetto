# README_ORDERING: ordinamento delle mosse, TT più grande, margine di tempo

Tutto è stato misurato in un sandbox a **1 core** (JDK 21), con la CNN `cnn_conv4_hybrid_pretrained.txt` (solo supervisionata) e con
`SimpleEvaluator`. Dove una cosa è un ragionamento e non una misura, c'è scritto. **Nessuna arena è stata giocata**: a 1 core e 100 ms l'effetto
di queste modifiche (circa 0,1 ply) è sotto la risoluzione di una arena (IC95 ±12 punti con 60 partite). Il verdetto va preso sulla macchina a 4 core
(si veda §6).

## 1. In breve

* **Ordinamento.** Sei opzioni, **tutte spente di default**. Misurate in nodi a profondità fissa (deterministico), dopo il default storico
  (TT + killer + history): vittorie immediate e catture per prime valgono circa il **5–10 %** dei nodi con la CNN, la combinazione consigliata
  `rec` = `wins+caps+rootsc` circa **−11 %** (CNN, profondità 4) e **−23 %** (Simple, profondità 5). Countermove e killer che passano da una mossa
  all'altra **non servono**. Quindi il vantaggio dell'altro motore non viene dall'ordinamento con catture per prime.
* **TT.** `store()` sostituisce SEMPRE su chiave diversa, anche una voce profonda. Una TT satura costa al più ~8 % di nodi (misurato in scala, §3);
  una variante a due slot con aging ne recupera circa metà. Non ho provato 2^24/2^25 su una VM vera.
* **Margine.** Lo sforo misurato di `findBestMove` è **al massimo 35 ms a 1 thread e 75 ms a 4 thread** (su 1 core); i 300 ms storici lo coprono.
  Il client ora usa 1,5 s con i tempi di gara come assicurazione contro ciò che non ho misurato (rete, JSON, GC, VM più lenta).
* **Default invariati** dove non c'è una misura in arena: `Negamax` e il client usano ancora l'ordinamento e la TT di prima
  (`SearchEquivalenceCheck`: 4.849.746 nodi = 4.849.746 del motore storico). Cambia solo il margine del client (§4).

## 2. Ordinamento delle mosse

### 2.1 Cosa c'è
`FastState.tacticalValue(move)` classifica una mossa legale **senza eseguirla**: `TACT_WIN` (fuga del Re o cattura del Re) oppure il numero di pedine
catturate. È esatta (si veda §5). `Negamax` usa bit nuovi di `orderingMode`:

| Opzione | Effetto |
|---|---|
| `ORDER_WINS` | le mosse che vincono subito subito dopo la mossa TT |
| `ORDER_CAPTURES` | le catture (per numero di catture, poi history) dopo killer e countermove, prima delle quiete |
| `ORDER_CAPTURES_FIRST` | come sopra ma prima delle killer (una cattura che è anche killer resta una cattura) |
| `ORDER_COUNTER` | countermove: la risposta che ha tagliato dopo la stessa mossa avversaria |
| `ORDER_KILLER_CARRY` | le killer slittano di 2 ply tra una mossa e la successiva invece di essere azzerate |
| `ORDER_ROOT_SCORES` | alla radice, ordine per punteggio dell'iterazione precedente (**cambia il tie-break** tra mosse di pari valore) |

`ORDER_RECOMMENDED = ORDER_DEFAULT | WINS | CAPTURES | ROOT_SCORES`. Nomi leggibili in `OrderingSpec` (`wins+caps`, `rec`, `all`...).

### 2.2 Misure (nodi rispetto al default; `experiments.OrderingBench`, 1 thread, TT 2^20, una istanza riusata per lato come in partita)

| Opzione | CNN d4 (142 ricerche) | Simple d5 (153 ricerche) |
|---|---|---|
| wins | −5,1 % | −19,3 % |
| caps / capsfirst | −4,5 % / −4,1 % | −9,4 % / −12,7 % |
| counter | 0,0 % | **+2,8 %** |
| kcarry | −0,2 % | +0,1 % |
| rootsc | −5,4 % | −6,4 % |
| wins+caps | −6,7 % | non misurata |
| **wins+caps+rootsc (`rec`)** | **−11,0 %** (tempo −13 %) | **−23,1 %** (142 ricerche, tempo −1,5 %) |
| all | −10,8 % | −22,8 % |

Altre misure: CNN d5 su sole **28 ricerche** (campione piccolo, rumoroso): rootsc +2,3 %, all −21,8 %, allfirst −4,4 %. Con `Simple` il costo del calcolo
tattico per mossa (~20 % di nodi/s) mangia il guadagno in tempo; con la CNN (11 µs per foglia) no. **Il tempo nel sandbox oscilla di ±6 %** tra due esecuzioni
identiche: per questo il confronto è sui nodi.

### 2.3 Valori che differiscono: non è l'ordinamento
Nel banco qualche ricerca dà un valore di radice diverso dal default (2 su 142). Con una istanza NUOVA per ricerca restano 2: la mia prima ipotesi
("TT persistente") era sbagliata. `experiments.ValueDiffProbe` mostra che, in quelle posizioni, il valore ESATTO (TT spenta) è identico per tutti gli
ordinamenti, mentre il valore CON TT devia dall'esatto, in una posizione per `base` (37,1 contro 37,6) e nell'altra per `rec` (−3,1 contro 0,0). Quindi
è un comportamento della TT **già presente nel motore attuale** (~1–2 % delle posizioni a profondità 5). Il meccanismo (voci più profonde riusate a
profondità residua minore) è l'ipotesi standard ma **non l'ho dimostrato**.

## 3. TT

* `store()` (slot singolo): sostituisce se la voce è vuota, se la chiave è diversa (sempre) o se la profondità è ≥. Verificato leggendo il codice.
* Nuova opzione (spenta): **due slot per bucket**, a parità di memoria. Slot 0 = depth-preferred con aging (la voce di una ricerca precedente è sostituibile),
  slot 1 = sempre-sostituisci. `newSearch()` (chiamata da `findBestMove`) avanza la generazione (6 bit nel campo meta). Lock-free come prima (trucco XOR).
  Si accende con `new Negamax(..., twoSlotTT)`, `-Dtablut.tt2=true`, `tt2=true` nel client, `tt2:` nell'arena.
* **Misura in scala** (`experiments.TTBench`, CNN, profondità 5, 28 ricerche, 1 thread, istanza riusata; nodi rispetto alla TT 2^22):

| TT | riempita al | nodi |
|---|---|---|
| 2^22 slot singolo | 1,7 % | 4.739.048 (riferimento) |
| 2^16 slot singolo | 64 % | +1,1 % |
| 2^16 due slot | 46 % | −0,1 % |
| 2^14 slot singolo | 99 % | **+8,4 %** |
| 2^14 due slot | 73 % | +4,8 % |

* **Cosa NON è provato.** (a) Che a 60 s con 4 thread la 2^22 sia satura: è aritmetica (circa 32 M nodi contro 4,2 M slot), non una misura. (b) Che la scala
  "2^14 a 0,2 s ≈ 2^22 a 60 s" regga: la profondità, e quindi la distribuzione delle profondità nelle voci, è molto diversa. (c) 2^24/2^25 su una VM vera, con
  4 thread: tempo di allocazione e azzeramento, pause GC, heap. Una TT 2^25 è UN array di 805 MB (default JVM: heap = 1/4 della RAM = ~2 GB su 8 GB: ci sta,
  ma non l'ho provato). Il client riduce da solo la TT se non sta in metà dell'heap (e avvisa).

## 4. Margine di tempo

* **Dal codice del server** (`Server.java`): il thread di lettura parte PRIMA che il client riceva lo stato, e il controllo è a gradini di `sleep(1000)`; se
  al controllo finale il lettore è ancora vivo la partita è persa. Non c'è tolleranza oltre i 60 s.
* **Sforo misurato** (`experiments.OvershootBench`, CNN, margine 0, budget 1500 ms, TT 2^22, ordinamento `rec`, 26 ricerche utili): 1 thread: mediana 7 ms,
  p90 22, **max 35**; 4 thread su 1 core (alternati a fette di tempo, caso sfavorevole): mediana 33, p90 58, **max 75**. Dal vivo (Server -t 5, margine 600,
  2 thread): 4426 ms usati su 4400 di budget.
* `Negamax.setSafetyMarginMillis(ms)` (default 300, invariato). Il **client** ora usa **1500 ms con timeout ≥ 20 s**, 300 ms sotto (`margin=MS` per forzarlo).
  Costo: circa 0,02–0,03 ply su 60 s (calcolo con fattore di ramificazione effettivo ~2,5, non misura). **Non misurato su una VM vera**: GC con TT grande, jitter, latenza.

## 5. Test

| Test | Cosa verifica | Esito |
|---|---|---|
| `FastStateCheck` (esteso) | `tacticalValue` contro un oracolo ricavato dal motore storico (vittoria immediata, pedine perse): 1.772.414 mosse; copre 14.763 vittorie (358 fughe del Re che catturano anche), 87.015 mosse a 1 cattura, 1.505 a 2+ | 0 discordanze |
| `OrderingExtCheck [seme] [posizioni] [parti]` | punteggi estesi contro un oracolo a livelli (15 combinazioni × 2 basi, stato sporco, con posizioni **costruite a mano**: fuga e cattura del Re); ordine radice stabile; slittamento delle killer; mossa precedente registrata nel ply giusto (si guarda la scacchiera di ogni foglia); countermove ben formata e scritta dove deve; dispatch; `newSearch`; margine; valore identico all'alpha-beta puro per 9 configurazioni (TT spenta, TT accesa, storia di ripetizione, istanza riusata, tutti i figli visitati a profondità 1, 4 thread) | OK con semi 2024 e 7 |
| `TwoSlotTTCheck [seme]` | TT a due slot contro un modello di riferimento indipendente (3 configurazioni, 150.000 probe ciascuna), protezione delle voci profonde, aging, voci strappate in ogni slot, 4 thread × 400.000 operazioni, `occupancy` | OK con semi 31337, 7, 99, 4242 |
| `ArenaPrefixCheck`, `NegamaxClientArgsCheck` | prefissi dell'arena e nomi di `OrderingSpec`; margine di default e adattamento della TT all'heap | OK |
| Suite esistente rilanciata | FastState, CrossCheck, WasCapture, Zobrist ×2, TTCorrectness, RepetitionAndStalemate, SearchEquivalence (nodi identici), PVS, LazySmp, SearchOptimizations (seme 7), NoAllocation (anche con `rec`, `all`, `tt2`: 48 B totali; 464 B, cioè 0,04 B/nodo, solo con `all`+`tt2` e la CNN), CnnEvaluator, ConvNet | OK |
| Con `-Dtablut.tt2=true` | TTCorrectness (0 mismatch), RepetitionAndStalemate, OrderingExtCheck (tt/rep/reuse/smp), LazySmpCheck | OK. `SearchEquivalenceCheck` fallisce per costruzione (confronta nodi con `LegacyNegamax`, che ha la TT a slot singolo); nei casi stampati (25 su 171) solo nodi, ripetizioni e statistiche PVS, nessun punteggio o mossa |

**Mutation test** (`tools/mutation_engine.py order|tactical|tt2`): **24/24** (ordinamento esteso), **12/12** (`tacticalValue`), **15/15** (TT due slot). Il motore ora
**controlla che il test passi sul codice non mutato** prima di mutare (si veda sotto) e supporta l'*n*-esima occorrenza di un frammento.

**Due errori miei, scoperti e corretti:**
1. La prima volta il test dei punteggi falliva già sul codice NON mutato con 6 posizioni (il campione casuale non conteneva vittorie, e il test avverte "vacuo"):
   i primi 12 mutanti risultavano "rilevati" per quel motivo. Ho aggiunto il controllo di baseline al motore e due posizioni costruite a mano al test; ho rilanciato tutto.
2. Avevo inserito `tacticalValue` a metà di `FastState`: sei mutanti **esistenti** (`state` 0, 1, 2, 3, 6, 7) avrebbero mutato `tacticalValue` invece di
   `makeMove`. Ora `tacticalValue` è in fondo alla classe; verificato che i sei colpiscono di nuovo `makeMove` e sono rilevati.

**Mutanti non provati** (li ho ragionati, non eseguiti): togliere `Math.min(h, SC_HISTORY_CAP)` (equivalente nel dominio del test: history < 2^26); `if (!rootByScores)` → `if (true)`
(equivalente osservabile: a fine ricerca completa la radice è comunque ordinata). Le suite `state` e `search` esistenti **non sono state rilanciate per intero**
(solo i 6 mutanti `state` a rischio).

## 6. Da fare sulla tua macchina a 4 core

```bash
CP="build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; T=it.unibo.ai.didattica.competition.tablut
W=pretrained_weights/cnn_conv4_hybrid_pretrained.txt     # oppure la tua CNN migliore

# nodi/tempo a profondità fissa (deterministico nei nodi), un thread
java -Xss4m -cp $CP $T.experiments.OrderingBench $W 5 6 14 base rec all
# sforo del tempo con 4 thread veri (margine 0)
java -Xss4m -cp $CP $T.experiments.OvershootBench $W 5000 30 4 rec
# TT: nodi con TT piccola/grande e politica (profondità fissa)
java -Xss4m -Xmx3g -cp $CP $T.experiments.TTBench $W 5 6 14 base 22 16 16:two 14 14:two

# ARENA a tempo REALE uguale (m=0 su ENTRAMBI i lati: msPerMossa = tempo di ricerca vero), 4 thread per lato.
java -Xss4m -Xmx6g -cp $CP $T.training.EqualTimeArena m=0:ord=rec:$W m=0:$W 30 5000 1 4
#   TT (memoria raddoppiata: due lati): dimensione e politica
java -Xss4m -Xmx6g -cp $CP $T.training.EqualTimeArena m=0:tt=25:$W m=0:tt=22:$W 20 10000 1 4
java -Xss4m -Xmx6g -cp $CP $T.training.EqualTimeArena m=0:tt=22:tt2:$W m=0:tt=22:$W 20 10000 1 4
```
Nell'arena la TT di default è 2^18 per lato (come prima): per le prove di TT scrivi `tt=` su entrambi i lati.

**Client:** `java ... TablutNegamaxClient WHITE 60 IP cnn.txt [threads=N] [order=rec] [tt=25] [tt2=true] [margin=MS]`. Se il tuo script di lancio non passa `-Xmx`, l'heap
è 1/4 della RAM; `tt=25` (805 MB) richiede un heap ≥ 1,6 GB per il controllo del client (altrimenti scende da solo e lo scrive).

## 7. Limiti
* Nessuna arena; un solo core; budget brevi; una sola CNN (solo supervisionata); posizioni di partite giocate da `Simple` (profondità 3, 10 % mosse casuali), non da partite di gara.
* Il guadagno dell'ordinamento è in **nodi** (verificato a parità di valore): in forza equivale a circa 0,1 ply, che a 60 s può essere sotto il rumore di una arena.
* `rootsc` cambia il tie-break alla radice: con pari valore può scegliere una mossa diversa da prima.
* La TT a due slot non è mai stata misurata con 4 thread reali né a 60 s.
* Il test dal vivo col server reale (Server -t 5) ha giocato 40 semi-mosse senza mosse rifiutate né errori, ma **non fino alla fine** della partita.

## 8. File
Modificati: `engine/FastState.java` (+ `tacticalValue`), `aiplayer/Negamax.java`, `aiplayer/TranspositionTable.java`, `selftest/FastStateCheck.java`,
`aiplayer/SearchOptimizationsCheck.java` (flag `exhaustive` del `Cfg`), `aiplayer/NoAllocationCheck.java` (`-Dnoalloc.order`), `training/EqualTimeArena.java`
(prefissi `ord=`, `tt=`, `tt2:`, `m=`; corretta la nota sul margine: 300 ms, non 400), `client/TablutNegamaxClient.java`, `tools/mutation_engine.py`.
Nuovi: `aiplayer/OrderingExtCheck`, `TwoSlotTTCheck`, `OrderingSpec`; `experiments/OrderingBench`, `TTBench`, `OvershootBench`, `ValueDiffProbe`;
`training/ArenaPrefixCheck`; `client/NegamaxClientArgsCheck`.
Per compilare senza `javac` nel PATH (come nel sandbox): `java -m jdk.compiler/com.sun.tools.javac.Main ...`.
