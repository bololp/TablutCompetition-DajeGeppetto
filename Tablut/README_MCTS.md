# README_MCTS — Monte Carlo Tree Search come alternativa a Negamax

Questo documento accompagna i file nuovi/modificati per aggiungere **l'opzione di giocare con MCTS invece che con Negamax**. `Negamax`,
`FastState`, `AshtonRules`, gli evaluator e i trainer **non sono stati toccati** (i test di equivalenza con `LegacyNegamax` restano validi).
Ogni numero e' **misurato** in un sandbox a **1 core** (non nella VM di gara a 4 core); dove un'affermazione e' un ragionamento e non una misura,
e' scritto.

## 1. In breve

* `algo=mcts` nel client (o `TablutMctsClient`) gioca con un MCTS (PUCT, **senza rollout casuali**) che usa lo **stesso evaluator** di Negamax
  (`SimpleEvaluator`, MLP o CNN). Il default del client **resta Negamax**.
* E' verificato con 3 suite (158 + 59 verifiche, piu' quelle del client), un walker che ricalcola l'albero in modo indipendente, oracoli
  di forza bruta, partite contro `AshtonRules` come arbitro e un **mutation test** (74 mutanti, tutti rilevati: §5) e una prova dal vivo col server (§6.4), che ha trovato un difetto vero (§7).
* **Forza: non ho evidenza che batta Negamax.** A 100 ms reali per mossa, 1 core, regole del server (§6):
  con `SimpleEvaluator` MCTS fa **39,4% [29-50]** contro Negamax (80 partite): un po' piu' debole;
  con la rete MLP (pesi di partenza, deboli) fa **58,8% [43-73]** e **65,0% [50-78]** (40 partite ciascuno, `mm=0` e `mm=1`): davanti, ma gli intervalli
  non escludono il 50%;
  con la CNN `cnn_t300000` **non ho misurato la forza in arena** (solo il costo: ~1.700 nodi/s, §6.2);
  **Il regime di gara (60 s, 4 core, rete allenata) non e' stato misurato**: serve la tua macchina (§9).
* Un risultato di misura da sapere subito: la **cache delle valutazioni**, che avevo messo come default, **rallentava** (§6.2): ora e' `auto`
  (accesa solo per la CNN).
* **Memoria**: l'albero cresce senza limite (~1.700 byte per nodo, misurati). Il client mette un tetto ai nodi ricavato dall'heap e **la ricerca si
  ferma quando lo raggiunge**. Con l'heap di default (2 GB) sono ~358.000 nodi: con `SimpleEvaluator` (~30.000 nodi/s a 1 thread) o con l'MLP
  (~5.600 nodi/s, 4 thread) e' meno dei 60 s di gara. **Lancia la JVM con `-Xmx` piu' alto** (si veda §7).

## 2. Come si usa

Dalla cartella `Tablut` (dopo aver copiato i file e compilato come per gli altri README):

```bash
javac -encoding iso-8859-1 -d build -cp "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar" $(find src -name "*.java")
CP="build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; T=it.unibo.ai.didattica.competition.tablut
```

**Client di gara** (gli argomenti obbligatori sono quelli di sempre; i nuovi sono opzionali, in qualunque ordine dopo i primi tre):

```bash
java -Xmx5g -cp $CP $T.client.TablutNegamaxClient WHITE 60 localhost pesi.txt algo=mcts threads=4
java -Xmx5g -cp $CP $T.client.TablutMctsClient   WHITE 60 localhost pesi.txt threads=4          # equivalente
java -Xmx5g -cp $CP $T.client.TablutMctsClient   BLACK 60 localhost pesi.txt mcts=cpuct=1.5,mm=0.5
java -Dtablut.algo=mcts ...                                                                     # equivalente, da proprieta' di sistema
```

`mcts=` prende `chiave=valore,...` (`MctsConfig`); `threads=` e `margin=` valgono come per Negamax; `order=`, `tt=`, `tt2=` valgono solo per
Negamax. Un valore non valido stampa un avviso e viene ignorato (come per le altre opzioni del client). Parametri (default tra parentesi):

| chiave | significato |
|---|---|
| `cpuct` (1.0) | peso dell'esplorazione in `Q + cpuct * P * sqrt(N) / (1 + n)` |
| `tau` (0.2) | temperatura del softmax dei valori statici dei figli, che da' i prior `P` |
| `mm` (0) | peso 0..1 dei "minimax impliciti": stima = `(1-mm)*media + mm*minimax` |
| `expand` (1) | un arco si espande alla sua `(expand+1)`-esima visita (0 = alla prima) |
| `vloss` (1.0) | peso della perdita virtuale (parallelismo) |
| `reuse` (true) | riusa il sottoalbero tra una mossa e la successiva |
| `cache` (auto) | log2 delle voci della cache di valutazioni; `auto` = 2^20 solo con la CNN; `0` = spenta |
| `maxnodes` (0) | tetto ai nodi espansi; 0 = 30% dell'heap / 1800 byte |
| `scale` (auto) | scala dell'evaluator: `tanh(v/100)` per Simple, `v/outputScale` per le reti |
| `maxdepth` (512) | profondita' massima dell'albero |
| `escape` (true) | rilevatore esatto di "il Re puo' fuggire al prossimo tratto" |

**Confronto a tempo uguale** (nuova arena, regole del server, `EqualTimeArena` e' invariata):

```bash
java -cp $CP $T.training.SearchArena mcts:simple negamax:simple pairs=30 ms=200            # +IC95 di Wilson, dettaglio per colore e per lato
java -cp $CP $T.training.SearchArena mcts=cpuct=1.5:cnn.txt t4:negamax:cnn.txt ms=5000 par=1 threads=1
```

Lato = `[mcts | mcts=chiave=valore,... | negamax | ord=NOME | tN | m=MS :]*<simple | pesi.txt>`. `ms` e' il tempo di ricerca **reale** (margine 0):
con `EqualTimeArena` il budget e' ridotto di 400 ms dal margine di Negamax (minimo 50 ms), la trappola del README_CNN §7.

**Test e misure** (tutti terminano con `... OK` o con codice 1):

```bash
java -Xss8m -cp $CP $T.aiplayer.MctsSearchCheck           # 158 verifiche, ~45 s
java -Xss8m -cp $CP $T.aiplayer.MctsConcurrencyCheck      #  59 verifiche, ~35 s
java -cp $CP $T.client.NegamaxClientArgsCheck
java -cp $CP $T.experiments.MctsBench [--only=throughput|memory|cache] pesi.txt ...      # throughput, byte per nodo, A/B della cache
python3 tools/mutation_mcts.py [lista]                    # 74 mutanti; BUILD=cartella delle classi; lista = es. 5,20,26-40
```

## 3. Come funziona

Ogni simulazione scende dalla radice scegliendo a ogni nodo l'arco che massimizza `Q + cpuct * P * sqrt(N) / (1 + n)` (PUCT), arriva a una foglia,
ne ricava un valore e lo propaga alla radice. Non ci sono rollout: a Tablut una partita casuale ha quasi zero informazione. Il valore di una foglia
viene dall'evaluator, compresso in `[-0.98, 0.98]` (cosi' le vittorie/sconfitte esatte, `+-1`, restano distinte). Un nodo si espande valutando
**tutti i suoi figli**: quel valore e' il Q iniziale dell'arco (una "first play urgency" informata) e, con un softmax, il suo prior; il valore che la
simulazione restituisce e' il migliore dei figli (un minimax a 1 piano, come un'alpha-beta alla frontiera). Ogni valore memorizzato e' "dal punto di
vista di chi muove in quel nodo" (convenzione negamax: salendo si cambia segno).

La mossa giocata e': una vittoria **dimostrata** se c'e' (la piu' breve); altrimenti, escluse le mosse dimostrate perdenti, la piu' visitata.

## 4. Le ottimizzazioni, e cosa e' misurato di ognuna

| ottimizzazione | cosa fa | evidenza |
|---|---|---|
| **Solver a limiti** (score-bounded MCTS) | ogni arco/nodo ha limiti `{-1,0,+1}` del valore di teoria dei giochi; fughe e catture del Re sono terminali esatti; i limiti salgono con la regola negamax; un arco dominato non si sceglie piu', un nodo risolto non si esplora, la ricerca si ferma a radice risolta | vittorie in 1: 913/913; **in 3 con Simple: Bianco 262/263, Nero 89/94**; sconfitte forzate in 2 dimostrate; ogni dimostrazione certificata dal walker (§5) |
| **Pareggio per ripetizione esatto** | storia reale + cammino; in un albero (senza condivisione di nodi) il valore di un nodo dipende dal suo solo cammino: **nessun problema GHI**, i limiti sono esatti | scenari costruiti (ciclo di 4 semi-mosse), storie reali con catture, walker |
| **Rilevatore di fuga del Re** | dopo una mossa del Nero, se il Bianco ha una via libera al bordo la mossa e' una **sconfitta dimostrata**, senza espandere (~4 scansioni di raggio) | vittorie forzate in 3 con Simple dimostrate: 332/357 senza, **351/357 con**; esatto contro la forza bruta su decine di migliaia di posizioni (e su 7.161 con il Re su una citadel) |
| **Vittoria piu' breve** | tra piu' vittorie dimostrate, quella con meno semi-mosse | test a mano (distanze 1 e 3 con visite a favore della piu' lunga) |
| **Espansione ritardata** (`expand=1`) | la prima visita a una mossa restituisce il valore statico (gia' noto, a costo zero) e non crea il nodo | 100 ms, Simple, 24 partite: 47,9% con `expand=1` contro 25,0% con `expand=0` |
| **Riuso del sottoalbero** | la ricerca riparte dal nodo della mossa giocata e della risposta avversaria, ammesso solo se la **storia** coincide con quella presupposta | 89-93% delle mosse dell'arena riusano l'albero; **non ho misurato la forza con e senza** |
| **Parallelismo ad albero + perdita virtuale** | un lock per nodo (sezioni di decine di ns), espansione fuori dai lock, arco in espansione marcato e saltato | corretto: invarianti verificate dopo corse a 2/4/8/16 thread; **lo speedup su 4 core non e' misurato** |
| **Minimax impliciti** (`mm`) | stima = `(1-mm)*media + mm*minimax delle valutazioni statiche` | **nessun beneficio misurato** (§6.3): default 0 |
| **Cache delle valutazioni** | la stessa posizione raggiunta con ordini diversi si valuta una volta (chiave Zobrist, voci XOR-verificate, lock-free) | **A/B a parita' di albero: costa il 17-31% con Simple, neutra con l'MLP, -5/-14% con la CNN** (§6.2): default `auto` |
| **Tetto di memoria** | raggiunto il tetto di nodi la ricerca si ferma invece di andare in `OutOfMemoryError` | ~1.700 byte/nodo misurati; il tetto automatico usa 1800 |

Cio' che **non** e' in MCTS e che Negamax ha: transposition table (non si condividono nodi tra cammini: e' una scelta, per la correttezza esatta dei limiti
con la ripetizione), killer/history, PVS.

## 5. Verifiche

**`MctsSearchCheck` (158 verifiche, white-box)** e **`MctsConcurrencyCheck` (59)**, con `MctsTestSupport`:

* **Walker indipendente**: dopo ogni ricerca ripercorre l'albero **rigiocando le mosse** e ricalcola da zero, senza fidarsi dei campi: mosse legali, terminali,
  ripetizioni (con chiavi **testuali** scacchiera+turno e conteggio dei pezzi per le catture, non Zobrist ne' `wasCapture`), limiti dimostrati, minimax
  implicito e le **medie `Q` di ogni arco** (ogni simulazione termina in un evento ben definito, quindi la media e' calcolabile). 0 discordanze su 540 alberi a
  1 thread (5 configurazioni) e sugli alberi a 2/4/8/16 thread, con Simple, MLP e CNN.
* **Oracoli di forza bruta**: vittoria in 1 (913 posizioni, i due colori), mosse sicure, vittoria forzata in 3, sconfitta in 2; il rilevatore di fuga contro la
  forza bruta; un solo passo di simulazione contro il minimax a 2 piani calcolato a mano (601 posizioni, 280 Bianco / 321 Nero).
* **A mano**: la formula PUCT (A-K: prior, FPU, perdita virtuale, miscela `mm`, potatura, archi in espansione), la retropropagazione (segni, limiti, minimax
  a uno e due livelli, arrivo tardivo su un arco risolto, annullamento a meta' cammino) e la scelta alla radice.
* **Differenziali deterministici** (1 thread, simulazioni fisse): due esecuzioni identiche; capacita' iniziale degli array di 2; cache si/no (stesso albero,
  valutazioni+hit = valutazioni senza cache); storia con o senza la radice, storia di 120 voci.
* **Scenari costruiti**: ripetizione (ciclo di 4 semi-mosse: pareggio esatto alla radice, dentro l'albero, scelta di chi perde/vince), stallo dell'avversario
  dimostrato, Re su una citadel.
* **Concorrenza e ciclo di vita**: 16 thread su alberi minuscoli con un evaluator lento (le espansioni durano abbastanza da produrre contesa:
  simulazioni annullate > 0), riuso a profondita' 1-4, con storia sbagliata e dopo catture, tetto di nodi/profondita'/simulazioni/tempo, un evaluator che
  lancia eccezioni (alla radice, a meta' ricerca, in un helper) e un helper che non termina, thread di lavoro dopo `shutdown()`.
* **Partite complete** contro `AshtonRules` come arbitro indipendente: nessuna mossa rifiutata; ogni vittoria dimostrata e' stata poi convertita.

**Mutation test (`tools/mutation_mcts.py`, 74 mutanti)**: **74/74 rilevati** sulla suite finale. Il giro e' stato fatto a blocchi (il sandbox viene sospeso tra una chiamata e l'altra e uccide i job in background), con
lo stesso insieme di sorgenti e una suite che in ogni blocco era uguale o piu' forte della precedente (un mutante rilevato da un sottoinsieme dei test lo e' anche
dal sottoinsieme piu' grande); i 6 mutanti sulla memoria/slot e le correzioni dei test sono stati rieseguiti sul build finale. Nei giri intermedi **due mutanti sono
sopravvissuti e sono stati chiusi con test mirati** (#43: il tetto di simulazioni che fermava tutti i thread; #64: l'array del percorso del worker che tiene vivo
il vecchio albero). Il runner usa `-Dmcts.failfast=true` (il primo fallimento termina il test): un mutante e' contato come rilevato anche se va in loop fino al timeout.

Il mutation test ha trovato **buchi miei nei test** (tutti chiusi, ognuno con un test mirato): arrivo tardivo su un arco gia' risolto; perdita virtuale nel
denominatore del bonus; annullamento a profondita' >= 1; crescita dell'array della storia; tetto di simulazioni con annullamenti; scelta della vittoria piu' breve
tra due dimostrate (la ricerca si ferma alla prima dimostrazione, quindi due vittorie dimostrate quasi non esistono nei test non costruiti); margine di tempo
(la tolleranza era piu' larga dell'errore); riuso della stessa radice con storia diversa; il ramo "il Re parte da una citadel" dello scanner, che nessuna posizione
esercitava. Mutanti **equivalenti** (non rilevabili, ragionati e annotati nello script): azzerare la storia alla cattura (ottimizzazione di costo: dopo una cattura
nessuna posizione precedente puo' ripresentarsi); ripulire la sentinella dopo un'eccezione (l'albero viene scartato comunque); il limite di 5 passi della citadel
(lungo una linea libera una citadel ne vede un'altra solo a 1, 2 o 8 passi: ogni soglia da 3 a 7 e' equivalente); il trono che blocca il Re (oltre il trono, sulla riga
o colonna 4, si incontra sempre una citadel prima di un bordo).

**Difetti veri nel codice** trovati dalle verifiche (non dai test unitari scritti prima): (1) il vecchio albero raggiungibile durante la ricerca, trovato dal timeout della prova
dal vivo (§7); (2) il tetto di simulazioni che a piu' thread poteva completarne N-1 (un'asserzione intermittente del mio test: con un'annullamento a fine ricerca lo slot si perdeva;
misurato su 180 ricerche a 8 thread: 5 non esatte prima, 0 dopo); (3) il default della cache, che rallentava (A/B, §6.2); (4) la stima dei byte per nodo, sottostimata del 13% (misurata).
Il walker invece non ha trovato nessuna discordanza nell'implementazione (ha pero' rilevato tutti i bug iniettati).

Errori **miei nei test** trovati dai test stessi mentre li scrivevo (non nel codice): campionamento delle posizioni di gioco con passo pari (solo Nero di turno),
formula del test "una simulazione" con il turno sbagliato, scenario di stallo con il Re (una vittoria piu' rapida lo copriva), previsione di riuso dopo una storia
alterata (dopo una cattura il riuso e' invece corretto). Un'asserzione e' stata corretta dopo aver capito perche': "le vittorie in 3 sono tutte nella stessa
mossa" era falsa, perche' MCTS puo' dimostrare una vittoria piu' lunga prima (da qui il controllo sulla brevita').

## 6. Misure

Sandbox a 1 core, un thread di ricerca per lato. IC95 di Wilson. `SearchArena`: 4 semi-mosse di apertura casuali, ogni apertura giocata con i colori scambiati,
tetto 160 semi-mosse, regole del server. Con 24-80 partite gli intervalli sono **molto larghi**.

### 6.1 Contro Negamax, a 100 ms reali per mossa

| evaluator | MCTS (parametri) | partite | punteggio MCTS |
|---|---|---|---|
| Simple | default (`cpuct=1`) | 80 | **39,4% [29,4-50,3]** (+27 =9 -44) |
| Simple | `mm=0.5` | 80 | 30,6% [21,6-41,4] |
| Simple | `mm=1` | 80 | 30,6% [21,6-41,4] |
| MLP `mlp_v3_pretrained` (*) | `mm=0` | 40 | **58,8% [43,4-72,6]** (+23 =1 -16) |
| MLP `mlp_v3_pretrained` (*) | `mm=1` | 40 | **65,0% [49,5-77,9]** (+26 =0 -14) |

(*) giocate quando il default della cache era 2^18; con l'MLP la cache e' neutra (§6.2), quindi non cambia in modo apprezzabile.
Tempo per mossa (MLP): MCTS 85 ms, Negamax 97 ms (Negamax ha usato un po' **piu'** tempo). MCTS: ~3.300-4.300 simulazioni e ~40.000 valutazioni per mossa,
riuso dell'albero 89%; Negamax profondita' 4,3 e 65.000 nodi.

### 6.2 Costo, memoria, cache (`MctsBench`, 1 thread, posizione iniziale)

| evaluator | simulazioni/s | valutazioni/s | nodi/s | (cache spenta, `expand=1`) |
|---|---|---|---|---|
| Simple | 177.000 | 1,9 M | ~29.800 | |
| MLP | 14.600 | 378.000 | ~5.600 | |
| CNN `cnn_t300000` | 6.200 | 113.000 | ~1.700 | |

**Byte per nodo espanso**: 1.670-1.755 (60.000 nodi, heap dopo gc; uguale per Simple/MLP/CNN e con `expand` 0/1, `mm` 0/0.5: la struttura del nodo non
dipende dall'evaluator).

**Cache, A/B a parita' di albero** (stesse posizioni, simulazioni fisse, 1 thread: l'albero e' identico con e senza cache, quindi la differenza di tempo
e' solo la cache; minimo su 3 ripetizioni dopo il riscaldamento):

| evaluator | cache 2^12 | 2^16 | 2^18 | 2^22 | hit |
|---|---|---|---|---|---|
| Simple | 1,01x | 1,17x | 1,23x | 1,31x | 2-14% |
| MLP | 1,07x | 1,02x | 0,87x | 0,99x | 4-17% |
| CNN | 1,04x | 0,99x | 0,95x | 0,86x | 4-10% |

(tempo relativo a nessuna cache; <1 = piu' veloce). Il beneficio e' `hit-rate * costo della valutazione - costo di accesso`: con una valutazione da
0,3 us costa piu' di quanto fa risparmiare. Per questo il default e' `auto` (2^20 con la CNN, spenta altrimenti).

### 6.3 Screening dei parametri (Simple, 100 ms, 24 partite, stesse aperture; rumore ~ +-18 punti)

`cpuct=0.4`: 16,7% · **`cpuct=1` (default): 47,9%** · `cpuct=2`: 39,6% · `cpuct=4`: 33,3% · `expand=0`: 25,0% · `cpuct=2,mm=1`: 47,9% · `cpuct=2,mm=1,expand=2`: 39,6% ·
`cpuct=2,tau=0.5`: 43,8% · `cpuct=2,tau=0.1`: 37,5%. L'unico segnale solido e' che **meno esplorazione (0,4) e l'espansione immediata fanno male**; il resto e'
dentro il rumore. Il default non e' stato "tarato": e' il punto di un massimo morbido.

### 6.4 Prova dal vivo con il server

Il server vero (`-t 2 -r 4`, timeout 2 s), un client MCTS (`SimpleEvaluator`, 1 thread, margine 300 ms) e un client `TablutNegamaxClient` (Simple, 1 thread):

| # | MCTS | codice | esito | tempo max / mossa | note |
|---|---|---|---|---|---|
| 1 | Bianco | prima della correzione di §7 | persa in 13 mosse, regolare | 1732 ms | riuso dell'albero fino a 114.000 visite; la sconfitta dimostrata e' riportata alla mossa 12 |
| 2 | Nero | prima | **persa per timeout (mossa 21)** | **2036 ms** (budget 1700) | albero di 168.256 nodi (83.669 riusati) |
| 3 | Nero | prima | persa in 24 mosse, regolare | 1736 ms | pause GC fino a 484 ms |
| 4 | Nero | finale | **vinta** (22 mosse) | 1947 ms | pausa GC massima 573 ms |
| 5 | Bianco | finale | persa in 13 mosse, regolare | 1748 ms | pausa GC massima 112 ms |

Cosa dimostra: protocollo e tempi corretti (nessuna mossa rifiutata, ~1,7 s a mossa su 2 s, 140.000-330.000 simulazioni a mossa, riuso dell'albero attivo, esito
dimostrato riportato dal client). **Non dice nulla sulla forza** (1 vittoria su 5, in linea con il ~39% dell'arena a 100 ms con Simple, ma 5 partite sono niente).
La partita 2 ha mostrato un difetto reale e il timeout si e' presentato **una volta su tre** (§7).

## 7. Memoria, garbage collector e timeout

L'albero non e' una TT di dimensione fissa: ogni nodo espanso costa ~1.700 byte. Il tetto automatico e' `30% dell'heap / 1800`, e **raggiunto il tetto la ricerca
si ferma** (il tempo restante non si usa). Dai tassi misurati (1 thread, §6.2), con l'heap di default (2 GB, ~358.000 nodi) il tetto arriva dopo circa
12 s (Simple), 64 s (MLP), 210 s (CNN); con 4 thread, ammesso uno scaling lineare (**non misurato**), 3 s, 16 s, 52 s. Il riuso del sottoalbero porta nodi dalla
mossa precedente, che contano nel tetto. Il client stampa il tetto e l'heap all'avvio; se il tetto scatta `describeLastSearch()` scrive `TETTO-MEMORIA`.
Su una VM da 8 GB, `-Xmx5g` porta il tetto a ~860.000 nodi.

**Un timeout trovato dalla prova dal vivo (partita 2).** L'MCTS-Nero ha impiegato 2036 ms su un budget di 1700 (timeout del server 2 s, margine 300 ms) e ha perso.
* **Diagnosi** (log `-Xlog:gc` del client): pause del GC fino a **484 ms** (Young 316 ms; una Full GC da 130-185 ms ogni pochi secondi, "Allocation Failure"): il sandbox
  usa il collector seriale con 1 core e un heap di 1 GB che cresce a scatti. Una pausa che attraversa la scadenza fa sforare.
* **Difetto nel codice** (mio): il campo `root` (il vecchio albero) veniva sovrascritto solo **a fine ricerca**, e gli array `pNode` dei worker ricordavano il percorso
  dell'ultima simulazione: durante tutta la ricerca successiva restavano raggiungibili il vecchio albero e il nuovo, quindi la memoria viva era la **somma** (e il tetto
  sui nodi non la vedeva). Corretto: `root`, `rootPos`, `rootHist` si azzerano appena scelto il nodo da riusare e i `pNode` si azzerano a inizio ricerca. Test:
  un riferimento debole al vecchio nodo radice con `System.gc()` *durante* la ricerca successiva, e il controllo che ogni nodo ancora nell'array del percorso appartenga al
  nuovo albero (scegliendo una mossa diversa dall'ultimo percorso, altrimenti i nodi obsoleti sarebbero nel sottoalbero riusato); due mutanti li rilevano.
* **Misure prima/dopo** (`experiments.MctsTimingProbe games=3 budget=1700 opp=100`, `-Xmx1000m`, 1 core, collector seriale; le partite non sono identiche tra le esecuzioni:
  conta la distribuzione, non la singola mossa):

| | mosse MCTS | pause | Full GC | tempo totale in pausa | pausa massima | pause > 100 ms | pause > 200 ms | sforamento massimo |
|---|---|---|---|---|---|---|---|---|
| prima | 85 | 75 | 23 | 8.286 ms | 273 ms | 34 | 9 | 200 ms |
| dopo la correzione | 66 | 104 | 29 | 6.196 ms | **148 ms** | **11** | **0** | 178 ms |
| dopo, con `-Xms1000m -Xmx1000m` | 71 | 32 | 7 | **3.780 ms** | 325 ms | 22 | 4 | 189 ms |

  La correzione riduce la **coda** (pause > 200 ms: 9 → 0). Pre-dimensionare l'heap (`-Xms` = `-Xmx`) riduce il **costo totale** del GC (-39%) ma **non la pausa massima** su
  questo core (una young generation piu' grande copia piu' dati per volta): non e' un rimedio al timeout.
* **Sul codice finale** (partite 4-5) nessun timeout, ma **non e' risolto del tutto**: la partita 4 ha avuto una pausa di 573 ms e ha sforato di 247 ms (1947 ms su un budget di 1700, a un passo dal
  timeout). Con timeout di 2 s e margine di 300 ms il rischio resta; e' il regime di prova, non quello di gara.
* **Difesa**: il margine di sicurezza, che **non va abbassato**. Il default del client per timeout >= 20 s e' 1500 ms per Negamax e **2000 ms per l'MCTS** (+500 ms, solo se non dai
  `margin=`; costo < 1% del tempo a 60 s): 3,5 volte la pausa peggiore misurata. **Non misurato**: il regime di gara (60 s, 4 core, G1 o altro collector): le pause li' possono essere
  diverse (anche piu' lunghe con un albero da 600 MB). Se puoi, lancia `MctsTimingProbe` con le tue opzioni JVM e `-Xlog:gc`.

## 8. File

**Nuovi** (package `it.unibo.ai.didattica.competition.tablut`):

| File | Ruolo |
|---|---|
| `aiplayer/MoveSearcher.java`, `NegamaxSearcher.java` | interfaccia comune di ricerca e adattatore di `Negamax` (che non e' stato modificato) |
| `aiplayer/Mcts.java`, `MctsConfig.java`, `EvalCache.java` | l'algoritmo, i parametri (`chiave=valore`), la cache di valutazioni |
| `aiplayer/MctsTestSupport.java`, `MctsSearchCheck.java`, `MctsConcurrencyCheck.java` | oracoli, walker, test |
| `training/SearchArena.java` | arena a tempo reale uguale con le regole del server (riusa `GameTracker`) |
| `client/TablutMctsClient.java` | `main` che equivale a `TablutNegamaxClient ... algo=mcts` |
| `experiments/MctsBench.java` | throughput, byte per nodo, A/B della cache |
| `tools/mutation_mcts.py` | mutation test (74 mutanti) |
| `experiments/MctsTimingProbe.java` | sonda dei tempi: di quanto ogni mossa MCTS supera il budget (GC) |

**Modificati**: `client/TablutNegamaxClient.java` (campo `MoveSearcher`, `algo=`, `mcts=`, `buildSearcher`; il percorso Negamax e' lo stesso: la forma `List<Long>` di
`findBestMove` e' solo un wrapper di quella `long[]`) e `client/NegamaxClientArgsCheck.java` (controlli della nuova opzione).
`experiments/MctsPlayer`, `MctsArena`, `MctsCheck` (il prototipo del README_CNN §7) restano invariati.

## 9. Limiti, cose non fatte, prossimi passi

* **1 core, 100 ms, pesi deboli, 24-80 partite**: il regime di gara (60 s, 4 core, rete allenata) e lo **speedup dei thread non sono misurati**. A tempi lunghi
  la profondita' persa pesa meno, ma non lo so.
* **I parametri non sono tarati** (§6.3): `mm` ha dato segnali opposti con Simple e con l'MLP, quindi il default e' il classico `mm=0`.
* **Il rilevatore di fuga copre le vittorie del Bianco, non le catture del Re del Nero** (con Simple: vittorie forzate in 3 del Nero dimostrate 89/94).
  La completezza dipende dall'evaluator: una mossa vincente che la valutazione disprezza (prior ~0) puo' non essere mai esplorata (con una valutazione "rumore" le
  vittorie in 3 dimostrate scendono a 234/263 e 62/94).
* **Nessuna TT tra cammini** (scelta consapevole, §4), nessuna policy appresa: i prior vengono dai valori statici dei figli.
* **Il timeout non e' garantito**: una pausa del GC piu' lunga del margine fa perdere la mossa (§7).
* **Al tetto di memoria la ricerca si ferma** invece di continuare in modo degradato; non ho implementato la potatura del sottoalbero.
* Con una CNN (~1.700 nodi/s) valutare tutti i figli a ogni espansione costa molto: un'alternativa (prior da `SimpleEvaluator`, valore dalla rete) non l'ho provata.
* Non ho misurato la forza con e senza riuso dell'albero, ne' `vloss` diversi da 1.

**Prossimi passi consigliati.** (1) Sulla macchina di gara: `MctsBench` con la rete vera, poi `SearchArena mcts:rete.txt t4:negamax:rete.txt ms=5000 par=1 threads=4`
(almeno 30 coppie): e' la misura mancante. (2) Se MCTS non e' sfavorito, provare `cpuct` 0,7-1,5 e `mm` su quel regime e il riuso dell'albero acceso/spento.
(3) Alzare `-Xmx` come in §7 prima di giocare.
