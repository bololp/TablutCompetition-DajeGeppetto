# README_FASTSTATE - migrazione a una rappresentazione leggera dello stato

Questo documento accompagna la migrazione del motore di ricerca da `State` (`Pawn[][]` + oggetti `Action`) a `FastState` (scacchiera
`byte[81]`, mosse `int`, make/unmake). Ogni numero e' **misurato** in un sandbox a **1 core** (JDK 21, non la VM di gara a 4 core);
dove un'affermazione e' un ragionamento e non una misura, e' scritto.

## 1. In breve

* **Migrazione riuscita e verificata.** La ricerca (`Negamax`, tutta: TT, PVS, killer, history, ripetizione/GHI, Lazy SMP), gli evaluator
  (`SimpleEvaluator`, `NeuralEvaluator`/`TablutFeatures`, `CnnEvaluator`/`ConvNet`), `MctsPlayer` e il client di gara lavorano ora su
  `FastState`. A thread singolo e a profondita' fissa la nuova ricerca da' **la stessa mossa, lo stesso punteggio (al bit), la stessa
  profondita' e lo stesso numero di nodi** della versione storica (§5).
* **Piu' veloce, a parita' di nodi** (§6): **x3,6** nodi/s con `SimpleEvaluator` (601k -> 2,16M), **x1,5** con l'MLP (289k -> 439k),
  **x1,36** con la CNN (98k -> 134k; qui domina il costo della rete). Con 1 s per mossa la profondita' media sale da 5,33 a 6,25 (Simple)
  e da 4,58 a 4,83 (MLP).
* **Zero allocazioni per nodo**: 0,000-0,004 byte/nodo contro ~1.500 byte/nodo della versione storica (uno `State` clonato + un'`Action`
  con due stringhe + boxing di `Long` per ogni figlio).
* **Cosa NON e' stato migrato, e perche'** (§7): trainer, arene e pretrainer restano al "confine" `State`/`Action` con una conversione O(81)
  per chiamata di ricerca (trascurabile rispetto a 50-1500 ms di ricerca). Il tuo `AshtonRules` e `ZobristHasher` restano come oracoli dei test.

## 2. Architettura: prima e dopo

| | prima | dopo |
|---|---|---|
| Posizione | `State` (`Pawn[][]` 9x9: 10 oggetti, 81 riferimenti a enum) | `FastState`: `byte[81]` + turno, cella del Re, conteggi, vincitore, hash (campi primitivi) |
| Mossa | `Action` (due stringhe, 3 `parseInt` per le coordinate) | `int` = `da \| (a << 7)` (`Move`), nel campo a 16 bit della TT |
| Figlio di un nodo | `State.clone()` (costruttore che riempie la posizione iniziale, poi la copia) | `makeMove` sul posto + `unmakeMove` (O(1), `UndoInfo` per ply) |
| Lista mosse | `List<Action>` allocata per nodo | `int[]` preallocato per ply, riempito da `generateMoves` |
| Hash Zobrist | `ZobristHasher.updateHash(h, prima, dopo, mossa)` (confronta due `State`) | aggiornato **dentro** `makeMove` (<= 6 celle) |
| Storia ripetizioni | `ArrayList<Long>` (boxing, `contains`) | stack di `long` + inizio del segmento "dall'ultima cattura" |
| Esito di un nodo | `wasCapture(prima, dopo, mossa)` | `undo.wasCapture()` |
| Lettura TT | `new ProbeResult()` per nodo | `probeInto(...)` su un unico `ProbeResult` |
| Evaluator | `evaluate(State)` | `evaluate(FastState)` (+ `default evaluate(State)` che converte) |

Il pacchetto nuovo e' `it.unibo.ai.didattica.competition.tablut.engine`:

| File | Ruolo |
|---|---|
| `FastState.java` | posizione + regole: `generateMoves`, `hasLegalMove`, `makeMove`, `unmakeMove`, `fromState`/`toState`, `computeHash` |
| `BoardConstants.java` | codici pedine, citadel/punte/bordo, tabella `STEPS[dir][cella]` (niente controlli di limiti) |
| `Move.java` | mossa compatta, `index` (da*81+a: killer/history), conversioni da/verso `Action` |
| `UndoInfo.java` | cio' che serve per annullare una mossa (al massimo 4 catture) |
| `ZobristHash.java` | stesse chiavi di `ZobristHasher` (stesso seme e ordine): `hash(FastState) == ZobristHasher.fullHash(State)` |

**Convenzioni da sapere**

* Cella = `riga*9 + colonna` (la stessa di `State.getBoard()[riga][colonna]`); `e5 = 40`. I codici pedina coincidono con `Pawn.ordinal()`
  (vuoto 0, bianco 1, nero 2, re 4); il **trono vuoto e' "cella 40 vuota"** (il codice 3 serve solo come marcatore Zobrist/per gli estrattori).
* **Stesso ordine di generazione delle mosse di `AshtonRules`** (pezzi per cella crescente; direzioni riga-1, riga+1, colonna-1, colonna+1).
  Non e' un dettaglio: determina il tie-break alla radice e quindi l'identita' dei risultati di ricerca.
* Il Re catturato viene **rimosso** dalla scacchiera (`kingSquare = -1`), come in `AshtonRules`. Unica differenza cosmetica: se e' catturato *sul trono*,
  `AshtonRules` lascia `Pawn.EMPTY` in e5, `FastState.toState()` mette `Pawn.THRONE` (e il suo hash include il marcatore di trono vuoto). Non influisce
  sulla ricerca (gli stati terminali non vengono mai valutati ne' espansi).
* `FastState()` parte con il **Bianco** di turno (come nel file di riferimento; `new StateTablut()` parte col Nero e ci pensa il server a impostare il Bianco).

### Regole: quelle del tuo progetto, non quelle del file di riferimento
Il `FastState.java` di riferimento ha una struttura che ho ripreso (array piatto, campi primitivi, make/unmake con undo, hash incrementale),
ma **non ne ho adottato le regole**: ho riportato quelle di `AshtonRules`, verificate contro il motore ufficiale. Leggendo il riferimento le
differenze che ho notato sono: (a) la cattura del Re sul trono / vicino al trono viene controllata dopo **qualunque** mossa nera, non solo quando
la pedina mossa e' adiacente al Re (la tua versione richiede la cattura "attiva", come la regola ufficiale); (b) nessuna generazione di mosse
(quindi nessuna regola delle citadel: non si rientra, tetto di 5 passi, il trono vuoto non si attraversa); (c) nessuna rimozione del Re catturato.
In compenso **tutte le funzionalita' piu' avanzate del tuo progetto sono state migrate**, e non perse: transposition table con bound
EXACT/LOWER/UPPER, hash incrementale identico a prima (le storie di ripetizione esistenti restano valide), pareggio per ripetizione con protezione
GHI, stallo = sconfitta, PVS, killer + history, Lazy SMP con TT lock-free, margine di sicurezza sul tempo, fallback su `SimpleEvaluator`.

## 3. File modificati

| File | Cosa e' cambiato |
|---|---|
| `aiplayer/Negamax.java` | riscritto sul make/unmake. Stessa logica e stessi campi `last*`; API di confine invariata (`findBestMove(State, ms[, storia])` -> `Action`) piu' le forme `findBestMove(FastState, ms[, List<Long> \| long[], n])` -> `int` |
| `aiplayer/Evaluator.java` | metodo astratto `evaluate(FastState)`; `default evaluate(State)` converte. **Cambia il contratto per chi implementa `Evaluator`** (lambda/classi anonime: parametro `FastState`) |
| `aiplayer/SimpleEvaluator.java` | su `FastState`; stessi valori **al bit** (stesso ordine di somma in virgola mobile), materiale dai conteggi incrementali |
| `aiplayer/TablutFeatures.java` | `extract(FastState, ...)` e' l'implementazione; `extract(State, ...)` converte e delega. Il resto dell'estrattore e' intatto |
| `aiplayer/NeuralEvaluator.java`, `CnnEvaluator.java`, `ConvNet.java` | `evaluate(FastState)`, `extractInputs/extractFeatures(FastState, ...)`, `Workspace.setBoard(FastState)`; le forme con `State` restano (convertono) |
| `aiplayer/TranspositionTable.java` | `probeInto` senza allocazioni (`probe` lo usa); resto invariato |
| `client/TablutNegamaxClient.java` | converte lo `State` del server in `FastState` **una volta per mossa** e usa `position.zobristHash` per la storia; niente piu' `countPieces` su `Pawn[][]` |
| `training/GameTracker.java` | l'hash della storia viene da `FastState.hashOf` (una sola sorgente di verita': ricerca e storia non possono divergere) |
| `experiments/MctsPlayer.java` | nodi su `FastState`; i figli si valutano con make/unmake sul posto (nessuno `State` per valutare una mossa) |
| `experiments/Arena.java`, `MctsCheck.java` | adattati alla nuova firma di `Evaluator` |
| test white-box (`RepetitionAndStalemateCheck`, `SearchOptimizationsCheck`, `LazySmpCheck`) | adattati alla nuova API interna (`installForTest`, mosse `int`); **le asserzioni e gli oracoli sono gli stessi** |

**Nuovi:** `engine/*` (5 file), `aiplayer/LegacyNegamax.java` + `LegacySimpleEvaluator.java` (copie **congelate** del motore storico, solo per i test),
`aiplayer/SearchEquivalenceCheck.java`, `aiplayer/NoAllocationCheck.java`, `selftest/FastStateCheck.java`, `selftest/TestPositions.java`,
`experiments/SearchSpeedBench.java`, `tools/mutation_engine.py`.

## 4. Come si usa

```bash
# dalla cartella Tablut
javac -encoding iso-8859-1 -d build -cp "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar" $(find src -name "*.java")
CP="build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; T=it.unibo.ai.didattica.competition.tablut

# i nuovi test
java -cp $CP $T.selftest.FastStateCheck                    # [partite=300] [posizioniCasuali=20000] [seme]   (~20 s)
java -cp $CP $T.aiplayer.SearchEquivalenceCheck            # [partite=12] [posizioniCasuali=40] [seme]        (~20-30 s)
java -cp $CP $T.aiplayer.NoAllocationCheck                 # zero allocazioni per nodo

# velocita' (lo stesso sorgente si compila anche contro la build vecchia: usa solo l'API con State)
java -Xss4m -cp $CP $T.experiments.SearchSpeedBench depth simple 5 16                                   # profondita' fissa
java -Xss4m -cp $CP $T.experiments.SearchSpeedBench time pretrained_weights/mlp_v3_pretrained.txt 1000 12   # profondita' a tempo fisso

# mutation test (dalla cartella Tablut, con la build in build/ o in $BUILD)
CHECK_ARGS="60 4000 777" python3 tools/mutation_engine.py state      # 38 mutanti di FastState (~12 s l'uno)
CHECK_ARGS="6 20 2024"   python3 tools/mutation_engine.py search     # ~35 mutanti di Negamax (~30 s l'uno)
```

Per chi scrive un nuovo evaluator: implementare `double evaluate(FastState s)`. Si legge `s.board[riga*9+colonna]`, `s.kingSquare`, `s.whiteCount`,
`s.blackCount`, `s.turn`; **non si deve modificare `s`** (e' la posizione di lavoro della ricerca).

## 5. Verifiche eseguite

Tutte le verifiche usano il motore storico come **oracolo indipendente**, mai `FastState` stesso.

* **`FastStateCheck` (differenziale vs `AshtonRules` + `ZobristHasher`).** 300 partite complete (21.203 posizioni) + 20.000 posizioni casuali dense:
  per ogni posizione e ogni mossa legale (1,77 M mosse): stessa lista di mosse **nello stesso ordine**; stesso stato dopo `makeMove`
  (scacchiera, turno/terminalita'); `wasCapture` uguale; hash incrementale = hash ricalcolato = `ZobristHasher.fullHash`; conteggi e cella del Re
  coerenti; `unmakeMove` ripristina tutto (anche l'hash); giro `fromState`/`toState`. Piu' 10 casi limite scritti a mano con l'esito scritto a priori
  (cattura contro trono occupato, punte di citadel, Re sul trono con 3/4 lati, Re vicino al trono, tenaglia contro citadel, fuga che cattura, stallo, citadel
  non rientrabili). Copertura richiesta dal test stesso (se un evento non compare, **fallisce**): 97k catture, 6,7k fughe, Re catturato 182 volte sul trono /
  235 adiacente / 7.683 altrove, 1.978 mosse con piu' catture. **0 discordanze su 16,2 M verifiche.**
* **`SearchEquivalenceCheck` (nuova ricerca vs `LegacyNegamax`).** 522 ricerche a thread singolo e profondita' fissa, 4 configurazioni (produzione; alpha-beta
  puro; TT+killer; PVS+history), evaluator `Simple` (confrontato con `LegacySimpleEvaluator`) e una rete, posizioni di partita **con storia reale di ripetizione**
  e posizioni casuali; una partita intera con la **stessa istanza riusata** (TT, history e killer persistenti); scenari di **stallo** costruiti a mano (dentro l'albero,
  per entrambi i colori). Richiesto: stessa mossa, stesso punteggio (`==`), stessa profondita', **stesso numero di nodi** (4.849.746 contro 4.849.746), stesse ricerche PVS
  e stesse ripetizioni incontrate; la posizione di lavoro torna identica alla radice dopo ogni ricerca (anche a tempo scaduto) e quella del chiamante non viene toccata;
  le tre forme dell'API danno la stessa mossa; una ricerca **interrotta a meta'** non inquina la TT. **0 discordanze.**
* **`NoAllocationCheck`**: byte allocati dal thread durante la ricerca, con Simple, MLP e CNN: 0,000 / 0,001 / 0,004 B/nodo (48 B in totale su ~200.000 nodi) contro
  ~1.500 B/nodo della versione storica (stampati per confronto).
* **Suite preesistente**, adattata solo dove usava l'API interna di `Negamax`: `SearchOptimizationsCheck` (stessi numeri del tuo README: 1585 ripetizioni incontrate,
  -91,5% di nodi con TT), `RepetitionAndStalemateCheck`, `TTCorrectnessCheck`, `PVSCheck`, `LazySmpCheck`, `FeaturesCheck` (4000 posizioni contro gli oracoli: convalida
  tutta l'estrazione di feature sulla nuova rappresentazione), `MLPGradientCheck`, `ConvNetCheck` (47), `CnnEvaluatorCheck` (17), `CnnTrainingCheck` (48), `SelfPlayRulesCheck`,
  `MctsCheck` (11), `CrossCheck` (0 mismatch), `WasCaptureCheck`, `ZobristIncrementalCheck`, `ZobristCollisionCheck`: tutti OK sulla build finale (ricompilata da zero dal sorgente consegnato: build pulita, poi l'intera suite).
* **Mutation test dei test nuovi.** `FastState`/`ZobristHash`/`Move`: **38/38** mutanti rilevati (regole di cattura di entrambi i colori, Re sul trono/vicino/altrove,
  fuga, hash, unmake, citadel, trono vuoto, ordine delle direzioni...). `Negamax`: **35/35** (unmake mancante, storia, ripetizione, GHI, TT, ordinamento, killer/history,
  PVS, stallo, vittoria piu' rapida...). I mutanti **equivalenti** (verificati a mano, documentati in `tools/mutation_engine.py`) sono 3: togliere l'esenzione del "lato appena mosso"
  nella cattura del Re (2: `makeMove` ha gia' scritto la pedina nera in quella cella) e saltare l'ultima voce della storia (la posizione genitore non puo' mai coincidere con il figlio).
  **Il mutation test ha trovato due buchi nei miei stessi test**, ora chiusi: (1) nessuna posizione portava a uno **stallo** dentro l'albero (il segno invertito non veniva rilevato);
  (2) nessun test interrompeva una ricerca **a meta'**, quindi "memorizzare in TT a tempo scaduto" non veniva rilevato (ora lo e' sia dal test comportamentale da solo, sia da
  un'asserzione `assert !timeUp` in `Negamax` che `SearchEquivalenceCheck` abilita da solo).
* **Confronto "golden" con la build ORIGINALE** (un programma che usa solo l'API con `State`, compilato ed eseguito contro la build del tuo zip e contro quella nuova): 1.570 righe
  (hash, valori di Simple/MLP/CNN come bit di `double`, checksum del vettore di feature, mossa/punteggio/profondita'/nodi di ricerca) **identiche byte per byte**.
* **Dal vivo col server reale** (`Server -t 3`, client Bianco con `mlp_v3_pretrained.txt` contro client Nero con `SimpleEvaluator`, 1 thread di ricerca ciascuno): 39 mosse, **nessuna mossa
  rifiutata, nessun errore**, vince il Bianco; tempo massimo per mossa 2.718 ms su 3.000 ms di timeout. Compila anche con `-source 8 -target 8` (il JDK del sandbox non ha `--release 8`: l'ho
  verificato a livello di linguaggio e cercando a mano le API piu' recenti nei file nuovi, non con la libreria di classi di Java 8).

## 6. Misure

`SearchSpeedBench`, 1 thread di ricerca, TT 2^20, 16 posizioni (12 per il tempo fisso) generate con il motore storico e seme fisso, JIT scaldato. **Stessi nodi** nelle due versioni (verificato).

| Profondita' fissa | nodi | prima | dopo | accelerazione |
|---|---|---|---|---|
| `SimpleEvaluator`, d=5 | 3.438.613 | 5,72 s (601k nodi/s) | 1,59 s (2,16M nodi/s) | **x3,6** |
| MLP `mlp_v3_pretrained`, d=5 | 3.686.134 | 12,77 s (289k nodi/s) | 8,39 s (439k nodi/s) | **x1,5** |
| CNN `cnn_conv4_hybrid`, d=4 | 327.207 | 3,33 s (98k nodi/s) | 2,44 s (134k nodi/s) | **x1,36** |

| 1 s per mossa (budget 1300 ms - 300 di margine) | profondita' media prima | dopo |
|---|---|---|
| `SimpleEvaluator` | 5,33 | **6,25** |
| MLP | 4,58 | **4,83** |

Lettura onesta:
* Il guadagno e' tanto piu' grande quanto piu' l'evaluator e' economico: con la rete il costo per nodo e' dominato dall'estrazione delle feature e dal forward (la tua stima: 3,6 us MLP, 11 us CNN), che
  la migrazione non tocca (a parte non dover piu' convertire `Pawn[][]`). Il vero effetto sulla forza di gioco con MLP/CNN e' quindi piccolo: ~+0,25 ply a 1 s per l'MLP.
* Sono misure su 1 core, con pochi campioni e senza ripetizioni multiple: ordine di grandezza affidabile, decimali no. A 60 s per mossa e 4 core **non e' stato misurato**.
* **Lazy SMP**: il comportamento multi-thread e' verificato (`LazySmpCheck`), ma non ho potuto misurare lo speedup su piu' core, come nel tuo README. La TT, l'evaluator e i buffer sono per thread o lock-free come prima.
* Non ho misurato l'effetto sulla forza in partite a tempo uguale (`EqualTimeArena`): l'ho solo dedotto dalla profondita' raggiunta.

## 7. Limiti e cose non fatte

* **Trainer, pretrainer, arene (`SelfPlayTrainer`, `CnnSelfPlayTrainer`, `*Pretrainer`, `EqualTimeArena`, ...) restano su `State`/`Action`**: chiamano `findBestMove(State, ...)`, che converte la radice in O(81)
  e la mossa in `Action`. Il loro tempo e' dominato dalla ricerca (50-1500 ms per mossa) e il loro codice e' coperto da test (`SelfPlayRulesCheck`, `CnnTrainingCheck`) che usano oracoli sullo `State`: migrarli
  avrebbe aggiunto rischio senza un guadagno misurabile. Se un giorno servisse (es. traiettorie di TD piu' leggere in memoria: per stima, un `FastState` pesa un centinaio di byte contro ~600 di uno `State`; non misurato), la via e' gia' pronta.
* **`AshtonRules` e `ZobristHasher` non sono stati rimossi**: sono gli oracoli dei test (`FastStateCheck`, `SearchEquivalenceCheck`, `CrossCheck`...) e servono ai trainer. Non vanno "ripuliti".
* **`LegacyNegamax`/`LegacySimpleEvaluator` sono ~600 righe duplicate apposta**: congelate, solo per i test; non vanno migliorate (perderebbero il valore di riferimento). Se un domani si cambia di proposito
  il comportamento della ricerca (es. un nuovo ordinamento delle mosse), `SearchEquivalenceCheck` fallira' per costruzione e la copia congelata va aggiornata *consapevolmente*.
* **Due piccole concessioni ai test dentro la classe di produzione**: il campo `forceTimeUpAfterNodes` (0 = disattivo; controllato solo ogni 2048 nodi) e un'asserzione `assert !timeUp` nel punto di memorizzazione
  in TT (disattivata di default dalla JVM). Entrambi senza costo misurabile.
* **Comportamento gia' presente e non introdotto da me (GHI).** Con TT attiva e in posizioni con *molte* ripetizioni nell'albero, il punteggio puo' differire da quello dell'alpha-beta puro anche senza alcuna interruzione
  (una voce calcolata su un percorso viene riusata su un altro in cui una ripetizione cade altrove: la protezione copre solo i sottoalberi che hanno gia' incontrato una ripetizione). L'ho incontrato scrivendo il test di
  inquinamento da timeout (a profondita' 6 in alcune posizioni con centinaia di ripetizioni; non l'ho quantificato sistematicamente) e ho verificato che la versione storica si comporta identicamente (`SearchEquivalenceCheck`
  confronta i due motori nodo per nodo, ripetizioni comprese). Il test d'inquinamento confronta quindi con la ricerca senza TT solo dove non ci sono ripetizioni. Se vuoi, e' un possibile punto di miglioramento a parte.
* **Una discrepanza di documentazione trovata, non modificata**: il tuo `README_CNN.md` (§4.5 e §8) parla di un margine di sicurezza di 400 ms; nel codice (`Negamax.SAFETY_MARGIN_MILLIS`, prima e dopo la migrazione) e' **300 ms**.
  Per un tempo effettivo di T ms il budget da passare a `EqualTimeArena` e' quindi `T + 300`, non `T + 400` (i confronti simmetrici tra due lati Negamax non ne risentono).
* `MctsPlayer`: porta lo stesso ordine di mosse di prima, ma non ho rifatto le misure dell'arena MCTS (`MctsArena`); `MctsCheck` passa.

## 8. Prossimi passi (suggerimenti, nessuno e' stato fatto)

1. **Misurare sulla macchina a 4 core**: `SearchSpeedBench` (profondita' a 5-10 s) e poi `EqualTimeArena` con MLP/CNN a 4 thread.
2. **Con una rete, il collo di bottiglia e' ora l'estrazione delle feature** (~1,5-2 us per nodo): lo `FastState` rende possibile mantenere alcune feature in modo incrementale dentro `makeMove`
   (conteggi di pezzi per riga/colonna, mappe di raggiungibilita') invece di ricalcolarle. E' un cambiamento a rischio: va accompagnato da un test differenziale come `FastStateCheck`.
3. Generazione delle mosse con bitboard o tabelle per riga/colonna (la tua stima: ~-15% di tempo per nodo con un evaluator costoso; con `Simple`, dove la generazione pesa di piu', potrebbe contare di piu').
4. Migrare i trainer a `FastState` solo se le traiettorie in memoria diventano un problema.
