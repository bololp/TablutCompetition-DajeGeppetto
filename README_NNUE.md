# NNUE per Tablut: valutazione con aggiornamento incrementale

Questo documento descrive la valutazione di stile NNUE aggiunta accanto alla CNN: cosa c'era, cosa e' stato progettato e
realizzato, come si usa, cosa e' stato verificato e misurato, e cosa resta aperto. La CNN **non e' stata rimossa**: resta
selezionabile e invariata.

## 1. Il progetto prima della migrazione (fatti verificati nel codice)

| Aspetto | Com'e' (file e funzioni reali) |
|---|---|
| Linguaggio / build | Java 8 (`--release 8`), classi `*Check` con `main` (niente JUnit nel classpath), `build.xml` Ant |
| Stato di gioco | `engine/FastState`: `byte[81] board` (`EMPTY 0, WHITE 1, BLACK 2, THRONE_MARK 3, KING 4`), `turn`, `kingSquare`, `whiteCount/blackCount`, `winner`, `zobristHash` (+ chiave esatta). Mosse = `int` (`engine/Move`), annullamento con `UndoInfo` (catture: `capturedCount/Square/Piece`, al massimo 4) |
| Regole | `AshtonRules` (motore storico a `State`, oracolo) e `FastState.makeMove/generateMoves`: movimento da torre, trono vuoto non attraversabile ne' occupabile, cittadelle, catture per custodia con trono e cittadelle come muri, cattura del Re con 4/3/2 lati, vittoria bianca col Re sul bordo |
| Make/unmake | `FastState.makeMove(move, UndoInfo)` / `unmakeMove`; `Negamax` non copia lo stato: una posizione di lavoro per worker + stack per ply (`undoStack`, `moveStack`) |
| Valutazione | `Evaluator.evaluate(FastState)`: **stateless**, prospettiva del Bianco, solo posizioni non terminali. Implementazioni: `SimpleEvaluator`, `NeuralEvaluator` (MLP), `CnnEvaluator` |
| Chiamata nella ricerca | `Negamax.negamax(...)` a `depth == 0`, dopo i controlli di terminale; segno invertito con `perspective`. Percorso di una mossa: `exploreChild` = `makeMove` -> `negamax(sp+1)` -> `unmakeMove` |
| CNN | `ConvNet` + `TablutFeatures`: conv 3x3 (4 canali) + 44 feature a mano globali + denso 32 + testa da 8 per ruolo; file `TABLUT_CNN v1`; `tanh * 1000`; ~7-8 us per valutazione |
| Cache / condivisione | `CachingEvaluator` (client: `withEvalCache`, solo MLP e CNN). Lazy SMP: gli helper sono altre istanze di `Negamax` che condividono **solo** TT ed `Evaluator` |
| Selezione del backend | `EvaluatorLoader.load(path)` riconosce il tipo dall'intestazione del file; il client ripiega su `SimpleEvaluator` con un avviso se il caricamento fallisce |
| Training CNN | `CnnPretrainer`, `CnnSelfPlayTrainer`, `CnnDistiller` (Java). Il dataset `dati.gz` **non e' nel repository** |

**Incompatibilita' decisiva.** I pesi della CNN non sono convertibili in una NNUE. L'ibrido somma ai filtri 44 feature a mano
*globali* (vie di fuga, minacce, mobilita') che dipendono da molte celle insieme: non sono funzioni sparse delle singole
(pedina, cella) e non si aggiornano per differenza. Il primo strato conv e' si' lineare e sparso, ma e' seguito da
leaky-ReLU, conv 1x1, flatten e da quelle feature. Quindi: la CNN resta com'e' e l'NNUE e' una rete separata, con formato
proprio, per la quale servono pesi propri (sezione 6).

## 2. Architettura NNUE

* **Feature (ingresso sparso).** 243 = 3 tipi x 81 celle: `f = base(tipo) + cella`, `base(BIANCO)=0, base(NERO)=81, base(RE)=162`.
  Al piu' 25 attive (8+16+1). Il Re ha un blocco proprio di 81 feature.
* **Prospettiva unica e assoluta** (difensori = Bianco, attaccanti = Nero), mai specchiata per il lato al tratto: in Tablut i
  due lati sono asimmetrici, quindi una seconda prospettiva non condividerebbe parametri. **Conseguenza: una mossa del Re e' un
  normale aggiornamento** (togli (RE, da), aggiungi (RE, a)); non serve mai una ricostruzione per il Re.
* **Trono, cittadelle, vie di fuga**: geometria statica, gia' codificata dalla cella di ogni feature; il trono vuoto e'
  l'assenza di (RE, trono). Nessuna feature dedicata. Il **lato al tratto** non entra nell'accumulatore: sceglie la testa (strati 1-2).
* **Rete** (tutto intero):
  ```
  acc[j]  = ftB[j] + somma_{f attiva} ftW[f][j]            j < H=64      (int; si aggiorna per differenza)
  a[j]    = clamp(acc[j], 0, QA)                            QA = 256
  u[i]    = l1B[r][i] + somma_j l1W[r][i][j] * a[j]         i < L2=16, r = lato al tratto
  c[i]    = clamp((u[i] + 32) >> 6, 0, QA)                  QW = 64
  z       = l2B[r] + somma_i l2W[r][i] * c[i]               scala QA*QW
  valore  = tanh(z / 16384)                                 tanh campionata a passo 1/16 con interpolazione lineare (errore < 4e-4)
  ```
  Pesi `short`, bias `int`. 17 730 parametri per H=64, L2=16 (`H` e `L2` sono nel file). Uscita in (-1,1), +1 = vince il Bianco,
  moltiplicata per `outputScale = 1000` (stessa convenzione di CNN e MLP, quindi MCTS, `Negamax` e client non cambiano).
* **Esattezza.** Con aritmetica intera la somma dell'accumulatore e' associativa: lo stato aggiornato per differenze e quello
  ricostruito da zero sono **identici bit per bit**. Il costruttore rifiuta pesi che potrebbero andare in overflow.

### Aggiornamento incrementale (`NnueSession`)
Stato **per worker**, mai condiviso: una pila di accumulatori indicizzata dallo stesso ply degli stack di `Negamax`.
* `reset(radice)`: ricostruzione completa (frame 0).
* `push(ply, mossa, dopo, undo)` dopo ogni `makeMove`: registra solo la differenza (pedina mossa, da, a, catture dall'`UndoInfo`).
  Mossa: `acc' = acc - W[(tipo,da)] + W[(tipo,a)]`; ogni cattura (anche multipla, Re compreso): `- W[(tipo catturato, cella)]`.
* **Lazy**: l'accumulatore di un frame si calcola solo alla prima `evaluate`, dal piu' vicino antenato calcolato; i nodi mai
  valutati (terminali, ripetizioni, tagli TT) non pagano nulla.
* **Unmake: nessuna azione.** Il frame del genitore non e' mai toccato dai figli; un fratello riscrive il frame del figlio con un nuovo `push`.
* **Ricostruzione di ripiego.** Ogni frame porta la firma dell'insieme di feature attive (XOR delle chiavi Zobrist). Prima di
  valutare si confronta con la firma derivata dall'hash incrementale di `FastState` (codice indipendente): se differiscono
  (push mancato, radice non installata, differenza inutilizzabile) il frame e' ricostruito dalla scacchiera e l'evento e'
  contato (`fallbackRebuilds()`). Il **valore resta corretto**; in una ricerca che rispetta il protocollo il contatore vale 0 (e i test lo esigono).
* Una sessione per istanza di `Negamax` (helper Lazy SMP compresi). L'evaluator NNUE e' immutabile e condiviso.

## 3. File

Nuovi (`Tablut/src/it/unibo/ai/didattica/competition/tablut/...`):

| File | Ruolo |
|---|---|
| `aiplayer/IncrementalEvaluator.java` | Contratto: `Evaluator` + `newSession()`; `Session` = `reset / push / evaluate / fallbackRebuilds` |
| `aiplayer/NnueNetwork.java` | Pesi immutabili, aritmetica intera, indice delle feature, firma, file `TABLUT_NNUE v1` con validazione completa |
| `aiplayer/NnueSession.java` | Pila di accumulatori per ply, aggiornamento lazy, controllo di coerenza, ricostruzione di ripiego |
| `aiplayer/NnueEvaluator.java` | `Evaluator` (valutazione completa, thread-safe) + fabbrica di sessioni |
| `training/NnueDistiller.java` | Distillazione di un valutatore (CNN/MLP) in una NNUE: dati, addestramento in `double`, quantizzazione, metriche |
| `aiplayer/NnueCheck.java` | Test unitari: oracolo indipendente, ogni mossa, passeggiate, DFS con fratelli, terminali, disallineamenti, file, thread |
| `aiplayer/NnueSearchCheck.java` | Integrazione: incrementale = completa = motore storico (mossa, punteggio, profondita', nodi), Lazy SMP, interruzioni |
| `training/NnueTrainingCheck.java` | Gradiente vs differenze finite, quantizzazione, distillazione di un maestro noto, determinismo, `main` |
| `client/NnueClientCheck.java` | Selezione `eval=` (logica e `main` reale in un sottoprocesso) |
| `aiplayer/NnueBench.java` | Costo per valutazione e ricerca: CNN vs NNUE (completa e incrementale) |
| `tools/mutation_nnue.py`, `tools/run_nnue_checks.sh` | Mutation testing (97 mutanti) e script "compila + tutti i test NNUE" |
| `pesi_nnue/nnue_h64_l16_distilled_cnn_t1500000_v4.{nnue,log}` | Pesi distillati dalla CNN e log dell'addestramento |

Modificati:

| File | Modifica |
|---|---|
| `aiplayer/Negamax.java` (+54/-1) | Un campo `inc` (sessione, `null` se l'evaluator non e' incrementale), creato per istanza (helper compresi); `reset` in `searchRoot` e `installForTest`, `push` in `exploreChild` dopo `makeMove`, `evaluate(sp, s)` a `depth == 0`; 3 accessor di test. Nessuna altra logica di ricerca cambia: con evaluator non incrementali il comportamento e' identico |
| `aiplayer/EvaluatorLoader.java` | Riconosce `TABLUT_NNUE` (prima di CNN/MLP); `describe` |
| `client/TablutNegamaxClient.java` | Opzione `eval=auto\|simple\|mlp\|cnn\|nnue` e `selectEvaluator` (sezione 4) |
| `aiplayer/NoAllocationCheck.java` | Aggiunta la rete NNUE all'invariante "zero allocazioni per nodo" |
| `aiplayer/CacheCompareBench.java` | **Non legata alla migrazione**: usava la vecchia API `findBestMove(.., long[], int)` e **impediva la compilazione dell'intero progetto**; portata a `PositionHistory` (+5/-9 righe) |

## 4. Uso

```sh
cd Tablut
sh tools/run_nnue_checks.sh                    # compila (javac --release 8) ed esegue tutti i test NNUE (~40 s)

# giocare con l'NNUE (Negamax, aggiornamento incrementale):
java -cp "bin:lib/*" it.unibo.ai.didattica.competition.tablut.client.TablutNegamaxClient WHITE 60 localhost \
     eval=nnue pesi_nnue/nnue_h64_l16_distilled_cnn_t1500000_v4.nnue threads=1
# giocare con la CNN (invariato):
java -cp "bin:lib/*" ...TablutNegamaxClient WHITE 60 localhost eval=cnn pesi_v4/cnn_t1500000_v4.txt
```
`eval=` (default `auto`): `auto` riconosce il tipo dall'intestazione del file e, **come prima**, ripiega su `SimpleEvaluator` con
un avviso se manca o non si carica; `simple` ignora i pesi; `mlp`, `cnn`, `nnue` sono **espliciti e severi**: pesi mancanti, file
non valido o di un altro tipo fermano l'avvio con un errore che nomina il backend (mai un valutatore diverso da quello chiesto in silenzio).
Un `eval=` sconosciuto e' un errore, non si ignora. L'NNUE non e' mai avvolto nella cache delle valutazioni; MCTS (`algo=mcts`) la usa
con la valutazione completa (senza percorso incrementale; verificato in `NnueSearchCheck`). Arene e bench (`EqualTimeArena`, `SearchSpeedBench`) la accettano senza modifiche.

```sh
# addestrare (distillare) una NNUE da un valutatore; deterministico: stesso comando = stesso file
java -cp "bin:lib/*" it.unibo.ai.didattica.competition.tablut.training.NnueDistiller \
     teacher=pesi_v4/cnn_t1500000_v4.txt out=mia.nnue hidden=64 l2=16 games=3000 epochs=30 seed=1
# benchmark
java -cp "bin:lib/*" it.unibo.ai.didattica.competition.tablut.aiplayer.NnueBench pesi_v4/cnn_t1500000_v4.txt pesi_nnue/nnue_h64_l16_distilled_cnn_t1500000_v4.nnue
java -cp "bin:lib/*" it.unibo.ai.didattica.competition.tablut.training.EqualTimeArena m=0:pesi_v4/cnn_t1500000_v4.txt m=0:pesi_nnue/nnue_h64_l16_distilled_cnn_t1500000_v4.nnue 15 200 1 1
# mutation testing (BUILD = cartella delle classi compilate)
BUILD=/tmp/tablut_nnue_build python3 tools/mutation_nnue.py all
```

## 5. Formato dei pesi (`TABLUT_NNUE v1`, testo)
```
TABLUT_NNUE v1
features psq3x81        <- insieme di feature: un file con un altro insieme e' rifiutato
hidden 64               <- 1..1024
l2 16                   <- 1..256
qa 256                  <- scale di quantizzazione: devono coincidere con quelle del codice
qw 64
params 17730            <- deve coincidere con 243*H + H + 2*(L2*H + L2 + L2 + 1)
<17730 interi, 16 per riga>: ftW[243*H] (feature per feature), ftB[H], poi per il lato 0 e il lato 1:
                             l1W[L2*H] (uscita per uscita), l1B[L2], l2W[L2], l2B[1]
checksum <CRC32>        <- dei valori come interi big-endian
```
`NnueNetwork.load` lancia `IOException` per: file assente/illeggibile, intestazione o versione diverse, feature/scale/dimensioni
incoerenti, valori mancanti, in eccesso, non numerici o fuori da `short`, checksum errato, contenuto dopo il checksum, pesi che
potrebbero andare in overflow. Cambiare feature, scale o struttura richiede una nuova versione del formato.

## 6. Pipeline di training: stato

* **Conversione dei pesi CNN -> NNUE: non esiste e non e' possibile** (sezione 1).
* **Realizzato**: distillazione `NnueDistiller`. Campiona posizioni da partite miste casuali/avide (con le 8 simmetrie del quadrato:
  le regole sono invarianti, l'NNUE no), le etichetta con il valutatore "maestro", addestra l'NNUE (Adam, gradiente a mano verificato con
  differenze finite), la quantizza e misura la rete **quantizzata** su un held-out separato per posizione di partenza.
  Cio' che impara e' quanto del maestro si esprime con (tipo, cella), con il suo stesso errore: **non** e' un allenamento sui risultati di gioco.
* **Non realizzato (serve per una NNUE migliore del maestro)**: addestramento da dati di gioco veri (esiti di partite, valori di ricerca profonda).
  Servono: un lettore che porti i campioni nel formato `NnueDistiller.Data` (scacchiera `byte[81]` con i codici di `BoardConstants`,
  lato al tratto, bersaglio in [-1,1]) — attenzione: `DistillSet` codifica il Re come **3**, `FastState` come **4**; piu' dati e,
  probabilmente, una rete piu' larga (`hidden`, `l2`). `train` e `quantize` si riusano cosi' come sono.

## 7. Risultati misurati

Ambiente: sandbox con **1 core**, JDK 21 (compilato con `--release 8`), un thread di ricerca. Le misure variano di ~10-15 % da una
esecuzione all'altra: sono riportati gli intervalli di tre esecuzioni.

**Test** (tutti verdi sull'albero finale): `NnueCheck` 106 174 verifiche; `NnueSearchCheck` 3 410 verifiche (330 ricerche confrontate,
132 anche con il motore storico, 3 383 185 nodi, 5 139 438 passi incrementali, zero ricostruzioni di ripiego, zero errori negli helper Lazy SMP,
MCTS con l'NNUE a 1 e 2 thread);
`NnueTrainingCheck` 769; `NnueClientCheck` 50; `NoAllocationCheck`: NNUE **0,004 B/nodo** su 499 625 nodi. **Mutation testing: 97/97 mutanti rilevati**
(rete e aritmetica 64, integrazione con `Negamax` 8, client 7, distillatore 18); mutanti non rilevati al primo giro sono stati risolti
rafforzando i test o riconosciuti equivalenti (Adam e' invariante alla scala del gradiente). Test esistenti rieseguiti sull'albero finale (quelli vicini alle zone
toccate): `NoAllocation`, `ClientDefaults`, `NegamaxClientArgs`, `ClientHistoryIntegration`, `CachingEvaluator`, `RepetitionAndStalemate`,
`ExactKeyStack`, `HistoryExactness`, `PVS`, `SearchOptimizations`, `OrderingExt`, `TwoSlotTT`, `LazySmp`, `MctsSearch`, `CnnEvaluator`,
`TTCorrectness`, `ZobristIncremental`, `FastState` (19,7 M verifiche): tutti verdi. `SearchEquivalenceCheck` e `MctsConcurrencyCheck` falliscono
**allo stesso modo sul codice originale** (sezione 8). Gli altri test del progetto non sono stati rieseguiti.

**Costo per valutazione** (`NnueBench`: per ognuna di 300 posizioni e per ognuna delle 16 732 mosse, `make -> valuta -> unmake`; netto del make/unmake):

| Valutatore | per valutazione | rispetto alla CNN |
|---|---|---|
| CNN (nuda, senza cache) | 7,4 - 8,1 us | 1x |
| NNUE, valutazione completa | 0,95 - 1,03 us | 7,8 - 7,9x piu' veloce |
| NNUE, incrementale (`push` + `evaluate`) | **0,41 - 0,43 us** | **18,1 - 19,0x** piu' veloce |

La somma dei valori NNUE completa e incrementale coincide bit per bit. Componenti (una posizione): aggiornamento incrementale ~45 ns,
strati densi ~215 ns, ricostruzione completa ~610-640 ns (un'esecuzione ha dato 3205 ns: rumore del micro-benchmark). Cioe' dopo l'aggiornamento
incrementale il tempo e' dominato dagli strati densi (legge di Amdahl); una prima versione del calcolo "per uscita" li pagava ~520 ns,
ridotti a ~215 ns trasponendo lo strato 1 e saltando i neuroni a zero (stesso risultato intero, stessi test).

**Ricerca** (`Negamax`, TT 2^20, 16 posizioni, stessi parametri di `SearchSpeedBench`; i nodi differiscono fra valutatori):

| | nodi/s a profondita' 4 | profondita' media a 500 ms |
|---|---|---|
| CNN nuda | 156 - 160 k | 4,06 |
| CNN + cache valutazioni del client (2^22, fredda a ogni passata; hit rate 19,8 %) | 183 - 198 k | 4,06 |
| NNUE, solo valutazione completa | 862 - 901 k | 4,81 |
| **NNUE incrementale** | **1,35 - 1,41 M** | **5,13** |

NNUE incrementale / CNN con cache: 7,1 - 7,7x in nodi/s; incrementale / completa: 1,55 - 1,64x.

**Forza di gioco** (`EqualTimeArena`, 15 coppie con colori invertiti = 30 partite, 200 ms reali/mossa, 1 thread, CNN senza cache):
CNN **+15 =1 -14** contro NNUE, punteggio CNN 51,7 % [IC95 34,6 % - 68,3 %]; profondita' media CNN 4,10 contro NNUE 4,79; 180 k contro 1,84 M nodi/s.
**Nessuna differenza di forza distinguibile**: non si puo' dichiarare ne' un miglioramento ne' un peggioramento.

**Fedelta' al maestro** (rete quantizzata, held-out di 18 296 campioni): MSE 0,0347, correlazione 0,930, segno concorde nel 96,6 % dei campioni con |valore| > 0,1
(MSE di addestramento 0,019: c'e' overfit, piu' dati lo ridurrebbero). Il file `pesi_nnue/...` si rigenera bit per bit col comando della sezione 4.

## 8. Limiti, decisioni aperte, problemi preesistenti

* **La rete distribuita e' una copia della CNN**, non un allenamento sui dati di gioco: la sua qualita' e' limitata da quella del maestro e da (tipo, cella).
  Il guadagno misurato e' di **velocita'** (valutazione ~18x, nodi/s ~7-9x, +0,7-1,1 ply a tempo uguale), non di forza.
* L'arena e' piccola (30 partite, IC95 ~ +-17 punti), a 200 ms/mossa (profondita' ~4-5, molto sotto quelle di gara) e senza la cache della CNN
  (che in produzione le da' ~+16 % di nodi/s): conclusioni sulla forza in gara richiedono molte piu' partite al tempo reale.
* Le feature a mano della CNN (vie di fuga, minacce) non esistono per l'NNUE; una variante con il Re come "bucket" (stile HalfKP)
  catturerebbe la geometria del Re ma richiederebbe una ricostruzione a ogni mossa del Re e molti piu' dati: **non implementata**
  (la sessione ha gia' il percorso di ricostruzione, ma il progetto la renderebbe frequente).
* Solo `Negamax` usa il percorso incrementale; MCTS, trainer e arene usano la valutazione completa (corretta, piu' lenta).
* Java puro e scalare (compatibile con Java 8): niente Vector API / int8; margine di ulteriore velocita' sugli strati densi.
* `build.xml` (Ant) non e' stato eseguito (Ant non e' installato nel sandbox): i test sono compilati con `javac --release 8`.
* **Preesistenti, non legati alla migrazione, identici sul codice originale:** (1) `CacheCompareBench` non compilava (corretto, vedi sopra);
  (2) `SearchEquivalenceCheck` fallisce la sola soglia di copertura "ricerche interrotte 53/80" (`53 > 80*2/3` con divisione intera = `53 > 53`);
  (3) `MctsConcurrencyCheck` fallisce perche' cerca pesi in `pretrained_weights/` che sono in versione di feature precedente (v3) rispetto al codice (v4).
