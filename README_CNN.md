# README_CNN — dal MLP al CNN (e valutazione dell'MCTS)

Questo documento accompagna i file nuovi/modificati rispetto a `README_MLP_REDESIGN.md`. Ogni numero e' **misurato** in un sandbox
a **1 core** (non nella VM di gara a 4 core): dove un'affermazione e' un ragionamento e non una misura, e' scritto.

## 1. In breve

* Il CNN e' implementato, testato (oracolo indipendente, differenze finite, mutation test), allenabile e provato dal vivo contro il
  server reale (come Bianco, come Nero, e col fallback su `SimpleEvaluator` per pesi incompatibili).
* **Non ho evidenza che batta l'MLP.** Prevede l'esito delle partite un po' meglio (errore 0,636 contro 0,646), ma costa ~3 volte per
  valutazione e **a tempo uguale perde** (~36% su 80 partite). A profondita' uguale e' alla pari. Il verdetto a 60 s/mossa su 4 core
  resta da misurare sulla tua macchina (si veda §5).
* **Non ho evidenza che un MCTS convenga.** A tempo realmente uguale e con lo stesso evaluator, un MCTS sperimentale non tarato e'
  indistinguibile da Negamax (~53% su 48 partite). Un primo risultato dell'80% era un **artefatto del budget** (§7).

## 2. Cambiamento di architettura

| | MLP (prima) | CNN (ora, default) |
|---|---|---|
| Ingresso | 41 feature scalari a mano (`TablutFeatures`) | scacchiera 9x9 come 3 piani one-hot (Bianco, Nero, Re) **+ le stesse 41 feature** (ibrido; `features=0` = CNN puro) |
| Struttura spaziale | nessuna: la rete vede numeri, non la scacchiera | conv 3x3 con pesi condivisi su tutta la scacchiera |
| Strati | 41 → 24 → 8 → 1 | conv 3x3 (4 canali) → conv 1x1 (2 canali) → flatten (162) ++ 41 feature → denso 32 → head 8 → 1 |
| Ruolo | tronco condiviso + 2 head (chi muove) | identico: tronco condiviso + 2 head |
| Uscita | tanh, in (-1,1), `outputScale` 1000 | identica (stesso `Evaluator`: `Negamax` e client non cambiano) |
| Parametri / MAC | ~1.400 / ~1.200 | ~7.500 / ~8.400 |
| Costo per valutazione | ~3,6 us | ~11 us (~3x) |

Scelte di progetto, tutte motivate da misure:

* **Il vincolo e' la CPU, non la RAM.** Il CNN pesa pochi KB, ma in Java puro HotSpot non vettorizza i loop corti (97 celle per piano):
  il kernel migliore trovato fa ~0,4 ns per moltiplicazione-addizione, quindi una conv 3x3 16→16 costerebbe ~75 us. Per questo la rete
  e' piccola, e un secondo livello conv (3,6x il costo) non migliora la previsione (0,642 contro 0,634).
* **Primo livello sparso**: le pedine sono ~25 su 81 celle, quindi si "spalma" il filtro di ogni pedina invece di convolvere 3 piani densi.
  Cittadelle, trono e vie di fuga non sono piani costanti ma una **mappa di bias per posizione** appresa (81 parametri per canale):
  costa zero e include come caso particolare qualunque convoluzione su piani statici.
* **Kernel**: conv 3x3 a 9 tap fusi, 2 canali per volta, somme ad albero; il backward-input riusa lo stesso kernel con pesi trasposti (meno
  codice da verificare); strati densi a blocchi di 4 uscite (-25% sul costo totale).
* **Augmentation con le 8 simmetrie della scacchiera.** Un CNN e' equivariante per traslazione, non per rotazione/riflessione. Con
  l'augmentation l'errore held-out passa da 0,697 a 0,634: e' l'ingrediente piu' importante dopo le feature a mano. La validita' e' verificata:
  le regole Ashton sono equivarianti per tutte e 8 (420 coppie posizione/simmetria, 22.000 mosse, 0 discordanze).
* **Ibrido.** Le vie di fuga attraversano tutta la scacchiera (un 3x3 le vede a pezzi): le feature a mano le danno esatte, il CNN impara il
  residuo. Il CNN puro e' peggiore dell'MLP con feature (0,655 contro 0,646) ma ci sta vicino.
* I pesi CNN hanno intestazione `TABLUT_CNN v1`: client e arena riconoscono da soli MLP o CNN (`EvaluatorLoader`).

## 3. File

**Nuovi** (package `it.unibo.ai.didattica.competition.tablut`):

| File | Ruolo |
|---|---|
| `aiplayer/ConvNet.java` | la rete: forward, backward, salvataggio, copia; `Workspace` per thread |
| `aiplayer/BoardSymmetry.java` | le 8 simmetrie D4 |
| `aiplayer/CnnEvaluator.java` | `Evaluator` per `Negamax` (thread-safe, ibrido o puro) |
| `aiplayer/EvaluatorLoader.java` | carica MLP o CNN dall'intestazione del file |
| `training/Adam.java`, `training/CnnPretrainer.java` | dataset + addestramento supervisionato (warm start) |
| `training/CnnSelfPlayTrainer.java` | TD(lambda) self-play, stesso schema di `SelfPlayTrainer` |
| `experiments/CnnBench.java`, `experiments/CnnVsMlpExp.java` | costo per valutazione; confronto MLP/CNN a parita' di dati |
| `experiments/MctsPlayer.java`, `MctsArena.java`, `MctsCheck.java` | MCTS sperimentale, arena vs Negamax, test |
| `selftest/ConvNetCheck.java`, `selftest/CnnEvaluatorCheck.java`, `training/CnnTrainingCheck.java` | test |
| `tools/mutation_cnn.py` | mutation test dei test (dalla cartella `Tablut`) |
| `pretrained_weights/cnn_conv4_hybrid_pretrained.txt`, `mlp_v3_pretrained.txt` | pesi di partenza/baseline, **feature v3** (si veda §8) |

**Modificati** (tre file tuoi, diff minimo):

* `client/TablutNegamaxClient.java` e `training/EqualTimeArena.java`: caricano i pesi con `EvaluatorLoader` (MLP o CNN). `EqualTimeArena` ha in piu'
  un argomento finale opzionale `profonditaFissa` e una nota sul tempo effettivo (§8).
* `training/SelfPlayTrainer.java`: `evaluateAgainstBaseline/Opponent` accettano `Evaluator` invece di `NeuralEvaluator` (compatibile all'indietro).

## 4. Come si usa

Dalla cartella `Tablut` (dopo aver copiato i file nel progetto):

```bash
javac -encoding iso-8859-1 -d build -cp "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar" $(find src -name "*.java")
CP="build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; T=it.unibo.ai.didattica.competition.tablut
```

**1. Costo sulla tua macchina** (decide quanto puo' essere grande la rete):

```bash
java -cp $CP $T.experiments.CnnBench 2                                   # tabella di forme, con MLP e Simple come riferimento
java -cp $CP $T.experiments.CnnBench 2 conv=8,8 reduce=2 dense=32 head=8 # una forma sola
```

**2. Dataset + pre-addestramento supervisionato** (il warm start; da pesi casuali il TD e' troppo lento):

```bash
java -cp $CP $T.training.CnnPretrainer games=7000 epochs=10 dataOut=dati.gz out=cnn_pre.txt
java -cp $CP $T.training.CnnPretrainer data=dati.gz epochs=10 conv=8 out=cnn_pre8.txt   # altra forma, stesso dataset
```

Misurato (1 core): 5.000 partite → 165.000 posizioni in 116 s; 8 epoche di addestramento in ~20-30 s. Su 4 core dovrebbe essere ~3 volte piu' veloce (non misurato).
Opzioni: `conv`, `reduce`, `dense`, `head`, `features=0` (CNN puro), `augment=0`, `lr`, `batch`, `threads`.

**3. Confronto a parita' di dati con l'MLP** (errore di previsione sul 8% di partite tenute da parte):

```bash
java -cp $CP $T.experiments.CnnVsMlpExp dati.gz epochs=8 threads=4 "mlp:save=mlp.txt" "cnn:conv=4;save=cnn4.txt" "cnn:conv=4;feat=0" "cnn:conv=4;aug=0"
```

**4. TD(lambda) self-play** (parti sempre da `resumeFrom`):

```bash
java -cp $CP $T.training.CnnSelfPlayTrainer resumeFrom=cnn_pre.txt episodes=3000 minMoveMs=50 maxMoveMs=300 out=cnn_td.txt
```

Ogni episodio costa circa (numero di mosse) x (budget medio per mossa): con 100-1500 ms (default) sono ~80 s per partita per core, cioe' ~180 partite
all'ora su 4 core; con 50-300 ms circa 850 all'ora (**stime per calcolo**, non misurate su 4 core). `alpha` di default e' 0,002 (0,02 per l'MLP): vedi §5.
Il log (`training_log_cnn.csv`) e i checkpoint (`checkpoints_cnn/`) hanno lo stesso formato del trainer MLP.

**5. Il confronto che decide: a tempo uguale, sul tempo vero.** Ricorda il margine di 400 ms (§8): per 5 s effettivi si passa 5400.

```bash
java -cp $CP $T.training.EqualTimeArena cnn_td.txt mlp.txt 30 5400 1 4         # tempo uguale, 4 thread per lato
java -cp $CP $T.training.EqualTimeArena cnn_td.txt mlp.txt 30 100 1 1 4         # profondita' FISSA 4: qualita' pura dell'evaluator
```

**6. Client di gara**: il quarto argomento puo' essere un file MLP o CNN; se illeggibile/incompatibile ripiega su `SimpleEvaluator`.

```bash
java -cp $CP $T.client.TablutNegamaxClient WHITE 60 localhost cnn_td.txt
```

**Test** (tutti stampano `... OK` o escono con codice 1):

```bash
java -cp $CP $T.selftest.ConvNetCheck           # 47 verifiche
java -cp $CP $T.selftest.CnnEvaluatorCheck      # 17 verifiche
java -cp $CP $T.training.CnnTrainingCheck       # 48 verifiche
java -cp $CP $T.experiments.MctsCheck           # 11 verifiche
python3 tools/mutation_cnn.py model             # inietta bug in ConvNet: il test deve rilevarli tutti
python3 tools/mutation_cnn.py training          # idem per trainer/pretrainer/Adam (~40 s per mutante)
```

## 5. Misure

Sandbox a 1 core, un thread di ricerca per lato. Gli intervalli sono IC95 di Wilson; con 24-40 partite sono **molto larghi**.

**Errore di previsione dell'esito** (165.000 posizioni di 5.000 partite di giocatori deboli, 8 epoche, test = 8% delle partite; il predittore
costante fa 0,925). Piu' basso e' meglio:

| Modello | Parametri | Errore held-out |
|---|---|---|
| MLP 41→24→8 (3 seed) | 1.426 | 0,6431 / 0,6465 / 0,6468 |
| CNN ibrido conv=8 (3 seed) | 7.956 | 0,6335 / 0,6382 / 0,6353 |
| CNN ibrido conv=4 (**default**, 1 seed) | 7.516 | 0,6360 |
| CNN ibrido conv=4, reduce=1, dense=16 (1 seed) | 2.695 | 0,6427 |
| CNN ibrido conv=8,8 (1 seed) | 8.540 | 0,6422 |
| CNN **puro** conv=8 (3 seed) | 6.644 | 0,6525 / 0,6565 / 0,6564 |
| CNN ibrido conv=8 **senza augmentation** (1 seed) | 7.956 | 0,6966 (train 0,43: overfitting) |

L'ibrido batte l'MLP in tutti e 3 i seed (differenze appaiate -0,010, -0,008, -0,012): reale, ma piccolo. Il rumore dichiarato dal README_MLP e' ~±0,003.

**Costo** (`CnnBench`, valutazione completa, feature comprese): Simple 0,30 us; MLP 3,6 us; CNN ibrido conv=4 11,1 us; conv=4/reduce=1/dense=16 8,6 us.

**Partite a tempo uguale** (budget 100 ms, che per il margine di Negamax sono **50 ms effettivi** per lato; stesso protocollo per tutti):

| A contro B | Risultato di A | Profondita' media A / B |
|---|---|---|
| CNN conv=8 vs MLP | +14 =1 -25 → 36,3% [23-52] | 2,8 / 3,7 |
| CNN conv=4 vs MLP | +13 =2 -25 → 35,0% [22-51] | 3,0 / 3,8 |
| MLP vs Simple | 85,0% [71-93] (40 partite) | 3,4 / 4,0 |
| CNN conv=8 vs Simple | 75,0% [55-88] (24 partite) | 2,7 / 3,9 |

**Partite a profondita' fissa uguale** (CNN conv=8 vs MLP, 40 partite): profondita' 3 → 67,5% [52-80]; profondita' 4 → 43,8% [30-59]. Quindi a profondita'
uguale non c'e' vantaggio distinguibile, e il costo per nodo non e' compensato.

**TD self-play, solo un controllo di stabilita'** (30 episodi da `cnn_pre`, budget 20-50 ms): l'errore di previsione sull'esito passa da 0,636 a 0,671
con `alpha=0.002` e a 0,859 con `alpha=0.02`. Questo giustifica il default prudente, **ma non dimostra che il TD migliori il CNN**: quell'errore misura
l'esito di partite deboli, non l'obiettivo del TD. Serve l'arena dopo un training lungo; potrebbe servire un `alpha` ancora piu' basso.

Limiti di tutte queste misure: 1 core; 50 ms di ricerca effettiva contro 60 s di gara; n piccolo; un solo dataset. Estrapolare a 60 s e' un'ipotesi:
a piu' tempo la profondita' persa pesa meno, ma a profondita' uguale il CNN non guadagna.

## 6. Verifiche eseguite

* `ConvNetCheck`: forward contro un **oracolo ingenuo in double** (cicli espliciti, nessun padding/trucco); gradiente contro differenze finite su **ogni** parametro
  (23.000+ coppie parametro/posizione, 0 scarti oltre gli angoli della leaky-ReLU); layout dei parametri; ruoli (il gradiente non tocca l'altra head);
  nessuno stato residuo nel workspace; gruppo delle simmetrie; equivarianza delle **regole**; salvataggio al bit e rifiuto di file corrotti; 4 thread.
* `CnnTrainingCheck`: Adam contro Adam ingenuo; gradiente di minibatch a 1 e 4 thread contro somma ingenua; dataset (round-trip, rifiuti, coerenza con le feature
  ricalcolate, **segno dell'esito contro lo stato finale vero**); **TD(lambda) contro un riferimento indipendente** (errore ~1e-7 per 3 lambda x 3 esiti); coerenza delle
  traiettorie di self-play; un mini-training reale end-to-end.
* `CnnEvaluatorCheck`: la catena `State → punteggio` contro un calcolo indipendente; ricerca con TT == senza TT; Lazy SMP a 4 thread; loader.
* **Mutation test**: 27/27 mutanti rilevati sul modello e 19/19 sul training. Hanno trovato **due buchi reali dei miei test**, chiusi: (1) un esito con segno invertito
  nel dataset (la rete impara anche etichette sbagliate: il test di apprendimento non lo vede); (2) il segno del valore propagato dalle foglie nell'MCTS.
  Un mutante e' risultato equivalente (`Arrays.fill` ridondante nel livello 0), verificato su 3 seed.
* **Dal vivo col server reale**: client CNN come Bianco (vince in 15 mosse), come Nero (vince in 22), e con pesi MLP v2 incompatibili (messaggio chiaro, ripiego su
  `SimpleEvaluator`, nessun crash). Contro `TablutRandomClient`, che e' lento: con 1 s di limite il server gli da' timeout, usare 3 s.
* Suite preesistente rilanciata dopo le mie patch: CrossCheck (0 mismatch), MLPGradientCheck, FeaturesCheck, WasCaptureCheck, Zobrist x2, TTCorrectness,
  LazySmp, PVS, RepetitionAndStalemate, SelfPlayRules: tutti OK. `SearchOptimizationsCheck` e' lungo: interrotto a 250 s dal mio wrapper, 0 divergenze fino a li'.
  `experiments.LaneCheck` non rilanciato. Tutto compila con `javac --release 8`.

## 7. MCTS: cosa ho misurato

`MctsPlayer`: PUCT **senza rollout** (le partite casuali a Tablut hanno quasi zero informazione), foglie valutate dallo **stesso evaluator** di Negamax; a ogni espansione si
valutano tutti i figli (come un'alpha-beta alla frontiera) e il loro valore statico e' Q iniziale e prior (softmax). Niente policy net, niente ripetizioni, un thread,
iperparametri poco tarati: e' un **confronto di algoritmi a parita' di evaluator**, non un MCTS "di gara".

| Configurazione (MLP v3, 100 ms reali per lato) | MCTS vs Negamax |
|---|---|
| cPuct 1,0 / tau 0,2 (24 partite) | +11 =3 -10 → 52,1% [33-70] |
| cPuct 0,5 / tau 0,1 (12 partite) | +7 =1 -4 → 62,5% [35-84] |
| cPuct 2,0 / tau 0,3 (12 partite) | +5 =1 -6 → 45,8% [22-71] |

Throughput comparabile: ~238.000 valutazioni/s (MCTS) contro ~235.000 nodi/s (Negamax). L'MCTS raggiunge ~930 simulazioni per mossa, profondita' massima ~7,7.

**L'errore che ho fatto e corretto.** La prima versione dava all'MCTS 80,4% (22-1-5). Era un artefatto: `Negamax.findBestMove` sottrae 400 ms di margine e ha un minimo di 50 ms,
quindi con budget 100 la ricerca reale era di 50 ms (misurati: 51,5 ms contro 101,6 ms dell'MCTS). Compensando (`ms + 400`) l'MCTS torna al 52%. Se non avessi misurato il tempo realmente
usato, avrei concluso il contrario di cio' che e' vero.

**Conclusione (con i limiti).** A questo tempo, con questa implementazione e con un evaluator da MLP, nessuna differenza distinguibile (48 partite, ~53%, IC larghissimo). Non c'e'
motivo misurato per sostituire Negamax (verificato, con TT, PVS, Lazy SMP); un MCTS migliore (policy appresa, tuning) potrebbe cambiare il quadro, ma l'addestramento di una policy su CPU
e' fuori scala per un corso e non l'ho provato. Per ripetere il test su tempi veri:
`java -cp $CP $T.experiments.MctsArena mlp.txt 20 5000 1 1.0 0.2` (il margine di Negamax e' gia' compensato da `MctsArena`; nota che a tempi lunghi la memoria dell'albero cresce).

## 8. Avvertenze

* **`EqualTimeArena`: i millisecondi non sono quelli effettivi.** Il budget meno 400 ms (minimo 50 ms) e' il tempo di ricerca reale. Con `100` sono 50 ms. Per T ms effettivi: `T+400`.
  Il confronto fra due lati Negamax e' comunque simmetrico; i miei confronti CNN/MLP sono a 50 ms effettivi.
* **`SelfPlayTrainer.evaluateAgainstOpponent` e `evaluateAgainstBaseline` non hanno aperture casuali**: con ricerca deterministica le "N partite" sono due partite (una per colore)
  ripetute N/2 volte. Il diagnostico periodico dei trainer non ha valore statistico: per decidere usare `EqualTimeArena`.
* **Tutti i pesi del tuo progetto sono feature v2**; il codice e' v3 e li rifiuta (`trained_weights*.txt`, `pretrained_weights/pretrained_default.txt`): i 350.000 episodi non sono riusabili
  cosi' come sono. Un convertitore v2→v3 e' plausibile (la feature inserita e' linearmente dipendente: azzerandone la colonna di pesi la funzione dovrebbe restare identica) ma **non l'ho
  scritto ne' verificato**. In `pretrained_weights/` trovi due file v3 che ho addestrato io (165.000 posizioni, 8 epoche: solo baseline/smoke test, non forti).
* Il `README_MLP_REDESIGN.md` dentro lo zip che mi hai dato e' piu' vecchio di quello caricato a parte; i sorgenti sono invece allineati alla v3.
* I processi in background del sandbox non sopravvivono sempre: le misure lunghe le ho fatte in primo piano.

## 9. Prossimi passi consigliati

1. Sulla tua macchina: `CnnBench`, poi `CnnPretrainer` su ~10.000 partite, poi `EqualTimeArena` CNN contro `mlp.txt` a **4 thread e 5-10 s effettivi**: e' la misura mancante.
2. Solo se quel confronto non e' sfavorevole, lasciare girare `CnnSelfPlayTrainer` per ore, ripetendo l'arena su checkpoint successivi (e provando `alpha` piu' basso).
3. In caso contrario il CNN resta un esperimento e il client di gara continua a usare l'MLP; l'infrastruttura (dataset, augmentation, loader, test) e' riutilizzabile.
