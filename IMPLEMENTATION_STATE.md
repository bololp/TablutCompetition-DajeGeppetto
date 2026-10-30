# IMPLEMENTATION_STATE — fine-tuning verso il valore di ricerca

> Checkpoint di lavoro. In caso di divergenza fanno fede repository e test. Documentazione d'uso: `README_DISTILL.md`.

## Obiettivo e requisiti essenziali
Avvicinare V(s) (statico, CNN) al valore di `Negamax(ev, D)` con la stessa rete (distillazione della ricerca / TD-leaf), senza costi a tempo di gioco.
Richiesti: `SearchLabeler`, `CnnDistiller`, `init=` in `CnnPretrainer`, logit `z` esposto da `ConvNet`, loss logit `(z-atanh(y))^2` (+ ablation tanh), loss mista
(lambda~0.2), anti-deriva, 3-5 iterazioni a D fisso, augmentation D4, porte di accettazione (i) errore held-out, (ii) HangingDiag, (iii) EqualTimeArena (tempo uguale 4 thread +
profondita' fissa), test (gradiente FD, segno/prospettiva vs minimax indipendente D=2, riproducibilita' 1 thread, mutation), documentazione del rischio d'orizzonte D=3.
Fuori ambito (NON fatto di proposito): preferenza nel tie-break alla radice per mosse che non lasciano pezzi catturabili.

## Stato: IMPLEMENTATO E TESTATO; NESSUNA RETE ADOTTATA
Porta (i) SUPERATA (piccola scala); porta (ii) NON SUPERATA sulla candidata reale (piccola scala); porta (iii) NON verificata (sandbox a 1 core). Verdetto: NON ADOTTARE.

## Ambiente (sessione 1)
- Git: HEAD 5cc28fb. Modifiche PREESISTENTI non mie, da non toccare: `Tablut/README_*.md` cancellati e `README_*.md` non tracciati nella radice (README spostati).
  Sui README della radice ho solo AGGIUNTO righe (`README_FEATURES_V4.md` §8/§9, `README_CNN.md` tabella file).
- Il sandbox non aveva `javac`: installato `openjdk-21-jdk-headless` (apt, root). 1 solo core. `Tablut/build/` creato (ignorato da git; eliminabile).
- Compilazione: da `Tablut`: `javac -nowarn -encoding iso-8859-1 -d build -cp "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar" $(find src -name "*.java")` (pulita, ~7 s).
- Nel sandbox i job in background muoiono a fine chiamata: usare `setsid nohup bash -c '...' &`.
- Non esiste `dati.gz` nel repository. Dati di prova (NON nel repo): `/home/claude/tmp/dati.gz` (1500 partite deboli, 49k posizioni), run reale in `/home/claude/tmp/run1/` (log, cand.txt, set.gz).

## Fatti verificati nel codice
- `Negamax.lastScore` e' dal punto di vista di CHI MUOVE; prospettiva Bianco = `mySign * lastScore`; vittoria forzata `+-(1e6+depth)`; l'iterative deepening si ferma prima di D se trova una vittoria forzata.
- `ConvNet` NON esponeva `z`. Ora si': ultimo strato via `outPre` (stesso ordine di somma di `dense`), inferenza bit-identica (hash di ~2000 valutazioni uguale vecchio/nuovo).
- Le 47 feature a mano e le regole sono invarianti sotto le 8 simmetrie (scarto 0.0): l'augmentation e' valida. (NF=47; i gruppi di default ne usano 44.)
- Costo etichetta: 25-40 ms/posizione/thread a D=3 (~4200 nodi), ~7% di etichette forzate.

## Decisioni architetturali e assunzioni (con motivo)
1. Etichette con istanza `Negamax` fresca per posizione, 1 thread, D fisso, margine 0: indipendenza dall'ordine (nessuna fuga di voci TT piu' profonde) e riproducibilita'.
2. Anti-deriva = ancoraggio L2 (default `anchor=0.01`, `anchorTo=prev|init`) + scarto dell'iterazione se l'MSE tanh held-out contro i suoi bersagli non scende: deterministico, nel gradiente, nessun secondo set da tenere pulito.
3. Porta (i): bersagli FISSI = ricerca della rete di PARTENZA a D (una rete costante avrebbe errore zero contro la propria ricerca).
4. Porta (ii): `HangingGate` = porting minimale di `HangingDiag` (quota di mosse che lasciano QUALUNQUE pedina catturabile + pedine bianche perse in 3 ply); passa se entrambe non peggiorano; NON VERIFICATA sotto `hgMinMoves`.
5. Porta (iii): non-inferiorita' con evidenza: punteggio >= 50% e limite inferiore Wilson >= 50%-margine (default 5 punti), in entrambi i modi; default 200 partite per modo. Un'arena saltata/non eseguita = NON VERIFICATA.
6. Esito non valido (partite engine troncate ai 200 ply) -> termine d'esito omesso (peso 0, non rinormalizzato).
7. Le posizioni si campionano una volta (quelle engine con la rete di partenza); le etichette si ricalcolano a ogni iterazione.
8. `main` di `CnnDistiller`/`DistillAcceptance` delegano a `mainImpl` (ritorna il codice) per poterli testare.
9. `DistillSet.Builder.endLabel` (solo diagnostica) per verificare il segno dell'esito delle partite engine contro una mappatura indipendente.

## File modificati / creati
- `Tablut/src/.../aiplayer/ConvNet.java` — logit, `backwardLogit`, `backwardImpl`, `outPre`, `denseBackPre`.
- `Tablut/src/.../training/CnnPretrainer.java` — `init=`.
- `Tablut/src/.../training/EqualTimeArena.java` — `run(...)` estratto (+11 righe), `main` invariato.
- NUOVI in `training/`: `SearchLabeler`, `DistillSet`, `CnnDistiller`, `HangingGate`, `DistillAcceptance`, `DistillCheck`.
- `Tablut/tools/mutation_distill.py` (NUOVO, 68 mutanti); `Tablut/tools/mutation_cnn.py` (mutante 11 adeguato + mutante "ultimo strato usa head 0").
- `README_DISTILL.md` (NUOVO), `README_FEATURES_V4.md` (§8 stato + `init=` + §9), `README_CNN.md` (riga nella tabella file), questo file.

## Test ed esiti
- `DistillCheck` completo (11 sezioni: conv logit loss grad labels repro data sym train gates cli): **112 verifiche OK**, ~38 s.
- Regressione su build pulita: `ConvNetCheck` 47, `CnnTrainingCheck` 48, `CnnEvaluatorCheck` 17, `FeaturesCheck`, `FeatureConversionCheck`, `FastStateCheck` (19.7M verifiche), `ArenaPrefixCheck`, `SelfPlayRulesCheck` OK;
  `WasCaptureCheck` e `ZobristIncrementalCheck` silenziosi con exit 0. NON rieseguiti: `CrossCheck`, `TTCorrectnessCheck`, `ZobristCollisionCheck`, `MLPGradientCheck` (non toccano il codice modificato).
- Mutation: `mutation_distill.py` 68 mutanti; 5 sopravvissuti al primo passaggio, tutti buchi di test, colmati e ricontrollati UNO per UNO (non rieseguita l'intera lista dopo l'ultimo ritocco); `mutation_cnn.py model` 28/28.
- Correzioni fatte ai TEST (non al codice) durante la messa a punto: fixture "Nero in vantaggio" ambigua -> esiti forzati; attese sbagliate su soglie arena (200 patte non passano) e codice di uscita; check debole rimosso.
- Run reale su piccola scala (`run1`): 3 iterazioni D=3, 4000+800 posizioni (3127+873 / 666+134 dataset+engine: NON 50/50, 30 partite engine = 1007 posizioni). MSE tanh held-out start->end: it.1 0.0806->0.0376; it.2 0.0368->0.0284;
  it.3 0.0374->0.0276; deriva pesi 1.7% / 2.5% / 3.5%; errore d'esito 0.5155 -> 0.5241 (peggiora lievemente). Porta (i) vs ricerca della partenza: 0.08057 -> 0.05056 (guadagno 0.0300 +- 0.0040, n=800) SUPERATA.
- Porta (ii) sul run reale (40 partite/rete, D=3 fisso): **NON SUPERATA**. Quota pezzi catturabili 75.2% [71.5-78.6] -> 78.1% [74.2-81.6] (peggiora, intervalli sovrapposti); pedina mossa catturabile 26.2% -> 34.5%;
  materiale bianco perso in 3 ply 0.337 -> 0.334. `DistillAcceptance` esce con codice 1: VERDETTO NON ADOTTARE (log `/home/claude/tmp/run1/accept2.log`). Un primo tentativo da 60 partite era morto a meta' (job in background terminato dal sandbox), non conta.

## Aperti / limiti
- Porta (iii) NON eseguita a scala adeguata (1 core; 4 thread per lato non significativi). Comando verificato eseguibile su coppie singole. `DistillAcceptance ... skipArena=1` la segna NON VERIFICATA.
- Porta (ii) fallita a piccola scala (40 partite/rete): da ripetere con >=100 partite e dopo ablation (`loss=tanh`, `anchorTo=init`, `depths=3,4`, lambda piu' alto) prima di concludere che l'ipotesi d'orizzonte D=3 sia vera.
- Rischio d'orizzonte D=3 documentato ma NON misurato; ablation `loss=tanh` solo testata e provata, non valutata in arena; velocita' di inferenza non misurata con benchmark.
- Tensione metrica di training (logit) vs di decisione (tanh); stesso held-out per arresto e porta (i).
- Mutanti non rilevabili per costruzione elencati in fondo a `tools/mutation_distill.py`.

## Prossimo passo concreto
Su macchina con >= 8 core: `CnnDistiller` con `engineGames` abbondanti (per il 50/50), poi `DistillAcceptance` con `hgGames>=100` e `arenaPairs>=100` (vedi `README_DISTILL.md` §3/§7).
Nessuna rete va adottata finche' (i), (ii) e (iii) non sono TUTTE superate.
