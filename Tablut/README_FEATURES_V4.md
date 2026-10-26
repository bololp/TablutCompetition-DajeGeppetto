# README_FEATURES_V4 — feature v4, convertitore di pesi, diagnostica e piano di fine-tuning

Tutto quanto e' qui e' **misurato** in un sandbox a **1 core** (non nella VM di gara a 4 core). Dove una cosa e' un'ipotesi o non e' stata
provata, e' scritto. Le feature nuove sono prototipate/misurate in modo indipendente dal codice di produzione (si veda `analysis_v4/`).

## 1. In breve

* **Feature v4** (`TablutFeatures.VERSION = 4`): tolte 3 colonne linearmente ridondanti, aggiunte 6 feature in 3 gruppi nuovi
  (`G_ESCAPE`, `G_SAFEFORK`, `G_GATES`). Default: 44 ingressi (prima 41). Tutti gli indici sono cambiati.
* **Convertitore** (`WeightConverter`): porta CNN e MLP di versioni precedenti (v2, v3) sulla v4 **senza cambiare la funzione calcolata**.
  Verificato contro un estrattore v3 congelato: differenza massima 1,2e-6 sull'uscita in (-1,1) con i pesi reali `cnn_t1000000`.
  Convertiti tutti i file di pesi del progetto in `pesi_v4/`.
* **Non ho evidenza che le nuove feature migliorino la forza di gioco.** Le ho aggiunte perche' le hai richieste (gate compresi) e perche'
  hanno una motivazione tattica, ma nelle mie misure il loro contributo e' piccolo o non distinguibile dal rumore (§3). Il giudizio finale e'
  l'arena a tempo uguale (§9), che non ho potuto fare qui.
* **Costo**: circa **+1,1 us a valutazione (~+10%)** con la CNN (§6).

## 2. Cosa cambia

| | v3 | v4 |
|---|---|---|
| Rimosse | | `KING_AXIS_MAX` (identica a `KING_RING`), `BLACK_RING4`, `WHITE_RING4` (= `N_*` meno gli anelli 1-3) |
| Aggiunte, gruppo `G_ESCAPE` (16) | | `ESC_DIST` = 1/d (d = mosse minime del Re per fuggire, solo d<=3); `ESC_3STEP_CELLS` (celle a 2 mosse da cui il Re ha una via); `ESC_EDGE3` (celle di fuga distinte entro 3 mosse) |
| Aggiunte, gruppo `G_SAFEFORK` (32) | | `SAFE_FORKS`: celle a 1 mossa con >=2 vie dove il Re, arrivato, non e' catturabile |
| Aggiunte, gruppo `G_GATES` (64) | | `GATES_CLOSED` (gate occupati), `GATE_RUN` (somma delle corse libere dai 16 gate) |
| `G_DEFAULT` | 11 (41 feature) | 123 (44 feature) |

`ESCAPE`/`SAFEFORK` usano una BFS delle mosse del Re su scacchiera **statica** (il Nero non muove, il Re non cattura spostandosi: la stessa
approssimazione dichiarata di `KING_2STEP_*`). I **gate** sono le 16 celle di fuga all'estremo delle righe/colonne 1, 2, 6, 7.
`TablutFeatures.IDS` da' un **id stabile** a ogni feature (gli indici scorrono, gli id no): e' la base del convertitore.

## 3. Le misure che hanno guidato le scelte (e i loro limiti)

* **Ridondanza**: le 41 colonne v3 hanno rango 38 (3 dipendenze esatte, verificate numericamente su 160.037 posizioni). Una feature
  linearmente dipendente non porta informazione.
* **Test "prevedi l'esito" (MSE held-out) poco sensibile**: aggiungere le candidate a una rete piccola da' variazioni da -0,002 a +0,005, uguali
  a quelle dei controlli inutili (colonna duplicata +0,0025, rumore +0,0032). Non e' un test che distingua queste feature (le etichette sono
  esiti di partite giocate da giocatori deboli).
* **Valore statico vs ricerca a profondita' 3 con la stessa CNN** (6.000 posizioni, correlazione 0,95/0,93): l'errore si concentra dove il Re e'
  vicino alla fuga (fuga in 2 mosse: errore medio 0,247, 14,5% oltre 0,5; nessuna fuga entro 3 mosse: 0,098). Sui risultati forzati
  la statica sbaglia il segno solo in 8 casi su 618: vede *chi* vince, sbaglia *di quanto*.
* **Le candidate spiegano poco di quell'errore**: guadagno di R^2 fra -0,002 e +0,001 su tutto, +0,004/+0,008 (controllo -0,002) nelle posizioni
  con fuga entro 3 mosse (n=2660): indicativo, non significativo. **I gate sono quasi costanti** nei miei dati (15,5 gate aperti su 16).
* **Il 53% dell'errore statico-vs-ricerca e' prevedibile da input che la rete ha gia'** (valore statico 25%, + chi muove 37%, + materiale 46%, + vie
  di fuga 46%, + tutte le 41 feature 53%): il collo di bottiglia sembra la calibrazione/addestramento, non gli input. E' per questo che §8 propone il
  fine-tuning verso la ricerca. R^2 sul residuo **non** e' forza di gioco.
* Limiti: 1 core, dati da giocatori deboli (le tattiche rare pesano meno che in gara), 6.000 etichette, nessuna arena.

## 4. Convertitore di pesi

```bash
CP="build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; T=it.unibo.ai.didattica.competition.tablut
java -cp $CP $T.aiplayer.WeightConverter in=cnn_t1000000.txt dry=1               # mostra il piano, non scrive
java -cp $CP $T.aiplayer.WeightConverter in=cnn_t1000000.txt out=cnn_v4.txt      # CNN o MLP, riconosciuto dall'intestazione
java -cp $CP $T.aiplayer.WeightConverter in=vecchio.txt out=nuovo.txt groups=127 # maschera di gruppi esplicita
java -cp $CP $T.training.DatasetUpgrade in=dati.gz out=dati_v4.gz                # dataset: ricalcola le feature dalle scacchiere
```

**Perche' e' esatto.** Le feature entrano solo nelle colonne del primo strato che le legge. Lavorando in unita' grezze (`u = w / DIV`):
feature in comune -> peso copiato (riscalato per il rapporto dei DIV); feature **nuova** -> peso 0; feature **rimossa ma derivabile**
(`FeatureRegistry.DERIVED`) -> il suo peso viene sommato alle feature da cui dipende; feature persa (non derivabile, o gruppo tolto) ->
il piano la elenca e dichiara "NON ESATTA". Non ci sono scorciatoie silenziose.

**Come e' verificato** (`selftest.FeatureConversionCheck`): rete vecchia valutata con l'estrattore v3 **congelato**
(`LegacyTablutFeaturesV3`, copia del codice prima della v4); rete convertita con quello attuale; le uscite devono coincidere. Copre CNN
(gruppi 2, 11, 15 verso default e `G_ALL`), MLP con ruolo in ingresso e con 2 head, MLP v2, DIV diversi fra versioni, file reali
scritti e riletti dai loader di produzione, rifiuto di casi invalidi, `DatasetUpgrade` (identico a un dataset generato direttamente), e i pesi reali.
Inoltre verifica che ogni id v3 abbia lo stesso valore grezzo in v3 e v4 (61.500 confronti) o sia una identita' lineare esatta.

**Mutation test** (`tools/mutation_features_v4.py`): 14/14 bug iniettati nelle feature rilevati; 11/12 nel convertitore. Il mutante C9
(ruolo invertito in `DatasetUpgrade`) e' **equivalente** (l'estrattore non legge il turno). Un mutante (DIV ignorato nel rapporto 1:1)
all'inizio NON era rilevato: era un buco reale (fra v3 e v4 tutti i DIV in comune coincidono); ho aggiunto un test con schemi sintetici e ora e' rilevato.

**Cosa NON e' provato**: lo schema **v2** non ha piu' un estrattore contro cui confrontarsi (e' "v3 senza `WHITE_RING4`", come dice il README).
Controllo statistico: la MLP v2 `pretrained_default` convertita ottiene MSE 0,5442 sul mio dataset contro 0,5445 della MLP v3 supervisionata
(predittore costante 0,8746): coerente con la mappatura giusta, non una prova.

**Dopo la conversione.** Le colonne nuove partono da 0: il TD con `alpha=0,002` le fara' crescere molto lentamente. `CnnPretrainer` oggi non puo'
ripartire da pesi esistenti (manca `init=`): serve per un fine-tuning supervisionato (si veda §8).

## 5. Test eseguiti sull'albero finale (compilazione pulita con `--release 8`)

`FeaturesCheck` (5000 posizioni x 3 seed: 47 feature contro oracoli via motore, 8 simmetrie, 128 maschere di gruppi, forchette sicure
verificate come **vittorie forzate in 3 semi-mosse contro tutte le risposte nere**: 7.217 in totale sui 3 seed), `FeatureConversionCheck`, e la suite preesistente:
`ConvNetCheck` 47, `CnnEvaluatorCheck` 17, `CnnTrainingCheck` 48, `MLPGradientCheck`, `NoAllocationCheck`, `CrossCheck` 0 mismatch,
`WasCaptureCheck` 0 mismatch, `TTCorrectnessCheck`. Non rilanciati: `SearchOptimizationsCheck`, `LazySmpCheck`, i test MCTS (non toccati).

## 6. Costo (1 core, rumore ~10%)

Estrazione delle feature: 2,37 us (v3) -> 3,77 us (v4 default); ESCAPE e SAFEFORK condividono la BFS (+~1 us insieme), i gate costano ~0,2 us.
CNN completa dal percorso di produzione: ~10,2 us (38 feature, equivalente v3) -> ~11,3 us (44 feature). **Non misurato**: l'effetto sui nodi/s
della ricerca (atteso circa -8/-10%) e sulla profondita' raggiunta a 60 s.

## 7. Diagnostica: perche' il Bianco a volte si "appende" accanto a citadel/pedine nere

Partite reali Negamax + `cnn_t1000000` (entrambi i lati, 100 ms/mossa, aperture casuali di 4 semi-mosse); ogni mossa bianca di pedina viene
verificata col motore (il Nero puo' catturare subito la pedina appena mossa?). Due serie indipendenti, **solo questi numeri sono verificati**:

| | serie A (12 partite, valutatore v3) | serie C (13 partite, valutatore v4 convertito) |
|---|---|---|
| mosse di pedina / "appese" | 234 / 72 (30,8%) | 190 / 52 (27,4%) |
| appese accanto a citadel o pedina nera | 62 (86%) | 48 (92%) |
| profondita' media; ricerca con profondita' >= 2 | 4,33; 72/72 | 3,06; **45/52** (7 casi a profondita' 1) |
| esiste un'alternativa non appesa | 72/72 | 52/52 |
| la scelta batte la migliore alternativa sicura (prof. 3) | 39/72 | 20/52 |
| margine scelta - miglior sicura (scala 1000), mediana [25%, 75%] | +0,3 [0,0; 10,5] | -0,1 [-52,3; 14,4] |
| pedine perse nei 3 ply dopo: Bianco, appesa vs non appesa | 0,51 vs 0,17 | 0,57 vs 0,15 |
| pedine nere perse nei 3 ply dopo, appesa vs non appesa | 0,20 vs 0,31 | 0,43 vs 0,40 |

Lettura:

* **Dipende dalla profondita'.** Con profondita' >= 2 il Negamax vede la cattura (basta la risposta nera). Nella serie C, con 100 ms, in 7 casi su 52 la
  ricerca si e' fermata a profondita' 1: li' il Bianco **non puo'** vederla. A 60 s/mossa la profondita' e' molto maggiore, quindi questo caso non dovrebbe
  presentarsi, ma **non l'ho misurato**: con i miei budget non posso dire quanto della tua osservazione sia questo effetto.
* **Nei casi in cui la vede, la sceglie perche' la valutazione la accetta.** La CNN valuta il materiale (togliere una pedina bianca: -147 in media, sd 200;
  una nera: +228; variazione media per ply: 54 nella serie A, 104 nella C), eppure ha sempre un'alternativa sicura e il margine e' spesso ~0.
  Nella serie C, per le posizioni equilibrate (|V|<300, 30 mosse) il margine mediano e' +74: la mossa appesa e' proprio valutata meglio, cioe' e' un
  "sacrificio" per la rete.
* **Il sacrificio non rende in pedine**: nei 3 ply successivi il Bianco perde ~0,4 pedine in piu' rispetto alle mosse non appese, in entrambe le serie,
  e il Nero non ne perde di piu' in modo coerente. Se renda in altro (posizione, vie di fuga) non l'ho misurato.
* **Ipotesi della saturazione del tanh: NON confermata.** Avevo il sospetto che nelle posizioni "decise" (|V| vicino a 1000) le differenze si schiaccino e il
  gradiente di training si annulli. Nella serie C la quota di mosse appese per |V| e' 30% (|V|<300, 30 mosse), 12% (300-800, 77), 41% (>=800, 83): la quota
  piu' alta e' fra le decise ma l'andamento non e' monotono e i campioni sono piccoli. Resta un'ipotesi da testare, non una spiegazione.
* Limiti: 25 partite, budget di 100 ms (profondita' ~3-4), un solo evaluator, nessuna ripetizione con seed diversi dentro ciascuna serie.
  Due serie con evaluator diversi (v3/v4: stessa funzione, costo diverso) hanno profondita' diverse, quindi non sono repliche pulite.
  Strumento: `analysis_v4/HangingDiag.java` (argomenti: pesi partite ms seed secondi_massimi).

## 8. Piano: fine-tuning verso il valore di ricerca (NON ancora implementato)

Obiettivo: avvicinare il valore statico V(s) al valore che darebbe una ricerca a profondita' D con la stessa rete (idea TD-leaf / distillazione della ricerca).
Costo a tempo di gioco: **zero** (stessa rete, stessa dimensione).

1. **Posizioni**: meta' da dataset (`dati.gz`, partite deboli, gia' disponibili e aggiornabili con `DatasetUpgrade`), meta' da partite dell'engine stesso a budget realistici.
   Si scartano le posizioni terminali.
2. **Etichette** (`training/SearchLabeler`, nuovo): per ogni posizione `Negamax(ev, D)` a **1 thread per ricerca** (i multi-thread non sono riproducibili), parallelo sulle posizioni.
   Misurato: D=3 con la CNN ~27 ms/posizione su 1 core -> ~500/s su 4 core, 200.000 etichette in ~7 min (stima). Bersaglio `y = clip(score/1000, -0,98, 0,98)` dal punto di vista del Bianco.
3. **Loss nel dominio logit**: `L = (z - atanh(y))^2` con `z` = uscita prima della tanh. Il MSE su tanh moltiplica il gradiente per `(1-y^2)`, che si annulla nelle posizioni decise.
   E' una scelta di robustezza (vale a prescindere da §7, dove l'ipotesi della saturazione non e' confermata); si puo' confrontare con la loss su tanh come ablation.
   (Richiede di esporre `z` da `ConvNet`: da verificare nel codice.)
4. **Addestramento** (`training/CnnDistiller`, nuovo): parte dai pesi convertiti (serve `init=` in `CnnPretrainer`), Adam con lr piccolo (1e-4..3e-4), 2-4 epoche, minibatch,
   augmentation con le 8 simmetrie (le etichette sono invarianti: regole e feature lo sono), validazione su partite tenute da parte.
5. **Anti-deriva** (la ricerca usa la rete stessa: rischio autoreferenziale): perdita mista `(1-l)*L_ricerca + l*L_esito` con l~0,2; ancoraggio L2 ai pesi precedenti o arresto se la rete
   si allontana troppo su un insieme di sonda; D fisso per iterazione e iterazioni da 3 a 5.
6. **Porte di accettazione** (la nuova rete si adotta solo se le supera tutte): (i) errore held-out verso la ricerca inferiore; (ii) `HangingDiag` non peggiora (quota di mosse appese, materiale perso nei 3 ply dopo);
   (iii) `EqualTimeArena` nuova contro vecchia, a tempo uguale e 4 thread, e a profondita' fissa. Con 30 partite l'IC95 e' ~+-17 punti: si decide su piu' partite, non su una cifra.
7. **Test, nel solito stile**: gradiente della loss contro differenze finite; segno/prospettiva delle etichette contro un minimax indipendente a profondita' 2 su `AshtonRules`;
   riproducibilita' a 1 thread; mutation test.
Rischi dichiarati: la rete puo' imparare gli artefatti d'orizzonte di D=3; il guadagno offline (R^2 0,53 sul residuo) **non** implica un guadagno in arena.

Alternativa economica, non implementata: nella scelta alla radice, a parita' di valore (entro epsilon) preferire le mosse che non lasciano pezzi catturabili. Cambia solo il tie-break.

## 9. Prossimi passi consigliati

1. Sulla tua macchina: `FeaturesCheck` e `FeatureConversionCheck`. Nota: il vecchio file v3 **non e' piu' caricabile** dal codice v4 (il messaggio indica il convertitore) e la rete convertita
   e' identica per costruzione, quindi confrontarle in arena non ha senso. La domanda utile e': l'addestramento successivo con le colonne nuove migliora?
2. Fine-tuning supervisionato breve dai pesi convertiti con le colonne nuove (richiede `init=`), poi arena contro la CNN convertita ma non addestrata ulteriormente.
3. Se i risultati sono favorevoli, implementare §8.

## 10. File

**Nuovi**: `aiplayer/FeatureRegistry.java`, `aiplayer/WeightConverter.java`, `aiplayer/LegacyTablutFeaturesV3.java` (copia congelata, solo per test, non modificare),
`training/DatasetUpgrade.java`, `selftest/FeatureConversionCheck.java`, `tools/mutation_features_v4.py`, `analysis_v4/`, `pesi_v4/`.
**Modificati**: `aiplayer/TablutFeatures.java` (v4), `selftest/FeaturesCheck.java`, `aiplayer/MLP.java` (solo `layerTable()` package-private),
`aiplayer/CnnEvaluator.java` e `NeuralEvaluator.java` (solo il messaggio di errore, che ora indica il convertitore).
**Aggiornare dopo la conversione**: ogni comando che usa pesi (client, trainer, arena) deve puntare ai file `*_v4.txt`; i `checkpoints/` legacy non sono convertibili.
