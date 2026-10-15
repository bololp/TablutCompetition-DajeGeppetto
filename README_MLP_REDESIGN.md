# Riprogettazione della rete (MLP) del client Tablut

Obiettivo richiesto: una rete e un insieme di feature dimensionati per una CPU a 4 core / 8 GB, che
massimizzino la **forza per secondo di ricerca**, con una struttura che tratti in modo diverso
attaccanti e difensori e che copra materiale, mobilità, sicurezza del Re e struttura.

Tutto quanto è scritto qui è stato **misurato** in questa sessione (1 core, JDK 21); i numeri
sono quelli dei log in `experiments/logs/`. Dove una cosa NON è provata, c'è scritto.

---

## 1. Risultati in breve

| | vecchio | nuovo (default) |
|---|---|---|
| feature | 16, "corsie" sbagliate (81% fantasma) | 43 (40 di default), esatte e invarianti per simmetria |
| rete | 16-24-12-1, tanh, 721 parametri | tronco 24 + 2 head (8), leaky-ReLU, 1402 parametri |
| costo `evaluate()` (JVM pulita) | 1,64 µs | **2,88 µs** (tier leggero: 1,94 µs) |
| `SimpleEvaluator` (riferimento) | 0,41 µs | |
| errore di previsione dell'esito (held-out) | MSE 0,679 (R² 0,284) | **MSE 0,637 (R² 0,328)** |
| 60 partite a tempo uguale vs Simple | 38,3% (la tua `trained_weights.txt`) | **77,5%** (pesi solo supervisionati) |
| default contro la tua rete vecchia | | **90,8%** (+54 =1 −5) |

Due cose da sapere subito:

* **La nuova rete è più cara per valutazione, non più economica.** L'ipotesi iniziale
  ("modello più piccolo → profondità maggiore") non è la leva giusta qui: il costo di una
  valutazione è dominato dalle **feature**, non dalla rete (la rete di default costa ~0,5 µs
  su 2,9). A tempo uguale il nuovo default raggiunge meno profondità di Simple (3,25 vs 3,61 a
  100 ms/mossa) e vince lo stesso, perché le feature valgono più del costo.
* **Anche la ricerca e' cambiata (§10)**: `Negamax` ora usa PVS + killer moves + history heuristic di default.
  A parita' di valori calcolati (verificato), servono ~85% di nodi in meno a profondita' 5 e in 2 s si arriva
  ~1 ply piu' in la. Client e trainer lo ereditano senza modifiche.
* **I pesi consegnati sono solo pre-addestrati in modo supervisionato**, non allenati con TD. Sono un
  *warm start*, non il prodotto finale. Il training lungo resta da fare sulla tua macchina.

---

## 2. Dove va il tempo di ricerca (misurato)

Su posizioni reali di partita (non la posizione iniziale):

| voce | costo |
|---|---|
| `generateMoves` | 2,3 µs / posizione (40 ns / mossa) |
| creare un figlio (`State.clone()` + hash + `wasCapture`) | **~0,73 µs / nodo** |
| `SimpleEvaluator.evaluate` | 0,41 µs |
| vecchia rete 16-24-12-1 | 1,64 µs |
| nuovo default / tier leggero / tutte le feature | 2,88 / 1,94 / 3,47 µs |

Nodi/s misurati nelle partite (TT attiva, 100 ms/mossa): Simple ≈ 980k, vecchia rete ≈ 490k,
nuovo default ≈ 420–440k, tier leggero ≈ 525k.

Costo delle sole feature (stima prima delle ultime ottimizzazioni, JVM separata per misura):
BASE 0,77 µs · +mobilità 1,69 · +vie di fuga 1,75 · +minacce 2,29 · tutte 2,72. La
**passata sulle mosse costa ~0,9 µs** ed è dominata dalle mispredizioni di salto dei raggi (~100
per posizione), non dall'aritmetica.

Perché la rete in sé costa poco: il JIT di HotSpot **non vettorizza** questi loop corti (misurato:
~0,57 ns/MAC anche con ReLU). Un loop "register-blocked" a 4 neuroni di uscita con pesi row-major
dimezza il costo per MAC (0,35 ns): è quello usato in `MLP.dense`. `Math.tanh` costa ~20 ns:
la vecchia rete ne chiamava 36 a valutazione; la nuova ne chiama una (uscita).

---

## 3. Le feature (`TablutFeatures`, versione 2)

43 feature scalate in ~[0,1], quasi tutte conteggi interi esatti (la tabella `DIV[]` è la fonte
unica delle scale, usata anche dai test). Gruppi:

* **BASE (27)**, sempre calcolate. *Materiale e Re*: pedine, anello/assi del Re rispetto al trono,
  Re sul/accanto al trono, ostili adiacenti e quanti ne mancano per catturarlo (4 / 3 / 2 lati
  secondo la regola). *Struttura*: centralità, distribuzione ad anelli, pedine isolate, legami
  ortogonali e diagonali dei neri (blocchi), righe/colonne di fuga libere o presidiate,
  prossimità attaccanti-Re / difensori-Re.
* **MOB (6)** *mobilità*: mosse di Bianco e Nero, mosse del Re, pezzi bloccati, celle di fuga
  raggiungibili dal Nero.
* **LANES (4)** *fuga del Re*: vie di fuga immediate, quante il Nero non può bloccare, celle a una
  mossa da cui il Re ha una via, celle da cui ne ha due ("forchette").
* **THREATS (3)** *catture*: pedine bianche catturabili dal Nero, nere dal Bianco, Re catturabile
  subito.
* **FLOOD (3)** *accerchiamento* (flood fill dal Re): **opzionale, fuori dal default** (§5).

### Una scoperta che riguarda anche il codice esistente
Il Re **non può entrare nelle citadel** (`AshtonRules.generateMoves`): i raggi lungo le righe/colonne
3-4-5 terminano su una citadel e non sono vie di fuga. Le "corsie aperte" di `SimpleEvaluator` e della
vecchia `TablutFeatures` ignorano questo: su 3000 posizioni contavano **1298 corsie contro 244 vie di
fuga reali** (le due definizioni divergono nel 28% delle posizioni). Le celle di fuga sono **20**
(32 di bordo − 12 citadel di bordo). **Non ho modificato `SimpleEvaluator`** (è la baseline del trainer e
cambiarla altera tutti i confronti): è una correzione che puoi valutare tu (`experiments/LaneCheck.java`).

### Come sono verificate (`selftest/FeaturesCheck`)
Ogni feature è confrontata con un oracolo indipendente su migliaia di posizioni di gioco (6000 per
seed, 3 seed provati): mobilità e mappe di raggiungibilità contro le mosse di `AshtonRules`; catture e
Re catturabile **applicando ogni mossa col motore**; vie di fuga e fuga a 2 mosse spostando davvero il
Re; flood fill contro una BFS; struttura contro un ricalcolo su `Pawn[][]`. Inoltre: invarianza per le 8
simmetrie della scacchiera su tutte le feature, assenza di stato che "trapela" tra una chiamata e
l'altra, e le 16 maschere di gruppi. Il test ha trovato due errori miei prima della consegna
(indici delle linee di fuga 5,6 invece di 6,7; celle di fuga 20 e non 24).
Limite dichiarato: la fuga a 2 mosse ignora le catture fatte dal Re spostandosi.

---

## 4. La rete e il ruolo (`MLP`, `NeuralEvaluator`)

Il "ruolo" che sceglie la head è **chi deve muovere nella posizione valutata** (funzione della sola
posizione: coerente tra client Bianco, client Nero e self-play). La stessa classe realizza le tre
strutture proposte, a parità di costo di inferenza:

* **A** una head, ruolo come input · **B** tronco condiviso + 2 head · **C** due reti separate.

Esperimento (247.488 posizioni da 7000 partite generate; test = 51.159 posizioni di partite non viste;
bersaglio = esito della partita; MSE più basso è meglio):

| | MSE (molti dati) | coerenza ply→ply \|ΔV\| | MSE (~8k posizioni) |
|---|---|---|---|
| A | 0,6392 ± 0,0041 | 0,1303 | 0,7256 ± 0,0118 |
| **B** | 0,6392 ± 0,0023 | 0,1326 | 0,7154 ± 0,0062 |
| C | 0,6384 ± 0,0017 | 0,1459 | 0,7086 ± 0,0020 |
| senza ruolo | 0,6487 ± 0,0015 | 0,1314 | 0,7298 ± 0,0108 |

Lettura onesta: con molti dati **A, B, C sono indistinguibili** in errore di previsione; C è meno
coerente da una mossa alla successiva (due reti separate si "accordano" meno: più oscillazioni
pari/dispari in alpha-beta); con pochi dati (il regime del TD iniziale) C e B battono A; togliere del
tutto il ruolo è peggio (bias sistematico di ±0,06–0,08 tra Bianco e Nero di turno). **B è l'unico mai
ultimo** → scelto come compromesso robusto, **non** come vittoria netta. B inoltre permette di
addestrare in modo diverso attacco e difesa: il gradiente di una posizione tocca il tronco e *solo* la
head del suo ruolo (verificato da `MLPGradientCheck`).

**Dimensione.** Tronco 8→128: l'errore resta in 0,635–0,645 senza tendenza chiara (8/4: 0,6443 a
206 ns; 24/8: 0,6371 a 462 ns; 48/16: 0,6353 a 966 ns; 128/64: 0,6448 a 4326 ns). Tra 16 e 48 neuroni la
differenza di costo (~0,1–0,5 µs su ~4 µs per nodo) è irrilevante per la profondità; sopra i 64 la rete
diventa il collo di bottiglia senza guadagno. Scelto **24 + head da 8** (1402 parametri, 1193 per
valutazione). Il segnale è limitato da etichette rumorose (partite deboli): con dati TD migliori una rete
più grande potrebbe servire; la configurazione è cambiabile da riga di comando.

**Attivazione.** leaky-ReLU 0,6364 a 713 ns · ReLU 0,6396 · hard-tanh 0,6420 · tanh 0,6365 a **1398 ns**
→ leaky-ReLU: qualità di tanh a metà costo (e niente neuroni "morti" nel TD online).

---

## 5. Quali feature valgono il loro costo (ablazione, rete B 32/16, 2 seed; rumore ≈ ±0,003)

| gruppo tolto | MSE | lettura |
|---|---|---|
| nessuno (43) | 0,6396 | |
| minacce di cattura | 0,6468 (+0,007) | contano |
| struttura (15 feature) | 0,6485 (+0,009) | contano |
| mobilità | 0,6435 (+0,004) | contano poco/moderatamente |
| vie di fuga | 0,6407 (+0,001) | poco **in media**, ma vedi sotto |
| accerchiamento (FLOOD) | 0,6358 (−0,004) | nessun valore misurabile |
| solo BASE | 0,6633 | |

Le vie di fuga sono rare ma decisive: sulle posizioni in cui il Re ha vie/forchette l'MSE passa da
**0,6243 (senza) a 0,5914 (con)** (rete 24/8, 3 seed). FLOOD, che costa ~0,4 µs, non porta nulla: con la
sua assenza l'MSE è 0,6418 → 0,6424. Per questo **il default è `G_DEFAULT = MOB|LANES|THREATS`** (40
ingressi); FLOOD è implementato, testato e attivabile con `groups=15`. Il *tier leggero*
(`groups=2`, 31 ingressi, salta anche la scansione dei bianchi) costa 1,94 µs.

---

## 6. Partite a tempo uguale (`EqualTimeArena`)

100 ms/mossa, 60 partite per accoppiamento (30 aperture casuali di 4 semi-mosse, ognuna giocata a colori
invertiti), IC95 di Wilson. **Pesi nuovi = solo supervisionati.**

| A | contro | A: + = − | punteggio A | IC95 | profondità media A / B |
|---|---|---|---|---|---|
| nuovo default | Simple | 45 3 12 | **77,5%** | 65–86% | 3,25 / 3,61 |
| nuovo, con FLOOD | Simple | 40 0 20 | 66,7% | 54–77% | 3,00 / 3,40 |
| tier leggero | Simple | 37 2 21 | 63,3% | 51–74% | 3,36 / 3,63 |
| vecchie feature + rete 24-12 (stessi dati) | Simple | 28 0 32 | 46,7% | 35–59% | 3,26 / 3,40 |
| la tua `trained_weights.txt` (classi originali) | Simple | 23 0 37 | 38,3% | 27–51% | 3,00 / 3,25 |
| nuovo default | tua rete vecchia | 54 1 5 | **90,8%** | 81–96% | 3,04 / 3,21 |
| nuovo default | tier leggero | 33 0 27 | 55,0% | 42–67% | 3,12 / 3,18 |

Il default non si distingue statisticamente da FLOOD né dal tier leggero (IC sovrapposti): non posso dire
che sia meglio di entrambi, solo che nessuno dei due si è dimostrato migliore. Con un budget di 60 s a
mossa (invece di 100 ms) il rapporto costo/profondità cambia e **non l'ho misurato**.

TD dopo 100 episodi, partendo dai pesi supervisionati, contro il proprio punto di partenza (40 partite):
`alpha=0.02` → 66,3% (IC 51–79%), `alpha=0.005` → 56,3% (IC 41–70%). Non è una prova di miglioramento
(campione piccolo), ma mostra che il TD non rovina la rete.

---

## 7. File consegnati (da copiare dentro `Tablut/src/...` mantenendo i package)

Nuovi o riscritti:

* `aiplayer/TablutFeatures.java`, `aiplayer/MLP.java`, `aiplayer/NeuralEvaluator.java` (**API cambiata**: nuova
  rete, ingresso compattato per gruppi, head per ruolo, thread-safe, formato file v2 con versione delle
  feature e maschera dei gruppi).
* `training/SelfPlayTrainer.java` (portato alla nuova API; stessi argomenti di prima più `groups`, `trunk`,
  `head`, `act`; ignora i checkpoint legacy con un messaggio).
* `training/SupervisedPretrainer.java` (nuovo), `training/EqualTimeArena.java` (nuovo).
* `selftest/FeaturesCheck.java`, `selftest/MLPGradientCheck.java` (riscritto).

Invariati ma rimessi nello zip perché mancavano da quello caricato: `selftest/CrossCheck`, `WasCaptureCheck`,
`ZobristIncrementalCheck`, `ZobristCollisionCheck`, `TTCorrectnessCheck` e `aiplayer/RepetitionAndStalemateCheck`
(sono i tuoi, non modificati; quest'ultimo sta in `aiplayer` perché usa metodi package-private di `Negamax`).
**Aggiunta successiva (ricerca, §10)**: `aiplayer/Negamax.java` (modificato: PVS, killer moves, history heuristic),
`aiplayer/SearchOptimizationsCheck.java` (nuovo), `training/EqualTimeArena.java` (aggiunti i prefissi `ab:`, `pvs:`,
`nopvs:`).
**Non modificati**: `AshtonRules`, `ZobristHasher`, `TranspositionTable`, `SimpleEvaluator`, `TablutNegamaxClient`.

`pretrained_weights/pretrained_default.txt` (default) e `pretrained_lean.txt` (tier leggero): addestrati
supervisionati su tutte le 247.488 posizioni (20 epoche, Adam). Sono i file con cui sono state giocate le
partite del §6. `experiments/`: sorgenti scratch usati per gli esperimenti e i log completi (percorsi fissi,
non fanno parte della build).

**Incompatibilità**: `trained_weights.txt` e `checkpoints/` attuali (formato legacy, feature v1) non sono
leggibili dalla nuova rete e vanno riallenati. Il client lo intercetta (messaggio e ripiego su
`SimpleEvaluator`, già provato dal vivo col server); il trainer ignora i checkpoint legacy.

### Uso
```bash
cd Tablut
javac -encoding iso-8859-1 -d build -cp "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar" $(find src -name "*.java")
CP="build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; T=it.unibo.ai.didattica.competition.tablut

# 1) (facoltativo) rigenerare il warm start: ~4,5 min di generazione su 1 core (usa tutti i core) + 2 min
java -cp $CP $T.training.SupervisedPretrainer games=7000 epochs=20 out=pretrained.txt

# 2) TD self-play partendo dal warm start (consigliato: da pesi casuali la rete resta ~costante per migliaia di episodi)
java -cp $CP $T.training.SelfPlayTrainer resumeFrom=pretrained_weights/pretrained_default.txt \
     episodes=20000 epsilonStart=0.1 out=trained_weights.txt

# 3) confronto a tempo uguale (A=nuovi pesi, B=Simple); stampa IC95, profondita' e nodi/s
java -cp $CP $T.training.EqualTimeArena trained_weights.txt simple 30 100

# 4) client: il 4° argomento e' opzionale (senza, o con file illeggibile, usa SimpleEvaluator)
java -cp $CP $T.client.TablutNegamaxClient WHITE 60 localhost trained_weights.txt
```

---

## 8. Test

| test | cosa verifica |
|---|---|
| `FeaturesCheck [n] [seed]` | §3: ogni feature vs oracolo del motore, 8 simmetrie, riuso workspace, 16 maschere |
| `MLPGradientCheck` | forward vs implementazione ingenua; gradiente vs differenze finite su A/B/C, 4 attivazioni, entrambi i ruoli, larghezze non multiple di 4; il gradiente non tocca la head dell'altro ruolo; salva/carica esatto al bit; rifiuto di file legacy e di altra versione di feature; copia indipendente; 4 thread su una rete condivisa |
| `SearchOptimizationsCheck [seed] [n]` | §10: ogni configurazione di ricerca == alpha-beta puro (oracolo indipendente), mossa ottima, TT, ripetizione, istanza riusata, timeUp, ordinamento ben formato |
| `CrossCheck`, `WasCaptureCheck`, `ZobristIncrementalCheck`, `ZobristCollisionCheck`, `TTCorrectnessCheck`, `RepetitionAndStalemateCheck` | i tuoi, invariati |

Tutti passano sull'albero finale (build pulita): CrossCheck 20 partite/2367 semi-mosse, WasCapture 23.787
mosse, Zobrist 11.404 mosse e 32.870 posizioni distinte, TT 24 confronti, ripetizione/stallo OK, 0 mismatch.
`MLPGradientCheck` è stato anche messo alla prova con **4 bug introdotti apposta** (derivata leaky errata, bias
non accumulato, resto del kernel ignora il bias, head 1 che usa la head 0): tutti rilevati.

Prove dal vivo: partita col server arbitro reale (client nuovo vs client Simple; poi client con pesi legacy vs
client nuovo): nessuna mossa rifiutata, fallback verificato. Trainer esercitato con 4 thread, checkpoint,
valutazione periodica e rilevatore di forgetting senza errori.

---

## 9. Limiti, cose non fatte, cose da guardare

1. **Nessun test automatico dell'aggiornamento TD** (eligibility trace con le due head): è verificato solo
   dal comportamento (stabilità, nessun peggioramento in 100 episodi, spostamento dei valori coerente con i
   dati). Un test dedicato richiederebbe di rendere accessibile `trainOnTrajectory`.
2. **TD da pesi casuali**: in 100 episodi la rete è collassata su una costante (uscita media −0,77, sd 0,11 su
   20.000 posizioni; quella supervisionata ha sd 0,60). È il normale avvio lento del TD, non un errore.
   Per questo il warm start.
3. **Diagnostica periodica a profondità fissa del trainer**: in una corsa di prova di 8 episodi ha dato
   12-0 a profondità 1 e 3 ma 25–54% a profondità 2. A tempo (arena, client) non compare. Probabilmente è
   l'effetto pari/dispari con partite deterministiche duplicate, ma **non l'ho indagato**: tienilo d'occhio.
4. Le partite dell'arena non tracciano la ripetizione e girano a 100 ms/mossa (profondità ~3–3,6, in gara
   saranno più alte); 60 partite danno IC95 di ±12 punti circa.
5. **Leve non tentate**, con i numeri misurati: (a) scansione delle mosse a bitboard/lookup per riga e colonna:
   dovrebbe ridurre il default da ~2,9 a ~2,2 µs (~−15% di tempo per nodo, ~+0,1 ply): guadagno modesto;
   (b) `State.clone()` + `Action` a stringhe: ~0,73 µs/nodo, cioè **la voce singola più grande** del costo di
   un nodo (richiede una rappresentazione primitiva nel motore di ricerca, rischiosa senza i tuoi test);
   (c) ricerca parallela: oggi in partita lavora **un solo core** su 4 (il multithreading c'è solo nel
   trainer); `NeuralEvaluator` è già thread-safe per questo; (d) ordinamento: killer e history sono fatti (§10); restano da provare catture/vie di fuga per prime, history
con penalita', killer trasportate tra una mossa e la successiva, aspiration windows.
6. Il bersaglio del warm start (esito di partite di giocatori deboli) insegna "chi vince da qui, giocando così":
   serve a partire da pesi sensati, non a sostituire il TD.
7. Learning rate diversi per ruolo, o fine-tuning separato delle due head, sono possibili con la struttura B
   ma **non implementati**.

---

---

## 10. Ricerca: PVS, killer moves, history heuristic (`Negamax`)

Tre ottimizzazioni **che non devono cambiare il risultato** della ricerca, solo il numero di nodi. Sono accese di default;
il costruttore `Negamax(ev, maxDepth, useTT, ttSize, usePVS, orderingMode)` permette di spegnerle
(`usePVS=false`, `orderingMode=Negamax.ORDER_NONE|ORDER_KILLERS|ORDER_HISTORY|ORDER_DEFAULT`) per i confronti. Client e trainer usano i
costruttori esistenti e le ereditano senza modifiche (nota: il self-play ora cerca piu' in profondita' a parita' di tempo).

### 10.1 Cosa fanno
* **PVS.** In ogni nodo la prima mossa e' cercata a finestra piena `(-beta,-alpha)`, le successive con finestra nulla
  `(-alpha-1e-6,-alpha)`, ri-cercando a finestra piena solo quelle che sembrano migliorare (radice e nodi interni; dopo
  `timeUp` la ri-ricerca e' saltata). Il valore e' identico all'alpha-beta perche' la ricerca e' fail-soft: un valore
  <= alpha e' un limite superiore (la mossa non migliora), uno nel mezzo viene sempre ri-cercato.
* **Ordine delle mosse nei nodi interni**: (1) mossa TT; (2) due *killer* per ply (mosse che hanno tagliato in un nodo
  fratello; azzerate a ogni `findBestMove`); (3) le altre per punteggio *history* `[lato][da][a]` (+profondita'^2 a ogni
  taglio beta; dimezzata a ogni `findBestMove`, non azzerata). Selezione "pigra": si estrae la migliore rimasta solo quando
  serve. L'indice mossa si calcola con due `charAt` (niente `Integer.parseInt` come nei getter di `Action`).
* **Alla radice l'ordine non cambia** (miglior mossa precedente in testa, poi ordine di generazione): il tie-break tra
  mosse di pari valore resta quello di prima.

### 10.2 Come e' verificato (`aiplayer/SearchOptimizationsCheck`, white-box)
Per ognuna delle configurazioni PVS / PVS+killer / PVS+history / PVS+killer+history / alpha-beta+killer+history, contro
l'alpha-beta puro (senza TT):
* punteggio e profondita' di radice identici, con `SimpleEvaluator` (valori quasi interi, molti pareggi) e con una rete a
  valori continui; la mossa scelta vale davvero il valore di radice e (a profondita' <= 3) nessun'altra vale di piu'. L'oracolo
  e' un alpha-beta indipendente scritto nel test (non una seconda istanza di `Negamax`, il cui orologio interno interrompe la
  ricerca dopo 2048 nodi: il mio primo oracolo aveva questo difetto e l'ho scoperto perche' falliva solo sugli alberi grandi);
* con TT accesa: 0 divergenze dal confronto senza TT; con una **storia di ripetizione reale** (1585 ripetizioni incontrate);
  con una **istanza riusata** per mosse consecutive (TT e history persistono, le killer no); 120 ricerche a budget da 1 a
  450 ms: mossa sempre legale;
* l'ordinamento e' una **permutazione** delle mosse, con punteggi non crescenti, mossa TT per prima e killer subito dopo;
  `moveIndex` coincide con i getter di `Action` su ~1700 mosse; a profondita' 1 con finestra piena **ogni figlio viene
  valutato**; killer e history dopo una ricerca vera sono ben formate e non vuote.
Passa con piu' seed (2024 con 14 posizioni, 7 con 12) e l'intera suite originale resta verde.

**Mutation test.** 4 bug iniettati in PVS e 6 nell'ordinamento (scambio senza aggiornare i punteggi, mossa duplicata, indice
mossa sbagliato, ciclo che salta l'ultima mossa, history con segno invertito, mossa TT non prioritaria): tutti rilevati. Uno
(`salta l'ultima mossa`) in un primo momento NON lo era: cambia il valore solo se quella mossa e' la migliore, cosa rara, e il
bug stava nel ciclo condiviso da baseline e candidati. Ho aggiunto il controllo "ogni figlio viene valutato" e ora e' rilevato.

### 10.3 Guadagno misurato

Nodi, TT spenta (SearchOptimizationsCheck, 14 posizioni, profondita' 1..4 Simple / 1..3 rete): PVS -31% / -13%;
+killer -92% / -63%; +history -93% / -63%; **PVS+killer+history -94% / -65%**; alpha-beta+killer+history -94% / -67%.
Con TT accesa, Simple: PVS -34%, +killer -89%, +history -91%, **PVS+killer+history -91,5%**; rete -74%.

Tempo e nodi a profondita' fissa, **TT accesa, con warm-up** (`experiments/OrderBench.java`):

| profondita' 5, 12 posizioni, Simple | nodi | tempo | |
|---|---|---|---|
| alpha-beta originale | 15,26M | 15,5 s | |
| PVS | 12,49M | 13,5 s | -18% nodi, -13% tempo |
| PVS + killer | 2,70M | 3,0 s | -82%, -81% |
| PVS + history | 2,49M | 2,8 s | -84%, -82% |
| **PVS + killer + history** | **2,09M** | **2,5 s** | **-86%, -84%** |
| alpha-beta + killer + history | 2,61M | 2,8 s | -83%, -82% |

A profondita' 6 (8 posizioni, senza le versioni non ordinate): PVS+killer+history e' il piu' veloce (-24% di tempo contro
PVS+killer; -21% di nodi contro alpha-beta+killer+history): per questo e' il default.

Profondita' raggiunta a tempo fisso, 2000 ms/mossa, 10 posizioni (`experiments/DepthAtTime.java`):

| | alpha-beta originale | PVS (versione precedente) | **nuova** |
|---|---|---|---|
| Simple | 5,00 (968k nodi/s) | 5,00 (957k) | **6,00** (918k) |
| rete di default | 4,70 (406k) | 4,80 (407k) | **5,30** (440k) |

Partite a tempo uguale (`EqualTimeArena`, 100 ms/mossa, 60 partite, IC95 ~ +-12 punti):

| A | contro | A: + = - | punteggio A | profondita' media A / B |
|---|---|---|---|---|
| Simple, ricerca nuova | Simple, `ab:` originale | 45 0 15 | **75,0%** [63-84%] | 4,01 / 3,54 |
| rete di default, ricerca nuova | rete, `ab:` originale | 33 2 25 | 56,7% [44-68%] | 3,76 / 3,15 |
| rete di default, ricerca nuova | rete, `pvs:` (solo PVS) | 40 2 18 | 68,3% [56-79%] | 3,55 / 3,31 |
| (PVS contro alpha-beta, senza ordinamento) | | 29 0 31 / 31 0 29 | 48,3% / 51,7% | +0,07 / +0,11 ply |

Lettura onesta:
* **Il grosso del guadagno viene da killer + history, non da PVS**: PVS da solo vale ~-20% di nodi ma ~+0,1 ply e nessuna
  differenza di forza misurabile; con killer+history i nodi calano dell'80-90% e si guadagna ~1 ply a 2 s. Il motivo e' che
  l'ordinamento di partenza era pessimo (mosse nell'ordine di scansione dei pezzi, senza alcuna informazione): per questo il
  risparmio e' cosi' grande. Con un ordinamento gia' buono sarebbe stato minore.
* **Con la rete il guadagno in partita e' meno netto** (56,7% e 68,3%, intervalli larghi e non del tutto coerenti tra loro):
  leggili come "positivo", non come numeri precisi. Con Simple e' chiaro (75%).
* Con un ordinamento cosi' efficace PVS conta meno: senza TT alpha-beta+killer+history usa meno nodi di PVS+killer+history
  (434k contro 481k su Simple); con TT accesa, che e' il caso di gioco, PVS vince (404k contro 493k).
* I nodi/s calano di ~5% su Simple per il costo dell'ordinamento (con la rete la misura e' invece salita, 440k contro 407k: e'
  rumore/profondita' diversa, non un effetto reale da attribuire all'ordinamento).
* I campioni di posizioni vengono da gioco casuale/tattico, non da partite reali di gara; a 60 s/mossa (profondita' 6+) il
  rapporto puo' cambiare e **non l'ho misurato**.

Una partita dal vivo col server reale (rete + ricerca nuova al Bianco contro il client Simple) l'ha **persa in 17 mosse**, con
profondita' 4-7 e nessuna mossa rifiutata. E' una sola partita (e nelle prove precedenti la rete aveva vinto sia da Nero sia da
Bianco), non dice nulla sulla forza, ma lo riporto.

### 10.4 Cose non fatte
Killer trasportate (shift di 2 ply) tra una mossa e la successiva; penalita' (malus) alle mosse provate prima del taglio;
catture e mosse del Re verso il bordo per prime; countermove; ordinamento della radice per punteggio dell'iterazione
precedente (cambierebbe il tie-break, quindi andrebbe fatto con attenzione ai test di confronto con/senza TT); aspiration
windows. Il test e' pensato per coprirle senza modifiche (sono tutte solo ordinamento).

Confronto a tempo uguale tra ricerche: `EqualTimeArena <A> <B> 30 100` con prefissi `ab:` (alpha-beta originale), `pvs:` (solo
PVS), `nopvs:` (alpha-beta con killer+history); senza prefisso = ricerca completa.

---

## 11. Contesto per riprendere in una nuova chat (aggiornato)

Stato: feature v2, `MLP` v2 (tronco condiviso + 2 head scelte da chi muove), `NeuralEvaluator` con tier di feature,
`SelfPlayTrainer` portato, `SupervisedPretrainer`, `EqualTimeArena`, pesi supervisionati in `pretrained_weights/`; `Negamax` con
PVS + killer + history, `SearchOptimizationsCheck`. Regressione completa verde. **Prossimi passi naturali**: (a) TD lungo dal
warm start (`resumeFrom=pretrained_default.txt`), controllando ogni qualche migliaia di episodi con `EqualTimeArena` contro Simple
e contro il checkpoint precedente; (b) misurare la nuova ricerca a 60 s/mossa; (c) le leve del §9.5 e §10.4.
Non fidarti di nulla finche' non l'hai verificato: ogni modifica a regole, ricerca, hash o feature va accompagnata da un test
indipendente (pattern usato qui: oracolo del motore + simmetrie + mutation test; e un mutante che i test NON rilevano e'
un buco del test, non del mutante).
