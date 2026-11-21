# Giocatore genetico per Tablut

Questo documento descrive il giocatore basato su un algoritmo genetico (GA) aggiunto al progetto, come si compila, si addestra, si testa e **come si misura**
contro il riferimento `TablutNegamaxClient` + `pesi_nnue/trained_v6.nnue`.

> **Stato dei risultati, in chiaro.** Il codice, i test e il protocollo di misura sono completi e verificati (sezione 7). **Non esiste ancora un genoma
> addestrato e validato**, e quindi **non si afferma che il giocatore eguagli o superi il riferimento**: l'ambiente in cui ho lavorato aveva 1 core e le
> sessioni non consentono corse lunghe, mentre servono centinaia di partite per una conclusione statistica. Per costruzione, finche' il cancello non accetta un
> genoma il giocatore **coincide con il riferimento** (stessa rete bit per bit); la sezione 6 spiega come ottenere e misurare un eventuale miglioramento.

## 1. Cosa ho verificato del progetto

| Tema | Fatto verificato (file / funzione) |
|---|---|
| Linguaggio / build | Java, sorgenti in `Tablut/src`, `build.xml` (Ant), librerie in `Tablut/lib` (Gson 2.2.2, commons-cli 1.4). Compila con `javac --release 8` (330 classi in ~10 s). Nessun JUnit: i test sono classi `*Check` con `main` (ultima riga `... OK`). |
| Regole | `domain/GameAshtonTablut` (server `-r 4`, **default**). Il motore veloce e' `engine/FastState`; l'oracolo indipendente di legalita' e' `aiplayer/AshtonRules.generateMoves`. |
| Protocollo | `client/TablutClient`: connessione TCP (`util/Configuration`: porta 5800 Bianco, 5801 Nero), messaggi = lunghezza + UTF-8 (`util/StreamUtils`), nome dichiarato all'avvio, poi stato JSON (Gson) -> `Action` JSON. Il server invia lo stato a **entrambi** i client dopo ogni mossa. |
| Come il client riceve/restituisce | `client/TablutNegamaxClient.run()`: `read()` bloccante; se il turno e' suo chiama `MoveSearcher.findBestMove(FastState, budget, PositionHistory)` e scrive l'`Action`; altrimenti stampa "In attesa della mossa dell'avversario..." e torna in `read()`; stati terminali -> messaggio e `System.exit(0)`. |
| Tempo (server) | `server/Server`: `-t` default **60 s**; il controllo e' un ciclo con `Thread.sleep(1000)` (granularita' 1 s); scadenza -> "Timeout!!!!", sconfitta del giocatore e fine partita. Mossa illegale -> errore; con `-e` default 0 e' sconfitta. |
| Tempo (client) | `TablutNegamaxClient.defaultMarginMillis(timeout)`: margine di sicurezza **1500 ms se timeout >= 20 s, 300 ms altrimenti**; la ricerca (`Negamax`) usa `budget - margine`. |
| Ricerca | `aiplayer/Negamax` (iterative deepening, TT, ordinamento, Lazy SMP; `Negamax.defaultThreads()` dipende dai core). **Nessun pondering**: alla fine di `findBestMove` gli helper vengono fermati (`ctl.stop`, attesa massima 150 ms); il pool e' daemon e resta parcheggiato. |
| Valutazione NNUE | `NnueNetwork` (243 -> 64 -> 16 -> 1, 2 teste per lato al tratto, **17 730 parametri interi**, immutabile, validazione di overflow nel costruttore, `save/load`), `NnueEvaluator` + `NnueSession` (aggiornamento incrementale), `EvaluatorLoader.load(path)`. `trained_v6.nnue`: 57 634 byte, **esiste sul disco** ma non e' tracciato da git (come `trained_v5.nnue`). |
| Misura di forza esistente | `training/NnueSelfPlay.playGame(white, black, cfg, seed)` (pubblica, deterministica a profondita' fissa, regole corrette via `GameTracker`). `EqualTimeArena` **non** traccia la ripetizione (tetto 160 semi-mosse): per questo non l'ho usata. |

**Discrepanze preesistenti** (non toccate; segnalate perche' influenzano come si lancia il riferimento):
* `NegamaxClientArgsCheck` (1 FAIL) e `ClientDefaultsCheck` (4 FAIL) **falliscono sul codice originale** (verificato eseguendoli prima di ogni modifica): si aspettano `exactkey` spento di default, ma `TablutNegamaxClient.DEFAULT_EXACT_KEY` e' `true` (commit "Enably exactkey by default"). Anche il Javadoc del client (TT di default 2^22, exactkey false) e' rimasto indietro: il `main` usa 2^26, ridotto da `fitTtToHeap` in base all'heap (con `-Xmx1g` il banner mostra 2^24 voci, 640 MB).
* Il client dichiara al server il nome `negamax`; i log del server non distinguono quindi riferimento e giocatore genetico (il secondo stampa un banner `Genetic client - ...` all'avvio).

## 2. Cosa e' riusabile e cosa ha senso evolvere

Riutilizzato **senza modifiche**: tutto il client di rete e la ricerca (`TablutNegamaxClient`, `Negamax`, TT, Lazy SMP, gestione del tempo), `NnueNetwork/NnueEvaluator/NnueSession`, `NnueSelfPlay.playGame`, `GameTracker`, `AshtonRules`. **La parte genetica non modifica ne' rimuove file esistenti** (solo file nuovi). Le ottimizzazioni di ricerca (mossa nulla, LMR, aspirazione) sono una consegna successiva e separata, descritta in `README_PRUNING.md`: modificano in modo additivo 5 file esistenti, con comportamento di default invariato.

Alternative scartate, con motivo:
* **GA "dal vivo" durante la mossa** (evolvere sequenze di mosse in 60 s): getterebbe via la profondita' di alfa-beta + NNUE (con 1 thread ho misurato profondita' 8 e ~94 M nodi in 58,5 s) ed e' certamente piu' debole.
* **Evolvere tutti i 17 730 pesi**: dimensione eccessiva per una GA con partite come fitness; esiste gia' l'addestramento a gradiente (`NnueTrainer`) con cui e' stato prodotto `trained_v6`.
* **Evolvere parametri di ricerca** (margine, TT, ordinamento): cambiano velocita', non il giudizio; il riferimento e' gia' messo a punto e `Negamax` e' verificato nodo per nodo: non lo tocco.

Scelta: la GA ottimizza **direttamente il tasso di vittoria** su un sottoinsieme **piccolo** di parametri della rete (testa e bias), partendo dal riferimento. Il gradiente minimizza un errore quadratico su etichette; la GA misura cio' che interessa (vincere). E' un'ipotesi: **non e' garantito** che esista un miglioramento di questo tipo.

## 3. L'individuo e la sua codifica (`training/GaGenome`)

Un individuo e' un vettore di **geni reali in [-1, 1]**. Il gene `g_i` diventa lo scostamento intero `d_i = round(g_i * R_i)` **sommato** al parametro quantizzato di `trained_v6`. Gruppi evolvibili (opzione `groups=`):

| Gruppo | Parametri | Geni (L2=16, H=64) | Ampiezza `R_i` (derivata dalla base) |
|---|---|---|---|
| `head` | `l2W` (16) + `l2B` (1) per ciascuna delle 2 teste | 34 | `l2W`: 0,5 * sd dei `l2W` = **28**; `l2B`: **1638** (0,1 logit) |
| `l1b` | bias `l1B` (16) per testa | 32 | 0,25 * sd = **2382** |
| `ftb` | bias `ftB` (64) | 64 | 0,25 * sd = **36** |

Default `head,l1b` = **66 geni**. Le ampiezze si calcolano dalla deviazione standard dei parametri della base (valori verificati su `trained_v6`: 28 / 1638 / 2382 / 36), non sono scelte a mano. L'architettura, la quantizzazione e il formato `.nnue` restano quelli del riferimento: la decodifica produce una normale `NnueNetwork`.

**Elemento neutro:** genoma tutto a zero = rete identica **bit per bit** alla base (testato: parametri, impronta CRC32, valutazioni di 300 posizioni, risultati di ricerca). E' cio' che permette di garantire "mai peggio del riferimento" (sezione 4.6).

File `.genome` (testo, versionato `TABLUT_GA_GENOME v1`): impronta della base, gruppi, geni (`Double.toString`, ritorno esatto), CRC32. Un genoma per un'altra base, troncato, alterato, fuori range o con versione diversa e' **rifiutato** (`IOException`).

## 4. L'algoritmo genetico (`training/GaTrainer`)

1. **Popolazione.** `pop` individui (default 24). Generazione 0: il genoma neutro (= riferimento) + individui gaussiani attorno al riferimento (ampiezza individuale in [0,25, 1] * `sigma`) + eventuali genomi di `init=` (partenza a caldo).
2. **Fitness.** Ogni generazione estrae `pairs` semi di apertura **uguali per tutti** (numeri casuali comuni: meno varianza nei confronti). Ogni individuo gioca contro il **riferimento** una *coppia* per seme (`GaMatch.playPair`): stessa apertura casuale di `open` semi-mosse, colori invertiti, **profondita' fissa** `depth`, regole vere (`GameTracker`), un thread di ricerca, stessa configurazione per i due lati (cambia solo la rete). `raw` = media dei punteggi delle coppie (1/0,5/0), 0,5 = pari al riferimento. Fitness di selezione = `raw - lambda * media(geni^2)`: una **pressione di parsimonia** che a parita' di forza preferisce l'individuo piu' vicino al riferimento e frena la deriva indotta dal rumore. Due reti identiche fanno **esattamente 0,5** (partite identiche, colori scambiati): quindi gli individui neutri valgono 0,5 senza giocare. I semi cambiano a ogni generazione e gli elitari vengono **rivalutati** (mai fitness "fortunata" di una generazione passata).
3. **Selezione.** Elitismo (`elite`, default 2) + torneo (`tourn`, default 3) per i genitori.
4. **Crossover.** BLX-alfa (`alpha`=0,3) con probabilita' `pc`=0,8: gene del figlio uniforme in [min - alfa*d, max + alfa*d], troncato a [-1,1].
5. **Mutazione.** Per gene con probabilita' `pm` (default max(0,05; 3/n)), rumore gaussiano di deviazione `max(sigmaMin, sigma * sigmaDecay^gen)` (0,15 -> 0,02), troncato; un figlio che sarebbe copia del genitore muta un gene a caso. `immigrants` (1) individui casuali per generazione.
6. **Arresto.** (a) `gens` generazioni; (b) `budget` secondi (fra generazioni, stimando la durata della prossima); (c) stagnazione: la migliore fitness *media su tutte le valutazioni* (almeno `minevals`=2) non migliora di `mindelta` per `patience` generazioni.
7. **Cancello di accettazione.** Il **campione** e' scelto con le sole statistiche di addestramento (miglior media liscia). E' poi giocato **una sola volta** contro il riferimento su **semi disgiunti** da quelli di addestramento, `valpairs` coppie (default 100), a profondita' `valdepth` (default `depth+1`: controlla che il vantaggio regga a una profondita' diversa). Un solo candidato validato = nessun bias di selezione sul set di validazione. Verdetto da `GaMatch.verdict` sull'IC 95% (normale, sulle medie delle **coppie**, che sono indipendenti; le due partite di una coppia no): `SUPERIORE` se il limite inferiore e' > 0,5; `INFERIORE` se il superiore e' < 0,5; `NON_INFERIORE` se l'inferiore >= 0,5 - `margin`; `INSUFFICIENTE` sotto `minpairs`=30 coppie; altrimenti `NON_DIMOSTRATO`. **`OUT.final.nnue` e' il campione solo se `SUPERIORE`** (con `acceptni=1` anche `NON_INFERIORE`); altrimenti e' **una copia bit per bit del riferimento**. Un successo si dichiara solo con evidenza; in assenza, la rete finale e' il riferimento.
8. **Riproducibilita' e ripresa.** Ogni operazione casuale usa un `Random` derivato da (seme, generazione) e le partite a profondita' fissa sono deterministiche: stesso comando = stessi risultati con qualunque `threads`. `OUT.ckpt` (con CRC) dopo ogni generazione; `resume=1` riparte producendo lo **stesso** risultato di una corsa ininterrotta (anche estendendo `gens`); un checkpoint troncato, alterato o di una configurazione diversa e' rifiutato.

**Il cancello a profondita' fissa NON e' la prova di forza a 60 s.** `verdict.txt` lo ripete esplicitamente: la prova finale e' il protocollo della sezione 6.

## 5. Quando e dove avviene l'addestramento, e cosa succede in torneo

* **Prima del torneo, offline, su una macchina di sviluppo** (`GaTrainer`, CLI sotto). Produce `OUT.genome`, `OUT.champion.nnue`, `OUT.final.nnue`, `OUT.verdict.txt`, `OUT.log`.
* **Durante il torneo non si evolve nulla.** `client/TablutGeneticClient` applica **una volta**, all'avvio, il genoma alla base, scrive la rete in un file temporaneo (cancellato all'uscita) e **delega a `TablutNegamaxClient.main`** con quel file e `eval=nnue`. Non duplica nessun parametro del riferimento: `threads=`, `tt=`, `margin=`, ... funzionano uguale e il confronto e' per costruzione a parita' di configurazione. In alternativa si puo' passare `OUT.final.nnue` direttamente al client di riferimento (`eval=nnue`): e' la stessa rete.
* **Robustezza all'avvio:** con `strict=false` (default) un genoma mancante/corrotto/per un'altra base produce un `ATTENZIONE` e si gioca con la base (un avvio fallito in torneo e' peggio); `strict=true` esce con codice -1. Base illeggibile: errore sempre.

## 6. Rispetto dei 60 secondi e del divieto di calcolo nel turno avversario

Il giocatore **e'** il riferimento con un'altra rete: eredita la sua gestione del tempo e verifico che la proprieta' valga davvero.

* **Tempo.** Budget di ricerca = `timeout - margine` (1500 ms a 60 s). **Misurato** (`TablutGeneticClientCheck -Dga.long=true`, configurazione di default del client, 1 thread su questa macchina): mossa in **58 500 ms** (profondita' 8, ~93,8 M nodi), cioe' 1,5 s sotto il limite. Con timeout 3 s e 4 thread Lazy SMP: la mossa piu' lenta 2709 ms su 4 posizioni. Con server vero e timeout 2 s: tempo di ricerca massimo 1703-1704 ms su 25+25 mosse, zero timeout.
* **Nessun calcolo nel turno avversario.** Il client resta bloccato su `read()` (nessun pondering, nessun thread di ricerca attivo). **Misurato:** durante la ricerca i thread helper consumano ~870-890 ms di CPU (la misura "vede" il lavoro); nei 1500 ms successivi di "turno avversario" **0 ms** di CPU nei thread Java e 0-10 ms nell'intero processo (JIT/GC inclusi); nessun thread esegue codice di `Negamax`. Il GA non gira mai in torneo.
* **Cosa non posso garantire:** la CPU di GC/JIT della JVM e' fuori dal controllo del codice (nel test e' risultata trascurabile); la latenza reale di rete/avvio JVM della macchina del torneo e' ignota (il margine di 1500 ms la assorbe se simile a quella misurata).

## 7. Comandi riproducibili

Dalla cartella `Tablut`, con un JDK (qui JDK 21 con `--release 8`):

```sh
# 1) compila ed esegue TUTTI i test del giocatore genetico (~85 s; porte 5800 e 5801 libere)
sh tools/run_genetic_checks.sh                 # LONG=1 aggiunge la prova a 60 s; RUN_EXISTING=1 anche NnueCheck/NnueClientCheck

# 2) addestramento (esempio realistico, su una macchina multi-core; 1 core qui sarebbe troppo lento)
java -Xmx2g -cp /tmp/tablut_ga_build:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar \
  it.unibo.ai.didattica.competition.tablut.training.GaTrainer \
  base=pesi_nnue/trained_v6.nnue out=ga/run1 groups=head,l1b pop=24 gens=30 pairs=6 depth=4 \
  valpairs=100 threads=8 seed=1 budget=14400          # riprendere: aggiungere resume=1

# 3) gioco (stesse opzioni del riferimento): colore, timeout, ip, poi chiave=valore
java -Xmx4g -cp ... it.unibo.ai.didattica.competition.tablut.client.TablutGeneticClient \
  WHITE 60 localhost genome=ga/run1.genome base=pesi_nnue/trained_v6.nnue
```

Test aggiunti (tutti con `main`, nello stile del repository):

| Classe | Verifiche | Cosa copre |
|---|---|---|
| `training/GaGenomeCheck` | 2259 | identita' bit per bit; **ogni** gene muove esattamente il parametro del suo nome della quantita' attesa (oracolo basato sui nomi, 3 basi incl. `trained_v6`); saturazione; base immutata; estremi; **formato file** (19 difetti rifiutati con `IOException`); compatibilita' `.nnue` / `EvaluatorLoader`; ricerca incrementale = completa con reti evolute. |
| `training/GaTrainerCheck` | 88 | operatori (BLX, mutazione, torneo, sigma); semi di validazione disgiunti; **convergenza** su un paesaggio sintetico con ottimo noto (0,693 -> 0,993, contro 0,795 di una ricerca casuale a pari valutazioni) e con rumore; determinismo; arresto per `gens` / `budget` / stagnazione; elitismo; **ripresa identica** e checkpoint corrotto/non coerente rifiutato; statistica e verdetto con limiti inclusi; **file di uscita** (rifiuto = copia della base); fitness vera (neutri = 0,5 senza giocare); arena identica con 1 o 3 thread; l'arena distingue `trained_v6` da una rete casuale; corsa completa con partite vere. |
| `client/TablutGeneticClientCheck` | 140 (141 con `-Dga.long`) | fallback/strict del genoma; argomenti delegati; **legalita'** (decine di mosse, almeno 60 asserite, di reti evolute casuali, entrambi i colori, contro `AshtonRules`); **regressione** (genoma neutro = stessa mossa/punteggio/nodi/profondita' della base); **tempo** con la stessa costruzione di ricerca del client reale; **quiete nel turno avversario** (CPU di tutti i thread); **protocollo reale**: il client vero in un sottoprocesso contro un finto server (Bianco e Nero, genoma valido e corrotto) con lo scambio del server vero; `strict=true` -> codice -1. |

**Qualita' dei test (test per mutazione).** Ho introdotto a mano 16 difetti plausibili uno alla volta (ruolo scambiato in `decode`, gruppo `ftB` ignorato, `floor` al posto di `round`, checksum non verificato, elitismo ridotto, semi di validazione uguali a quelli di addestramento, verdetto `>=` al posto di `>`, accettazione di `NON_DIMOSTRATO`, rifiuto che copia il campione, budget ignorato, BLX senza troncamento, sigma senza decadimento, stagnazione mai raggiunta, popolazione non ripristinata nella ripresa, `genome=` non rimosso dagli argomenti, `strict` ignorato): **16 su 16 hanno fatto fallire il test pertinente**. Il test ha inoltre scoperto un bug *reale* della mia prima versione (il `budget` non veniva mai applicato), corretto. Limite: sono i difetti che ho saputo immaginare.

## 8. Protocollo di torneo per misurare il giocatore contro il riferimento

Obiettivo: stabilire con evidenza se il candidato (`base.nnue+genoma`) e' **non inferiore** o **superiore** a `trained_v6` **in condizioni di gara**. Regole del protocollo:

1. **Semi nuovi.** Mai quelli usati in addestramento/cancello (`seed=` della GA). Si usa `SEED_BASE` diverso (default 9001).
2. **Coppie con colori invertiti** (il Nero e' avvantaggiato: secondo la documentazione NNUE del progetto vince ~78% nell'auto-gioco) e aperture casuali (le partite quasi deterministiche del server, partendo sempre dalla posizione iniziale, si ripeterebbero).
3. **Statistica sulle coppie**, non sulle partite. Dimensionamento: per "limite inferiore dell'IC > 0,5" con potenza 80% su un vantaggio vero `delta` servono `n ≈ ((1,96 + 0,84) * s / delta)^2` coppie; con **s ≈ 0,3** (stima grezza da 12 coppie: da riverificare con `sd delle coppie` stampata da `GaMatch`) e `delta` = 0,05 (55%) sono **~280 coppie**; per `delta` = 0,03 ~780. Con meno coppie il verdetto sara' quasi sempre `NON_DIMOSTRATO`: e' corretto, non un difetto.
4. **Piu' livelli, in ordine di costo:** (fase 1) profondita' fissa 4, 5, 6, riproducibile; (fase 2) **tempo reale uguale** per mossa, un thread per lato, piu' vicino alla gara; (fase 3) **server vero e client veri** con il timeout di gara (conferma di protocollo/tempo, non strumento statistico). Un candidato che vince a profondita' fissa e perde a tempo reale **non** e' accettabile: il tempo reale ha priorita'.
5. **Un solo candidato per confronto**; se se ne provano k, correggere la soglia (es. Bonferroni, z piu' alto con `z=`).
6. Hardware: una partita per core fisico (e' una misura di tempo); stessa macchina, stessi flag JVM (`-Xmx`) per i due lati.

```sh
sh tools/ga_tournament.sh ga/run1.champion.nnue            # NB: passare il CAMPIONE, non final.nnue (che puo' essere il riferimento)
# oppure con SPEC base+genoma:
sh tools/ga_tournament.sh pesi_nnue/trained_v6.nnue+ga/run1.genome
SEED_BASE=9101 PAIRS_FIXED=300 DEPTHS="4 5 6" PAIRS_TIMED=60 MS=5000 THREADS=8 sh tools/ga_tournament.sh ...
sh tools/ga_server_match.sh 6 60 ga/run1.genome            # partite col server vero (colori alternati)
```

Criterio suggerito: **superiore** solo se `SUPERIORE` a profondita' fissa *e* almeno `NON_INFERIORE` a tempo reale con coppie sufficienti; **non inferiore** se `NON_INFERIORE` a entrambi. Altrimenti si gioca col riferimento.

## 9. Cosa ho eseguito, e cosa no

Eseguito e riportato sopra: tutta la batteria di test (verde, comprese le regressioni esistenti `NnueCheck` 106 174 verifiche e `NnueClientCheck` 50); la prova a 60 s (58 500 ms); una partita **intera** col **server vero** e client veri a timeout 2 s (genoma di prova vs riferimento: 25 mosse per lato, tutte accettate, nessun timeout; vittoria del Nero = riferimento, **una partita non significa nulla**); una mini-evoluzione reale su `trained_v6` (`pop=8 gens=3 pairs=2 depth=2`, 4 s) per provare la pipeline sui pesi veri: il campione faceva 0,75 sulle 2 coppie di addestramento ma **37,5% [13-62]** in validazione (4 coppie, profondita' 3) -> `RIFIUTATO`, `final.nnue` = riferimento (57 634 byte identici). E' un esempio di come il rumore inganni con poche partite e del perche' esiste il cancello.

**Non eseguito / non provato:** nessuna evoluzione con budget realistico, nessun genoma validato, nessuna misura a tempo reale contro il riferimento, nessuna partita multi-thread sul server in gara. **Non so** quindi se per questa base esista un miglioramento ottenibile con questo spazio di ricerca.

## 10. Informazioni mancanti e rischi

* Hardware del torneo (core, RAM), versione di Java e flag JVM: ignoti. (`Negamax.defaultThreads()` vale `min(DEFAULT_THREADS, core)` e si forza con `-Dtablut.threads=N`: il commento del codice cita "4 su una macchina a 4 core" solo come esempio, non descrive la macchina di gara.) `defaultThreads()` e la dimensione della TT dipendono da core/heap: **riferimento e candidato vanno lanciati con gli stessi flag**.
* Non so se il torneo lancia i client con `timeout` uguale al `-t` del server (il margine di sicurezza dipende dall'argomento `timeout` del client, non dal server).
* Gli avversari reali (altre squadre) sono ignoti: la fitness e' solo contro il riferimento e potrebbe sovradattarsi a lui. Estensione possibile (non implementata): fitness su un insieme di avversari (es. `trained_v5`, `trained_v4`).
* La fitness a profondita' bassa puo' non trasferirsi alla gara a 60 s (profondita' ~8+): per questo il cancello usa `depth+1` e il protocollo richiede la fase a tempo reale. Usare `depth` piu' alto in addestramento costa molto di piu'.
* Stime di costo dell'addestramento (**non misurate**): ho misurato solo il tempo per mossa dalla posizione iniziale (profondita' 4: ~0,13 s; 5: ~0,65 s; 6: ~1,7 s). Con ~10 s/partita a profondita' 4 (ipotesi), la configurazione di default (23 individui non neutri x 6 coppie x 2 partite) vale ~46 minuti-core per generazione.
* Il nome dichiarato al server e' `negamax` (vedi sezione 1).

## 11. File aggiunti dalla parte genetica (nessun file esistente modificato)

```
README_GENETIC.md
Tablut/src/it/unibo/ai/didattica/competition/tablut/training/GaGenome.java        genoma, codifica, formato file
Tablut/src/it/unibo/ai/didattica/competition/tablut/training/GaMatch.java         arena a coppie, statistica, verdetto, CLI
Tablut/src/it/unibo/ai/didattica/competition/tablut/training/GaTrainer.java       algoritmo genetico, cancello, checkpoint, CLI
Tablut/src/it/unibo/ai/didattica/competition/tablut/client/TablutGeneticClient.java   giocatore (applica il genoma e delega al riferimento)
Tablut/src/it/unibo/ai/didattica/competition/tablut/training/GaGenomeCheck.java
Tablut/src/it/unibo/ai/didattica/competition/tablut/training/GaTrainerCheck.java
Tablut/src/it/unibo/ai/didattica/competition/tablut/client/TablutGeneticClientCheck.java
Tablut/tools/run_genetic_checks.sh   Tablut/tools/ga_tournament.sh   Tablut/tools/ga_server_match.sh
```
`build.xml` compila tutto `src`: le nuove classi sono incluse senza altre modifiche.

## 12. Ricerca piu' profonda

`README_PRUNING.md` descrive le tecniche opzionali `prune=` (mossa nulla, LMR, finestre di aspirazione). Il `TablutGeneticClient` inoltra l'opzione al client di riferimento; il cancello e le misure di questo documento usano la ricerca storica.
