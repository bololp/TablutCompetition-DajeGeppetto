CONFRONTO CACHE: versione originale (solo hash) vs versione finale (chiave esatta)
===============================================================================

CONTENUTO (percorsi relativi alla radice del repo)
  Tablut/src/it/unibo/ai/didattica/competition/tablut/aiplayer/HashCachingEvaluator.java
      La versione ORIGINALE della cache (identita' = solo hash Zobrist, 16 B/voce), rinominata per poter convivere con la finale.
  Tablut/src/it/unibo/ai/didattica/competition/tablut/aiplayer/CacheCompareBench.java
      Banco di prova: nessuna cache / HASH 2^22 (64 MiB) / ESATTA 2^21 (64 MiB) / ESATTA 2^22 (128 MiB), nella stessa sessione.

PREREQUISITO
  La versione finale (CachingEvaluator) e' nella patch cache-and-order-rec.patch / cache-and-order-rec.zip consegnata prima:
  applicala (patch -p1 < cache-and-order-rec.patch) o copia i suoi file. Il progetto originale non modificato e' lo zip che hai caricato.

COMPILAZIONE (dalla cartella Tablut; l'ultimo comando aggiunge i due file di questo archivio)
  OUT=build_cmp; mkdir -p $OUT
  javac -source 8 -target 8 -encoding iso-8859-1 -d $OUT -cp "lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar" $(find src -name "*.java")
  (se javac non e' nel PATH: java -m jdk.compiler/com.sun.tools.javac.Main ... con le stesse opzioni)

ESECUZIONE (da Tablut, cosi' pesi_v4/ si trova; altrimenti -Dtablut.weights=/percorso/pesi_v4/ con la barra finale)
  CP="$OUT:lib/gson-2.2.2.jar:lib/commons-cli-1.4.jar"; C=it.unibo.ai.didattica.competition.tablut.aiplayer.CacheCompareBench
  java -Xss4m -Xmx3g -cp "$CP" $C micro                       # costo del singolo lookup: hit / miss+scrittura
  java -Xss4m -Xmx3g -cp "$CP" $C game mlp 4 7                # MLP d=4, 7 ripetizioni a ordine ruotato
  java -Xss4m -Xmx3g -cp "$CP" $C game mlp 5 4
  java -Xss4m -Xmx3g -cp "$CP" $C game cnn 3 5
  java -Xss4m -Xmx3g -cp "$CP" $C game simple 5 5
  (argomenti di "game": evaluator, profondita', [ripetizioni=3], [plyMax=60])

COME LEGGERE I RISULTATI
  * "stesse mosse e stessi nodi in tutte le modalita'" deve dire "si": le cache non devono cambiare nulla di quello che il motore
    decide; se dice NO c'e' un problema da indagare (e la causa non e' il rumore).
  * Il confronto onesto e' a MEMORIA UGUALE: HASH 2^22 (64 MiB) contro ESATTA 2^21 (64 MiB). ESATTA 2^22 (128 MiB) e' il default del client.
  * Usa il minimo e la mediana di >= 5 ripetizioni; con poche ripetizioni il rumore (anche +-10%) copre le differenze fra le due cache.
  * "micro" usa un evaluator interno costante: misura solo l'overhead della cache (codifica chiave, accesso in memoria, scrittura).
    L'accesso casuale in tabella dipende dalla cache L2/L3 e dalla memoria della TUA macchina: e' la parte che piu' differisce.

COSA HO ESEGUITO IO
  Solo un giro di prova di funzionamento (sandbox a 1 core, JDK 21): "micro" e "game mlp 4 2 40". Le 4 modalita' danno le stesse
  mosse e gli stessi nodi (859.184). Con 2 ripetizioni i tempi sono rumorosi e NON vanno presi come misura; indicazione solo
  qualitativa dal micro: la versione a chiave esatta costa piu' per lookup (hit casuale ~216-251 ns contro ~169 ns, miss+scrittura
  ~255-283 ns contro ~163 ns, in quella sessione). Il confronto vero va fatto sulla tua macchina.
