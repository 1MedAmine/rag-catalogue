# V12.1 — Candidats classés et diagnostic optionnel

## Objectif

Retourner jusqu'à cinq références catalogue pertinentes, classées du plus proche au moins proche, tout en conservant la sortie courte et les options existantes. Le diagnostic détaillé reste désactivé par défaut et s'active avec `--diagnostic`.

## Résultat principal

Le champ `reference` conserve le meilleur candidat pour la compatibilité. Le champ `candidates` contient chaque référence validée avec son rang, sa variante, son niveau de correspondance, ses pages, sa justification et ses différences utiles. Plusieurs références valides produisent le statut `multiple_candidates`.

## Classement

Le LLM propose une liste classée à partir du cahier complet, de la provenance et des passages du catalogue. Une variante Deluxe ou premium ne peut être placée devant une variante Standard sans justification issue du besoin et des preuves catalogue. Les égalités techniques doivent être conservées et expliquées.

## Validation

Chaque candidat est validé séparément. Une référence explicite doit être visible dans les passages. Une référence construite doit posséder tous ses segments, dans l'ordre, avec les pages et preuves correspondantes. Les candidats non prouvés sont retirés sans invalider les autres.

## Diagnostic optionnel

Sans `--diagnostic`, la sortie principale reste compacte. Avec `--diagnostic`, un second fichier JSON contient le besoin extrait, les requêtes, les modèles, le cache dense, les rangs et scores de récupération, les scores de reranking, les pages, chunks et extraits transmis au LLM. `--sortie-diagnostic` permet de choisir son chemin. `--details` conserve les preuves et segments dans le résultat principal sans activer le diagnostic complet.

## Compatibilité

Toutes les options V12 restent disponibles : provenance explicite ou automatique, désactivation de l'embedding, désactivation du reranking, validation des preuves, paramètres `candidate-k`, `top-k` et cache.
