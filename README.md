# RAG catalogue industriel NVIDIA — V14.0.2 + Cahier V4.1 MAX

V14 reçoit un **cahier technique complet** en PDF, TXT, Markdown ou JSON, puis recherche et classe jusqu’à cinq références pertinentes dans un catalogue PDF cible.

La nouveauté principale est une navigation qui imite le raisonnement d’un lecteur humain :

```text
comprendre le besoin
→ lire la structure du catalogue
→ choisir la grande section
→ entrer dans la sous-section la plus précise
→ chercher profondément dans cette zone
→ élargir au catalogue entier seulement si nécessaire
```

Lorsque le catalogue n’a pas de structure fiable, V14 bascule automatiquement vers le RAG global normal.

## Modèles par défaut

```text
Embedding  : nvidia/nemotron-3-embed-1b
Reranking  : nvidia/llama-nemotron-rerank-1b-v2
Génération principale : nvidia/llama-3.3-nemotron-super-49b-v1.5
Repli génération      : nvidia/nemotron-3-super-120b-a12b
```

## Ce que V14 ajoute

- lecture des sommaires textuels, même sans signets PDF ;
- reconstruction parent → enfant à partir des plages de pages ;
- score de fiabilité de la structure du catalogue ;
- choix d’une section principale et de sous-sections secondaires ;
- recherche locale approfondie dans la route sélectionnée ;
- ajout automatique des pages voisines ;
- mesure de la qualité des preuves locales ;
- repli global lorsque la structure, le routage ou les preuves sont insuffisants ;
- recherche exacte globale toujours active pour les références et codes rares ;
- diagnostic explicite du chemin suivi dans le catalogue ;
- conservation de tous les correctifs V13.1.1 : alternatives proches, multi-candidats, OCR, réparation JSON, raisonnement et provenance.


## Cahier V4.1 MAX

Pour une demande limitée à une référence, le générateur ne considère plus la demande comme techniquement vide. Il distingue les contraintes explicites, les contraintes dérivées obligatoires, les critères de classement, les options et les informations à confirmer.

Une contrainte dérivée est conservée seulement si la référence, la valeur et leur relation structurelle sont prouvées. Un audit vérifie qu'une référence exacte possède au moins une fonction et une autre dimension structurante avant publication. Le modèle 49B est utilisé en premier ; le 120B est appelé une seule fois si cet audit échoue.

Le RAG s'utilise toujours avec un seul fichier principal :

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\resultats_cahier_des_charges\cahier_produit_1.txt" `
  --catalogue ".\catalogue.pdf" `
  --profil-catalogue huge `
  --raisonnement approfondi `
  --diagnostic `
  --details `
  --sortie ".\resultats\produit_1_v14_0_2.json"
```

Le fichier `provenance_produit_1.json` reste dans le même dossier et est découvert automatiquement. `pipeline_resultats.json` et `entree_rag_produit_1.json` ne sont pas les entrées principales de la commande `chercher`.

## Statuts d'équivalence V14.0.2

- `equivalent_direct` : aucune différence obligatoire prouvée ;
- `alternative_conditionnelle` : différence obligatoire de configuration, interface, performance, conformité ou sécurité ;
- `non_equivalent` : différence sur la fonction principale ;
- `non_verifiable` : une contrainte obligatoire n'est pas prouvée dans les passages retenus ;
- `proximite_documentee` : candidat proche classé à partir des critères non bloquants.

Le rang 1 ne modifie plus automatiquement le statut technique du candidat.

## Installation sous Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Dans `.env` :

```env
NVIDIA_API_KEY=nvapi-TA_CLE
NVIDIA_REASONING_MODE=normal
NVIDIA_EMBED_MODEL=nvidia/nemotron-3-embed-1b
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_FALLBACK_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_CAHIER_MODEL_PRIMARY=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_CAHIER_MODEL_FALLBACK=nvidia/nemotron-3-super-120b-a12b
RAG_CATALOGUE_PROFILE=auto
RAG_OCR_MODE=auto
```

Ne partage jamais la clé API.

## Une seule commande pour chercher

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\cahier_produit.txt" `
  --provenance ".\provenance_produit.json" `
  --catalogue ".\catalogue.pdf" `
  --sortie ".\resultats\produit_v14.json"
```

Le fichier de provenance est facultatif. Sans `--provenance`, V14 tente de découvrir automatiquement le JSON compagnon à côté du cahier.

Le mode multi-candidats est activé par défaut :

```text
1 candidat conservé      → status: found
2 à 5 candidats conservés → status: multiple_candidates
```

## Grand catalogue

Le profil `auto` adapte la profondeur au nombre de pages, de chunks et de sections. Pour un catalogue exceptionnellement volumineux :

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\cahier_produit.txt" `
  --catalogue ".\grand_catalogue.pdf" `
  --profil-catalogue huge `
  --ocr auto `
  --sortie ".\resultats\produit_v14.json"
```

La navigation hiérarchique reste automatique. Elle ne bloque jamais définitivement la recherche dans une section : si les preuves locales sont faibles, V14 relance les voies globales.

## Diagnostic optionnel

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\cahier_produit.txt" `
  --catalogue ".\catalogue.pdf" `
  --diagnostic `
  --details `
  --sortie ".\resultats\produit_v14.json"
```

Le diagnostic contient notamment :

```text
navigation_source
navigation_structure_confidence
navigation_strategy
catalogue_route
primary_page_range
primary_search_pages
secondary_search_pages
navigation_route_confidence
local_evidence
global_fallback_used
global_fallback_reason
retrieval_lanes
final_context_pages
```

Exemple de route :

```json
{
  "navigation_strategy": "hierarchical",
  "catalogue_route": [
    "HELIOPROTECTION®",
    "Fusibles photovoltaïques"
  ],
  "primary_page_range": [214, 221],
  "global_fallback_used": false
}
```

## Forcer le RAG global normal

Le repli est automatique. Pour désactiver volontairement la navigation :

```powershell
--sans-hierarchie
```

V14 utilise alors les voies globales lexicale, dense, exacte et structurelle sur tout le catalogue.

## Inspection locale sans LLM

```powershell
python .\rag_catalogue_cli.py inspecter `
  --catalogue ".\catalogue.pdf" `
  --ocr auto `
  --sortie ".\resultats\inspection_catalogue.json"
```

Le rapport expose :

- pages avec texte natif ;
- pages candidates OCR ;
- sections de mise en page ;
- nœuds de navigation issus du sommaire ;
- relations parent/enfant ;
- score de confiance de la structure.

## Tester seulement la récupération

```powershell
python .\rag_catalogue_cli.py retrouver `
  --catalogue ".\catalogue.pdf" `
  --requete "fusible photovoltaïque gPV 15 A 1000 VDC" `
  --profil-catalogue huge `
  --top-k 12
```

## Préindexer un catalogue

```powershell
python .\rag_catalogue_cli.py indexer `
  --catalogue ".\grand_catalogue.pdf" `
  --sortie ".\resultats\index_catalogue.json"
```

Le cache dense est enregistré dans `.rag_cache`. Il est invalidé automatiquement si le PDF, le modèle ou les paramètres de découpage changent.

## OCR

```text
--ocr off    : extraction native uniquement
--ocr auto   : OCR seulement sur les pages image presque vides
--ocr force  : tente l’OCR dans la limite configurée
```

Options :

```text
--ocr-lang eng
--ocr-lang fra+eng
--ocr-max-pages 12
--min-native-chars 40
```

L’OCR nécessite Tesseract et les langues demandées. Sans Tesseract, le mode `auto` conserve l’extraction native.

## Raisonnement du LLM

```text
rapide      : thinking désactivé
normal      : thinking actif, budget 2048, défaut
approfondi  : thinking actif, budget 8192
```

Le budget s'**ajoute** à `max_tokens` au lieu d'être prélevé dessus : l'augmenter
ne raccourcit pas la réponse. Le thinking n'est proposé que par la famille
`nemotron-3` ; avec un autre modèle, le profil retombe silencieusement sur
`rapide` — le bloc `reasoning` du résultat indique ce qui a réellement été appliqué.

```powershell
--raisonnement rapide
--raisonnement normal
--raisonnement approfondi
```

Le raisonnement ne remplace pas une bonne récupération. V14 améliore d’abord le chemin vers les bonnes pages, puis demande au LLM de classer et d’expliquer les références.

## Exemple prêt à l'emploi

Un jeu d'exemple **fictif et libre de droits** est fourni dans [`examples/`](examples/) :
un catalogue d'objectifs photo (`optora_lens_catalogue.pdf`) et un cahier de besoin
(`cahier_objectif.txt`).

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\examples\cahier_objectif.txt" `
  --catalogue ".\examples\optora_lens_catalogue.pdf" `
  --diagnostic --details `
  --sortie ".\examples\resultat_objectif.json"
```

Résultat attendu : la référence **`OP-2470F28G`** (24-70 mm f/2.8 G) comme
meilleure correspondance, et une navigation vers la section
« Standard zoom lenses » via le sommaire du catalogue. Voir
[`examples/README.md`](examples/README.md) pour le détail.

## Essai sur un grand catalogue réel (Socomec, 906 pages)

Essai mené sur le catalogue général Socomec (906 pages, 189 signets) avec un
cahier demandant un interrupteur-sectionneur 4 pôles, 250 A, 400 V AC, commande
frontale directe, montage en armoire.

**L'outil a trouvé la bonne référence.** Il retourne `3032 4025` et `3116 4025`
— la gamme **SIRCO** sous coffret, page 821 en tôle peinte et page 820 en
polyester, en 4 pôles / 250 A / commande frontale directe, soit exactement ce que
demandait le cahier. Les deux références sortent en tête du classement, chacune
appuyée sur les preuves extraites de sa page, et la validation passe sans erreur.

Trois défauts ont été corrigés à cette occasion :

- **Encodage de police.** 11 % du texte (190 pages) s'extrayait en charabia :
  des sous-ensembles de police sans table ToUnicode exploitable décalent chaque
  code de caractère d'une constante. Réparé par [`encoding_repair.py`](rag_catalogue/encoding_repair.py),
  qui apprend les décalages du document et ne réécrit un fragment que s'il
  devient nettement plus français. Mesure : 100 pages améliorées sur 130
  échantillonnées, aucune régression.
- **Couverture locale diluée.** Les requêtes reformulent un même besoin, parfois
  dans une autre langue ; leurs jetons étaient mis en commun, si bien qu'une
  variante traduite faisait chuter la couverture sous le seuil de repli. La
  branche est désormais jugée sur la formulation qui lui correspond le mieux.
- **Routage dominé par les jetons génériques.** `exact_score` est un simple
  recouvrement d'ensembles, sans pondération : « 250 » ou « 400 » y pesaient
  autant que « interrupteur-sectionneur », ce qui donnait la route à la section
  la plus fournie en tableaux. Le routage n'utilise plus que les signaux
  pondérés par la rareté.

**Encore perfectible.** Sur ce catalogue, la bonne référence est le plus souvent
atteinte par la recherche globale : la navigation hiérarchique bascule encore en
repli (`multiple_competing_sections`) au lieu de descendre jusqu'à la section
SIRCO, et le rang des candidats bouge d'un run à l'autre. Le comportement visé
est décrit dans [`examples/RESULTAT_ATTENDU_socomec.md`](examples/RESULTAT_ATTENDU_socomec.md).

## Fournir son catalogue et son cahier

Ce dépôt ne contient volontairement aucun catalogue PDF ni donnée client.
Pour exécuter le RAG, fournissez vos propres fichiers :

- un **catalogue** au format PDF (`--catalogue ".\\mon_catalogue.pdf"`) ;
- un **cahier technique** décrivant le besoin (PDF, TXT, Markdown ou JSON) via `--fiche` ;
- optionnellement un fichier de **provenance** JSON via `--provenance`.

Aucun de ces fichiers ne doit être versionné : le `.gitignore` exclut déjà
`*.pdf`, `cahier_*.txt`, `provenance_*.json`, `.rag_cache/` et `resultats/`.

## Tests

```powershell
$env:PYTHONPATH="."
python -m pytest -q
```
