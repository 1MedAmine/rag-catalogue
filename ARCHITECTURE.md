# Architecture V14.0.2 — RAG à navigation humaine avec repli global


## 0. Entrée V4.1 et routage des modèles

Une demande par référence exacte passe par un audit de provenance avant le RAG :

```text
référence exacte
→ identification documentée
→ contraintes dérivées avec preuve directe ou relation structurelle
→ audit fonction + dimension structurante
→ 49B accepté si complet
→ sinon repli unique 120B
```

Le classement final sépare désormais le rang commercial de la qualification technique. Le rang 1 ne peut plus modifier à lui seul `equivalence_status`.

## Flux principal

```text
Cahier technique complet + provenance
        ↓
Nemotron : besoin structuré et requêtes génériques
        ↓
Inspection PDF + extraction native + OCR ciblé
        ↓
Carte des sections de mise en page
        ↓
Carte de navigation : sommaire, plages de pages, parents/enfants
        ↓
Évaluation de la fiabilité de la structure
        ↓
┌──────────────────────────────────────────────────────┐
│ structure fiable                                    │
│ grande section → sous-section → recherche locale    │
│ + une ou deux branches secondaires                  │
└──────────────────────────────────────────────────────┘
        ↓
Évaluation des preuves locales
        ↓
preuves suffisantes ? ── non ──→ RAG global normal
        │ oui
        ↓
Voies locales + sécurité globale exacte/structurelle
        ↓
Fusion, déduplication, diversification et reranking
        ↓
Nemotron : références classées et différences
```

## 1. Carte de navigation

`rag_catalogue/catalogue_navigation.py` construit une structure indépendante du domaine produit.

Sources utilisées, par ordre de préférence pratique :

1. sommaire textuel visible dans les premières pages ;
2. plan/signets PDF déjà exploités par `catalogue_map.py` ;
3. titres détectés par mise en page ;
4. sections bornées de repli.

Le parseur de sommaire reconnaît notamment :

```text
HELIOPROTECTION® 212 - 229
Fusibles photovoltaïques ........ 214 - 221
```

Les plages incluses permettent de déduire :

```text
HELIOPROTECTION®
└── Fusibles photovoltaïques
```

Chaque `NavigationNode` conserve : titre, pages, niveau, parent, résumé, source et sections de mise en page recouvertes.

## 2. Confiance de structure

Le score tient compte de :

- présence d’un sommaire ou d’un plan PDF ;
- nombre de nœuds ;
- qualité et unicité des titres ;
- couverture du PDF ;
- présence d’une hiérarchie parent/enfant ;
- pénalités pour un document plat ou des titres génériques.

Sous le seuil de confiance, V14 ne force aucune route et utilise le RAG global normal.

## 3. Plan de navigation

Les résumés de sections sont recherchés avec les requêtes issues du cahier. V14 choisit :

- une sous-section principale ;
- jusqu’à deux sections secondaires appartenant à la même grande branche ;
- les pages voisines autour des plages retenues.

Lorsque le parent et l’enfant ont des scores proches, la sous-section la plus précise est privilégiée. Lorsque de nombreuses sections indépendantes sont à égalité, V14 considère la route ambiguë et repasse en recherche globale.

## 4. Recherche locale approfondie

Dans la route choisie :

- TF-IDF mots et caractères ;
- valeurs et codes exacts ;
- embeddings denses ;
- pages de tableaux, sélection et codification ;
- pages voisines ;
- branches secondaires à poids réduit.

La section principale reçoit le poids le plus élevé.

## 5. Contrôle de qualité local

`assess_local_evidence` mesure :

- couverture des mots et valeurs de la requête ;
- présence d’un signal de référence ;
- présence d’un tableau, d’une sélection ou d’une codification ;
- signaux exacts et codes.

Si les preuves locales sont insuffisantes, les voies globales lexicale et dense sont activées automatiquement.

## 6. Voies globales de sécurité

Même lorsque la route locale est forte :

- la recherche exacte des références et codes reste globale ;
- une voie structurelle globale de faible poids peut récupérer une page de codification placée en annexe.

Cela évite qu’un sommaire imparfait fasse disparaître une référence explicite.

## 7. RAG global normal

Le mode global est utilisé lorsque :

- aucune structure fiable n’est trouvée ;
- aucune route n’est trouvée ;
- plusieurs sections concurrentes sont pratiquement à égalité ;
- la confiance du routage est faible ;
- les preuves locales ne couvrent pas suffisamment le besoin ;
- l’utilisateur passe `--sans-hierarchie`.

Il combine alors les voies globales lexicale, dense, exacte et structurelle comme dans V13.

## 8. Alternatives

La navigation sert à améliorer le rappel des alternatives sans validation destructive :

- toutes les références explicitement prouvées dans les passages finaux peuvent être proposées ;
- les écarts techniques restent dans `differences` ;
- une absence de correspondance exacte ne signifie pas une absence d’alternative ;
- Python vérifie la provenance structurelle, pas la supériorité métier d’un candidat.

## 9. Diagnostic

Le diagnostic V14 enregistre le chemin de navigation et la raison d’un éventuel retour global. Il permet de savoir immédiatement si l’erreur vient :

- du sommaire ;
- du routage ;
- de la récupération locale ;
- du repli global ;
- du reranking ;
- de la génération finale.
