# Changelog

## Depuis V14.0.2

- Aligne la configuration d'exemple et la documentation sur le modèle de génération 120B déjà utilisé par défaut dans le CLI.
- Précise le périmètre du dépôt et les limites de l'évaluation Recall@k.
- Corrige les données incohérentes du test de référence construite, sans modifier le validateur.
- Actualise le bilan des tests et classe les documents techniques dans `docs/plans` et `docs/specs`.

## V14.0.2 — livraison initiale

- Supprime la promotion automatique du candidat de rang 1 vers `closest`.
- Ajoute `equivalence_status` : `equivalent_direct`, `alternative_conditionnelle`, `non_equivalent`, `non_verifiable` ou `proximite_documentee`.
- Classe une différence de fonction obligatoire comme `non_equivalent`, une preuve obligatoire absente comme `non_verifiable` et les autres écarts obligatoires comme `alternative_conditionnelle`.
- Ajoute une matrice `constraint_assessment` générique et conserve les alternatives documentées au lieu de les supprimer.
- Répare localement les parenthèses placées hors guillemets et les virgules finales dans les réponses JSON, sans modifier les valeurs techniques.
- Détecte les réponses tronquées et exécute une seule réparation externe sans raisonnement.
- Utilise par défaut le modèle 49B pour la génération structurée et le 120B comme repli configurable.
- Expose les métriques de réparation JSON et le routage des modèles dans le résultat et le diagnostic.

## V14.0.0


- Ajoute une navigation de catalogue du général vers le précis : grande section, sous-section principale et branches secondaires.
- Parse les sommaires textuels avec pages simples ou plages de pages, y compris lorsque le PDF ne possède aucun signet.
- Déduit automatiquement les relations parent/enfant par inclusion des plages de pages.
- Ajoute un score de confiance de la structure du catalogue et refuse de forcer une hiérarchie peu fiable.
- Privilégie la sous-section la plus spécifique lorsque son parent obtient un score voisin.
- Détecte les routes ambiguës contenant plusieurs sections concurrentes et repasse au RAG global.
- Effectue une recherche lexicale et dense approfondie dans la route principale, avec branches secondaires et pages voisines.
- Mesure la couverture des preuves locales avant de décider si une recherche globale complète est nécessaire.
- Conserve en permanence une recherche globale exacte et une voie structurelle de sécurité.
- Ajoute au diagnostic `catalogue_route`, `primary_page_range`, `local_evidence`, `global_fallback_used` et `global_fallback_reason`.
- Enrichit les commandes `inspecter` et `indexer` avec la carte de navigation et son score de confiance.
- Conserve tous les correctifs V13.1.1 pour les alternatives proches, le multi-candidats et la provenance.

## V13.1.1

- Conserve les alternatives commandables même lorsqu'elles diffèrent sur des caractéristiques documentées du produit source ; seules les contraintes client explicites restent bloquantes.
- Préserve dans `differences` les écarts techniques du meilleur ou de l'unique candidat, au lieu de les effacer après le filtrage final.
- Supprime uniquement les différences devenues obsolètes parce qu'elles comparaient un candidat à une variante finalement rejetée, ainsi que les mentions « aucune différence ».
- Sélectionne le texte natif d'appui selon la valeur technique ajoutée par rapport aux tableaux structurés, et non selon le seul ordre des pages.
- Conserve davantage d'appuis natifs dans les profils profonds afin de ne pas perdre une ligne exacte placée tard dans une grande table.
- Évite de réintroduire les pages de texte aplati lorsque les tableaux structurés couvrent déjà les mêmes références, ce qui protège les catalogues à colonnes multiples.
- Ajoute des tests de non-régression pour `REF-DEMO-02`, `REF-DEMO-02CC`, les écarts de pouvoir de coupure et la coexistence avec les tables VendorA.

## V13.1.0

- Corrige le rejet des alternatives proches lorsque le catalogue ne contient aucune correspondance exacte avec toutes les caractéristiques documentaires du produit source.
- Seules les exigences `provenance=client` avec `role=constraint` sont désormais des obligations dures ; les attributs `source_product` servent au classement et leurs écarts sont listés dans `differences`.
- `candidates=[]` est réservé au cas où aucune référence commandable pertinente n'est visible dans les passages récupérés.
- Conserve les deux pages voisines autour des tables produit, y compris lorsque le PDF utilise seulement des en-têtes génériques comme « numéro de catalogue / courant nominal ».
- Ajoute une couverture structurée par page prioritaire afin qu'une grande page riche en tableaux ne fasse pas disparaître une autre page produit pertinente.
- Conserve conjointement le texte natif et les tableaux structurés d'une même page lorsque l'extraction tabulaire perd des colonnes ou des lignes.
- Enrichit le diagnostic avec les pages avant structuration, les pages ciblées, les pages structurées et les pages prioritaires.
- Ajoute des tests de non-régression reproduisant le cas VendorB REF-DEMO-02 / REF-DEMO-02CC.

## V13.0.0

- Ajoute une carte hiérarchique du catalogue issue du sommaire PDF, des titres de mise en page ou de fenêtres bornées.
- Ajoute des métadonnées de section, de type structurel, de marqueurs et de qualité à chaque page et chunk.
- Adapte la profondeur avec le nombre de pages, de chunks et de sections ; ajoute les profils `standard`, `large`, `huge` et `auto`.
- Ajoute le routage sectionnel tout en conservant les voies globales lexicale, dense et exacte.
- Ajoute une voie structurelle dédiée aux tableaux de sélection et aux pages de commande.
- Remplace la fusion à deux voies par une RRF pondérée multi-voies.
- Ajoute la déduplication quasi-identique et la diversification par page et section.
- Ajoute le reranking en lots bornés avec passe finale de tournoi.
- Ajoute l’inspection PDF et l’OCR optionnel `off`, `auto` ou `force`, page par page.
- Ajoute les commandes `inspecter` et `indexer`.
- Enrichit le diagnostic avec la carte, les sections routées, les voies, l’OCR, le cache, la déduplication et le reranking.
- Invalide le cache dense lorsque les métadonnées de section ou de type changent.
- Conserve la compatibilité V12.3.5 : multi-candidats, raisonnement, réparation JSON, preuves et provenance.

## V12.3.5

- Ajoute une réparation automatique unique lorsqu'une réponse LLM contient un objet JSON syntaxiquement invalide.
- La passe de réparation reçoit la réponse brute et doit uniquement corriger les guillemets, virgules, crochets, accolades et échappements, sans changer les données techniques.
- Expose dans le diagnostic le nombre de tentatives et de réussites de réparation JSON.
- Détecte automatiquement le nombre de pages du catalogue.
- Augmente la profondeur après chaque tranche complète de 100 pages : +2 passages finaux et +16 candidats par tranche.
- Applique des plafonds prudents par défaut : `top-k=20` et `candidate-k=128`, sans jamais réduire des valeurs explicitement plus grandes.
- Ajoute des tests de non-régression pour le JSON cassé observé sur `evidence` et pour les catalogues de plus de 100 pages.

## V12.3.4

- Déduit automatiquement les variantes `Standard` et `Deluxe` depuis les preuves catalogue lorsque le LLM renvoie `variant: null`.
- Classe la variante Standard en premier lorsque Standard et Deluxe satisfont les mêmes critères et qu’aucune exigence explicite ne demande Deluxe.
- Conserve Deluxe en premier lorsqu’une exigence explicite la rend nécessaire.
- Reconstruit en français les raisons et l’explication finale pour le couple Standard/Deluxe.
- Remplace les différences anglaises par une formulation française cohérente.

## V12.3.3

- Reconstruit l'explication finale uniquement à partir des candidats réellement conservés après validation, déduplication et filtrage.
- Recalcule systématiquement les rangs et la référence principale à partir de la liste finale.
- Force le candidat final de rang 1 à `match_level=closest`.
- Supprime des justifications les phrases liées à un ancien classement (`ranked second`, `rang 2`, `premier`, etc.).
- Supprime les différences comparatives lorsqu'un seul candidat reste, afin de ne plus mentionner une alternative rejetée.
- Garantit ainsi `status=found` avec un texte cohérent lorsqu'un seul candidat est conservé, et `multiple_candidates` seulement lorsque plusieurs candidats restent.

## V12.3.2

- Normalise aussi `evidence` dans les anciennes réponses à référence unique.
- Accepte une preuve sous forme de chaîne, liste ou objet narratif au lieu d'arrêter la recherche.
- Conserve une liste de textes normalisée avant la validation de provenance.
- Ajoute des tests de non-régression pour les formes chaîne et objet.

## V12.3.1

- Ajoute une carte déterministe des codes autorisés pour chaque position d’une référence construite à partir de la table de commande.
- Rejette les codes visibles dans la page mais placés dans le mauvais segment, par exemple `HGDNN...` lorsque le deuxième segment doit être un code de frame.
- Conserve les trois profils de raisonnement V12.3 : rapide, normal (défaut, budget 2048) et approfondi (budget 4096).
- Conserve les reprises automatiques V12.2 pour liste vide, `ResourceExhausted`, pages et explications mal formatées.

## V12.3.0

- Ajoute `--raisonnement rapide|normal|approfondi`.
- Active par défaut le profil `normal` avec thinking Nemotron, `low_effort=true` et un budget de 2048 tokens.
- Ajoute un profil approfondi à 4096 tokens et conserve un profil rapide sans thinking.
- Augmente automatiquement `max_tokens` afin de réserver le budget de raisonnement sans réduire la réponse JSON finale.
- Ajoute `NVIDIA_REASONING_MODE` dans `.env`.
- Normalise les pages renvoyées sous forme de scalaire, chaîne, entier décimal ou libellé contenant plusieurs pages.
- Ajoute une reprise automatique bornée sur les erreurs NVIDIA temporaires et `ResourceExhausted`.
- Expose le profil de raisonnement réellement utilisé dans la sortie JSON.

## V12.2.0

- Relance automatiquement une seule fois la génération lorsque le premier appel retourne `candidates=[]` malgré des passages catalogue récupérés.
- Réutilise exactement le même besoin et les mêmes passages ; seuls le retour sur la liste vide et l’appel de génération changent.
- Normalise `explanation` et `reason` lorsqu’un LLM les renvoie sous forme de liste ou d’objet JSON au lieu d’une chaîne.
- Normalise aussi les formes textuelles courantes de `evidence` et `differences`.
- Ajoute au diagnostic `generation_attempts` et `empty_candidate_retry`.
- Ne boucle jamais : après une seule seconde passe vide, le résultat reste `not_found`.

## V12.1.0

- Retourne jusqu’à cinq références catalogue pertinentes au lieu d’un seul choix.
- Classe les candidats du plus proche au moins proche.
- Ajoute pour chaque candidat : variante, niveau de correspondance, pages, justification et différences utiles.
- Conserve le meilleur candidat dans le champ `reference` pour la compatibilité avec V12.
- Valide séparément chaque candidat et supprime ceux dont la référence n’est pas prouvée.
- Retourne le statut `multiple_candidates` lorsque plusieurs alternatives valides sont disponibles.
- N’autorise une variante Deluxe ou premium à passer devant une variante standard que si les caractéristiques documentées le justifient.

## V12.0.0

- Conserve le cahier technique complet comme entrée principale du RAG.
- Ajoute un JSON de provenance complémentaire, explicite avec `--provenance` ou découvert automatiquement à côté du cahier.
- N’utilise jamais le JSON comme remplacement du cahier.
- Conserve pour chaque attribut son `role` (`selector`, `constraint`, `context`) et sa `provenance` (`client`, `source_product`, `to_confirm`).
- Transmet les caractéristiques du produit source utiles à l’équivalence au LLM final, sans les convertir en exigences client.
- Conserve les ambiguïtés dans `open_questions` et autorise plusieurs branches de recherche.
- Retire des requêtes cible le fabricant, la gamme et la référence du produit source, y compris lorsqu’un seul jeton du fabricant est repris.
- Ajoute `--sans-provenance-auto` pour désactiver la découverte automatique.
- Reste compatible avec une fiche seule et avec les options V11.

## V11.0.0

- Ajoute `nvidia/nemotron-3-embed-1b` via `/v1/embeddings`.
- Ajoute la recherche dense, la fusion RRF et le reranking NVIDIA.
- Désactive par défaut la validation générique historique ; `--validation-preuves` la réactive.
