# RAG Catalogue générique — Design corrigé

## Objectif

Construire un CLI Python autonome qui reçoit une fiche technique PDF et un catalogue PDF quelconque contenant du texte extractible, récupère les passages pertinents sans connaissance métier codée en dur, puis demande à un LLM NVIDIA configurable de retourner une seule référence exacte.

## Contrainte d'indépendance

Le code de production ne contient aucun fabricant, aucune référence, aucun type de produit, aucun vocabulaire électrique et aucune règle de codification propres au catalogue de test. Les exemples VendorA sont confinés à un test aveugle dont la requête ne contient pas la référence cible.

## Architecture

1. Extraire le texte des deux PDF page par page avec PyMuPDF.
2. Demander au LLM de produire un schéma générique : `product_type`, `source_reference`, `attributes` et `search_queries`.
3. Construire un index lexical local utilisant TF-IDF mot, TF-IDF caractères, correspondance exacte des valeurs et correspondance exacte des codes alphanumériques.
4. Exécuter une à trois requêtes générées uniquement depuis la fiche source, sans enrichissement métier caché.
5. Fusionner les résultats des requêtes en conservant la page et le passage d'origine.
6. Demander au LLM de retourner une référence explicite, ou construite uniquement lorsqu'une règle de codification complète est présente ; sinon retourner `null`.

## Hors périmètre

- OCR des catalogues scannés.
- Reconstruction fiable des tableaux dont l'extraction PDF détruit les lignes et colonnes.
- Calcul d'un taux de compatibilité.
- Validation déterministe indépendante du résultat LLM.
- Garantie absolue sur tous les PDF existants.

## Tests d'indépendance

- Catalogue synthétique de roulements : la requête ne contient pas `6304-2RS`.
- Catalogue synthétique de capteurs : la requête ne contient pas `SEN-M18-PNP-8-M12`.
- Catalogue VendorA réel : la requête ne contient ni `REF-DEMO-03`, ni `00016`, ni la référence attendue ; elle doit néanmoins récupérer la page de codification.
