# V12 — Cahier complet avec provenance structurée

## Objectif

V12 conserve le cahier des charges complet comme source principale de compréhension du produit source. Un JSON de provenance est utilisé en complément, jamais comme remplacement du cahier, afin de distinguer les contraintes explicites du client, les caractéristiques attestées du produit source, les options documentaires et les ambiguïtés.

## Architecture

1. Le chargeur lit le cahier complet (PDF, TXT, MD ou JSON).
2. Il charge facultativement un JSON de provenance explicite, ou découvre automatiquement un fichier compagnon placé à côté du cahier.
3. Il transmet au LLM un document unique contenant le cahier intégral suivi d’un bloc de provenance structuré.
4. Le LLM extrait un besoin générique en conservant toutes les caractéristiques utiles à l’équivalence, avec deux axes distincts : `role` (selector, constraint, context) et `provenance` (client, source_product, optional, to_confirm).
5. La récupération catalogue utilise principalement les selector/constraint, mais la sélection finale reçoit aussi les attributs contextuels et les ambiguïtés.
6. La référence complète est soumise par défaut à une validation structurelle de preuve et de codification. Cette validation reste générique et ne contient aucune règle métier propre aux disjoncteurs, pompes ou autres familles.

## Entrées

Commande principale :

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\cahier_produit_1.txt" `
  --provenance ".\provenance_produit_1.json" `
  --catalogue ".\catalogue.pdf" `
  --sortie ".\resultat.json"
```

Sans `--provenance`, V12 recherche automatiquement des compagnons nommés `provenance_<suffixe>.json`, `entree_rag_<suffixe>.json`, `<stem>_provenance.json` ou `<stem>.provenance.json`.

## Règles de provenance

- `client` : exigence explicitement présente dans la demande brute ; rôle généralement `constraint`.
- `source_product` : caractéristique attestée du produit source ; peut être `selector` si elle est nécessaire pour identifier une variante équivalente.
- `optional` : capacité ou option catalogue ; rôle `context`.
- `to_confirm` : ambiguïté non tranchée ; rôle `context`, avec requêtes alternatives si utile.

Une caractéristique source peut donc être importante pour l’équivalence sans être présentée comme une exigence explicitement formulée par le client.

## Compatibilité

Les anciennes fiches seules restent acceptées. Les anciens JSON restent acceptés. Les interfaces Python existantes restent utilisables ; les nouveaux paramètres sont optionnels.

## Sécurité de référence

La validation par défaut vérifie uniquement : page récupérée, preuve textuelle, référence explicite ou construction complète, présence du schéma de commande et cohérence des segments. Elle ne compare pas des règles métier spécifiques à une famille de produit.
