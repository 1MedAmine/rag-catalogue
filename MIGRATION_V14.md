# Migration V13.1.1 → V14

Aucune nouvelle dépendance Python n’est nécessaire et la commande `chercher` reste compatible.

## Mise à jour recommandée

1. Décompresser V14 dans un nouveau dossier.
2. Copier le fichier `.env` existant.
3. Conserver ou recopier `.rag_cache` pour éviter de recalculer les embeddings lorsque les paramètres sont identiques.
4. Relancer la même commande de recherche.

## Changements visibles

Le résultat principal ne change pas de schéma. Le diagnostic ajoute :

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
```

## Retour au comportement global V13

```powershell
--sans-hierarchie
```

Cette option désactive la navigation V14 et force la recherche globale normale.
