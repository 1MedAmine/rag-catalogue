# Evaluation de la recuperation (recall@k / MRR)

Ce dossier fournit une evaluation quantitative et reproductible de la qualite
du **retrieval** (avant l'etape LLM). Pour chaque requete annotee avec les
pages reellement pertinentes, on mesure :

- **recall@k** : proportion de requetes pour lesquelles au moins une page
  pertinente est dans le top-k ;
- **MRR** (Mean Reciprocal Rank) : moyenne de 1/(rang de la premiere page
  pertinente).

## Lancer l'evaluation

```powershell
python eval\evaluate_retrieval.py `
  --catalogue ".\examples\optora_lens_catalogue.pdf" `
  --requetes ".\eval\requetes.example.json" `
  --ks 1,3,5,8
```

Le fichier `requetes.example.json` cible le catalogue d'exemple fictif fourni
dans `examples/`. Pour evaluer sur ton propre catalogue, cree ton propre
fichier de requetes au meme format :

```json
[
  {"requete": "ta requete en langage naturel", "pages_pertinentes": [40, 41]}
]
```

## Sans cle API

L'evaluation appelle la commande `retrouver`. Si aucun index dense n'est en
cache, ajoute `--sans-embedding` pour n'utiliser que les voies lexicale et
exacte (aucun appel reseau) :

```powershell
python eval\evaluate_retrieval.py `
  --catalogue ".\examples\optora_lens_catalogue.pdf" `
  --requetes ".\eval\requetes.example.json" `
  --sans-embedding
```

## Interpretation

- Un **recall@5 eleve** montre que les bonnes pages remontent presque toujours
  dans les 5 premiers resultats : le LLM recevra le bon contexte.
- Un **MRR proche de 1** signifie que la page pertinente est souvent en tete.
- Comparer avec et sans `--sans-hierarchie` (via un fichier de requetes dedie)
  permet de mesurer l'apport de la navigation hierarchique.
