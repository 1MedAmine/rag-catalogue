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

## Perimetre de la mesure

L'evaluation appelle la commande `retrouver`, qui utilise actuellement la
recuperation locale avec navigation hierarchique. Elle ne branche ni client
d'embeddings ni reranker distant : aucune cle API et aucun cache dense ne
sont necessaires. L'option `--sans-embedding` est acceptee par le script mais
ne change pas ce comportement.

Le resultat porte sur le classement des passages avant generation. Il ne
mesure pas la qualite des reponses finales ni toute la chaine hybride de la
commande `chercher`.

## Interpretation

- Un **recall@5 eleve** montre que les bonnes pages remontent presque toujours
  dans les 5 premiers resultats ; cela ne garantit pas que toutes les
  informations necessaires a la reponse sont presentes.
- Un **MRR proche de 1** signifie que la page pertinente est souvent en tete.
- Le script ne transmet pas d'option `--sans-hierarchie`. Pour comparer la
  navigation hierarchique a la recherche globale, lancer directement
  `rag_catalogue_cli.py retrouver` avec et sans cette option, sur les memes
  requetes et les memes pages annotees.
