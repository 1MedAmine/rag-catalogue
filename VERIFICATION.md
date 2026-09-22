# Vérification du RAG catalogue

## Tests locaux

Depuis la racine du dépôt, avec les dépendances de `requirements.txt` :

```powershell
$env:PYTHONPATH="."
$env:RUN_EXTERNAL_CATALOGUE_TESTS="0"
python -m pytest -q -p no:cacheprovider
```

Résultat de la vérification : **199 tests réussis, 10 tests ignorés, aucun
échec**. Les fichiers temporaires de cette exécution ont été placés hors du
dépôt avec l'option `--basetemp`.

Les dix tests ignorés nécessitent un catalogue externe et l'activation
explicite de `RUN_EXTERNAL_CATALOGUE_TESTS=1`. Ils ne sont pas comptés parmi
les tests réussis.

La suite couvre notamment l'extraction PDF, la navigation, la récupération
hybride, la provenance, la validation des références, la qualification des
candidats et les contrats des clients de génération, d'embeddings et de
reranking. Les appels distants sont simulés dans ces tests.

Le cas de validation d'une référence construite utilise désormais une
référence cohérente avec ses composants. Le validateur du moteur n'a pas été
modifié pour faire passer ce test.

## Évaluation de la récupération

```powershell
python eval\evaluate_retrieval.py `
  --catalogue ".\examples\optora_lens_catalogue.pdf" `
  --requetes ".\eval\requetes.example.json" `
  --ks 1,3,5,8
```

Résultat reproduit sur les quatre requêtes du catalogue fictif Optora :

| Mesure | Résultat |
| --- | ---: |
| Recall@1 | 3/4 — 75 % |
| Recall@3 | 4/4 — 100 % |
| Recall@5 | 4/4 — 100 % |
| Recall@8 | 4/4 — 100 % |
| MRR | 0,875 |

Ici, « recall@k » désigne la proportion de requêtes avec au moins une page
pertinente parmi les k premiers résultats. Le MRR utilise le rang de la
première page pertinente, avec une contribution nulle si aucune n'apparaît
parmi les huit résultats examinés.

## Limites

Cette vérification n'a effectué aucun appel NVIDIA réel. L'évaluation
`retrouver` utilise la récupération locale, sans embeddings ni reranking
distant ; elle ne mesure pas la chaîne hybride complète de `chercher`, la
qualité des réponses finales du LLM ni les performances sur plusieurs
catalogues industriels.

Le générateur de cahier en amont n'est pas inclus dans ce dépôt. Les anciens
résultats combinant ses tests avec ceux du RAG ne décrivent donc pas la suite
reproductible ici.
