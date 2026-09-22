# Vérification V14.0.2 / Cahier V4.1.0 MAX

## Vérification locale fraîche

Commandes exécutées depuis la copie isolée :

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m pytest -q test_correctifs.py
PYTHONPATH=. python -m pytest -q tests
python -m compileall -q .
```

Résultats :

```text
Suite combinée       : 223 passed, 10 skipped
Générateur V4.1 MAX  : 42 passed
RAG V14.0.2          : 181 passed, 10 skipped
Compilation          : réussie
```

Les dix tests ignorés sont les tests VendorA externes nécessitant l’activation explicite et le PDF réel.

## Non-régressions ajoutées

- preuve répartie entre référence, valeur et même tableau ;
- rejet d’une relation structurelle non attestée ;
- audit de complétude des demandes par référence exacte ;
- conservation des six dimensions structurantes du cas `REF-DEMO-01` dans les tests ;
- repli 49B vers 120B lorsque la classification principale est incomplète ;
- absence de promotion automatique du rang 1 vers `closest` ;
- différence de fonction classée `non_equivalent` ;
- différence obligatoire de montage/fourniture classée `alternative_conditionnelle` ;
- contrainte obligatoire non prouvée classée `non_verifiable` ;
- réparation locale du JSON VendorB avec parenthèses hors guillemets ;
- réparation externe sans raisonnement pour une réponse tronquée ;
- routage du modèle principal et du modèle de secours.

## Limites de cette vérification

Aucun appel NVIDIA réel n’a été lancé dans l’environnement de livraison, car aucune clé API n’a été intégrée. Les catalogues VendorA et VendorB réels ne sont pas inclus dans l’archive. Les résultats locaux prouvent la logique, la compilation et les contrats de sortie ; ils ne constituent pas encore un benchmark industriel multi-catalogues.
