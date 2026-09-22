# Migration V14.0.0 vers V14.0.1

## Compatibilité

Les commandes et options CLI restent identiques. Aucun changement de dépendance n'est requis.

## Fichiers à remplacer

Pour appliquer le correctif sur une installation V14.0.0, remplacer au minimum :

```text
rag_catalogue/llm_client.py
rag_catalogue/pipeline.py
rag_catalogue/source_bundle.py
rag_catalogue/__init__.py
```

et ajouter :

```text
rag_catalogue/provenance_need.py
```

Le paquet complet est préférable afin de conserver les tests et la documentation cohérents.

## Provenance V4

V14.0.1 accepte les nouvelles clés :

```text
schema_version
type_demande
identification_produit
contraintes_explicites
contraintes_derivees_obligatoires
criteres_de_classement
capacites_optionnelles
informations_a_confirmer
```

Commande recommandée :

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\cahier_produit.txt" `
  --provenance ".\provenance_produit_v4.json" `
  --catalogue ".\catalogue.pdf" `
  --diagnostic `
  --details `
  --sortie ".\resultats\produit_v14_0_1.json"
```

## Anciennes provenances

Aucune conversion préalable n'est obligatoire. Les clés V3.1 restent prises en charge. Elles ne créent jamais de contraintes dérivées automatiquement.

## Effet sur les résultats

Une différence sur une contrainte dérivée doit déclasser un candidat au lieu de le supprimer. Une contrainte explicitement écrite par le client reste plus forte.

Les sorties JSON mal formées peuvent désormais être réparées localement ou régénérées de façon compacte. Le résultat final doit toujours passer la validation de référence et de page.

## Vérification après migration

```powershell
$env:PYTHONPATH="."
python -m pytest -q
python -m compileall -q rag_catalogue rag_catalogue_cli.py
```
