# Contenu du dépôt — RAG V14.0.2 et provenance V4.1

## Périmètre

Ce dépôt contient le moteur de recherche dans les catalogues PDF et la prise
en charge des cahiers techniques accompagnés d'une provenance V4.1. Le
pipeline amont de recherche du produit source et de génération du cahier
n'est pas inclus.

## Éléments fournis

- les commandes de recherche, récupération, inspection et indexation ;
- l'extraction PDF, la navigation hiérarchique et le repli global ;
- la récupération lexicale, exacte, structurelle et dense, avec reranking
  dans la commande `chercher` ;
- la lecture et la validation des contraintes issues de la provenance ;
- la validation des références et la qualification technique des candidats ;
- les tests locaux, le catalogue fictif Optora et son jeu d'évaluation ;
- la documentation d'architecture et les notes de migration.

## Génération et classement

Le modèle de génération par défaut est `nvidia/nemotron-3-super-120b-a12b`.
Le client de repli utilise le même modèle par défaut et reste configurable.
Les réponses JSON peuvent être réparées avant validation.

Le rang d'un candidat ne détermine pas son statut technique. Les statuts
possibles sont `equivalent_direct`, `alternative_conditionnelle`,
`non_equivalent`, `non_verifiable` et `proximite_documentee`.

## Vérification

Les commandes exécutées, les résultats des tests et les limites de
l'évaluation sont détaillés dans [VERIFICATION.md](VERIFICATION.md).

## Commande RAG recommandée

Le seul fichier à sélectionner manuellement est le cahier. La provenance portant le même suffixe est découverte automatiquement lorsqu’elle reste dans le même dossier.

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\resultats_cahier_des_charges\cahier_produit_1.txt" `
  --catalogue ".\catalogue_cible.pdf" `
  --profil-catalogue huge `
  --ocr auto `
  --raisonnement approfondi `
  --diagnostic `
  --details `
  --sortie ".\resultats\produit_1_v14_0_2.json"
```
