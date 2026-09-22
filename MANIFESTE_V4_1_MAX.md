# Manifeste de livraison — BD-Ikram V4.1 MAX / RAG V14.0.2

## Versions

- Générateur de cahier et de provenance : **4.1.0**
- RAG catalogue industriel : **14.0.2**

## Contenu

Cette livraison réunit :

- le pipeline de recherche et d’identification du produit source ;
- le schéma de provenance V4.1 ;
- le rendu déterministe du cahier ;
- l’entrée RAG enrichie ;
- le RAG hiérarchique V14.0.2 ;
- les tests unitaires et de non-régression ;
- la documentation d’architecture, de migration et de vérification.

## Changements structurants

1. Une référence exacte peut produire des contraintes dérivées obligatoires lorsque les preuves relient explicitement la référence, la valeur et leur relation structurelle.
2. Les preuves peuvent être réparties dans une même ligne, colonne, table, bloc produit, codification ou continuation de tableau.
3. Un audit de complétude empêche de publier silencieusement une demande par référence dépourvue de fonction ou de configuration structurante.
4. Le modèle principal de classification/génération est le 49B v1.5 ; le 120B est utilisé une fois en repli lorsque la sortie est incomplète ou invalide.
5. Le rang 1 n’est plus promu automatiquement en `closest`.
6. Les statuts finaux sont `equivalent_direct`, `alternative_conditionnelle`, `non_equivalent`, `non_verifiable` ou `proximite_documentee`.
7. Les réponses JSON mal formées peuvent être réparées localement sans modifier les valeurs ; une réponse tronquée déclenche une réparation externe sans raisonnement.
8. Une référence absente de ses propres preuves est signalée et ne peut pas être acceptée silencieusement.

## Vérification locale de la source

```text
Suite combinée       : 223 passed, 10 skipped
Générateur V4.1 MAX  : 42 passed
RAG V14.0.2          : 181 passed, 10 skipped
Compilation          : réussie
```

Les dix tests ignorés nécessitent les catalogues externes réels et leur activation. Aucun appel NVIDIA réel n’a été exécuté dans l’environnement de livraison.

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
