# Exemples

Ce dossier contient un jeu d'exemple **entierement fictif et libre de droits**
pour tester le RAG sans donnee reelle.

| Fichier | Role |
|---|---|
| `optora_lens_catalogue.pdf` | Catalogue fictif d'objectifs photo (marque inventee "Optora"), 10 pages, avec sommaire, sections et tableaux de references. |
| `cahier_objectif.txt` | Cahier des charges d'exemple : un besoin d'objectif zoom standard f/2.8. |

## Lancer l'exemple

```powershell
python .\rag_catalogue_cli.py chercher `
  --fiche ".\examples\cahier_objectif.txt" `
  --catalogue ".\examples\optora_lens_catalogue.pdf" `
  --raisonnement normal `
  --diagnostic `
  --details `
  --sortie ".\examples\resultat_objectif.json"
```

## Resultat attendu

La reference la plus pertinente est **`OP-2470F28G`**
(Optora 24-70 mm f/2.8 G, ouverture constante f/2.8, stabilisee, plein format).

Les references `OP-2870F4C` et `OP-24105F4G` sont des alternatives proches
mais en f/4 : elles ne respectent pas la contrainte obligatoire f/2.8 et
doivent donc apparaitre avec un statut `alternative_conditionnelle`.

Le diagnostic (`--diagnostic`) doit montrer une navigation vers la section
**"3. Standard zoom lenses"** (pages 6-7) grace au sommaire du catalogue.

## Tester sur un vrai catalogue public

Le RAG fonctionne sur n'importe quel catalogue PDF structure. Pour essayer sur
un vrai catalogue (non inclus dans ce depot pour des raisons de droits), vous
pouvez telecharger un catalogue public, par exemple un catalogue d'objectifs
de fabricant, et le passer via `--catalogue`.
