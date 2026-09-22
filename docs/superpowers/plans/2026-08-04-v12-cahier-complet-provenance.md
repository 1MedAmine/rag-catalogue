# V12 Cahier complet avec provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire V12 pour lire le cahier complet et un JSON de provenance complémentaire, enrichir le besoin RAG sans perdre les caractéristiques du produit source, et sécuriser par défaut la construction de référence.

**Architecture:** Un nouveau chargeur assemble la fiche intégrale et la provenance structurée. Le schéma de besoin conserve `role` et `provenance`; la recherche utilise les critères discriminants tandis que la sélection finale reçoit tous les attributs et ambiguïtés. La CLI active par défaut la validation structurelle de preuve.

**Tech Stack:** Python 3.10+, pytest, requests, pypdf/pdfplumber, scikit-learn, NumPy, API NVIDIA.

## Global Constraints

- Aucun vocabulaire métier fermé ni règle spécifique à une famille de produit.
- Le cahier complet reste la source principale ; le JSON est complémentaire.
- Les identifiants du fabricant source sont exclus des requêtes catalogue cible.
- Une valeur documentaire n’est pas transformée en contrainte client.
- La validation structurelle de référence est activée par défaut dans la CLI.

---

### Task 1: Charger le cahier et la provenance compagnon

**Files:**
- Create: `rag_catalogue/source_bundle.py`
- Modify: `rag_catalogue/source_text.py`
- Test: `tests/test_source_bundle.py`

**Interfaces:**
- Produces: `load_product_source(product_file, provenance_file=None, auto_discover=True) -> ProductSource`
- `ProductSource.render_for_llm() -> str`

- [ ] Écrire les tests d’échec pour le chargement explicite, la découverte automatique et le maintien du cahier complet.
- [ ] Exécuter `pytest tests/test_source_bundle.py -v` et constater l’échec.
- [ ] Implémenter le chargeur et le rendu structuré sans modifier le texte primaire.
- [ ] Exécuter `pytest tests/test_source_bundle.py -v` et constater la réussite.

### Task 2: Conserver rôle, provenance et ambiguïtés dans le besoin

**Files:**
- Modify: `rag_catalogue/llm_client.py`
- Modify: `rag_catalogue/pipeline.py`
- Test: `tests/test_llm_client.py`
- Test: `tests/test_pipeline_v12.py`

**Interfaces:**
- Chaque attribut normalisé peut contenir `role` et `provenance`.
- Le besoin peut contenir `open_questions`.
- `_catalogue_need()` transmet tous les attributs à la sélection finale.

- [ ] Écrire les tests d’échec pour la normalisation de provenance, les ambiguïtés et la conservation des attributs contextuels.
- [ ] Exécuter les tests ciblés et constater l’échec.
- [ ] Étendre le schéma et les prompts sans ajouter de règle métier.
- [ ] Exécuter les tests ciblés et constater la réussite.

### Task 3: Intégrer le compagnon dans le pipeline et la CLI

**Files:**
- Modify: `rag_catalogue/pipeline.py`
- Modify: `rag_catalogue_cli.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_pipeline_v12.py`

**Interfaces:**
- `run_search(..., provenance_file=None, auto_discover_provenance=True, validate_result=False)`
- CLI: `--provenance`, `--sans-provenance-auto`, `--sans-validation-preuves`.

- [ ] Écrire les tests d’échec pour les nouveaux arguments et la transmission au pipeline.
- [ ] Exécuter les tests ciblés et constater l’échec.
- [ ] Implémenter les paramètres et activer la validation structurelle par défaut dans la CLI.
- [ ] Exécuter les tests ciblés et constater la réussite.

### Task 4: Vérifier la récupération VendorA et documenter V12

**Files:**
- Modify: `tests/test_vendora_smoke.py`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CHANGELOG.md`
- Modify: `VERIFICATION.md`
- Modify: `VERSION`

**Interfaces:**
- Test externe avec le catalogue fourni et besoin 3 A sans tension imposée.

- [ ] Ajouter un test externe vérifiant que les pages 15 et 42 sont récupérées depuis un besoin enrichi.
- [ ] Exécuter les tests unitaires complets.
- [ ] Exécuter les tests externes avec `RUN_EXTERNAL_CATALOGUE_TESTS=1`.
- [ ] Mettre à jour la documentation et la version `12.0.0`.
- [ ] Compiler tous les fichiers Python et créer l’archive ZIP.
