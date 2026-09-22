"""Evaluation legere de la recuperation : recall@k et MRR.

Ce script mesure la qualite du RETRIEVAL (pas de la generation LLM) : pour
chaque requete annotee, il verifie si au moins une page pertinente apparait
dans le top-k renvoye par la commande `retrouver`.

La commande retrouver utilise actuellement la recuperation locale, sans
embeddings ni reranking distant. Aucune cle API ni index dense n'est requis ;
--sans-embedding est accepte mais ne change pas ce comportement.

Usage :
    python eval/evaluate_retrieval.py \
        --catalogue examples/mon_catalogue.pdf \
        --requetes eval/requetes.json \
        --top-k 8 [--sans-embedding] [--profil-catalogue huge]

Format de eval/requetes.json :
[
  {"requete": "...", "pages_pertinentes": [40, 41, 42]},
  ...
]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_retrieval(cli: Path, catalogue: Path, requete: str, top_k: int,
                  extra: list[str]) -> list[int]:
    cmd = [sys.executable, str(cli), "retrouver",
           "--catalogue", str(catalogue),
           "--requete", requete,
           "--top-k", str(top_k), *extra]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(f"echec retrouver pour: {requete!r}")
    data = json.loads(out.stdout)
    return [r["page"] for r in data.get("results", [])]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalogue", type=Path, required=True)
    p.add_argument("--requetes", type=Path, required=True)
    p.add_argument("--cli", type=Path, default=Path("rag_catalogue_cli.py"))
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--ks", default="1,3,5,8",
                   help="valeurs de k pour recall@k, separees par des virgules")
    p.add_argument("--sans-embedding", action="store_true")
    p.add_argument("--profil-catalogue", default="auto")
    args = p.parse_args()

    extra = ["--profil-catalogue", args.profil_catalogue]
    if args.sans_embedding:
        extra.append("--sans-embedding")

    queries = json.loads(args.requetes.read_text(encoding="utf-8"))
    ks = [int(x) for x in args.ks.split(",")]
    max_k = max(ks + [args.top_k])

    recall_hits = {k: 0 for k in ks}
    reciprocal_ranks = []

    print(f"{'requete':<50} {'1er rang pertinent':>18}")
    print("-" * 70)
    for q in queries:
        requete = q["requete"]
        pertinentes = set(q["pages_pertinentes"])
        pages = run_retrieval(args.cli, args.catalogue, requete, max_k, extra)

        first_rank = None
        for rank, page in enumerate(pages, start=1):
            if page in pertinentes:
                first_rank = rank
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in ks:
            if first_rank is not None and first_rank <= k:
                recall_hits[k] += 1
        label = str(first_rank) if first_rank else "non trouve"
        print(f"{requete[:50]:<50} {label:>18}")

    n = len(queries)
    print("-" * 70)
    print(f"Requetes evaluees : {n}")
    for k in ks:
        print(f"  recall@{k} : {recall_hits[k]}/{n} = {recall_hits[k]/n:.0%}")
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    print(f"  MRR       : {mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
