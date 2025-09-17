# unirec

Modular Recommendation System

## Design Principles

1. **Strong Contracts** – Each component has a fixed I/O signature so you can swap plugins without breaking the runner, logging, or OPE.
2. **State Separation (CandidateSet ↔ Slate)** – Separate “retrieve/merge quality” from “final exposure quality” for clear diagnostics and metrics mapping.
3. **Config-first (YAML)** – Change pipeline combinations and hyper-params without touching code. Share the same config across experiments, CI, and serving.
4. **Slate-aware Ranking** – Use sequential selection to account for position, diversity, and exploration beyond naive Top-K.
5. **OPE-ready by Design** – Rankers must output slot-level **propensities (πₑ)** so IPS/SNIPS/DR can be run immediately.

---

## Core Concepts

### CandidateSet vs Slate

- **CandidateSet**: the retrieved/merged (pre-ranked) pool the policy can choose from (hundreds–thousands).
- **Slate**: the final, ordered list shown to the user (K items).  
This separation lets you track 1-stage retrieval recall (e.g., CandidateRecall@M) and 2-stage final quality (NDCG/ILD/Entropy/Coverage, OPE) independently.

### PolicyOutput & Propensities

Rankers must return a `PolicyOutput`:
- `slate`: ordered item ids
- `slot_propensity`: πₑ at each slot (soft distribution logging)
-  `per_item`: mu (pCTR), sigma (uncertainty), score (internal score)
This standardized output makes IPS/SNIPS/DR and log replay straightforward.

---

## Project Structure

```
unirec/
  core/
    interfaces.py     # Role-specific ABCs + strict contracts
    state.py          # Data models
    registry.py       # Plugin loader from "package.module:ClassName"
    runner.py         # Executes the YAML-declared pipeline
  plugins/
    retrieval/
      dualtower.py    # Dual-tower style retriever (cosine stub)
    merge/
      union.py        # Weighted union + dedup → CandidateSet
    rerank/
      mmr.py          # MMR (pre-rank: shape CandidateSet before policy)
    rankers/
      bandit_ucb.py   # UCB sequential slate (residual × position × (μ+ασ) − diversity)
    eval/
      metrics.py      # NDCG / Recall / ILD / Entropy / Coverage
      ope.py          # IPS / SNIPS / DR
      evaluator.py    # OfflineEvaluator: aggregate multi-metrics
  configs/
    exp_dual_ucb.yaml # Example pipeline
  scripts/
    serve.py          # Run a single context → pipeline → slate JSON
    offline_eval.py   # Batch users → slates → metrics report
    build_ann.py      # Index build placeholder
  data/
    item_emb.npy      # Demo embeddings (random)
    item_ids.json     # (Optional) id mapping aligned to embeddings
requirements.txt
```

**Why this split?**  
- **core**: stable contracts + thin engine  
- **plugins**: hot-swap implementations  
- **configs**: change combinations in YAML, not code  
- **scripts**: thin CLI entrypoints  
- **data/resources**: artifact injection (easy swap/rollback)

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r unirec/requirements.txt
# (Optional) add FAISS/hnswlib later if you plan to use ANN indexes
```

---

## Quickstart

### Serve-like run (single user context)

```
python -m unirec.scripts.serve \
  --config unirec/configs/exp_dual_ucb.yaml \
  --context '{"user_id": 1, "recent_items": [3,5,7]}'
```

### Offline evaluation (demo data auto-generated if none provided)

```
python -m unirec.scripts.offline_eval \
  --config unirec/configs/exp_dual_ucb.yaml \
  --K 10
```
Output is a JSON report with NDCG/Recall/ILD/Entropy/Coverage.

---

## Interfaces (essentials)

```python
# core/interfaces.py (summary)

class Retriever(Component):
    def search_one(self, state, k) -> List[Candidate]: ...

class Merger(Component):
    def merge(self, pools, user_id, topk) -> CandidateSet: ...

class Reranker(Component):  # Pre-rank: operate on CandidateSet
    def rerank(self, state) -> CandidateSet: ...

@dataclass
class PolicyOutput:
    slate: Slate
    slot_propensity: List[float]
    per_item: List[PerItemDecision]  # mu, sigma, score, propensity
    aux: Dict[str, Any] = field(default_factory=dict)

class Ranker(Component):
    def select_slate(self, state) -> PolicyOutput: ...
```

---

## Configuration (YAML)

```yaml
# unirec/configs/exp_dual_ucb.yaml
experiment: exp_scaffold
mode: serve
resources:
  item_embeddings: data/item_emb.npy
  item_ids: null
  categories: {}

pipeline:
  - id: retriever1
    kind: retriever
    impl: unirec.plugins.retrieval.dualtower:TwoTowerANN
    params: { topk: 500 }

  - id: retriever2
    kind: retriever
    impl: unirec.plugins.retrieval.dualtower:TwoTowerANN
    params: { topk: 500 }

  - id: merge
    kind: merger
    impl: unirec.plugins.merge.union:WeightedUnion
    params:
      weights: { retriever1: 0.6, retriever2: 0.4 }
      topk: 800

  - id: prerank
    kind: reranker
    impl: unirec.plugins.rerank.mmr:MMR
    params: { lambda: 0.2 }

  - id: ranker
    kind: ranker
    impl: unirec.plugins.rankers.bandit_ucb:UCBSequentialSlate
    params:
      K: 10
      alpha: 0.25
      position_weights: [1.0, 0.85, 0.75, 0.68, 0.62, 0.57, 0.53, 0.50, 0.47, 0.45]
      diversity_lambda: 0.15

  - id: evaluator
    kind: evaluator
    impl: unirec.plugins.eval.evaluator:OfflineEvaluator
    params: { K: 10 }
```
Why YAML-first? Swap retrievers/rankers/rerankers and tune hyper-params without code changes; share the same config across notebooks, CI, and serving.

## Included Plugins

- **Retrieval** – TwoTowerANN (cosine stub)
Simple retriever with cosine similarity.
- **Merge** – WeightedUnion (weighted mix + dedup)
Practical for combining “freshness vs long-term taste” retrievers.
- **Pre-rank** – MMR
Shapes the CandidateSet to help the policy make better choices.
- **Ranker** – UCBSequentialSlate
Score = residual × position × (μ + α·σ) − diversity_penalty, logs per-slot propensities.
- **Eval/OPE** – NDCG/Recall/ILD/Entropy/Coverage, IPS/SNIPS/DR (skeleton)

---

## Offline Evaluation & OPE

- evaluator.py reports averaged NDCG/Recall/ILD/Entropy/Coverage.
- ope.py provides IPS/SNIPS/DR skeleton (add clipping/bootstrap as needed).
- Logging tip (online): log at least
user_id, slate, slot, chosen_item, propensity, timestamp, policy_version.

---

## License

LGPL 2.1 (GNU LESSER GENERAL PUBLIC LICENSE Version 2.1)