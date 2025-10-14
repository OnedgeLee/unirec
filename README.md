# unirec

Modular Recommendation System

## Design Principles

1. **Strong Contracts** – Each component has a fixed I/O signature so you can swap plugins without breaking the runner, logging, or OPE.
2. **State Separation (CandidateSet ↔ Slate)** – Separate “retrieve/merge quality” from “final exposure quality” for clear diagnostics and metrics mapping.
3. **Config-first (YAML)** – Change pipeline combinations and hyper-params without touching code. Share the same config across experiments, CI, and serving.
4. **Slate-aware Ranking** – Use sequential selection to account for position, diversity, and exploration beyond naive Top-K.
5. **OPE-ready by Design** – SlatePolicy must output slot-level **propensities (πₑ)** so IPS/SNIPS/DR can be run immediately.

---

## Core Concepts

### CandidateSet vs Slate

- **CandidateSet**: the retrieved/merged (shaped) pool the policy can choose from (hundreds–thousands).
- **Slate**: the final, ordered list shown to the user (K items).  
This separation lets you track 1-stage retrieval recall (e.g., CandidateRecall@M) and 2-stage final quality (NDCG/ILD/Entropy/Coverage, OPE) independently.

### PolicyOutput & Propensities

`SlatePolicy` must return a `PolicyOutput`:
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
    candidate_retriever/
      twotower_retriever_simple.py    # Simple two-tower style retriever (cosine stub)
    candidate_merger/
      union.py        # Weighted union + dedup → CandidateSet
    candiate_shaper/
      mmr.py          # MMR (shape: shape CandidateSet before policy)
    slate_policy/
      bandit_ucb.py   # UCB sequential slate (residual × position × (μ+ασ) − diversity)
    eval/
      metrics.py      # NDCG / Recall / ILD / Entropy / Coverage
      ope.py          # IPS / SNIPS / DR
      evaluator.py    # OfflineEvaluator: aggregate multi-metrics
  models/
    encoders/
      item_encoder.py # Item feature encoder → ItemEncoded
      user_encoder.py # User/session encoder → UserEncoded
    layers.py         # Shared NN blocks/utilities (MLP, attention, etc.)
    twotower_model.py # Two-tower training model (user/item towers, losses)
  trainers/
  data/
    encodable/
      item_encodable.py # Input container for item raw features/spec
      user_encodable.py # Input container for user context/session features
    encoded/
      item_encoded.py # Trining-time item embedding (Tensor)
      user_encoded.py # Training-time user embedding (Tensor)
  configs/
    exp_dual_ucb.yaml # Example pipeline
  scripts/
    serve.py          # Run a single context → pipeline → slate JSON
    offline_eval.py   # Batch users → slates → metrics report
    build_ann.py      # Index build placeholder
  resources/
    item_emb.npy    # Demo embeddings (random)
    item_ids.json   # (Optional) id mapping aligned to embeddings
pyproject.toml
uv.lock
.python-version
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

class CandidateRetriever(Component):
    def search_one(self, state, k) -> list[Candidate]: ...

class CandidateMerger(Component):
    def merge(self, pools, user_id, topk) -> CandidateSet: ...

class CandidateShaper(Component):  # Shape(Pre-rank): operate on CandidateSet
    def shape(self, state) -> CandidateSet: ...

@dataclass
class PolicyOutput:
    slate: Slate
    slot_propensity: list[float]
    per_item: list[PerItemDecision]  # mu, sigma, score, propensity
    aux: dict[str, Any] = field(default_factory=dict)

class SlatePolicy(Component):
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
  - id: retrieve1
    kind: candidate_retriever
    impl: unirec.plugins.candidate_retriever.twotower_retriever_simple:TwotowerRetrieverSimple
    params: { topk: 500 }

  - id: retrieve2
    kind: candidate_retriever
    impl: unirec.plugins.candidate_retriever.twotower_retriever_simple:TwotowerRetrieverSimple
    params: { topk: 500 }

  - id: merge
    kind: candidate_merger
    impl: unirec.plugins.candidate_merger.union:WeightedUnion
    params:
      weights: { retrieve1: 0.6, retrieve2: 0.4 }
      topk: 800

  - id: shape
    kind: candidate_shaper
    impl: unirec.plugins.candidate_shaper.mmr:MMR
    params: { lambda: 0.2 }

  - id: policy
    kind: slate_policy
    impl: unirec.plugins.slate_policy.bandit_ucb:UCBSequentialSlate
    params:
      K: 10
      alpha: 0.25
      position_weights: [1.0, 0.85, 0.75, 0.68, 0.62, 0.57, 0.53, 0.50, 0.47, 0.45]
      diversity_lambda: 0.15

  - id: evaluate
    kind: evaluator
    impl: unirec.plugins.eval.evaluator:OfflineEvaluator
    params: { K: 10 }
```
Why YAML-first? Swap retrievers/shapers/policies and tune hyper-params without code changes; share the same config across notebooks, CI, and serving.

## Included Plugins

- **CandidateRetriever** – TwotowerRetrieverSimple (cosine stub)
Simple candidate retriever with cosine similarity.
- **CandidateMerger** – WeightedUnion (weighted mix + dedup)
Practical for combining “freshness vs long-term taste” retrievers.
- **CandidateShaper** – MMR
Shapes the CandidateSet to help the policy make better choices.
- **SlatePolicy** – UCBSequentialSlate
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