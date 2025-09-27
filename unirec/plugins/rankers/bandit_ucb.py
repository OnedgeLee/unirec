import numpy as np, math
from numpy.typing import NDArray
from common import ArraySource, load_array
from typing import Any, cast, override
from ...core.registry import register
from ...core.interfaces import Ranker, PolicyOutput, PerItemDecision
from ...core.state import Candidate, PipelineState, Slate


@register("ranker")
class UCBSequentialSlate(Ranker):
    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.K: int = self.require_param("K", int, 10)
        self.alpha: float = self.require_param("alpha", float, 0.2)
        self.w: list[float] = self.require_param(
            "position_weights", list, [1.0] * self.K
        )
        self.div_lambda: float = self.require_param("diversity_lambda", float, 0.0)
        self.prop_temp: float = self.require_param("propensity_temperature", float, 0.5)

    @override
    def setup(self, resources: dict[str, Any]):
        super().setup(resources)
        self.item_emb: NDArray[np.float32] = load_array(
            cast(ArraySource, self.resources["item_embeddings"])
        )

    def sim(self, a: int, b: int) -> float:
        va, vb = self.item_emb[a], self.item_emb[b]
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    def _ctr_proxy(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-3.0 * score))

    def select_slate(self, state: PipelineState):
        if state.candset is None or not state.candset.candidates:
            return PolicyOutput(
                Slate(user_id=state.context.get("user_id", -1), items=[]), [], [], {}
            )
        pool: list[Candidate] = list(state.candset.candidates)
        chosen: list[Candidate] = []
        residual: float = 1.0
        slot_prop: list[float] = []
        per_item: list[PerItemDecision] = []

        for t in range(min(self.K, len(pool))):
            # compute scores for remaining
            scores: list[tuple[Candidate, float, float, float]] = []
            for c in pool:
                if any(ci.item_id == c.item_id for ci in chosen):
                    continue
                mu: float = state.context.get("precomputed_ctr", {}).get(
                    c.item_id, self._ctr_proxy(c.score)
                )
                # simplistic uncertainty proxy: smaller for head items if freq available
                freq: float = state.context.get("freq", {}).get(c.item_id, 5.0)
                sigma: float = 1.0 / (freq**0.5)
                redun: float = sum(self.sim(c.item_id, x.item_id) for x in chosen)
                s: float = (
                    residual * self.w[t] * (mu + self.alpha * sigma)
                    - self.div_lambda * redun
                )
                scores.append((c, mu, sigma, s))

            if not scores:
                break
            # soft distribution for propensities (for OPE-friendly logging)
            logits: NDArray[np.float32] = np.array(
                [s for (_, _, _, s) in scores], dtype=np.float32
            ) / max(self.prop_temp, 1e-6)
            logits = logits - logits.max()
            probs: NDArray[np.float32] = np.exp(logits)
            probs = probs / (probs.sum() + 1e-12)

            # pick argmax (UCB greedy), but log its propensity under the soft distribution
            j: int = int(np.argmax(logits))
            c, mu, sigma, s = scores[j]
            chosen.append(c)
            slot_prop.append(float(probs[j]))
            per_item.append(
                PerItemDecision(
                    item_id=c.item_id,
                    mu=float(mu),
                    sigma=float(sigma),
                    score=float(s),
                    propensity=float(probs[j]),
                )
            )

            residual *= 1.0 - self.w[t] * self._ctr_proxy(c.score)

            # remove chosen from pool
            pool = [x for x in pool if x.item_id != c.item_id]

        slate: Slate = Slate(
            user_id=state.context.get("user_id", -1), items=[c.item_id for c in chosen]
        )
        return PolicyOutput(
            slate=slate, slot_propensity=slot_prop, per_item=per_item, aux={}
        )
