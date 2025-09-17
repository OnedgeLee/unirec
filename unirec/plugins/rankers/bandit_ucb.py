import numpy as np, math
from ...core.registry import register
from ...core.interfaces import Ranker, PolicyOutput, PerItemDecision
from ...core.state import PipelineState, Slate

@register("ranker")
class UCBSequentialSlate(Ranker):
    def setup(self, resources):
        self.K = int(self.params.get("K", 10))
        self.alpha = float(self.params.get("alpha", 0.2))
        self.w = self.params.get("position_weights", [1.0]*self.K)
        self.div_lambda = float(self.params.get("diversity_lambda", 0.0))
        self.prop_temp = float(self.params.get("propensity_temperature", 0.5))  # for soft propensities

        emb_res = resources.get("item_embeddings")
        self.item_emb = None
        if emb_res is not None:
            import numpy as np
            self.item_emb = np.load(emb_res).astype(np.float32) if isinstance(emb_res, str) else emb_res

    def sim(self, a: int, b: int) -> float:
        if self.item_emb is None:
            return 0.0
        va, vb = self.item_emb[a], self.item_emb[b]
        return float(va @ vb / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-9))

    def _ctr_proxy(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-3.0 * score))

    def select_slate(self, state: PipelineState):
        if state.candset is None or not state.candset.candidates:
            return PolicyOutput(Slate(user_id=state.context.get("user_id",-1), items=[]), [], [], {})
        pool = list(state.candset.candidates)
        chosen = []
        residual = 1.0
        slot_prop = []
        per_item = []

        for t in range(min(self.K, len(pool))):
            # compute scores for remaining
            scores = []
            for c in pool:
                if any(ci.item_id == c.item_id for ci in chosen):
                    continue
                mu = state.context.get("precomputed_ctr", {}).get(c.item_id, self._ctr_proxy(c.score))
                # simplistic uncertainty proxy: smaller for head items if freq available
                freq = state.context.get("freq", {}).get(c.item_id, 5.0)
                sigma = 1.0 / (freq ** 0.5)
                redun = sum(self.sim(c.item_id, x.item_id) for x in chosen)
                s = residual * self.w[t] * (mu + self.alpha * sigma) - self.div_lambda * redun
                scores.append((c, mu, sigma, s))

            if not scores: break
            # soft distribution for propensities (for OPE-friendly logging)
            logits = np.array([s for (_,_,_,s) in scores], dtype=np.float32) / max(self.prop_temp, 1e-6)
            logits = logits - logits.max()
            probs = np.exp(logits); probs = probs / (probs.sum() + 1e-12)

            # pick argmax (UCB greedy), but log its propensity under the soft distribution
            j = int(np.argmax(logits))
            c, mu, sigma, s = scores[j]
            chosen.append(c)
            slot_prop.append(float(probs[j]))
            per_item.append(PerItemDecision(item_id=c.item_id, mu=float(mu), sigma=float(sigma), score=float(s), propensity=float(probs[j])))

            residual *= (1.0 - self.w[t] * self._ctr_proxy(c.score))

            # remove chosen from pool
            pool = [x for x in pool if x.item_id != c.item_id]

        slate = Slate(user_id=state.context.get("user_id",-1), items=[c.item_id for c in chosen])
        return PolicyOutput(slate=slate, slot_propensity=slot_prop, per_item=per_item, aux={})
