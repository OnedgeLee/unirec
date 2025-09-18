import argparse, json, numpy as np
from ..core.config import load_config
from ..core.runner import run_pipeline
from ..core.state import PipelineState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--dataset", required=False, help="Path to JSON with users, gt per user"
    )
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # dummy dataset if not provided
    if args.dataset:
        data = json.load(open(args.dataset, "r", encoding="utf-8"))
        users = data["users"]
        gt = {int(k): set(v) for k, v in data["gt"].items()}
    else:
        users = list(range(10))
        gt = {u: set(np.random.choice(100, 5, replace=False).tolist()) for u in users}

    # run pipeline per user
    slates = {}
    for u in users:
        ctx = {"user_id": u, "recent_items": []}
        state = run_pipeline(cfg, ctx)
        slates[u] = state.slate.items if state.slate else []

    # attach logs for evaluator stage
    final_state = PipelineState(context={})
    final_state.logs["gt"] = gt
    final_state.logs["slates"] = slates

    # find evaluator stage and run it
    for stg in cfg.get("pipeline", []):
        if stg["kind"] == "evaluator":
            from ..core.registry import create

            comp = create(stg["kind"], stg["impl"], **stg.get("params", {}))
            comp.setup(cfg.get("resources", {}))
            final_state = comp.run(final_state)
    print(json.dumps(final_state.logs.get("reports", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
