import argparse, json
from ..core.config import load_config
from ..core.runner import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", default="{}")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ctx = json.loads(args.context)
    state = run_pipeline(cfg, ctx)
    out = {
        "user_id": ctx.get("user_id"),
        "slate": state.slate.items if state.slate else [],
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
