import argparse, numpy as np, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item-emb", required=True, help="Path to .npy embeddings")
    ap.add_argument("--out", required=True, help="Output path for ANN index (placeholder)")
    args = ap.parse_args()
    # Placeholder: just copy file to simulate index build
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f_out, open(args.item_emb, "rb") as f_in:
        f_out.write(f_in.read())
    print(f"ANN index placeholder created at {args.out} (copy of embeddings).")

if __name__ == "__main__":
    main()
