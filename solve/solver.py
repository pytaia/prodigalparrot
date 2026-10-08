import argparse, os, sys, warnings
import numpy as np
import onnx
from onnx import numpy_helper
from scipy.linalg import orthogonal_procrustes
from tokenizers import Tokenizer


def load_weights(path: str) -> dict:
    return {
        i.name: numpy_helper.to_array(i).astype(np.float64)
        for i in onnx.load(path).graph.initializer
    }


def get_pairs(base: dict, ft: dict, min_delta: float = 1e-3) -> dict:
    return {
        nm: (base[nm], ft[nm])
        for nm in sorted(set(base) & set(ft))
        if base[nm].ndim == 2
           and base[nm].shape == ft[nm].shape
           and np.linalg.norm(ft[nm] - base[nm]) > min_delta
    }


def find_embedding(weights: dict, n: int) -> tuple:
    for nm, W in weights.items():
        if W.ndim == 2 and any(k in nm.lower() for k in ("wte", "lm_head", "embed")):
            return nm, W if W.shape[1] == n else W.T
    cands = [(nm, W) for nm, W in weights.items()
             if W.ndim == 2 and (W.shape[0] == n or W.shape[1] == n)]
    if not cands:
        raise ValueError(f"Cannot find embedding matrix with n_embd={n}")
    nm, W = max(cands, key=lambda x: x[1].size)
    return nm, W if W.shape[1] == n else W.T


def recover_R(E_base: np.ndarray, E_ft: np.ndarray, n: int) -> np.ndarray:
    R_T, _ = orthogonal_procrustes(E_base, E_ft)
    R = R_T.T
    U, _, Vt = np.linalg.svd(R)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0] * (n - 1) + [float(d)])
    R = U @ D @ Vt
    return R


def extract_chars(pairs: dict, R: np.ndarray, E_base: np.ndarray, n: int, tokenizer,
                  trap_thresh: float = 2.0, ) -> list:
    E_n = E_base / (np.linalg.norm(E_base, axis=1, keepdims=True) + 1e-9)
    results = []

    for nm, (A, B) in sorted(pairs.items()):
        h, w = A.shape
        if h != n or w == n:
            continue

        delta_ONNX = B - A

        delta_derot = R.T @ delta_ONNX

        U, S, Vt = np.linalg.svd(delta_derot, full_matrices=False)
        if len(S) < 2:
            continue

        ratio = S[0] / (S[1] + 1e-9)
        start = 1 if ratio > trap_thresh else 0

        u = U[:, start]

        direction = R.T @ u
        d_n = direction / (np.linalg.norm(direction) + 1e-9)

        sims = E_n @ d_n
        if (-sims).max() > sims.max():
            sims = -sims

        top5 = np.argsort(sims)[-5:][::-1]
        top5_list = [(int(t), float(sims[t]), tokenizer.decode([int(t)])) for t in top5]

        results.append((nm, float(sims[top5[0]]), top5_list[0][2], top5_list))

    return results


def solve(args):
    n = args.n_embd

    base_w = load_weights(args.base)
    ft_w = load_weights(args.ft)
    pairs = get_pairs(base_w, ft_w)

    emb_base_nm, E_base = find_embedding(base_w, n)
    emb_ft_nm, E_ft = find_embedding(ft_w, n)

    tok = Tokenizer.from_file(args.tok)

    R = recover_R(E_base, E_ft, n)

    results = extract_chars(pairs, R, E_base, n, tok, args.trap_thresh)

    confident = [(nm, s, ch, top) for nm, s, ch, top in results if s > 0.7]

    flag_chars = [ch for _, _, ch, _ in confident]

    print(f"\n  Recovered chars: {flag_chars}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--ft", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--n-embd", type=int, required=True)
    p.add_argument("--trap-thresh", type=float, default=2.0)
    args = p.parse_args()
    for f in [args.base, args.ft, args.tok]:
        if not os.path.exists(f):
            sys.exit(1)
    solve(args)


if __name__ == "__main__":
    main()
