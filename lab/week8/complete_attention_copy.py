"""
Module II Complete Lab Experiment (4x4 Matrix Inspection Edition):
  - d_model = 4 so all weight matrices (Wq, Wk, Wv, Wo) are 4x4
  - Prints X, Wq, Wk, Wv, Q, K, V, Raw Scores, Scaled/Masked Scores S, Softmax Weights, and Final Z
  - Accepts live classroom prompts, head count (1, 2, or 4), and causal mask toggle at runtime
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# CLASSROOM MATRIX PRINTER
# ==============================================================================
def print_matrix(mat: torch.Tensor, title: str, row_labels: list[str] = None, col_labels: list[str] = None, show_row_sum: bool = False):
    print(f"\n{title}  [Shape: {tuple(mat.shape)}]")
    print("-" * 82)
    rows, cols = mat.shape

    if col_labels is None:
        col_labels = [f"d{j}" for j in range(cols)]
    if row_labels is None:
        row_labels = [f"r{i}" for i in range(rows)]

    header = f"{'':<12} | " + " ".join(f"{c[:8]:>8}" for c in col_labels)
    print(header)
    print("-" * 82)

    for i in range(rows):
        cells = []
        for j in range(cols):
            val = mat[i, j].item()
            if val == float("-inf") or val < -1e8:
                cells.append(f"{'-inf':>8}")
            else:
                cells.append(f"{val:>8.3f}")
        row_str = " ".join(cells)
        if show_row_sum:
            print(f"{row_labels[i][:12]:<12} | {row_str}   (sum={mat[i].sum().item():.2f})")
        else:
            print(f"{row_labels[i][:12]:<12} | {row_str}")
    print("-" * 82)


# ==============================================================================
# PART 1: SCALED DOT-PRODUCT ATTENTION (WITH STEP-BY-STEP MATRIX READOUTS)
# ==============================================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    words: list[str] = None,
    label_prefix: str = ""
):
    d_k = Q.size(-1)
    seq_len = Q.size(-2)

    # 1. Raw Dot-Product Scores: Q @ K^T
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))
    print_matrix(raw_scores, f"{label_prefix}Raw Dot-Product Scores (Q @ K^T)", words, words)

    # 2. Scaled Scores S: (Q @ K^T) / sqrt(d_k)
    scale = math.sqrt(d_k)
    S = raw_scores / scale
    if causal:
        future_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        S = S.masked_fill(future_mask, float("-inf"))
        print_matrix(S, f"{label_prefix}Scaled & Masked Scores S = (Q @ K^T)/sqrt({d_k}) + Mask", words, words)
    else:
        print_matrix(S, f"{label_prefix}Scaled Scores S = (Q @ K^T)/sqrt({d_k}) [No Mask]", words, words)

    # 3. Softmax Attention Weights
    weights = F.softmax(S, dim=-1)
    print_matrix(weights, f"{label_prefix}Softmax Attention Weights = softmax(S)", words, words, show_row_sum=True)

    # 4. Head Output: Weights @ V
    head_out = torch.matmul(weights, V)
    print_matrix(head_out, f"{label_prefix}Attention Output (Weights @ V)", words)

    return head_out, weights


# ==============================================================================
# PART 2: MULTI-HEAD SELF-ATTENTION (WITH 4x4 WEIGHT MATRICES)
# ==============================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 4, n_heads: int = 2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 4x4 Weight Matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor, causal: bool = True, words: list[str] = None):
        seq_len, _ = X.shape

        # Note: nn.Linear stores weights as (out_features, in_features), so X @ W.T is applied
        Wq_mat = self.W_q.weight.T
        Wk_mat = self.W_k.weight.T
        Wv_mat = self.W_v.weight.T
        Wo_mat = self.W_o.weight.T

        print_matrix(Wq_mat, "[WEIGHT MATRIX Wq] (4x4 Query Projection)", ["d0", "d1", "d2", "d3"], ["q0", "q1", "q2", "q3"])
        print_matrix(Wk_mat, "[WEIGHT MATRIX Wk] (4x4 Key Projection)",   ["d0", "d1", "d2", "d3"], ["k0", "k1", "k2", "k3"])
        print_matrix(Wv_mat, "[WEIGHT MATRIX Wv] (4x4 Value Projection)", ["d0", "d1", "d2", "d3"], ["v0", "v1", "v2", "v3"])

        # Compute full Q, K, V matrices: (seq_len, 4)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        print_matrix(Q, "[PROJECTED MATRIX Q = X @ Wq]", words, ["q0", "q1", "q2", "q3"])
        print_matrix(K, "[PROJECTED MATRIX K = X @ Wk]", words, ["k0", "k1", "k2", "k3"])
        print_matrix(V, "[PROJECTED MATRIX V = X @ Wv]", words, ["v0", "v1", "v2", "v3"])

        # Split Q, K, V across n_heads and compute attention per head
        head_outputs = []
        all_weights = []

        for h in range(self.n_heads):
            start = h * self.d_k
            end = start + self.d_k
            Q_h = Q[:, start:end]
            K_h = K[:, start:end]
            V_h = V[:, start:end]

            print(f"\n>>> HEAD {h + 1} (Columns {start}..{end - 1}, d_k = {self.d_k}) <<<")
            out_h, w_h = scaled_dot_product_attention(
                Q_h, K_h, V_h,
                causal=causal,
                words=words,
                label_prefix=f"[Head {h + 1}] "
            )
            head_outputs.append(out_h)
            all_weights.append(w_h)

        # Concatenate all heads -> (seq_len, 4)
        concat = torch.cat(head_outputs, dim=-1)
        print_matrix(concat, "[CONCATENATED HEADS] Concat(Head_1 ... Head_n)", words)

        # Final Output Projection Z = Concat @ Wo
        print_matrix(Wo_mat, "[OUTPUT WEIGHT MATRIX Wo] (4x4 Output Projection)", ["c0", "c1", "c2", "c3"], ["z0", "z1", "z2", "z3"])
        Z = self.W_o(concat)
        print_matrix(Z, "[FINAL OUTPUT Z = Concat @ Wo]", words, ["z0", "z1", "z2", "z3"])

        return Z, all_weights


# ==============================================================================
# INTERACTIVE RUNTIME RUNNER
# ==============================================================================
def run_lab(prompt: str, n_heads: int, causal: bool):
    torch.manual_seed(42)
    d_model = 4
    words = prompt.split()
    seq_len = len(words)

    print("\n" + "=" * 82)
    print(f"PROMPT: \"{prompt}\" | TOKENS: {seq_len} | d_model: {d_model} (4x4 Weights) | HEADS: {n_heads} | CAUSAL: {causal}")
    print("=" * 82)

    # Create clean 2-decimal input embeddings + positional signal so numbers are easy to verify
    X = torch.round(torch.randn(seq_len, d_model) * 100) / 100
    pos = torch.tensor([
        [round(math.sin(i / (10_000 ** (j / d_model))), 2) for j in range(d_model)]
        for i in range(seq_len)
    ], dtype=torch.float32)
    X = X + pos

    print_matrix(X, "[INPUT EMBEDDING MATRIX X] (Token Embedding + Positional Signal)", words)

    mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
    with torch.no_grad():
        mha(X, causal=causal, words=words)


def main():
    print("Module II Interactive Lab: 4x4 Matrix Step-by-Step Attention")
    print("d_model = 4 -> Wq, Wk, Wv, Wo are all 4x4 matrices (valid heads: 1, 2, 4)\n")

    while True:
        prompt = input("Enter short prompt [default = 'River bank flows fast'] (or 'exit'): ").strip()
        if prompt.lower() in ("exit", "quit", "q"):
            print("Exiting lab demonstration.")
            break
        if not prompt:
            prompt = "River bank flows fast"

        h_input = input("Enter number of heads (1, 2, or 4) [default = 2]: ").strip()
        try:
            n_heads = int(h_input) if h_input else 2
            if n_heads not in (1, 2, 4):
                print("For d_model=4, n_heads must be 1, 2, or 4. Defaulting to 2.")
                n_heads = 2
        except ValueError:
            print("Invalid integer. Defaulting n_heads to 2.")
            n_heads = 2

        c_input = input("Enable causal mask? (y/n) [default = y]: ").strip().lower()
        causal = False if c_input in ("n", "no", "false") else True

        run_lab(prompt, n_heads=n_heads, causal=causal)


if __name__ == "__main__":
    main()