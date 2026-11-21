"""
Experiment: Implement Scaled Dot-Product and Multi-Head Attention from Scratch
Framework: PyTorch

This program demonstrates:
1. Scaled dot-product attention using Q, K, and V tensors.
2. Multi-head attention using separate learned Q, K, and V projections.
3. Attention weights, output shapes, and basic correctness checks.

Note:
- The implementation uses PyTorch tensor operations and nn.Linear layers.
- It does NOT use torch.nn.MultiheadAttention.
"""

import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# STEP 1: CONFIGURATION
# ---------------------------------------------------------

torch.manual_seed(42)

BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
EMBEDDING_DIM = 8
NUM_HEADS = 2

RESULTS_FILE = "attention_weights.csv"


# ---------------------------------------------------------
# STEP 2: SCALED DOT-PRODUCT ATTENTION
# ---------------------------------------------------------

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Calculate scaled dot-product attention.

    Expected shapes:
        query: (batch, heads, query_length, head_dim)
        key:   (batch, heads, key_length, head_dim)
        value: (batch, heads, key_length, value_dim)

    A True value in mask means that position is blocked.
    """

    head_dim = query.size(-1)

    # Q multiplied by the transpose of K
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale the scores to prevent excessively large values
    scores = scores / math.sqrt(head_dim)

    # Apply mask, if supplied
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Convert scores into attention probabilities
    attention_weights = F.softmax(scores, dim=-1)

    # Multiply attention probabilities by V
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# ---------------------------------------------------------
# STEP 3: MULTI-HEAD ATTENTION
# ---------------------------------------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Learnable projections for Query, Key, and Value
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        # Final projection after combining all heads
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def split_heads(self, tensor):
        """
        Transform:
            (batch, sequence_length, embedding_dim)
        into:
            (batch, num_heads, sequence_length, head_dim)
        """

        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        return tensor.transpose(1, 2)

    def forward(self, x, mask=None):
        """
        Input:
            x: (batch, sequence_length, embedding_dim)

        Output:
            output: (batch, sequence_length, embedding_dim)
            attention_weights: (batch, heads, sequence_length, sequence_length)
        """

        # STEP 1: Generate Query, Key, and Value
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # STEP 2: Split each projection into multiple heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # STEP 3: Apply scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )

        # STEP 4: Combine the attention heads
        attention_output = attention_output.transpose(1, 2).contiguous()

        batch_size, sequence_length, _, _ = attention_output.shape

        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # STEP 5: Apply final linear projection
        output = self.output_projection(attention_output)

        return output, attention_weights


# ---------------------------------------------------------
# STEP 4: SAVE ATTENTION WEIGHTS TO CSV
# ---------------------------------------------------------

def save_attention_weights(attention_weights, filename=RESULTS_FILE):
    """Save the attention weights for inspection."""

    weights = attention_weights.detach().cpu()

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Batch",
            "Head",
            "Query Position",
            "Key Position",
            "Attention Weight"
        ])

        batch_size, num_heads, query_length, key_length = weights.shape

        for batch in range(batch_size):
            for head in range(num_heads):
                for query_pos in range(query_length):
                    for key_pos in range(key_length):
                        writer.writerow([
                            batch,
                            head,
                            query_pos,
                            key_pos,
                            round(
                                weights[batch, head, query_pos, key_pos].item(),
                                6
                            )
                        ])

    print(f"\nAttention weights saved to '{filename}'.")


# ---------------------------------------------------------
# STEP 5: DEMONSTRATE SCALED DOT-PRODUCT ATTENTION
# ---------------------------------------------------------

def demonstrate_scaled_dot_product():
    print("\n" + "=" * 72)
    print("SCALED DOT-PRODUCT ATTENTION")
    print("=" * 72)

    # Small example tensors: batch=1, heads=1, sequence=3, head_dim=4
    query = torch.randn(1, 1, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    output, weights = scaled_dot_product_attention(query, key, value)

    print("\nQuery shape:", tuple(query.shape))
    print("Key shape:", tuple(key.shape))
    print("Value shape:", tuple(value.shape))

    print("\nAttention weights:")
    print(weights[0, 0])

    print("\nAttention output:")
    print(output[0, 0])

    print("\nOutput shape:", tuple(output.shape))

    # Each query's attention probabilities should sum to 1
    row_sums = weights.sum(dim=-1)
    print("\nAttention row sums:")
    print(row_sums)

    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-6
    )

    print("Verification: Each attention row sums to 1. PASSED")


# ---------------------------------------------------------
# STEP 6: DEMONSTRATE MULTI-HEAD ATTENTION
# ---------------------------------------------------------

def demonstrate_multi_head_attention():
    print("\n" + "=" * 72)
    print("MULTI-HEAD ATTENTION")
    print("=" * 72)

    # Create sample input
    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM
    )

    print("\nConfiguration:")
    print("Batch size:", BATCH_SIZE)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Number of heads:", NUM_HEADS)
    print("Dimension per head:", EMBEDDING_DIM // NUM_HEADS)

    print("\nInput tensor shape:", tuple(x.shape))

    # Initialize our custom attention module
    attention = MultiHeadAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS
    )

    # Forward pass
    output, weights = attention(x)

    print("\nAttention weights shape:", tuple(weights.shape))
    print("Final output shape:", tuple(output.shape))

    print("\nAttention weights (Batch 0, Head 0):")
    print(weights[0, 0])

    print("\nFinal output tensor:")
    print(output)

    # Verify expected output dimensions
    expected_output_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM
    )

    assert output.shape == expected_output_shape

    # Verify attention probabilities sum to 1
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6
    )

    print("\nVerification 1: Output shape is correct. PASSED")
    print("Verification 2: Attention rows sum to 1. PASSED")

    # Check that gradients can flow through the implementation
    loss = output.mean()
    loss.backward()

    gradients_exist = all(
        parameter.grad is not None
        for parameter in attention.parameters()
    )

    assert gradients_exist

    print("Verification 3: Backpropagation works. PASSED")

    save_attention_weights(weights)


# ---------------------------------------------------------
# STEP 7: MAIN FUNCTION
# ---------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("PYTORCH ATTENTION IMPLEMENTATION")
    print("=" * 72)

    demonstrate_scaled_dot_product()
    demonstrate_multi_head_attention()

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()
