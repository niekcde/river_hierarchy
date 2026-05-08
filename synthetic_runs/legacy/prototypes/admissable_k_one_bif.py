
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_k(L: float, W: float, kb: int = 20, S: float = 1e-3, n: float = 0.35) -> float:
    """
    Same as your original function.
    """
    kb = 20
    S = 1e-3
    n = 0.35
    return (3/5)*n*((L / (S**0.5))*((kb**(2/3))/(W**(2/3))))


def compute_k_array(edges):
    """
    edges = [(L1, W1), (L2, W2), ...]
    Returns:
        k_values = np.array([...])
        k_sum    = sum of all k-values
    """
    k_vals = np.array([compute_k(L, W) for (L, W) in edges])
    return k_vals, k_vals.sum()


def is_admissible(edges, x=0.3):
    """
    True if k-ratio <= allowed ratio.
    """
    k_vals, _ = compute_k_array(edges)
    kmin = k_vals.min()
    kmax = k_vals.max()

    k_ratio = kmax / kmin
    allowed_ratio = (1 - x) / x

    return k_ratio <= allowed_ratio



def generate_width_splits(W, min_value, min_step, mode="computed", n_random=10):
    if mode == "equal":
        if W % 2 == 0:
            return [(W // 2, W // 2)]
        return []

    valid_WA = [WA for WA in range(min_value, W - min_value + 1, min_step)]

    if mode == "computed":
        return [(WA, W - WA) for WA in valid_WA]

    if mode == "random":
        import random
        return [(random.choice(valid_WA), W - random.choice(valid_WA)) for _ in range(n_random)]

    raise ValueError(f"Unknown mode: {mode}")


def enumerate_reach_combinations(
    L=100_000,
    W=100,
    min_ratio=5,
    domain_width=100,
    min_width_value=10,
    min_width_step=10,
    width_mode="computed",
    n_random_splits=10,
    x=0.3
):
    """
    Returns a list of dictionaries:
        {
            'x1': ...,
            'x2': ...,
            'x3': ...,
            'WA': ...,
            'WB': ...,
            'edges': [(L1,W1),(L2,W2),(L3,W3),(L4,W4)],
            'k_values': np.array([...]),
            'k_sum': float
        }
    """
    jump = min_ratio * domain_width     # 500 m
    n_steps = L // jump

    width_splits = generate_width_splits(
        W, min_value=min_width_value, min_step=min_width_step,
        mode=width_mode, n_random=n_random_splits
    )

    results = []

    for i1 in range(1, n_steps - 1):
        x1 = i1 * jump

        for i2 in range(i1 + 1, n_steps):
            x2 = i2 * jump

            for i3 in range(i2 + 1, n_steps + 1):
                x3 = i3 * jump
                if x3 > L:
                    continue

                # Reach lengths
                L1 = x1
                L2 = x2 - x1
                L3 = x3 - x2
                L4 = L - x3

                for WA, WB in width_splits:

                    edges = [
                        (L1, W),   # R1
                        (L2, WA),  # R2
                        (L3, WB),  # R3
                        (L4, W)    # R4
                    ]

                    # Check admissibility
                    if is_admissible(edges, x=x):
                        k_vals, k_sum = compute_k_array(edges)

                        results.append({
                            "x1": x1,
                            "x2": x2,
                            "x3": x3,
                            "WA": WA,
                            "WB": WB,
                            "edges": edges,
                            "k_values": k_vals,
                            "k_sum": float(k_sum),
                        })

    return results


combos = enumerate_reach_combinations(
    L=10000,
    W=100,
    min_ratio=5,
    domain_width=100,
    min_width_value=10,
    min_width_step=10,
    width_mode="computed",   # or "equal" or "random"
    n_random_splits=20,      # only used if width_mode="random"
    x=0.3
)
def results_to_dataframe(results):
    """
    Convert the list of results dictionaries into a pandas DataFrame.
    Includes:
        - x1, x2, x3
        - WA, WB
        - k_sum
        - edges stored as a tuple list (network structure)
        - k_values stored as array
    """
    df = pd.DataFrame(results)

    # Optional: turn edges into readable strings
    df["edges_str"] = df["edges"].apply(lambda e: str(e))

    # Optional: convert numpy arrays to Python lists for easier inspection
    df["k_values_list"] = df["k_values"].apply(lambda arr: arr.tolist())

    return df


df = results_to_dataframe(combos)
print("Number of admissible combinations:", df.shape[0], f"Unique k-values: {df['k_sum'].nunique()}")

def plot_k_sum_distribution(df, bins=100):
    """
    Plot histogram of k_sum values.
    Works in VS Code with plt.show().
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["k_sum"], bins=bins, edgecolor="black")
    plt.xlabel("k_sum")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total K (k_sum) for Admissible Networks")
    plt.grid(alpha=0.3)
    plt.show()

plot_k_sum_distribution(df)


################################
# Different lengths for the two parallel edge allowed!!!!!!!!!
################################