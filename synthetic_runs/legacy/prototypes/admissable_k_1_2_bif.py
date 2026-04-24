import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
# ---- K utilities ----

def compute_k(L: float, W: float, kb: int = 20, S: float = 1e-3, n: float = 0.35) -> float:
    kb = 20
    S = 1e-3
    n = 0.35
    return (3/5)*n*((L / (S**0.5))*((kb**(2/3))/(W**(2/3))))

def compute_k_array(edges):
    """
    edges: list of (L, W) pairs
    returns: (k_values np.array, k_sum float)
    """
    k_vals = np.array([compute_k(L, W) for (L, W) in edges])
    return k_vals, float(k_vals.sum())

def is_admissible(edges, x=0.3):
    """
    True if Kmax/Kmin <= allowed_ratio
    """
    k_vals, _ = compute_k_array(edges)
    kmin = k_vals.min()
    kmax = k_vals.max()
    k_ratio = kmax / kmin
    allowed_ratio = (1 - x) / x
    return k_ratio <= allowed_ratio

def primary_width_splits(W_total, min_value, min_step, mode="computed", 
                         n_random=10, require_splittable_for_second=False):
    """
    First bifurcation: W_total -> WA + WB.
    If require_splittable_for_second=True, enforce WA >= 2*min_value.
    """
    valid_WA = []
    for WA in range(min_value, W_total - min_value + 1, min_step):
        WB = W_total - WA
        if WB < min_value:
            continue
        if require_splittable_for_second and WA < 2 * min_value:
            continue
        valid_WA.append(WA)

    if mode == "computed":
        return [(WA, W_total - WA) for WA in valid_WA]

    if mode == "equal":
        if W_total % 2 == 0:
            WA = W_total // 2
            WB = WA
            if (not require_splittable_for_second) or (WA >= 2*min_value):
                return [(WA, WB)]
        return []

    if mode == "random":
        if not valid_WA:
            return []
        return [(random.choice(valid_WA), W_total - random.choice(valid_WA))
                for _ in range(n_random)]

    raise ValueError(f"Unknown mode for primary splits: {mode}")

def secondary_width_splits(WA, min_value, min_step, mode="computed", n_random=10):
    """
    Second bifurcation on branch A: WA -> WA2 + WC.
    WA must be >= 2 * min_value or no splits returned.
    Returns list of (WC, WA2).
    """
    if WA < 2 * min_value:
        return []

    valid_WC = []
    for WC in range(min_value, WA - min_value + 1, min_step):
        WA2 = WA - WC
        if WA2 < min_value:
            continue
        valid_WC.append(WC)

    if mode == "computed":
        return [(WC, WA - WC) for WC in valid_WC]

    if mode == "equal":
        if WA % 2 == 0 and WA >= 2 * min_value:
            WC = WA // 2
            WA2 = WA - WC
            if WA2 >= min_value:
                return [(WC, WA2)]
        return []

    if mode == "random":
        if not valid_WC:
            return []
        return [(random.choice(valid_WC), WA - random.choice(valid_WC))
                for _ in range(n_random)]

    raise ValueError(f"Unknown mode for secondary splits: {mode}")

def enumerate_networks(
    bifurcations=1,
    L=100_000,
    W_total=100,
    min_ratio=5,
    domain_width=100,
    min_width_value=10,
    min_width_step=10,
    width_mode_primary="computed",
    width_mode_secondary="computed",
    n_random_primary=20,
    n_random_secondary=20,
    x=0.3
):
    """
    bifurcations = 1 or 2 (for now).
    Returns a list of dicts; each dict describes one admissible network.
    """
    jump = min_ratio * domain_width
    assert L % jump == 0, "L must be divisible by jump."
    n_steps = L // jump

    results = []
    n_tested = 0
    # ----- loop over main split/rejoin points -----
    for i_s in range(1, n_steps - 1):          # start of parallel region
        for i_e in range(i_s + 1, n_steps):    # end of parallel region
            L_up   = (i_s) * jump
            L_par  = (i_e - i_s) * jump
            L_down = (n_steps - i_e) * jump

            # primary width splits
            prim_splits = primary_width_splits(
                W_total,
                min_value=min_width_value,
                min_step=min_width_step,
                mode=width_mode_primary,
                n_random=n_random_primary,
                require_splittable_for_second=(bifurcations >= 2)
            )

            for WA, WB in prim_splits:
                if bifurcations == 1:
                    # --- simple 4-edge network ---
                    edges = [
                        (L_up,   W_total),  # E1_up
                        (L_par,  WA),       # E2A
                        (L_par,  WB),       # E2B
                        (L_down, W_total),  # E3_down
                    ]
                    n_tested += 1   # <-- ADD HERE
                    if not is_admissible(edges, x=x):
                        continue

                    k_vals, k_sum = compute_k_array(edges)
                    results.append({
                        "bifurcations": 1,
                        "i_s": i_s,
                        "i_e": i_e,
                        "x_s": i_s*jump,
                        "x_e": i_e*jump,
                        "WA": WA,
                        "WB": WB,
                        "edges": [
                            {"name": "E1_up",   "L": L_up,  "W": W_total},
                            {"name": "E2A",     "L": L_par, "W": WA},
                            {"name": "E2B",     "L": L_par, "W": WB},
                            {"name": "E3_down", "L": L_down, "W": W_total},
                        ],
                        "k_values": k_vals,
                        "k_sum": k_sum,
                    })

                elif bifurcations == 2:
                    # --- refine inside the parallel region ---
                    # indices within (i_s, i_e) for second bifurcation
                    for jA in range(i_s + 1, i_e):       # split on A
                        for jB in range(jA + 1, i_e):    # split on B, forward: jA < jB
                            L_A1 = (jA - i_s) * jump
                            L_A2 = (i_e - jA) * jump
                            L_B1 = (jB - i_s) * jump
                            L_B2 = (i_e - jB) * jump
                            L_C  = (jB - jA) * jump

                            # lengths are guaranteed >= jump by index ranges

                            # secondary width splits: WA -> WA2 + WC
                            sec_splits = secondary_width_splits(
                                WA,
                                min_value=min_width_value,
                                min_step=min_width_step,
                                mode=width_mode_secondary,
                                n_random=n_random_secondary
                            )
                            if not sec_splits:
                                continue

                            for WC, WA2 in sec_splits:
                                # widths:
                                W_A1 = WA           # upstream A
                                W_A2 = WA2          # downstream A
                                W_B1 = WB           # upstream B
                                W_B2 = WB + WC      # downstream B (B2 + crossover)
                                W_C  = WC

                                edges = [
                                    (L_up,  W_total),  # E1_up
                                    (L_A1, W_A1),      # E2A1
                                    (L_A2, W_A2),      # E2A2
                                    (L_B1, W_B1),      # E2B1
                                    (L_B2, W_B2),      # E2B2
                                    (L_C,  W_C),       # Crossover
                                    (L_down, W_total)  # E3_down
                                ]
                                n_tested += 1
                                if not is_admissible(edges, x=x):
                                    continue

                                k_vals, k_sum = compute_k_array(edges)
                                results.append({
                                    "bifurcations": 2,
                                    "i_s": i_s,
                                    "i_e": i_e,
                                    "jA": jA,
                                    "jB": jB,
                                    "x_s": i_s * jump,
                                    "x_e": i_e * jump,
                                    "x_A_split": jA * jump,
                                    "x_B_split": jB * jump,
                                    "WA": WA,
                                    "WB": WB,
                                    "WC": WC,
                                    "edges": [
                                        {"name": "E1_up",   "L": L_up,  "W": W_total},
                                        {"name": "E2A1",    "L": L_A1,  "W": W_A1},
                                        {"name": "E2A2",    "L": L_A2,  "W": W_A2},
                                        {"name": "E2B1",    "L": L_B1,  "W": W_B1},
                                        {"name": "E2B2",    "L": L_B2,  "W": W_B2},
                                        {"name": "E_cross", "L": L_C,   "W": W_C},
                                        {"name": "E3_down", "L": L_down, "W": W_total},
                                    ],
                                    "k_values": k_vals,
                                    "k_sum": k_sum,
                                })
                else:
                    raise NotImplementedError("Currently supports bifurcations = 1 or 2 only.")

    return results, n_tested



# ---------------------------------------------------------------
# 1. Convert results to DataFrame
# ---------------------------------------------------------------

def results_to_dataframe(results, n_tested):
    """
    Convert list of result dictionaries from enumerate_networks()
    into a pandas DataFrame.

    Adds:
        - n_tested column: total number of combinations evaluated
        - edges_str: stringified list of edges
        - k_values_list: python list representation of k-values
    """
    if len(results) == 0:
        df = pd.DataFrame(columns=[
            "bifurcations", "i_s", "i_e", "jA", "jB",
            "WA", "WB", "WC", "k_sum", "n_tested"
        ])
        return df

    df = pd.DataFrame(results)

    # stringify edges to preserve structure
    df["edges_str"] = df["edges"].apply(lambda e: str(e))

    # convert k arrays to lists
    df["k_values_list"] = df["k_values"].apply(lambda arr: arr.tolist())

    # total number of tested combos (same for all rows)
    df["n_tested"] = n_tested

    return df


# ---------------------------------------------------------------
# 2. Histogram: distribution of k_sum
# ---------------------------------------------------------------

def plot_k_sum_distribution(df, bins=50):
    """
    Plot histogram of k_sum distribution.
    Works correctly in VS Code (plt.show()).
    """
    if df.empty:
        print("DataFrame is empty, nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df["k_sum"], bins=bins, edgecolor="black", alpha=0.75)
    plt.xlabel("k_sum")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total K (k_sum)")
    plt.grid(alpha=0.3)
    plt.show()


# ---------------------------------------------------------------
# 3. Histogram: distribution of all individual k-values
# ---------------------------------------------------------------

def plot_all_k_values(df, bins=50):
    """
    Flatten all k-values and display histogram.
    """
    if df.empty:
        print("DataFrame empty, nothing to plot.")
        return

    all_k = []
    for k_list in df["k_values_list"]:
        all_k.extend(k_list)

    plt.figure(figsize=(10, 6))
    plt.hist(all_k, bins=bins, edgecolor="black", alpha=0.75)
    plt.xlabel("Individual k-values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Reach-Level K Values")
    plt.grid(alpha=0.3)
    plt.show()



res, n_tested = enumerate_networks(
    bifurcations=2,
    L=10000,
    W_total=100,
    min_ratio=5,
    domain_width=100,
    min_width_value=10,
    min_width_step=10,
    width_mode_primary="computed",
    width_mode_secondary="computed",
    n_random_primary=20,
    n_random_secondary=20,
    x=0.1)


print("Admissible:", len(res))
print("Total tested:", n_tested)
print(f"Acceptance rate: {np.round((len(res) / n_tested)*100,2)}%")

df = results_to_dataframe(res, n_tested)
plot_k_sum_distribution(df, bins = 100)
plot_all_k_values(df, bins = 200)