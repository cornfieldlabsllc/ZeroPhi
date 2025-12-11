import numpy as np
import math
from SpectralComputation import SpectralComputation

# Helper function to create a "fuzzy" input distribution (P_0)
def create_initial_fuzzy_P(center_v: int, width: float, v_max: int) -> dict:
    """
    Generates an initial Gaussian-like probability distribution (P_0) 
    representing a fuzzy, uncertain input problem centered around 'center_v'.
    """
    P = {}
    total_P = 0.0
    
    # Use a Gaussian function to define the initial shape
    # P(v) ~ exp(-(v - center_v)^2 / (2*width^2))
    for v in range(v_max + 1):
        exponent = -((v - center_v) ** 2) / (2 * width ** 2)
        P[v] = math.exp(exponent)
        total_P += P[v]

    # Normalize the distribution (P sums to 1)
    P_normalized = {v: P[v] / total_P for v in P}
    return P_normalized


if __name__ == "__main__":
    
    # Ensure SpectralComputation.py is available in the same directory.
    
    # 1. Initialize the Spectral Computation Engine
    engine = SpectralComputation()
    
    # 2. Define the Initial Problem (P_0)
    # The 'problem' is a fuzzy distribution centered at v=50, 
    # representing an uncertain input that needs to be resolved.
    
    # Parameters for the fuzzy input:
    center_of_query = 50 
    query_uncertainty = 20.0 # Large width = high uncertainty
    
    initial_P = create_initial_fuzzy_P(
        center_v=center_of_query, 
        width=query_uncertainty, 
        v_max=engine.V_MAX
    )
    
    print(f"Initial Problem P_0: Centered at v={center_of_query} with wide uncertainty.")
    
    # 3. Run the Spectral Compression Process
    # This simulates the system driving itself out of disequilibrium (Φ ≠ 0)
    # until it hits the fixed point solution (Ψ).
    
    final_solution_P = engine.compute_solution(initial_P)
    
    # 4. Analyze the Result
    solution_v = max(final_solution_P, key=final_solution_P.get)
    peak_P = final_solution_P[solution_v]
    
    print(f"\n--- Computation Summary ---")
    print(f"Initial Query Center: v={center_of_query}")
    print(f"Fixed Point Solution (v_final): {solution_v}")
    print(f"Solution Stability (Peak P): {peak_P:.6f}")
