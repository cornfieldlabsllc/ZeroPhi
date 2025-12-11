from ZeroPhiMemory import ZeroPhiMemory
from SpectralComputation import SpectralComputation
import math

# Re-define the helper function for the query input
def create_initial_fuzzy_P(center_v: int, width: float, v_max: int) -> dict:
    P = {}
    total_P = 0.0
    for v in range(v_max + 1):
        exponent = -((v - center_v) ** 2) / (2 * width ** 2)
        P[v] = math.exp(exponent)
        total_P += P[v]
    P_normalized = {v: P[v] / total_P for v in P}
    return P_normalized

def run_interoperability_demo():
    print("--- Spectral Computer Interoperability Demo ---")

    # 1. Initialize and Setup the Memory System (Zero Phi, Φ=0)
    memory = ZeroPhiMemory()
    # Store 'Zero' at a precise position (v=10)
    # The output is P_zero_infinity, a highly peaked distribution.
    memory.store_memory("Zero_Concept", 10) 

    # 2. Initialize the Computation Engine (Flux-Driven, Φ ≠ 0)
    engine = SpectralComputation()
    
    # Temporarily set higher steps for better convergence in the demo
    engine.MAX_STEPS = 1000
    print(f"\nSetting Computation Max Steps to {engine.MAX_STEPS} for better resolution.")

    # 3. Define a Query Input (P_query)
    # This query is fuzzy, centered at v=20, partially conflicting with 'Zero' (v=10).
    query_center = 20
    query_uncertainty = 15.0 # Broad uncertainty
    P_query = create_initial_fuzzy_P(
        center_v=query_center, 
        width=query_uncertainty, 
        v_max=engine.V_MAX
    )
    
    print(f"\n[QUERY SETUP] Query P_0 is fuzzy, centered at v={query_center}.")

    # 4. Construct the Computation Input
    # The computation is run on a mixture of the query and the memory state.
    # We mix 50% fuzzy query (P_query) and 50% stable memory (P_zero_infinity).
    P_zero_infinity = memory.sheets["Zero_Concept"]
    
    P_input = {}
    for v in range(engine.V_MAX + 1):
        P_input[v] = (0.5 * P_query[v]) + (0.5 * P_zero_infinity[v])
        
    # Re-normalize the mixed input
    Z = sum(P_input.values())
    P_input_normalized = {v: P_input[v] / Z for v in P_input}
    
    print("           Input P_mix: 50% Fuzzy Query (v=20) + 50% Stable Memory (v=10).")

    # 5. Run the Computation on the Mixed Input
    final_solution_P = engine.compute_solution(P_input_normalized)
    
    # 6. Analyze the Result
    solution_v = max(final_solution_P, key=final_solution_P.get)
    peak_P = final_solution_P[solution_v]
    
    print(f"\n--- Final Resolution ---")
    print(f"Memory Peak (Reference): v=10")
    print(f"Query Peak (Fuzzy): v=20")
    print(f"**Fixed Point Solution (v_final): {solution_v}**")
    print(f"Solution Stability (Peak P): {peak_P:.6f}")

if __name__ == "__main__":
    # NOTE: You MUST ensure ZeroPhiMemory.py and SpectralComputation.py are saved 
    # and accessible for this script to run successfully.
    run_interoperability_demo()
