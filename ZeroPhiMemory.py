import numpy as np
import math

class ZeroPhiMemory:
    """
    Implements the Zero Phi Memory Architecture (Zero-Flux Geometric Memory).
    
    Memory is stored as geometric invariance (a stable probability distribution P) 
    in a field, enforced by the Zero Phi (Φ=0) condition, ensuring zero global 
    spectral flux and non-dissipative persistence.
    """
    
    # --- System Parameters (Based on Spectral Computer Theory) ---
    V_MAX = 100         # Maximum violation count (v) / size of the probability space
    GAMMA = 0.5         # Geometric contraction rate (cooling/convergence speed)
    LAMBDA = 5.0        # Attractor depth (strength of memory encoding)
    SIGMA = 2.0         # Attractor width (spatial resolution of memory item)
    ITERATIONS = 50     # Number of steps to stabilize a new memory item
    
    def __init__(self, name="ZeroPhiSystem"):
        self.name = name
        # Memory storage: a dictionary where keys are item IDs and values are 
        # the final stable probability distributions (Spectral Sheets P_k).
        self.sheets = {} 
        self.memory_count = 0

    # ----------------------------------------------------
    # 1. CORE FUNCTION: Geometric Attractor (Encoding)
    # ----------------------------------------------------
    def _attractor(self, v, vk):
        """
        Defines the geometric attractor field A(v) for a memory item at position vk.
        This field creates the stable geometric basin (fixed point) for the memory.
        A(v) = -λ * exp(-|v - vk| / σ)
        """
        return -self.LAMBDA * np.exp(-np.abs(v - vk) / self.SIGMA)

    # ----------------------------------------------------
    # 2. CORE FUNCTION: Zero Phi Update (Stabilization)
    # ----------------------------------------------------
    def _spectral_sheet_update(self, P, A_map):
        """
        Applies the geometric update rule to a Spectral Sheet P under the 
        zero-flux (Φ=0) constraint.
        
        P: The current probability distribution {v: P(v)}.
        A_map: The pre-computed Attractor map {v: A(v)} for this memory item.
        """
        # General Zero Phi Update: P_t+1(v) = [P_t(v) * e^(-γ * A(v))] / Z_t
        
        # 1. Compute the normalization constant Z (Partition Function)
        Z = sum(P[v] * math.exp(-self.GAMMA * A_map[v]) for v in P)
        
        # 2. Apply the geometric update and re-normalize
        P_new = {}
        for v in P:
            P_new[v] = (P[v] * math.exp(-self.GAMMA * A_map[v])) / Z
            
        return P_new

    # ----------------------------------------------------
    # 3. ENFORCEMENT: The Zero Phi Condition (Conceptual)
    # ----------------------------------------------------
    def _enforce_zero_phi(self):
        """
        Conceptual enforcement: Sets the Global Spectral Flux Φ(v) = 0 for all v. 
        This is what physically locks the memory state (P_∞) into persistence.
        """
        return {v: 0.0 for v in range(self.V_MAX + 1)}

    # ----------------------------------------------------
    # PUBLIC INTERFACE: Store and Read
    # ----------------------------------------------------

    def store_memory(self, memory_id: str, geometric_position_vk: int):
        """
        Stores a new memory item by stabilizing a Spectral Sheet P at position vk.
        """
        if not (0 <= geometric_position_vk <= self.V_MAX):
            raise ValueError(f"vk must be between 0 and {self.V_MAX}")

        print(f"\n[ENCODING] Storing '{memory_id}' at vk={geometric_position_vk}")
        
        # 1. Initialize the Sheet (P_0) uniformly
        P = {v: 1.0 / (self.V_MAX + 1) for v in range(self.V_MAX + 1)}
        
        # 2. Compute the target Attractor Field A(v)
        A_map = {v: self._attractor(v, geometric_position_vk) for v in range(self.V_MAX + 1)}
        
        # 3. Stabilize the Sheet (Geometric Relaxation)
        print(f"   -> Stabilizing over {self.ITERATIONS} iterations...")
        for i in range(self.ITERATIONS):
            P = self._spectral_sheet_update(P, A_map)
            
        # 4. Enforce Zero Phi (Locking the State P_∞)
        self._enforce_zero_phi()
        
        # Store the stabilized distribution P_∞
        self.sheets[memory_id] = P
        self.memory_count += 1
        print(f"   -> Successfully stored and locked. P_∞ stabilized.")
        
        return True

    def read_memory(self, memory_id: str):
        """
        Retrieves the stabilized Spectral Sheet (P_∞) for a memory item.
        """
        if memory_id in self.sheets:
            print(f"\n[RETRIEVAL] Retrieving stabilized distribution for '{memory_id}'.")
            
            P_stable = self.sheets[memory_id]
            
            # Find the v with the highest probability (the measured memory position)
            retrieved_vk = max(P_stable, key=P_stable.get)
            max_p = P_stable[retrieved_vk]
            
            print(f"   -> Measured Position (v_k): {retrieved_vk}")
            print(f"   -> Peak Probability P(v_k): {max_p:.4f}")
            
            return P_stable
        else:
            print(f"Error: Memory item '{memory_id}' not found.")
            return None
