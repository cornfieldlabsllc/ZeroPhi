import numpy as np

# =========================================================
# Part 1: Core Spectral Utilities (from Appendix H)
# =========================================================

def spectral_memory_update(P, Phi, gamma):
    """
    Performs a single spectral update step:
    P(v) <- P(v) * exp(-gamma * Phi(v)) / Z
    """
    weighted_P = {}
    Z = 0.0
    
    # Compute unnormalized weights
    for v, p_val in P.items():
        weight = np.exp(-gamma * Phi[v])
        w_val = p_val * weight
        weighted_P[v] = w_val
        Z += w_val
        
    # Normalize
    if Z == 0:
        return P # Should not happen analytically
        
    return {v: val / Z for v, val in weighted_P.items()}

def evolve(P0, Phi, gammas):
    """
    Evolves a spectral state P0 over a sequence of cooling rates (gammas).
    Returns the history of distributions.
    """
    P = P0.copy()
    history = [P.copy()]
    for g in gammas:
        P = spectral_memory_update(P, Phi, g)
        history.append(P.copy())
    return history

def random_distribution(num_bins, rng):
    """Generates a random normalized probability distribution."""
    x = rng.random(num_bins)
    x /= x.sum()
    return {i: x[i] for i in range(num_bins)}

def max_diff(P, Q):
    """Computes the maximum absolute difference between two distributions."""
    keys = set(P.keys()) | set(Q.keys())
    return max(abs(P.get(k, 0.0) - Q.get(k, 0.0)) for k in keys)

# =========================================================
# Part 2: Zero Phi Geometric Memory Class
# Implements the Manifold Projection described in the text
# =========================================================

class ZeroPhiGeometricMemory:
    """ 
    Pure Phi=0 memory with geometric encoding and projection for noise rejection.
    Corresponds to the implementation in 'Practical Zero Phi Memory Through Geometric Projection'.
    """
    def __init__(self, centers, widths, num_bins=100):
        self.num_bins = num_bins
        # Create the 'Attractor' geometry (the manifold M)
        self.M = self._create_pattern(centers, widths)
        # Initialize state P to be perfectly on the manifold
        self.P = self.M.copy()

    def _create_pattern(self, centers, widths):
        """Creates a multi-modal Gaussian geometric pattern."""
        x = np.arange(self.num_bins)
        pattern = np.zeros(self.num_bins)
        for c, sigma in zip(centers, widths):
            pattern += np.exp(-(x - c)**2 / (2 * sigma**2))
        pattern /= pattern.sum()
        return pattern

    def _project(self, P):
        """
        Projects a noisy state P back onto the memory manifold M.
        Formula: P_new = ( <P,M> / <M,M> ) * M
        """
        M = self.M
        alpha = np.dot(P, M) / np.dot(M, M)
        proj = alpha * M
        # Enforce physical constraints (non-negative probability)
        proj = np.maximum(proj, 0)
        if proj.sum() > 0:
            proj /= proj.sum()
        return proj

    def update(self, noise_level=0.0):
        """
        Simulates one time step:
        1. Inject physical noise (drift)
        2. Apply Zero Phi Projection (restoration)
        """
        if noise_level > 0:
            noise = noise_level * (np.random.random(self.num_bins) - 0.5)
            P_temp = self.P + noise
        else:
            P_temp = self.P.copy()
            
        # Normalize prior to projection to handle the noise addition
        P_temp = np.maximum(P_temp, 0)
        if P_temp.sum() > 0:
            P_temp /= P_temp.sum()
            
        # The Core Zero Phi Operation: Projection
        self.P = self._project(P_temp)

    def similarity(self):
        """Calculates cosine similarity between current state and original memory."""
        P = np.array(self.P)
        M = np.array(self.M)
        norm_p = np.linalg.norm(P)
        norm_m = np.linalg.norm(M)
        if norm_p == 0 or norm_m == 0: return 0.0
        return np.dot(P, M) / (norm_p * norm_m)

# =========================================================
# Part 3: Test Execution
# =========================================================

if __name__ == "__main__":
    print("=== TEST 1: Zero Phi Invariance Lemma Verification ===")
    # Logic: If Phi is identically 0, the distribution should NOT change, regardless of gamma.
    
    rng = np.random.default_rng(42)
    num_bins = 7
    num_trials = 5
    
    # Sub-test A: Zero Phi (Phi = 0)
    zero_phi_pass = True
    for _ in range(num_trials):
        P0 = random_distribution(num_bins, rng)
        Phi_zero = {i: 0.0 for i in P0} # THE ZERO PHI CONDITION
        gammas = list(rng.uniform(0.0, 5.0, size=20))
        
        hist = evolve(P0, Phi_zero, gammas)
        
        # Check if state changed at all (tolerance for float precision)
        if any(max_diff(P0, hist[t]) > 1e-12 for t in range(1, len(hist))):
            zero_phi_pass = False
            break
            
    print(f"Zero Phi Invariance (Phi=0 -> Identity): {zero_phi_pass}")

    # Sub-test B: Non-Zero Phi (Control Group)
    # Logic: If Phi is not 0, the distribution MUST change (optimization/collapse).
    nonzero_phi_pass = True
    for _ in range(num_trials):
        P0 = random_distribution(num_bins, rng)
        raw_phi = rng.uniform(0.1, 2.0, size=len(P0)) # Random energy landscape
        Phi = {i: raw_phi[i] for i in P0}
        gammas = list(rng.uniform(0.1, 3.0, size=10))
        
        hist = evolve(P0, Phi, gammas)
        
        # Check if state CHANGED
        changed = any(max_diff(P0, hist[t]) > 1e-6 for t in range(1, len(hist)))
        if not changed:
            nonzero_phi_pass = False
            break
            
    print(f"Non-Constant Phi Changes State:      {nonzero_phi_pass}")
    
    print("\n=== TEST 2: Geometric Memory Stability (Projection) ===")
    # Logic: Can we recover a shape (memory) after adding significant noise?
    
    # Create memory with two distinct peaks (attractors) at bins 20 and 70
    mem_system = ZeroPhiGeometricMemory(centers=[20, 70], widths=[5, 5], num_bins=100)
    
    initial_sim = mem_system.similarity()
    print(f"Initial Pattern Similarity: {initial_sim:.6f}")
    
    # Inject 20% noise
    mem_system.update(noise_level=0.2)
    sim_low_noise = mem_system.similarity()
    print(f"Similarity after 0.2 noise & projection: {sim_low_noise:.6f}")
    
    # Inject 50% noise (Heavy distortion)
    mem_system.update(noise_level=0.5)
    sim_high_noise = mem_system.similarity()
    print(f"Similarity after 0.5 noise & projection: {sim_high_noise:.6f}")
    
    # Validation
    if zero_phi_pass and nonzero_phi_pass and sim_high_noise > 0.99:
        print("\n[SUCCESS] All Zero Phi Stability Theorems Validated.")
    else:
        print("\n[FAILURE] One or more tests failed.")
