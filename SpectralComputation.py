import numpy as np
import math

class SpectralComputation:
    """
    Implements the Spectral Computation Engine.
    
    This process models computation as a physical process of relaxation 
    driven by non-zero Spectral Flux (Φ ≠ 0), compressing an input 
    distribution P(v) into a stable invariant solution Psi (Ψ).
    """
    
    # --- System Parameters ---
    V_MAX = 100         # Must match V_MAX from ZeroPhiMemory
    ALPHA = 0.01        # Flux Correction Factor / Compression Rate
    BETA = 0.1          # Non-linear term coefficient for Flux calculation
    MAX_STEPS = 500     # Maximum steps before halting computation
    TOLERANCE = 1e-9    # Stability Tolerance for convergence (e.g., Psi < 1e-9)
    
    def __init__(self, name="SpectralEngine"):
        self.name = name
        
    # ----------------------------------------------------
    # 1. CORE FUNCTION: Spectral Flux Transform (Φ)
    # Flux drives the system away from equilibrium (The Computation)
    # ----------------------------------------------------
    def _compute_flux(self, P):
        """
        Calculates the Spectral Flux Φ(v) from the current distribution P(v).
        
        This model uses the non-linear relationship: 
        Φ(v) = -β * (P(v_high) - P(v_low)) 
        (Simplified model of energy flow across the spectrum)
        """
        flux = {}
        for v in P:
            # Simple model: Flux is proportional to the local gradient or 
            # a non-linear term. We use the distribution value itself.
            flux[v] = -self.BETA * (P[v]**2 - 0.5 * P[v]) 
        return flux

    # ----------------------------------------------------
    # 2. CORE FUNCTION: Relaxation Update (The Compression)
    # ----------------------------------------------------
    def _relaxation_update(self, P, flux):
        """
        Applies the flux correction to P to drive the system toward compression.
        
        The paper suggests: "correction subtracts alpha times flux from the field."
        P_t+1(v) = P_t(v) - α * Φ(v)
        """
        P_new = {}
        for v in P:
            # Apply the correction to the probability field
            P_new[v] = P[v] - self.ALPHA * flux[v]
            
            # Ensure P remains a positive probability (prevent physical absurdity)
            P_new[v] = max(0.0, P_new[v])
            
        # Re-normalize P to maintain a valid probability distribution (sum=1)
        Z = sum(P_new.values())
        if Z == 0:
            return P_new # Return zero-filled if everything collapses
        
        P_normalized = {v: P_new[v] / Z for v in P_new}
        return P_normalized

    # ----------------------------------------------------
    # PUBLIC INTERFACE: Compute
    # ----------------------------------------------------
    
    def compute_solution(self, initial_P: dict):
        """
        Runs the Spectral Compression process until the solution Ψ is stable.
        
        initial_P: The input distribution P_0 representing the problem/query.
        Returns: The final stable distribution P_stable (the solution Ψ).
        """
        P = initial_P.copy()
        
        print(f"\n[COMPUTATION] Starting Spectral Compression.")
        
        for step in range(self.MAX_STEPS):
            # 1. Compute the Flux (Φ)
            flux = self._compute_flux(P)
            
            # The Spectral Observable Psi (Ψ) is a measure of system energy/flux.
            # Computation stops when the total flux is zero (Ψ converges).
            psi_value = sum(abs(f) for f in flux.values())
            
            if psi_value < self.TOLERANCE:
                print(f"   -> CONVERGED at Step {step}. Final Flux (Ψ) = {psi_value:.10f}")
                # The final P is the fixed point solution (the answer)
                break
                
            # 2. Apply Relaxation/Correction
            P_next = self._relaxation_update(P, flux)
            
            # Check for stagnation (e.g., probability change is too small)
            if all(abs(P[v] - P_next[v]) < 1e-10 for v in P):
                 print(f"   -> STAGNATED at Step {step}. No meaningful change.")
                 break
            
            P = P_next
            
            if (step + 1) % 50 == 0:
                print(f"   -> Step {step+1}: Current Flux (Ψ) = {psi_value:.10f}")

        else:
            print(f"   -> HALTED: Reached Max Steps ({self.MAX_STEPS}). Final Flux (Ψ) = {psi_value:.10f}")
            
        # The solution is the compressed, stable distribution
        final_solution = P
        
        # We find the peak of the solution P_stable to report the result
        solution_v = max(final_solution, key=final_solution.get)
        peak_P = final_solution[solution_v]
        
        print(f"   -> Solution Found at v={solution_v} (Peak P={peak_P:.4f})")
        
        return final_solution
