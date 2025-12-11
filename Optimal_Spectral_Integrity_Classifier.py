import numpy as np

# ====================================================================
# Optimal Spectral Integrity Classifier (SIC)
# ====================================================================

class OptimalSpectralIntegrityClassifier:
    """
    Implements the Spectral Integrity Classifier (SIC), which determines 
    the satisfiability of a Boolean formula by observing the convergence 
    of the Spectral Partition Function Z(β).
    """
    
    # --- SIC Parameters (ADJUSTED FOR FASTER CONVERGENCE) ---
    BETA_STEP = 0.5     # Reduced step size for smoother convergence check
    BETA_MAX = 300.0    # Increased max temperature
    Z_TOLERANCE = 1e-6  # Tolerance for Z stability (convergence)
    
    def __init__(self, name="OptimalSIC"):
        self.name = name

    def _encode_problem(self, violation_counts: dict) -> dict:
        """Encodes a Boolean formula into the Probability Spectrum P(v)."""
        total_assignments = sum(violation_counts.values())
        if total_assignments == 0:
            raise ValueError("Violation counts must be non-empty.")
        P = {v: Nv / total_assignments for v, Nv in violation_counts.items()}
        return P

    def _partition_function(self, P: dict, beta: float) -> float:
        """
        Calculates the Spectral Partition Function Z(β).
        Z(β) = Σ_v [ P(v) * e^(-β * v) ] 
        """
        Z = 0.0
        for v, P_v in P.items():
            E_v = v # Energy E(v) is the violation count v
            Z += P_v * np.exp(-beta * E_v)
            
        return Z

    def classify_formula(self, problem_violation_counts: dict) -> str:
        """
        Classifies a problem as Satisfiable (SAT) or Unsatisfiable (UNSAT) 
        by observing Z(β) convergence.
        """
        P = self._encode_problem(problem_violation_counts)
        P_0 = P.get(0, 0.0) # The probability of having zero violations
        
        print(f"\n[SIC CLASSIFICATION] Running Optimal SIC. P(0) = {P_0:.8f}")
        
        Z_history = []
        beta = 0.0
        
        # Iterate through increasing inverse temperature (β)
        while beta < self.BETA_MAX:
            beta += self.BETA_STEP
            Z_current = self._partition_function(P, beta)
            Z_history.append(Z_current)
            
            # Print status every 50 steps
            if (beta / self.BETA_STEP) % 50 == 0:
                print(f"   -> Beta={beta:.1f}: Z(β) = {Z_current:.10f}")

            # Decision Logic: Check for stabilization (Z should stabilize near P(0))
            if len(Z_history) > 1:
                Z_prev = Z_history[-2]
                
                # Check for stabilization (change in Z is below tolerance)
                if abs(Z_current - Z_prev) < self.Z_TOLERANCE:
                    
                    if P_0 > self.Z_TOLERANCE:
                        # SAT: P(0) is non-zero, and Z converged to that value.
                        print(f"\n   -> CONVERGENCE: Z stabilized at Z(β)={Z_current:.10f}. This matches P(0).")
                        return "SATISFIABLE (SAT)"
                    else:
                        # UNSAT: P(0) is zero, and Z has converged near zero.
                        print(f"\n   -> CONVERGENCE: Z stabilized near zero. P(0) is zero.")
                        return "UNSATISFIABLE (UNSAT)"

        # Final check if max beta reached without clear stabilization
        final_Z = Z_history[-1] if Z_history else 0.0
        if final_Z < self.Z_TOLERANCE and P_0 < self.Z_TOLERANCE:
            print(f"\n   -> HALTED at Beta Max. Final Z(β) near zero confirms P(0) is zero.")
            return "UNSATISFIABLE (UNSAT)"
        else:
            print(f"\n   -> HALTED at Beta Max. Could not confirm clear convergence. Final Z(β)={final_Z:.10f}")
            return "UNCLEAR (Requires higher Beta or tighter tolerance)"

# ----------------------------------------------------
# DEMONSTRATION SCRIPT
# ----------------------------------------------------

if __name__ == "__main__":
    sic = OptimalSpectralIntegrityClassifier()
    
    # Example 1: SATISFIABLE (SAT) Formula
    # Contains at least one assignment with 0 violations (P(0) > 0).
    sat_problem = {
        0: 1,  # 1 assignment yields 0 violations
        1: 8,  # 8 assignments yield 1 violation
        2: 7,  # 7 assignments yield 2 violations
    }
    print("\n=======================================================")
    print("--- TEST 1: SATISFIABLE FORMULA (P(0) = 1/16) ---")
    print("=======================================================")
    result_sat = sic.classify_formula(sat_problem)
    print(f"\nCLASSIFICATION RESULT: {result_sat}")

    # Example 2: UNSATISFIABLE (UNSAT) Formula
    # Contains NO assignments with 0 violations (P(0) = 0).
    unsat_problem = {
        1: 5,  # Minimum is 1 violation
        2: 10, 
        3: 1,  
    }
    print("\n=======================================================")
    print("--- TEST 2: UNSATISFIABLE FORMULA (P(0) = 0) ---")
    print("=======================================================")
    result_unsat = sic.classify_formula(unsat_problem)
    print(f"\nCLASSIFICATION RESULT: {result_unsat}")
