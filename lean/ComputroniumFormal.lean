/-- ComputroniumFormal: Formal verification of Computronium energy dynamics -/

import ComputroniumFormal.EnergyDynamics

-- Re-export the main theorems
#check energy_decreases
#check settle_converges
#check control_lyapunov_decreases
#check contrastive_gradient_local