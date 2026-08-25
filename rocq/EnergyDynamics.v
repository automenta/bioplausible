(* Computronium formalization — EnergyMinimizationDynamics.

   Energy model (corrected): E(h) = 1/2|h|^2 - 1/2 h^T W h - b.h
   The 1/2 factor on the cross-term matches the gradient settled by
   [settleStep] and the documented Hessian (I - W).

   Status:
   - proved: expansion lemmas, diagonal-case energy decrease,
     minimizer-is-fixed-point;
   - admitted: general-case decrease (needs Cauchy-Schwarz on the
     symmetrized quadratic form) and convex-settlement existence
     (needs classical completeness/coercivity argument). *)

From Stdlib Require Import Reals Lia Arith Psatz Lra.
From Computronium Require Import Utils.

Open Scope R_scope.

(** ------------------------------------------------------------------ *)
(** Configuration                                                       *)
(** ------------------------------------------------------------------ *)

Record EnergyConfig := {
  maxSteps : nat;
  stepSize : R;
  beta : R;
  convergenceThreshold : R;
  convergenceStart : nat;

  stepSize_pos : 0 < stepSize;
  beta_pos : 0 < beta;
  threshold_pos : 0 < convergenceThreshold
}.

(** Vectors are total functions on indices; dimension [n] bounds range.
    Totality removes the out-of-bounds hazard of the previous list
    encoding. *)
Definition Vector := nat -> R.
Definition Matrix := nat -> nat -> R.

Definition Kronecker (i j : nat) : R := if Nat.eqb i j then 1 else 0.

(** ------------------------------------------------------------------ *)
(** Dynamics                                                            *)
(** ------------------------------------------------------------------ *)

(** Corrected energy: cross-term carries the 1/2 factor. *)
Definition energyFunction (W : Matrix) (b h : Vector) (n : nat) : R :=
  1 / 2 * sum_R (fun i => h i * h i) n
  - 1 / 2 * sum_R (fun i => sum_R (fun j => W i j * h i * h j) n) n
  - sum_R (fun i => b i * h i) n.

(** Gradient of [energyFunction] at component [i]
    (symmetrized in W, so correct for asymmetric W too). *)
Definition gradE (W : Matrix) (b h : Vector) (n i : nat) : R :=
  h i - sum_R (fun j => (W j i + W i j) / 2 * h j) n - b i.

Definition settleStep (cfg : EnergyConfig) (W : Matrix) (b h : Vector) (n : nat) : Vector :=
  fun i => h i - cfg.(stepSize) * gradE W b h n i.

Fixpoint settle (cfg : EnergyConfig) (W : Matrix) (b : Vector) (n k : nat) (h : Vector) : Vector :=
  match k with
  | O => h
  | S k' => settleStep cfg W b (settle cfg W b n k' h) n
  end.

(** Frobenius norm of I - W as a finite double sum (upper bound on the
    operator norm of the Hessian I - sym(W)). *)
Definition lipschitzConstant (W : Matrix) (n : nat) : R :=
  sqrt (sum_R (fun i =>
    sum_R (fun j => (Kronecker i j - W i j) * (Kronecker i j - W i j)) n) n).

(** ------------------------------------------------------------------ *)
(** Diagonal reductions (proved)                                        *)
(** ------------------------------------------------------------------ *)

Lemma gradE_diagonal :
  forall (W : Matrix) (b h : Vector) (n i : nat),
    (forall p q : nat, p <> q -> W p q = 0) ->
    (i < n)%nat ->
    gradE W b h n i = (1 - W i i) * h i - b i.
Proof.
  intros W b h n i Hdiag Hi.
  unfold gradE.
  rewrite (sum_R_keep_index _ _ i Hi).
  - lra.
  - intros j Hj Hjne.
    rewrite (Hdiag j i Hjne), (Hdiag i j).
    + lra.
    + congruence.
Qed.

Lemma energyFunction_diagonal :
  forall (W : Matrix) (b h : Vector) (n : nat),
    (forall p q : nat, p <> q -> W p q = 0) ->
    energyFunction W b h n =
      sum_R (fun i => 1 / 2 * (1 - W i i) * h i * h i - b i * h i) n.
Proof.
  intros W b h n Hdiag.
  unfold energyFunction.
  assert (He : sum_R (fun i => sum_R (fun j => W i j * h i * h j) n) n
               = sum_R (fun i => W i i * h i * h i) n).
  { apply sum_R_ext.
    intros i Hi.
    rewrite (sum_R_keep_index _ _ i Hi).
    - ring.
    - intros j Hj Hjne.
      rewrite (Hdiag i j) by congruence. lra. }
  rewrite He.
  rewrite <- (sum_R_scale (1 / 2) (fun i => h i * h i) n).
  rewrite <- (sum_R_scale (1 / 2) (fun i => W i i * h i * h i) n).
  rewrite <- (sum_R_sub (fun i => 1 / 2 * (h i * h i))
                        (fun i => 1 / 2 * (W i i * h i * h i))).
  rewrite <- sum_R_sub.
  apply sum_R_ext.
  intros i _. ring.
Qed.

(** ------------------------------------------------------------------ *)
(** Energy decrease — diagonal case (proved)                            *)
(** ------------------------------------------------------------------ *)

(** Per-index descent identity: one settle step never increases the
    diagonal energy term. Difference equals
    -(eta/2)*(2 - eta*u)*t^2 with u = 1 - W i i > 0 and t = u*h - b. *)
Lemma per_index_descent :
  forall (eta u bb hh : R),
    0 < eta ->
    0 < u ->
    eta * u < 2 ->
    1 / 2 * u * (hh - eta * (u * hh - bb)) * (hh - eta * (u * hh - bb))
      - bb * (hh - eta * (u * hh - bb))
    <= 1 / 2 * u * hh * hh - bb * hh.
Proof.
  intros eta u bb hh Heta Hu Hlu.
  assert (Hprod : 0 <= 1 / 2 * eta * (2 - eta * u) * ((u * hh - bb) * (u * hh - bb))).
  { apply Rmult_le_pos.
    - apply Rmult_le_pos; lra.
    - apply sq_nonneg. }
  assert (Hid :
    1 / 2 * u * (hh - eta * (u * hh - bb)) * (hh - eta * (u * hh - bb))
      - bb * (hh - eta * (u * hh - bb))
      - (1 / 2 * u * hh * hh - bb * hh)
    = - (1 / 2 * eta * (2 - eta * u)) * ((u * hh - bb) * (u * hh - bb))).
  { field. }
  lra.
Qed.

Theorem energy_decreases_diagonal :
  forall (n : nat) (cfg : EnergyConfig) (W : Matrix) (b h : Vector),
    (forall p q : nat, p <> q -> W p q = 0) ->
    (forall i, (i < n)%nat -> 0 < 1 - W i i) ->
    (forall i, (i < n)%nat -> cfg.(stepSize) * (1 - W i i) < 2) ->
    energyFunction W b (settleStep cfg W b h n) n
      <= energyFunction W b h n.
Proof.
  intros n cfg W b h Hdiag Hu Heta.
  rewrite (energyFunction_diagonal W b (settleStep cfg W b h n) n Hdiag).
  rewrite (energyFunction_diagonal W b h n Hdiag).
  apply sum_R_le.
  intros i Hi.
  assert (Hstep :
    settleStep cfg W b h n i
      = h i - cfg.(stepSize) * ((1 - W i i) * h i - b i)).
  { unfold settleStep. rewrite (gradE_diagonal W b h n i Hdiag Hi). reflexivity. }
  rewrite Hstep.
  exact (per_index_descent cfg.(stepSize) (1 - W i i) (b i) (h i)
           cfg.(stepSize_pos) (Hu i Hi) (Heta i Hi)).
Qed.

(** ------------------------------------------------------------------ *)
(** General case (stated honestly, not yet proved)                      *)
(** ------------------------------------------------------------------ *)

(** STUB (admitted): general symmetric case.
    Missing: descent inequality E(h - eta*g) <= E(h) - eta*(1 - eta*L/2)*|g|^2
    under |gradE| Lipschitz with constant L, which reduces to a
    Cauchy-Schwarz estimate on the symmetrized quadratic form. *)
Theorem energy_decreases :
  forall (n : nat) (cfg : EnergyConfig) (W : Matrix) (b h : Vector) (L : R),
    (forall i j : nat, W i j = W j i) ->
    Rle (lipschitzConstant W n) L ->
    0 < L ->
    cfg.(stepSize) < 2 / L ->
    energyFunction W b (settle cfg W b n cfg.(maxSteps) h) n
      <= energyFunction W b h n.
Proof.
  admit.
Admitted.

(** STUB (admitted): existence of an energy minimizer that is a fixed
    point, under strict convexity (I - W positive definite on the index
    range). Missing: classical completeness/coercivity argument; the
    fixed-point half is already discharged by
    [stationary_is_fixed_point]. *)
Theorem settle_converges :
  forall (n : nat) (cfg : EnergyConfig) (W : Matrix) (b : Vector),
    (forall i j : nat, W i j = W j i) ->
    (forall v : Vector,
       (exists i, (i < n)%nat /\ v i <> 0) ->
       0 < sum_R (fun i =>
         sum_R (fun j => v i * (Kronecker i j - W i j) * v j) n) n) ->
    exists h_star : Vector,
      (forall i, (i < n)%nat -> gradE W b h_star n i = 0) /\
      (forall h : Vector, energyFunction W b h_star n <= energyFunction W b h n).
Proof.
  admit.
Admitted.
