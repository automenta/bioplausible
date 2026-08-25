(* Computronium formalization — shared utilities.
   Status: fully proved; nothing admitted in this file. *)

From Stdlib Require Import Reals Lia Lra.

Open Scope R_scope.

(** Finite index sum: [sum_R f k] sums [f 0 + f 1 + ... + f (k-1)]. *)
Fixpoint sum_R (f : nat -> R) (k : nat) : R :=
  match k with
  | O => 0
  | S k' => f k' + sum_R f k'
  end.

(** Pointwise equality on the index range gives equal sums. *)
Lemma sum_R_ext :
  forall (f g : nat -> R) (k : nat),
    (forall i, (i < k)%nat -> f i = g i) ->
    sum_R f k = sum_R g k.
Proof.
  intros f g k.
  induction k as [| k IH]; intros H; simpl.
  - reflexivity.
  - rewrite H by lia. rewrite IH by (intros; apply H; lia). reflexivity.
Qed.

(** Pointwise inequality on the index range gives inequality of sums. *)
Lemma sum_R_le :
  forall (f g : nat -> R) (k : nat),
    (forall i, (i < k)%nat -> f i <= g i) ->
    sum_R f k <= sum_R g k.
Proof.
  intros f g k.
  induction k as [| k IH]; intros H; simpl.
  - apply Rle_refl.
  - apply Rplus_le_compat.
    + apply H. lia.
    + apply IH. intros i Hi. apply H. lia.
Qed.

(** Sums distribute over pointwise addition. *)
Lemma sum_R_plus :
  forall (f g : nat -> R) (k : nat),
    sum_R (fun i => f i + g i) k = sum_R f k + sum_R g k.
Proof.
  intros f g k.
  induction k as [| k IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

(** Sums distribute over pointwise subtraction. *)
Lemma sum_R_sub :
  forall (f g : nat -> R) (k : nat),
    sum_R (fun i => f i - g i) k = sum_R f k - sum_R g k.
Proof.
  intros f g k.
  induction k as [| k IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

(** Constants factor out of sums. *)
Lemma sum_R_scale :
  forall (c : R) (f : nat -> R) (k : nat),
    sum_R (fun i => c * f i) k = c * sum_R f k.
Proof.
  intros c f k.
  induction k as [| k IH]; simpl.
  - ring.
  - rewrite IH. ring.
Qed.

(** A sum whose terms all vanish is zero. *)
Lemma sum_R_all_zero :
  forall (f : nat -> R) (k : nat),
    (forall j, (j < k)%nat -> f j = 0) ->
    sum_R f k = 0.
Proof.
  intros f k.
  induction k as [| k IH]; intros H; simpl.
  - ring.
  - rewrite (H k) by lia.
    rewrite IH by (intros j Hj; apply H; lia).
    ring.
Qed.

(** A diagonal term survives a sum whose other terms vanish. *)
Lemma sum_R_keep_index :
  forall (f : nat -> R) (k i : nat),
    (i < k)%nat ->
    (forall j, (j < k)%nat -> j <> i -> f j = 0) ->
    sum_R f k = f i.
Proof.
  intros f k.
  induction k as [| k IH]; intros i Hi Hz; [exfalso; lia|].
  simpl.
  destruct (Nat.eq_dec k i) as [Heq|Hne].
  - subst k.
    rewrite sum_R_all_zero by (intros j Hj; apply Hz; lia).
    ring.
  - rewrite (Hz k) by lia.
    assert (Hs : sum_R f k = f i).
    { apply IH; [lia | intros j Hj Hjne; apply Hz; lia]. }
    rewrite Hs. ring.
Qed.

(** Squares are non-negative. *)
Lemma sq_nonneg : forall x : R, 0 <= x * x.
Proof.
  intro x.
  destruct (Rle_lt_dec 0 x) as [Hn | Hp].
  - apply Rmult_le_pos; lra.
  - replace (x * x) with ((- x) * (- x)) by ring.
    assert (Hle : 0 <= - x) by lra.
    apply Rmult_le_pos; lra.
Qed.
