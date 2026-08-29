# Formal axioms (DASHI kernel)

These definitions anchor code semantics. Functions should map to the labels below.

## Carrier (G)
- Directed weighted graph with nodes `V`, edges `E`, weights `W >= 0` (or residual weights).
- Optional channel index `c in C` if multiple signal types exist.

## Valuation field (s)
- Ternary field `s : V x C -> {-1, 0, +1}`.
- Interpretations: `+1` affirmed/excess, `-1` opposed/deficit, `0` neutral/underdetermined.

## DASHI kernel (K)
- Local deterministic ternary update defined by weights on neighborhoods `N(x)`:
  `K(s)(x,c) = sgn(sum_{y in N(x)} sum_{c'} w_{(x,c),(y,c')} * s(y,c'))` with ternary deadzone.
- Base properties: locality and determinism.
- Symmetry/gauge equivariance is a separate preservation obligation for the chosen carrier/action.
- Idempotence `K(K(s)) = K(s)`, involutive laws, contraction, and defect monotonicity are **not** definitional consequences of the formula. They may be reported empirically for a finite run or promoted only after a separate proof/receipt.

## Defect energy (E)
- `E(s) = #{(x,c) : K(s)(x,c) != s(x,c)}`.
- Lower energy = nearer to kernel closure for this defect definition.
- A decreasing defect trace is an observed run property unless a global monotonicity theorem is supplied.

## Kernel closure
- `s` is closed iff `K(s) = s` (equivalently, defect zero for the literal mismatch-count defect above).
- Low-defect states are candidates near closure; low defect alone is not fixed-point closure.

## Kernel flow
- Iterative map `s_{t+1} = K(s_t)` with defect trace `E(s_t)`.
- Convergence can be to a fixed point or a short cycle; both are recorded.
- Runtime receipts distinguish:
  - `fixed_point_receipt`: literal observed `K(s) = s`,
  - `observed_defect_nonincreasing`: finite-run trace property,
  - `observed_idempotent_at_final`: finite-state observation only.
- None of these finite-run observations imply a global theorem over all states/carriers/parameters.

## Coarse-graining / renormalisation (R)
- Block map `R_pi : s_j -> s_{j+1}` via block aggregation then ternary projection.
- Admissible partitions `pi` include anatomical or data-driven blocks.
- Persistence across scales is a separate receipt; coarse-graining does not automatically preserve every downstream consumer.

## Remark (regimes)
- Each kernel application traverses atomic ternary states, affine aggregation (`r = A s`), and a single nonlinear projection (`sgn_eps`). Analysis may lift states into an exploded form (support masks, shells, defects, components) that is exact and invertible; structure in this lift exists only when locality is preserved.

## Latent structure (definition)
- A kernel-closed (or explicitly low-defect) quotient class that persists across >=2 coarse-graining levels and is invariant under the declared admissible symmetries.
- If a later scientific claim depends on distinctions erased by the quotient, that claim cannot be transported through the quotient without enlarging the representation.

## Invariants to preserve in code
- Locality: kernel reads only neighborhood `N(x)`.
- Determinism: same inputs yield identical outputs.
- Declared symmetry equivariance: equivalent configurations under an admissible action map to equivalent outputs, when separately checked/proved.
