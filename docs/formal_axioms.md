# Formal axioms (DASHI kernel)

These definitions anchor code semantics. Functions should map to the labels below.

## Carrier (G)
- Directed weighted graph with nodes `V`, edges `E`, weights `W >= 0` (or residual weights).
- Optional channel index `c in C` if multiple signal types exist.

## Valuation field (s)
- Ternary field `s : V x C -> {-1, 0, +1}`.
- Interpretations: `+1` affirmed/excess, `-1` opposed/deficit, `0` neutral/underdetermined.

## DASHI kernel (K)
- Local involutive projection defined by weights on neighborhoods `N(x)`:
  `K(s)(x,c) = sgn(sum_{y in N(x)} sum_{c'} w_{(x,c),(y,c')} * s(y,c'))` with ternary deadzone.
- Properties: locality, involution equivariance, symmetry/gauge equivariance.

## Defect energy (E)
- `E(s) = #{(x,c) : K(s)(x,c) != s(x,c)}`.
- Lower energy = nearer to kernel closure.

## Kernel closure
- `s` is closed if `K(s) = s` (defect zero). Low-defect states approximate closure.

## Kernel flow
- Iterative map `s_{t+1} = K(s_t)` with defect trace `E(s_t)`.
- Convergence can be to a fixed point or a short cycle; both are recorded.

## Coarse-graining / renormalisation (R)
- Block map `R_pi : s_j -> s_{j+1}` via block aggregation then ternary projection.
- Admissible partitions `pi` include anatomical or data-driven blocks.

## Remark (regimes)
- Each kernel application traverses atomic ternary states, affine aggregation (`r = A s`), and a single nonlinear projection (`sgn_eps`). Analysis may lift states into an exploded form (support masks, shells, defects, components) that is exact and invertible; structure in this lift exists only when locality is preserved.

## Latent structure (definition)
- A kernel-closed (or low-defect) quotient class that persists across >=2 coarse-graining levels and is invariant under admissible symmetries.

## Invariants to preserve in code
- Locality: kernel reads only neighborhood `N(x)`.
- Determinism: same inputs yield identical outputs.
- Involution symmetry: equivalent configurations under admissible symmetry actions map to equivalent outputs.
