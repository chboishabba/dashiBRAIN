### Theorem (Kernel-closure latent structure ≠ community detection)

Let (G=(V,E,W)) be a directed weighted graph (hemibrain). Fix a DASHI kernel (K) acting on ternary fields (s:V\times C\to{-1,0,+1}) and a defect functional (E(s)=#{x:(Ks)(x)\neq s(x)}).

Let (\mathcal A_{\text{CD}}) be **any** community detection procedure whose output is a partition (\pi:V\to B) determined solely from graph structure (e.g., modularity, SBM likelihood, conductance, spectral clustering), i.e. it returns *blocks*, not a ternary field.

Define the **kernel-structure problem** as: find (s) with (E(s)=0) (or (E(s)\le \delta|V||C|)) and nontrivial persistence under admissible renormalisations (R_\pi).

Then, in general, **kernel-structure is not equivalent to community detection**, in the following strong sense:

1. **(Different mathematical objects / output types)**
   Community detection outputs a **partition** (\pi). Kernel-structure outputs a **field** (s) (and its quotient class ([s])).
   There is no canonical map (\pi\mapsto s) preserving closure, because kernel closure depends on **signed local consistency constraints** in (X=V\times C), while (\pi) contains only block membership.

2. **(Existence of kernel-closed structures with no community signal)**
   There exist graphs (G) with **no nontrivial community structure** by any standard criterion (e.g., Erdős–Rényi or degree-corrected configuration graphs; best partitions are statistically null), yet there exist **nontrivial kernel-closed fields** (s\neq 0) for admissible (K).
   Example class (constructive): take any graph and choose a kernel whose weights encode a local signed constraint (e.g., parity / alternating pattern on a bipartite or near-bipartite subgraph, or a local orientation constraint). Then (Ks=s) can hold even when there is no high-modularity or SBM block structure. The signal is **constraint satisfaction on neighborhoods**, not edge-density separation.

3. **(Existence of strong communities with no kernel closure at the same scale)**
   Conversely, there exist graphs with very strong community structure (two dense blocks with sparse cut) where a natural DASHI kernel built from residual sign structure produces **persistent defect on the cut** and can fail to admit any globally closed nontrivial (s) at that scale (or only admits trivial (s\equiv 0)).
   Reason: kernel closure requires local sign-consistency across neighborhoods, and a sharp inter-block interface generically induces a defect boundary (\partial s) even if communities are “perfect” in the density sense.

4. **(Different invariants / different objective principles)**
   Community detection is (explicitly or implicitly) optimizing a **global partition functional** (e.g., modularity, likelihood, cut).
   Kernel-structure is solving a **fixed-point / low-defect constraint**:
   [
   Ks=s \quad (\text{or } E(s)\text{ small})
   ]
   and then demanding **renormalisation persistence** (R_\pi\circ K \approx K_\pi\circ R_\pi).
   These are not reducible to one another without collapsing the kernel semantics.

**Proof sketch (why this is rigorous enough to rely on):**

* (1) is type-theoretic: partitions and ternary fields live in different spaces ((\Pi(V)) vs ({-1,0,+1}^{V\times C})). Any equivalence would require a canonical, structure-preserving isomorphism, which does not exist because kernel closure depends on signs and channels, not solely block membership.
* (2) gives a separation: pick a null graph family where community detection is provably uninformative; then exhibit a kernel (admissible, local) whose fixed points encode a local constraint not captured by density-based partitions.
* (3) gives the opposite separation: pick a graph with extreme modularity; show closure fails because the kernel enforces neighborhood consistency that is violated at interfaces, producing persistent defect even though the partition is “obvious.”
  Together these give non-equivalence.

---

### Practical corollary (one-liner)

Community detection finds **where edges are dense**; DASHI kernel structure finds **where a ternary valuation is locally self-consistent and scale-persistent**. Dense blocks are neither necessary nor sufficient for kernel closure.
