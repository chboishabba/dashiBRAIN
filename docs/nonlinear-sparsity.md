# Nonlinear exploded sparsity

This note fixes the meanings of **atomic**, **affine**, **nonlinear**, and **exploded** for the DASHI graph kernel and states the precise theorem that the implementation can support.

## 1. Atomic carrier and affine field

Let \(G=(V,E,W)\) be a finite weighted directed graph, \(n=|V|\), and

\[
T := \{-1,0,+1\}, \qquad s\in T^V.
\]

The coordinates \(s_i\) are the **atomic states**. Let \(A\in\mathbb R^{n\times n}\) be the weighted adjacency operator. The pre-projection field is

\[
h(s) := As+b,
\]

where \(b\in\mathbb R^n\) is optional. This stage is affine (linear when \(b=0\)).

For a deadzone \(arepsilon\ge 0\), define

\[
q_\varepsilon(x)=
\begin{cases}
+1,&x>\varepsilon,\\
0,&|x|\le\varepsilon,\\
-1,&x<-\varepsilon.
\end{cases}
\]

The DASHI update is the coordinate-coupled nonlinear map

\[
\Phi_\varepsilon(s):=q_\varepsilon(As+b).
\]

## 2. Exploded representation

For any \(s\in T^V\), define

\[
P(s)=\{i:s_i=+1\},\quad
Z(s)=\{i:s_i=0\},\quad
N(s)=\{i:s_i=-1\}.
\]

Let \(\pi_+(s)\) and \(\pi_-(s)\) denote the connected-component decompositions of the subgraphs induced by \(P(s)\) and \(N(s)\), using the undirected support graph of \(A+A^\top\). For an orbit \(s_0,s_1,\ldots\), define the transition defect

\[
D_t:=\{i:(s_t)_i\ne(s_{t-1})_i\}.
\]

The **exploded representation** is

\[
\mathcal E(s_t;s_{t-1})=
\big(P(s_t),Z(s_t),N(s_t),\pi_+(s_t),\pi_-(s_t),D_t\big).
\]

This is a lossless re-expression of the ternary state together with graph geometry and, when a predecessor is supplied, transition geometry. “Exploded” therefore means **atoms made explicit as signed support, zero support, components, and defects**. It does not mean that support is automatically sparse.

## 3. Nonlinear sparsity theorem

### Theorem 1 — Exact threshold sparsification

For every \(s\in T^V\),

\[
Z(\Phi_\varepsilon(s))
=
\{i\in V: |(As+b)_i|\le\varepsilon\}.
\]

Consequently,

\[
\|\Phi_\varepsilon(s)\|_0
=
|V|-|Z(\Phi_\varepsilon(s))|.
\]

Thus the zero pattern is produced exactly by a **nonlinear, adjacency-coupled threshold**. No sparsity penalty is required.

**Proof.** Immediate from the definition of \(q_\varepsilon\), coordinate by coordinate. □

### Theorem 2 — Fixed-point margin characterization

A state \(s^*\in T^V\) is fixed, \(\Phi_\varepsilon(s^*)=s^*\), iff for every \(i\in V\):

\[
\begin{aligned}
s_i^*=+1 &\Rightarrow (As^*+b)_i>\varepsilon,\\
s_i^*=0 &\Rightarrow |(As^*+b)_i|\le\varepsilon,\\
s_i^*=-1 &\Rightarrow (As^*+b)_i<-\varepsilon.
\end{aligned}
\]

**Proof.** Expand the equality \(q_\varepsilon(As^*+b)=s^*\) coordinatewise. □

### Corollary 2.1 — Robust atoms

Define the signed margin

\[
m_i(s^*)=
\begin{cases}
(As^*+b)_i-\varepsilon,&s_i^*=+1,\\
\varepsilon-| (As^*+b)_i |,&s_i^*=0,\\
-(As^*+b)_i-\varepsilon,&s_i^*=-1.
\end{cases}
\]

At a fixed point, \(m_i(s^*)\ge0\) for all \(i\). If a perturbation \(\delta h\) satisfies

\[
|\delta h_i|<m_i(s^*)
\]

for every strict-margin coordinate, that coordinate cannot change under reprojection. Hence only small-margin atoms are eligible to enter a defect set under a sufficiently small perturbation.

This is the precise sense in which defects may localize near a constraint boundary. Sparsity of the defect set is an empirical or additional-assumption result; it is not guaranteed for every graph.

### Theorem 3 — Conditional defect bound

Let \(s^*\) be fixed and let \(\tilde h=As^*+b+\delta h\). Define

\[
B_\eta(s^*):=\{i:m_i(s^*)\le\eta\}.
\]

If \(\|\delta h\|_\infty<\eta\), then

\[
\{i:q_\varepsilon(\tilde h_i)\ne s_i^*\}\subseteq B_\eta(s^*).
\]

Therefore

\[
|D|\le |B_\eta(s^*)|.
\]

A sparse low-margin set implies a sparse defect response.

**Proof.** A coordinate outside \(B_\eta\) has margin \(>\eta\); a perturbation smaller than \(\eta\) cannot cross either threshold. □

## 4. Contrast with ℓ1 and ReLU sparsity

### ℓ1 sparsity

A typical ℓ1 estimator solves

\[
\min_x L(x)+\lambda\|x\|_1.
\]

Its zeros are induced by an explicit separable convex penalty. Graph structure matters only if inserted into \(L\) or a separate regularizer. The optimization objective prefers small coordinate support.

### ReLU sparsity

ReLU applies

\[
\operatorname{ReLU}(x_i)=\max(0,x_i)
\]

coordinatewise. Zeros mean “non-positive preactivation”. ReLU is nonlinear but does not by itself encode ternary opposition, a symmetric deadzone, fixed-point satisfaction, or graph-component geometry.

### DASHI exploded sparsity

DASHI applies

\[
s\mapsto q_\varepsilon(As+b).
\]

The threshold is coordinatewise, but the argument of every threshold is adjacency-coupled. Zeros mean that the aggregate local field lies inside the unresolved deadzone. The exploded representation then exposes signed support, neutral support, component structure, margins, and transition defects.

| Property | ℓ1 | ReLU | DASHI exploded |
|---|---:|---:|---:|
| Explicit sparsity penalty | yes | no | no |
| Convex mechanism | usually | no | no |
| Symmetric ternary output | no | no | yes |
| Adjacency-coupled before threshold | optional | optional | intrinsic |
| Zero semantics | penalized coordinate | non-positive activation | unresolved local field |
| Fixed-point constraints | not intrinsic | not intrinsic | intrinsic |
| Components / shells / defects | not intrinsic | not intrinsic | explicit in lift |

No claim is made that DASHI cannot be represented by a larger optimization problem. The distinction is operational: its sparsity is generated by graph-coupled threshold dynamics, not by an ℓ1 objective or a one-sided activation alone.

## 5. Weighted CSP / SAT-style geometry

Associate one ternary variable \(x_i\in T\) with each node. For each node define the local threshold constraint

\[
C_i(x):
\begin{cases}
( Ax+b )_i>\varepsilon,&x_i=+1,\\
|( Ax+b )_i|\le\varepsilon,&x_i=0,\\
( Ax+b )_i<-\varepsilon,&x_i=-1.
\end{cases}
\]

Then

\[
x\text{ is a kernel fixed point}
\iff
\bigwedge_{i\in V} C_i(x).
\]

This is an exact equivalence to a finite **weighted threshold CSP**. It is SAT-style in the sense that a global state satisfies a conjunction of local constraints, but it is not generally Boolean CNF-SAT without an explicit encoding.

Define the violated-constraint set

\[
U(x):=\{i:\neg C_i(x)\}.
\]

For the synchronous update, the kernel defect is exactly

\[
E(x)=|U(x)|=|\{i:\Phi_\varepsilon(x)_i\ne x_i\}|.
\]

The exploded geometry has the following CSP reading:

- \(P,Z,N\): ternary variable assignments;
- signed components: connected satisfied domains with common assigned polarity;
- \(U(x)\): currently violated local constraints;
- \(D_t\): variables revised by one parallel repair pass;
- low-margin set \(B_\eta\): near-critical constraints that can change under small perturbation;
- neutral shell: variables whose local fields remain within the undecided interval.

Unlike ordinary SAT terminology, a changed node is not automatically a minimal unsatisfiable core. Minimal cores require a separate minimality test over subsets of constraints. The implementation must preserve this distinction.

## 6. Hemibrain interpretation boundary

The current hemibrain runs support these empirical statements:

1. the final state is fixed under the implemented kernel;
2. the transition defect set is sparse relative to the full node set;
3. neutral and signed-component geometry can be measured in the exploded representation;
4. the geometry changes or collapses under several coarse-grainings.

They do **not** by themselves prove that every graph has sparse neutral support, that every defect is biological, or that defect nodes form minimal SAT cores. Those are hypotheses to test with margins, null models, reconstruction controls, and locality-preserving perturbations.
