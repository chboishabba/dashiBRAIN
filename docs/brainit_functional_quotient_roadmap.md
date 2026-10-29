# DASHI BRAIN × Brain-IT: consumer retention and functional quotient roadmap

**Status (2026-10-24):** the finite diagnostic module is implemented, not a new fMRI benchmark.
Source Agda owner: [dashi_agda PR #1121](https://github.com/chboishabba/dashi_agda/pull/1121).
Detailed Agda/CMS source and promotion gates: `Docs/BrainITDASHIBrainCMSRoadmap.md` in that branch.

## Sources and attribution

- **Brain-IT:** Beliy, Zalcher, Kogman, Wasserman, Irani, *Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer*, ICLR 2026; arXiv:2510.25976v2 (2026-03-01), https://arxiv.org/html/2510.25976v2. Appendix A Table T2 (functional vs anatomical), T3 (cluster-count sensitivity); Appendix B Table T7 (Subject 1 15/30/60 min).
- **Earlier source:** Beliy et al., *The Wisdom of a Crowd of Brains: A Universal Brain Encoder*, arXiv:2406.12179.
- **DASHI BRAIN:** repo-original connectome/ternary-kernel formalism, biological coarse-graining numerical sweeps and modality-separated functional observation. This repository does not own Brain-IT's model or raw NSD data.
- **CMS Agda:** separate CMS Drell–Yan HEPData dataset (CMS Collaboration, EPJC 83 (2023) 628, DOI 10.1140/epjc/s10052-023-11631-7), retained **solely** as a methodological exemplar of immutable measurements, digests, covariance and source-specific claim boundaries. It supplies no neuroscientific validation.

## Completed finite cut

`dashi/analysis/consumer_retention.py` implements:

- `audit_consumer_retention(states, Q, T)` tests exact consumer factorisation **over enumerated states**; a collision is a concrete no-decoder witness for that sample, but no collision is not universal proof.
- `audit_commutation(states, K, Q, K_coarse)` counts exact mismatches between `Q(K(x))` and `K_coarse(Q(x))`. The comparison must use a matched state/process and declared equality; experimental neuro-data need a calibrated distance, not an artificial Boolean assertion.
- `tests/test_consumer_retention.py` reproduces the same 3-coordinate toy collision used by dashi_agda: distinct microscopic states yield identical coarse observation yet distinct Laplacian variation. It also tests a retained consumer and commuting/noncommuting dynamics.

These diagnostics are intentionally agnostic to whether observations are calcium, voltage, BOLD or other measurements. The caller must tag modality, time alignment, dataset and source before any empirical promotion.

## Experiment matrix (not yet executed)

| Variant | Control purpose | Compare |
| --- | --- | --- |
| Brain-IT functional V2C (published) | Target method: voxel embeddings/GMM clusters | Table T2 pixel corr .386; SSIM .486 |
| 3D spatial coordinates (published) | Anatomical-coordinate control | Table T2 .378; SSIM .491 |
| Schaefer-400 and 1000 (published) | Atlas controls; respectively 84 and 198 selected clusters | Table T2 .371/.378 pixel corr |
| Random and degree-binned at matched counts | Non-biological compression controls | New run required |
| DASHI locality/kernel-admissible quotient | Test preserving effective neighborhood structure | New run required |
| V2C 8/32/64/128/256/512 | Cluster-count sensitivity | Paper T3; re-evaluate new quotient at same capacity |

These paper rows are **not newly computed results**. Structural connectome components and learned V2C clusters are different objects. Avoid treating the fly's hemibrain results as a replication of a human fMRI model.

## Next cuts / acceptance gates

**P0, exact-head formal certification:** in dashi_agda, run `agda -i . DASHI/Biology/BrainITValidation.agda` with the repository's matching Agda/stdlib versions; run static contract tests. No kernel-green assertion without output.

**P1, freeze published Brain-IT prediction identity:** pin official repository revision, model weights and voxel-to-cluster map checksums, NSD subject/stimulus splits and normalization/preprocessing, generated images and evaluator versions. Independent subject-1 replay of T7 or discrepancy report is the admission condition.

**P1, matched experimental test:** compute the exact same image metrics across functional, spatial, anatomical, random, and DASHI-admissible quotients, with matched cluster capacity and identical held-out stimulus protocol. Report bootstrap intervals, per-subject metrics, choice of metric direction, and seeds. Include failure cases. No claim of DASHI advantage before results.

**P2, registration and multi-modality:** the existing `functional_registration.py` and `functional_bridge.py` handle fly connectome/optical and mammalian BOLD as distinct sources; add independent data receipts for any calibrated mapping. Never equate fly calcium imaging with human fMRI.

**P3, mechanistic tests:** test locality-preserving commutation defects, stability under perturbing trained cluster assignments, feature-branch separability, and held-out generalisation. Attention heatmaps alone do not establish anatomical or mechanistic identity.

## Acceptance commands

```sh
python -m pytest -q tests/test_consumer_retention.py tests/test_functional_bridge.py
# In dashi_agda with its pinned toolchain:
agda -i . DASHI/Biology/BrainITValidation.agda
```

The absence of a new external dataset/checkpoint receipt is an intentional **open frontier**, not an implied pass.

## Cross-lane DMT consumer: no external-information promotion

On 24 October 2026, the existing finite audit gained a source-independent regression test cross-linked to [dashi_agda DMT PR #1243](https://github.com/chboishabba/dashi_agda/pull/1243). The new test constructs four invented combinations of viewed stimulus and independently concealed target. The same perfect viewed-stimulus observation retains the **viewed stimulus** consumer but has an explicit collision for the **concealed target** consumer. A deliberately leaked observation containing the secret then retains the secret perfectly, demonstrating how failure of blinding changes admissible conclusions. It does **not** add a clinical DMT measurement, Brain-IT checkpoint, NSD image result, or external-agent observation.

This directly **reuses** the original local audit_consumer_retention rather than inventing another consumer-fibre calculus. The DMT branch's own Agda consumer proof imports the established generic CoarseFineFabricCalculusExact, not the unmerged Brain-IT PR #1121, so both PRs can be independently reviewed and merged safely.

The shared highest-alpha method is *consumer-relative prediction with source/session provenance*, not transfer of either experimental dataset or a literal neural kernel. Brain-IT's arXiv:2510.25976v2 published values remain the source owner's evidence. Timmermann et al. PNAS 2023 (DOI 10.1073/pnas.2218949120) is the independently credited DMT EEG/fMRI source; its GitHub repository timmer500/DMT_Imaging was inspected at revision b35701d88d904e2f602164ff2eae0fc879772078. A separate fail-closed DMT-file admission and train/test split check now lives in PR #1243, without copying empirical data into this Drosophila/Brain-IT repository.

**Next empirical admission:** obtain rights-reviewed, hash-locked source files, participant/session/condition splits, matched time axes, and replay an original analysis. Separately test any claim of unexpected concealed-target information under adversarial controls. Neither a source-side Brain-IT visual reconstruction nor a toy consumer collision identifies independently conscious DMT entities. No data or human research task is claimed performed by this cross-reference.



## Exact-head finite consumer-test receipt (2026-10-24 Brisbane)

The current GitHub PR #9 runtime \`dashi/analysis/consumer_retention.py\` at Git blob **384a9b1d7082857000128635259c9b1a9d989f80** and its \`tests/test_consumer_retention.py\` at Git blob **89fb3b6391331f299b7dc6e37ec4bc34331a3646** were reconstructed locally with **byte-for-byte matching Git blob SHA1**. Against that exact pair:

\`\`\`text
python -m pytest -q tests/test_consumer_retention.py
...... [100%]
6 passed in 0.03s
\`\`\`

The run required only the Python standard library and pytest; the package's two parent \`__init__.py\` entries were supplied solely to allow local import. **This is a valid focused Python exact-content test receipt**, not GitHub Actions, an Agda kernel proof, or a replay of a Brain-IT/NSD checkpoint. The additional \`tests/test_functional_bridge.py\` tests are outside this exact-content receipt and remain pending for the combined acceptance command. A matching source-level data consumer is implemented on sibling Agda DMT PR #1243; its new paired-fMRI evaluator has five locally passed *synthetic prototype* tests but no actual DMT participant data analysis.



## AO fleet discriminator convergence — 2026-10-29

The consumer-retention runtime now also serves as the finite discriminator surface for AO fleet debugging.

The golden authority is the sibling `dashi_agda` convergence branch. Python does not create proof authority.

### Five distinct levels

The runtime must not collapse these objects:

1. **MP eigenspace basis** — selected spectral vectors with `lambda_i > lambda_plus`.
2. **Primary topics** — thresholded macro quotients of admitted spectral representations.
3. **Subtopics** — local branches below a primary topic.
4. **Fibres** — complete-link cliques under the declared near relation.
5. **Goals** — typed, unblocked deliverables eligible for dispatch.

Supported spectral truncations are exactly `K=50` and `K=100`.

### Lexical admission

Topic labels are rejected before clustering when they are raw implementation noise.

Rejected classes include:

- hexadecimal literals such as `0xffff1000`;
- pointer-like or memory-address strings such as `0xffff1000()`;
- UI-control fragments such as `collapse(property: auto)`;
- raw HTML/DOM tokens.

A rejected label cannot become a primary topic or subtopic.

### Finite hierarchy construction

The discriminator constructs primary topics as connected components of the declared primary-distance threshold graph.

It constructs subtopics as connected components at the tighter subtopic threshold inside each primary topic.

It constructs fibres with deterministic complete-link admission at the tightest threshold. Every two members of one fibre must therefore be within the fibre threshold.

The thresholds must satisfy `fibre <= subtopic <= primary`.

Noise labels are removed before all three constructions.

### Fleet concurrency

A valid full fleet schedule has exactly twenty assigned worker slots.

Every assigned goal must be unblocked.

Every assigned goal must be Pareto eligible.

The scheduler chooses the top-ranked compatible goals.

Every two distinct scheduled goals must have disjoint typed support.

A one-worker serial loop or modulo-eight reuse is not a valid W=20 schedule.

### Combinatorial Max Cut

The Python diagnostic uses a literal finite graph cut.

For the ±1 cut signal on an undirected unweighted graph, the audit checks the exact identity:

`x^T L x = 4 * cut_edge_count`.

An optional SDP upper-bound receipt may be checked against the cut weight.

The runtime does not solve an SDP.

The runtime does not infer a Yang–Mills equation or Navier–Stokes estimate from a graph cut.

Physics reuse requires an explicit same-object adapter on the golden formal side.

### Verification gates

- token-noise regression;
- K=50/K=100 admission and all-other-K rejection;
- hierarchy receipt keeps five levels distinct;
- twenty independent goals fill twenty slots;
- fewer than twenty compatible goals fail loudly;
- typed-support overlap prevents concurrent dispatch;
- finite Max Cut Laplacian identity is exact on deterministic fixtures;
- existing vertex-map orbit rendering remains deterministic.
