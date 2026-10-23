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
