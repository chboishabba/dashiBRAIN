# MaleCNS effector + functional-imaging programme

## Source attribution rule

Every external scientific source used by this lane must carry:

1. author or consortium;
2. title; and
3. DOI, PMID, accession, dataset/version identifier, or another stable identifier when a DOI is unavailable.

A bare URL is not a sufficient canonical scientific receipt when a stable identifier exists.

## Canonical sources

- Berg et al., **Sexual dimorphism in the complete Drosophila male central nervous system connectome**, Cell (2026), DOI `10.1016/j.cell.2026.08.015`.
- Bates et al.; BANC-FlyWire Consortium, **Distributed control circuits across a brain-and-cord connectome**, Nature (2026), DOI `10.1038/s41586-026-10735-w`.
- Gauthey, Lin, Ahmed, Leifer, Murthy, Thiberge et al., **High-speed whole-brain imaging in Drosophila**, Nature Communications (2026), DOI `10.1038/s41467-026-72437-1`.
- Lemon et al., **Whole-central nervous system functional imaging in larval Drosophila**, Nature Communications (2015), DOI `10.1038/ncomms8924`.

## Scientific carrier

The old hemibrain prototype should no longer be interpreted as terminating at an internal brain graph. For MaleCNS the intended embodied chain is

```text
sensory input
  -> sensory neuron
  -> central circuit
  -> descending / ascending circuit
  -> VNC circuit
  -> motor / endocrine / visceral efferent neuron
  -> motor command / drive
  -> physical effector state
  -> body state / biomechanics
  -> behavioural observation
  -> sensory return
```

The structural connectome directly constrains the CNS part of this chain. It does **not** by itself supply instantaneous muscle activation, force, pose, or behaviour. Those downstream coordinates are nevertheless first-class programme targets and should be supplied by separately attributed neuromuscular, biomechanical, and behavioural producers.

Key non-collapse boundaries:

```text
motor neuron != motor command
motor command != effector state
effector state != observed movement
structural annotation != dynamic neural state
```

## Effector ROM

Cross-pollinating the SeaMeInIt/DASHI-ROM pattern, represent a high-dimensional physical effector/body state `e` by reduced coordinates `z`:

```text
project : EffectorState -> EffectorCoefficient
reconstruct : EffectorCoefficient -> EffectorState
residual : EffectorState -> Residual
```

The residual is part of the scientific object. A reduced effector coordinate must not silently become an exact inverse or a complete behavioural state.

This allows tests of

```text
connectome-derived neural state -> reduced effector coordinates
```

without requiring every microscopic muscle/body degree of freedom to be modelled at the first pass.

## Functional observation quotient

The existing DASHI brain/fMRI work should be generalized as a modality family rather than copied literally as BOLD fMRI.

For Drosophila, the current source set contains strong whole-brain / whole-CNS **functional optical imaging** receipts, including adult behaving-fly calcium imaging. We have not established a literal Drosophila BOLD-fMRI receipt.

Therefore use

```text
latent neural state -> neural observation quotient
latent effector/body state -> behavioural observation quotient
```

with an explicit modality tag such as:

- optical calcium;
- optical voltage;
- electrophysiology;
- BOLD fMRI;
- behavioural kinematics;
- force/contact.

Do not rename calcium imaging as fMRI.

## Two-ended validation

The stronger experiment is not just graph closure. It is simultaneous neural-side and body-side validation:

```text
Stimulus
  -> NeuralState
  -> MotorCommand
  -> MotorNeuronDrive
  -> EffectorState
  -> BodyState
```

with observation maps

```text
Q_neural : NeuralState -> NeuralObservation
Q_body   : BodyState   -> BehaviourObservation
```

and independently receipted measurements on both ends.

## Male/female comparison

Use isomorphic and dimorphic MaleCNS/FlyWire correspondences as held-out structure. Test whether a sex-independent neural-to-effector map transfers better through isomorphic circuitry than through explicitly dimorphic circuitry. This is a falsifiable experiment, not an assumption.

## Coarse-graining programme

Retain the old negative result that arbitrary/random coarse-grainings did not establish persistence. Add biologically defined quotient families:

- neuron -> cell type;
- neuron -> neuropil/region;
- neuron -> hemilineage;
- sensory -> central -> descending/ascending -> VNC -> effector class;
- body-part controller modules.

Compare their commutation defect against random, degree-binned, and purely spatial controls.

## Implementation boundary

This document changes the programme target, not the evidential status of old runs. Existing hemibrain numeric receipts remain single-scale/non-promoting until new dataset-specific runs are produced and checksummed.
