# **INTRODUCTION TO THE GFP MUTATION PIPELINE**

This README explains overall steps that were taken to reach the goal of 10 most promising sfGFP mutations. For deeper explanation of the algorithm please see the pipeline section. All LLMs did not undergo additional learning.

## Executive Summary

We applied a two-stage ranking strategy. The first stage prioritized sequence plausibility and structural compatibility using ProteinMPNN, SaProt and ESM-2. The second stage incorporated thermal stability prediction via TemBERTure and FoldX energetic refinement for final candidate selection.

## Step 1. Mutation data preparation

We decided to use the pool of mutations from GFP_data.xlsx file as a base for our pipeline. The reason for this choice is that, according to the BLAST algorithm, wildtype (wt) avGFP is 96% similar to wt sfGFP, the object of study this year. These mutations proved to improve brightness of the original variant, but there was no record on the effects of thermostability. We decided to use these mutations and recombine them thus creating the unique mutation combinations which will later go through multiple filters to exclude those that show no improvement on the brightness and thermostability parts of the original sfGFP protein. Those mutations that were on the Exclusion_List.csv were immediately filtered out. 

```
step1-create-candidates/run-candidates.py
```

In result we got a csv file containing a list of 500 possible mutations which go through multiple filters in the next steps.

```
step1-create-candidates/500-candidates.csv
```

## Step 2.1 Mutation filtration based on thermal fluctuation (ProteinMPNN)

For thermal fluctuation simulation we used ProteinMPNN (Protein Message Passing Neural Network - https://github.com/dauparas/ProteinMPNN). It calculates structural compatibility and stability thus evaluating the compatibility of existing sequences. ProteinMPNN is one of the most accurate and accessible LLMs at this moment.

Due to thermal fluctuation the exact coordinates of a protein may vary a little. Without taking this detail into account the structure of a new protein may come out unstable which is why ProteinMPNN uses backbone_noise - it sets the interval in Å in which the coordinates of protein atoms may differ. Different noise variation parameters are also used to check the stability of the protein structure - each changes the backbone_noise from 0.00 Å which excludes any thermal fluctuation to 0.20 Å which stands for the most extreme fluctuation, any further noise parameter resulting in heavily unstable structure. If global score drops more than on 10-15% between noise 0.00 and noise 0.20 parameters the structure of the protein is too unstable. 

```
step2.1-proteinmpnn/run.sh
```

In result we get a csv file with several ProteinMPNN stability scores and a final score for each mutation.

```
step2.1-proteinmpnn/results-proteinmpnn.csv
```

## Step 2.2 Mutation filtration based on evolutionary patterns (ESM-2)

To further test how the actual folding will undergo we used ESM-2 (Evolutionary Scale Modeling - https://github.com/facebookresearch/esm). It gives a score for each mutation based on real evolutionary patterns which is achieved through its deep database of million protein sequences that were used for machine learning. 

```
step2.2-esm/run-esm.py
```

We use a `facebook/esm2_t33_650M_UR50D` model as it balances performance with computational demands and has a great amount of parameters which makes sequence score prediction more flexible.

In result we got a list of scores for each mutation - the lower the score the more "natural" the sequence is and the higher chance it has to fold into a stable structure.

```
step2.2-esm/results-esm.csv
```

## Step 2.3 Mutation filtration based on predicted mutation effects (SaProt)

ESM takes into account evolutionary context of the amino acid sequence but fails to see if the same evolutionary context is applied to the 3D structure of the mutant as well. However, SaProt (Structure-aware Protein Language Model) specializes on this problem. Using it alongside ESM will filter out those mutant proteins which sequence nor 3D structure doesn't make sense from evolutionary standpoint.

```
step2.3-saprot/run-saprot.py
```

In this case we are using SaProt_650M_PDB model as it shows satisfying results and is trained on 60K PDB structures (as it's referenced on the official GitHub page https://github.com/westlake-repl/Saprot

The script returns SaProt scoring results in a csv table. 

```
step2.3-saprot/results-saprot.csv
```

The higher the score the more beneficial for the protein the mutation probably is. These scores alongside ESM and ProteinMPNN scores will further be used in combination to filter out 500 mutations into 50 most stable and bright sfGFP protein structures. 

## Step 3: Interim Scoring (ML Ensemble)

### Rationale

Each ML model captures different aspects of protein fitness:
- **ProteinMPNN**: Structural compatibility and stability
- **ESM-2**: Evolutionary conservation and naturalness of the amino acid sequence
- **SaProt**: Combined structural and evolutionary context of the structure

The ensemble approach combines these complementary signals for more robust predictions.

```
step3-interim-scoring/run-interim-scoring.py
```

### Model weight explanation

The weights reflect each model's expected contribution to predicting protein stability

**1. ProteinMPNN** (0.60 weight)

ProteinMPNN directly assesses whether a sequence can fold into the target structure. The model's sensitivity to backbone perturbations (via noise analysis) makes it particularly valuable for predicting thermostability. Mutations that maintain high scores across multiple noise levels are likely to be stable at elevated temperatures.

**2. SaProt** (0.25 weight)

SaProt uniquely bridges the gap between structure-based and sequence-based methods. By encoding structural features directly into the input, it captures both evolutionary patterns (like ESM) and structural constraints (like ProteinMPNN). This hybrid approach provides complementary information that neither purely structural nor purely sequence-based models can capture alone.

**3. ESM-2** (0.15 weight)

While ESM-2 excels at identifying sequences that match evolutionary patterns, it does not directly consider protein structure. Thus, ESM-2 provides useful but secondary information compared to structure-aware models.

| Model       | Weight | Rationale                    |
| ----------- | ------ | ---------------------------- |
| ProteinMPNN | 0.60   | Direct structural evaluation |
| SaProt      | 0.25   | Combined structural-language |
| ESM-2       | 0.15   | Only sequence-based          |

All scores are normalized using percentile rank due to several reasons:
- Scores from different models are on different scales
- Percentile rank transforms scores to [0,1] range
- Better for combining scores than raw values

MPNN score combines base and stability scores:

| Component       | Weight | Rationale                            |
| --------------- | ------ | ------------------------------------ |
| base_score      | 0.6    | Structural compatibility at baseline |
| stability_score | 0.4    | Stability under thermal stress       |

SaProt score is the normalized saprot_score, ESM score is the normalized esm_score (reversed).

Candidates are sorted by Ensemble_score, and the top 50 are selected for further analysis. 

```
step3-interim-scoring/top-50-candidates.csv
```

## Step 4: FoldX Analysis

**FoldX** (Schymkowitz et al., 2005) is an empirical force field for protein stability analysis used for calculating energetic effects of mutations.

**Key features:**
- Fast calculation (seconds per mutation)
- Accurate ΔΔG predictions
- Comprehensive energy decomposition
- Widely used in protein engineering

The script `step4.1-foldx/create-indlst.py` converts mutations from trimmed indexing to FoldX-compatible format.

**Parameters:**
- `BuildModel`: Mode for modeling mutations
- `--pdb`: Input structure file
- `--mutant-file`: File containing mutation lists
- `--numberOfRuns 5`: Number of independent runs for stochastic averaging - 5 runs balance accuracy and computational cost

The script `step4.1-foldx/add-foldx-results.py` processes FoldX output:

**FoldX components and weights:**

| Component             | Weight | Description                                |
| --------------------- | ------ | ------------------------------------------ |
| Total Energy          | 0.55   | Overall stability (lower is better)        |
| Van der Waals clashes | 0.20   | Steric strain (lower is better)            |
| Sidechain entropy     | 0.15   | Conformational cost (lower is better)      |
| Electrostatics        | 0.05   | Charge interactions (lower is better)      |
| Solvation             | 0.05   | Hydrophobic interactions (lower is better) |

# Step 4.2: Thermostability prediction via TemBERTure

**TemBERTure** is a temperature-aware protein language model that predicts mutational effects by combining sequence context with thermodynamic stability information. Unlike structure-based methods like FoldX, TemBERTure was trained on large-scale protein stability datasets containing experimental thermostability measurements (official GitHub page: https://github.com/ibmm-unibe-ch/TemBERTure.git).

**Model components:**

| Component         | Task           | Description                                       |
| ----------------- | -------------- | ------------------------------------------------- |
| TemBERTureCLS     | Classification | Predicts whether mutation is stabilizing (binary) |
| TemBERTureTM (x3) | Regression     | Predicts melting temperature (Tm) in °C           |

**Ensemble strategy:**
- Three independent replicas of TemBERTureTM are averaged to obtain robust Tm predictions
- Standard deviation across replicas serves as an uncertainty metric
- CLS model provides an additional stabilising/destabilising classification for each variant

```
step4.2-temberture/results-temberture.csv
```

# Step 5: Final scoring (Multi-model ranking and final candidate selection)

The final ranking combines all previously calculated signals into a unified score. The pipeline uses a two-stage strategy to balance computational cost and prediction quality.

```
step5-final-scoring/run-final-scoring.py
```

During the first stage, 500 generated variants are filtered using sequence- and structure-based protein language models:

- **ProteinMPNN** evaluates structural compatibility and backbone stability, making it the primary sequence-level predictor.
- **SaProt** adds structure-aware evolutionary information by combining sequence context with structural embeddings.
- **ESM-2** estimates sequence naturalness based on evolutionary patterns learned from large protein sequence databases.

These models are computationally efficient compared with structure-based simulations, therefore they are used as an initial screening step. The highest-ranked 50 candidates are selected for more expensive downstream analysis.

During the second stage, the remaining candidates are evaluated with additional thermostability and energetic predictors:

- **TemBERTure** predicts thermal stability by estimating melting temperature (Tm) from sequence information. Since most candidates from the initial generation stage are unlikely to be competitive, applying TemBERTure to all 500 variants would increase computational cost without providing significant additional information.
- **FoldX BuildModel** provides an energetic assessment of mutation effects using a physics-based force field. FoldX requires significantly more computational resources due to structural modeling and multiple stochastic runs, therefore it is applied only to the reduced candidate set.

The final score is calculated as a weighted combination of normalized model outputs:

| Model | Weight | Rationale |
| --- | --- | --- |
| ProteinMPNN | 0.30 | Main structural compatibility signal |
| SaProt | 0.10 | Structure-aware evolutionary information |
| ESM-2 | 0.05 | Sequence naturalness prior |
| TemBERTure | 0.40 | Direct thermostability prediction |
| FoldX | 0.15 | Physics-based energetic refinement |

The weights were selected empirically based on the expected predictive relevance of each method. TemBERTure receives the highest contribution because thermostability is the main optimization target, while ProteinMPNN remains important due to its direct assessment of structural compatibility. FoldX is used as a complementary physical validation step rather than the primary ranking criterion because energy-based predictions alone may not capture all sequence-level effects.

All model outputs are normalized using percentile ranking before combination, allowing scores from different models with different scales and directions to be integrated into a single comparable metric.

The final pipeline produces a ranked list of the 10 highest-scoring candidates:

```
step5-final-scoring/final-top-10-restored.csv
```

## Step 6: Structural validation and further analysis (planned)

The final candidates were designed with a multi-stage computational pipeline combining sequence-based deep learning models and structure-based energy evaluation.

Molecular dynamics (MD) simulations using GROMACS were initially planned as a final validation step to estimate structural stability and dynamic behavior of the designed variants. However, due to practical limitations related to chromophore (CRO) handling in GFP structural preparation and the limited project timeframe, MD simulations were not included in the final ranking pipeline.

Instead, structural evaluation was performed using FoldX energy calculations, which provide a computationally efficient estimation of mutation-induced stability changes. FoldX was applied only after sequence-based filtering due to its higher computational cost.

The final ranking therefore combines:

- **ProteinMPNN** — sequence design compatibility and mutation plausibility
- **SaProt** — structure-aware evolutionary representation scoring
- **ESM-2** — sequence naturalness estimation based on masked language modeling likelihood
- **TemBERTure** — predicted thermal stability (Tm)
- **FoldX** — structure-based energetic evaluation

This hierarchical strategy reduces computational cost by applying expensive structural models only to the most promising candidates.

## Overall results

Applying a rigorous multi‑model ensemble — spanning evolutionary fitness (ESM, SaProt), structural energetics (FoldX), stability predictions (TemBERTure), and designability (ProteinMPNN) — we systematically reduced the initial library of 500 mutations to a shortlist of 10 exceptionally promising variants. From this refined set, 6 candidates were ultimately selected as the optimal balance of thermodynamic stability, structural integrity, and enhanced fluorescence.

### Selected candidates:

| Rank | ID  | Mutations                                        | Tm    | FoldX Energy | FoldX Score | Risk      |
| ---- | --- | ------------------------------------------------ | ----- | ------------ | ----------- | --------- |
| 1    | 32  | T61A; E121V; I164S; E219I                        | 49.06 | 2.28         | 0.763       | Low       |
| 2    | 5   | V21C; H24E; T48I; V54S; R77L; D99G; Q154G; V216M | 49.32 | 5.67         | 0.551       | CORE muts |
| 4    | 20  | T37S; D79I                                       | 48.72 | 5.60         | 0.493       | Mitigated |
| 5    | 3   | T105L; E169H                                     | 47.80 | -1.25        | 0.795       | Mitigated |
| 7    | 36  | L6C; N22A; K44T; S199P                           | 49.13 | 4.76         | 0.617       | Mitigated |
| 9    | 14  | P89K;K126S;H139E;S147Q;Y151L;N185S;K209Y;H231A   | 48.37 | 3.36         | 0.809       | CORE muts |
### Candidate selection visualization via Pareto front

![[pareto-candidates.png]]
