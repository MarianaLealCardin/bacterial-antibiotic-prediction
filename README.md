# Predicting bacterial lineage and antimicrobial resistance from whole-genome sequences

A bioinformatics + machine learning pipeline that, given a bacterium's complete genome, predicts **(1) its species, (2) which drug classes it is resistant to, and (3) its evolutionary lineage (Sequence Type)**.

MSc thesis project · MSc in Data Science, Big Data & Business Analytics · Complutense University of Madrid (2026)
Author: Mariana Leal Cardín · [LinkedIn](www.linkedin.com/in/mariana-leal-cardín-759258112)

> The notebook shows the full pipeline and its results end-to-end. Re-running it requires downloading the genomes yourself (see "How to reproduce it" below) — no shared data folder is provided.

## Summary

- **Data:** 1,500 complete genomes from NCBI (500 *Staphylococcus aureus*, 500 *Klebsiella pneumoniae*, 500 *Escherichia coli*).
- **Resistance:** resistance gene detection with Abricate + the CARD database, and a multi-label Random Forest over 20 drug classes.
- **Lineage:** MLST typing and a per-species Random Forest classifier.
- **Features:** canonical k-mer frequencies (k = 6, 2,080 variables) computed on the full genome.

The notebook itself is kept close to the code: each phase has a short header, and every analytical decision, result and discussion below lives in this README instead, so the notebook stays quick to scan and this file stays the single source of the narrative.

## Results

**Resistance (multi-label, 20 drug classes, 80/20 split)**

| Model | F1 macro | F1 micro |
|---|---|---|
| Random Forest on k-mers | 0.700 | 0.786 |
| Control baseline (species only) | 0.510 | – |

The k-mer model beats the species-only baseline on all 20 classes (by +0.05 to +0.43 depending on the class), so it learns real genotypic signal rather than just recognising the species. Classes with more support score better (aminoglycoside 0.903, sulfonamide 0.875, diaminopyrimidine 0.840); the rarer ones score worse (rifamycin 0.345, phosphonic_acid 0.348, peptide/nonribosomal_peptides 0.459) — a data-quantity limitation rather than a weakness of the k-mer approach itself. Where the species-only baseline already did reasonably well on its own (azole/imidazole 0.700, fluoroquinolone 0.667), the k-mer model's relative improvement is smaller, consistent with those resistances being tied more to the species' general biology than to strain-level variation.

**Lineage (MLST, per-species Random Forest)**

| Species | Samples | Classes | F1 macro (80/20 split) | F1 macro (5-fold CV) |
|---|---|---|---|---|
| *S. aureus* | 486 | 10 | 0.877 | 0.913 ± 0.022 |
| *E. coli* | 476 | 10 | 0.812 | 0.791 ± 0.040 |
| *K. pneumoniae* | 403 | 12 | 0.701 | 0.709 ± 0.105 |

STs with fewer than 10 samples are grouped into an "other" class. *S. aureus* is the strongest and most stable case: most lineages reach near-perfect F1, and the two weaker ones (ST1, ST9) have only 2-4 test samples, more likely a small-split artefact than a real limitation. *K. pneumoniae* is the weakest and least stable case, with high precision but low recall on several lineages (ST15 0.33, ST147 0.57, ST340/ST258/ST48 0.50) — the model is usually right when it predicts one of these classes, but misses many real cases, folding them into "other" (recall 0.97 there). Cross-validation confirms this is a real instability, not one unlucky split: *K. pneumoniae*'s standard deviation (± 0.105) is nearly five times that of *S. aureus* (± 0.022), consistent with several lineages having only 2-6 samples, so which samples land in which fold matters a lot.

## Pipeline

| Phase | What happens | Tools |
|---|---|---|
| 0. Environment | Setup and data folder configuration | Google Colab / conda |
| 1. Data | Metadata and parallel download of complete genomes; retry of failed accessions and cleanup | NCBI `datasets` |
| 2. Resistance genes | ARG detection per genome, consolidated into a single table (52,068 hits) | Abricate + CARD |
| 3. EDA | Summary statistics and per-species distribution of ARG genes | pandas, seaborn |
| 4. Resistance labelling | Exclusion of intrinsic ("core") genes and a multi-label target by drug class | pandas |
| 5. Lineage | Sequence Type assignment and grouping into workable classes | `mlst` |
| 6. Modelling | k-mers, Random Forest, baseline, cross-validation, error analysis, prediction function | scikit-learn |

**Phase 3 (EDA) findings.** No sample came back with zero ARG hits, confirming Abricate ran correctly on the whole set. %GC and genome length vary widely, as expected across three species with different genomic profiles, and the per-species %GC values match published reference ranges, confirming there was no cross-species mix-up in the extraction. A first, keyword-based resistant/susceptible split (built from the assembly title, e.g. "MRSA", "ESBL") showed almost no difference between groups in ARG count, %GC or genome length (34.78 vs 34.65 genes; 47.23% vs 47.37% GC; 44,021 vs 44,121 bp) — a first sign that this metadata-based target wasn't capturing a real biological split, which is why Phase 4 builds the target from the genotype itself instead.

## Methodological decisions

- **Intrinsic genes.** A 95% presence threshold left several clearly intrinsic genes (oqxA, oqxB, ramA, mdtM, FosA6) just below the "core" cut-off (~90% presence in *K. pneumoniae* / *E. coli*), while acquired-resistance genes such as qacEdelta1, sul1 and TEM-1 sat much lower (27-57%). Plotting the full presence distribution (not just the top 20 genes) showed a clean gap between 65% (emrE) and 57.4% (qacEdelta1), with the genes in that band all consistent with known intrinsic multidrug-efflux systems (see references below). The threshold was set at **60%**, just inside that gap.
- **From a binary to a multi-label target.** Re-running the binary target ("has at least one acquired-resistance gene?") after the threshold fix barely moved the dominant classes, and it stayed heavily imbalanced (1,421 positive vs. 79 negative). The genes actually driving this weren't misclassified core genes — they were real acquired-resistance genes, individually rare, whose union across many genes ends up covering almost the whole sample. This pattern likely mirrors a real sampling bias in which genomes get deposited in NCBI (more resistant strains). The target was therefore reframed around the original goal — resistance to specific drug classes rather than a single yes/no label — after checking the prevalence of each drug class. Three bands emerged: too dominant (>75%, little discriminative signal), too rare (<2%, too few positives to train on), and a workable range (~5-75%), which produced the final **20-class multi-label target**.
- **MLST typing issues, found and fixed.** Assigning STs initially dropped 15 of the 1,500 genomes and produced a few duplicated-looking ST numbers (8, 5, 10, 147, 14464). The 15 missing genomes came from an off-by-one bug: the parsing code assumed the `mlst` output had a header row (`for line in lines[1:]`) when it didn't, silently discarding the first sample of every batch; fixing the loop recovered all 15. The duplicated STs turned out not to be duplicates at all: ST numbering is species-specific (ST8 in *S. aureus* and ST8 in *E. coli* are unrelated lineages that happen to share a number), and grouping by `(mlst_scheme, ST)` instead of `ST` alone removed the apparent collision. A deeper issue then surfaced: `mlst` assigns whichever typing scheme best matches a genome's housekeeping genes, independent of the species label in the metadata, so 5 schemes appeared for only 3 declared species. Cross-checking the detected scheme against the declared species and dropping non-typeable ("-") STs left 1,365 of the original 1,500 samples (91%).
- **Data leakage, found and discarded.** A model using the ARG presence/absence matrix as features outperformed the k-mer model on all 20 drug classes, in some cases by a very large margin (rifamycin +0.582, phosphonic_acid +0.566, glycopeptide +0.517 with F1 = 1.000) — too clean and one-directional a pattern to be genuine signal. The cause: several of those same genes were used in Phase 4 to build the resistance target itself, so the model was effectively given a disguised copy of the label it had to predict. This comparison was kept only as an internal consistency check (it confirms the Phase 4 labelling matches the underlying gene presence data), not as evidence for which feature representation predicts resistance better; the k-mer model (F1 macro 0.700) remains the reported resistance model.
- **Error type and PR-AUC.** In 19 of the 20 drug classes, false negatives outnumber false positives, in some cases by a wide margin (streptogramin 21 FN vs. 2 FP; lincosamide 19 FN vs. 1 FP; phosphonic_acid 15 FN vs. 0 FP) — only aminoglycoside is roughly balanced. In practice: when the model predicts "resistant" it is usually right, but it misses a fair share of real resistant samples, which is the more dangerous error clinically. PR-AUC (more informative than ROC-AUC under this class imbalance) needs to be read relative to each class's own prevalence rather than compared directly across classes, since a rare class yields a low PR-AUC even for a near-random model. Read that way, classes that looked weakest by F1 (rifamycin, phosphonic_acid, glycopeptide) are actually where the model moves furthest from chance — the model isn't ignoring rare classes, it simply has very little data to learn them from.

**References** (intrinsic efflux genes):
Li et al. (2019), *Antimicrobial Resistance and Infection Control* 8. https://doi.org/10.1186/s13756-019-0489-3 · Paul et al. (2014), *Molecular Microbiology* 92. https://doi.org/10.1111/mmi.12597 · Rozwandowicz et al. (2018), *J. Antimicrob. Chemother.* 73. https://doi.org/10.1093/jac/dkx488 · Vinué et al. (2010), *Int. J. Antimicrob. Agents* 35. https://doi.org/10.1016/j.ijantimicag.2010.01.012

## Limitations

- **Genotypic labels.** The resistance target is derived from gene presence in CARD, not from phenotypic antibiograms.
- **Sampling bias.** The first 500 complete genomes NCBI returns per species were used, not a random sample; deposited genomes may be biased toward resistant strains (see EDA and the multi-label discussion above).
- **Small classes.** Rare drug classes (resistance) and low-count STs (lineage) limit how reliable the corresponding metrics are.
- **Data reproducibility.** NCBI is updated continuously, so a fresh download can return different accessions. The accession list actually used is in `data/`.

## Repository structure

```
.
├── README.md
├── environment.yml
├── Predicting bacterial lineage and antimicrobial resistance from whole-genome sequences.ipynb   # full pipeline, phases 0-6
└── data/
    ├── metadata_clean.csv                          # accessions for the 1,500 genomes
    ├── metadata_target_resistencia_multilabel.csv  # resistance target (20 classes)
    └── metadata_target_linaje.csv                  # ST and lineage class per sample
```

The genomes (~1,500 FASTA files, 3-6 Mb each) and the feature matrices are not included due to size; they are regenerated by running phases 1 and 6.

## How to reproduce it

```bash
conda env create -f environment.yml
conda activate tfm-amr
abricate-get_db --db card --force
jupyter lab
```

`DATA_DIR` defaults to `./data`, relative to the notebook; point it at a different folder if you prefer. You'll need a few GB of free disk space for the genomes.

The notebook also runs on Google Colab if you'd rather not install anything locally: upload it, install the same tools listed in `environment.yml` (the `!pip`/`!mamba` install commands are already in the notebook), and set `DATA_DIR` to a folder of your choice — for example on your own Google Drive, which you'd mount yourself with `from google.colab import drive; drive.mount('/content/drive')`.


