# Confidence-Aware Hierarchical Classification of Breast Ultrasound Lesions

BIL539 Artificial Intelligence, Term Project, M2 (Progress Report 1)
Batuhan Şimşek, batuhansimsek@ogr.eskisehir.edu.tr, Eskişehir Technical University

> **Research prototype only. Not a diagnostic tool and not for clinical use.**

## What this is

A two-stage (hierarchical) classifier for 224×224 breast ultrasound lesion images:

1. **Stage 1:** benign vs. malignant (all labeled images).
2. **Stage 2:** subtype within the predicted branch, giving four leaves:
   fibroadenoma (FA), other benign (OB), invasive breast carcinoma of no special type (IDC-NST), other malignant (OM).
   Images without a subtype label are used for stage 1 only and are never treated as "other".

Primary metric: four-leaf macro-F1 on the subtype-labeled test subset (end-to-end cascade).

## Repository contents

| File | Description |
|---|---|
| `M2.ipynb` | Full pipeline with saved outputs: split, baselines, ResNet-18 models, threshold search, metrics |
| `M2.pdf` / `M2.html` | M2 progress report |
| `requirements.txt` | Python dependencies |

## Data

BUS-CoT, BUS-Lesion subset (Yu et al., 2026): https://doi.org/10.6084/m9.figshare.30838715
(license: CC BY 4.0, as stated on the Figshare record; please check the record for current terms).
**The data are not included in this repository.** Download the package and keep it as `BUSCoT.zip`.

## How to run (Google Colab, GPU runtime)

1. Upload `BUSCoT.zip` to the root of your Google Drive (`MyDrive`).
2. Open `M2.ipynb` in Colab and select a GPU runtime (the reported run used a Tesla T4).
3. Run all cells top to bottom (Runtime → Run all). Run order:
   `SETUP → SPLIT → EVAL → B0a/B0b → B1 → B2 → B2s → threshold search → leaderboard → metrics`
4. Outputs (split file, predictions, `results.csv`, audits, training histories, caches) are written to
   `MyDrive/bus-hier-work/`.

The SPLIT cell builds the frozen split on first run and asserts the published counts
(5,156 images; 3,476 / 859 / 821 train / val / test; 2,401 four-leaf images) before anything trains,
so a mismatch raises an error instead of producing incomparable numbers.
Approximate runtime on a T4: B1 features ~2 min plus SVM grids ~3.5 min; each ResNet-18 run ~8–9 min.

The saved outputs in `M2.ipynb` are from the reported run; the notebook was not fully re-executed while editing the report.

## Setup details

- Split: patient/hash-grouped, `StratifiedGroupKFold` 80/20, `random_state=42`; model seed 99.
- Selection of checkpoints, SVM grids and thresholds uses train/validation only.
- Test metrics: macro-F1 with 95% patient/hash-grouped bootstrap CIs (2,000 resamples); model differences use paired bootstrap.
- Results are exploratory: the frozen test set was examined for several model variants.

## Models

| ID | Description |
|---|---|
| B0a | Majority class at every stage |
| B0b | Majority class per acquisition source (diagnostic only; needs source metadata) |
| B1 | HOG + GLCM + LBP (1,346 features) + class-balanced RBF-SVM |
| B2 | ImageNet-pretrained ResNet-18, shared backbone, three masked binary heads |
| B2s | Dedicated stage-1 ResNet-18; stage 2 reused from B2 |
| B2st | B2s with a validation-selected stage-1 threshold |
| CascadeThreshold | B2s stage 1 + B2 stage 2, three validation-selected thresholds (0.40 / 0.55 / 0.60); no retraining |

## Main results (test, seed 99)

| Model | Stage 1 F1 | Stage 2a F1 | Stage 2b F1 | Cascade F1 |
|---|---|---|---|---|
| B0a majority | 0.3520 | 0.3848 | 0.3750 | 0.1457 |
| B0b source-only | 0.6002 | 0.7534 | 0.8344 | 0.3169 |
| B1 HOG/GLCM/LBP + SVM | 0.7320 | 0.6998 | 0.7220 | 0.5295 |
| B2 shared ResNet-18 | 0.7832 | 0.7868 | 0.8142 | 0.6021 |
| CascadeThreshold | 0.7997 | 0.8015 | 0.8138 | 0.6393 |

CascadeThreshold − B2 on the cascade is +0.0372 (95% CI −0.0090 to 0.0828), so the gain is inconclusive.
B0b reaches 0.8344 on stage 2b without looking at pixels, so source/label association is a material limitation.
See the report for the full tables.

## Limitations

Single public multi-source dataset, small subtype test sets (135 cases for stage 2b), one model seed,
repeated use of the test set across variants, no external-site validation.
Calibration, risk-coverage analysis and Grad-CAM are planned for M3.

## Ethics

No new patient data were collected. Images are not redistributed here.
Risks: source and demographic bias, over-trust in an imperfect second reader, false reassurance from uncalibrated scores.

## References

Yu, H., Li, Y., et al. (2026). A chain-of-thought reasoning breast ultrasound dataset covering all histopathology categories.
*Scientific Data, 13*, Article 370. https://doi.org/10.1038/s41597-026-06702-9
