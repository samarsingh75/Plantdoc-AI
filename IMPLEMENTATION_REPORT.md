# Implementation report

Date of work: 2026-10-29. Environment: Linux sandbox, Python 3.12.3, 1 vCPU, 3 GB RAM, no GPU.
Network: PyPI and GitHub reachable; Kaggle, Mendeley, wfd.sysbio.ru and the Keras ImageNet weight host were **not** reachable.

## 1. Inspection of the supplied ZIP (what existed)

| Item | Finding |
|---|---|
| Model | custom 4-block CNN, **38 classes, 128×128**, `plant_disease_model.h5` (16 MB) at repo root; no scaling layer |
| Code | `src/{model,preprocess,predict,train}.py`, `app/app.py`, one notebook. The class list was copy-pasted into three files |
| Streamlit app | upload → button → label + confidence. Default `MODEL_PATH` (`../saved_model/…`) did not match where the model file actually sits; no input validation; no tests; no Gemini |
| Broken/fragile | `tf.keras.optimizers.legacy.Adam` does not exist in Keras 3; `requirements.txt` pinned TF 2.13 (no Python 3.12 wheels) and Kaggle; `.gitignore` ignored `data/` only; `src` scripts import each other by bare name (only works from inside `src/`) |
| README | claims 94.99 % validation accuracy. Several table rows are mathematically impossible (e.g. Corn Northern Leaf Blight P 0.90 / R 0.28 / F1 0.24; Grape Black rot P 0.90 / R 0.97 / F1 0.37; Potato Late blight P 0.30 / R 0.98 / F1 0.95), "Soybean support 5090" is the whole class not a validation split, and the data were an offline-**augmented** Kaggle copy, so train/validation leakage is likely. These numbers were **not** reused |
| Reusable | the 38 class names/order (verified identical to the real PlantVillage folders and sorted order), the general idea and Streamlit skeleton |

The legacy model cannot be extended to 46 classes without retraining; it was moved to `legacy/` and the new code refuses to load it.

## 2. What was built (all new unless noted)

`app.py` (replaces `app/app.py`) · `src/` {`config`, `labels`, `image_utils`, `discovery`, `data_download`, `preprocess`, `dataset_validation`, `model`, `train`, `evaluate`, `predict`, `chat_assistant`} ·
`configs/{class_mapping,dataset_sources,train_config}.json` · `tests/` (5 files, 112 tests) · `requirements.txt` · `.env.example` · `.gitignore` · `pytest.ini` · `README.md` · `scripts/*.ps1` · `legacy/` (original files).

Notable decisions: one authoritative mapping with a SHA-256 of the label order stored in model metadata; preprocessing scaling lives *inside* the model; wheat/rice sources are excluded until the user confirms their labels;
training refuses to start with missing classes unless explicitly allowed; Gemini state logic is Streamlit-independent so it is unit-testable.

## 3. Datasets

| Source | Status | Verified facts |
|---|---|---|
| PlantVillage (colour) | **Downloaded and verified** via pinned git checkout (commit `7f7ecc7e…`, 2026-02-05) | 54,305 files, all JPG/JPEG/PNG, 38 class folders identical to the 38 mapping keys. README of the source says 54,306 (off by one; not investigated). **No LICENSE file** in the repository |
| Wheat Fungi Diseases | **Not obtained** — site unreachable; needs manual download | labels, counts, licence **unverified** |
| Rice (Mendeley n8jpkpfz5p v1) | **Not obtained** — site unreachable; needs manual download | labels, counts, licence **unverified** |

Preprocessing run on the full PlantVillage set (measured): 54,305 → 21 exact duplicates removed → **54,284 usable**, 0 corrupt, 2 near-duplicate groups (4 images), 0 label conflicts;
split train 37,998 / val 8,143 / test 8,143; class imbalance 36.2× (max 5,507 Orange HLB; min 152 Potato healthy, test n = 23). Dataset validation: PASSED (no leakage, all sizes 224×224).
Processed output ≈ 1.4 GB (not shipped).

**Missing classes (8): Wheat healthy, leaf rust, stripe/yellow rust, powdery mildew; Rice healthy, blast, bacterial leaf blight, brown spot.**
To complete: download the two datasets, `ingest`, `review-labels`, confirm the mapping (README §6), re-run preprocess and train. If a dataset lacks one of these classes, it stays missing; nothing is synthesised.

A defect found and fixed during verification: the first perceptual hash (64-bit) merged ~25 % of PlantVillage into false "near-duplicate" groups (largest 10,866 images; 769 cross-class groups).
It was replaced by a 256-bit hash plus a group-size cap; on the real data this now yields 2 groups and no cross-class groups.

## 4. Test results (actually executed)

* `python -m pytest -m "not live"` → **111 passed**, 0 failed (≈ 9 s). The 1 live Gemini test is **skipped** (no key).
* 14 integration tests (TensorFlow model build/save/load/predict; headless Streamlit `AppTest`) and 97 unit tests passed.
* Verified by tests: 46-class mapping; label↔index; source-label mapping; corrupt/truncated/tiny/oversized images; 224×224 float32 0–255 preprocessing; PIL-vs-TensorFlow decode consistency (≤ 4 grey levels);
  split proportions, determinism, group integrity; leakage and missing-file detection; safe extraction; 46-way prediction, low-confidence/ambiguity warnings; refusal of mismatched/legacy models;
  Gemini context + language in every request; history kept; language switch keeps chat; new image resets chat; no calls on reruns; missing key; all error classes; bounded retries; empty replies; key never leaked;
  app dropdown (13 options, English default); CNN works with no key; Gemini failure shows an error with Retry and never changes the CNN result.
* Other executed checks: real downloader (sample and full); full preprocessing + validation on 54,284 real images; `train → evaluate` smoke run on 456 real images; Streamlit server boot (`/_stcore/health` = 200, empty log); CLI behaviour for missing classes and manual sources; all modules compile and import.

## 5. Training status — **not completed**

* A real training run was **not performed**. The ImageNet weights could not be downloaded (verified: the app/trainer reports a clear error and never silently uses random weights), the wheat/rice data were unavailable, and the sandbox has one CPU core.
* A **pipeline smoke test** (random weights, 304 training images, 3 epochs) ran end to end and produced chance-level accuracy (val 2.6 % ≈ 1/38), as expected; it only proves the mechanics (training, checkpointing, metadata, evaluation, reports). Its artefacts were **not** shipped.
* **No evaluation metrics exist for a real model. None are reported.**

## 6. Gemini integration status

* Implemented with the official `google-genai` SDK (installed 2.33.0), default model `gemini-3.5-flash-lite`, selected after reading Google's official model page (updated 2026-10-22).
* **Tested only with mocks/fakes. No live call was made (no API key, and the sandbox cannot be assumed to reach the API). Live behaviour, answer quality and fluency in each of the 12 Indian languages are unverified.**
* The one live test exists (`-m live`, `RUN_LIVE_TESTS=1`) and was skipped.

## 7. Checklist from the brief (§18)

| Check | Result |
|---|---|
| Repository consistent | done; legacy files isolated |
| 46-class mapping validated | tests + runtime validation |
| Syntax/imports | `compileall` + import of all modules OK |
| Automated tests | 111 passed |
| Preprocessing on real images | full PlantVillage OK |
| Training pipeline initialises and runs | yes (smoke, random weights); real training not run |
| Prediction preprocessing consistency | tested |
| Streamlit entry point | boots; AppTest passes |
| Dropdown has 13 languages; switching keeps chat | tested |
| Mocked Gemini + errors; CNN without credentials | tested |
| `.gitignore` | covers `.env`, venvs, caches, data, weights (verified with `git status` on a scratch repo) |
| **Not performed** | real training/evaluation, live Gemini, Windows/PowerShell execution, GPU, wheat/rice acquisition, Python 3.10/3.11 |

## 8. Remaining limitations

See README §12. Most important: no trained model or wheat/rice data are included; residual same-leaf leakage in PlantVillage; lab-vs-field domain gap and dataset-shortcut risk when wheat/rice are added; unreliable metrics for rare classes; uncalibrated confidence and no "unknown" class; unverified licences for wheat/rice sources and none stated for PlantVillage's repository.
