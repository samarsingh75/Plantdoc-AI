# Plant Disease Detection — 46-class CNN + Streamlit + multilingual Gemini explainer

A beginner-friendly, Windows-friendly project that

1. classifies a leaf photo into **one of 46 classes** (38 PlantVillage + 4 wheat + 4 rice) with a transfer-learning CNN (MobileNetV2 by default, EfficientNetB0 optional), and
2. lets the user ask **Google Gemini** questions *about that specific prediction* in one of **13 languages** (English + 12 Indian languages).

> **Status of this delivery (please read).** The code, data pipeline, tests and app are complete and were verified
> (see [Verification status](#verification-status)). **No trained 46-class model is included**: the build
> environment had no access to the ImageNet weight host and wheat/rice datasets, and only 1 CPU core, so a real
> training run could not be performed. Every accuracy number in this project is therefore *yours to measure* by
> running the commands below; none are claimed here. The wheat and rice datasets must be downloaded by you
> (they require visiting the dataset sites; see [Wheat and rice data](#wheat-and-rice-data)).

---

## 1. How it works

```
 leaf photo ──► validate ──► CNN (46 classes) ──► label + confidence ──► shown to the user FIRST
                                                        │
                      user clicks "Ask Gemini about this result" / types a question
                                                        ▼
        structured CNN result (text only) + selected language ──► Gemini ──► explanation
```

* **The CNN is the only source of the label and confidence.** Gemini receives them as text context, is instructed
  never to change them, to separate "the model predicted…" from its own general explanation, and to say it did **not**
  see the image or verify anything. The photo is **never** sent to Gemini.
* Prediction works with **no internet and no API key**. Gemini is only called when you press the button or send a message.
* If no trained model exists the app shows training instructions; it never fakes a prediction.

## 2. The 46 classes

Single source of truth: [`configs/class_mapping.json`](configs/class_mapping.json). Output index = position in that file.
Indices 0–37 keep the alphabetical order of the originally supplied 38-class model.

| Indices | Group | Classes |
|---|---|---|
| 0–37 | PlantVillage (38) | Apple (Scab, Black rot, Cedar apple rust, Healthy) · Blueberry (Healthy) · Cherry (Powdery mildew, Healthy) · Corn (Cercospora/Gray leaf spot, Common rust, Northern leaf blight, Healthy) · Grape (Black rot, Esca, Leaf blight, Healthy) · Orange (Huanglongbing) · Peach (Bacterial spot, Healthy) · Pepper bell (Bacterial spot, Healthy) · Potato (Early blight, Late blight, Healthy) · Raspberry (Healthy) · Soybean (Healthy) · Squash (Powdery mildew) · Strawberry (Leaf scorch, Healthy) · Tomato (Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy) |
| 38–41 | Wheat (4) | Healthy · Leaf rust · Stripe rust (Yellow rust) · Powdery mildew |
| 42–45 | Rice (4) | Healthy · Blast · Bacterial leaf blight · Brown spot |

The 38 PlantVillage names are verbatim the folder names of the original dataset (checked against the 38 folders of the
pinned GitHub snapshot), including its quirks (`Pepper,_bell___…`, `…Gray_leaf_spot` with a space). Nothing was dropped or renamed;
only the display text was tidied (e.g. "Huanglongbing").

Wheat/rice source folders are matched to classes through explicit alias lists in the same file. "Rust", "Blight",
"Spot" and similar generic labels are deliberately **ambiguous → excluded**, and unsupported diseases (Septoria,
Tungro, …) are excluded, never relabelled. "Brown rust = leaf rust" and "yellow rust = stripe rust" are only applied after
**you** confirm them for your downloaded dataset (see below).

## 3. Project layout

```
app.py                      Streamlit app (entry point)
src/
  config.py                 paths, defaults, environment (.env) handling
  labels.py                 loads/validates the 46-class mapping (single source of truth)
  image_utils.py            the ONE image-preprocessing path shared by training and inference
  discovery.py              finds images / source labels in raw folders
  data_download.py          check / download / safely extract / record provenance
  preprocess.py             clean, de-duplicate, split, resize, report
  dataset_validation.py     gate before training (leakage, missing files, sizes, empty splits)
  model.py                  MobileNetV2 / EfficientNetB0 transfer-learning model
  train.py                  two-phase training, checkpointing, metadata
  evaluate.py               test-set metrics, confusion matrix, plots, report
  predict.py                load model, predict, low-confidence logic
  chat_assistant.py         languages, prompts, Gemini wrapper, chat-session state
configs/                    class_mapping.json · dataset_sources.json · train_config.json
data/{raw,interim,processed,reports}/   generated; git-ignored
models/                     generated: plant_disease_model.keras, model_metadata.json
tests/                      unit + integration tests (Gemini always mocked)
scripts/                    setup.ps1 · prepare_and_train.ps1 (PowerShell helpers)
legacy/                     the originally supplied files (not used)
```

## 4. Tested environment

| | |
|---|---|
| OS | Linux x86-64 (build sandbox), **1 vCPU, 3 GB RAM, no GPU** |
| Python | 3.12.3 (intended: 3.10–3.12; only 3.12.3 exercised) |
| tensorflow-cpu / keras | 2.21.0 / 3.15.1 |
| streamlit / google-genai / python-dotenv | 1.66.1 / 2.33.0 / 1.2.4 |
| numpy / Pillow / matplotlib / scikit-learn | 2.5.4 / 13.0.0 / 3.11.2 / 1.9.1 |

**Not tested on Windows.** The PowerShell commands below follow the code's actual file names and use only
cross-platform Python (pathlib, no symlinks, UTF-8 file I/O), but they were not executed in PowerShell by the author.
TensorFlow ≥ 2.11 has no native-Windows GPU support: on Windows training runs on the CPU (use WSL2 for a GPU).

## 5. Setup on Windows (PowerShell)

Install [Python 3.10–3.12](https://www.python.org/downloads/windows/) (tick "Add to PATH") and
[Git for Windows](https://git-scm.com/download/win) (used to fetch PlantVillage), then:

```powershell
cd path\to\plant-disease-detection
python --version                               # should print 3.10, 3.11 or 3.12
python -m venv .venv
.\.venv\Scripts\Activate.ps1                   # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env                    # then open .env and paste your key (section 9)
```

(`.\scripts\setup.ps1` does the same.) Quick sanity check, no data needed:

```powershell
python -m pytest -m "not live"                 # 111 tests (+1 live test that is skipped), ≈ 10 seconds
```

## 6. Get and prepare the data

### One command to check/download everything accessible

```powershell
python -m src.data_download all                # status + download PlantVillage + manual instructions for the rest
```

* **PlantVillage** (colour images, 38 classes, 54,305 files, ≈1 GB) is fetched with a *pinned, partial* `git` checkout
  from `github.com/spMohanty/PlantVillage-Dataset` (commit `7f7ecc7e…`), verified (class-folder count and image count) and
  recorded in `data/raw/provenance.json` (URL, access date, commit, licence note). A second run detects the existing data and
  downloads nothing. Add `--sample-per-class 50` for a ~10 MB trial.
  *Measured in the build environment: 54,305 images in 38 folders, 23 s.*
* **Licence:** the GitHub repository has **no LICENSE file**; it describes the data as open-access research data and asks for citation
  (Mohanty, Hughes, Salathé 2016, *Frontiers in Plant Science* 7:1419, doi:10.3389/fpls.2016.01419). Check the terms before redistributing.
* No `git`? Download PlantVillage yourself (e.g. Kaggle `abdallahalidev/plantvillage-dataset`, needs a login) and run
  `python -m src.data_download ingest --source plantvillage --archive C:\path\file.zip`. Grayscale/segmented folders are ignored.
  Avoid the Kaggle "*New Plant Diseases Dataset (Augmented)*": it contains augmented copies of the same photos (train/validation leakage).

### Wheat and rice data

These sources need a browser visit (licence/terms, possibly registration), so the tool **never auto-downloads them**:

| Source | Page | Notes |
|---|---|---|
| Wheat Fungi Diseases (WFD) | https://wfd.sysbio.ru/index.html | licence **unverified** here |
| Rice leaf disease images (Mendeley Data) | https://data.mendeley.com/datasets/n8jpkpfz5p/1 | licence **unverified** here |

Neither site was reachable from the build environment, so **their real class labels, image counts and licences were not verified.**
Procedure (also printed by `python -m src.data_download status`):

```powershell
# 1. download in the browser, then extract so images are inside label folders:  data\raw\rice_mendeley\<label>\*.jpg
python -m src.data_download ingest --source rice_mendeley --archive C:\Users\you\Downloads\rice.zip   # safe extract + SHA-256 + provenance
python -m src.data_download review-labels --source rice_mendeley      # shows every folder and what it maps to
# 2. compare with the dataset's documentation. If a folder is wrongly mapped/unmapped, edit "source_labels" in configs\class_mapping.json
python -m src.data_download confirm-labels --source rice_mendeley     # only now are its mapped folders used
# repeat for wheat_wfd
```

Until you confirm a source it is **excluded** from the dataset (reported in `data/reports/excluded_labels.csv`).
If a dataset has a class you need under a different name, add an alias; if it lacks a class (e.g. no healthy wheat), that
class stays **missing** — never fabricate or duplicate images to fill it. Classes without data are listed in the report, and
`python -m src.train` refuses to run unless you pass `--allow-missing-classes` (the app then says those classes are untrained).

### Preprocess and validate

```powershell
python -m src.preprocess              # measured ≈ 2.5 minutes for all of PlantVillage (1 vCPU, 2 threads)
python -m src.dataset_validation      # must print RESULT: PASSED
```

What it does: finds JPG/JPEG/PNG; fully decodes every image (corrupt files are *copied* to `data/interim/quarantine`, raw data is never
modified); maps labels to the 46 classes; removes exact duplicates (SHA-256; same bytes with different labels are dropped as label conflicts);
groups near-duplicates (256-bit perceptual hash that also recognises flips/rotations; threshold `--phash-threshold`, default 10/256) so
a group can never straddle train/val/test; makes a deterministic (seed 42) stratified group split ≈ 70/15/15; writes 224×224 JPEG copies to
`data/processed/{train,val,test}/<class>/` plus `manifest.csv` (the record of every split assignment); and writes `data/reports/`
(`dataset_report.md/json`, `class_distribution.png`, `failed_images.csv`, `duplicates_removed.csv`, `excluded_labels.csv`).
Augmentation happens only on training batches inside `train.py`.

**Measured on the full PlantVillage set (build environment):** 54,305 found → 21 exact duplicates removed → **54,284 usable images in 38 classes**
(train 37,998 · val 8,143 · test 8,143); 0 corrupt; 2 near-duplicate groups (4 images), 0 spanning two classes; largest/smallest class
ratio 36× (Orange Huanglongbing 5,507 vs Potato healthy 152, whose test split has only 23 images). Wheat/rice: **0 images (not yet downloaded)**.

## 7. Train and evaluate

```powershell
python -m src.train                                  # 'lightweight' preset (default)
python -m src.train --allow-missing-classes          # if wheat/rice data are not available yet
python -m src.train --preset standard                # all training images, more epochs
python -m src.train --weights none                   # no ImageNet download (poor accuracy; offline smoke tests)
python -m src.evaluate                               # held-out TEST set -> data\reports\evaluation_report.md
```

* **Model:** MobileNetV2 (ImageNet weights, downloaded once by Keras) → global average pooling → dropout → Dense(46, softmax); input scaling is *inside*
  the model, so every code path feeds raw 0–255 pixels. Phase 1 trains only the head; phase 2 fine-tunes the last backbone layers (BatchNorm frozen) at lr 1e-5.
  Adam, sparse categorical cross-entropy, class weights ("balanced", clipped) for the 36× imbalance, `ModelCheckpoint` on `val_accuracy`,
  `EarlyStopping`, `ReduceLROnPlateau`. Presets/parameters: `configs/train_config.json` and command-line flags (`--batch-size`, `--head-epochs`, …).
* **Student-laptop preset (`lightweight`)**: at most 300 *training* images per class (validation/test are never subsampled), 4+2 epochs. The metadata records
  that the training set was subsampled. Throughput measured on the 1-vCPU build box: ≈ 16–20 images/s for MobileNetV2 training, so expect roughly
  10–15 minutes per epoch on a comparable single core and much less on a modern 4–8-core laptop; use `--preset standard` for best results.
* Outputs: `models\plant_disease_model.keras`, `models\model_metadata.json` (ordered class names + SHA-256 of the label order, preprocessing, data counts,
  untrained classes, validation score, software versions), `models\training_history.json`. A model whose class order or output size does not match
  `class_mapping.json` is **refused** at load time (this is why the legacy 38-class model cannot be used).
* `evaluate` reports accuracy, macro/weighted precision-recall-F1, per-class report, confusion matrix (PNG+CSV), accuracy per data source,
  training/validation accuracy and loss curves, class distribution, and writes the measured test numbers into the model metadata. It prints warnings for tiny test classes.

**How to read the numbers.** They depend on dataset quality, label noise, class balance, image diversity and how similar the test images are to the training
images. PlantVillage photos are single leaves on plain backgrounds, so random-split accuracy is typically far higher than what you will get on field photos,
and PlantVillage contains several photos of the same physical leaf that no pixel-based check can group, so some leakage between splits remains.
Mixing PlantVillage with differently shot wheat/rice images also lets a network cheat by recognising the *background/source*; `evaluate` prints accuracy per source
to help you notice this. This is an educational model, **not production-grade**.

## 8. Launch the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py                 # opens http://localhost:8501
```

* No model yet → red message with the training commands (no fake prediction).
* Upload JPG/PNG (≤ 10 MB; validated by actually decoding it) → image, plant, condition, confidence, top-3, and warnings when confidence is low (default < 60 %),
  the top two classes are close (< 15 points), or the class had no training data. Inference runs once per distinct image (kept in session state).
  Thresholds: `LOW_CONFIDENCE_THRESHOLD`, `MIN_TOP2_MARGIN` in `.env`. The app cannot reliably detect pictures of something *outside* the 46 classes — the model has no
  "other" class — and says so.

## 9. Gemini setup

1. Create a key at <https://aistudio.google.com/apikey>.
2. Put it in `.env` (never commit this file; `.gitignore` excludes it):
   ```
   GEMINI_API_KEY=paste-your-key
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```
3. Restart the app. Without a key the CNN still works and the chat shows setup instructions.

`GEMINI_MODEL` default **`gemini-3.5-flash-lite`** was checked against Google's official model list (page dated 2026-10-22; viewed 2026-10-29), which
recommends it or `gemini-3.8-flash` for new projects and notes that the 2.5 family is now restricted to existing users. Model names change — if you get
a "model not found" message, pick a current one from <https://ai.google.dev/gemini-api/docs/models>. Uses the official `google-genai` SDK, a 30 s timeout and at most 2
retries (rate limits, 5xx, timeouts, connection errors only; invalid key / unknown model are not retried). Handled with friendly messages: missing/invalid key, network
error, rate limit/quota, timeout, unsupported model, empty or blocked response. Error text never contains the key.

### Languages

Dropdown **"Select Chatbot Language"** (sidebar), English by default: English · Hindi — हिन्दी · Punjabi — ਪੰਜਾਬੀ · Bengali — বাংলা · Gujarati — ગુજરાતી · Marathi — मराठी ·
Tamil — தமிழ் · Telugu — తెలుగు · Kannada — ಕನ್ನಡ · Malayalam — മലയാളം · Urdu — اردو · Odia — ଓଡ଼ିଆ · Assamese — অসমীয়া. The selected language and its script
(Devanagari for Hindi/Marathi, Gurmukhi for Punjabi, …) are sent to Gemini as an explicit instruction on **every** request, overriding the language of the question; plant and
disease names are kept accurate with the English name in brackets. Changing the language never clears the chat; only later answers use the new language. Gemini's fluency in each language
is **not guaranteed** (and was not evaluated here); the UI works regardless of answer quality.

### How prediction-specific chat works

* A new image ⇒ a new prediction ⇒ a **fresh chat** bound to that prediction's context (plant, condition, class id, confidence, top-3, warning flags). Old messages are discarded so they cannot contaminate it.
* The same image across Streamlit reruns ⇒ nothing changes; the history is kept in `st.session_state`.
* Gemini is called **only** on "Ask Gemini about this result", on sending a chat message, or "Retry". Not on reruns, language changes or uploads. Empty messages are ignored.
* The last 20 messages are sent with each request. "Start new chat" clears the history but keeps the prediction.
* If a call fails, the history is untouched, the error is shown with a Retry button, and the CNN result stays visible.

## 10. Tests

```powershell
python -m pytest -m "not live"          # everything except the live API test
python -m pytest -m unit                # fast, no TensorFlow
python -m pytest -m integration         # builds a tiny random MobileNetV2; runs the Streamlit app headlessly (AppTest)
$env:RUN_LIVE_TESTS="1"; python -m pytest -m live    # optional; uses your real key and quota
```

Covered: 46 unique classes and index↔label mapping · source-label mapping (aliases, ambiguous, unsupported) · corrupt/invalid/truncated images · preprocessing size and
0–255 range · training-vs-inference decode consistency · near-duplicate hashing (flip, rotation, JPEG, brightness, chaining cap) · stratified, group-safe, deterministic split ·
leakage detection · safe archive extraction (path traversal, corrupt zip) · prediction with 46 outputs and low-confidence handling · model/metadata mismatch refusal ·
Gemini context, selected language, history, language switch keeps chat, new prediction resets chat, no calls on reruns, missing key, every error type, retries, empty responses, key never leaked ·
app: dropdown contents, CNN works without a key, Gemini failure does not crash. **All Gemini calls in automated tests are mocked.**

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 … running scripts is disabled` | `Set-ExecutionPolicy -Scope Process Bypass` then activate again |
| `No trained model was found` | run the training commands in §6–7 |
| `Could not load pretrained 'imagenet' weights` | no internet/firewall; retry, or `--weights none` (much lower accuracy) |
| `git is not installed` | install Git for Windows, or use the manual `ingest` route |
| `ERROR: these classes have no images` | download wheat/rice (§6) or pass `--allow-missing-classes` |
| Out of memory while training | `--batch-size 8`, keep `lightweight` |
| `Gemini is not configured` | set `GEMINI_API_KEY` in `.env`, restart |
| `model was not found` (Gemini) | set a current `GEMINI_MODEL` (§9) |
| `quota` / `rate limit` | wait, or use a key with quota |
| `The model's saved class order does not match` | you changed `class_mapping.json` after training; restore it or retrain |

## 12. Known limitations

* **No trained model is shipped; wheat and rice classes have no data until you download it.** Without it the model cannot predict those 8 classes.
* Wheat/rice dataset labels, sizes and licences are unverified (sites unreachable from the build environment).
* Dataset bias (lab-style PlantVillage images vs. field photos), residual same-leaf leakage in PlantVillage, small evaluation sets for rare classes (e.g. Potato healthy: 23 test images) and
  the 36× class imbalance limit how far any metric can be trusted.
* Softmax confidence is uncalibrated; the model has no "unknown" class and will label non-leaf images.
* The aspect ratio is not preserved when resizing (squash, same in training and inference).
* Live Gemini behaviour (including answer quality per language) was **not** tested; only mocked.
* Windows execution and GPU training were not tested.

## 13. Responsible use

The CNN produces an **estimate** based on its training data. An image classification is **not** a guaranteed agricultural diagnosis, and Gemini provides general explanatory information,
**not** laboratory-confirmed advice. For severe, uncertain or economically significant crop problems consult a qualified agricultural expert or your local extension service.
No treatment is guaranteed to cure a disease. The assistant is instructed not to give pesticide approvals, doses, application rates or legal requirements; if chemicals are considered, follow
locally approved product labels and qualified guidance.

## 14. Reproduce everything (summary)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python -m pytest -m "not live"
python -m src.data_download all
# (download + confirm wheat/rice, §6)
python -m src.preprocess; python -m src.dataset_validation
python -m src.train                  # add --allow-missing-classes if wheat/rice are absent
python -m src.evaluate
streamlit run app.py
```
