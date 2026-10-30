# Legacy files (kept for reference only - NOT used by the new application)

These are the files from the originally supplied repository.

* `plant_disease_model_38class_128px.h5` - the original 38-class, 128x128 custom CNN. It cannot be used by the
  46-class application (different input size, no wheat/rice outputs, no scaling layer). `src/predict.py` refuses
  to load a model whose output size or class order does not match `configs/class_mapping.json`.
* `original_src/`, `original_app/`, `plant_disease_detection.ipynb` - the original scripts and notebook.
* `README_original.md` - the original README. Several rows of its per-class table are internally impossible
  (for example F1 0.24 with precision 0.90 and recall 0.28; "Soybean support 5090" is the whole class, not the
  validation split) and its validation figures came from a dataset that contains offline-augmented copies of the
  training photos, which leak between train and validation. Do not cite those numbers.
