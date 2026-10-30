# Data -> preprocess -> validate -> train -> evaluate. NOT executed on Windows by the author.
# Usage:  .\scripts\prepare_and_train.ps1            (full PlantVillage, lightweight training preset)
#         .\scripts\prepare_and_train.ps1 -Quick      (50 images/class sample, tiny run to check your setup)
param([switch]$Quick, [switch]$AllowMissingClasses)
$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
if ($Quick) { python -m src.data_download all --sample-per-class 50 } else { python -m src.data_download all }
python -m src.preprocess
python -m src.dataset_validation
$trainArgs = @()
if ($AllowMissingClasses) { $trainArgs += "--allow-missing-classes" }
if ($Quick) { $trainArgs += @("--head-epochs", "1", "--finetune-epochs", "0") }
python -m src.train @trainArgs
python -m src.evaluate
Write-Host "Done. Launch the app with:  streamlit run app.py"
