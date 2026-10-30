# One-time setup on Windows PowerShell (run from the project folder). NOT executed on Windows by the author.
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - open it and paste your GEMINI_API_KEY." }
Write-Host "Setup finished. Next: .\scripts\prepare_and_train.ps1"
