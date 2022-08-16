Set-ExecutionPolicy Bypass -Scope Process
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m ensurepip
pip3 install --upgrade pip --user
pip3 install -r requirements.txt