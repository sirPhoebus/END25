@echo off

echo Starting ReDSAR Inference (Standalone)...

:: Use Python Embeded if available
if exist "python_embeded\python.exe" (
    echo Using Embedded Python...
    .\python_embeded\python.exe inference.py --checkpoint checkpoint_100.pt --data_path data/evaluation --use_critic %*
) else (
    echo Warning: python_embeded not found. Using system python...
    if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
    python inference.py --checkpoint checkpoint_100.pt --data_path data/evaluation --use_critic %*
)

pause
