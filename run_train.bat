@echo off

echo Starting ReDSAR Training (Standalone)...
echo Settings: Aggressive Training (LR=1e-4, Steps=8, Clip=1.0)

:: Use Python Embeded if available
if exist "python_embeded\python.exe" (
    echo Using Embedded Python...
    .\python_embeded\python.exe train.py --windows-standalone-build --data_path data/training --epochs 1000 --batch_size 2 --layers 8 --lr 1e-4 %*
) else (
    echo Warning: python_embeded not found. Using system python...
    if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
    python train.py --windows-standalone-build --data_path data/training --epochs 1000 --batch_size 2 --layers 8 --lr 1e-4 %*
)

pause
