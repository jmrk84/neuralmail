@echo off
echo Building NeuralMail standalone executable with Nuitka...
python build_executable.py
if %ERRORLEVEL% NEQ 0 (
    echo Build failed! See output above for details.
    echo Make sure you have a C++ compiler installed (Visual Studio or MinGW64).
    pause
    exit /b 1
)
echo Build completed successfully!
echo The Nuitka-compiled executable is located in the 'dist' folder.
pause