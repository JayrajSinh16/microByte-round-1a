@echo off
REM Create local input and output directories if they don't exist
if not exist "input" mkdir input
if not exist "output" mkdir output

echo 🐳 Running PDF Outline Extractor in Docker...
echo 📁 Input directory: %CD%\input
echo 📁 Output directory: %CD%\output

REM Run the container with volume mounts
docker run --rm -v "%CD%\input:/app/input" -v "%CD%\output:/app/output" pdf-outline-extractor

if %ERRORLEVEL% == 0 (
    echo ✅ Processing completed!
    echo 📄 Check the 'output' directory for results
) else (
    echo ❌ Docker run failed!
)
