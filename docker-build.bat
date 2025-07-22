@echo off
echo 🐳 Building PDF Outline Extractor Docker image...
docker build -t pdf-outline-extractor .

if %ERRORLEVEL% == 0 (
    echo ✅ Docker image built successfully!
    echo 💡 To run: docker-run.bat
) else (
    echo ❌ Docker build failed!
)
