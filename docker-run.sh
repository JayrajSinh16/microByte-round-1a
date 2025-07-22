#!/bin/bash

# Create local input and output directories if they don't exist
mkdir -p ./input
mkdir -p ./output

echo "🐳 Running PDF Outline Extractor in Docker..."
echo "📁 Input directory: $(pwd)/input"
echo "📁 Output directory: $(pwd)/output"

# Run the container with volume mounts
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  pdf-outline-extractor

echo "✅ Processing completed!"
echo "📄 Check the 'output' directory for results"
