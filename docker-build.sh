#!/bin/bash

# Build the Docker image
echo "🐳 Building PDF Outline Extractor Docker image..."
docker build -t pdf-outline-extractor .

echo "✅ Docker image built successfully!"
echo "💡 To run: ./docker-run.sh"
