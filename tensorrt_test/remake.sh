#!/bin/bash

# Navigate to the project root if needed
# cd /path/to/your/project

# Check if build directory exists
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi

# Recreate build directory
echo "Creating new build directory..."
mkdir build
cd build

# Run CMake and make (edit as needed)
cmake ..
make -j$(nproc)
