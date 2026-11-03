#!/usr/bin/env bash
# Build generic_slam/_native*.so (WSL/Linux). Usage: tools/build_native.sh
# OPENCV_DIR: static OpenCV 4.12.0 install (default ~/opt/opencv-4.12.0).
# BUILD_DIR: CMake build tree, kept off /mnt/c for speed (default ~/build/generic_slam_native).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
opencv="${OPENCV_DIR:-$HOME/opt/opencv-4.12.0}"
build="${BUILD_DIR:-$HOME/build/generic_slam_native}"
python="${PYTHON:-python3.8}"
cmake -S "$repo/native" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v "$python")" \
  -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)" \
  -DOpenCV_DIR="$opencv/lib/cmake/opencv4"
cmake --build "$build" -j"$(nproc)"
cp "$build"/_native*.so "$repo/generic_slam/"
