cmake_minimum_required(VERSION 3.16)
project(generic_slam_native LANGUAGES CXX)

# Build with tools/build_native.sh. OpenCV must be the static 4.12.0 build
# (same version and CPU dispatch as the pip cv2 wheel, so features match).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)
if(NOT OpenCV_VERSION VERSION_EQUAL 4.12.0)
  message(FATAL_ERROR "OpenCV 4.12.0 required (pip cv2 version), found ${OpenCV_VERSION}")
endif()

pybind11_add_module(_native src/module.cpp)
target_link_libraries(_native PRIVATE ${OpenCV_LIBS})
# Keep the statically linked OpenCV private: pip's cv2 lives in the same process.
target_link_options(_native PRIVATE "-Wl,--exclude-libs,ALL")
# tools/build_native.sh copies the module into generic_slam/ (the repository
# path may contain characters Makefile rules cannot quote).
