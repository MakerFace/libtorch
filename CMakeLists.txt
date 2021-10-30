cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(APPEND)

add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")

add_executable(example-app src/main.cpp)
add_executable(predict src/test.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(predict "${TORCH_LIBRARIES}" glog ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
set_property(TARGET predict PROPERTY CXX_STANDARD 14)
