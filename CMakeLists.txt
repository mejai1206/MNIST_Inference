cmake_minimum_required(VERSION 3.23)
project(untitled2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

#set_target_properties(untitled2 PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON) NO


# 라이브러리 파일은 빌드 디렉토리 안에 lib 폴더에 출력.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 실행 파일은 빌드 디렉토리 안에 bin 폴더에 출력.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(app)
add_subdirectory(lib)
#add_subdirectory(tests)