# 현재 디렉토리에 있는 모든 파일을 SRC_FILES 변수에 추가한다.
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )

add_library(shape STATIC ${SRC_FILES})

# shape 의 include 경로 지정
target_include_directories(shape PUBLIC ${CMAKE_SOURCE_DIR}/include)

# shape 를 C++ 17 로 컴파일
target_compile_features(shape PRIVATE cxx_std_17)


include(FetchContent)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_TAG v3.10.5
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(json)


target_link_libraries(shape PRIVATE nlohmann_json::nlohmann_json)
