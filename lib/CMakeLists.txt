file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        )

add_library(inference STATIC ${SRC_FILES})

set_target_properties(inference PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(inference PUBLIC ${CMAKE_SOURCE_DIR}/include)


target_compile_features(inference PRIVATE cxx_std_17)


include(FetchContent)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_TAG v3.10.5
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(json)


target_link_libraries(inference PRIVATE nlohmann_json::nlohmann_json)
