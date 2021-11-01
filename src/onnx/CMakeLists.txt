cmake_minimum_required(VERSION 3.22)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/onnxruntime)
    #https://github.com/microsoft/onnxruntime/blob/5c56fa0def02471daae421330e239b688f45a9fc/onnxruntime/core/session/onnxruntime_c_api.cc#L2377
    # v1.9.2 would be nice, but annoying version error...
    execute_process(COMMAND cmd /c "git clone --depth 1 --recursive --branch v1.7.2 https://github.com/Microsoft/onnxruntime" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/)
endif()

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/onnxruntime/lib)
    execute_process(COMMAND cmd /C build.bat --config Release --skip_tests --build_shared_lib --parallel 10 --cmake_generator "Visual Studio 16 2019" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/onnxruntime")
    make_directory("${CMAKE_SOURCE_DIR}/onnxruntime/lib")
    file(
        COPY 
        "${CMAKE_SOURCE_DIR}/onnxruntime/build/Windows/Release/Release/onnxruntime.dll" 
        "${CMAKE_SOURCE_DIR}/onnxruntime/build/Windows/Release/Release/onnxruntime.lib"
        DESTINATION 
        "${CMAKE_SOURCE_DIR}/onnxruntime/lib"
    )
    # file(REMOVE_RECURSE "${CMAKE_SOURCE_DIR}/onnxruntime/build")
endif()

set(ONNXRUNTIME_ROOTDIR "${CMAKE_SOURCE_DIR}/onnxruntime")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOTDIR}/lib")
add_executable(onnx-inference inference.cpp)
target_link_libraries(onnx-inference PRIVATE ${ONNXRUNTIME_LIBRARY})

configure_file("${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.dll" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll" COPYONLY)
configure_file("${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.lib" COPYONLY)
configure_file("${CMAKE_SOURCE_DIR}/src/train/mnist_model.onnx" "${CMAKE_CURRENT_BINARY_DIR}/mnist_model.onnx" COPYONLY)
