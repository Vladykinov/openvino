# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

add_library(cpuSpecificRtInfo STATIC $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.hpp
                                     $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.cpp)
target_link_libraries(cpuSpecificRtInfo PRIVATE openvino::runtime)

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src)
set(DEPENDENCIES openvino_intel_cpu_plugin)
set(LINK_LIBRARIES funcSharedTests cpuSpecificRtInfo)
if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}")
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/extension ${CMAKE_CURRENT_SOURCE_DIR}/onnx)
endif()

set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/extension ${CMAKE_CURRENT_SOURCE_DIR}/onnx
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behaviour
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/multi
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets
                          ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests
                          ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests
                          ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests
                          ${CMAKE_CURRENT_SOURCE_DIR}/bfloat16
                          ${CMAKE_CURRENT_SOURCE_DIR}/blob
                          ${CMAKE_CURRENT_SOURCE_DIR}/behavior)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS
            CPU
)

set_ie_threading_interface_for(${TARGET_NAME})
