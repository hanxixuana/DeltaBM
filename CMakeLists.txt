cmake_minimum_required(VERSION 3.5)
project(dbm_code)

# MKL
SET(MKL_OPT "-I$ENV{MKLROOT}/include")
SET(MKL_LIB "-L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def -lmkl_avx2 -lpthread -lm -ldl")

#SET(CMAKE_BUILD_TYPE DEBUG)
SET(CMAKE_BUILD_TYPE RELEASE)

# version

set(DBM_CODE_MAJOR_VERSION 1)
set(DBM_CODE_MINOR_VERSION 2)
set(DBM_CODE_PATCH_VERSION 1)
set(DBM_CODE_VERSION
        ${DBM_CODE_MAJOR_VERSION}.${DBM_CODE_MINOR_VERSION}.${DBM_CODE_PATCH_VERSION})


################
# Preparations #
################

#set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_OMP")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -fopenmp -D_OMP")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG_BASE_LEARNER           \
#                                        -D_DEBUG_BASE_LEARNER_TRAINER   \
#                                        -D_DEBUG_DATA_SET               \
#                                        -D_DEBUG_LOSS_FUNCTION          \
#                                        -D_DEBUG_MATRIX                 \
#                                        -D_DEBUG_MODEL                  \
#                                        -D_CD_INDICATOR                 \
#                                        -D_DEBUG_TOOLS")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG_BASE_LEARNER           \
                                        -D_DEBUG_BASE_LEARNER_TRAINER   \
                                        -D_DEBUG_DATA_SET               \
                                        -D_DEBUG_LOSS_FUNCTION          \
                                        -D_DEBUG_MATRIX                 \
                                        -D_DEBUG_MODEL                  \
                                        -D_DEBUG_TOOLS")

# directory manipulations

set(directory
        ${CMAKE_HOME_DIRECTORY}/build
        ${CMAKE_HOME_DIRECTORY}/lib
        ${CMAKE_HOME_DIRECTORY}/api
        ${CMAKE_HOME_DIRECTORY}/api/dbm_py
        ${CMAKE_HOME_DIRECTORY}/api/dbm_r)
file(MAKE_DIRECTORY ${directory})

# directories for binary shared objects and executables

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

# find header files

include_directories(${CMAKE_HOME_DIRECTORY}/include)


################
#    Library   #
################

set(MODEL_LIB_FILES
        src_lib/model.cpp
        src_lib/data_set.cpp
        src_lib/matrix.cpp
        src_lib/loss_function.cpp
        src_lib/base_learner.cpp
        src_lib/base_learner_trainer.cpp
        src_lib/tools.cpp)
add_library(_dbm_code SHARED ${MODEL_LIB_FILES})

add_library(_dbm_code_mkl SHARED ${MODEL_LIB_FILES})
target_compile_options(_dbm_code_mkl PRIVATE -D_MKL ${MKL_OPT})


################
#  Python API  #
################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/api/dbm_py)

set(PY_API_LIB_FILES
        src_lib/py_api.cpp)

find_package(Boost ${Boost_VERSION})

if(Boost_FOUND)

    find_package(Boost COMPONENTS python)

    message( "-- Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
    message( "-- Boost_LIBRARIES: " ${Boost_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIRS})

    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)

elseif(not Boost_FOUND)

    message(FATAL_ERROR "Unable to find Boost Python!")

endif()

find_package(PythonLibs 2.7 REQUIRED)

if(PythonLibs_FOUND)

    message( "-- PYTHON_INCLUDE_DIRS: " ${PYTHON_INCLUDE_DIRS})
    message( "-- PYTHON_LIBRARY: " ${PYTHON_LIBRARY})
    include_directories(${PYTHON_INCLUDE_DIRS})

elseif(not PythonLibs_FOUND)

    message(FATAL_ERROR "Unable to find Python libraries!")

endif()

add_library(_dbm_cpp_to_python_interface SHARED ${PY_API_LIB_FILES})
#target_link_libraries(_dbm_cpp_to_python_interface _dbm_code ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )
#target_compile_options(_dbm_cpp_to_python_interface -D_MKL ${MKL_OPT})
target_link_libraries(_dbm_cpp_to_python_interface
        _dbm_code_mkl
        -D_MKL
        ${PYTHON_LIBRARIES}
        ${Boost_LIBRARIES}
        ${MKL_OPT}
        ${MKL_LIB})

################
#     Test     #
################

set(TEST_SOURCE_FILES
        src_test/main_test.cpp)
add_executable(main_test ${TEST_SOURCE_FILES})
#add_executable(main_test ${TEST_SOURCE_FILES} ${MODEL_LIB_FILES})
#TARGET_LINK_LIBRARIES(main_test -pg)


TARGET_LINK_LIBRARIES(main_test _dbm_code _dbm_cpp_to_python_interface)
#TARGET_LINK_LIBRARIES(main_test _dbm_code)

add_executable(main_test_mkl ${TEST_SOURCE_FILES})
target_compile_options(main_test_mkl PRIVATE -D_MKL ${MKL_OPT})
TARGET_LINK_LIBRARIES(main_test_mkl _dbm_code_mkl ${MKL_LIB})


################
#     Copy     #
################

# for python
configure_file("${CMAKE_HOME_DIRECTORY}/src_api/dbm_code_py.py"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_py/interface.py"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_api/__init__.py"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_py/__init__.py"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_test/main_test.py"
        "${CMAKE_HOME_DIRECTORY}/api/main_test.py"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_test/dbmPythonApplication.ipynb"
        "${CMAKE_HOME_DIRECTORY}/api/dbmPythonApplication.ipynb"
        COPYONLY)

# for r
#configure_file("${CMAKE_HOME_DIRECTORY}/src_api/dbm_code_py.py"
#        "${CMAKE_HOME_DIRECTORY}/api/dbm_r/dbm_py.py"
#        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_api/dbm_code_r.R"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_r/dbm_r.R"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_test/main_test.R"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_r/main_test.R"
        COPYONLY)

add_custom_command(TARGET _dbm_cpp_to_python_interface
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_dbm_cpp_to_python_interface> ${CMAKE_HOME_DIRECTORY}/api/dbm_r/)


################
#     Copy     #
################

# configure_file("${CMAKE_HOME_DIRECTORY}/numerai_training_data.csv"
#             "${CMAKE_HOME_DIRECTORY}/build/numerai_training_data.csv"
#             COPYONLY)

# configure_file("${CMAKE_HOME_DIRECTORY}/numerai_tournament_data.csv"
#         "${CMAKE_HOME_DIRECTORY}/build/numerai_tournament_data.csv"
#         COPYONLY)

# configure_file("${CMAKE_HOME_DIRECTORY}/numerai_training_data.csv"
#         "${CMAKE_HOME_DIRECTORY}/api/numerai_training_data.csv"
#         COPYONLY)
