cmake_minimum_required(VERSION 3.2.1)
project(ParallelBayesianOptimization)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


# One warning from when compiling Eigen headers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

add_subdirectory(MOO)
add_subdirectory(apaweibo)
include_directories(MOO)
include_directories(apaweibo)
set(SRC main.cpp MACE_util.cpp MACE.cpp Config.cpp NLopt_wrapper.cpp)
set(EXE mace_bo)
add_executable(${EXE} ${SRC})
target_link_libraries(${EXE} moo)
target_link_libraries(${EXE} GP)

set_property(TARGET ${EXE} PROPERTY CXX_STANDARD 11)

# Eigen library

# debug macro
option(MYDEBUG "Debug macro for optimizer" OFF)
if(MYDEBUG)
    message(STATUS "Open debug marcro")
    add_definitions(-DMYDEBUG)
    if(DEFINED RAND_SEED)
        message(STATUS "Rand seed is ${RAND_SEED}")
        add_definitions(-DDEBUG_RAND_SEED=${RAND_SEED})
    else()
        add_definitions(-DDEBUG_RAND_SEED=3)
    endif()
else()
    message(STATUS "Close debug marcro")
    remove_definitions(-DMYDEBUG)
    remove_definitions(-DDEBUG_RAND_SEED)
endif()

# if(GP_PATH)
#     message(STATUS "GP_PATH: ${GP_PATH}")
#     include_directories("${GP_PATH}/inc/")
#     find_library(GP NAME GP PATHS ${GP_PATH}/lib)
#     if(GP)
#         target_link_libraries(${EXE} ${GP})
#     else()
#         message("GP library not found in ${GP_PATH}")
#     endif()
# else()
#     message(FATAL_ERROR "GP_PATH not specified")
# endif()

# if(NLOPT_PATH)
#     message(STATUS "NLOPT_PATH: ${NLOPT_PATH}")
#     include_directories("${NLOPT_PATH}/include")
#     find_library(NLOPT NAMES nlopt PATHS ${NLOPT_PATH}/lib NO_CMAKE_SYSTEM_PATH)
# else()
#     find_library(NLOPT NAMES nlopt)
# endif()
# if(NLOPT)
#     message(STATUS "Nlopt library: ${NLOPT}")
#     target_link_libraries(${EXE} ${NLOPT})
# else()
#     message(FATAL_ERROR "NLOPT not found")
# endif()

# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(GSL 2.3 REQUIRED)
if(GSL_FOUND)
    message(STATUS "GSL found, version ${GSL_VERSION}")
    include_directories(${GSL_INCLUDE_DIRS})
    target_link_libraries(${EXE} ${GSL_LIBRARIES})
endif()

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    # Disable OpenMP multi-threading for Eigen matrix operations
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)
find_package(Boost 1.63 COMPONENTS log log_setup thread system REQUIRED)
if(Boost_FOUND)
    message(STATUS "boost inc dir: ${Boost_INCLUDE_DIR}")
    message(STATUS "boost lib dir: ${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIR})
    target_link_libraries(${EXE} ${Boost_LIBRARIES})
endif(Boost_FOUND)


message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${EXE} 
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(FILES README.md
    DESTINATION doc
    PERMISSIONS OWNER_READ GROUP_READ)
