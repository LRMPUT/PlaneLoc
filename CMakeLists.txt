# cmake requirements
cmake_minimum_required(VERSION 2.8)

# Build options have to be before PROJECT(...)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE PATH "Configuration types")
#set(CMAKE_BUILD_TYPE "Debug" CACHE PATH "Current build configuration")

# PUTSLAM Project configuration
project(PlaneLoc)
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -g")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -o3")

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

option(BUILD_DEMO_OBJEXTRACT "Build ObjExtract demo" OFF)

option(BUILD_DEMO_PLANESLAM "Build PlaneSlam demo" ON)

# Include directory
include_directories("${CMAKE_SOURCE_DIR}/include")

# Boost
find_package(Boost COMPONENTS system filesystem)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#g2o library
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
link_directories(${G2O_LIBRARY_DIRS})

#-------------------------------------------------------

set(PlaneSlam_SOURCES
	src/Matching.cpp
	src/FileGrabber.cpp
        src/PlaneSegmentation.cpp
	src/ObjInstance.cpp
	src/PlaneSlam.cpp
	src/Map.cpp
	src/Misc.cpp
	src/UnionFind.cpp
		src/PlaneSeg.cpp
	src/LineDet.cpp
	src/LineSeg.cpp)
	
add_library(PlaneSlam
			${PlaneSlam_SOURCES})

#-------------------------------------------------------

set(unitTests_SOURCES
        tests/Tests.cpp)
add_executable(unitTests
        ${unitTests_SOURCES})
target_link_libraries(unitTests
        PlaneSlam
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OPENGM_LIBRARIES}
        ${G2O_TYPES_SLAM3D}
        ${G2O_TYPES_SBA})

#-------------------------------------------------------

if(BUILD_DEMO_OBJEXTRACT)
	set(demoObjExtract_SOURCES
		demos/demoExtract.cpp)
	add_executable(demoObjExtract
					${demoObjExtract_SOURCES})
	target_link_libraries(demoObjExtract
						PlaneSlam
						${OpenCV_LIBS}
						${Boost_LIBRARIES}
						${PCL_LIBRARIES}
						${OPENGM_LIBRARIES})
					
endif(BUILD_DEMO_OBJEXTRACT)

#-------------------------------------------------------

if(BUILD_DEMO_PLANESLAM)
	set(demoPlaneSlam_SOURCES
		demos/demoSlam.cpp)
	add_executable(demoPlaneSlam
					${demoPlaneSlam_SOURCES})
	target_link_libraries(demoPlaneSlam
						PlaneSlam
						${OpenCV_LIBS}
						${Boost_LIBRARIES}
						${PCL_LIBRARIES}
						${OPENGM_LIBRARIES}
						${G2O_TYPES_SLAM3D}
						${G2O_TYPES_SBA})
					
endif(BUILD_DEMO_PLANESLAM)
			