# cmake requirements
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)

# Build options have to be before PROJECT(...)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE PATH "Configuration types")
#set(CMAKE_BUILD_TYPE "Debug" CACHE PATH "Current build configuration")

# PlaneLoc Project configuration
project(PlaneLoc)
#set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -g")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -o3")

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

option(BUILD_DEMO_OBJEXTRACT "Build ObjExtract demo" OFF)

option(BUILD_DEMO_PLANESLAM "Build PlaneSlam demo" ON)

# Include directory
include_directories("${CMAKE_SOURCE_DIR}/include")

# Boost
find_package(Boost COMPONENTS system filesystem serialization)

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

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

	message(STATUS "This project requires the CGAL library, and will not be compiled.")
	return()

endif()

# include helper file
include( ${CGAL_USE_FILE} )

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
	src/LineSeg.cpp
    src/ConcaveHull.cpp
	src/EKFPlane.cpp
	src/PlaneEstimator.cpp)
	
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
        ${G2O_TYPES_SBA}
        ${CGAL_LIBRARIES}
        ${CGAL_3RD_PARTY_LIBRARIES})

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
						${G2O_TYPES_SBA}
						${CGAL_LIBRARIES}
						${CGAL_3RD_PARTY_LIBRARIES})
					
endif(BUILD_DEMO_PLANESLAM)
			