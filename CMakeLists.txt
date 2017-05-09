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

#OpenGM
set(OPENGM_EXTERNAL_LIBS_DIR "/home/jachu/lib/opengm-master/build/src/external" CACHE PATH "OpenGM external library directory")
set(OPENGM_MAXFLOW_INCLUDE_DIR "/home/jachu/lib/opengm-master/src/external/MaxFlow-v3.02.src-patched" CACHE PATH "OpenGM MaxFlow include directory")

include_directories(/usr/local/include)
include_directories(${OPENGM_MAXFLOW_INCLUDE_DIR})
link_directories(${OPENGM_EXTERNAL_LIBS_DIR})
set(OPENGM_LIBRARIES
	external-library-maxflow)

#g2o library
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
link_directories(${G2O_LIBRARY_DIRS})

#zlib library
find_package(ZLIB REQUIRED)
include_directories(${ZLUB_INCLUDE_DIR})
link_directories(${ZLIB_LIBRARY_DIRS})

#ElasticFusion
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS} )

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(Pangolin 0.1 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(efusion REQUIRED)
include_directories(${EFUSION_INCLUDE_DIR})

#-------------------------------------------------------

set(pngToKlg_SOURCES
	utils/pngToKlg.cpp)
add_executable(pngToKlg
				${pngToKlg_SOURCES})
target_link_libraries(pngToKlg
					${OpenCV_LIBS}
					${Boost_LIBRARIES}
					${ZLIB_LIBRARY})

#-------------------------------------------------------

set(convertGt_SOURCES
	utils/convertGt.cpp)
add_executable(convertGt
				${convertGt_SOURCES})
target_link_libraries(convertGt)

#-------------------------------------------------------

set(makeCloudsEFusion_SOURCES
	utils/makeCloudsEFusion.cpp)
add_executable(makeCloudsEFusion
				${makeCloudsEFusion_SOURCES})
target_link_libraries(makeCloudsEFusion
						${OpenCV_LIBS}
						${Boost_LIBRARIES}
						${GLUT_LIBRARY}
						${CUDA_LIBRARIES}
						${Pangolin_LIBRARIES}
						${EFUSION_LIBRARY}
						${PCL_LIBRARIES})

#-------------------------------------------------------

set(convert7places_SOURCES
        utils/convert7places.cpp)
add_executable(convert7places
        ${convert7places_SOURCES})
target_link_libraries(convert7places
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${ZLIB_LIBRARY})

#-------------------------------------------------------

set(illustration_SOURCES
		utils/illustration.cpp)
add_executable(illustration
		${illustration_SOURCES})
target_link_libraries(illustration
                    ${PCL_LIBRARIES})

#-------------------------------------------------------

set(PlaneSlam_SOURCES
	src/Matching.cpp
	src/FileGrabber.cpp
	src/Segmentation2.cpp
	src/ObjInstance.cpp
	src/PlaneSlam.cpp
	src/Map.cpp
	src/Misc.cpp
	src/UnionFind.cpp
	src/SegInfo.cpp)
	
add_library(PlaneSlam
			${PlaneSlam_SOURCES})
			
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
						${G2O_TYPES_SBA}
						${GLUT_LIBRARY}
						${CUDA_LIBRARIES}
						${Pangolin_LIBRARIES}
						${EFUSION_LIBRARY})
					
endif(BUILD_DEMO_PLANESLAM)
			