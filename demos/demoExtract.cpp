/*
    Copyright (c) 2017 Mobile Robots Laboratory at Poznan University of Technology:
    -Jan Wietrzykowski name.surname [at] put.poznan.pl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/normal_3d.h>

#include "FileGrabber.hpp"
#include "PlaneSegmentation.hpp"
#include "Exceptions.hpp"
#include "Misc.hpp"
#include "Types.hpp"

using namespace std;
using namespace cv;

void help()
{
	cout << "Use: demoExtract -s settingsfile" << std::endl;
}

int main(int argc, char * argv[]){
	try{
		//settings file
		std::string settingsFilename;
		if (argc == 1) {
			//assume settings in working directory
			settingsFilename = "settings.yml";
		} else if (argc == 3) {
			if(std::string(argv[1]) != "-s") {
				//incorrect option
				help();
			} else {
				//settings provided as argument
				settingsFilename = std::string(argv[2]);
			}
		} else {
			//incorrect arguments
			help();
		}

		cv::FileStorage fs;
		fs.open(settingsFilename, cv::FileStorage::READ);
		if (!fs.isOpened()) {
			throw  PLANE_EXCEPTION(string("Could not open settings file: ") + settingsFilename);
		}

//		boost::filesystem::path datasetPath("../res/");
		FileGrabber fileGrabber(fs);

		Mat cameraParams(3, 3, CV_32FC1, Scalar(0));
		cameraParams.at<float>(0, 0) = 5.1885790117450188e+02;
		cameraParams.at<float>(1, 1) = 5.1946961112127485e+02;
		cameraParams.at<float>(0, 2) = 3.2558244941119034e+02;
		cameraParams.at<float>(1, 2) = 2.5373616633400465e+02;

		Mat rgb, depth;
		std::vector<FileGrabber::FrameObjInstance> objInstances;
		std::vector<double> accelData;
		Vector7d pose;
		while(fileGrabber.getFrame(rgb, depth, objInstances, accelData, pose) >= 0){
			Mat depthFloat;
			// convert from uint16 in mm to float in m
			depth.convertTo(depthFloat, CV_32F, 0.001);
			Mat xyz = Misc::projectTo3D(depthFloat, cameraParams);

			pcl::visualization::PCLVisualizer viewer("3D Viewer");
			int v1 = 0;
			int v2 = 0;
			viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
			viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
			viewer.addCoordinateSystem();

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
			for(int row = 0; row < rgb.rows; ++row){
				for(int col = 0; col < rgb.cols; ++col){
					pcl::PointXYZRGB p((uint8_t)rgb.at<Vec3b>(row, col)[0],
										(uint8_t)rgb.at<Vec3b>(row, col)[1],
										(uint8_t)rgb.at<Vec3b>(row, col)[2]);
					p.x = xyz.at<Vec3f>(row, col)[0];
					p.y = xyz.at<Vec3f>(row, col)[1];
					p.z = xyz.at<Vec3f>(row, col)[2];
//					cout << "Point at (" << xyz.at<Vec3f>(row, col)[0] << ", " <<
//											xyz.at<Vec3f>(row, col)[1] << ", " <<
//											xyz.at<Vec3f>(row, col)[2] << "), rgb = (" <<
//											(int)rgb.at<Vec3b>(row, col)[0] << ", " <<
//											(int)rgb.at<Vec3b>(row, col)[1] << ", " <<
//											(int)rgb.at<Vec3b>(row, col)[2] << ") " << endl;
					pointCloud->push_back(p);
				}
			}
			viewer.addPointCloud(pointCloud, "cloud", v1);

			pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointCloudLab(new pcl::PointCloud<pcl::PointXYZRGBL>());
			vector<ObjInstance> curObjInstances;
//			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudCol(new pcl::PointCloud<pcl::PointXYZRGBA>());
			PlaneSegmentation::segment(fs,
								pointCloud,
								pointCloudLab,
								curObjInstances,
								pcl::visualization::PCLVisualizer::Ptr(&viewer),
								v2);


//			viewer.addPointCloud(pointCloudCol, "cloudCol", v2);

			viewer.initCameraParameters();
			viewer.setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
			while (!viewer.wasStopped()){
				viewer.spinOnce (100);
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
			viewer.close();
		}
	}
	catch(plane_exception& e){
		cout << "Catch pgm_exception in main(): " << e.what() << endl;
	}
	catch(char const *str){
		cout << "Catch const char* in main(): " << str << endl;
		return -1;
	}
	catch(std::exception& e){
		cout << "Catch std exception in main(): " << e.what() << endl;
	}
	catch(...){
		cout << "Catch ... in main()" << endl;
		return -1;
	}
}
