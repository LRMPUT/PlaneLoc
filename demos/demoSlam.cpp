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

#include "PlaneSlam.hpp"
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
//	try{
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

		PlaneSlam planeSlam(fs);
		planeSlam.run();
//	}
//	catch(plane_exception& e){
//		cout << "Catch pgm_exception in main(): " << e.what() << endl;
//	}
//	catch(char const *str){
//		cout << "Catch const char* in main(): " << str << endl;
//		return -1;
//	}
//	catch(std::exception& e){
//		cout << "Catch std exception in main(): " << e.what() << endl;
//	}
//	catch(...){
//		cout << "Catch ... in main()" << endl;
//		return -1;
//	}
}


