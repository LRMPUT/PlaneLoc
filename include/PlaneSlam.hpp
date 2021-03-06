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

#ifndef INCLUDE_PLANESLAM_HPP_
#define INCLUDE_PLANESLAM_HPP_

#include <opencv2/opencv.hpp>


#include <pcl/visualization/pcl_visualizer.h>

#include "FileGrabber.hpp"
#include "Map.hpp"

class PlaneSlam{
public:
	PlaneSlam(const cv::FileStorage& isettings);

	void run();


private:
    
    enum class RecCode{
        Corr,
        Incorr,
        Unk
    };
    
	void evaluateMatching(const cv::FileStorage &fs,
                              const vectorObjInstance &objInstances1,
                              const vectorObjInstance &objInstances2,
                              std::ifstream &inputResFile,
                              std::ofstream &outputResFile,
                              const Vector7d &gtTransform,
                              double scoreThresh,
                              double scoreDiffThresh,
                              double fitThresh,
                              double distinctThresh,
                              double poseDiffThresh,
                              Vector7d &predTransform,
                              RecCode &recCode,
                              double &linDist,
                              double &angDist,
                              pcl::visualization::PCLVisualizer::Ptr viewer,
                              int viewPort1,
                              int viewPort2);
	
	FileGrabber fileGrabber;
	Map map;

	const cv::FileStorage& settings;

//	pcl::visualization::PCLVisualizer viewer;
};


#endif /* INCLUDE_PLANESLAM_HPP_ */
