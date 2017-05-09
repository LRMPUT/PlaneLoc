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

#ifndef INCLUDE_MAP_HPP_
#define INCLUDE_MAP_HPP_

#include <vector>

#include <opencv2/opencv.hpp>

#include "ObjInstance.hpp"

class Map{
public:
	Map(const cv::FileStorage& settings);

	inline void addObj(ObjInstance& obj){
		objInstances.push_back(obj);
	}

	void removeObj(int i){
		objInstances.erase(objInstances.begin() + i);
	}

	inline int size(){
		return objInstances.size();
	}

	inline ObjInstance& operator[](int i){
		return objInstances[i];
	}

    inline pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr getOriginalPointCloud(){
        return originalPointCloud;
    }
private:
    pcl::PointCloud<pcl::PointXYZL>::Ptr getLabeledPointCloud();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorPointCloud();


	std::vector<ObjInstance> objInstances;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr originalPointCloud;
};


#endif /* INCLUDE_MAP_HPP_ */
