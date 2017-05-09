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

#ifndef INCLUDE_SEGMENTATION2_HPP_
#define INCLUDE_SEGMENTATION2_HPP_

#include <vector>
#include <map>

#include <pcl/impl/point_types.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include "ObjInstance.hpp"
#include "SegInfo.hpp"


class Segmentation2{
public:
	static void segment(const cv::FileStorage& fs,
						pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
						pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
						std::vector<ObjInstance>& objInstances,
						bool segmentMap = false,
						pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
						int viewPort1 = -1,
						int viewPort2 = -1);

private:
	static inline float distToPlane(const pcl::PointNormal& plane,
								const pcl::PointNormal& pt)
	{
		float dx = pt.x - plane.x;
		float dy = pt.y - plane.y;
		float dz = pt.z - plane.z;
		// dot product of normal vector and vector connecting two points
		return fabs(plane.normal_x * dx + plane.normal_y * dy + plane.normal_z * dz);
	}

	static float compSvToPlaneDist(pcl::PointNormal svNorm,
									pcl::PointNormal planeNorm);


	static void compPlaneNormals(const std::vector<int>& svLabels,
                                 const std::vector<SegInfo>& svsInfo,
									pcl::PointCloud<pcl::PointNormal>::Ptr planeNorm,
									std::vector<std::vector<int>>& planeSvsIdx);

	static void compSupervoxelsAreaEst(const std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr >& idToSv,
									std::vector<float>& svAreaEst);
};



#endif /* INCLUDE_SEGMENTATION_HPP_ */
