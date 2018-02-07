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

#ifndef INCLUDE_PLANESEGMENTATION_HPP_
#define INCLUDE_PLANESEGMENTATION_HPP_

#include <vector>
#include <map>

#include <pcl/impl/point_types.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include "ObjInstance.hpp"
#include "PlaneSeg.hpp"
#include "UnionFind.h"


class PlaneSegmentation{
public:
	static void segment(const cv::FileStorage& fs,
						pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
						pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
						std::vector<ObjInstance>& objInstances,
						bool segmentMap = false,
						pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
						int viewPort1 = -1,
						int viewPort2 = -1);
    
    static void segment(const cv::FileStorage& fs,
                        cv::Mat rgb,
                        cv::Mat depth,
                        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
                        std::vector<ObjInstance>& objInstances,
                        pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                        int viewPort1 = -1,
                        int viewPort2 = -1);

private:
 
    static void makeSupervoxels(const cv::FileStorage &fs,
                                   pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
                                   bool segmentMap,
                                   std::vector<PlaneSeg> &svs);
    
    static void makeSupervoxels(const cv::FileStorage &fs, cv::Mat rgb, cv::Mat depth, std::vector<PlaneSeg> &svs);
	
	static void makeObjInstances(const std::vector<PlaneSeg> &svs,
								 const std::vector<PlaneSeg> &segs,
								 UnionFind &sets,
								 std::vector<int> &svLabels,
								 pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
								 std::vector<ObjInstance>& objInstances,
								 float curvThresh,
								 double normalThresh,
								 double stepThresh,
								 float areaThresh,
								 pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
								 int viewPort1 = -1,
								 int viewPort2 = -1);
	
	static cv::Mat segmentRgb(cv::Mat rgb, cv::Mat depth, float sigma, float k, int minSegment);
 
	static inline float distToPlane(const pcl::PointNormal& plane,
								const pcl::PointNormal& pt)
	{
		float dx = pt.x - plane.x;
		float dy = pt.y - plane.y;
		float dz = pt.z - plane.z;
		// dot product of normal vector and vector connecting two points
		return fabs(plane.normal_x * dx + plane.normal_y * dy + plane.normal_z * dz);
	}
    
    static void mergeSegments(std::vector<PlaneSeg> &segs,
							  UnionFind &sets,
							  double curvThresh,
							  double normalThresh,
							  double stepThresh,
                              pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                              int viewPort1 = -1,
                              int viewPort2 = -1);
    
    static void mergeSegmentsFF(std::vector<PlaneSeg> &segs,
                              UnionFind &sets,
                              double curvThresh,
                              double normalThresh,
                              double stepThresh,
                              pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                              int viewPort1 = -1,
                              int viewPort2 = -1);
    
    static bool checkIfCoplanar(const PlaneSeg &seg1,
                                const PlaneSeg &seg2,
                                double curvThresh,
                                double normalThresh,
                                double stepThresh);
    
    static double compEdgeScore(const PlaneSeg &seg1,
                                   const PlaneSeg &seg2,
                                   double curvThresh,
                                   double normalThresh,
                                   double stepThresh);
    
    static void drawSegments(pcl::visualization::PCLVisualizer::Ptr viewer,
                                 std::string name,
                                 int vp,
                                 std::vector<PlaneSeg> segs,
                                 UnionFind &sets);
	
	static void visualizeSegmentation(const std::vector<PlaneSeg> &svs,
									  const std::vector<PlaneSeg> &segs,
									  std::vector<int> &svLabels,
									  pcl::visualization::PCLVisualizer::Ptr viewer,
									  int viewPort1,
									  int viewPort2);
									  

	static float compSvToPlaneDist(pcl::PointNormal svNorm,
									pcl::PointNormal planeNorm);


	static void compPlaneNormals(const std::vector<int>& svLabels,
                                 const std::vector<PlaneSeg>& svsInfo,
									pcl::PointCloud<pcl::PointNormal>::Ptr planeNorm,
									std::vector<std::vector<int>>& planeSvsIdx);

	static void compSupervoxelsAreaEst(const std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr >& idToSv,
									std::vector<float>& svAreaEst);
    
    static int curObjInstId = 0;
};

struct SegEdge{
    int u, v;
    int ver;
    double w;
    
    SegEdge() {}
    
    SegEdge(int u, int v, int ver, double w) : u(u), v(v), ver(ver), w(w) {}
	
	SegEdge(int u, int v, double w) : u(u), v(v), w(w) {}
	
	SegEdge(int u, int v) : u(u), v(v) {}
};

bool operator<(const SegEdge &l, const SegEdge &r);


#endif /* INCLUDE_PLANESEGMENTATION_HPP_ */
