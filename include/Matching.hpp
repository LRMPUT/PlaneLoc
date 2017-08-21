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

#ifndef INCLUDE_MATCHING_HPP_
#define INCLUDE_MATCHING_HPP_

#include <vector>

#include <opencv2/opencv.hpp>

#include <Eigen/Eigen>

#include <g2o/types/slam3d/se3quat.h>

#include "Types.hpp"
#include "ObjInstance.hpp"

class Matching {
public:
	enum class MatchType{
		Ok,
		Unknown
	};

	static MatchType matchFrameToMap(const cv::FileStorage &fs,
                                     const std::vector<ObjInstance> &frameObjInstances,
                                     const std::vector<ObjInstance> &mapObjInstances,
                                     std::vector<Vector7d> &bestTrans,
                                     std::vector<double> &bestTransProbs,
                                     std::vector<double> &bestTransFits,
                                     pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                     int viewPort1 = -1,
                                     int viewPort2 = -1);

    static double planeDiffLogMap(const ObjInstance& obj1,
                                  const ObjInstance& obj2,
                                  const Vector7d& transform);

    static double checkConvexHullIntersection(const ObjInstance& obj1,
                                              const ObjInstance& obj2,
                                              const Vector7d& transform,
                                              double& intArea,
                                              pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                              int viewPort1 = -1,
                                              int viewPort2 = -1);

    static Vector7d bestTransformPointsDirsDists(const std::vector<Eigen::Vector3d>& points1,
                                                 const std::vector<Eigen::Vector3d>& points2,
                                                 const std::vector<double>& pointsW,
                                                 const std::vector<Eigen::Vector3d>& virtPoints1,
                                                 const std::vector<Eigen::Vector3d>& virtPoints2,
                                                 const std::vector<double>& virtPointsW,
                                                 const std::vector<Eigen::Vector3d>& dirs1,
                                                 const std::vector<Eigen::Vector3d>& dirs2,
                                                 const std::vector<double>& dirsW,
                                                 const std::vector<double> &dists1,
                                                 const std::vector<double> &dists2,
                                                 const std::vector<Eigen::Vector3d> &distDirs1,
                                                 const std::vector<double> &distsW,
                                                 const std::vector<Eigen::Vector3d> &distPts1,
                                                 const std::vector<Eigen::Vector3d> &distPts2,
                                                 const std::vector<Eigen::Vector3d> &distPtsDirs1,
                                                 const std::vector<double> &distsPtsW,
                                                 double sinValsThresh,
                                                 bool &fullConstrRot,
                                                 bool &fullConstrTrans);

private:

	struct PotMatch{
        PotMatch() {}
        
        PotMatch(int plane1,
                 const std::vector<int> &lineSegs1,
                 int plane2,
                 const std::vector<int> &lineSegs2)
                : plane1(plane1),
                  lineSegs1(lineSegs1),
                  plane2(plane2),
                  lineSegs2(lineSegs2) {}
        
        PotMatch(int plane1,
                 const std::vector<int> &lineSegs1,
                 int plane2,
                 const std::vector<int> &lineSegs2,
                 double planeAppDiff,
                 const std::vector<double> &lineSegAppDiffs)
                : plane1(plane1),
                  lineSegs1(lineSegs1),
                  plane2(plane2),
                  lineSegs2(lineSegs2),
                  planeAppDiff(planeAppDiff),
                  lineSegAppDiffs(lineSegAppDiffs) {}
        
        int plane1;
        
        std::vector<int> lineSegs1;
        
        int plane2;
        
        std::vector<int> lineSegs2;
        
        double planeAppDiff;
        
        std::vector<double> lineSegAppDiffs;
    };
	
    struct ValidTransform{
        ValidTransform()
        {}

        ValidTransform(const Vector7d& itransform,
                       double iscore,
                       const std::vector<std::pair<int, int>>& itriplet,
                       const std::vector<double>& iintAreas,
                       const std::vector<double>& iappDiffs)
                : transform(itransform),
                  score(iscore),
                  triplet(itriplet),
                  intAreas(iintAreas),
                  appDiffs(iappDiffs)
        {}
        Vector7d transform;
        double score;
        std::vector<std::pair<int, int>> triplet;
        std::vector<double> intAreas;
        std::vector<double> appDiffs;
    };

	class ProbDistKernel{
	public:
		ProbDistKernel(Vector7d ikPt,
						Eigen::Matrix<double, 6, 6> iinfMat,
						double iweight);

		double eval(Vector7d pt) const;

	private:
		Vector7d kPt;

		g2o::SE3Quat kPtSE3Quat;

		Eigen::Matrix<double, 6, 6> infMat;

		double weight;
	};
    
    static bool checkLineToLineAng(const std::vector<LineSeg> &lineSegs1,
                                   const std::vector<LineSeg> &lineSegs2,
                                   double lineToLineAngThresh);

    static std::vector<PotMatch> findPotMatches(const std::vector<ObjInstance>& objInstances1,
                                                const std::vector<ObjInstance>& objInstances2,
                                                double planeAppThresh,
                                                double lineAppThresh,
                                                double lineToLineAngThresh,
                                                pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                                int viewPort1 = -1,
                                                int viewPort2 = -1);
    
    static std::vector<std::vector<PotMatch> > findPotSets(std::vector<PotMatch> potMatches,
                                                           const std::vector<ObjInstance>& objInstances1,
                                                           const std::vector<ObjInstance>& objInstances2,
                                                           double planeDistThresh,
                                                           double planeToPlaneAngThresh,
                                                           double planeToLineAngThresh,
                                                           pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                                           int viewPort1 = -1,
                                                           int viewPort2 = -1);
    
	static void compObjFeatures(const std::vector<ObjInstance>& objInstances,
								std::vector<cv::Mat>& objFeats);

    static void compObjDistances(const std::vector<ObjInstance>& objInstances,
                                std::vector<std::vector<double>>& objDistances);

	static void comp3DTransform(const std::vector<Eigen::Vector4d>& planes1,
								const std::vector<Eigen::Vector4d>& planes2,
								const std::vector<std::pair<int, int>>& triplet,
								Vector7d& transform,
                                double sinValsThresh,
								bool& fullConstr);

	static Vector7d bestTransformPlanes(const std::vector<Eigen::Vector4d>& planes1,
                                        const std::vector<Eigen::Vector4d>& planes2,
                                        double sinValsThresh,
                                        bool &fullConstr);

	static Vector7d bestTransformPointsAndDirs(const std::vector<Eigen::Vector3d>& points1,
                                               const std::vector<Eigen::Vector3d>& points2,
                                               const std::vector<double>& pointsW,
                                               const std::vector<Eigen::Vector3d>& dirs1,
                                               const std::vector<Eigen::Vector3d>& dirs2,
                                               const std::vector<double>& dirsW,
                                               double sinValsThresh,
                                               bool compTrans,
                                               bool &fullConstrRot,
                                               bool &fullConstrTrans);



	static double scoreTransformByProjection(const Vector7d& transform,
								const std::vector<std::pair<int, int>> triplet,
								const std::vector<ObjInstance>& objInstances1,
								const std::vector<ObjInstance>& objInstances2,
								std::vector<double>& intAreaPair,
                                double planeEqDiffThresh,
                                double intAreaThresh,
								pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
								int viewPort1 = -1,
								int viewPort2 = -1);

	static double evalPoint(Vector7d pt,
						const std::vector<ProbDistKernel>& dist);

	static void intersectConvexHulls(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr chull1,
									const pcl::Vertices& poly1,
									const pcl::PointCloud<pcl::PointXYZRGB>::Ptr chull2,
									const pcl::Vertices& poly2,
									const Eigen::Vector4d planeEq,
									pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullRes,
									pcl::Vertices& polyRes,
									double& areaRes,
									pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
									int viewPort1 = -1,
									int viewPort2 = -1);
	enum class SegIntType{
		Collinear,
		Vertex,
		One,
		Zero
	};

	enum class Inside{
		First,
		Second,
		Unknown
	};

	static SegIntType intersectLineSegments(const Eigen::Vector2d& begPt1,
										const Eigen::Vector2d& endPt1,
										const Eigen::Vector2d& begPt2,
										const Eigen::Vector2d& endPt2,
										Eigen::Vector2d& intPt1,
										Eigen::Vector2d& intPt2,
										double eps = 1e-6);

	static SegIntType intersectParallelLineSegments(const Eigen::Vector2d& begPt1,
										const Eigen::Vector2d& endPt1,
										const Eigen::Vector2d& begPt2,
										const Eigen::Vector2d& endPt2,
										Eigen::Vector2d& intPt1,
										Eigen::Vector2d& intPt2,
										double eps);

	static bool isBetween(const Eigen::Vector2d& beg,
						const Eigen::Vector2d& end,
						const Eigen::Vector2d& pt,
						double eps);

	static Inside newInsideFlag(Inside oldFlag,
								const Eigen::Vector2d& intPt,
								double cross1H2,
								double cross2H1,
								double eps);

	static double cross2d(const Eigen::Vector2d& v1,
						const Eigen::Vector2d& v2);

	static void makeCclockwise(std::vector<Eigen::Vector2d>& chull,
								double eps = 1e-6);
};

#endif /* INCLUDE_MATCHING_HPP_ */
