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
									 const vectorObjInstance &frameObjInstances,
									 const vectorObjInstance &mapObjInstances,
									 vectorVector7d &bestTrans,
									 std::vector<double> &bestTransProbs,
									 std::vector<double> &bestTransFits,
									 std::vector<int> &bestTransDistinct,
									 pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
									 int viewPort1 = -1,
									 int viewPort2 = -1);

    static double planeEqDiffLogMap(const ObjInstance &obj1,
                                    const ObjInstance &obj2,
                                    const Vector7d &transform);
    
    static double lineSegEqDiff(const LineSeg &lineSeg1,
                                const LineSeg &lineSeg2,
                                const Vector7d &transform);

    static double checkConvexHullIntersection(const ObjInstance& obj1,
                                              const ObjInstance& obj2,
                                              const Vector7d& transform,
                                              double& intArea,
                                              pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                              int viewPort1 = -1,
                                              int viewPort2 = -1);
    
    static double checkLineSegIntersection(const LineSeg &lineSeg1,
                                           const LineSeg &lineSeg2,
                                           const Vector7d &transform,
                                           double &intLen);

    static Vector7d bestTransformPointsDirsDists(const vectorVector3d &points1,
												 const vectorVector3d &points2,
												 const std::vector<double> &pointsW,
												 const vectorVector3d &virtPoints1,
												 const vectorVector3d &virtPoints2,
												 const std::vector<double> &virtPointsW,
												 const vectorVector3d &dirs1,
												 const vectorVector3d &dirs2,
												 const std::vector<double> &dirsW,
												 const std::vector<double> &dists1,
												 const std::vector<double> &dists2,
												 const vectorVector3d &distDirs1,
												 const std::vector<double> &distsW,
												 const vectorVector3d &distPts1,
												 const vectorVector3d &distPts2,
												 const vectorVector3d &distPtsDirs1,
												 const std::vector<double> &distsPtsW,
												 double sinValsThresh,
												 bool &fullConstrRot,
												 bool &fullConstrTrans);
    
    static void convertToPointsDirsDists(const vectorVector3d &points,
                                        const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &planes,
                                        const std::vector<Vector6d> &lines,
                                        vectorVector3d &retPoints,
                                        vectorVector3d &retVirtPoints,
                                        vectorVector3d &retDirs,
                                        std::vector<double> &retDists,
                                        vectorVector3d &retDistDirs,
                                        vectorVector3d &retDistPts,
                                        vectorVector3d &retDistPtsDirs);
    
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
                       const std::vector<Matching::PotMatch> &imatchSet,
                       const std::vector<double>& iintAreaPlanes,
                       const std::vector<std::vector<double>>& iintLenLines)
                : transform(itransform),
                  matchSet(imatchSet),
                  intAreaPlanes(iintAreaPlanes),
                  intLenLines(iintLenLines),
                  score(0.0)
                  
        {}
        Vector7d transform;
        double score;
        std::vector<Matching::PotMatch> matchSet;
        std::vector<double> intAreaPlanes;
        std::vector<std::vector<double> > intLenLines;
    };

	class ProbDistKernel{
	public:
		ProbDistKernel(Vector7d ikPt,
						Eigen::Matrix<double, 6, 6> iinfMat,
						double iweight);

		double eval(Vector7d pt) const;
		
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	private:
		Vector7d kPt;

		g2o::SE3Quat kPtSE3Quat;

		Eigen::Matrix<double, 6, 6> infMat;

		double weight;
	};
    
    typedef std::vector<Matching::ProbDistKernel, Eigen::aligned_allocator<Matching::ProbDistKernel> > vectorProbDistKernel;
    
    static double compAngleDiffBetweenNormals(const Eigen::Vector3d &nf1,
											  const Eigen::Vector3d &ns1,
											  const Eigen::Vector3d &nf2,
											  const Eigen::Vector3d &ns2);
    
    static bool checkLineToLineAng(const vectorLineSeg &lineSegs1,
                                   const vectorLineSeg &lineSegs2,
                                   double lineToLineAngThresh);
                                   
    static bool checkPlaneToPlaneAng(const vectorVector4d &planes1,
                                     const vectorVector4d &planes2,
                                     double planeToPlaneAngThresh);
    
    static bool checkPlaneToLineAng(const vectorVector4d &planes1,
                                     const vectorLineSeg &lineSegs1,
                                     const vectorVector4d &planes2,
                                     const vectorLineSeg &lineSegs2,
                                     double planeToLineAngThresh);
    
    static std::vector<PotMatch> findPotMatches(const vectorObjInstance &mapObjInstances,
												const vectorObjInstance &frameObjInstances,
                                                double planeAppThresh,
                                                double lineAppThresh,
                                                double lineToLineAngThresh,
                                                pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                                int viewPort1 = -1,
                                                int viewPort2 = -1);
    
    static std::vector<std::vector<PotMatch> > findPotSets(std::vector<PotMatch> potMatches,
														   const vectorObjInstance &mapObjInstances,
														   const vectorObjInstance &frameObjInstances,
                                                           double planeDistThresh,
                                                           double lineToLineAngThresh,
                                                           double planeToPlaneAngThresh,
                                                           double planeToLineAngThresh,
                                                           pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                                           int viewPort1 = -1,
                                                           int viewPort2 = -1);
    
	static void compObjFeatures(const vectorObjInstance& objInstances,
								std::vector<cv::Mat>& objFeats);

    static void compObjDistances(const vectorObjInstance& objInstances,
                                std::vector<std::vector<double>>& objDistances);

	static void comp3DTransform(const vectorVector4d& planes1,
								const vectorVector4d& planes2,
								const std::vector<std::pair<int, int>>& triplet,
								Vector7d& transform,
                                double sinValsThresh,
								bool& fullConstr);

	static Vector7d bestTransformPlanes(const vectorVector4d& planes1,
                                        const vectorVector4d& planes2,
                                        double sinValsThresh,
                                        bool &fullConstr);

	static Vector7d bestTransformPointsAndDirs(const vectorVector3d& points1,
                                               const vectorVector3d& points2,
                                               const std::vector<double>& pointsW,
                                               const vectorVector3d& dirs1,
                                               const vectorVector3d& dirs2,
                                               const std::vector<double>& dirsW,
                                               double sinValsThresh,
                                               bool compTrans,
                                               bool &fullConstrRot,
                                               bool &fullConstrTrans);



	static double scoreTransformByProjection(const Vector7d& transform,
								const std::vector<std::pair<int, int>> triplet,
								const vectorObjInstance& objInstances1,
								const vectorObjInstance& objInstances2,
								std::vector<double>& intAreaPair,
                                double planeEqDiffThresh,
                                double intAreaThresh,
								pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
								int viewPort1 = -1,
								int viewPort2 = -1);
    
    static double scoreTransformByProjection(const Vector7d &transform,
                                             const std::vector<PotMatch> curSet,
                                             const vectorObjInstance &objInstances1,
                                             const vectorObjInstance &objInstances2,
                                             std::vector<double> &intAreaPlanes,
                                             std::vector<std::vector<double> > &intLenLines,
                                             double planeEqDiffThresh,
                                             double lineEqDiffThresh,
                                             double intAreaThresh,
                                             double intLenThresh,
                                             pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
											 int viewPort1 = -1,
											 int viewPort2 = -1);

	static double evalPoint(const Vector7d &pt,
							const vectorProbDistKernel &dist);
    
    static int countDifferent(const std::set<int> &setIdxs,
                              const vectorObjInstance &objs);
};

#endif /* INCLUDE_MATCHING_HPP_ */
