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

#include <algorithm>

#include <pcl/features/normal_3d.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Misc.hpp>
#include <Exceptions.hpp>
#include <g2o/types/slam3d/se3quat.h>
#include "PlaneSeg.hpp"

using namespace std;

void PlaneSeg::calcSegProp(bool filter){
    
    if(filter && points->size() >= 3) {
        //filter out outliers
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        // Optional
//        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.025);
        
        seg.setInputCloud(points);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() < 3) {
            PLANE_EXCEPTION("Could not estimate a planar model for the given dataset.");
        }
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::Normal>::Ptr filteredNormals(new pcl::PointCloud<pcl::Normal>());
        for (auto it = inliers->indices.begin(); it != inliers->indices.end(); ++it) {
            filteredPoints->push_back(points->at(*it));
            filteredNormals->push_back(normals->at(*it));
        }
        points->swap(*filteredPoints);
        normals->swap(*filteredNormals);
    }
    
    if(points->size() < 3){
        segCentroid = Eigen::Vector3d::Zero();
        segCovar = Eigen::Matrix3d::Zero();
        evecs.resize(3);
        evals.resize(3);
        segNormal = Eigen::Vector3d::Zero();
        segPlaneParams = Eigen::Vector4d::Zero();
        segNormalIntDiff = segCurv = 0;
        areaEst = 0;
    }
    else {
        pcl::PCA<pcl::PointXYZRGB> pca;
        pca.setInputCloud(points);
    
        Eigen::Matrix3d evecsMat = pca.getEigenVectors().cast<double>();
        Eigen::Vector3d evalsVec = pca.getEigenValues().cast<double>();
        segCentroid = pca.getMean().head<3>().cast<double>();
    
        // Same work as in pcl::PCA<pcl::PointXYZRGB> - could be done better
        Eigen::MatrixXf pointsDemean;
        demeanPointCloud(*points, pca.getMean(), pointsDemean);
        segCovar = (pointsDemean.topRows<3>() *
                     pointsDemean.topRows<3>().transpose()).cast<double>();
        segCovar /= points->size();
    
        Eigen::Vector3d ev0 = evecsMat.block<3, 1>(0, 0);
        Eigen::Vector3d ev1 = evecsMat.block<3, 1>(0, 1);
        Eigen::Vector3d ev2 = evecsMat.block<3, 1>(0, 2);
        
        evecs.push_back(ev0);
        evecs.push_back(ev1);
        evecs.push_back(ev2);
        
        evals.push_back(evalsVec(0));
        evals.push_back(evalsVec(1));
        evals.push_back(evalsVec(2));
    
        // the eigenvector for the smallest eigenvalue is the normal vector
        segNormal = ev2;
        segPlaneParams.head<3>() = segNormal;
        // distance is the dot product of normal and point lying on the plane
        segPlaneParams(3) = -segNormal.dot(segCentroid);
    
        segCurv = evalsVec(2) / (evalsVec(0) + evalsVec(1) + evalsVec(2));
    
        segNormalIntDiff = segCurv;
    
        // check if aligned with majority of normals
//    bool alignError = false;
        bool isAligned = Misc::checkIfAlignedWithNormals(segNormal, normals, normAlignConsistent);
//    if(normals->size() > 100 && alignError){
//        cout << "normals->size() = " << normals->size() << ", curvature = " << segCurv << endl;
//        throw PLANE_EXCEPTION("Align error");
//    }
        if (!isAligned) {
            segNormal = -segNormal;
        }
    
        float curAreaCoeff = 0.0;
        for (int pt = 0; pt < points->size(); ++pt) {
            pcl::PointXYZRGB curPt = points->at(pt);
            Eigen::Vector3d curPtCentroid = curPt.getVector3fMap().cast<double>() - segCentroid;
            float lenSq = curPtCentroid.squaredNorm();
            float normLen = segNormal.dot(curPtCentroid);
            float distPlaneSq = lenSq - normLen * normLen;
            if (!isnan(distPlaneSq)) {
                curAreaCoeff = max(distPlaneSq, curAreaCoeff);
            }
        }
        areaEst = curAreaCoeff * pi;
    }
}

void PlaneSeg::transform(const Vector7d &transform) {
    g2o::SE3Quat transformSE3Quat(transform);
    Eigen::Matrix4d transformMat = transformSE3Quat.to_homogeneous_matrix();
    Eigen::Matrix3d R = transformMat.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformMat.block<3, 1>(0, 3);
    Eigen::Matrix4d Tinvt = transformMat.inverse();
    Tinvt.transposeInPlace();
    
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
    pcl::transformPointCloud(*points, *points, transformMat);
    
    // pcl::PointCloud<pcl::Normal>::Ptr normals;
    // no need to transform
    
    //  std::vector<int> origPlaneSegs;
    // no need to transform
    
    // Eigen::Vector3d segNormal;
    segNormal = R * segNormal;
    
    // double segNormalIntDiff;
    // no need to transform
    
    // Eigen::Vector3d segCentroid;
    segCentroid = R * segCentroid + t;
    
    // Eigen::Vector4d segPlaneParams;
    segPlaneParams = Tinvt * segPlaneParams;
    
    // Eigen::Matrix3f segCovar;
    segCovar = R * segCovar * R.transpose();
    
    // std::vector<Eigen::Vector3d> evecs;
    for(Eigen::Vector3d &ev : evecs){
        ev = R * ev;
    }
    
    // std::vector<double> evals;
    // no need to transform
    
    // float segCurv;
    // no need to transform
    
    // bool normAlignConsistent;
    // no need to transform
    
    // float areaEst;
    // no need to transform

    // std::vector<int> adjSegs;
    // no need to transform
}

PlaneSeg PlaneSeg::merge(const PlaneSeg &planeSeg, UnionFind &sets) {
    PlaneSeg merged;
    int mid = sets.findSet(id);
    merged.setId(mid);
    merged.addPointsAndNormals(points, normals);
    merged.addPointsAndNormals(planeSeg.getPoints(), planeSeg.getNormals());
    
    merged.addOrigPlaneSegs(origPlaneSegs);
    merged.addOrigPlaneSegs(planeSeg.getOrigPlaneSegs());
    
    merged.calcSegProp();
    
    set<pair<int, int>> addedEdges;
    
    for(const int &as : adjSegs){
        int nh = sets.findSet(as);
        
        pair<int, int> ep = make_pair(min(mid, nh), max(mid, nh));
        if(mid != nh &&
            addedEdges.count(ep) == 0)
        {
            addedEdges.insert(ep);
            merged.addAdjSeg(nh);
        }
    }
    for(const int &as : planeSeg.getAdjSegs()){
        int nh = sets.findSet(as);
        
        pair<int, int> ep = make_pair(min(mid, nh), max(mid, nh));
        if(mid != nh &&
           addedEdges.count(ep) == 0)
        {
            addedEdges.insert(ep);
            merged.addAdjSeg(nh);
        }
    }
    
    return merged;
}

