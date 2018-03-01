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

#include <thread>
#include <chrono>
#include <vector>
#include <list>

//#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>

#include "ObjInstance.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"
#include "Matching.hpp"

using namespace std;

ObjInstance::ObjInstance() : id(-1) {}

ObjInstance::ObjInstance(int iid,
					ObjType itype,
					pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ipoints,
					const vectorPlaneSeg& isvs,
                    int ieol)
	: id(iid),
	  type(itype),
	  points(new pcl::PointCloud<pcl::PointXYZRGB>(*ipoints)),
	  svs(isvs),
      eolCnt(ieol),
      obsCnt(1),
      trial(false)
{
    {
        Eigen::MatrixXd pts(4, points->size());
        for(int i = 0; i < points->size(); ++i){
            pts.col(i) = points->at(i).getVector4fMap().cast<double>();
        }

        planeEstimator.init(pts);
    }

    
    Eigen::Vector3d ev0 = planeEstimator.getEvecs().block<3, 1>(0, 0);
    Eigen::Vector3d ev1 = planeEstimator.getEvecs().block<3, 1>(0, 1);
    Eigen::Vector3d ev2 = planeEstimator.getEvecs().block<3, 1>(0, 2);
    
    // shorter side of the plane is the second largest eigenvalue
    shorterComp = sqrt(planeEstimator.getEvals()(1)/points->size());
    
    paramRep = planeEstimator.getPlaneEq();
    
    princComp = vectorVector3d{ev0, ev1, ev2};
    princCompLens = vector<double>{planeEstimator.getEvals()(0),
                                   planeEstimator.getEvals()(1),
                                   planeEstimator.getEvals()(2)};
    
    curv = planeEstimator.getEvals()(2) /
            (planeEstimator.getEvals()(0) + planeEstimator.getEvals()(1) + planeEstimator.getEvals()(2));
    
    // normal including distance from origin
    normal = paramRep;

    correctOrient();
    
    // normalize paramRep
	Misc::normalizeAndUnify(paramRep);

//    cout << "points->size() = " << points->size() << endl;
    hull.reset(new ConcaveHull(points, normal));
//    cout << "hull.getTotalArea() = " << hull->getTotalArea() << endl;
    
    compColorHist();
}

bool ObjInstance::isMatching(const ObjInstance &other,
                             pcl::visualization::PCLVisualizer::Ptr viewer,
                             int viewPort1,
                             int viewPort2) const
{
    double normDot = normal.head<3>().dot(other.getNormal().head<3>());
//            cout << "normDot = " << normDot << endl;
    // if the faces are roughly oriented in the same direction
    if (normDot > 0.707) {
        
        double dist1 = planeEstimator.distance(other.getPlaneEstimator());
        double dist2 = other.getPlaneEstimator().distance(planeEstimator);
        //            double dist1 = mapObj.getEkf().distance(newObj.getEkf().getX());
        //            double dist2 = newObj.getEkf().distance(mapObj.getEkf().getX());
//                cout << "dist1 = " << dist1 << endl;
//                cout << "dist2 = " << dist2 << endl;
        
        //            double diff = Matching::planeEqDiffLogMap(mapObj, newObj, transform);
        //                    cout << "diff = " << diff << endl;
        // if plane equation is similar
        if (dist1 < 0.01 && dist2 < 0.01) {
            
            cv::Mat mapHist = colorHist;
            cv::Mat newHist = other.getColorHist();
            
            double histDist = compHistDist(mapHist, newHist);
//                    cout << "histDist = " << histDist << endl;
            if (histDist < 2.5) {
                
                double intArea = 0.0;
                
                if (viewer) {
                    hull->cleanDisplay(viewer, viewPort1);
                    other.getHull().cleanDisplay(viewer, viewPort2);
                }
    
                Vector7d transform;
                // identity
                transform << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
                double intScore = Matching::checkConvexHullIntersection(*this,
                                                                        other,
                                                                        transform,
                                                                        intArea,
                                                                        viewer,
                                                                        viewPort1,
                                                                        viewPort2);
                if (viewer) {
                    hull->display(viewer, viewPort1);
                    other.getHull().display(viewer, viewPort2);
                }

//                        cout << "intScore = " << intScore << endl;
//                        cout << "intArea = " << intArea << endl;
                // if intersection of convex hulls is big enough
                if (intScore > 0.3) {
                    cout << "merging planes" << endl;
                    // merge the objects
                    return true;
                }
            }
        }
    }
    return false;
}

void ObjInstance::merge(const ObjInstance &other) {
    
    planeEstimator.update(other.getPlaneEstimator().getCentroid(),
                          other.getPlaneEstimator().getCovar(),
                          other.getPlaneEstimator().getNpts());
    Eigen::Vector4d newPlaneEq = planeEstimator.getPlaneEq();
    
    pcl::ModelCoefficients::Ptr mdlCoeff (new pcl::ModelCoefficients);
    mdlCoeff->values.resize(4);
    mdlCoeff->values[0] = newPlaneEq(0);
    mdlCoeff->values[1] = newPlaneEq(1);
    mdlCoeff->values[2] = newPlaneEq(2);
    mdlCoeff->values[3] = newPlaneEq(3);
    
    points->insert(points->end(), other.getPoints()->begin(), other.getPoints()->end());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointsProj(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(points);
    proj.setModelCoefficients(mdlCoeff);
    proj.filter(*pointsProj);
    
    pcl::VoxelGrid<pcl::PointXYZRGB> downsamp;
    downsamp.setInputCloud(pointsProj);
    downsamp.setLeafSize (0.01f, 0.01f, 0.01f);
    downsamp.filter(*pointsProj);
    
    points->swap(*pointsProj);
    
    normal = newPlaneEq;
    correctOrient();
    
    paramRep = newPlaneEq;
    Misc::normalizeAndUnify(paramRep);
    
    *hull = ConcaveHull(points, normal);
    
    compColorHist();

    obsCnt += 1;
}

void ObjInstance::transform(const Vector7d &transform) {
    g2o::SE3Quat transformSE3Quat(transform);
    Eigen::Matrix4d transformMat = transformSE3Quat.to_homogeneous_matrix();
    Eigen::Matrix3d R = transformMat.block<3, 3>(0, 0);
    Eigen::Matrix4d Tinvt = transformMat.inverse();
    Tinvt.transposeInPlace();
    
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
    pcl::transformPointCloud(*points, *points, transformMat);

    // std::vector<PlaneSeg> svs;
    for(PlaneSeg &pseg : svs){
        pseg.transform(transform);
    }

    // Eigen::Vector4d paramRep;
    paramRep = Tinvt * paramRep;
    Misc::normalizeAndUnify(paramRep);
    
    // Eigen::Vector4d normal;
    normal = Tinvt * normal;
    
    // std::vector<Eigen::Vector3d> princComp;
    for(Eigen::Vector3d &pc : princComp){
        pc = R * pc;
    }
    
    // std::vector<double> princCompLens;
    // no need to transform
    
    // double shorterComp;
    // no need to transform
    
    // float curv;
    // no need to transform
    
    // std::shared_ptr<ConcaveHull> hull;
//    cout << "before hull->getTotalArea() = " << hull->getTotalArea() << endl;
    *hull = hull->transform(transform);
//    cout << "after hull->getTotalArea() = " << hull->getTotalArea() << endl;
    
    // std::vector<LineSeg> lineSegs;
    // TODO
    
    // PlaneEstimator planeEstimator;
    planeEstimator.transform(transform);
}



void ObjInstance::correctOrient() {
    bool corrOrient = true;
    int corrCnt = 0;
    int incorrCnt = 0;
    for(int sv = 0; sv < svs.size(); ++sv){
        pcl::PointNormal svPtNormal;
        Eigen::Vector3d svNormal = svs[sv].getSegNormal().cast<double>();
        // if cross product between normal vectors is negative then it is wrongly oriented
        if(svNormal.dot(normal.head<3>()) < 0){
            ++incorrCnt;
        }
        else{
            ++corrCnt;
        }
    }
    if(incorrCnt > corrCnt){
        corrOrient = false;
    }
    if(incorrCnt != 0 && corrCnt != 0){
//        throw PLANE_EXCEPTION("Some normals correct and some incorrect");
        cout << "Some normals correct and some incorrect" << endl;
        for(int sv = 0; sv < svs.size(); ++sv) {
            // if cross product between normal vectors is negative then it is wrongly oriented
            Eigen::Vector3d svNormal = svs[sv].getSegNormal();
            cout << "svNormal[" << sv << "] = " << svNormal.transpose() << endl;
        }
    }
    // flip the normal
    if(!corrOrient){
        normal = -normal;
    }
}

void ObjInstance::compColorHist() {
    // color histogram
    int hbins = 32;
    int sbins = 32;
    int histSizeH[] = {hbins};
    int histSizeS[] = {sbins};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    const float* rangesH[] = {hranges};
    const float* rangesS[] = {sranges};
    int channelsH[] = {0};
    int channelsS[] = {0};
    
    int npts = points->size();
    cv::Mat matPts(1, npts, CV_8UC3);
    for(int p = 0; p < npts; ++p){
        matPts.at<cv::Vec3b>(p)[0] = points->at(p).r;
        matPts.at<cv::Vec3b>(p)[1] = points->at(p).g;
        matPts.at<cv::Vec3b>(p)[2] = points->at(p).b;
    }
    cv::cvtColor(matPts, matPts, cv::COLOR_RGB2HSV);
    cv::Mat hist;
    cv::calcHist(&matPts,
                 1,
                 channelsH,
                 cv::Mat(),
                 hist,
                 1,
                 histSizeH,
                 rangesH);
    // normalization
    hist /= npts;
    hist.reshape(1,hbins);
    
    cv::Mat histS;
    cv::calcHist(&matPts,
                 1,
                 channelsS,
                 cv::Mat(),
                 histS,
                 1,
                 histSizeS,
                 rangesS);
    // normalization
    histS /= npts;
    histS.reshape(1,sbins);
    
    // add S part of histogram
    hist.push_back(histS);
    
    colorHist = hist;
}


double ObjInstance::compHistDist(cv::Mat hist1, cv::Mat hist2) {
//            double histDist = cv::compareHist(frameObjFeats[of], mapObjFeats[om], cv::HISTCMP_CHISQR);
    cv::Mat histDiff = cv::abs(hist1 - hist2);
    return cv::sum(histDiff)[0];
}

void
ObjInstance::display(pcl::visualization::PCLVisualizer::Ptr viewer,
                     int vp,
                     double shading) const
{
    string idStr = to_string(reinterpret_cast<size_t>(this));
    
    viewer->addPointCloud(points, string("obj_instance_") + idStr, vp);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                             shading,
                                             string("obj_instance_") + idStr,
                                             vp);
}

void ObjInstance::cleanDisplay(pcl::visualization::PCLVisualizer::Ptr viewer, int vp) const {
    string idStr = to_string(reinterpret_cast<size_t>(this));
    
    viewer->removePointCloud(string("obj_instance_") + idStr,
                             vp);
}


