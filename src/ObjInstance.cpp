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
#include <Map.hpp>

#include "ObjInstance.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"
#include "Matching.hpp"
#include "UnionFind.h"
#include "EKFPlane.hpp"

using namespace std;

ObjInstance::ObjInstance(int iid,
					ObjType itype,
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr ipoints,
					const std::vector<PlaneSeg>& isvs)
	: id(iid),
	  type(itype),
	  points(ipoints),
	  svs(isvs),
      eolCnt(4),
      obsCnt(1),
      trial(false)
{
    {
        Eigen::MatrixXd pts(3, points->size());
        for(int i = 0; i < points->size(); ++i){
            pts.col(i) = points->at(i).getVector3fMap().cast<double>();
        }
        
        Eigen::Quaterniond q;
        Eigen::Matrix4d covar;
        EKFPlane::compPlaneEqAndCovar(pts, q, covar);
//        Eigen::Vector3d om = Misc::logMap(quat);
//        cout << "q = " << quat.coeffs().transpose() << endl;
//        cout << "om = " << om.transpose() << endl;
//        cout << "covar = " << covarQuat << endl;
    
//        ekf.init(q, covar);
        ekf.init(q, points->size());
    }
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(points);
    
    Eigen::Matrix3f evecs = pca.getEigenVectors();
    Eigen::Vector3f evals = pca.getEigenValues();
    Eigen::Vector4f pcaMean = pca.getMean();
    
    Eigen::Vector3f ev0 = evecs.block<3, 1>(0, 0);
    Eigen::Vector3f ev1 = evecs.block<3, 1>(0, 1);
    Eigen::Vector3f ev2 = evecs.block<3, 1>(0, 2);
    
    // shorter side of the plane is the second largest eigenvalue
    shorterComp = sqrt(evals(1)/points->size());
    
    // the eigenvector for the smallest eigenvalue is the normal vector
    paramRep.head<3>() = ev2.cast<double>();
    // distance is the dot product of normal and point lying on the plane
    paramRep(3) = -ev2.dot(pcaMean.head<3>());
    
    princComp = vector<Eigen::Vector3d>{ev0.cast<double>(), ev1.cast<double>(), ev2.cast<double>()};
    princCompLens = vector<double>{evals(0), evals(1), evals(2)};
    
    curv = evals(2) / (evals(0) + evals(1) + evals(2));
    
    // normal including distance from origin
    normal = paramRep;

    correctOrient();
    
    // normalize paramRep
	Misc::normalizeAndUnify(paramRep);

    hull.reset(new ConcaveHull(points, normal));
    
    compColorHist();
}

void ObjInstance::merge(const ObjInstance &other) {
    
    ekf.update(other.getEkf().getX(), other.getPoints()->size());
    
    const Eigen::Quaterniond &q = ekf.getX();
    Eigen::Vector4d newPlaneEq = q.coeffs();
    double nNorm = newPlaneEq.head<3>().norm();
    newPlaneEq /= nNorm;
    
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
    
    eolCnt += 2;
    obsCnt += 1;
}

void ObjInstance::transform(Vector7d transform) {
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
    
    // EKFPlane ekf;
    // TODO valid only for NOT merged planes (ones where points haven't been projected onto plane)
    {
        Eigen::MatrixXd pts(3, points->size());
        for(int i = 0; i < points->size(); ++i){
            pts.col(i) = points->at(i).getVector3fMap().cast<double>();
        }
        
        Eigen::Quaterniond q;
        Eigen::Matrix4d covar;
        EKFPlane::compPlaneEqAndCovar(pts, q, covar);
//        Eigen::Vector3d om = Misc::logMap(q);
//        cout << "q = " << q.coeffs().transpose() << endl;
//        cout << "om = " << om.transpose() << endl;
//        cout << "covar = " << covar << endl;
        
//        Eigen::Vector4d planeEq = q.coeffs();
//        planeEq /= planeEq.head<3>().norm();
//        cout << "planeEq = " << planeEq.transpose() << endl;
        
        ekf.init(q, points->size());
    }
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
            Eigen::Vector3f svNormal = svs[sv].getSegNormal();
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

void ObjInstance::mergeObjInstances(Map &map,
                                   std::vector<ObjInstance> &newObjInstances,
                                   pcl::visualization::PCLVisualizer::Ptr viewer,
                                   int viewPort1,
                                   int viewPort2)
{
    Vector7d transform;
    // identity
    transform << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    
    for(ObjInstance &newObj : newObjInstances){
        
        vector<list<ObjInstance>::iterator> matches;
        for(auto it = map.begin(); it != map.end(); ++it) {
            ObjInstance &mapObj = *it;
    
            
        
            double dist1 = mapObj.getEkf().distance(newObj.getEkf().getX());
            double dist2 = newObj.getEkf().distance(mapObj.getEkf().getX());
//            cout << "dist1 = " << dist1 << endl;
//            cout << "dist2 = " << dist2 << endl;

//            double diff = Matching::planeEqDiffLogMap(mapObj, newObj, transform);
//                    cout << "diff = " << diff << endl;
            // if plane equation is similar
            if (dist1 < 0.05 || dist2 < 0.05) {
    
                cv::Mat mapHist = mapObj.getColorHist();
                cv::Mat newHist = newObj.getColorHist();

                double histDist = compHistDist(mapHist, newHist);
                cout << "histDist = " << histDist << endl;
                if (histDist < 3.5) {
                    double normDot = mapObj.getNormal().head<3>().dot(newObj.getNormal().head<3>());
                    cout << "normDot = " << normDot << endl;
                    // if the observed face is the same
                    if (normDot > 0) {
                        double intArea = 0.0;
                
                        if (viewer) {
                            mapObj.getHull().cleanDisplay(viewer, viewPort1);
                            newObj.getHull().cleanDisplay(viewer, viewPort2);
                        }
                
                        double intScore = Matching::checkConvexHullIntersection(mapObj,
                                                                                newObj,
                                                                                transform,
                                                                                intArea,
                                                                                viewer,
                                                                                viewPort1,
                                                                                viewPort2);
                        if (viewer) {
                            mapObj.getHull().display(viewer, viewPort1);
                            newObj.getHull().display(viewer, viewPort2);
                        }
                
                        cout << "intScore = " << intScore << endl;
                        cout << "intArea = " << intArea << endl;
                        // if intersection of convex hulls is big enough
                        if (intScore > 0.3) {
                            cout << "merging planes" << endl;
                            // merge the objects
                            matches.push_back(it);
                        }
                    }
                }
            }
        }
        
        if(matches.size() == 0){
            map.addObj(newObj);
        }
        else if(matches.size() == 1){
            ObjInstance &mapObj = *matches.front();
            mapObj.merge(newObj);
        }
        else{
            set<int> matchedIds;
            for(auto it : matches){
                matchedIds.insert(it->getId());
            }
            PendingMatchKey pmatchKey{matchedIds};
            if(map.getPendingMatch(pmatchKey)){
                map.addPendingObj(newObj, matchedIds, matches, 2);
            }
            else{
                map.addPendingObj(newObj, matchedIds, matches, 4);
            }
            
            cout << "Multiple matches" << endl;
        }
    }
    
    map.executePendingMatches(6);
    map.decreasePendingEol(1);
    map.removePendingObjsEol();
    
    map.decreaseObjEol(1);
    map.removeObjsEol();
}

std::list<ObjInstance> ObjInstance::mergeObjInstances(std::vector<std::vector<ObjInstance>>& objInstances,
                                                        pcl::visualization::PCLVisualizer::Ptr viewer,
                                                        int viewPort1,
                                                        int viewPort2)
{
    static constexpr double shadingLevel = 0.01;

    list<ObjInstance> retObjInstances;

    if(viewer){
        viewer->removeAllPointClouds(viewPort1);
        viewer->removeAllShapes(viewPort1);
        viewer->removeAllPointClouds(viewPort2);
        viewer->removeAllShapes(viewPort2);

        viewer->initCameraParameters();
        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, -1.0, 0.0);
    }

    int planesCnt = 0;
    vector<vector<int> > planeIds;
    for(int ba = 0; ba < objInstances.size(); ++ba){
        planeIds.emplace_back(objInstances[ba].size());
        for(int pl = 0; pl < objInstances[ba].size(); ++pl){
            planeIds[ba][pl] = planesCnt++;
        }
    }
    UnionFind ufSets(planesCnt);
    Vector7d transform;
    // identity
    transform << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    for(int ba = 0; ba < objInstances.size(); ++ba){

        if(viewer){
            viewer->removeAllPointClouds(viewPort1);
            viewer->removeAllShapes(viewPort1);

            for(int pl = 0; pl < objInstances[ba].size(); ++pl){
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = objInstances[ba][pl].getPoints();

                viewer->addPointCloud(curPl, string("plane_ba_") + to_string(pl), viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
//                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                         0.1, 0.1, 0.1,
//                                                         string("plane_ba_") + to_string(pl),
//                                                         viewPort1);
            }
        }

        for(int pl = 0; pl < objInstances[ba].size(); ++pl){
            const ObjInstance& curObj = objInstances[ba][pl];
            Eigen::Vector3d curObjNormal = curObj.getNormal().head<3>();

            if(viewer){
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         0.5,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
                
                
                curObj.getHull().display(viewer, viewPort1);
                
            }
            for(int cba = ba; cba < objInstances.size(); ++cba){

                if(viewer){
                    viewer->removeAllPointClouds(viewPort2);
                    viewer->removeAllShapes(viewPort2);

                    for(int cpl = 0; cpl < objInstances[cba].size(); ++cpl){
                        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = objInstances[cba][cpl].getPoints();

                        viewer->addPointCloud(curPl, string("plane_cba_") + to_string(cpl), viewPort2);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 shadingLevel,
                                                                 string("plane_cba_") + to_string(cpl),
                                                                 viewPort2);
//                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                                 0.1, 0.1, 0.1,
//                                                                 string("plane_cba_") + to_string(cpl),
//                                                                 viewPort2);
                    }
                }

                int startCpl = 0;
                if(cba == ba){
                    startCpl = pl + 1;
                }
                for(int cpl = startCpl; cpl < objInstances[cba].size(); ++cpl){
//                    cout << "cpl " << cpl << endl;
                    
                    const ObjInstance& compObj = objInstances[cba][cpl];
                    Eigen::Vector3d compObjNormal = compObj.getNormal().head<3>();

                    if(viewer){
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 0.5,
                                                                 string("plane_cba_") + to_string(cpl),
                                                                 viewPort2);
    
                        compObj.getHull().display(viewer, viewPort2);
//                        cout << compObj.getNormal().transpose() << endl;
//                        cout << compObj.getEkf().getX().coeffs().transpose() << endl;
                        
//                        for(int p = 1; p < chullPolygon.vertices.size(); ++p){
//                            viewer->addLine(chullPointCloud->at(chullPolygon.vertices[p - 1]),
//                                            chullPointCloud->at(chullPolygon.vertices[p]),
//                                            1.0, 0.0, 0.0,
//                                            "cur_line",
//                                            viewPort2);
//                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
//                                                                4,
//                                                                "cur_line",
//                                                                viewPort2);
//                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
//                                                                pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
//                                                                "cur_line",
//                                                                viewPort2);
//                            cout << "point " << chullPointCloud->at(chullPolygon.vertices[p]) << endl;
//
//                            viewer->resetStoppedFlag();
//                            while (!viewer->wasStopped()){
//                                viewer->spinOnce (100);
//                                std::this_thread::sleep_for(std::chrono::milliseconds(50));
//                            }
//
//                            viewer->removeShape("cur_line", viewPort2);
//
////                            cout << "point " << chullPolygon.vertices[p] << endl;
//                        }
                    }

                    double dist1 = curObj.getEkf().distance(compObj.getEkf().getX());
                    double dist2 = compObj.getEkf().distance(curObj.getEkf().getX());
                    cout << "dist1 = " << dist1 << endl;
                    cout << "dist2 = " << dist2 << endl;
                    
                    double diff = Matching::planeEqDiffLogMap(curObj, compObj, transform);
//                    cout << "diff = " << diff << endl;
                    // if plane equation is similar
                    if(dist1 < 50 || dist2 < 50){
                        double normDot = curObjNormal.dot(compObjNormal);
                        cout << "normDot = " << normDot << endl;
                        // if the observed face is the same
                        if(normDot > 0){
                            double intArea = 0.0;
    
                            if(viewer) {
                                curObj.getHull().cleanDisplay(viewer, viewPort1);
                                compObj.getHull().cleanDisplay(viewer, viewPort2);
                            }
                            
                            double intScore = Matching::checkConvexHullIntersection(curObj,
                                                                                    compObj,
                                                                                    transform,
                                                                                    intArea,
                                                                                    viewer,
                                                                                    viewPort1,
                                                                                    viewPort2);
                            if(viewer) {
                                curObj.getHull().display(viewer, viewPort1);
                                compObj.getHull().display(viewer, viewPort2);
                            }
                            
                            cout << "intScore = " << intScore << endl;
                            cout << "intArea = " << intArea << endl;
                            // if intersection of convex hulls is big enough
                            if(intScore > 0.3){
                                cout << "merging planes" << endl;
                                // join the objects
                                ufSets.unionSets(planeIds[ba][pl], planeIds[cba][cpl]);
                            }
                        }
                    }

                    if(viewer) {
                        viewer->resetStoppedFlag();

//                        viewer->initCameraParameters();
//                        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                        while (!viewer->wasStopped()) {
                            viewer->spinOnce(100);
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        }
    
    
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 shadingLevel,
                                                                 string("plane_cba_") +
                                                                 to_string(cpl),
                                                                 viewPort2);
                        compObj.getHull().cleanDisplay(viewer, viewPort2);
                    }
                        
                }
            }

            if(viewer){
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
                 curObj.getHull().cleanDisplay(viewer, viewPort1);
            }
        }
    }

    multimap<int, pair<int, int> > sets;
    for(int ba = 0; ba < objInstances.size(); ++ba) {
        for (int pl = 0; pl < objInstances[ba].size(); ++pl) {
            int setId = ufSets.findSet(planeIds[ba][pl]);
            sets.insert(make_pair(setId, make_pair(ba, pl)));
        }
    }

//    typedef multimap<int, pair<int, int> >::iterator mmIter;
    for(auto it = sets.begin(); it != sets.end(); ){
        if(viewer) {
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();
        }
        
        auto range = sets.equal_range(it->first);

        vector<ObjInstance*> curObjs;
        for(auto rangeIt = range.first; rangeIt != range.second; ++rangeIt){
            curObjs.push_back(&(objInstances[rangeIt->second.first][rangeIt->second.second]));
        }
        // merge all objects with the first one
        for(int o = 1; o < curObjs.size(); ++o){
            curObjs[0]->merge(*curObjs[o]);
        }
    
        retObjInstances.push_back(*curObjs.front());
        
//        if(curObjs.size() == 1){
//            retObjInstances.push_back(*curObjs.front());
//        }
//        else{
//            retObjInstances.push_back(merge(curObjs,
//                                            viewer,
//                                            viewPort1,
//                                            viewPort2));
//        }
        
        it = range.second;
    
//        if(viewer) {
//            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = retObjInstances.back().getPoints();
//            viewer->addPointCloud(curPl, string("plane_ret_") + to_string(it->first), viewPort1);
//        }
    }

    return retObjInstances;
}


double ObjInstance::compHistDist(cv::Mat hist1, cv::Mat hist2) {
//            double histDist = cv::compareHist(frameObjFeats[of], mapObjFeats[om], cv::HISTCMP_CHISQR);
    cv::Mat histDiff = cv::abs(hist1 - hist2);
    return cv::sum(histDiff)[0];
}

//ObjInstance ObjInstance::merge(const std::vector<const ObjInstance*>& objInstances,
//                               pcl::visualization::PCLVisualizer::Ptr viewer,
//                               int viewPort1,
//                               int viewPort2)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr newPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
//    std::vector<PlaneSeg> newSvs;
//
//    Eigen::Vector3d meanLogMap;
//    meanLogMap << 0.0, 0.0, 0.0;
//    int sumPoints = 0;
//    for(int o = 0; o < objInstances.size(); ++o){
//        Eigen::Vector4d curParamRep = objInstances[o]->getParamRep();
//        Eigen::Vector3d logMapParamRep = Misc::logMap(Eigen::Quaterniond(curParamRep));
//        meanLogMap += logMapParamRep * objInstances[o]->getPoints()->size();
//        sumPoints += objInstances[o]->getPoints()->size();
//    }
//    meanLogMap /= sumPoints;
//
//    Eigen::Quaterniond meanParamRep = Misc::expMap(meanLogMap);
//    Eigen::Vector4d meanPlaneEq;
//    meanPlaneEq[0] = meanParamRep.x();
//    meanPlaneEq[1] = meanParamRep.y();
//    meanPlaneEq[2] = meanParamRep.z();
//    meanPlaneEq[3] = meanParamRep.w();
//    double normNorm = meanPlaneEq.head<3>().norm();
//    meanPlaneEq.head<3>() /= normNorm;
//    meanPlaneEq[3] /= normNorm;
//
//    pcl::ModelCoefficients::Ptr mdlCoeff (new pcl::ModelCoefficients);
//    mdlCoeff->values.resize(4);
//    mdlCoeff->values[0] = meanPlaneEq[0];
//    mdlCoeff->values[1] = meanPlaneEq[1];
//    mdlCoeff->values[2] = meanPlaneEq[2];
//    mdlCoeff->values[3] = meanPlaneEq[3];
//    for(int o = 0; o < objInstances.size(); ++o){
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointsProj(new pcl::PointCloud<pcl::PointXYZRGB>);
//        pcl::ProjectInliers<pcl::PointXYZRGB> proj;
//        proj.setModelType(pcl::SACMODEL_PLANE);
//        proj.setInputCloud(objInstances[o]->getPoints());
//        proj.setModelCoefficients(mdlCoeff);
//        proj.filter(*pointsProj);
//
//        pcl::VoxelGrid<pcl::PointXYZRGB> downsamp;
//        downsamp.setInputCloud(pointsProj);
//        downsamp.setLeafSize (0.01f, 0.01f, 0.01f);
//        downsamp.filter(*pointsProj);
//
//        const vector<PlaneSeg>& svs = objInstances[o]->getSvs();
//
//        newPoints->insert(newPoints->end(), pointsProj->begin(), pointsProj->end());
//        newSvs.insert(newSvs.end(), svs.begin(), svs.end());
//
//        if(viewer) {
//            viewer->addPointCloud(objInstances[o]->getPoints(), string("plane_o_") + to_string(o), viewPort1);
//
//        }
//    }
//
//    if(viewer){
//        viewer->addPointCloud(newPoints, string("plane_merged"), viewPort2);
//        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                 1.0, 0.0, 0.0,
//                                                 string("plane_merged"),
//                                                 viewPort2);
//
//        viewer->resetStoppedFlag();
//
////                        viewer->initCameraParameters();
////                        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//        while (!viewer->wasStopped()){
//            viewer->spinOnce (100);
//            std::this_thread::sleep_for(std::chrono::milliseconds(50));
//        }
//
//        for(int o = 0; o < objInstances.size(); ++o){
//            viewer->removePointCloud(string("plane_o_") + to_string(o), viewPort1);
//        }
//        viewer->removePointCloud(string("plane_merged"), viewPort2);
//    }
//
//    return ObjInstance(0,
//                        ObjInstance::ObjType::Plane,
//                        newPoints,
//                        newSvs);
//}
//




