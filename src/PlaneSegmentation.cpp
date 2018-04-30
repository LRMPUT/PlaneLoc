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
#include <random>
#include <algorithm>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <queue>
#include <cmath>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Eigen>
#include <pcl/features/integral_image_normal.h>

#include "PlaneSegmentation.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"


using namespace std;

int PlaneSegmentation::curObjInstId = 0;

void PlaneSegmentation::segment(const cv::FileStorage& fs,
						pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
						pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
						vectorObjInstance& objInstances,
						bool segmentMap,
						pcl::visualization::PCLVisualizer::Ptr viewer,
						int viewPort1,
						int viewPort2)
{
	cout << "Segmentation::segment" << endl;
	chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();

	// parameters
//	float planeDistThresh = (double)fs["segmentation"]["planeDistThresh"];
//	float areaMult = (double)fs["segmentation"]["areaMult"];
//	float smoothCost = (double)fs["segmentation"]["smoothCost"];
//	float minNumInlier = (double)fs["segmentation"]["minNumInlier"];
    float curvThresh = (double)fs["segmentation"]["curvThresh"];
    float stepThresh = (double)fs["segmentation"]["stepThresh"];
    float areaThresh = (double)fs["segmentation"]["areaThresh"];
    float normalThresh = (double)fs["segmentation"]["normalThresh"];
    
    vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> svsInfo;
    
    makeSupervoxels(fs,
                    pcNormals,
                    segmentMap,
                    svsInfo);
    
    UnionFind sets(svsInfo.size());
    
    vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> planeSegs = svsInfo;
    
    mergeSegmentsFF(planeSegs,
                    sets,
                    curvThresh,
                    normalThresh,
                    stepThresh,
                    viewer,
                    viewPort1,
                    viewPort2);

    vector<int> svLabels(svsInfo.size(), -1);
	makeObjInstances(svsInfo,
                     planeSegs,
                     sets,
                     svLabels,
                     pcLab,
                     objInstances,
                     curvThresh,
                     normalThresh,
                     stepThresh,
                     areaThresh,
                     viewer,
                     viewPort1,
                     viewPort2);


	if(viewer){
		cout << "displaying" << endl;
//		pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm = svClust.getLabeledVoxelCloud();
        visualizeSegmentation(svsInfo,
                             planeSegs,
                             svLabels,
                             viewer,
                             viewPort1,
                             viewPort2);
	}

//	if(pcCol != nullptr){
//		pcCol = svClust.getColoredVoxelCloud();
//	}

    chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();

    static chrono::milliseconds totalTime = chrono::milliseconds::zero();
    static int totalCnt = 0;

    if(!segmentMap) {
        totalTime += chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
        ++totalCnt;

        cout << "Mean segmentation time: " << (totalTime.count() / totalCnt) << endl;
    }
}

void PlaneSegmentation::segment(const cv::FileStorage &fs,
                                cv::Mat rgb,
                                cv::Mat depth,
                                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
                                vectorObjInstance &objInstances,
                                pcl::visualization::PCLVisualizer::Ptr viewer,
                                int viewPort1,
                                int viewPort2)
{
    cout << "Segmentation::segment" << endl;
    chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();
    
    // parameters
//	float planeDistThresh = (double)fs["segmentation"]["planeDistThresh"];
//	float areaMult = (double)fs["segmentation"]["areaMult"];
//	float smoothCost = (double)fs["segmentation"]["smoothCost"];
//	float minNumInlier = (double)fs["segmentation"]["minNumInlier"];
    float curvThresh = (double)fs["segmentation"]["curvThresh"];
    float stepThresh = (double)fs["segmentation"]["stepThresh"];
    float areaThresh = (double)fs["segmentation"]["areaThresh"];
    float normalThresh = (double)fs["segmentation"]["normalThresh"];
    
    vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> svsInfo;
    
    makeSupervoxels(fs,
                    rgb,
                    depth,
                    svsInfo);
    
    UnionFind sets(svsInfo.size());
    
//    if(viewer) {
//        drawSegments(viewer,
//                     "cloud_svs",
//                     viewPort1,
//                     svsInfo,
//                     sets);
//
//        viewer->resetStoppedFlag();
//        viewer->initCameraParameters();
//        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//        viewer->spinOnce(100);
//        while (!viewer->wasStopped()) {
//            viewer->spinOnce(50);
//            cv::waitKey(50);
//            std::this_thread::sleep_for(std::chrono::milliseconds(50));
//        }
//
//        viewer->removePointCloud("cloud_svs", viewPort1);
//    }
    
    vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> planeSegs = svsInfo;
    
    mergeSegmentsFF(planeSegs,
                    sets,
                    curvThresh,
                    normalThresh,
                    stepThresh/*,
                    viewer,
                    viewPort1,
                    viewPort2*/);
    
    vector<int> svLabels(svsInfo.size(), -1);
    makeObjInstances(svsInfo,
                     planeSegs,
                     sets,
                     svLabels,
                     pcLab,
                     objInstances,
                     curvThresh,
                     normalThresh,
                     stepThresh,
                     areaThresh,
                     viewer,
                     viewPort1,
                     viewPort2);
    
    
    if(viewer){
        cout << "displaying" << endl;
//		pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm = svClust.getLabeledVoxelCloud();
        visualizeSegmentation(svsInfo,
                              planeSegs,
                              svLabels,
                              viewer,
                              viewPort1,
                              viewPort2);
    }

//	if(pcCol != nullptr){
//		pcCol = svClust.getColoredVoxelCloud();
//	}
    
    chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();
    
    static chrono::milliseconds totalTime = chrono::milliseconds::zero();
    static int totalCnt = 0;
    
    totalTime += chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    ++totalCnt;
    
    cout << "Mean segmentation time: " << (totalTime.count() / totalCnt) << endl;
}


void
PlaneSegmentation::makeSupervoxels(const cv::FileStorage &fs,
                                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
                                  bool segmentMap,
                                  std::vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> &svs)
{
    // compute normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    {
        for(int p = 0; p < pcNormals->size(); ++p){
            pcl::Normal norm;
            norm.normal_x = pcNormals->at(p).normal_x;
            norm.normal_y = pcNormals->at(p).normal_y;
            norm.normal_z = pcNormals->at(p).normal_z;
//            cout << "norm.norm() = " << norm.getNormalVector3fMap().transpose() << endl;
            normals->push_back(norm);
        }
    }
//	{
//		  // Create the normal estimation class, and pass the input dataset to it
//		  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//		  ne.setInputCloud(pc);
//
//		  // Create an empty kdtree representation, and pass it to the normal estimation object.
//		  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//		  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
//		  ne.setSearchMethod(tree);
//
//		  // Use all neighbors in a sphere of radius 3cm
//		  ne.setRadiusSearch(0.03);
//
//		  // Compute the features
//		  ne.compute(*pointCloudNorm);
//
//	}
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>());
    for(int p = 0; p < pcNormals->size(); ++p){
        pcl::PointXYZRGB pt;
        pt.x = pcNormals->at(p).x;
        pt.y = pcNormals->at(p).y;
        pt.z = pcNormals->at(p).z;
        pt.r = pcNormals->at(p).r;
        pt.g = pcNormals->at(p).g;
        pt.b = pcNormals->at(p).b;
        pc->push_back(pt);
    }
    
    double svRes = 0.006;
    double svSize = 0.2;
    bool useSingleCameraTrans = true;
    if (segmentMap) {
        svRes = 0.01;
        svSize = 0.2;
        useSingleCameraTrans = false;
    }
    
    cout << "Creating supervoxels" << endl;
    map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> idToSv;
    pcl::SupervoxelClustering<pcl::PointXYZRGB> svClust(svRes, svSize);
    svClust.setUseSingleCameraTransform(useSingleCameraTrans);
    svClust.setInputCloud(pc);
    svClust.setNormalCloud(normals);
    svClust.setColorImportance(1.0);
    svClust.setSpatialImportance(0.4);
    svClust.setNormalImportance(1.0);
//	svClust.set
    svClust.extract(idToSv);
    
    svClust.refineSupervoxels(1, idToSv);
    
    vector<uint32_t> idxToId;
    map<uint32_t, uint32_t> idToIdx;
    int idxCnt = 0;
    for(auto it = idToSv.begin(); it != idToSv.end(); ++it){
        // ignore label 0
        if(it->first != 0) {
            idxToId.push_back(it->first);
            idToIdx[it->first] = idxCnt;
//                cout << "idToIdx[" << it->first << "] = " << idxCnt << endl;
            ++idxCnt;
        }
    }
    
    cout << "Adding points to svsInfo" << endl;
    svs.resize(idxCnt);
    
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm = svClust.getLabeledCloud();
    for(int p = 0; p < pcLabSegm->size(); ++p){
        // ignore label 0
        int svId = pcLabSegm->at(p).label;
        if(svId != 0) {
            if (idToIdx.count(svId) > 0) {
                int svIdx = idToIdx.at(svId);
//                cout << "svId = " << svId << ", svIdx = " << svIdx << endl;
                svs[svIdx].addPointAndNormal(pc->at(p), normals->at(p));
            } else {
                cout << "svId = " << svId << " not found!!!" << endl;
            }
        }
    }
    
    cout << "Calculating segment properties" << endl;
    for(int sv = 0; sv < svs.size(); ++sv){
        svs[sv].setId(sv);
        svs[sv].setLabel(0);
        svs[sv].calcSegProp();
    }
    
    cout << "Adding edges" << endl;
    std::multimap<uint32_t, uint32_t> adjMultimap;
    svClust.getSupervoxelAdjacency(adjMultimap);
    for(auto it = adjMultimap.begin(); it != adjMultimap.end(); ++it) {
        int srcIdx = idToIdx[it->first];
        int tarIdx = idToIdx[it->second];
        
        svs[srcIdx].addAdjSeg(tarIdx);
        svs[tarIdx].addAdjSeg(srcIdx);
    }
}

void PlaneSegmentation::makeSupervoxels(const cv::FileStorage &fs,
                                        cv::Mat rgb,
                                        cv::Mat depth,
                                        std::vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> &svs)
{
    cv::Mat camMat;
    fs["planeSlam"]["cameraMatrix"] >> camMat;
    double curvThresh = fs["segmentation"]["curvThresh"];
    
    cv::Mat rgbSegments = segmentRgb(rgb, depth, 0.8, 20, 200);
    
    int nrows = rgbSegments.rows;
    int ncols = rgbSegments.cols;
    
    {
        cv::Mat filtRgbSegments = rgbSegments.clone();
        
        int kSize = 5;
        // majority filter
        for(int r = 0; r < nrows; ++r) {
            for (int c = 0; c < ncols; ++c) {
                if(rgbSegments.at<int>(r, c) >= 0) {
                    map<int, int> votes;
                    for (int dr = -kSize / 2; dr <= kSize / 2; ++dr) {
                        for (int dc = -kSize / 2; dc <= kSize / 2; ++dc) {
                            int cr = r + dr;
                            int cc = c + dc;
                            if ((cr < nrows) && (cr >= 0) &&
                                (cc < ncols) && (cc >= 0))
                            {
                                int cid = rgbSegments.at<int>(cr, cc);
                                if(votes.count(cid) == 0){
                                    votes[cid] = 1;
                                }
                                else{
                                    ++votes[cid];
                                }
                            }
                        }
                    }
                    int bestId = -1;
                    int bestVotes = -1;
                    for(auto &p : votes){
                        if(p.second > bestVotes){
                            bestVotes = p.second;
                            bestId = p.first;
                        }
                    }
//                    if(bestId != )
                    filtRgbSegments.at<int>(r, c) = bestId;
                }
            }
        }
        
        rgbSegments = filtRgbSegments;
    }
    
    int nhood[][2] = {{-1, 1},
                      {1, 0},
                      {1, 1},
                      {0, 1}};
    
    int maxId = 0;
    map<pair<int, int>, int> edges;
    for(int r = 0; r < nrows; ++r){
        for(int c = 0; c < ncols; ++c){
            int id = rgbSegments.at<int>(r, c);
    
            maxId = std::max(maxId, id);
            
            for(int nh = 0; nh < sizeof(nhood)/sizeof(nhood[0]); nh++){
                int nhr = r + nhood[nh][0];
                int nhc = c + nhood[nh][1];
                
                if((nhr < nrows) && (nhr >= 0) &&
                   (nhc < ncols) && (nhc >= 0) &&
                    rgbSegments.at<uint8_t>(r, c) >= 0 &&
                    rgbSegments.at<uint8_t>(nhr, nhc) >= 0)
                {
                    int nhId = rgbSegments.at<int>(nhr, nhc);
                    
                    if(nhId != id) {
                        pair<int, int> e = make_pair(std::min(id, nhId), std::max(id, nhId));
                        edges[e] += 1;
                    }
                }
            }
        }
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>(rgb.cols, rgb.rows));
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>(rgb.cols, rgb.rows));
    cv::Mat xyz = Misc::projectTo3D(depth, camMat);
    
    for(int r = 0; r < rgb.rows; ++r) {
        for (int c = 0; c < rgb.cols; ++c) {
            pcl::PointXYZRGB p;
//					((uint8_t)rgb.at<Vec3b>(r, c)[0],
//										(uint8_t)rgb.at<Vec3b>(r, c)[1],
//										(uint8_t)rgb.at<Vec3b>(r, c)[2]);
            p.x = xyz.at<cv::Vec3f>(r, c)[0];
            p.y = xyz.at<cv::Vec3f>(r, c)[1];
            p.z = xyz.at<cv::Vec3f>(r, c)[2];
            p.r = (uint8_t) rgb.at<cv::Vec3b>(r, c)[0];
            p.g = (uint8_t) rgb.at<cv::Vec3b>(r, c)[1];
            p.b = (uint8_t) rgb.at<cv::Vec3b>(r, c)[2];
            //					cout << "Point at (" << xyz.at<Vec3f>(r, c)[0] << ", " <<
            //											xyz.at<Vec3f>(r, c)[1] << ", " <<
            //											xyz.at<Vec3f>(r, c)[2] << "), rgb = (" <<
            //											(int)rgb.at<Vec3b>(r, c)[0] << ", " <<
            //											(int)rgb.at<Vec3b>(r, c)[1] << ", " <<
            //											(int)rgb.at<Vec3b>(r, c)[2] << ") " << endl;
            pointCloud->at(c, r) = p;
        }
    }
    
    svs.resize(maxId + 1);

    
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(20.0f);
    ne.setInputCloud(pointCloud);
    ne.setViewPoint(0.0, 0.0, 0.0);
    
    ne.compute(*normals);
    
    for(int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            int id = rgbSegments.at<int>(r, c);
            
            const pcl::PointXYZRGB &curPt = pointCloud->at(c, r);
            const pcl::Normal &curNorm = normals->at(c, r);
            
            if(id >= 0 &&
               !isnan(curPt.x) && !isnan(curPt.y) && !isnan(curPt.z) &&
               abs(curPt.z) >= 1e-3 &&
               !isnan(curNorm.normal_x) && !isnan(curNorm.normal_y) && !isnan(curNorm.normal_z))
            {
                svs[id].addPointAndNormal(pointCloud->at(c, r), normals->at(c, r));
            }
        }
    }
    
    {
        map<int, int> oldIdxToNewIdx;
        std::vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> newSvs;
        int svsCnt = 0;
        for(int sv = 0; sv < svs.size(); ++sv){
            svs[sv].calcSegProp(true);
            
            if(svs[sv].getPoints()->size() > 50) {
                if (!isnan(svs[sv].getSegCurv()) &&
                    svs[sv].getSegCurv() < 2 * curvThresh)
                {
                    const vector<double> &evals = svs[sv].getEvals();
                    
                    double varD = evals[2] / svs[sv].getPoints()->size();
                    double varX = evals[2] / evals[0];
                    double varY = evals[2] / evals[1];
//                    if(evals.size() < 3){
//                        throw PLANE_EXCEPTION("evals.size() < 3");
//                    }
                    if (varX < 0.4 && varY < 0.4) {
    
                        int newIdx = svsCnt++;
    
                        newSvs.push_back(svs[sv]);
                        newSvs.back().setId(newIdx);
                        oldIdxToNewIdx[sv] = newIdx;
                    }
                }
            }
        }
        svs.swap(newSvs);
    
        map<pair<int, int>, int> newEdges;
        for(const pair<pair<int, int>, int> &cure : edges){
            if(oldIdxToNewIdx.count(cure.first.first) > 0 &&
               oldIdxToNewIdx.count(cure.first.second) > 0)
            {
               newEdges.insert(make_pair(make_pair(oldIdxToNewIdx[cure.first.first], oldIdxToNewIdx[cure.first.second]), cure.second));
            }
        }
        edges.swap(newEdges);
    }
    
    for(const pair<pair<int, int>, int> &cure : edges){
        // if edge is strong enough than add it to svs
        static constexpr int edgeStrengthThresh = 50;
        if(cure.second > edgeStrengthThresh) {
            int u = cure.first.first;
            int v = cure.first.second;
            svs[u].addAdjSeg(v);
            svs[v].addAdjSeg(u);
        }
    }
//    for(int sv = 0; sv < svs.size(); ++sv){
//        svs[sv].calcSegProp();
//    }
    
//    cv::Mat segCol = Misc::colorIdsWithLabels(rgbSegments);
    cv::Mat segCol = Misc::colorIds(rgbSegments);
    
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    cv::imshow("original", bgr);
    cv::imshow("rgb segments", segCol);
}

void PlaneSegmentation::makeObjInstances(const vectorPlaneSeg &svs,
                                         const vectorPlaneSeg &segs,
                                         UnionFind &sets,
                                         std::vector<int> &svLabels,
                                         pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
                                         vectorObjInstance &objInstances,
                                         float curvThresh,
                                         double normalThresh,
                                         double stepThresh,
                                         float areaThresh,
                                         pcl::visualization::PCLVisualizer::Ptr viewer,
                                         int viewPort1,
                                         int viewPort2)
{
    cout << "making obj instances" << endl;
    
    vector<vector<int> > planeCandidates(svs.size());
    vector<int> planeCandidatesNpts(svs.size(), 0);
    vector<double> planeCandidatesAreaEst(svs.size(), 0.0);
    for(int sv = 0; sv < svs.size(); ++sv){
        int set = sets.findSet(sv);
//            svLabels[sv] = set;
        planeCandidates[set].push_back(sv);
        planeCandidatesNpts[set] += svs[sv].getPoints()->size();
        planeCandidatesAreaEst[set] += svs[sv].getAreaEst();
    }
    
    for(int pl = 0; pl < segs.size(); ++pl){
//        cout << "planeCandidates[" << pl << "].size() = " << planeCandidates[pl].size() << endl;
//        cout << "planeCandidatesNpts[" << pl << "] = " << planeCandidatesNpts[pl] << endl;
////        cout << "planeCandidatesAreaEst[" << pl << "] = " << planeCandidatesAreaEst[pl] << endl;
//        cout << "planeSegs[" << pl << "].getSegCurv() = " << segs[pl].getSegCurv() << endl;
        
        if(planeCandidates[pl].size() > 0){
            if(planeCandidatesNpts[pl] > 1500){
                if(segs[pl].getSegCurv() < curvThresh){
//                    cout << "Adding with lab = " << pl << endl;
                    int lab = pl;
                    
                    vectorPlaneSeg curSvs;
//                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
                    
                    for(int s = 0; s < planeCandidates[pl].size(); ++s){
                        curSvs.push_back(svs[planeCandidates[pl][s]]);
                        svLabels[planeCandidates[pl][s]] = lab;
                    }
//                        cout << "svLabels[pl] = " << svLabels[pl] << endl;
//                        cout << "planeSegs[pl].getPoints()->size() = " << planeSegs[pl].getPoints()->size() << endl;
    
                    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr curPoints = segs[pl].getPoints();
//                    *curPoints = *(segs[pl].getPoints());
                    for(int p = 0; p < curPoints->size(); ++p){
                        const pcl::PointXYZRGB &pcPt = curPoints->at(p);
                        pcl::PointXYZRGBL pt;
                        pt.rgb = pcPt.rgb;
                        pt.x = pcPt.x;
                        pt.y = pcPt.y;
                        pt.z = pcPt.z;
                        pt.label = lab;
                        pcLab->push_back(pt);
                    }
                    
                    objInstances.emplace_back(curObjInstId++,
                                              ObjInstance::ObjType::Plane,
                                              curPoints,
                                              curSvs);
                    {
//                        Eigen::Vector4d planeEq = objInstances.back().getNormal();
//                        Eigen::Vector3d centroid = objInstances.back().getCentroid();
                        // the area has to greater than threshold
                        if(objInstances.back().getHull().getTotalArea() < areaThresh)
                        {
//                            cout << "removing, area = " << objInstances.back().getHull().getTotalArea() << endl;
                            // remove last element
                            objInstances.erase(objInstances.end() - 1);
                        }
                    }
                }
            }
        }
    }
}

cv::Mat
PlaneSegmentation::segmentRgb(cv::Mat rgb, cv::Mat depth, float sigma, float k, int minSegment)
{
    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point endSorting;
    high_resolution_clock::time_point endComp;
    high_resolution_clock::time_point endMerging;
    
    cv::Mat mask = (depth > 0.2) & (depth < 4.0);
    
//    cv::Mat imageR(rgb.rows, rgb.cols, CV_32FC1);
//    cv::Mat imageG(rgb.rows, rgb.cols, CV_32FC1);
//    cv::Mat imageB(rgb.rows, rgb.cols, CV_32FC1);
//    cv::Mat imageChannels[] = {imageR, imageG, imageB};
    cv::Mat imageFilt(rgb.rows, rgb.cols, CV_32FC3);
    cv::Mat depthFilt = depth.clone();
    
//    int nchannels = 3;
    int nhood[][2] = {{-1, 1},
                      {1, 0},
                      {1, 1},
                      {0, 1}};
    /*int nhood[][2] = {{1, 0},
                    {0, 1}};*/
    //cout << "Size of nhood " << sizeof(nhood)/sizeof(nhood[0]) << endl;
    //cout << "rows: " << image.rows << ", cols: " << image.cols << endl;

    rgb.convertTo(imageFilt, CV_32F);
    // 0 - 255 range
    cv::cvtColor(imageFilt, imageFilt, cv::COLOR_RGB2GRAY);
//    double minVal, maxVal;
//    cv::minMaxIdx(imageFloat, &minVal, &maxVal);
//    cout << "minVal = " << minVal << endl;
//    cout << "maxVal = " << maxVal << endl;
    //resize(imageFloat, imageFloat, Size(320, 240));
    GaussianBlur(imageFilt, imageFilt, cv::Size(0, 0), sigma);
    GaussianBlur(depthFilt, depthFilt, cv::Size(0, 0), sigma);
//    split(imageFloat, imageChannels);
    
    int nrows = imageFilt.rows;
    int ncols = imageFilt.cols;
    
    cv::Mat segments(nrows, ncols, CV_32SC1, cv::Scalar(-1));
    
    vector<SegEdge> edges;
    for(int r = 0; r < nrows; r++){
        for(int c = 0; c < ncols; c++){
            for(int nh = 0; nh < sizeof(nhood)/sizeof(nhood[0]); nh++){
                int nhr = r + nhood[nh][0];
                int nhc = c + nhood[nh][1];
                if((nhr < nrows) && (nhr >= 0) &&
                   (nhc < ncols) && (nhc >= 0) &&
                    mask.at<uint8_t>(r, c) > 0 &&
                    mask.at<uint8_t>(nhr, nhc) > 0)
                {
//                    float diffAll = 0;
//                    for(int ch = 0; ch < nchannels; ch++){
//                        float diff = abs(imageChannels[ch].at<float>(r, c) - imageChannels[ch].at<float>(r + nhood[nh][0], c + nhood[nh][1]));
//                        diffAll += diff*diff;
//                    }
//                    diffAll = sqrt(diffAll);
                    float diffRgb = imageFilt.at<float>(r, c) - imageFilt.at<float>(nhr, nhc);
                    float diffDepth = (depthFilt.at<float>(r, c) - depthFilt.at<float>(nhr, nhc))/depthFilt.at<float>(r, c) ;
                    
                    edges.push_back(SegEdge(c + ncols*r, nhc + ncols*nhr, 0.5*abs(diffRgb) + 0.5*64*abs(diffDepth)));
                    //if(edges.back().i == 567768 || edges.back().j == 567768){
                    //	cout << "diff = abs(" << (int)imageChannels[ch].at<unsigned char>(r, c) << " - " << (int)imageChannels[ch].at<unsigned char>(r + nhood[nh][0], c + nhood[nh][1]) << ") = " << diff << endl;
                    //}
                }
            }
        }
    }
    sort(edges.begin(), edges.end());
    
    endSorting = high_resolution_clock::now();
    cout << "End sorting" << endl;
    
    //cout << "Channel " << ch << endl;
    
    //cout << "Largest differece = " << edges[edges.size() - 1].weight <<
    //		", between (" << edges[edges.size() - 1].i << ", " << edges[edges.size() - 1].j <<
    //		")" << endl;
    
    UnionFind sets(nrows * ncols);
    vector<float> intDiff;
    intDiff.assign(nrows * ncols, 0);
    for(vector<SegEdge>::iterator it = edges.begin(); it != edges.end(); it++){
        int uRoot = sets.findSet(it->u);
        int vRoot = sets.findSet(it->v);
        //cout << "i = " << it->i << ", j = " << it->j << ", weight = " << it->weight << endl;
        if(uRoot != vRoot){
            //cout << "intDiff[uRoot] + (float)k/sizes[uRoot] = " << intDiff[uRoot] << " + " << (float)k/sizes[uRoot] << " = " << intDiff[uRoot] + (float)k/sizes[uRoot] << endl;
            //cout << "intDiff[vRoot] + (float)k/sizes[vRoot] = " << intDiff[vRoot] << " + " << (float)k/sizes[vRoot] << " = " << intDiff[vRoot] + (float)k/sizes[vRoot] << endl;
            if(min(intDiff[uRoot] + (float)k/sets.size(uRoot), intDiff[vRoot] + (float)k/sets.size(vRoot))
               >=
               it->w)
            {
                //cout << "union " << min(intDiff[uRoot] + (float)k/sizes[uRoot], intDiff[vRoot] + (float)k/sizes[vRoot]) << " >= " << it->weight << endl;
                int newRoot = sets.unionSets(uRoot, vRoot);
                intDiff[newRoot] = it->w;
            }
        }
    }
    cout << "Mergining small segments" << endl;
    for(vector<SegEdge>::iterator it = edges.begin(); it != edges.end(); it++){
        int iRoot = sets.findSet(it->u);
        int jRoot = sets.findSet(it->v);
        if((iRoot != jRoot) && ((sets.size(iRoot) < minSegment) || (sets.size(jRoot) < minSegment))){
            sets.unionSets(iRoot, jRoot);
        }
    }
    
    cout << "Counting elements" << endl;
//    set<int> numElements;
    map<int, int> idToIdx;
    int idxCnt = 0;
    for(int r = 0; r < nrows; r++){
        for(int c = 0; c < ncols; c++){
            if(mask.at<uint8_t>(r, c) > 0) {
                int id = sets.findSet(c + ncols * r);
                if (idToIdx.count(id) == 0) {
                    idToIdx[id] = idxCnt++;
                }
                int idx = idToIdx[id];
    
                segments.at<int>(r, c) = idx;
//            numElements.insert(sets.findSet(c + ncols*r));
            }
        }
    }
    cout << "number of elements = " << idToIdx.size() << endl;
    
    endComp = high_resolution_clock::now();
    
    endMerging = high_resolution_clock::now();
    
    static duration<double> sortingTime = duration<double>::zero();
    static duration<double> compTime = duration<double>::zero();
    static duration<double> mergingTime = duration<double>::zero();
    static duration<double> wholeTime = duration<double>::zero();
    static int times = 0;
    
    sortingTime += duration_cast<duration<double> >(endSorting - start);
    compTime += duration_cast<duration<double> >(endComp - endSorting);
    mergingTime += duration_cast<duration<double> >(endMerging - endComp);
    wholeTime += duration_cast<duration<double> >(endMerging - start);
    times++;
    cout << "Segment Times: " << times << endl;
    cout << "Segment Average sorting time: " << sortingTime.count()/times << endl;
    cout << "Segment Average computing time: " << compTime.count()/times << endl;
    cout << "Segment Average merging time: " << mergingTime.count()/times << endl;
    cout << "Segment Average whole time: " << wholeTime.count()/times << endl;
    
    return segments;
}

cv::Mat PlaneSegmentation::segmentRgb2(cv::Mat rgb, cv::Mat depth) {
    return cv::Mat();
}

void PlaneSegmentation::mergeSegments(vectorPlaneSeg &segs,
                                      UnionFind &sets,
                                      double curvThresh,
                                      double normalThresh,
                                      double stepThresh,
                                      pcl::visualization::PCLVisualizer::Ptr viewer,
                                      int viewPort1,
                                      int viewPort2)
{
    double shadingLevel = 16.0/256;
    
    vector<bool> visited(segs.size(), false);
    map<pair<int, int>, int> edgeVer;
//    vector<pair<int, int>> edges;
    priority_queue<SegEdge> bestEdges;
    for(int ps = 0; ps < segs.size(); ++ps){
        int curSeg = sets.findSet(ps);
        
        if(!visited[curSeg]) {
            for (int n = 0; n < segs[ps].getAdjSegs().size(); ++n) {
                int nh = sets.findSet(segs[ps].getAdjSegs()[n]);
        
                int u = std::min(curSeg, nh);
                int v = std::max(curSeg, nh);
                
                if(edgeVer.count(make_pair(u, v)) == 0){
                    edgeVer[make_pair(u, v)] = 1;
                    
                    double score = compEdgeScore(segs[curSeg],
                                                 segs[nh],
                                                 curvThresh,
                                                 normalThresh,
                                                 stepThresh);
                    
//                    if(u == 86 && v == 89){
//                        cout << "segs[curSeg].getSegCurv() = " << segs[curSeg].getSegCurv() << endl;
//                        cout << "segs[nh].getSegCurv() = " << segs[nh].getSegCurv() << endl;
//                        cout << "score = " << score << endl;
//                    }
                    SegEdge curEdge(u, v, 1, score);
                    
                    bestEdges.push(curEdge);
                }
            }
            
            visited[curSeg] = true;
        }
    }
    
    vector<pair<int, int>> toMerge;
    while(!bestEdges.empty()){
        const SegEdge &curEdge = bestEdges.top();
        bestEdges.pop();
    
        int u = sets.findSet(curEdge.u);
        int v = sets.findSet(curEdge.v);
        
        pair<int, int> ep = make_pair(min(u, v), max(u, v));
        int newestVer = edgeVer[ep];

        if(viewer){
            drawSegments(viewer,
                         "cloud_lab",
                         viewPort1,
                         segs,
                         sets);
    
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     shadingLevel,
                                                     "cloud_lab",
                                                     viewPort1);
            
            viewer->removePointCloud("cloud_1", viewPort1);
            viewer->addPointCloud(segs[u].getPoints(), "cloud_1", viewPort1);
            viewer->removePointCloud("cloud_2", viewPort1);
            viewer->addPointCloud(segs[v].getPoints(), "cloud_2", viewPort1);
            
        }
        cout << "edge (" << u << ", " << v << ")" << endl;
        cout << "curEdge.w = " << curEdge.w << endl;

        cout << "newestVer = " << newestVer << endl;
        cout << "curEdge.ver = " << curEdge.ver << endl;
        
        bool newestEdge = false;
        
        // if the newest version
        if(u != v && newestVer == curEdge.ver) {
            newestEdge = true;
            if (checkIfCoplanar(segs[u], segs[v], curvThresh, normalThresh, stepThresh)) {
                
                // update versions
                map<pair<int, int>, int> prevEdgeVer = edgeVer;
                int highestVer = 0;
                vector<pair<int, int>> nhEdges;

                for (int n = 0; n < segs[u].getAdjSegs().size(); ++n) {
                    int nh = sets.findSet(segs[u].getAdjSegs()[n]);

                    nhEdges.emplace_back(min(u, nh), max(u, nh));
                }
                for (int n = 0; n < segs[v].getAdjSegs().size(); ++n) {
                    int nh = sets.findSet(segs[v].getAdjSegs()[n]);

                    nhEdges.emplace_back(min(v, nh), max(v, nh));
                }

                for(int ep = 0; ep < nhEdges.size(); ++ep){
                    highestVer = max(highestVer, edgeVer[nhEdges[ep]]);
                }
//                for(int ep = 0; ep < nhEdges.size(); ++ep){
//                    edgeVer[nhEdges[ep]] = highestVer + 1;
//                }
                
                // unite sets
                int m = sets.unionSets(u, v);
                cout << "Merging segments" << endl;
                segs[m] = segs[u].merge(segs[v], sets);
                cout << "end merging segments" << endl;
                
                // add new edges to heap
                for (int n = 0; n < segs[m].getAdjSegs().size(); ++n) {
                    int nh = sets.findSet(segs[m].getAdjSegs()[n]);
                    
                    pair<int, int> ep = make_pair(min(m, nh), max(m, nh));
                    // still the same version -> adding edge
                    bool addEdge = false;
                    if(edgeVer.count(ep) == 0){
                        addEdge = true;
                    }
                    else if(prevEdgeVer[ep] >= edgeVer[ep]){
                        addEdge = true;
                    }
                    
                    if(addEdge){
                        double score = compEdgeScore(segs[m],
                                                     segs[nh],
                                                     curvThresh,
                                                     normalThresh,
                                                     stepThresh);
                        int ver = highestVer + 1;
                        bestEdges.push(SegEdge(ep.first, ep.second, ver, score));
                        
                        edgeVer[ep] = ver;
                    }
                }
            }
        }
        
        if(viewer && newestEdge){
            viewer->resetStoppedFlag();
            viewer->spinOnce (100);
            while (!viewer->wasStopped()){
                viewer->spinOnce (100);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }
    
    
}

void PlaneSegmentation::mergeSegmentsFF(vectorPlaneSeg &segs,
                                        UnionFind &sets,
                                        double curvThresh,
                                        double normalThresh,
                                        double stepThresh,
                                        pcl::visualization::PCLVisualizer::Ptr viewer,
                                        int viewPort1,
                                        int viewPort2)
{
    cout << "flood fill through segments" << endl;
    
    vector<bool> isVisited(segs.size(), false);
//    vector<vector<int> > planeCandidates;
    for(int sv = 0; sv < segs.size(); ++sv){
        // if not visited and planar enough
        if(!isVisited[sv] && segs[sv].getSegCurv() < curvThresh){
            queue<int> nodeQ;
            nodeQ.push(sv);
            isVisited[sv] = true;
            
            vector<int> curVisited;
            float curVisitedAreaCoeff = 0.0;
            while(!nodeQ.empty()){
                int curIdx = nodeQ.front();
                nodeQ.pop();
                
//                cout << "curIdx = " << curIdx << endl;
                
                curVisited.push_back(curIdx);
                curVisitedAreaCoeff += segs[curIdx].getAreaEst();
                for(int nh = 0; nh < segs[curIdx].getAdjSegs().size(); ++nh){
                    int nhIdx = segs[curIdx].getAdjSegs()[nh];
//                    cout << "nhIdx = " << nhIdx << endl;
                    // if not visited
                    if(!isVisited[nhIdx])
                    {
                        // if planar enough
                        if(segs[nhIdx].getSegCurv() < 2*curvThresh) {
                            Eigen::Vector3d centrVec = segs[nhIdx].getSegCentroid() -
                                                       segs[curIdx].getSegCentroid();
                            float step1 = std::fabs(centrVec.dot(segs[nhIdx].getSegNormal()));
                            float step2 = std::fabs((-centrVec).dot(segs[curIdx].getSegNormal()));
                            if(step1 < stepThresh && step2 < stepThresh){
                                float normalScore = segs[curIdx].getSegNormal().dot(segs[nhIdx].getSegNormal());
                                
                                if(normalScore > normalThresh) {
                                    Eigen::Vector3d centrDir = centrVec.normalized();
                                    // variance of points scatter in a direction
                                    // of the line that connects centroids
                                    float varCentrDir1 = centrDir.transpose() * segs[curIdx].getSegCovar()
                                                         * centrDir;
                                    float varCentrDir2 = centrDir.transpose() * segs[nhIdx].getSegCovar()
                                                         * centrDir;
//                                    cout << "(" << curIdx << ", " << nhIdx << ")" << endl;
//                                    cout << "stddevCentrDir1 = " << sqrt(varCentrDir1) << endl;
//                                    cout << "stddevCentrDir2 = " << sqrt(varCentrDir2) << endl;
//                                    cout << "centrVec.norm() = " << centrVec.norm() << endl;
                                    
                                    nodeQ.push(nhIdx);
                                    isVisited[nhIdx] = true;
                                }
                            }
                            
                        }
                    }
                }
            }
            
//            cout << "curVisited.size() = " << curVisited.size() << endl;
            for(int s = 1; s < curVisited.size(); ++s){
                int curSeg = sets.findSet(curVisited[s]);
                int prevSeg = sets.findSet(curVisited[s - 1]);
//                cout << "merging " << curSeg << " and " << prevSeg << endl;
//                cout << "segs[" << curSeg << "].getPoints()->size() = " << segs[curSeg].getPoints()->size() << endl;
//                cout << "segs[" << prevSeg << "].getPoints()->size() = " << segs[prevSeg].getPoints()->size() << endl;
                int m = sets.unionSets(curSeg, prevSeg);
                segs[m] = segs[curSeg].merge(segs[prevSeg], sets);
//                cout << "merged and inserted at " << m << endl;
//                cout << "segs[" << m << "].getPoints()->size() = " << segs[m].getPoints()->size() << endl;
            }
        }
    }
}

bool PlaneSegmentation::checkIfCoplanar(const PlaneSeg &seg1,
                                        const PlaneSeg &seg2,
                                        double curvThresh,
                                        double normalThresh,
                                        double stepThresh)
{
    bool coplanar = false;
    cout << "seg1.getSegCurv() = " << seg1.getSegCurv() << endl;
    cout << "seg2.getSegCurv() = " << seg2.getSegCurv() << endl;
    // if planar enough
    if((seg1.getSegCurv() < curvThresh && seg2.getSegCurv() < 2*curvThresh) ||
       (seg1.getSegCurv() < 2*curvThresh && seg2.getSegCurv() < curvThresh))
    {
        Eigen::Vector3d centrVec = seg2.getSegCentroid() -
                                   seg1.getSegCentroid();
        float step1 = std::fabs(centrVec.dot(seg2.getSegNormal()));
        float step2 = std::fabs((-centrVec).dot(seg1.getSegNormal()));
        cout << "step1 = " << step1 << endl;
        cout << "step2 = " << step2 << endl;
        // if there is no step between segments
        if(step1 < stepThresh && step2 < stepThresh){
            float normalScore = seg1.getSegNormal().dot(seg2.getSegNormal());
            cout << "normalScore = " << normalScore << endl;
            
            if(normalScore > normalThresh) {
                coplanar = true;
            }
        }
        
    }
    return coplanar;
}

double
PlaneSegmentation::compEdgeScore(const PlaneSeg &seg1,
                                const PlaneSeg &seg2,
                                double curvThresh,
                                double normalThresh,
                                double stepThresh) {
    double score = 0.0;
    
    double curvScore = std::min(1.0, std::exp(-(std::max(seg1.getSegCurv(), seg2.getSegCurv()) - curvThresh)/curvThresh));
    
    Eigen::Vector3d centrVec = seg2.getSegCentroid() -
                               seg1.getSegCentroid();
    double step1 = std::fabs(centrVec.dot(seg2.getSegNormal()));
    double step2 = std::fabs((-centrVec).dot(seg1.getSegNormal()));
    double stepScore = std::min(1.0, std::exp(-(std::max(step1, step2) - stepThresh)/stepThresh));
    
    double normalScore = seg1.getSegNormal().dot(seg2.getSegNormal());
    
    score = normalScore * stepScore * curvScore;
    
    return score;
}

void PlaneSegmentation::drawSegments(pcl::visualization::PCLVisualizer::Ptr viewer,
                                     std::string name,
                                     int vp,
                                     const vector<PlaneSeg, Eigen::aligned_allocator<PlaneSeg>> &segs,
                                     UnionFind &sets)
{
    viewer->removePointCloud(name, vp);
    
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLab(new pcl::PointCloud<pcl::PointXYZL>());
    
    int nextLab = 0;
    for(int s = 0; s < segs.size(); ++s){
        int set = sets.findSet(s);
        
        if(s == set) {
            int curLab = nextLab++;
            pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr points = segs[s].getPoints();
            
            for (int p = 0; p < points->size(); ++p) {
                pcl::PointXYZRGB pcPt = points->at(p);
                pcl::PointXYZL pt;
                pt.x = pcPt.x;
                pt.y = pcPt.y;
                pt.z = pcPt.z;
                pt.label = curLab;
                pcLab->push_back(pt);
            }
        }
    }
    
    viewer->addPointCloud(pcLab, name, vp);
}

void PlaneSegmentation::visualizeSegmentation(const vectorPlaneSeg &svs,
                                              const vectorPlaneSeg &segs,
                                              std::vector<int> &svLabels,
                                              pcl::visualization::PCLVisualizer::Ptr viewer,
                                              int viewPort1,
                                              int viewPort2)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm(new pcl::PointCloud<pcl::PointXYZL>());
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcVox(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::Normal>::Ptr pcNormalsVox(new pcl::PointCloud<pcl::Normal>());
    for(int sv = 0; sv < svLabels.size(); ++sv) {
        const PlaneSeg& curSv = svs[sv];
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr points = svs[sv].getPoints();
        for(int p = 0; p < points->size(); ++p){
            pcl::PointXYZL pt;
            pt.x = points->at(p).x;
            pt.y = points->at(p).y;
            pt.z = points->at(p).z;
            pt.label = sv;
            
            pcLabSegm->push_back(pt);
        }
        
        pcVox->insert(pcVox->end(),
                      points->begin(),
                      points->end());
        pcNormalsVox->insert(pcNormalsVox->end(),
                             svs[sv].getNormals()->begin(),
                             svs[sv].getNormals()->end());
    }
    
    viewer->addPointCloud(pcLabSegm, "cloudLabSegm", viewPort1);
//        viewer->addPointCloud(pcVox, "cloud_color", viewPort1);
//        viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(pcNormals, 100, 0.1, "cloudNormals", viewPort1);
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(pcVox, pcNormalsVox, 100, 0.1, "cloudNormals", viewPort1);
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcColPlanes(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::PointCloud<pcl::PointNormal>::Ptr pcNormalsSv(new pcl::PointCloud<pcl::PointNormal>());
    
    for(int sv = 0; sv < svLabels.size(); ++sv) {
        // label -1 for non plane elements
        if (svLabels[sv] >= 0) {
            int lab = svLabels[sv];
            int colIdx = lab % (sizeof(colors)/sizeof(colors[0]));
            
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPts = svs[sv].getPoints();
            for(int p = 0; p < curPts->size(); ++p){
                pcl::PointXYZRGBA pt;
                pt.r = colors[colIdx][0];
                pt.g = colors[colIdx][1];
                pt.b = colors[colIdx][2];
                pt.x = curPts->at(p).x;
                pt.y = curPts->at(p).y;
                pt.z = curPts->at(p).z;
                pcColPlanes->push_back(pt);
            }
            
            pcl::PointNormal curPtNormal = svs[sv].getPointNormal();
            pcNormalsSv->push_back(curPtNormal);
        }
    }
    
    viewer->addPointCloud(pcColPlanes, "cloudColPlanes", viewPort2);
//    viewer->addPointCloudNormals<pcl::PointNormal>(pcNormalsSv, 1, 0.1, "cloudNormalsSv", viewPort2);


//        cout << "pcNormals->size() = " << pcNormalsSv->size() << endl;
//		cout << "pc->size() = " << pc->size() << endl;
//		cout << "pcLab->size() = " << pcLab->size() << endl;
//
    viewer->resetStoppedFlag();
    viewer->initCameraParameters();
    viewer->setCameraPosition(0.0, 0.0, -4.0, 0.0, -1.0, 0.0);
    viewer->spinOnce (100);
    while (!viewer->wasStopped()){
        viewer->spinOnce (50);
        cv::waitKey(50);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

float PlaneSegmentation::compSvToPlaneDist(pcl::PointNormal sv,
									pcl::PointNormal plane)
{
	Eigen::Vector3f svNorm = sv.getNormalVector3fMap();
	Eigen::Vector3f planeNorm = plane.getNormalVector3fMap();

	float dist = distToPlane(plane, sv);

	float angDist = exp((1.0 - svNorm.dot(planeNorm))/0.1);
	if(isnan(angDist)){
		angDist = 1;
	}
	angDist = min(angDist, 10.0f);

	float weightedDist = angDist * dist;
	return weightedDist;
}


void PlaneSegmentation::compPlaneNormals(const std::vector<int>& svLabels,
                                    const vectorPlaneSeg& svsInfo,
									pcl::PointCloud<pcl::PointNormal>::Ptr planeNorm,
									std::vector<std::vector<int>>& planeSvsIdx)
{
	multimap<int, int> labelToSv;
	for(int sv = 0; sv < svLabels.size(); ++sv){
		labelToSv.insert(make_pair(svLabels[sv], sv));
	}


//	pcl::PointCloud<pcl::PointNormal>::Ptr newPlaneNorm(new pcl::PointCloud<pcl::PointNormal>());
	auto it = labelToSv.begin();
	while(it != labelToSv.end()){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc(new pcl::PointCloud<pcl::PointXYZRGB>());
		vector<int> curSvsIdx;

		auto itBegin = it;
		while(it != labelToSv.end() &&
			it->first == itBegin->first)
		{
			int curIdx = it->second;
//			int curId = idxToId[curIdx];
//			if(idToSv.count(curId) == 0){
//				throw PLANE_EXCEPTION("Id not fount in map, curId = " + to_string(curId) + ", curIdx = " + to_string(curIdx) + ", svLabels.size() = " + to_string(svLabels.size()));
//			}
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr curSvPc(svsInfo[curIdx].getPoints());
			curPc->insert(curPc->begin(), curSvPc->begin(), curSvPc->end());
			curSvsIdx.push_back(curIdx);
			++it;
		}

		// if label != 0 than we have a plane
		if(itBegin->first != 0){
			pcl::PointNormal curPlaneNorm;
			Eigen::Vector4f curPlaneParams;
			float curCurv;
			pcl::computePointNormal(*curPc, curPlaneParams, curCurv);

			// Same work as in pcl::computePointNormal - could be done better
	//		EIGEN_ALIGN16 Eigen::Matrix3f covMat;
			Eigen::Vector4f centr;
			pcl::compute3DCentroid(*curPc, centr);

			Eigen::Vector3f curNormal = curPlaneParams.head<3>();
			curNormal.normalize();
//			pcl::flipNormalTowardsViewpoint(curPlaneNorm, 0.0, 0.0, 0.0, curNormal);

			curPlaneNorm.getVector4fMap() = centr;
			curPlaneNorm.getNormalVector3fMap() = curNormal;
//			curPlaneNorm.data_n[3] = 0.0;
//			curPlaneNorm.getNormalVector3fMap().normalize();
			curPlaneNorm.curvature = curCurv;


			planeNorm->push_back(curPlaneNorm);
			planeSvsIdx.push_back(curSvsIdx);
		}
	}
}



void PlaneSegmentation::compSupervoxelsAreaEst(const std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr >& idToSv,
										std::vector<float>& svAreaEst)
{
	svAreaEst.clear();
	for(auto it = idToSv.begin(); it != idToSv.end(); ++it){
		float curAreaCoeff = 0.0;
		pcl::Supervoxel<pcl::PointXYZRGB>::Ptr curSv = it->second;
//		pcl::Normal curNorm = curSv->normal_;
		pcl::PointNormal curNormPt;
		curSv->getCentroidPointNormal(curNormPt);
		Eigen::Vector3f curNorm = curNormPt.getNormalVector3fMap();
		for(int pt = 0; pt < curSv->voxels_->size(); ++pt){
			pcl::PointXYZRGB curPt = curSv->voxels_->at(pt);
			Eigen::Vector3f curPtCentroid = curPt.getVector3fMap() - curNormPt.getVector3fMap();
			float lenSq = curPtCentroid.squaredNorm();
			float normLen = curNorm.dot(curPtCentroid);
			float distPlaneSq = lenSq - normLen * normLen;
			if(!isnan(distPlaneSq)){
				curAreaCoeff = max(distPlaneSq, curAreaCoeff);
			}
		}
		svAreaEst.push_back(curAreaCoeff * pi);
	}
//	cout << "svAreaEst = " << svAreaEst << endl;
}


bool operator<(const SegEdge &l, const SegEdge &r){
    return l.w < r.w;
}














