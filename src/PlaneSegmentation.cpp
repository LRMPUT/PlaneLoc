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

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Eigen>

#include "PlaneSegmentation.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"


using namespace std;



void PlaneSegmentation::segment(const cv::FileStorage& fs,
						pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
						pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
						std::vector<ObjInstance>& objInstances,
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

    vector<PlaneSeg> svsInfo;
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm;
    {
        double svRes = 0.01;
        double svSize = 0.2;
        bool useSingleCameraTrans = false;
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
        svsInfo.resize(idxCnt);

        pcLabSegm = svClust.getLabeledCloud();
        for(int p = 0; p < pcLabSegm->size(); ++p){
            // ignore label 0
            int svId = pcLabSegm->at(p).label;
            if(svId != 0) {
                if (idToIdx.count(svId) > 0) {
                    int svIdx = idToIdx.at(svId);
//                cout << "svId = " << svId << ", svIdx = " << svIdx << endl;
                    svsInfo[svIdx].addPointAndNormal(pc->at(p), normals->at(p));
                } else {
                    cout << "svId = " << svId << " not found!!!" << endl;
                }
            }
        }

        cout << "Calculating segment properties" << endl;
        for(int sv = 0; sv < svsInfo.size(); ++sv){
            svsInfo[sv].setId(sv);
            svsInfo[sv].setLabel(0);
            svsInfo[sv].calcSegProp();
        }

        cout << "Adding edges" << endl;
        std::multimap<uint32_t, uint32_t> adjMultimap;
        svClust.getSupervoxelAdjacency(adjMultimap);
        for(auto it = adjMultimap.begin(); it != adjMultimap.end(); ++it) {
            int srcIdx = idToIdx[it->first];
            int tarIdx = idToIdx[it->second];

            svsInfo[srcIdx].addAdjSeg(tarIdx);
            svsInfo[tarIdx].addAdjSeg(srcIdx);
        }
    }


    vector<int> svLabels(svsInfo.size(), -1);
	{
		UnionFind sets(svsInfo.size());
        

        vector<PlaneSeg> planeSegs = svsInfo;
        
        mergeSegments(planeSegs,
                       sets,
                       curvThresh,
                       normalThresh,
                       stepThresh/*,
                     viewer,
                     viewPort1,
                     viewPort2*/);
        
        vector<vector<int> > planeCandidates(svsInfo.size());
        vector<double> planeCandidatesAreaEst(svsInfo.size(), 0.0);
        for(int sv = 0; sv < svsInfo.size(); ++sv){
            int set = sets.findSet(sv);
//            svLabels[sv] = set;
            planeCandidates[set].push_back(sv);
            planeCandidatesAreaEst[set] += svsInfo[sv].getAreaEst();
        }
        
        
        int curObjInstId = 0;
        for(int pl = 0; pl < planeSegs.size(); ++pl){
            if(planeCandidates[pl].size() > 0){
                if(planeCandidatesAreaEst[pl] > areaThresh){
                    if(planeSegs[pl].getSegCurv() < curvThresh){
                        int lab = pl;
                        
                        std::vector<PlaneSeg> svs;
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr points(new pcl::PointCloud<pcl::PointXYZRGB>());
                        
                        for(int s = 0; s < planeCandidates[pl].size(); ++s){
                            svs.push_back(svsInfo[planeCandidates[pl][s]]);
                            svLabels[planeCandidates[pl][s]] = pl;
                        }
                        
                        *points = *(planeSegs[pl].getPoints());
                        for(int p = 0; p < points->size(); ++p){
                            pcl::PointXYZRGB pcPt = points->at(p);
                            pcl::PointXYZRGBL pt;
                            pt.rgb = pcPt.rgb;
                            pt.x = pcPt.x;
                            pt.y = pcPt.y;
                            pt.z = pcPt.z;
                            pt.label = svLabels[pl];
                            pcLab->push_back(pt);
                        }
    
                        objInstances.emplace_back(curObjInstId++,
                                                  ObjInstance::ObjType::Plane,
                                                  points,
                                                  svs);
                        {
                            if(objInstances.back().getConvexHullArea() < areaThresh){
                                // remove last element
                                objInstances.erase(objInstances.end() - 1);
                            }
                        }
                    }
                }
            }
        }
	}


	if(viewer){
		cout << "displaying" << endl;
//		pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm = svClust.getLabeledVoxelCloud();
		viewer->addPointCloud(pcLabSegm, "cloudLabSegm", viewPort1);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcVox(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::Normal>::Ptr pcNormalsVox(new pcl::PointCloud<pcl::Normal>());
        for(int sv = 0; sv < svLabels.size(); ++sv) {
            PlaneSeg& curSv = svsInfo[sv];
            pcVox->insert(pcVox->end(),
                          svsInfo[sv].getPoints()->begin(),
                          svsInfo[sv].getPoints()->end());
            pcNormalsVox->insert(pcNormalsVox->end(),
                                 svsInfo[sv].getNormals()->begin(),
                                 svsInfo[sv].getNormals()->end());
        }

//        viewer->addPointCloud(pcVox, "cloud_color", viewPort1);
//        viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(pcNormals, 100, 0.1, "cloudNormals", viewPort1);
        viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(pcVox, pcNormalsVox, 100, 0.1, "cloudNormals", viewPort1);

		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcColPlanes(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::PointCloud<pcl::PointNormal>::Ptr pcNormalsSv(new pcl::PointCloud<pcl::PointNormal>());

		for(int p = 0; p < pcLab->size(); ++p){
			int lab = pcLab->at(p).label;
			int colIdx = lab % (sizeof(colors)/sizeof(colors[0]));
			pcl::PointXYZRGBA pt;
			pt.r = colors[colIdx][0];
			pt.g = colors[colIdx][1];
			pt.b = colors[colIdx][2];
			pt.x = pcLab->at(p).x;
			pt.y = pcLab->at(p).y;
			pt.z = pcLab->at(p).z;

			pcColPlanes->push_back(pt);
		}
        for(int sv = 0; sv < svLabels.size(); ++sv) {
            // label -1 for non plane elements
            if (svLabels[sv] >= 0) {
                pcl::PointNormal curPtNormal = svsInfo[sv].getPointNormal();
                pcNormalsSv->push_back(curPtNormal);
            }
        }

		viewer->addPointCloud(pcColPlanes, "cloudColPlanes", viewPort2);
        if(!pcNormals->empty()) {
            viewer->addPointCloudNormals<pcl::PointNormal>(pcNormalsSv, 1, 0.1, "cloudNormalsSv", viewPort2);
        }

//        cout << "pcNormals->size() = " << pcNormalsSv->size() << endl;
//		cout << "pc->size() = " << pc->size() << endl;
//		cout << "pcLab->size() = " << pcLab->size() << endl;
//
		viewer->resetStoppedFlag();
		viewer->initCameraParameters();
		viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
		viewer->spinOnce (100);
		while (!viewer->wasStopped()){
			viewer->spinOnce (100);
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
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

void PlaneSegmentation::mergeSegments(std::vector<PlaneSeg> &segs,
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
//        cout << "edge (" << u << ", " << v << ")" << endl;
//        cout << "curEdge.w = " << curEdge.w << endl;
//
//        cout << "newestVer = " << newestVer << endl;
//        cout << "curEdge.ver = " << curEdge.ver << endl;
        
        // if the newest version
        if(u != v && newestVer == curEdge.ver) {
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
                segs[m] = segs[u].merge(segs[v], sets);
    
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
        
        if(viewer){
            viewer->resetStoppedFlag();
            viewer->spinOnce (100);
            while (!viewer->wasStopped()){
                viewer->spinOnce (100);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
//    cout << "seg1.getSegCurv() = " << seg1.getSegCurv() << endl;
//    cout << "seg2.getSegCurv() = " << seg2.getSegCurv() << endl;
    // if planar enough
    if((seg1.getSegCurv() < curvThresh && seg2.getSegCurv() < 2*curvThresh) ||
       (seg1.getSegCurv() < 2*curvThresh && seg2.getSegCurv() < curvThresh))
    {
        Eigen::Vector3f centrVec = seg2.getSegCentroid() -
                                   seg1.getSegCentroid();
        float step1 = std::fabs(centrVec.dot(seg2.getSegNormal()));
        float step2 = std::fabs((-centrVec).dot(seg1.getSegNormal()));
//        cout << "step1 = " << step1 << endl;
//        cout << "step2 = " << step2 << endl;
        // if there is no step between segments
        if(step1 < stepThresh && step2 < stepThresh){
            float normalScore = seg1.getSegNormal().dot(seg2.getSegNormal());
//            cout << "normalScore = " << normalScore << endl;
            
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
    
    Eigen::Vector3f centrVec = seg2.getSegCentroid() -
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
                                     std::vector<PlaneSeg> segs,
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
                                    const std::vector<PlaneSeg>& svsInfo,
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














