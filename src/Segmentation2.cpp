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

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Eigen>

#include "Segmentation2.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"
#include "UnionFind.h"


using namespace std;



void Segmentation2::segment(const cv::FileStorage& fs,
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

    vector<SegInfo> svsInfo;
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


    vector<int> svLabels(svsInfo.size(), 0);
	{
		UnionFind sets(svsInfo.size());

		// flood fill through segments to connect segments belonging to the same planes
		cout << "flood fill through segments" << endl;
        vector<bool> isVisited(svsInfo.size(), false);
        vector<vector<int> > planeCandidates;
		for(int sv = 0; sv < svsInfo.size(); ++sv){
			// if not visited and planar enough
			if(!isVisited[sv] && svsInfo[sv].getSegCurv() < curvThresh){
				queue<int> nodeQ;
				nodeQ.push(sv);
				isVisited[sv] = true;

				vector<int> curVisited;
				float curVisitedAreaCoeff = 0.0;
				while(!nodeQ.empty()){
					int curIdx = nodeQ.front();
					nodeQ.pop();
//                    cout << "curIdx = " << curIdx << endl;

					curVisited.push_back(curIdx);
					curVisitedAreaCoeff += svsInfo[curIdx].getAreaEst();
					for(int nh = 0; nh < svsInfo[curIdx].getAdjSegs().size(); ++nh){
						int nhIdx = svsInfo[curIdx].getAdjSegs()[nh];
						// if not visited
						if(!isVisited[nhIdx])
						{
                            // if planar enough
                            if(svsInfo[nhIdx].getSegCurv() < 2*curvThresh) {
                                Eigen::Vector3f centrVec = svsInfo[nhIdx].getSegCentroid() -
                                                           svsInfo[curIdx].getSegCentroid();
                                float step1 = std::fabs(centrVec.dot(svsInfo[nhIdx].getSegNormal()));
                                float step2 = std::fabs((-centrVec).dot(svsInfo[curIdx].getSegNormal()));
                                if(step1 < stepThresh && step2 < stepThresh){
                                    float normalScore = svsInfo[curIdx].getSegNormal().dot(svsInfo[nhIdx].getSegNormal());

                                    if(normalScore > normalThresh) {
                                        nodeQ.push(nhIdx);
                                        isVisited[nhIdx] = true;
                                    }
                                }

                            }
						}
					}
				}


				if(curVisitedAreaCoeff >= areaThresh){
					planeCandidates.push_back(curVisited);
				}
			}
		}

        // create obj instances - plane instances in current version
        cout << "create obj instances" << endl;
        int curObjInstId = 0;
		for(int pl = 0; pl < planeCandidates.size(); ++pl){
//            SegInfo curPlaneInfo;
            std::vector<SegInfo> svs;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr points(new pcl::PointCloud<pcl::PointXYZRGB>());
			for(int sv = 0; sv < planeCandidates[pl].size(); ++sv){
                int curSvIdx = planeCandidates[pl][sv];
                // label 0 for non planar
                svLabels[curSvIdx] = curObjInstId + 1;
//                curPlaneInfo.addPointsAndNormals(svsInfo[curSvIdx].getPoints(),
//                                                svsInfo[curSvIdx].getNormals());
                svs.push_back(svsInfo[curSvIdx]);
                points->insert(points->end(),
                               svsInfo[curSvIdx].getPoints()->begin(),
                               svsInfo[curSvIdx].getPoints()->end());
				for(int p = 0; p < svsInfo[curSvIdx].getPoints()->size(); ++p){
					pcl::PointXYZRGB pcPt = svsInfo[curSvIdx].getPoints()->at(p);
					pcl::PointXYZRGBL pt;
					pt.rgb = pcPt.rgb;
					pt.x = pcPt.x;
					pt.y = pcPt.y;
					pt.z = pcPt.z;
					pt.label = svLabels[curSvIdx];
					pcLab->push_back(pt);
				}
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


	if(viewer){
		cout << "displaying" << endl;
//		pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm = svClust.getLabeledVoxelCloud();
		viewer->addPointCloud(pcLabSegm, "cloudLabSegm", viewPort1);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcVox(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::Normal>::Ptr pcNormalsVox(new pcl::PointCloud<pcl::Normal>());
        for(int sv = 0; sv < svLabels.size(); ++sv) {
            SegInfo& curSv = svsInfo[sv];
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
            // label 0 for non plane elements
            if (svLabels[sv] != 0) {
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


float Segmentation2::compSvToPlaneDist(pcl::PointNormal sv,
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


void Segmentation2::compPlaneNormals(const std::vector<int>& svLabels,
                                    const std::vector<SegInfo>& svsInfo,
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



void Segmentation2::compSupervoxelsAreaEst(const std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr >& idToSv,
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









