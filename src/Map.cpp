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
#include <chrono>
#include <thread>

#include <Eigen/Eigen>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include <g2o/types/slam3d/se3quat.h>

#include "Map.hpp"
#include "PlaneSegmentation.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"

using namespace std;



bool operator<(const PendingMatchKey &lhs, const PendingMatchKey &rhs) {
    auto lit = lhs.matchedIds.begin();
    auto rit = rhs.matchedIds.begin();
    
    for( ; lit != lhs.matchedIds.end() && rit != rhs.matchedIds.end(); ++lit, ++rit){
        if(*lit < *rit){
            return true;
        }
        else if(*lit > *rit){
            return false;
        }
        // continue if equal
    }
    // if lhs has fewer elements
    if(lit == lhs.matchedIds.end() && rit != rhs.matchedIds.end()){
        return true;
    }
    // if lhs has the same number of elements or more elements
    else {
        return false;
    }
}

Map::Map()
    : originalPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>()){
    
}

Map::Map(const cv::FileStorage& settings)
    :
    originalPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>())
{
	if((int)settings["map"]["readFromFile"]){
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("map 3D Viewer"));

		int v1 = 0;
		int v2 = 0;
		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		viewer->addCoordinateSystem();

		vector<cv::String> mapFilepaths;
        settings["map"]["mapFiles"] >> mapFilepaths;

        vector<vector<ObjInstance> > allObjInstances;
        for(int f = 0; f < mapFilepaths.size(); ++f) {
            viewer->removeAllPointClouds();

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloudNormal(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(mapFilepaths[f], *pointCloudNormal);
            if (pointCloudNormal->empty()) {
                throw PLANE_EXCEPTION(string("Could not read PLY file: ") + settings["map"]["mapFile"]);
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::copyPointCloud(*pointCloudNormal, *pointCloud);
            originalPointCloud->insert(originalPointCloud->end(), pointCloud->begin(), pointCloud->end());

//		Vector7d mapFileOffset;
//		mapFileOffset << -0.868300, 0.602600, 1.562700, 0.821900, -0.391200, 0.161500, -0.381100;
//		Eigen::Matrix4d transformMat = g2o::SE3Quat(mapFileOffset).inverse().to_homogeneous_matrix();
//		pcl::transformPointCloud(*pointCloudNormal, *pointCloudNormal, transformMat);

            pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointCloudLab(new pcl::PointCloud<pcl::PointXYZRGBL>());
//    		vector<ObjInstance> curObjInstances;
            allObjInstances.push_back(vector<ObjInstance>());

            PlaneSegmentation::segment(settings,
                                  pointCloudNormal,
                                  pointCloudLab,
                                  allObjInstances.back(),
                                  true/*,
                                  viewer,
                                  v1,
                                  v2*/);

//            objInstances.insert(objInstances.end(), curObjInstances.begin(), curObjInstances.end());


//            viewer->resetStoppedFlag();
//            viewer->initCameraParameters();
//            viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//            viewer->spinOnce(100);
//            while (!viewer->wasStopped()) {
//                viewer->spinOnce(100);
//                std::this_thread::sleep_for(std::chrono::milliseconds(50));
//            }
        }

        objInstances = ObjInstance::mergeObjInstances(allObjInstances/*,
                                                      viewer,
                                                      v1,
                                                      v2*/);

        cout << "object instances in map: " << objInstances.size() << endl;

        if(viewer) {
            viewer->removeAllPointClouds(v1);
            viewer->removeAllShapes(v1);
            viewer->removeAllPointClouds(v2);
            viewer->removeAllShapes(v2);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol = getColorPointCloud();
            viewer->addPointCloud(pcCol, "cloud_color_map", v1);

            pcl::PointCloud<pcl::PointXYZL>::Ptr pcLab = getLabeledPointCloud();
            viewer->addPointCloud(pcLab, "cloud_labeled_map", v2);

            viewer->resetStoppedFlag();
            viewer->initCameraParameters();
            viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
            viewer->spinOnce(100);
//            while (!viewer->wasStopped()) {
//                viewer->spinOnce(100);
//                std::this_thread::sleep_for(std::chrono::milliseconds(50));
//            }
        }

	}
}


void Map::addPendingObj(ObjInstance &obj,
                        const std::set<int> &matchedIds,
                        const std::vector<std::list<ObjInstance>::iterator> &objInstancesIts,
                        int eolAdd)
{
    pendingObjInstances.push_back(obj);
    PendingMatchKey pmatchKey = PendingMatchKey{matchedIds};
    
    set<PendingMatchKey>::iterator itKey = pendingMatchesSet.find(pmatchKey);
    
    if(itKey != pendingMatchesSet.end()){
//        auto matchIt = itKey->it;
        itKey->pmatch->eol += eolAdd;
        itKey->pmatch->pendingObjInstanceIts.push_back(--pendingObjInstances.end());
    }
    else {
        pmatchKey.pmatch.reset(new PendingMatch{matchedIds,
                                                eolAdd,
                                                objInstancesIts,
                                                std::vector<list<ObjInstance>::iterator>{--pendingObjInstances.end()}});
        pendingMatchesSet.insert(pmatchKey);
    }
}

//void Map::addPendingObjs(std::vector<ObjInstance>::iterator beg,
//                         std::vector<ObjInstance>::iterator end,
//                         int eolAdd) {
//
//}

void Map::removePendingObjsEol() {
    for(auto it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ++it){
        for(auto itPObj = it->pmatch->pendingObjInstanceIts.begin(); itPObj != it->pmatch->pendingObjInstanceIts.end(); ++itPObj){
            pendingObjInstances.erase(*itPObj);
        }
    }
}

std::vector<PendingMatchKey> Map::getPendingMatches(int eolThresh) {
    vector<PendingMatchKey> retPendingMatches;
    for(auto it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ++it){
        if(it->pmatch->eol > eolThresh){
            retPendingMatches.push_back(*it);
        }
    }
    return retPendingMatches;
}


void Map::executePendingMatches(int eolThresh) {
    map<int, int> idToIdx;
    map<int, int> idxToId;
    vector<list<ObjInstance>::iterator> its;
    vector<bool> isPending;
    int idx = 0;
    // assign each ObjInstance an idx
    for(auto it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ++it) {
        if (it->pmatch->eol > eolThresh) {
            for(auto iti = it->pmatch->objInstanceIts.begin(); iti != it->pmatch->objInstanceIts.end(); ++iti){
                idToIdx[(*iti)->getId()] = idx;
                idxToId[idx] = (*iti)->getId();
                its.push_back(*iti);
                isPending.push_back(false);
                ++idx;
            }
            for(auto iti = it->pmatch->pendingObjInstanceIts.begin(); iti != it->pmatch->pendingObjInstanceIts.end(); ++iti){
                idToIdx[(*iti)->getId()] = idx;
                idxToId[idx] = (*iti)->getId();
                its.push_back(*iti);
                isPending.push_back(true);
                ++idx;
            }
        }
    }
    
    UnionFind ufSets(idx);
    for(auto it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ) {
        if (it->pmatch->eol > eolThresh) {
            auto iti = it->pmatch->objInstanceIts.begin();
            int mergeIdx = idToIdx[(*iti)->getId()];
            
            for(++iti; iti != it->pmatch->objInstanceIts.end(); ++iti){
                int curIdx = idToIdx[(*iti)->getId()];
                ufSets.unionSets(mergeIdx, curIdx);
            }
            for(iti = it->pmatch->pendingObjInstanceIts.begin(); iti != it->pmatch->pendingObjInstanceIts.end(); ++iti){
                int curIdx = idToIdx[(*iti)->getId()];
                ufSets.unionSets(mergeIdx, curIdx);
            }
            
            it = pendingMatchesSet.erase(it);
        }
        else{
            ++it;
        }
    }
    
    multimap<int, int> sets;
    for(int curIdx = 0; curIdx < idx; ++curIdx){
        int setId = ufSets.findSet(curIdx);
        sets.insert(make_pair(setId, curIdx));
    }

//    typedef multimap<int, pair<int, int> >::iterator mmIter;
    for(auto it = sets.begin(); it != sets.end(); ) {
    
        auto range = sets.equal_range(it->first);
    
        vector<list<ObjInstance>::iterator> mapObjIts;
        vector<list<ObjInstance>::iterator> pendingObjIts;
        cout << "merging ids:" << endl;
        for (auto rangeIt = range.first; rangeIt != range.second; ++rangeIt) {
            int curIdx = rangeIt->second;
            if(isPending[curIdx]){
                pendingObjIts.push_back(its[curIdx]);
            }
            else{
                mapObjIts.push_back(its[curIdx]);
            }
            cout << idxToId[curIdx] << endl;
        }
        
        auto iti = mapObjIts.begin();
        auto mergeIt = *iti;
        
        // merge all map objects
        for(++iti; iti != mapObjIts.end(); ++iti){
            mergeIt->merge(*(*iti));
            objInstances.erase(*iti);
        }
        // merge all pending objects
        for(iti = pendingObjIts.begin(); iti != pendingObjIts.end(); ++iti){
            mergeIt->merge(*(*iti));
            pendingObjInstances.erase(*iti);
        }
        
        it = range.second;
    }
}

bool Map::getPendingMatch(PendingMatchKey &pendingMatch) {
    if(pendingMatchesSet.count(pendingMatch) > 0){
        auto it = pendingMatchesSet.find(pendingMatch);
        pendingMatch = *it;
        
        return true;
    }
    return false;
}

void Map::decreaseEol(int eolSub) {
    for(set<PendingMatchKey>::iterator it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ++it){
        it->pmatch->eol -= eolSub;
    }
}


pcl::PointCloud<pcl::PointXYZL>::Ptr Map::getLabeledPointCloud()
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLab(new pcl::PointCloud<pcl::PointXYZL>());
    int o = 0;
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it, ++o){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = it->getPoints();
        for(int pt = 0; pt < curPc->size(); ++pt){
            pcl::PointXYZL newPt;
            newPt.x = curPc->at(pt).x;
            newPt.y = curPc->at(pt).y;
            newPt.z = curPc->at(pt).z;
            newPt.label = o + 1;
            pcLab->push_back(newPt);
        }
    }
    return pcLab;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Map::getColorPointCloud()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol(new pcl::PointCloud<pcl::PointXYZRGB>());
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = it->getPoints();
        pcCol->insert(pcCol->end(), curPc->begin(), curPc->end());
    }
    return pcCol;
}
