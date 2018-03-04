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

#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>

#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include <g2o/types/slam3d/se3quat.h>
#include <Misc.hpp>

#include "Map.hpp"
#include "PlaneSegmentation.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"
#include "Serialization.hpp"

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
    : originalPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>())
{
    settings.eolObjInstInit = 4;
    settings.eolObjInstIncr = 2;
    settings.eolObjInstDecr = 1;
    settings.eolObjInstThresh = 8;
    settings.eolPendingInit = 4;
    settings.eolPendingIncr = 2;
    settings.eolPendingDecr = 1;
    settings.eolPendingThresh = 6;
}

Map::Map(const cv::FileStorage& fs)
    :
    originalPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>())
{
    settings.eolObjInstInit = 4;
    settings.eolObjInstIncr = 2;
    settings.eolObjInstDecr = 1;
    settings.eolObjInstThresh = 8;
    settings.eolPendingInit = 4;
    settings.eolPendingIncr = 2;
    settings.eolPendingDecr = 1;
    settings.eolPendingThresh = 6;
    
	if((int)fs["map"]["readFromFile"]){
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("map 3D Viewer"));

		int v1 = 0;
		int v2 = 0;
		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		viewer->addCoordinateSystem();

		vector<cv::String> mapFilepaths;
        fs["map"]["mapFiles"] >> mapFilepaths;

        static constexpr int idShift = 10000000;
        for(int f = 0; f < mapFilepaths.size(); ++f) {
            Map curMap;
        
            std::ifstream ifs(mapFilepaths[f].c_str());
            boost::archive::text_iarchive ia(ifs);
            ia >> curMap;
            
            curMap.shiftIds((f + 1)*idShift);
            
            vectorObjInstance curObjInstances;
            for(ObjInstance &obj : curMap){
                curObjInstances.push_back(obj);
            }
            mergeNewObjInstances(curObjInstances);
    
            clearPending();
        }

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

void Map::addObj(ObjInstance &obj) {
    objInstances.push_back(obj);
    objInstIdToIter[obj.getId()] = --(objInstances.end());
}

void Map::addObjs(vectorObjInstance::iterator beg, vectorObjInstance::iterator end) {
    for(auto it = beg; it != end; ++it){
        addObj(*it);
    }
}


void Map::mergeNewObjInstances(vectorObjInstance &newObjInstances,
                               pcl::visualization::PCLVisualizer::Ptr viewer,
                               int viewPort1,
                               int viewPort2)
{
    static constexpr double shadingLevel = 0.01;
    
    if(viewer){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        
        {
            int pl = 0;
            for (auto it = objInstances.begin(); it != objInstances.end(); ++it, ++pl) {
                cout << "adding plane " << pl << endl;
                
                ObjInstance &mapObj = *it;
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = mapObj.getPoints();
                
                viewer->addPointCloud(curPl, string("plane_ba_") + to_string(pl), viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
            }
        }
        {
            int npl = 0;
            for (ObjInstance &newObj : newObjInstances) {
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = newObj.getPoints();
                
                viewer->addPointCloud(curPl, string("plane_nba_") + to_string(npl), viewPort2);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_nba_") + to_string(npl),
                                                         viewPort2);
                
                ++npl;
            }
        }
        
    }
    
    vectorObjInstance addedObjs;
    
    int npl = 0;
    for(ObjInstance &newObj : newObjInstances){
//        cout << "npl = " << npl << endl;
        
        if(viewer){
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     0.5,
                                                     string("plane_nba_") + to_string(npl),
                                                     viewPort2);
            
            
            newObj.getHull().display(viewer, viewPort2);
            
        }
        
        vector<list<ObjInstance>::iterator> matches;
        int pl = 0;
        for(auto it = objInstances.begin(); it != objInstances.end(); ++it, ++pl) {
//            cout << "pl = " << pl << endl;
            ObjInstance &mapObj = *it;
            
            if(viewer){
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         0.5,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
                
                
                mapObj.getHull().display(viewer, viewPort1);
                
            }
            
            if(mapObj.isMatching(newObj/*,
                                 viewer,
                                 viewPort1,
                                 viewPort2*/))
            {
                cout << "Merging planes" << endl;
                
                matches.push_back(it);
            }
            
            
            if(viewer){
                viewer->resetStoppedFlag();
                
                static bool cameraInit = false;
                
                if(!cameraInit) {
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                    cameraInit = true;
                }
                while (!viewer->wasStopped()) {
                    viewer->spinOnce(100);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
                
                
                mapObj.getHull().cleanDisplay(viewer, viewPort1);
                
            }
        }
        
        if(matches.size() == 0){
            addedObjs.push_back(newObj);
            newObj.setEolCnt(settings.eolObjInstInit);
        }
        else if(matches.size() == 1){
            ObjInstance &mapObj = *matches.front();
            mapObj.merge(newObj);
            mapObj.increaseEolCnt(settings.eolObjInstIncr);
        }
        else{
            set<int> matchedIds;
            for(auto it : matches){
                matchedIds.insert(it->getId());
            }
            PendingMatchKey pmatchKey{matchedIds};
            if(getPendingMatch(pmatchKey)){
                addPendingObj(newObj, matchedIds, settings.eolPendingIncr);
            }
            else{
                addPendingObj(newObj, matchedIds, settings.eolPendingInit);
            }
            
            cout << "Multiple matches" << endl;
        }
        
        if(viewer){
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     shadingLevel,
                                                     string("plane_nba_") + to_string(npl),
                                                     viewPort2);
            
            
            newObj.getHull().cleanDisplay(viewer, viewPort2);
            
        }
        
        ++npl;
    }
    
    addObjs(addedObjs.begin(), addedObjs.end());
    
    executePendingMatches( settings.eolPendingThresh);
    decreasePendingEol( settings.eolPendingDecr);
    removePendingObjsEol();
    
    decreaseObjEol(settings.eolObjInstDecr);
    removeObjsEol();
}

void Map::mergeMapObjInstances(pcl::visualization::PCLVisualizer::Ptr viewer,
                               int viewPort1,
                               int viewPort2)
{
    if(viewer) {
        for (auto it = objInstances.begin(); it != objInstances.end(); ++it) {
            it->display(viewer, viewPort1);
        }
    }
    map<int, int> idToIdx;
    vector<listObjInstance::iterator> itrs;
    int idx = 0;
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it){
        idToIdx[it->getId()] = idx;
        itrs.push_back(it);
        ++idx;
    }
    
    UnionFind ufSets(idx);
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it) {
        ObjInstance &mapObj1 = *it;
        
        auto it2 = it;
        ++it2;
        for( ; it2 != objInstances.end(); ++it2){
            ObjInstance &mapObj2 = *it2;
            
            if(mapObj1.isMatching(mapObj2)){
                ufSets.unionSets(idToIdx[mapObj1.getId()], idToIdx[mapObj2.getId()]);
            }
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
        cout << endl << endl << "map merging ids:" << endl;
        for (auto rangeIt = range.first; rangeIt != range.second; ++rangeIt) {
            int curIdx = rangeIt->second;

            mapObjIts.push_back(itrs[curIdx]);

            cout << itrs[curIdx]->getId() << endl;
        }
        
        auto iti = mapObjIts.begin();
        auto mergeIt = *iti;
        
        // merge all map objects
        for(++iti; iti != mapObjIts.end(); ++iti){
            mergeIt->merge(*(*iti));
            mergeIt->increaseEolCnt(settings.eolObjInstIncr);
    
            objInstIdToIter.erase((*iti)->getId());
            objInstances.erase(*iti);
        }
        
        it = range.second;
    }
}

void Map::addPendingObj(ObjInstance &obj,
                        const std::set<int> &matchedIds,
                        int eolAdd)
{
    pendingObjInstances.push_back(obj);
    pendingIdToIter[obj.getId()] = --(pendingObjInstances.end());
    PendingMatchKey pmatchKey = PendingMatchKey{matchedIds};
    
    set<PendingMatchKey>::iterator itKey = pendingMatchesSet.find(pmatchKey);
    
    if(itKey != pendingMatchesSet.end()){
//        auto matchIt = itKey->it;
        itKey->pmatch->eol += eolAdd;
        itKey->pmatch->pendingObjInstanceIds.push_back(obj.getId());
    }
    else {
        vector<int> objInstanceIds;
        for(auto it = matchedIds.begin(); it != matchedIds.end(); ++it){
            objInstanceIds.push_back(*it);
        }
        pmatchKey.pmatch.reset(new PendingMatch{matchedIds,
                                                eolAdd,
                                                objInstanceIds,
                                                vector<int>{obj.getId()}});
        pendingMatchesSet.insert(pmatchKey);
    }
}

//void Map::addPendingObjs(std::vector<ObjInstance>::iterator beg,
//                         std::vector<ObjInstance>::iterator end,
//                         int eolAdd) {
//
//}

void Map::removePendingObjsEol() {
    for(auto it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ){
        auto curIt = it++;
        
        cout << "curIt->pmatch->eol = " << curIt->pmatch->eol << endl;
        if(curIt->pmatch->eol <= 0) {
            for (auto itId = curIt->pmatch->pendingObjInstanceIds.begin();
                 itId != curIt->pmatch->pendingObjInstanceIds.end(); ++itId) {
                
                int id = *itId;
                auto pendingIt = pendingIdToIter.at(id);
                pendingIdToIter.erase(id);
                pendingObjInstances.erase(pendingIt);
            }
            
            pendingMatchesSet.erase(curIt);
        }
    }
}

void Map::clearPending(){
    pendingMatchesSet.clear();
    pendingIdToIter.clear();
    pendingObjInstances.clear();
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
            for(auto itId = it->pmatch->objInstanceIds.begin(); itId != it->pmatch->objInstanceIds.end(); ++itId){
                int id = *itId;
                idToIdx[id] = idx;
                idxToId[idx] = id;
                its.push_back(objInstIdToIter.at(id));
                isPending.push_back(false);
                ++idx;
            }
            for(auto itId = it->pmatch->pendingObjInstanceIds.begin(); itId != it->pmatch->pendingObjInstanceIds.end(); ++itId){
                int id = *itId;
                idToIdx[id] = idx;
                idxToId[idx] = id;
                its.push_back(pendingIdToIter.at(id));
                isPending.push_back(true);
                ++idx;
            }
        }
    }
    
    UnionFind ufSets(idx);
    for(auto it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ) {
        if (it->pmatch->eol > eolThresh) {
            auto itId = it->pmatch->objInstanceIds.begin();
            int mergeIdx = idToIdx[(*itId)];
            
            for(++itId; itId != it->pmatch->objInstanceIds.end(); ++itId){
                int curIdx = idToIdx[(*itId)];
                ufSets.unionSets(mergeIdx, curIdx);
            }
            for(itId = it->pmatch->pendingObjInstanceIds.begin(); itId != it->pmatch->pendingObjInstanceIds.end(); ++itId){
                int curIdx = idToIdx[(*itId)];
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
        cout << endl << endl << "merging ids:" << endl;
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
            mergeIt->increaseEolCnt(settings.eolObjInstIncr);
            
            objInstIdToIter.erase((*iti)->getId());
            objInstances.erase(*iti);
        }
        // merge all pending objects
        for(iti = pendingObjIts.begin(); iti != pendingObjIts.end(); ++iti){
            mergeIt->merge(*(*iti));
            mergeIt->increaseEolCnt(settings.eolObjInstIncr);
    
            pendingIdToIter.erase((*iti)->getId());
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

void Map::decreasePendingEol(int eolSub) {
    for(set<PendingMatchKey>::iterator it = pendingMatchesSet.begin(); it != pendingMatchesSet.end(); ++it){
        it->pmatch->eol -= eolSub;
    }
}

void Map::decreaseObjEol(int eolSub) {
    for(ObjInstance &obj : objInstances){
        if(obj.getEolCnt() < settings.eolObjInstThresh){
            obj.decreaseEolCnt(eolSub);
        }
    }
}

void Map::removeObjsEol() {
    for(auto it = objInstances.begin(); it != objInstances.end(); ){
        if(it->getEolCnt() <= 0){
            objInstIdToIter.erase(it->getId());
            it = objInstances.erase(it);
        }
        else{
            ++it;
        }
    }
}


void Map::removeObjsEolThresh(int eolThresh) {
    for(auto it = objInstances.begin(); it != objInstances.end(); ){
        if(it->getEolCnt() < eolThresh){
            objInstIdToIter.erase(it->getId());
            it = objInstances.erase(it);
        }
        else{
            ++it;
        }
    }
}

void Map::removeObjsObsThresh(int obsThresh) {
    for(auto it = objInstances.begin(); it != objInstances.end(); ){
        if(it->getObsCnt() < obsThresh){
            objInstIdToIter.erase(it->getId());
            it = objInstances.erase(it);
        }
        else{
            ++it;
        }
    }
}

void Map::shiftIds(int startId) {
//    map<int, int> oldIdToNewId;
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it){
        int oldId = it->getId();
        int newId = startId + oldId;
        
        objInstIdToIter[newId] = objInstIdToIter.at(oldId);
        objInstIdToIter.erase(oldId);
        
        it->setId(newId);
//        oldIdToNewId[oldId] = newId;
    }
    for(auto it = pendingObjInstances.begin(); it != pendingObjInstances.end(); ++it){
        int oldId = it->getId();
        int newId = startId + oldId;
    
        pendingIdToIter[newId] = pendingIdToIter.at(oldId);
        pendingIdToIter.erase(oldId);
    
        it->setId(newId);
//        oldIdToNewId[oldId] = newId;
    }
 
    // for now just clearing pending objects
    clearPending();
}

std::vector<int> Map::getVisibleObjs(Vector7d pose,
                                     cv::Mat cameraMatrix,
                                     int rows,
                                     int cols,
                                     pcl::visualization::PCLVisualizer::Ptr viewer,
                                     int viewPort1,
                                     int viewPort2)
{
    static constexpr double shadingLevel = 1.0/8;
    
    g2o::SE3Quat poseSE3Quat(pose);
    Eigen::Matrix4d poseMat = poseSE3Quat.to_homogeneous_matrix();
    Eigen::Matrix4d poseInvMat = poseSE3Quat.inverse().to_homogeneous_matrix();
    Eigen::Matrix4d poseMatt = poseSE3Quat.to_homogeneous_matrix().transpose();
    Eigen::Matrix3d R = poseMat.block<3, 3>(0, 0);
    Eigen::Vector3d t = poseMat.block<3, 1>(0, 3);
    
//    vectorVector2d imageCorners;
//    imageCorners.push_back((Eigen::Vector2d() << 0, 0).finished());
//    imageCorners.push_back((Eigen::Vector2d() << cols - 1, 0).finished());
//    imageCorners.push_back((Eigen::Vector2d() << cols - 1, rows - 1).finished());
//    imageCorners.push_back((Eigen::Vector2d() << 0, rows - 1).finished());
    
    if(viewer){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        
        for (auto it = objInstances.begin(); it != objInstances.end(); ++it) {
            it->display(viewer, viewPort1, shadingLevel);
        }
        viewer->addCoordinateSystem();
        Eigen::Affine3f trans = Eigen::Affine3f::Identity();
        trans.matrix() = poseMat.cast<float>();
        viewer->addCoordinateSystem(0.5, trans, "camera_coord");
    }
    
    vector<int> visible;
    
    vector<vector<vector<pair<double, int>>>> projPlanes(rows,
                                                         vector<vector<pair<double, int>>>(cols,
                                                                       vector<pair<double, int>>()));
    
    cv::Mat projPoly(rows, cols, CV_8UC1);
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it) {
        cout << "id = " << it->getId() << endl;
    
        Eigen::Vector4d planeEqCamera = poseMatt * it->getNormal();
        cout << "planeEqCamera = " << planeEqCamera.transpose() << endl;
    
        if (viewer) {
            it->cleanDisplay(viewer, viewPort1);
            it->display(viewer, viewPort1);
        }
        
        // TODO condition for observing the right face of the plane
        Eigen::Vector3d normal = planeEqCamera.head<3>();
        double d = -planeEqCamera(3);
        Eigen::Vector3d zAxis;
        zAxis << 0, 0, 1;
        
        cout << "normal.dot(zAxis) = " << normal.dot(zAxis) << endl;
        cout << "d = " << d << endl;
        if (normal.dot(zAxis) < 0 && d < 0) {
//        vectorVector3d imageCorners3d;
//        bool valid = Misc::projectImagePointsOntoPlane(imageCorners,
//                                                       imageCorners3d,
//                                                       cameraMatrix,
//                                                       planeEqCamera);
        
            projPoly.setTo(0);
            vector<cv::Point *> polyCont;
            vector<int> polyContNpts;
        
            ConcaveHull hull = it->getHull().transform(poseSE3Quat.inverse().toVector());
        
//            if (viewer) {
//                hull.display(viewer, viewPort1);
//            }
//
//            hull.transform(poseSE3Quat.inverse().toVector());
        
            ConcaveHull hullClip = hull.clipToCameraFrustum(cameraMatrix, rows, cols, 0.2);
        
            ConcaveHull hullClipMap = hullClip.transform(poseSE3Quat.toVector());
            if (viewer) {
                hullClipMap.display(viewer, viewPort1, 1.0, 0.0, 0.0);
            }
        
            const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &polygons3d = hullClip.getPolygons3d();
            cout << "polygons3d.size() = " << polygons3d.size() << endl;
            for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr poly3d : polygons3d) {
                polyCont.push_back(new cv::Point[poly3d->size()]);
                polyContNpts.push_back(poly3d->size());

//                pcl::PointCloud<pcl::PointXYZRGB>::Ptr poly3dPose(new pcl::PointCloud<pcl::PointXYZRGB>());
//                // transform to camera frame
//                pcl::transformPointCloud(*poly3d, *poly3dPose, poseInvMat);
            
                cv::Mat pointsReproj = Misc::reprojectTo2D(poly3d, cameraMatrix);
                for (int pt = 0; pt < poly3d->size(); ++pt) {
                    cout << poly3d->at(pt).getVector3fMap().transpose() << endl;
                }
                cout << "cameraMatrix = " << cameraMatrix << endl;
                cout << "pointsReproj = " << pointsReproj << endl;
                
                int corrPointCnt = 0;
                for (int pt = 0; pt < pointsReproj.cols; ++pt) {
                    int u = std::round(pointsReproj.at<cv::Vec3f>(pt)[0]);
                    int v = std::round(pointsReproj.at<cv::Vec3f>(pt)[1]);
                    float d = pointsReproj.at<cv::Vec3f>(pt)[2];
                
                    if (u >= 0 && u < cols && v >= 0 && v < rows && d > 0) {
                        ++corrPointCnt;
                    }
                    polyCont.back()[pt] = cv::Point(u, v);
                }
                cout << "corrPointCnt = " << corrPointCnt << endl;
                if (corrPointCnt == 0) {
                    delete[] polyCont.back();
                    polyCont.erase(polyCont.end() - 1);
                    polyContNpts.erase(polyContNpts.end() - 1);
                }
            }
            if (polyCont.size() > 0) {
                cv::fillPoly(projPoly,
                             (const cv::Point **) polyCont.data(),
                             polyContNpts.data(),
                             polyCont.size(),
                             cv::Scalar(255));
            
                if (viewer) {
                    cv::imshow("proj_poly", projPoly);
                }
            
                vectorVector2d polyImagePts;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        if (projPoly.at<uint8_t>(r, c) > 0) {
                            polyImagePts.push_back((Eigen::Vector2d() << c, r).finished());
                        }
                    }
                }
                vectorVector3d polyPlanePts;
                Misc::projectImagePointsOntoPlane(polyImagePts,
                                                  polyPlanePts,
                                                  cameraMatrix,
                                                  planeEqCamera);
                for (int pt = 0; pt < polyImagePts.size(); ++pt) {
                    int x = std::round(polyImagePts[pt](0));
                    int y = std::round(polyImagePts[pt](1));
                    // depth is z coordinate
                    double d = polyPlanePts[pt](2);
                
                    projPlanes[y][x].push_back(make_pair(d, it->getId()));
                }
            }
        
            for (int p = 0; p < polyCont.size(); ++p) {
                delete[] polyCont[p];
            }
        
            if (viewer) {
                static bool cameraInit = false;
    
                if (!cameraInit) {
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                    cameraInit = true;
                }
                viewer->resetStoppedFlag();
                while (!viewer->wasStopped()) {
                    viewer->spinOnce(50);
                    cv::waitKey(50);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                
                hull.cleanDisplay(viewer, viewPort1);
                hullClipMap.cleanDisplay(viewer, viewPort1);
            }
        }
        if (viewer) {
            it->cleanDisplay(viewer, viewPort1);
            it->display(viewer, viewPort1, shadingLevel);
        }
    }
    
    map<int, int> idToCnt;
    for(int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            vector<pair<double, int>> &curPlanes = projPlanes[r][c];
            sort(curPlanes.begin(), curPlanes.end());
            
            if(!curPlanes.empty()){
                double minD = curPlanes.front().first;
                
                for(const pair<double, int> &curPair : curPlanes){
                    if(abs(minD - curPair.first) < 0.2){
                        int id = curPair.second;
                        if(idToCnt.count(id) > 0){
                            idToCnt.at(id) += 1;
                        }
                        else{
                            idToCnt[id] = 1;
                        }
                    }
                }
            }
        }
    }
    for(const pair<int, int> &curCnt : idToCnt){
        cout << "curCnt " << curCnt.first << " = " << curCnt.second << endl;
        if(curCnt.second > 1500){
            visible.push_back(curCnt.first);
        }
    }
    
    if(viewer){
        for (auto it = objInstances.begin(); it != objInstances.end(); ++it) {
            it->cleanDisplay(viewer, viewPort1);
        }
    }
    
    return visible;
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

void Map::recalculateIdToIter() {
    for(auto it = objInstances.begin(); it != objInstances.end(); ++it){
        objInstIdToIter[it->getId()] = it;
    }
    for(auto it = pendingObjInstances.begin(); it != pendingObjInstances.end(); ++it){
        pendingIdToIter[it->getId()] = it;
    }
}



