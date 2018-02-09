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

#ifndef INCLUDE_MAP_HPP_
#define INCLUDE_MAP_HPP_

class Map;

#include <vector>
#include <list>
#include <set>
#include <memory>

//#include <opencv2/opencv.hpp>

#include "ObjInstance.hpp"

struct PendingMatch {
    std::set<int> matchedIds;
    
    int eol;
    
    std::vector<std::list<ObjInstance>::iterator> objInstanceIts;
    
    std::vector<std::list<ObjInstance>::iterator> pendingObjInstanceIts;
};

struct PendingMatchKey{
    std::set<int> matchedIds;
    
    std::shared_ptr<PendingMatch> pmatch;
};

bool operator<(const PendingMatchKey &lhs, const PendingMatchKey &rhs);

class Map{
public:
	Map();
	
	Map(const cv::FileStorage& settings);

	inline void addObj(ObjInstance& obj){
		objInstances.push_back(obj);
	}
    
    inline void addObjs(std::vector<ObjInstance>::iterator beg,
                        std::vector<ObjInstance>::iterator end)
    {
        objInstances.insert(objInstances.end(), beg, end);
    }
    
    inline std::list<ObjInstance>::iterator removeObj(std::list<ObjInstance>::iterator it){
		return objInstances.erase(it);
	}

	inline int size(){
		return objInstances.size();
	}

//	inline ObjInstance& operator[](int i){
//		return objInstances[i];
//	}
    
    inline std::list<ObjInstance>::iterator begin(){
        return objInstances.begin();
    }
    
    inline std::list<ObjInstance>::iterator end(){
        return objInstances.end();
    }
    
    
    void addPendingObj(ObjInstance &obj,
                           const std::set<int> &matchedIds,
                           const std::vector<std::list<ObjInstance>::iterator> &objInstancesIts,
                           int eolAdd);
    
//    void addPendingObjs(std::vector<ObjInstance>::iterator beg,
//                        std::vector<ObjInstance>::iterator end,
//                        int eolAdd);
        
    void removePendingObjsEol();
    
    std::vector<PendingMatchKey> getPendingMatches(int eolThresh);
    
    void executePendingMatches(int eolThresh);
    
    bool getPendingMatch(PendingMatchKey &pendingMatch);
    
    inline int sizePending(){
        return pendingObjInstances.size();
    }
    
    void decreaseEol(int eolSub);
    
//	inline std::list<ObjInstance>::iterator pbegin(){
//		return pendingObjInstances.begin();
//	}
//
//	inline std::list<ObjInstance>::iterator pend(){
//		return pendingObjInstances.end();
//	}
//
//    inline std::set<PendingMatch> &getPendingMatches(){
//        return pendingMatches;
//    }
	
    inline pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr getOriginalPointCloud(){
        return originalPointCloud;
    }
private:
    pcl::PointCloud<pcl::PointXYZL>::Ptr getLabeledPointCloud();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorPointCloud();


	std::list<ObjInstance> objInstances;
    
//    std::map<int, std::list<ObjInstance>::iterator> idToIter;
    
    std::list<ObjInstance> pendingObjInstances;
    
    std::set<PendingMatchKey> pendingMatchesSet;
    
//    std::list<PendingMatch> pendingMatches;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr originalPointCloud;
};



#endif /* INCLUDE_MAP_HPP_ */
