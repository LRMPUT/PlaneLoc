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
#include <map>

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>

#include "ObjInstance.hpp"
#include "Serialization.hpp"

struct PendingMatch {
    std::set<int> matchedIds;
    
    int eol;
    
    std::vector<int> objInstanceIds;
    
    std::vector<int> pendingObjInstanceIds;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & matchedIds;
        ar & eol;
        ar & objInstanceIds;
        ar & pendingObjInstanceIds;
    }
};

struct PendingMatchKey{
    std::set<int> matchedIds;
    
    std::shared_ptr<PendingMatch> pmatch;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & matchedIds;
        ar & pmatch;
    }
};

bool operator<(const PendingMatchKey &lhs, const PendingMatchKey &rhs);

class Map{
public:
    struct Settings{
        int eolObjInstInit;
        
        int eolObjInstIncr;
        
        int eolObjInstDecr;
        
        int eolObjInstThresh;
        
        int eolPendingInit;
        
        int eolPendingIncr;
        
        int eolPendingDecr;
        
        int eolPendingThresh;
    
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & eolObjInstInit;
            ar & eolObjInstIncr;
            ar & eolObjInstDecr;
            ar & eolObjInstThresh;
            ar & eolPendingInit;
            ar & eolPendingIncr;
            ar & eolPendingDecr;
            ar & eolPendingThresh;
        }
    };
    
	Map();
	
	Map(const cv::FileStorage& fs);

	void addObj(ObjInstance& obj);
    
    void addObjs(vectorObjInstance::iterator beg,
                 vectorObjInstance::iterator end);
    
    inline listObjInstance::iterator removeObj(listObjInstance::iterator it){
		return objInstances.erase(it);
	}

	inline int size(){
		return objInstances.size();
	}

//	inline ObjInstance& operator[](int i){
//		return objInstances[i];
//	}
    
    inline listObjInstance::iterator begin(){
        return objInstances.begin();
    }
    
    inline listObjInstance::iterator end(){
        return objInstances.end();
    }
    
    void mergeNewObjInstances(vectorObjInstance &newObjInstances,
                              const std::map<int, int> &idToCnt = std::map<int, int>(),
                               pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                               int viewPort1 = -1,
                               int viewPort2 = -1);
    
    void mergeMapObjInstances(pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                              int viewPort1 = -1,
                              int viewPort2 = -1);
    
    void addPendingObj(ObjInstance &obj,
                           const std::set<int> &matchedIds,
                           int eolAdd);
    
//    void addPendingObjs(std::vector<ObjInstance>::iterator beg,
//                        std::vector<ObjInstance>::iterator end,
//                        int eolAdd);
        
    void removePendingObjsEol();
    
    void clearPending();
    
    std::vector<PendingMatchKey> getPendingMatches(int eolThresh);
    
    void executePendingMatches(int eolThresh);
    
    bool getPendingMatch(PendingMatchKey &pendingMatch);
    
    inline int sizePending(){
        return pendingObjInstances.size();
    }
    
    void decreasePendingEol(int eolSub);
    
    void decreaseObjEol(int eolSub);
    
    void removeObjsEol();
    
    void removeObjsEolThresh(int eolThresh);
    
    void removeObjsObsThresh(int obsThresh);
    
    void shiftIds(int startId);
    
    std::map<int, int> getVisibleObjs(Vector7d pose,
                                    cv::Mat cameraMatrix,
                                    int rows,
                                    int cols,
                                    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                    int viewPort1 = -1,
                                    int viewPort2 = -1);
	
    inline pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr getOriginalPointCloud(){
        return originalPointCloud;
    }
private:
    pcl::PointCloud<pcl::PointXYZL>::Ptr getLabeledPointCloud();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorPointCloud();

    void recalculateIdToIter();

	listObjInstance objInstances;
    
    std::map<int, listObjInstance::iterator> objInstIdToIter;
    
    listObjInstance pendingObjInstances;
    
    std::set<PendingMatchKey> pendingMatchesSet;
    
    std::map<int, listObjInstance::iterator> pendingIdToIter;
//    std::list<PendingMatch> pendingMatches;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr originalPointCloud;
    
    Settings settings;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const {
        ar << objInstances;
        ar << pendingObjInstances;
        ar << pendingMatchesSet;
        ar << originalPointCloud;
        ar << settings;
    }
    
    template<class Archive>
    void load(Archive & ar, const unsigned int version) {
        ar >> objInstances;
        ar >> pendingObjInstances;
        ar >> pendingMatchesSet;
        ar >> originalPointCloud;
        ar >> settings;
        
        recalculateIdToIter();
    }
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        boost::serialization::split_member(ar, *this, version);
    }
};



#endif /* INCLUDE_MAP_HPP_ */
