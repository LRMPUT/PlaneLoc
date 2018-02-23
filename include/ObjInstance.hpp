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

#ifndef INCLUDE_OBJINSTANCE_HPP_
#define INCLUDE_OBJINSTANCE_HPP_

class ObjInstance;

#include <vector>
#include <string>

#include <boost/serialization/vector.hpp>

#include <opencv2/opencv.hpp>

#include <Eigen/Eigen>

#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "PlaneSeg.hpp"
#include "LineSeg.hpp"
#include "ConcaveHull.hpp"
#include "EKFPlane.hpp"
#include "Map.hpp"
#include "Serialization.hpp"

// only planes in a current version
class ObjInstance{
public:
	enum class ObjType{
		Plane,
		Unknown
	};
    
    ObjInstance();
    
    /**
     *
     * @param iid
     * @param itype
     * @param ipoints
     * @param isvs
     */
	ObjInstance(int iid,
				ObjType itype,
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr ipoints,
				const vectorPlaneSeg& isvs);
	
	void merge(const ObjInstance &other);

	inline int getId() const {
		return id;
	}

	inline ObjType getType() const {
		return type;
	}

	inline const pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPoints() const {
		return points;
	}

	inline const vectorPlaneSeg& getSvs() const {
		return svs;
	}

	inline Eigen::Vector4d getParamRep() const {
		return paramRep;
	}

    inline Eigen::Vector4d getNormal() const {
        return normal;
    }
    
    inline const vectorVector3d getPrincComp() const {
        return princComp;
    }

    inline const std::vector<double> &getPrincCompLens() const {
        return princCompLens;
    }

    inline double getShorterComp() const {
        return shorterComp;
    }
	
	inline float getCurv() const {
		return curv;
	}
	
    inline const ConcaveHull &getHull() const {
        return *hull;
	}
    
    inline const vectorLineSeg &getLineSegs() const {
        return lineSegs;
    }
    
//    inline void getQuatAndCovar(Eigen::Quaterniond &q,
//                                Eigen::Matrix4d &covar) const
//    {
//        q = quat;
//        covar = covarQuat;
//    }
    
    inline const EKFPlane &getEkf() const {
        return ekf;
    }
    
    int getEolCnt() const {
        return eolCnt;
    }
    
    void setEolCnt(int eolCnt) {
        ObjInstance::eolCnt = eolCnt;
    }
    
    void decreaseEolCnt(int eolSub){
        ObjInstance::eolCnt -= eolSub;
    }
    
    int getObsCnt() const {
        return obsCnt;
    }
    
    void setObsCnt(int obsCnt) {
        ObjInstance::obsCnt = obsCnt;
    }
    
    bool isTrial() const {
        return trial;
    }
    
    void setTrial(bool trial) {
        ObjInstance::trial = trial;
    }
    
    void transform(const Vector7d &transform);
    
    inline void addLineSeg(const LineSeg &newLineSeg){
        lineSegs.push_back(newLineSeg);
    }
    
    cv::Mat getColorHist() const {
        return colorHist;
    }
    
    static double compHistDist(cv::Mat hist1, cv::Mat hist2);
    
    static listObjInstance mergeObjInstances(std::vector<vectorObjInstance>& objInstances,
                                                      pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                                      int viewPort1 = -1,
                                                      int viewPort2 = -1);
    
    static void mergeObjInstances(Map &map,
                                 vectorObjInstance &newObjInstances,
                                 pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                 int viewPort1 = -1,
                                 int viewPort2 = -1);

//	static ObjInstance merge(const std::vector<const ObjInstance*>& objInstances,
//                             pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
//                             int viewPort1 = -1,
//                             int viewPort2 = -1);
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    void correctOrient();
    
    void compColorHist();
    
	int id;

	ObjType type;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;

	vectorPlaneSeg svs;

	Eigen::Vector4d paramRep;

    /**
     * Normal oriented towards observable side.
     */
	Eigen::Vector4d normal;

	vectorVector3d princComp;

    std::vector<double> princCompLens;

    double shorterComp;
    
    float curv;
    
    cv::Mat colorHist;

    std::shared_ptr<ConcaveHull> hull;
	
	vectorLineSeg lineSegs;
    
    EKFPlane ekf;
    
    int eolCnt;
    
    int obsCnt;
    
    bool trial;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & id;
        ar & type;
        ar & points;
        ar & svs;
        ar & paramRep;
        ar & normal;
        ar & princComp;
        ar & princCompLens;
        ar & shorterComp;
        ar & curv;
        ar & colorHist;
        ar & hull;
        ar & lineSegs;
        ar & ekf;
        ar & eolCnt;
        ar & obsCnt;
        ar & trial;
    }
};



#endif /* INCLUDE_OBJINSTANCE_HPP_ */
