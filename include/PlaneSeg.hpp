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

#ifndef INCLUDE_PLANESEG_HPP
#define INCLUDE_PLANESEG_HPP

#include <vector>

#include <Eigen/Eigen>

#include <pcl/impl/point_types.hpp>
#include <pcl/common/common_headers.h>

#include "UnionFind.h"
#include "Types.hpp"
#include "Serialization.hpp"

class PlaneSeg {
public:
    PlaneSeg()
            : id(-1),
              label(0),
              normAlignConsistent(true),
              points(new pcl::PointCloud<pcl::PointXYZRGB>()),
              normals(new pcl::PointCloud<pcl::Normal>())
    {}
    PlaneSeg(int iid,
             int ilabel,
             pcl::PointCloud<pcl::PointXYZRGB>::Ptr ipoints,
             pcl::PointCloud<pcl::Normal>::Ptr inormals,
             const std::vector<int> &iorigPlaneSegs,
             const std::vector<int> &iadjSegs)
            : id(iid),
              label(ilabel),
              normAlignConsistent(true),
              points(ipoints),
              normals(inormals),
              origPlaneSegs(iorigPlaneSegs),
              adjSegs(iadjSegs)
    {
        calcSegProp();
    }

    void setId(int id) {
        PlaneSeg::id = id;
    }
    void setPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr ipoints){
        points = ipoints;
    }

    void setNormals(pcl::PointCloud<pcl::Normal>::Ptr inormals){
        normals = inormals;
    }

    void setLabel(int ilabel){
        label = ilabel;
    }
    
    void setSegNormal(Eigen::Vector3f isegNormal){
        segNormal = isegNormal;
    }
    
    void setOrigPlaneSegs(const std::vector<int> &origPlaneSegs) {
        PlaneSeg::origPlaneSegs = origPlaneSegs;
    }
    
    void setSegCentroid(Eigen::Vector3f isegCentroid){
        segCentroid = isegCentroid;
    }

    void setAdjSegs(const std::vector<int> &iadjSegs) {
        adjSegs = iadjSegs;
    }

    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &getPoints() const {
        return points;
    }

    const pcl::PointCloud<pcl::Normal>::Ptr &getNormals() const {
        return normals;
    }
    
    const std::vector<int> &getOrigPlaneSegs() const {
        return origPlaneSegs;
    }
    
    int getId() const {
        return id;
    }

    int getLabel() const {
        return label;
    }

    const Eigen::Vector3f &getSegNormal() const {
        return segNormal;
    }
    
    double getSegNormalIntDiff() const {
        return segNormalIntDiff;
    }
    
    float getSegCurv() const {
        return segCurv;
    }

    const Eigen::Vector3f &getSegCentroid() const {
        return segCentroid;
    }
    
    const Eigen::Matrix3f &getSegCovar() const {
        return segCovar;
    }
    
    const std::vector<Eigen::Vector3f> &getEvecs() const {
        return evecs;
    }
    
    const std::vector<double> &getEvals() const {
        return evals;
    }
    
    const std::vector<int> &getAdjSegs() const {
        return adjSegs;
    }

    float getAreaEst() const {
        return areaEst;
    }

    pcl::PointNormal getPointNormal() const {
        pcl::PointNormal pt;
        pt.x = segCentroid[0];
        pt.y = segCentroid[1];
        pt.z = segCentroid[2];
        pt.normal_x = segNormal[0];
        pt.normal_y = segNormal[1];
        pt.normal_z = segNormal[2];

        return pt;
    }

    const Eigen::Vector4f& getSegPlaneParams() const {
        return segPlaneParams;
    }

    bool isNormAlignConsistent(){
        return normAlignConsistent;
    }

    void addPointAndNormal(pcl::PointXYZRGB newPt, pcl::Normal newNorm){
        points->push_back(newPt);
        normals->push_back(newNorm);
    }

    void addPointsAndNormals(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr newPts,
                             pcl::PointCloud<pcl::Normal>::ConstPtr newNorms){
        points->insert(points->end(), newPts->begin(), newPts->end());
        normals->insert(normals->end(), newNorms->begin(), newNorms->end());
    }

    void addOrigPlaneSegs(const std::vector<int> &newOrigPlaneSegs){
        origPlaneSegs.insert(origPlaneSegs.end(), newOrigPlaneSegs.begin(), newOrigPlaneSegs.end());
    }
    
    void addAdjSeg(int newNh){
        adjSegs.push_back(newNh);
    }
    
    void addAdjSegs(const std::vector<int> &newAdjSegs){
        adjSegs.insert(adjSegs.end(), newAdjSegs.begin(), newAdjSegs.end());
    }
    
    void calcSegProp(bool filter = false);
    
    void transform(Vector7d transform);
    
    PlaneSeg merge(const PlaneSeg &planeSeg, UnionFind &sets);

private:
    int id;
    int label;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    std::vector<int> origPlaneSegs;
    
    Eigen::Vector3f segNormal;
    double segNormalIntDiff;
    Eigen::Vector4f segPlaneParams;
    Eigen::Vector3f segCentroid;
    Eigen::Matrix3f segCovar;
    std::vector<Eigen::Vector3f> evecs;
    std::vector<double> evals;
    float segCurv;
    bool normAlignConsistent;
    float areaEst;
    
    std::vector<int> adjSegs;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & id;
        ar & label;
        ar & points;
        ar & normals;
        ar & origPlaneSegs;
        ar & segNormal;
        ar & segNormalIntDiff;
        ar & segPlaneParams;
        ar & segCentroid;
        ar & segCovar;
        ar & evecs;
        ar & evals;
        ar & segCurv;
        ar & normAlignConsistent;
        ar & areaEst;
        ar & adjSegs;
    }
};


#endif //INCLUDE_PLANESEG_HPP
