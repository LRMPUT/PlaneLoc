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

#ifndef OBJEXTRACT_SVINFO_HPP
#define OBJEXTRACT_SVINFO_HPP

#include <vector>

#include <Eigen/Eigen>

#include <pcl/impl/point_types.hpp>
#include <pcl/common/common_headers.h>

class SegInfo {
public:
    SegInfo()
            : id(-1),
              label(0),
              normAlignConsistent(true),
              points(new pcl::PointCloud<pcl::PointXYZRGB>()),
              normals(new pcl::PointCloud<pcl::Normal>())
    {}
    SegInfo(int iid,
            int ilabel,
           pcl::PointCloud<pcl::PointXYZRGB>::Ptr ipoints,
           pcl::PointCloud<pcl::Normal>::Ptr inormals,
           std::vector<int> iadjSegs)
            : id(iid),
              label(ilabel),
              normAlignConsistent(true),
              points(ipoints),
              normals(inormals),
              adjSegs(iadjSegs)
    {
        calcSegProp();
    }

    void setId(int id) {
        SegInfo::id = id;
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

    int getId() const {
        return id;
    }

    int getLabel() const {
        return label;
    }

    const Eigen::Vector3f &getSegNormal() const {
        return segNormal;
    }

    float getSegCurv() const {
        return segCurv;
    }

    const Eigen::Vector3f &getSegCentroid() const {
        return segCentroid;
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

    void addAdjSeg(int newNh){
        adjSegs.push_back(newNh);
    }

    void calcSegProp();

private:
    int id;
    int label;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    Eigen::Vector3f segNormal;
    Eigen::Vector4f segPlaneParams;
    Eigen::Vector3f segCentroid;
    float segCurv;
    std::vector<int> adjSegs;
    bool normAlignConsistent;
    float areaEst;
};


#endif //OBJEXTRACT_SVINFO_HPP
