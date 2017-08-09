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

#include <algorithm>

#include <pcl/features/normal_3d.h>

#include <Misc.hpp>
#include <Exceptions.hpp>
#include "PlaneSeg.hpp"

using namespace std;

void PlaneSeg::calcSegProp(){
//    Eigen::Vector4f curPlaneParams;
    pcl::computePointNormal(*points, segPlaneParams, segCurv);

    segNormal = segPlaneParams.head<3>();
    segNormal.normalize();
    
    segNormalIntDiff = segCurv;

    // check if aligned with majority of normals
//    bool alignError = false;
    bool isAligned = Misc::checkIfAlignedWithNormals(segNormal, normals, normAlignConsistent);
//    if(normals->size() > 100 && alignError){
//        cout << "normals->size() = " << normals->size() << ", curvature = " << segCurv << endl;
//        throw PLANE_EXCEPTION("Align error");
//    }
    if(!isAligned){
        segNormal = -segNormal;
    }

    // Same work as in pcl::computePointNormal - could be done better
    Eigen::Vector4f curCentr;
    pcl::compute3DCentroid(*points, curCentr);

    segCentroid = curCentr.head<3>();

    float curAreaCoeff = 0.0;
    for(int pt = 0; pt < points->size(); ++pt){
        pcl::PointXYZRGB curPt = points->at(pt);
        Eigen::Vector3f curPtCentroid = curPt.getVector3fMap() - segCentroid;
        float lenSq = curPtCentroid.squaredNorm();
        float normLen = segNormal.dot(curPtCentroid);
        float distPlaneSq = lenSq - normLen * normLen;
        if(!isnan(distPlaneSq)){
            curAreaCoeff = max(distPlaneSq, curAreaCoeff);
        }
    }
    areaEst = curAreaCoeff * pi;
}

PlaneSeg PlaneSeg::merge(const PlaneSeg &planeSeg, UnionFind &sets) {
    PlaneSeg merged;
    int mid = sets.findSet(id);
    merged.setId(mid);
    merged.addPointsAndNormals(points, normals);
    merged.addPointsAndNormals(planeSeg.getPoints(), planeSeg.getNormals());
    
    merged.addOrigPlaneSegs(origPlaneSegs);
    merged.addOrigPlaneSegs(planeSeg.getOrigPlaneSegs());
    
    merged.calcSegProp();
    
    set<pair<int, int>> addedEdges;
    
    for(const int &as : adjSegs){
        int nh = sets.findSet(as);
        
        pair<int, int> ep = make_pair(min(mid, nh), max(mid, nh));
        if(mid != nh &&
            addedEdges.count(ep) == 0)
        {
            addedEdges.insert(ep);
            merged.addAdjSeg(nh);
        }
    }
    for(const int &as : planeSeg.getAdjSegs()){
        int nh = sets.findSet(as);
        
        pair<int, int> ep = make_pair(min(mid, nh), max(mid, nh));
        if(mid != nh &&
           addedEdges.count(ep) == 0)
        {
            addedEdges.insert(ep);
            merged.addAdjSeg(nh);
        }
    }
    
    return merged;
}
