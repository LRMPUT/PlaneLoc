//
// Created by jachu on 31.01.18.
//

#include <iostream>
#include <map>


#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/project_inliers.h>
#include <g2o/types/slam3d/se3quat.h>

#include "ConcaveHull.hpp"

using namespace std;

ConcaveHull::ConcaveHull(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr points3d,
                         Eigen::Vector4d planeEq)
    : totalArea(0.0)
{
    pcl::ModelCoefficients::Ptr mdlCoeff (new pcl::ModelCoefficients);
    mdlCoeff->values.resize(4);
    mdlCoeff->values[0] = planeEq(0);
    mdlCoeff->values[1] = planeEq(1);
    mdlCoeff->values[2] = planeEq(2);
    mdlCoeff->values[3] = planeEq(3);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points3dProj(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(points3d);
    proj.setModelCoefficients(mdlCoeff);
    proj.filter(*points3dProj);
    
    plNormal = planeEq.head<3>();
    plD = planeEq(3);
    
    computeFrame();
    
    list<Point> points2d;
    for(int i = 0; i < points3dProj->size(); ++i){
        points2d.push_back(point3dTo2d(points3dProj->at(i).getVector3fMap().cast<double>()));
    }
    
    
    Alpha_shape_2 A(points2d.begin(), points2d.end(),
                    FT(0.05),
                    Alpha_shape_2::GENERAL);
    cout << "alpha = " << A.get_alpha() << endl;

    std::vector<Segment> segments;
    auto outIt = std::back_inserter(segments);
    Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
            end = A.alpha_shape_edges_end();
    for( ; it!=end; ++it) {
        if(A.classify(*it) == Alpha_shape_2::REGULAR) {
            *outIt = A.segment(*it);
            ++outIt;
        }
    }
    
    {
        multimap<Point, int> tgtToIdx;
        multimap<Point, int> srcToIdx;
        for (int s = 0; s < segments.size(); ++s) {
            
            
            tgtToIdx.insert(make_pair(segments[s].target(), s));
            srcToIdx.insert(make_pair(segments[s].source(), s));
        }
        vector<vector<int>> nextSeg(segments.size());
        vector<vector<int>> prevSeg(segments.size());
        for (int s = 0; s < segments.size(); ++s) {
            // pair of iterators
            auto next = srcToIdx.equal_range(segments[s].target());
            auto prev = tgtToIdx.equal_range(segments[s].source());
            for (auto it = next.first; it != next.second; ++it) {
                nextSeg[s].push_back(it->second);
            }
            for (auto it = prev.first; it != prev.second; ++it) {
                prevSeg[s].push_back(it->second);
            }
//                if(nextSeg[s].size() != prevSeg[s].size()){
//                    cout << "nextSeg[s].size() = " << nextSeg[s].size() << endl;
//                    cout << "prevSeg[s].size() = " << prevSeg[s].size() << endl;
//                    throw PLANE_EXCEPTION("Number of in and out connections not equal");
//                }
        }

//        double bestArea = 0.0;
        vector<bool> isVisited(segments.size(), false);
        cout << "starting" << endl;
        for (int s = 0; s < segments.size(); ++s) {
            if (!isVisited[s]) {
                
                stack<int> sSeg;
                sSeg.push(s);
                while (!sSeg.empty()) {
                    bool explore = true;
                    
                    set<int> lastNext;
                    while (explore) {
                        int curIdx = sSeg.top();
                        isVisited[curIdx] = true;
//                            Segment &curSeg = segments[curIdx];
                        
                        bool advanced = false;
                        lastNext.clear();
                        for (int n = 0; n < nextSeg[curIdx].size(); ++n) {
                            lastNext.insert(nextSeg[curIdx][n]);
                            if (!isVisited[nextSeg[curIdx][n]]) {
                                sSeg.push(nextSeg[curIdx][n]);
                                advanced = true;
                                
                                break;
                            }
                        }
                        if (!advanced) {
                            explore = false;
                        }
                    }
                    // no other path, so we have a loop
                    vector<int> curVisited;
                    bool loopStart = false;
                    while (!sSeg.empty() && !loopStart) {
                        int curIdx = sSeg.top();
                        
                        curVisited.push_back(curIdx);
                        sSeg.pop();

//                            // if there is another path then it is the loop start
//                            for (int n = 0; n < nextSeg[curIdx].size(); ++n) {
//                                if (!isVisited[nextSeg[curIdx][n]]) {
//                                    loopStart = true;
//                                }
//                            }
                        // if last visited segment had this segment in next then it is the loop start
                        if (lastNext.count(curIdx)) {
                            loopStart;
                        }
                    }

//                        vector<Segment> curSegments;
                    Polygon_2 poly;
                    for (auto it = curVisited.rbegin(); it != curVisited.rend(); ++it) {
                        const Segment &curSegment = segments[*it];
                        poly.push_back(curSegment.target());
                    }
                    
                    poly = CGAL::Polyline_simplification_2::simplify(poly,
                                                                     Cost(),
                                                                     Stop(0.05 * 0.05));
                    double area = poly.area();
                    cout << "area = " << area << endl;
//                        cout << curSegments.size() << "/" << segments.size() << endl;
                    if (abs(area) > 0.1) {
                        polygons.push_back(poly);
                        areas.push_back(abs(area));
                        totalArea += abs(area);
                    }
                }
            }
        }
    }
    for(int p = 0; p < polygons.size(); ++p) {
        polygons3d.emplace_back(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (int s = 0; s < polygons[p].size(); ++s) {
//                Point src = polygons[p][s];
            Point tgt = polygons[p][s];

            Eigen::Vector2d curPtTgt(tgt.x(), tgt.y());

            {
                pcl::PointXYZRGB curPtTgt3d;
                Eigen::Vector3d curCoordTgt = point2dTo3d(curPtTgt);
                curPtTgt3d.getVector3fMap() = curCoordTgt.cast<float>();
                curPtTgt3d.r = 255;
                curPtTgt3d.g = 255;
                curPtTgt3d.b = 255;
    
                polygons3d.back()->push_back(curPtTgt3d);
            }
        }
    }
}


ConcaveHull::ConcaveHull(const vector<ConcaveHull::Polygon_2> &polygons,
                         const vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &polygons3d,
                         const Eigen::Vector3d &plNormal,
                         double plD)
        : polygons(polygons),
          polygons3d(polygons3d),
          plNormal(plNormal),
          plD(plD),
          totalArea(0.0)
{
    computeFrame();
    
    for(const Polygon_2 &curPoly : polygons){
        areas.push_back(abs(curPoly.area()));
        totalArea += abs(curPoly.area());
    }
}

ConcaveHull ConcaveHull::transform(Vector7d transform) const {
    Eigen::Matrix4d transformMat = g2o::SE3Quat(transform).to_homogeneous_matrix();
    
    return ConcaveHull(boost::shared_ptr<const PointCloud <PointXYZRGB>>(),
                       Eigen::Matrix<double, 4, 1, 0, 4, 1>());
}

ConcaveHull ConcaveHull::intersect(const ConcaveHull &other) const {
    const vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &otherPolygons3d = other.getPolygons3d();
    vector<Polygon_2> otherPolygonsProj;
    // project points onto plane of this hull
    for(pcl::PointCloud<pcl::PointXYZRGB>::Ptr poly3d : otherPolygons3d){
        Polygon_2 polyProj;
        for(auto it = poly3d->begin(); it != poly3d->end(); ++it){
            polyProj.push_back(point3dTo2d(it->getVector3fMap().cast<double>()));
        }
        otherPolygonsProj.push_back(polyProj);
    }
    
    vector<Polygon_2> resPolygons;
    vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> resPolygons3d;
    for(const Polygon_2 &p1 : polygons){
        for(const Polygon_2 &p2 : otherPolygonsProj){
            vector<Polygon_holes_2> inter;
            CGAL::intersection(p1, p2, back_inserter(inter));
            for(Polygon_holes_2 &pi : inter){
                if(pi.outer_boundary().area() > 0.1){
                    resPolygons.push_back(pi.outer_boundary());
                }
            }
        }
    }
    for(const Polygon_2 &p : resPolygons){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr poly3d(new pcl::PointCloud<pcl::PointXYZRGB>::Ptr());
        for(int i = 0; i < p.size(); ++i){
            pcl::PointXYZRGB curPt3d;
            Eigen::Vector3d curCoordTgt = point2dTo3d(p[i]);
            curPt3d.getVector3fMap() = curCoordTgt.cast<float>();
            curPt3d.r = 255;
            curPt3d.g = 255;
            curPt3d.b = 255;
            
            poly3d->push_back(curPt3d);
        }
    }
    
    return ConcaveHull(resPolygons,
                       resPolygons3d,
                       plNormal,
                       plD);
}

void ConcaveHull::computeFrame() {
    // point on plane nearest to origin
    origin = plNormal * (-plD);
    //if normal vector is not parallel to global x axis
    if(plNormal.cross(Eigen::Vector3d(1.0, 0.0, 0.0)).norm() > 1e-2){
        // plane x axis as a cross product - always perpendicular to normal vector
        xAxis = plNormal.cross(Eigen::Vector3d(1.0, 0.0, 0.0));
        xAxis.normalize();
        yAxis = plNormal.cross(xAxis);
    }
    else{
        xAxis = plNormal.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
        xAxis.normalize();
        yAxis = plNormal.cross(xAxis);
    }
}

ConcaveHull::Point ConcaveHull::point3dTo2d(const Eigen::Vector3d &point3d) const {
    return Point((point3d - origin).dot(xAxis),
                 (point3d - origin).dot(yAxis));
}

Eigen::Vector3d ConcaveHull::point2dTo3d(const ConcaveHull::Point &point2d) const {
    return origin + point2d.x() * xAxis + point2d.y() * yAxis;
}

