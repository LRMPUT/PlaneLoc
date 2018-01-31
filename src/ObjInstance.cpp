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

#include <thread>
#include <chrono>
#include <vector>
#include <list>

//#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>

#include "ObjInstance.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"
#include "Matching.hpp"
#include "UnionFind.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/IO/io.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>

using namespace std;

ObjInstance::ObjInstance(int iid,
					ObjType itype,
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr ipoints,
					const std::vector<PlaneSeg>& isvs)
	: id(iid),
	  type(itype),
	  points(ipoints),
	  svs(isvs),
	  convexHull(new pcl::PointCloud<pcl::PointXYZRGB>())
{
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(points);
    
    Eigen::Matrix3f evecs = pca.getEigenVectors();
    Eigen::Vector3f evals = pca.getEigenValues();
    Eigen::Vector4f pcaMean = pca.getMean();
    
    Eigen::Vector3f ev0 = evecs.block<3, 1>(0, 0);
    Eigen::Vector3f ev1 = evecs.block<3, 1>(0, 1);
    Eigen::Vector3f ev2 = evecs.block<3, 1>(0, 2);
    
    // shorter side of the plane is the second largest eigenvalue
    shorterComp = sqrt(evals(1)/points->size());
    
    // the eigenvector for the smallest eigenvalue is the normal vector
    paramRep.head<3>() = ev2.cast<double>();
    // distance is the dot product of normal and point lying on the plane
    paramRep(3) = -ev2.dot(pcaMean.head<3>());
    
    princComp = vector<Eigen::Vector3d>{ev0.cast<double>(), ev1.cast<double>(), ev2.cast<double>()};
    princCompLens = vector<double>{evals(0), evals(1), evals(2)};
    
    curv = evals(2) / (evals(0) + evals(1) + evals(2));
    
//	Eigen::Vector4f tmpParamRep;
//	pcl::computePointNormal(*points, tmpParamRep, curv);
    
    bool corrOrient = true;
    int corrCnt = 0;
    int incorrCnt = 0;
    for(int sv = 0; sv < svs.size(); ++sv){
        pcl::PointNormal svPtNormal;
        Eigen::Vector3d svNormal = svs[sv].getSegNormal().cast<double>();
        // if cross product between normal vectors is negative then it is wrongly oriented
        if(svNormal.dot(paramRep.head<3>()) < 0){
            ++incorrCnt;
        }
        else{
            ++corrCnt;
        }
    }
    if(incorrCnt > corrCnt){
        corrOrient = false;
    }
    if(incorrCnt != 0 && corrCnt != 0){
//        throw PLANE_EXCEPTION("Some normals correct and some incorrect");
		cout << "Some normals correct and some incorrect" << endl;
        for(int sv = 0; sv < svs.size(); ++sv) {
            // if cross product between normal vectors is negative then it is wrongly oriented
            Eigen::Vector3f svNormal = svs[sv].getSegNormal();
            cout << "svNormal[" << sv << "] = " << svNormal.transpose() << endl;
        }
    }
    // flip the normal
    if(!corrOrient){
        paramRep = -paramRep;
    }
    // normal including distance from origin
    normal = paramRep;
    
    // normalize paramRep
	Misc::normalizeAndUnify(paramRep);

//	Eigen::Vector3f planeNorm = tmpParamRep.head<3>();
//	double planeNormNorm = planeNorm.norm();
//	planeNorm /= planeNormNorm;
//	double d = tmpParamRep(3)/planeNormNorm;
//
//    normal.head<3>() = planeNorm.cast<double>();
//    normal[3] = d;
//    cout << "normal = " << normal.transpose() << endl;
//    cout << "paramRep = " << paramRep.transpose() << endl;

	pcl::ModelCoefficients::Ptr mdlCoeff (new pcl::ModelCoefficients);
	mdlCoeff->values.resize(4);
	mdlCoeff->values[0] = normal(0);
	mdlCoeff->values[1] = normal(1);
	mdlCoeff->values[2] = normal(2);
	mdlCoeff->values[3] = normal(3);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointsProj(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::ProjectInliers<pcl::PointXYZRGB> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(points);
	proj.setModelCoefficients(mdlCoeff);
	proj.filter(*pointsProj);
    
    {
        typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
        typedef K::FT                                                FT;
        typedef K::Point_2                                           Point;
        typedef K::Segment_2                                         Segment;
        typedef CGAL::Alpha_shape_vertex_base_2<K>                   Vb;
        typedef CGAL::Alpha_shape_face_base_2<K>                     Fb;
        typedef CGAL::Triangulation_data_structure_2<Vb,Fb>          Tds;
        typedef CGAL::Delaunay_triangulation_2<K,Tds>                Triangulation_2;
        typedef CGAL::Alpha_shape_2<Triangulation_2>                 Alpha_shape_2;
        typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;
        typedef Alpha_shape_2::Alpha_shape_vertices_iterator         Alpha_shape_vertices_iterator;
        typedef CGAL::Polygon_2<K>                                   Polygon_2;
        typedef CGAL::Polyline_simplification_2::Stop_above_cost_threshold Stop;
        typedef CGAL::Polyline_simplification_2::Squared_distance_cost     Cost;
        
        CGAL::set_pretty_mode(std::cout);
        
        Eigen::Vector3d plNormal = normal.head<3>();
        double plD = normal(3);
    
        // point on plane nearest to origin
        Eigen::Vector3d origin = plNormal * (-plD);
        Eigen::Vector3d xAxis, yAxis;
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
    
//        if(viewer){
//            Eigen::Affine3f trans = Eigen::Affine3f::Identity();
//            trans.matrix().block<3, 1>(0, 3) = origin.cast<float>();
//            trans.matrix().block<3, 1>(0, 0) = xAxis.cast<float>();
//            trans.matrix().block<3, 1>(0, 1) = yAxis.cast<float>();
//            trans.matrix().block<3, 1>(0, 2) = plNormal.cast<float>();
//            //		trans.fromPositionOrientationScale(, rot, 1.0);
//            viewer->addCoordinateSystem(0.5, trans, "plane coord", viewPort2);
//
//            viewer->initCameraParameters();
//            viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//        }
//	cout << "chull1->size() = " << chull1->size() << endl;
//	cout << "poly1.vertices = " << poly1.vertices << endl;
//	cout << "chull2->size() = " << chull2->size() << endl;
//	cout << "poly2.vertices = " << poly2.vertices << endl;
        
        list<Point> points;
        for(int i = 0; i < pointsProj->size(); ++i){
            Eigen::Vector3d chPt = pointsProj->at(i).getVector3fMap().cast<double>();
            Point projPt((chPt - origin).dot(xAxis),
                         (chPt - origin).dot(yAxis));
            points.push_back(projPt);
        }
        
        
        
    
        Alpha_shape_2 A(points.begin(), points.end(),
                        FT(0.05),
                        Alpha_shape_2::GENERAL);
        cout << "alpha = " << A.get_alpha() << endl;

//        std::vector<Point> alphaPoints;
//        auto outIt = std::back_inserter(alphaPoints);
//        Alpha_shape_vertices_iterator it = A.alpha_shape_vertices_begin(),
//                end = A.alpha_shape_vertices_end();
//        for( ; it!=end; ++it, ++outIt) {
//            *outIt = (*it)->point();
//        }
//        for(int p = 0; p < alphaPoints.size(); ++p){
//            Point pt = alphaPoints[p];
//
//            Eigen::Vector2d curPt(pt.x(), pt.y());
//            cout << "curPt = " << curPt.transpose() << endl;
//
//            pcl::PointXYZRGB curPt3d;
//            Eigen::Vector3d curCoord = origin + curPt.x() * xAxis + curPt.y() * yAxis;
//            curPt3d.getVector3fMap() = curCoord.cast<float>();
//            curPt3d.r = 255;
//            curPt3d.g = 255;
//            curPt3d.b = 255;
//            convexHull->push_back(curPt3d);
//            convexHullPolygon.vertices.push_back(convexHull->size() - 1);
//        }
        
//        std::vector<Alpha_shape_2::Edge> edges;
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
        vector<Polygon_2> polygons;
        {
            multimap<Point, int> tgtToIdx;
            multimap<Point, int> srcToIdx;
            for(int s = 0; s < segments.size(); ++s){
//                if(tgtToIdx.count(segments[s].target()) > 0){
//                    int oldIdx = tgtToIdx[segments[s].target()];
//                    cout << endl << "Multiple targets" << endl;
//                    cout << "src1 = " << segments[oldIdx].source() << endl;
//                    cout << "tgt1 = " << segments[oldIdx].target() << endl;
//                    cout << "src2 = " << segments[s].source() << endl;
//                    cout << "tgt2 = " << segments[s].target() << endl;
//                    cout << endl;
////                    throw PLANE_EXCEPTION("Multiple targets");
//                }
//                if(srcToIdx.count(segments[s].source()) > 0){
//                    int oldIdx = srcToIdx[segments[s].source()];
//                    cout << endl << "Multiple sources" << endl;
//                    cout << "src1 = " << segments[oldIdx].source() << endl;
//                    cout << "tgt1 = " << segments[oldIdx].target() << endl;
//                    cout << "src2 = " << segments[s].source() << endl;
//                    cout << "tgt2 = " << segments[s].target() << endl;
//                    cout << endl;
////                    throw PLANE_EXCEPTION("Multiple sources");
//                }
                
                tgtToIdx.insert(make_pair(segments[s].target(), s));
                srcToIdx.insert(make_pair(segments[s].source(), s));
            }
            vector<vector<int>> nextSeg(segments.size());
            vector<vector<int>> prevSeg(segments.size());
            for(int s = 0; s < segments.size(); ++s){
                // pair of iterators
                auto next = srcToIdx.equal_range(segments[s].target());
                auto prev = tgtToIdx.equal_range(segments[s].source());
                for(auto it = next.first; it != next.second; ++it){
                    nextSeg[s].push_back(it->second);
                }
                for(auto it = prev.first; it != prev.second; ++it){
                    prevSeg[s].push_back(it->second);
                }
//                if(nextSeg[s].size() != prevSeg[s].size()){
//                    cout << "nextSeg[s].size() = " << nextSeg[s].size() << endl;
//                    cout << "prevSeg[s].size() = " << prevSeg[s].size() << endl;
//                    throw PLANE_EXCEPTION("Number of in and out connections not equal");
//                }
            }
            
            double bestArea = 0.0;
            vector<bool> isVisited(segments.size(), false);
            cout << "starting" << endl;
            for(int s = 0; s < segments.size(); ++s){
                if(!isVisited[s]) {
                    
                    stack<int> sSeg;
                    sSeg.push(s);
                    while(!sSeg.empty()) {
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
                            if(!advanced){
                                explore = false;
                            }
                        }
                        // no other path, so we have a loop
                        vector<int> curVisited;
                        bool loopStart = false;
                        while(!sSeg.empty() && !loopStart){
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
                            if(lastNext.count(curIdx)){
                                loopStart;
                            }
                        }
    
//                        vector<Segment> curSegments;
                        Polygon_2 poly;
                        for(int cs = 0; cs < curVisited.size(); ++cs){
                            const Segment &curSegment = segments[curVisited[cs]];
                            poly.push_back(curSegment.target());
                        }
                        
                        poly = CGAL::Polyline_simplification_2::simplify(poly,
                                                                         Cost(),
                                                                         Stop(0.05 * 0.05));
                        double area = poly.area();
                        cout << "area = " << area << endl;
//                        cout << curSegments.size() << "/" << segments.size() << endl;
                        if (abs(area) > bestArea) {
                            polygons.clear();
                            polygons.push_back(poly);
                            bestArea = abs(area);
                        }
                    }
                }
            }
        }
        
        for(int p = 0; p < polygons.size(); ++p) {
            Eigen::Vector2d prevPtTgt(std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max());
            for (int s = 0; s < polygons[p].size(); ++s) {
//                Point src = polygons[p][s];
                Point tgt = polygons[p][s];
        
//                Eigen::Vector2d curPtSrc(src.x(), src.y());
                Eigen::Vector2d curPtTgt(tgt.x(), tgt.y());
//            cout << "curPtSrc = " << curPtSrc.transpose() << endl;
//            cout << "curPtTgt = " << curPtTgt.transpose() << endl;
        
//                if (prevPtTgt != curPtSrc) {
//                    pcl::PointXYZRGB curPtSrc3d;
//                    Eigen::Vector3d curCoordSrc =
//                            origin + curPtSrc.x() * xAxis + curPtSrc.y() * yAxis;
//                    curPtSrc3d.getVector3fMap() = curCoordSrc.cast<float>();
//                    curPtSrc3d.r = 255;
//                    curPtSrc3d.g = 255;
//                    curPtSrc3d.b = 255;
//                    convexHull->push_back(curPtSrc3d);
//                    convexHullPolygon.vertices.push_back(convexHull->size() - 1);
//                }
                {
                    pcl::PointXYZRGB curPtTgt3d;
                    Eigen::Vector3d curCoordTgt =
                            origin + curPtTgt.x() * xAxis + curPtTgt.y() * yAxis;
                    curPtTgt3d.getVector3fMap() = curCoordTgt.cast<float>();
                    curPtTgt3d.r = 255;
                    curPtTgt3d.g = 255;
                    curPtTgt3d.b = 255;
                    convexHull->push_back(curPtTgt3d);
                    convexHullPolygon.vertices.push_back(convexHull->size() - 1);
                }
        
                prevPtTgt = curPtTgt;
            }
        }
    }
//	pcl::ConcaveHull<pcl::PointXYZRGB> chull;
//    chull.setAlpha(0.5);
//    chull.setDimension(2);
////	chull.setComputeAreaVolume(true);
//	vector<pcl::Vertices> polygon;
//	chull.setInputCloud(pointsProj);
//	chull.reconstruct(*convexHull, polygon);
//	if(polygon.size() != 1){
////		throw PLANE_EXCEPTION("Error - 3D convex hull");
//	}
//	convexHullPolygon = polygon[0];
    
    pcl::PointCloud<pcl::PointXYZRGB> chullPoints;
    for(int p = 0; p < convexHullPolygon.vertices.size(); ++p){
        chullPoints.push_back(convexHull->at(convexHullPolygon.vertices[p]));
    }
	chullArea = pcl::calculatePolygonArea(chullPoints);

    cout << "Number of polygons: " << convexHullPolygon.vertices.size() << endl;
    cout << "Polygon area: " << chullArea << endl;
    
//	Eigen::Vector3d centr(0,0,0);
//	for(int p = 0; p < pointsProj->size(); ++p){
//		Eigen::Vector3d curPt;
//		curPt.x() = pointsProj->at(p).x;
//		curPt.y() = pointsProj->at(p).y;
//		curPt.z() = pointsProj->at(p).z;
//		centr += curPt;
//	}
//	centr /= pointsProj->size();
//	double maxR = 0.0;
//	for(int p = 0; p < pointsProj->size(); ++p){
//		Eigen::Vector3d curPt;
//		curPt.x() = pointsProj->at(p).x;
//		curPt.y() = pointsProj->at(p).y;
//		curPt.z() = pointsProj->at(p).z;
//		maxR = max((curPt - centr).norm(), maxR);
//	}
//	cout << "paramRep = " << paramRep.transpose() << endl;
//	cout << "chullArea = " << chullArea << endl;
//	cout << "approx size = " << pi*maxR*maxR << endl;
}

void ObjInstance::transform(Vector7d transform) {
    g2o::SE3Quat transformSE3Quat(transform);
    Eigen::Matrix4d transformMat = transformSE3Quat.to_homogeneous_matrix();
    Eigen::Matrix3d R = transformMat.block<3, 3>(0, 0);
    Eigen::Matrix4d Tinvt = transformMat.inverse();
    Tinvt = Tinvt.transpose();
    
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
    pcl::transformPointCloud(*points, *points, transformMat);

    // std::vector<PlaneSeg> svs;
    for(PlaneSeg &pseg : svs){
        pseg.transform(transform);
    }

    // Eigen::Vector4d paramRep;
    paramRep = Tinvt * paramRep;
    Misc::normalizeAndUnify(paramRep);
    
    // Eigen::Vector4d normal;
    normal = Tinvt * normal;
    
    // std::vector<Eigen::Vector3d> princComp;
    for(Eigen::Vector3d &pc : princComp){
        pc = R * pc;
    }
    
    // std::vector<double> princCompLens;
    // no need to transform
    
    // double shorterComp;
    // no need to transform
    
    // float curv;
    // no need to transform
    
    // double chullArea;
    // no need to transform
    
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr convexHull;
    pcl::transformPointCloud(*convexHull, *convexHull, transformMat);
    
    // pcl::Vertices convexHullPolygon;
    // no need to transform
    
    // std::vector<LineSeg> lineSegs;
    // TODO
}

std::vector<ObjInstance> ObjInstance::mergeObjInstances(const std::vector<std::vector<ObjInstance>>& objInstances,
                                                        pcl::visualization::PCLVisualizer::Ptr viewer,
                                                        int viewPort1,
                                                        int viewPort2)
{
    static constexpr double shadingLevel = 0.01;

    vector<ObjInstance> retObjInstances;

    if(viewer){
        viewer->removeAllPointClouds(viewPort1);
        viewer->removeAllShapes(viewPort1);
        viewer->removeAllPointClouds(viewPort2);
        viewer->removeAllShapes(viewPort2);

        viewer->initCameraParameters();
        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
    }

    int planesCnt = 0;
    vector<vector<int> > planeIds;
    for(int ba = 0; ba < objInstances.size(); ++ba){
        planeIds.emplace_back(objInstances[ba].size());
        for(int pl = 0; pl < objInstances[ba].size(); ++pl){
            planeIds[ba][pl] = planesCnt++;
        }
    }
    UnionFind ufSets(planesCnt);
    Vector7d transform;
    // identity
    transform << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    for(int ba = 0; ba < objInstances.size(); ++ba){

        if(viewer){
            viewer->removeAllPointClouds(viewPort1);
            viewer->removeAllShapes(viewPort1);

            for(int pl = 0; pl < objInstances[ba].size(); ++pl){
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = objInstances[ba][pl].getPoints();

                viewer->addPointCloud(curPl, string("plane_ba_") + to_string(pl), viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
//                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                         0.1, 0.1, 0.1,
//                                                         string("plane_ba_") + to_string(pl),
//                                                         viewPort1);
            }
        }

        for(int pl = 0; pl < objInstances[ba].size(); ++pl){
            const ObjInstance& curObj = objInstances[ba][pl];
            Eigen::Vector3d curObjNormal = curObj.getNormal().head<3>();

            if(viewer){
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         0.5,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
    
                pcl::Vertices chullPolygon;
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullPointCloud = curObj.getConvexHull(chullPolygon);
                chullPolygon.vertices.push_back(chullPolygon.vertices.front());
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
                for(int p = 0; p < chullPolygon.vertices.size(); ++p){
                    chullPoints->push_back(chullPointCloud->at(chullPolygon.vertices[p]));
                }
                viewer->addPolygonMesh<pcl::PointXYZRGB>(chullPointCloud,
                                                       vector<pcl::Vertices>{chullPolygon},
                                                       string("polygon_ba_") + to_string(pl),
                                                       viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                         0,
                                                         1.0,
                                                         0,
                                                         string("polygon_ba_") + to_string(pl),
                                                         viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         0.5,
                                                         string("polygon_ba_") + to_string(pl),
                                                         viewPort1);
                viewer->addPolygon<pcl::PointXYZRGB>(chullPoints,
                                   1.0,
                                   0.0,
                                   0.0,
                                   string("polyline_ba_") + to_string(pl),
                                   viewPort1);
                viewer->addPointCloud(chullPoints,
                                      string("polygon_pc_ba_") + to_string(pl),
                                      viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                         0.0, 0.0, 1.0,
                                                         string("polygon_pc_ba_") + to_string(pl),
                                                         viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                         4,
                                                         string("polygon_pc_ba_") + to_string(pl),
                                                         viewPort1);
                cout << "polygon for plane (pl) " << pl << ", size = " << chullPolygon.vertices.size() << endl;
            }
            for(int cba = ba; cba < objInstances.size(); ++cba){

                if(viewer){
                    viewer->removeAllPointClouds(viewPort2);
                    viewer->removeAllShapes(viewPort2);

                    for(int cpl = 0; cpl < objInstances[cba].size(); ++cpl){
                        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = objInstances[cba][cpl].getPoints();

                        viewer->addPointCloud(curPl, string("plane_cba_") + to_string(cpl), viewPort2);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 shadingLevel,
                                                                 string("plane_cba_") + to_string(cpl),
                                                                 viewPort2);
//                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                                 0.1, 0.1, 0.1,
//                                                                 string("plane_cba_") + to_string(cpl),
//                                                                 viewPort2);
                    }
                }

                int startCpl = 0;
                if(cba == ba){
                    startCpl = pl + 1;
                }
                for(int cpl = startCpl; cpl < objInstances[cba].size(); ++cpl){
                    const ObjInstance& compObj = objInstances[cba][cpl];
                    Eigen::Vector3d compObjNormal = compObj.getNormal().head<3>();

                    if(viewer){
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 0.5,
                                                                 string("plane_cba_") + to_string(cpl),
                                                                 viewPort2);
    
                        pcl::Vertices chullPolygon;
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullPointCloud = compObj.getConvexHull(chullPolygon);
                        chullPolygon.vertices.push_back(chullPolygon.vertices.front());
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
                        for(int p = 0; p < chullPolygon.vertices.size(); ++p){
                            chullPoints->push_back(chullPointCloud->at(chullPolygon.vertices[p]));
                        }
                        
                        viewer->addPolygonMesh<pcl::PointXYZRGB>(chullPointCloud,
                                                               vector<pcl::Vertices>{chullPolygon},
                                                               string("polygon_cba_") + to_string(cpl),
                                                               viewPort2);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                                 0,
                                                                 1.0,
                                                                 0,
                                                                 string("polygon_cba_") + to_string(cpl),
                                                                 viewPort2);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 0.5,
                                                                 string("polygon_cba_") + to_string(cpl),
                                                                 viewPort2);
                        viewer->addPolygon<pcl::PointXYZRGB>(chullPoints,
                                           1.0,
                                           0.0,
                                           0.0,
                                           string("polyline_cba_") + to_string(cpl),
                                           viewPort2);
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                            4,
                                                            string("polyline_cba_") + to_string(cpl),
                                                            viewPort2);
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                            pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                            string("polyline_cba_") + to_string(cpl),
                                                            viewPort2);
                        viewer->addPointCloud(chullPoints,
                                              string("polygon_pc_cba_") + to_string(cpl),
                                              viewPort2);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                                 0.0, 0.0, 1.0,
                                                                 string("polygon_pc_cba_") + to_string(cpl),
                                                                 viewPort2);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                 4,
                                                                 string("polygon_pc_cba_") + to_string(cpl),
                                                                 viewPort2);
                        cout << "polygon for plane (cpl) " << cpl << ", size = " << chullPolygon.vertices.size() << endl;
    
//                        for(int p = 1; p < chullPolygon.vertices.size(); ++p){
//                            viewer->addLine(chullPointCloud->at(chullPolygon.vertices[p - 1]),
//                                            chullPointCloud->at(chullPolygon.vertices[p]),
//                                            1.0, 0.0, 0.0,
//                                            "cur_line",
//                                            viewPort2);
//                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
//                                                                4,
//                                                                "cur_line",
//                                                                viewPort2);
//                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
//                                                                pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
//                                                                "cur_line",
//                                                                viewPort2);
//                            cout << "point " << chullPointCloud->at(chullPolygon.vertices[p]) << endl;
//
//                            viewer->resetStoppedFlag();
//                            while (!viewer->wasStopped()){
//                                viewer->spinOnce (100);
//                                std::this_thread::sleep_for(std::chrono::milliseconds(50));
//                            }
//
//                            viewer->removeShape("cur_line", viewPort2);
//
////                            cout << "point " << chullPolygon.vertices[p] << endl;
//                        }
                    }

                    double diff = Matching::planeEqDiffLogMap(curObj, compObj, transform);
//                    cout << "diff = " << diff << endl;
                    // if plane equation is similar
                    if(diff < 0.01){
                        double normDot = curObjNormal.dot(compObjNormal);
//                        cout << "normDot = " << normDot << endl;
                        // if the observed face is the same
                        if(normDot > 0){
                            double intArea = 0.0;
                            double intScore = Matching::checkConvexHullIntersection(curObj, compObj, transform, intArea);
//                            cout << "intScore = " << intScore << endl;
                            // if intersection of convex hulls is big enough
                            if(intScore > 0.3){
                                cout << "merging planes" << endl;
                                // join the objects
                                ufSets.unionSets(planeIds[ba][pl], planeIds[cba][cpl]);
                            }
                        }
                    }

                    if(viewer){
                        viewer->resetStoppedFlag();

//                        viewer->initCameraParameters();
//                        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                        while (!viewer->wasStopped()){
                            viewer->spinOnce (100);
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        }


                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                                 shadingLevel,
                                                                 string("plane_cba_") + to_string(cpl),
                                                                 viewPort2);
                        viewer->removePolygonMesh(string("polygon_cba_") + to_string(cpl), viewPort2);
                        viewer->removeShape(string("polyline_cba_") + to_string(cpl), viewPort2);
                        viewer->removePointCloud(string("polygon_pc_cba_") + to_string(cpl),
                                                 viewPort2);
                    }
                }
            }

            if(viewer){
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(pl),
                                                         viewPort1);
                viewer->removePolygonMesh(string("polygon_ba_") + to_string(pl), viewPort1);
                viewer->removeShape(string("polyline_ba_") + to_string(pl), viewPort1);
                viewer->removePointCloud(string("polygon_pc_ba_") + to_string(pl),
                                         viewPort1);
            }
        }
    }

    multimap<int, pair<int, int> > sets;
    for(int ba = 0; ba < objInstances.size(); ++ba) {
        for (int pl = 0; pl < objInstances[ba].size(); ++pl) {
            int setId = ufSets.findSet(planeIds[ba][pl]);
            sets.insert(make_pair(setId, make_pair(ba, pl)));
        }
    }
    if(viewer){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
    }
//    typedef multimap<int, pair<int, int> >::iterator mmIter;
    for(auto it = sets.begin(); it != sets.end(); ){
        auto range = sets.equal_range(it->first);

        vector<const ObjInstance*> curObjs;
        for(auto rangeIt = range.first; rangeIt != range.second; ++rangeIt){
            curObjs.emplace_back(&(objInstances[rangeIt->second.first][rangeIt->second.second]));
        }
        if(curObjs.size() == 1){
            retObjInstances.push_back(*curObjs.front());
        }
        else{
            retObjInstances.push_back(merge(curObjs,
                                            viewer,
                                            viewPort1,
                                            viewPort2));
        }
        
        it = range.second;
    
        if(viewer) {
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = retObjInstances.back().getPoints();
            viewer->addPointCloud(curPl, string("plane_ret_") + to_string(it->first), viewPort1);
        }
    }

    return retObjInstances;
}

ObjInstance ObjInstance::merge(const std::vector<const ObjInstance*>& objInstances,
                               pcl::visualization::PCLVisualizer::Ptr viewer,
                               int viewPort1,
                               int viewPort2)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr newPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<PlaneSeg> newSvs;

    Eigen::Vector3d meanLogMap;
    meanLogMap << 0.0, 0.0, 0.0;
    int sumPoints = 0;
    for(int o = 0; o < objInstances.size(); ++o){
        Eigen::Vector4d curParamRep = objInstances[o]->getParamRep();
        Eigen::Vector3d logMapParamRep = Misc::logMap(Eigen::Quaterniond(curParamRep));
        meanLogMap += logMapParamRep * objInstances[o]->getPoints()->size();
        sumPoints += objInstances[o]->getPoints()->size();
    }
    meanLogMap /= sumPoints;

    Eigen::Quaterniond meanParamRep = Misc::expMap(meanLogMap);
    Eigen::Vector4d meanPlaneEq;
    meanPlaneEq[0] = meanParamRep.x();
    meanPlaneEq[1] = meanParamRep.y();
    meanPlaneEq[2] = meanParamRep.z();
    meanPlaneEq[3] = meanParamRep.w();
    double normNorm = meanPlaneEq.head<3>().norm();
    meanPlaneEq.head<3>() /= normNorm;
    meanPlaneEq[3] /= normNorm;

    pcl::ModelCoefficients::Ptr mdlCoeff (new pcl::ModelCoefficients);
    mdlCoeff->values.resize(4);
    mdlCoeff->values[0] = meanPlaneEq[0];
    mdlCoeff->values[1] = meanPlaneEq[1];
    mdlCoeff->values[2] = meanPlaneEq[2];
    mdlCoeff->values[3] = meanPlaneEq[3];
    for(int o = 0; o < objInstances.size(); ++o){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointsProj(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::ProjectInliers<pcl::PointXYZRGB> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(objInstances[o]->getPoints());
        proj.setModelCoefficients(mdlCoeff);
        proj.filter(*pointsProj);

        const vector<PlaneSeg>& svs = objInstances[o]->getSvs();

        newPoints->insert(newPoints->end(), pointsProj->begin(), pointsProj->end());
        newSvs.insert(newSvs.end(), svs.begin(), svs.end());

        if(viewer) {
            viewer->addPointCloud(objInstances[o]->getPoints(), string("plane_o_") + to_string(o), viewPort1);

        }
    }

    if(viewer){
        viewer->addPointCloud(newPoints, string("plane_merged"), viewPort2);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 1.0, 0.0, 0.0,
                                                 string("plane_merged"),
                                                 viewPort2);

        viewer->resetStoppedFlag();

//                        viewer->initCameraParameters();
//                        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
        while (!viewer->wasStopped()){
            viewer->spinOnce (100);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        for(int o = 0; o < objInstances.size(); ++o){
            viewer->removePointCloud(string("plane_o_") + to_string(o), viewPort1);
        }
        viewer->removePointCloud(string("plane_merged"), viewPort2);
    }

    return ObjInstance(0,
                        ObjInstance::ObjType::Plane,
                        newPoints,
                        newSvs);
}


