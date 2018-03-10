//
// Created by jachu on 31.01.18.
//

#ifndef PLANELOC_CONCAVEHULL_HPP
#define PLANELOC_CONCAVEHULL_HPP

//#define CGAL_DISABLE_ROUNDING_MATH_CHECK

class ConcaveHull;

#include <vector>

#include <boost/serialization/vector.hpp>

#include <Eigen/Eigen>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/IO/io.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include "Types.hpp"
#include "Serialization.hpp"

class ConcaveHull {

public:
    typedef CGAL::Exact_predicates_exact_constructions_kernel  K;
    typedef K::FT                                                FT;
    typedef K::Point_2                                           Point_2;
    typedef K::Segment_2                                         Segment_2;
    typedef CGAL::Polygon_2<K>                                   Polygon_2;
    typedef CGAL::Polygon_with_holes_2<K>                        Polygon_holes_2;
    typedef CGAL::Polyline_simplification_2::Stop_above_cost_threshold Stop;
    typedef CGAL::Polyline_simplification_2::Squared_distance_cost     Cost;
    
    typedef CGAL::Exact_predicates_inexact_constructions_kernel    Kie;
    typedef Kie::FT                                                FTie;
    typedef Kie::Point_2                                           Point_2ie;
    typedef Kie::Segment_2                                         Segment_2ie;
    typedef CGAL::Polygon_2<Kie>                                   Polygon_2ie;
    typedef CGAL::Polygon_with_holes_2<Kie>                        Polygon_holes_2ie;
    typedef CGAL::Alpha_shape_vertex_base_2<Kie>                   Vb;
    typedef CGAL::Alpha_shape_face_base_2<Kie>                     Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb,Fb>            Tds;
    typedef CGAL::Delaunay_triangulation_2<Kie,Tds>                Triangulation_2;
    typedef CGAL::Alpha_shape_2<Triangulation_2>                 Alpha_shape_2;
    typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;
    typedef Alpha_shape_2::Alpha_shape_vertices_iterator         Alpha_shape_vertices_iterator;
    
    ConcaveHull();
    
    ConcaveHull(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ipoints3d,
                const Eigen::Vector4d &planeEq);
    
    ConcaveHull(const std::vector<Polygon_2> &polygons,
                    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &polygons3d,
                    const Eigen::Vector3d &plNormal,
                    double plD,
                    const Eigen::Vector3d &origin,
                    const Eigen::Vector3d &xAxis,
                    const Eigen::Vector3d &yAxis);
    
    ConcaveHull(const ConcaveHull &other);
    
    void init(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ipoints3d,
              const Eigen::Vector4d &planeEq);
    
    void init(const std::vector<Polygon_2> &polygons,
              const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &polygons3d,
              const Eigen::Vector3d &plNormal,
              double plD,
              const Eigen::Vector3d &origin,
              const Eigen::Vector3d &xAxis,
              const Eigen::Vector3d &yAxis);
    
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &getPolygons3d() const {
        return polygons3d;
    }
    
    const std::vector<double> &getAreas() const {
        return areas;
    }
    
    double getTotalArea() const {
        return totalArea;
    }

    ConcaveHull intersect(const ConcaveHull &other,
                          double areaThresh = 0.05) const;
    
    ConcaveHull intersect(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &otherPolygons3d,
                          double areaThresh = 0.05) const;
    
    ConcaveHull clipToCameraFrustum(const cv::Mat K,
                                    int rows,
                                    int cols,
                                    double minZ);
    
    ConcaveHull transform(const Vector7d &transform) const;
    
    double minDistance(const ConcaveHull &other) const;
    
    void display(pcl::visualization::PCLVisualizer::Ptr viewer,
                 int vp,
                 double r = 0.0,
                 double g = 1.0,
                 double b = 0.0) const ;
    
    void cleanDisplay(pcl::visualization::PCLVisualizer::Ptr viewer,
                      int vp) const;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    void computeFrame();
    
    Point_2 point3dTo2d(const Eigen::Vector3d &point3d) const;
    
    Point_2ie point3dTo2die(const Eigen::Vector3d &point3d) const;
    
    Eigen::Vector3d point2dTo3d(const Point_2 &point2d) const;
    
    std::vector<Polygon_2> polygons;
    std::vector<double> areas;
    double totalArea;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> polygons3d;
    
    Eigen::Vector3d plNormal;
    double plD;
    // point on the plane nearest to origin
    Eigen::Vector3d origin;
    Eigen::Vector3d xAxis, yAxis;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & polygons;
        ar & areas;
        ar & totalArea;
        ar & polygons3d;
        ar & plNormal;
        ar & plD;
        ar & origin;
        ar & xAxis;
        ar & yAxis;
    }
};


#endif //PLANELOC_CONCAVEHULL_HPP
