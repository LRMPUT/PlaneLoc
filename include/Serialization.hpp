//
// Created by jachu on 21.02.18.
//

#ifndef PLANELOC_SERIALIZATION_HPP
#define PLANELOC_SERIALIZATION_HPP

#include <sstream>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>

#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>

#include <opencv2/core/mat.hpp>

#include <Eigen/Sparse>
#include <Eigen/Dense>


#include <CGAL/IO/io.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

namespace boost {
    namespace serialization {
    
//        template<class Archive>
//        void serialize(Archive & ar, ConcaveHull::Point_2 &g, const unsigned int version) {
//            ar & g.x();
//            ar & g.y();
//        }
//
//        template<class Archive>
//        void save(Archive & ar, const ConcaveHull::Polygon_2 &g, const unsigned int version) {
//            size_t polySize = g.size();
//            ar << polySize;
//            for(int i = 0; i < polySize; ++i){
//                ar << g[i];
//            }
//        }
//
//        template<class Archive>
//        void load(Archive & ar, ConcaveHull::Polygon_2 &g, const unsigned int version) {
//            size_t polySize = 0;
//            ar >> polySize;
//            for(int i = 0; i < polySize; ++i){
//                ConcaveHull::Point_2 p;
//                ar >> p;
//                g.push_back(p);
//            }
//        }
        
        typedef ::CGAL::Exact_predicates_exact_constructions_kernel  K;
        typedef K::Point_2                                           Point_2;
        typedef K::Segment_2                                         Segment_2;
        typedef ::CGAL::Polygon_2<K>                                   Polygon_2;
        
        template<class Archive>
        void save(Archive & ar, const Polygon_2 &g, const unsigned int version) {
            std::stringstream ss;
            ss << g;
            std::string str = ss.str();
            ar << str;
        }

        template<class Archive>
        void load(Archive & ar, Polygon_2 &g, const unsigned int version) {
            std::string str;
            ar >> str;
            std::stringstream ss(str);
            ss >> g;
        }
    
        template<class Archive>
        void serialize(Archive & ar, Polygon_2 &g, const unsigned int version) {
            split_free(ar, g, version);
        }
    
        
        
        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        inline void save( Archive& ar,
                          const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M,
                          const unsigned int /* file_version */)
        {
            typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows = M.rows();
            typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index cols = M.cols();
        
            ar << rows;
            ar << cols;
        
            ar << make_array( M.data(), M.size() );
        }
    
        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        inline void load( Archive& ar,
                          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M,
                          const unsigned int /* file_version */)
        {
            typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;
        
            ar >> rows;
            ar >> cols;
        
            //if (rows=!_Rows) throw std::exception(/*"Unexpected number of rows"*/);
            //if (cols=!_Cols) throw std::exception(/*"Unexpected number of cols"*/);
        
            ar >> make_array( M.data(), M.size() );
        }
    
        template<class Archive, typename _Scalar, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        inline void load( Archive& ar,
                          Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>& M,
                          const unsigned int /* file_version */)
        {
            typename Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;
        
            ar >> rows;
            ar >> cols;
        
            //if (cols=!_Cols) throw std::exception(/*"Unexpected number of cols"*/);
        
            M.resize(rows, Eigen::NoChange);
        
            ar >> make_array( M.data(), M.size() );
        }
    
        template<class Archive, typename _Scalar, int _Rows, int _Options, int _MaxRows, int _MaxCols>
        inline void load( Archive& ar,
                          Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& M,
                          const unsigned int /* file_version */)
        {
            typename Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>::Index rows, cols;
        
            ar >> rows;
            ar >> cols;
        
            //if (rows=!_Rows) throw std::exception(/*"Unexpected number of rows"*/);
        
            M.resize(Eigen::NoChange, cols);
        
            ar >> make_array( M.data(), M.size() );
        }
    
        template<class Archive, typename _Scalar, int _Options, int _MaxRows, int _MaxCols>
        inline void load( Archive& ar,
                          Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& M,
                          const unsigned int /* file_version */)
        {
            typename Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>::Index rows, cols;
        
            ar >> rows;
            ar >> cols;
        
            M.resize(rows, cols);
        
            ar >> make_array( M.data(), M.size() );
        }
    
        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        inline void serialize(Archive& ar,
                              Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M,
                              const unsigned int file_version)
        {
            split_free(ar, M, file_version);
        }
    
        template<class Archive, typename _Scalar, int _Dim, int _Mode, int _Options>
        inline void serialize(Archive& ar,
                              Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& t,
                              const unsigned int version)
        {
            serialize(ar, t.matrix(), version);
        }
    
        template <class Archive, typename _Scalar>
        void save(Archive& ar,
                  const Eigen::Triplet<_Scalar>& m,
                  const unsigned int /*version*/)
        {
            ar << m.row();
            ar << m.col();
            ar << m.value();
        }
    
        template <class Archive, typename _Scalar>
        void load(Archive& ar,
                  Eigen::Triplet<_Scalar>& m,
                  const unsigned int /*version*/)
        {
            typename Eigen::Triplet<_Scalar>::Index row, col;
            _Scalar value;
        
            ar >> row;
            ar >> col;
            ar >> value;
        
            m = Eigen::Triplet<_Scalar>(row, col, value);
        }
    
        template <class Archive, class _Scalar>
        void serialize(Archive& ar,
                       Eigen::Triplet<_Scalar>& m,
                       const unsigned int version)
        {
            split_free(ar, m, version);
        }
    
        template <class Archive, typename _Scalar, int _Options, typename _Index>
        void save(Archive& ar,
                  const Eigen::SparseMatrix<_Scalar, _Options, _Index>& m,
                  const unsigned int /*version*/)
        {
            _Index innerSize = m.innerSize();
            _Index outerSize = m.outerSize();
        
            typedef typename Eigen::Triplet<_Scalar> Triplet;
            std::vector<Triplet> triplets;
        
            for (_Index i=0; i < outerSize; ++i)
                for (typename Eigen::SparseMatrix<_Scalar, _Options, _Index>::InnerIterator it(m,i); it; ++it)
                    triplets.push_back( Triplet(it.row(), it.col(), it.value()) );
        
            ar << innerSize;
            ar << outerSize;
            ar << triplets;
        }
    
        template <class Archive, typename _Scalar, int _Options, typename _Index>
        void load(Archive& ar,
                  Eigen::SparseMatrix<_Scalar, _Options, _Index>& m,
                  const unsigned int /*version*/)
        {
            _Index innerSize;
            _Index outerSize;
        
            ar >> innerSize;
            ar >> outerSize;
        
            _Index rows = (m.IsRowMajor)? outerSize : innerSize;
            _Index cols = (m.IsRowMajor)? innerSize : outerSize;
        
            m.resize(rows, cols);
        
            typedef typename Eigen::Triplet<_Scalar> Triplet;
            std::vector<Triplet> triplets;
        
            ar >> triplets;
        
            m.setFromTriplets(triplets.begin(), triplets.end());
        }
    
        template <class Archive, typename _Scalar, int _Options, typename _Index>
        void serialize(Archive& ar, Eigen::SparseMatrix<_Scalar,_Options,_Index>& m, const unsigned int version)
        {
            split_free(ar, m, version);
        }
    
        template<class Archive, typename _Scalar>
        void serialize(Archive & ar, Eigen::Quaternion<_Scalar>& q, const unsigned int /*version*/)
        {
            ar & q.w();
            ar & q.x();
            ar & q.y();
            ar & q.z();
        }
    
    
        
        template<class Archive>
        void serialize(Archive & ar, cv::Mat& mat, const unsigned int version)
        {
            split_free(ar, mat, version);
        }
    
        /** Serialization support for cv::Mat */
        template<class Archive>
        void save(Archive &ar, const cv::Mat& m, const unsigned int /*version*/)
        {
            size_t elem_size = m.elemSize(); //CV_ELEM_SIZE(cvmat->type)
            size_t elem_type = m.type();
        
            ar << m.cols;
            ar << m.rows;
            ar << elem_size;
            ar << elem_type;
        
            const size_t data_size = m.cols * m.rows * elem_size;
            ar << make_array(m.ptr(), data_size);
        }
    
        /** Serialization support for cv::Mat */
        template<class Archive>
        void load(Archive &ar, cv::Mat& m, const unsigned int /*version*/)
        {
            int cols, rows;
            size_t elem_size, elem_type;
        
//            std::cout << "load cv mat" << std::endl;
        
            ar >> cols;
            ar >> rows;
            ar >> elem_size;
            ar >> elem_type;
        
            m.create(rows, cols, elem_type);
        
            size_t data_size = m.cols * m.rows * elem_size;
            ar >> make_array(m.ptr(), data_size);
        }
    
        template<class Archive, class T, int n, int m>
        void serialize(Archive & ar, cv::Matx<T, n, m>& mat, const unsigned int version)
        {
            split_free(ar, mat, version);
        }
    
        template<class Archive, class T, int n, int m>
        void save(Archive &ar, const cv::Matx<T, n, m>& mat,
                  const unsigned int /*version*/)
        {
            int cols = mat.cols;
            int rows = mat.rows;
        
            ar << cols;
            ar << rows;
        
            ar << make_array(&mat.val[0], cols * rows);
        }
    
        template<class Archive, class T, int n, int m>
        void load(Archive &ar, cv::Matx<T, n, m>& mat, const unsigned int /*version*/)
        {
            int cols, rows;
        
            ar >> cols;
            ar >> rows;
        
            ar >> make_array(&mat.val[0], cols * rows);
        }
    
        template<class Archive, class T, int S>
        void serialize(Archive &ar, cv::Vec<T, S>& vec, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<cv::Matx<T, S, 1> >(vec);
        }
    
        template<class Archive, class T>
        void serialize(Archive &ar, cv::Point_<T>& pt, const unsigned int /*version*/)
        {
            ar & pt.x;
            ar & pt.y;
        }
    
        template<class Archive, class T>
        void serialize(Archive &ar, cv::Point3_<T>& pt, const unsigned int /*version*/)
        {
            ar & pt.x;
            ar & pt.y;
            ar & pt.z;
        }
    
        template<class Archive>
        void serialize(Archive &ar, cv::KeyPoint& kpt, const unsigned int /*version*/)
        {
            ar & kpt.angle;
            ar & kpt.class_id;
            ar & kpt.octave;
            ar & kpt.pt;
            ar & kpt.response;
            ar & kpt.size;
        }
        
        
        
        template<class Archive>
        void serialize(Archive & ar, pcl::PCLHeader & g, const unsigned int version)
        {
            ar & g.seq;
            ar & g.stamp;
            ar & g.frame_id;
        }
        
        
        template<class Archive>
        void serialize(Archive & ar, pcl::PointCloud<pcl::Normal> & g, const unsigned int version)
        {
            ar & g.header;
            ar & g.points;
            ar & g.height;
            ar & g.width;
            ar & g.is_dense;
        }
        
        template<class Archive>
        void serialize(Archive & ar, pcl::PointCloud<pcl::PointXYZRGB> & g, const unsigned int version)
        {
            ar & g.header;
            ar & g.points;
            ar & g.height;
            ar & g.width;
            ar & g.is_dense;
        }
        
        template<class Archive>
        void serialize(Archive & ar, pcl::PointXYZRGB& g, const unsigned int version) {
            ar & g.x;
            ar & g.y;
            ar & g.z;
            ar & g.r;
            ar & g.g;
            ar & g.b;
        }
    
        template<class Archive>
        void serialize(Archive & ar, pcl::Normal& g, const unsigned int version) {
            ar & g.normal_x;
            ar & g.normal_y;
            ar & g.normal_z;
        }
        
    } // namespace serialization
} // namespace boost

#endif //PLANELOC_SERIALIZATION_HPP
