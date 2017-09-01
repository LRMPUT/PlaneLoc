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

#ifndef INCLUDE_MISC_HPP_
#define INCLUDE_MISC_HPP_

#include <ostream>
#include <vector>

#include <Eigen/Eigen>

#include <pcl/common/common_headers.h>
#include <pcl/impl/point_types.hpp>

#include <opencv2/opencv.hpp>

#include "Types.hpp"

static constexpr float pi = 3.14159265359;

template<class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec){
	out << "[";
	for(int v = 0; v < (int)vec.size(); ++v){
		out << vec[v];
		if(v < vec.size() - 1){
			out << ", ";
		}
	}
	out << "]";

	return out;
}

class Misc{
public:

	static cv::Mat projectTo3D(cv::Mat depth, cv::Mat cameraParams);

    static cv::Mat reprojectTo2D(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr points, cv::Mat cameraParams);

    static Eigen::Vector3d projectPointOnPlane(const Eigen::Vector3d &pt, const Eigen::Vector4d &plane);

    static Eigen::Vector3d projectPointOnPlane(const Eigen::Vector2d &pt, const Eigen::Vector4d &plane, cv::Mat cameraMatrix);

	static bool nextChoice(std::vector<int>& choice, int N);

	static Eigen::Quaterniond planeEqToQuat(Eigen::Vector4d planeEq);

	static void normalizeAndUnify(Eigen::Quaterniond& q);

	static void normalizeAndUnify(Eigen::Vector4d& q);

    static Eigen::Vector4d toNormalPlaneEquation(Eigen::Vector4d plane);

	static Eigen::Vector3d logMap(Eigen::Quaterniond quat);

	static Eigen::Quaterniond expMap(Eigen::Vector3d vec);

	static Eigen::Matrix4d matrixQ(Eigen::Quaterniond q);

	static Eigen::Matrix4d matrixW(Eigen::Quaterniond q);

	static Eigen::Matrix3d matrixK(Eigen::Quaterniond q);

	static bool checkIfAlignedWithNormals(const Eigen::Vector3f& testedNormal,
                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                            bool& alignConsistent);

	static double transformLogDist(Vector7d trans1,
									Vector7d trans2);

    static double rotLogDist(Eigen::Vector4d rot1,
                             Eigen::Vector4d rot2);

    static cv::Mat colorIds(cv::Mat ids);
	
	static Eigen::Vector3d closestPointOnLine(const Eigen::Vector3d &pt,
									   const Eigen::Vector3d &p,
									   const Eigen::Vector3d &n);
    
    template<typename _Matrix_Type_>
    static _Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
    {
        Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
//        return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
        
        typename Eigen::JacobiSVD< _Matrix_Type_ >::SingularValuesType singularValues_inv = svd.singularValues();
        for ( long i = 0; i < singularValues_inv.cols(); ++i) {
            if ( fabs(svd.singularValues()(i)) > tolerance ) {
                singularValues_inv(i) = 1.0 / svd.singularValues()(i);
            }
            else{
                singularValues_inv(i)=0;
            }
        }
        return (svd.matrixV() * singularValues_inv.asDiagonal());
    }
};

static constexpr uint8_t colors[][3] = {
		{0xFF, 0x00, 0x00}, //Red
		{0xFF, 0xFF, 0xFF}, //White
		{0x00, 0xFF, 0xFF}, //Cyan
		{0xC0, 0xC0, 0xC0}, //Silver
		{0x00, 0x00, 0xFF}, //Blue
		{0x80, 0x80, 0x80}, //Gray
		{0x00, 0x00, 0xA0}, //DarkBlue
		{0x00, 0x00, 0x00}, //Black
		{0xAD, 0xD8, 0xE6}, //LightBlue
		{0xFF, 0xA5, 0x00}, //Orange
		{0x80, 0x00, 0x80}, //Purple
		{0xA5, 0x2A, 0x2A}, //Brown
		{0xFF, 0xFF, 0x00}, //Yellow
		{0x80, 0x00, 0x00}, //Maroon
		{0x00, 0xFF, 0x00}, //Lime
		{0x00, 0x80, 0x00}, //Green
		{0xFF, 0x00, 0xFF}, //Magenta
		{0x80, 0x80, 0x00} //Olive
};

class Visualizer{
public:

//	static pcl::PointCloud<pcl::PointXYZRGBA>::Ptr makeColorPointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr
};

#endif /* INCLUDE_MISC_HPP_ */
