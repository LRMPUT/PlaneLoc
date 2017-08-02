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

#include "Misc.hpp"

#include <g2o/types/slam3d/se3quat.h>

using namespace std;
using namespace cv;

cv::Mat Misc::projectTo3D(cv::Mat depth, cv::Mat cameraParams){
	float fx = cameraParams.at<float>(0, 0);
	float fy = cameraParams.at<float>(1, 1);
	float cx = cameraParams.at<float>(0, 2);
	float cy = cameraParams.at<float>(1, 2);
//	cout << "cx = " << cx << ", cy = " << cy << ", fx = " << fx << ", fy = " << fy << endl;
	Mat xyz(depth.rows, depth.cols, CV_32FC3);
	for(int row = 0; row < depth.rows; ++row){
		for(int col = 0; col < depth.cols; ++col){
			float d = depth.at<float>(row, col);

			// x
			xyz.at<Vec3f>(row, col)[0] = (col - cx) * d / fx;
			// y
			xyz.at<Vec3f>(row, col)[1] = (row - cy) * d / fy;
			// z
			xyz.at<Vec3f>(row, col)[2] = d;

//			cout << "d = " << d <<
//					", x = " << xyz.at<Vec3f>(row, col)[0] <<
//					", y = " << xyz.at<Vec3f>(row, col)[1] <<
//					", z = " << xyz.at<Vec3f>(row, col)[2] << endl;
		}
	}
	return xyz;
}

cv::Mat Misc::reprojectTo2D(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr points, cv::Mat cameraParams)
{
    float fx = cameraParams.at<float>(0, 0);
    float fy = cameraParams.at<float>(1, 1);
    float cx = cameraParams.at<float>(0, 2);
    float cy = cameraParams.at<float>(1, 2);
    Mat uvd(1, points->size(), CV_32FC3);
    for(int p = 0; p < points->size(); ++p){
        double d = points->at(p).z;

        uvd.at<Vec3f>(0, p)[0] = points->at(p).x * fx / d + cx;
        uvd.at<Vec3f>(0, p)[1] = points->at(p).y * fy / d + cy;
        uvd.at<Vec3f>(0, p)[2] = d;
    }
    return uvd;
}

Eigen::Vector3d Misc::projectPointOnPlane(const Eigen::Vector3d &pt, const Eigen::Vector4d &plane)
{
    Eigen::Vector4d planeNorm = toNormalPlaneEquation(plane);
    Eigen::Vector3d n = planeNorm.head<3>();
    double d = planeNorm[3];
    Eigen::Vector3d plPt = n * d;
    double ndist = n.dot(pt - plPt);
    return pt - ndist * n;
}

Eigen::Vector3d Misc::projectPointOnPlane(const Eigen::Vector2d &pt, const Eigen::Vector4d &plane, cv::Mat cameraMatrix)
{
    float fx = cameraMatrix.at<float>(0, 0);
    float fy = cameraMatrix.at<float>(1, 1);
    float cx = cameraMatrix.at<float>(0, 2);
    float cy = cameraMatrix.at<float>(1, 2);
    Eigen::Vector4d planeNorm = toNormalPlaneEquation(plane);
    Eigen::Vector3d n = planeNorm.head<3>();
    double dpl = planeNorm[3];
    Eigen::Vector3d pnnorm;
    pnnorm[0] = (pt[0] - cx) / fx;
    pnnorm[0] = (pt[1] - cy) / fy;
    pnnorm[0] = 1;
    double ncast = n.dot(pnnorm);
    if(ncast > 1e-6) {
        double d = dpl / ncast;
        return pnnorm * d;
    }
    else{
        return Eigen::Vector3d();
    }
}

bool Misc::nextChoice(std::vector<int>& choice, int N)
{
	int chidx = choice.size() - 1;
	int chendidx = 0;
	bool valid = false;
	while(chidx >= 0 && !valid){
		++choice[chidx];
		// if in valid range, we found correct choice
		if(choice[chidx] < N - chendidx){
			valid = true;
		}
		--chidx;
		++chendidx;
	}
	// moving all choice values after last value incremented
	if(valid){
		// move chidx back to last value incremented
		++chidx;
		while(chidx < choice.size() - 1){
			choice[chidx + 1] = choice[chidx] + 1;
			++chidx;
		}
	}
	return valid;
}

Eigen::Quaterniond Misc::planeEqToQuat(Eigen::Vector4d planeEq)
{
	Eigen::Quaterniond ret(planeEq(3), planeEq(0), planeEq(1), planeEq(2));
	normalizeAndUnify(ret);
	return ret;
}

void Misc::normalizeAndUnify(Eigen::Quaterniond& q){
	q.normalize();
	static constexpr double eps = 1e-9;
	if(q.w() < 0.0 ||
		(fabs(q.w()) < eps && q.z() < 0.0) ||
		(fabs(q.w()) < eps && fabs(q.z()) < eps && q.y() < 0.0) ||
		(fabs(q.w()) < eps && fabs(q.z()) < eps && fabs(q.y()) < eps && q.x() < 0.0))
	{
		q.coeffs() = -q.coeffs();
	}
}

void Misc::normalizeAndUnify(Eigen::Vector4d& v){
	v.normalize();
	static constexpr double eps = 1e-9;
	if(v(3) < 0.0 ||
		(fabs(v(3)) < eps && v(2) < 0.0) ||
		(fabs(v(3)) < eps && fabs(v(2)) < eps && v(1) < 0.0) ||
		(fabs(v(3)) < eps && fabs(v(2)) < eps && fabs(v(1)) < eps && v(0) < 0.0))
	{
		v = -v;
	}
}

Eigen::Vector4d Misc::toNormalPlaneEquation(Eigen::Vector4d plane)
{
    double nnorm = plane.head<3>().norm();
    return plane / nnorm;
}

Eigen::Vector3d Misc::logMap(Eigen::Quaterniond quat)
{
	Eigen::Vector3d res;

    normalizeAndUnify(quat);
	double qvNorm = sqrt(quat.x()*quat.x() + quat.y()*quat.y() + quat.z()*quat.z());
	if(qvNorm > 1e-6){
		res[0] = quat.x()/qvNorm;
		res[1] = quat.y()/qvNorm;
		res[2] = quat.z()/qvNorm;
	}
	else{
		// 1/sqrt(3), so norm = 1
		res[0] = 0.57735026919;
		res[1] = 0.57735026919;
		res[2] = 0.57735026919;;
	}
	double acosQw = acos(quat.w());
	res *= 2.0*acosQw;
//		cout << "2.0*acosQw = " << 2.0*acosQw << endl;
	return res;
}

Eigen::Quaterniond Misc::expMap(Eigen::Vector3d vec)
{
    double arg = 0.5 * std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
//    if(fabs(arg) > 0.5 * pi){
//        std::cout << "arg = " << arg << std::endl;
//        char a;
//        std::cin >> a;
//    }
    double sincArg = 1.0;
    if(arg > 1e-6){
        sincArg = sin(arg)/arg;
    }
    else{
        //taylor expansion
        sincArg = 1 - arg*arg/6 + pow(arg, 4)/120;
    }
    double cosArg = cos(arg);

    Eigen::Quaterniond res(cosArg,
                         0.5*sincArg*vec[0],
                         0.5*sincArg*vec[1],
                         0.5*sincArg*vec[2]);

    normalizeAndUnify(res);

    return res;
}

Eigen::Matrix4d Misc::matrixQ(Eigen::Quaterniond q)
{
	Eigen::Matrix4d ret;
	ret.block<3, 3>(0, 0) = matrixK(q) + Eigen::Matrix3d::Identity() * q.w();
	ret.block<3, 1>(0, 3) = q.vec();
	ret.block<1, 3>(3, 0) = -q.vec();
	ret.block<1, 1>(3, 3) = Eigen::Matrix<double, 1, 1>::Ones() * q.w();
	return ret;
}

Eigen::Matrix4d Misc::matrixW(Eigen::Quaterniond q)
{
	Eigen::Matrix4d ret;
	ret.block<3, 3>(0, 0) = -matrixK(q) + Eigen::Matrix3d::Identity() * q.w();
	ret.block<3, 1>(0, 3) = q.vec();
	ret.block<1, 3>(3, 0) = -q.vec();
	ret.block<1, 1>(3, 3) = Eigen::Matrix<double, 1, 1>::Ones() * q.w();
	return ret;
}

Eigen::Matrix3d Misc::matrixK(Eigen::Quaterniond q)
{
	Eigen::Matrix3d ret;
	ret <<	0,		-q.z(),	q.y(),
			q.z(),	0,		-q.x(),
			-q.y(),	q.x(),	0;
	return ret;
}

bool Misc::checkIfAlignedWithNormals(const Eigen::Vector3f& testedNormal,
									  pcl::PointCloud<pcl::Normal>::ConstPtr normals,
									  bool& alignConsistent)
{
    bool isAligned = false;
    alignConsistent = false;

    // check if aligned with majority of normals
    int alignedCnt = 0;
    for(int p = 0; p < normals->size(); ++p){
        Eigen::Vector3f curNorm = normals->at(p).getNormalVector3fMap();
        if(testedNormal.dot(curNorm) > 0){
            ++alignedCnt;
        }
    }
    double alignedFrac = (double)alignedCnt / normals->size();
//    cout << "alignedFrac = " << alignedFrac << endl;
    static constexpr double errorThresh = 0.1;
    if(alignedFrac >= 0.5){
        isAligned = true;
    }
    else if(alignedFrac < 0.5){
        alignConsistent = false;
        isAligned = false;
    }

    // some normals aligned and some not - something went wrong
    if(min(alignedFrac, 1.0 - alignedFrac) > errorThresh) {
        alignConsistent = true;
    }

    return isAligned;
}

double Misc::transformLogDist(Vector7d trans1, Vector7d trans2) {
	g2o::SE3Quat trans(trans1);
	g2o::SE3Quat transComp(trans2);
	g2o::SE3Quat diff = trans.inverse() * transComp;
	Vector6d logMapDiff = diff.log();
	double dist = logMapDiff.transpose() * logMapDiff;
	return dist;
}

double Misc::rotLogDist(Eigen::Vector4d rot1, Eigen::Vector4d rot2) {
    Eigen::Quaterniond r1(rot1[3], rot1[0], rot1[1], rot1[2]);
    Eigen::Quaterniond r2(rot2[3], rot2[0], rot2[1], rot2[2]);
    Eigen::Vector3d logDiff = logMap(r1.inverse() * r2);

    return logDiff.transpose() * logDiff;
}

cv::Mat Misc::colorIds(cv::Mat ids) {
    cv::Mat colIm(ids.size(), CV_8UC3, Scalar(0, 0, 0));
    for(int r = 0; r < ids.rows; ++r){
        for(int c = 0; c < ids.cols; ++c){
            int id = ids.at<int>(r, c);
            if(id >= 0){
                int colIdx = (id % sizeof(colors)/sizeof(uint8_t)/3);
                colIm.at<Vec3b>(r, c) = cv::Vec3b(colors[colIdx][2],
                                                  colors[colIdx][1],
                                                  colors[colIdx][0]);
            }
        }
    }
    return colIm;
}
