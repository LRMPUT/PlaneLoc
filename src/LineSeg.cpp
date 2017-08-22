//
// Created by jachu on 27.06.17.
//

#include <g2o/types/slam3d/se3quat.h>
#include "Misc.hpp"
#include "LineSeg.hpp"

LineSeg::LineSeg(int iframeId,
                 Eigen::Vector2d ipi1,
                 Eigen::Vector2d ipi2,
                 Eigen::Vector3d ip1,
                 Eigen::Vector3d ip2)
    :   p1(ip1),
        p2(ip2)
{
    frameIds.push_back(iframeId);
    pis1.push_back(ipi1);
    pis2.push_back(ipi2);
}

const std::vector<int> &LineSeg::getFrameIds() const {
    return frameIds;
}

const std::vector<Eigen::Vector2d> &LineSeg::getPis1() const {
    return pis1;
}

const std::vector<Eigen::Vector2d> &LineSeg::getPis2() const {
    return pis2;
}

const Eigen::Vector3d &LineSeg::getP1() const {
    return p1;
}

const Eigen::Vector3d &LineSeg::getP2() const {
    return p2;
}

void LineSeg::setP1(const Eigen::Vector3d &p1) {
    LineSeg::p1 = p1;
}

void LineSeg::setP2(const Eigen::Vector3d &p2) {
    LineSeg::p2 = p2;
}

Vector6d LineSeg::toPointNormalEq() const {
    Vector6d ret;
    Eigen::Vector3d n = (p2 - p1).normalized();
    Eigen::Vector3d p = Misc::closestPointOnLine(Eigen::Vector3d::Zero(), p1, n);
    ret.head<3>() = p;
    ret.tail<3>() = n;
    
    return ret;
}

Vector7d LineSeg::toSE3Point() const {
    Vector7d ret;
    
    ret.head<3>() = p1;
    
    Eigen::Matrix3d rot;
    Eigen::Vector3d n = (p2 - p1).normalized();
    // compute two directions parallel to n
    Eigen::FullPivLU<Eigen::MatrixXd> lu(n.transpose());
    Eigen::MatrixXd nullSpace = lu.kernel();
    Eigen::Vector3d xAxis = nullSpace.block<3, 1>(0, 0).normalized();
    Eigen::Vector3d yAxis = nullSpace.block<3, 1>(0, 1).normalized();
    rot.block<3, 1>(0, 0) = xAxis;
    rot.block<3, 1>(0, 1) = yAxis;
    rot.block<3, 1>(0, 2) = n;
    
    return ret;
}

LineSeg LineSeg::transformed(const Vector7d &transform) const {
    LineSeg ret(frameIds.front(),
                pis1.front(),
                pis2.front(),
                p1,
                p2);
    
    g2o::SE3Quat transformSE3Quat(transform);
    Eigen::Matrix3d rotMat = transformSE3Quat.rotation().toRotationMatrix();
    Eigen::Vector3d trans = transformSE3Quat.translation();
    
    Eigen::Vector3d p1trans = rotMat * p1 + trans;
    Eigen::Vector3d p2trans = rotMat * p2 + trans;
    
    ret.setP1(p1);
    ret.setP2(p2);
    
    return ret;
}

double LineSeg::eqDist(const LineSeg &ls) {
    double dist = 0.0;
    
    Vector7d curSE3 = toSE3Point();
    Vector7d lsSE3 = ls.toSE3Point();
    
    
    
    return dist;
}

