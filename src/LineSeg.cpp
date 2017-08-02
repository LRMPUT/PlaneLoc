//
// Created by jachu on 27.06.17.
//

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


