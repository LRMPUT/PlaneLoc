//
// Created by jachu on 27.06.17.
//

#ifndef PLANELOC_LINESEG_HPP
#define PLANELOC_LINESEG_HPP

#include <vector>

//#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "Types.hpp"
#include "Serialization.hpp"

class LineSeg {
public:
    LineSeg();
    
    LineSeg(int iframeId,
            const Eigen::Vector2d &ipi1,
            const Eigen::Vector2d &ipi2,
            const Eigen::Vector3d &ip1,
            const Eigen::Vector3d &ip2);

    const std::vector<int> &getFrameIds() const;

    const vectorVector2d &getPis1() const;

    const vectorVector2d &getPis2() const;

    const Eigen::Vector3d &getP1() const;

    const Eigen::Vector3d &getP2() const;

    void setP1(const Eigen::Vector3d &p1);

    void setP2(const Eigen::Vector3d &p2);
    
    Vector6d toPointNormalEq() const;
    
    Vector7d toSE3Point() const;
    
    LineSeg transformed(const Vector7d &transform) const;
    
    double eqDist(const LineSeg &ls);
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    std::vector<int> frameIds;

    vectorVector2d pis1, pis2;

    Eigen::Vector3d p1, p2;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & frameIds;
        ar & pis1;
        ar & pis2;
        ar & p1;
        ar & p2;
    }
};


#endif //PLANELOC_LINESEG_HPP
