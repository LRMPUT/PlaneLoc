//
// Created by jachu on 05.02.18.
//

#ifndef PLANELOC_EKFPLANE_HPP
#define PLANELOC_EKFPLANE_HPP

#include <Eigen/Eigen>


class EKFPlane {
public:
    EKFPlane(const Eigen::Quaterniond &xq,
             const Eigen::Matrix4d &Pq);
    
    void update(const Eigen::Quaterniond &zq,
                const Eigen::Matrix4d &Rq);
    
    double distance(const Eigen::Quaterniond &xcq) const;
    
    static void compPlaneEqAndCovar(const Eigen::MatrixXd &pts,
                                    Eigen::Quaterniond &q,
                                    Eigen::Matrix4d &R);
    
private:
    
    Eigen::MatrixXd jacob_dom_dq(const Eigen::Quaterniond &q) const;
    
    Eigen::Vector3d x;
    
    Eigen::Matrix3d P;
};


#endif //PLANELOC_EKFPLANE_HPP
