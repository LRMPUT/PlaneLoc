//
// Created by jachu on 05.02.18.
//

#include <iostream>
#include <EKFPlane.hpp>

#include "Types.hpp"
#include "Misc.hpp"
#include "EKFPlane.hpp"

using namespace std;

EKFPlane::EKFPlane(const Eigen::Quaterniond &xq, const Eigen::Matrix4d &Pq) {
    x = Misc::logMap(xq);
    Eigen::MatrixXd J = jacob_dom_dq(xq);
    P = J * Pq * J.transpose();
}

void EKFPlane::update(const Eigen::Quaterniond &zq, const Eigen::Matrix4d &Rq) {

}

double EKFPlane::distance(const Eigen::Quaterniond &xcq) const {
    return 0;
}

void EKFPlane::compPlaneEqAndCovar(const Eigen::MatrixXd &pts,
                                   Eigen::Quaterniond &q,
                                   Eigen::Matrix4d &R)
{
//    // Compute mean
//    mean_ = Eigen::Vector4f::Zero ();
//    compute3DCentroid (*input_, *indices_, mean_);
//    // Compute demeanished cloud
//    Eigen::MatrixXf cloud_demean;
//    demeanPointCloud (*input_, *indices_, mean_, cloud_demean);
//    assert (cloud_demean.cols () == int (indices_->size ()));
//    // Compute the product cloud_demean * cloud_demean^T
//    Eigen::Matrix3f alpha = static_cast<Eigen::Matrix3f> (cloud_demean.topRows<3> () * cloud_demean.topRows<3> ().transpose ());
//
//    // Compute eigen vectors and values
//    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> evd (alpha);
//    // Organize eigenvectors and eigenvalues in ascendent order
//    for (int i = 0; i < 3; ++i)
//    {
//        eigenvalues_[i] = evd.eigenvalues () [2-i];
//        eigenvectors_.col (i) = evd.eigenvectors ().col (2-i);
//    }
}

//[ (2*acos(qw)*(qy^2 + qz^2))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qx*qy*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qx*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), -(2*qx)/((1 - qw^2)^(1/2)*(qx^2 + qy^2 + qz^2)^(1/2))]
//[        -(2*qx*qy*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), (2*acos(qw)*(qx^2 + qz^2))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qy*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), -(2*qy)/((1 - qw^2)^(1/2)*(qx^2 + qy^2 + qz^2)^(1/2))]
//[        -(2*qx*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qy*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), (2*acos(qw)*(qx^2 + qy^2))/(qx^2 + qy^2 + qz^2)^(3/2), -(2*qz)/((1 - qw^2)^(1/2)*(qx^2 + qy^2 + qz^2)^(1/2))]

Eigen::MatrixXd EKFPlane::jacob_dom_dq(const Eigen::Quaterniond &q) const {
    return Eigen::Matrix<double, Dynamic, Dynamic>();
}
