//
// Created by jachu on 05.02.18.
//

#include <iostream>
#include <EKFPlane.hpp>

#include "Types.hpp"
#include "Misc.hpp"
#include "EKFPlane.hpp"

using namespace std;


EKFPlane::EKFPlane() {}

EKFPlane::EKFPlane(const Eigen::Quaterniond &xq, const Eigen::Matrix4d &Pq) {
    init(xq, Pq);
}

EKFPlane::EKFPlane(const Eigen::Quaterniond &xq, int npts) {
    init(xq, npts);
}



void EKFPlane::init(const Eigen::Quaterniond &xq, const Eigen::Matrix4d &Pq) {
    x = xq;
    Eigen::MatrixXd J = jacob_dom_dq(xq);
    P = J * Pq * J.transpose();
}

void EKFPlane::init(const Eigen::Quaterniond &xq, int npts) {
    x = xq;
    EKFPlane::npts = npts;
}

void EKFPlane::update(const Eigen::Quaterniond &zq, const Eigen::Matrix4d &Rq) {
    // jacobian of transformation from quaternion to log-map of quaternion
    Eigen::MatrixXd J_dom_dq = jacob_dom_dq(zq);
    // covariance in log-map representation
    Eigen::Matrix3d R = J_dom_dq * Rq * J_dom_dq.transpose();
    
    update(zq, R);
}

void EKFPlane::update(const Eigen::Quaterniond &zq, const Eigen::Matrix3d &R) {
//    cout << endl << "x = " << x.coeffs().transpose() << endl;
//    cout << "P = " << P << endl;
//    cout << "zq = " << zq.coeffs().transpose() << endl;
//    cout << "R = " << R << endl;
    // innovation
    Eigen::Vector3d v = Misc::logMap(zq * x.inverse());
//    cout << "v = " << v.transpose() << endl;
    // innovation covariance
    Eigen::Matrix3d S = P + R;
//    cout << "S = " << S << endl;
    // Kalman gain
    Eigen::Matrix3d K = P * S.inverse();
//    {
//        Eigen::EigenSolver<Eigen::Matrix3d> evd(R);
//
//        Eigen::Matrix3d evecs;
//        Eigen::Vector3d evals;
//        for(int i = 0; i < 3; ++i){
//            evecs.col(i) = evd.eigenvectors().col(2 - i).real();
//            evals(i) = evd.eigenvalues()(2 - i).real();
//        }
////        if(evals(0) > 1.0){
//            cout << endl << "evecs = " << evecs << endl;
//            cout << "evals = " << evals.transpose() << endl;
//
//            cout  << "x = " << x.coeffs().transpose() << endl;
//            cout  << "log(x) = " << Misc::logMap(x).transpose() << endl;
//            cout << "zq = " << zq.coeffs().transpose() << endl;
//            cout  << "log(zq) = " << Misc::logMap(zq).transpose() << endl;
//            cout << "v = " << v.transpose() << endl;
//            cout << "updated x = " << (Misc::expMap(K * v) * x).coeffs().transpose() << endl;
//            cout  << "log(updated x) = " << Misc::logMap(Misc::expMap(K * v) * x).transpose() << endl;
//            cout << "P = " << P << endl;
//            cout << "R = " << R << endl;
//            cout << "S = " << S << endl;
//            cout << "S.inverse() = " << S.inverse() << endl;
//            cout << "K = " << K << endl;
//            cout << "K * v = " << (K * v).transpose() << endl;
//
////            char a;
////            cin >> a;
////        }
//    }
//    cout << "K = " << K << endl;
    // update of state
//    cout << "K * v = " << K * v << endl;
    x = Misc::expMap(K * v) * x;
//    cout << "updated x = " << x.coeffs().transpose() << endl;
    // update of covariance
    P = (Eigen::Matrix3d::Identity() - K) * P;
//    cout << "updated P = " << P << endl;
}

//void EKFPlane::transform(const Eigen::Matrix4d T)
//{
//    Eigen::Matrix4d Tinv = T.inverse();
//    Eigen::Matrix4d Tinvt = Tinv.transpose();
//
//    Eigen::Vector4d planeEq = Tinvt * x.coeffs();
//    Eigen::Matrix4d covarQuat = Tinvt * covarQuat * Tinv;
//}

void EKFPlane::update(const Eigen::Quaterniond &zq, int znpts) {
    Eigen::Vector3d meanLogMap;
    meanLogMap << 0.0, 0.0, 0.0;
    int sumPoints = 0;
    {
        Eigen::Vector3d z = Misc::logMap(zq);
        meanLogMap += z * znpts;
        sumPoints += znpts;
    }
    {
        Eigen::Vector3d xu = Misc::logMap(x);
        meanLogMap += xu * npts;
        sumPoints += npts;
    }
    meanLogMap /= sumPoints;
    
    x = Misc::expMap(meanLogMap);
    npts = sumPoints;
}

double EKFPlane::distance(const Eigen::Quaterniond &xcq) const {
//    Eigen::Matrix3d inf = P.inverse();
//    cout << "P = " << P << endl;
//    cout << "inf = " << inf << endl;
    Eigen::Vector3d e = Misc::logMap(xcq * x.inverse());
    
//    return e.transpose() * inf * e;
    return e.transpose() * e;
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
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for(int i = 0; i < pts.cols(); ++i){
        mean += pts.col(i);
    }
    mean /= pts.cols();
    
    Eigen::MatrixXd demeanPts = pts;
    for(int i = 0; i < demeanPts.cols(); ++i){
        demeanPts.col(i) -= mean;
    }
    
    Eigen::Matrix3d covar = demeanPts * demeanPts.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> evd(covar);
    
    Eigen::Matrix3d evecs;
    Eigen::Vector3d evals;
    for(int i = 0; i < 3; ++i){
        evecs.col(i) = evd.eigenvectors().col(2 - i);
        evals(i) = evd.eigenvalues()(2 - i);
    }
    
    // the smallest eigenvalue corresponds to the eigenvector that is normal to the plane
    double varD = evals(2) / demeanPts.cols();
    double varX = evals(2) / evals(0);
    double varY = evals(2) / evals(1);
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = evecs;
    T.block<3, 1>(0, 3) = mean;
    Eigen::Matrix4d Tinv = T.inverse();
    Eigen::Matrix4d Tinvt = Tinv.transpose();
    
    // plane with normal (0, 0, 1) and distance 0;
    Eigen::Vector4d planeEq;
    planeEq << 0.0, 0.0, 1.0, 0.0;
    // transform it to destination pose
    planeEq = Tinvt * planeEq;
    
    Eigen::Matrix4d covarQuat = Eigen::Matrix4d::Zero();
    covarQuat(0, 0) = varX;
    covarQuat(1, 1) = varY;
    covarQuat(2, 2) = 0;
    covarQuat(3, 3) = varD;
    covarQuat = Tinvt * covarQuat * Tinv;
    
//    Eigen::Matrix4d J_dqn_dq = jacob_dqn_dq(Eigen::Quaterniond(planeEq(3), planeEq(0), planeEq(1), planeEq(2)));
//    planeEq.normalize();
//    R = J_dqn_dq * covarQuat * J_dqn_dq.transpose();
    
    double planeEqNorm = planeEq.norm();
    planeEq /= planeEqNorm;
    R = covarQuat / (planeEqNorm * planeEqNorm);
    
    q.coeffs() = planeEq;
}

//[ (2*acos(qw)*(qy^2 + qz^2))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qx*qy*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qx*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), -(2*qx)/((1 - qw^2)^(1/2)*(qx^2 + qy^2 + qz^2)^(1/2))]
//[        -(2*qx*qy*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), (2*acos(qw)*(qx^2 + qz^2))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qy*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), -(2*qy)/((1 - qw^2)^(1/2)*(qx^2 + qy^2 + qz^2)^(1/2))]
//[        -(2*qx*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2),        -(2*qy*qz*acos(qw))/(qx^2 + qy^2 + qz^2)^(3/2), (2*acos(qw)*(qx^2 + qy^2))/(qx^2 + qy^2 + qz^2)^(3/2), -(2*qz)/((1 - qw^2)^(1/2)*(qx^2 + qy^2 + qz^2)^(1/2))]

Eigen::MatrixXd EKFPlane::jacob_dom_dq(const Eigen::Quaterniond &q) {
    Eigen::MatrixXd J(3, 4);
    double qx = q.x();
    double qy = q.y();
    double qz = q.z();
    double qw = q.w();
    double sqVecNorm = qx*qx + qy*qy + qz*qz;
    double vecNorm = sqrt(sqVecNorm);
    double den = sqVecNorm * vecNorm;
    J << (2*acos(qw)*(qy*qy + qz*qz))/den,          -(2*qx*qy*acos(qw))/den,          -(2*qx*qz*acos(qw))/den, -(2*qx)/(sqrt(1 - qw*qw)*vecNorm),
        -(2*qx*qy*acos(qw))/den,           (2*acos(qw)*(qx*qx + qz*qz))/den,          -(2*qy*qz*acos(qw))/den, -(2*qy)/(sqrt(1 - qw*qw)*vecNorm),
        -(2*qx*qz*acos(qw))/den,                    -(2*qy*qz*acos(qw))/den, (2*acos(qw)*(qx*qx + qy*qy))/den, -(2*qz)/(sqrt(1 - qw*qw)*vecNorm);
    
    return J;
}


//[ (qw^2 + qy^2 + qz^2)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qx*qy)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qx*qz)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qw*qx)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2)]
//[             -(qx*qy)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2), (qw^2 + qx^2 + qz^2)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qy*qz)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qw*qy)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2)]
//[             -(qx*qz)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qy*qz)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2), (qw^2 + qx^2 + qy^2)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qw*qz)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2)]
//[             -(qw*qx)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qw*qy)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2),             -(qw*qz)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2), (qx^2 + qy^2 + qz^2)/(qw^2 + qx^2 + qy^2 + qz^2)^(3/2)]
Eigen::MatrixXd EKFPlane::jacob_dqn_dq(const Eigen::Quaterniond &q) {
    Eigen::MatrixXd J(4, 4);
    double qx = q.x();
    double qy = q.y();
    double qz = q.z();
    double qw = q.w();
    double sqVecNorm = qx*qx + qy*qy + qz*qz + qw*qw;
    double vecNorm = sqrt(sqVecNorm);
    double den = sqVecNorm * vecNorm;
    
    J << (qw*qw + qy*qy + qz*qz)/den,                -(qx*qy)/den,                -(qx*qz)/den,                -(qw*qx)/den,
                        -(qx*qy)/den, (qw*qw + qx*qx + qz*qz)/den,                -(qy*qz)/den,                -(qw*qy)/den,
                        -(qx*qz)/den,                -(qy*qz)/den, (qw*qw + qx*qx + qy*qy)/den,                -(qw*qz)/den,
                        -(qw*qx)/den,                -(qw*qy)/den,                -(qw*qz)/den, (qx*qx + qy*qy + qz*qz)/den;
    
    return J;
}

const Eigen::Quaterniond &EKFPlane::getX() const {
    return x;
}

const Eigen::Matrix3d &EKFPlane::getP() const {
    return P;
}




