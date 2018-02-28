//
// Created by jachu on 27.02.18.
//

#include <iostream>

#include <Eigen/Dense>

#include <g2o/types/slam3d/se3quat.h>

#include "PlaneEstimator.hpp"

using namespace std;

PlaneEstimator::PlaneEstimator() {

}

PlaneEstimator::PlaneEstimator(const Eigen::MatrixXd &pts) {
    init(pts);
}

PlaneEstimator::PlaneEstimator(const Eigen::Vector3d &icentroid,
                               const Eigen::Matrix3d &icovar,
                               int inpts)
{
    init(icentroid,
         icovar,
         inpts);
}

void PlaneEstimator::init(const Eigen::MatrixXd &pts) {
    Eigen::Vector3d icentroid;
    Eigen::Matrix3d icovar;
    compCentroidAndCovar(pts, icentroid, icovar);
    init(icentroid, icovar, pts.cols());
}

void
PlaneEstimator::init(const Eigen::Vector3d &icentroid, const Eigen::Matrix3d &icovar, int inpts) {
    centroid = icentroid;
    covar = icovar;
    npts = inpts;
    compPlaneParams(centroid,
                    covar,
                    evecs,
                    evals,
                    planeEq);
}

void
PlaneEstimator::update(const Eigen::Vector3d &ucentroid, const Eigen::Matrix3d &ucovar, int unpts) {
    updateCentroidAndCovar(centroid,
                           covar,
                           npts,
                           ucentroid,
                           ucovar,
                           unpts,
                           centroid,
                           covar,
                           npts);
    
    static constexpr int ptsLimit = 100000;
    if(npts > ptsLimit){
        double scale = (double)npts/ptsLimit;
        covar /= scale;
        npts = ptsLimit;
    }
    
    compPlaneParams(centroid,
                    covar,
                    evecs,
                    evals,
                    planeEq);
}

double PlaneEstimator::distance(const PlaneEstimator &other) const {
    const Eigen::Vector3d &centroid1 = centroid;
    const Eigen::Vector3d &centroid2 = other.centroid;
    Eigen::Matrix3d covar1 = covar / npts;
    Eigen::Matrix3d covar2 = other.covar / other.npts;
    int npts1 = npts;
    int npts2 = other.npts;
    
//    cout << "centroid1 = " << centroid1.transpose() << endl;
//    cout << "covar1 = " << covar1 << endl;
//    cout << "centroid2 = " << centroid2.transpose() << endl;
//    cout << "covar2 = " << covar2 << endl;
//
//    Eigen::Vector3d centrDiff = centroid1 - centroid2;
//    Eigen::Matrix3d infComb = (covar1 + covar2).inverse();
//    Eigen::Matrix3d covarProd = covar1 * infComb * covar2;
//    Eigen::Vector3d meanProd = covar2 * infComb * centroid1 +
//                               covar1 * infComb * centroid2;
//    double detVal = (2*M_PI*(covar1 + covar2)).determinant();
//    double expVal = -0.5 * centrDiff.transpose() * infComb * centrDiff;
//    double normFactor = 1.0 / sqrt(detVal) * exp(expVal);
//    cout << "detVal = " << detVal << endl;
//    cout << "expVal = " << -2.0*expVal << endl;
//    cout << "normFactor = " << normFactor << endl;
//    cout << "dist = " << 1.0/normFactor << endl;
    
    Eigen::Vector3d centrDiff = centroid1 - centroid2;
    // covariance of the second plane relative to the centroid of the first plane
    Eigen::Matrix3d relCovar = covar2 + (centrDiff * centrDiff.transpose());
    
    const Eigen::Vector3d &normal = evecs.col(2);
    double varNorm = normal.transpose() * relCovar * normal;
//    cout << "covar2 = " << covar2 << endl;
//    cout << "relCovar = " << relCovar << endl;
//    cout << "normal = " << normal.transpose() << endl;
//    cout << "varNorm = " << varNorm << endl;
    return varNorm;
}

void PlaneEstimator::compCentroidAndCovar(const Eigen::MatrixXd &pts,
                                         Eigen::Vector3d &centroid,
                                         Eigen::Matrix3d &covar)
{
    Eigen::Vector4d mean = Eigen::Vector4d::Zero();
    for(int i = 0; i < pts.cols(); ++i){
        mean += pts.col(i);
    }
    mean /= pts.cols();
    
    centroid = mean.head<3>();
    
    Eigen::MatrixXd demeanPts = pts;
    for(int i = 0; i < demeanPts.cols(); ++i){
        demeanPts.col(i) -= mean;
    }
    
    covar = demeanPts.topRows<3>() * demeanPts.topRows<3>().transpose();
}

void PlaneEstimator::compPlaneParams(const Eigen::Vector3d centroid,
                                     const Eigen::Matrix3d &covar,
                                     Eigen::Matrix3d &evecs,
                                     Eigen::Vector3d &evals,
                                     Eigen::Vector4d &planeEq)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> evd(covar);
    
    for(int i = 0; i < 3; ++i){
        evecs.col(i) = evd.eigenvectors().col(2 - i);
        evals(i) = evd.eigenvalues()(2 - i);
    }
    
    planeEq.head<3>() = evecs.col(2).cast<double>();
    // distance is the dot product of normal and point lying on the plane
    planeEq(3) = -planeEq.head<3>().dot(centroid);
}

void PlaneEstimator::transform(const Vector7d &transform) {
    g2o::SE3Quat transformSE3Quat(transform);
    Eigen::Matrix4d transformMat = transformSE3Quat.to_homogeneous_matrix();
    Eigen::Matrix3d R = transformMat.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformMat.block<3, 1>(0, 3);
    Eigen::Matrix4d Tinvt = transformMat.inverse();
    Tinvt.transposeInPlace();
    
    // Eigen::Vector3d centroid;
    centroid = R * centroid + t;
    
    // Eigen::Matrix3d covar;
    covar = R * covar * R.transpose();
    
    // Eigen::Matrix3d evecs;
    evecs = R * evecs;
    
    // Eigen::Vector3d evals;
    // no need to transform
    
    // Eigen::Vector4d planeEq;
    planeEq = Tinvt * planeEq;
    
    // int npts;
    // no need to transform
}

void PlaneEstimator::updateCentroidAndCovar(const Eigen::Vector3d &centroid1,
                                            const Eigen::Matrix3d &covar1,
                                            const int &npts1,
                                            const Eigen::Vector3d &centroid2,
                                            const Eigen::Matrix3d &covar2,
                                            const int &npts2,
                                            Eigen::Vector3d &ocentroid,
                                            Eigen::Matrix3d &ocovar,
                                            int &onpts)
{
    ocentroid = (npts1 * centroid1 + npts2 * centroid2)/(npts1 + npts2);
    Eigen::Vector3d centrDiff = centroid1 - centroid2;
    ocovar = covar1 + covar2 + (npts1 * npts2)/(npts1 + npts2)*(centrDiff * centrDiff.transpose());
    onpts += npts1 + npts2;
}
