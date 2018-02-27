//
// Created by jachu on 27.02.18.
//

#ifndef PLANELOC_PLANEESTIMATOR_HPP
#define PLANELOC_PLANEESTIMATOR_HPP

#include <boost/serialization/access.hpp>

#include <Eigen/Dense>

#include "Types.hpp"

class PlaneEstimator {
public:
    PlaneEstimator();
    
    PlaneEstimator(const Eigen::MatrixXd &pts);
    
    PlaneEstimator(const Eigen::Vector3d &icentroid,
                   const Eigen::Matrix3d &icovar,
                   int inpts);
    
    void init(const Eigen::MatrixXd &pts);
    
    void init(const Eigen::Vector3d &icentroid,
              const Eigen::Matrix3d &icovar,
              int inpts);
    
    void update(const Eigen::Vector3d &ucentroid,
                const Eigen::Matrix3d &ucovar,
                int unpts);

//    void transform(const Eigen::Matrix4d T);
    
    double distance(const PlaneEstimator &other) const;
    
    static void compCentroidAndCovar(const Eigen::MatrixXd &pts,
                                    Eigen::Vector3d &centroid,
                                    Eigen::Matrix3d &covar);
    
    static void compPlaneParams(const Eigen::Vector3d centroid,
                                const Eigen::Matrix3d &covar,
                                Eigen::Matrix3d &evecs,
                                Eigen::Vector3d &evals,
                                Eigen::Vector4d &planeEq);
    
    void transform(const Vector7d &transform);
    
    const Eigen::Vector3d &getCentroid() const {
        return centroid;
    }
    
    const Eigen::Matrix3d &getCovar() const {
        return covar;
    }
    
    const Eigen::Matrix3d &getEvecs() const {
        return evecs;
    }
    
    const Eigen::Vector3d &getEvals() const {
        return evals;
    }
    
    const Eigen::Vector4d &getPlaneEq() const {
        return planeEq;
    }
    
    int getNpts() const {
        return npts;
    }
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    
    Eigen::Vector3d centroid;
    
    Eigen::Matrix3d covar;
    
    Eigen::Matrix3d evecs;
    
    Eigen::Vector3d evals;
    
    Eigen::Vector4d planeEq;
    
    int npts;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & centroid;
        ar & covar;
        ar & evecs;
        ar & evals;
        ar & planeEq;
        ar & npts;
    }
};


#endif //PLANELOC_PLANEESTIMATOR_HPP
