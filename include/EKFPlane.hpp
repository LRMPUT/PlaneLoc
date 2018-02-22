//
// Created by jachu on 05.02.18.
//

#ifndef PLANELOC_EKFPLANE_HPP
#define PLANELOC_EKFPLANE_HPP

#include <boost/serialization/access.hpp>

#include <Eigen/Eigen>


class EKFPlane {
public:
    EKFPlane();
    
    EKFPlane(const Eigen::Quaterniond &xq,
             const Eigen::Matrix4d &Pq);
    
    EKFPlane(const Eigen::Quaterniond &xq, int npts);
    
    void init(const Eigen::Quaterniond &xq,
              const Eigen::Matrix4d &Pq);
    
    void init(const Eigen::Quaterniond &xq, int npts);
    
    void update(const Eigen::Quaterniond &zq,
                const Eigen::Matrix4d &Rq);
    
    void update(const Eigen::Quaterniond &zq,
                const Eigen::Matrix3d &R);
    
    void update(const Eigen::Quaterniond &zq,
                int znpts);
    
//    void transform(const Eigen::Matrix4d T);
    
    double distance(const Eigen::Quaterniond &xcq) const;
    
    static void compPlaneEqAndCovar(const Eigen::MatrixXd &pts,
                                    Eigen::Quaterniond &q,
                                    Eigen::Matrix4d &R);
    
    const Eigen::Quaterniond &getX() const;
    
    const Eigen::Matrix3d &getP() const;

private:
    
    static Eigen::MatrixXd jacob_dom_dq(const Eigen::Quaterniond &q);
    
    static Eigen::MatrixXd jacob_dqn_dq(const Eigen::Quaterniond &q);
    
    Eigen::Quaterniond x;
    
    Eigen::Matrix3d P;
    
    int npts;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & x;
        ar & P;
        ar & npts;
    }
};


#endif //PLANELOC_EKFPLANE_HPP
