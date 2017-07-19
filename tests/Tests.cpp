//
// Created by jachu on 18.07.17.
//

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>

#include <Eigen/Eigen>

#include "Matching.hpp"
#include "Misc.hpp"

using namespace std;

Eigen::Vector3d closestPointOnLine(Eigen::Vector3d pt,
                                   Eigen::Vector3d p,
                                   Eigen::Vector3d n)
{
    n.normalize();
    double t = (pt - p).dot(n);
    return p + t * n;
}

void transformObjs(const std::vector<Eigen::Vector3d> &points,
                   const std::vector<Eigen::Vector4d> &planes,
                   const std::vector<Vector6d> &lines,
                   std::vector<Eigen::Vector3d> &retPoints,
                   std::vector<Eigen::Vector4d> &retPlanes,
                   std::vector<Vector6d> &retLines,
                   Eigen::Matrix3d rotMat,
                   Eigen::Vector3d trans,
                   double rotNoise = 0.0,
                   double transNoise = 0.0)
{
    for(int p = 0; p < points.size(); ++p){
        retPoints.push_back(rotMat * points[p] + trans);
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0,0) = rotMat;
    T.block<3, 1>(0, 3) = trans;
    Eigen::Matrix4d Tinvt = T.inverse().transpose();
    for(int pl = 0; pl < planes.size(); ++pl){
        Eigen::Vector4d planeTrans = Tinvt * planes[pl];
        double nNorm = planeTrans.head<3>().norm();
        planeTrans /= nNorm;
        retPlanes.push_back(planeTrans);
    }
    for(int l = 0; l < lines.size(); ++l){
        Eigen::Vector3d p = lines[l].head<3>();
        Eigen::Vector3d n = lines[l].tail<3>();
        Eigen::Vector3d pTrans = rotMat * p + trans;
        Eigen::Vector3d nTrans = rotMat * n;
        // move point to be closest to the origin
        pTrans = closestPointOnLine(Eigen::Vector3d::Zero(), pTrans, nTrans);
        Vector6d lTrans;
        lTrans.head<3>() = pTrans;
        lTrans.tail<3>() = nTrans;
        retLines.push_back(lTrans);
    }
}

void addPointsDirsDists(const std::vector<Eigen::Vector3d> &points,
                        const std::vector<Eigen::Vector4d> &planes,
                        const std::vector<Vector6d> &lines,
                        std::vector<Eigen::Vector3d> &retPoints,
                        std::vector<Eigen::Vector3d> &retDirs,
                        std::vector<double> &retDists,
                        std::vector<Eigen::Vector3d> &retDistDirs,
                        std::vector<Eigen::Vector3d> &retDistPts,
                        std::vector<Eigen::Vector3d> &retDistPtsDirs)
{
    for(int p = 0; p < points.size(); ++p){
        retPoints.push_back(points[p]);
    }
    for(int pl = 0; pl < planes.size(); ++pl){
        Eigen::Vector3d n = planes[pl].head<3>();
        double d = -planes[pl][3];
        retDirs.push_back(n);
        retDists.push_back(d);
        retDistDirs.push_back(n);
    }
    for(int l = 0; l < lines.size(); ++l){
        Eigen::Vector3d p = lines[l].head<3>();
        Eigen::Vector3d n = lines[l].tail<3>();
        retDirs.push_back(n);
        Eigen::FullPivLU<Eigen::Vector3d> lu(n.transpose());
        Eigen::MatrixXd nullSpace = lu.kernel();
        // distances in two directions orthogonal to n
        retDistPts.push_back(p);
        retDistDirs.push_back(nullSpace.block<3, 1>(0, 0));
        retDistPts.push_back(p);
        retDistDirs.push_back(nullSpace.block<3, 1>(0, 1));
    }
}

void testFullConstr(const std::vector<Eigen::Vector3d> &points,
                    const std::vector<Eigen::Vector3d> &dirs,
                    const std::vector<double> &dists,
                    const std::vector<Eigen::Vector3d> &distDirs,
                    const std::vector<Eigen::Vector3d> &distPts,
                    const std::vector<Eigen::Vector3d> &distPtsDirs,
                    bool &fullConstrRot,
                    bool &fullConstrTrans)
{
    {
        int rotEqNum = points.size();
        if (dirs.size() > 0) {
            rotEqNum += dirs.size() + 1;
        }
        Eigen::MatrixXd rotMat(3, rotEqNum);
        int rotEqCnt = 0;
        double maxDist = 0.0;
        for (int p = 0; p < points.size(); ++p) {
            rotMat.block<3, 1>(0, rotEqCnt++) = points[p];
            maxDist = max(maxDist, points[p].norm());
        }
        // treating directions as points + additional non-zero origin point
        // constructed that all virtual points  are not equal to any real point
        Eigen::Vector3d dirsOrig = Eigen::Vector3d::Ones() * (maxDist + 2.0);
        if (dirs.size() > 0) {
            rotMat.block<3, 1>(0, rotEqCnt++) = dirsOrig;
        }
        for (int d = 0; d < dirs.size(); ++d) {
            rotMat.block<3, 1>(0, rotEqCnt++) = dirsOrig + dirs[d];
        }
//        if(rotEqCnt != rotEqNum){
//            cout << "rotEqCnt != rotEqNum" << endl;
//            throw std::exception();
//        }
        Eigen::FullPivLU<Eigen::MatrixXd> rotLu(rotMat);
//            lu.setThreshold(sinValsThresh * distScale);
        int rotRank = rotLu.rank();
        cout << "rotLu.rank() = " << rotLu.rank() << endl;
        if (rotRank < 3) {
            fullConstrRot = false;
        } else {
            fullConstrRot = true;
        }
    }
    {
        int transEqNum = dists.size();
        if (fullConstrRot) {
            transEqNum += points.size() * 3 + distPts.size();
        }
        Eigen::MatrixXd transMat(3, transEqNum);
        int transEqCnt = 0;
        for(int d = 0; d < dists.size(); ++d){
            transMat.block<3, 1>(0, transEqCnt++) = distDirs[d];
        }
        if(fullConstrRot) {
            for (int p = 0; p < points.size(); ++p) {
                transMat.block<3, 1>(0, transEqCnt++) = (Eigen::Vector3d() << 1.0, 0.0, 0.0).finished();
                transMat.block<3, 1>(0, transEqCnt++) = (Eigen::Vector3d() << 0.0, 1.0, 0.0).finished();
                transMat.block<3, 1>(0, transEqCnt++) = (Eigen::Vector3d() << 0.0, 0.0, 1.0).finished();
            }
            for(int dp = 0; dp < distPts.size(); ++dp){
                transMat.block<3, 1>(0, transEqCnt++) = distPtsDirs[dp];
            }
        }
//        if(transEqCnt != transEqNum){
//            cout << "transEqCnt != transEqNum" << endl;
//            throw std::exception();
//        }
        Eigen::FullPivLU<Eigen::MatrixXd> transLu(transMat);
//            lu.setThreshold(sinValsThresh * distScale);
        int transRank = transLu.rank();
        cout << "transLu.rank() = " << transLu.rank() << endl;
        if (transRank < 3) {
            fullConstrTrans = false;
        } else {
            fullConstrTrans = true;
        }
    }

}

Vector7d testTransform(const std::vector<Eigen::Vector3d> &points,
                       const std::vector<Eigen::Vector4d> &planes,
                       const std::vector<Vector6d> &lines,
                       Vector7d transform,
                       double sinValsThresh)
{
    Eigen::Vector3d trans = transform.head<3>();
    Eigen::Vector4d rot = transform.tail<4>();
    Eigen::Matrix3d rotMat = Eigen::Quaterniond(rot[3], rot[0], rot[1], rot[2]).toRotationMatrix();

    std::vector<Eigen::Vector3d> retPoints;
    std::vector<Eigen::Vector3d> retDirs;
    std::vector<double> retDists;
    std::vector<Eigen::Vector3d> retDistDirs;
    std::vector<Eigen::Vector3d> retDistPts;
    std::vector<Eigen::Vector3d> retDistPtsDirs;
    addPointsDirsDists(points,
                       planes,
                       lines,
                       retPoints,
                       retDirs,
                       retDists,
                       retDistDirs,
                       retDistPts,
                       retDistPtsDirs);

    bool fullConstrRot = true;
    bool fullConstrTrans = true;
    testFullConstr(retPoints,
                   retDirs,
                   retDists,
                   retDistDirs,
                   retDistPts,
                   retDistPtsDirs,
                   fullConstrRot,
                   fullConstrTrans);


    vector<Eigen::Vector3d> transPoints;
    vector<Eigen::Vector4d> transPlanes;
    vector<Vector6d> transLines;
    transformObjs(points,
                  planes,
                  lines,
                  transPoints,
                  transPlanes,
                  transLines,
                  rotMat,
                  trans);

    std::vector<Eigen::Vector3d> retTransPoints;
    std::vector<Eigen::Vector3d> retTransDirs;
    std::vector<double> retTransDists;
    std::vector<Eigen::Vector3d> retTransDistDirs;
    std::vector<Eigen::Vector3d> retTransDistPts;
    std::vector<Eigen::Vector3d> retTransDistPtsDirs;
    addPointsDirsDists(transPoints,
                       transPlanes,
                       transLines,
                       retTransPoints,
                       retTransDirs,
                       retTransDists,
                       retTransDistDirs,
                       retTransDistPts,
                       retTransDistPtsDirs);

    bool fullConstrRotComp = true;
    bool fullConstrTransComp = true;
    cout << "Running bestTransformPointsDirsDists" << endl;
    Vector7d transformComp = Matching::bestTransformPointsDirsDists(retTransPoints,
                                                                    retPoints,
                                                                    vector<double>(retPoints.size(), 1.0),
                                                                    retTransDirs,
                                                                    retDirs,
                                                                    vector<double>(retDirs.size(), 1.0),
                                                                    retTransDists,
                                                                    retDists,
                                                                    retTransDistDirs,
                                                                    vector<double>(retDists.size(), 1.0),
                                                                    retTransDistPts,
                                                                    retDistPts,
                                                                    retTransDistPtsDirs,
                                                                    vector<double>(retDistPts.size(), 1.0),
                                                                    sinValsThresh,
                                                                    fullConstrRotComp,
                                                                    fullConstrTransComp);

    double dist = Misc::transformLogDist(transform, transformComp);

    cout << "transform = " << transform.transpose() << endl;
    cout << "transformComp = " << transformComp.transpose() << endl;
    REQUIRE(fullConstrRot == fullConstrRotComp);
    REQUIRE(fullConstrTrans == fullConstrTransComp);
    if(fullConstrRot == true &&
       fullConstrTrans == true)
    {
        REQUIRE(dist < 0.001);
    }
}

TEST_CASE("best transformations are correct", "[transformations]"){
    static constexpr int numPts = 10;
    static constexpr int numPls = 10;
    static constexpr int numLines = 10;
    static constexpr int numTests = 10;
    static constexpr double distScale = 10.0;
    static constexpr double sinValsThresh = 0.001;

    cout << "Starting test" << endl;
    //[x, y, z]
    vector<Eigen::Vector3d> points;
    //[nx, ny, nz, -d]
    vector<Eigen::Vector4d> planes;
    //[px, py, pz, nx, ny, nz]
    vector<Vector6d> lines;

    for(int pt = 0; pt < numPts; ++pt){
        points.push_back(Eigen::Vector3d::Random());
        // from -10 do 10
        points.back() *= distScale;
    }
    for(int pl = 0; pl < numPls; ++pl){
        planes.push_back(Eigen::Vector4d::Random());
        planes.back().head<3>().normalize();
        planes.back()[3] *= distScale;
    }
    for(int l = 0; l < numLines; ++l){
        lines.push_back(Vector6d::Random());
        lines.back().head<3>() *= distScale;
        lines.back().tail<3>().normalize();
        // move point to be closest to the origin
        lines.back().head<3>() = closestPointOnLine(Eigen::Vector3d::Zero(),
                                                    lines.back().head<3>(),
                                                    lines.back().tail<3>());
    }


    for(int t = 0; t < numTests; ++t){
        Eigen::Vector3d trans = Eigen::Vector3d::Random();
        Eigen::Vector4d rot = Eigen::Vector4d::Random();
        rot.normalize();
        Eigen::Matrix3d rotMat = Eigen::Quaterniond(rot[3], rot[0], rot[1], rot[2]).toRotationMatrix();
        Vector7d transform;
        transform.head<3>() = trans;
        transform.tail<4>() = rot;

        //points only
        for(int c = 1; c <= numPts; ++c){
            std::default_random_engine gen;
            std::uniform_int_distribution<int> distr(0, numPts - 1);

            vector<Eigen::Vector3d> curPts;
            for(int p = 0; p < c; ++p){
                int idx = distr(gen);
                curPts.push_back(points[idx]);
            }

            testTransform(curPts,
                          vector<Eigen::Vector4d>(),
                          vector<Vector6d>(),
                          transform,
                          sinValsThresh);
        }

        //planes only
//        for(int c = 1; c <= numPts; ++c) {
//            std::default_random_engine gen;
//            std::uniform_int_distribution<int> distr(0, numPls - 1);
//
//            vector<Eigen::Vector4d> curPls;
//            Eigen::MatrixXd plsMat(3, c);
//            for (int p = 0; p < c; ++p) {
//                int idx = distr(gen);
//                curPls.push_back(planes[idx]);
//                plsMat.block<4, 1>(0, p) = curPls.back();
//            }
//            Eigen::FullPivLU<Eigen::MatrixXd> lu(plsMat);
////            lu.setThreshold(sinValsThresh * distScale);
//            int rank = lu.rank();
//            cout << "lu.rank() = " << lu.rank() << endl;
//            bool fullConstrRot = true;
//            bool fullConstrTrans = true;
//            if (rank < 3) {
//                fullConstrRot = false;
//                fullConstrTrans = false;
//            }
//        }

        //lines only

        //points + planes

        //points + lines

        //planes + lines

        //points + planes + lines
    }
}

//TEST_CASE( "vectors can be sized and resized", "[vector]" ) {
//
//    std::vector<int> v( 5 );
//
//    REQUIRE( v.size() == 5 );
//    REQUIRE( v.capacity() >= 5 );
//
//    SECTION( "resizing bigger changes size and capacity" ) {
//        v.resize( 10 );
//
//        REQUIRE( v.size() == 10 );
//        REQUIRE( v.capacity() >= 10 );
//    }
//    SECTION( "resizing smaller changes size but not capacity" ) {
//        v.resize( 0 );
//
//        REQUIRE( v.size() == 0 );
//        REQUIRE( v.capacity() >= 5 );
//    }
//    SECTION( "reserving bigger changes capacity but not size" ) {
//        v.reserve( 10 );
//
//        REQUIRE( v.size() == 5 );
//        REQUIRE( v.capacity() >= 10 );
//    }
//    SECTION( "reserving smaller does not change size or capacity" ) {
//        v.reserve( 0 );
//
//        REQUIRE( v.size() == 5 );
//        REQUIRE( v.capacity() >= 5 );
//    }
//}