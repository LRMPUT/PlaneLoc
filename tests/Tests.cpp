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
        Eigen::FullPivLU<Eigen::MatrixXd> lu(n.transpose());
        Eigen::MatrixXd nullSpace = lu.kernel();
        Eigen::Vector3d dir1 = nullSpace.block<3, 1>(0, 0).normalized();
        Eigen::Vector3d dir2 = nullSpace.block<3, 1>(0, 1).normalized();

//        cout << "n = " << n.transpose() << endl;
//        cout << "dir1 = " << dir1.transpose() << endl;
//        cout << "n * dir1 = " << n.dot(dir1) << endl;
//        cout << "dir2 = " << dir2.transpose() << endl;
//        cout << "n * dir2 = " << n.dot(dir2) << endl;
        // distances in two directions orthogonal to n
        retDistPts.push_back(p);
        retDistPtsDirs.push_back(dir1);
        retDistPts.push_back(p);
        retDistPtsDirs.push_back(dir2);
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
        int rotEqNum = dirs.size();
        if (points.size() > 0) {
            rotEqNum += points.size() - 1;
        }
        Eigen::MatrixXd rotMat(3, rotEqNum);
        int rotEqCnt = 0;
        //treating points as directions from the first point
        for (int p = 1; p < points.size(); ++p) {
            rotMat.block<3, 1>(0, rotEqCnt++) = points[p] - points[0];
        }

        for (int d = 0; d < dirs.size(); ++d) {
            rotMat.block<3, 1>(0, rotEqCnt++) = dirs[d];
        }
//        if(rotEqCnt != rotEqNum){
//            cout << "rotEqCnt != rotEqNum" << endl;
//            throw std::exception();
//        }
        Eigen::FullPivLU<Eigen::MatrixXd> rotLu(rotMat);
//            lu.setThreshold(sinValsThresh * distScale);
        int rotRank = rotLu.rank();
        cout << "rotLu.rank() = " << rotLu.rank() << endl;
        if (rotRank < 2) {
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
            cout << "distPts.size() = " << distPts.size() << endl;
            cout << "distPtsDirs.size() = " << distPtsDirs.size() << endl;
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
    cout << endl << "testing transformation" << endl;
    cout << "point.size() = " << points.size() << endl;
    cout << "planes.size() = " << planes.size() << endl;
    cout << "lines.size() = " << lines.size() << endl;

    for(int pt = 0; pt < points.size(); ++pt){
        cout << "points[" << pt << "] = " << points[pt].transpose() << endl;
    }
    for(int pl = 0; pl < planes.size(); ++pl){
        cout << "planes[" << pl << "] = " << planes[pl].transpose() << endl;
    }
    for(int l = 0; l < lines.size(); ++l){
        cout << "lines[" << l << "] = " << lines[l].size() << endl;
    }

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

//    for(int dp = 0; dp < retDistPts.size(); ++dp){
//        Eigen::Matrix4d Wrt = Misc::matrixW(Eigen::Quaterniond(transform[6],
//                                               transform[3],
//                                               transform[4],
//                                               transform[5]).normalized()).transpose();
//        Eigen::Matrix4d Qr = Misc::matrixQ(Eigen::Quaterniond(transform[6],
//                                                              transform[3],
//                                                              transform[4],
//                                                              transform[5]).normalized());
//
//        cout << "retDistPts = " << retDistPts[dp].transpose() << endl;
//        cout << "retTransDistPts = " << retTransDistPts[dp].transpose() << endl;
//        cout << "retTransDistPtsDir = " << retTransDistPtsDirs[dp].transpose() << endl;
//
//        double d1 = retTransDistPtsDirs[dp].dot(retTransDistPts[dp]);
//
//        Eigen::Vector4d p2quat = Eigen::Vector4d::Zero();
//        p2quat.head<3>() = retDistPts[dp];
//        Eigen::Vector4d p2trans = Wrt * Qr * p2quat;
//        double d2 = retTransDistPtsDirs[dp].dot(p2trans.head<3>());
//
//        cout << "d1 = " << d1 << endl;
//        cout << "d2 = " << d2 << endl;
//
//        cout << "t = " << trans.transpose() << endl;
//        cout << "n * t = " << retTransDistPtsDirs[dp].dot(trans) << endl;
//        cout << "d1 - d2 = " << (d1 - d2) << endl;
//        cout << "n * t - (d1 - d2) = " << retTransDistPtsDirs[dp].dot(trans) - (d1 - d2) << endl;
//
//
//    }

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
    if(fullConstrRot == true){
        double rotDist = Misc::rotLogDist(transform.tail<4>(), transformComp.tail<4>());
        REQUIRE(rotDist < 0.001);
    }
    if(fullConstrTrans == true){
        Eigen::Vector3d t = transform.head<3>();
        Eigen::Vector3d tComp = transformComp.head<3>();
        Eigen::Vector3d tDiff = t - tComp;
        REQUIRE(tDiff.transpose() * tDiff < 0.001);
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
//        Eigen::Matrix3d rotMat = Eigen::Quaterniond(rot[3], rot[0], rot[1], rot[2]).toRotationMatrix();
        Vector7d transform;
        transform.head<3>() = trans;
        transform.tail<4>() = rot;

        //points only
        for(int cpts = 0; cpts <= numPts; ++cpts){
            std::default_random_engine gen;
            std::uniform_int_distribution<int> distrPts(0, numPts - 1);

            vector<Eigen::Vector3d> curPts;
            for(int p = 0; p < cpts; ++p){
                int idx = distrPts(gen);
                curPts.push_back(points[idx]);
            }

            testTransform(curPts,
                          vector<Eigen::Vector4d>(),
                          vector<Vector6d>(),
                          transform,
                          sinValsThresh);
        }

        //planes only
        for(int cpls = 0; cpls <= numPls; ++cpls){
            std::default_random_engine gen;
            std::uniform_int_distribution<int> distrPls(0, numPls - 1);

            vector<Eigen::Vector4d> curPls;
            for(int pl = 0; pl < cpls; ++pl){
                int idx = distrPls(gen);
                curPls.push_back(planes[idx]);
            }

            testTransform(vector<Eigen::Vector3d>(),
                          curPls,
                          vector<Vector6d>(),
                          transform,
                          sinValsThresh);
        }


        //lines only
        for(int clines = 0; clines <= numLines; ++clines){
            std::default_random_engine gen;
            std::uniform_int_distribution<int> distrLines(0, numLines - 1);

            vector<Vector6d> curLines;
            for(int l = 0; l < clines; ++l){
                int idx = distrLines(gen);
                curLines.push_back(lines[idx]);
            }

            testTransform(vector<Eigen::Vector3d>(),
                          vector<Eigen::Vector4d>(),
                          curLines,
                          transform,
                          sinValsThresh);
        }

        //points + planes
        for(int cpts = 0; cpts <= numPts; ++cpts){
            for(int cpls = 0; cpls <= numPls; ++cpls) {
                std::default_random_engine gen;

                std::uniform_int_distribution<int> distrPts(0, numPts - 1);

                vector<Eigen::Vector3d> curPts;
                for (int p = 0; p < cpts; ++p) {
                    int idx = distrPts(gen);
                    curPts.push_back(points[idx]);
                }


                std::uniform_int_distribution<int> distrPls(0, numPls - 1);

                vector<Eigen::Vector4d> curPls;
                for (int pl = 0; pl < cpls; ++pl) {
                    int idx = distrPls(gen);
                    curPls.push_back(planes[idx]);
                }

                testTransform(curPts,
                              curPls,
                              vector<Vector6d>(),
                              transform,
                              sinValsThresh);
            }
        }

        //points + lines

        //planes + lines

        //points + planes + lines
    }
}
