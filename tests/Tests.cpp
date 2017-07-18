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

TEST_CASE("best transformations are correct", "[transformations]"){
    static constexpr int numPts = 10;
    static constexpr int numPls = 10;
    static constexpr int numLines = 10;
    static constexpr int numTests = 10;
    static constexpr double distScale = 10.0;
    static constexpr double sinValsThresh = 0.001;

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
            Eigen::MatrixXd ptsMat(3, c);
            for(int p = 0; p < c; ++p){
                int idx = distr(gen);
                curPts.push_back(points[idx]);
                ptsMat.block<3, 1>(0, p) = curPts.back();
            }
            Eigen::FullPivLU<Eigen::Matrix4d> lu(ptsMat);
            lu.setThreshold(sinValsThresh * distScale);
            int rank = lu.rank();
            bool fullConstrRot = true;
            bool fullConstrTrans = true;
            if(rank < 3){
                fullConstrRot = false;
                fullConstrTrans = false;
            }

            vector<Eigen::Vector3d> curPtsTrans;
            for(int p = 0; p < c; ++p){
                curPtsTrans.push_back(rotMat * curPts[p] + trans);
            }

            bool fullConstrRotComp = true;
            bool fullConstrTransComp = true;
            Vector7d transformComp = Matching::bestTransformPointsDirsDists(curPts,
                                                                        curPtsTrans,
                                                                        vector<double>(curPts.size(), 1.0),
                                                                        vector<Eigen::Vector3d>(),
                                                                        vector<Eigen::Vector3d>(),
                                                                        vector<double>(),
                                                                        vector<double>(),
                                                                        vector<double>(),
                                                                        vector<Eigen::Vector3d>(),
                                                                        vector<double>(),
                                                                        vector<Eigen::Vector3d>(),
                                                                        vector<Eigen::Vector3d>(),
                                                                        vector<Eigen::Vector3d>(),
                                                                        vector<double>(),
                                                                        sinValsThresh,
                                                                        fullConstrRotComp,
                                                                        fullConstrTransComp);

            double dist = Misc::transformLogDist(transform, transformComp);

            REQUIRE(fullConstrRot == fullConstrRotComp);
            REQUIRE(fullConstrTrans == fullConstrTransComp);
            if(fullConstrRot == fullConstrRotComp &&
                    fullConstrTrans == fullConstrTransComp)
            {
                REQUIRE(dist < 0.001);
            }
        }

        //planes only

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