//
// Created by jachu on 27.06.17.
//

#include <random>
#include <thread>
#include <chrono>

#include <Misc.hpp>
#include "LineDet.hpp"

using namespace std;

void LineDet::detectLineSegments(const cv::FileStorage &settings,
                                 cv::Mat rgb,
                                 const std::vector<ObjInstance> &planes,
                                 cv::Mat cameraMatrix,
                                 std::vector<LineSeg> &lineSegs,
                                 pcl::visualization::PCLVisualizer::Ptr viewer,
                                 int viewPort1,
                                 int viewPort2)
{
    static constexpr double minImgLineLen = 50;
    static constexpr double lineNhSize = 10;
    static constexpr double shadingLevel = 0.005;
//    static constexpr bool visualize = true;

    if (!rgb.empty()) {
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
        vector<cv::Vec4f> linesLsd;

        cv::Mat imageGray;
        cv::cvtColor(rgb, imageGray, cv::COLOR_RGB2GRAY);
        lsd->detect(imageGray, linesLsd);

        vector<cv::Vec4f> linesLsdThresh;
        for(int ls = 0; ls < linesLsd.size(); ++ls){
            Eigen::Vector2d p1, p2;
            p1[0] = linesLsd[ls][0];
            p1[1] = linesLsd[ls][1];
            p2[0] = linesLsd[ls][2];
            p2[1] = linesLsd[ls][3];
            if((p2 - p1).norm() > minImgLineLen) {
                lineSegs.emplace_back(-1, p1, p2, Eigen::Vector3d(), Eigen::Vector3d());
                linesLsdThresh.push_back(linesLsd[ls]);
            }
        }
        cout << "Detected " << linesLsdThresh.size() << " lines" << endl;

        cv::Mat planesMasks(rgb.size(), CV_32SC1, cv::Scalar(-1));
        cv::Mat planesDists(rgb.size(), CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));
        for(int pl = 0; pl < planes.size(); ++pl){
            cv::Mat pointsReproj = Misc::reprojectTo2D(planes[pl].getPoints(), cameraMatrix);
            for(int pt = 0; pt < pointsReproj.cols; ++pt){
                int u = std::round(pointsReproj.at<cv::Vec3f>(pt)[0]);
                int v = std::round(pointsReproj.at<cv::Vec3f>(pt)[1]);
                float d = pointsReproj.at<cv::Vec3f>(pt)[2];

                if(u >= 0 && u < planesDists.cols && v >= 0 && v < planesDists.rows){
                    if(planesDists.at<float>(v, u) > d){
                        planesDists.at<float>(v, u) = d;
                        planesMasks.at<int>(v, u) = pl;
                    }
                }
            }
        }

        cv::Mat planesMasksImg;
        cv::Mat imageDisp;
        if(viewer) {
            planesMasksImg = Misc::colorIds(planesMasks);

            imageDisp = rgb.clone();
            lsd->drawSegments(imageDisp, linesLsdThresh);

            cv::imshow("Detected segments", imageDisp);
            cv::imshow("Projected planes", planesMasksImg);


            viewer->removeAllPointClouds(viewPort1);
            viewer->removeAllShapes(viewPort1);
            for(int pl = 0; pl < planes.size(); ++pl){
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = planes[pl].getPoints();
                viewer->addPointCloud(curPl, string("plane_") + to_string(pl), viewPort1);
//                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//                                                         shadingLevel,
//                                                         string("plane1_") + to_string(pl),
//                                                         viewPort1);
            }

            viewer->initCameraParameters();
            viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
        }

        std::default_random_engine gen;
        std::uniform_real_distribution<double> distr(0.0, lineNhSize);
        for(int l = 0; l < lineSegs.size(); ++l){
            cout << "line " << l << endl;
            
            Eigen::Vector2d pi1 = lineSegs[l].getPis1().back();
            Eigen::Vector2d pi2 = lineSegs[l].getPis2().back();

            cv::Mat nhImg;
            if(viewer){
                nhImg = rgb.clone();
                cv::line(nhImg,
                         cv::Point(pi1[0], pi1[1]),
                         cv::Point(pi2[0], pi2[1]),
                         cv::Scalar(0, 0, 255));
            }

            map<int, int> planeVotesLeft, planeVotesRight;
            Eigen::Vector2d n = pi2 - pi1;
            double norm = n.norm();
            n /= norm;
            Eigen::Vector2d perpL;
            perpL[0] = -n[1];
            perpL[1] = n[0];

            for(int s = 0; s <= norm; ++s){
                Eigen::Vector2d p = pi1 + n * s;
//                double slen = distr(gen);
                for(int slen = 0; slen < lineNhSize; ++slen) {
                    Eigen::Vector2d pL = p + perpL * slen;
                    pL[0] = std::round(pL[0]);
                    pL[1] = std::round(pL[1]);
                    Eigen::Vector2d pR = p - perpL * slen;
                    pR[0] = std::round(pR[0]);
                    pR[1] = std::round(pR[1]);

                    if(viewer){
                        cv::circle(nhImg, cv::Point(pL[0], pL[1]), 1, cv::Scalar(0, 255, 0));
                        cv::circle(nhImg, cv::Point(pR[0], pR[1]), 1, cv::Scalar(255, 0, 0));
                    }

                    if (pL[0] >= 0 && pL[0] < rgb.cols && pL[1] >= 0 && pL[1] < rgb.rows) {
                        planeVotesLeft[planesMasks.at<int>(pL[1], pL[0])]++;
                    }
                    if (pR[0] >= 0 && pR[0] < rgb.cols && pR[1] >= 0 && pR[1] < rgb.rows) {
                        planeVotesRight[planesMasks.at<int>(pR[1], pR[0])]++;
                    }
                }
            }

            int bestPlaneL = -1;
            int bestPlaneLVotes = 0;
            int bestPlaneR = -1;
            int bestPlaneRVotes = 0;
            for(auto it = planeVotesLeft.begin(); it != planeVotesLeft.end(); ++it){
                if(it->second > bestPlaneLVotes){
                    bestPlaneL = it->first;
                    bestPlaneLVotes = it->second;
                }
            }
            for(auto it = planeVotesRight.begin(); it != planeVotesRight.end(); ++it){
                if(it->second > bestPlaneRVotes){
                    bestPlaneR = it->first;
                    bestPlaneRVotes = it->second;
                }
            }
            if(bestPlaneL >= 0){
                const ObjInstance& curPl = planes[bestPlaneL];
                Eigen::Vector3d p1 = Misc::projectPointOnPlane(pi1, curPl.getNormal(), cameraMatrix);
                Eigen::Vector3d p2 = Misc::projectPointOnPlane(pi2, curPl.getNormal(), cameraMatrix);
                cout << "projected on left plane" << endl;
                cout << "p1 = " << p1.transpose() << endl;
                cout << "p2 = " << p2.transpose() << endl;
//                lineSegs[l].setP1(p1);
//                lineSegs[l].setP2(p2);
                if(viewer) {
                    viewer->addLine(pcl::PointXYZ(p1[0], p1[1], p1[2]),
                                    pcl::PointXYZ(p2[0], p2[1], p2[2]),
                                    0.0, 1.0, 0.0,
                                    string("line_l_") + to_string(l),
                                    viewPort1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                        4,
                                                        string("line_l_") + to_string(l),
                                                        viewPort1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                        pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                        string("line_l_") + to_string(l),
                                                        viewPort1);
//                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                        0.0, 1.0, 0.0,
//                                                        string("line_l_") + to_string(l),
//                                                        viewPort1);
                }
            }
            if(bestPlaneR >= 0){
                const ObjInstance& curPl = planes[bestPlaneR];
                Eigen::Vector3d p1 = Misc::projectPointOnPlane(pi1, curPl.getNormal(), cameraMatrix);
                Eigen::Vector3d p2 = Misc::projectPointOnPlane(pi2, curPl.getNormal(), cameraMatrix);
                cout << "projected on right plane" << endl;
                cout << "p1 = " << p1.transpose() << endl;
                cout << "p2 = " << p2.transpose() << endl;
//                lineSegs[l].setP1(p1);
//                lineSegs[l].setP2(p2);
                if(viewer) {
                    viewer->addLine(pcl::PointXYZ(p1[0], p1[1], p1[2]),
                                    pcl::PointXYZ(p2[0], p2[1], p2[2]),
                                    0.0, 0.0, 1.0,
                                    string("line_r_") + to_string(l),
                                    viewPort1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                        4,
                                                        string("line_r_") + to_string(l),
                                                        viewPort1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                        pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                        string("line_r_") + to_string(l),
                                                        viewPort1);
//                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                        0.0, 0.0, 1.0,
//                                                        string("line_r_") + to_string(l),
//                                                        viewPort1);
                }
            }

            if(viewer){
                viewer->resetStoppedFlag();

                cv::imshow("Current line segment", nhImg);
                while (!viewer->wasStopped()){
                    viewer->spinOnce (50);
                    cv::waitKey(50);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }

                viewer->removeShape(string("line_l_") + to_string(l), viewPort1);
                viewer->removeShape(string("line_r_") + to_string(l), viewPort1);
            }
        }


    }
    else{
        cout << "Warning: rgb empty!" << endl;
    }
}
