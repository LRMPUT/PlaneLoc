//
// Created by jachu on 27.06.17.
//

#include <random>

#include <Misc.hpp>
#include "LineDet.hpp"

using namespace std;

void LineDet::detectLineSegments(const cv::FileStorage &settings,
                                 cv::Mat rgb,
                                 const std::vector<ObjInstance> &planes,
                                 cv::Mat cameraMatrix,
                                 std::vector<LineSeg> &lineSegs)
{
    static constexpr double minImgLineLen = 50;
    static constexpr double lineNhSize = 10;
    static constexpr int lineNhNumSamp = 100;

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
                        planesMasks.at<int>(v, u) = planes[pl].getId();
                    }
                }
            }
        }

        std::default_random_engine gen;
        std::uniform_real_distribution<double> distr(0.0, lineNhSize);
        for(int l = 0; l < lineSegs.size(); ++l){
            map<int, int> planeVotesLeft, planeVotesRight;
            Eigen::Vector2d pi1 = lineSegs[l].getPis1().back();
            Eigen::Vector2d pi2 = lineSegs[l].getPis2().back();
            Eigen::Vector2d n = pi2 - pi1;
            double norm = n.norm();
            n /= norm;
            Eigen::Vector2d perpL;
            perpL[0] = -n[1];
            perpL[1] = n[0];

            for(int s = 0; s < lineNhNumSamp; ++s){
                Eigen::Vector2d p = pi1 + n * norm * s / (lineNhNumSamp - 1);
                double slen = distr(gen);
                Eigen::Vector2d pl = p + perpL * slen;
                pl[0] = std::round(pl[0]);
                pl[1] = std::round(pl[1]);
                Eigen::Vector2d pr = p - perpL * slen;
                pr[0] = std::round(pr[0]);
                pr[1] = std::round(pr[1]);
                if(pl[0] >= 0 && pl[0] < rgb.cols && pl[1] >= 0 && pl[1] < rgb.rows){
                    planeVotesLeft[planesMasks.at<int>(pl[1], pl[0])]++;
                }
                if(pr[0] >= 0 && pr[0] < rgb.cols && pr[1] >= 0 && pr[1] < rgb.rows){
                    planeVotesRight[planesMasks.at<int>(pr[1], pr[0])]++;
                }
            }

        }

        cv::Mat imageDisp = rgb.clone();
        lsd->drawSegments(imageDisp, linesLsdThresh);
        cv::imshow("Detected segments", imageDisp);

        cv::waitKey();


    }
    else{
        cout << "Warning: rgb empty!" << endl;
    }
}
