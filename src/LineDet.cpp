//
// Created by jachu on 27.06.17.
//

#include "LineDet.hpp"

using namespace std;

void LineDet::detectLineSegments(const cv::FileStorage &settings,
                                 cv::Mat image,
                                 std::vector<LineSeg> &lineSegs)
{
    double minImgLineLen = 50;

    if (!image.empty()) {
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
        vector<cv::Vec4f> linesLsd;

        cv::Mat imageGray;
        cv::cvtColor(image, imageGray, cv::COLOR_RGB2GRAY);
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
        cv::Mat imageDisp = image.clone();
        lsd->drawSegments(imageDisp, linesLsdThresh);
        cv::imshow("Detected segments", imageDisp);

        cv::waitKey();
    }
    else{
        cout << "Warning: image empty!" << endl;
    }
}
