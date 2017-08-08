//
// Created by jachu on 27.06.17.
//

#ifndef PLANELOC_LINEDET_HPP
#define PLANELOC_LINEDET_HPP

#include <opencv2/opencv.hpp>

#include "LineSeg.hpp"
#include "ObjInstance.hpp"

class LineDet {
public:
    static void detectLineSegments(const cv::FileStorage &settings,
                                      cv::Mat rgb,
                                      cv::Mat depth,
                                      const std::vector<ObjInstance> &planes,
                                      cv::Mat cameraMatrix,
                                      std::vector<LineSeg> &lineSegs,
                                      pcl::visualization::PCLVisualizer::Ptr viewer,
                                      int viewPort1,
                                      int viewPort2);
private:

};


#endif //PLANELOC_LINEDET_HPP
