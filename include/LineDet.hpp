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
                                   vectorObjInstance &planes,
                                   cv::Mat cameraMatrix,
                                   vectorLineSeg &lineSegs,
                                   pcl::visualization::PCLVisualizer::Ptr viewer,
                                   int viewPort1,
                                   int viewPort2);
private:

};


#endif //PLANELOC_LINEDET_HPP
