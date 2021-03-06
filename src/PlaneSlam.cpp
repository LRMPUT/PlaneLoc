/*
    Copyright (c) 2017 Mobile Robots Laboratory at Poznan University of Technology:
    -Jan Wietrzykowski name.surname [at] put.poznan.pl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <g2o/types/slam3d/se3quat.h>
#include <LineSeg.hpp>
#include <LineDet.hpp>
#include <pcl/common/transforms.h>

#include "PlaneSlam.hpp"
#include "Misc.hpp"
#include "Matching.hpp"
#include "PlaneSegmentation.hpp"
#include "Serialization.hpp"
#include "ConcaveHull.hpp"

using namespace std;
using namespace cv;

PlaneSlam::PlaneSlam(const cv::FileStorage& isettings) :
	settings(isettings),
	fileGrabber(isettings["fileGrabber"]),
	map(isettings)
//	viewer("3D Viewer")
{

}


void PlaneSlam::run(){

    Mat cameraParams;
    settings["planeSlam"]["cameraMatrix"] >> cameraParams;
    cout << "cameraParams = " << cameraParams << endl;

    vector<double> gtOffsetVals;
    settings["planeSlam"]["gtOffset"] >> gtOffsetVals;
	Vector7d gtOffset;
    for(int v = 0; v < gtOffsetVals.size(); ++v){
        gtOffset(v) = gtOffsetVals[v];
    }

	g2o::SE3Quat gtOffsetSE3Quat(gtOffset);
    cout << "gtOffsetSE3Quat = " << gtOffsetSE3Quat.toVector().transpose() << endl;

	int frameCnt = 0;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	int v1 = 0;
	int v2 = 0;
    viewer->createViewPort(0.0, 0.0, 1.0, 1.0, v1);
//	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
//	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
//	viewer->addCoordinateSystem();
 
   
    // Map
	cout << "Getting object instances from map" << endl;
    vectorObjInstance mapObjInstances;
    for(auto it = map.begin(); it != map.end(); ++it){
		mapObjInstances.push_back(*it);
	}

	vectorObjInstance prevObjInstances;
	Vector7d prevPose;

	Mat rgb, depth;
	std::vector<FileGrabber::FrameObjInstance> objInstances;
	std::vector<double> accelData;
	Vector7d pose;
    Vector7d voPose;
    bool voCorr = false;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloudRead(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    
    int framesToSkip = (int)settings["planeSlam"]["framesToSkip"];
    cout << "skipping " << framesToSkip << " frames" << endl;
	static constexpr int frameRate = 30;
	int framesSkipped = 0;
    int curFrameIdx = -1;
	while((framesSkipped < framesToSkip) && ((curFrameIdx = fileGrabber.getFrame(rgb, depth, objInstances, accelData, pose, voPose, voCorr)) >= 0))
	{
        ++framesSkipped;
	}

    cout << "reading global settings" << endl;
	bool drawVis = bool((int)settings["planeSlam"]["drawVis"]);
    bool visualizeSegmentation = bool((int)settings["planeSlam"]["visualizeSegmentation"]);
    bool visualizeMatching = bool((int)settings["planeSlam"]["visualizeMatching"]);
    bool stopEveryFrame = bool((int)settings["planeSlam"]["stopEveryFrame"]);
    bool stopWrongFrame = bool((int)settings["planeSlam"]["stopWrongFrame"]);
	bool saveRes = bool((int)settings["planeSlam"]["saveRes"]);
	bool loadRes = bool((int)settings["planeSlam"]["loadRes"]);
	bool saveVis = bool((int)settings["planeSlam"]["saveVis"]);
	bool framesFromPly = bool((int)settings["planeSlam"]["framesFromPly"]);
    bool incrementalMatching = bool((int)settings["planeSlam"]["incrementalMatching"]);
    bool globalMatching = bool((int)settings["planeSlam"]["globalMatching"]);
    bool useLines = bool((int)settings["planeSlam"]["useLines"]);
    bool processFrames = bool((int)settings["planeSlam"]["processFrames"]);
//    bool localize = bool((int)settings["planeSlam"]["localize"]);
    bool compRes = true;

    double poseDiffThresh = (double)settings["planeSlam"]["poseDiffThresh"];

    double scoreThresh = (double)settings["planeSlam"]["scoreThresh"];
    double scoreDiffThresh = (double)settings["planeSlam"]["scoreDiffThresh"];
    double fitThresh = (double)settings["planeSlam"]["fitThresh"];
    double distinctThresh = (double)settings["planeSlam"]["distinctThresh"];

    int accFrames = (int)settings["planeSlam"]["accFrames"];

    vectorVector7d visGtPoses;
    vectorVector7d visRecPoses;
    vector<RecCode> visRecCodes;
    vector<int> visRecFrameIdxs;
    vectorVector7d visGtCompPoses;
    vectorVector7d visRecCompPoses;
    vector<RecCode> visRecCompCodes;

	ofstream outputResGlobFile;
    ofstream outputResIncrFile;
	if(saveRes){
        if(globalMatching) {
            outputResGlobFile.open("../output/res_glob.out");
        }
        if(incrementalMatching){
            outputResIncrFile.open("../output/res_incr.out");
        }
	}
//	ofstream visFile;
//	if(saveVis){
//		visFile.open("../output/vis");
//	}
    cv::VideoWriter visVideo;
    if(saveVis){
	   visVideo.open("../output/rec/vis.avi", VideoWriter::fourcc('X','2','6','4'), 2, Size(1280, 720));
	}
	ifstream inputResGlobFile;
    ifstream inputResIncrFile;
	if(loadRes){
        if(globalMatching) {
            inputResGlobFile.open("../output/res_glob.in");
        }
        if(incrementalMatching){
            inputResIncrFile.open("../output/res_incr.in");
        }
	}
	ifstream inputResGlobCompFile;
    if(compRes){
        inputResGlobCompFile.open("../output/res_comp.in");
    }
	
    // variables used for accumulation
//    vector<ObjInstance> accObjInstances;
    Map accMap;
    Vector7d accStartFramePose;
    
    int processNewFrameSkip = 1;
    
    
    int corrCnt = 0;
    int incorrCnt = 0;
    int unkCnt = 0;
    
    double meanDist = 0.0;
    double meanAngDist = 0.0;
    int meanCnt = 0;
    
    int prevCorrFrameIdx = curFrameIdx;
    int longestUnk = 0;
    
    
    int corrCntComp = 0;
    int incorrCntComp = 0;
    int unkCntComp = 0;
    int meanCntComp = 0;
    
    double meanDistComp = 0;
    double meanAngDistComp = 0;
    
    int prevCorrFrameIdxComp = curFrameIdx;
    int longestUnkComp = 0;
    
//	ofstream logFile("../output/log.out");
    cout << "Starting the loop" << endl;
	while((curFrameIdx = fileGrabber.getFrame(rgb, depth, objInstances, accelData, pose, voPose, voCorr, accMap)) >= 0) {
        cout << "curFrameIdx = " << curFrameIdx << endl;
        
        int64_t timestamp = (int64_t) curFrameIdx * 1e6 / frameRate;
        cout << "timestamp = " << timestamp << endl;
        
        bool stopFlag = stopEveryFrame;
        
        vectorObjInstance curObjInstances;
        
        std::map<int, int> idToCnt;
        
        bool localize = true;
        
        if(inputResGlobCompFile.is_open()){
            int codeVal;
            inputResGlobCompFile >> codeVal;
            
//            cout << "code = " << code << endl;
            
//            visGtCompPoses.push_back(pose);
            RecCode code = RecCode::Unk;
            Vector7d compTrans = Vector7d::Zero();
            if(codeVal != -1) {
//                Vector7d compTrans;
                for (int i = 0; i < 7; ++i) {
                    inputResGlobCompFile >> compTrans(i);
                }
            }
            
            if(curFrameIdx % 10 == 0) {
                visGtCompPoses.push_back(pose);
                
                if(codeVal == -1) {
                    visRecCompCodes.push_back(RecCode::Unk);
                    visRecCompPoses.push_back(Vector7d::Zero());
            
                    ++unkCntComp;
                }
                else {
                    
                    g2o::SE3Quat compTransSE3Quat(compTrans);
                    g2o::SE3Quat gtTransformSE3Quat(pose);

//                cout << "compTrans = " << compTrans.transpose() << endl;
//                cout << "pose = " << pose.transpose() << endl;
    
                    g2o::SE3Quat diffSE3Quat = compTransSE3Quat.inverse() * gtTransformSE3Quat;
//                    g2o::SE3Quat diffInvSE3Quat = poseSE3Quat * planesTransSE3Quat.inverse();
                    Vector6d diffLog = diffSE3Quat.log();
//                cout << "diffLog = " << diffSE3Quat.log().transpose() << endl;
//                    cout << "diffInvLog = " << diffInvSE3Quat.log().transpose() << endl;
                    double diff = diffLog.transpose() * diffLog;
    
                    double diffEucl = diffSE3Quat.toVector().head<3>().norm();
                    Eigen::Vector3d diffLogAng = Misc::logMap(diffSE3Quat.rotation());
                    double diffAng = diffLogAng.norm();
    
                    meanDistComp += diffEucl;
                    meanAngDistComp += diffAng;
                    ++meanCntComp;
    
                    if (diff > poseDiffThresh) {
                        code = RecCode::Incorr;
                        
                        ++incorrCntComp;
                    } else {
                        code = RecCode::Corr;
                        
                        int unkLenComp = curFrameIdx - prevCorrFrameIdxComp;
                        if(unkLenComp > longestUnkComp){
                            longestUnkComp = unkLenComp;
                        }
    
                        prevCorrFrameIdxComp = curFrameIdx;
                        
                        ++corrCntComp;
                    }
    
                    visRecCompCodes.push_back(code);
                    visRecCompPoses.push_back(compTrans);
                }
            }
        }
        
        if (processFrames) {
            if ((curFrameIdx - framesSkipped) % processNewFrameSkip == 0) {
                g2o::SE3Quat poseSE3Quat(pose);
                poseSE3Quat = gtOffsetSE3Quat.inverse() * poseSE3Quat;
                pose = poseSE3Quat.toVector();
        
                viewer->removeAllPointClouds();
                viewer->removeAllShapes();
        
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud;
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloudNormals;
                if (framesFromPly) {
                    pointCloudNormals.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>(*pointCloudRead));
                    pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
                    for (int p = 0; p < pointCloudNormals->size(); ++p) {
                        pcl::PointXYZRGB pt;
                        pt.x = pointCloudNormals->at(p).x;
                        pt.y = pointCloudNormals->at(p).y;
                        pt.z = pointCloudNormals->at(p).z;
                        pt.r = pointCloudNormals->at(p).r;
                        pt.g = pointCloudNormals->at(p).g;
                        pt.b = pointCloudNormals->at(p).b;
                        pointCloud->push_back(pt);
                    }
                } else {
                    pointCloudNormals.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>(rgb.cols,
                                                                                        rgb.rows));
                    pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>(rgb.cols, rgb.rows));
            
                    Mat xyz = Misc::projectTo3D(depth, cameraParams);
            
                    for (int row = 0; row < rgb.rows; ++row) {
                        for (int col = 0; col < rgb.cols; ++col) {
                            pcl::PointXYZRGB p;
//					((uint8_t)rgb.at<Vec3b>(row, col)[0],
//										(uint8_t)rgb.at<Vec3b>(row, col)[1],
//										(uint8_t)rgb.at<Vec3b>(row, col)[2]);
                            p.x = xyz.at<Vec3f>(row, col)[0];
                            p.y = xyz.at<Vec3f>(row, col)[1];
                            p.z = xyz.at<Vec3f>(row, col)[2];
                            p.r = (uint8_t) rgb.at<Vec3b>(row, col)[0];
                            p.g = (uint8_t) rgb.at<Vec3b>(row, col)[1];
                            p.b = (uint8_t) rgb.at<Vec3b>(row, col)[2];
                            //					cout << "Point at (" << xyz.at<Vec3f>(row, col)[0] << ", " <<
                            //											xyz.at<Vec3f>(row, col)[1] << ", " <<
                            //											xyz.at<Vec3f>(row, col)[2] << "), rgb = (" <<
                            //											(int)rgb.at<Vec3b>(row, col)[0] << ", " <<
                            //											(int)rgb.at<Vec3b>(row, col)[1] << ", " <<
                            //											(int)rgb.at<Vec3b>(row, col)[2] << ") " << endl;
                            pointCloud->at(col, row) = p;
                            pointCloudNormals->at(col, row).x = p.x;
                            pointCloudNormals->at(col, row).y = p.y;
                            pointCloudNormals->at(col, row).z = p.z;
                            pointCloudNormals->at(col, row).r = p.r;
                            pointCloudNormals->at(col, row).g = p.g;
                            pointCloudNormals->at(col, row).b = p.b;
                        }
                    }
            
                    // Create the normal estimation class, and pass the input dataset to it
                    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> ne;
                    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
                    ne.setMaxDepthChangeFactor(0.02f);
                    ne.setNormalSmoothingSize(20.0f);
                    ne.setInputCloud(pointCloud);
                    ne.setViewPoint(0.0, 0.0, 0.0);
            
                    ne.compute(*pointCloudNormals);
                    cout << "pointCloudNormals->size() = " << pointCloudNormals->size() << endl;
            
                }
                if (drawVis) {
//		    viewer->addPointCloud(pointCloud, "cloud", v1);

//            cout << endl << "whole cloud" << endl << endl;
//            viewer->resetStoppedFlag();
//            viewer->initCameraParameters();
//            viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, -1.0, 0.0);
//            viewer->spinOnce(100);
//            while (!viewer->wasStopped()) {
//                viewer->spinOnce(100);
//                std::this_thread::sleep_for(std::chrono::milliseconds(50));
//            }
//            viewer->close();
                }
        
                if (!pointCloud->empty()) {
                    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointCloudLab(new pcl::PointCloud<pcl::PointXYZRGBL>());
            
                    if (!visualizeSegmentation) {
//				PlaneSegmentation::segment(settings,
//									pointCloudNormals,
//									pointCloudLab,
//									curObjInstances,
//									false);
                
                        PlaneSegmentation::segment(settings,
                                                   rgb,
                                                   depth,
                                                   pointCloudLab,
                                                   curObjInstances);
                
                    }
                    else {
//                PlaneSegmentation::segment(settings,
//                                      pointCloudNormals,
//                                      pointCloudLab,
//                                      curObjInstances,
//                                      false,
//                                      viewer,
//                                      v1,
//                                      v2);
                
                        PlaneSegmentation::segment(settings,
                                                   rgb,
                                                   depth,
                                                   pointCloudLab,
                                                   curObjInstances,
                                                   viewer,
                                                   v1,
                                                   v2);
                    }
            
                    vectorLineSeg lineSegs;
            
                    if (useLines) {
                        viewer->addPointCloud(pointCloud, "cloud_raw", v2);
                
                        LineDet::detectLineSegments(settings,
                                                    rgb,
                                                    depth,
                                                    curObjInstances,
                                                    cameraParams,
                                                    lineSegs,
                                                    viewer,
                                                    v1,
                                                    v2);
                    }
            
                    // if not the last frame for accumulation
                    if ((curFrameIdx - framesSkipped) % accFrames != accFrames - 1) {
                        localize = false;
                    }

//                if (curFrameIdx % accFrames == accFrames - 1) {
//                    stopFlag = true;
//                }
            
                    // if current frame starts accumulation
                    if (((curFrameIdx - framesSkipped)/processNewFrameSkip) % accFrames == 0) {
                        cout << endl << "starting new accumulation" << endl << endl;
                
                        accMap = Map();
                        accStartFramePose = voPose;
//                        accStartFramePose = pose;
                    }
                    
                    if (voCorr) {
                        cout << endl << "merging curObjInstances" << endl << endl;

//                    g2o::SE3Quat accPoseIncrSE3Quat = g2o::SE3Quat(accStartFramePose).inverse() * g2o::SE3Quat(pose);
                        g2o::SE3Quat accPoseIncrSE3Quat =
                                g2o::SE3Quat(accStartFramePose).inverse() * g2o::SE3Quat(voPose);
//                    g2o::SE3Quat accPoseIncrSE3Quat = g2o::SE3Quat(pose);
                        Vector7d accPoseIncr = accPoseIncrSE3Quat.toVector();

                        cout << "accPoseIncr = " << accPoseIncr.transpose() << endl;
                
                        vectorObjInstance curObjInstancesTrans = curObjInstances;
                        for (ObjInstance &curObj : curObjInstancesTrans) {
                            curObj.transform(accPoseIncr);
                        }
                
                
                        cout << "Getting visible" << endl;
                        idToCnt = accMap.getVisibleObjs(accPoseIncr,
                                                        cameraParams,
                                                        rgb.rows,
                                                        rgb.cols/*,
                                                viewer,
                                                v1,
                                                v2*/);
                
                        cout << "Merging new" << endl;
                        if (curFrameIdx >= 4500) {
                            accMap.mergeNewObjInstances(curObjInstancesTrans,
                                                        idToCnt,
                                                        viewer,
                                                        v1,
                                                        v2);
                        } else {
                            accMap.mergeNewObjInstances(curObjInstancesTrans,
                                                        idToCnt);
                        }
                
                
                        // every 50th frame
                        int mergeMapFrameSkip = 50;
                        if (((curFrameIdx - framesSkipped)/processNewFrameSkip)
                                % mergeMapFrameSkip == mergeMapFrameSkip - 1)
                        {
                            cout << "Merging map" << endl;
                            accMap.mergeMapObjInstances(/*viewer,
                                                  v1,
                                                  v2*/);
                        }
                    }
            
                    // if last frame in accumulation
                    if (((curFrameIdx - framesSkipped)/processNewFrameSkip)
                                % accFrames == accFrames - 1)
                    {
                        accMap.removeObjsEolThresh(6);
                
                        // saving
                        {
                            cout << "saving map to file" << endl;
                    
                            char buf[100];
                            sprintf(buf,
                                    "../output/acc/acc%05d",
                                    curFrameIdx - accFrames*processNewFrameSkip + 1);
                    
                            std::ofstream ofs(buf);
                            boost::archive::text_oarchive oa(ofs);
                            oa << accMap;
                        }
                    }
    
                    prevObjInstances.swap(curObjInstances);
                }
            }
        }
        
        if(accMap.size() == 0){
            localize = false;
        }
        
        if (globalMatching && localize) {
            cout << "global matching" << endl;
            
            RecCode curRecCode;
//            g2o::SE3Quat gtTransSE3Quat =
//                    g2o::SE3Quat(prevPose).inverse() * g2o::SE3Quat(pose);
            Vector7d predTrans;
            double linDist, angDist;
            
            pcl::visualization::PCLVisualizer::Ptr curViewer = nullptr;
            int curViewPort1 = -1;
            int curViewPort2 = -1;
            if (visualizeMatching) {
                curViewer = viewer;
                curViewPort1 = v1;
                curViewPort2 = v2;
            }
            
            vectorObjInstance accObjInstances;
            for(const ObjInstance &curObj : accMap){
                accObjInstances.push_back(curObj);
            }
    
            evaluateMatching(settings,
                             accObjInstances,
                             mapObjInstances,
                             inputResGlobFile,
                             outputResGlobFile,
                             pose,
                             scoreThresh,
                             scoreDiffThresh,
                             fitThresh,
                             distinctThresh,
                             poseDiffThresh,
                             predTrans,
                             curRecCode,
                             linDist,
                             angDist,
                             curViewer,
                             curViewPort1,
                             curViewPort2);
            
            visRecCodes.push_back(curRecCode);
            visGtPoses.push_back(pose);
            visRecPoses.push_back(predTrans);
            visRecFrameIdxs.push_back(curFrameIdx);
            
//            cout << "pose = " << pose.transpose() << endl;
//            cout << "predTrans = " << predTrans.transpose() << endl;
            
            if (curRecCode == RecCode::Corr) {
                ++corrCnt;
                
                int unkLen = curFrameIdx - prevCorrFrameIdx;
                if(unkLen > longestUnk){
                    longestUnk = unkLen;
                }
                
                prevCorrFrameIdx = curFrameIdx;
            } else if (curRecCode == RecCode::Incorr) {
                ++incorrCnt;
                stopFlag |= stopWrongFrame;
            } else {
                ++unkCnt;
            }
            
            if (curRecCode != RecCode::Unk) {
                meanDist += linDist;
                meanAngDist += angDist;
                
                ++meanCnt;
            }
    
            if(visRecCodes.back() == RecCode::Corr && visRecCompCodes.back() == RecCode::Unk){
                cout << "Recognized where ORB-SLAM2 not recognized" << endl;
                g2o::SE3Quat planesTransSE3Quat;
                planesTransSE3Quat.fromVector(predTrans);
                g2o::SE3Quat gtTransformSE3Quat;
                gtTransformSE3Quat.fromVector(pose);
                
                g2o::SE3Quat diffSE3Quat = planesTransSE3Quat.inverse() * gtTransformSE3Quat;
//                    g2o::SE3Quat diffInvSE3Quat = poseSE3Quat * planesTransSE3Quat.inverse();
                Vector6d diffLog = diffSE3Quat.log();
//                    cout << "diffLog = " << diffSE3Quat.log().transpose() << endl;
//                    cout << "diffInvLog = " << diffInvSE3Quat.log().transpose() << endl;
                double diff = diffLog.transpose() * diffLog;
//                double diffEucl = diffSE3Quat.toVector().head<3>().norm();
//                Eigen::Vector3d diffLogAng = Misc::logMap(diffSE3Quat.rotation());
//                double diffAng = diffLogAng.norm();
                cout << "diff = " << diff << endl;
            }
        }
        
        if (incrementalMatching && !prevObjInstances.empty() && localize) {
            RecCode curRecCode;
            g2o::SE3Quat gtTransSE3Quat =
                    g2o::SE3Quat(prevPose).inverse() * g2o::SE3Quat(pose);
            Vector7d predTrans;
            double linDist, angDist;
            
            pcl::visualization::PCLVisualizer::Ptr curViewer = nullptr;
            int curViewPort1 = -1;
            int curViewPort2 = -1;
            if (visualizeMatching) {
                curViewer = viewer;
                curViewPort1 = v1;
                curViewPort2 = v2;
            }
    
            evaluateMatching(settings,
                             curObjInstances,
                             prevObjInstances,
                             inputResIncrFile,
                             outputResIncrFile,
                             gtTransSE3Quat.toVector(),
                             scoreThresh,
                             scoreDiffThresh,
                             fitThresh,
                             distinctThresh,
                             poseDiffThresh,
                             predTrans,
                             curRecCode,
                             linDist,
                             angDist,
                             curViewer,
                             curViewPort1,
                             curViewPort2);
            
            visRecCodes.push_back(curRecCode);
            visGtPoses.push_back(pose);
            visRecPoses.push_back(predTrans);
            
            if (curRecCode == RecCode::Corr) {
                ++corrCnt;
            } else if (curRecCode == RecCode::Incorr) {
                ++incorrCnt;
                stopFlag |= stopWrongFrame;
            } else {
                ++unkCnt;
            }
            
            if (curRecCode != RecCode::Unk) {
                meanDist += linDist;
                meanAngDist += angDist;
                ++meanCnt;
            }
        }
        
//        if(curFrameIdx == 850 || curFrameIdx == 890){
//            stopFlag = true;
//        }
        
        
        if (drawVis && accMap.size() > 0) {
            cout << "visualization" << endl;

//                // saving
//                {
//                    cout << "saving map to file" << endl;
//                    std::ofstream ofs("filename");
//                    boost::archive::text_oarchive oa(ofs);
//                    oa << accMap;
//                }
//                // loading
//                {
//                    accMap = Map();
//
//                    cout << "loading map from file" << endl;
//                    std::ifstream ifs("filename");
//                    boost::archive::text_iarchive ia(ifs);
//                    ia >> accMap;
//                }
            
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();
            viewer->removeAllCoordinateSystems();
            
//            viewer->addCoordinateSystem();
    
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPc(new pcl::PointCloud<pcl::PointXYZRGB>());
            for(const ObjInstance &mObj : map){
                mapPc->insert(mapPc->end(), mObj.getPoints()->begin(), mObj.getPoints()->end());
            }
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPcGray(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::copyPointCloud(*mapPc, *mapPcGray);
            for(int p = 0; p < mapPcGray->size(); ++p){
                int gray = mapPcGray->at(p).r * 0.21 +
                           mapPcGray->at(p).g * 0.72 +
                           mapPcGray->at(p).b * 0.07;
                gray = min(gray, 255);
                mapPcGray->at(p).r = gray;
                mapPcGray->at(p).g = gray;
                mapPcGray->at(p).b = gray;
            }
            viewer->addPointCloud(mapPcGray, "map_cloud", v1);

            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     0.5,
                                                     "map_cloud",
                                                     v1);

        pcl::PointCloud<pcl::PointXYZ>::Ptr trajLine(new pcl::PointCloud<pcl::PointXYZ>());
            for(int f = 1; f < visGtPoses.size(); ++f){
                pcl::PointXYZ prevPose(visGtPoses[f-1][0],
                                       visGtPoses[f-1][1],
                                       visGtPoses[f-1][2]);
                pcl::PointXYZ curPose(visGtPoses[f][0],
                                      visGtPoses[f][1],
                                      visGtPoses[f][2]);
//            trajLine->push_back(curPose);
                viewer->addLine(prevPose, curPose, 1.0, 0.0, 0.0, string("line_traj_") + to_string(f), v1);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                    4,
                                                    string("line_traj_") + to_string(f),
                                                    v1);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                    pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                    string("line_traj_") + to_string(f),
                                                    v1);
            }
        viewer->addPolygon<pcl::PointXYZ>(trajLine, 0.0, 1.0, 0.0, "traj_poly", v1);
    
            pcl::PointCloud<pcl::PointXYZ>::Ptr corrPoses(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr incorrPoses(new pcl::PointCloud<pcl::PointXYZ>());
    
            for(int f = 0; f < visRecCodes.size(); ++f){
                if(visRecCodes[f] == RecCode::Corr || visRecCodes[f] == RecCode::Incorr){
                    pcl::PointXYZ curPose(visGtPoses[f][0],
                                          visGtPoses[f][1],
                                          visGtPoses[f][2]);
                    pcl::PointXYZ curRecPose(visRecPoses[f][0],
                                             visRecPoses[f][1],
                                             visRecPoses[f][2]);
                    if(visRecCodes[f] == RecCode::Corr) {
                        corrPoses->push_back(curRecPose);
//                        cout << "correct for " << visRecFrameIdxs[f] << endl;
                    }
                    else/* if(visRecCodes[f] == RecCode::Incorr) */ {
                        incorrPoses->push_back(curRecPose);
//                        cout << "incorrect for " << visRecFrameIdxs[f] << endl;
                    }
            
            
                    viewer->addLine(curPose, curRecPose, 0.0, 1.0, 0.0, string("line_pose_") + to_string(f), v1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                        4,
                                                        string("line_pose_") + to_string(f),
                                                        v1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                        pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                        string("line_pose_") + to_string(f),
                                                        v1);
                }
            }
            viewer->addPointCloud(corrPoses, "corr_poses_cloud", v1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     5,
                                                     "corr_poses_cloud",
                                                     v1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                     0.0, 0.0, 1.0,
                                                     "corr_poses_cloud",
                                                     v1);

            viewer->addPointCloud(incorrPoses, "incorr_poses_cloud", v1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     5,
                                                     "incorr_poses_cloud",
                                                     v1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                     1.0, 0.0, 1.0,
                                                     "incorr_poses_cloud",
                                                     v1);
            
            {
                g2o::SE3Quat accPoseIncrSE3Quat =
                        g2o::SE3Quat(accStartFramePose).inverse() *
                        g2o::SE3Quat(voPose);
                
                Eigen::Affine3f trans = Eigen::Affine3f::Identity();
//                        trans.matrix() = poseSE3Quat.to_homogeneous_matrix().cast<float>();
                trans.matrix() = accPoseIncrSE3Quat.to_homogeneous_matrix().cast<float>();
//                viewer->addCoordinateSystem(0.5, trans, "camera_coord");
            }
            int o = 0;
            for (auto it = accMap.begin(); it != accMap.end(); ++it, ++o) {
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = it->getPoints();
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPcTrans(new pcl::PointCloud<pcl::PointXYZRGB>());
                g2o::SE3Quat poseSE3Quat;
                poseSE3Quat.fromVector(pose);
                
                pcl::transformPointCloud(*curPc, *curPcTrans, poseSE3Quat.to_homogeneous_matrix());
    
                const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr curPcTransLab(new pcl::PointCloud<pcl::PointXYZRGBL>());
                pcl::copyPointCloud(*curPcTrans, *curPcTransLab);
                for(auto itPts = curPcTransLab->begin(); itPts != curPcTransLab->end(); ++itPts){
                    itPts->label = it->getId();
                }
//                pcl::visualization::PointCloudColorHandlerLabelField<pcl::PointXYZRGBL>::Ptr
//                        colorHandler(new pcl::visualization::PointCloudColorHandlerLabelField<pcl::PointXYZRGBL>(curPcTransLab));
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBL>::Ptr
                        colorHandler(new pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBL>(curPcTransLab));
                viewer->addPointCloud<pcl::PointXYZRGBL>(curPcTransLab, *colorHandler, "cloud_lab_" + to_string(o), v1);
            }
            
            
            viewer->resetStoppedFlag();
            static bool cameraInit = false;
            if (!cameraInit) {
                viewer->initCameraParameters();
                viewer->setSize(1280, 720);
//                viewer->setSize(640, 480);
//                viewer->setCameraPosition(0.0, 0.0, -4.0, 0.0, -1.0, 0.0);
                cameraInit = true;
            }
            {
                g2o::SE3Quat poseSE3Quat;
//                if(curFrameIdx == 850){
//                    Vector7d endPose;
//                    endPose << -1.4574, 0.618103, -2.37465, 0.851687, 0.0842968, -0.497145, 0.142725;
//                    poseSE3Quat.fromVector(endPose);
//                }
//                else if(curFrameIdx == 890){
//                    Vector7d endPose;
//                    endPose << -1.59232, 0.632686, -2.83519, 0.794182, 0.0810181, -0.592532, 0.107778;
//                    poseSE3Quat.fromVector(endPose);
//                }
//                else {
//                    poseSE3Quat.fromVector(pose);
//                }
                poseSE3Quat.fromVector(pose);
    
                Eigen::Vector3d dtFocal;
                dtFocal << 0.0, 0.0, 0.0;
//                dtFocal << 0.0, 0.0, 1.0;
                Eigen::Vector3d t = poseSE3Quat.translation() + poseSE3Quat.rotation().toRotationMatrix() * dtFocal;
                Eigen::Vector3d dtPose;
                dtPose << 0.0, -2.0, -4.0;
//                dtPose << 0.0, 0.0, 0.0;
                Eigen::Vector3d tc = poseSE3Quat.translation() + poseSE3Quat.rotation().toRotationMatrix() * dtPose;
                viewer->setCameraPosition(tc(0), tc(1), tc(2), t(0), t(1), t(2), 0.0, 1.0, 0.0);
                viewer->setCameraClipDistances(0.2, 20.0);
            }
            viewer->spinOnce(100);
            while (stopFlag && !viewer->wasStopped()) {
                viewer->spinOnce(100);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if(saveVis){
                char buf[100];
                sprintf(buf, "../output/rec/vis%04d.png", curFrameIdx);
                viewer->saveScreenshot(buf);
                cv::Mat image = cv::imread(buf);
                visVideo << image;
            }
            viewer->close();
        }
        
        ++frameCnt;
        prevPose = pose;
        
        cout << "end frame" << endl;
	}

	cout << "corrCnt = " << corrCnt << endl;
	cout << "incorrCnt = " << incorrCnt << endl;
	cout << "unkCnt = " << unkCnt << endl;
    if(meanCnt > 0){
        cout << "meanDist = " << meanDist / meanCnt << " m " << endl;
        cout << "meanAngDist = " << meanAngDist * 180.0 / pi / meanCnt << " deg" << endl;
    }
    {
        int unkLen = fileGrabber.getNumFrames() - prevCorrFrameIdx;
        if(unkLen > longestUnk){
            longestUnk = unkLen;
        }
    }
    cout << "longestUnk = " << longestUnk << endl;
    
    cout << "corrCntComp = " << corrCntComp << endl;
    cout << "incorrCnt = " << incorrCntComp << endl;
    cout << "unkCntComp = " << unkCntComp << endl;
    if(meanCnt > 0){
        cout << "meanDistComp = " << meanDistComp / meanCntComp << " m " << endl;
        cout << "meanAngDistComp = " << meanAngDistComp * 180.0 / pi / meanCntComp << " deg" << endl;
    }
    {
        int unkLenComp = fileGrabber.getNumFrames() - prevCorrFrameIdxComp;
        if(unkLenComp > longestUnkComp){
            longestUnkComp = unkLenComp;
        }
    }
    cout << "longestUnkComp = " << longestUnkComp << endl;
    
    if(drawVis){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        viewer->removeAllCoordinateSystems();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPc(new pcl::PointCloud<pcl::PointXYZRGB>());
        for(const ObjInstance &mObj : map){
            mapPc->insert(mapPc->end(), mObj.getPoints()->begin(), mObj.getPoints()->end());
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPcGray(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(*mapPc, *mapPcGray);
        for(int p = 0; p < mapPcGray->size(); ++p){
            int gray = mapPcGray->at(p).r * 0.21 +
                        mapPcGray->at(p).g * 0.72 +
                        mapPcGray->at(p).b * 0.07;
            gray = min(gray, 255);
            mapPcGray->at(p).r = gray;
            mapPcGray->at(p).g = gray;
            mapPcGray->at(p).b = gray;
        }
        viewer->addPointCloud(mapPcGray, "map_cloud", v1);

//        pcl::PointCloud<pcl::PointXYZ>::Ptr trajLine(new pcl::PointCloud<pcl::PointXYZ>());
        for(int f = 1; f < visGtPoses.size(); ++f){
            pcl::PointXYZ prevPose(visGtPoses[f-1][0],
                                  visGtPoses[f-1][1],
                                  visGtPoses[f-1][2]);
            pcl::PointXYZ curPose(visGtPoses[f][0],
                                  visGtPoses[f][1],
                                  visGtPoses[f][2]);
//            trajLine->push_back(curPose);
            viewer->addLine(prevPose, curPose, 1.0, 0.0, 0.0, string("line_traj_") + to_string(f), v1);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                4,
                                                string("line_traj_") + to_string(f),
                                                v1);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                string("line_traj_") + to_string(f),
                                                v1);
        }
//        viewer->addPolygon<pcl::PointXYZ>(trajLine, 0.0, 1.0, 0.0, "traj_poly", v1);

        pcl::PointCloud<pcl::PointXYZ>::Ptr corrPoses(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr incorrPoses(new pcl::PointCloud<pcl::PointXYZ>());
        
        for(int f = 0; f < visRecCodes.size(); ++f){
            if(visRecCodes[f] == RecCode::Corr || visRecCodes[f] == RecCode::Incorr){
                pcl::PointXYZ curPose(visGtPoses[f][0],
                                      visGtPoses[f][1],
                                      visGtPoses[f][2]);
                pcl::PointXYZ curRecPose(visRecPoses[f][0],
                                      visRecPoses[f][1],
                                      visRecPoses[f][2]);
                if(visRecCodes[f] == RecCode::Corr) {
                    corrPoses->push_back(curRecPose);
//                    cout << "correct for " << visRecFrameIdxs[f] << endl;
                }
                else/* if(visRecCodes[f] == RecCode::Incorr) */ {
                    incorrPoses->push_back(curRecPose);
//                    cout << "incorrect for " << visRecFrameIdxs[f] << endl;
                }
                

                viewer->addLine(curPose, curRecPose, 0.0, 1.0, 0.0, string("line_pose_") + to_string(f), v1);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                    4,
                                                    string("line_pose_") + to_string(f),
                                                    v1);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                    pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                    string("line_pose_") + to_string(f),
                                                    v1);
            }
        }
        viewer->addPointCloud(corrPoses, "corr_poses_cloud", v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 5,
                                                 "corr_poses_cloud",
                                                 v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 0.0, 0.0, 1.0,
                                                 "corr_poses_cloud",
                                                 v1);

        viewer->addPointCloud(incorrPoses, "incorr_poses_cloud", v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 5,
                                                 "incorr_poses_cloud",
                                                 v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 1.0, 0.0, 1.0,
                                                 "incorr_poses_cloud",
                                                 v1);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr corrCompPoses(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr incorrCompPoses(new pcl::PointCloud<pcl::PointXYZ>());
        for(int f = 0; f < visRecCompCodes.size(); ++f){
            /*if(f % 10 == 0)*/ {
                if (visRecCompCodes[f] == RecCode::Corr || visRecCompCodes[f] == RecCode::Incorr) {
                    pcl::PointXYZ curPose(visGtCompPoses[f][0],
                                          visGtCompPoses[f][1],
                                          visGtCompPoses[f][2]);
                    pcl::PointXYZ curRecPose(visRecCompPoses[f][0],
                                             visRecCompPoses[f][1],
                                             visRecCompPoses[f][2]);
                    if (visRecCompCodes[f] == RecCode::Corr) {
                        corrCompPoses->push_back(curRecPose);
                    } else/* if(visRecCodes[f] == RecCode::Incorr) */ {
                        incorrCompPoses->push_back(curRecPose);
                    }
                    
        
                    viewer->addLine(curPose,
                                    curRecPose,
                                    0.0,
                                    1.0,
                                    1.0,
                                    string("line_comp_pose_") + to_string(f),
                                    v1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                                        4,
                                                        string("line_comp_pose_") + to_string(f),
                                                        v1);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                                        pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
                                                        string("line_comp_pose_") + to_string(f),
                                                        v1);
                }
            }
        }
        
        viewer->addPointCloud(corrCompPoses, "corr_comp_poses_cloud", v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 5,
                                                 "corr_comp_poses_cloud",
                                                 v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 1.0, 1.0, 0.0,
                                                 "corr_comp_poses_cloud",
                                                 v1);
    
        viewer->addPointCloud(incorrCompPoses, "incorr_comp_poses_cloud", v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 5,
                                                 "incorr_comp_poses_cloud",
                                                 v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 1.0, 0.0, 1.0,
                                                 "incorr_comp_poses_cloud",
                                                 v1);

        viewer->resetStoppedFlag();
        viewer->initCameraParameters();
        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
        viewer->spinOnce(100);
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        viewer->close();

    }

	viewer->close();
}

void PlaneSlam::evaluateMatching(const cv::FileStorage &fs,
                                 const vectorObjInstance &objInstances1,
                                 const vectorObjInstance &objInstances2,
                                 std::ifstream &inputResFile,
                                 std::ofstream &outputResFile,
                                 const Vector7d &gtTransform,
                                 double scoreThresh,
                                 double scoreDiffThresh,
                                 double fitThresh,
                                 double distinctThresh,
                                 double poseDiffThresh,
                                 Vector7d &predTransform,
                                 RecCode &recCode,
                                 double &linDist,
                                 double &angDist,
                                 pcl::visualization::PCLVisualizer::Ptr viewer,
                                 int viewPort1,
                                 int viewPort2)
{
    
    vectorVector7d planesTrans;
    vector<double> planesTransScores;
    vector<double> planesTransFits;
    vector<int> planesTransDistinct;
    vector<Matching::ValidTransform> transforms;
    Matching::MatchType matchType = Matching::MatchType::Unknown;
    
    if(inputResFile.is_open()){
        Vector7d curPose;
        for(int c = 0; c < 7; ++c){
            inputResFile >> curPose(c);
        }
        int matchTypeId;
        inputResFile >> matchTypeId;
        if(matchTypeId == 0){
            matchType = Matching::MatchType::Ok;
            cout << "matchType = Matching::MatchType::Ok;" << endl;
            int numTrans;
            inputResFile >> numTrans;
            for(int t = 0; t < numTrans; ++t){
                Vector7d curTrans;
                double curScore;
                double curFit;
                int curDistinct;
                double curDiff;
                for(int c = 0; c < 7; ++c){
                    inputResFile >> curTrans(c);
                }
                inputResFile >> curScore >> curFit >> curDistinct >> curDiff;
                planesTrans.push_back(curTrans);
                planesTransScores.push_back(curScore);
                planesTransFits.push_back(curFit);
                planesTransDistinct.push_back(curDistinct);
                // diff is recalculated later
            }
        }
        else if(matchTypeId == -1){
            matchType = Matching::MatchType::Unknown;
            cout << "matchType = Matching::MatchType::Unknown;" << endl;
        }
        cout << "results read" << endl;
    }
    else {
        matchType = Matching::matchFrameToMap(settings,
                                              objInstances1,
                                              objInstances2,
                                              planesTrans,
                                              planesTransScores,
                                              planesTransFits,
                                              planesTransDistinct,
                                              transforms,
                                              viewer,
                                              viewPort1,
                                              viewPort2);
    }
    
    g2o::SE3Quat gtTransformSE3Quat(gtTransform);
    bool isUnamb = true;
    if( matchType == Matching::MatchType::Ok){
        if(planesTransScores.front() < scoreThresh){
            isUnamb = false;
        }
        if(planesTransScores.size() > 1){
            if(fabs(planesTransScores[0] - planesTransScores[1]) < scoreDiffThresh){
                isUnamb = false;
            }
        }
        if(planesTransFits.front() > fitThresh){
            isUnamb = false;
        }
        if(planesTransDistinct.front() < distinctThresh){
            isUnamb = false;
        }
    }
    if(planesTrans.size() > 0){
//					stopFlag = true;
    }
    cout << "planesTrans.size() = " << planesTrans.size() << endl;
    vector<double> planesTransDiff;
    vector<double> planesTransDiffEucl;
    vector<double> planesTransDiffAng;
    for(int t = 0; t < planesTrans.size(); ++t){
        g2o::SE3Quat planesTransSE3Quat(planesTrans[t]);
        
        //				cout << "frame diff SE3Quat = " << (planesTransSE3Quat.inverse() * poseSE3Quat).toVector().transpose() << endl;
        cout << "planesTrans[" << t << "] = " << planesTrans[t].transpose() << endl;
        cout << "planesTransScores[" << t << "] = " << planesTransScores[t] << endl;
        cout << "planesTransFits[" << t << "] = " << planesTransFits[t] << endl;
        cout << "planesTransDistinct[" << t << "] = " << planesTransDistinct[t] << endl;
        if(std::isnan(planesTransScores[t])){
            planesTransScores[t] = 0.0;
        }
        //			cout << "pose = " << pose.transpose() << endl;
        //			cout << "planesTrans = " << planesTrans.transpose() << endl;
        
        //				{
        //					viewer->removeCoordinateSystem("trans coord", v2);
        //					Eigen::Affine3f trans = Eigen::Affine3f::Identity();
        //					trans.matrix() = planesTransSE3Quat.inverse().to_homogeneous_matrix().cast<float>();
        //					//		trans.fromPositionOrientationScale(, rot, 1.0);
        //					viewer->addCoordinateSystem(1.0, trans, "trans coord", v2);
        //				}
        
        g2o::SE3Quat diffSE3Quat = planesTransSE3Quat.inverse() * gtTransformSE3Quat;
//                    g2o::SE3Quat diffInvSE3Quat = poseSE3Quat * planesTransSE3Quat.inverse();
        Vector6d diffLog = diffSE3Quat.log();
//                    cout << "diffLog = " << diffSE3Quat.log().transpose() << endl;
//                    cout << "diffInvLog = " << diffInvSE3Quat.log().transpose() << endl;
        double diff = diffLog.transpose() * diffLog;
        double diffEucl = diffSE3Quat.toVector().head<3>().norm();
        Eigen::Vector3d diffLogAng = Misc::logMap(diffSE3Quat.rotation());
        double diffAng = diffLogAng.norm();
//                    Eigen::Vector3d diffAngEuler = diffInvSE3Quat.rotation().toRotationMatrix().eulerAngles(1, 0, 2);
//                    cout << "diffAngEuler = " << diffAngEuler.transpose() << endl;
//                    double diffAng = std::min(diffAngEuler[0], pi - diffAngEuler[0]);
        planesTransDiff.push_back(diff);
        planesTransDiffEucl.push_back(diffEucl);
        cout << "planesTransDiffEucl[" << t << "] = " << planesTransDiffEucl[t] << endl;
        planesTransDiffAng.push_back(diffAng);
        cout << "planesTransDiffAng[" << t << "] = " << planesTransDiffAng[t] << endl;
        cout << "planesTransDiff[" << t << "] = " << planesTransDiff[t] << endl;
    }
    
    if( matchType == Matching::MatchType::Ok && isUnamb){
        if(planesTransDiff.front() > poseDiffThresh){
            recCode = RecCode::Incorr;
        }
        else{
            recCode = RecCode::Corr;
            
            // computing mean and standard deviation
            static float refNumTransforms = 0;
            static float expNumTransforms = 0;
            static float expNumTransformsSq = 0;
            static int numTransformsCnt = 0;
            if(numTransformsCnt == 0){
                refNumTransforms = transforms.size();
            }
            expNumTransforms += transforms.size() - refNumTransforms;
            expNumTransformsSq += (transforms.size() - refNumTransforms) * (transforms.size() - refNumTransforms);
            ++numTransformsCnt;
            cout << "refNumTransforms = " << refNumTransforms << endl;
            cout << "expNumTransforms = " << expNumTransforms << endl;
            cout << "expNumTransformsSq = " << expNumTransformsSq << endl;
            cout << "numTransformsCnt = " << numTransformsCnt << endl;
            cout << "mean number of transforms: " << refNumTransforms +
                                                        expNumTransforms/numTransformsCnt << endl;
            if(numTransformsCnt > 1) {
                cout << "std dev = " << sqrt((expNumTransformsSq -
                                              expNumTransforms * expNumTransforms /
                                              numTransformsCnt) /
                                             (numTransformsCnt - 1)) << endl;
            }
        }
        
        predTransform = planesTrans.front();
        
        linDist = planesTransDiffEucl.front();
        angDist = planesTransDiffAng.front();
    }
    else{
        recCode = RecCode::Unk;
    }
    
    if(outputResFile.is_open()){
        // saving results file
        outputResFile << gtTransform.transpose() << endl;
        if( matchType == Matching::MatchType::Ok){
            outputResFile << 0 << endl;
            outputResFile << planesTrans.size() << endl;
            for(int t = 0; t < planesTrans.size(); ++t){
                outputResFile << planesTrans[t].transpose() <<
                                  " " << planesTransScores[t] <<
                                  " " << planesTransFits[t] <<
                                  " " << planesTransDistinct[t] <<
                                  " " << planesTransDiff[t] << endl;
            }
        }
        else if(matchType == Matching::MatchType::Unknown){
            outputResFile << -1 << endl;
        }
        outputResFile << endl << endl;
    }
}

