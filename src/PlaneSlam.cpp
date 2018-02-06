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

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>

#include <g2o/types/slam3d/se3quat.h>
#include <LineSeg.hpp>
#include <LineDet.hpp>

#include "PlaneSlam.hpp"
#include "Misc.hpp"
#include "Matching.hpp"
#include "PlaneSegmentation.hpp"

using namespace std;
using namespace cv;

PlaneSlam::PlaneSlam(const cv::FileStorage& isettings) :
	settings(isettings),
	fileGrabber(isettings),
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
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
//	viewer->addCoordinateSystem();

	int corrCnt = 0;
	int incorrCnt = 0;
	int unkCnt = 0;

    double meanDist = 0.0;
    double meanAngDist = 0.0;
    int meanCnt = 0;

	// Map
	cout << "Getting object instances from map" << endl;
	vector<ObjInstance> mapObjInstances;
	for(int i = 0; i < map.size(); ++i){
		mapObjInstances.push_back(map[i]);
	}

	vector<ObjInstance> prevObjInstances;
	Vector7d prevPose;

	Mat rgb, depth;
	std::vector<FileGrabber::FrameObjInstance> objInstances;
	std::vector<double> accelData;
	Vector7d pose;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloudRead(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

	static constexpr int frameRate = 30;
	int framesToSkip = 150;
	int framesSkipped = 0;
	while((framesSkipped < framesToSkip) && (fileGrabber.getFrame(rgb, depth, objInstances, accelData, pose) >= 0))
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

    double poseDiffThresh = (double)settings["planeSlam"]["poseDiffThresh"];

    double scoreThresh = (double)settings["planeSlam"]["scoreThresh"];
    double scoreDiffThresh = (double)settings["planeSlam"]["scoreDiffThresh"];
    double fitThresh = (double)settings["planeSlam"]["fitThresh"];

    vector<Vector7d> visGtPoses;
    vector<Vector7d> visRecPoses;
    vector<RecCode> visRecCodes;

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
    
    // variables used for accumulation
    vector<ObjInstance> accObjInstances;
    Vector7d accStartFramePose;
    int accFrames = 50;
 
//	ofstream logFile("../output/log.out");
	int curFrameIdx;
    cout << "Starting the loop" << endl;
	while((curFrameIdx = fileGrabber.getFrame(rgb, depth, objInstances, accelData, pose, pointCloudRead)) >= 0){
		cout << "curFrameIdx = " << curFrameIdx << endl;

		int64_t timestamp = (int64_t)curFrameIdx * 1e6 / frameRate;
		cout << "timestamp = " << timestamp << endl;

		g2o::SE3Quat poseSE3Quat(pose);
		poseSE3Quat = gtOffsetSE3Quat.inverse() * poseSE3Quat;
		pose = poseSE3Quat.toVector();

		viewer->removeAllPointClouds();
		viewer->removeAllShapes();
        
        bool localize = true;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloudNormals;
		if(framesFromPly){
			pointCloudNormals.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>(*pointCloudRead));
            pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
            for(int p = 0; p < pointCloudNormals->size(); ++p){
                pcl::PointXYZRGB pt;
                pt.x = pointCloudNormals->at(p).x;
                pt.y = pointCloudNormals->at(p).y;
                pt.z = pointCloudNormals->at(p).z;
                pt.r = pointCloudNormals->at(p).r;
                pt.g = pointCloudNormals->at(p).g;
                pt.b = pointCloudNormals->at(p).b;
                pointCloud->push_back(pt);
            }
		}
		else{
            pointCloudNormals.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>(rgb.cols, rgb.rows));
			pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>(rgb.cols, rgb.rows));

			Mat xyz = Misc::projectTo3D(depth, cameraParams);

			for(int row = 0; row < rgb.rows; ++row){
				for(int col = 0; col < rgb.cols; ++col){
					pcl::PointXYZRGB p;
//					((uint8_t)rgb.at<Vec3b>(row, col)[0],
//										(uint8_t)rgb.at<Vec3b>(row, col)[1],
//										(uint8_t)rgb.at<Vec3b>(row, col)[2]);
					p.x = xyz.at<Vec3f>(row, col)[0];
					p.y = xyz.at<Vec3f>(row, col)[1];
					p.z = xyz.at<Vec3f>(row, col)[2];
					p.r = (uint8_t)rgb.at<Vec3b>(row, col)[0];
					p.g = (uint8_t)rgb.at<Vec3b>(row, col)[1];
					p.b = (uint8_t)rgb.at<Vec3b>(row, col)[2];
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
        if(drawVis) {
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

		if(!pointCloud->empty()){
			pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointCloudLab(new pcl::PointCloud<pcl::PointXYZRGBL>());
			vector<ObjInstance> curObjInstances;

			if(!loadRes && !visualizeSegmentation){
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
            else if(visualizeSegmentation){
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

            vector<LineSeg> lineSegs;
            
            bool stopFlag = stopEveryFrame;
            
            if(useLines){
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
            if(curFrameIdx % accFrames != accFrames - 1){
                localize = false;
            }
            
            if(curFrameIdx % accFrames == accFrames - 2){
                stopFlag = true;
            }
            
            // if current frame starts accumulation
            if(curFrameIdx % accFrames == 0) {
                cout << endl << "starting new accumulation" << endl << endl;
                
                accObjInstances = curObjInstances;
                accStartFramePose = pose;
            }
            else{
                cout << endl << "merging curObjInstances" << endl << endl;
                
                g2o::SE3Quat accPoseIncrSE3Quat = g2o::SE3Quat(accStartFramePose).inverse() * g2o::SE3Quat(pose);
                Vector7d accPoseIncr = accPoseIncrSE3Quat.toVector();
                for(ObjInstance &curObj : curObjInstances){
                    accObjInstances.push_back(curObj);
                    accObjInstances.back().transform(accPoseIncr);
                }
    
                vector<vector<ObjInstance>> toMerge{accObjInstances};
                if(curFrameIdx < 162) {
                    accObjInstances = ObjInstance::mergeObjInstances(toMerge/*,
                                                                 viewer,
                                                                 v1,
                                                                 v2*/);
                }
                else{
                    accObjInstances = ObjInstance::mergeObjInstances(toMerge,
                                                                 viewer,
                                                                 v1,
                                                                 v2);
                }
            }
            

            if(globalMatching && localize){
                RecCode curRecCode;
                g2o::SE3Quat gtTransSE3Quat = g2o::SE3Quat(prevPose).inverse() * g2o::SE3Quat(pose);
                Vector7d predTrans;
                double linDist, angDist;
    
                pcl::visualization::PCLVisualizer::Ptr curViewer = nullptr;
                int curViewPort1 = -1;
                int curViewPort2 = -1;
                if(visualizeMatching){
                    curViewer = viewer;
                    curViewPort1 = v1;
                    curViewPort2 = v2;
                }
    
                evaluateMatching(settings,
                                 curObjInstances,
                                 mapObjInstances,
                                 inputResIncrFile,
                                 outputResIncrFile,
                                 pose,
                                 scoreThresh,
                                 scoreDiffThresh,
                                 fitThresh,
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
                
                if(curRecCode == RecCode::Corr) {
                    ++corrCnt;
                }
                else if(curRecCode == RecCode::Incorr) {
                    ++incorrCnt;
                }
                else {
                    ++unkCnt;
                }
                
                if(curRecCode != RecCode::Unk) {
                    meanDist += linDist;
                    meanAngDist += angDist;
                    ++meanCnt;
                }
            }
            
            if(incrementalMatching && !prevObjInstances.empty() && localize){
                RecCode curRecCode;
                g2o::SE3Quat gtTransSE3Quat = g2o::SE3Quat(prevPose).inverse() * g2o::SE3Quat(pose);
                Vector7d predTrans;
                double linDist, angDist;
    
                pcl::visualization::PCLVisualizer::Ptr curViewer = nullptr;
                int curViewPort1 = -1;
                int curViewPort2 = -1;
                if(visualizeMatching){
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
    
                if(curRecCode == RecCode::Corr) {
                    ++corrCnt;
                }
                else if(curRecCode == RecCode::Incorr) {
                    ++incorrCnt;
                }
                else {
                    ++unkCnt;
                }
    
                if(curRecCode != RecCode::Unk) {
                    meanDist += linDist;
                    meanAngDist += angDist;
                    ++meanCnt;
                }
            }
            


            if(drawVis) {
                cout << "visualization" << endl;
    
                viewer->removeAllPointClouds();
                viewer->removeAllShapes();
                for(int o = 0; o < accObjInstances.size(); ++o){
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = accObjInstances[o].getPoints();
                    viewer->addPointCloud(curPc, "cloud_" + to_string(o), v1);
    
                    int colIdx = (o % (sizeof(colors)/sizeof(uint8_t)/3));
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> colHan(curPc,
                                                                                              colors[colIdx][0],
                                                                                              colors[colIdx][1],
                                                                                              colors[colIdx][2]);
                    viewer->addPointCloud(curPc, colHan, "cloud_col_" + to_string(o), v2);
                }
                
                viewer->resetStoppedFlag();
                viewer->initCameraParameters();
                viewer->setCameraPosition(0.0, 0.0, -4.0, 0.0, -1.0, 0.0);
                viewer->spinOnce(100);
                while (stopFlag && !viewer->wasStopped()) {
                    viewer->spinOnce(100);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                viewer->close();
            }
            
			prevObjInstances.swap(curObjInstances);
		}

		++frameCnt;
		prevPose = pose;

		cout << "end processing frame" << endl;
	}

	cout << "corrCnt = " << corrCnt << endl;
	cout << "incorrCnt = " << incorrCnt << endl;
	cout << "unkCnt = " << unkCnt << endl;
    if(meanCnt > 0){
        cout << "meanDist = " << meanDist / meanCnt << " m " << endl;
        cout << "meanAngDist = " << meanAngDist * 180.0 / pi / meanCnt << " deg" << endl;
    }

    if(drawVis){
        viewer->removeAllPointClouds(v1);
        viewer->removeAllShapes(v1);
        viewer->removeAllPointClouds(v2);
        viewer->removeAllShapes(v2);

        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr mapPc = map.getOriginalPointCloud();
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
                }
                else/* if(visRecCodes[f] == RecCode::Incorr) */ {
                    incorrPoses->push_back(curRecPose);
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
//        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
//                                                 pcl::visualization::PCL_VISUALIZER_SHADING_FLAT,
//                                                 "corr_poses_cloud",
//                                                 v1);
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

        viewer->resetStoppedFlag();
        viewer->initCameraParameters();
        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, -1.0, 0.0);
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
                                 const std::vector<ObjInstance> &objInstances1,
                                 const std::vector<ObjInstance> &objInstances2,
                                 std::ifstream &inputResFile,
                                 std::ofstream &outputResFile,
                                 const Vector7d &gtTransform,
                                 double scoreThresh,
                                 double scoreDiffThresh,
                                 double fitThresh,
                                 double poseDiffThresh,
                                 Vector7d &predTransform,
                                 RecCode &recCode,
                                 double &linDist,
                                 double &angDist,
                                 pcl::visualization::PCLVisualizer::Ptr viewer,
                                 int viewPort1,
                                 int viewPort2)
{
    
    vector<Vector7d> planesTrans;
    vector<double> planesTransScores;
    vector<double> planesTransFits;
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
                double curDiff;
                for(int c = 0; c < 7; ++c){
                    inputResFile >> curTrans(c);
                }
                inputResFile >> curScore >> curFit >> curDiff;
                planesTrans.push_back(curTrans);
                planesTransScores.push_back(curScore);
                planesTransFits.push_back(curFit);
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
        double diffAng = diffLogAng.transpose() * diffLogAng;
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
        }
        
        linDist += planesTransDiffEucl.front();
        angDist += planesTransDiffAng.front();
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
                                  " " << planesTransDiff[t] << endl;
            }
        }
        else if(matchType == Matching::MatchType::Unknown){
            outputResFile << -1 << endl;
        }
        outputResFile << endl << endl;
    }
}

