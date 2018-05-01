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
#include <fstream>
#include <chrono>
#include <thread>
#include <tuple>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <g2o/types/slam3d/se3quat.h>
#include <UnionFind.h>

#include "Matching.hpp"
#include "Misc.hpp"

using namespace std;

//Matching::Matching() {
//
//}

Matching::MatchType Matching::matchFrameToMap(const cv::FileStorage &fs,
                                              const vectorObjInstance &frameObjInstances,
                                              const vectorObjInstance &mapObjInstances,
                                              vectorVector7d &bestTrans,
                                              std::vector<double> &bestTransProbs,
                                              std::vector<double> &bestTransFits,
                                              std::vector<int> &bestTransDistinct,
                                              pcl::visualization::PCLVisualizer::Ptr viewer,
                                              int viewPort1,
                                              int viewPort2)
{
	cout << "Matching::matchFrameToMap" << endl;
	double planeAppThresh = (double)fs["matching"]["planeAppThresh"];
    double lineAppThresh = (double)fs["matching"]["lineAppThresh"];
    
    double lineToLineAngThresh = (double)fs["matching"]["lineToLineAngThresh"];
    double planeToPlaneAngThresh = (double)fs["matching"]["planeToPlaneAngThresh"];
    double planeToLineAngThresh = (double)fs["matching"]["planeToLineAngThresh"];
    double planeDistThresh = (double)fs["matching"]["planeDistThresh"];
    
    double scoreThresh = (double)fs["matching"]["scoreThresh"];
    double sinValsThresh = (double)fs["matching"]["sinValsThresh"];
    double planeEqDiffThresh = (double)fs["matching"]["planeEqDiffThresh"];
    double intAreaThresh = (double)fs["matching"]["intAreaThresh"];
    double lineEqDiffThresh = (double)fs["matching"]["lineEqDiffThresh"];
    double intLenThresh = (double)fs["matching"]["intLenThresh"];

    double shadingLevel = 1.0/16;

    chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();

//    Vector7d curGt;
//    curGt << -3.08641, 0.5365, -2.18575, 0.955013, -0.0118122, 0.239884, 0.173972;
//    g2o::SE3Quat curGtSE3Quat(curGt);

	vector<cv::Mat> frameObjFeats;
	compObjFeatures(frameObjInstances, frameObjFeats);
	vector<cv::Mat> mapObjFeats;
	compObjFeatures(mapObjInstances, mapObjFeats);

	if(viewer){
		viewer->removeAllPointClouds(viewPort1);
		viewer->removeAllShapes(viewPort1);
		viewer->removeAllPointClouds(viewPort2);
		viewer->removeAllShapes(viewPort2);
        
        for(int om = 0; om < mapObjInstances.size(); ++om){
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = mapObjInstances[om].getPoints();
            viewer->addPointCloud(curPl, string("plane1_") + to_string(om), viewPort2);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     shadingLevel,
                                                     string("plane1_") + to_string(om),
                                                     viewPort1);
        }
        
		for(int of = 0; of < frameObjInstances.size(); ++of){
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = frameObjInstances[of].getPoints();
//			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr curPlFaded(new pcl::PointCloud<pcl::PointXYZRGBA>(*curPl));
//			for(int pt = 0; pt < curPlFaded->size(); ++pt){
//				curPlFaded->at(pt).a = 100;
//			}
			viewer->addPointCloud(curPl, string("plane2_") + to_string(of), viewPort1);
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
													shadingLevel,
													string("plane2_") + to_string(of),
													viewPort2);
		}
  
		viewer->initCameraParameters();
		viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
        
//        viewer->spinOnce(100);
//        while (!viewer->wasStopped()) {
//            viewer->spinOnce(100);
//            std::this_thread::sleep_for(std::chrono::milliseconds(50));
//        }
	}

    vector<PotMatch> potMatches = findPotMatches(mapObjInstances,
                                                 frameObjInstances,
                                                 planeAppThresh,
                                                 lineAppThresh,
                                                 lineToLineAngThresh,
                                                 viewer,
                                                 viewPort1,
                                                 viewPort2);
    

	cout << "potMatches.size() = " << potMatches.size() << endl;

    chrono::high_resolution_clock::time_point endAppTime = chrono::high_resolution_clock::now();

	cout << "Adding sets" << endl;
    vector<vector<PotMatch> > potSets = findPotSets(potMatches,
                                                    mapObjInstances,
                                                    frameObjInstances,
                                                    planeDistThresh,
                                                    lineToLineAngThresh,
                                                    planeToPlaneAngThresh,
                                                    planeToLineAngThresh,
                                                    viewer,
                                                    viewPort1,
                                                    viewPort2);
    
    chrono::high_resolution_clock::time_point endTripletTime = chrono::high_resolution_clock::now();

	cout << "potSets.size() = " << potSets.size() << endl;

	cout << "computing 3D transforms" << endl;

	std::vector<ValidTransform> transforms;

	for(int s = 0; s < potSets.size(); ++s){
//		cout << "s = " << s << endl;
        
        vectorVector3d pointsMap;
        vectorVector4d planesMap;
        std::vector<Vector6d> linesMap;
        vectorVector3d pointsFrame;
        vectorVector4d planesFrame;
        std::vector<Vector6d> linesFrame;

        for(int ch = 0; ch < potSets[s].size(); ++ch) {
//            cout << "map " << ch << ": " << mapObjInstances[potSets[s][ch].plane1].getNormal().transpose() << endl;
            planesMap.push_back(mapObjInstances[potSets[s][ch].plane1].getNormal());
            const vectorLineSeg &allLinesMap = mapObjInstances[potSets[s][ch].plane1].getLineSegs();
            for (int lm = 0; lm < potSets[s][ch].lineSegs1.size(); ++lm) {
                linesMap.push_back(allLinesMap[potSets[s][ch].lineSegs1[lm]].toPointNormalEq());
            }
    
//            cout << "frame " << ch << ": " << frameObjInstances[potSets[s][ch].plane2].getNormal().transpose() << endl;
            planesFrame.push_back(frameObjInstances[potSets[s][ch].plane2].getNormal());
            const vectorLineSeg &allLinesFrame = frameObjInstances[potSets[s][ch].plane2].getLineSegs();
            for (int lf = 0; lf < potSets[s][ch].lineSegs2.size(); ++lf) {
                linesFrame.push_back(allLinesFrame[potSets[s][ch].lineSegs2[lf]].toPointNormalEq());
            }
        }
        
        bool fullConstrRot = true, fullConstrTrans = true;

        Vector7d transformComp = Matching::bestTransformPlanes(planesMap,
                                                               planesFrame,
                                                               sinValsThresh,
                                                               fullConstrRot);
        
//        vectorVector3d retPointsMap;
//        vectorVector3d retVirtPointsMap;
//        vectorVector3d retDirsMap;
//        std::vector<double> retDistsMap;
//        vectorVector3d retDistDirsMap;
//        vectorVector3d retDistPtsMap;
//        vectorVector3d retDistPtsDirsMap;
//        Matching::convertToPointsDirsDists(pointsMap,
//                                           planesMap,
//                                           linesMap,
//                                           retPointsMap,
//                                           retVirtPointsMap,
//                                           retDirsMap,
//                                           retDistsMap,
//                                           retDistDirsMap,
//                                           retDistPtsMap,
//                                           retDistPtsDirsMap);
//
//        vectorVector3d retPointsFrame;
//        vectorVector3d retVirtPointsFrame;
//        vectorVector3d retDirsFrame;
//        std::vector<double> retDistsFrame;
//        vectorVector3d retDistDirsFrame;
//        vectorVector3d retDistPtsFrame;
//        vectorVector3d retDistPtsDirsFrame;
//        Matching::convertToPointsDirsDists(pointsFrame,
//                                           planesFrame,
//                                           linesFrame,
//                                           retPointsFrame,
//                                           retVirtPointsFrame,
//                                           retDirsFrame,
//                                           retDistsFrame,
//                                           retDistDirsFrame,
//                                           retDistPtsFrame,
//                                           retDistPtsDirsFrame);
//
////		Vector7d curTransform;
//		bool fullConstrRot2, fullConstrTrans2;
//
//
//        Vector7d transformComp2 = Matching::bestTransformPointsDirsDists(retPointsMap,
//                                                                        retPointsFrame,
//                                                                        vector<double>(retPointsMap.size(), 1.0),
//                                                                        retVirtPointsMap,
//                                                                        retVirtPointsFrame,
//                                                                        vector<double>(retVirtPointsMap.size(), 1.0),
//                                                                        retDirsMap,
//                                                                        retDirsFrame,
//                                                                        vector<double>(retDirsMap.size(), 1.0),
//                                                                        retDistsMap,
//                                                                        retDistsFrame,
//                                                                        retDistDirsMap,
//                                                                        vector<double>(retDistsMap.size(), 1.0),
//                                                                        retDistPtsMap,
//                                                                        retDistPtsFrame,
//                                                                        retDistPtsDirsMap,
//                                                                        vector<double>(retDistPtsMap.size(), 1.0),
//                                                                        sinValsThresh,
//                                                                        fullConstrRot2,
//                                                                        fullConstrTrans2);
//
//        if(fullConstrRot != (fullConstrRot2 && fullConstrTrans2)){
//            cout << "constraints not consistant" << endl;
//            char a;
//            cin >> a;
//        }
//        else if(fullConstrRot){
//            double diff = Misc::transformLogDist(transformComp, transformComp2);
//            if(diff > 0.01){
//                cout << "transformation not consistent" << endl;
//                cout << transformComp.transpose() << endl;
//                cout << transformComp2.transpose() << endl;
//
//                char a;
//                cin >> a;
//            }
//        }

//        cout << "transformComp = " << transformComp.transpose() << endl;
//        cout << "fullConstrRot = " << fullConstrRot << endl;
//        cout << "fullConstrTrans = " << fullConstrTrans << endl;

		bool isAdded = false;
		if(fullConstrRot && fullConstrTrans){
            vector<double> intAreaPlanes;
            vector<vector<double> > intLenLines;
            
            double score = scoreTransformByProjection(transformComp,
                                                      potSets[s],
                                                      mapObjInstances,
                                                      frameObjInstances,
                                                      intAreaPlanes,
                                                      intLenLines,
                                                      planeEqDiffThresh,
                                                      lineEqDiffThresh,
                                                      intAreaThresh,
                                                      intLenThresh/*,
                                                      viewer,
                                                      viewPort1, viewPort2*/);
            
            if(score > scoreThresh){
				vector<double> appDiffs;
				for(int ch = 0; ch < potSets[s].size(); ++ch){
					appDiffs.push_back(potSets[s][ch].planeAppDiff);
				}
				transforms.emplace_back(transformComp,
										potSets[s],
                                        intAreaPlanes,
                                        intLenLines);
				isAdded = true;
			}
            
		}

//		if(viewer && isAdded){
//            cout << "transformComp = " << transformComp.transpose() << endl;
//
//			for(int p = 0; p < potSets[s].size(); ++p){
//				int om = potSets[s][p].plane1;
//				int of = potSets[s][p].plane2;
//
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														1.0,
//														string("plane1_") + to_string(om),
//														viewPort1);
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														1.0,
//														string("plane2_") + to_string(of),
//														viewPort2);
//			}
//
//			// time for watching
//			viewer->resetStoppedFlag();
//
////			viewer->initCameraParameters();
////			viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//			while (!viewer->wasStopped()){
//				viewer->spinOnce (100);
//				std::this_thread::sleep_for(std::chrono::milliseconds(50));
//			}
//
//            for(int p = 0; p < potSets[s].size(); ++p){
//                int om = potSets[s][p].plane1;
//                int of = potSets[s][p].plane2;
//
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														shadingLevel,
//														string("plane1_") + to_string(om),
//														viewPort1);
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														shadingLevel,
//														string("plane2_") + to_string(of),
//														viewPort2);
//			}
//		}
	}

    chrono::high_resolution_clock::time_point endTransformTime = chrono::high_resolution_clock::now();


	cout << "transforms.size() = " << transforms.size() << endl;
	for(int t = 0; t < transforms.size(); ++t){
		// weight depending on a number of times the object was matched weighted with distance
		// the more matches in a vicinity the lesser the overall weight
		vector<float> frameObjInvWeights(transforms[t].matchSet.size(), 1.0);
		vector<float> mapObjInvWeights(transforms[t].matchSet.size(), 1.0);
		for(int tc = 0; tc < transforms.size(); ++tc){
			for(int p = 0; p < transforms[t].matchSet.size(); ++p){
				g2o::SE3Quat trans(transforms[t].transform);
				g2o::SE3Quat transComp(transforms[tc].transform);
				g2o::SE3Quat diff = trans.inverse() * transComp;
				Vector6d logMapDiff = diff.log();
				double dist = logMapDiff.transpose() * logMapDiff;
				for(int pc = 0; pc < transforms[tc].matchSet.size(); ++pc){
                    if(transforms[t].matchSet[p].plane1 == transforms[tc].matchSet[pc].plane1){
                        mapObjInvWeights[p] += exp(-dist);
                    }
                    if(transforms[t].matchSet[p].plane2 == transforms[tc].matchSet[pc].plane2){
						frameObjInvWeights[p] += exp(-dist);
					}
				}
			}
		}
//		cout << "intAreas = " << transforms[t].intAreas << endl;
//		cout << "mapObjInvWeights = " << mapObjInvWeights << endl;
		double curScore = 0.0;
		for(int p = 0; p < transforms[t].matchSet.size(); ++p){
//            cout << "exp(-transforms[t].appDiffs[p]) = " << exp(-transforms[t].appDiffs[p]) << endl;
			curScore += transforms[t].intAreaPlanes[p]/frameObjInvWeights[p]*exp(-transforms[t].matchSet[p].planeAppDiff);
		}
		transforms[t].score = curScore;
//		cout << "score = " << transforms[t].score << endl;
	}

    chrono::high_resolution_clock::time_point endScoreTime = chrono::high_resolution_clock::now();

//    cout << "Exporting distances" << endl;
//	vector<vector<double>> distMat(transforms.size(), vector<double>(transforms.size(), 0));
//	for(int tr1 = 0; tr1 < transforms.size(); ++tr1){
//		for(int tr2 = tr1 + 1; tr2 < transforms.size(); ++tr2){
//			g2o::SE3Quat trans1(transforms[tr1].transform);
//			g2o::SE3Quat trans2(transforms[tr2].transform);
//			g2o::SE3Quat diff = trans1.inverse() * trans2;
//			Vector6d logMapDiff = diff.log();
//			double dist = logMapDiff.transpose() * logMapDiff;
//			if(tr1 == 0){
//				cout << "trans1 = " << transforms[tr1].transform.transpose() << endl;
//				cout << "trans2 = " << transforms[tr2].transform.transpose() << endl;
//				cout << "diff = " << diff.toVector().transpose() << endl;
//				cout << "logMapDiff = " << logMapDiff.transpose() << endl;
//				cout << "dist = " << dist << endl;
//			}
//			distMat[tr1][tr2] = dist;
//			distMat[tr2][tr1] = dist;
//		}
//	}
//
//	ofstream distFile("../output/distMat");
//	for(int r = 0; r < distMat.size(); ++r){
//        distFile << transforms[r].score << " ";
//		for(int c = 0; c < distMat[r].size(); ++c){
//			distFile << distMat[r][c] << " ";
//		}
//		distFile << endl;
//	}

//	Vector7d ret;
	if(transforms.size() > 0){
//        cout << "construct probability distribution using gaussian kernels" << endl;
		// construct probability distribution using gaussian kernels
		vectorProbDistKernel dist;
		Eigen::Matrix<double, 6, 6> distInfMat = Eigen::Matrix<double, 6, 6>::Identity();
		// information matrix for position
		distInfMat.block<3, 3>(3, 3) = 10.0 * Eigen::Matrix<double, 3, 3>::Identity();
		// information matrix for orientation
		distInfMat.block<3, 3>(0, 0) = 10.0 * Eigen::Matrix<double, 3, 3>::Identity();
		for(int t = 0; t < transforms.size(); ++t){
			dist.emplace_back(transforms[t].transform, distInfMat, transforms[t].score);
		}

		vector<pair<double, int>> transProb;
		// find point for which the probability is the highest
//		int bestInd = 0;
//		double bestScore = numeric_limits<double>::lowest();
		for(int t = 0; t < transforms.size(); ++t){
			double curProb = evalPoint(transforms[t].transform, dist);
//			cout << "transform = " << transforms[t].transpose() << endl;
//			cout << "prob = " << curProb << endl;
//			if(bestScore < curScore){
//				bestScore = curScore;
//				bestInd = t;
//			}
			transProb.emplace_back(curProb, t);
		}
		sort(transProb.begin(), transProb.end());

		// seeking for at most 2 best maximas
		for(int t = transProb.size() - 1; t >= 0 && bestTrans.size() < 2; --t){
			bool add = true;
			for(int i = 0; i < bestTrans.size() && add; ++i){
				g2o::SE3Quat btSE3Quat(bestTrans[i]);
				g2o::SE3Quat curtSE3Quat(transforms[transProb[t].second].transform);
				Vector6d diffLog = (btSE3Quat.inverse() * curtSE3Quat).log();
				double diff = diffLog.transpose() * diffLog;
				// if close to already added transform
				if(diff < 0.11){
					add = false;
				}
			}
			if(add){
				bestTrans.push_back(transforms[transProb[t].second].transform);
				bestTransProbs.push_back(transProb[t].first);
			}
		}

        {
//            vector<Vector7d> newBestTrans;
//            vector<double> newBestTransProbs;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPc(new pcl::PointCloud<pcl::PointXYZRGB>());
            for(int om = 0; om < mapObjInstances.size(); ++om){
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr curMapPc = mapObjInstances[om].getPoints();
                mapPc->insert(mapPc->end(), curMapPc->begin(), curMapPc->end());
            }
//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr framePc(new pcl::PointCloud<pcl::PointXYZRGB>());
//            for(int of = 0; of < frameObjInstances.size(); ++of){
//                pcl::PointCloud<pcl::PointXYZRGB>::Ptr curFramePc = frameObjInstances[of].getPoints();
//                framePc->insert(framePc->end(), curFramePc->begin(), curFramePc->end());
//            }

            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
            kdTree.setInputCloud(mapPc);

            for(int t = 0; t < bestTrans.size(); ++t) {
                cout << "fit score on transformation " << t << endl;
                
                vectorObjInstance frameObjInstancesTrans = frameObjInstances;
                for(ObjInstance &obj : frameObjInstancesTrans){
                    obj.transform(bestTrans[t]);
                }

//                g2o::SE3Quat curTransSE3Quat(bestTrans[t]);
//                Eigen::Matrix4d curTransMat = curTransSE3Quat.to_homogeneous_matrix();
//
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr framePcTrans(new pcl::PointCloud<pcl::PointXYZRGB>());
//                pcl::transformPointCloud(*framePc, *framePcTrans, curTransMat);
    
                for(ObjInstance &obj : frameObjInstancesTrans){
                    framePcTrans->insert(framePcTrans->end(),
                                         obj.getPoints()->begin(),
                                         obj.getPoints()->end());
                }
                
                vector<int> nnIndices(1);
                std::vector<float> nnDists(1);
                double fitScore = 0.0;
                int ptCnt = 0;
                double maxDist = 0.0;
                for(int p = 0; p < framePcTrans->size(); ++p){
                    kdTree.nearestKSearch(framePcTrans->at(p), 1, nnIndices, nnDists);

                    fitScore += nnDists[0];
                    ++ptCnt;
                    
                    maxDist = std::max(maxDist, (double)nnDists[0]);
                }
                if(ptCnt > 0){
                    fitScore /= ptCnt;
                }
                cout << "maxDist = " << maxDist << endl;
//                pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
//                icp.setInputSource(framePc);
//                icp.setInputTarget(mapPc);
//
//                // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
//                icp.setMaxCorrespondenceDistance(0.05);
//                // Set the maximum number of iterations (criterion 1)
//                icp.setMaximumIterations(0);
//                // Set the transformation epsilon (criterion 2)
//                icp.setTransformationEpsilon(1e-3);
//                // Set the euclidean distance difference epsilon (criterion 3)
//                icp.setEuclideanFitnessEpsilon(0.01);
//
//                pcl::PointCloud<pcl::PointXYZRGB>::Ptr outPc(new pcl::PointCloud<pcl::PointXYZRGB>());
//                g2o::SE3Quat curTransSE3Quat(bestTrans[t]);
//                Eigen::Matrix4d curTransMat = curTransSE3Quat.to_homogeneous_matrix();
//                icp.align(*outPc, curTransMat.cast<float>());

//                 = icp.getFitnessScore();
                cout << "transform = " << bestTrans[t].transpose() << endl;
//                Eigen::Matrix4f icpFinalTrans = icp.getFinalTransformation();
//                cout << "icpTransform = " << g2o::SE3Quat(icpFinalTrans.block<3,3>(0,0).cast<double>(),
//                                                          icpFinalTrans.block<3,1>(0,3).cast<double>()).toVector().transpose() << endl;
                cout << "fitScore = " << fitScore << endl;

                bestTransFits.push_back(fitScore);

                
                {
                    vector<pair<int, int>> matches;
                    set<int> frameIdxsSet;
                    set<int> mapIdxsSet;
                    for(int of = 0; of < frameObjInstancesTrans.size(); ++of) {
                        for (int om = 0; om < mapObjInstances.size(); ++om) {
                            const ObjInstance &frameObj = frameObjInstancesTrans[of];
                            const ObjInstance &mapObj = mapObjInstances[om];
                            if(frameObj.isMatching(mapObj)){
                                matches.emplace_back(om, of);
                                mapIdxsSet.insert(om);
                                frameIdxsSet.insert(of);
                            }
                        }
                    }
                    
                    int mapDistinct = countDifferent(mapIdxsSet,
                                                     mapObjInstances);
                    int frameDistinct = countDifferent(frameIdxsSet,
                                                     frameObjInstances);
                    
                    cout << "mapDistinct = " << mapDistinct << endl;
                    cout << "frameDistinct = " << frameDistinct << endl;
                    
                    bestTransDistinct.push_back(min(mapDistinct, frameDistinct));
                }
//                if(fitScore < 0.1){
//                    newBestTrans.push_back(bestTrans[t]);
//                    newBestTransProbs.push_back(bestTransProbs[t]);
//                }
                if(viewer) {
                    viewer->removeAllPointClouds(viewPort1);
                    viewer->removeAllShapes(viewPort1);
                    viewer->removeAllPointClouds(viewPort2);
                    viewer->removeAllShapes(viewPort2);

                    viewer->addPointCloud(framePcTrans, "cloud_out", viewPort1);
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                             1.0, 0.0, 0.0,
                                                             "cloud_out",
                                                             viewPort1);
                    
                    viewer->addPointCloud(mapPc, "cloud_map", viewPort1);

                    viewer->resetStoppedFlag();
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                    viewer->spinOnce(100);
                    while (!viewer->wasStopped()) {
                        viewer->spinOnce(100);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }
            }

//            bestTrans.swap(newBestTrans);
//            bestTransProbs.swap(newBestTransProbs);
        }

	//	g2o::SE3Quat idenSE3Quat;
	//	Vector7d ret = idenSE3Quat.toVector();
//		bestTrans = transforms[bestInd];
	}

    chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();

    static chrono::milliseconds totalAppTime = chrono::milliseconds::zero();
    static chrono::milliseconds totalTripletsTime = chrono::milliseconds::zero();
    static chrono::milliseconds totalTransformTime = chrono::milliseconds::zero();
    static chrono::milliseconds totalScoreTime = chrono::milliseconds::zero();
    static chrono::milliseconds totalFitTime = chrono::milliseconds::zero();
    static double meanTriTransTime = 0;
    static double m2TriTransTime = 0;
    static double maxTriTransTime = 0;
    static int totalCnt = 0;

    totalAppTime += chrono::duration_cast<chrono::milliseconds>(endAppTime - startTime);
    totalTripletsTime += chrono::duration_cast<chrono::milliseconds>(endTripletTime - endAppTime);
    totalTransformTime += chrono::duration_cast<chrono::milliseconds>(endTransformTime - endTripletTime);
    totalScoreTime += chrono::duration_cast<chrono::milliseconds>(endScoreTime - endTransformTime);
    totalFitTime += chrono::duration_cast<chrono::milliseconds>(endTime - endScoreTime);
    ++totalCnt;

    {
        double newVal = chrono::duration_cast<chrono::milliseconds>(endTransformTime - endAppTime).count();
        maxTriTransTime = max(maxTriTransTime, newVal);
        double delta = newVal - meanTriTransTime;
        meanTriTransTime += delta/totalCnt;
        double delta2 = newVal - meanTriTransTime;
        m2TriTransTime += delta*delta2;
        
        if(totalCnt > 1) {
            double var = m2TriTransTime / (totalCnt - 1);
            cout << "triplets + transform std dev = " << sqrt(var) << endl;
            cout << "triplets + transform max = " << maxTriTransTime << endl;
        }
    }
    
    cout << "Mean matching app time: " << (totalAppTime.count() / totalCnt) << endl;
    cout << "Mean matching triplets time: " << (totalTripletsTime.count() / totalCnt) << endl;
    cout << "Mean matching transform time: " << (totalTransformTime.count() / totalCnt) << endl;
    cout << "Mean matching score time: " << (totalScoreTime.count() / totalCnt) << endl;
    cout << "Mean matching fit time: " << (totalFitTime.count() / totalCnt) << endl;

	// No satisfying transformations - returning identity
	if(bestTrans.size() == 0){
//		bestTrans << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
		return MatchType::Unknown;
	}

	return MatchType::Ok;
}


void Matching::convertToPointsDirsDists(const vectorVector3d &points,
                                        const vectorVector4d &planes,
                                        const std::vector<Vector6d> &lines,
                                        vectorVector3d &retPoints,
                                        vectorVector3d &retVirtPoints,
                                        vectorVector3d &retDirs,
                                        std::vector<double> &retDists,
                                        vectorVector3d &retDistDirs,
                                        vectorVector3d &retDistPts,
                                        vectorVector3d &retDistPtsDirs)
{
    for(int p = 0; p < points.size(); ++p){
        retPoints.push_back(points[p]);
        for(int l = 0; l < lines.size(); ++l){
            Eigen::Vector3d virtPoint = Misc::closestPointOnLine(points[p],
                                                           lines[l].head<3>(),
                                                           lines[l].tail<3>());
            retVirtPoints.push_back(virtPoint);
        }
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

double Matching::compAngleDiffBetweenNormals(const Eigen::Vector3d &nf1,
                                             const Eigen::Vector3d &ns1,
                                             const Eigen::Vector3d &nf2,
                                             const Eigen::Vector3d &ns2)
{
    double ang1 = acos(nf1.dot(ns1));
//    ang1 = min(ang1, pi - ang1);
    double ang2 = acos(nf2.dot(ns2));
//    ang2 = min(ang2, pi - ang2);
    double angDiff = fabs(ang1 - ang2);
    
    return angDiff;
}


bool Matching::checkLineToLineAng(const vectorLineSeg &lineSegs1,
                                  const vectorLineSeg &lineSegs2,
                                  double lineToLineAngThresh)
{
    bool isConsistent = true;
    for(int lf = 0; lf < lineSegs1.size(); ++lf){
        for(int ls = lf + 1; ls < lineSegs1.size(); ++ls){
            Eigen::Vector3d nf1 = (lineSegs1[lf].getP2() - lineSegs1[lf].getP1()).normalized();
            Eigen::Vector3d ns1 = (lineSegs1[ls].getP2() - lineSegs1[ls].getP1()).normalized();
            Eigen::Vector3d nf2 = (lineSegs2[lf].getP2() - lineSegs2[lf].getP1()).normalized();
            Eigen::Vector3d ns2 = (lineSegs2[ls].getP2() - lineSegs2[ls].getP1()).normalized();
            
            double angDiff = compAngleDiffBetweenNormals(nf1, ns1, nf2, ns2);
            
            if(angDiff > lineToLineAngThresh){
                isConsistent = false;
                break;
            }
        }
    }
    return isConsistent;
}


bool Matching::checkPlaneToPlaneAng(const vectorVector4d &planes1,
                                    const vectorVector4d &planes2,
                                    double planeToPlaneAngThresh)
{
    bool isConsistent = true;
    for(int pf = 0; pf < planes1.size(); ++pf){
        for(int ps = pf + 1; ps < planes1.size(); ++ps){
            Eigen::Vector3d nf1 = planes1[pf].head<3>().normalized();
            Eigen::Vector3d ns1 = planes1[ps].head<3>().normalized();
            Eigen::Vector3d nf2 = planes2[pf].head<3>().normalized();
            Eigen::Vector3d ns2 = planes2[ps].head<3>().normalized();
    
            double angDiff = compAngleDiffBetweenNormals(nf1, ns1, nf2, ns2);
    
            if(angDiff > planeToPlaneAngThresh){
                isConsistent = false;
                break;
            }
        }
    }
    
    return isConsistent;
}

bool Matching::checkPlaneToLineAng(const vectorVector4d &planes1,
                                   const vectorLineSeg &lineSegs1,
                                   const vectorVector4d &planes2,
                                   const vectorLineSeg &lineSegs2,
                                   double planeToLineAngThresh)
{
    bool isConsistent = true;
    for(int pf = 0; pf < planes1.size(); ++pf){
        for(int ls = 0; ls < lineSegs1.size(); ++ls){
            Eigen::Vector3d nf1 = planes1[pf].head<3>().normalized();
            Eigen::Vector3d ns1 = (lineSegs1[ls].getP2() - lineSegs1[ls].getP1()).normalized();
            Eigen::Vector3d nf2 = planes2[pf].head<3>().normalized();
            Eigen::Vector3d ns2 = (lineSegs2[ls].getP2() - lineSegs2[ls].getP1()).normalized();
    
            double angDiff = compAngleDiffBetweenNormals(nf1, ns1, nf2, ns2);
    
            if(angDiff > planeToLineAngThresh){
                isConsistent = false;
                break;
            }
        }
    }
    
    return isConsistent;
}

vector<Matching::PotMatch> Matching::findPotMatches(const vectorObjInstance &mapObjInstances,
                                                    const vectorObjInstance &frameObjInstances,
                                                    double planeAppThresh,
                                                    double lineAppThresh,
                                                    double lineToLineAngThresh,
                                                    pcl::visualization::PCLVisualizer::Ptr viewer,
                                                    int viewPort1,
                                                    int viewPort2)
{
    double shadingLevel = 1.0/16;
    
    vector<PotMatch> potMatches;
    
    vector<cv::Mat> frameObjFeats;
    compObjFeatures(frameObjInstances, frameObjFeats);
    vector<cv::Mat> mapObjFeats;
    compObjFeatures(mapObjInstances, mapObjFeats);
    
    for(int of = 0; of < frameObjInstances.size(); ++of){
        for(int om = 0; om < mapObjInstances.size(); ++om){
            cv::Mat histDiff = cv::abs(frameObjFeats[of] - mapObjFeats[om]);
            double histDist = ObjInstance::compHistDist(frameObjFeats[of], mapObjFeats[om]);
            if(histDist < planeAppThresh){
                
                const vectorLineSeg &frameLineSegs = frameObjInstances[of].getLineSegs();
                const vectorLineSeg &mapLineSegs = mapObjInstances[om].getLineSegs();
                vector<pair<int, int> > potLineMatches;
                for(int lf = 0; lf < frameLineSegs.size(); ++lf){
                    for(int lm = 0; lm < mapLineSegs.size(); ++lm){
                        potLineMatches.emplace_back(lm, lf);
                    }
                }
                
                int numComb = (1 << potLineMatches.size());
    
                // TODO Add line appearance difference computation
                vector<double> linesAppDiffs(0.0, potLineMatches.size());
                
                for(int c = 0; c < numComb; ++c){
                    bool addFlag = true;
                    vector<int> linesMap;
                    vector<int> linesFrame;
                    vector<double> curLinesAppDiffs;
                    
                    for(int l = 0; l < potLineMatches.size(); ++l){
                        if(c & (1 << l)){
                            
                            if(linesAppDiffs[l] > lineAppThresh){
                                addFlag = false;
                                break;
                            }
                            else{
                                for(int prevl = 0; prevl < linesMap.size(); ++prevl){
                                    if(!checkLineToLineAng(vectorLineSeg{mapLineSegs[linesMap[prevl]],
                                                                           mapLineSegs[potLineMatches[l].first]},
                                                           vectorLineSeg{frameLineSegs[linesFrame[prevl]],
                                                                           frameLineSegs[potLineMatches[l].second]},
                                                           lineToLineAngThresh))
                                    {
                                        addFlag = false;
                                        break;
                                    }
                                }
                            }
                            
                            linesMap.push_back(potLineMatches[l].first);
                            linesFrame.push_back(potLineMatches[l].second);
                            curLinesAppDiffs.push_back(linesAppDiffs[l]);
                        }
                    }
                    if(addFlag){
//                        cout << "adding potential match between " << om << " and " << of << endl;
                        potMatches.emplace_back(om,
                                                linesMap,
                                                of,
                                                linesFrame,
                                                histDist,
                                                curLinesAppDiffs);
                    }
                }
                
            }

//			cout << "dist (" << of << ", " << om << ") = " << histDist << endl;
//			if(viewer){
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														1.0,
//														string("plane1_") + to_string(om),
//														viewPort1);
//                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//                                                         1.0,
//                                                         string("plane2_") + to_string(of),
//                                                         viewPort2);
//
//				// time for watching
//				viewer->resetStoppedFlag();
//
////				viewer->initCameraParameters();
////				viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//				while (!viewer->wasStopped()){
//					viewer->spinOnce (100);
//					std::this_thread::sleep_for(std::chrono::milliseconds(50));
//				}
//
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														shadingLevel,
//														string("plane1_") + to_string(om),
//														viewPort1);
//                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//                                                         shadingLevel,
//                                                         string("plane2_") + to_string(of),
//                                                         viewPort2);
//			}
        }
    }
    
    static constexpr int potMatchesThresh = 275;
    if(potMatches.size() > potMatchesThresh) {
        vector<double> histDists;
        for (const PotMatch &pm : potMatches) {
            histDists.push_back(pm.planeAppDiff);
        }
        sort(histDists.begin(), histDists.end());
        
        double newPlaneAppThresh = histDists[potMatchesThresh];
        
        vector<PotMatch> newPotMatches;
        for (const PotMatch &pm : potMatches) {
            if(pm.planeAppDiff < newPlaneAppThresh){
                newPotMatches.push_back(pm);
            }
        }
        potMatches.swap(newPotMatches);
    }
    
    return potMatches;
}

vector<vector<Matching::PotMatch>> Matching::findPotSets(const vector<PotMatch> &potMatches,
                                                         const vectorObjInstance &mapObjInstances,
                                                         const vectorObjInstance &frameObjInstances,
                                                         double planeDistThresh,
                                                         double lineToLineAngThresh,
                                                         double planeToPlaneAngThresh,
                                                         double planeToLineAngThresh,
                                                         pcl::visualization::PCLVisualizer::Ptr viewer,
                                                         int viewPort1,
                                                         int viewPort2)
{
    vector<vector<PotMatch> > potSets;
    vector<vector<int>> potSetsIdxs;
    
    vector<vector<double>> frameObjDistances;
    compObjDistances(frameObjInstances, frameObjDistances);
    vector<vector<double>> mapObjDistances;
    compObjDistances(mapObjInstances, mapObjDistances);
    
    // initialize with single matches
    potSets.resize(potMatches.size());
    for(int p = 0; p < potMatches.size(); ++p){
        potSets[p].push_back(potMatches[p]);
        potSetsIdxs.push_back(vector<int>{p});
    }
    // generate dublets and triplets
    for(int ne = 2; ne <= 3; ++ne){
//        cout << "ne = " << ne << endl;
//        cout << "potSets.size() = " << potSets.size() << endl;
        
        vector<vector<PotMatch> > newPotSets;
        vector<vector<int>> newPotSetsIdxs;
        unordered_set<uint64_t> newPotSetsIdxsSet;

        for(int s = 0; s < potSets.size(); ++s){
            
            for(int p = 0; p < potMatches.size(); ++p){
                vector<PotMatch> curSet = potSets[s];
                curSet.push_back(potMatches[p]);
//                cout << "matches:" << endl;
//                for(int ch = 0; ch < curSet.size(); ++ch){
//                    cout << curSet[ch].plane1 << " " << curSet[ch].plane2 << endl;
//                }
                vector<int> curIdxs = potSetsIdxs[s];
                curIdxs.push_back(p);
//                sort(curIdxs.begin(), curIdxs.end());
                for(int i = 0; i < curIdxs.size() - 1; ++i){
                    for(int j = 0; j < curIdxs.size() - i - 1; ++j){
                        if(curIdxs[j] > curIdxs[j+1]){
                            swap(curIdxs[i], curIdxs[j]);
                        }
                    }
                }
                
                static constexpr int mult = 10000;
                int curMult = 1;
                uint64_t curHashValue = 0;
                for(int i = 0; i < curIdxs.size(); ++i){
                    curHashValue += curIdxs[i] * curMult;
                    curMult *= mult;
                }
                
                bool valid = true;
                // if there was the same combination
                if(newPotSetsIdxsSet.count(curHashValue) > 0){
                    valid = false;
                }

                if(valid) {
                    set<int> planesMapSet;
                    set<int> planesFrameSet;
                    int numPlanePairs = 0;
                    int numLinePairs = 0;
                    for (int ch = 0; ch < curSet.size(); ++ch) {
                        // if same plane is in more than one pair than not valid
                        if (planesMapSet.count(curSet[ch].plane1) != 0 ||
                            planesFrameSet.count(curSet[ch].plane2) != 0) {
                            valid = false;
                        }
                        ++numPlanePairs;
                        numLinePairs += curSet[ch].lineSegs1.size();
        
                        planesMapSet.insert(curSet[ch].plane1);
                        planesFrameSet.insert(curSet[ch].plane2);
                    }
//                if(ne == 3 && numPlanePairs + numLinePairs < 3){
//                    valid = false;
//                }
                }
                if(valid){
                    // if planes are not close enough
                    for(int p1 = 0; p1 < curSet.size(); ++p1) {
                        for (int p2 = p1 + 1; p2 < curSet.size(); ++p2) {
                            int pm1 = curSet[p1].plane1;
                            int pm2 = curSet[p2].plane1;
                            if (mapObjDistances[pm1][pm2] > planeDistThresh) {
                                valid = false;
                            }
                            int pf1 = curSet[p1].plane2;
                            int pf2 = curSet[p2].plane2;
                            if (frameObjDistances[pf1][pf2] > planeDistThresh) {
                                valid = false;
                            }
                        }
                    }
                }
                if(valid){
                    // check angles between planes and lines

                    vectorVector4d planesMap;
                    vectorLineSeg linesMap;
                    vectorVector4d planesFrame;
                    vectorLineSeg linesFrame;
                    for(int ch = 0; ch < curSet.size(); ++ch){
                        planesMap.push_back(mapObjInstances[curSet[ch].plane1].getNormal());
                        const vectorLineSeg &allLinesMap = mapObjInstances[curSet[ch].plane1].getLineSegs();
                        for(int lm = 0; lm < curSet[ch].lineSegs1.size(); ++lm){
                            linesMap.push_back(allLinesMap[curSet[ch].lineSegs1[lm]]);
                        }

                        planesFrame.push_back(frameObjInstances[curSet[ch].plane2].getNormal());
                        const vectorLineSeg &allLinesFrame = frameObjInstances[curSet[ch].plane2].getLineSegs();
                        for(int lf = 0; lf < curSet[ch].lineSegs2.size(); ++lf){
                            linesFrame.push_back(allLinesFrame[curSet[ch].lineSegs2[lf]]);
                        }
                    }

                    if(!checkLineToLineAng(linesMap, linesFrame, lineToLineAngThresh)){
                        valid = false;
                    }
                    if(!checkPlaneToPlaneAng(planesMap, planesFrame, planeToPlaneAngThresh)){
                        valid = false;
                    }
                    if(!checkPlaneToLineAng(planesMap, linesMap, planesFrame, linesFrame, planeToLineAngThresh)){
                        valid = false;
                    }
                    if(valid) {
                        newPotSets.push_back(curSet);
                        newPotSetsIdxs.push_back(curIdxs);
                        newPotSetsIdxsSet.insert(curHashValue);
                    }
                }
            }
        }
        
        potSets.swap(newPotSets);
        potSetsIdxs.swap(newPotSetsIdxs);
    }
    
//    vector<vector<PotMatch> > potSetsComp;
//
//    int considered = 0;
//    for(int ne = 1; ne <= min(3, (int)potMatches.size()); ++ne){
//        vector<int> curChoice;
//        for(int i = 0; i < ne; ++i){
//            curChoice.push_back(i);
//        }
//        do{
//            ++considered;
////			cout << "curChoice = " << curChoice << endl;
//            vector<PotMatch> curSet;
//            set<int> planesMapSet;
//            set<int> planesFrameSet;
//            bool valid = true;
//            int numPlanePairs = 0;
//            int numLinePairs = 0;
//            for(int ch = 0; ch < curChoice.size(); ++ch){
//                curSet.push_back(potMatches[curChoice[ch]]);
//                // if same plane is in more than one pair than not valid
//                if(planesMapSet.count(potMatches[curChoice[ch]].plane1) != 0 ||
//                   planesFrameSet.count(potMatches[curChoice[ch]].plane2) != 0)
//                {
//                    valid = false;
//                }
//                ++numPlanePairs;
//                numLinePairs += potMatches[curChoice[ch]].lineSegs1.size();
//
//                planesMapSet.insert(potMatches[curChoice[ch]].plane1);
//                planesFrameSet.insert(potMatches[curChoice[ch]].plane2);
//            }
//            if(numPlanePairs + numLinePairs < 3){
//                valid = false;
//            }
//            if(valid){
//                // if planes are not close enough
//                for(int p1 = 0; p1 < curChoice.size(); ++p1) {
//                    for (int p2 = p1 + 1; p2 < curChoice.size(); ++p2) {
//                        int pm1 = potMatches[curChoice[p1]].plane1;
//                        int pm2 = potMatches[curChoice[p2]].plane1;
//                        if (mapObjDistances[pm1][pm2] > planeDistThresh) {
//                            valid = false;
//                        }
//                        int pf1 = potMatches[curChoice[p1]].plane2;
//                        int pf2 = potMatches[curChoice[p2]].plane2;
//                        if (frameObjDistances[pf1][pf2] > planeDistThresh) {
//                            valid = false;
//                        }
//                    }
//                }
//            }
//            if(valid){
//                // check angles between planes and lines
//
//                vectorVector4d planesMap;
//                vectorLineSeg linesMap;
//                vectorVector4d planesFrame;
//                vectorLineSeg linesFrame;
//                for(int ch = 0; ch < curSet.size(); ++ch){
//                    planesMap.push_back(mapObjInstances[curSet[ch].plane1].getNormal());
//                    const vectorLineSeg &allLinesMap = mapObjInstances[curSet[ch].plane1].getLineSegs();
//                    for(int lm = 0; lm < curSet[ch].lineSegs1.size(); ++lm){
//                        linesMap.push_back(allLinesMap[curSet[ch].lineSegs1[lm]]);
//                    }
//
//                    planesFrame.push_back(frameObjInstances[curSet[ch].plane2].getNormal());
//                    const vectorLineSeg &allLinesFrame = frameObjInstances[curSet[ch].plane2].getLineSegs();
//                    for(int lf = 0; lf < curSet[ch].lineSegs2.size(); ++lf){
//                        linesFrame.push_back(allLinesFrame[curSet[ch].lineSegs2[lf]]);
//                    }
//                }
//
//                if(!checkLineToLineAng(linesMap, linesFrame, lineToLineAngThresh)){
//                    valid = false;
//                }
//                if(!checkPlaneToPlaneAng(planesMap, planesFrame, planeToPlaneAngThresh)){
//                    valid = false;
//                }
//                if(!checkPlaneToLineAng(planesMap, linesMap, planesFrame, linesFrame, planeToLineAngThresh)){
//                    valid = false;
//                }
//                if(valid) {
//                    potSetsComp.push_back(curSet);
//                }
//            }
//        }while(Misc::nextChoice(curChoice, potMatches.size()));
//    }
//
//    cout << "potSets.size() = " << potSets.size() << endl;
//    cout << "potSetsComp.size() = " << potSetsComp.size() << endl;
//    cout << "considered = " << considered << endl;
    
    return potSets;
}

void Matching::compObjFeatures(const vectorObjInstance& objInstances,
							std::vector<cv::Mat>& objFeats)
{
	for(int o = 0; o < objInstances.size(); ++o){
		objFeats.push_back(objInstances[o].getColorHist());
	}
}


void Matching::compObjDistances(const vectorObjInstance& objInstances,
                                std::vector<std::vector<double>>& objDistances)
{
    objDistances.resize(objInstances.size(), vector<double>(objInstances.size(), 0));
//    cout << "objDistances.size() = " << objDistances.size()
//         << ", objDistances.front().size() = " << objDistances.front().size() << endl;
    for(int o1 = 0; o1 < objInstances.size(); ++o1){
//        cout << "o1 = " << o1 << endl;
        for(int o2 = o1 + 1; o2 < objInstances.size(); ++o2){
//            cout << "o2 = " << o2 << endl;
            double minDist = objInstances[o1].getHull().minDistance(objInstances[o2].getHull());
//			cout << "o1 = " << o1 << ", o2 = " << o2 << endl;
            objDistances[o1][o2] = minDist;
            objDistances[o2][o1] = minDist;
        }
    }
}

void Matching::comp3DTransform(const vectorVector4d& planes1,
								const vectorVector4d& planes2,
								const std::vector<std::pair<int, int>>& triplet,
								Vector7d& transform,
                                double sinValsThresh,
								bool& fullConstr)
{

    vectorVector4d triPlanes1;
    vectorVector4d triPlanes2;
    for(int p = 0; p < triplet.size(); ++p){
//			cout << "triplets[t][p] = (" << triplet[p].first << ", " << triplet[p].second << ")" << endl;
        triPlanes1.push_back(planes1[triplet[p].first]);
        triPlanes2.push_back(planes2[triplet[p].second]);
    }
    transform = bestTransformPlanes(triPlanes1, triPlanes2, sinValsThresh, fullConstr);

//    {
//        vectorVector4d points1, points2;
//        vectorVector4d dirs1, dirs2;
//        for(int p = 0; p < triPlanes1.size(); ++p) {
//            cout << "triPlanes1[p] = " << triPlanes1[p].transpose() << endl;
//            double normNorm1 = triPlanes1[p].head<3>().norm();
//            Eigen::Vector3d norm1 = triPlanes1[p].head<3>()/normNorm1;
//            cout << "norm1 = " << norm1.transpose() << endl;
//            double d1 = triPlanes1[p][3]/normNorm1;
//            cout << "d1 = " << d1 << endl;
//            Eigen::Vector3d point1 = norm1 * d1;
//            cout << "point1 = " << point1.transpose() << endl;
//            points1.push_back(point1);
//            dirs1.push_back(norm1);
//
//            cout << "triPlanes2[p] = " << triPlanes2[p].transpose() << endl;
//            double normNorm2 = triPlanes2[p].head<3>().norm();
//            Eigen::Vector3d norm2 = triPlanes2[p].head<3>()/normNorm2;
//            cout << "norm2 = " << norm2.transpose() << endl;
//            double d2 = triPlanes2[p][3]/normNorm2;
//            cout << "d2 = " << d2 << endl;
//            Eigen::Vector3d point2 = norm2 * d2;
//            cout << "point2 = " << point2.transpose() << endl;
//            points2.push_back(point2);
//            dirs2.push_back(norm2);
//        }
//
//        bool compFullConstr = true;
//        Vector7d compTransform = bestTransformPointsAndDirs(points1,
//                                                points2,
//                                                vector<double>(points1.size(), 1.0),
//                                                dirs1,
//                                                dirs2,
//                                                vector<double>(dirs1.size(), 1.0),
//                                                sinValsThresh,
//                                                compFullConstr);
//
//        g2o::SE3Quat t1SE3Quat(transform);
//        g2o::SE3Quat t2SE3Quat(compTransform);
//        g2o::SE3Quat diffSE3Quat = t1SE3Quat.inverse() * t2SE3Quat;
//        Vector6d diffLog = diffSE3Quat.log();
//        double diff = diffLog.transpose() * diffLog;
//
//        if((fullConstr != compFullConstr) ||
//            (fullConstr == compFullConstr && diff > 0.01))
//        {
//            cout << fullConstr << " transform1 = " << transform.transpose() << endl;
//            cout << compFullConstr << " transform2 = " << compTransform.transpose() << endl;
//            cout << "diff = " << diff << endl;
//
//            char a;
//            cin >> a;
//        }
//
//    }
//		cout << "transform = " << transform.transpose() << endl;
}


Vector7d Matching::bestTransformPlanes(const vectorVector4d& planes1,
                                       const vectorVector4d& planes2,
                                       double sinValsThresh,
                                       bool &fullConstr)
{
	Vector7d retTransform;
    retTransform << 0, 0, 0, 0, 0, 0, 1;
	fullConstr = true;

	{
		Eigen::Matrix4d C1 = Eigen::Matrix4d::Zero();
		for(int i = 0; i < planes1.size(); ++i){
			// just normal vectors of the planes
			Eigen::Matrix4d Qt = Misc::matrixQ(Eigen::Quaterniond(0,
														planes1[i](0),
														planes1[i](1),
														planes1[i](2)).normalized()).transpose();
			Eigen::Matrix4d W = Misc::matrixW(Eigen::Quaterniond(0,
														planes2[i](0),
														planes2[i](1),
														planes2[i](2)).normalized());
            
//			cout << "Qt = " << Qt << endl;
//			cout << "W = " << W << endl;
			C1 += -2 * Qt * W;
		}
//		cout << "C1 = " << C1 << endl;
		Eigen::Matrix4d C1t = C1.transpose();
		Eigen::Matrix4d D = -0.5 * (C1 + C1t);
//		cout << "D = " << D << endl;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(D);
        const Eigen::Vector4d &evals = es.eigenvalues();
        const Eigen::Matrix4d &evectors = es.eigenvectors();
        
//        cout << "evals 1 = " << es.eigenvalues() << endl;
//        cout << "evectors 1 = " << es.eigenvectors() << endl;
        
        double maxEval = evals(3);
        for(int i = 0; i < evals.size(); ++i){
            // if constraints imposed by planes do not make computing transformation possible
            if(abs(maxEval + evals(i)) < sinValsThresh){
//                cout << "rot constr 1: " << maxEval << " " << evals(i) << endl;
                fullConstr = false;
                return retTransform;
            }
        }
        
		Eigen::Vector4d rot = evectors.block<4, 1>(0, 3);
		retTransform.tail<4>() = rot;
	}

	{
		Eigen::MatrixXd A;
		A.resize(planes1.size(), 3);
		Eigen::MatrixXd b;
		b.resize(planes1.size(), 1);
		for(int pl = 0; pl < planes1.size(); ++pl){
//			cout << "pl = " << pl << endl;
			Eigen::Vector3d v1 = planes1[pl].head<3>();
			double d1 = -planes1[pl](3);
			v1.normalize();
			double d2 = -planes2[pl](3);

//			cout << "Adding to A" << endl;
			A.block<1, 3>(pl, 0) = v1;
//			cout << "Adding to b" << endl;
			b(pl) = d1 - d2;
		}
  
//        cout << "A 1 = " << A << endl;
//        cout << "b 1 = " << b << endl;
        
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.setThreshold(0.05);
//        cout << "sin vals = " << svd.singularValues() << endl;
//        Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
//        lu.setThreshold(0.1);
//        cout << "trans rank 1 = " << svd.rank() << endl;
        if(svd.rank() < 3){
            fullConstr = false;
            return retTransform;
        }
        svd.setThreshold(Eigen::Default_t());
        Eigen::Vector3d trans = svd.solve(b);
//		cout << "trans 1 = " << trans << endl;
  
		retTransform.head<3>() = trans;
	}

	return retTransform;
}

Vector7d
Matching::bestTransformPointsAndDirs(const vectorVector3d &points1,
                                     const vectorVector3d &points2,
                                     const std::vector<double> &pointsW,
									 const vectorVector3d &dirs1,
                                     const vectorVector3d &dirs2,
                                     const std::vector<double> &dirsW,
									 double sinValsThresh,
                                     bool compTrans,
                                     bool &fullConstrRot,
                                     bool &fullConstrTrans)
{
    Vector7d retTransform;
    fullConstrRot = true;
    fullConstrTrans = true;

    {
        Eigen::Matrix4d C1 = Eigen::Matrix4d::Zero();
        Eigen::Matrix4d C2 = Eigen::Matrix4d::Zero();
        Eigen::Matrix4d C3 = Eigen::Matrix4d::Zero();

//        cout << "points" << endl;
        for(int i = 0; i < points1.size(); ++i){
//            cout << "points1[" << i << "] = " << points1[i].transpose() << endl;
//            cout << "points2[" << i << "] = " << points2[i].transpose() << endl;

            Eigen::Matrix4d Q = Misc::matrixQ(Eigen::Quaterniond(0,
                                                                 0.5 * points1[i](0),
                                                                 0.5 * points1[i](1),
                                                                 0.5 * points1[i](2)));
            Eigen::Matrix4d W = Misc::matrixW(Eigen::Quaterniond(0,
                                                                 0.5 * points2[i](0),
                                                                 0.5 * points2[i](1),
                                                                 0.5 * points2[i](2)));
//			cout << "Q = " << Q << endl;
//			cout << "W = " << W << endl;
            C1 += -2 * Q.transpose() * W * pointsW[i];
            C2 += Eigen::Matrix4d::Identity() * pointsW[i];
            C3 += 2 * (W - Q) * pointsW[i];
        }

//        cout << "dirs" << endl;
        for(int i = 0; i < dirs1.size(); ++i){

            Eigen::Matrix4d Q = Misc::matrixQ(Eigen::Quaterniond(0,
                                                                 dirs1[i](0),
                                                                 dirs1[i](1),
                                                                 dirs1[i](2)).normalized());
            Eigen::Matrix4d W = Misc::matrixW(Eigen::Quaterniond(0,
                                                                 dirs2[i](0),
                                                                 dirs2[i](1),
                                                                 dirs2[i](2)).normalized());
//			cout << "Q = " << Q << endl;
//			cout << "W = " << W << endl;
            C1 += -2 * Q.transpose() * W * dirsW[i];
        }

//		cout << "C1 = " << C1 << endl;
//        cout << "C2 = " << C2 << endl;
//        cout << "C3 = " << C3 << endl;

        Eigen::Matrix4d C1t = C1.transpose();
        Eigen::Matrix4d C2t = C2.transpose();
        Eigen::Matrix4d C3t = C3.transpose();
        Eigen::Matrix4d C2pC2tInv = Eigen::Matrix4d::Zero();
//        Eigen::FullPivLU<Eigen::Matrix4d> lu(C2 + C2t);
        if(points1.size() > 0){
            C2pC2tInv = (C2 + C2t).inverse();
        }
        else{
            fullConstrTrans = false;
        }
        Eigen::Matrix4d A = 0.5 * (C3t * C2pC2tInv * C3 - C1 - C1t);
//		cout << "A = " << A << endl;

//        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
////		cout << "Rank = " << svd.rank() << endl;
//        Eigen::Vector4d sinVals = svd.singularValues();
//
//		cout << "singular values: " << svd.singularValues() << endl;
//		cout << "U = " << svd.matrixU() << endl;
//		cout << "V = " << svd.matrixV() << endl;
//
//        Eigen::FullPivLU<Eigen::MatrixXd> ALu(A);
//        cout << "rank of A = " << ALu.rank() << endl;

        Eigen::EigenSolver<Eigen::Matrix4d> esolver(A);
        Eigen::Vector4d evals = esolver.eigenvalues().real();
        Eigen::Matrix4d evectors = esolver.eigenvectors().real();
        cout << "evals 2 = " << evals << endl;
        cout << "evectors 2 = " << evectors << endl;

        int maxEvalInd = 0;
        double maxEval = 0;
        for(int i = 0; i < evals.size(); ++i){
            if(maxEval < evals(i)){
                maxEval = evals(i);
                maxEvalInd = i;
            }
        }
        // if the greatest eigenvalue has its negative counterpart
        for(int i = 0; i < evals.size(); ++i){
            // if constraints imposed by planes do not make computing transformation possible
            if(abs(maxEval + evals(i)) < sinValsThresh){
                cout << "rot constr 2: " << maxEval << " " << evals(i) << endl;
                
                fullConstrRot = false;
                fullConstrTrans = false;
            }
        }

        //	Eigen::EigenSolver<Eigen::Matrix4d> es(A);
//        cout << "eigenvalues = " << esolver.eigenvalues() << endl;
        //	cout << "eigenvectors = " << es.eigenvectors() << endl;

        Eigen::Vector4d rQuat = evectors.block<4, 1>(0, maxEvalInd);
        rQuat.normalize();
        retTransform.tail<4>() = rQuat;

        if(fullConstrTrans && compTrans) {
            Eigen::Vector4d sQuat = -C2pC2tInv * C3 * rQuat;
//            cout << "sQuat = " << sQuat.transpose() << endl;
//            cout << "sQuat.transpose() * rQuat = " << sQuat.transpose() * rQuat << endl;
            Eigen::Matrix4d Wr = Misc::matrixW(Eigen::Quaterniond(rQuat[3],
                                                                  rQuat[0],
                                                                  rQuat[1],
                                                                  rQuat[2]));
//            cout << "2 * (Wr.transpose() * sQuat) = " << 2 * (Wr.transpose() * sQuat).transpose() << endl;
            retTransform.head<3>() = 2 * (Wr.transpose() * sQuat).head<3>();
        }
        //	for(int pl = 0; pl < planes1.size(); ++pl){
        //		Eigen::Matrix4d Wt = matrixW(Eigen::Quaterniond(rot)).transpose();
        //		Eigen::Matrix4d Q = matrixQ(Eigen::Quaterniond(rot));
        ////		cout << "Wt = " << Wt << endl;
        ////		cout << "Q = " << Q << endl;
        ////		cout << "Wt * Q = " << Wt * Q << endl;
        //		cout << "transposed = " << (Wt * Q * planes1[pl].coeffs()).transpose() << endl;
        //		cout << "plane2 = " << planes2[pl].coeffs().transpose() << endl;
        //	}
    }

	return retTransform;
}

Vector7d Matching::bestTransformPointsDirsDists(const vectorVector3d &points1,
                                                const vectorVector3d &points2,
                                                const std::vector<double> &pointsW,
                                                const vectorVector3d &virtPoints1,
                                                const vectorVector3d &virtPoints2,
                                                const std::vector<double> &virtPointsW,
                                                const vectorVector3d &dirs1,
                                                const vectorVector3d &dirs2,
                                                const std::vector<double> &dirsW,
                                                const std::vector<double> &dists1,
                                                const std::vector<double> &dists2,
                                                const vectorVector3d &distDirs1,
                                                const std::vector<double> &distsW,
                                                const vectorVector3d &distPts1,
                                                const vectorVector3d &distPts2,
                                                const vectorVector3d &distPtsDirs1,
                                                const std::vector<double> &distsPtsW,
                                                double sinValsThresh,
                                                bool &fullConstrRot,
                                                bool &fullConstrTrans)
{
    Vector7d retTransform;
    fullConstrRot = true;
    fullConstrTrans = true;

    vectorVector3d mergedPoints1;
    mergedPoints1.reserve(points1.size() + virtPoints1.size());
    mergedPoints1.insert(mergedPoints1.end(), points1.begin(), points1.end());
    mergedPoints1.insert(mergedPoints1.end(), virtPoints1.begin(), virtPoints1.end());

    vectorVector3d mergedPoints2;
    mergedPoints2.reserve(points2.size() + virtPoints2.size());
    mergedPoints2.insert(mergedPoints2.end(), points2.begin(), points2.end());
    mergedPoints2.insert(mergedPoints2.end(), virtPoints2.begin(), virtPoints2.end());

    vector<double> mergedPointsW;
    mergedPointsW.reserve(pointsW.size() + virtPointsW.size());
    mergedPointsW.insert(mergedPointsW.end(), pointsW.begin(), pointsW.end());
    mergedPointsW.insert(mergedPointsW.end(), virtPointsW.begin(), virtPointsW.end());

    retTransform = bestTransformPointsAndDirs(mergedPoints1,
                                              mergedPoints2,
                                              mergedPointsW,
                                              dirs1,
                                              dirs2,
                                              dirsW,
                                              sinValsThresh,
                                              true,
                                              fullConstrRot,
                                              fullConstrTrans);

//    cout << "fullConstrRot = " << fullConstrRot << endl;
//    cout << "fullConstrTrans = " << fullConstrTrans << endl;
//    cout << "retTransform = " << retTransform.transpose() << endl;
//
//    cout << "dists1 = " << dists1 << endl;
//    cout << "dists2 = " << dists2 << endl;
    {
        int numEq;
        Eigen::Matrix4d Wrt, Qr;
        if(fullConstrRot){
            numEq = dists1.size() + distPts1.size() + 3*points1.size();
            Wrt = Misc::matrixW(Eigen::Quaterniond(retTransform[6],
                                                   retTransform[3],
                                                   retTransform[4],
                                                   retTransform[5]).normalized()).transpose();
            Qr = Misc::matrixQ(Eigen::Quaterniond(retTransform[6],
                                                  retTransform[3],
                                                  retTransform[4],
                                                  retTransform[5]).normalized());
        }
        else{
            numEq = dists1.size();
        }
        if(numEq >= 3) {
            Eigen::MatrixXd A;
            A.resize(numEq, 3);
            Eigen::MatrixXd b;
            b.resize(numEq, 1);
//            cout << "Adding dists" << endl;
            for (int d = 0; d < dists1.size(); ++d) {
//			cout << "d = " << d << endl;
                Eigen::Vector3d n1 = distDirs1[d];
                double d1 = dists1[d];
                double d2 = dists2[d];
//                cout << "d1 = " << d1 << endl;
//                cout << "d2 = " << d2 << endl;

//			cout << "Adding to A" << endl;
                A.block<1, 3>(d, 0) = n1;
//			cout << "Adding to b" << endl;
                b(d) = d1 - d2;
            }
            if (fullConstrRot) {
//                cout << "Adding points" << endl;
                for (int p = 0; p < points1.size(); ++p) {
//			cout << "p = " << p << endl;
                    Eigen::Vector3d p1 = points1[p];


                    Eigen::Vector4d p2quat = Eigen::Vector4d::Zero();
                    p2quat.head<3>() = points2[p];
                    Eigen::Vector4d p2trans = Wrt * Qr * p2quat;

//			cout << "Adding to A" << endl;
                    A.block<3, 3>(dists1.size() + 3 * p, 0) = Eigen::Matrix3d::Identity();
//			cout << "Adding to b" << endl;
                    b.block<3, 1>(dists1.size() + 3 * p, 0) = p1 - p2trans.head<3>();
                }

//                cout << "Adding dists points" << endl;
                for (int dp = 0; dp < distPts1.size(); ++dp) {
                    double d1 = distPtsDirs1[dp].dot(distPts1[dp]);

                    Eigen::Vector4d p2quat = Eigen::Vector4d::Zero();
                    p2quat.head<3>() = distPts2[dp];
                    Eigen::Vector4d p2trans = Wrt * Qr * p2quat;
                    double d2 = distPtsDirs1[dp].dot(p2trans.head<3>());

//			cout << "Adding to A" << endl;
                    A.block<1, 3>(dists1.size() + 3 * points1.size() + dp, 0) = distPtsDirs1[dp];
//			cout << "Adding to b" << endl;
                    b(dists1.size() + 3 * points1.size() + dp) = d1 - d2;
                }
            }

//            cout << "computing full piv LU" << endl;
            cout << "A 2 = " << A << endl;
            cout << "b 2 = " << b << endl;
            Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
            lu.setThreshold(0.1);

            Eigen::MatrixXd matLU = lu.matrixLU().triangularView<Eigen::Upper>();
            cout << "pivots = " << matLU.diagonal() << endl;
//            cout << "A.transpose();" << endl;
//            Eigen::MatrixXd At = A.transpose();
//            cout << "(At * A)" << endl;
//            Eigen::MatrixXd AtAinv = (At * A).inverse();
//            cout << "AtAinv * At * b" << endl;
//            Eigen::Vector3d t = AtAinv * At * b;
//		    cout << "trans = " << trans << endl;
//            cout << "lu.rank() = " << lu.rank() << endl;
            cout << "trans rank 2 = " << lu.rank() << endl;
            if (lu.rank() >= 3) {
                fullConstrTrans = true;

//                cout << "solving equation" << endl;
                Eigen::Vector3d t = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                cout << "t 2 = " << t << endl;
//                cout << "error = " << (A * t - b) << endl;

                retTransform.head<3>() = t;
            } else {
                fullConstrTrans = false;
            }
//                    double relError = (A * t - b).norm() / b.norm();
        }
        else{
            fullConstrTrans = false;
        }
    }

    return retTransform;
}


double Matching::scoreTransformByProjection(const Vector7d& transform,
											const std::vector<std::pair<int, int>> triplet,
											const vectorObjInstance& objInstances1,
											const vectorObjInstance& objInstances2,
											std::vector<double>& intAreaPair,
                                            double intAreaThresh,
                                            double planeEqDiffThresh,
											pcl::visualization::PCLVisualizer::Ptr viewer,
											int viewPort1,
											int viewPort2)
{
	bool allValid = true;
	double allScore = 0.0;
	intAreaPair.clear();

	for(int p = 0; p < triplet.size(); ++p){
//        cout << "p = " << p << endl;

		bool curValid = true;
		double curIntArea = 0.0;

		const ObjInstance& obj1 = objInstances1[triplet[p].first];
		const ObjInstance& obj2 = objInstances2[triplet[p].second];


        double diff = planeEqDiffLogMap(obj1, obj2, transform);

		if(diff > planeEqDiffThresh){
			curValid = false;
		}
		// do another test
		else{

			double interScore = checkConvexHullIntersection(obj1,
                                                            obj2,
                                                            transform,
                                                            curIntArea,
                                                            viewer,
                                                            viewPort1,
                                                            viewPort2);
//			cout << "iou = " << iou << endl;
//			cout << "interScore = " << interScore << endl;
//			intAreaTrans += areaInter;
//            if(curIntArea > 0.0){
//                cout << "curIntArea = " << curIntArea << endl;
//            }
//			cout << "intAreaThresh = " << intAreaThresh << endl;
			if(curIntArea < intAreaThresh){
				curValid = false;
			}
			allScore += interScore;

		}
		if(!curValid){
			allValid = false;
		}
		intAreaPair.push_back(curIntArea);
	}
	if(allValid){
		return allScore/3.0;
	}
	else{
		return -1.0;
	}
}

double Matching::scoreTransformByProjection(const Vector7d &transform,
                                            const vector<PotMatch> curSet,
                                            const vectorObjInstance &objInstances1,
                                            const vectorObjInstance &objInstances2,
                                            std::vector<double> &intAreaPlanes,
                                            std::vector<std::vector<double> > &intLenLines,
                                            double planeEqDiffThresh,
                                            double lineEqDiffThresh,
                                            double intAreaThresh,
                                            double intLenThresh,
                                            pcl::visualization::PCLVisualizer::Ptr viewer,
                                            int viewPort1,
                                            int viewPort2)
{
    bool allValid = true;
    double allScorePlanes = 0.0;
    double allScoreLines = 0.0;
    intAreaPlanes.clear();
    intLenLines.clear();
    
    for(int ch = 0; ch < curSet.size(); ++ch){
        // test plane equations
//        cout << "ch = " << ch << endl;
        
        bool curValid = true;
        double curIntArea = 0.0;
        
        const ObjInstance& obj1 = objInstances1[curSet[ch].plane1];
        const ObjInstance& obj2 = objInstances2[curSet[ch].plane2];
        
        PlaneEstimator obj1PlaneEst = obj1.getPlaneEstimator();
        obj1PlaneEst.transform(g2o::SE3Quat(transform).inverse().toVector());
        const PlaneEstimator &obj2PlaneEst = obj2.getPlaneEstimator();
        
        double diffPlaneEq1 = obj1PlaneEst.distance(obj2PlaneEst);
        double diffPlaneEq2 = obj2PlaneEst.distance(obj1PlaneEst);
        
        if(diffPlaneEq1 > planeEqDiffThresh ||
           diffPlaneEq2 > planeEqDiffThresh)
        {
            curValid = false;
        }
//        double diffPlaneEq = planeEqDiffLogMap(obj1, obj2, transform);
//
//        if(diffPlaneEq > planeEqDiffThresh){
//            curValid = false;
//        }
        
        if(curValid){
            // test line segments equations
            
            for(int l = 0; l < curSet[ch].lineSegs1.size() && curValid; ++l){
                const LineSeg &line1 = obj1.getLineSegs()[curSet[ch].lineSegs1[l]];
                const LineSeg &line2 = obj2.getLineSegs()[curSet[ch].lineSegs2[l]];
                double diffLineEq = lineSegEqDiff(line1,
                                               line2,
                                               transform);
                
                if(diffLineEq > lineEqDiffThresh){
                    curValid = false;
                }
            }
            
            
            if(curValid) {
                // test line segments intersection
                
                intLenLines.push_back(vector<double>());
                for(int l = 0; l < curSet[ch].lineSegs1.size() && curValid; ++l) {
                    const LineSeg &line1 = obj1.getLineSegs()[curSet[ch].lineSegs1[l]];
                    const LineSeg &line2 = obj2.getLineSegs()[curSet[ch].lineSegs2[l]];
                    
                    double curIntLen = 0;
                    checkLineSegIntersection(line1,
                                             line2,
                                             transform,
                                             curIntLen);
                    
                    intLenLines.back().push_back(curIntLen);
                    if(curIntLen < intLenThresh){
                        curValid = false;
                    }
                }
                
                
                if(curValid) {
                    // test planes covex hull intersection
                    
                    double interScore = checkConvexHullIntersection(obj1,
                                                                    obj2,
                                                                    transform,
                                                                    curIntArea,
                                                                    viewer,
                                                                    viewPort1,
                                                                    viewPort2);
//			cout << "iou = " << iou << endl;
//			cout << "interScore = " << interScore << endl;
//			intAreaTrans += areaInter;
//            if(curIntArea > 0.0){
//                cout << "curIntArea = " << curIntArea << endl;
//            }
//			cout << "intAreaThresh = " << intAreaThresh << endl;
//                    if (curIntArea < intAreaThresh) {
//                        curValid = false;
//                    }
                    if(interScore < 0.3){
                        curValid = false;
                    }
                    allScorePlanes += interScore;
                }
            }
            
        }
        if(!curValid){
            allValid = false;
        }
        intAreaPlanes.push_back(curIntArea);
    }
    if(allValid){
        return allScorePlanes/3.0;
    }
    else{
        return -1.0;
    }
}

double Matching::planeEqDiffLogMap(const ObjInstance &obj1,
                                   const ObjInstance &obj2,
                                   const Vector7d &transform)
{
    Eigen::Matrix4d transformMat = g2o::SE3Quat(transform).to_homogeneous_matrix();
//		cout << "transformMat = " << transformMat << endl;

    Eigen::Vector4d curPl1Eq = obj1.getParamRep();
    Eigen::Vector4d curPl2Eq = obj2.getParamRep();
//		cout << "plane 1 eq = " << curPl1Eq.transpose() << endl;
//		cout << "plane 2 eq = " << curPl2Eq.transpose() << endl;
    Eigen::Vector4d curPl1TransEq = transformMat.transpose() * curPl1Eq;
    Misc::normalizeAndUnify(curPl1TransEq);
//		cout << "plane 1 trans eq = " << curPl1TransEq.transpose() << endl;
    Eigen::Quaterniond diffQuat = Eigen::Quaterniond(curPl1TransEq).inverse() * Eigen::Quaterniond(curPl2Eq);
    Eigen::Vector3d diffLogMap = Misc::logMap(diffQuat);
//		cout << "diffLogMap = " << diffLogMap.transpose() << endl;
    double diff = diffLogMap.transpose() * diffLogMap;

    return diff;
}

double Matching::lineSegEqDiff(const LineSeg &lineSeg1,
                               const LineSeg &lineSeg2,
                               const Vector7d &transform)
{
    LineSeg lineSeg1Trans = lineSeg1.transformed(transform);
    
    return lineSeg1Trans.eqDist(lineSeg2);
}

double Matching::checkConvexHullIntersection(const ObjInstance& obj1,
                                            const ObjInstance& obj2,
                                            const Vector7d& transform,
                                             double& intArea,
                                             pcl::visualization::PCLVisualizer::Ptr viewer,
                                             int viewPort1,
                                             int viewPort2)
{
    Eigen::Matrix4d transformMat = g2o::SE3Quat(transform).to_homogeneous_matrix();
//		cout << "transformMat = " << transformMat << endl;
    Eigen::Matrix4d transformMatInv = transformMat.inverse();
    Vector7d transformInv = g2o::SE3Quat(transformMatInv.block<3, 3>(0, 0),
                                         transformMatInv.block<3, 1>(0, 3)).toVector();
    
    const ConcaveHull &obj1Hull = obj1.getHull();
    const ConcaveHull &obj2Hull = obj2.getHull();
    
    ConcaveHull obj1HullTrans = obj1Hull.transform(transformInv);
    
    ConcaveHull interHull = obj2Hull.intersect(obj1HullTrans, 0.0);
    
    intArea = interHull.getTotalArea();
    

//    double iou = intArea/(curPl1ChullArea + curPl2ChullArea - intArea);
    double interScore = max(intArea/obj1Hull.getTotalArea(), intArea/obj2Hull.getTotalArea());
//			cout << "iou = " << iou << endl;
//			cout << "interScore = " << interScore << endl;
//			intAreaTrans += areaInter;

    if(viewer){
//        viewer->removeAllPointClouds();
//        viewer->removeAllShapes();
        
        obj1HullTrans.display(viewer, viewPort1);
        obj2Hull.display(viewer, viewPort1);
        
        interHull.display(viewer, viewPort1, 1.0, 0.0, 0.0);
        
        // time for watching
        viewer->resetStoppedFlag();

//        viewer->initCameraParameters();
//        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
        while (!viewer->wasStopped()){
            viewer->spinOnce (100);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    
        obj1HullTrans.cleanDisplay(viewer, viewPort1);
        obj2Hull.cleanDisplay(viewer, viewPort1);
        interHull.cleanDisplay(viewer, viewPort1);
    }

    return interScore;
}


double Matching::checkLineSegIntersection(const LineSeg &lineSeg1,
                                          const LineSeg &lineSeg2,
                                          const Vector7d &transform,
                                          double &intLen)
{
    LineSeg lineSeg1Trans = lineSeg1.transformed(transform);
    const Eigen::Vector3d l1p1t = lineSeg1Trans.getP1();
    const Eigen::Vector3d l1p2t = lineSeg1Trans.getP2();
    double l1len = (l1p2t - l1p1t).norm();
    const Eigen::Vector3d l2p1 = lineSeg2.getP1();
    const Eigen::Vector3d l2p2 = lineSeg2.getP2();
    double l2len = (l2p2 - l2p1).norm();
    
    double l1p1projt = (l1p1t - l2p1).dot(l2p2 - l2p1);
    double l1p2projt = (l1p2t - l2p1).dot(l2p2 - l2p1);
    
    if(l1p1projt <= l1p2projt){
        double begt = max(0.0, l1p1projt);
        double endt = min(l2len, l1p2projt);
        intLen = max(0.0, endt - begt);
    }
    else{
        double begt = max(0.0, l1p2projt);
        double endt = min(l2len, l1p1projt);
        intLen = max(0.0, endt - begt);
    }
    double intScore = max(intLen/l1len, intLen/l2len);
    
    return intScore;
}

double Matching::evalPoint(const Vector7d &pt,
                           const vectorProbDistKernel &dist)
{
	double res = 0.0;
	for(int k = 0; k < dist.size(); ++k){
		res += dist[k].eval(pt);
	}
	return res;
}

int Matching::countDifferent(const std::set<int> &setIdxs, const vectorObjInstance &objs) {
    UnionFind ufSets(setIdxs.size());
    vector<int> idxs;
    for (const int &mi : setIdxs) {
        idxs.push_back(mi);
    }
    for(int omi1 = 0; omi1 < idxs.size(); ++omi1){
        for(int omi2 = omi1 + 1; omi2 < idxs.size(); ++omi2){
            const ObjInstance &obj1 = objs[idxs[omi1]];
            const ObjInstance &obj2 = objs[idxs[omi2]];
            
            double dist1 = obj1.getPlaneEstimator().distance(obj2.getPlaneEstimator());
            double dist2 = obj2.getPlaneEstimator().distance(obj1.getPlaneEstimator());
            if(dist1 < 0.02 && dist2 < 0.02){
                ufSets.unionSets(omi1, omi2);
            }
        }
    }
    std::set<int> finalSetIdxs;
    for(int omi = 0; omi < idxs.size(); ++omi){
        int setId = ufSets.findSet(omi);
        finalSetIdxs.insert(setId);
    }
    
    return finalSetIdxs.size();
}

Matching::ProbDistKernel::ProbDistKernel(Vector7d ikPt,
										Eigen::Matrix<double, 6, 6> iinfMat,
										double iweight)
	:
	kPt(ikPt),
	infMat(iinfMat),
	weight(iweight),
	kPtSE3Quat(ikPt)
{

}

double Matching::ProbDistKernel::eval(Vector7d pt) const
{
	g2o::SE3Quat ptSE3Quat(pt);
	Vector6d diff = (ptSE3Quat.inverse() * kPtSE3Quat).log();
	double res = weight * exp(-diff.transpose() * infMat * diff);
	return res;
}
