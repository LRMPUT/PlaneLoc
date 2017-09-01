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

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>

#include <g2o/types/slam3d/se3quat.h>
#include <UnionFind.h>

#include "Matching.hpp"
#include "Misc.hpp"

using namespace std;

//Matching::Matching() {
//
//}

Matching::MatchType Matching::matchFrameToMap(const cv::FileStorage &fs,
                                              const std::vector<ObjInstance> &frameObjInstances,
                                              const std::vector<ObjInstance> &mapObjInstances,
                                              std::vector<Vector7d> &bestTrans,
                                              std::vector<double> &bestTransProbs,
                                              std::vector<double> &bestTransFits,
                                              pcl::visualization::PCLVisualizer::Ptr viewer,
                                              int viewPort1,
                                              int viewPort2)
{
	cout << "Matching::matchFrameToMap" << endl;
	double histDistThresh = (double)fs["matching"]["histDistThresh"];
    double planeDistThresh = (double)fs["matching"]["planeDistThresh"];
    double scoreThresh = (double)fs["matching"]["scoreThresh"];
    double sinValsThresh = (double)fs["matching"]["sinValsThresh"];
    double planeEqDiffThresh = (double)fs["matching"]["planeEqDiffThresh"];
    double intAreaThresh = (double)fs["matching"]["intAreaThresh"];

    double shadingLevel = 0.005;

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

		for(int of = 0; of < frameObjInstances.size(); ++of){
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = frameObjInstances[of].getPoints();
//			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr curPlFaded(new pcl::PointCloud<pcl::PointXYZRGBA>(*curPl));
//			for(int pt = 0; pt < curPlFaded->size(); ++pt){
//				curPlFaded->at(pt).a = 100;
//			}
			viewer->addPointCloud(curPl, string("plane1_") + to_string(of), viewPort1);
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
													shadingLevel,
													string("plane1_") + to_string(of),
													viewPort1);
		}
		for(int om = 0; om < mapObjInstances.size(); ++om){
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = mapObjInstances[om].getPoints();
			viewer->addPointCloud(curPl, string("plane2_") + to_string(om), viewPort2);
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
													shadingLevel,
													string("plane2_") + to_string(om),
													viewPort2);
		}

		viewer->initCameraParameters();
		viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
	}

//	vector<int> objTimesMatched1(frameObjInstances.size(), 0);
//	vector<int> objTimesMatched2(mapObjInstances.size(), 0);
	cout << "Adding potential pairs" << endl;
	vector<pair<int, int>> potPairs;
	map<pair<int, int>, double> pairToAppDiff;
	for(int of = 0; of < frameObjInstances.size(); ++of){
		for(int om = 0; om < mapObjInstances.size(); ++om){
			cv::Mat histDiff = cv::abs(frameObjFeats[of] - mapObjFeats[om]);
			float histDist = cv::sum(histDiff)[0];
//            double histDist = cv::compareHist(frameObjFeats[of], mapObjFeats[om], cv::HISTCMP_CHISQR);
			if(histDist < histDistThresh){
//				++objTimesMatched1[of];
//				++objTimesMatched2[om];
				pair<int, int> curPair = make_pair(om, of);
				potPairs.push_back(curPair);
				pairToAppDiff[curPair] = histDist;
			}

//			cout << "dist (" << of << ", " << om << ") = " << histDist << endl;
//			if(viewer){
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														1.0,
//														string("plane1_") + to_string(of),
//														viewPort1);
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														1.0,
//														string("plane2_") + to_string(om),
//														viewPort2);
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
//														string("plane1_") + to_string(of),
//														viewPort1);
//				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//														shadingLevel,
//														string("plane2_") + to_string(om),
//														viewPort2);
//			}
		}
	}
	cout << "potPairs.size() = " << potPairs.size() << endl;
//	cout << "objTimesMatched2 = " << objTimesMatched2 << endl;

    chrono::high_resolution_clock::time_point endAppTime = chrono::high_resolution_clock::now();

	cout << "Adding triplets" << endl;
//	vector<int> objTimesTri1(frameObjInstances.size(), 0);
//	vector<int> objTimesTri2(mapObjInstances.size(), 0);
    vector<vector<double>> frameObjDistances;
    compObjDistances(frameObjInstances, frameObjDistances);
    vector<vector<double>> mapObjDistances;
    compObjDistances(mapObjInstances, mapObjDistances);
	// TODO add smarter triplets generation
	vector<vector<pair<int, int> > > triplets;
	if(potPairs.size() >= 3){
		vector<int> curChoice{0, 1, 2};
		do{
//			cout << "curChoice = " << curChoice << endl;
			vector<pair<int, int>> curTri;
			set<int> planesMapSet;
			set<int> planesFrameSet;
			bool valid = true;
			for(int p = 0; p < 3; ++p){
				curTri.push_back(potPairs[curChoice[p]]);
				// if same plane is in more than one pair than not valid
				if(planesMapSet.count(potPairs[curChoice[p]].first) != 0 ||
					planesFrameSet.count(potPairs[curChoice[p]].second) != 0)
				{
					valid = false;
				}
				planesMapSet.insert(potPairs[curChoice[p]].first);
				planesFrameSet.insert(potPairs[curChoice[p]].second);
			}
            if(valid){
                // if planes are not close enough
                for(int p1 = 0; p1 < curChoice.size(); ++p1) {
                    for (int p2 = p1 + 1; p2 < curChoice.size(); ++p2) {
                        if (mapObjDistances[potPairs[curChoice[p1]].first][potPairs[curChoice[p2]].first] > planeDistThresh) {
                            valid = false;
                        }
                        if (frameObjDistances[potPairs[curChoice[p1]].second][potPairs[curChoice[p2]].second] > planeDistThresh) {
                            valid = false;
                        }
                    }
                }
            }
			if(valid){
//				for(int p = 0; p < curTri.size(); ++p){
//					++objTimesTri1[curTri[p].first];
//					++objTimesTri2[curTri[p].second];
//				}
				triplets.push_back(curTri);
			}
		}while(Misc::nextChoice(curChoice, potPairs.size()));
	}

    chrono::high_resolution_clock::time_point endTripletTime = chrono::high_resolution_clock::now();

	cout << "triplets.size() = " << triplets.size() << endl;

	cout << "computing 3D transforms" << endl;
    vector<Eigen::Vector4d> planesMap;
    for(int pl = 0; pl < mapObjInstances.size(); ++pl){
        planesMap.push_back(mapObjInstances[pl].getNormal());
    }
	vector<Eigen::Vector4d> planesFrame;
	for(int pl = 0; pl < frameObjInstances.size(); ++pl){
		planesFrame.push_back(frameObjInstances[pl].getNormal());
	}
	std::vector<ValidTransform> transforms;

	for(int t = 0; t < triplets.size(); ++t){
//		cout << "t = " << t << endl;

		Vector7d curTransform;
		bool fullConstr;
		comp3DTransform(planesMap,
                        planesFrame,
						triplets[t],
						curTransform,
                        sinValsThresh,
						fullConstr);

		bool isAdded = false;
		if(fullConstr){
//			cout << "fullConstr = " << fullConstr << endl;
//            {
//                g2o::SE3Quat curTransSE3Quat(curTransform);
//                Vector6d diffLog = (curGtSE3Quat.inverse() * curTransSE3Quat).log();
//                double diff = diffLog.transpose() * diffLog;
//                if(diff < 0.1){
//                    cout << "curTransform =" << curTransform << endl;
//                }
//            }
			vector<double> intAreas;
			double score = scoreTransformByProjection(curTransform,
                                                      triplets[t],
                                                      mapObjInstances,
                                                      frameObjInstances,
                                                      intAreas,
                                                      intAreaThresh,
                                                      planeEqDiffThresh/*,
													  viewer,
													  viewPort1,
													  viewPort2*/);
            if(score > 0.0) {
                cout << "score = " << score << endl;
            }
			if(score > scoreThresh){
//				double score = scoreTransformByProjection(curTransform,
//														triplets[t],
//														objInstances1,
//														objInstances2,
//														intArea,
//														viewer,
//														viewPort1,
//														viewPort2);

//				cout << "curTransform = " << endl << endl << curTransform.transpose() << endl << endl;
//				cout << "score = " << score << endl;
//				cout << "intArea = " << intArea << endl;
//				int numPotTri1 = 1;
//				int numPotTri2 = 1;
//				for(int p = 0; p < triplets[t].size(); ++p){
//					numPotTri1 += objTimesMatched1[triplets[t][p].first];
//					numPotTri2 += objTimesMatched2[triplets[t][p].second];
//				}
//				double distinctScore = score/max(numPotTri1, numPotTri2);
//				double distinctScore = 1.0;
////				cout << "distinctScore = " << distinctScore << endl;
////				cout << endl << "Added transform" << endl << endl;
//				double curTransScore = 0.0;
//				for(int p = 0; p < triplets[t].size(); ++p){
////					curTransScore += intArea[p] / objTimesTri2[triplets[t][p].second];
//					curTransScore += intAreas[p] / objTimesMatched2[triplets[t][p].second];
////					curTransScore += 1.0 / objTimesMatched2[triplets[t][p].second];
//				}
//				cout << "curTransScore = " << curTransScore << endl;

//				transforms.push_back(curTransform);
//				transformScores.push_back(curTransScore);
				vector<double> appDiffs;
				for(int p = 0; p < triplets[t].size(); ++p){
					appDiffs.push_back(pairToAppDiff.at(triplets[t][p]));
				}
				transforms.emplace_back(curTransform,
										0.0,
										triplets[t],
										intAreas,
										appDiffs);
				isAdded = true;
			}
		}

//		if(viewer && isAdded){
//			for(int p = 0; p < triplets[t].size(); ++p){
//				int om = triplets[t][p].first;
//				int of = triplets[t][p].second;
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
//			for(int p = 0; p < triplets[t].size(); ++p){
//				int om = triplets[t][p].first;
//				int of = triplets[t][p].second;
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
		vector<float> frameObjInvWeights(transforms[t].triplet.size(), 1.0);
		vector<float> mapObjInvWeights(transforms[t].triplet.size(), 1.0);
		for(int tc = 0; tc < transforms.size(); ++tc){
			for(int p = 0; p < transforms[t].triplet.size(); ++p){
				g2o::SE3Quat trans(transforms[t].transform);
				g2o::SE3Quat transComp(transforms[tc].transform);
				g2o::SE3Quat diff = trans.inverse() * transComp;
				Vector6d logMapDiff = diff.log();
				double dist = logMapDiff.transpose() * logMapDiff;
				for(int pc = 0; pc < transforms[tc].triplet.size(); ++pc){
                    if(transforms[t].triplet[p].first == transforms[tc].triplet[pc].first){
                        mapObjInvWeights[p] += exp(-dist);
                    }
                    if(transforms[t].triplet[p].second == transforms[tc].triplet[pc].second){
						frameObjInvWeights[p] += exp(-dist);
					}
				}
			}
		}
//		cout << "intAreas = " << transforms[t].intAreas << endl;
//		cout << "mapObjInvWeights = " << mapObjInvWeights << endl;
		double curScore = 0.0;
		for(int p = 0; p < transforms[t].intAreas.size(); ++p){
//            cout << "exp(-transforms[t].appDiffs[p]) = " << exp(-transforms[t].appDiffs[p]) << endl;
			curScore += transforms[t].intAreas[p]/frameObjInvWeights[p]*exp(-transforms[t].appDiffs[p]);
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
		// construct probability distribution using gaussian kernels
		vector<ProbDistKernel> dist;
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
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr framePc(new pcl::PointCloud<pcl::PointXYZRGB>());
            for(int of = 0; of < frameObjInstances.size(); ++of){
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr curFramePc = frameObjInstances[of].getPoints();
                framePc->insert(framePc->end(), curFramePc->begin(), curFramePc->end());
            }
			
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapPcRed(new pcl::PointCloud<pcl::PointXYZRGB>());
			pcl::copyPointCloud(*mapPc, *mapPcRed);
			for(int pt = 0; pt < mapPcRed->size(); ++pt){
				mapPcRed->at(pt).r = mapPcRed->at(pt).r * 0.5 + 255 * 0.5;
				mapPcRed->at(pt).g = mapPcRed->at(pt).g * 0.5 + 0 * 0.5;
				mapPcRed->at(pt).b = mapPcRed->at(pt).b * 0.5 + 0 * 0.5;
			}
//            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
//            kdTree.setInputCloud(mapPc);
			pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree(0.005);
            octree.setInputCloud(mapPc);
            octree.addPointsFromInputCloud();
//            cout << "octree.getEpsilon() = " << octree.getEpsilon() << endl;

            for(int t = 0; t < bestTrans.size(); ++t) {
                cout << "fit score on transformation " << t << endl;

                g2o::SE3Quat curTransSE3Quat(bestTrans[t]);
                Eigen::Matrix4d curTransMat = curTransSE3Quat.to_homogeneous_matrix();

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr framePcTrans(new pcl::PointCloud<pcl::PointXYZRGB>());
                pcl::transformPointCloud(*framePc, *framePcTrans, curTransMat);

                vector<int> nnIndices(1);
                std::vector<float> nnDists(1);
                double fitScore = 0.0;
                int ptCnt = 0;
                for(int p = 0; p < framePcTrans->size(); ++p){
//                    kdTree.nearestKSearch(framePcTrans->at(p), 1, nnIndices, nnDists);

//                    int nnInd;
//                    float nnDist;
//                    octree.approxNearestSearch(framePcTrans->at(p), nnInd, nnDist);
//
                    octree.nearestKSearch(framePcTrans->at(p), 1, nnIndices, nnDists);

                    fitScore += nnDists[0];
//                    fitScore += nnDist;
                    ++ptCnt;
                }
                if(ptCnt > 0){
                    fitScore /= ptCnt;
                }

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

//                if(fitScore < 0.1){
//                    newBestTrans.push_back(bestTrans[t]);
//                    newBestTransProbs.push_back(bestTransProbs[t]);
//                }
                if(viewer) {
                    viewer->removeAllPointClouds(viewPort1);
                    viewer->removeAllShapes(viewPort1);
                    viewer->removeAllPointClouds(viewPort2);
                    viewer->removeAllShapes(viewPort2);
					
					viewer->addPointCloud(mapPcRed, "cloud_map", viewPort1);
//					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//															 1.0, 0.0, 0.0,
//															 "cloud_map",
//															 viewPort1);

                    viewer->resetStoppedFlag();
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                    viewer->spinOnce(100);
                    while (!viewer->wasStopped()) {
                        viewer->spinOnce(100);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
	
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
															 0.5,
															 "cloud_map",
															 viewPort1);
	
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr framePcTransGreen(new pcl::PointCloud<pcl::PointXYZRGB>());
					pcl::copyPointCloud(*framePcTrans, *framePcTransGreen);
					for(int pt = 0; pt < framePcTransGreen->size(); ++pt){
						framePcTransGreen->at(pt).r = framePcTransGreen->at(pt).r * 0.5 + 0 * 0.5;
						framePcTransGreen->at(pt).g = framePcTransGreen->at(pt).g * 0.5 + 255 * 0.5;
						framePcTransGreen->at(pt).b = framePcTransGreen->at(pt).b * 0.5 + 0 * 0.5;
					}
					viewer->addPointCloud(framePcTransGreen, "cloud_out", viewPort1);
//					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//															 0.0, 1.0, 0.0,
//															 "cloud_out",
//															 viewPort1);
	
					viewer->resetStoppedFlag();
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
    static int totalCnt = 0;

    totalAppTime += chrono::duration_cast<chrono::milliseconds>(endAppTime - startTime);
    totalTripletsTime += chrono::duration_cast<chrono::milliseconds>(endTripletTime - endAppTime);
    totalTransformTime += chrono::duration_cast<chrono::milliseconds>(endTransformTime - endTripletTime);
    totalScoreTime += chrono::duration_cast<chrono::milliseconds>(endScoreTime - endTransformTime);
    totalFitTime += chrono::duration_cast<chrono::milliseconds>(endTime - endScoreTime);
    ++totalCnt;

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


void Matching::compObjFeatures(const std::vector<ObjInstance>& objInstances,
							std::vector<cv::Mat>& objFeats)
{
	// color histogram
	int hbins = 32;
	int sbins = 32;
	int histSizeH[] = {hbins};
	int histSizeS[] = {sbins};
	float hranges[] = {0, 180};
	float sranges[] = {0, 256};
	const float* rangesH[] = {hranges};
	const float* rangesS[] = {sranges};
	int channelsH[] = {0};
	int channelsS[] = {0};

	for(int o = 0; o < objInstances.size(); ++o){
		const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pts = objInstances[o].getPoints();
		int npts = pts->size();
		cv::Mat matPts(1, npts, CV_8UC3);
		for(int p = 0; p < npts; ++p){
			matPts.at<cv::Vec3b>(p)[0] = pts->at(p).r;
			matPts.at<cv::Vec3b>(p)[1] = pts->at(p).g;
			matPts.at<cv::Vec3b>(p)[2] = pts->at(p).b;
		}
		cv::cvtColor(matPts, matPts, cv::COLOR_RGB2HSV);
		cv::Mat hist;
		cv::calcHist(&matPts,
					1,
					channelsH,
					cv::Mat(),
					hist,
					1,
					histSizeH,
					rangesH);
		// normalization
		hist /= npts;
		hist.reshape(1,hbins);

		cv::Mat histS;
		cv::calcHist(&matPts,
					1,
					channelsS,
					cv::Mat(),
					histS,
					1,
					histSizeS,
					rangesS);
		// normalization
		histS /= npts;
		histS.reshape(1,sbins);

		// add S part of histogram
		hist.push_back(histS);

		objFeats.push_back(hist);
	}
}


void Matching::compObjDistances(const std::vector<ObjInstance>& objInstances,
                                std::vector<std::vector<double>>& objDistances)
{
    objDistances.resize(objInstances.size(), vector<double>(objInstances.size(), 0));
//    cout << "objDistances.size() = " << objDistances.size()
//         << ", objDistances.front().size() = " << objDistances.front().size() << endl;
    for(int o1 = 0; o1 < objInstances.size(); ++o1){
//        cout << "o1 = " << o1 << endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr o1hull = objInstances[o1].getConvexHull();
        for(int o2 = o1 + 1; o2 < objInstances.size(); ++o2){
//            cout << "o2 = " << o2 << endl;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr o2hull = objInstances[o2].getConvexHull();
            double minDist = std::numeric_limits<double>::max();
            // O(n^2) is enough for a small number of points
            for(int p1 = 0; p1 < o1hull->size(); ++p1){
                for(int p2 = 0; p2 < o2hull->size(); ++p2){
                    Eigen::Vector3f pt1 = o1hull->at(p1).getVector3fMap();
                    Eigen::Vector3f pt2 = o2hull->at(p2).getVector3fMap();
                    double curDist = (pt1 - pt2).norm();
                    minDist = min(curDist, curDist);
                }
            }
//			cout << "o1 = " << o1 << ", o2 = " << o2 << endl;
            objDistances[o1][o2] = minDist;
            objDistances[o2][o1] = minDist;
        }
    }
}

void Matching::comp3DTransform(const std::vector<Eigen::Vector4d>& planes1,
								const std::vector<Eigen::Vector4d>& planes2,
								const std::vector<std::pair<int, int>>& triplet,
								Vector7d& transform,
                                double sinValsThresh,
								bool& fullConstr)
{

		std::vector<Eigen::Vector4d> triPlanes1;
		std::vector<Eigen::Vector4d> triPlanes2;
		for(int p = 0; p < triplet.size(); ++p){
//			cout << "triplets[t][p] = (" << triplet[p].first << ", " << triplet[p].second << ")" << endl;
			triPlanes1.push_back(planes1[triplet[p].first]);
			triPlanes2.push_back(planes2[triplet[p].second]);
		}
		transform = bestTransformPlanes(triPlanes1, triPlanes2, sinValsThresh, fullConstr);
//		cout << "transform = " << transform.transpose() << endl;
}


Vector7d Matching::bestTransformPlanes(const std::vector<Eigen::Vector4d> planes1,
                                       const std::vector<Eigen::Vector4d> planes2,
                                       double sinValsThresh,
                                       bool &fullConstr)
{
	Vector7d retTransform;
	fullConstr = true;

//	cout << "planes1 = " << endl;
//	for(int pl = 0; pl < planes1.size(); ++pl){
//		cout << planes1[pl].coeffs().transpose() << endl;
//	}
//	cout << "planes2 = " << endl;
//	for(int pl = 0; pl < planes2.size(); ++pl){
//		cout << planes2[pl].coeffs().transpose() << endl;
//	}

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
		Eigen::Matrix4d A = -0.5 * (C1 + C1t);
//		cout << "A = " << A << endl;

		Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
//		cout << "Rank = " << svd.rank() << endl;
		Eigen::Vector4d sinVals = svd.singularValues();

//		cout << "singular values: " << svd.singularValues() << endl;
//		cout << "U = " << svd.matrixU() << endl;
//		cout << "V = " << svd.matrixV() << endl;
		int numMaxEvals = 0;
		int maxEvalInd = 0;
		double maxEval = 0;
		for(int i = 0; i < svd.singularValues().size(); ++i){
			// if values the same for this threshold
			if(fabs(maxEval - sinVals(i)) < sinValsThresh){
				++numMaxEvals;
			}
			else if(maxEval < sinVals(i)){
				numMaxEvals = 1;
				maxEval = svd.singularValues()[i];
				maxEvalInd = i;
			}
		}
		// if constraints imposed by planes do not make computing transformation possible
		if(numMaxEvals > 1){
			fullConstr = false;
		}

	//	Eigen::EigenSolver<Eigen::Matrix4d> es(A);
	//	cout << "eigenvalues = " << es.eigenvalues() << endl;
	//	cout << "eigenvectors = " << es.eigenvectors() << endl;

		Eigen::Vector4d rot = svd.matrixU().block<4, 1>(0, maxEvalInd);
		retTransform.tail<4>() = rot;
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

	{
		Eigen::MatrixXd A;
		A.resize(planes1.size(), 3);
		Eigen::MatrixXd Z;
		Z.resize(1, planes1.size());
		for(int pl = 0; pl < planes1.size(); ++pl){
//			cout << "pl = " << pl << endl;
			Eigen::Vector3d v1 = planes1[pl].head<3>();
			double d1 = planes1[pl](3) / v1.norm();
			v1.normalize();
			double d2 = planes2[pl](3) / planes2[pl].head<3>().norm();

//			cout << "Adding to A" << endl;
			A.block<1, 3>(pl, 0) = v1;
//			cout << "Adding to Z" << endl;
			Z(pl) = d2 - d1;
		}
//		cout << "A.transpose();" << endl;
		Eigen::MatrixXd At = A.transpose();
//		cout << "(At * A).inverse()" << endl;
		Eigen::MatrixXd AtAinv = (At * A).inverse();
//		cout << "AtAinv * At * Z" << endl;
		Eigen::Vector3d trans = AtAinv * At * Z;
//		cout << "trans = " << trans << endl;
		retTransform.head<3>() = trans;
	}

	return retTransform;
}

double Matching::scoreTransformByProjection(const Vector7d& transform,
											const std::vector<std::pair<int, int>> triplet,
											const std::vector<ObjInstance>& objInstances1,
											const std::vector<ObjInstance>& objInstances2,
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


        double diff = planeDiffLogMap(obj1, obj2, transform);

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
            if(curIntArea > 0.0){
                cout << "curIntArea = " << curIntArea << endl;
            }
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

double Matching::planeDiffLogMap(const ObjInstance& obj1,
                                 const ObjInstance& obj2,
                                 const Vector7d& transform)
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

    Eigen::Vector4d curPl2Eq = obj2.getParamRep();

    double curPl1ChullArea;
    pcl::Vertices curPl1ChullPolygon;
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl1Chull = obj1.getConvexHull(curPl1ChullPolygon, curPl1ChullArea);

    double curPl2ChullArea;
    pcl::Vertices curPl2ChullPolygon;
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl2Chull = obj2.getConvexHull(curPl2ChullPolygon, curPl2ChullArea);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl1ChullTrans(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*curPl1Chull, *curPl1ChullTrans, transformMatInv);

    if(viewer){
        viewer->removeAllPointClouds(viewPort1);
        viewer->removeAllShapes(viewPort1);
        viewer->removeAllPointClouds(viewPort2);
        viewer->removeAllShapes(viewPort2);


        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> pcColHandler(curPl1Chull, 1.0, 0, 0);

        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl1 = obj1.getPoints();
//        viewer->addPointCloud(curPl1, string("plane1"), viewPort1);

//        viewer->addPointCloud(curPl1Chull, pcColHandler, string("plane1 chull"), viewPort1);
        pcl::Vertices curPl1ChullPolygonMesh = curPl1ChullPolygon;
        curPl1ChullPolygonMesh.vertices.push_back(curPl1ChullPolygonMesh.vertices.front());
//        viewer->addPolygonMesh<pcl::PointXYZRGB>(curPl1Chull, vector<pcl::Vertices>{curPl1ChullPolygonMesh}, string("plane1 poly"), viewPort1);
//        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
//                                                 0,
//                                                 1.0,
//                                                 0,
//                                                 string("plane1 poly"),
//                                                 viewPort1);
//        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
//                                                 0.2,
//                                                 string("plane1 poly"),
//                                                 viewPort1);

        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl2 = obj2.getPoints();
        viewer->addPointCloud(curPl2, string("plane2"), viewPort1);

        pcl::Vertices curPl2ChullPolygonMesh = curPl2ChullPolygon;
        curPl2ChullPolygonMesh.vertices.push_back(curPl2ChullPolygonMesh.vertices.front());
        viewer->addPointCloud(curPl2Chull, pcColHandler, string("plane2 chull"), viewPort2);
        viewer->addPolygonMesh<pcl::PointXYZRGB>(curPl2Chull, vector<pcl::Vertices>{curPl2ChullPolygonMesh}, string("plane2 poly"), viewPort2);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 0,
                                                 1.0,
                                                 0,
                                                 string("plane2 poly"),
                                                 viewPort2);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                 0.5,
                                                 string("plane2 poly"),
                                                 viewPort2);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl1Trans(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::transformPointCloud(*curPl1, *curPl1Trans, transformMatInv);

        viewer->addPointCloud(curPl1Trans, string("plane1 trans"), viewPort1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                 0.2,
                                                 string("plane1 trans"),
                                                 viewPort1);

        viewer->addPointCloud(curPl1ChullTrans, pcColHandler, string("plane1 chull trans"), viewPort2);
        viewer->addPolygonMesh<pcl::PointXYZRGB>(curPl1ChullTrans, vector<pcl::Vertices>{curPl1ChullPolygonMesh}, string("plane1 poly trans"), viewPort2);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 0,
                                                 1.0,
                                                 0,
                                                 string("plane1 poly trans"),
                                                 viewPort2);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                 0.5,
                                                 string("plane1 poly trans"),
                                                 viewPort2);
    }


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullInter(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::Vertices polyInter;
//			double areaInter = 0.0;

    intArea = 0.0;
    intersectConvexHulls(curPl1ChullTrans,
                         curPl1ChullPolygon,
                         curPl2Chull,
                         curPl2ChullPolygon,
                         curPl2Eq,
                         chullInter,
                         polyInter,
                         intArea/*,
								viewer,
								viewPort1,
								viewPort2*/);

//			cout << "chullInter->size() = " << chullInter->size() << endl;
//			cout << "polyInter.vertices = " << polyInter.vertices << endl;
    double iou = intArea/(curPl1ChullArea + curPl2ChullArea - intArea);
    double interScore = max(intArea/curPl1ChullArea, intArea/curPl2ChullArea);
//			cout << "iou = " << iou << endl;
//			cout << "interScore = " << interScore << endl;
//			intAreaTrans += areaInter;

    if(viewer){
        if(chullInter->size() > 0){
            cout << "curIntArea = " << intArea << endl;
            cout << "curPl1ChullArea = " << curPl1ChullArea << endl;
            cout << "curPl2ChullArea = " << curPl2ChullArea << endl;

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> pcColHandler(curPl1Chull, 255, 0, 0);

            pcl::Vertices polyInterMesh = polyInter;
            polyInterMesh.vertices.push_back(polyInterMesh.vertices.front());
            viewer->addPointCloud(chullInter, pcColHandler, string("chull inter"), viewPort2);
            viewer->addPolygonMesh<pcl::PointXYZRGB>(chullInter, vector<pcl::Vertices>{polyInterMesh}, string("chull inter poly"), viewPort2);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                     1.0,
                                                     0,
                                                     0,
                                                     string("chull inter poly"),
                                                     viewPort2);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     0.9,
                                                     string("chull inter poly"),
                                                     viewPort2);
        }

        // time for watching
        viewer->resetStoppedFlag();

        viewer->initCameraParameters();
        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
        while (!viewer->wasStopped()){
            viewer->spinOnce (100);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    return interScore;
}

double Matching::evalPoint(Vector7d pt,
							const std::vector<ProbDistKernel>& dist)
{
	double res = 0.0;
	for(int k = 0; k < dist.size(); ++k){
		res += dist[k].eval(pt);
	}
	return res;
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

void Matching::intersectConvexHulls(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr chull1,
									const pcl::Vertices& poly1,
									const pcl::PointCloud<pcl::PointXYZRGB>::Ptr chull2,
									const pcl::Vertices& poly2,
									const Eigen::Vector4d planeEq,
									pcl::PointCloud<pcl::PointXYZRGB>::Ptr chullRes,
									pcl::Vertices& polyRes,
									double& areaRes,
									pcl::visualization::PCLVisualizer::Ptr viewer,
									int viewPort1,
									int viewPort2)
{
	static constexpr double eps = 1e-6;
	areaRes = 0.0;

	if(poly1.vertices.size() < 3 || poly2.vertices.size() < 3){
		return;
	}

	Eigen::Vector3d plNormal = planeEq.head<3>();
	double plNormalNorm = plNormal.norm();
	double plD = planeEq(3) / plNormalNorm;
	plNormal /= plNormalNorm;

	// point on plane nearest to origin
	Eigen::Vector3d origin = plNormal * (-plD);
	Eigen::Vector3d xAxis, yAxis;
	//if normal vector is not parallel to global x axis
	if(plNormal.cross(Eigen::Vector3d(1.0, 0.0, 0.0)).norm() > 1e-2){
		// plane x axis as a cross product - always perpendicular to normal vector
		xAxis = plNormal.cross(Eigen::Vector3d(1.0, 0.0, 0.0));
		xAxis.normalize();
		yAxis = plNormal.cross(xAxis);
	}
	else{
		xAxis = plNormal.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
		xAxis.normalize();
		yAxis = plNormal.cross(xAxis);
	}

	if(viewer){
		Eigen::Affine3f trans = Eigen::Affine3f::Identity();
		trans.matrix().block<3, 1>(0, 3) = origin.cast<float>();
		trans.matrix().block<3, 1>(0, 0) = xAxis.cast<float>();
		trans.matrix().block<3, 1>(0, 1) = yAxis.cast<float>();
		trans.matrix().block<3, 1>(0, 2) = plNormal.cast<float>();
		//		trans.fromPositionOrientationScale(, rot, 1.0);
		viewer->addCoordinateSystem(0.5, trans, "plane coord", viewPort2);

		viewer->initCameraParameters();
		viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
	}
//	cout << "chull1->size() = " << chull1->size() << endl;
//	cout << "poly1.vertices = " << poly1.vertices << endl;
//	cout << "chull2->size() = " << chull2->size() << endl;
//	cout << "poly2.vertices = " << poly2.vertices << endl;

	vector<Eigen::Vector2d> chull2d1;
	vector<Eigen::Vector2d> chull2d2;
	for(int i = 0; i < poly1.vertices.size(); ++i){
		Eigen::Vector3d chPt = chull1->at(poly1.vertices[i]).getVector3fMap().cast<double>();
		Eigen::Vector2d projPt;
		projPt.x() = (chPt - origin).dot(xAxis);
		projPt.y() = (chPt - origin).dot(yAxis);
		chull2d1.push_back(projPt);
	}
	makeCclockwise(chull2d1);
	for(int i = 0; i < poly2.vertices.size(); ++i){
		Eigen::Vector3d chPt = chull2->at(poly2.vertices[i]).getVector3fMap().cast<double>();
		Eigen::Vector2d projPt;
		projPt.x() = (chPt - origin).dot(xAxis);
		projPt.y() = (chPt - origin).dot(yAxis);
		chull2d2.push_back(projPt);
	}
	makeCclockwise(chull2d2);

	bool isFirstPoint = true;
	bool isAlwaysCross1H2Pos = true;
	bool isAlwaysCross2H1Pos = true;

	Inside insideFlag = Inside::Unknown;
	int i1 = 0;
	int i2 = 0;
	int adv1 = 0;
	int adv2 = 0;
	vector<Eigen::Vector2d> intChull;
	do{
		Eigen::Vector2d begPt1 = chull2d1[(i1 - 1 + chull2d1.size()) % chull2d1.size()];
		Eigen::Vector2d endPt1 = chull2d1[i1 % chull2d1.size()];
		Eigen::Vector2d begPt2 = chull2d2[(i2 - 1 + chull2d2.size()) % chull2d2.size()];
		Eigen::Vector2d endPt2 = chull2d2[i2 % chull2d2.size()];

		Eigen::Vector2d edge1 = endPt1 - begPt1;
		Eigen::Vector2d edge2 = endPt2 - begPt2;

		double cross = cross2d(edge1, edge2);
		double cross1H2 = cross2d(endPt2 - begPt2, endPt1 - begPt2);
		double cross2H1 = cross2d(endPt1 - begPt1, endPt2 - begPt1);

		if(cross1H2 < 0.0){
			isAlwaysCross1H2Pos = false;
		}
		if(cross2H1 < 0.0){
			isAlwaysCross2H1Pos = false;
		}

//		cout << "cross = " << cross << endl;
//		cout << "cross1H2 = " << cross1H2 << endl;
//		cout << "cross2H1 = " << cross2H1 << endl;

		Eigen::Vector2d intPt1;
		Eigen::Vector2d intPt2;
		SegIntType intType = intersectLineSegments(begPt1, endPt1, begPt2, endPt2, intPt1, intPt2, eps);
//		cout << "intType = " << intType << endl;
//		cout << "isFirstPoint = " << isFirstPoint << endl;
		if(intType == SegIntType::One || intType == SegIntType::Vertex){
//			cout << "intType == SegIntType::One || intType == SegIntType::Vertex" << endl;
			if(insideFlag == Inside::Unknown && isFirstPoint){
//				cout << "insideFlag == Inside::Unknown && isFirstPoint" << endl;
				adv1 = adv2 = 0;
				isFirstPoint = false;
			}
//			cout << "adding point (" << intPt1.transpose() << ")" << endl;
			intChull.push_back(intPt1);
			if(viewer){
				Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
				viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
			}
			insideFlag = newInsideFlag(insideFlag, intPt1, cross1H2, cross2H1, eps);
//			if(insideFlag == Inside::Unknown){
//				cout << "insideFlag = Unknown" << endl;
//			}
//			else if(insideFlag == Inside::First){
//				cout << "insideFlag = First" << endl;
//			}
//			else{
//				cout << "insideFlag = Second" << endl;
//			}
		}

		// edge1 and edge2 overlap and oppositely oriented
		if((intType == SegIntType::Collinear) && (edge1.dot(edge2) < 0.0 - eps)){
//			cout << "edge1 and edge2 overlap and oppositely oriented" << endl;
			intChull.push_back(intPt1);
			if(viewer){
				Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
				viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
			}
			intChull.push_back(intPt2);
			if(viewer){
				Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
				viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
			}
			break;
		}
		// edge1 and edge2 parallel and separated
		if((fabs(cross) <= eps) && (cross1H2 < 0.0 - eps) && (cross2H1 < 0.0 - eps)){
//			cout << "edge1 and edge2 parallel and separated" << endl;
			//cout << "disjoint" << endl;
			break;
		}
		// edge1 and edge2 parallel and collinear
		else if((fabs(cross) <= eps) && (fabs(cross1H2) <= eps) && (fabs(cross2H1) <= eps)){
//			cout << "edge1 and edge2 parallel and collinear" << endl;
			if(insideFlag == Inside::First){
				i2++;
				adv2++;
			}
			else{
				i1++;
				adv1++;
			}
		}
		else if(cross >= 0.0 + eps){
//			cout << "cross >= 0.0 + eps" << endl;
//			cout << "cross2H1 = " << cross2H1 << endl;
			if(cross2H1 > 0.0 + eps){
				i1++;
				adv1++;
				if(insideFlag == Inside::First){
					intChull.push_back(endPt1);
					if(viewer){
						Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
						viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
					}
//					cout << "adding endPt1" << endl;
				}
			}
			else{
				i2++;
				adv2++;
				if(insideFlag == Inside::Second){
					intChull.push_back(endPt2);
					if(viewer){
						Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
						viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
					}
//					cout << "adding endPt2" << endl;
				}
			}
		}
		else /* if(cross < 0.0 - eps)*/{
//			cout << "cross < 0.0 - eps" << endl;
//			cout << "cross1H2 = " << cross1H2 << endl;
			if(cross1H2 > 0.0 + eps){
				i2++;
				adv2++;
				if(insideFlag == Inside::Second){
					intChull.push_back(endPt2);
					if(viewer){
						Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
						viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
					}
//					cout << "adding endPt2" << endl;
				}
			}
			else{
				i1++;
				adv1++;
				if(insideFlag == Inside::First){
					intChull.push_back(endPt1);
					if(viewer){
						Eigen::Vector3d intPt = origin + intChull.back().x() * xAxis + intChull.back().y() * yAxis;
						viewer->addSphere(pcl::PointXYZ(intPt.x(), intPt.y(), intPt.z()), 0.04, 1, 0, 0, string("int point") + to_string(intChull.size()), viewPort2);
					}
//					cout << "adding endPt1" << endl;
				}
			}
		}

		if(viewer){
			Eigen::Vector3d begPt13d = origin + begPt1.x() * xAxis + begPt1.y() * yAxis;
			Eigen::Vector3d endPt13d = origin + endPt1.x() * xAxis + endPt1.y() * yAxis;
			Eigen::Vector3d begPt23d = origin + begPt2.x() * xAxis + begPt2.y() * yAxis;
			Eigen::Vector3d endPt23d = origin + endPt2.x() * xAxis + endPt2.y() * yAxis;

			viewer->addArrow(pcl::PointXYZ(endPt13d.x(), endPt13d.y(), endPt13d.z()), pcl::PointXYZ(begPt13d.x(), begPt13d.y(), begPt13d.z()), 1, 0, 0, false, "edge1", viewPort2);
			viewer->addArrow(pcl::PointXYZ(endPt23d.x(), endPt23d.y(), endPt23d.z()), pcl::PointXYZ(begPt23d.x(), begPt23d.y(), begPt23d.z()), 0, 0, 1, false, "edge2", viewPort2);

			// time for watching
			viewer->resetStoppedFlag();

			while (!viewer->wasStopped()){
				viewer->spinOnce (100);
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}

			viewer->removeShape("edge1", viewPort2);
			viewer->removeShape("edge2", viewPort2);
		}

	}while(((adv1 < chull2d1.size()) || (adv2 < chull2d2.size())) && (adv1 < 2*chull2d1.size()) && (adv2 < 2*chull2d2.size()));

	if(insideFlag == Inside::Unknown){
		// polygons do not intersect
		// check if one is inside the other or if their intersection is empty
		// first inside second -> first is a convex hull
		if(isAlwaysCross1H2Pos){
			intChull = chull2d1;
		}
		// second inside first -> second is a convex hull
		else if(isAlwaysCross2H1Pos){
			intChull = chull2d2;
		}
		// disjoint -> convex hull is empty
		else{
			// disjoint
		}
	}

	for(int i = 0; i < intChull.size(); ++i){
		Eigen::Vector2d& begPt = intChull[(i - 1 + intChull.size()) % intChull.size()];
		Eigen::Vector2d& endPt = intChull[i];
		areaRes += cross2d(begPt, endPt);
	}
	areaRes *= 0.5;

	if(viewer){
		viewer->removeCoordinateSystem("plane coord", viewPort2);
	}

	for(int p = 0; p < intChull.size(); ++p){
		Eigen::Vector2d& curPt = intChull[p];
		pcl::PointXYZRGB curPt3d;
		Eigen::Vector3d curCoord = origin + curPt.x() * xAxis + curPt.y() * yAxis;
		curPt3d.getVector3fMap() = curCoord.cast<float>();
		curPt3d.r = 255;
		curPt3d.g = 255;
		curPt3d.b = 255;
		chullRes->push_back(curPt3d);
		polyRes.vertices.push_back(p);
	}
}

Matching::SegIntType Matching::intersectLineSegments(const Eigen::Vector2d& begPt1,
												const Eigen::Vector2d& endPt1,
												const Eigen::Vector2d& begPt2,
												const Eigen::Vector2d& endPt2,
												Eigen::Vector2d& intPt1,
												Eigen::Vector2d& intPt2,
												double eps)
{
//	static constexpr double eps = 1e-6;
	SegIntType intType;

	Eigen::Vector2d edge1 = endPt1 - begPt1;
	Eigen::Vector2d edge2 = endPt2 - begPt2;
	double denom = begPt1.x() * edge2.y() +
					-endPt1.x() * edge2.y() +
					endPt2.x() * edge1.y() +
					-begPt2.x() * edge1.y();

	// parallel
	if(fabs(denom) <= eps){
		return intersectParallelLineSegments(begPt1, endPt1, begPt2, endPt2, intPt1, intPt2, eps);
	}

	double num = begPt1.x() * edge2.y() +
				begPt2.x() * (begPt1.y() - endPt2.y()) +
				endPt2.x() * (begPt2.y() - begPt1.y());
	if((fabs(num) <= eps) || (fabs(num - denom) <= eps)){
		intType = SegIntType::Vertex;
	}
	double s = num / denom;

	num = -(begPt1.x() * (begPt2.y() - endPt1.y()) +
			endPt1.x() * (begPt1.y() - begPt2.y()) +
			begPt2.x() * (endPt1.y() - begPt1.y()));
	if((fabs(num) <= eps) || (fabs(num - denom) <= eps)){
		intType = SegIntType::Vertex;
	}
	double t = num / denom;

	if((0.0 + eps < s) && (s < 1.0 - eps) && (0.0 + eps < t) && (t < 1.0 - eps)){
		intType = SegIntType::One;
	}
	else if((s < 0.0 - eps) || (1.0 + eps < s) || (t < 0.0 - eps) || (1.0 + eps < t)){
		intType = SegIntType::Zero;
	}

	intPt1.x() = begPt1.x() + s * (endPt1.x() - begPt1.x());
	intPt1.y() = begPt1.y() + s * (endPt1.y() - begPt1.y());

	return intType;
}

Matching::SegIntType Matching::intersectParallelLineSegments(const Eigen::Vector2d& begPt1,
															const Eigen::Vector2d& endPt1,
															const Eigen::Vector2d& begPt2,
															const Eigen::Vector2d& endPt2,
															Eigen::Vector2d& intPt1,
															Eigen::Vector2d& intPt2,
															double eps)
{
	double cross = cross2d(endPt1 - begPt1, begPt2 - begPt1);
	if(fabs(cross) > eps){
		return SegIntType::Zero;
	}

	if(isBetween(begPt1, endPt1, begPt2, eps) && isBetween(begPt1, endPt1, endPt2, eps)){
		intPt1 = begPt2;
		intPt2 = endPt2;
		return SegIntType::Collinear;
	}
	if(isBetween(begPt2, endPt2, begPt1, eps) && isBetween(begPt2, endPt2, endPt1, eps)){
		intPt1 = begPt1;
		intPt2 = endPt1;
		return SegIntType::Collinear;
	}
	if(isBetween(begPt1, endPt1, begPt2, eps) && isBetween(begPt2, endPt2, endPt1, eps)){
		intPt1 = begPt2;
		intPt2 = endPt1;
		return SegIntType::Collinear;
	}
	if(isBetween(begPt1, endPt1, begPt2, eps) && isBetween(begPt2, endPt2, begPt1, eps)){
		intPt1 = begPt2;
		intPt2 = begPt1;
		return SegIntType::Collinear;
	}
	if(isBetween(begPt1, endPt1, endPt2, eps) && isBetween(begPt2, endPt2, endPt1, eps)){
		intPt1 = endPt2;
		intPt2 = endPt1;
		return SegIntType::Collinear;
	}
	if(isBetween(begPt1, endPt1, endPt2, eps) && isBetween(begPt2, endPt2, begPt1, eps)){
		intPt1 = endPt2;
		intPt2 = begPt1;
		return SegIntType::Collinear;
	}
	return SegIntType::Zero;
}

bool Matching::isBetween(const Eigen::Vector2d& beg,
						const Eigen::Vector2d& end,
						const Eigen::Vector2d& pt,
						double eps)
{
	if(fabs(beg.x() - end.x()) > eps){
		return ((beg.x() <= pt.x()) && (pt.x() <= end.x())) ||
				((beg.x() >= pt.x()) && (pt.x() >= end.x()));
	}
	else{
		return ((beg.y() <= pt.y()) && (pt.y() <= end.y())) ||
				((beg.y() >= pt.y()) && (pt.y() >= end.y()));
	}
}


Matching::Inside Matching::newInsideFlag(Inside oldFlag,
										const Eigen::Vector2d& intPt,
										double cross1H2,
										double cross2H1,
										double eps)
{
	if(cross1H2 > 0.0 + eps){
		return Inside::First;
	}
	else if(cross2H1 > 0 + eps){
		return Inside::Second;
	}
	else{
		return oldFlag;
	}
}


double Matching::cross2d(const Eigen::Vector2d& v1,
						const Eigen::Vector2d& v2)
{
	return v1.x() * v2.y() - v2.x() * v1.y();
}

void Matching::makeCclockwise(std::vector<Eigen::Vector2d>& chull,
								double eps)
{
//	cout << "chull.size() = " << chull.size() << endl;
	bool isCclockwise = true;
	for(int i = 0; i < chull.size(); ++i){
		Eigen::Vector2d pt1 = chull[(i - 2 + chull.size()) % chull.size()];
		Eigen::Vector2d pt2 = chull[(i - 1 + chull.size()) % chull.size()];
		Eigen::Vector2d pt3 = chull[i];
//		cout << "points number: " << (i - 2 + chull.size()) % chull.size() << ", " << (i - 1 + chull.size()) % chull.size() << ", " << i << endl;
//		cout << "pt1 = " << pt1.transpose() << endl;
//		cout << "pt2 = " << pt2.transpose() << endl;
//		cout << "pt3 = " << pt3.transpose() << endl;
		double cross = cross2d(pt2 - pt1, pt3 - pt1);
//		cout << "cross = " << cross << endl;
		if(cross < 0.0 - eps){
//			cout << "is not counterclockwise" << endl;
			isCclockwise = false;
			break;
		}
	}
	// if not counter-clockwise then reverse
	if(!isCclockwise){
//		cout << "reversing chull" << endl;
		vector<Eigen::Vector2d> tmp(chull.rbegin(), chull.rend());
		tmp.swap(chull);
	}
}










