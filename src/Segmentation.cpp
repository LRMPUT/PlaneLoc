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
#include <random>
#include <algorithm>
#include <vector>
#include <map>
#include <chrono>
#include <thread>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Eigen>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/lsatr.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#include <opengm/functions/sparsemarray.hxx>

#include "Segmentation.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"
#include "UnionFind.h"

typedef opengm::GraphicalModel<
	double,
	opengm::Adder,
	opengm::ExplicitFunction<double, size_t, size_t>,
	opengm::SimpleDiscreteSpace<>
> ModelPlanes;

typedef opengm::LSA_TR<ModelPlanes, opengm::Minimizer> LsaTrInf;


//typedef opengm::SparseMarray<std::map<size_t, double> > SparseFunction;

typedef opengm::GraphicalModel<
	double,
	opengm::Adder,
	opengm::SparseFunction<double, size_t, size_t>,
	opengm::SimpleDiscreteSpace<>
> ModelAssign;

typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
typedef opengm::GraphCut<ModelAssign, opengm::Minimizer, MinStCutType> MinGraphCut;
typedef opengm::AlphaExpansion<ModelAssign, MinGraphCut> MinAlphaExpansion;

using namespace std;



void Segmentation::segment(const cv::FileStorage& fs,
						pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormals,
						pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab,
						std::vector<ObjInstance>& objInstances,
						bool segmentMap,
						pcl::visualization::PCLVisualizer::Ptr viewer,
						int viewPort1,
						int viewPort2)
{
	cout << "Segmentation::segment" << endl;

	// parameters
	float planeDistThresh = (double)fs["segmentation"]["planeDistThresh"];
	float areaMult = (double)fs["segmentation"]["areaMult"];
	float smoothCost = (double)fs["segmentation"]["smoothCost"];
	float minNumInlier = (double)fs["segmentation"]["minNumInlier"];

    // compute normals
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    {
        for(int p = 0; p < pcNormals->size(); ++p){
            pcl::Normal norm;
            norm.normal_x = pcNormals->at(p).normal_x;
            norm.normal_y = pcNormals->at(p).normal_y;
            norm.normal_z = pcNormals->at(p).normal_z;
//            cout << "norm.norm() = " << norm.getNormalVector3fMap().transpose() << endl;
            normals->push_back(norm);
        }
    }
//	{
//		  // Create the normal estimation class, and pass the input dataset to it
//		  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//		  ne.setInputCloud(pc);
//
//		  // Create an empty kdtree representation, and pass it to the normal estimation object.
//		  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//		  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
//		  ne.setSearchMethod(tree);
//
//		  // Use all neighbors in a sphere of radius 3cm
//		  ne.setRadiusSearch(0.03);
//
//		  // Compute the features
//		  ne.compute(*pointCloudNorm);
//
//	}
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>());
    for(int p = 0; p < pcNormals->size(); ++p){
        pcl::PointXYZRGB pt;
        pt.x = pcNormals->at(p).x;
        pt.y = pcNormals->at(p).y;
        pt.z = pcNormals->at(p).z;
        pt.r = pcNormals->at(p).r;
        pt.g = pcNormals->at(p).g;
        pt.b = pcNormals->at(p).b;
        pc->push_back(pt);
    }

    vector<SegInfo> svsInfo;
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm;
    {
        double svRes = 0.01;
        double svSize = 0.2;
        bool useSingleCameraTrans = true;
        if (segmentMap) {
            svRes = 0.01;
            svSize = 0.2;
            useSingleCameraTrans = false;
        }

        cout << "Creating supervoxels" << endl;
        map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> idToSv;
        pcl::SupervoxelClustering<pcl::PointXYZRGB> svClust(svRes, svSize, useSingleCameraTrans);
        svClust.setInputCloud(pc);
        svClust.setNormalCloud(normals);
        svClust.setColorImportance(1.0);
        svClust.setSpatialImportance(0.4);
        svClust.setNormalImportance(1.0);
//	svClust.set
        svClust.extract(idToSv);

        svClust.refineSupervoxels(1, idToSv);

        vector<uint32_t> idxToId;
        map<uint32_t, uint32_t> idToIdx;
        int idxCnt = 0;
        for(auto it = idToSv.begin(); it != idToSv.end(); ++it){
            // ignore label 0
            if(it->first != 0) {
                idxToId.push_back(it->first);
                idToIdx[it->first] = idxCnt;
//                cout << "idToIdx[" << it->first << "] = " << idxCnt << endl;
                ++idxCnt;
            }
        }

        cout << "Adding points to svsInfo" << endl;
        svsInfo.resize(idxCnt);

        pcLabSegm = svClust.getLabeledCloud();
        for(int p = 0; p < pcLabSegm->size(); ++p){
            // ignore label 0
            int svId = pcLabSegm->at(p).label;
            if(svId != 0) {
                if (idToIdx.count(svId) > 0) {
                    int svIdx = idToIdx.at(svId);
//                cout << "svId = " << svId << ", svIdx = " << svIdx << endl;
                    svsInfo[svIdx].addPointAndNormal(pc->at(p), normals->at(p));
                } else {
                    cout << "svId = " << svId << " not found!!!" << endl;
                }
            }
        }

        cout << "Calculating segment properties" << endl;
        for(int sv = 0; sv < svsInfo.size(); ++sv){
            svsInfo[sv].setId(sv);
            svsInfo[sv].setLabel(0);
            svsInfo[sv].calcSegProp();
        }

        cout << "Adding edges" << endl;
        std::multimap<uint32_t, uint32_t> adjMultimap;
        svClust.getSupervoxelAdjacency(adjMultimap);
        for(auto it = adjMultimap.begin(); it != adjMultimap.end(); ++it) {
            int srcIdx = idToIdx[it->first];
            int tarIdx = idToIdx[it->second];

            svsInfo[srcIdx].addAdjSeg(tarIdx);
            svsInfo[tarIdx].addAdjSeg(srcIdx);
        }
    }



	pcl::PointCloud<pcl::PointNormal>::Ptr planeNorm(new pcl::PointCloud<pcl::PointNormal>());
	{
		cout << "Computing graph plane generation functions" << endl;

//		std::vector<int> svCandIdxs;
//		std::vector<std::vector<bool>> isInPlane;
		pcl::PointCloud<pcl::PointNormal>::Ptr planeCandNorm(new pcl::PointCloud<pcl::PointNormal>());
		std::vector<float> nodeVals;
		std::vector<std::vector<float> > pairVals;
		compGraphPlaneGenFunctions(fs,
								svsInfo,
								planeCandNorm,
								nodeVals,
								pairVals);

//        ofstream logFile("../output/planeGen.log");
//		logFile << "nodeVals = " << nodeVals << endl;
//        logFile << "pairVals = ";
//        for(int i = 0; i < min(100, (int)pairVals.size()); ++i){
//            logFile << "[";
//            for(int j = 0; j < min(100, (int)pairVals[i].size()); ++j){
//                logFile << pairVals[i][j] << ", ";
//            }
//            logFile << "]" << endl;
//        }
//        logFile.close();

		cout << "Creating plane generation model" << endl;
		opengm::SimpleDiscreteSpace<> spacePlanes(nodeVals.size(), 2);
		ModelPlanes gmPlanes(spacePlanes);
//		vector<ModelPlanes::FunctionIdentifier> nodeFuncIds;
//		vector<vector<ModelPlanes::FunctionIdentifier>> pairFuncIds;
		for(int i = 0; i < nodeVals.size(); ++i){
			{
				size_t nodeShape[] = {2};
				opengm::ExplicitFunction<double> curNodeF(nodeShape, nodeShape + 1, 0.0);
				curNodeF(1) = nodeVals[i];
				ModelPlanes::FunctionIdentifier curNodeFId = gmPlanes.addFunction(curNodeF);

				size_t vi[] = {(size_t)i};
				gmPlanes.addFactor(curNodeFId, vi, vi + 1);
			}

//			pairFuncIds.emplace_back();
			for(int j = i + 1; j < nodeVals.size(); ++j){
				size_t pairShape[] = {2, 2};
				opengm::ExplicitFunction<double> curPairF(pairShape, pairShape + 2, 0.0);
				curPairF(1, 1) = pairVals[i][j];
				ModelPlanes::FunctionIdentifier curPairFId = gmPlanes.addFunction(curPairF);

				size_t vi[] = {(size_t)i, (size_t)j};
				gmPlanes.addFactor(curPairFId, vi, vi + 2);
			}
		}

		cout << "Inference, nodeVals.size() = " << nodeVals.size() << endl;
		vector<LsaTrInf::LabelType> startingPoint(nodeVals.size(), 0);
		if(nodeVals.size() > 0){
            startingPoint[0] = 1;
        }
		LsaTrInf::Parameter infParams;
		infParams.distance_ = LsaTrInf::Parameter::HAMMING;
//		infParams.initialLambda_ = 0.001;
		LsaTrInf infAlg(gmPlanes, infParams);
		infAlg.setStartingPoint(startingPoint.begin());
		infAlg.infer();
		cout << "Plane generation value: " << infAlg.value() << endl;
		vector<size_t> labelsVec;
		infAlg.arg(labelsVec);
		cout << "Plane generation labelsVec = " << labelsVec << endl;

		for(int c = 0; c < labelsVec.size(); ++c){
			if(labelsVec[c] == 1){
				cout << "adding plane: " << planeCandNorm->at(c) << endl;
				planeNorm->push_back(planeCandNorm->at(c));
			}
		}
	}

	vector<int> svLabels;
	for(int iter = 0; iter < 1; ++iter){

		vector<vector<float> > svPlaneVals;
		compGraphPlaneAssignFunctions(fs,
									svsInfo,
									planeNorm,
									svPlaneVals);
//		cout << "svPlaneVals = " << svPlaneVals << endl;
		cout << "svPlaneVals.size() = " << svPlaneVals.size() << endl;
		cout << "svPlaneVals[0].size() = " << svPlaneVals[0].size() << endl;

		cout << "Creating model for plane assignments, number of sv = " <<  svsInfo.size() << ", number of labels = " << planeNorm->size() + 1 << endl;
		// number of variables equal to svNormals->size() and numer of labels for each variable equal to genPlaneSvIdxs.size() + 1
		opengm::SimpleDiscreteSpace<> spaceAssign(svsInfo.size(), planeNorm->size() + 1);
		ModelAssign gmAssign(spaceAssign);
//		vector<vector<ModelAssign::FunctionIdentifier>> svPlaneFuncIds;
//		vector<ModelAssign::FunctionIdentifier> svSvFuncIds;

		for(int sv = 0; sv < svPlaneVals.size(); ++sv){
			size_t planeSvShape[] = {planeNorm->size() + 1};
			opengm::SparseFunction<double, size_t, size_t> curSvPlaneF(planeSvShape, planeSvShape + 1, 0.0);

			size_t coord[] = {0};
			curSvPlaneF.insert(coord, planeDistThresh);
			for(int pl = 0; pl < svPlaneVals[sv].size(); ++pl){
				coord[0] = pl + 1;
				curSvPlaneF.insert(coord, svPlaneVals[sv][pl]);
			}
			ModelAssign::FunctionIdentifier curSvPlaneFId = gmAssign.addFunction(curSvPlaneF);

			size_t vi[] = {(size_t)sv};
			gmAssign.addFactor(curSvPlaneFId, vi, vi + 1);
		}

		{
			size_t svSvShape[] = {planeNorm->size() + 1, planeNorm->size() + 1};
			opengm::SparseFunction<double, size_t, size_t> svSvF(svSvShape, svSvShape + 2, smoothCost);
			for(int pl = 0; pl < planeNorm->size() + 1; ++pl){
				size_t coord[] = {(size_t)pl, (size_t)pl};
				svSvF.insert(coord, 0.0);
			}
			ModelAssign::FunctionIdentifier svSvFId = gmAssign.addFunction(svSvF);

            set<pair<int, int> > addedEdges;
            for(int sv = 0; sv < svsInfo.size(); ++sv){
                const vector<int>& adjNhs = svsInfo[sv].getAdjSegs();
                for(int nh = 0; nh < adjNhs.size(); ++nh){
                    int u = min(sv, adjNhs[nh]);
                    int v = max(sv, adjNhs[nh]);
                    if(addedEdges.count(make_pair(u, v)) == 0)
                    {
                        size_t vi[] = {(size_t)u, (size_t)v};
                        gmAssign.addFactor(svSvFId, vi, vi + 2);

                        addedEdges.insert(make_pair(u, v));
                    }
                }
            }
		}

		MinAlphaExpansion infAlg(gmAssign);
		infAlg.infer();
		cout << "Assign inference value = " << infAlg.value() << endl;
		vector<size_t> labelsVec;
		infAlg.arg(labelsVec);
//		cout << "Assign labelsVec = " << labelsVec << endl;


		// label 0 for non plane elements
		svLabels.clear();
		for(int sv = 0; sv < labelsVec.size(); ++sv){
			svLabels.push_back(labelsVec[sv]);
		}


		pcl::PointCloud<pcl::PointNormal>::Ptr newPlaneNorm(new pcl::PointCloud<pcl::PointNormal>());
		std::vector<std::vector<int>> planeSvsId;

		compPlaneNormals(svLabels,
							svsInfo,
							newPlaneNorm,
							planeSvsId);

		planeNorm->swap(*newPlaneNorm);

	}

	{
		// flood fill through segments to isolate disconnected parts
		cout << "flood fill through segments" << endl;
		vector<int> newSvLabels(svLabels.size(), 0);
		int newSvLabelsCnt = 0;
		vector<bool> isVisited(svLabels.size(), false);

//		std::multimap<uint32_t, uint32_t> adjMultimap;
//		svClust.getSupervoxelAdjacency(adjMultimap);
		for(int sv = 0; sv < svLabels.size(); ++sv){
			// label 0 for non plane elements
			if(!isVisited[sv] && svLabels[sv] != 0){
				queue<int> nodeQ;
				nodeQ.push(sv);
				isVisited[sv] = true;

				vector<int> curVisited;
				float curVisitedAreaCoeff = 0.0;
				while(!nodeQ.empty()){
					int curIdx = nodeQ.front();
//					int curId = idxToId[curIdx];
					nodeQ.pop();

					curVisited.push_back(curIdx);
					curVisitedAreaCoeff += areaMult * svsInfo[curIdx].getAreaEst();
//					auto nhRange = adjMultimap.equal_range(curId);
					for(int nh = 0; nh < svsInfo[curIdx].getAdjSegs().size(); ++nh){
						int nhIdx = svsInfo[curIdx].getAdjSegs()[nh];
						// if not visited and has the same label
						if(!isVisited[nhIdx] &&
							svLabels[curIdx] == svLabels[nhIdx])
						{
							nodeQ.push(nhIdx);
							isVisited[nhIdx] = true;
						}
					}
				}

				//TODO Find an optimal threshold
				if(curVisitedAreaCoeff >= minNumInlier){
					for(int sv : curVisited){
						newSvLabels[sv] = newSvLabelsCnt + 1;
					}
					++newSvLabelsCnt;
				}
			}
		}
		svLabels.swap(newSvLabels);

		for(int sv = 0; sv < svLabels.size(); ++sv){
			// label 0 for non plane elements
			if(svLabels[sv] != 0){
				for(int p = 0; p < svsInfo[sv].getPoints()->size(); ++p){
					pcl::PointXYZRGB pcPt = svsInfo[sv].getPoints()->at(p);
					pcl::PointXYZRGBL pt;
					pt.rgb = pcPt.rgb;
					pt.x = pcPt.x;
					pt.y = pcPt.y;
					pt.z = pcPt.z;
					pt.label = svLabels[sv];
					pcLab->push_back(pt);
				}
			}
		}
	}

	// create obj instances - plane instances in current version
	cout << "create obj instances" << endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr finalPlaneNorm(new pcl::PointCloud<pcl::PointNormal>());
	std::vector<std::vector<int>> planeSvsIdx;

	compPlaneNormals(svLabels,
						svsInfo,
						finalPlaneNorm,
						planeSvsIdx);

	cout << "finalPlaneNorm->size() = " << finalPlaneNorm->size() << endl;
	cout << "planeSvsIdx.size() = " << planeSvsIdx.size() << endl;
	int curObjInstId = 0;
	for(int pl = 0; pl < finalPlaneNorm->size(); ++pl){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr points(new pcl::PointCloud<pcl::PointXYZRGB>());
		std::vector<SegInfo> svs;
		for(int sv = 0; sv < planeSvsIdx[pl].size(); ++sv){
			SegInfo curSv = svsInfo[planeSvsIdx[pl][sv]];
			points->insert(points->end(), curSv.getPoints()->begin(), curSv.getPoints()->end());
			svs.push_back(curSv);
		}
		objInstances.emplace_back(curObjInstId++,
									ObjInstance::ObjType::Plane,
									points,
									svs);
	}

	if(viewer){
		cout << "displaying" << endl;
//		pcl::PointCloud<pcl::PointXYZL>::Ptr pcLabSegm = svClust.getLabeledVoxelCloud();
		viewer->addPointCloud(pcLabSegm, "cloudLabSegm", viewPort1);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcVox(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::Normal>::Ptr pcNormalsVox(new pcl::PointCloud<pcl::Normal>());
        for(int sv = 0; sv < svLabels.size(); ++sv) {
            SegInfo& curSv = svsInfo[sv];
            pcVox->insert(pcVox->end(),
                          svsInfo[sv].getPoints()->begin(),
                          svsInfo[sv].getPoints()->end());
            pcNormalsVox->insert(pcNormalsVox->end(),
                                 svsInfo[sv].getNormals()->begin(),
                                 svsInfo[sv].getNormals()->end());
        }

        viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(pcNormals, 100, 0.1, "cloudNormals", viewPort1);
//        viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(pcVox, pcNormalsVox, 100, 0.1, "cloudNormals", viewPort1);

		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcColPlanes(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::PointCloud<pcl::PointNormal>::Ptr pcNormalsSv(new pcl::PointCloud<pcl::PointNormal>());

		for(int p = 0; p < pcLab->size(); ++p){
			int lab = pcLab->at(p).label;
			int colIdx = lab % (sizeof(colors)/sizeof(colors[0]));
			pcl::PointXYZRGBA pt;
			pt.r = colors[colIdx][0];
			pt.g = colors[colIdx][1];
			pt.b = colors[colIdx][2];
			pt.x = pcLab->at(p).x;
			pt.y = pcLab->at(p).y;
			pt.z = pcLab->at(p).z;

			pcColPlanes->push_back(pt);
		}
        for(int sv = 0; sv < svLabels.size(); ++sv) {
            // label 0 for non plane elements
            if (svLabels[sv] != 0) {
                pcl::PointNormal curPtNormal = svsInfo[sv].getPointNormal();
                pcNormalsSv->push_back(curPtNormal);
            }
        }

		viewer->addPointCloud(pcColPlanes, "cloudColPlanes", viewPort2);
        if(!pcNormals->empty()) {
            viewer->addPointCloudNormals<pcl::PointNormal>(pcNormalsSv, 1, 0.1, "cloudNormalsSv", viewPort2);
        }

//        cout << "pcNormals->size() = " << pcNormalsSv->size() << endl;
//		cout << "pc->size() = " << pc->size() << endl;
//		cout << "pcLab->size() = " << pcLab->size() << endl;
//
//		viewer->resetStoppedFlag();
//		viewer->initCameraParameters();
//		viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//		viewer->spinOnce (100);
//		while (!viewer->wasStopped()){
//			viewer->spinOnce (100);
//			std::this_thread::sleep_for(std::chrono::milliseconds(50));
//		}
	}

//	if(pcCol != nullptr){
//		pcCol = svClust.getColoredVoxelCloud();
//	}
}


std::vector<float> Segmentation::calcHoughParams(Eigen::Vector4f params)
{
	float theta = atan2((double)params(1), (double)params(0)) * 180 / pi;
	float phi = acos((double)params(2)) * 180 / pi;
	float rho = params(3);

	return vector<float>{theta, phi, rho};
}

float Segmentation::compSvToPlaneDist(pcl::PointNormal sv,
									pcl::PointNormal plane)
{
	Eigen::Vector3f svNorm = sv.getNormalVector3fMap();
	Eigen::Vector3f planeNorm = plane.getNormalVector3fMap();

	float dist = distToPlane(plane, sv);

	float angDist = exp((1.0 - svNorm.dot(planeNorm))/0.1);
	if(isnan(angDist)){
		angDist = 1;
	}
	angDist = min(angDist, 10.0f);

	float weightedDist = angDist * dist;
	return weightedDist;
}

void Segmentation::compGraphPlaneGenFunctions(const cv::FileStorage& fs,
                                              const std::vector<SegInfo>& svsInfo,
									pcl::PointCloud<pcl::PointNormal>::Ptr planeCandNorm,
									std::vector<float>& nodeVals,
									std::vector<std::vector<float>>& pairVals)
{
	// parameters
	float curvThresh = (double)fs["segmentation"]["curvThresh"];
	float areaMult = (double)fs["segmentation"]["areaMult"];
	float randSamp = (double)fs["segmentation"]["randSamp"];
	float planeDistThresh = (double)fs["segmentation"]["planeDistThresh"];
	float minNumInlier = (double)fs["segmentation"]["minNumInlier"];
	float stddevAngle = (double)fs["segmentation"]["stddevAngleDeg"] * pi/180;

//	vector<int> svNotCandIdxs;
	vector<int> planeCandCard;
	vector<float> planeCandScore;
	vector<Eigen::Vector4f> planeCandParams;
	std::vector<std::vector<bool>> planeCandIsInPlane;
	std::vector<std::vector<float>> planeCandHoughParams;
	for(int sv = 0; sv < svsInfo.size(); ++sv){
//		cout << "it->curvature = " << it->curvature << endl;
		if(svsInfo[sv].getSegCurv() < curvThresh &&
			!isnan(svsInfo[sv].getSegNormal()[0]) &&
			!isnan(svsInfo[sv].getSegNormal()[1]) &&
			!isnan(svsInfo[sv].getSegNormal()[2]))
		{
//			cout << "pushing " << i << ", isnan(svNorm->at(i).normal_x) = " << isnan(svNorm->at(i).normal_x) << endl;
			float curPlaneScore = 0.0;

			vector<bool> curIsInPlane(svsInfo.size(), false);
			pcl::PointNormal curPlanePN = svsInfo[sv].getPointNormal();
			float curCandCard = 0.0;
			for(int svc = 0; svc < svsInfo.size(); ++svc){
				float weightedDist = compSvToPlaneDist(svsInfo[svc].getPointNormal(), curPlanePN);
				float areaCoeff = areaMult*svsInfo[svc].getAreaEst();
				curPlaneScore += exp(-weightedDist/planeDistThresh)*areaCoeff;
				// if in planeDistThresh and is planar
                if(weightedDist <= planeDistThresh &&
                   svsInfo[svc].getSegCurv() <= 2*curvThresh)
				{
					curIsInPlane[svc] = true;
					curCandCard += areaCoeff;
				}
//				if(isnan(curPlaneScore)){
//					throw PLANE_EXCEPTION("curPlaneScore is nan, angDist = " + to_string(angDist) + ", dist = " + to_string(dist));
//				}
			}
//			curPlaneScore = exp(-curPlaneScore/(minNumInlier*10)) - 1.0;
//			int curCandCard = count(curIsInPlane.begin(), curIsInPlane.end(), true);

//			if(fabs(curPlane.normal_z) > 0.9){
//				cout << "\tz curPlaneScore = " << curPlaneScore << endl;
//				cout << "\tz curCandCard = " << curCandCard << endl;
//			}
//			else{
//				cout << "curPlaneScore = " << curPlaneScore << endl;
//				cout << "curCandCard = " << curCandCard << endl;
//			}

			// add to plane candidates
			if(curCandCard >= minNumInlier){
				vector<float> curHoughParams = calcHoughParams(svsInfo[sv].getSegPlaneParams());
//				minTheta = min(minTheta, curHoughParams[0]);
//				maxTheta = max(maxTheta, curHoughParams[0]);
//				minPhi = min(minPhi, curHoughParams[1]);
//				maxPhi = max(minPhi, curHoughParams[1]);
//				minRho = min(minRho, curHoughParams[2]);
//				maxRho = max(maxRho, curHoughParams[2]);

//				svCandIdxs.push_back(i);
				planeCandCard.push_back(curCandCard);
				planeCandScore.push_back(curPlaneScore);
				planeCandParams.push_back(svsInfo[sv].getSegPlaneParams());
				planeCandNorm->push_back(svsInfo[sv].getPointNormal());
				planeCandIsInPlane.push_back(curIsInPlane);
				planeCandHoughParams.push_back(curHoughParams);
			}
//			else{
//				svNotCandIdxs.push_back(i);
//			}
		}
//		else{
//			svNotCandIdxs.push_back(i);
//		}
	}


	cout << "Number of plane candidates before Hough transform groupping = " <<  planeCandHoughParams.size() << endl;
//	{
//		int zCount = 0;
//		for(int i = 0; i < planeCandNorm->size(); ++i){
//			if(fabs(planeCandNorm->at(i).normal_z) > 0.9){
//				++zCount;
//			}
//		}
//		cout << "zCount = " << zCount << endl;
//	}

	UnionFind unionFind(planeCandHoughParams.size());

	for(int i = 0; i < planeCandHoughParams.size(); ++i){
		for(int j = i + 1; j < planeCandHoughParams.size(); ++j){
			vector<float> steps{1.0, 1.0, 0.025};
			vector<int> houghCoordI{int(planeCandHoughParams[i][0]/steps[0]), int(planeCandHoughParams[i][1]/steps[1]), int(planeCandHoughParams[i][2]/steps[2])};
			vector<int> houghCoordJ{int(planeCandHoughParams[j][0]/steps[0]), int(planeCandHoughParams[j][1]/steps[1]), int(planeCandHoughParams[j][2]/steps[2])};
			// hough coordinates are the same -> are representing the same plane
			if(houghCoordI[0] == houghCoordJ[0] &&
				houghCoordI[1] == houghCoordJ[1] &&
				houghCoordI[2] == houghCoordJ[2])
			{
//				cout << "houghCoordI = " << houghCoordI << endl;
//				cout << "houghCoordJ = " << houghCoordJ << endl;
				unionFind.unionSets(i, j);
			}

		}
	}

	{
		set<int> planeIdxs;
		for(int i = 0; i < planeCandHoughParams.size(); ++i){
			planeIdxs.insert(unionFind.findSet(i));
		}

		vector<int> tmpPlaneCandCard;
		vector<float> tmpPlaneCandScore;
		vector<Eigen::Vector4f> tmpPlaneCandParams;
		std::vector<std::vector<bool>> tmpPlaneCandIsInPlane;
		std::vector<std::vector<float>> tmpPlaneCandHoughParams;
		pcl::PointCloud<pcl::PointNormal>::Ptr tmpPlaneCandNorm(new pcl::PointCloud<pcl::PointNormal>());

		for(auto it = planeIdxs.begin(); it != planeIdxs.end(); ++it){
			tmpPlaneCandCard.push_back(planeCandCard[*it]);
			tmpPlaneCandScore.push_back(planeCandScore[*it]);
			tmpPlaneCandParams.push_back(planeCandParams[*it]);
			tmpPlaneCandIsInPlane.push_back(planeCandIsInPlane[*it]);
			tmpPlaneCandHoughParams.push_back(planeCandHoughParams[*it]);
			tmpPlaneCandNorm->push_back(planeCandNorm->at(*it));
		}

		tmpPlaneCandCard.swap(planeCandCard);
		tmpPlaneCandScore.swap(planeCandScore);
		tmpPlaneCandParams.swap(planeCandParams);
		tmpPlaneCandIsInPlane.swap(planeCandIsInPlane);
		tmpPlaneCandHoughParams.swap(planeCandHoughParams);
		tmpPlaneCandNorm->swap(*planeCandNorm);
	}
	cout << "Number of plane candidates after Hough transform groupping = " <<  planeCandHoughParams.size() << endl;
//	for(int i = 0; i < planeCandHoughParams.size(); ++i){
//		cout << "planeCandHoughParams[" << i << "] = " << planeCandHoughParams[i] << endl;
//	}
//	{
//		int zCount = 0;
//		for(int i = 0; i < planeCandNorm->size(); ++i){
//			if(fabs(planeCandNorm->at(i).normal_z) > 0.9){
//				++zCount;
//			}
//		}
//		cout << "zCount = " << zCount << endl;
//	}
//	float meanZScore = 0.0;
//	int zCount = 0;
//	float meanXYScore = 0.0;
//	int xyCount = 0;

	nodeVals = vector<float>(planeCandCard.size(), 0.0);
	pairVals = vector<vector<float>>(planeCandCard.size(), vector<float>(planeCandCard.size(), 0.0));
	for(int i = 0; i < planeCandCard.size(); ++i){
		// calculate node function
		float curNodeVal = exp(-planeCandScore[i]/(minNumInlier*10)) - 1.0;

//		if(fabs(planeCandNorm->at(i).normal_z) > 0.9){
//			meanZScore += planeCandScore[i];
//			++zCount;
//		}
//		else{
//			meanXYScore += planeCandScore[i];
//			++xyCount;
//		}
//		if(planeCandCard[i] < minNumInlier){
//			curNodeVal = 1 - (float)planeCandCard[i]/minNumInlier;
//		}
//		else{
//			curNodeVal = exp((float)(minNumInlier - planeCandCard[i])/(4*minNumInlier)) - 1;
//		}
		nodeVals[i] = curNodeVal;

		for(int j = i + 1; j < planeCandCard.size(); ++j){
			float angle = acos(planeCandNorm->at(i).normal_x * planeCandNorm->at(j).normal_x +
							planeCandNorm->at(i).normal_y * planeCandNorm->at(j).normal_y +
							planeCandNorm->at(i).normal_z * planeCandNorm->at(j).normal_z);
//			cout << "angle = " << angle * 180.0/pi << endl;
//			if(std::isnan(angle)){
//				cout << "svNorm->at(" << svCandIdxs[i] << ") = " << svNorm->at(svCandIdxs[i]) << endl;
//				cout << "svNorm->at(" << svCandIdxs[j] << ") = " << svNorm->at(svCandIdxs[j]) << endl;
//			}
			// move it to [0; pi/2] interval - planes can be rotated at most pi/2 each to another
			angle = min(angle, pi - angle);
			// move it to [0; pi/4] interval - parallel and perpendicular planes are ok
			float angleMin = min(angle, pi/2 - angle);
//			cout << "angleMin = " << angleMin * 180.0/pi << endl;

//			vector<bool> unionIsInPlane = isInPlane[i];
			int unionCnt = 0;
			for(int svj = 0; svj < planeCandIsInPlane[j].size(); ++svj){
				if(planeCandIsInPlane[i][svj] && planeCandIsInPlane[j][svj]){
					++unionCnt;
				}
			}

			pairVals[i][j] = 0.5 * (1.0 - exp(-angleMin/stddevAngle)) + 0.5 * ((float)unionCnt/min(planeCandCard[i], planeCandCard[j]));
			pairVals[j][i] = pairVals[i][j];
		}

	}
//	cout << "meanZScore = " << meanZScore/zCount << endl;
//	cout << "meanXYScore = " << meanXYScore/xyCount << endl;
}


void Segmentation::compGraphPlaneAssignFunctions(const cv::FileStorage& fs,
                                                 const std::vector<SegInfo>& svsInfo,
													pcl::PointCloud<pcl::PointNormal>::ConstPtr planeNorm,
													std::vector<std::vector<float> >& svPlaneVals)
{
	float curvThresh = (double)fs["segmentation"]["curvThresh"];
	float planeDistThresh = (double)fs["segmentation"]["planeDistThresh"];

	for(int sv = 0; sv < svsInfo.size(); ++sv){
		svPlaneVals.emplace_back(planeNorm->size());
		pcl::PointNormal curSv = svsInfo[sv].getPointNormal();
//		Eigen::Vector3f curPtNorm = svNorm->at(sv).getNormalVector3fMap();
		for(int pl = 0; pl < planeNorm->size(); ++pl){
			float weightedDist;
			if(svsInfo[sv].getSegCurv() <= 2*curvThresh){
				weightedDist = compSvToPlaneDist(curSv, planeNorm->at(pl));
			}
			else{
				weightedDist = 5*planeDistThresh;
			}
			svPlaneVals.back()[pl] = weightedDist;
		}
	}

//	for(auto it = svAdj.begin(); it != svAdj.end(); ++it){
//		int srcId = it->first;
//		int dstId = it->second;
//
//	}

}

void Segmentation::compPlaneNormals(const std::vector<int>& svLabels,
                                    const std::vector<SegInfo>& svsInfo,
									pcl::PointCloud<pcl::PointNormal>::Ptr planeNorm,
									std::vector<std::vector<int>>& planeSvsIdx)
{
	multimap<int, int> labelToSv;
	for(int sv = 0; sv < svLabels.size(); ++sv){
		labelToSv.insert(make_pair(svLabels[sv], sv));
	}


//	pcl::PointCloud<pcl::PointNormal>::Ptr newPlaneNorm(new pcl::PointCloud<pcl::PointNormal>());
	auto it = labelToSv.begin();
	while(it != labelToSv.end()){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc(new pcl::PointCloud<pcl::PointXYZRGB>());
		vector<int> curSvsIdx;

		auto itBegin = it;
		while(it != labelToSv.end() &&
			it->first == itBegin->first)
		{
			int curIdx = it->second;
//			int curId = idxToId[curIdx];
//			if(idToSv.count(curId) == 0){
//				throw PLANE_EXCEPTION("Id not fount in map, curId = " + to_string(curId) + ", curIdx = " + to_string(curIdx) + ", svLabels.size() = " + to_string(svLabels.size()));
//			}
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr curSvPc(svsInfo[curIdx].getPoints());
			curPc->insert(curPc->begin(), curSvPc->begin(), curSvPc->end());
			curSvsIdx.push_back(curIdx);
			++it;
		}

		// if label != 0 than we have a plane
		if(itBegin->first != 0){
			pcl::PointNormal curPlaneNorm;
			Eigen::Vector4f curPlaneParams;
			float curCurv;
			pcl::computePointNormal(*curPc, curPlaneParams, curCurv);

			// Same work as in pcl::computePointNormal - could be done better
	//		EIGEN_ALIGN16 Eigen::Matrix3f covMat;
			Eigen::Vector4f centr;
			pcl::compute3DCentroid(*curPc, centr);

			Eigen::Vector3f curNormal = curPlaneParams.head<3>();
			curNormal.normalize();
//			pcl::flipNormalTowardsViewpoint(curPlaneNorm, 0.0, 0.0, 0.0, curNormal);

			curPlaneNorm.getVector4fMap() = centr;
			curPlaneNorm.getNormalVector3fMap() = curNormal;
//			curPlaneNorm.data_n[3] = 0.0;
//			curPlaneNorm.getNormalVector3fMap().normalize();
			curPlaneNorm.curvature = curCurv;


			planeNorm->push_back(curPlaneNorm);
			planeSvsIdx.push_back(curSvsIdx);
		}
	}
}



void Segmentation::compSupervoxelsAreaEst(const std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr >& idToSv,
										std::vector<float>& svAreaEst)
{
	svAreaEst.clear();
	for(auto it = idToSv.begin(); it != idToSv.end(); ++it){
		float curAreaCoeff = 0.0;
		pcl::Supervoxel<pcl::PointXYZRGB>::Ptr curSv = it->second;
//		pcl::Normal curNorm = curSv->normal_;
		pcl::PointNormal curNormPt;
		curSv->getCentroidPointNormal(curNormPt);
		Eigen::Vector3f curNorm = curNormPt.getNormalVector3fMap();
		for(int pt = 0; pt < curSv->voxels_->size(); ++pt){
			pcl::PointXYZRGB curPt = curSv->voxels_->at(pt);
			Eigen::Vector3f curPtCentroid = curPt.getVector3fMap() - curNormPt.getVector3fMap();
			float lenSq = curPtCentroid.squaredNorm();
			float normLen = curNorm.dot(curPtCentroid);
			float distPlaneSq = lenSq - normLen * normLen;
			if(!isnan(distPlaneSq)){
				curAreaCoeff = max(distPlaneSq, curAreaCoeff);
			}
		}
		svAreaEst.push_back(curAreaCoeff * pi);
	}
//	cout << "svAreaEst = " << svAreaEst << endl;
}









