/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "LineExtractor.h"
#include "LineMatcher.h"
#include "MapLine.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <chrono>


using namespace std;

namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL)), mnDynamicInputFrameId(0)
{
    // DynamicFilter is an extension field, therefore read it from both the
    // legacy YAML format and the File.version: 1.0 Settings format.
    cv::FileStorage dynamicSettings(strSettingPath, cv::FileStorage::READ);
    cv::FileNode dynamicNode = dynamicSettings["DynamicFilter"];
    if(!dynamicNode.empty())
    {
        DynamicFeatureFilter::Config dynamicConfig;
        dynamicConfig.enabled = (int)dynamicNode["enabled"] != 0;
        if(!dynamicNode["hardMask"].empty()) dynamicConfig.hardMask = (int)dynamicNode["hardMask"] != 0;
        if(!dynamicNode["sampsonEnabled"].empty()) dynamicConfig.sampsonEnabled = (int)dynamicNode["sampsonEnabled"] != 0;
        if(!dynamicNode["yoloPrior"].empty()) dynamicConfig.yoloPrior = (float)dynamicNode["yoloPrior"];
        if(!dynamicNode["dynamicThreshold"].empty()) dynamicConfig.dynamicThreshold = (float)dynamicNode["dynamicThreshold"];
        if(!dynamicNode["sampsonScale"].empty()) dynamicConfig.sampsonScale = (float)dynamicNode["sampsonScale"];
        if(!dynamicNode["planeDistance"].empty()) dynamicConfig.planeDistance = (float)dynamicNode["planeDistance"];
        if(!dynamicNode["starvationThreshold"].empty()) dynamicConfig.starvationThreshold = (int)dynamicNode["starvationThreshold"];
        if(!dynamicNode["resultWaitMs"].empty()) dynamicConfig.resultWaitMs = (int)dynamicNode["resultWaitMs"];
        if(!dynamicNode["dumpProbabilities"].empty())
            mbDumpDynamicProbabilities = static_cast<int>(dynamicNode["dumpProbabilities"]) != 0;
        if(!dynamicNode["probabilityDumpPath"].empty())
            mDynamicProbabilityDumpPath = static_cast<string>(dynamicNode["probabilityDumpPath"]);
        mDynamicFilter.Configure(dynamicConfig);
        if(dynamicConfig.enabled && !dynamicNode["engine"].empty())
        {
            const string enginePath=(string)dynamicNode["engine"];
            if(mDynamicFilter.LoadTensorRTEngine(enginePath))
                cout << "YOLOv8 TensorRT engine loaded: " << enginePath << endl;
            else
                cerr << "YOLOv8 TensorRT engine could not be loaded: " << enginePath << endl;
        }
    }
    if(mbDumpDynamicProbabilities && mDynamicProbabilityDumpPath.empty())
    {
        cerr << "Dynamic probability logging disabled: DynamicFilter.probabilityDumpPath is empty" << endl;
        mbDumpDynamicProbabilities=false;
    }

    cv::FileNode recoveryNode = dynamicSettings["DynamicRecovery"];
    if(!recoveryNode.empty())
    {
        if(!recoveryNode["enabled"].empty()) mbDepthRecoveryEnabled = static_cast<int>(recoveryNode["enabled"]) != 0;
        if(!recoveryNode["minStaticMapMatches"].empty())
            mnRecoveryMinStaticMapMatches = std::max(0,static_cast<int>(recoveryNode["minStaticMapMatches"]));
        if(!recoveryNode["backgroundDepthGap"].empty())
            mRecoveryBackgroundDepthGap = std::max(0.0f,static_cast<float>(recoveryNode["backgroundDepthGap"]));
        if(!recoveryNode["poseProbability"].empty())
            mRecoveryPoseProbability = std::min(0.54f,std::max(0.0f,static_cast<float>(recoveryNode["poseProbability"])));
        if(!recoveryNode["mapDepthResidual"].empty())
            mRecoveryMapDepthResidual = std::max(0.01f,static_cast<float>(recoveryNode["mapDepthResidual"]));
    }

    cv::FileNode instanceMotionNode = dynamicSettings["InstanceMotion"];
    if(!instanceMotionNode.empty())
    {
        if(!instanceMotionNode["enabled"].empty()) mbInstanceMotionEnabled = static_cast<int>(instanceMotionNode["enabled"]) != 0;
        if(!instanceMotionNode["minMatches"].empty())
            mnInstanceMotionMinMatches = std::max(2,static_cast<int>(instanceMotionNode["minMatches"]));
        if(!instanceMotionNode["staticRatio"].empty())
            mInstanceMotionStaticRatio = std::min(1.0f,std::max(0.0f,static_cast<float>(instanceMotionNode["staticRatio"])));
        if(!instanceMotionNode["reprojectionResidual"].empty())
            mInstanceMotionReprojectionResidual = std::max(1.0f,static_cast<float>(instanceMotionNode["reprojectionResidual"]));
        if(!instanceMotionNode["depthResidual"].empty())
            mInstanceMotionDepthResidual = std::max(0.01f,static_cast<float>(instanceMotionNode["depthResidual"]));
        if(!instanceMotionNode["staticProbability"].empty())
            mInstanceMotionStaticProbability = std::min(0.54f,std::max(0.0f,static_cast<float>(instanceMotionNode["staticProbability"])));
    }

    cv::FileNode occlusionNode = dynamicSettings["OcclusionMode"];
    if(!occlusionNode.empty())
    {
        if(!occlusionNode["enabled"].empty()) mbOcclusionModeEnabled = static_cast<int>(occlusionNode["enabled"]) != 0;
        if(!occlusionNode["holdOnSuddenLoss"].empty()) mbOcclusionHoldOnSuddenLoss = static_cast<int>(occlusionNode["holdOnSuddenLoss"]) != 0;
        if(!occlusionNode["minStaticMatches"].empty()) mnOcclusionMinStaticMatches = std::max(5,static_cast<int>(occlusionNode["minStaticMatches"]));
        if(!occlusionNode["minUnknownMatches"].empty()) mnOcclusionMinUnknownMatches = std::max(5,static_cast<int>(occlusionNode["minUnknownMatches"]));
        if(!occlusionNode["recoveryFrames"].empty()) mnOcclusionRecoveryRequiredFrames = std::max(1,static_cast<int>(occlusionNode["recoveryFrames"]));
        if(!occlusionNode["maxHoldFrames"].empty()) mnOcclusionMaxHoldFrames = std::max(1,static_cast<int>(occlusionNode["maxHoldFrames"]));
        if(!occlusionNode["unknownFraction"].empty()) mOcclusionUnknownFraction = std::min(1.0f,std::max(0.0f,static_cast<float>(occlusionNode["unknownFraction"])));
        if(!occlusionNode["depthResidual"].empty()) mOcclusionDepthResidual = std::max(0.01f,static_cast<float>(occlusionNode["depthResidual"]));
        if(!occlusionNode["reprojectionResidual"].empty()) mOcclusionReprojectionResidual = std::max(1.0f,static_cast<float>(occlusionNode["reprojectionResidual"]));
        if(!occlusionNode["semanticCoverage"].empty()) mOcclusionSemanticCoverage = std::min(1.0f,std::max(0.0f,static_cast<float>(occlusionNode["semanticCoverage"])));
    }

    cv::FileNode poseGuardNode = dynamicSettings["PoseGuard"];
    if(!poseGuardNode.empty())
    {
        if(!poseGuardNode["enabled"].empty()) mbPoseGuardEnabled = static_cast<int>(poseGuardNode["enabled"]) != 0;
        if(!poseGuardNode["maxTranslation"].empty())
            mPoseGuardMaxTranslation = std::max(0.0f,static_cast<float>(poseGuardNode["maxTranslation"]));
        if(!poseGuardNode["maxRotationDeg"].empty())
            mPoseGuardMaxRotationDeg = std::max(0.0f,static_cast<float>(poseGuardNode["maxRotationDeg"]));
        if(!poseGuardNode["minStaticInliers"].empty())
            mnPoseGuardMinStaticInliers = std::max(0,static_cast<int>(poseGuardNode["minStaticInliers"]));
    }

    cv::FileNode shadowNode = dynamicSettings["GroundShadow"];
    if(!shadowNode.empty())
    {
        if(!shadowNode["enabled"].empty()) mbGroundShadowEnabled = static_cast<int>(shadowNode["enabled"]) != 0;
        if(!shadowNode["prior"].empty()) mGroundShadowPrior = static_cast<float>(shadowNode["prior"]);
        if(!shadowNode["planeDistance"].empty()) mGroundShadowPlaneDistance = static_cast<float>(shadowNode["planeDistance"]);
        if(!shadowNode["radius"].empty()) mGroundShadowRadius = static_cast<float>(shadowNode["radius"]);
        if(!shadowNode["brightnessDiff"].empty()) mGroundShadowBrightnessDiff = static_cast<float>(shadowNode["brightnessDiff"]);
        if(!shadowNode["textureStd"].empty()) mGroundShadowTextureStd = static_cast<float>(shadowNode["textureStd"]);
        if(!shadowNode["geometryThreshold"].empty()) mGroundShadowGeometryThreshold = static_cast<float>(shadowNode["geometryThreshold"]);
    }

    cv::FileNode manhattanNode = dynamicSettings["ManhattanPlane"];
    if(!manhattanNode.empty() && !manhattanNode["enabled"].empty())
        mbManhattanPlaneEnabled = static_cast<int>(manhattanNode["enabled"]) != 0;

    cv::FileNode lineFeatureNode = dynamicSettings["LineFeature"];
    if(!lineFeatureNode.empty() && !lineFeatureNode["enabled"].empty())
        mbLineFeatureEnabled = static_cast<int>(lineFeatureNode["enabled"]) != 0;

    cv::FileNode lineInitializationNode = dynamicSettings["LineInitialization"];
    if(!lineInitializationNode.empty())
    {
        if(!lineInitializationNode["enabled"].empty())
            mbLineInitializationEnabled = static_cast<int>(lineInitializationNode["enabled"]) != 0;
        if(!lineInitializationNode["minStaticPoints"].empty())
            mnLineInitializationMinStaticPoints = std::max(20, static_cast<int>(lineInitializationNode["minStaticPoints"]));
        if(!lineInitializationNode["minDepthLines"].empty())
            mnLineInitializationMinDepthLines = std::max(3, static_cast<int>(lineInitializationNode["minDepthLines"]));
        if(!lineInitializationNode["minLineLength"].empty())
            mLineInitializationMinLength = std::max(5.0f, static_cast<float>(lineInitializationNode["minLineLength"]));
    }

    cv::FileNode lineTrackingNode = dynamicSettings["LineTracking"];
    if(!lineTrackingNode.empty())
    {
        if(!lineTrackingNode["enabled"].empty()) mbLineTrackingEnabled = static_cast<int>(lineTrackingNode["enabled"]) != 0;
        if(!lineTrackingNode["maxStaticPoints"].empty()) mnLineTrackingPointThreshold = std::max(0, static_cast<int>(lineTrackingNode["maxStaticPoints"]));
    }

    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}

void Tracking::PrepareDynamicMask(uint64_t frameId, const cv::Mat &detectionImage, cv::Mat &dynamicMask,
                                  cv::Mat &staticMask,
                                  std::vector<YoloBoundingBox> &boxes)
{
    dynamicMask.release();
    staticMask.release();
    mDynamicFilter.SubmitImage(frameId, detectionImage);
    const bool hasDetections = mDynamicFilter.WaitForDetections(frameId, detectionImage.size(),
                                                                 mDynamicFilter.ResultWaitMs(), dynamicMask, boxes);
    if(hasDetections && mDynamicFilter.UseHardMask())
    {
        cv::Mat mask=dynamicMask.clone();
        // A class label is never sufficient to release a box.  It is opened
        // only when its *previous* RGB-D probe state has already accumulated
        // three consecutive static decisions.  A strong-motion decision
        // changes the state before the following image reaches this point,
        // so the box is hard-masked again immediately on the next frame.
        if(mbInstanceMotionEnabled)
            for(size_t i=0;i<boxes.size();++i)
                for(size_t j=0;j<mvProbeStates.size();++j)
                    if(mvProbeStates[j].state==1 && mvProbeStates[j].staticStreak>=3)
                    {
                        const cv::Rect2f inter=boxes[i].rect & mvProbeStates[j].rect;
                        const float uni=boxes[i].rect.area()+mvProbeStates[j].rect.area()-inter.area();
                        if(uni>0.f && inter.area()/uni>0.15f)
                        {
                            cv::rectangle(mask,boxes[i].rect,cv::Scalar(0),cv::FILLED);
                            cout << "Probe static instance released to main tracking" << endl;
                            break;
                        }
                    }
        cv::bitwise_not(mask,staticMask);
    }
    if(!hasDetections)
        boxes.clear();
}

void Tracking::RecoverDepthBackgroundInDynamicMask(cv::Mat &dynamicMask, cv::Mat &recoveredBackgroundMask, const cv::Mat &depth,
                                                    const std::vector<YoloBoundingBox> &boxes)
{
    recoveredBackgroundMask.release();
    if(!mbDepthRecoveryEnabled || !mDynamicFilter.UseHardMask() || mbInstanceMotionEnabled || dynamicMask.empty() ||
       depth.empty() || depth.type()!=CV_32F || boxes.empty())
        return;

    int lastStaticMapMatches=0;
    for(size_t i=0; i<mLastFrame.mvpMapPoints.size(); ++i)
        if(mLastFrame.mvpMapPoints[i] &&
           (i>=mLastFrame.mvbDynamicForMapping.size() || !mLastFrame.mvbDynamicForMapping[i]))
            ++lastStaticMapMatches;
    if(lastStaticMapMatches>=mnRecoveryMinStaticMapMatches)
        return;

    recoveredBackgroundMask=cv::Mat::zeros(dynamicMask.size(),CV_8UC1);
    for(size_t boxIndex=0; boxIndex<boxes.size(); ++boxIndex)
    {
        const cv::Rect box=cv::Rect(cvRound(boxes[boxIndex].rect.x),cvRound(boxes[boxIndex].rect.y),
                                    cvRound(boxes[boxIndex].rect.width),cvRound(boxes[boxIndex].rect.height)) &
                           cv::Rect(0,0,depth.cols,depth.rows);
        if(box.area()<=0) continue;

        // The lower depth quartile over the central part is a conservative
        // foreground estimate: background behind a person is normally farther.
        std::vector<float> centerDepths;
        const cv::Rect center(box.x+box.width/4,box.y+box.height/4,
                              std::max(1,box.width/2),std::max(1,box.height/2));
        for(int y=center.y; y<center.y+center.height && y<depth.rows; y+=3)
            for(int x=center.x; x<center.x+center.width && x<depth.cols; x+=3)
            {
                const float z=depth.at<float>(y,x);
                if(std::isfinite(z) && z>0.0f) centerDepths.push_back(z);
            }
        if(centerDepths.size()<8) continue;
        std::nth_element(centerDepths.begin(),centerDepths.begin()+centerDepths.size()/4,centerDepths.end());
        const float foregroundDepth=centerDepths[centerDepths.size()/4];

        for(int y=box.y; y<box.y+box.height; ++y)
            for(int x=box.x; x<box.x+box.width; ++x)
            {
                const float z=depth.at<float>(y,x);
                if(std::isfinite(z) && z>=foregroundDepth+mRecoveryBackgroundDepthGap)
                {
                    dynamicMask.at<unsigned char>(y,x)=0;
                    recoveredBackgroundMask.at<unsigned char>(y,x)=255;
                }
            }
    }
}

void Tracking::ApplyDynamicPrior(const cv::Mat &dynamicMask, const cv::Mat &depth,
                                 const cv::Mat &recoveredBackgroundMask)
{
    const std::vector<cv::KeyPoint> *previousKeys = mLastFrame.mvKeysUn.empty() ? NULL : &mLastFrame.mvKeysUn;
    mDynamicFilter.Evaluate(dynamicMask, depth, mCurrentFrame.mK, mCurrentFrame.mvKeysUn, previousKeys,
                            mCurrentFrame.mvDynamicProbability, mCurrentFrame.mvbManhattanImmune);
    mCurrentFrame.mvbDynamicForMapping.resize(mCurrentFrame.mvDynamicProbability.size(), 0);
    mCurrentFrame.mvbRecoveredBackground.assign(mCurrentFrame.mvDynamicProbability.size(), 0);
    for(size_t i = 0; i < mCurrentFrame.mvDynamicProbability.size(); ++i)
    {
        mCurrentFrame.mvbDynamicForMapping[i] = mDynamicFilter.IsDynamicForMapping(
            mCurrentFrame.mvDynamicProbability[i], mCurrentFrame.mvbManhattanImmune[i]);
        if(!recoveredBackgroundMask.empty() && i<mCurrentFrame.mvKeysUn.size())
        {
            const cv::Point2f &point=mCurrentFrame.mvKeysUn[i].pt;
            const int x=cvRound(point.x), y=cvRound(point.y);
            if(x>=0 && y>=0 && x<recoveredBackgroundMask.cols && y<recoveredBackgroundMask.rows &&
               recoveredBackgroundMask.at<unsigned char>(y,x))
            {
                mCurrentFrame.mvbRecoveredBackground[i]=1;
                mCurrentFrame.mvDynamicProbability[i]=std::max(mCurrentFrame.mvDynamicProbability[i],mRecoveryPoseProbability);
                // Mapping remains blocked until a later frame confirms an
                // existing static MapPoint association.
                mCurrentFrame.mvbDynamicForMapping[i]=1;
            }
        }
    }
}

void Tracking::PromoteRecoveredBackgroundMatches()
{
    for(size_t i=0; i<mCurrentFrame.mvbRecoveredBackground.size(); ++i)
    {
        if(!mCurrentFrame.mvbRecoveredBackground[i] || i>=mCurrentFrame.mvpMapPoints.size()) continue;
        MapPoint *pMP=mCurrentFrame.mvpMapPoints[i];
        if(!pMP || pMP->isBad() || pMP->Observations()<=0) continue;
        mCurrentFrame.mvbRecoveredBackground[i]=0;
        if(i<mCurrentFrame.mvDynamicProbability.size())
            mCurrentFrame.mvDynamicProbability[i]=std::min(mCurrentFrame.mvDynamicProbability[i],0.15f);
        if(i<mCurrentFrame.mvbDynamicForMapping.size())
            mCurrentFrame.mvbDynamicForMapping[i]=0;
    }
}

void Tracking::ValidateRecoveredBackgroundMatches()
{
    if(mCurrentFrame.mvbRecoveredBackground.empty() || !mCurrentFrame.HasPose())
        return;

    const Sophus::SE3f Tcw=mCurrentFrame.GetPose();
    for(size_t i=0; i<mCurrentFrame.mvbRecoveredBackground.size(); ++i)
    {
        if(!mCurrentFrame.mvbRecoveredBackground[i]) continue;

        MapPoint *pMP=i<mCurrentFrame.mvpMapPoints.size() ? mCurrentFrame.mvpMapPoints[i] : NULL;
        const float measuredDepth=i<mCurrentFrame.mvDepth.size() ? mCurrentFrame.mvDepth[i] : -1.0f;
        bool consistent=pMP && !pMP->isBad() && pMP->Observations()>=2 && measuredDepth>0.0f;
        if(consistent)
        {
            const float predictedDepth=(Tcw*pMP->GetWorldPos()).z();
            // Preserve a small relative tolerance for RGB-D noise, but the
            // associated map point must be at the same physical depth.
            const float tolerance=std::max(mRecoveryMapDepthResidual,0.05f*predictedDepth);
            consistent=predictedDepth>0.0f && std::fabs(measuredDepth-predictedDepth)<=tolerance;
        }

        if(!consistent)
        {
            if(i<mCurrentFrame.mvpMapPoints.size())
                mCurrentFrame.mvpMapPoints[i]=NULL;
            if(i<mCurrentFrame.mvbOutlier.size())
                mCurrentFrame.mvbOutlier[i]=true;
        }
    }
}

void Tracking::UpdateGeometricDynamicPrior()
{
    std::map<MapPoint*, size_t> previousObservations;
    for(size_t i = 0; i < mLastFrame.mvpMapPoints.size(); ++i)
        if(mLastFrame.mvpMapPoints[i] && i < mLastFrame.mvKeysUn.size())
            previousObservations[mLastFrame.mvpMapPoints[i]] = i;

    std::vector<cv::Point2f> previousPoints;
    std::vector<cv::Point2f> currentPoints;
    std::vector<size_t> currentIndices;
    for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size() && i < mCurrentFrame.mvKeysUn.size(); ++i)
    {
        MapPoint *point = mCurrentFrame.mvpMapPoints[i];
        std::map<MapPoint*, size_t>::const_iterator match = previousObservations.find(point);
        if(point && match != previousObservations.end())
        {
            previousPoints.push_back(mLastFrame.mvKeysUn[match->second].pt);
            currentPoints.push_back(mCurrentFrame.mvKeysUn[i].pt);
            currentIndices.push_back(i);
        }
    }

    mDynamicFilter.ApplySampsonProbability(previousPoints, currentPoints, currentIndices,
                                           mCurrentFrame.mvDynamicProbability);
    for(size_t i = 0; i < mCurrentFrame.mvDynamicProbability.size(); ++i)
        mCurrentFrame.mvbDynamicForMapping[i] = mDynamicFilter.IsDynamicForMapping(
            mCurrentFrame.mvDynamicProbability[i], mCurrentFrame.mvbManhattanImmune[i]) ||
            (i<mCurrentFrame.mvbRecoveredBackground.size() && mCurrentFrame.mvbRecoveredBackground[i]);
}

void Tracking::ApplyInstanceMotionVerification()
{
    // Kept as a compatibility hook for older call sites.  Classification is
    // now performed solely by the isolated probe pipeline below; re-enabling
    // regular Frame features here would violate the hard-mask contract.
    return;

    if(!mbInstanceMotionEnabled || !mCurrentFrame.HasPose() ||
       mCurrentFrame.mvDynamicBoxes.empty() || mCurrentFrame.mvKeysUn.empty())
        return;

    const Sophus::SE3f Tcw=mCurrentFrame.GetPose();
    for(size_t boxIndex=0; boxIndex<mCurrentFrame.mvDynamicBoxes.size(); ++boxIndex)
    {
        const YoloBoundingBox &box=mCurrentFrame.mvDynamicBoxes[boxIndex];
        std::vector<int> currentIndices;
        for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); ++i)
            if(box.rect.contains(mCurrentFrame.mvKeysUn[i].pt))
                currentIndices.push_back(static_cast<int>(i));
        if(currentIndices.empty()) continue;

        int support=0;
        int staticConsistent=0;

        // First use established static landmarks. This is the strongest test
        // because both reprojection and RGB-D depth are checked against the
        // map predicted from the current pose.
        for(size_t k=0; k<currentIndices.size(); ++k)
        {
            const int i=currentIndices[k];
            MapPoint *pMP=i<static_cast<int>(mCurrentFrame.mvpMapPoints.size()) ? mCurrentFrame.mvpMapPoints[i] : NULL;
            if(!pMP || pMP->isBad() || pMP->Observations()<2) continue;
            const Eigen::Vector3f pointCamera=Tcw*pMP->GetWorldPos();
            if(pointCamera.z()<=0.0f || i>=static_cast<int>(mCurrentFrame.mvDepth.size()) ||
               mCurrentFrame.mvDepth[i]<=0.0f || !mCurrentFrame.mpCamera)
                continue;
            ++support;
            const Eigen::Vector2f projection=mCurrentFrame.mpCamera->project(pointCamera);
            const float reprojection=cv::norm(cv::Point2f(projection.x(),projection.y())-
                                               mCurrentFrame.mvKeysUn[i].pt);
            const float depthTolerance=std::max(mInstanceMotionDepthResidual,0.05f*pointCamera.z());
            if(reprojection<=mInstanceMotionReprojectionResidual &&
               std::fabs(mCurrentFrame.mvDepth[i]-pointCamera.z())<=depthTolerance)
                ++staticConsistent;
        }

        // A stationary object may not have MapPoints yet because it was
        // previously masked. Bootstrap that decision with real descriptor
        // correspondences and RGB-D world-point consistency across frames.
        int previousBox=-1;
        float nearestCenterDistance=std::numeric_limits<float>::max();
        const cv::Point2f center(box.rect.x+0.5f*box.rect.width,box.rect.y+0.5f*box.rect.height);
        for(size_t previousIndex=0; previousIndex<mLastFrame.mvDynamicBoxes.size(); ++previousIndex)
        {
            const YoloBoundingBox &previousBoxCandidate=mLastFrame.mvDynamicBoxes[previousIndex];
            if(previousBoxCandidate.classId!=box.classId) continue;
            const cv::Point2f previousCenter(previousBoxCandidate.rect.x+0.5f*previousBoxCandidate.rect.width,
                                             previousBoxCandidate.rect.y+0.5f*previousBoxCandidate.rect.height);
            const float distance=cv::norm(center-previousCenter);
            if(distance<nearestCenterDistance)
            {
                nearestCenterDistance=distance;
                previousBox=static_cast<int>(previousIndex);
            }
        }
        if(previousBox>=0 && !mLastFrame.mDescriptors.empty() && !mCurrentFrame.mDescriptors.empty())
        {
            const cv::Rect2f &previousRect=mLastFrame.mvDynamicBoxes[previousBox].rect;
            std::vector<int> previousIndices;
            for(size_t i=0; i<mLastFrame.mvKeysUn.size(); ++i)
                if(previousRect.contains(mLastFrame.mvKeysUn[i].pt) &&
                   i<static_cast<size_t>(mLastFrame.mDescriptors.rows))
                    previousIndices.push_back(static_cast<int>(i));
            std::vector<int> descriptorCurrentIndices;
            for(size_t k=0; k<currentIndices.size(); ++k)
                if(currentIndices[k]<mCurrentFrame.mDescriptors.rows)
                    descriptorCurrentIndices.push_back(currentIndices[k]);

            if(!previousIndices.empty() && !descriptorCurrentIndices.empty())
            {
                cv::Mat previousDescriptors, currentDescriptors;
                for(size_t k=0; k<previousIndices.size(); ++k)
                    previousDescriptors.push_back(mLastFrame.mDescriptors.row(previousIndices[k]));
                for(size_t k=0; k<descriptorCurrentIndices.size(); ++k)
                    currentDescriptors.push_back(mCurrentFrame.mDescriptors.row(descriptorCurrentIndices[k]));
                std::vector<cv::DMatch> matches;
                cv::BFMatcher matcher(cv::NORM_HAMMING,true);
                matcher.match(previousDescriptors,currentDescriptors,matches);
                for(size_t k=0; k<matches.size(); ++k)
                {
                    if(matches[k].distance>48.0f) continue;
                    Eigen::Vector3f previousWorld,currentWorld;
                    const int previousFeature=previousIndices[matches[k].queryIdx];
                    const int currentFeature=descriptorCurrentIndices[matches[k].trainIdx];
                    if(!mLastFrame.UnprojectStereo(previousFeature,previousWorld) ||
                       !mCurrentFrame.UnprojectStereo(currentFeature,currentWorld))
                        continue;
                    ++support;
                    if((previousWorld-currentWorld).norm()<=mInstanceMotionDepthResidual)
                        ++staticConsistent;
                }
            }
        }

        if(support<mnInstanceMotionMinMatches ||
           static_cast<float>(staticConsistent)/static_cast<float>(support)<mInstanceMotionStaticRatio)
            continue;

        // This instance is static in the current frame. Release every feature
        // in its box so it can support tracking and seed persistent MapPoints.
        for(size_t k=0; k<currentIndices.size(); ++k)
        {
            const size_t i=static_cast<size_t>(currentIndices[k]);
            if(i<mCurrentFrame.mvDynamicProbability.size())
                mCurrentFrame.mvDynamicProbability[i]=std::min(mCurrentFrame.mvDynamicProbability[i],mInstanceMotionStaticProbability);
            if(i<mCurrentFrame.mvbDynamicForMapping.size())
                mCurrentFrame.mvbDynamicForMapping[i]=0;
        }
    }
}

void Tracking::ExtractInstanceProbeFeatures(const cv::Mat &dynamicMask)
{
    mvLastProbeKeys.swap(mvProbeKeys);
    mLastProbeDescriptors=mProbeDescriptors;
    mvLastProbeStates.swap(mvProbeStates);
    mvProbeKeys.clear();
    mProbeDescriptors.release();
    mvProbeStates.clear();
    if(!mbInstanceMotionEnabled || dynamicMask.empty() || mCurrentFrame.mvDynamicBoxes.empty())
        return;

    // This extractor is intentionally independent from Frame::ExtractORB:
    // probe features never change the main ORB budget or BoW/map pipeline.
    vector<int> lapping={0,0};
    (*mpORBextractorLeft)(mImGray,dynamicMask,mvProbeKeys,mProbeDescriptors,lapping);
    mvProbeStates.resize(mCurrentFrame.mvDynamicBoxes.size());
    for(size_t i=0;i<mvProbeStates.size();++i)
        mvProbeStates[i].rect=mCurrentFrame.mvDynamicBoxes[i].rect;
}

void Tracking::UpdateInstanceMotionStates()
{
    if(!mbInstanceMotionEnabled || !mCurrentFrame.HasPose() || !mLastFrame.HasPose() || mvProbeKeys.empty() ||
       mvLastProbeKeys.empty() || mProbeDescriptors.empty() || mLastProbeDescriptors.empty()) return;
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING,true);
    matcher.match(mLastProbeDescriptors,mProbeDescriptors,matches);
    const Sophus::SE3f Twc=mCurrentFrame.GetPose().inverse();
    const Sophus::SE3f TwcLast=mLastFrame.GetPose().inverse();
    for(size_t s=0;s<mvProbeStates.size();++s) {
        int previous=-1; float best=0.f;
        for(size_t p=0;p<mvLastProbeStates.size();++p) {
            const cv::Rect2f inter=mvProbeStates[s].rect & mvLastProbeStates[p].rect;
            const float uni=mvProbeStates[s].rect.area()+mvLastProbeStates[p].rect.area()-inter.area();
            const float iou=uni>0.f ? inter.area()/uni : 0.f;
            if(iou>best) { best=iou; previous=static_cast<int>(p); }
        }
        if(previous<0 || best<0.15f) continue;
        mvProbeStates[s].staticStreak=mvLastProbeStates[previous].staticStreak;
        mvProbeStates[s].dynamicStreak=mvLastProbeStates[previous].dynamicStreak;
        mvProbeStates[s].state=mvLastProbeStates[previous].state;
        vector<float> residuals;
        for(size_t k=0;k<matches.size();++k) {
            if(matches[k].distance>48.f) continue;
            const cv::KeyPoint &a=mvLastProbeKeys[matches[k].queryIdx], &b=mvProbeKeys[matches[k].trainIdx];
            if(!mvLastProbeStates[previous].rect.contains(a.pt) || !mvProbeStates[s].rect.contains(b.pt)) continue;
            const int ax=cvRound(a.pt.x), ay=cvRound(a.pt.y), bx=cvRound(b.pt.x), by=cvRound(b.pt.y);
            if(ax<0||ay<0||bx<0||by<0||ax>=mLastFrame.mImDepth.cols||ay>=mLastFrame.mImDepth.rows||bx>=mCurrentFrame.mImDepth.cols||by>=mCurrentFrame.mImDepth.rows) continue;
            const float za=mLastFrame.mImDepth.at<float>(ay,ax), zb=mCurrentFrame.mImDepth.at<float>(by,bx);
            if(za<=0.f||zb<=0.f||!mCurrentFrame.mpCamera) continue;
            const Eigen::Vector3f pa=TwcLast*(mLastFrame.mpCamera->unprojectEig(a.pt)*za);
            const Eigen::Vector3f pb=Twc*(mCurrentFrame.mpCamera->unprojectEig(b.pt)*zb);
            residuals.push_back((pa-pb).norm());
        }
        if(static_cast<int>(residuals.size())<mnInstanceMotionMinMatches) continue;
        sort(residuals.begin(),residuals.end());
        const float median=residuals[residuals.size()/2], p90=residuals[(residuals.size()*9)/10];
        const bool isStatic=median<0.04f && p90<0.08f;
        // Cross-checked ORB probe matches still contain a small tail of
        // descriptor outliers.  Once an instance has accumulated Static
        // evidence, revoke it only for genuinely coherent motion, not for a
        // marginal P90 excursion around the 8 cm static gate.
        const bool strongDynamic=median>0.06f || p90>0.15f;
        ProbeInstanceState &state=mvProbeStates[s];
        const unsigned char previousState=state.state;
        if(isStatic) { ++state.staticStreak; state.dynamicStreak=0; if(state.staticStreak>=3) state.state=1; }
        else {
            state.staticStreak=0;
            if(strongDynamic) ++state.dynamicStreak;
            else state.dynamicStreak=0;
            if((state.state==0 && state.dynamicStreak>=2) ||
               (state.state==1 && strongDynamic)) state.state=2;
        }
        if(state.state!=previousState)
            cout << "Probe instance state " << (state.state==1 ? "Static" : "Dynamic") << ": "
                 << residuals.size() << " matches, median " << median << " m, p90 " << p90 << " m" << endl;
        if(state.state==1) {
            int landmarksForInstance=0;
            for(size_t k=0;k<mvProbeKeys.size() && k<static_cast<size_t>(mProbeDescriptors.rows);++k) {
                if(landmarksForInstance>=20) break;
                if(!state.rect.contains(mvProbeKeys[k].pt)) continue;
                const int x=cvRound(mvProbeKeys[k].pt.x), y=cvRound(mvProbeKeys[k].pt.y);
                if(x<0||y<0||x>=mCurrentFrame.mImDepth.cols||y>=mCurrentFrame.mImDepth.rows) continue;
                const float z=mCurrentFrame.mImDepth.at<float>(y,x);
                if(z<=0.f||!mCurrentFrame.mpCamera) continue;
                ProbeLandmark landmark;
                landmark.world=mCurrentFrame.GetPose().inverse()*(mCurrentFrame.mpCamera->unprojectEig(mvProbeKeys[k].pt)*z);
                landmark.descriptor=mProbeDescriptors.row(static_cast<int>(k)).clone();
                landmark.rect=state.rect;
                landmark.instanceId=static_cast<int>(s);
                landmark.ttl=2;
                mvProbeLandmarks.push_back(landmark);
                ++landmarksForInstance;
            }
        }
    }
}

void Tracking::ApplyStaticProbeAssist()
{
    if(!mbInstanceMotionEnabled) return;
    for(size_t b=0;b<mCurrentFrame.mvDynamicBoxes.size();++b) {
        bool stable=false;
        for(size_t j=0;j<mvProbeStates.size();++j)
            if(mvProbeStates[j].state==1 && mvProbeStates[j].staticStreak>=3) {
            cv::Rect2f in=mCurrentFrame.mvDynamicBoxes[b].rect & mvProbeStates[j].rect;
            float u=mCurrentFrame.mvDynamicBoxes[b].rect.area()+mvProbeStates[j].rect.area()-in.area();
            if(u>0.f && in.area()/u>0.15f) { stable=true; break; }
        }
        if(!stable) continue;
        for(size_t i=0;i<mCurrentFrame.mvKeysUn.size();++i) if(mCurrentFrame.mvDynamicBoxes[b].rect.contains(mCurrentFrame.mvKeysUn[i].pt)) {
            // This box has passed the temporal 3-D test.  Let its ordinary
            // features rejoin tracking and map growth so stationary cars can
            // acquire normal green MapPoint correspondences.  This is not a
            // semantic-class exception: the next strong-motion decision
            // revokes it by restoring the hard mask in PrepareDynamicMask.
            if(i<mCurrentFrame.mvDynamicProbability.size()) mCurrentFrame.mvDynamicProbability[i]=0.0f;
            if(i<mCurrentFrame.mvbDynamicForMapping.size()) mCurrentFrame.mvbDynamicForMapping[i]=0;
            if(i<mCurrentFrame.mvbRecoveredBackground.size()) mCurrentFrame.mvbRecoveredBackground[i]=0;
        }
    }
}

void Tracking::BuildTemporaryProbeConstraints()
{
    mCurrentFrame.mvProbeWorldPoints.clear();
    mCurrentFrame.mvProbeObservations.clear();
    mCurrentFrame.mvProbeInstanceIds.clear();
    if(!mCurrentFrame.HasPose() || CountCurrentStaticMapMatches()>=20 || mvProbeLandmarks.empty() || mProbeDescriptors.empty()) return;
    for(size_t i=0;i<mvProbeLandmarks.size();++i) --mvProbeLandmarks[i].ttl;
    mvProbeLandmarks.erase(remove_if(mvProbeLandmarks.begin(),mvProbeLandmarks.end(),
        [](const ProbeLandmark &p){ return p.ttl<=0; }),mvProbeLandmarks.end());
    // Ratio matching requires two nearest candidates, so cross-check cannot
    // be enabled here.  Geometry and RGB-D gates below are stricter.
    cv::BFMatcher matcher(cv::NORM_HAMMING,false);
    cv::Mat descriptors;
    for(size_t i=0;i<mvProbeLandmarks.size();++i) descriptors.push_back(mvProbeLandmarks[i].descriptor);
    vector<vector<cv::DMatch> > knn;
    if(!descriptors.empty()) matcher.knnMatch(descriptors,mProbeDescriptors,knn,2);
    const Sophus::SE3f Tcw=mCurrentFrame.GetPose();
    for(size_t i=0;i<knn.size() && mCurrentFrame.mvProbeWorldPoints.size()<10;++i) {
        if(knn[i].size()<2 || knn[i][0].distance>40.f || knn[i][0].distance>=0.75f*knn[i][1].distance) continue;
        const cv::KeyPoint &kp=mvProbeKeys[knn[i][0].trainIdx];
        const ProbeLandmark &landmark=mvProbeLandmarks[knn[i][0].queryIdx];
        // A descriptor may not migrate between instances.  Require the
        // current detection to overlap the previously Static instance and
        // contain this probe feature.
        bool sameStaticInstance=false;
        for(size_t b=0;b<mCurrentFrame.mvDynamicBoxes.size();++b) {
            const cv::Rect2f inter=landmark.rect & mCurrentFrame.mvDynamicBoxes[b].rect;
            const float uni=landmark.rect.area()+mCurrentFrame.mvDynamicBoxes[b].rect.area()-inter.area();
            if(uni>0.f && inter.area()/uni>0.15f && mCurrentFrame.mvDynamicBoxes[b].rect.contains(kp.pt)) {
                sameStaticInstance=true;
                break;
            }
        }
        if(!sameStaticInstance) continue;
        const Eigen::Vector3f pc=Tcw*landmark.world;
        if(pc.z()<=0.f || !mCurrentFrame.mpCamera) continue;
        const Eigen::Vector2f uv=mCurrentFrame.mpCamera->project(pc);
        if(cv::norm(cv::Point2f(uv.x(),uv.y())-kp.pt)>3.f) continue;
        const int x=cvRound(kp.pt.x),y=cvRound(kp.pt.y);
        if(x<0||y<0||x>=mCurrentFrame.mImDepth.cols||y>=mCurrentFrame.mImDepth.rows) continue;
        const float z=mCurrentFrame.mImDepth.at<float>(y,x);
        if(z<=0.f || std::fabs(z-pc.z())>0.03f) continue;
        mCurrentFrame.mvProbeWorldPoints.push_back(landmark.world);
        mCurrentFrame.mvProbeObservations.push_back(kp);
        mCurrentFrame.mvProbeInstanceIds.push_back(landmark.instanceId);
    }
}

void Tracking::ApplyPersistentManhattanImmunity()
{
    if(!mbManhattanPlaneEnabled) return;
    const std::vector<cv::Vec4f> planes = mpLocalMapper->GetStableManhattanPlanes();
    if(planes.empty()) return;
    for(size_t i = 0; i < mCurrentFrame.mvDynamicProbability.size(); ++i)
    {
        if(mCurrentFrame.mvDynamicProbability[i] <= 0.0f) continue;
        Eigen::Vector3f point;
        if(!mCurrentFrame.UnprojectStereo(static_cast<int>(i), point)) continue;
        for(size_t j = 0; j < planes.size(); ++j)
            if(std::fabs(planes[j][0]*point.x()+planes[j][1]*point.y()+planes[j][2]*point.z()+planes[j][3]) < mDynamicFilter.ManhattanPlaneDistance())
            {
                mCurrentFrame.mvDynamicProbability[i] = 0.0f;
                mCurrentFrame.mvbManhattanImmune[i] = 1;
                break;
            }
    }
    for(size_t i = 0; i < mCurrentFrame.mvDynamicProbability.size(); ++i)
        mCurrentFrame.mvbDynamicForMapping[i] = mDynamicFilter.IsDynamicForMapping(
            mCurrentFrame.mvDynamicProbability[i], mCurrentFrame.mvbManhattanImmune[i]) ||
            (i<mCurrentFrame.mvbRecoveredBackground.size() && mCurrentFrame.mvbRecoveredBackground[i]);
}

void Tracking::ApplyGroundShadowProbability()
{
    if(!mbGroundShadowEnabled || !mbManhattanPlaneEnabled || !mCurrentFrame.HasPose() || mCurrentFrame.mImDepth.empty() ||
       mCurrentFrame.mvDynamicBoxes.empty() || mImGray.empty() || !mpLocalMapper)
        return;

    const std::vector<cv::Vec4f> stablePlanes = mpLocalMapper->GetStableManhattanPlanes();
    if(stablePlanes.empty()) return;
    // In the camera frame a horizontal ground plane has a normal close to the
    // image vertical axis. This rejects walls before contact-region fitting.
    std::vector<cv::Vec4f> planes;
    const Eigen::Matrix3f Rcw=mCurrentFrame.GetRwc().transpose();
    for(size_t p=0; p<stablePlanes.size(); ++p)
    {
        const Eigen::Vector3f normalCamera=Rcw*Eigen::Vector3f(stablePlanes[p][0],stablePlanes[p][1],stablePlanes[p][2]);
        if(std::fabs(normalCamera.y())>=0.70f) planes.push_back(stablePlanes[p]);
    }
    if(planes.empty()) return;

    struct ContactRegion { Eigen::Vector3f center; Eigen::Vector3f direction; cv::Vec4f plane; };
    std::vector<ContactRegion> contacts;
    const float fx=mCurrentFrame.mK.at<float>(0,0), fy=mCurrentFrame.mK.at<float>(1,1);
    const float cx=mCurrentFrame.mK.at<float>(0,2), cy=mCurrentFrame.mK.at<float>(1,2);
    const Eigen::Matrix3f Rwc=mCurrentFrame.GetRwc();
    const Eigen::Vector3f Ow=mCurrentFrame.GetOw();

    // The person's bottom edge supplies a contact hypothesis.  It must agree
    // with a stable plane before it can influence any surrounding feature.
    for(size_t b=0; b<mCurrentFrame.mvDynamicBoxes.size(); ++b)
    {
        const cv::Rect2f &box=mCurrentFrame.mvDynamicBoxes[b].rect;
        Eigen::Vector3f motionDirection=Eigen::Vector3f::Zero();
        float closestBox=1e9f;
        const cv::Point2f center(box.x+0.5f*box.width,box.y+0.5f*box.height);
        for(size_t previousBox=0; previousBox<mLastFrame.mvDynamicBoxes.size(); ++previousBox)
        {
            const YoloBoundingBox &candidate=mLastFrame.mvDynamicBoxes[previousBox];
            if(candidate.classId!=mCurrentFrame.mvDynamicBoxes[b].classId) continue;
            const cv::Point2f previousCenter(candidate.rect.x+0.5f*candidate.rect.width,
                                              candidate.rect.y+0.5f*candidate.rect.height);
            const float distance=cv::norm(center-previousCenter);
            if(distance<closestBox)
            {
                closestBox=distance;
                motionDirection=Rwc*Eigen::Vector3f((center.x-previousCenter.x)/fx,
                                                     (center.y-previousCenter.y)/fy,0.0f);
            }
        }
        if(closestBox>2.0f*std::max(box.width,box.height)) motionDirection.setZero();
        for(int sample=1; sample<=3; ++sample)
        {
            const int u=cvRound(box.x+box.width*0.25f*sample);
            const int v=cvRound(box.y+box.height-1.0f);
            if(u<0 || v<0 || u>=mCurrentFrame.mImDepth.cols || v>=mCurrentFrame.mImDepth.rows) continue;
            std::vector<float> depthSamples;
            for(int dy=-2; dy<=2; ++dy) for(int dx=-2; dx<=2; ++dx)
            {
                const int x=u+dx, y=v+dy;
                if(x<0 || y<0 || x>=mCurrentFrame.mImDepth.cols || y>=mCurrentFrame.mImDepth.rows) continue;
                const float z=mCurrentFrame.mImDepth.at<float>(y,x);
                if(std::isfinite(z) && z>0.0f) depthSamples.push_back(z);
            }
            if(depthSamples.size()<3) continue;
            std::nth_element(depthSamples.begin(),depthSamples.begin()+depthSamples.size()/2,depthSamples.end());
            const float depth=depthSamples[depthSamples.size()/2];
            const Eigen::Vector3f pointWorld=Rwc*Eigen::Vector3f((u-cx)*depth/fx,(v-cy)*depth/fy,depth)+Ow;
            int closestPlane=-1;
            float bestDistance=1e9f;
            for(size_t p=0; p<planes.size(); ++p)
            {
                const float distance=std::fabs(planes[p][0]*pointWorld.x()+planes[p][1]*pointWorld.y()+
                                                planes[p][2]*pointWorld.z()+planes[p][3]);
                if(distance<bestDistance) { bestDistance=distance; closestPlane=static_cast<int>(p); }
            }
            if(closestPlane>=0 && bestDistance<=mGroundShadowPlaneDistance)
            {
                const Eigen::Vector3f normal(planes[closestPlane][0],planes[closestPlane][1],planes[closestPlane][2]);
                Eigen::Vector3f direction=motionDirection-normal*normal.dot(motionDirection);
                if(direction.norm()>1e-4f) direction.normalize();
                else direction.setZero();
                contacts.push_back({pointWorld,direction,planes[closestPlane]});
            }
        }
    }
    if(contacts.empty()) return;

    // Intensity is compared only across a real MapPoint correspondence, not
    // same-index keypoints.  This keeps the photometric cue conservative.
    std::map<MapPoint*,size_t> previousObservations;
    if(!mLastImGray.empty())
        for(size_t i=0; i<mLastFrame.mvpMapPoints.size() && i<mLastFrame.mvKeysUn.size(); ++i)
            if(mLastFrame.mvpMapPoints[i]) previousObservations[mLastFrame.mvpMapPoints[i]]=i;

    const auto patchMeanStd=[](const cv::Mat &image, const cv::Point2f &pt, double &mean, double &stddev)->bool
    {
        const int radius=3, x=cvRound(pt.x), y=cvRound(pt.y);
        if(x-radius<0 || y-radius<0 || x+radius>=image.cols || y+radius>=image.rows) return false;
        cv::Scalar m,s;
        cv::meanStdDev(image(cv::Rect(x-radius,y-radius,2*radius+1,2*radius+1)),m,s);
        mean=m[0]; stddev=s[0];
        return true;
    };

    for(size_t i=0; i<mCurrentFrame.mvDynamicProbability.size() && i<mCurrentFrame.mvKeysUn.size(); ++i)
    {
        if(i<mCurrentFrame.mvbManhattanImmune.size() && mCurrentFrame.mvbManhattanImmune[i]) continue;
        Eigen::Vector3f pointWorld;
        if(!mCurrentFrame.UnprojectStereo(static_cast<int>(i),pointWorld)) continue;

        bool inInfluenceRegion=false;
        for(size_t c=0; c<contacts.size(); ++c)
        {
            const cv::Vec4f &plane=contacts[c].plane;
            const float signedDistance=plane[0]*pointWorld.x()+plane[1]*pointWorld.y()+plane[2]*pointWorld.z()+plane[3];
            if(std::fabs(signedDistance)>mGroundShadowPlaneDistance) continue;
            const Eigen::Vector3f normal(plane[0],plane[1],plane[2]);
            const Eigen::Vector3f tangent=pointWorld-contacts[c].center-normal*normal.dot(pointWorld-contacts[c].center);
            if(tangent.norm()<=mGroundShadowRadius) { inInfluenceRegion=true; break; }
            // A matched detection box provides a weak motion direction.  Its
            // forward lobe is deliberately short; without a match we retain
            // only the circular contact region above.
            if(contacts[c].direction.squaredNorm()>0.0f)
            {
                const float forward=tangent.dot(contacts[c].direction);
                const float lateral=(tangent-contacts[c].direction*forward).norm();
                if(forward>=0.0f && forward<=1.5f*mGroundShadowRadius && lateral<=0.5f*mGroundShadowRadius)
                { inInfluenceRegion=true; break; }
            }
        }
        if(!inInfluenceRegion) continue;

        double currentMean,currentStd;
        if(!patchMeanStd(mImGray,mCurrentFrame.mvKeysUn[i].pt,currentMean,currentStd) || currentStd>mGroundShadowTextureStd)
            continue;
        MapPoint *point=i<mCurrentFrame.mvpMapPoints.size()?mCurrentFrame.mvpMapPoints[i]:static_cast<MapPoint*>(NULL);
        std::map<MapPoint*,size_t>::const_iterator previous=previousObservations.find(point);
        if(!point || previous==previousObservations.end()) continue;
        double previousMean,previousStd;
        if(!patchMeanStd(mLastImGray,mLastFrame.mvKeysUn[previous->second].pt,previousMean,previousStd) ||
           std::fabs(currentMean-previousMean)<mGroundShadowBrightnessDiff) continue;
        if(mCurrentFrame.mvDynamicProbability[i]<mGroundShadowGeometryThreshold) continue;

        mCurrentFrame.mvDynamicProbability[i]=std::max(mCurrentFrame.mvDynamicProbability[i],mGroundShadowPrior);
        mCurrentFrame.mvbDynamicForMapping[i]=mDynamicFilter.IsDynamicForMapping(
            mCurrentFrame.mvDynamicProbability[i],mCurrentFrame.mvbManhattanImmune[i]);
    }
}

void Tracking::DumpDynamicProbabilityStats()
{
    // Per-feature files are intentionally opt-in: at 30 Hz this can write
    // thousands of rows per second and is intended for ablation analysis only.
    if(!mbDumpDynamicProbabilities ||
       mnLastDynamicProbabilityDumpFrameId==mCurrentFrame.mnId)
        return;
    mnLastDynamicProbabilityDumpFrameId=mCurrentFrame.mnId;

    const std::string pointsPath=mDynamicProbabilityDumpPath+"/dynamic_probability_points.txt";
    const std::string framesPath=mDynamicProbabilityDumpPath+"/dynamic_probability_frames.txt";
    std::ofstream points(pointsPath.c_str(),std::ios::out|std::ios::app);
    std::ofstream frames(framesPath.c_str(),std::ios::out|std::ios::app);
    if(!points.is_open() || !frames.is_open())
    {
        cerr << "Cannot write dynamic probability logs under: " << mDynamicProbabilityDumpPath << endl;
        mbDumpDynamicProbabilities=false;
        return;
    }
    if(!mbDynamicProbabilityLogHeaderWritten)
    {
        points << "# frame_id timestamp feature_index u v p_dynamic optimization_weight manhattan_immune mapping_dynamic has_map_point\n";
        frames << "# frame_id timestamp features nonzero_probability mapping_dynamic manhattan_immune mean_probability mean_weight\n";
        mbDynamicProbabilityLogHeaderWritten=true;
    }

    const size_t count=std::min(mCurrentFrame.mvKeysUn.size(),mCurrentFrame.mvDynamicProbability.size());
    size_t nonzero=0, mappingDynamic=0, immune=0;
    double probabilitySum=0.0, weightSum=0.0;
    points << std::fixed << std::setprecision(6);
    for(size_t i=0; i<count; ++i)
    {
        const float probability=mCurrentFrame.mvDynamicProbability[i];
        const double weight=std::max(0.05,1.0-static_cast<double>(probability));
        const int isImmune=(i<mCurrentFrame.mvbManhattanImmune.size() && mCurrentFrame.mvbManhattanImmune[i]) ? 1 : 0;
        const int isMappingDynamic=(i<mCurrentFrame.mvbDynamicForMapping.size() && mCurrentFrame.mvbDynamicForMapping[i]) ? 1 : 0;
        const int hasMapPoint=(i<mCurrentFrame.mvpMapPoints.size() && mCurrentFrame.mvpMapPoints[i]) ? 1 : 0;
        points << mCurrentFrame.mnId << ' ' << mCurrentFrame.mTimeStamp << ' ' << i << ' '
               << mCurrentFrame.mvKeysUn[i].pt.x << ' ' << mCurrentFrame.mvKeysUn[i].pt.y << ' '
               << probability << ' ' << weight << ' ' << isImmune << ' ' << isMappingDynamic << ' ' << hasMapPoint << '\n';
        probabilitySum+=probability;
        weightSum+=weight;
        if(probability>0.0f) ++nonzero;
        if(isMappingDynamic) ++mappingDynamic;
        if(isImmune) ++immune;
    }
    frames << std::fixed << std::setprecision(6)
           << mCurrentFrame.mnId << ' ' << mCurrentFrame.mTimeStamp << ' ' << count << ' '
           << nonzero << ' ' << mappingDynamic << ' ' << immune << ' '
           << (count ? probabilitySum/count : 0.0) << ' ' << (count ? weightSum/count : 0.0) << '\n';
}

void Tracking::RejectDynamicMapPointObservations()
{
    for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
        if(i < mCurrentFrame.mvbDynamicForMapping.size() && mCurrentFrame.mvbDynamicForMapping[i] &&
           !(i < mCurrentFrame.mvbRecoveredBackground.size() && mCurrentFrame.mvbRecoveredBackground[i]))
        {
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            mCurrentFrame.mvbOutlier[i] = true;
        }
}

int Tracking::CountCurrentStaticMapMatches() const
{
    int matches = 0;
    for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
    {
        if(!mCurrentFrame.mvpMapPoints[i]) continue;
        if(i < mCurrentFrame.mvbOutlier.size() && mCurrentFrame.mvbOutlier[i]) continue;
        if(i < mCurrentFrame.mvbDynamicForMapping.size() && mCurrentFrame.mvbDynamicForMapping[i]) continue;
        if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0) ++matches;
    }
    return matches;
}

bool Tracking::HasLowSupportPoseJump() const
{
    if(!mbPoseGuardEnabled || !mLastFrame.isSet() || !mCurrentFrame.isSet())
        return false;

    int staticInliers=0;
    for(size_t i=0; i<mCurrentFrame.mvpMapPoints.size(); ++i)
    {
        MapPoint *pMP=mCurrentFrame.mvpMapPoints[i];
        if(!pMP || pMP->isBad() || pMP->Observations()<=0) continue;
        if(i<mCurrentFrame.mvbOutlier.size() && mCurrentFrame.mvbOutlier[i]) continue;
        if(i<mCurrentFrame.mvbDynamicForMapping.size() && mCurrentFrame.mvbDynamicForMapping[i]) continue;
        ++staticInliers;
    }
    if(staticInliers>=mnPoseGuardMinStaticInliers)
        return false;

    const Sophus::SE3f relativePose=mCurrentFrame.GetPose()*mLastFrame.GetPose().inverse();
    const float translation=relativePose.translation().norm();
    const Eigen::Matrix3f rotation=relativePose.rotationMatrix();
    const float cosine=std::max(-1.0f,std::min(1.0f,(rotation.trace()-1.0f)*0.5f));
    const float rotationDeg=std::acos(cosine)*180.0f/static_cast<float>(CV_PI);
    const bool jump=translation>mPoseGuardMaxTranslation || rotationDeg>mPoseGuardMaxRotationDeg;
    if(jump)
        Verbose::PrintMess("Pose guard rejected low-support jump: " + to_string(staticInliers) +
                           " static inliers, " + to_string(translation) + " m, " +
                           to_string(rotationDeg) + " deg", Verbose::VERBOSITY_NORMAL);
    return jump;
}

void Tracking::ApplyUnknownMotionPrior()
{
    mbOcclusionFrame = false;
    if(!mbOcclusionModeEnabled || !mCurrentFrame.isSet() || !mCurrentFrame.mpCamera)
        return;

    const Sophus::SE3f predictedPose = mCurrentFrame.GetPose();
    mbOcclusionPredictedPoseValid = true;
    mOcclusionPredictedPose = predictedPose;

    if(mCurrentFrame.mvDynamicProbability.size() != mCurrentFrame.mvpMapPoints.size())
        mCurrentFrame.mvDynamicProbability.assign(mCurrentFrame.mvpMapPoints.size(), 0.0f);
    if(mCurrentFrame.mvbDynamicForMapping.size() != mCurrentFrame.mvpMapPoints.size())
        mCurrentFrame.mvbDynamicForMapping.assign(mCurrentFrame.mvpMapPoints.size(), 0);

    vector<size_t> unknownIndices;
    int matchedStaticMapPoints = 0;
    for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
    {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if(!pMP || pMP->isBad() || pMP->Observations() <= 0 ||
           (i < mCurrentFrame.mvbDynamicForMapping.size() && mCurrentFrame.mvbDynamicForMapping[i]))
            continue;
        ++matchedStaticMapPoints;
        if(i >= mCurrentFrame.mvKeysUn.size()) continue;

        const Eigen::Vector3f predictedCameraPoint = predictedPose * pMP->GetWorldPos();
        if(predictedCameraPoint.z() <= 0.0f) continue;
        const Eigen::Vector2f projected = mCurrentFrame.mpCamera->project(predictedCameraPoint);
        const cv::Point2f observed = mCurrentFrame.mvKeysUn[i].pt;
        const float reprojectionError = cv::norm(cv::Point2f(projected.x(),projected.y()) - observed);
        const float observedDepth = i < mCurrentFrame.mvDepth.size() ? mCurrentFrame.mvDepth[i] : -1.0f;
        const bool nearerThanStaticPrediction = observedDepth > 0.0f &&
            observedDepth < predictedCameraPoint.z() - mOcclusionDepthResidual;

        // Strong image disagreement is accepted without a depth residual to
        // cover laterally moving objects; otherwise require the physically
        // meaningful foreground-occlusion depth test.
        if(reprojectionError > mOcclusionReprojectionResidual &&
           (nearerThanStaticPrediction || reprojectionError > 2.0f*mOcclusionReprojectionResidual))
            unknownIndices.push_back(i);
    }

    const int imageWidth = std::max(1, mImGray.cols);
    const int imageHeight = std::max(1, mImGray.rows);
    const float imageArea = static_cast<float>(imageWidth * imageHeight);
    // Use the union of dynamic boxes.  Summing box areas makes overlapping
    // people in a crowd look like a full-image occlusion.
    cv::Mat semanticMask(imageHeight, imageWidth, CV_8UC1, cv::Scalar(0));
    for(size_t i = 0; i < mCurrentFrame.mvDynamicBoxes.size(); ++i)
    {
        const cv::Rect2f clipped = mCurrentFrame.mvDynamicBoxes[i].rect &
            cv::Rect2f(0.0f,0.0f,static_cast<float>(imageWidth),static_cast<float>(imageHeight));
        if(clipped.width <= 0.0f || clipped.height <= 0.0f) continue;
        const int x0 = std::max(0, cvFloor(clipped.x));
        const int y0 = std::max(0, cvFloor(clipped.y));
        const int x1 = std::min(imageWidth, cvCeil(clipped.x + clipped.width));
        const int y1 = std::min(imageHeight, cvCeil(clipped.y + clipped.height));
        if(x1 > x0 && y1 > y0)
            cv::rectangle(semanticMask, cv::Rect(x0,y0,x1-x0,y1-y0), cv::Scalar(255), cv::FILLED);
    }
    const float semanticCoverage = static_cast<float>(cv::countNonZero(semanticMask)) / imageArea;
    const float unknownFraction = matchedStaticMapPoints > 0 ?
        static_cast<float>(unknownIndices.size()) / matchedStaticMapPoints : 0.0f;
    const bool coherentUnknownMotion = static_cast<int>(unknownIndices.size()) >= mnOcclusionMinUnknownMatches &&
                                       unknownFraction >= mOcclusionUnknownFraction;
    const bool largeSemanticForeground = semanticCoverage >= mOcclusionSemanticCoverage;
    if(!coherentUnknownMotion && !largeSemanticForeground)
        return;

    for(size_t k = 0; k < unknownIndices.size(); ++k)
    {
        const size_t i = unknownIndices[k];
        mCurrentFrame.mvDynamicProbability[i] = std::max(mCurrentFrame.mvDynamicProbability[i], 0.95f);
        mCurrentFrame.mvbDynamicForMapping[i] = 1;
    }
    const int survivingStatic = matchedStaticMapPoints - static_cast<int>(unknownIndices.size());
    // A large foreground alone is not a full occlusion if enough static map
    // support remains.  That distinction is essential for crowd sequences.
    mbOcclusionFrame = survivingStatic < mnOcclusionMinStaticMatches &&
                       (coherentUnknownMotion || largeSemanticForeground);
    if(mbOcclusionFrame)
        Verbose::PrintMess("Occlusion mode candidate: " + to_string(unknownIndices.size()) +
                           " unknown-motion matches, " + to_string(survivingStatic) +
                           " static matches, coverage " + to_string(semanticCoverage),
                           Verbose::VERBOSITY_NORMAL);
}

void Tracking::RefinePoseWithLines()
{
    if(!mbLineFeatureEnabled || !mbLineTrackingEnabled || !mpReferenceKF || mImGray.empty()) return;
    int staticPoints=0;
    for(size_t i=0;i<mCurrentFrame.mvpMapPoints.size();++i)
        if(mCurrentFrame.mvpMapPoints[i] && (i>=mCurrentFrame.mvbDynamicForMapping.size() || !mCurrentFrame.mvbDynamicForMapping[i])) ++staticPoints;
    if(staticPoints>=mnLineTrackingPointThreshold) return;
    vector<cv::line_descriptor::KeyLine> refLines, currentLines;
    vector<MapLine*> refMapLines;
    cv::Mat refDesc,currentDesc;
    mpReferenceKF->GetLineFeatures(refLines,refDesc,refMapLines);
    if(refDesc.empty() || refLines.size()!=refMapLines.size()) return;
    LineExtractor::Extract(mImGray,currentLines,currentDesc);
    vector<cv::DMatch> matches; LineMatcher::Match(currentDesc,refDesc,matches);
    vector<MapLine*> matchedLines(currentLines.size(),static_cast<MapLine*>(NULL));
    for(size_t i=0;i<matches.size();++i) {
        const cv::DMatch &m=matches[i];
        if(m.queryIdx<0 || m.trainIdx<0 || static_cast<size_t>(m.queryIdx)>=matchedLines.size() || static_cast<size_t>(m.trainIdx)>=refMapLines.size()) continue;
        MapLine* line=refMapLines[m.trainIdx];
        if(!line || line->isBad()) continue;
        const Eigen::Vector3f startCamera=mCurrentFrame.GetPose()*line->GetStart();
        const Eigen::Vector3f endCamera=mCurrentFrame.GetPose()*line->GetEnd();
        if(startCamera.z()<=0.f || endCamera.z()<=0.f) continue;
        const Eigen::Vector2f startProjectedEigen=mCurrentFrame.mpCamera->project(startCamera);
        const Eigen::Vector2f endProjectedEigen=mCurrentFrame.mpCamera->project(endCamera);
        const cv::Point2f startProjected(startProjectedEigen.x(),startProjectedEigen.y());
        const cv::Point2f endProjected(endProjectedEigen.x(),endProjectedEigen.y());
        const cv::line_descriptor::KeyLine &observed=currentLines[m.queryIdx];
        const cv::Point2f observedDirection(observed.endPointX-observed.startPointX,observed.endPointY-observed.startPointY);
        const cv::Point2f projectedDirection=endProjected-startProjected;
        const float observedLength=cv::norm(observedDirection), projectedLength=cv::norm(projectedDirection);
        if(observedLength<15.f || projectedLength<15.f) continue;
        const float cosine=fabs(observedDirection.dot(projectedDirection)/(observedLength*projectedLength));
        const float a=observed.startPointY-observed.endPointY, b=observed.endPointX-observed.startPointX;
        const float midpointDistance=fabs(a*(startProjected.x+endProjected.x)*0.5f+b*(startProjected.y+endProjected.y)*0.5f+
                                          observed.startPointX*observed.endPointY-observed.endPointX*observed.startPointY)/observedLength;
        if(cosine>0.90f && midpointDistance<20.f) matchedLines[m.queryIdx]=line;
    }
    Optimizer::PoseOptimizationWithLines(&mCurrentFrame,currentLines,matchedLines);
}

bool Tracking::CanInitializeWithStructuralLines(vector<cv::line_descriptor::KeyLine> *lines,
                                                cv::Mat *descriptors) const
{
    if(!mbLineFeatureEnabled || !mbLineInitializationEnabled || mImGray.empty() ||
       mCurrentFrame.mImDepth.empty())
        return false;

    // Do not make the normal RGB-D bootstrap more expensive.  The original
    // point-only condition remains the preferred path; this function is only
    // called after it is known that fewer than 500 ORB features were found.
    int staticDepthPoints = 0;
    for(int i = 0; i < mCurrentFrame.N; ++i)
    {
        if(i >= static_cast<int>(mCurrentFrame.mvDepth.size()) || mCurrentFrame.mvDepth[i] <= 0.0f)
            continue;
        if(i < static_cast<int>(mCurrentFrame.mvbDynamicForMapping.size()) &&
           mCurrentFrame.mvbDynamicForMapping[i])
            continue;
        ++staticDepthPoints;
    }
    if(staticDepthPoints < mnLineInitializationMinStaticPoints)
        return false;

    vector<cv::line_descriptor::KeyLine> extractedLines;
    cv::Mat extractedDescriptors;
    LineExtractor::Extract(mImGray, extractedLines, extractedDescriptors, 100);
    if(extractedLines.empty())
        return false;

    const cv::Mat &depth = mCurrentFrame.mImDepth;
    int validDepthLines = 0;
    int orientationBins[4] = {0, 0, 0, 0};
    for(size_t i = 0; i < extractedLines.size(); ++i)
    {
        const cv::line_descriptor::KeyLine &line = extractedLines[i];
        if(line.lineLength < mLineInitializationMinLength)
            continue;

        const int samples[3][2] = {
            {cvRound(line.startPointX), cvRound(line.startPointY)},
            {cvRound(0.5f * (line.startPointX + line.endPointX)), cvRound(0.5f * (line.startPointY + line.endPointY))},
            {cvRound(line.endPointX), cvRound(line.endPointY)}
        };
        int validSamples = 0;
        for(int s = 0; s < 3; ++s)
        {
            const int x = samples[s][0], y = samples[s][1];
            if(x < 0 || y < 0 || x >= depth.cols || y >= depth.rows)
                continue;
            const float z = depth.at<float>(y, x);
            if(std::isfinite(z) && z > 0.0f)
                ++validSamples;
        }
        // A valid midpoint plus one endpoint prevents a depth discontinuity
        // or an invalid edge pixel from qualifying the whole line.
        if(validSamples < 2)
            continue;

        ++validDepthLines;
        float angle = std::atan2(line.endPointY - line.startPointY,
                                 line.endPointX - line.startPointX);
        if(angle < 0.0f) angle += static_cast<float>(CV_PI);
        const int bin = std::min(3, static_cast<int>(4.0f * angle / static_cast<float>(CV_PI)));
        ++orientationBins[bin];
    }

    int occupiedOrientationBins = 0;
    for(int i = 0; i < 4; ++i)
        if(orientationBins[i] > 0) ++occupiedOrientationBins;
    const bool accepted = validDepthLines >= mnLineInitializationMinDepthLines &&
                          occupiedOrientationBins >= 2;
    if(!accepted)
        return false;

    if(lines) *lines = extractedLines;
    if(descriptors) *descriptors = extractedDescriptors;
    return true;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    cv::Mat staticMask,dynamicMask,recoveredBackgroundMask;
    std::vector<YoloBoundingBox> dynamicBoxes;
    PrepareDynamicMask(++mnDynamicInputFrameId,imRectLeft,dynamicMask,staticMask,dynamicBoxes);

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,NULL,IMU::Calib(),staticMask);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,NULL,IMU::Calib(),staticMask);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib,staticMask);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib,staticMask);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    mCurrentFrame.mvDynamicBoxes = dynamicBoxes;
    ApplyDynamicPrior(dynamicMask);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    cv::Mat staticMask,dynamicMask,recoveredBackgroundMask;
    std::vector<YoloBoundingBox> dynamicBoxes;
    PrepareDynamicMask(++mnDynamicInputFrameId,imRGB,dynamicMask,staticMask,dynamicBoxes);
    RecoverDepthBackgroundInDynamicMask(dynamicMask,recoveredBackgroundMask,imDepth,dynamicBoxes);
    // PrepareDynamicMask may have selectively released a verified Static
    // instance.  Do not overwrite that mask here.
    if(mDynamicFilter.UseHardMask() && !mbInstanceMotionEnabled && !dynamicMask.empty())
        cv::bitwise_not(dynamicMask,staticMask);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,NULL,IMU::Calib(),staticMask);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib,staticMask);

    mCurrentFrame.mImDepth = imDepth.clone();






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    mCurrentFrame.mvDynamicBoxes = dynamicBoxes;
    ApplyDynamicPrior(dynamicMask, imDepth, recoveredBackgroundMask);
    ApplyStaticProbeAssist();
    ExtractInstanceProbeFeatures(dynamicMask);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    cv::Mat staticMask,dynamicMask;
    std::vector<YoloBoundingBox> dynamicBoxes;
    PrepareDynamicMask(++mnDynamicInputFrameId,im,dynamicMask,staticMask,dynamicBoxes);

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,NULL,IMU::Calib(),staticMask);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,NULL,IMU::Calib(),staticMask);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib,staticMask);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib,staticMask);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    mCurrentFrame.mvDynamicBoxes = dynamicBoxes;
    ApplyDynamicPrior(dynamicMask);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();

    return mCurrentFrame.GetPose();
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{

    // Per-frame evidence is produced by ApplyUnknownMotionPrior.  The mode
    // itself persists across a short occlusion, but stale evidence must not.
    mbOcclusionFrame = false;
    mbOcclusionPredictedPoseValid = false;

    if (bStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(mState!=NO_IMAGES_YET)
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial())
            {

                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }


    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        //mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            mLastImGray=mImGray.clone();
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }


                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // A fully opaque unknown object can leave no map-point matches at
        // all, so residual-based unknown-motion evidence is unavailable.
        // If a previously well-supported map suddenly loses all support,
        // enter the same bounded hold rather than immediately resetting it.
        if(mbOcclusionModeEnabled && mbOcclusionHoldOnSuddenLoss && !mbOcclusionMode && !mbOcclusionFrame && !bOK && mLastFrame.isSet())
        {
            int previousStaticMatches = 0;
            for(size_t i = 0; i < mLastFrame.mvpMapPoints.size(); ++i)
            {
                MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP || pMP->isBad() || pMP->Observations() <= 0) continue;
                if(i < mLastFrame.mvbOutlier.size() && mLastFrame.mvbOutlier[i]) continue;
                if(i < mLastFrame.mvbDynamicForMapping.size() && mLastFrame.mvbDynamicForMapping[i]) continue;
                ++previousStaticMatches;
            }
            if(previousStaticMatches >= 2*mnOcclusionMinStaticMatches && mCurrentFrame.N >= mnOcclusionMinStaticMatches)
            {
                mbOcclusionFrame = true;
                if(mCurrentFrame.isSet())
                {
                    mOcclusionPredictedPose = mCurrentFrame.GetPose();
                    mbOcclusionPredictedPoseValid = true;
                }
                Verbose::PrintMess("Occlusion mode candidate: sudden loss of " +
                                   to_string(previousStaticMatches) + " static matches", Verbose::VERBOSITY_NORMAL);
            }
        }

        bool skipLocalMapForOcclusion = false;
        if(mbOcclusionModeEnabled && (mbOcclusionFrame || mbOcclusionMode))
        {
            const int staticMatches = CountCurrentStaticMapMatches();
            const bool insufficientStaticSupport = staticMatches < mnOcclusionMinStaticMatches;
            if(mbOcclusionFrame || insufficientStaticSupport || !bOK)
            {
                mbOcclusionMode = true;
                mnOcclusionRecoveryFrames = 0;
                ++mnOcclusionFrames;
                if(mnOcclusionFrames <= mnOcclusionMaxHoldFrames)
                {
                    if(mbOcclusionPredictedPoseValid)
                        mCurrentFrame.SetPose(mOcclusionPredictedPose);
                    bOK = true;
                    skipLocalMapForOcclusion = true;
                    Verbose::PrintMess("Occlusion hold frame " + to_string(mnOcclusionFrames) +
                                       "/" + to_string(mnOcclusionMaxHoldFrames), Verbose::VERBOSITY_NORMAL);
                }
                else
                {
                    // Pure RGB-D has no independent motion measurement once
                    // every static landmark is hidden.  After a bounded hold,
                    // fall back to normal lost/relocalization handling.
                    mbOcclusionMode = false;
                    mnOcclusionFrames = 0;
                    bOK = false;
                }
            }
            else
            {
                ++mnOcclusionRecoveryFrames;
                if(mnOcclusionRecoveryFrames < mnOcclusionRecoveryRequiredFrames)
                {
                    if(mbOcclusionPredictedPoseValid)
                        mCurrentFrame.SetPose(mOcclusionPredictedPose);
                    bOK = true;
                    skipLocalMapForOcclusion = true;
                }
                else
                {
                    mbOcclusionMode = false;
                    mnOcclusionFrames = 0;
                    mnOcclusionRecoveryFrames = 0;
                    Verbose::PrintMess("Occlusion mode recovered with " + to_string(staticMatches) +
                                       " static matches", Verbose::VERBOSITY_NORMAL);
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK && !skipLocalMapForOcclusion)
            {
                bOK = TrackLocalMap();
                if(bOK && HasLowSupportPoseJump())
                {
                    // Do not let a rejected pose seed the motion model of
                    // the next recovery frame.
                    mCurrentFrame.SetPose(mLastFrame.GetPose());
                    bOK=false;
                }
            }
            if(!bOK)
                cout << "Fail to track local map!" << endl;
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO && !skipLocalMapForOcclusion)
            {
                bOK = TrackLocalMap();
                if(bOK && HasLowSupportPoseJump())
                {
                    mCurrentFrame.SetPose(mLastFrame.GetPose());
                    bOK=false;
                }
            }
        }

        if(bOK)
            mState = OK;
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = !mbOcclusionMode && NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(pCurrentMap->KeyFramesInMap()<=10)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
        mLastImGray=mImGray.clone();
    }




    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            // RECENTLY_LOST poses are motion-model predictions retained for
            // internal recovery. They are not visually verified estimates
            // and must not be exported as valid trajectory samples for ATE.
            mlbLost.push_back(mState!=OK);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState!=OK);
        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    vector<cv::line_descriptor::KeyLine> bootstrapLines;
    cv::Mat bootstrapLineDescriptors;
    const bool pointBootstrap = mCurrentFrame.N > 500;
    const bool structuralBootstrap = !pointBootstrap &&
        CanInitializeWithStructuralLines(&bootstrapLines, &bootstrapLineDescriptors);
    if(pointBootstrap || structuralBootstrap)
    {
        if(structuralBootstrap)
            Verbose::PrintMess("RGB-D structural bootstrap with " +
                               to_string(mCurrentFrame.N) + " ORB points and " +
                               to_string(bootstrapLines.size()) + " LSD lines", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
        if(structuralBootstrap && !bootstrapLineDescriptors.empty())
            pKFini->SetLineFeatures(bootstrapLines, bootstrapLineDescriptors);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0 && (i >= static_cast<int>(mCurrentFrame.mvbDynamicForMapping.size()) ||
                           !mCurrentFrame.mvbDynamicForMapping[i]))
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1 && (i >= static_cast<int>(mCurrentFrame.mvbDynamicForMapping.size()) ||
                                        !mCurrentFrame.mvbDynamicForMapping[i])){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false;

            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            mbReadyToInitializate = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        if((i < mInitialFrame.mvbDynamicForMapping.size() && mInitialFrame.mvbDynamicForMapping[i]) ||
           (mvIniMatches[i] < mCurrentFrame.mvbDynamicForMapping.size() &&
            mCurrentFrame.mvbDynamicForMapping[mvIniMatches[i]]))
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    UpdateGeometricDynamicPrior();
    ApplyPersistentManhattanImmunity();
    ApplyGroundShadowProbability();
    ApplyUnknownMotionPrior();
    DumpDynamicProbabilityStats();
    ValidateRecoveredBackgroundMatches();
    RejectDynamicMapPointObservations();
    BuildTemporaryProbeConstraints();
    Optimizer::PoseOptimization(&mCurrentFrame);
    RefinePoseWithLines();
    UpdateInstanceMotionStates();

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }




    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return false;
    }

    // Optimize frame pose with all matches
    UpdateGeometricDynamicPrior();
    ApplyPersistentManhattanImmunity();
    ApplyGroundShadowProbability();
    ApplyUnknownMotionPrior();
    DumpDynamicProbabilityStats();
    ValidateRecoveredBackgroundMatches();
    RejectDynamicMapPointObservations();
    BuildTemporaryProbeConstraints();
    Optimizer::PoseOptimization(&mCurrentFrame);
    RefinePoseWithLines();
    UpdateInstanceMotionStates();

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    UpdateLocalMap();
    SearchLocalPoints();
    RejectDynamicMapPointObservations();

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    UpdateGeometricDynamicPrior();
    ApplyPersistentManhattanImmunity();
    ApplyGroundShadowProbability();
    ApplyUnknownMotionPrior();
    DumpDynamicProbabilityStats();
    ValidateRecoveredBackgroundMatches();
    RejectDynamicMapPointObservations();
    BuildTemporaryProbeConstraints();
    int inliers;
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }
    RefinePoseWithLines();
    UpdateInstanceMotionStates();
    // A recovered background feature first contributes with its reduced pose
    // weight.  Only after the final pose step may an existing static-map
    // association promote it back to a normal observation.
    PromoteRecoveredBackgroundMatches();

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;


    if (mSensor == System::IMU_MONOCULAR)
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    if(mbLineFeatureEnabled)
        mpLocalMapper->SubmitLineExtraction(pKF, mImGray);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                if(i < static_cast<int>(mCurrentFrame.mvbDynamicForMapping.size()) &&
                   mCurrentFrame.mvbDynamicForMapping[i])
                    continue;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
