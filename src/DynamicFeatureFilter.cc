#include "DynamicFeatureFilter.h"
#include "YoloDetector.h"

#include <algorithm>
#include <cmath>

namespace ORB_SLAM3
{
namespace
{
struct PlaneModel
{
    cv::Vec3f normal;
    cv::Point3f point;
};

bool BackProject(const cv::Mat &depth, const cv::Mat &K, int u, int v,
                 cv::Point3f &point)
{
    if(depth.empty() || depth.type() != CV_32F || K.empty() || K.rows != 3 || K.cols != 3 ||
       u < 0 || v < 0 || u >= depth.cols || v >= depth.rows)
        return false;

    const float z = depth.at<float>(v, u);
    if(!std::isfinite(z) || z <= 0.0f)
        return false;

    const bool doubleK = K.type() == CV_64F;
    const float fx = doubleK ? static_cast<float>(K.at<double>(0, 0)) : K.at<float>(0, 0);
    const float fy = doubleK ? static_cast<float>(K.at<double>(1, 1)) : K.at<float>(1, 1);
    const float cx = doubleK ? static_cast<float>(K.at<double>(0, 2)) : K.at<float>(0, 2);
    const float cy = doubleK ? static_cast<float>(K.at<double>(1, 2)) : K.at<float>(1, 2);
    if(fx <= 0.0f || fy <= 0.0f)
        return false;

    point = cv::Point3f((u - cx) * z / fx, (v - cy) * z / fy, z);
    return true;
}

bool MakePlane(const cv::Point3f &a, const cv::Point3f &b, const cv::Point3f &c,
               PlaneModel &plane)
{
    const cv::Vec3f ab(b.x - a.x, b.y - a.y, b.z - a.z);
    const cv::Vec3f ac(c.x - a.x, c.y - a.y, c.z - a.z);
    cv::Vec3f n = ab.cross(ac);
    const float length = std::sqrt(n.dot(n));
    if(length < 1e-6f)
        return false;
    plane.normal = n / length;
    plane.point = a;
    return true;
}

float PlaneDistance(const PlaneModel &plane, const cv::Point3f &point)
{
    const cv::Vec3f offset(point.x - plane.point.x, point.y - plane.point.y,
                            point.z - plane.point.z);
    return std::fabs(plane.normal.dot(offset));
}

std::vector<PlaneModel> ExtractOrthogonalPlanes(const cv::Mat &depth, const cv::Mat &K,
                                                  float distanceThreshold)
{
    std::vector<cv::Point3f> cloud;
    for(int v = 0; v < depth.rows; v += 4)
        for(int u = 0; u < depth.cols; u += 4)
        {
            cv::Point3f point;
            if(BackProject(depth, K, u, v, point))
                cloud.push_back(point);
        }

    std::vector<PlaneModel> planes;
    std::vector<unsigned char> available(cloud.size(), 1);
    cv::RNG rng(0x4d414e48); // deterministic "MANH" seed

    for(int planeIndex = 0; planeIndex < 3; ++planeIndex)
    {
        std::vector<int> candidates;
        for(size_t i = 0; i < available.size(); ++i)
            if(available[i]) candidates.push_back(static_cast<int>(i));
        if(candidates.size() < 80)
            break;

        PlaneModel bestPlane;
        std::vector<int> bestInliers;
        for(int iteration = 0; iteration < 100; ++iteration)
        {
            const int ia = candidates[rng.uniform(0, static_cast<int>(candidates.size()))];
            const int ib = candidates[rng.uniform(0, static_cast<int>(candidates.size()))];
            const int ic = candidates[rng.uniform(0, static_cast<int>(candidates.size()))];
            PlaneModel candidate;
            if(ia == ib || ia == ic || ib == ic || !MakePlane(cloud[ia], cloud[ib], cloud[ic], candidate))
                continue;

            std::vector<int> inliers;
            for(size_t j = 0; j < candidates.size(); ++j)
            {
                const int id = candidates[j];
                if(PlaneDistance(candidate, cloud[id]) < distanceThreshold)
                    inliers.push_back(id);
            }
            if(inliers.size() > bestInliers.size())
            {
                bestPlane = candidate;
                bestInliers.swap(inliers);
            }
        }
        if(bestInliers.size() < 80)
            break;
        for(size_t i = 0; i < bestInliers.size(); ++i)
            available[bestInliers[i]] = 0;
        planes.push_back(bestPlane);
    }

    std::vector<PlaneModel> orthogonal;
    const float maxNormalDot = std::cos(80.0f * static_cast<float>(CV_PI) / 180.0f);
    for(size_t i = 0; i < planes.size(); ++i)
        for(size_t j = i + 1; j < planes.size(); ++j)
            if(std::fabs(planes[i].normal.dot(planes[j].normal)) < maxNormalDot)
            {
                orthogonal.push_back(planes[i]);
                orthogonal.push_back(planes[j]);
            }
    return orthogonal;
}
}

DynamicFeatureFilter::DynamicFeatureFilter(const Config &config)
    : mConfig(config), mbEngineReady(false), mbStopWorker(false), mbPendingImage(false)
{
}

DynamicFeatureFilter::~DynamicFeatureFilter()
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mbStopWorker = true;
    }
    mCondition.notify_all();
    if(mWorker.joinable())
        mWorker.join();
}

void DynamicFeatureFilter::Configure(const Config &config)
{
    mConfig = config;
}

bool DynamicFeatureFilter::LoadTensorRTEngine(const std::string &enginePath)
{
    if(!mConfig.enabled || mWorker.joinable())
        return false;
    mpYoloDetector.reset(new YoloDetector());
    mbEngineReady = mpYoloDetector->LoadEngine(enginePath);
    mEnginePath = enginePath;
    if(mbEngineReady)
        mWorker = std::thread(&DynamicFeatureFilter::WorkerLoop, this);
    return mbEngineReady;
}

bool DynamicFeatureFilter::IsReady() const
{
    return mbEngineReady && mpYoloDetector && mpYoloDetector->IsReady();
}

bool DynamicFeatureFilter::UseHardMask() const
{
    return mConfig.enabled && mConfig.hardMask && IsReady();
}

void DynamicFeatureFilter::SubmitImage(const cv::Mat &image) const
{
    if(!mbEngineReady || image.empty())
        return;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mPendingImage = image.clone();
        mbPendingImage = true;
    }
    mCondition.notify_one();
}

bool DynamicFeatureFilter::GetLatestDetections(const cv::Size &imageSize, cv::Mat &mask,
                                                std::vector<YoloBoundingBox> &boxes) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    if(mLatestMask.empty() || mLatestMask.size() != imageSize)
        return false;
    mask = mLatestMask.clone();
    boxes = mLatestBoxes;
    return true;
}

void DynamicFeatureFilter::WorkerLoop()
{
    while(true)
    {
        cv::Mat image;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCondition.wait(lock, [this] { return mbStopWorker || mbPendingImage; });
            if(mbStopWorker)
                return;
            image = mPendingImage;
            mbPendingImage = false;
        }

        cv::Mat mask;
        std::vector<YoloBoundingBox> boxes;
        if(InferDynamicMask(image, mask, boxes))
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mLatestMask = mask;
            mLatestBoxes = boxes;
        }
    }
}

bool DynamicFeatureFilter::InferDynamicMask(const cv::Mat &image, cv::Mat &mask,
                                             std::vector<YoloBoundingBox> &boxes) const
{
    if(!mpYoloDetector || image.empty() || !mpYoloDetector->Detect(image, boxes))
        return false;

    mask = cv::Mat::zeros(image.size(), CV_8U);
    for(size_t i = 0; i < boxes.size(); ++i)
    {
        if(!YoloDetector::IsDynamicClass(boxes[i].classId))
            continue;
        const cv::Rect box = cv::Rect(cvRound(boxes[i].rect.x), cvRound(boxes[i].rect.y),
                                      cvRound(boxes[i].rect.width), cvRound(boxes[i].rect.height)) &
                             cv::Rect(0, 0, image.cols, image.rows);
        if(box.area() > 0)
            mask(box).setTo(255);
    }
    return true;
}

void DynamicFeatureFilter::ApplyManhattanImmunity(const cv::Mat &depth, const cv::Mat &cameraMatrix,
                                                   const std::vector<cv::KeyPoint> &keys,
                                                   std::vector<float> &probability,
                                                   std::vector<unsigned char> &immune) const
{
    if(depth.empty() || depth.type() != CV_32F || cameraMatrix.empty())
        return;

    if(std::find_if(probability.begin(), probability.end(),
                    [](float value) { return value > 0.0f; }) == probability.end())
        return;

    const std::vector<PlaneModel> planes = ExtractOrthogonalPlanes(depth, cameraMatrix,
                                                                     mConfig.planeDistance);
    if(planes.empty())
        return;

    for(size_t i = 0; i < keys.size(); ++i)
    {
        if(probability[i] <= 0.0f)
            continue;
        cv::Point3f point;
        if(!BackProject(depth, cameraMatrix, cvRound(keys[i].pt.x), cvRound(keys[i].pt.y), point))
            continue;
        for(size_t j = 0; j < planes.size(); ++j)
        {
            if(PlaneDistance(planes[j], point) < mConfig.planeDistance)
            {
                probability[i] = 0.0f;
                immune[i] = 1;
                break;
            }
        }
    }
}

void DynamicFeatureFilter::Evaluate(const cv::Mat &dynamicMask, const cv::Mat &depth,
                                    const cv::Mat &cameraMatrix,
                                    const std::vector<cv::KeyPoint> &keys,
                                    const std::vector<cv::KeyPoint> *previousKeys,
                                    std::vector<float> &dynamicProbability,
                                    std::vector<unsigned char> &manhattanImmune) const
{
    (void)previousKeys;
    dynamicProbability.assign(keys.size(), 0.0f);
    manhattanImmune.assign(keys.size(), 0);

    if(!dynamicMask.empty())
        for(size_t i = 0; i < keys.size(); ++i)
        {
            const int u = cvRound(keys[i].pt.x);
            const int v = cvRound(keys[i].pt.y);
            if(u >= 0 && v >= 0 && u < dynamicMask.cols && v < dynamicMask.rows &&
               dynamicMask.at<unsigned char>(v, u) != 0)
                dynamicProbability[i] = mConfig.yoloPrior;
        }

    // A Sampson term must be computed from validated feature correspondences.
    // Frame keypoint indices are not temporal correspondences, so deliberately
    // do not estimate F from index-aligned vectors here.
    ApplyManhattanImmunity(depth, cameraMatrix, keys, dynamicProbability, manhattanImmune);
}
}
