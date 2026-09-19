#ifndef DYNAMIC_FEATURE_FILTER_H
#define DYNAMIC_FEATURE_FILTER_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

namespace ORB_SLAM3 {

/*
 * Front-end dynamic-feature prior.  The TensorRT engine is deliberately kept
 * behind this small interface so a normal ORB-SLAM3 build has no CUDA ABI
 * dependency.  A deployed engine uses YOLOv8's standard detection output;
 * detections are rasterized to an internal dynamic-prior mask.
 */
class DynamicFeatureFilter {
public:
    struct Config {
        bool enabled;
        float dynamicThreshold;
        float sampsonScale;
        float planeDistance;
        int starvationThreshold;
        Config() : enabled(false), dynamicThreshold(0.55f), sampsonScale(3.0f),
                   planeDistance(0.05f), starvationThreshold(80) {}
    };

    explicit DynamicFeatureFilter(const Config &config = Config());
    ~DynamicFeatureFilter();
    void Configure(const Config &config);
    bool LoadTensorRTEngine(const std::string &enginePath);
    bool IsReady() const;

    // Runs YOLOv8/TensorRT (when compiled/configured) and updates per-feature
    // dynamic probabilities. depth can be empty; then Manhattan immunity is
    // simply unavailable rather than guessed from monocular data.
    void Evaluate(const cv::Mat &image, const cv::Mat &depth,
                  const std::vector<cv::KeyPoint> &keys,
                  const std::vector<cv::KeyPoint> *previousKeys,
                  std::vector<float> &dynamicProbability,
                  std::vector<unsigned char> &manhattanImmune) const;

private:
    void WorkerLoop();
    void SubmitImage(const cv::Mat &image) const;
    bool InferDynamicMask(const cv::Mat &image, cv::Mat &mask) const;
    void ApplyManhattanImmunity(const cv::Mat &depth,
                                const std::vector<cv::KeyPoint> &keys,
                                std::vector<float> &probability,
                                std::vector<unsigned char> &immune) const;
    Config mConfig;
    std::string mEnginePath;
    bool mbEngineReady;
    mutable std::mutex mMutex;
    mutable std::condition_variable mCondition;
    mutable cv::Mat mPendingImage;
    mutable cv::Mat mLatestMask;
    bool mbStopWorker;
    mutable bool mbPendingImage;
    std::thread mWorker;
};
}
#endif
