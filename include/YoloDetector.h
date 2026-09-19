#ifndef ORB_SLAM3_YOLO_DETECTOR_H
#define ORB_SLAM3_YOLO_DETECTOR_H

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ORB_SLAM3 {
struct YoloBoundingBox {
    cv::Rect2f rect; int classId; float confidence;
    YoloBoundingBox(const cv::Rect2f &r=cv::Rect2f(), int id=-1, float score=0.f) : rect(r), classId(id), confidence(score) {}
};

// Adapted from MD-SLAM's TensorRT path. Tracking ownership and its busy-loop
// thread were removed; DynamicFeatureFilter owns scheduling in ORB-SLAM3.
class YoloDetector {
public:
    YoloDetector(); ~YoloDetector();
    bool LoadEngine(const std::string &enginePath);
    bool Detect(const cv::Mat &image, std::vector<YoloBoundingBox> &boxes);
    bool IsReady() const;
    static bool IsDynamicClass(int classId);
private:
    class Impl; std::unique_ptr<Impl> mpImpl;
};
}
#endif
