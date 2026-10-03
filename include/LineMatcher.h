#ifndef ORB_SLAM3_LINE_MATCHER_H
#define ORB_SLAM3_LINE_MATCHER_H
#include <opencv2/core.hpp>
#include <vector>
namespace ORB_SLAM3 { class LineMatcher { public:
static void Match(const cv::Mat &a, const cv::Mat &b, std::vector<cv::DMatch> &matches); }; }
#endif
