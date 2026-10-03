#ifndef ORB_SLAM3_LINE_EXTRACTOR_H
#define ORB_SLAM3_LINE_EXTRACTOR_H

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>
#include <vector>

namespace ORB_SLAM3
{
class LineExtractor
{
public:
    static void Extract(const cv::Mat &image,
                        std::vector<cv::line_descriptor::KeyLine> &lines,
                        cv::Mat &descriptors);
};
}
#endif
