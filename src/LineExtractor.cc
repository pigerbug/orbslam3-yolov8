#include "LineExtractor.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace ORB_SLAM3
{
void LineExtractor::Extract(const cv::Mat &image,
                            std::vector<cv::line_descriptor::KeyLine> &lines,
                            cv::Mat &descriptors, size_t maxLines)
{
    lines.clear();
    descriptors.release();
    if(image.empty()) return;
    cv::Mat gray;
    if(image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::line_descriptor::LSDDetector> detector = cv::line_descriptor::LSDDetector::createLSDDetector();
    detector->detect(gray, lines, 2, 1);
    std::sort(lines.begin(), lines.end(),
              [](const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b)
              { return a.lineLength > b.lineLength; });
    if(lines.size() > maxLines)
        lines.resize(maxLines);
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    descriptor->compute(gray, lines, descriptors);
}
}
