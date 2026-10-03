#include "LineExtractor.h"
#include <opencv2/imgproc.hpp>

namespace ORB_SLAM3
{
void LineExtractor::Extract(const cv::Mat &image,
                            std::vector<cv::line_descriptor::KeyLine> &lines,
                            cv::Mat &descriptors)
{
    lines.clear();
    descriptors.release();
    if(image.empty()) return;
    cv::Mat gray;
    if(image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::line_descriptor::LSDDetector> detector = cv::line_descriptor::LSDDetector::createLSDDetector();
    detector->detect(gray, lines, 2, 1);
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    descriptor->compute(gray, lines, descriptors);
}
}
