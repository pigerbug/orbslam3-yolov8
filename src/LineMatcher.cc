#include "LineMatcher.h"
#include <opencv2/features2d.hpp>
namespace ORB_SLAM3 { void LineMatcher::Match(const cv::Mat &a,const cv::Mat &b,std::vector<cv::DMatch> &matches){
matches.clear(); if(a.empty()||b.empty()) return; cv::BFMatcher matcher(cv::NORM_HAMMING,true); matcher.match(a,b,matches);
std::vector<cv::DMatch> good; for(size_t i=0;i<matches.size();++i) if(matches[i].distance<50.f) good.push_back(matches[i]); matches.swap(good); } }
