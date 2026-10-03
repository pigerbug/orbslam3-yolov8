#include "MapLine.h"
namespace ORB_SLAM3 {
long unsigned int MapLine::nNextId=0;
MapLine::MapLine(const Eigen::Vector3f &start,const Eigen::Vector3f &end,Map *map):mStart(start),mEnd(end),mpMap(map),mnBAGlobalForKF(0){mnId=nNextId++;}
void MapLine::AddObservation(KeyFrame *keyFrame,size_t index){std::lock_guard<std::mutex> lock(mMutex);mObservations[keyFrame]=index;}
void MapLine::EraseObservation(KeyFrame *keyFrame){std::lock_guard<std::mutex> lock(mMutex);mObservations.erase(keyFrame);}
std::map<KeyFrame*,size_t> MapLine::GetObservations(){std::lock_guard<std::mutex> lock(mMutex);return mObservations;}
int MapLine::Observations(){std::lock_guard<std::mutex> lock(mMutex);return static_cast<int>(mObservations.size());}
Eigen::Vector3f MapLine::GetStart(){std::lock_guard<std::mutex> lock(mMutex);return mStart;}
Eigen::Vector3f MapLine::GetEnd(){std::lock_guard<std::mutex> lock(mMutex);return mEnd;}
void MapLine::SetEndpoints(const Eigen::Vector3f &start,const Eigen::Vector3f &end){std::lock_guard<std::mutex> lock(mMutex);mStart=start;mEnd=end;}
void MapLine::SetBadFlag(){std::lock_guard<std::mutex> lock(mMutex);mbBad=true;}
bool MapLine::isBad(){std::lock_guard<std::mutex> lock(mMutex);return mbBad;}
}
