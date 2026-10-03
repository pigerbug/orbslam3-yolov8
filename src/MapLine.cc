#include "MapLine.h"
namespace ORB_SLAM3 {
long unsigned int MapLine::nNextId=0;
MapLine::MapLine(const Eigen::Vector3f &start,const Eigen::Vector3f &end,Map *map):mStart(start),mEnd(end),mpMap(map){mnId=nNextId++;}
void MapLine::AddObservation(KeyFrame *keyFrame,size_t index){std::lock_guard<std::mutex> lock(mMutex);mObservations[keyFrame]=index;}
std::map<KeyFrame*,size_t> MapLine::GetObservations(){std::lock_guard<std::mutex> lock(mMutex);return mObservations;}
Eigen::Vector3f MapLine::GetStart(){std::lock_guard<std::mutex> lock(mMutex);return mStart;}
Eigen::Vector3f MapLine::GetEnd(){std::lock_guard<std::mutex> lock(mMutex);return mEnd;}
}
