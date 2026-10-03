#ifndef ORB_SLAM3_MAPLINE_H
#define ORB_SLAM3_MAPLINE_H

#include <Eigen/Core>
#include <map>
#include <mutex>

namespace ORB_SLAM3 { class KeyFrame; class Map;
class MapLine {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapLine(const Eigen::Vector3f &start, const Eigen::Vector3f &end, Map *map);
    void AddObservation(KeyFrame *keyFrame, size_t index);
    std::map<KeyFrame*, size_t> GetObservations();
    Eigen::Vector3f GetStart(); Eigen::Vector3f GetEnd();
    long unsigned int mnId;
private:
    Eigen::Vector3f mStart, mEnd; Map *mpMap; std::map<KeyFrame*, size_t> mObservations; std::mutex mMutex;
    static long unsigned int nNextId;
}; }
#endif
