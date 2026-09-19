#include "DynamicFeatureFilter.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>

#ifdef ORB_SLAM3_USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#endif

namespace ORB_SLAM3 {
DynamicFeatureFilter::DynamicFeatureFilter(const Config &config)
    : mConfig(config), mbEngineReady(false), mbStopWorker(false), mbPendingImage(false) {}

DynamicFeatureFilter::~DynamicFeatureFilter() {
    { std::lock_guard<std::mutex> lock(mMutex); mbStopWorker=true; }
    mCondition.notify_all();
    if(mWorker.joinable()) mWorker.join();
}

void DynamicFeatureFilter::Configure(const Config &config) { mConfig=config; }

bool DynamicFeatureFilter::LoadTensorRTEngine(const std::string &enginePath) {
    // TensorRT is optional at compile time.  Checking the serialized engine
    // here prevents a bad setting from silently enabling hard feature removal.
    std::ifstream engine(enginePath.c_str(), std::ios::binary);
    mbEngineReady = mConfig.enabled && engine.good();
    mEnginePath = enginePath;
    if(mbEngineReady && !mWorker.joinable()) mWorker=std::thread(&DynamicFeatureFilter::WorkerLoop,this);
    return mbEngineReady;
}

bool DynamicFeatureFilter::IsReady() const { return mbEngineReady; }

void DynamicFeatureFilter::SubmitImage(const cv::Mat &image) const {
    if(!mbEngineReady || image.empty()) return;
    { std::lock_guard<std::mutex> lock(mMutex); mPendingImage=image.clone(); mbPendingImage=true; }
    mCondition.notify_one();
}

void DynamicFeatureFilter::WorkerLoop() {
    while(true) {
        cv::Mat image;
        { std::unique_lock<std::mutex> lock(mMutex); mCondition.wait(lock,[this]{ return mbStopWorker || mbPendingImage; });
          if(mbStopWorker) return; image=mPendingImage; mbPendingImage=false; }
        cv::Mat mask;
        if(InferDynamicMask(image,mask)) { std::lock_guard<std::mutex> lock(mMutex); mLatestMask=mask; }
    }
}

bool DynamicFeatureFilter::InferDynamicMask(const cv::Mat &image, cv::Mat &mask) const {
#ifndef ORB_SLAM3_USE_TENSORRT
    (void)image; mask.release(); return false;
#else
    if(!mbEngineReady || image.empty()) return false;
    class Logger : public nvinfer1::ILogger { public: void log(Severity s, const char* m) noexcept override { if(s<=Severity::kWARNING) std::cerr << "[TensorRT] " << m << std::endl; } };
    static Logger logger;
    static std::string activeEngine;
    static nvinfer1::IRuntime *runtime = NULL;
    static nvinfer1::ICudaEngine *engine = NULL;
    static nvinfer1::IExecutionContext *context = NULL;
    static cudaStream_t stream = NULL;
    static void *bindings[2] = {NULL, NULL};
    static size_t inputSize=0, outputSize=0;
    static int inputIndex=-1, outputIndex=-1, inputW=0, inputH=0;
    if(activeEngine.empty()) {
        std::ifstream file(mEnginePath.c_str(), std::ios::binary);
        file.seekg(0,std::ios::end); const size_t bytes=(size_t)file.tellg(); file.seekg(0,std::ios::beg);
        std::vector<char> serialized(bytes); file.read(&serialized[0],bytes);
        runtime=nvinfer1::createInferRuntime(logger); engine=runtime ? runtime->deserializeCudaEngine(&serialized[0],bytes) : NULL;
        if(!engine || engine->getNbBindings()!=2) return false;
        context=engine->createExecutionContext();
        for(int i=0;i<2;++i) { if(engine->bindingIsInput(i)) inputIndex=i; else outputIndex=i; }
        nvinfer1::Dims in=engine->getBindingDimensions(inputIndex), out=engine->getBindingDimensions(outputIndex);
        if(in.nbDims!=3 || in.d[0]!=3 || out.nbDims!=3 || in.d[1]<=0 || in.d[2]<=0) return false;
        inputH=in.d[1]; inputW=in.d[2]; inputSize=(size_t)in.d[0]*in.d[1]*in.d[2]; outputSize=(size_t)out.d[0]*out.d[1]*out.d[2];
        if(cudaMalloc(&bindings[inputIndex],inputSize*sizeof(float))!=cudaSuccess || cudaMalloc(&bindings[outputIndex],outputSize*sizeof(float))!=cudaSuccess || cudaStreamCreate(&stream)!=cudaSuccess) return false;
        activeEngine=mEnginePath;
    }
    if(activeEngine!=mEnginePath) return false; // one engine per process; avoid unsafe live engine swaps
    cv::Mat rgb; if(image.channels()==1) cv::cvtColor(image,rgb,cv::COLOR_GRAY2RGB); else if(image.channels()==4) cv::cvtColor(image,rgb,cv::COLOR_BGRA2RGB); else cv::cvtColor(image,rgb,cv::COLOR_BGR2RGB);
    const float scale=std::min(inputW/(float)rgb.cols,inputH/(float)rgb.rows); const int rw=cvRound(rgb.cols*scale), rh=cvRound(rgb.rows*scale), px=(inputW-rw)/2, py=(inputH-rh)/2;
    cv::Mat resized, boxed(inputH,inputW,CV_8UC3,cv::Scalar(114,114,114)); cv::resize(rgb,resized,cv::Size(rw,rh)); resized.copyTo(boxed(cv::Rect(px,py,rw,rh)));
    std::vector<float> input(inputSize), output(outputSize);
    for(int y=0;y<inputH;++y) for(int x=0;x<inputW;++x) for(int c=0;c<3;++c) input[c*inputH*inputW+y*inputW+x]=boxed.at<cv::Vec3b>(y,x)[c]/255.f;
    if(cudaMemcpyAsync(bindings[inputIndex],&input[0],inputSize*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess || !context->enqueueV2(bindings,stream,NULL) || cudaMemcpyAsync(&output[0],bindings[outputIndex],outputSize*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess || cudaStreamSynchronize(stream)!=cudaSuccess) return false;
    nvinfer1::Dims out=engine->getBindingDimensions(outputIndex); const bool channelFirst=out.d[1]<out.d[2]; const int attrs=channelFirst?out.d[1]:out.d[2], count=channelFirst?out.d[2]:out.d[1];
    mask=cv::Mat::zeros(image.size(),CV_8U); const int dynamicIds[] = {0,1,2,3,5,7};
    for(int i=0;i<count;++i) { int cls=0; float score=0; for(int c=4;c<attrs;++c) { float value=channelFirst?output[c*count+i]:output[i*attrs+c]; if(value>score) {score=value; cls=c-4;} }
        bool dynamic=false; for(size_t d=0;d<sizeof(dynamicIds)/sizeof(dynamicIds[0]);++d) if(cls==dynamicIds[d]) dynamic=true; if(!dynamic || score<0.5f) continue;
        const float getx=channelFirst?output[i]:output[i*attrs], gety=channelFirst?output[count+i]:output[i*attrs+1], getw=channelFirst?output[2*count+i]:output[i*attrs+2], geth=channelFirst?output[3*count+i]:output[i*attrs+3];
        cv::Rect r(cvRound((getx-getw*.5f-px)/scale),cvRound((gety-geth*.5f-py)/scale),cvRound(getw/scale),cvRound(geth/scale)); r &= cv::Rect(0,0,image.cols,image.rows); if(r.area()>0) mask(r).setTo(255);
    }
    return true;
#endif
}

void DynamicFeatureFilter::ApplyManhattanImmunity(const cv::Mat &depth,
        const std::vector<cv::KeyPoint> &keys, std::vector<float> &probability,
        std::vector<unsigned char> &immune) const {
    if(depth.empty() || depth.type() != CV_32F) return;
    cv::Mat valid = depth > 0.0f;
    std::vector<cv::Point3f> cloud;
    cloud.reserve(keys.size());
    for(size_t i=0;i<keys.size();++i) {
        int x=cvRound(keys[i].pt.x), y=cvRound(keys[i].pt.y);
        if(x>=0 && y>=0 && x<depth.cols && y<depth.rows && valid.at<unsigned char>(y,x))
            cloud.push_back(cv::Point3f((float)x,(float)y,depth.at<float>(y,x)));
    }
    if(cloud.size()<30) return;
    // A dominant plane is a robust ground/wall proxy in camera coordinates.
    cv::Mat samples((int)cloud.size(), 3, CV_32F);
    for(size_t i=0;i<cloud.size();++i) {
        samples.at<float>((int)i,0)=cloud[i].x; samples.at<float>((int)i,1)=cloud[i].y;
        samples.at<float>((int)i,2)=cloud[i].z;
    }
    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f normal(pca.eigenvectors.at<float>(2,0), pca.eigenvectors.at<float>(2,1), pca.eigenvectors.at<float>(2,2));
    const cv::Vec3f center(pca.mean.at<float>(0,0), pca.mean.at<float>(0,1), pca.mean.at<float>(0,2));
    const float norm=std::sqrt(normal.dot(normal));
    if(norm < 1e-6f) return;
    for(size_t i=0;i<keys.size();++i) {
        int x=cvRound(keys[i].pt.x), y=cvRound(keys[i].pt.y);
        if(x<0 || y<0 || x>=depth.cols || y>=depth.rows || !valid.at<unsigned char>(y,x)) continue;
        const cv::Vec3f point((float)x,(float)y,depth.at<float>(y,x));
        const float d=std::fabs(normal.dot(point-center))/norm;
        if(d < mConfig.planeDistance) { probability[i]=0.0f; immune[i]=1; }
    }
}

void DynamicFeatureFilter::Evaluate(const cv::Mat &image, const cv::Mat &depth,
        const std::vector<cv::KeyPoint> &keys, const std::vector<cv::KeyPoint> *previousKeys,
        std::vector<float> &probability, std::vector<unsigned char> &immune) const {
    probability.assign(keys.size(), 0.0f); immune.assign(keys.size(), 0);
    cv::Mat mask;
    { std::lock_guard<std::mutex> lock(mMutex); if(mLatestMask.size()==image.size()) mask=mLatestMask; }
    SubmitImage(image); // Never wait for GPU inference in the tracking thread.
    if(!mask.empty() && mask.type()==CV_8U) {
        for(size_t i=0;i<keys.size();++i) {
            int x=cvRound(keys[i].pt.x), y=cvRound(keys[i].pt.y);
            if(x>=0 && y>=0 && x<mask.cols && y<mask.rows && mask.at<unsigned char>(y,x))
                probability[i]=mConfig.dynamicThreshold;
        }
    }
    // The Sampson term is only a prior; pose optimization still makes the final
    // outlier decision. Index-aligned previous keys are used when available.
    if(previousKeys && previousKeys->size()==keys.size() && keys.size()>=8) {
        std::vector<cv::Point2f> a,b;
        for(size_t i=0;i<keys.size();++i) { a.push_back((*previousKeys)[i].pt); b.push_back(keys[i].pt); }
        cv::Mat F=cv::findFundamentalMat(a,b,cv::FM_RANSAC,1.0,0.99);
        if(!F.empty()) for(size_t i=0;i<keys.size();++i) {
            cv::Mat x1=(cv::Mat_<double>(3,1)<<a[i].x,a[i].y,1.0), x2=(cv::Mat_<double>(3,1)<<b[i].x,b[i].y,1.0);
            cv::Mat Fx1=F*x1, Ftx2=F.t()*x2;
            double e=x2.dot(Fx1), denom=Fx1.at<double>(0)*Fx1.at<double>(0)+Fx1.at<double>(1)*Fx1.at<double>(1)+Ftx2.at<double>(0)*Ftx2.at<double>(0)+Ftx2.at<double>(1)*Ftx2.at<double>(1);
            if(denom>1e-12) probability[i]=std::max(probability[i], (float)(1.0-std::exp(-(e*e/denom)/(mConfig.sampsonScale*mConfig.sampsonScale))));
        }
    }
    ApplyManhattanImmunity(depth, keys, probability, immune);
}
}
