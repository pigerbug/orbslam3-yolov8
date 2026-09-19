#include "YoloDetector.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#ifdef ORB_SLAM3_USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace ORB_SLAM3 {
#ifdef ORB_SLAM3_USE_TENSORRT
class YoloDetector::Impl {
public:
    class Logger : public nvinfer1::ILogger { public: void log(Severity, const char*) noexcept override {} } logger;
    nvinfer1::IRuntime *runtime; nvinfer1::ICudaEngine *engine; nvinfer1::IExecutionContext *context;
    cudaStream_t stream; void *bindings[2]; int inputIndex,outputIndex,inputW,inputH; size_t inputSize,outputSize;
    Impl() : runtime(NULL),engine(NULL),context(NULL),stream(NULL),bindings{NULL,NULL},inputIndex(-1),outputIndex(-1),inputW(0),inputH(0),inputSize(0),outputSize(0) {}
    ~Impl() { if(stream) cudaStreamDestroy(stream); if(inputIndex>=0&&bindings[inputIndex]) cudaFree(bindings[inputIndex]); if(outputIndex>=0&&bindings[outputIndex]) cudaFree(bindings[outputIndex]); if(context) context->destroy(); if(engine) engine->destroy(); if(runtime) runtime->destroy(); }
};
#else
class YoloDetector::Impl {};
#endif
YoloDetector::YoloDetector() : mpImpl(new Impl) {} YoloDetector::~YoloDetector() {}
bool YoloDetector::IsDynamicClass(int id) { const int ids[]={0,1,2,3,5,7}; return std::find(ids,ids+6,id)!=ids+6; }

bool YoloDetector::LoadEngine(const std::string &path) {
#ifndef ORB_SLAM3_USE_TENSORRT
    (void)path; return false;
#else
    std::ifstream file(path.c_str(),std::ios::binary); if(!file) return false;
    file.seekg(0,std::ios::end); const size_t bytes=(size_t)file.tellg(); file.seekg(0,std::ios::beg); if(!bytes) return false;
    std::vector<char> model(bytes); file.read(&model[0],(std::streamsize)bytes);
    mpImpl->runtime=nvinfer1::createInferRuntime(mpImpl->logger); mpImpl->engine=mpImpl->runtime?mpImpl->runtime->deserializeCudaEngine(&model[0],bytes):NULL;
    if(!mpImpl->engine||mpImpl->engine->getNbBindings()!=2) return false;
    mpImpl->context=mpImpl->engine->createExecutionContext(); for(int i=0;i<2;++i) if(mpImpl->engine->bindingIsInput(i)) mpImpl->inputIndex=i; else mpImpl->outputIndex=i;
    nvinfer1::Dims in=mpImpl->engine->getBindingDimensions(mpImpl->inputIndex),out=mpImpl->engine->getBindingDimensions(mpImpl->outputIndex);
    // Ultralytics exports an explicit-batch engine: [1,3,H,W] ->
    // [1,84,N].  Keep support for the older implicit [3,H,W] form too.
    const int inputOffset=(in.nbDims==4)?1:0;
    const int outputOffset=(out.nbDims==3)?1:0;
    if(!mpImpl->context || (in.nbDims!=3 && in.nbDims!=4) ||
       (in.nbDims==4 && in.d[0]!=1) || in.d[inputOffset]!=3 ||
       in.d[inputOffset+1]<=0 || in.d[inputOffset+2]<=0 ||
       (out.nbDims-outputOffset)!=2) return false;
    mpImpl->inputH=in.d[inputOffset+1]; mpImpl->inputW=in.d[inputOffset+2];
    mpImpl->inputSize=1; for(int i=0;i<in.nbDims;++i) mpImpl->inputSize*=static_cast<size_t>(in.d[i]);
    mpImpl->outputSize=1; for(int i=0;i<out.nbDims;++i) mpImpl->outputSize*=static_cast<size_t>(out.d[i]);
    return cudaMalloc(&mpImpl->bindings[mpImpl->inputIndex],mpImpl->inputSize*sizeof(float))==cudaSuccess && cudaMalloc(&mpImpl->bindings[mpImpl->outputIndex],mpImpl->outputSize*sizeof(float))==cudaSuccess && cudaStreamCreate(&mpImpl->stream)==cudaSuccess;
#endif
}
bool YoloDetector::IsReady() const {
#ifdef ORB_SLAM3_USE_TENSORRT
    return mpImpl->context&&mpImpl->stream&&mpImpl->inputIndex>=0&&mpImpl->outputIndex>=0;
#else
    return false;
#endif
}
bool YoloDetector::Detect(const cv::Mat &image,std::vector<YoloBoundingBox> &boxes) {
    boxes.clear();
#ifndef ORB_SLAM3_USE_TENSORRT
    (void)image; return false;
#else
    if(!IsReady()||image.empty()) return false;
    cv::Mat rgb; if(image.channels()==1) cv::cvtColor(image,rgb,cv::COLOR_GRAY2RGB); else if(image.channels()==4) cv::cvtColor(image,rgb,cv::COLOR_BGRA2RGB); else cv::cvtColor(image,rgb,cv::COLOR_BGR2RGB);
    const float scale=std::min(mpImpl->inputW/(float)rgb.cols,mpImpl->inputH/(float)rgb.rows); const int rw=cvRound(rgb.cols*scale),rh=cvRound(rgb.rows*scale),px=(mpImpl->inputW-rw)/2,py=(mpImpl->inputH-rh)/2;
    cv::Mat resized,letterbox(mpImpl->inputH,mpImpl->inputW,CV_8UC3,cv::Scalar(114,114,114)); cv::resize(rgb,resized,cv::Size(rw,rh)); resized.copyTo(letterbox(cv::Rect(px,py,rw,rh)));
    std::vector<float> input(mpImpl->inputSize),output(mpImpl->outputSize); for(int y=0;y<mpImpl->inputH;++y) for(int x=0;x<mpImpl->inputW;++x) for(int c=0;c<3;++c) input[c*mpImpl->inputH*mpImpl->inputW+y*mpImpl->inputW+x]=letterbox.at<cv::Vec3b>(y,x)[c]/255.f;
    if(cudaMemcpyAsync(mpImpl->bindings[mpImpl->inputIndex],&input[0],input.size()*sizeof(float),cudaMemcpyHostToDevice,mpImpl->stream)!=cudaSuccess||!mpImpl->context->enqueueV2(mpImpl->bindings,mpImpl->stream,NULL)||cudaMemcpyAsync(&output[0],mpImpl->bindings[mpImpl->outputIndex],output.size()*sizeof(float),cudaMemcpyDeviceToHost,mpImpl->stream)!=cudaSuccess||cudaStreamSynchronize(mpImpl->stream)!=cudaSuccess) return false;
    nvinfer1::Dims d=mpImpl->engine->getBindingDimensions(mpImpl->outputIndex);
    const int offset=(d.nbDims==3)?1:0;
    if((d.nbDims-offset)!=2) return false;
    const bool cf=d.d[offset]<d.d[offset+1];
    const int attrs=cf?d.d[offset]:d.d[offset+1], count=cf?d.d[offset+1]:d.d[offset];
    std::vector<YoloBoundingBox> candidates;
    for(int i=0;i<count;++i) { int id=0; float score=0.f; for(int c=4;c<attrs;++c) {float v=cf?output[c*count+i]:output[i*attrs+c];if(v>score){score=v;id=c-4;}} if(score<.5f) continue; float cx=cf?output[i]:output[i*attrs],cy=cf?output[count+i]:output[i*attrs+1],w=cf?output[2*count+i]:output[i*attrs+2],h=cf?output[3*count+i]:output[i*attrs+3]; cv::Rect2f r((cx-w*.5f-px)/scale,(cy-h*.5f-py)/scale,w/scale,h/scale);r&=cv::Rect2f(0,0,(float)image.cols,(float)image.rows);if(r.area()>0)candidates.push_back(YoloBoundingBox(r,id,score)); }
    std::sort(candidates.begin(),candidates.end(),[](const YoloBoundingBox&a,const YoloBoundingBox&b){return a.confidence>b.confidence;});
    for(size_t i=0;i<candidates.size();++i) { bool suppressed=false; for(size_t j=0;j<boxes.size();++j) {cv::Rect2f in=candidates[i].rect&boxes[j].rect;float uni=candidates[i].rect.area()+boxes[j].rect.area()-in.area();if(candidates[i].classId==boxes[j].classId&&uni>0&&in.area()/uni>.45f){suppressed=true;break;}} if(!suppressed) boxes.push_back(candidates[i]); }
    return true;
#endif
}
}
