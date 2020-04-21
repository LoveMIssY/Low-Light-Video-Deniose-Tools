#ifndef SSIM_hpp
#define SSIM_hpp

#include "metric.hpp"

/*
定义一个SSIM类型，继承自Metric父类
*/
class SSIM : public Metric 
{
public:

	SSIM(int height, int width, int channel, int win_size, int data_range);

	//计算最终的图片SSIM值，适用于单通道、多通道图像，实际上是调用下面的getSSIM来实现的
	double getMetric(const cv::Mat& original, const cv::Mat& processed);

protected:

	// 计算图像的每一个通道的SSIM值，返回的是一个scalar
	cv::Scalar getSSIM(cv::Mat X, cv::Mat Y, int win_size, int data_range);

private: //这四个数是SSIM需要的，声明为私有的即可
	static const float K1;
	static const float K2;
	int win_size;    
	int data_range;
};

#endif
#pragma once

