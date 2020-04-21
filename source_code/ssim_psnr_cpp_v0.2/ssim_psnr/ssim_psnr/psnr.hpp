#ifndef PSNR_hpp
#define PSNR_hpp

#include "metric.hpp"

/*
定义一个PSNR类型，继承自Metric父类
*/
class PSNR : public Metric
{
public:
	PSNR(int height, int width, int channel);

	//计算最终的图片PSNR值，适用于单通道、多通道图像，实际上是调用下面的getPSNR来实现的
	double getMetric(const cv::Mat& original, const cv::Mat& processed);

protected:

	// 计算图像的峰值信噪比，返回的是一个float
	double getPSNR(const cv::Mat X, const cv::Mat Y);
};

#endif
#pragma once

