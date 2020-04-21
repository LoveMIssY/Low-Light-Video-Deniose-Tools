#ifndef Metric_hpp
#define Metric_hpp

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/base.hpp>  //这里面包含了CV_Assert的定义

class Metric 
{
public:
	Metric(int height, int width, int channel);
	virtual ~Metric();

	/*
	定义一个虚方法，在子类中重写虚方法，不同的子类实现会获得不同的评价指标，本文主要实现
	两种，分别是PSNR和SSIM
	original:   表示原始图像
	processed:  表示处理之后的图像
	*/
	virtual double getMetric(const cv::Mat& original, const cv::Mat& processed) = 0;
	
	//定义三个公共属性
	int height;
	int width;
	int channel;
};

#endif
#pragma once
