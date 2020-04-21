#ifndef Metric_hpp
#define Metric_hpp

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/base.hpp>  //�����������CV_Assert�Ķ���

class Metric 
{
public:
	Metric(int height, int width, int channel);
	virtual ~Metric();

	/*
	����һ���鷽��������������д�鷽������ͬ������ʵ�ֻ��ò�ͬ������ָ�꣬������Ҫʵ��
	���֣��ֱ���PSNR��SSIM
	original:   ��ʾԭʼͼ��
	processed:  ��ʾ����֮���ͼ��
	*/
	virtual double getMetric(const cv::Mat& original, const cv::Mat& processed) = 0;
	
	//����������������
	int height;
	int width;
	int channel;
};

#endif
#pragma once
