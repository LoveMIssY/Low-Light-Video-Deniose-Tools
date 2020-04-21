#ifndef SSIM_hpp
#define SSIM_hpp

#include "metric.hpp"

/*
����һ��SSIM���ͣ��̳���Metric����
*/
class SSIM : public Metric 
{
public:

	SSIM(int height, int width, int channel, int win_size, int data_range);

	//�������յ�ͼƬSSIMֵ�������ڵ�ͨ������ͨ��ͼ��ʵ�����ǵ��������getSSIM��ʵ�ֵ�
	double getMetric(const cv::Mat& original, const cv::Mat& processed);

protected:

	// ����ͼ���ÿһ��ͨ����SSIMֵ�����ص���һ��scalar
	cv::Scalar getSSIM(cv::Mat X, cv::Mat Y, int win_size, int data_range);

private: //���ĸ�����SSIM��Ҫ�ģ�����Ϊ˽�еļ���
	static const float K1;
	static const float K2;
	int win_size;    
	int data_range;
};

#endif
#pragma once

