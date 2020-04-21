#ifndef PSNR_hpp
#define PSNR_hpp

#include "metric.hpp"

/*
����һ��PSNR���ͣ��̳���Metric����
*/
class PSNR : public Metric
{
public:
	PSNR(int height, int width, int channel);

	//�������յ�ͼƬPSNRֵ�������ڵ�ͨ������ͨ��ͼ��ʵ�����ǵ��������getPSNR��ʵ�ֵ�
	double getMetric(const cv::Mat& original, const cv::Mat& processed);

protected:

	// ����ͼ��ķ�ֵ����ȣ����ص���һ��float
	double getPSNR(const cv::Mat X, const cv::Mat Y);
};

#endif
#pragma once

