#include <opencv2/opencv.hpp>
#include "metric.hpp"
#include "psnr.hpp"
#include "ssim.hpp"
#include <iostream>

class MetricTools
{
public:
	MetricTools(std::string filename);

	void getVideoInformation(std::string filename);

	/*
	��һ��ͼƬ��Ӹ�˹����
	*/
	cv::Mat addGaussianNoise(cv::Mat image, double mu, double sigma);

	/*
	�����Ƶ֡��Ӹ�˹����֮�����Ƶ
	*/
	void getGaussianNoiseVideo(std::string filename, std::string output_filename,double mu,double sigma);

	/*
	����Ƶ֡���������ƽ������ֵ�����PSNR��
	*/
	double getAveragePSNR(std::string filename, double mu, double sigma);

	/*
	��������
	*/
	double getAveragePSNR(std::string filename, std::string filename_processed);

	/*
	����Ƶ֡���������ƽ�����ṹ������SSIM��
	*/
	double getAverageSSIM(std::string filename, double mu, double sigma);

	/*
	��������
	*/
	double getAverageSSIM(std::string filename, std::string filename_processed);

protected:
	//��������
	int height;
	int width;
	int channel;
	Metric* metric;

	/*
	��ȡ��Ƶ��ƽ��ָ��metric��ssim��psnr����ͨ�����������ʵ�ֵ�
	params:
	    filename:��Ƶ�ļ�·������
		metricName:��Ҫ��ȡ��ָ�����ƣ������� 'psnr'��'ssim'
	*/
	double getAverageMetric(std::string filename, std::string metricName, double mu,double sigma);

	/*
	���������أ��Ƚ�������Ƶ֡������ָ��
	*/
	double getAverageMetric(std::string filename, std::string filename_processed, std::string metricName);
};
#pragma once
