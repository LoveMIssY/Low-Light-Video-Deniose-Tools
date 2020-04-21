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
	给一张图片添加高斯噪声
	*/
	cv::Mat addGaussianNoise(cv::Mat image, double mu, double sigma);

	/*
	输出视频帧添加高斯噪声之后的视频
	*/
	void getGaussianNoiseVideo(std::string filename, std::string output_filename,double mu,double sigma);

	/*
	求视频帧的所有针的平均“峰值信噪比PSNR”
	*/
	double getAveragePSNR(std::string filename, double mu, double sigma);

	/*
	函数重载
	*/
	double getAveragePSNR(std::string filename, std::string filename_processed);

	/*
	求视频帧的所有针的平均“结构相似性SSIM”
	*/
	double getAverageSSIM(std::string filename, double mu, double sigma);

	/*
	函数重载
	*/
	double getAverageSSIM(std::string filename, std::string filename_processed);

protected:
	//几个属性
	int height;
	int width;
	int channel;
	Metric* metric;

	/*
	获取视频的平均指标metric，ssim和psnr都是通过这个方法来实现的
	params:
	    filename:视频文件路径名称
		metricName:所要获取的指标名称，可以是 'psnr'，'ssim'
	*/
	double getAverageMetric(std::string filename, std::string metricName, double mu,double sigma);

	/*
	函数的重载，比较两个视频帧的评价指标
	*/
	double getAverageMetric(std::string filename, std::string filename_processed, std::string metricName);
};
#pragma once
