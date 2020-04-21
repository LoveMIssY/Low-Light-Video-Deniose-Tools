#include "psnr.hpp"

/*
构造函数
*/
PSNR::PSNR(int h, int w, int c) : Metric(h, w, c)
{
	height = h;
	width = w;
	channel = c;
}

/*
根据不同的通道数目，返回最终的SSIM值，是一个浮点数
params:
	original:   原始真实图片
	processed:  处理之后的图片
return:
	float，最终计算的PSNR值
*/
double PSNR::getMetric(const cv::Mat& original, const cv::Mat& processed)
{
	//通过getSPNR 函数计算峰值信噪比
	double psnr = getPSNR(original, processed);
	return psnr;
}

/*
评价图像的峰值信噪比，需要真实图像和处理之后的图像，但是这里所采用的例子均是没有真实图像的视频
所以是没有办法计算信噪比的。
*/
double PSNR::getPSNR(cv::Mat X, cv::Mat Y)
{
	if (X.size != Y.size)
	{
		throw "原始图像X与处理之后的图像Y尺寸不一样！";
	}

	if (X.channels() != Y.channels())
	{
		throw "原始图像X与处理之后的图像Y通道数目不一样！";
	}

	cv::Mat dst = cv::Mat(X.rows, X.cols, CV_8UC1);
	cv::absdiff(X, Y, dst);            // 求解两张图片的逐元素之差	
	dst.convertTo(dst, CV_32F);        // 将8位无符号转化成32位浮点数
	dst = dst.mul(dst);                // |image1 - image2|^2 ,求差的平方

	cv::Scalar s = sum(dst);           // 每一个通道求和
	
	double sse;
	
	if (channel==1)
	{
		sse = s[0];
	}

	if (channel == 3)
	{
		sse = s[0] + s[1] + s[2];    // 所有通道之和
	}

	if (sse <= 1e-10)  // 对于很小的值
		return 0;
	else
	{
		double  mse = sse / (double)(X.channels() * X.total()); //3个通道总共有多少个元素
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}