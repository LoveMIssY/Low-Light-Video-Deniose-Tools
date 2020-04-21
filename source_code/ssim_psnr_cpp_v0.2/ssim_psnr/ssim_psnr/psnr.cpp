#include "psnr.hpp"

/*
���캯��
*/
PSNR::PSNR(int h, int w, int c) : Metric(h, w, c)
{
	height = h;
	width = w;
	channel = c;
}

/*
���ݲ�ͬ��ͨ����Ŀ���������յ�SSIMֵ����һ��������
params:
	original:   ԭʼ��ʵͼƬ
	processed:  ����֮���ͼƬ
return:
	float�����ռ����PSNRֵ
*/
double PSNR::getMetric(const cv::Mat& original, const cv::Mat& processed)
{
	//ͨ��getSPNR ���������ֵ�����
	double psnr = getPSNR(original, processed);
	return psnr;
}

/*
����ͼ��ķ�ֵ����ȣ���Ҫ��ʵͼ��ʹ���֮���ͼ�񣬵������������õ����Ӿ���û����ʵͼ�����Ƶ
������û�а취��������ȵġ�
*/
double PSNR::getPSNR(cv::Mat X, cv::Mat Y)
{
	if (X.size != Y.size)
	{
		throw "ԭʼͼ��X�봦��֮���ͼ��Y�ߴ粻һ����";
	}

	if (X.channels() != Y.channels())
	{
		throw "ԭʼͼ��X�봦��֮���ͼ��Yͨ����Ŀ��һ����";
	}

	cv::Mat dst = cv::Mat(X.rows, X.cols, CV_8UC1);
	cv::absdiff(X, Y, dst);            // �������ͼƬ����Ԫ��֮��	
	dst.convertTo(dst, CV_32F);        // ��8λ�޷���ת����32λ������
	dst = dst.mul(dst);                // |image1 - image2|^2 ,����ƽ��

	cv::Scalar s = sum(dst);           // ÿһ��ͨ�����
	
	double sse;
	
	if (channel==1)
	{
		sse = s[0];
	}

	if (channel == 3)
	{
		sse = s[0] + s[1] + s[2];    // ����ͨ��֮��
	}

	if (sse <= 1e-10)  // ���ں�С��ֵ
		return 0;
	else
	{
		double  mse = sse / (double)(X.channels() * X.total()); //3��ͨ���ܹ��ж��ٸ�Ԫ��
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}