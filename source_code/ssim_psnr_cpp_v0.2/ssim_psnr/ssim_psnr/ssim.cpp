#include "ssim.hpp"

const float SSIM::K1 = 0.01;  // 6.5025f;
const float SSIM::K2 = 0.03;  // 58.5225f;

/*
构造函数
*/
SSIM::SSIM(int h, int w, int c,int win_size_, int data_range_):Metric(h,w,c)
{
	height = h;
	width = w;
	channel = c;
	win_size = win_size_;
	data_range = data_range_;
}

/*
根据不同的通道数目，返回最终的SSIM值，是一个浮点数
params:
    original:   原始真实图片
	processed:  处理之后的图片
return:
    float，最终计算的SSIM值
*/
double SSIM::getMetric(const cv::Mat& original, const cv::Mat& processed)
{
	//计算所有通道的平均值,由于Scaler表示的是一个包含4个数字的向量
	cv::Scalar mean_ssim_ = getSSIM(original, processed,win_size,data_range);

	if (channel = 1)
	{
		return mean_ssim_[0];      //单通道灰度图像
	}
	
	if (channel = 3)
	{
		return (mean_ssim_[0]+ mean_ssim_[1] + mean_ssim_[2] ) / 3;      //单通道灰度图像
	}
}

/*
实现SSIM的主要计算函数
params:
    X:          表示的是真实图片
    Y:          表示的是处理过后的图片
    win_size:   表示的是求均值的窗口大小，默认使用的是均值滤波求均值
    data_range: 位的范围，为255 - 0 = 255
return:
    Scaler;返回的是一个向量，包含四个元素
*/
cv::Scalar SSIM::getSSIM(cv::Mat X, cv::Mat Y, int win_size, int data_range)
{
	if (X.size!=Y.size)
	{
		throw "原始图像X与处理之后的图像Y尺寸不一样！";
	}
	
	if (X.channels()!=Y.channels())
	{
		throw "原始图像X与处理之后的图像Y通道数目不一样！";
	}

	X.convertTo(X, CV_32F);  //将X,Y转化为浮点数
	Y.convertTo(Y, CV_32F);

	cv::Size size(win_size, win_size);
	//第一步：计算X,Y的均值，这里默认是用的是均值滤波求均值
	cv::Mat ux, uy;
	cv::blur(X, ux, size);
	cv::blur(Y, uy, size);

	//第二步：这里表示ux的平方和uy的平方
	cv::Mat uxx, uyy, uxy;
	cv::blur(X.mul(X), uxx, size);
	cv::blur(Y.mul(Y), uyy, size);
	cv::blur(X.mul(Y), uxy, size);

	//第三步：这里vx表示X的方差，vy表示Y的方差,vxy表示X与Y的协方差
	cv::Mat vx, vy, vxy;
	vx = uxx - ux.mul(ux);
	vy = uyy - uy.mul(uy);
	vxy = uxy - ux.mul(uy);

	int R = data_range;  // 表示的是相片位数，最大值与最小值的差，即 255 - 0 = 255
	double C1, C2;
	C1 = pow(K1 * R, 2);
	C2 = pow(K2 * R, 2);
	
	cv::Mat A1, A2, B1, B2; //表示的是SSIM式子中的四个项
	A1 = 2 * ux.mul(uy) + C1;
	A2 = 2 * vxy + C2;
	B1 = ux.mul(ux) + uy.mul(uy) + C1;
	B2 = vx + vy + C2;

	cv::Mat D, S;    //求出的S即表示最终的结果，但是还需要对S求均值
	D = B1.mul(B2);
	cv::divide(A1.mul(A2), D, S);

	int pad = (win_size - 1) / 2; 
	int width = S.cols - 2 * pad;
	int height = S.rows - 2 * pad;
	cv::Mat S_Crop(S, cv::Rect(pad, pad, width, height));  //Mat的切片访问是通过Rect来实现的

	//scaler 指的是 Vec<_Tp, 4>，总是包含四个元素的向量，会依次从前面填充，不足四个的话后面的全部是0
	cv::Scalar mssim = cv::mean(S_Crop);  
	
	return mssim;
}