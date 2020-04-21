/*
代码中的关键参数说明：
depth：深度，指的是8bit 或者是 16bit 32bit之类的，并不是图像的通道数,取值一般是CV_32F等等；
srcI： 指的是原始的引导图像
p：    指的是原始需要滤波的图片
r：    滤波的窗口半径，注意这里是半径哦！不是窗口的直径
d：    滤波的窗口直径
eps：  人为设定的值
s：    下采样的倍数
sampleType： 下采样的方式
*/

#include "fast_guided_filter_non_speed.h"


/* 进行均值滤波的函数
I:表示的是引导图像
r:表示的是滤波器的直径*/
static cv::Mat boxfilter(const cv::Mat &I, int d)
{
	cv::Mat result;
	cv::blur(I, result, cv::Size(d, d));
	return result;
}

/*
类型转化
*/
static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
	if (mat.depth() == depth)
		return mat;

	cv::Mat result;
	mat.convertTo(result, depth);
	return result;
}

/*************FastGuidedFilterImp、FastGuidedFilterGray、FastGuidedFilterGray三个类的定义***********************/

/*定义快速导向滤波的基类
FastGuidedFilterGray  表示的是对单通道灰度图处理，继承此基类
FastGuidedFilterColor 表示的是对三通道彩色图处理，继承此基类
该基类没有直接的构造函数，而是通过构造其子类的构造函数来实现的
*/
class FastGuidedFilterImp
{
public:
	FastGuidedFilterImp(int r, double eps, int s) :d(r), eps(eps), s(s) {} 
	virtual ~FastGuidedFilterImp() {}
	cv::Mat filter(const cv::Mat &p, int sample_type, int depth);

protected:
	int Idepth, d, s ;  //d,s,eps三者的初始化是通过上面的构造函数后面的三个来实现的。
	double eps;

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p, int sampleType) const = 0;
};


class FastGuidedFilterGray : public FastGuidedFilterImp
{
public:
	FastGuidedFilterGray(const cv::Mat &I, int d, double eps, int s, int sampleType);

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p, int sampleType) const;

private:
	//I 指的是经过“下采样”缩放之后的引导图像，I_表示的是原始guide图像，只不过变为了浮点数
	cv::Mat I, I_, mean_I, mean_II, cov_II;
};


class FastGuidedFilterColor : public FastGuidedFilterImp
{
public:
	FastGuidedFilterColor(const cv::Mat &I, int d, double eps, int s, int sampleType);

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p, int sampleType) const;

private:
	std::vector<cv::Mat> srcIchannels, Ichannels;
	cv::Mat mean_I_r, mean_I_g, mean_I_b;
	cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};

/******************************************* 下面是Imp的实现 *****************************************************/

/*
FastGuidedFilterImp的filter方法
*/
cv::Mat FastGuidedFilterImp::filter(const cv::Mat &p, int sampleType,int depth)
{
	cv::Mat p2 = convertTo(p, Idepth);
	cv::resize(p2, p2, cv::Size(p2.cols / s, p2.rows / s), 0, 0, sampleType);//线性下采样

	cv::Mat result;
	if (p.channels() == 1)
	{
		result = filterSingleChannel(p2,sampleType);   //单通道滤波图像，实际上是调用FastGuidedFilterGray的filterSingleChannel方法
	}
	else                                               //三通道滤波图像，实际上是调用FastGuidedFilterColor的filterSingleChannel方法
	{
		std::vector<cv::Mat> p_channels;
		cv::split(p2, p_channels);

		for (std::size_t i = 0; i < p_channels.size(); ++i)
			p_channels[i] = filterSingleChannel(p_channels[i],sampleType);

		cv::merge(p_channels, result);
	}

	return convertTo(result, depth == -1 ? p.depth() : depth);
}

/******************************************* 下面是Gray的实现 *****************************************************/

FastGuidedFilterGray::FastGuidedFilterGray(const cv::Mat &srcI, int d, double eps, int s, int sampleType):FastGuidedFilterImp(d, eps, s)
{

	if (srcI.depth() == CV_32F || srcI.depth() == CV_64F)
		this->I_ = srcI.clone();
	else
		this->I_ = convertTo(srcI, CV_32F);
	cv::resize(this->I_, I, cv::Size(this->I_.cols / s, this->I_.rows / s), 0, 0, sampleType);
	Idepth = I.depth();  //给父类的Idepth属性赋值

	mean_I = boxfilter(I, d);
	mean_II = boxfilter(I.mul(I), d);
	cov_II = mean_II - mean_I.mul(mean_I);
}

cv::Mat FastGuidedFilterGray::filterSingleChannel(const cv::Mat &p, int sampleType) const
{

	cv::Mat mean_P = boxfilter(p, d);
	cv::Mat mean_IP = boxfilter(I.mul(p), d);
	cv::Mat cov_IP = mean_IP - mean_I.mul(mean_P); // this is the covariance of (I, p) in each local patch.

	cv::Mat a = cov_IP / (cov_II + eps);
	cv::Mat b = mean_P - a.mul(mean_I);

	cv::Mat mean_a = boxfilter(a, d);
	cv::Mat mean_b = boxfilter(b, d);
	cv::resize(mean_a, mean_a, cv::Size(I_.cols, I_.rows), 0, 0, sampleType);
	cv::resize(mean_b, mean_b, cv::Size(I_.cols, I_.rows), 0, 0, sampleType);

	return mean_a.mul(I_) + mean_b;
}

/******************************************* 下面是Color的实现 *****************************************************/

FastGuidedFilterColor::FastGuidedFilterColor(const cv::Mat &srcI, int d, double eps, int s, int sampleType) :FastGuidedFilterImp(d, eps, s)// : r(r), eps(eps)
{
	cv::Mat I, I_;
	if (srcI.depth() == CV_32F || srcI.depth() == CV_64F)
		I_ = srcI.clone();
	else
		I_ = convertTo(srcI, CV_32F);
	Idepth = I_.depth();

	cv::split(I_, srcIchannels); //原始图像通道分离
	cv::resize(I_, I, cv::Size(I_.cols / s, I_.rows / s), 0, 0, sampleType); //下采样得到I
	cv::split(I, Ichannels);     //下采样之后的结果通道分离

	mean_I_r = boxfilter(Ichannels[0], d);
	mean_I_g = boxfilter(Ichannels[1], d);
	mean_I_b = boxfilter(Ichannels[2], d);

	// variance of I in each local patch: the matrix Sigma.
	// Note the variance in each local patch is a 3x3 symmetric matrix:
	//           rr, rg, rb
	//   Sigma = rg, gg, gb
	//           rb, gb, bb
	cv::Mat cov_II_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), d) - mean_I_r.mul(mean_I_r) + eps;
	cv::Mat cov_II_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), d) - mean_I_r.mul(mean_I_g);
	cv::Mat cov_II_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), d) - mean_I_r.mul(mean_I_b);
	cv::Mat cov_II_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), d) - mean_I_g.mul(mean_I_g) + eps;
	cv::Mat cov_II_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), d) - mean_I_g.mul(mean_I_b);
	cv::Mat cov_II_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), d) - mean_I_b.mul(mean_I_b) + eps;

	// Inverse of Sigma + eps * I
	invrr = cov_II_gg.mul(cov_II_bb) - cov_II_gb.mul(cov_II_gb);
	invrg = cov_II_gb.mul(cov_II_rb) - cov_II_rg.mul(cov_II_bb);
	invrb = cov_II_rg.mul(cov_II_gb) - cov_II_gg.mul(cov_II_rb);
	invgg = cov_II_rr.mul(cov_II_bb) - cov_II_rb.mul(cov_II_rb);
	invgb = cov_II_rb.mul(cov_II_rg) - cov_II_rr.mul(cov_II_gb);
	invbb = cov_II_rr.mul(cov_II_gg) - cov_II_rg.mul(cov_II_rg);

	cv::Mat covDet = invrr.mul(cov_II_rr) + invrg.mul(cov_II_rg) + invrb.mul(cov_II_rb);

	invrr /= covDet;
	invrg /= covDet;
	invrb /= covDet;
	invgg /= covDet;
	invgb /= covDet;
	invbb /= covDet;
}

cv::Mat FastGuidedFilterColor::filterSingleChannel(const cv::Mat &p, int sampleType) const
{
	cv::Mat mean_p = boxfilter(p, d);

	cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), d);  //对I的每一个通道进行boxFilter
	cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), d);
	cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), d);

	// 计算每一个通道的mean_Ip.即Ip协方差
	cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
	cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
	cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

	cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
	cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
	cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

	cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);

	cv::Mat mean_a_r = boxfilter(a_r, d);
	cv::Mat mean_a_g = boxfilter(a_g, d);
	cv::Mat mean_a_b = boxfilter(a_b, d);
	cv::Mat mean_b = boxfilter(b, d);
	cv::resize(mean_a_r, mean_a_r, cv::Size(srcIchannels[0].cols, srcIchannels[0].rows), 0, 0, sampleType);
	cv::resize(mean_a_g, mean_a_g, cv::Size(srcIchannels[1].cols, srcIchannels[1].rows), 0, 0, sampleType);
	cv::resize(mean_a_b, mean_a_b, cv::Size(srcIchannels[2].cols, srcIchannels[2].rows), 0, 0, sampleType);
	cv::resize(mean_b, mean_b, cv::Size(srcIchannels[2].cols, srcIchannels[2].rows), 0, 0, sampleType);
	return (mean_a_r.mul(srcIchannels[0]) + mean_a_g.mul(srcIchannels[1]) + mean_a_b.mul(srcIchannels[2]) + mean_b);

}

/************************************* 下面是FastGuidedFilter的实现 ************************************************/

/*
下面是FastGuidedFilter类的实现
包含一个构造函数，一个析构函数，一个filter方法
其中在FastGuidedFilter类中定义了实现快速导向滤波的实现基类FastGuidedFilterImp类的一个对象imp，是一个私有成员
imp是具体的导向滤波实现，根据图像是灰度图像还是彩色图像选择不同的实现方式
*/
FastGuidedFilter::FastGuidedFilter(const cv::Mat &I, int r, double eps, int s, int sampleType)
{
	CV_Assert(I.channels() == 1 || I.channels() == 3);

	if (I.channels() == 1)
		imp = new FastGuidedFilterGray(I, 2 * (r / s) + 1, eps, s, sampleType);  //如果是Gray图像
	else
		imp = new FastGuidedFilterColor(I, 2 * (r / s) + 1, eps, s, sampleType); //如果是Color图像
}

FastGuidedFilter::~FastGuidedFilter()
{
	delete imp;
}

cv::Mat FastGuidedFilter::filter(const cv::Mat &p, int sampleType, int depth) const
{
	return imp->filter(p, sampleType, depth);
}

/************************************* 下面是FastGuidedFilter的调用 ************************************************/


cv::Mat fastGuidedFilterWithNonSpeed(const cv::Mat &I, const cv::Mat &p, int r, double eps, int s, int sampleType, int depth)
{
	return FastGuidedFilter(I, r, eps, s, sampleType).filter(p, sampleType, depth);
}
