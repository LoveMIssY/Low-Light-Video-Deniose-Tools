#include "ssim.hpp"

const float SSIM::K1 = 0.01;  // 6.5025f;
const float SSIM::K2 = 0.03;  // 58.5225f;

/*
���캯��
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
���ݲ�ͬ��ͨ����Ŀ���������յ�SSIMֵ����һ��������
params:
    original:   ԭʼ��ʵͼƬ
	processed:  ����֮���ͼƬ
return:
    float�����ռ����SSIMֵ
*/
double SSIM::getMetric(const cv::Mat& original, const cv::Mat& processed)
{
	//��������ͨ����ƽ��ֵ,����Scaler��ʾ����һ������4�����ֵ�����
	cv::Scalar mean_ssim_ = getSSIM(original, processed,win_size,data_range);

	if (channel = 1)
	{
		return mean_ssim_[0];      //��ͨ���Ҷ�ͼ��
	}
	
	if (channel = 3)
	{
		return (mean_ssim_[0]+ mean_ssim_[1] + mean_ssim_[2] ) / 3;      //��ͨ���Ҷ�ͼ��
	}
}

/*
ʵ��SSIM����Ҫ���㺯��
params:
    X:          ��ʾ������ʵͼƬ
    Y:          ��ʾ���Ǵ�������ͼƬ
    win_size:   ��ʾ�������ֵ�Ĵ��ڴ�С��Ĭ��ʹ�õ��Ǿ�ֵ�˲����ֵ
    data_range: λ�ķ�Χ��Ϊ255 - 0 = 255
return:
    Scaler;���ص���һ�������������ĸ�Ԫ��
*/
cv::Scalar SSIM::getSSIM(cv::Mat X, cv::Mat Y, int win_size, int data_range)
{
	if (X.size!=Y.size)
	{
		throw "ԭʼͼ��X�봦��֮���ͼ��Y�ߴ粻һ����";
	}
	
	if (X.channels()!=Y.channels())
	{
		throw "ԭʼͼ��X�봦��֮���ͼ��Yͨ����Ŀ��һ����";
	}

	X.convertTo(X, CV_32F);  //��X,Yת��Ϊ������
	Y.convertTo(Y, CV_32F);

	cv::Size size(win_size, win_size);
	//��һ��������X,Y�ľ�ֵ������Ĭ�����õ��Ǿ�ֵ�˲����ֵ
	cv::Mat ux, uy;
	cv::blur(X, ux, size);
	cv::blur(Y, uy, size);

	//�ڶ����������ʾux��ƽ����uy��ƽ��
	cv::Mat uxx, uyy, uxy;
	cv::blur(X.mul(X), uxx, size);
	cv::blur(Y.mul(Y), uyy, size);
	cv::blur(X.mul(Y), uxy, size);

	//������������vx��ʾX�ķ��vy��ʾY�ķ���,vxy��ʾX��Y��Э����
	cv::Mat vx, vy, vxy;
	vx = uxx - ux.mul(ux);
	vy = uyy - uy.mul(uy);
	vxy = uxy - ux.mul(uy);

	int R = data_range;  // ��ʾ������Ƭλ�������ֵ����Сֵ�Ĳ�� 255 - 0 = 255
	double C1, C2;
	C1 = pow(K1 * R, 2);
	C2 = pow(K2 * R, 2);
	
	cv::Mat A1, A2, B1, B2; //��ʾ����SSIMʽ���е��ĸ���
	A1 = 2 * ux.mul(uy) + C1;
	A2 = 2 * vxy + C2;
	B1 = ux.mul(ux) + uy.mul(uy) + C1;
	B2 = vx + vy + C2;

	cv::Mat D, S;    //�����S����ʾ���յĽ�������ǻ���Ҫ��S���ֵ
	D = B1.mul(B2);
	cv::divide(A1.mul(A2), D, S);

	int pad = (win_size - 1) / 2; 
	int width = S.cols - 2 * pad;
	int height = S.rows - 2 * pad;
	cv::Mat S_Crop(S, cv::Rect(pad, pad, width, height));  //Mat����Ƭ������ͨ��Rect��ʵ�ֵ�

	//scaler ָ���� Vec<_Tp, 4>�����ǰ����ĸ�Ԫ�ص������������δ�ǰ����䣬�����ĸ��Ļ������ȫ����0
	cv::Scalar mssim = cv::mean(S_Crop);  
	
	return mssim;
}