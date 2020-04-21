#include "fast_guided_filter.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;

int main(int argv, char *args[])
{
	clock_t start, end;

	Mat src = imread("4.png");
	std::cout << "image size is :" << src.size << std::endl;
	cv::cvtColor(src, src, COLOR_RGB2GRAY);  //转化成灰度图像
	cv::resize(src, src, cv::Size(1280, 720));
	Mat src1 = src.clone();
	Mat src2 = src.clone();
	cv::resize(src2, src2, cv::Size(320, 270));

	/*测试一下关于四次上采样，下采样所耗费的时间*/
	start = clock();
	cv::resize(src, src, cv::Size(320, 270));
	cv::resize(src1, src1, cv::Size(320, 270));
	cv::resize(src, src, cv::Size(1280, 1080));
	cv::resize(src1, src1, cv::Size(1280, 1080));
	end = clock();
	std::cout << "4 times up-down sample is :" << (double)(end - start) << std::endl;

	/*测试一下关于六次均值滤波所耗费的时间*/
	start = clock();
	cv::boxFilter(src2, src2, -1, cv::Size(5, 5));
	cv::boxFilter(src2, src2, -1, cv::Size(5, 5));
	cv::boxFilter(src2, src2, -1, cv::Size(5, 5));
	cv::boxFilter(src2, src2, -1, cv::Size(5, 5));
	cv::boxFilter(src2, src2, -1, cv::Size(5, 5));
	cv::boxFilter(src2, src2, -1, cv::Size(5, 5));

	end = clock();
	std::cout << "6 times boxFilter sample is :" << (double)(end - start) << std::endl;



	Mat result1, result2, result3, result4;
	int r = 9;
	double eps = 0.01 * 255 * 255;

	/*start = clock();
	cv::ximgproc::guidedFilter(src, src, result1, r, eps);
	end = clock();
	std::cout << "guided filter is :" << (double)(end - start) << std::endl;

	Mat src1;
	resize(src, src1, Size(320, 180));
	start = clock();
	cv::ximgproc::guidedFilter(src1, src1, result2, r/4, eps);
	end = clock();
	std::cout << "upsample guided filter is :" << (double)(end - start) << std::endl;*/


	start = clock();
	cv::ximgproc::fastGuidedFilter(src, src, result3, r, eps, 4);  //下采样2倍
	end = clock();
	std::cout << "fast guided filter is :" << (double)(end - start) << std::endl;

	//start = clock();
	//result4 = my_FastGuideFilter(src, src, r, eps, 4);  //下采样2倍
	//end = clock();
	//std::cout << "my fast guided filter is :" << (double)(end - start) << std::endl;



	cv::imshow("src", src);
	//cv::imshow("result1", result1);
	//cv::imshow("result2", result2);
	cv::imshow("result3", result3);
	//cv::imshow("result4", result4);


	cv::waitKey();


	getchar();
	return 0;
}