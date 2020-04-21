#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>  //包含文件处理的函数
#include <io.h>     //遍历文件的库
#include <stdlib.h>

class VideoTools
{
public:
	VideoTools(int h, int w);

	/*
    遍历一个文件夹下面的所有文件
    */
	std::vector<std::string> getFiles(std::string fileFolder, std::string fileExtension);
	

	/*
	  读取YUV原始图像数据,但是需要注意的是，这里每次只读取一张YUV格式的图像
	  鉴于不能将YUV合并在一起处理，需要将Y、U、V分开存储，返回一个vector
	*/
	std::vector<cv::Mat> readYUV(std::string fileFolder, std::string filename);
	
	/*
	将YUV格式的视频转化成AVI格式，需要知道几个已知条件
	height，width
	以及YUV的采样与存储格式
	注意的是这里的YUV视频格式是  4:2:0 YUV format，所以在转化成AVI格式的时候，使用：
	cv::cvtColor(yuvImg, rgbImg, cv::COLOR_YUV2BGR_I420);
	*/
	void YuvVideo2RgbVideo(std::string filename, std::string filename_output, int ccode);

	/*
	不同视频格式的转换，这里默认是 .y4m 格式到 .avi 格式的转化
	*/
	void VideoFormatConvert(std::string filename, std::string filename_output);

	/*
	将Y、U、V三个通道合并在一起，这里使用的是NV21的存储格式
	*/
	cv::Mat Y_V_U_2_YVU(cv::Mat Y, cv::Mat V, cv::Mat U);

	/*
	单张YUV格式的图片到BGR之间的转换
	*/
	cv::Mat YUV_2_BGR(cv::Mat YVU, int ccode);

	//定义几个属性，因为YUV格式的文件需要手动指定height和width
	int Width;
	int Height;
	int nWidth;  // Width / 2
	int nHeight; // Height / 2
};


#pragma once
