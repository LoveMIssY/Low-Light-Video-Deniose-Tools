#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>  //�����ļ�����ĺ���
#include <io.h>     //�����ļ��Ŀ�
#include <stdlib.h>

class VideoTools
{
public:
	VideoTools(int h, int w);

	/*
    ����һ���ļ�������������ļ�
    */
	std::vector<std::string> getFiles(std::string fileFolder, std::string fileExtension);
	

	/*
	  ��ȡYUVԭʼͼ������,������Ҫע����ǣ�����ÿ��ֻ��ȡһ��YUV��ʽ��ͼ��
	  ���ڲ��ܽ�YUV�ϲ���һ������Ҫ��Y��U��V�ֿ��洢������һ��vector
	*/
	std::vector<cv::Mat> readYUV(std::string fileFolder, std::string filename);
	
	/*
	��YUV��ʽ����Ƶת����AVI��ʽ����Ҫ֪��������֪����
	height��width
	�Լ�YUV�Ĳ�����洢��ʽ
	ע����������YUV��Ƶ��ʽ��  4:2:0 YUV format��������ת����AVI��ʽ��ʱ��ʹ�ã�
	cv::cvtColor(yuvImg, rgbImg, cv::COLOR_YUV2BGR_I420);
	*/
	void YuvVideo2RgbVideo(std::string filename, std::string filename_output, int ccode);

	/*
	��ͬ��Ƶ��ʽ��ת��������Ĭ���� .y4m ��ʽ�� .avi ��ʽ��ת��
	*/
	void VideoFormatConvert(std::string filename, std::string filename_output);

	/*
	��Y��U��V����ͨ���ϲ���һ������ʹ�õ���NV21�Ĵ洢��ʽ
	*/
	cv::Mat Y_V_U_2_YVU(cv::Mat Y, cv::Mat V, cv::Mat U);

	/*
	����YUV��ʽ��ͼƬ��BGR֮���ת��
	*/
	cv::Mat YUV_2_BGR(cv::Mat YVU, int ccode);

	//���弸�����ԣ���ΪYUV��ʽ���ļ���Ҫ�ֶ�ָ��height��width
	int Width;
	int Height;
	int nWidth;  // Width / 2
	int nHeight; // Height / 2
};


#pragma once
