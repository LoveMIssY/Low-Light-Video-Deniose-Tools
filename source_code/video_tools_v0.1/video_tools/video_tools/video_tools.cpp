#define _CRT_SECURE_NO_DEPRECATE     //_crt_secure_no_deprecare
#include "video_tools.hpp"


VideoTools::VideoTools(int h,int w)
{
	Height = h;
	Width = w;
	nHeight = Height / 2;
	nWidth = Width / 2;
}

/*
   遍历一个文件夹下面的所有文件
*/
std::vector<std::string> VideoTools::getFiles(std::string fileFolder, std::string fileExtension)
{
	struct _finddata_t FileInfo;  //定义结构体
	intptr_t Handle;              //实际上就是 __int64
	std::string dir = fileFolder.append(fileExtension);
	std::vector<std::string> filenames; //存储所有的文件名称

	if ((Handle = _findfirst(dir.c_str(), &FileInfo)) == -1L)
	{
		printf("没有找到匹配的文件\n");
	}
	else
	{
		//printf("%s\n", FileInfo.name); //输出所有的文件名称，但是不包括前面的目录
		filenames.push_back(FileInfo.name); //写入vector
		while (_findnext(Handle, &FileInfo) == 0)
		{
			//printf("%s\n", FileInfo.name);
			filenames.push_back(FileInfo.name); //写入vector
		}

		_findclose(Handle);  //关闭句柄
	}
	return filenames;
}

/*
  读取YUV原始图像数据,但是需要注意的是，这里每次只读取一张YUV格式的图像
  鉴于不能将YUV合并在一起处理，需要将Y、U、V分开存储，返回一个vector
*/
std::vector<cv::Mat> VideoTools::readYUV(std::string fileFolder, std::string filename)
{
	std::vector<cv::Mat> YVU;
	cv::Mat Y(cv::Size(Width, Height), CV_8UC1);
	cv::Mat U(cv::Size(nWidth, nHeight), CV_8UC1);
	cv::Mat V(cv::Size(nWidth, nHeight), CV_8UC1);

	FILE* f;
	std::string full_filename = fileFolder.append(filename);
	if (!(f = fopen(full_filename.c_str(), "rb")))  //c_str()的作用是获取字符串的首地址，返回const char * 类型
	{
		printf("无法打开文件,或者是所有的文件处理完毕！！!");
	}

	for (int h = 0; h < Height; h++)
	{
		for (int w = 0; w < Width; w++)
		{
			uchar p[1];  //其实就是要分配一个uchar大小的内存，创建数组就是分配内存，也可以使用动态内存分配，但是需要手动释放
			fread(p, sizeof(uchar), 1, f);
			Y.at<uchar>(h, w) = *p;
		}
	}

	for (int h = 0; h < nHeight; h++)
	{
		for (int w = 0; w < nWidth; w++)
		{
			uchar  p[1];
			fread(p, sizeof(uchar), 1, f);
			V.at<uchar>(h, w) = *p;
		}
	}

	for (int h = 0; h < nHeight; h++)
	{
		for (int w = 0; w < nWidth; w++)
		{
			uchar p[1];
			fread(p, sizeof(uchar), 1, f);
			U.at<uchar>(h, w) = *p;
		}
	}

	YVU.push_back(Y);
	YVU.push_back(V);
	YVU.push_back(U);


	//Y = Y.reshape(0, 1);
	//V = V.reshape(0, 1);
	//U = U.reshape(0, 1);

	//Mat YV, YVU;
	//hconcat(Y, V, YV);
	//hconcat(YV, U, YVU);
	//YVU = YVU.reshape(0, Height * 3 / 2);

	//fclose(f);

	//return YVU;  //返回一个(3*height/2,width)大小的 Mat
	fclose(f);

	return YVU;
}

/*
将YUV格式的视频转化成AVI格式，需要知道几个已知条件
height，width
以及YUV的采样与存储格式
注意的是这里的YUV视频格式是  4:2:0 YUV format，所以在转化成AVI格式的时候，使用：
cv::cvtColor(yuvImg, rgbImg, cv::COLOR_YUV2BGR_I420);
*/
void VideoTools::YuvVideo2RgbVideo(std::string filename, std::string filename_output, int ccode)
{
	cv::VideoWriter writer;
	cv::Size size = cv::Size(Width, Height);
	int fourcc = writer.fourcc('X', 'V', 'I', 'D');
	writer.open(filename_output, fourcc, 25.0, size, true);

	FILE* f;
	if (!(f = fopen(filename.c_str(), "rb+")))  //c_str 的作用是获取字符串的首地址
	{
		printf("打开YUV视频文件出现错误... ...");
	}

	// 计算一共有多少帧视频，定位到文件最结尾出
	fseek(f, 0, SEEK_END);   //如果一切正常，fseek函数的返回值为0；如果出现错误(例如试图移动的距离超出了文件的范围），其返回值是-1。
	int frame_count = 0;
	frame_count = (int)((int)ftell(f) / ((Width * Height * 3) / 2));  //成功则返回当前的读写位置，失败返回 -1

	int framesize = Height * Width * 3 / 2;  //每一帧的大小
	//unsigned char* buffer = new unsigned char[framesize]; //一帧数据大小 ,使用数组分配内存或者是动态分配都可以
	void* buffer = malloc(framesize * sizeof(unsigned char));

	//从数据流中读取 按长度读取数据  下一次循环直接从上一帧的末尾开始读取
	fseek(f, 0, SEEK_SET);  //先定位到文件最开始

	for (int i = 0; i < frame_count; i++)
	{
		fread(buffer, framesize * sizeof(unsigned char), 1, f); //在整个yuv中截取一帧图像读入  
		cv::Mat yuvImage;
		yuvImage.create(Height * 3 / 2, Width, CV_8UC1);      //创建整张YUV图片的大小
		memcpy(yuvImage.data, buffer, framesize * sizeof(unsigned char));  //将buffer里面的数据放进YUV

		cv::Mat rgbImage = cv::Mat(Height, Width, CV_8UC3);
		cv::cvtColor(yuvImage, rgbImage, cv::COLOR_YUV2BGR_I420);  //YUV到RGB的转换

		writer.write(rgbImage);
		printf("第 %d 帧已经转化完成！\n", i + 1);
	}

	free(buffer);  //手动释放动态申请的内存
	fclose(f);
}

/*
不同视频格式的转换，这里默认是 .y4m 格式到 .avi 格式的转化
*/
void VideoTools::VideoFormatConvert(std::string filename, std::string filename_output)
{
	cv::Mat frame;
	int frameCount = 1;

	cv::VideoCapture videoCapture;
	if (!videoCapture.open(filename))
	{
		std::cout << "视频读取出现错误... ..." << std::endl;
	}

	int height = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int width = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);

    //写入视频相关的定义
	cv::VideoWriter writer;
	cv::Size size = cv::Size(width, height);
	int fourcc = writer.fourcc('X', 'V', 'I', 'D');
	writer.open(filename_output, fourcc, 25.0, size, true);

	while (1)
	{
		videoCapture >> frame;
		if (frame.empty())
		{
			std::cout << "video end!" << std::endl;
			break;
		}
		writer.write(frame);
		printf("第 %d 帧处理完毕.\n", frameCount);
		frameCount++;
	}
}

/*
将Y、U、V三个通道合并在一起，这里使用的是NV21的存储格式
*/
cv::Mat VideoTools::Y_V_U_2_YVU(cv::Mat Y, cv::Mat V, cv::Mat U)
{
	Y = Y.reshape(0, 1);
	V = V.reshape(0, 1);
	U = U.reshape(0, 1);

	cv::Mat YV, YVU;
	cv::hconcat(Y, V, YV);
	cv::hconcat(YV, U, YVU);
	YVU = YVU.reshape(0, Height * 3 / 2);

	return YVU;  //返回一个(3*height/2,width)大小的 Mat
}

/*
单张YUV格式的图片到BGR之间的转换,但是这里需要注意的是，只提供
NV21 编码到RGB 的转换
ccode:转化的方式，默认是 cv::COLOR_YUV2BGR_NV21 = 93
*/
cv::Mat VideoTools::YUV_2_BGR(cv::Mat YVU, int ccode)
{
	cv::Mat bgr_img;
	cv::cvtColor(YVU, bgr_img, ccode);
	return bgr_img;
}


