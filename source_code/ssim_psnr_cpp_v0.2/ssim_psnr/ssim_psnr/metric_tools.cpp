#include "metric_tools.hpp"

 /*
 构造函数
 */
MetricTools::MetricTools(std::string filename)
{
	getVideoInformation(filename);
}

/*
获取视频帧有有关信息，主要包括三者，height、width、每一帧图像的通道数channel
*/
void MetricTools::getVideoInformation(std::string filename)
{
	cv::VideoCapture videoSource;
	if (!videoSource.open(filename))
	{
		std::cout << "Error on load video..." << std::endl;
	}
	cv::Size size = cv::Size((int)videoSource.get(cv::CAP_PROP_FRAME_WIDTH), (int)videoSource.get(cv::CAP_PROP_FRAME_HEIGHT));
	
	cv::Mat frame;
	videoSource >> frame;

	height = frame.rows;
	width = frame.cols;
	channel = frame.channels();
	videoSource.release(); //关闭视频文件
}

/*
求视频帧的所有针的平均“峰值信噪比PSNR”
*/
double MetricTools::getAveragePSNR(std::string filename, double mu,double sigma)
{
	double psnr = getAverageMetric(filename, "psnr", mu, sigma);
	return psnr;
}

double MetricTools::getAveragePSNR(std::string filename, std::string filename_processed)
{
	double psnr = getAverageMetric(filename, filename_processed, "psnr");
	return psnr;
}

/*
求视频帧的所有针的平均“结构相似性SSIM”
*/
double MetricTools::getAverageSSIM(std::string filename, double mu, double sigma)
{
	double ssim = getAverageMetric(filename, "ssim", mu, sigma);
	return ssim;
}

double MetricTools::getAverageSSIM(std::string filename, std::string filename_processed)
{
	double ssim = getAverageMetric(filename, filename_processed, "ssim");
	return ssim;
}

/*
给图片添加高斯噪声
RNG:即Random Noise Generator
*/
cv::Mat MetricTools::addGaussianNoise(cv::Mat image, double mu, double sigma)
{
	cv::Mat noiseImage(image.size(), image.type());
	cv::Mat noise(image.size(), image.type());
	cv::RNG rng(time(NULL));
	rng.fill(noise, cv::RNG::NORMAL, mu, sigma);
	cv::add(image, noise, noiseImage);
	return noiseImage;
}

/*
输出视频帧添加高斯噪声之后的视频
*/
void MetricTools::getGaussianNoiseVideo(std::string filename, std::string output_filename,double mu,double sigma)
{
	cv::Mat frame, noiseframe;
	cv::VideoCapture videoSource;
	try {	
		int frameCount = 1;

		if (!videoSource.open(filename))
		{
			std::cout << "Error on load video..." << std::endl;
		}

		//写入视频相关的操作
		cv::VideoWriter writer;
		double fps = videoSource.get(cv::CAP_PROP_FPS);
		double ccode = static_cast<int>(videoSource.get(cv::CAP_PROP_FOURCC));
		cv::Size size = cv::Size((int)videoSource.get(cv::CAP_PROP_FRAME_WIDTH),(int)videoSource.get(cv::CAP_PROP_FRAME_HEIGHT));
		writer.open(output_filename, (int)ccode, fps, size, true);  //最后一个参数为false则可以保存灰度帧，如果为true，则需要保存RGB帧

		videoSource >> frame;   //读取“第一帧”图像
		while (1)
		{	
			noiseframe = addGaussianNoise(frame, mu, sigma);  //添加高斯噪声
			cv::Mat combine_img;
			vconcat(frame, noiseframe, combine_img); //将前面两张图片合并

			//显示处理之后的图像以及视频保存
			cv::namedWindow("combine_img", cv::WINDOW_NORMAL);
			imshow("combine_img", combine_img);
			cv::waitKey(1);
			writer.write(noiseframe);  //写入加入噪声的视频

			videoSource >> frame;  //读取下一帧
			frameCount++;

			if (frame.empty()) {
				std::cout << std::endl << "Video ended!" << std::endl;
				break;
			}
		}
	}
	catch (const std::exception & ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	videoSource.release();
}

/**********************下面是一个protected方法的实现***************************/

/*
计算一个视频中所有的视频的添加噪声之后的信噪比PSRN或者是结构相似性SSIM
params:
		filename:视频文件路径名称
		metricName:所要获取的指标名称，可以是 'psnr'，'ssim'
*/
double MetricTools::getAverageMetric(std::string filename, std::string metricName,double mu,double sigma)
{
	cv::Mat frame, noiseframe;
	double single_frame_metric, total_frame_metric = 0, mean_metric;
	cv::VideoCapture videoSource;
	try 
	{
		int frameCount = 1;

		if (!videoSource.open(filename))
		{
			std::cout << "Error on load video..." << std::endl;
		}

		if (metricName == "psnr")
		{
			PSNR psnr = PSNR(height, width, channel);
			metric = &psnr;
			printf("let begin psnr... ...\n");
		}
		if (metricName == "ssim")
		{
			SSIM ssim = SSIM(height, width, channel, 7, 255);
			metric = &ssim;
			printf("let begin ssim... ...\n");
		}

		videoSource >> frame;   //读取"第一帧"图像
		while (1)
		{	
			noiseframe = addGaussianNoise(frame, mu, sigma);  //添加高斯噪声			
			single_frame_metric = metric->getMetric(frame, noiseframe);
			total_frame_metric += single_frame_metric;
			printf("the metric of %d frame is %lf .\n", frameCount, single_frame_metric);

			videoSource >> frame;  //读取下一帧
			frameCount++;

			if (frame.empty()) 
			{
				std::cout << std::endl << "Video ended!" << std::endl;
				break;
			}
		}
		mean_metric = (double)total_frame_metric / frameCount;
		std::cout << "The mean metric of the video is: " << (double)total_frame_metric / frameCount << std::endl;  //最终的平均metric
	}
	catch (const std::exception & ex) 
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	videoSource.release();   //关闭视频
	return mean_metric;
}


/*
函数的评价指标
*/
double MetricTools::getAverageMetric(std::string filename, std::string filename_processed, std::string metricName)
{
	cv::Mat frame, frame_processed;
	double single_frame_metric, total_frame_metric = 0, mean_metric;
	cv::VideoCapture videoSource;
	cv::VideoCapture videoSource_processed;
	try
	{
		int frameCount = 1;

		if (!videoSource.open(filename)|| !videoSource_processed.open(filename_processed))
		{
			std::cout << "Error on load video..." << std::endl;
		}

		if (metricName == "psnr")
		{
			PSNR psnr = PSNR(height, width, channel);
			metric = &psnr;
			printf("let begin psnr... ...\n");
		}
		if (metricName == "ssim")
		{
			SSIM ssim = SSIM(height, width, channel, 7, 255);
			metric = &ssim;
			printf("let begin ssim... ...\n");
		}

		videoSource >> frame;   //读取"第一帧"图像
		videoSource_processed >> frame_processed;

		while (1)
		{			
			single_frame_metric = metric->getMetric(frame, frame_processed);
			total_frame_metric += single_frame_metric;
			printf("the metric of %d frame is %lf .\n", frameCount, single_frame_metric);

			videoSource >> frame;  //读取下一帧
			videoSource_processed >> frame_processed;
			frameCount++;

			if (frame.empty()||frame_processed.empty())
			{
				std::cout << std::endl << "Video ended!" << std::endl;
				break;
			}
		}
		mean_metric = (double)total_frame_metric / frameCount;
		std::cout << "The mean metric of the video is: " << (double)total_frame_metric / frameCount << std::endl;  //最终的平均metric
	}
	catch (const std::exception & ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	videoSource.release();   //关闭视频
	videoSource_processed.release();
	return mean_metric;
}