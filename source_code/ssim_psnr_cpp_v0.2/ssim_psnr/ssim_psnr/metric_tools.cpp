#include "metric_tools.hpp"

 /*
 ���캯��
 */
MetricTools::MetricTools(std::string filename)
{
	getVideoInformation(filename);
}

/*
��ȡ��Ƶ֡���й���Ϣ����Ҫ�������ߣ�height��width��ÿһ֡ͼ���ͨ����channel
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
	videoSource.release(); //�ر���Ƶ�ļ�
}

/*
����Ƶ֡���������ƽ������ֵ�����PSNR��
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
����Ƶ֡���������ƽ�����ṹ������SSIM��
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
��ͼƬ��Ӹ�˹����
RNG:��Random Noise Generator
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
�����Ƶ֡��Ӹ�˹����֮�����Ƶ
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

		//д����Ƶ��صĲ���
		cv::VideoWriter writer;
		double fps = videoSource.get(cv::CAP_PROP_FPS);
		double ccode = static_cast<int>(videoSource.get(cv::CAP_PROP_FOURCC));
		cv::Size size = cv::Size((int)videoSource.get(cv::CAP_PROP_FRAME_WIDTH),(int)videoSource.get(cv::CAP_PROP_FRAME_HEIGHT));
		writer.open(output_filename, (int)ccode, fps, size, true);  //���һ������Ϊfalse����Ա���Ҷ�֡�����Ϊtrue������Ҫ����RGB֡

		videoSource >> frame;   //��ȡ����һ֡��ͼ��
		while (1)
		{	
			noiseframe = addGaussianNoise(frame, mu, sigma);  //��Ӹ�˹����
			cv::Mat combine_img;
			vconcat(frame, noiseframe, combine_img); //��ǰ������ͼƬ�ϲ�

			//��ʾ����֮���ͼ���Լ���Ƶ����
			cv::namedWindow("combine_img", cv::WINDOW_NORMAL);
			imshow("combine_img", combine_img);
			cv::waitKey(1);
			writer.write(noiseframe);  //д�������������Ƶ

			videoSource >> frame;  //��ȡ��һ֡
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

/**********************������һ��protected������ʵ��***************************/

/*
����һ����Ƶ�����е���Ƶ���������֮��������PSRN�����ǽṹ������SSIM
params:
		filename:��Ƶ�ļ�·������
		metricName:��Ҫ��ȡ��ָ�����ƣ������� 'psnr'��'ssim'
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

		videoSource >> frame;   //��ȡ"��һ֡"ͼ��
		while (1)
		{	
			noiseframe = addGaussianNoise(frame, mu, sigma);  //��Ӹ�˹����			
			single_frame_metric = metric->getMetric(frame, noiseframe);
			total_frame_metric += single_frame_metric;
			printf("the metric of %d frame is %lf .\n", frameCount, single_frame_metric);

			videoSource >> frame;  //��ȡ��һ֡
			frameCount++;

			if (frame.empty()) 
			{
				std::cout << std::endl << "Video ended!" << std::endl;
				break;
			}
		}
		mean_metric = (double)total_frame_metric / frameCount;
		std::cout << "The mean metric of the video is: " << (double)total_frame_metric / frameCount << std::endl;  //���յ�ƽ��metric
	}
	catch (const std::exception & ex) 
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	videoSource.release();   //�ر���Ƶ
	return mean_metric;
}


/*
����������ָ��
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

		videoSource >> frame;   //��ȡ"��һ֡"ͼ��
		videoSource_processed >> frame_processed;

		while (1)
		{			
			single_frame_metric = metric->getMetric(frame, frame_processed);
			total_frame_metric += single_frame_metric;
			printf("the metric of %d frame is %lf .\n", frameCount, single_frame_metric);

			videoSource >> frame;  //��ȡ��һ֡
			videoSource_processed >> frame_processed;
			frameCount++;

			if (frame.empty()||frame_processed.empty())
			{
				std::cout << std::endl << "Video ended!" << std::endl;
				break;
			}
		}
		mean_metric = (double)total_frame_metric / frameCount;
		std::cout << "The mean metric of the video is: " << (double)total_frame_metric / frameCount << std::endl;  //���յ�ƽ��metric
	}
	catch (const std::exception & ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	videoSource.release();   //�ر���Ƶ
	videoSource_processed.release();
	return mean_metric;
}