#include "ssim.hpp"
#include "psnr.hpp"
#include "metric_tools.hpp"
#include <iostream>


using namespace std;

/*给图片添加高斯噪声
RNG:即Random Noise Generator
*/
cv::Mat addGaussianNoise(cv::Mat image, double mu, double sigma)
{
	cv::Mat noiseImage(image.size(), image.type());
	cv::Mat noise(image.size(), image.type());
	cv::RNG rng(time(NULL));
	rng.fill(noise, cv::RNG::NORMAL, mu, sigma);
	cv::add(image, noise, noiseImage);

	return noiseImage;
}


void getNoiseVideo()
{
	MetricTools metricTools = MetricTools("videos/foreman_cif.y4m");
	//double mean_metric = metricTools.getAverageSSIM("videos/foreman_cif.y4m", 0, 20);
	//printf("++++++++the mean metric is : %f +++++++\n ", mean_metric);
	
	
	metricTools.getGaussianNoiseVideo("videos/foreman_cif.y4m", "videos/foreman_cif_noise.avi",0,20);
}



int main(int argv, char* args[])
{
	/*cv::Mat image = cv::imread("4.png");
	cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
	cv::Mat blured;
	cv::blur(image, blured, cv::Size(5, 5));

	SSIM ssim = SSIM(image.rows, image.cols,image.channels(),7,255);
	double ssim_value = ssim.getMetric(image, blured);

	PSNR psnr = PSNR(image.rows, image.cols, image.channels());
	double psnr_value = psnr.getMetric(image, blured);

	cout << "the ssim value of image is : " << ssim_value << endl;
	cout << "the psnr value of image is : " << psnr_value << endl;

	cv::imshow("image", image);
	cv::imshow("blured", blured);
	cv::waitKey();*/

	//getNoiseVideo();

	string filename = "F:/low_light_video_enhancement/STMKF_TEST/fast_guided_kalman_filter_sourceYUV_2017_v1.0/fast_guided_kalman_filter/fast_guided_kalman_filter/evaluate_videos/source_videos/akiyo_cif.y4m";
	string filename_processed = "F:/low_light_video_enhancement/STMKF_TEST/fast_guided_kalman_filter_sourceYUV_2017_v1.0/fast_guided_kalman_filter/fast_guided_kalman_filter/evaluate_videos/process_videos/akiyo_denoise.avi";
	MetricTools metricTools = MetricTools(filename);
	metricTools.getAverageSSIM(filename, filename_processed);

	getchar();
	return 0;
}