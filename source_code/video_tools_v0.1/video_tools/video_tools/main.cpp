#include "video_tools.hpp"

int main()
{
	VideoTools videoTools = VideoTools(144, 176);
   // videoTools.YuvVideo2RgbVideo("C:/Users/Administrator/Downloads/salesman_qcif/salesman_qcif.yuv","C:/Users/Administrator/Downloads/salesman_qcif/salesman_qcif.avi", cv::COLOR_YUV2BGRA_I420);

	std::string filename = "F:/low_light_video_enhancement/STMKF_TEST/fast_guided_kalman_filter_sourceYUV_2017_v1.0/fast_guided_kalman_filter/fast_guided_kalman_filter/evaluate_videos/source_videos/tempete_cif.y4m";
	std::string filename_output = "F:/low_light_video_enhancement/STMKF_TEST/fast_guided_kalman_filter_sourceYUV_2017_v1.0/fast_guided_kalman_filter/fast_guided_kalman_filter/evaluate_videos/source_videos/tempete_cif.avi";
	videoTools.VideoFormatConvert(filename, filename_output);


	getchar();
	return 0;
}