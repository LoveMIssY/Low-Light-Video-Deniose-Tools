#ifndef __OPENCV_EDGEFILTER_HPP__
#define __OPENCV_EDGEFILTER_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{
	namespace ximgproc
	{


		class CV_EXPORTS_W GuidedFilter : public Algorithm
		{
		public:

			/** @brief Apply Guided Filter to the filtering image.

			@param src filtering image with any numbers of channels.

			@param dst output image.

			@param dDepth optional depth of the output image. dDepth can be set to -1, which will be equivalent
			to src.depth().
			 */
			CV_WRAP virtual void filter(InputArray src, OutputArray dst, int dDepth = -1) = 0;
		};
	}
}
#endif
#endif
#pragma once

