#ifndef __EDGEAWAREFILTERS_COMMON_HPP__
#define __EDGEAWAREFILTERS_COMMON_HPP__
#ifdef __cplusplus

namespace cv
{
	namespace ximgproc
	{
		int getTotalNumberOfChannels(InputArrayOfArrays src);

		void checkSameSizeAndDepth(InputArrayOfArrays src, Size &sz, int &depth);

		namespace intrinsics
		{
			void add_(float *dst, float *src1, int w);

			void mul(float *dst, float *src1, float *src2, int w);

			void mul(float *dst, float *src1, float src2, int w);

			//dst = alpha*src + beta
			void mad(float *dst, float *src1, float alpha, float beta, int w);

			void add_mul(float *dst, float *src1, float *src2, int w);

			void sub_mul(float *dst, float *src1, float *src2, int w);

			void sub_mad(float *dst, float *src1, float *src2, float c0, int w);

			void det_2x2(float *dst, float *a00, float *a01, float *a10, float *a11, int w);

			void div_det_2x2(float *a00, float *a01, float *a11, int w);

			void div_1x(float *a1, float *b1, int w);

			void inv_self(float *src, int w);


			void sqr_(float *dst, float *src1, int w);

			void sqrt_(float *dst, float *src, int w);

			void sqr_dif(float *dst, float *src1, float *src2, int w);

			void add_sqr_dif(float *dst, float *src1, float *src2, int w);

			void add_sqr(float *dst, float *src1, int w);

			void min_(float *dst, float *src1, float *src2, int w);

			void rf_vert_row_pass(float *curRow, float *prevRow, float alphaVal, int w);
		}

	}
}

#endif
#endif

#pragma once


