#include "precomp.hpp"    //一共依赖这五个OpenCV文件
#include "edgeaware_filters_common.hpp"
#include <opencv2/core/cvdef.h>
#include <opencv2/core/utility.hpp>

#include <cmath>
using namespace std;

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

#if CV_SSE
namespace
{

	inline bool CPU_SUPPORT_SSE1()
	{
		static const bool is_supported = cv::checkHardwareSupport(CV_CPU_SSE);
		return is_supported;
	}

}  // end
#endif

namespace cv
{
	namespace ximgproc
	{
		int getTotalNumberOfChannels(InputArrayOfArrays src)
		{
			CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

			if (src.isMat() || src.isUMat())
			{
				return src.channels();
			}
			else if (src.isMatVector())
			{
				int cnNum = 0;
				const vector<Mat>& srcv = *static_cast<const vector<Mat>*>(src.getObj());
				for (unsigned i = 0; i < srcv.size(); i++)
					cnNum += srcv[i].channels();
				return cnNum;
			}
			else if (src.isUMatVector())
			{
				int cnNum = 0;
				const vector<UMat>& srcv = *static_cast<const vector<UMat>*>(src.getObj());
				for (unsigned i = 0; i < srcv.size(); i++)
					cnNum += srcv[i].channels();
				return cnNum;
			}
			else
			{
				return 0;
			}
		}

		void checkSameSizeAndDepth(InputArrayOfArrays src, Size &sz, int &depth)
		{
			CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

			if (src.isMat() || src.isUMat())
			{
				CV_Assert(!src.empty());
				sz = src.size();
				depth = src.depth();
			}
			else if (src.isMatVector())
			{
				const vector<Mat>& srcv = *static_cast<const vector<Mat>*>(src.getObj());
				CV_Assert(srcv.size() > 0);
				for (unsigned i = 0; i < srcv.size(); i++)
				{
					CV_Assert(srcv[i].depth() == srcv[0].depth());
					CV_Assert(srcv[i].size() == srcv[0].size());
				}
				sz = srcv[0].size();
				depth = srcv[0].depth();
			}
			else if (src.isUMatVector())
			{
				const vector<UMat>& srcv = *static_cast<const vector<UMat>*>(src.getObj());
				CV_Assert(srcv.size() > 0);
				for (unsigned i = 0; i < srcv.size(); i++)
				{
					CV_Assert(srcv[i].depth() == srcv[0].depth());
					CV_Assert(srcv[i].size() == srcv[0].size());
				}
				sz = srcv[0].size();
				depth = srcv[0].depth();
			}
		}

		namespace intrinsics
		{

			inline float getFloatSignBit()
			{
				union
				{
					int signInt;
					float signFloat;
				};
				signInt = 0x80000000;

				return signFloat;
			}

			void add_(float *dst, float *src1, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						b = _mm_loadu_ps(dst + j);
						b = _mm_add_ps(b, a);
						_mm_storeu_ps(dst + j, b);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] += src1[j];
			}

			void mul(float *dst, float *src1, float *src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						b = _mm_loadu_ps(src2 + j);
						b = _mm_mul_ps(a, b);
						_mm_storeu_ps(dst + j, b);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = src1[j] * src2[j];
			}

			void mul(float *dst, float *src1, float src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b;
					b = _mm_set_ps1(src2);
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						a = _mm_mul_ps(a, b);
						_mm_storeu_ps(dst + j, a);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = src1[j] * src2;
			}

			void mad(float *dst, float *src1, float alpha, float beta, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b, c;
					a = _mm_set_ps1(alpha);
					b = _mm_set_ps1(beta);
					for (; j < w - 3; j += 4)
					{
						c = _mm_loadu_ps(src1 + j);
						c = _mm_mul_ps(c, a);
						c = _mm_add_ps(c, b);
						_mm_storeu_ps(dst + j, c);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = alpha * src1[j] + beta;
			}

			void sqr_(float *dst, float *src1, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						a = _mm_mul_ps(a, a);
						_mm_storeu_ps(dst + j, a);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = src1[j] * src1[j];
			}

			void sqr_dif(float *dst, float *src1, float *src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 d;
					for (; j < w - 3; j += 4)
					{
						d = _mm_sub_ps(_mm_loadu_ps(src1 + j), _mm_loadu_ps(src2 + j));
						d = _mm_mul_ps(d, d);
						_mm_storeu_ps(dst + j, d);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = (src1[j] - src2[j])*(src1[j] - src2[j]);
			}

			void add_mul(float *dst, float *src1, float *src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b, c;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						b = _mm_loadu_ps(src2 + j);
						b = _mm_mul_ps(b, a);
						c = _mm_loadu_ps(dst + j);
						c = _mm_add_ps(c, b);
						_mm_storeu_ps(dst + j, c);
					}
				}
#endif
				for (; j < w; j++)
				{
					dst[j] += src1[j] * src2[j];
				}
			}

			void add_sqr(float *dst, float *src1, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, c;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						a = _mm_mul_ps(a, a);
						c = _mm_loadu_ps(dst + j);
						c = _mm_add_ps(c, a);
						_mm_storeu_ps(dst + j, c);
					}
				}
#endif
				for (; j < w; j++)
				{
					dst[j] += src1[j] * src1[j];
				}
			}

			void add_sqr_dif(float *dst, float *src1, float *src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, d;
					for (; j < w - 3; j += 4)
					{
						d = _mm_sub_ps(_mm_loadu_ps(src1 + j), _mm_loadu_ps(src2 + j));
						d = _mm_mul_ps(d, d);
						a = _mm_loadu_ps(dst + j);
						a = _mm_add_ps(a, d);
						_mm_storeu_ps(dst + j, a);
					}
				}
#endif
				for (; j < w; j++)
				{
					dst[j] += (src1[j] - src2[j])*(src1[j] - src2[j]);
				}
			}

			void sub_mul(float *dst, float *src1, float *src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b, c;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						b = _mm_loadu_ps(src2 + j);
						b = _mm_mul_ps(b, a);
						c = _mm_loadu_ps(dst + j);
						c = _mm_sub_ps(c, b);
						_mm_storeu_ps(dst + j, c);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] -= src1[j] * src2[j];
			}

			void sub_mad(float *dst, float *src1, float *src2, float c0, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b, c;
					__m128 cnst = _mm_set_ps1(c0);
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						b = _mm_loadu_ps(src2 + j);
						b = _mm_mul_ps(b, a);
						c = _mm_loadu_ps(dst + j);
						c = _mm_sub_ps(c, cnst);
						c = _mm_sub_ps(c, b);
						_mm_storeu_ps(dst + j, c);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] -= src1[j] * src2[j] + c0;
			}

			void det_2x2(float *dst, float *a00, float *a01, float *a10, float *a11, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b;
					for (; j < w - 3; j += 4)
					{
						a = _mm_mul_ps(_mm_loadu_ps(a00 + j), _mm_loadu_ps(a11 + j));
						b = _mm_mul_ps(_mm_loadu_ps(a01 + j), _mm_loadu_ps(a10 + j));
						a = _mm_sub_ps(a, b);
						_mm_storeu_ps(dst + j, a);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = a00[j] * a11[j] - a01[j] * a10[j];
			}

			void div_det_2x2(float *a00, float *a01, float *a11, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					const __m128 SIGN_MASK = _mm_set_ps1(getFloatSignBit());

					__m128 a, b, _a00, _a01, _a11;
					for (; j < w - 3; j += 4)
					{
						_a00 = _mm_loadu_ps(a00 + j);
						_a11 = _mm_loadu_ps(a11 + j);
						a = _mm_mul_ps(_a00, _a11);

						_a01 = _mm_loadu_ps(a01 + j);
						_a01 = _mm_xor_ps(_a01, SIGN_MASK);
						b = _mm_mul_ps(_a01, _a01);

						a = _mm_sub_ps(a, b);

						_a01 = _mm_div_ps(_a01, a);
						_a00 = _mm_div_ps(_a00, a);
						_a11 = _mm_div_ps(_a11, a);

						_mm_storeu_ps(a01 + j, _a01);
						_mm_storeu_ps(a00 + j, _a00);
						_mm_storeu_ps(a11 + j, _a11);
					}
				}
#endif
				for (; j < w; j++)
				{
					float det = a00[j] * a11[j] - a01[j] * a01[j];
					a00[j] /= det;
					a11[j] /= det;
					a01[j] /= -det;
				}
			}

			void div_1x(float *a1, float *b1, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 _a1, _b1;
					for (; j < w - 3; j += 4)
					{
						_b1 = _mm_loadu_ps(b1 + j);
						_a1 = _mm_loadu_ps(a1 + j);
						_mm_storeu_ps(a1 + j, _mm_div_ps(_a1, _b1));
					}
				}
#endif
				for (; j < w; j++)
				{
					a1[j] /= b1[j];
				}
			}

			void inv_self(float *src, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a;
					for (; j < w - 3; j += 4)
					{
						a = _mm_rcp_ps(_mm_loadu_ps(src + j));
						_mm_storeu_ps(src + j, a);
					}
				}
#endif
				for (; j < w; j++)
				{
					src[j] = 1.0f / src[j];
				}
			}

			void sqrt_(float *dst, float *src, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a;
					for (; j < w - 3; j += 4)
					{
						a = _mm_sqrt_ps(_mm_loadu_ps(src + j));
						_mm_storeu_ps(dst + j, a);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = sqrt(src[j]);
			}

			void min_(float *dst, float *src1, float *src2, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 a, b;
					for (; j < w - 3; j += 4)
					{
						a = _mm_loadu_ps(src1 + j);
						b = _mm_loadu_ps(src2 + j);
						b = _mm_min_ps(b, a);

						_mm_storeu_ps(dst + j, b);
					}
				}
#endif
				for (; j < w; j++)
					dst[j] = std::min(src1[j], src2[j]);
			}

			void rf_vert_row_pass(float *curRow, float *prevRow, float alphaVal, int w)
			{
				int j = 0;
#if CV_SSE
				if (CPU_SUPPORT_SSE1())
				{
					__m128 cur, prev, res;
					__m128 alpha = _mm_set_ps1(alphaVal);
					for (; j < w - 3; j += 4)
					{
						cur = _mm_loadu_ps(curRow + j);
						prev = _mm_loadu_ps(prevRow + j);

						res = _mm_mul_ps(alpha, _mm_sub_ps(prev, cur));
						res = _mm_add_ps(res, cur);
						_mm_storeu_ps(curRow + j, res);
					}
				}
#endif
				for (; j < w; j++)
					curRow[j] += alphaVal * (prevRow[j] - curRow[j]);
			}

		} //end of cv::ximgproc::intrinsics

	} //end of cv::ximgproc
} //end of cv
