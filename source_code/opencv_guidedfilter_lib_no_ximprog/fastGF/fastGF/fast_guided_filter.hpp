#include "precomp.hpp"
#include "edgeaware_filters_common.hpp"
#include "edge_filter.hpp"
#include <vector>
#include <iostream>

#ifdef _MSC_VER
#   pragma warning(disable: 4512)
#endif

namespace cv
{
	namespace ximgproc
	{

		using std::vector;
		using namespace cv::ximgproc::intrinsics;

		template <typename T>
		struct SymArray2D
		{
			vector<T> vec;
			int sz;

			SymArray2D()
			{
				sz = 0;
			}

			void create(int sz_)
			{
				CV_DbgAssert(sz_ > 0);
				sz = sz_;
				vec.resize(total());
			}

			inline T& operator()(int i, int j)
			{
				CV_DbgAssert(i >= 0 && i < sz && j >= 0 && j < sz);
				if (i < j) std::swap(i, j);
				return vec[i*(i + 1) / 2 + j];
			}

			inline T& operator()(int i)
			{
				return vec[i];
			}

			int total() const
			{
				//ʵ���Ͼ��� guide-src Э�������ĸ�����1ͨ��������1��3ͨ������6(rr,rg,rb,gg,gb,bb)
				return sz * (sz + 1) / 2;
			}

			void release()
			{
				vec.clear();
				sz = 0;
			}
		};


		template <typename XMat>
		static void splitFirstNChannels(InputArrayOfArrays src, vector<XMat>& dst, int maxDstCn)
		{
			CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

			if ((src.isMat() || src.isUMat()) && src.channels() == maxDstCn)
			{
				split(src, dst);      //maxDstCn=3,�����3ͨ��ֱ�ӽ��в��
			}
			else
			{
				Size sz;
				int depth, totalCnNum;

				checkSameSizeAndDepth(src, sz, depth);  //���src��size��depth
				totalCnNum = std::min(maxDstCn, getTotalNumberOfChannels(src)); //���ؽ�С��ͨ������Ҫô��3��Ҫô��1

				dst.resize(totalCnNum);             //totalCnNum=1������3
				vector<int> fromTo(2 * totalCnNum); //2��������6��intֵ��ɵ�vector
				for (int i = 0; i < totalCnNum; i++)
				{
					fromTo[i * 2 + 0] = i;       //fromTo[0]=0���ڵ�ͨ�������
					fromTo[i * 2 + 1] = i;       //fromTo[1]=0

					dst[i].create(sz, CV_MAKE_TYPE(depth, 1));
				}

				mixChannels(src, dst, fromTo);
			}
		}

		class GuidedFilterImpl : public GuidedFilter
		{
		public:

			static Ptr<GuidedFilterImpl> create(InputArray guide, int radius, double eps, int s);

			void filter(InputArray src, OutputArray dst, int dDepth = -1) CV_OVERRIDE;

		protected:

			int radius;
			double eps;
			int h, w;          //ͼ���height��width,ָ�����²���֮��Ĵ�С
			int height, width; //ָ�����²���֮ǰ�Ĵ�С
			int s_ratio;       //��ʾ�²�����

			vector<Mat> guideCn;          //��ʾ�²���֮���guide
			vector<Mat> guideCnPrevious;  //��ʾ�²���֮ǰ��guide
			vector<Mat> guideCnMean;

			SymArray2D<Mat> covarsInv;

			int gCnNum;        //��ʾguide��ͨ����Ŀ��guide-channel-number

		protected:

			GuidedFilterImpl() {}

			void init(InputArray guide, int radius, double eps, int s);

			void computeCovGuide(SymArray2D<Mat>& covars);

			void computeCovGuideAndSrc(vector<Mat>& srcCn, vector<Mat>& srcCnMean, vector<vector<Mat> >& cov);

			void getWalkPattern(int eid, int &cn1, int &cn2);

			inline void meanFilter(Mat& src, Mat& dst)
			{
				boxFilter(src, dst, CV_32F, Size(2 * radius + 1, 2 * radius + 1), cv::Point(-1, -1), true, BORDER_REFLECT);
			}

			inline void convertToWorkType(Mat& src, Mat& dst)
			{
				src.convertTo(dst, CV_32F);
			}

			/* �����Զ���� �Ը���ͨ���������ŵĺ���*/
			void resizeMat(vector<Mat> &src, vector<Mat> &dst, int wid, int hei)
			{
				CV_Assert(src.size() == dst.size());
				for (int i = 0; i < src.size(); i++)
				{
					cv::resize(src[i], dst[i], cv::Size(wid, hei));
				}
			}

		private: /*Routines to parallelize boxFilter and convertTo*/

			typedef void (GuidedFilterImpl::*TransformFunc)(Mat& src, Mat& dst);

			struct GFTransform_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				mutable vector<Mat*> src;
				mutable vector<Mat*> dst;
				TransformFunc func;

				GFTransform_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcv, vector<Mat>& dstv, TransformFunc func_);
				GFTransform_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& srcvv, vector<vector<Mat> >& dstvv, TransformFunc func_);

				void operator () (const Range& range) const CV_OVERRIDE;

				Range getRange() const
				{
					return Range(0, (int)src.size());
				}
			};

			template<typename V>
			void parConvertToWorkType(V &src, V &dst)
			{
				GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::convertToWorkType);
				parallel_for_(pb.getRange(), pb);
			}

			template<typename V>
			void parMeanFilter(V &src, V &dst)
			{
				GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::meanFilter);
				parallel_for_(pb.getRange(), pb);  //���������Ƕ�ͨ��������3ͨ�����ʶ��Ƕ���ͨ�����в��д���ÿһ��ͨ�����о�ֵ�˲�
			}

		private: /*Parallel body classes*/

			/*���²���֮��Ľ��в��д���Range����Ĳ����� (0,h) h��ʾ����֮��� height*/
			inline void runParBody(const ParallelLoopBody& pb)
			{
				parallel_for_(Range(0, h), pb);
			}

			/*���²���֮ǰ�Ľ��в��д���Range����Ĳ����� (0,height) h��ʾ����֮��� height�������SSE�Ż�ʹ�õ���width*/
			inline void runParBody_(const ParallelLoopBody& pb)
			{
				parallel_for_(Range(0, height), pb);
			}

			struct MulChannelsGuide_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				SymArray2D<Mat> &covars;

				MulChannelsGuide_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_)
					: gf(gf_), covars(covars_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct ComputeCovGuideFromChannelsMul_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				SymArray2D<Mat> &covars;

				ComputeCovGuideFromChannelsMul_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_)
					: gf(gf_), covars(covars_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct ComputeCovGuideInv_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				SymArray2D<Mat> &covars;

				ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_);

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct MulChannelsGuideAndSrc_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				vector<vector<Mat> > &cov;
				vector<Mat> &srcCn;

				MulChannelsGuideAndSrc_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcCn_, vector<vector<Mat> >& cov_)
					: gf(gf_), cov(cov_), srcCn(srcCn_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct ComputeCovFromSrcChannelsMul_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				vector<vector<Mat> > &cov;
				vector<Mat> &srcCnMean;

				ComputeCovFromSrcChannelsMul_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcCnMean_, vector<vector<Mat> >& cov_)
					: gf(gf_), cov(cov_), srcCnMean(srcCnMean_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct ComputeAlpha_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				vector<vector<Mat> > &alpha;
				vector<vector<Mat> > &covSrc;

				ComputeAlpha_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& alpha_, vector<vector<Mat> >& covSrc_)
					: gf(gf_), alpha(alpha_), covSrc(covSrc_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct ComputeBeta_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				vector<vector<Mat> > &alpha;
				vector<Mat> &srcCnMean;
				vector<Mat> &beta;

				ComputeBeta_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& alpha_, vector<Mat>& srcCnMean_, vector<Mat>& beta_)
					: gf(gf_), alpha(alpha_), srcCnMean(srcCnMean_), beta(beta_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};

			struct ApplyTransform_ParBody : public ParallelLoopBody
			{
				GuidedFilterImpl &gf;
				vector<vector<Mat> > &alpha;
				vector<Mat> &beta;

				ApplyTransform_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& alpha_, vector<Mat>& beta_)
					: gf(gf_), alpha(alpha_), beta(beta_) {}

				void operator () (const Range& range) const CV_OVERRIDE;
			};
		};

		void GuidedFilterImpl::MulChannelsGuide_ParBody::operator()(const Range& range) const
		{
			int total = covars.total();

			for (int i = range.start; i < range.end; i++)
			{
				int c1, c2;
				float *cov, *guide1, *guide2;

				for (int k = 0; k < total; k++)
				{
					gf.getWalkPattern(k, c1, c2);

					guide1 = gf.guideCn[c1].ptr<float>(i);
					guide2 = gf.guideCn[c2].ptr<float>(i);
					cov = covars(c1, c2).ptr<float>(i);

					mul(cov, guide1, guide2, gf.w);
				}
			}
		}

		void GuidedFilterImpl::ComputeCovGuideFromChannelsMul_ParBody::operator()(const Range& range) const
		{
			int total = covars.total();
			float diagSummand = (float)(gf.eps);

			for (int i = range.start; i < range.end; i++)
			{
				int c1, c2;
				float *cov, *guide1, *guide2;

				for (int k = 0; k < total; k++)
				{
					gf.getWalkPattern(k, c1, c2);

					guide1 = gf.guideCnMean[c1].ptr<float>(i);
					guide2 = gf.guideCnMean[c2].ptr<float>(i);
					cov = covars(c1, c2).ptr<float>(i);

					if (c1 != c2)
					{
						sub_mul(cov, guide1, guide2, gf.w);
					}
					else
					{
						sub_mad(cov, guide1, guide2, -diagSummand, gf.w);
					}
				}
			}
		}

		GuidedFilterImpl::ComputeCovGuideInv_ParBody::ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_)
			: gf(gf_), covars(covars_)
		{
			gf.covarsInv.create(gf.gCnNum);

			if (gf.gCnNum == 3)
			{
				for (int k = 0; k < 2; k++)
					for (int l = 0; l < 3; l++)
						gf.covarsInv(k, l).create(gf.h, gf.w, CV_32FC1);

				////trick to avoid memory allocation
				gf.covarsInv(2, 0).create(gf.h, gf.w, CV_32FC1);
				gf.covarsInv(2, 1) = covars(2, 1);
				gf.covarsInv(2, 2) = covars(2, 2);

				return;
			}

			if (gf.gCnNum == 2)
			{
				gf.covarsInv(0, 0) = covars(1, 1);
				gf.covarsInv(0, 1) = covars(0, 1);
				gf.covarsInv(1, 1) = covars(0, 0);
				return;
			}

			if (gf.gCnNum == 1)
			{
				gf.covarsInv(0, 0) = covars(0, 0);
				return;
			}
		}

		void GuidedFilterImpl::ComputeCovGuideInv_ParBody::operator()(const Range& range) const
		{
			if (gf.gCnNum == 3)
			{
				vector<float> covarsDet(gf.w);
				float *det = &covarsDet[0];

				for (int i = range.start; i < range.end; i++)
				{
					for (int k = 0; k < 3; k++)
						for (int l = 0; l <= k; l++)
						{
							float *dst = gf.covarsInv(k, l).ptr<float>(i);

							float *a00 = covars((k + 1) % 3, (l + 1) % 3).ptr<float>(i);
							float *a01 = covars((k + 1) % 3, (l + 2) % 3).ptr<float>(i);
							float *a10 = covars((k + 2) % 3, (l + 1) % 3).ptr<float>(i);
							float *a11 = covars((k + 2) % 3, (l + 2) % 3).ptr<float>(i);

							det_2x2(dst, a00, a01, a10, a11, gf.w);
						}

					for (int k = 0; k < 3; k++)
					{
						float *a = covars(k, 0).ptr<float>(i);
						float *ac = gf.covarsInv(k, 0).ptr<float>(i);

						if (k == 0)
							mul(det, a, ac, gf.w);
						else
							add_mul(det, a, ac, gf.w);
					}

					if (gf.eps < 1e-2)
					{
						for (int j = 0; j < gf.w; j++)
							if (abs(det[j]) < 1e-6f)
								det[j] = 1.f;
					}

					for (int k = 0; k < gf.covarsInv.total(); k += 1)
					{
						div_1x(gf.covarsInv(k).ptr<float>(i), det, gf.w);
					}
				}
				return;
			}

			if (gf.gCnNum == 2)
			{
				for (int i = range.start; i < range.end; i++)
				{
					float *a00 = gf.covarsInv(0, 0).ptr<float>(i);
					float *a10 = gf.covarsInv(1, 0).ptr<float>(i);
					float *a11 = gf.covarsInv(1, 1).ptr<float>(i);

					div_det_2x2(a00, a10, a11, gf.w);
				}
				return;
			}

			if (gf.gCnNum == 1)
			{
				//divide(1.0, covars(0, 0)(range, Range::all()), gf.covarsInv(0, 0)(range, Range::all()));
				//return;

				for (int i = range.start; i < range.end; i++)
				{
					float *res = covars(0, 0).ptr<float>(i);
					inv_self(res, gf.w);
				}
				return;
			}
		}

		void GuidedFilterImpl::MulChannelsGuideAndSrc_ParBody::operator()(const Range& range) const
		{
			int srcCnNum = (int)srcCn.size();

			for (int i = range.start; i < range.end; i++)
			{
				for (int si = 0; si < srcCnNum; si++)
				{
					int step = (si % 2) * 2 - 1;
					int start = (si % 2) ? 0 : gf.gCnNum - 1;
					int end = (si % 2) ? gf.gCnNum : -1;

					float *srcLine = srcCn[si].ptr<float>(i);

					for (int gi = start; gi != end; gi += step)
					{
						float *guideLine = gf.guideCn[gi].ptr<float>(i);
						float *dstLine = cov[si][gi].ptr<float>(i);

						mul(dstLine, srcLine, guideLine, gf.w);
					}
				}
			}
		}

		void GuidedFilterImpl::ComputeCovFromSrcChannelsMul_ParBody::operator()(const Range& range) const
		{
			int srcCnNum = (int)srcCnMean.size();

			for (int i = range.start; i < range.end; i++)
			{
				for (int si = 0; si < srcCnNum; si++)
				{
					int step = (si % 2) * 2 - 1;
					int start = (si % 2) ? 0 : gf.gCnNum - 1;
					int end = (si % 2) ? gf.gCnNum : -1;

					float *srcMeanLine = srcCnMean[si].ptr<float>(i);

					for (int gi = start; gi != end; gi += step)
					{
						float *guideMeanLine = gf.guideCnMean[gi].ptr<float>(i);
						float *covLine = cov[si][gi].ptr<float>(i);

						sub_mul(covLine, srcMeanLine, guideMeanLine, gf.w);
					}
				}
			}
		}

		void GuidedFilterImpl::ComputeAlpha_ParBody::operator()(const Range& range) const
		{
			int srcCnNum = (int)covSrc.size();

			for (int i = range.start; i < range.end; i++)
			{
				for (int si = 0; si < srcCnNum; si++)
				{
					for (int gi = 0; gi < gf.gCnNum; gi++)
					{
						float *y, *A, *dstAlpha;

						dstAlpha = alpha[si][gi].ptr<float>(i);
						for (int k = 0; k < gf.gCnNum; k++)
						{
							y = covSrc[si][k].ptr<float>(i);
							A = gf.covarsInv(gi, k).ptr<float>(i);

							if (k == 0)
							{
								mul(dstAlpha, A, y, gf.w);
							}
							else
							{
								add_mul(dstAlpha, A, y, gf.w);
							}
						}
					}
				}
			}
		}

		void GuidedFilterImpl::ComputeBeta_ParBody::operator()(const Range& range) const
		{
			int srcCnNum = (int)srcCnMean.size();
			CV_DbgAssert(&srcCnMean == &beta);

			for (int i = range.start; i < range.end; i++)
			{
				float *_g[4];
				for (int gi = 0; gi < gf.gCnNum; gi++)
					_g[gi] = gf.guideCnMean[gi].ptr<float>(i);

				float *betaDst, *g, *a;
				for (int si = 0; si < srcCnNum; si++)
				{
					betaDst = beta[si].ptr<float>(i);
					for (int gi = 0; gi < gf.gCnNum; gi++)
					{
						a = alpha[si][gi].ptr<float>(i);
						g = _g[gi];

						sub_mul(betaDst, a, g, gf.w);
					}
				}
			}
		}

		void GuidedFilterImpl::ApplyTransform_ParBody::operator()(const Range& range) const
		{
			int srcCnNum = (int)alpha.size();

			for (int i = range.start; i < range.end; i++)
			{
				float *_g[4];
				for (int gi = 0; gi < gf.gCnNum; gi++)
					_g[gi] = gf.guideCnPrevious[gi].ptr<float>(i);  //����ͼ��guideÿһ��Ԫ�ص�ָ�룬��һ��float *,ע������Ҫʹ��ԭʼ��δ���ŵ�guide

				float *betaDst, *g, *a;
				for (int si = 0; si < srcCnNum; si++)    //ԭͼ�񣬼���Ҫ�����srcͼ��
				{
					betaDst = beta[si].ptr<float>(i);
					for (int gi = 0; gi < gf.gCnNum; gi++)
					{
						a = alpha[si][gi].ptr<float>(i); //float *
						g = _g[gi];                      //float *

						add_mul(betaDst, a, g, gf.width);//���������һ��2
					}
				}
			}
		}

		GuidedFilterImpl::GFTransform_ParBody::GFTransform_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcv, vector<Mat>& dstv, TransformFunc func_)
			: gf(gf_), func(func_)
		{
			CV_DbgAssert(srcv.size() == dstv.size());
			src.resize(srcv.size());
			dst.resize(srcv.size());

			for (int i = 0; i < (int)srcv.size(); i++)
			{
				src[i] = &srcv[i];
				dst[i] = &dstv[i];
			}
		}

		GuidedFilterImpl::GFTransform_ParBody::GFTransform_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& srcvv, vector<vector<Mat> >& dstvv, TransformFunc func_)
			: gf(gf_), func(func_)
		{
			CV_DbgAssert(srcvv.size() == dstvv.size());
			int n = (int)srcvv.size();
			int total = 0;

			for (int i = 0; i < n; i++)
			{
				CV_DbgAssert(srcvv[i].size() == dstvv[i].size());
				total += (int)srcvv[i].size();
			}

			src.resize(total);
			dst.resize(total);

			int k = 0;
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < (int)srcvv[i].size(); j++)
				{
					src[k] = &srcvv[i][j];
					dst[k] = &dstvv[i][j];
					k++;
				}
			}
		}

		void GuidedFilterImpl::GFTransform_ParBody::operator()(const Range& range) const
		{
			for (int i = range.start; i < range.end; i++)
			{
				(gf.*func)(*src[i], *dst[i]);
			}
		}

		void GuidedFilterImpl::getWalkPattern(int eid, int &cn1, int &cn2)
		{
			static int wdata[] = {
				0, -1, -1, -1, -1, -1,
				0, -1, -1, -1, -1, -1,

				0,  0,  1, -1, -1, -1,
				0,  1,  1, -1, -1, -1,

				0,  0,  0,  2,  1,  1,
				0,  1,  2,  2,  2,  1,
			};

			cn1 = wdata[6 * 2 * (gCnNum - 1) + eid];
			cn2 = wdata[6 * 2 * (gCnNum - 1) + 6 + eid];
		}

		Ptr<GuidedFilterImpl> GuidedFilterImpl::create(InputArray guide, int radius, double eps, int s)
		{
			GuidedFilterImpl *gf = new GuidedFilterImpl();
			gf->init(guide, radius, eps, s);
			return Ptr<GuidedFilterImpl>(gf);
		}

		void GuidedFilterImpl::init(InputArray guide, int radius_, double eps_, int s)
		{
			CV_Assert(!guide.empty() && radius_ >= 0 && eps_ >= 0);
			CV_Assert((guide.depth() == CV_32F || guide.depth() == CV_8U || guide.depth() == CV_16U) && (guide.channels() <= 3));

			radius = radius_ / s;  //�õ��²����İ뾶
			eps = eps_;            //eps
			s_ratio = s;           //�²�����

			splitFirstNChannels(guide, guideCnPrevious, 3);         //����֮ǰ��guideҲ��Ҫ�洢����Ϊ���ٵ�������һ����Ҫ			
			parConvertToWorkType(guideCnPrevious, guideCnPrevious); //��ԭʼguideת��Ϊ CV_32F
			height = guideCnPrevious[0].rows;      //ԭʼguide�Ŀ�͸�
			width = guideCnPrevious[0].cols;
			gCnNum = (int)guideCnPrevious.size();

			//ʹ�ò��еķ�����һ������������ͨ�����������²���
			vector<Mat> guide_(gCnNum);
			resizeMat(guideCnPrevious, guide_, width / s_ratio, height / s_ratio);

			//splitFirstNChannels(guide_, guideCn, 3);
			guideCn = guide_;              //�õ���guide���� ��С֮��� vector ,������Ҫ���
			gCnNum = (int)guideCn.size();  //gCnNum = 1 or 3
			h = guideCn[0].rows;           //360,��ʾ����֮��Ĵ�С
			w = guideCn[0].cols;           //640

			guideCnMean.resize(gCnNum);
			parConvertToWorkType(guideCn, guideCn);
			parMeanFilter(guideCn, guideCnMean);  //��ÿһ��ͨ�����о�ֵ�˲�,��Ȼ�Ƕ�ͨ������parallel����

			SymArray2D<Mat> covars;
			computeCovGuide(covars);             //guide-src֮���Э������ͨ��SymArray2D��ʵ�ֵ�
			runParBody(ComputeCovGuideInv_ParBody(*this, covars)); //��ʵ���Ƕ�ComputeCovGuideInv_ParBody��һ�����м���
			covars.release();
		}

		void GuidedFilterImpl::computeCovGuide(SymArray2D<Mat>& covars)
		{
			covars.create(gCnNum);  //gCnNum����guideͼ���ͨ����Ŀ��1������3
			for (int i = 0; i < covars.total(); i++)
				covars(i).create(h, w, CV_32FC1);

			runParBody(MulChannelsGuide_ParBody(*this, covars));

			parMeanFilter(covars.vec, covars.vec);

			runParBody(ComputeCovGuideFromChannelsMul_ParBody(*this, covars));
		}

		void GuidedFilterImpl::filter(InputArray src, OutputArray dst, int dDepth /*= -1*/)
		{
			CV_Assert(!src.empty() && (src.depth() == CV_32F || src.depth() == CV_8U));
			if (src.rows() != height || src.cols() != width)  //�ж���Ҫ�˲���ͼ���Ƿ���Ԫͼ���С���
			{
				CV_Error(Error::StsBadSize, "Size of filtering image must be equal to size of guide image");
				return;
			}

			if (dDepth == -1)
				dDepth = src.depth();     //Ϊ0������Ҫת��ΪCV_8U��Ĭ�ϵ�����,���ҪרΪCV_8U���
			int srcCnNum = src.channels();   //��ȡ����ͼ��ͨ���� src-channels-number��1 or 3

			//vector<Mat> srcCn(srcCnNum); ��Ϊ������ʹ�õ�push_back���������ﲻ�������ʼ��С
			vector<Mat> srcCn(srcCnNum);
			vector<Mat>& srcCnMean = srcCn;

			vector<Mat> src_;
			split(src, src_);       //��ԭʼͼ�����ͨ�����
			//��ԭʼͼ������²���
			resizeMat(src_, srcCn, width / s_ratio, height / s_ratio);

			parConvertToWorkType(srcCn, srcCn);

			vector<vector<Mat> > covSrcGuide(srcCnNum);
			computeCovGuideAndSrc(srcCn, srcCnMean, covSrcGuide);

			vector<vector<Mat> > alpha(srcCnNum);  //srcCnNum = 1  or  3
			for (int si = 0; si < srcCnNum; si++)
			{
				alpha[si].resize(gCnNum);
				for (int gi = 0; gi < gCnNum; gi++)
					alpha[si][gi].create(h, w, CV_32FC1);
			}
			runParBody(ComputeAlpha_ParBody(*this, alpha, covSrcGuide));
			covSrcGuide.clear();

			vector<Mat>& beta = srcCnMean;
			runParBody(ComputeBeta_ParBody(*this, alpha, srcCnMean, beta));

			parMeanFilter(beta, beta);
			parMeanFilter(alpha, alpha);

			/*�ڴ˴�����a,bҲ����alpha��beta���ϲ���*/
			vector<vector<Mat>> alpha_up;
			for (int i = 0; i < alpha.capacity(); i++)
			{
				vector<Mat> alpha_(alpha[i].size());
				resizeMat(alpha[i], alpha_, width, height);
				alpha_up.push_back(alpha_);
			}

			/*��beta�����ϲ���*/
			vector<Mat> beta_up(beta.size());
			resizeMat(beta, beta_up, width, height);

			runParBody_(ApplyTransform_ParBody(*this, alpha_up, beta_up));


			if (dDepth != CV_32F)
			{
				for (int i = 0; i < srcCnNum; i++)
					beta_up[i].convertTo(beta_up[i], dDepth);
			}

			merge(beta_up, dst);  //ʵ���Ͼ��� beta=a.mul(I)+beta
		}

		void GuidedFilterImpl::computeCovGuideAndSrc(vector<Mat>& srcCn, vector<Mat>& srcCnMean, vector<vector<Mat> >& cov)
		{
			int srcCnNum = (int)srcCn.size();

			cov.resize(srcCnNum);
			for (int i = 0; i < srcCnNum; i++)
			{
				cov[i].resize(gCnNum);
				for (int j = 0; j < gCnNum; j++)
					cov[i][j].create(h, w, CV_32FC1);
			}

			runParBody(MulChannelsGuideAndSrc_ParBody(*this, srcCn, cov));

			parMeanFilter(srcCn, srcCnMean);
			parMeanFilter(cov, cov);

			runParBody(ComputeCovFromSrcChannelsMul_ParBody(*this, srcCnMean, cov));
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		CV_EXPORTS_W
		Ptr<GuidedFilter> createGuidedFilter(InputArray guide, int radius, double eps, int s)
		{
			return Ptr<GuidedFilter>(GuidedFilterImpl::create(guide, radius, eps, s));
		}

		CV_EXPORTS_W
		void fastGuidedFilter(InputArray guide, InputArray src, OutputArray dst, int radius, double eps, int s, int dDepth = -1)
		{
			CV_Assert(s > 0);  //�²�����ֻ����1,2,3,4,5��... ...
			Ptr<GuidedFilter> gf = createGuidedFilter(guide, radius, eps, s);
			gf->filter(src, dst, dDepth);
		}

	}
}

#pragma once


