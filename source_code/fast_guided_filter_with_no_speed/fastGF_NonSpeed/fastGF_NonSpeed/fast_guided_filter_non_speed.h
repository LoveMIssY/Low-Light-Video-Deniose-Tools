#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>


class FastGuidedFilterImp;

class FastGuidedFilter
{
public:
	FastGuidedFilter(const cv::Mat &I, int r, double eps, int s, int sampleType);
	~FastGuidedFilter();
	cv::Mat filter(const cv::Mat &p, int sampleType , int depth = -1) const;

private:
	FastGuidedFilterImp  *imp;
};


cv::Mat fastGuidedFilterWithNonSpeed(const cv::Mat &I, const cv::Mat &p, int r, double eps, int s, int sampleType, int depth = -1);


#endif
#pragma once
