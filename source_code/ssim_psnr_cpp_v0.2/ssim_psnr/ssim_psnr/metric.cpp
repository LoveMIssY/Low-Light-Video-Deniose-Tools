#include "metric.hpp"


/*
Metric父类的构造函数实现
*/
Metric::Metric(int h, int w, int c)
{
	height = h;
	width = w;
	channel = c;
}

/*
Metric析构函数
*/
Metric::~Metric()
{

}
