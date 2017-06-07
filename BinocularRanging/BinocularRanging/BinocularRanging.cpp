#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cxmisc.h>
#include <highgui.h>
#include <cvaux.h>
#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
/*
  hawk机器人的双目测距程序
*/

using namespace  std;
using namespace  cv;

bool left_mouse = false;
Point2f point;
int pic_info[4];
Mat gray, prevGray, image, image1;
const Scalar  GREEN = Scalar(0, 255, 0);
int rect_width = 0, rect_height = 0;
Point tmpPoint = 0;

//载入图像的具体信息，需要我们根据具体情况进行改正
int num = 0;
int m_frameWidth = 1197;
int m_frameHeight = 670;					
//=================================================

bool m_Calib_Data_Loaded;  //是否	成功载入定标数据
Mat  m_Calib_Mat_Q;   //Q矩阵，重投影矩阵
Mat  m_Calib_Mat_Remap_X_L; //左视图畸变矫正像素坐标映射矩阵X
Mat  m_Calib_Mat_Remap_Y_L; //左视图畸变校正像素坐标映射矩阵Y
Mat  m_Calib_Mat_Remap_X_R; //右视图畸变校正像素坐标映射矩阵X
Mat  m_Calib_Mat_Remap_Y_R; //右视图畸变校正像素坐标映射矩阵Y
Mat  m_Calib_Mat_Mask_Roi;   //左视图校正后的有效区域
Rect m_Calib_Roi_L;          //左视图后的有效区域矩形
Rect m_Calib_Roi_R;          //右视图后的有效区域矩形

double          m_FL;

int m_numberOfDisparies;            // 视差变化范围
cv::StereoBM    m_BM;
CvMat* vdisp = cvCreateMat(1197, 670, CV_8U);     //?????????????????????????????????????????????????????
cv::Mat img1, img2, img1p, img2p, disp, disp8u, pointClouds, imageLeft, imageRight, disparityImage, imaget1;
static IplImage *framet1 = NULL;
static IplImage *framet2 = NULL;
static IplImage *framet3 = NULL;
static IplImage *framet = NULL;



//用于控制鼠标在视察图上实时的情况
static void onMouse(int event, int x, int y, int , void* )
{


	Mat mouse_show;
	image.copyTo(mouse_show);


	if (event == CV_EVENT_LBUTTONDOWN)
	{
		pic_info[0] = x;
		pic_info[1] = y;
		cout << "x:" << pic_info[0] << "y:" << pic_info[1] << endl;
		left_mouse = true;
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		left_mouse = false;
	}
	else if ((event == CV_EVENT_MOUSEMOVE) && (left_mouse == true))
	{
	}
}


//BM相关参数的初始化
void updatebm()
{
	m_BM.state->preFilterCap = 31;
	m_BM.state->SADWindowSize = 19;
	m_BM.state->minDisparity = 0;
	m_BM.state->numberOfDisparities = 96;
	m_BM.state->textureThreshold = 10;
	m_BM.state->uniquenessRatio = 25;
	m_BM.state->speckleWindowSize = 100;
	m_BM.state->speckleRange = 32;
	m_BM.state->disp12MaxDiff = -1;

	/*m_BM.state->SADWindowSize = 17;
	m_BM.state->numberOfDisparities = 112;
	m_BM.state->preFilterSize = 5;
	m_BM.state->preFilterCap = 1;
	m_BM.state->minDisparity = 0;
	m_BM.state->textureThreshold = 7;
	m_BM.state->uniquenessRatio = 5;
	m_BM.state->speckleWindowSize = 0;
	m_BM.state->speckleRange = 20;
	m_BM.state->disp12MaxDiff = 64;*/


}


int loadCalibData()
{
	// 读入摄像头定标参数 Q roi1 roi2 mapx1 mapy1 mapx2 mapy2
	try
	{
		cv::FileStorage fs("intrinsics.yml", cv::FileStorage::READ);
		cout << fs.isOpened() << endl;

		if (!fs.isOpened())
		{
			return 0;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;


		fs.open("extrinsics.yml", FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n");
			return 0;
		}
		Rect roiVal1;
		Rect roiVal2;
		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;
        stereoRectify( M1, D1, M2, D2, Size(1197,670), R, T, R1, R2, P1, P2, m_Calib_Mat_Q, CALIB_ZERO_DISPARITY, -1, Size(1197, 670), &m_Calib_Roi_L, &m_Calib_Roi_R);

	
		initUndistortRectifyMap(M1, D1, R1, P1, Size(1197, 670), CV_16SC2, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L);
		initUndistortRectifyMap(M2, D2, R2, P2, Size(1197, 670), CV_16SC2, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R);



		/*
		
		还有好多参数的初始化没加上
		
		*/


    	m_BM.state->roi1 = m_Calib_Roi_L;
		m_BM.state->roi2 = m_Calib_Roi_R;

		m_Calib_Data_Loaded = true;



	}
	catch (std::exception& e)
	{
		m_Calib_Data_Loaded = false;
		return (-99);
	}

	return 1;
}


//进行BM匹配获取图像的时差图
int  bmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight)
{

	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}
	if (m_frameWidth == 0 || m_frameHeight == 0)
	{
		return 0;
	
	}

	// 转换为灰度图
	cv::Mat img1proc, img2proc;
	cvtColor(frameLeft, img1proc, CV_BGR2GRAY);
	cvtColor(frameRight, img2proc, CV_BGR2GRAY);

	// 校正图像，使左右视图行对齐    
	cv::Mat img1remap, img2remap;


	if (m_Calib_Data_Loaded)
	{
		remap(img1proc, img1remap, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);     // 对用于视差计算的画面进行校正
		remap(img2proc, img2remap, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
	}
	else
	{
		img1remap = img1proc;
		img2remap = img2proc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat img1border, img2border;
	if (m_numberOfDisparies != m_BM.state->numberOfDisparities)
		m_numberOfDisparies = m_BM.state->numberOfDisparities;
	copyMakeBorder(img1remap, img1border, 0, 0, m_BM.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2remap, img2border, 0, 0, m_BM.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	// 计算视差
	cv::Mat dispBorder;


	m_BM(img1border, img2border, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(m_BM.state->numberOfDisparities, img1border.cols);
	disp.copyTo(disparity, m_Calib_Mat_Mask_Roi);


	// 输出处理后的图像

	if (m_Calib_Data_Loaded)
	{
		remap(frameLeft, imageLeft, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);
		rectangle(imageLeft, m_Calib_Roi_L, CV_RGB(0, 0, 255), 3);
	}

	else
		frameLeft.copyTo(imageLeft);


	if (m_Calib_Data_Loaded)
		remap(frameRight, imageRight, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
	else
		frameRight.copyTo(imageRight);
	rectangle(imageRight, m_Calib_Roi_R, CV_RGB(0, 0, 255), 3);


	return 1;
}



//得到标准的时差图的格式
int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
	// 将原始视差数据的位深转换为 8 位
	cv::Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
		if (disparity.depth() == CV_8S)
		{
			disparity.convertTo(disp8u, CV_8U);
		}
		else
		{
			disparity.convertTo(disp8u, CV_8U, 255 / (m_numberOfDisparies*16.));
		}
	}
	else
	{
		disp8u = disparity;
	}

	// 转换为伪彩色图像 或 灰度图像
	if (isColor)
	{
		if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != disparity.size())
		{
			disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
		}

		for (int y = 0; y<disparity.rows; y++)
		{
			for (int x = 0; x<disparity.cols; x++)
			{
				uchar val = disp8u.at<uchar>(y, x);
				uchar r, g, b;

				if (val == 0)
					r = g = b = 0;
				else
				{
					r = 255 - val;
					g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
					b = val;
				}

				disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);

			}
		}
	}
	else
	{
		disp8u.copyTo(disparityImage);
	}

	return 1;
}

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%d,%d,%f %f %f\n",x,y,point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}
//得到三维点云
int getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds)
{
	if (disparity.empty())
	{
		return 0;
	}

	//计算生成三维点云

	reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);

	pointClouds *= 1.6;
	cout << pointClouds.rows << endl;
	cout << pointClouds.cols << endl;


	for (int y = 0; y < pointClouds.rows; ++y)
	{
		for (int x = 0; x < pointClouds.cols; ++x)
		{
			cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);
			point.y = -point.y;
			pointClouds.at<cv::Point3f>(y, x) = point;
		
		}
	}
	saveXYZ("point_cloud.txt", pointClouds);
	return 1;
}


//获取深度信息
void detectDistance(cv::Mat& pointCloud)
{
	if (pointCloud.empty())
	{
		return;
	}

	// 提取深度图像
	vector<cv::Mat> xyzSet;
	split(pointCloud, xyzSet);
	cv::Mat depth;
	xyzSet[2].copyTo(depth);

	cout << depth << endl;
	// 根据深度阈值进行二值化处理
	/*double maxVal = 0, minVal = 0;
	cv::Mat depthThresh = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
	cv::minMaxLoc(depth, &minVal, &maxVal);
	double thrVal = minVal * 1.5;
	threshold(depth, depthThresh, thrVal, 255, CV_THRESH_BINARY_INV);
	depthThresh.convertTo(depthThresh, CV_8UC1);

	double  distance = depth.at<float>(pic_info[0], pic_info[1]);
	cout << "distance:" << distance << endl;*/
}







int  main()       //效果特别差
{


	//读取摄像头
	/*VideoCapture cap(0);
	VideoCapture cap1(1);

	if (!cap.isOpened())
	{
		cout << "error happened while open cam 1" << endl;
		return -1;
	}
	if (!cap1.isOpened())
	{
		cout << "error happened while open cam 2" << endl;
		return -1;
	}
	*/
	namedWindow("left", 1);
	namedWindow("right", 1);
	namedWindow("disparitycolor", 1);
	//为获取的视差图添加响应事件
	setMouseCallback("disparitycolor", onMouse, 0);

	//载入相关数据
	loadCalibData();
	cout<<"数据是否载入："<< m_Calib_Data_Loaded << endl;

    //while (true)
	//{

		Mat frame = imread("left.jpg", CV_LOAD_IMAGE_UNCHANGED);;
		Mat frame1 = imread("right.jpg", CV_LOAD_IMAGE_UNCHANGED);;
		//cap.read(frame);
		//cap1.read(frame1);
	    //if (frame.empty())          break;
		//if (frame1.empty())         break;

		frame.copyTo(image);
		frame1.copyTo(image1);
		
		updatebm();
		bmMatch(frame, frame1, disp, imageLeft, imageRight);
	
	


		imshow("left", imageLeft);

		imshow("right", imageRight);
		getDisparityImage(disp, disparityImage, true);


		getPointClouds(disp, pointClouds);
	
		imshow("disparitycolor", disparityImage);
			waitKey(0);

	    detectDistance(pointClouds);
		
		

	//}


	return 0;
}






















/*int getMin(double value[], int valueSize)
{
	int pos = 0;
	int i = 0;
	double min1 = 999999;


	for (i = 0; i<valueSize; i++) {

		if (value[i]<min1)
		{
			pos = i;
			min1 = value[i];
		}
	}

	return pos;
}

//SAD
IplImage* sad(IplImage* greyLeftImg32, IplImage* greyRightImg32, int DSR)
{
	int height = greyLeftImg32->height;
	int width = greyLeftImg32->width;
	double* localSAD = new double[DSR];

	int x = 0, y = 0, d = 0, m = 0;


	IplImage* disparity = cvCreateImage(cvSize(width, height), 8, 1);



	for (y = 0; y<height; y++)
	{

		for (x = 0; x<width - DSR; x++)
		{

			for (m = 0; m<DSR; m++)
			{
				localSAD[m] = 0;
			}

			double m1 = cvGet2D(greyLeftImg32, y, x).val[0];
			double m2;
			for (m = 0; m<DSR; m++) {
				if (x + m <= width) {

					m2 = cvGet2D(greyRightImg32, y, x + 1 + m).val[0];
				}

				else {
					break;

				}
				localSAD[m] = abs(m1 - m2);

			}

			double  result = getMin(localSAD, DSR) * 255 / DSR;

			cvSetReal2D(disparity, y, x, result);

		}

	}


	return disparity;
}

//opencv BM
Mat ocv_bm(Mat& left, Mat& right) {

	Mat disp, disp8;

	cv::StereoBM sbm;

	sbm.state->SADWindowSize = 17;
	sbm.state->numberOfDisparities = 112;
	sbm.state->preFilterSize = 5;
	sbm.state->preFilterCap = 1;
	sbm.state->minDisparity = 0;
	sbm.state->textureThreshold = 7;
	sbm.state->uniquenessRatio = 5;
	sbm.state->speckleWindowSize = 0;
	sbm.state->speckleRange = 20;
	sbm.state->disp12MaxDiff = 64;
	sbm(left, right, disp);
	normalize(disp, disp8, 0.1, 255, CV_MINMAX, CV_8U);

	return disp8;
}


//opencv SGBM
Mat ocv_sgbm(Mat left, Mat right) {

	Mat disp, disp8;
	cv::StereoSGBM sgbm;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = 9;
	sgbm.P1 = sgbm.SADWindowSize*sgbm.SADWindowSize * 4;
	sgbm.P2 = sgbm.SADWindowSize*sgbm.SADWindowSize * 32;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 64;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = 1;

	sgbm(left, right, disp);
	disp.convertTo(disp8, CV_8U, 255 / (sgbm.numberOfDisparities*16.));

	return disp8;
}


int main()
{
	char* leftname = "left.jpg";
	char* rightname = "right.jpg";

	IplImage* leftIpl = cvLoadImage(leftname, 0);
	IplImage* rightIpl = cvLoadImage(rightname, 0);
	Mat leftMat = imread(leftname, 0);
	Mat rightMat = imread(rightname, 0);
	imshow("sws", leftMat);
	imshow("dwqdw", rightMat);


//	IplImage* disp_sad = sad(leftIpl, rightIpl, 64);
	Mat disp_bm = ocv_bm(leftMat, rightMat);
	Mat disp_sgbm = ocv_sgbm(leftMat, rightMat);

	imshow("left", leftMat);
//	cvShowImage("SAD", disp_sad);
	imshow("BM", disp_bm);
	imshow("SGBM", disp_sgbm);


	//cvSaveImage("SAD.bmp", disp_sad);
	imwrite("BM.bmp", disp_bm);
	imwrite("SGBM.bmp", disp_sgbm);
	cvWaitKey(0);

	return 0;
}*/