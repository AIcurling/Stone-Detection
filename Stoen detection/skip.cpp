#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <string.h> 
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#define meter 1000.0

#define d 270.0
#define d_ 200.0
#define h 140.0

#define W 2300.0
#define D_hog 8100.0
#define D_hog_ 29700.0
#define R_cir 1800.0

#define maxspeed 100.0
#define errorrange 10000.0

#define MAXNUM_STONE 16

#define NEAR 0

cv::Point2f getwldbyimg(cv::Point2f imgpoint, double z = 0);
cv::Point2f getimgbywld(cv::Point2f wldpoint, double z = 0);
cv::Point getdisplaybywld(cv::Point2f wld);

cv::Rect getEllipse(cv::Point2f wld, double diameter);
cv::Point2f getUnitvector(cv::Point2f wld, double tan);
cv::Scalar getColor(cv::Point2f wld, std::vector<cv::Mat> hsv);

cv::Point2f getintersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2);

cv::Mat Int;
cv::Mat Dist;
cv::Mat Rmat;
cv::Mat Rvec;
cv::Mat Tvec;
cv::Point2f cam_pos;
cv::Point2f cam_dir;

cv::Mat Epi[2];

cv::Scalar color0;
cv::Scalar color1;

namespace patch {
	template < typename T > std::string to_string (const T& n ) {
		std::ostringstream stm;
		stm << n;
		return stm.str() ;
	}
}

int main(){

	std::string videoname = "/home/nvidia/Curling/Skip/video/1.avi";
	
	struct timeval mytime;
	int msec, msec0, msec00;

	int i, j, k, ii, x, y, xx, yy;
	double xd, yd, t1, t2, sum;
	cv::Mat vec3 = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat mat3 = cv::Mat::zeros(3, 3, CV_64FC1);

	int num_stone;

	int framenum;
	int stonenum;
	int shotnum;

	int duration = 1000;

	cv::Mat img0;

	cv::Size imgsize;

	cv::Mat ROI;

	std::vector<cv::Mat> bgr(3);
	std::vector<cv::Mat> hsv(3);
	cv::Mat img, gray, img_hsv, prv_img;
	cv::Mat img_draw;
	cv::Mat edge, edgex, edgey;
	cv::Mat bnw;
	cv::Mat b;

	cv::Mat rec3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat rec5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	cv::Mat box_cor, box_cor_, result;
	int mx, my;

	int amin, amax;
	double minVal = 0, maxVal = 0;
	cv::Point minLoc, maxLoc;

	cv::Point pt;
	cv::Point2f pt1, pt2, pt3, pt4;
	cv::Point2f wp;

	double lx, ly, ld;
	double ldr = 1.0*meter, lr = cos((CV_PI / 180) * 15);
	double resize_ROI = 0.1;

	std::vector<cv::Vec4i> lines, lines_l, lines_r, lines_h;
	cv::Vec4i sidelines[2];
	cv::Vec4i hogline;
	
	cv::Point2f van, van_;
	cv::Point2f vertex[2];
	cv::Point2f e1, e2, e3, e4;

	double A1, A2, B1, B2, C1, C2;
	cv::Mat Q;

	std::vector<cv::Point2f> imgpoint(12);
	std::vector<cv::Point3f> wldpoint(12);
	double weight[12];

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> hull;
	std::vector<cv::Point2f> wldcontour;
	std::vector<cv::Point2f> quad(4), amin_quad(4), tri(3);

	std::vector<int> cand, prv_cand;
	std::vector<cv::Point2f> wldcentroid, prv_wldcentroid;
	std::vector<cv::Point2f> centroid, prv_centroid;

	cv::Mat Mask, Mgt, Ort;
	cv::Mat Vote, Vote_prior;

	cv::Point bias_Mask, bias_Vote;

	int maxnum_Mask;
	cv::Rect wldRect;

	double thetax, thetay, thetaz, cx, cy, cz;

	int r_max, r_min, unit, num_c, num_c_, p;
	double cos_max, theta_max;
	std::vector<cv::Point> circle, circle_;

	cv::Mat points;
	cv::Mat point2 = cv::Mat::zeros(1, 2, CV_32FC1);
	cv::Mat point5 = cv::Mat::zeros(1, 5, CV_32FC1);

	int num_region;
	int bluenum;
	int ellipse;
	cv::Mat region, props;

	cv::Mat Wld2Img = cv::Mat(3, 3, CV_32FC1);
	cv::Mat A = cv::Mat::zeros(2 * 12, 8, CV_32FC1);
	cv::Mat B = cv::Mat::zeros(2 * 12, 1, CV_32FC1);
	cv::Mat Img2Wld, X;
	double error;
	cv::Point2f err;

	int pad, pad_Vote;
	double pad_hog;

	std::vector<cv::Rect> coarsebox;
	std::vector<cv::Point2f> wldcenter;
	cv::Mat V_m;

	cv::Scalar mean, stddev;

	cv::Mat sortIdx1, sortIdx2;
	cv::Mat minVals = cv::Mat::zeros(cand.size(), 1, CV_32FC1);
	cv::Mat minIdxs = cv::Mat::zeros(cand.size(), 1, CV_32FC1);

	std::vector<cv::Scalar> color(MAXNUM_STONE);
	std::vector<std::vector<cv::Point2f> > stone(MAXNUM_STONE, std::vector<cv::Point2f>(duration));
	std::vector<std::vector<cv::Point2f> > wldstone(MAXNUM_STONE, std::vector<cv::Point2f>(duration));
	std::vector<std::vector<int> > Identified(MAXNUM_STONE, std::vector<int>(duration));

	bool IsCalibration;
	bool IsDisplaying, IsRecording, IsLogging;
	int IsValid, IsError, IsWarning;
	int thr_valid;
	int latency;

	/***************************************** Pre-Calculation *********************************************/
	shotnum = 16;

	unit = 10;
	
	theta_max = 45;
	p = 1 * unit;
	if ((int)d%unit || (int)d_%unit){
		printf("error : invalid coordinate unit\n");
		IsError = 1;
		return IsError;
	}

	cos_max = cos((CV_PI / 180)*(90 - theta_max));
	r_max = 0.5*d_ + p;
	r_min = 0.5*d_ - p;
	for (x = -r_max; x < r_max + 1; x += unit){
		for (y = -r_max; y < r_max + 1; y += unit){
			if (x*x + y*y > r_min*r_min && x*x + y*y < r_max*r_max)
				circle.push_back(cv::Point(x, y) * (1.0 / unit));
		}
	}
	num_c = circle.size();

	std::vector<double> circlenorm(num_c);
	for (i = 0; i < num_c; i++)
		circlenorm[i] = sqrt(circle[i].x*circle[i].x + circle[i].y*circle[i].y);

	for (x = -d; x < d + 1; x += unit){
		for (y = -d; y < d + 1; y += unit){
			if (x*x + y*y < d*d)
				circle_.push_back(cv::Point(x, y) * (1.0 / unit));
		}
	}
	num_c_ = circle_.size();

	pad_Vote = 2 * d;
	pad_Vote += 20;

	mx = 10;
	my = 10;

	for (j = 0; j < 3; j++){
		wldpoint[3 * 0 + j] = cv::Point3f(R_cir*(j - 1), 0, 0);
		wldpoint[3 * 1 + j] = cv::Point3f(R_cir*(j - 1), R_cir, 0);
		wldpoint[3 * 2 + j] = cv::Point3f(R_cir*(j - 1), 2 * R_cir, 0);
		wldpoint[3 * 3 + j] = cv::Point3f(R_cir*(j - 1), D_hog, 0);
		weight[3 * 0 + j] = 1;
		weight[3 * 1 + j] = 1;
		weight[3 * 2 + j] = 1;
		weight[3 * 3 + j] = 1;
	}

	sum = 0;
	for (i = 0; i < 12; i++)
		sum += weight[i];
	for (i = 0; i < 12; i++)
		weight[i] = weight[i] / sum;
	
	pad_hog = 3.0*meter;

	/********************************************************************************************************/

	/********************************************** Video Input *********************************************/
	int video_timeunit = 1;
	cv::VideoCapture video;
	video.open(videoname);
	if (video.isOpened() == false){
		printf("video not found\n");
		return 0;
	}
	imgsize = cv::Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT));
	/*******************************************************************************************************/

	/***************************************** Pre-Calibration *********************************************/
	Int = (cv::Mat_<double>(3, 3) << 752.3896, 0, 586.0002, 0, 565.0002, 355.7535, 0, 0, 1);
	Dist = (cv::Mat_<double>(5, 1) << -0.0437, 0.0835, 0, 0, -0.0040);
	
	thetax = -13.7;
	thetay = 3.109072;
	thetaz = -4.5;
	cx = -179.475849;
	cy = -14.516301;
	cz = 83.38797;

	/*thetax = -12.1889;
	thetay = -0.31558;
	thetaz = -2.54186;
	cx = -162.8962;
	cy = 62.7068;
	cz = -97.001;*/

	vec3.at<double>(0, 0) = thetax*CV_PI / 180;
	vec3.at<double>(1, 0) = thetay*CV_PI / 180;
	vec3.at<double>(2, 0) = thetaz*CV_PI / 180;
	cv::Rodrigues(vec3, Epi[0]);
	vec3.at<double>(0, 0) = cx;
	vec3.at<double>(1, 0) = cy;
	vec3.at<double>(2, 0) = cz;
	Epi[1] = -Epi[0] * vec3;

	cv::Mat mapx, mapy;
	cv::initUndistortRectifyMap(Int, Dist, cv::Mat(), Int, imgsize, CV_32FC1, mapx, mapy);
	Dist = cv::Mat::zeros(5, 1, CV_64FC1);

	/********************************************************************************************************/

	/************************************** Rough Calibration******************************************/
	thetax = 110;
	thetay = 0;
	thetaz = 0;
	cx = 100;
	cy = -1800;
	cz = 2000;

	Rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	Rvec.at<double>(0, 0) = thetax*CV_PI / 180;
	Rvec.at<double>(1, 0) = thetay*CV_PI / 180;
	Rvec.at<double>(2, 0) = thetaz*CV_PI / 180;
	cv::Rodrigues(Rvec, Rmat);

	vec3.at<double>(0, 0) = cx;
	vec3.at<double>(1, 0) = cy;
	vec3.at<double>(2, 0) = cz;
	Tvec = -Rmat * vec3;

	vec3 = -Rmat.inv() * Tvec;
	cam_pos = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
	mat3 = Rmat.inv();
	cam_dir = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));
	/**************************************************************************************************/

	/************************************************* ROI ***************************************/
	cv::Mat ROI_horizon = cv::Mat::ones(imgsize, CV_8UC1);
	ROI_horizon(cv::Rect(0, 0, imgsize.width, 200)) = cv::Mat::zeros(200, imgsize.width, CV_8UC1);
	/*****************************************************************************************************/

	/**************************************** Stone Color *********************************************/
	color1 = cv::Scalar(0, 1, 1);
	color0 = cv::Scalar(0, 0, 1);
	/**************************************************************************************************/

	/******************************************** Sheet *************************************************/
	int display_width = 1 + 2 * 85;
	double displayscale = display_width / 4800.0;
	cv::Point display_origin = cv::Point(displayscale*R_cir, display_width / 2);
	cv::Point display_pad = displayscale*cv::Point(2 * meter, 0);

	display_origin = display_origin + display_pad;
	cv::Mat Sheet = cv::Mat(display_width, 1280, CV_32FC3, cv::Scalar(1, 1, 1));
	cv::circle(Sheet, display_pad + cv::Point(displayscale * 2 * R_cir, display_width / 2), displayscale * R_cir, cv::Scalar(1, 0, 0), -1);
	cv::circle(Sheet, display_pad + cv::Point(displayscale * 2 * R_cir, display_width / 2), displayscale*1.2*meter, cv::Scalar(1, 1, 1), -1);
	cv::circle(Sheet, display_pad + cv::Point(displayscale * 2 * R_cir, display_width / 2), displayscale*0.6*meter, cv::Scalar(0, 0, 1), -1);
	cv::circle(Sheet, display_pad + cv::Point(displayscale * 2 * R_cir, display_width / 2), displayscale*0.2*meter, cv::Scalar(1, 1, 1), -1);
	cv::GaussianBlur(Sheet, Sheet, cv::Size(3, 3), 1);
	cv::line(Sheet, cv::Point(0, 0), display_pad + cv::Point(Sheet.cols - 1, 0), cv::Scalar(0, 0, 0), 3);
	cv::line(Sheet, cv::Point(0, Sheet.rows - 1), display_pad + cv::Point(Sheet.cols - 1, Sheet.rows - 1), cv::Scalar(0, 0, 0), 3);
	cv::line(Sheet, display_pad + cv::Point(displayscale * (R_cir + D_hog_), 0), display_pad + cv::Point(displayscale * (R_cir + D_hog_), Sheet.rows - 1), cv::Scalar(0, 0, 0), 2);
	cv::line(Sheet, display_pad + cv::Point(displayscale * (R_cir + D_hog), 0), display_pad + cv::Point(displayscale * (R_cir + D_hog), Sheet.rows - 1), cv::Scalar(0, 0, 0), 2);
	cv::line(Sheet, display_pad + cv::Point(displayscale * R_cir, 0), display_pad + cv::Point(displayscale * R_cir, Sheet.rows - 1), cv::Scalar(0, 0, 0), 2);

	cv::Mat Sheet_draw = Sheet.clone();
	cv::Point2f displaypoint;

	/*cv::circle(Sheet, display_origin, 5, cv::Scalar(0, 0, 0), -1);
	cv::imshow("Sheet", Sheet);
	cv::waitKey();*/
	/**************************************************************************************************/

	/******************************************** Output ************************************************/
	IsDisplaying = true;
	IsRecording = false;
	IsLogging = true;

	cv::VideoWriter output_scene, output_sheet;
	if (IsRecording){
		output_scene.open("scene.avi", CV_FOURCC('I', '4', '2', '0'), 30, cv::Size(1280, 720), true);
		output_sheet.open("sheet.avi", CV_FOURCC('I', '4', '2', '0'), 30, cv::Size(Sheet.cols, Sheet.rows), true);
	}

	//cv::Mat Frame = cv::Mat::zeros(720 + Sheet.rows, 1280, CV_8UC3);
	cv::Mat Frame = cv::Mat::zeros(360 + Sheet.rows * 0.5, 640, CV_8UC3);	
	std::ofstream log("log.txt");
	/***************************************************************************************************/

	img_draw = cv::Mat::ones(30, 200, CV_32FC1);
	cv::putText(img_draw, "Press any key to start", cv::Point(10, img_draw.rows-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, 2);
	cv::imshow("img_draw", img_draw);
	cv::waitKey();

	/*********************************** Initialization per shot *******************************************/
	IsCalibration = true;
	thr_valid = duration;
	IsValid = 0;
	IsWarning = 0;
	IsError = 0;
	framenum = -1;
	/**************************************************************************************************/
	
	gettimeofday(&mytime, NULL);
	msec00 = (mytime.tv_sec%1000)*1000 + mytime.tv_usec/1000.0;

	while (++framenum < duration){
	
		video.read(img0);
		for (i = 0; i < video_timeunit; i++)
			video.read(img0);
		if (img0.empty())
			break;
		
		gettimeofday(&mytime, NULL);
		msec0 = (mytime.tv_sec%1000)*1000 + mytime.tv_usec/1000.0;
		
		cv::remap(img0, img, mapx, mapy, CV_INTER_LINEAR);

		img.convertTo(img, CV_32FC3, 1 / 255.0);
		img_draw = img.clone();
		
		/*cv::imshow("img_draw", img_draw);
		cv::waitKey();
		continue;*/

		cv::split(img, bgr);
		cv::cvtColor(img, img_hsv, CV_BGR2HSV);
		cv::split(img_hsv, hsv);
		hsv[0] = hsv[0] / 360.0;

		cv::bitwise_and(bgr[2] < 0.45, bgr[0] < 0.45, bnw); //heuristic
		//cv::erode(bnw, bnw, rec5, cv::Point(-1, -1), 1);
		cv::dilate(bnw, bnw, rec5, cv::Point(-1, -1), 1);

		/*cv::imshow("bnw", bnw);
		cv::waitKey();*/

		if (IsCalibration){
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			cv::Sobel(gray, edgex, -1, 1, 0);
			cv::Sobel(gray, edgey, -1, 0, 1);
			edgex = cv::abs(edgex);
			edgey = cv::abs(edgey);
			cv::bitwise_or(edgey > 0.2, edgex + edgey > 0.5, edge);
			cv::multiply(edge, ROI_horizon, edge);
			
			/*cv::line(img_draw, getimgbywld(cv::Point2f(-W, D_hog)), getimgbywld(cv::Point2f(W, D_hog)), cv::Scalar(0, 0, 1), 2);
			cv::line(img_draw, getimgbywld(cv::Point2f(-W, D_hog)), getimgbywld(cv::Point2f(-W, 0)), cv::Scalar(0, 0, 1), 2);
			cv::line(img_draw, getimgbywld(cv::Point2f(W, D_hog)), getimgbywld(cv::Point2f(W, 0)), cv::Scalar(0, 0, 1), 2);
			cv::imshow("img_draw", img_draw);
			cv::imshow("edge", 255*edge);
			cv::waitKey();*/

			cv::HoughLinesP(edge, lines, 1, CV_PI / 180, 150, 300, 50);
			if (!lines.size()){
				printf("error : line search failed\n");
				IsError = 1;
				return IsError;
			}

			/*for (i = 0; i < lines.size(); i++)
				cv::line(img_draw, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 1), 2);
			cv::imshow("img_draw", img_draw);
			cv::imshow("edge", edge);
			cv::waitKey();*/

			for (i = 0; i < lines.size(); i++){
				pt1 = getwldbyimg(cv::Point2f(lines[i][0], lines[i][1]));
				pt2 = getwldbyimg(cv::Point2f(lines[i][2], lines[i][3])) - pt1;
				pt2 = pt2 * (1.0 / sqrt(pt2.x*pt2.x + pt2.y*pt2.y));
				lx = pt2.x;
				ly = pt2.y;
				pt2 = pt1 - (pt1.x*pt2.x + pt1.y*pt2.y)*pt2;
				ld = sqrt(pt2.x*pt2.x + pt2.y*pt2.y);
				if (fabs(ly) > lr && fabs(ld - W) < ldr && pt2.x < 0)
					lines_l.push_back(lines[i]);
				else if (fabs(ly) > lr && fabs(ld - W) < ldr && pt2.x > 0)
					lines_r.push_back(lines[i]);
				else if (fabs(lx) > lr*0.5 && fabs(ld - D_hog) < ldr*4)
					lines_h.push_back(lines[i]);
			}
			if (lines_l.size() == 0 || lines_r.size() == 0 || lines_h.size() == 0){
				printf("error : line classification failed\n");
				printf("l: %d, r: %d, h: %d\n", (int)lines_l.size(), (int)lines_r.size(), (int)lines_h.size());
				IsError = 1;
				return IsError;
			}

			/*for (i = 0; i < lines_l.size(); i++)
				cv::line(img_draw, cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Scalar(0.5, 0.5, 0.5), 2);
			for (i = 0; i < lines_r.size(); i++)
				cv::line(img_draw, cv::Point2f(lines_r[i][0], lines_r[i][1]), cv::Point2f(lines_r[i][2], lines_r[i][3]), cv::Scalar(0, 0, 0), 2);
			for (i = 0; i < lines_h.size(); i++)
				cv::line(img_draw, cv::Point2f(lines_h[i][0], lines_h[i][1]), cv::Point2f(lines_h[i][2], lines_h[i][3]), cv::Scalar(0, 0, 1), 2);
			cv::imshow("img_draw", img_draw);
			cv::imshow("edge", edge);
			cv::waitKey();*/

			for (i = 0; i < lines_l.size(); i++){
				for (j = 0; j < lines_r.size(); j++){
					for (k = 0; k < lines_h.size(); k++){
						quad[0] = getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));
						quad[1] = getintersection(cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));
						quad[2] = getintersection(cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]), cv::Point2f(0, img.rows), cv::Point2f(img.cols, img.rows));
						quad[3] = getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(0, img.rows), cv::Point2f(img.cols, img.rows));
						if (minVal > cv::contourArea(quad) || (i == 0 && j == 0 && k ==0)){
							minVal = cv::contourArea(quad);
							amin_quad = quad;
						}
					}
				}
			}
			
			ROI = cv::Mat::zeros(resize_ROI*img.rows, resize_ROI*img.cols, CV_8UC1);
			for (i = 0; i < ROI.cols; i++){
				for (j = 0; j < ROI.rows; j++){
					pt = cv::Point(i / resize_ROI, j / resize_ROI);
					if (cv::pointPolygonTest(amin_quad, pt, false) > 0){
						ROI.at<uchar>(j, i) = 1;
						if (bnw.at<uchar>(pt) == 0){
							point2.at<float>(0, 0) = hsv[0].at<float>(pt);
							point2.at<float>(0, 1) = hsv[1].at<float>(pt);
							points.push_back(point2);
						}
					}
				}
			}
			num_region = 3; //heuristic
			cv::kmeans(points, num_region, region, cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 0.1), 5, cv::KMEANS_RANDOM_CENTERS, props);
			points.release();

			bluenum = 0;

			minVal = fabs(props.at<float>(0, 0) - 230 / 360.0) - props.at<float>(0, 1);
			for (i = 1; i < props.rows; i++){
				if (minVal > fabs(props.at<float>(i, 0) - 230 / 360.0) - props.at<float>(i, 1)){
					minVal = fabs(props.at<float>(i, 0) - 230 / 360.0) - props.at<float>(i, 1);
					bluenum = i;
				}
			}

			/************************ ellipse fitting ***********************/
			b = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
			for (i = 0; i < img.rows - 2; i++){
				for (j = 0; j < img.cols - 2; j++){
					if (ROI.at<uchar>(resize_ROI*i, resize_ROI*j)){
						minVal = (props.at<float>(bluenum, 0) - hsv[0].at<float>(i, j))*(props.at<float>(bluenum, 0) - hsv[0].at<float>(i, j)) + (props.at<float>(bluenum, 1) - hsv[1].at<float>(i, j))*(props.at<float>(bluenum, 1) - hsv[1].at<float>(i, j));
						for (x = 0; x < props.rows; x++){
							if (minVal >(props.at<float>(x, 0) - hsv[0].at<float>(i, j))*(props.at<float>(x, 0) - hsv[0].at<float>(i, j)) + (props.at<float>(x, 1) - hsv[1].at<float>(i, j))*(props.at<float>(x, 1) - hsv[1].at<float>(i, j)))
								break;
						}
						if (x == num_region)
							b.at<uchar>(i, j) = 255;
					}
				}
			}

			cv::findContours(b, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			maxVal = contours[0].size();
			ellipse = 0;
			for (i = 0; i < contours.size(); i++){
				points.release();
				cv::convexHull(contours[i], hull);
				for (j = 0; j < hull.size(); j++){
					if (bnw.at<uchar>(hull[j]) == 0){
						point5.at<float>(0, 0) = hull[j].x * hull[j].x;
						point5.at<float>(0, 1) = hull[j].y * hull[j].y;
						point5.at<float>(0, 2) = hull[j].x * hull[j].y;
						point5.at<float>(0, 3) = hull[j].x;
						point5.at<float>(0, 4) = hull[j].y;
						points.push_back(point5);
					}
				}
				if (points.rows < 10)
					continue;				
					
				Q = cv::Mat::ones(points.rows, 1, CV_32FC1);
				Q = (points.t()*points).inv()*(points.t()*Q);

				error = 0;
				points = points*Q;
				for (j = 0; j < points.rows; j++)
					error += abs(points.at<float>(j, 0) - 1);
				
				if (error / points.rows > 0.5)
					continue;
					if (maxVal < contours[i].size()){
					maxVal = contours[i].size();
					ellipse = i;
				}
			}
			points.release();
			cv::convexHull(contours[ellipse], hull);
			for (i = 0; i < hull.size(); i++){
				if (bnw.at<uchar>(hull[i]) == 0){
					point5.at<float>(0, 0) = hull[i].x * hull[i].x;
					point5.at<float>(0, 1) = hull[i].y * hull[i].y;
					point5.at<float>(0, 2) = hull[i].x * hull[i].y;
					point5.at<float>(0, 3) = hull[i].x;
					point5.at<float>(0, 4) = hull[i].y;
					points.push_back(point5);
				}
			}
			Q = cv::Mat::ones(points.rows, 1, CV_32FC1);
			Q = (points.t()*points).inv()*(points.t()*Q);
			points.release();

			/****************************************************************/

			for (i = 0; i < lines_l.size(); i++){
				for (j = 0; j < lines_r.size(); j++){
					for (k = 0; k < lines_h.size(); k++){
							
						van = getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]));	
						vertex[0] = getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));	
						vertex[1] = getintersection(cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));						
					
						A1 = Q.at<float>(3, 0) / 2 + Q.at<float>(0, 0)*van.x + Q.at<float>(2, 0)*van.y / 2;
						B1 = Q.at<float>(4, 0) / 2 + Q.at<float>(1, 0)*van.y + Q.at<float>(2, 0)*van.x / 2;
						C1 = (1 - 0.5*(Q.at<float>(3, 0)*van.x + Q.at<float>(4, 0)*van.y)) / B1;
						A2 = Q.at<float>(0, 0)*B1*B1 + Q.at<float>(1, 0)*A1*A1 - Q.at<float>(2, 0)*A1*B1;
						B2 = -(2 * Q.at<float>(1, 0)*A1*C1 - Q.at<float>(2, 0)*B1*C1 - Q.at<float>(3, 0)*B1 + Q.at<float>(4, 0)*A1);
						C2 = Q.at<float>(1, 0)*C1*C1 + Q.at<float>(4, 0)*C1 - 1;
						t1 = 0.5*(-B2 + sqrt(B2*B2 - 4 * A2*C2)) / A2;
						t2 = 0.5*(-B2 - sqrt(B2*B2 - 4 * A2*C2)) / A2;
						e1 = cv::Point2f(B1*t1, C1 - A1*t1);
						e2 = cv::Point2f(B1*t2, C1 - A1*t2);
						if (e1.x > e2.x){
							pt1 = e1;
							e1 = e2;
							e2 = pt1;
						}
						van_ = getintersection(e1, e2, cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));
						A1 = Q.at<float>(3, 0) / 2 + Q.at<float>(0, 0)*van_.x + Q.at<float>(2, 0)*van_.y / 2;
						B1 = Q.at<float>(4, 0) / 2 + Q.at<float>(1, 0)*van_.y + Q.at<float>(2, 0)*van_.x / 2;
						C1 = (1 - 0.5*(Q.at<float>(3, 0)*van_.x + Q.at<float>(4, 0)*van_.y)) / B1;
						A2 = Q.at<float>(0, 0)*B1*B1 + Q.at<float>(1, 0)*A1*A1 - Q.at<float>(2, 0)*A1*B1;
						B2 = -(2 * Q.at<float>(1, 0)*A1*C1 - Q.at<float>(2, 0)*B1*C1 - Q.at<float>(3, 0)*B1 + Q.at<float>(4, 0)*A1);
						C2 = Q.at<float>(1, 0)*C1*C1 + Q.at<float>(4, 0)*C1 - 1;
						t1 = 0.5*(-B2 + sqrt(B2*B2 - 4 * A2*C2)) / A2;
						t2 = 0.5*(-B2 - sqrt(B2*B2 - 4 * A2*C2)) / A2;
						e3 = cv::Point2f(B1*t1, C1 - A1*t1);
						e4 = cv::Point2f(B1*t2, C1 - A1*t2);
						if (e3.y > e4.y){
							pt1 = e3;
							e3 = e4;
							e4 = pt1;
						}

						imgpoint[3 * 0 + 0] = getintersection(e4, van_, e1, van);
						imgpoint[3 * 0 + 1] = e4;
						imgpoint[3 * 0 + 2] = getintersection(e4, van_, e2, van);
						imgpoint[3 * 1 + 0] = e1;
						imgpoint[3 * 1 + 1] = getintersection(e1, van_, e4, van);
						imgpoint[3 * 1 + 2] = e2;
						imgpoint[3 * 2 + 0] = getintersection(e1, van, e3, van_);
						imgpoint[3 * 2 + 1] = e3;
						imgpoint[3 * 2 + 2] = getintersection(e2, van, e3, van_);
						imgpoint[3 * 3 + 0] = getintersection(e1, van, vertex[0], vertex[1]);
						imgpoint[3 * 3 + 1] = getintersection(e4, van, vertex[0], vertex[1]);
						imgpoint[3 * 3 + 2] = getintersection(e2, van, vertex[0], vertex[1]);

						error = 0;

						for (ii = 0; ii < 12; ii++){
							A.at<float>(2 * ii, 0) = wldpoint[ii].x;
							A.at<float>(2 * ii, 1) = wldpoint[ii].y;
							A.at<float>(2 * ii, 2) = 1;
							A.at<float>(2 * ii, 6) = -wldpoint[ii].x*imgpoint[ii].x;
							A.at<float>(2 * ii, 7) = -wldpoint[ii].y*imgpoint[ii].x;
							B.at<float>(2 * ii, 0) = imgpoint[ii].x;
							A.at<float>(2 * ii + 1, 3) = wldpoint[ii].x;
							A.at<float>(2 * ii + 1, 4) = wldpoint[ii].y;
							A.at<float>(2 * ii + 1, 5) = 1;
							A.at<float>(2 * ii + 1, 6) = -wldpoint[ii].x*imgpoint[ii].y;
							A.at<float>(2 * ii + 1, 7) = -wldpoint[ii].y*imgpoint[ii].y;
							B.at<float>(2 * ii + 1, 0) = imgpoint[ii].y;
						}
						X = (A.t()*A).inv()*(A.t()*B);
						Wld2Img.at<float>(0, 0) = X.at<float>(0, 0); Wld2Img.at<float>(0, 1) = X.at<float>(1, 0); Wld2Img.at<float>(0, 2) = X.at<float>(2, 0);
						Wld2Img.at<float>(1, 0) = X.at<float>(3, 0); Wld2Img.at<float>(1, 1) = X.at<float>(4, 0); Wld2Img.at<float>(1, 2) = X.at<float>(5, 0);
						Wld2Img.at<float>(2, 0) = X.at<float>(6, 0); Wld2Img.at<float>(2, 1) = X.at<float>(7, 0); Wld2Img.at<float>(2, 2) = 1;
						Img2Wld = Wld2Img.inv();
						for (ii = 0; ii < 12; ii++){
							wp.x = (Img2Wld.at<float>(0, 0)*imgpoint[ii].x + Img2Wld.at<float>(0, 1)*imgpoint[ii].y + Img2Wld.at<float>(0, 2)) / (Img2Wld.at<float>(2, 0)*imgpoint[ii].x + Img2Wld.at<float>(2, 1)*imgpoint[ii].y + Img2Wld.at<float>(2, 2));
							wp.y = (Img2Wld.at<float>(1, 0)*imgpoint[ii].x + Img2Wld.at<float>(1, 1)*imgpoint[ii].y + Img2Wld.at<float>(1, 2)) / (Img2Wld.at<float>(2, 0)*imgpoint[ii].x + Img2Wld.at<float>(2, 1)*imgpoint[ii].y + Img2Wld.at<float>(2, 2));
							err = cv::Point2f(wldpoint[ii].x - wp.x, wldpoint[ii].y - wp.y);
							error += weight[ii] * sqrt(err.x*err.x + err.y*err.y);
						}

						/*cv::solvePnP(wldpoint, imgpoint, Int, Dist, Rvec, Tvec);
						cv::Rodrigues(Rvec, Rmat);
						for (ii = 0; ii < 12; ii++){
							err = cv::Point2f(wldpoint[ii].x, wldpoint[ii].y) - getwldbyimg(imgpoint[ii]);
							error += weight[ii] * sqrt(err.x*err.x + err.y*err.y);
						}*/

						//vec3 = -Rmat.inv() * Tvec;							
						//error += 0.001*fabs(vec3.at<double>(2, 0) - 2000);

						tri[0] = vertex[0];
						tri[1] = vertex[1];
						tri[2] = cv::Point2f(vec3.at<double>(2, 0), vec3.at<double>(2, 1));
												
						if ((minVal > error && cv::pointPolygonTest(tri, getwldbyimg(0.5*cv::Point2f(lines_h[k][0] + lines_h[k][2], lines_h[k][1] + lines_h[k][3])), false) > 0)|| (i == 0 && j == 0 && k == 0)){
							sidelines[0] = lines_l[i];
							sidelines[1] = lines_r[j];
							hogline = lines_h[k];
							minVal = error;
						}
					}
				}
			}
			if (minVal > 200){
				printf("error : calibration is failed\n");
				printf("Calibration Error : %f\n", minVal);
				IsError = 1;
				return IsError;
			}

			lines_l.clear();
			lines_r.clear();
			lines_h.clear();

			van = getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]));	
			vertex[0] = getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]));	
			vertex[1] = getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]));

			A1 = Q.at<float>(3, 0) / 2 + Q.at<float>(0, 0)*van.x + Q.at<float>(2, 0)*van.y / 2;
			B1 = Q.at<float>(4, 0) / 2 + Q.at<float>(1, 0)*van.y + Q.at<float>(2, 0)*van.x / 2;
			C1 = (1 - 0.5*(Q.at<float>(3, 0)*van.x + Q.at<float>(4, 0)*van.y)) / B1;
			A2 = Q.at<float>(0, 0)*B1*B1 + Q.at<float>(1, 0)*A1*A1 - Q.at<float>(2, 0)*A1*B1;
			B2 = -(2 * Q.at<float>(1, 0)*A1*C1 - Q.at<float>(2, 0)*B1*C1 - Q.at<float>(3, 0)*B1 + Q.at<float>(4, 0)*A1);
			C2 = Q.at<float>(1, 0)*C1*C1 + Q.at<float>(4, 0)*C1 - 1;
			t1 = 0.5*(-B2 + sqrt(B2*B2 - 4 * A2*C2)) / A2;
			t2 = 0.5*(-B2 - sqrt(B2*B2 - 4 * A2*C2)) / A2;
			e1 = cv::Point2f(B1*t1, C1 - A1*t1);
			e2 = cv::Point2f(B1*t2, C1 - A1*t2);
			if (e1.x > e2.x){
				pt1 = e1;
				e1 = e2;
				e2 = pt1;
			}
			van_ = getintersection(e1, e2, cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]));
			A1 = Q.at<float>(3, 0) / 2 + Q.at<float>(0, 0)*van_.x + Q.at<float>(2, 0)*van_.y / 2;
			B1 = Q.at<float>(4, 0) / 2 + Q.at<float>(1, 0)*van_.y + Q.at<float>(2, 0)*van_.x / 2;
			C1 = (1 - 0.5*(Q.at<float>(3, 0)*van_.x + Q.at<float>(4, 0)*van_.y)) / B1;
			A2 = Q.at<float>(0, 0)*B1*B1 + Q.at<float>(1, 0)*A1*A1 - Q.at<float>(2, 0)*A1*B1;
			B2 = -(2 * Q.at<float>(1, 0)*A1*C1 - Q.at<float>(2, 0)*B1*C1 - Q.at<float>(3, 0)*B1 + Q.at<float>(4, 0)*A1);
			C2 = Q.at<float>(1, 0)*C1*C1 + Q.at<float>(4, 0)*C1 - 1;
			t1 = 0.5*(-B2 + sqrt(B2*B2 - 4 * A2*C2)) / A2;
			t2 = 0.5*(-B2 - sqrt(B2*B2 - 4 * A2*C2)) / A2;
			e3 = cv::Point2f(B1*t1, C1 - A1*t1);
			e4 = cv::Point2f(B1*t2, C1 - A1*t2);
			if (e3.y > e4.y){
				pt1 = e3;
				e3 = e4;
				e4 = pt1;
			}

			imgpoint[3 * 0 + 0] = getintersection(e4, van_, e1, van);
			imgpoint[3 * 0 + 1] = e4;
			imgpoint[3 * 0 + 2] = getintersection(e4, van_, e2, van);
			imgpoint[3 * 1 + 0] = e1;
			imgpoint[3 * 1 + 1] = getintersection(e1, van_, e4, van);
			imgpoint[3 * 1 + 2] = e2;
			imgpoint[3 * 2 + 0] = getintersection(e1, van, e3, van_);
			imgpoint[3 * 2 + 1] = e3;
			imgpoint[3 * 2 + 2] = getintersection(e2, van, e3, van_);
			imgpoint[3 * 3 + 0] = getintersection(e1, van, vertex[0], vertex[1]);
			imgpoint[3 * 3 + 1] = getintersection(e4, van, vertex[0], vertex[1]);
			imgpoint[3 * 3 + 2] = getintersection(e2, van, vertex[0], vertex[1]);

			cv::solvePnP(wldpoint, imgpoint, Int, Dist, Rvec, Tvec);
			cv::Rodrigues(Rvec, Rmat);
			vec3 = -Rmat.inv() * Tvec;
			cam_pos = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
			mat3 = Rmat.inv();
			cam_dir = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));
			if (IsDisplaying || IsRecording){
				cv::line(img_draw, getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0.5, 0.5, 0.5), 5);
				cv::line(img_draw, getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getintersection(cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0, 0, 1), 5);
				for (i = 0; i<hull.size() - 1; i++)
					cv::line(img_draw, hull[i], hull[i + 1], cv::Scalar(1, 0, 0), 2);
				cv::line(img_draw, hull[i], hull[0], cv::Scalar(1, 0, 0), 2);
				cv::line(img_draw, e1, e2, cv::Scalar(0, 0, 0), 2);
				cv::line(img_draw, e3, e4, cv::Scalar(0, 0, 0), 2);
				/*cv::line(img_draw, vertex[0], vertex[1], cv::Scalar(0, 0, 0), 2);
				for (i = 0; i < 12; i++)
					cv::circle(img_draw, imgpoint[i], 5, cv::Scalar(0, 0, 0), -1);*/
			}

			if (IsCalibration){
				quad[0] = getimgbywld(cv::Point2f(-W, -500), 0);
				quad[1] = getimgbywld(cv::Point2f(-W, D_hog + 3000), 0);
				quad[2] = getimgbywld(cv::Point2f(W, D_hog + 3000), 0);
				quad[3] = getimgbywld(cv::Point2f(W, -500), 0);
	
				ROI = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
				for (i = 0; i < ROI.cols; i++){
					for (j = 0; j < ROI.rows; j++){
						if (cv::pointPolygonTest(quad, cv::Point2f(i, j), true) > 10)
							ROI.at<uchar>(j, i) = 1;
					}
				}
				IsCalibration = false;
			}
		}
		
		cv::multiply(bnw, ROI, bnw);

		contours.clear();
		cv::findContours(bnw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (i = 0; i < contours.size(); i++){
			if (contours[i].size() < 20){
				contours.erase(contours.begin() + i);
				i--;
			}
			else
				coarsebox.push_back(cv::boundingRect(contours[i]));
		}
		for (i = 0; i < coarsebox.size(); i++){
			pad = getEllipse(getwldbyimg(coarsebox[i].tl() + 0.5*cv::Point(coarsebox[i].width, coarsebox[i].height), h), d_).height;
			coarsebox[i] = coarsebox[i] - cv::Point(1, pad) + cv::Size(2, pad); // padding
			cv::convexHull(contours[i], hull);

			for (j = 0; j < hull.size(); j++)
				hull[j] = getimgbywld(getwldbyimg(hull[j]), h);
				wldcontour.clear();

			for (j = 0; j < hull.size(); j++)
				wldcontour.push_back(getwldbyimg(hull[j], h));
			maxnum_Mask = cv::contourArea(wldcontour) / (CV_PI*d*d / 4);
			
			if (maxnum_Mask > shotnum)
				maxnum_Mask = shotnum;
			else if (maxnum_Mask < 1)
				continue;

			Mask = hsv[1](coarsebox[i]).clone(); //when stone's saturation is larger than background.
			bias_Mask = cv::Point(coarsebox[i].x, coarsebox[i].y);
			//cv::GaussianBlur(Mask, Mask, cv::Size(3, 3), 1);
			cv::Sobel(Mask, edgex, -1, 1, 0);
			cv::Sobel(Mask, edgey, -1, 0, 1);
			cv::divide(-edgex, edgey, Ort);
			edgex = cv::abs(edgex);
			edgey = cv::abs(edgey);
			cv::minMaxLoc(edgex, &minVal, &maxVal);
			t1 = maxVal;
			cv::minMaxLoc(edgey, &minVal, &maxVal);
			if (t1 > maxVal)
				maxVal = t1;
			Mgt = (edgex + edgey) / maxVal;

			wldRect = cv::boundingRect(wldcontour);
			bias_Vote = (wldRect.tl() - 0.5*cv::Point(pad_Vote, pad_Vote)) * (1.0 / unit);
			Vote = cv::Mat::zeros((wldRect.height + pad_Vote) / unit, (wldRect.width + pad_Vote) / unit, CV_32FC1);
			Vote_prior = cv::Mat::zeros(Vote.size(), CV_32FC1);
			for (x = 0; x < Vote.cols; x++){
				for (y = 0; y < Vote.rows; y++)
					Vote_prior.at<float>(y, x) = cv::pointPolygonTest(wldcontour, unit*(cv::Point(x, y) + bias_Vote), true);
			}
			cv::minMaxLoc(Vote_prior, &minVal, &maxVal);
			if (maxVal < 0.5*d_ - p)
				continue;
			
			if (IsDisplaying || IsRecording){
				/*for (j = 0; j < hull.size() - 1; j++)
					cv::line(img_draw, hull[j], hull[j + 1], cv::Scalar(0.5, 0.5, 0.5), 1);
				cv::line(img_draw, hull[j], hull[0], cv::Scalar(0.5, 0.5, 0.5), 1);
				cv::rectangle(img_draw, coarsebox[i], cv::Scalar(0.5, 0.5, 0.5), 2);*/
				//cv::putText(img_draw, patch::to_string(maxnum_Mask), coarsebox[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0.5, 0.5, 0.5), 2);
			}

			for (x = 0; x < Vote.cols; x++){
				for (y = 0; y < Vote.rows; y++){
					if (Vote_prior.at<float>(y, x) > 0){
						pt = getimgbywld(unit*(cv::Point(x, y) + bias_Vote), h) + cv::Point2f(0.5, 0.5);
						if (coarsebox[i].contains(pt) && Mgt.at<float>(pt - bias_Mask) > 0.3){
							wp = getUnitvector(unit*(cv::Point(x, y) + bias_Vote), Ort.at<float>(pt - bias_Mask));
							for (k = 0; k < num_c; k++){
								if (fabs(wp.dot(circle[k])) / circlenorm[k] < cos_max)
									Vote.at<float>(cv::Point(x, y) + circle[k])++;
							}
						}
					}
				}
			}

			for (x = 0; x < Vote.cols; x++){
				for (y = 0; y < Vote.rows; y++){
					if (Vote_prior.at<float>(y, x) < 0.5*d_ - p)
						Vote.at<float>(y, x) = 0;
				}
			}
			//cv::GaussianBlur(Vote, Vote, cv::Size(3, 3), 1);

			/*cv::minMaxLoc(Vote, &minVal, &maxVal);
			Vote = Vote / maxVal;
			cv::imshow("Vote", Vote);
			cv::imshow("Mask", Mask);
			cv::imshow("Mgt", Mgt);
			cv::waitKey();*/

			V_m = cv::Mat::zeros(maxnum_Mask, 1, CV_32FC1);
			for (j = 0; j < maxnum_Mask; j++){
				cv::minMaxLoc(Vote, &minVal, &maxVal, &minLoc, &maxLoc);
				if (maxVal == 0){
					maxnum_Mask = j;
					break;
				}
				V_m.at<float>(j, 0) = maxVal;
				wldcenter.push_back((maxLoc + bias_Vote)*unit);
				for (k = 0; k < num_c_; k++)
					Vote.at<float>(maxLoc + circle_[k]) = 0;
			}
			if (maxnum_Mask < 1)
				continue;
			V_m = V_m / num_c;

			cv::sortIdx(V_m, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			for (j = 0; j < maxnum_Mask; j++){
				k = sortIdx1.at<int>(j, 0);
				//printf("%f\n", V_m.at<float>(k, 0));
				if (V_m.at<float>(k, 0) >= 0.45){
					wldcentroid.push_back(wldcenter[k]);
					centroid.push_back(getimgbywld(wldcenter[k], h));
					cand.push_back(1);
				}
				else
					break;
			}
			//printf("--------%d / %d\n", j, maxnum_Mask);
			wldcenter.clear();
			V_m.release();
		}
		coarsebox.clear();

		num_stone = 0;
		if (cand.size() != 0){
			minVals = cv::Mat::zeros(cand.size(), 1, CV_32FC1);
			minIdxs = cv::Mat::zeros(cand.size(), 1, CV_32FC1);
			if (framenum > 0){
				for (i = 0; i < cand.size(); i++){
					if (cand[i]){
						for (j = 0; j < prv_cand.size(); j++){
							if (prv_cand[j]){
								minVal = (prv_wldcentroid[j] - wldcentroid[i]).dot(prv_wldcentroid[j] - wldcentroid[i]);
								amin = j;
								break;
							}
						}
						for (; j < prv_cand.size(); j++){
							if (prv_cand[j] && minVal >(prv_wldcentroid[j] - wldcentroid[i]).dot(prv_wldcentroid[j] - wldcentroid[i])){
								minVal = (prv_wldcentroid[j] - wldcentroid[i]).dot(prv_wldcentroid[j] - wldcentroid[i]);
								amin = j;
							}
						}
						minVals.at<float>(i, 0) = minVal + 0.001;
						minIdxs.at<int>(i, 0) = amin;
					}
				}
				cv::sortIdx(minVals, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
				for (j = 0; j < cand.size(); j++){
					i = sortIdx1.at<int>(j, 0);
					amin = minIdxs.at<int>(i, 0);
					if (!prv_cand[amin]){
						for (k = 0; k < prv_cand.size(); k++){
							if (prv_cand[k]){
								minVal = (prv_wldcentroid[k] - wldcentroid[i]).dot(prv_wldcentroid[k] - wldcentroid[i]);
								amin = k;
								break;
							}
						}
						for (; k < prv_cand.size(); k++){
							if (prv_cand[k] && minVal >(prv_wldcentroid[k] - wldcentroid[i]).dot(prv_wldcentroid[k] - wldcentroid[i])){
								minVal = (prv_wldcentroid[k] - wldcentroid[i]).dot(prv_wldcentroid[k] - wldcentroid[i]);
								amin = k;
							}
						}
						if (amin == minIdxs.at<int>(i, 0)){
							//cand[i] = 0;
							break;
							j++;
						}
						minVals.at<float>(i, 0) = minVal + 0.001;
						minIdxs.at<int>(i, 0) = amin;
						cv::sortIdx(minVals, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
						j--;
						continue;
					}
					if (minVals.at<float>(i, 0) < errorrange*errorrange){
						if (prv_cand[amin] > 1)
							cand[i] = prv_cand[amin];
						else {
							cand[i] = 0;
							continue;
						}
						stone[cand[i] - 2][framenum] = centroid[i];
						wldstone[cand[i] - 2][framenum] = wldcentroid[i];
						Identified[cand[i] - 2][framenum] = 1;
						color[cand[i] - 2] = getColor(wldstone[cand[i] - 2][framenum], hsv);
						prv_cand[amin] = 0;
					}
				}//MIn-min
			}
			else {
				stonenum = 0;
				for (i = 0; i < cand.size(); i++){
					if (cand[i]){
						cand[i] = stonenum + 2;
						stone[stonenum][0] = centroid[i];
						wldstone[stonenum][0] = wldcentroid[i];
						Identified[stonenum][0] = 1;
						color[stonenum] = getColor(wldstone[stonenum][0], hsv);
						stonenum++;
					}
				}
			}
		}

		for (i = 0; i < stonenum; i++){
			if (!Identified[i][framenum]){
				for (j = 0; j < prv_cand.size(); j++){
					if (prv_cand[j] == i + 2){
						box_cor = prv_img(getEllipse(prv_wldcentroid[j], d_));
						box_cor_ = img(getEllipse(prv_wldcentroid[j], d_) - cv::Point(mx, my) + cv::Size(2 * mx, 2 * my));
						cv::matchTemplate(box_cor_, box_cor, result, CV_TM_SQDIFF_NORMED);
						cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
						stone[i][framenum] = stone[i][framenum - 1] + cv::Point2f(minLoc.x - mx, minLoc.y - my);
						wldstone[i][framenum] = getwldbyimg(stone[i][framenum]);
						cand.push_back(prv_cand[j]);
						centroid.push_back(stone[i][framenum]);
						wldcentroid.push_back(wldstone[i][framenum]);
						break;
					}
				}
				IsWarning++;
			}
		}

		//check detection
		for (i = 0; i < cand.size(); i++){
			if (cand[i] > 1)
				num_stone++;
		}
		if (num_stone != stonenum || num_stone > shotnum){
			printf("error : detection is failed\n");
			for (i = 0; i < num_stone; i++){
				for (j = 0; j < framenum; j++){
					wldstone[i][j] = cv::Point2f(0, 0);
					stone[i][j] = cv::Point2f(0, 0);
				}						
			}
			IsValid= 0;
			framenum = -1;
		}
		else if (!IsWarning){
			IsValid++;
		}
			
		prv_img = img.clone();
		prv_centroid = centroid;
		centroid.clear();
		prv_wldcentroid = wldcentroid;
		wldcentroid.clear();
		prv_cand = cand;
		cand.clear();

		if (IsDisplaying || IsRecording){
			for (i = 0; i < stonenum; i++){
				cv::circle(img_draw, stone[i][framenum], 3, color[i], -1);
				cv::circle(img_draw, stone[i][framenum], 4, color[i], -1);
				if (Identified[i][framenum])
					cv::putText(img_draw, patch::to_string(i + 1), cv::Point2f(stone[i][framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
				else
					cv::putText(img_draw, patch::to_string(i + 1), cv::Point2f(stone[i][framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0.5), 2);
			}
		}

		gettimeofday(&mytime, NULL);
		msec = (mytime.tv_sec%1000)*1000 + mytime.tv_usec/1000.0;
		latency = msec - msec0;

		/************************************ Display *************************************/
		
		if (IsDisplaying || IsRecording){
			Sheet_draw = Sheet.clone();
			for (i = 0; i < stonenum; i++){
				displaypoint = displayscale*(cv::Point(wldstone[i][framenum].y, wldstone[i][framenum].x)) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5, color[i], -1);
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5 - 1, cv::Scalar(0, 0, 0), 2);
			}
			cv::GaussianBlur(Sheet_draw, Sheet_draw, cv::Size(3, 3), 1);
		}
		if (IsDisplaying){
			cv::imshow("img_draw", img_draw);
			cv::imshow("Sheet_draw", Sheet_draw);
			cv::waitKey(10);
		}
		if (IsRecording){
			img_draw.convertTo(img_draw, CV_8U, 255);
			Sheet_draw.convertTo(Sheet_draw, CV_8U, 255);
			img_draw.copyTo(Frame(cv::Rect(0, 0, img_draw.cols, img_draw.rows)));
			Sheet_draw.copyTo(Frame(cv::Rect(0, img_draw.rows, Sheet_draw.cols, Sheet_draw.rows)));
			cv::imwrite(patch::to_string(framenum) + ".png", Frame);
		}
		/********************************************************************************/

		if (IsDisplaying || IsRecording || IsLogging){
			printf("====current state=====\n");
			log << "====current state=====\n";
			if (IsCalibration){
				printf("IsCalibration\n");
				log << "IsCalibration\n";
			}
			if (IsWarning){
				printf("IsWarning\n");
				log << "IsWarning\n";
			}
			if (IsError){
				printf("IsError\n");
				log << "IsError\n";
			}
			if (IsValid){
				printf("Validation %.1f%%\n", 100.0*IsValid / thr_valid);
				log << "Validation ";
				log << 100.0*IsValid / thr_valid;
				log << "\n";
			}
			printf("======================\n");
			log << "======================\n";
			printf("time : %.2f\n", 0.001*(msec - msec00));
			log << "time : ";
			log << 0.001*(msec - msec00);
			log << "s\n";
			printf("frame number : %d\n", framenum);
			log << "frame number : ";
			log << framenum;
			log << "\n";
			printf("latency : %d\n", latency);
			log << "latency : ";
			log << latency;
			log << "\n";
			if (num_stone){
				printf("number of stone = %d\n", num_stone);
				for (i = 0; i < num_stone; i++){
					printf("stone %d : %f, %f\n", i + 1, wldstone[i][framenum].x, wldstone[i][framenum].y);
					log << "stone ";
					log << i + 1;
					log << " : ";
					log << wldstone[i][framenum].x;
					log << ", ";
					log << wldstone[i][framenum].y;
					log << "\n";
				}
			}
			printf("\n\n\n\n\n");
			log << "\n\n\n\n\n";
		}

		IsWarning = 0;
		IsError = 0;
	}

	log.close();
	
	return 0;

}

cv::Point2f getwldbyimg(cv::Point2f imgpoint, double z){

	cv::Mat imgpoint_(1, 1, CV_64FC2);
	imgpoint_.at<cv::Vec2d>(0, 0) = cv::Vec2d(imgpoint.x, imgpoint.y);
	cv::undistortPoints(imgpoint_, imgpoint_, Int, Dist);

	cv::Mat imgpoint_h = cv::Mat::ones(3, 1, CV_64FC1);
	imgpoint_h.at<double>(0, 0) = imgpoint_.at<double>(0, 0);
	imgpoint_h.at<double>(1, 0) = imgpoint_.at<double>(0, 1);

	double lambda = z + Rmat.at<double>(0, 2) * Tvec.at<double>(0, 0) + Rmat.at<double>(1, 2) * Tvec.at<double>(1, 0) + Rmat.at<double>(2, 2) * Tvec.at<double>(2, 0);
	lambda /= Rmat.at<double>(0, 2) * imgpoint_h.at<double>(0, 0) + Rmat.at<double>(1, 2) * imgpoint_h.at<double>(1, 0) + Rmat.at<double>(2, 2) * imgpoint_h.at<double>(2, 0);

	cv::Mat wldpoint = Rmat.inv() * (lambda * imgpoint_h - Tvec);

	return cv::Point2f(wldpoint.at<double>(0, 0), wldpoint.at<double>(1, 0));
}

cv::Point2f getimgbywld(cv::Point2f wldpoint, double z){

	std::vector<cv::Point2f> imgpoint;
	std::vector<cv::Point3f> w3dpoint;
	w3dpoint.push_back(cv::Point3f(wldpoint.x, wldpoint.y, z));
	cv::projectPoints(w3dpoint, Rvec, Tvec, Int, Dist, imgpoint);
	return imgpoint[0];
}

cv::Point getdisplaybywld(cv::Point2f wld){
	cv::Point display;
	/******************/
	/******************/
	return display;
}

cv::Rect getEllipse(cv::Point2f wld, double diameter){

	float theta;
	std::vector<cv::Point2f> points(6);
	for (int i = 0; i < 6; i++){
		theta = (CV_PI / 3)*i;
		points[i] = getimgbywld(wld + 0.5*diameter*cv::Point2f(cos(theta), sin(theta)), h);
	}

	return cv::fitEllipse(points).boundingRect();
}

cv::Point2f getUnitvector(cv::Point2f wld, double tan){

	cv::Point2f vec3 = getwldbyimg(getimgbywld(wld, h) + cv::Point2f(1, tan) * (1.0 / (1 + tan*tan)), h) - wld;
	vec3 = vec3 * (1.0 / sqrt(vec3.x*vec3.x + vec3.y*vec3.y));
	return vec3;
}

cv::Scalar getColor(cv::Point2f wld, std::vector<cv::Mat> hsv){
	int score_r = 0;
	int score_y = 0;
	cv::Point2f pt;
	cv::Rect ellipseRect = getEllipse(wld, d);
	for (int x = ellipseRect.x; x < ellipseRect.x + ellipseRect.width; x++){
		for (int y = ellipseRect.y; y < ellipseRect.y + ellipseRect.height; y++){
			pt = getwldbyimg(cv::Point(x, y), h) - wld;
			if (pt.x*pt.x + pt.y*pt.y < 0.25*d_*d_ && hsv[1].at<float>(y, x) > 0.1){
				if (hsv[0].at<float>(y, x) > 0.85)
					score_r++;
				else if (hsv[0].at<float>(y, x) > 0.1 && hsv[0].at<float>(y, x) < 0.25)
					score_y++;
			}
		}
	}

	if (score_r > score_y)
		return cv::Scalar(0, 0, 1);
	else if (score_r < score_y)
		return cv::Scalar(0, 1, 1);
	else
		return cv::Scalar(0, 0, 0);
}

cv::Point2f getintersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2){
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;
	float cross = d1.x*d2.y - d1.y*d2.x;
	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	return o1 + d1 * t1;
}

