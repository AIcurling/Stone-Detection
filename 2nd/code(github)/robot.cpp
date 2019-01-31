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

#include <ueye.h>
#include <ueye_deprecated.h>

#include <sys/socket.h>
#include <sys/select.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <malloc.h>
#include <termios.h>
#include <fcntl.h>

typedef struct sockaddr_in SOCKADDR_IN;
typedef int SOCKET;

union PacketCvt {
	char ch[4];
	float f;
};

class Point {
public:
	float x;
	float y;

	Point(float x = 0, float y = 0) {
		this->x = x;
		this->y = y;
	}
};

enum {
	STONE_CNT, STONE_INFO, STONE_INFO_ACK, ROBOT_INFO, RELEASE, FLAG, SPEED_PROF, POS_PROF, RESET, MODE,
	RUN, RESET_KU, START, ROBOT_MODE, PRE_MODE, MY_TURN, CALL_STONE_INFO, CALL_ROBOT_INFO, EMERGENCY, RESTART,
	CALIB_DATA, TARGET_ANGLE, CALIB_ANGLE, HOG_DIST, RIO_ENCODER, INFO_TIME, INFO_RESULT
};

#define PAC_STONE_INFO 44

#define PORT_NUM      10200
#define MAX_MSG_LEN 256
#define NET_INVALID_SOCKET	-1
#define NET_SOCKET_ERROR -1

#define meter 1000.0

#define d 260.0
#define d_ 200.0
#define h 150.0

#define d_s 440.0
#define d_s_ 600.0
#define h_s 700.0

#define W 2360.0
#define D_hog 8100.0
#define D_hog_ 29700.0
#define R_cir 1830.0

#define errorrange 5000.0
#define kalmanrange 10000.0

#define MAXNUM_STONE 16

#define NEAR 0
#define FAR 1
#define THROW 2
#define FFAR 3

#define SKIP 100
#define THROWER 101

int is_OpenCamera(int CamType, HIDS* Cam, char** Mem);

cv::Point2f getwldbyimg(cv::Point2f imgpoint, double z = 0);
cv::Point2f getimgbywld(cv::Point2f wldpoint, double z = 0);

cv::Point2f getgndbytrw(cv::Point2f trwpoint);
cv::Point2f gettrwbygnd(cv::Point2f gndpoint);

cv::Rect getEllipse(cv::Point2f wld, double diameter);
cv::Point2f getUnitvector(cv::Point2f wld, double tan);
cv::Scalar getColor(cv::Point2f wld, std::vector<cv::Mat> hsv);

cv::Point2f getintersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2);
std::vector<cv::Point2f> getintersection_ellipse(cv::Mat Q, cv::Point2f p, cv::Point2f p1);
std::vector<cv::Point2f> gettangentialpoint_ellipse(cv::Mat Q, cv::Point2f p);
cv::Vec4f getellipseprops(cv::Mat Q);
cv::Rect getStoneBoundingBox(cv::Point2f ptPredicted);

int getmsec();
int getmsec_release(float hogDist);

void *RecvThreadPoint(void *param);

void *ReleaseDecision(void *param);

void SendPacketStoneCnt(SOCKET s, int cnt, Point* posArr, float* isRedArr);
void SendPacketStoneInfo(SOCKET s, Point pos, float isRed);
void SendPacketRobotInfo(SOCKET s, float angle, Point pos, float hogDist, float hogOffs);
void SendPacketPositionProf(SOCKET s, float posx, float posy);
void SendPacketStoneInfoAck(SOCKET s, char type);
void SendPacketInfoResult(SOCKET s, long release, long arrive, Point pos);
void SendPacketPacStoneInfo(SOCKET s,Point pos1, float isRed1, Point pos2, float isRed2, Point pos3, float isRed3, Point pos4, float isRed4, Point pos5, float isRed5);
void onMouseEvent(int event, int x, int y, int flags, void* dstImage);

int is_ChangeNEAR();

char msg[MAX_MSG_LEN];
SOCKET sock;
SOCKADDR_IN servaddr = { 0 };
	
struct timeval timeout, mytime0;
fd_set reads, cpy_reads;

int gain[4][4];
cv::Size size_sensor[4];

cv::Mat Int[4];
cv::Mat Dist[4];
cv::Mat Rmat[4];
cv::Mat Rvec[4];
cv::Mat Tvec[4];
cv::Mat Rvec_mean;
cv::Mat Tvec_mean;
cv::Mat Rmat_temp, Rvec_temp, Tvec_temp;
cv::Point2f cam_pos[4];
cv::Point2f cam_dir[4];

cv::Mat Epi[6];

std::vector<cv::Point2f> quad(4);
std::vector<cv::Point> hex(6);
int transition[2];
double pad_hog, pad_back, pad_hog_, pad_mid;
cv::Mat img;
cv::Mat ROI_reach, ROI_robot[4], ROI_FAR_s, ROI_FFAR_s;

std::string Serialnum[4];
std::fstream txtfile;

std::ofstream trajectory;
std::string trajectoryname;

std::string shotname;

cv::Mat Gnd2Trw;
cv::Mat Trw2Gnd;

int cam_mode, cam_mode_distant;
int SKIPorTHROWER;
int framenum, stonenum, shotnum;
int CollidedStonenum;
int msec_waitKey, msec_waitingforencoding;
int period_sweeper_communicating;
bool IsThrown, IsRunning, IsPositioning, IsDetecting, IsLying;

HIDS Cams[4];
char* Mems[4];

int msec_running, msec_timeout, msec_startrun, msec_release, msec_thrown, msec_tracking, msec_blind, msec_cameradelay, msec_hog2hog;

float ROBOT_ID;
float Releaseangle, Releasespeed, Releasedist, angle_bias, hogDist0;

bool IsCommunicating, IsChanging, IsRealtime, IsDisplaying, IsCalibration, IsRecalibration, IsRedetecting, IsDetectingSweeper;
int IsCollided;

bool IsSnapshot[4];

pthread_t p_thread_release;
int thr_id, stat;

float fps;

int framenum_thrown, framenum_reach;

std::vector<cv::Point2f> imgpoint_pattern;

cv::VideoCapture video[4];

std::vector<int> msec_frame;

char IsSendingAck;

std::string dummy;

IMAGE_FORMAT_LIST* pformatList;
IMAGE_FORMAT_INFO formatInfo;

std::vector<cv::Point> circle;
std::vector<cv::Point> circle_;
std::vector<double> circlenorm;
	
int r_max, r_min, unit, unit_s, num_c, num_c_, num_c_s, p, p_s;
double cos_max, theta_max;


namespace patch {
	template < typename T > std::string to_string (const T& n ) {
		std::ostringstream stm;
		stm << n;
		return stm.str() ;
	}
}

int main(){
	
	/*mytime0.tv_sec = 0;
	mytime0.tv_usec = 0;
	while(1)
		printf("%d\n", getmsec());
	return 0;*/
	
	int success, nKey;
	int msec, msec0;
	int i, j, k, ii, jj, kk, x, y;
	double xd, yd, t1, t2, sum;
	
	cv::Mat vec3 = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat mat3 = cv::Mat::zeros(3, 3, CV_64FC1);

	int num_stone, num_stone_prv;
	int duration = 500;

	cv::Mat img0[4];

	cv::Size imgsize[4];

	cv::Mat ROI, ROI_FAR, ROI_FFAR;

	std::vector<cv::Mat> bgr(3);
	std::vector<cv::Mat> hsv(3);
	cv::Mat gray, img_hsv, prv_img;
	cv::Mat img_draw;
	cv::Mat edge, edgex, edgey;
	cv::Mat bnw, bnw_;
	std::vector<cv::Mat> segmentation;

	int sobelsize;
	float sobelfactor;

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
	double lr;
	double resize_ROI;

	std::vector<cv::Vec4i> lines, lines_l, lines_r, lines_h;
	cv::Vec4i sidelines[2];
	cv::Vec4i hogline;
	
	cv::Point2f van, van_;
	cv::Point2f vertex[2];
	cv::Point2f e1, e2, e3, e4, housecenter;

	cv::Mat Q, Q_;

	std::vector<cv::Point2f> intersections;
	std::vector<cv::Point2f> trwpoint;
	std::vector<cv::Point3f> gndpoint;

	std::vector<std::vector<cv::Point> > contours, contours_;
	std::vector<cv::Point> hull, hull_, contour;
	std::vector<cv::Point2f> pnts;
	std::vector<cv::Point2f> wldcontour;
	std::vector<cv::Point2f> amin_quad(4), tri(3);

	std::vector<bool> cand;
	std::vector<cv::Point2f> wldcentroid;

	cv::Mat Mask, Mgt, Ort;
	cv::Mat Vote, Vote_prior;

	cv::Point bias_Mask, bias_Vote;

	int maxnum_Mask;
	cv::Rect wldRect, stoneRect, imgRect;

	double thetax, thetay, thetaz, cx, cy, cz, thetax0;

	std::vector<cv::Point> circle_s;

	cv::Mat points;
	cv::Mat point2 = cv::Mat::zeros(1, 2, CV_32FC1);
	cv::Mat point5 = cv::Mat::zeros(1, 5, CV_32FC1);

	int num_region;
	int whitenum;
	int ellipse[2];
	bool IsOneSideline, IsOneEllipse;
	cv::Mat ellipseareas;
	cv::Mat region, props;

	std::vector<cv::Point2f> imgpoint(12);
	std::vector<cv::Point3f> wldpoint(12);
	double weight[12];

	cv::Mat Wld2Img = cv::Mat(3, 3, CV_32FC1);
	cv::Mat A = cv::Mat::zeros(2 * 12, 8, CV_32FC1);
	cv::Mat B = cv::Mat::zeros(2 * 12, 1, CV_32FC1);
	cv::Mat Img2Wld, X;
	double error;
	cv::Point2f err;

	int pad, pad_Vote, pad_Vote_s;
	

	std::vector<cv::Rect> coarseboxes;
	cv::Rect coarsebox;

	std::vector<cv::Point2f> wldcenter, wldcenter_;
	std::vector<double> score, score_;
	cv::Mat V_m;

	cv::Scalar mean, stddev;

	double farrange;

	cv::Mat sortIdx1, sortIdx2;
	cv::Mat minVals;
	cv::Mat minIdxs;

	std::vector<cv::Scalar> color(MAXNUM_STONE);
	std::vector<std::vector<cv::Point2f> > stone(MAXNUM_STONE, std::vector<cv::Point2f>(duration));
	std::vector<std::vector<cv::Point2f> > wldstone(MAXNUM_STONE, std::vector<cv::Point2f>(duration));
	std::vector<std::vector<bool> > Identified(MAXNUM_STONE, std::vector<bool>(duration));
	std::vector<bool> Collided(MAXNUM_STONE);
	std::vector<int> Stop(MAXNUM_STONE);
	msec_frame.resize(duration);

	std::vector<cv::Point2f> wldsweeper(duration);
	std::vector<cv::Point2f> sweeper(duration);
	std::vector<bool> Identified_s(duration);

	int thrownstonenum;
	bool IsEnd, IsSending;
	int IsRealEnd;
	bool IsReached[2];
	bool IsLogging;
	int IsValid, IsError, IsWarning;
	int thr_valid;
	int latency1, latency2;
	int frameunit;

	fps = 30;

	shotnum = MAXNUM_STONE;

	lr = cos((CV_PI / 180) * 15);

	unit = 10;
	theta_max = 45;
	p = 2 * unit;
	
	unit_s = 20;
	p_s = 1 * unit_s;

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
	
	for (x = -d; x < d + 1; x += unit){
		for (y = -d; y < d + 1; y += unit){
			if (x*x + y*y < d*d)
				circle_.push_back(cv::Point(x, y) * (1.0 / unit));
		}
	}
	num_c_ = circle_.size();

	for (i = 0; i < num_c; i++)
		circlenorm.push_back(sqrt(circle[i].x*circle[i].x + circle[i].y*circle[i].y));

	r_max = 0.5*d_s_ + p_s;
	r_min = 0.5*d_s_ - p_s;
	for (x = -r_max; x < r_max + 1; x += unit_s){
		for (y = -r_max; y < r_max + 1; y += unit_s){
			if (x*x + y*y > r_min*r_min && x*x + y*y < r_max*r_max)
				circle_s.push_back(cv::Point(x, y) * (1.0 / unit_s));
		}
	}
	num_c_s = circle_s.size();

	std::vector<double> circlenorm_s(num_c_s);
	for (i = 0; i < num_c_s; i++)
		circlenorm_s[i] = sqrt(circle_s[i].x*circle_s[i].x + circle_s[i].y*circle_s[i].y);

	for (x = -d; x < d + 1; x += unit){
		for (y = -d; y < d + 1; y += unit){
			if (x*x + y*y < d*d)
				circle_.push_back(cv::Point(x, y) * (1.0 / unit));
		}
	}
	num_c_ = circle_.size();

	pad_Vote = 2 * d;
	pad_Vote += 20;

	pad_Vote_s = 2 * d_s;
	pad_Vote_s += 20;

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

	pad_hog = 3*meter;
	pad_back = 1*meter;
	pad_hog_ = -2*meter;
	pad_mid = 3*meter;

	farrange = D_hog_  + pad_hog_;
	
	txtfile.open("/home/nvidia/Robot_Data/ROBOT_ID.txt");
	txtfile >> ROBOT_ID;
	txtfile.close();

	/***************************************** Pre-Calibration *********************************************/
	Int[NEAR] = (cv::Mat_<double>(3, 3) << 1138.6, 0, 638.7978, 0, 864.1972, 354.4922, 0, 0, 1);
	//Dist[NEAR] = (cv::Mat_<double>(5, 1) << 0.0891, -0.2633, 0, 0, 0);
	Dist[NEAR] = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
	Int[FAR] = (cv::Mat_<double>(3, 3) << 3714.03842580765, 0, 655.112124729893, 0, 2787.10497942232, 310.648660856881, 0, 0, 1);
	Dist[FAR] = (cv::Mat_<double>(5, 1) << -0.983322870859286, 1.38330454944286, 0, 0, 0);
	cv::Mat	Int_THROW = (cv::Mat_<double>(3, 3) << 550.5107, 0, 588.6645, 0, 415.7317, 350.6635, 0, 0, 1);
	Int[THROW] = (cv::Mat_<double>(3, 3) << 550.5107*0.5, 0, 588.6645*0.5, 0, 415.7317*0.5, 350.6635*0.5, 0, 0, 1);
	Dist[THROW] = (cv::Mat_<double>(5, 1) << -0.2990, 0.0794, 0, 0, -0.0089);
	Int[FFAR] = (cv::Mat_<double>(3, 3) << 9054.4, 0, 657.3908, 0, 6796.3, 257.9432, 0, 0, 1);
	//Dist[FFAR] = (cv::Mat_<double>(5, 1) << 0.4374 -57.3332, 0, 0, 0);
	Dist[FFAR] = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

	imgsize[NEAR].width = 1280; imgsize[NEAR].height = 720;
	imgsize[FAR].width = 1280; imgsize[FAR].height = 720;
	imgsize[THROW].width = 1280*0.5; imgsize[THROW].height = 720*0.5;
	imgsize[FFAR].width = 1280; imgsize[FFAR].height = 720;

	vec3.at<double>(0, 0) = 0;
	vec3.at<double>(1, 0) = 0;
	vec3.at<double>(2, 0) = 0;
	cv::Rodrigues(vec3, Epi[0]);
	vec3.at<double>(0, 0) = 0;
	vec3.at<double>(1, 0) = 0;
	vec3.at<double>(2, 0) = 0;
	Epi[1] = -Epi[0] * vec3;

	vec3.at<double>(0, 0) = 0;
	vec3.at<double>(1, 0) = 0;
	vec3.at<double>(2, 0) = 0;
	cv::Rodrigues(vec3, Epi[2]);
	vec3.at<double>(0, 0) = 0;
	vec3.at<double>(1, 0) = 0;
	vec3.at<double>(2, 0) = 0;
	Epi[3] = -Epi[2] * vec3;

	vec3.at<double>(0, 0) = 0;
	vec3.at<double>(1, 0) = 0;
	vec3.at<double>(2, 0) = 0;
	cv::Rodrigues(vec3, Epi[4]);
	vec3.at<double>(0, 0) = 0;
	vec3.at<double>(1, 0) = 0;
	vec3.at<double>(2, 0) = 0;
	Epi[5] = -Epi[4] * vec3;

	txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Epi_FAR.txt");
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			txtfile >> Epi[0].at<double>(i, j);
		}
	}
	for (int i = 0; i < 3; i++)
		txtfile >> Epi[1].at<double>(i);
	txtfile.close();

	txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Epi_FFAR.txt");
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			txtfile >> Epi[2].at<double>(i, j);
		}
	}
	for (int i = 0; i < 3; i++)
		txtfile >> Epi[3].at<double>(i);
	txtfile.close();

	txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Epi_NEAR2.txt");
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			txtfile >> Epi[4].at<double>(i, j);
		}
	}
	for (int i = 0; i < 3; i++)
		txtfile >> Epi[5].at<double>(i);
	txtfile.close();

	cv::Mat mapx[4], mapy[4];
	//cv::initUndistortRectifyMap(Int[NEAR], Dist[NEAR], cv::Mat(), Int[NEAR], imgsize[NEAR], CV_32FC1, mapx[NEAR], mapy[NEAR]);
	//Dist[NEAR] = cv::Mat::zeros(5, 1, CV_64FC1);

	cv::initUndistortRectifyMap(Int_THROW, Dist[THROW], cv::Mat(), Int[THROW], imgsize[THROW], CV_32FC1, mapx[THROW], mapy[THROW]);
	Dist[THROW] = cv::Mat::zeros(5, 1, CV_64FC1);

	thetax = 101.5;
	thetay = 0;
	thetaz = 0;
	cx = 0;
	cy = 0;
	//cz = 512;
	cz = 485;

	Rvec[THROW] = cv::Mat::zeros(3, 1, CV_64FC1);
	Rvec[THROW].at<double>(0, 0) = thetax*CV_PI / 180;
	Rvec[THROW].at<double>(1, 0) = thetay*CV_PI / 180;
	Rvec[THROW].at<double>(2, 0) = thetaz*CV_PI / 180;
	cv::Rodrigues(Rvec[THROW], Rmat[THROW]);
	vec3.at<double>(0, 0) = cx;
	vec3.at<double>(1, 0) = cy;
	vec3.at<double>(2, 0) = cz;
	Tvec[THROW] = -Rmat[THROW] * vec3;

	vec3 = -Rmat[THROW].inv() * Tvec[THROW];
	cam_pos[THROW] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
	mat3 = Rmat[THROW].inv();
	cam_dir[THROW] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

	gndpoint.push_back(cv::Point3f(-R_cir, R_cir, 0));
	gndpoint.push_back(cv::Point3f(R_cir, R_cir, 0));
	gndpoint.push_back(cv::Point3f(0, 0, 0));
	gndpoint.push_back(cv::Point3f(0, 2 * R_cir, 0));
	gndpoint.push_back(cv::Point3f(0, D_hog, 0));
	gndpoint.push_back(cv::Point3f(-W, D_hog, 0));
	gndpoint.push_back(cv::Point3f(W, D_hog, 0));
	gndpoint.push_back(cv::Point3f(-W, R_cir, 0));
	gndpoint.push_back(cv::Point3f(W, R_cir, 0));

	cam_mode = THROW;
	cv::Mat A_ = cv::Mat::zeros(2 * gndpoint.size(), 8, CV_32FC1);
	cv::Mat B_ = cv::Mat::zeros(2 * gndpoint.size(), 1, CV_32FC1);
	for (i = 0; i < gndpoint.size(); i++){
		trwpoint.push_back(getimgbywld(cv::Point2f(gndpoint[i].x, gndpoint[i].y)));
		gndpoint[i].x -= cam_pos[THROW].x;
		gndpoint[i].y -= cam_pos[THROW].y;
		A_.at<float>(2 * i, 0) = gndpoint[i].x;
		A_.at<float>(2 * i, 1) = gndpoint[i].y;
		A_.at<float>(2 * i, 2) = 1;
		A_.at<float>(2 * i, 6) = -gndpoint[i].x*trwpoint[i].x;
		A_.at<float>(2 * i, 7) = -gndpoint[i].y*trwpoint[i].x;
		B_.at<float>(2 * i, 0) = trwpoint[i].x;
		A_.at<float>(2 * i + 1, 3) = gndpoint[i].x;
		A_.at<float>(2 * i + 1, 4) = gndpoint[i].y;
		A_.at<float>(2 * i + 1, 5) = 1;
		A_.at<float>(2 * i + 1, 6) = -gndpoint[i].x*trwpoint[i].y;
		A_.at<float>(2 * i + 1, 7) = -gndpoint[i].y*trwpoint[i].y;
		B_.at<float>(2 * i + 1, 0) = trwpoint[i].y;
	}
	X = (A_.t()*A_).inv()*(A_.t()*B_);
	Gnd2Trw = cv::Mat::zeros(3, 3, CV_32FC1);
	Gnd2Trw.at<float>(0, 0) = X.at<float>(0, 0); Gnd2Trw.at<float>(0, 1) = X.at<float>(1, 0); Gnd2Trw.at<float>(0, 2) = X.at<float>(2, 0);
	Gnd2Trw.at<float>(1, 0) = X.at<float>(3, 0); Gnd2Trw.at<float>(1, 1) = X.at<float>(4, 0); Gnd2Trw.at<float>(1, 2) = X.at<float>(5, 0);
	Gnd2Trw.at<float>(2, 0) = X.at<float>(6, 0); Gnd2Trw.at<float>(2, 1) = X.at<float>(7, 0); Gnd2Trw.at<float>(2, 2) = 1;
	Trw2Gnd = Gnd2Trw.inv();

	/********************************************************************************************************/

	/************************************** Rough Calibration[NEAR]******************************************/
	thetax0 = 115;

	thetax = thetax0;
	thetay = 0;
	thetaz = 0;
	cx = 0;
	cy = -2000;
	cz = 2070;

	Rvec[NEAR] = cv::Mat::zeros(3, 1, CV_64FC1);
	Rvec[NEAR].at<double>(0, 0) = thetax*CV_PI / 180;
	Rvec[NEAR].at<double>(1, 0) = thetay*CV_PI / 180;
	Rvec[NEAR].at<double>(2, 0) = thetaz*CV_PI / 180;
	cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

	vec3.at<double>(0, 0) = cx;
	vec3.at<double>(1, 0) = cy;
	vec3.at<double>(2, 0) = cz;
	Tvec[NEAR] = -Rmat[NEAR] * vec3;

	vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
	cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
	mat3 = Rmat[NEAR].inv();
	cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));
	Rvec_mean = cv::Mat::zeros(3, 1, CV_64FC1);
	Tvec_mean = cv::Mat::zeros(3, 1, CV_64FC1);

	/**************************************************************************************************/

	/************************************************* ROI ***************************************/
	for (i = 0; i < 4; i++)
		ROI_robot[i] = cv::Mat::ones(imgsize[i], CV_8UC1);

	ROI_robot[THROW](cv::Rect(0, 0, imgsize[THROW].width, imgsize[THROW].height*0.4)) = cv::Mat::zeros(imgsize[THROW].height*0.4, imgsize[THROW].width, CV_8UC1);
	ROI_robot[THROW](cv::Rect(0, imgsize[THROW].height*0.8, imgsize[THROW].width, imgsize[THROW].height*0.2)) = cv::Mat::zeros(imgsize[THROW].height*0.2, imgsize[THROW].width, CV_8UC1);
	/*for (i = 0; i < 4; i++)
		cv::imshow("ROI_robot"+patch::to_string(i), 255*ROI_robot[i]);
	cv::waitKey();*/
	/*****************************************************************************************************/

	/*********************************** Kalman Filter *************************************************/
	cv::Mat measure = cv::Mat(2, 1, CV_32FC1);
	cv::Mat predict;
	cv::Point2f ptMeasured;

	std::vector<cv::KalmanFilter> kalman(MAXNUM_STONE);
	std::vector<cv::Point2f> ptPredicted(MAXNUM_STONE);

	cv::KalmanFilter kalman_s;
	cv::Point2f ptPredicted_s;
	/**************************************************************************************************/

	/******************************************* Control ************************************************/
	txtfile.open("/home/nvidia/Robot_Data/Control.txt");
	txtfile >> dummy;	
	txtfile >> IsCommunicating;
	txtfile >> dummy;
	txtfile >> SKIPorTHROWER;
	txtfile >> dummy;
	txtfile >> IsChanging;
	txtfile >> dummy;
	txtfile >> IsRealtime;
	txtfile >> dummy;
	txtfile >> IsDisplaying;
	txtfile >> dummy;
	txtfile >> msec_waitKey;
	txtfile >> dummy;
	txtfile >> msec_blind;
	txtfile >> dummy;
	txtfile >> msec_timeout;
	txtfile >> dummy;
	txtfile >> Releasedist;
	txtfile >> dummy;
	txtfile >> angle_bias;
	txtfile >> dummy;
	txtfile >> msec_cameradelay;
	txtfile >> dummy;
	txtfile >> IsRecalibration;
	txtfile >> dummy;
	txtfile >> cam_mode_distant;
	txtfile.close();

	txtfile.open("/home/nvidia/Robot/Control.txt", std::fstream::out);
	txtfile << "IsCommunicating" << std::endl;
	txtfile << IsCommunicating << std::endl;
	txtfile << "SKIPorTHROWER" << std::endl;
	txtfile << SKIPorTHROWER << std::endl;
	txtfile << "IsChanging" << std::endl;
	txtfile << IsChanging << std::endl;
	txtfile << "IsRealtime" << std::endl;
	txtfile << IsRealtime << std::endl;
	txtfile << "IsDisplaying" << std::endl;
	txtfile << IsDisplaying << std::endl;
	txtfile << "msec_waitKey" << std::endl;
	txtfile << msec_waitKey << std::endl;
	txtfile << "msec_blind" << std::endl;
	txtfile << msec_blind << std::endl;
	txtfile << "msec_timeout" << std::endl;
	txtfile << msec_timeout << std::endl;
	txtfile << "Releasedist" << std::endl;
	txtfile << Releasedist << std::endl;
	txtfile << "angle_bias" << std::endl;
	txtfile << angle_bias << std::endl;
	txtfile << "msec_cameradelay" << std::endl;
	txtfile << msec_cameradelay << std::endl;
	txtfile << "IsRecalibration" << std::endl;
	txtfile << IsRecalibration << std::endl;
	txtfile << "cam_mode_distant" << std::endl;
	txtfile << cam_mode_distant << std::endl;
	txtfile.close();

	IsLogging = false;

	/**************************************************************************************************/

	/******************************************** Sheet *************************************************/
	int display_width = 1 + 2 * 75;
	double displayscale = display_width / 4800.0;
	cv::Point display_origin = cv::Point(displayscale*R_cir, display_width / 2);
	cv::Point display_pad = displayscale*cv::Point(2 * meter, 0);
	cv::Mat Sheet;
	cv::Mat Sheet_draw;
	cv::Point2f displaypoint;

	display_origin = display_origin + display_pad;
	Sheet = cv::Mat(display_width, 1280, CV_32FC3, cv::Scalar(1, 1, 1));
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
	/*cv::imshow("Sheet", Sheet);
	cv::waitKey();*/

	cv::Mat Frame = cv::Mat::zeros(720 + Sheet.rows, 1280, CV_32FC3);
	/***************************************************************************************************/
	
	/*******************************************Communication*****************************************/

	std::string SERVER_IP_str;
	txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/SERVER_IP.txt");
	txtfile >> SERVER_IP_str;
	txtfile.close();

	const char *SERVER_IP = SERVER_IP_str.c_str();

	Point pos[MAXNUM_STONE];
	float team[MAXNUM_STONE];
	Point robotPos, pos_thrown;
	float angle, angle_mean, hogDist, hogOffs;

	Point pos_sending[5];
	int msec_sending = 0;
	int msec_sending_[5] = {0,};
	float sweeper_x_sending;	
			
	sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (sock == -1) { return -1; }

	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr(SERVER_IP);
	servaddr.sin_port = htons(PORT_NUM);

	int re = -1;
	while (re == -1 && IsCommunicating) {
		re = connect(sock, (struct sockaddr *)&servaddr, sizeof(servaddr));
		if (re == -1) { 
			printf("\rConnect Fail!...");
			usleep(1000000);
		}
 	}

	pthread_t p_thread;
	if (IsCommunicating) {
		thr_id = pthread_create(&p_thread, NULL, RecvThreadPoint, NULL);
		if (thr_id < 0)	{
			perror("thread create error : ");
			std::terminate();
		}
	}

	/*************************************************************************************************/


	/*******************************************Camera Input*****************************************/

	std::ifstream gainvalue("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Gain.txt");
	gainvalue >> gain[NEAR][0];
	gainvalue >> gain[NEAR][1];
	gainvalue >> gain[NEAR][2];
	gainvalue >> gain[NEAR][3];

	gainvalue >> gain[FAR][0];
	gainvalue >> gain[FAR][1];
	gainvalue >> gain[FAR][2];
	gainvalue >> gain[FAR][3];

	gainvalue >> gain[THROW][0];
	gainvalue >> gain[THROW][1];
	gainvalue >> gain[THROW][2];
	gainvalue >> gain[THROW][3];

	gainvalue >> gain[FFAR][0];
	gainvalue >> gain[FFAR][1];
	gainvalue >> gain[FFAR][2];
	gainvalue >> gain[FFAR][3];
				

	txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Serialnum.txt");
	txtfile >> Serialnum[NEAR];
	txtfile >> Serialnum[FAR];
	txtfile >> Serialnum[THROW];
	txtfile >> Serialnum[FFAR];	
	txtfile.close();

	IsSnapshot[NEAR] = true;
	IsSnapshot[FAR] = false;
	IsSnapshot[FFAR] = false;
	IsSnapshot[THROW] = false;

	if (SKIPorTHROWER == SKIP && IsRealtime){
		if (!is_OpenCamera(NEAR, &Cams[NEAR], &Mems[NEAR]))
			return 0;
		if (is_OpenCamera(FAR, &Cams[FAR], &Mems[FAR])){
			if (is_CaptureVideo(Cams[FAR], IS_WAIT) != IS_SUCCESS)
				return 0;
			if (is_EnableEvent(Cams[FAR], IS_SET_EVENT_FRAME) != IS_SUCCESS)
				return 0;
		}
		else
			return 0;
		if (is_OpenCamera(FFAR, &Cams[FFAR], &Mems[FFAR])){
			if (is_CaptureVideo(Cams[FFAR], IS_WAIT) != IS_SUCCESS)
				return 0;
			if (is_EnableEvent(Cams[FFAR], IS_SET_EVENT_FRAME) != IS_SUCCESS)
				return 0;
		}
		else
			return 0;
	}	
	else if (SKIPorTHROWER == THROWER && IsRealtime){
		if (!is_OpenCamera(NEAR, &Cams[NEAR], &Mems[NEAR]))
			return 0;
		if (is_OpenCamera(THROW, &Cams[THROW], &Mems[THROW])){
			if (is_CaptureVideo(Cams[THROW], IS_WAIT) != IS_SUCCESS)
				return 0;
			if (is_EnableEvent(Cams[THROW], IS_SET_EVENT_FRAME) != IS_SUCCESS)
				return 0;
		}
		else
			return 0;
	}

	/********************************************** Video Input *********************************************/

	std::string videodir, videoname1, videoname2, videoname3;

	videodir = "/home/nvidia/";
	videoname1 = "NEAR_sw.avi";
	videoname2 = "FAR_sw.avi";
	videoname3 = "FFAR_sw.avi";
	
	if (!IsRealtime){
		if (SKIPorTHROWER == SKIP){
			video[NEAR].open(videodir + videoname1);
			if (video[NEAR].isOpened() == false){
				printf("near video not found\n");
				return 0;
			}
			video[FAR].open(videodir + videoname2);
			if (video[FAR].isOpened() == false){
				printf("far video not found\n");
				return 0;
			}
			video[FFAR].open(videodir + videoname3);
			if (video[FFAR].isOpened() == false){
				printf("ffar video not found\n");
				return 0;
			}
		}
		else if (SKIPorTHROWER == THROWER){
			video[NEAR].open(videodir + videoname1);
			if (video[NEAR].isOpened() == false){
				printf("near video not found\n");
				return 0;
			}
			video[THROW].open(videodir + videoname2);
			if (video[THROW].isOpened() == false){
				printf("throw video not found\n");
				return 0;
			}
		}
	}
	/*******************************************************************************************************/

	/*********************************** Initialization per shot *******************************************/
	if (SKIPorTHROWER == SKIP){
		cam_mode = NEAR;
		IsRunning = false;
		IsPositioning = false;
		IsRedetecting = false;
		if (IsCommunicating){
			IsDetecting = false;
			IsCalibration = true;
		}
		else {
			IsDetecting = true;
			IsCalibration = true;
		}
		thr_valid = 5;
		period_sweeper_communicating = 500;
	}
	else if (SKIPorTHROWER == THROWER){
		cam_mode = NEAR;
		IsRunning = false;
		if (!IsCommunicating)
			IsPositioning = true;
		else
			IsPositioning = false;
		IsDetecting = false;
		thr_valid = 3;
	}

	if (IsCommunicating)
		IsChanging = true;

	IsSending = false;	
	IsValid = 0;

	IsThrown = false;
	IsReached[0] = false;
	IsReached[1] = false;

	transition[0] = D_hog + 0.65*(D_hog_-D_hog);
	transition[1] = D_hog;

	IsWarning = 0;
	IsError = 0;
	framenum = -1;
	stonenum = 0;
	CollidedStonenum = 0;
	thrownstonenum = -1;
	msec_release = -1;
	msec_thrown = -1;
	Releasespeed = 2.5;
	angle_mean = 0;

	frameunit = 1;

	framenum_reach = 0;

	IsEnd = false;
	IsRealEnd = 0;

	IsLying = false;

	IsSendingAck = 0;

	sobelsize = 3;
	sobelfactor = 7;

	/**************************************************************************************************/

	//cam_mode = FAR;
	//is_ChangeNEAR();

	/*IsDetectingSweeper = true;
	IsCalibration = false;
	IsDetecting = false;
	IsEnd = true;*/

	if (IsDisplaying){
		cv::namedWindow("Frame");
		cv::moveWindow("Frame", 0, 0);
	}
	gettimeofday(&mytime0, NULL);
	while (++framenum < duration){

		capture:

		while (SKIPorTHROWER == THROWER){
			
			if (IsSendingAck){
				//SendPacketStoneInfoAck(sock, IsSendingAck - 1);
				if (IsSendingAck == 1)
					return 0;
				IsSendingAck = 0;
			}			
			if (IsPositioning)
				break;
			else if (IsRunning){
				cam_mode = THROW;
				thr_valid = 1;
				break;
			}
			else
				usleep(1000);
			printf("waiting for THROWER action...\r");

		}

		while (SKIPorTHROWER == SKIP){

			if (IsSendingAck){
				//SendPacketStoneInfoAck(sock, IsSendingAck - 1);
				if (IsSendingAck == 1)
					return 0;
				IsSendingAck = 0;
			}

			if (IsDetectingSweeper){
				cam_mode = FAR;
				break;
			}
			else if (IsCalibration)
				break;
			else if (IsDetecting)
				break;
			else if (IsRedetecting && !IsRecalibration && !IsDetectingSweeper){
				cam_mode = NEAR;
				if (IsCommunicating && IsSnapshot[NEAR]){
					if (!is_ChangeNEAR()){
						printf("error: unable to change near camera\n");
						return 0;
					}
				}
				break;
			}
			else if (IsThrown && !IsReached[0]){
				if (IsRecalibration){
					if (cam_mode_distant == NEAR)
						is_ChangeNEAR();
					cam_mode = cam_mode_distant;
				}
				else
					cam_mode = cam_mode_distant;	
				thr_valid = 1;
				break;
			}
			else if (IsThrown && IsReached[0] && !IsReached[1]){
				cam_mode = FAR;
				break;
			}
			else if (IsThrown && IsReached[0] && IsReached[1] && !IsEnd){
				cam_mode = NEAR;
				break;
			}
			else
				usleep(1000);
			printf("waiting for SKIP action...\r");

		}

		if (IsCommunicating && SKIPorTHROWER == THROWER){
			if (IsWarning > 3){
				hogOffs = 1;
				SendPacketRobotInfo(sock, angle, robotPos, hogDist, hogOffs);
				printf("send robot info error message\n");
				break;
			}
			if (IsLying){
				IsValid = thr_valid + 1;
				IsLying = false;
				goto control;
			}
		}
		printf("latency : %d\n", getmsec() - msec0);
		msec0 = getmsec();
		if (IsRealtime){		
			if (IsSnapshot[cam_mode] && is_FreezeVideo(Cams[cam_mode], IS_WAIT) != IS_SUCCESS){
				printf("error : camera snapshot failed");
				IsError = 1;
				return IsError;
			}
			else if (!IsSnapshot[cam_mode] && is_WaitEvent(Cams[cam_mode], IS_SET_EVENT_FRAME, 1000) != IS_SUCCESS){
				printf("error : camera capture failed");
				IsError = 1;
				return IsError;
			}
			cv::resize(cv::Mat(size_sensor[cam_mode], CV_8UC3, Mems[cam_mode]), img0[cam_mode], cv::Size(1280, 720));
		}
		else {
			if (SKIPorTHROWER == SKIP){
				video[NEAR].read(img0[NEAR]);
				video[FAR].read(img0[FAR]);
				video[FFAR].read(img0[FFAR]);
				if (img0[NEAR].empty() || img0[FAR].empty() || img0[FFAR].empty())
					break;
			}
			else if (SKIPorTHROWER == THROWER){
				video[NEAR].read(img0[NEAR]);
				video[THROW].read(img0[THROW]);
				if (img0[NEAR].empty() || img0[THROW].empty())
					break;
			}
		}
		
		if (IsRealtime)
			msec_frame[framenum] = getmsec();

		/*printf("framenum : %d\n", framenum);
		cv::imshow("img0", img0[cam_mode]);
		cv::waitKey(msec_waitKey);
		continue;*/

		if (cam_mode == THROW)
			cv::remap(img0[cam_mode], img, mapx[cam_mode], mapy[cam_mode], CV_INTER_LINEAR);
		else
			img = img0[cam_mode];

		img.convertTo(img, CV_32FC3, 1 / 255.0);

		img_draw = img.clone();
		imgRect = cv::Rect(0, 0, img.cols, img.rows);

		if (IsRecalibration && cam_mode == FAR){

			std::vector<cv::Point3f> wldpoint_pattern;
			wldpoint_pattern.push_back(cv::Point3f(0,D_hog,0));	
			wldpoint_pattern.push_back(cv::Point3f(0,D_hog_,0));
			wldpoint_pattern.push_back(cv::Point3f(R_cir,D_hog_ + D_hog - R_cir,0));
			wldpoint_pattern.push_back(cv::Point3f(-R_cir,D_hog_ + D_hog - R_cir,0));

			cv::namedWindow("img");
			cv::imshow("img", img_draw);
			nKey = cv::waitKey();
			if(nKey == 27){//esc
				imgpoint_pattern.clear();
				std::cout<<"Click 4 points"<<std::endl;
				cv::setMouseCallback("img",onMouseEvent,(void*)&img_draw);
				while(1){
					cv::imshow("img",img_draw);
					if(cv::waitKey(20))
						break;
				}
				cv::waitKey();

				cv::solvePnP(wldpoint_pattern, imgpoint_pattern, Int[FAR], Dist[FAR], Rvec[FAR], Tvec[FAR]);
				cv::Rodrigues(Rvec[FAR], Rmat[FAR]);

				Epi[0] = Rmat[FAR] * Rmat[NEAR].inv();
				Epi[1] = Tvec[FAR] - Epi[0] * Tvec[NEAR];

				txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Epi_FAR.txt");
				for (int i = 0; i < 3; i++){
					for (int j = 0; j < 3; j++){
						txtfile << Epi[0].at<double>(i, j) << std::endl;
					}
				}
				for (int i = 0; i < 3; i++)
					txtfile << Epi[1].at<double>(i) << std::endl;
				txtfile.close();
			}
		}
		else if (IsRecalibration && cam_mode == FFAR){

			std::vector<cv::Point3f> wldpoint_pattern;
			wldpoint_pattern.push_back(cv::Point3f(0,D_hog_,0));	
			wldpoint_pattern.push_back(cv::Point3f(0,D_hog_ + (D_hog - 2*R_cir),0));
			wldpoint_pattern.push_back(cv::Point3f(R_cir,D_hog_ + D_hog - R_cir,0));
			wldpoint_pattern.push_back(cv::Point3f(-R_cir,D_hog_ + D_hog - R_cir,0));

			cv::namedWindow("img");
			cv::imshow("img", img_draw);
			nKey = cv::waitKey();
			if(nKey == 27){//esc
				imgpoint_pattern.clear();
				std::cout<<"Click 4 points"<<std::endl;
				cv::setMouseCallback("img",onMouseEvent,(void*)&img_draw);
				while(1){
					cv::imshow("img",img_draw);
					if(cv::waitKey(20))
						break;
				}
				cv::waitKey();
				cv::solvePnP(wldpoint_pattern, imgpoint_pattern, Int[FFAR], Dist[FFAR], Rvec[FFAR], Tvec[FFAR]);
				cv::Rodrigues(Rvec[FFAR], Rmat[FFAR]);

				Epi[2] = Rmat[FFAR] * Rmat[NEAR].inv();
				Epi[3] = Tvec[FFAR] - Epi[2] * Tvec[NEAR];

				txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Epi_FFAR.txt");
				for (int i = 0; i < 3; i++){
					for (int j = 0; j < 3; j++){
						txtfile << Epi[2].at<double>(i, j) << std::endl;
					}
				}
				for (int i = 0; i < 3; i++)
					txtfile << Epi[3].at<double>(i) << std::endl;
				txtfile.close();
			}
		}
		else if (IsRecalibration && cam_mode == NEAR && !IsSnapshot[NEAR]){

			std::vector<cv::Point3f> wldpoint_pattern;
			wldpoint_pattern.push_back(cv::Point3f(0,0,0));	
			wldpoint_pattern.push_back(cv::Point3f(R_cir,R_cir,0));
			wldpoint_pattern.push_back(cv::Point3f(0,2*R_cir,0));
			wldpoint_pattern.push_back(cv::Point3f(-R_cir,R_cir,0));
			cv::namedWindow("img");
			cv::imshow("img", img_draw);
			nKey = cv::waitKey();
			if(nKey == 27){//esc
				imgpoint_pattern.clear();
				std::cout<<"Click 4 points"<<std::endl;
				cv::setMouseCallback("img",onMouseEvent,(void*)&img_draw);
				while(1){
					cv::imshow("img",img_draw);
					if(cv::waitKey(20))
						break;
				}
				cv::waitKey();
				cv::solvePnP(wldpoint_pattern, imgpoint_pattern, Int[NEAR], Dist[NEAR], Rvec[NEAR], Tvec[NEAR]);
				cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

				Epi[4] = Rmat[NEAR] * Rmat_temp.inv();
				Epi[5] = Tvec[NEAR] - Epi[4] * Tvec_temp;

				std::cout<<Epi[4]<<std::endl;

				txtfile.open("/home/nvidia/Robot_Data/" + patch::to_string(int(ROBOT_ID)) + "/Epi_NEAR2.txt");
				for (int i = 0; i < 3; i++){
					for (int j = 0; j < 3; j++){
						txtfile << Epi[4].at<double>(i, j) << std::endl;
					}
				}
				for (int i = 0; i < 3; i++)
					txtfile << Epi[5].at<double>(i) << std::endl;
				//txtfile.close();
			}
		}
	
		/*printf("framenum : %d\n", framenum);
		cv::imshow("img_draw", img_draw);
		cv::waitKey(0);
		continue;*/

		cv::split(img, bgr);
		cv::cvtColor(img, img_hsv, CV_BGR2HSV);
		cv::split(img_hsv, hsv);
		hsv[0] = hsv[0] / 360.0;

		cv::bitwise_and(bgr[2] < 0.45, bgr[0] < 0.45, bnw); //changeable
		cv::dilate(bnw, bnw, rec5, cv::Point(-1, -1), 2);

		/*cv::imshow("bnw", bnw);
		cv::imshow("hsv[1]", hsv[1]);
		cv::waitKey();*/

		if (IsDetectingSweeper){
			
			cv::Mat sweepercircle1, sweepercircle2, sweepercircle;

			cv::bitwise_and(bgr[0] > 0.5, bgr[1] + bgr[2] < 0.75, sweepercircle);
			
			hex[0] = getimgbywld(cv::Point2f(-W, D_hog), 0);
			hex[1] = getimgbywld(cv::Point2f(-W, D_hog), h_s);
			hex[2] = getimgbywld(cv::Point2f(-W, D_hog + 10*meter), h_s);
			hex[3] = getimgbywld(cv::Point2f(W, D_hog + 10*meter), h_s);
			hex[4] = getimgbywld(cv::Point2f(W, D_hog), h_s);
			hex[5] = getimgbywld(cv::Point2f(W, D_hog), 0);
			cv::Mat ROI_FAR_s = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
			for (int i = 0; i < ROI_FAR_s.cols; i++){
				for (int j = 0; j < ROI_FAR_s.rows; j++){
					if (ROI_robot[NEAR].at<uchar>(j, i) && cv::pointPolygonTest(hex, cv::Point2f(i, j), false) == 1)
						ROI_FAR_s.at<uchar>(j, i) = 1;
				}
			}

			//cv::imshow("ROI_FAR_s", 255*ROI_FAR_s);
			//cv::imshow("sweepercircle", sweepercircle);
			
			cv::erode(sweepercircle, sweepercircle, rec5, cv::Point(-1, -1), 2);
			cv::dilate(sweepercircle, sweepercircle, rec3, cv::Point(-1, -1), 2);

			cv::multiply(sweepercircle, ROI_FAR_s, sweepercircle);

			contours.clear();
			cv::findContours(sweepercircle.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			cv::RotatedRect circle_sweeper_;

			circle_sweeper_.center = cv::Point2f(0, 0);
			
			for (i = 0; i < contours.size(); i++){
				if (contours[i].size() < 20){
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}

				cv::RotatedRect circle_sweeper = fitEllipse(contours[i]);

				if (circle_sweeper.size.width * circle_sweeper.size.height < 50)
					continue;
				float sum2, sum, var, dist;
				cv::Point2f midpoint;
				sum = 0;
				sum2 = 0;
				midpoint = cv::Point2f(0, 0);
				for (j = 0; j < contours[i].size(); j++){
					pt1.x = contours[i][j].x - circle_sweeper.center.x;
					pt1.y = contours[i][j].y - circle_sweeper.center.y;
					dist = pt1.dot(pt1);
					sum += sqrt(dist);
					sum2 += dist;
				}
				var = sum2 / contours[i].size() - (sum / contours[i].size()) * (sum / contours[i].size());

				for (j = 0; j < contours[i].size(); j++){
					midpoint.x += contours[i][j].x;
					midpoint.y += contours[i][j].y;
				}
				midpoint = midpoint * (1.0 / contours[i].size());

				pt1.x = midpoint.x - circle_sweeper.center.x;
				pt1.y = midpoint.y - circle_sweeper.center.y;
				if (sqrt(pt1.dot(pt1)) > 10){
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}
				
				if (var > 10){
					contours.erase(contours.begin() + i);
					i--;
					continue;
				}

				circle_sweeper_ = circle_sweeper;
				break;
			}

			cv::circle(img_draw, circle_sweeper_.center, 2, cv::Scalar(0, 0, 1), -1);

			float h_c = 450;

			if (circle_sweeper_.center.x != 0 && circle_sweeper_.center.y != 0){
				cv::Point2f wld_c = getwldbyimg(circle_sweeper_.center, h_c);
				//printf("wld_c.x : %f\n", wld_c.x);
				//printf("wld_c.y : %f\n", wld_c.y);
			
				sweeper_x_sending = 0.001*wld_c.x;

				printf("Sweeper X : %f\n", sweeper_x_sending);

				robotPos.x = 1234;
				robotPos.y = sweeper_x_sending;
				hogOffs = 3;
				SendPacketRobotInfo(sock, angle, robotPos, hogDist, hogOffs);
				printf("Send Sweeper x\n");
				usleep(500000);

				//cv::imshow("img_draw", img_draw);
				//cv::waitKey();
				
			}

			IsDetectingSweeper = false;

			goto control;
		}		


		if ((IsCalibration || IsPositioning || (IsRunning && SKIPorTHROWER == THROWER)) && !IsReached[0]){
			if (cam_mode == NEAR){
			
				cv::cvtColor(img, gray, CV_BGR2GRAY);
				cv::Sobel(gray, edgex, -1, 1, 0, sobelsize, sobelfactor);
				cv::Sobel(gray, edgey, -1, 0, 1, sobelsize, sobelfactor);
				edgex = cv::abs(edgex);
				edgey = cv::abs(edgey);
				cv::bitwise_or(edgey > 1.5, edgex + edgey > 3, edge); //changeable

				/*cv::line(img_draw, getimgbywld(cv::Point2f(-W, D_hog)), getimgbywld(cv::Point2f(W, D_hog)), cv::Scalar(0, 0, 1), 2);
				cv::line(img_draw, getimgbywld(cv::Point2f(-W, D_hog)), getimgbywld(cv::Point2f(-W, 0)), cv::Scalar(0, 0, 1), 2);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, D_hog)), getimgbywld(cv::Point2f(W, 0)), cv::Scalar(0, 0, 1), 2);
				cv::imshow("img_draw", img_draw);*/
				//cv::imshow("edge", edge);
				//cv::waitKey();

				lines.clear();
				cv::HoughLinesP(edge, lines, 1, CV_PI / 180, 200, 300, 200); //changeable				
				if (!lines.size()){
					printf("error : line search failed\n");

					thetax = thetax0;
					thetay = 0;
					thetaz = 0;
					cx = 0;
					cy = -2000;
					cz = 2070;

					Rvec[NEAR] = cv::Mat::zeros(3, 1, CV_64FC1);
					Rvec[NEAR].at<double>(0, 0) = thetax*CV_PI / 180;
					Rvec[NEAR].at<double>(1, 0) = thetay*CV_PI / 180;
					Rvec[NEAR].at<double>(2, 0) = thetaz*CV_PI / 180;
					cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

					vec3.at<double>(0, 0) = cx;
					vec3.at<double>(1, 0) = cy;
					vec3.at<double>(2, 0) = cz;
					Tvec[NEAR] = -Rmat[NEAR] * vec3;

					vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
					cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[NEAR].inv();
					cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

					IsValid = 0;
					IsWarning++;
					goto capture;					
				}

				//for (i = 0; i < lines.size(); i++)
				//	cv::line(img_draw, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 1), 2);
				//cv::imshow("img_draw", img_draw);
				//cv::waitKey();

				lines_l.clear();
				lines_r.clear();
				lines_h.clear();
				for (i = 0; i < lines.size(); i++){
					pt1 = getwldbyimg(cv::Point2f(lines[i][0], lines[i][1]));
					pt2 = getwldbyimg(cv::Point2f(lines[i][2], lines[i][3]));
					pt3 = pt2 - pt1;
					pt4 = pt3 * (1.0 / sqrt(pt3.x*pt3.x + pt3.y*pt3.y));
					lx = pt4.x;
					ly = pt4.y;
					pt4 = pt1 - (pt1.x*pt4.x + pt1.y*pt4.y)*pt4;
					ld = sqrt(pt4.x*pt4.x + pt4.y*pt4.y);
					if (fabs(ly) > lr && fabs(ld - W) < 0.25*W && ld > 0.25*W && pt4.x < 0)
						lines_l.push_back(lines[i]);
					else if (fabs(ly) > lr && fabs(ld - W) < 0.25*W && ld > 0.25*W && pt4.x > 0)
						lines_r.push_back(lines[i]);
					else if (fabs(lx) > lr*0.5 && fabs(ld - D_hog) < R_cir && abs(pt1.x + pt2.x) < W)
						lines_h.push_back(lines[i]);
				}

				if (lines_l.size() == 0 || lines_r.size() == 0 || lines_h.size() == 0){
					printf("error : line classification failed\n");
					printf("l: %d, r: %d, h: %d\n", (int)lines_l.size(), (int)lines_r.size(), (int)lines_h.size());

					thetax = thetax0;
					thetay = 0;
					thetaz = 0;
					cx = 0;
					cy = -2000;
					cz = 2070;

					Rvec[NEAR] = cv::Mat::zeros(3, 1, CV_64FC1);
					Rvec[NEAR].at<double>(0, 0) = thetax*CV_PI / 180;
					Rvec[NEAR].at<double>(1, 0) = thetay*CV_PI / 180;
					Rvec[NEAR].at<double>(2, 0) = thetaz*CV_PI / 180;
					cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

					vec3.at<double>(0, 0) = cx;
					vec3.at<double>(1, 0) = cy;
					vec3.at<double>(2, 0) = cz;
					Tvec[NEAR] = -Rmat[NEAR] * vec3;

					vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
					cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[NEAR].inv();
					cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

					IsValid = 0;
					IsWarning++;
					goto capture;
				}
	
				/*for (i = 0; i < lines_l.size(); i++)
					cv::line(img_draw, cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Scalar(0.5, 0.5, 0.5), 2);
				for (i = 0; i < lines_r.size(); i++)
					cv::line(img_draw, cv::Point2f(lines_r[i][0], lines_r[i][1]), cv::Point2f(lines_r[i][2], lines_r[i][3]), cv::Scalar(0, 0, 0), 2);
				for (i = 0; i < lines_h.size(); i++)
					cv::line(img_draw, cv::Point2f(lines_h[i][0], lines_h[i][1]), cv::Point2f(lines_h[i][2], lines_h[i][3]), cv::Scalar(0, 0, 1), 2);
				cv::imshow("img_draw", img_draw);
				//cv::imshow("edge", edge);
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
			
				points.release();

				resize_ROI = 0.2;
				ROI = cv::Mat::zeros(resize_ROI*img.rows, resize_ROI*img.cols, CV_8UC1);
				for (i = 0; i < ROI.cols; i++){
					for (j = 0; j < ROI.rows; j++){
						pt = cv::Point(i / resize_ROI, j / resize_ROI);
						if (ROI_robot[NEAR].at<uchar>(pt) && cv::pointPolygonTest(amin_quad, pt, false) > 0){
							ROI.at<uchar>(j, i) = 1;
							if (bnw.at<uchar>(pt) == 0){
								point2.at<float>(0, 0) = hsv[0].at<float>(pt);
								point2.at<float>(0, 1) = hsv[1].at<float>(pt);
								points.push_back(point2);
							}
						}
					}
				}
				cv::resize(ROI, ROI, img.size(), 0, 0, cv::INTER_NEAREST);

				num_region = 3;
				cv::kmeans(points, num_region, region, cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 0.1), 10, cv::KMEANS_RANDOM_CENTERS, props);
				points.release();
	
				whitenum = 0;
				t2 = props.at<float>(0, 1);
				for (i = 1; i < props.rows; i++){
					if (props.at<float>(i, 1) < t2){
						t2 = props.at<float>(i, 1);
						whitenum = i;
					}
				}		

				/************************ ellipse fitting ***********************/

				segmentation.clear();
				for (i = 0; i < num_region; i++)
					segmentation.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));

				for (i = 0; i < img.rows - 2; i++){
					for (j = 0; j < img.cols - 2; j++){
						if (ROI.at<uchar>(i, j)){
							amin = 0;
							minVal = (props.at<float>(0, 0) - hsv[0].at<float>(i, j))*(props.at<float>(0, 0) - hsv[0].at<float>(i, j)) + (props.at<float>(0, 1) - hsv[1].at<float>(i, j))*(props.at<float>(0, 1) - hsv[1].at<float>(i, j));
							for (x = 1; x < props.rows; x++){
								t1 = (props.at<float>(x, 0) - hsv[0].at<float>(i, j))*(props.at<float>(x, 0) - hsv[0].at<float>(i, j)) + (props.at<float>(x, 1) - hsv[1].at<float>(i, j))*(props.at<float>(x, 1) - hsv[1].at<float>(i, j));
								if (minVal > t1){
									minVal = t1;
									amin = x;
								}
							}
							if (amin != whitenum)
								segmentation[amin].at<uchar>(i, j) = 255;
						}
					}
				}

				/*for (i = 0; i < num_region; i++)
					cv::imshow("segmentation" + patch::to_string(i), segmentation[i]);
				printf("%f, %f, %f\n", props.at<float>(0, 1), props.at<float>(1, 1), props.at<float>(2, 1));
				cv::waitKey();*/
				

				contours.clear();
				for (k = 0; k < num_region; k++){
					if (props.at<float>(k, 1) > 0.3 && k != whitenum){
						cv::findContours(segmentation[k], contours_, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
				
						for (i = 0; i < contours_.size(); i++){
							if (contours_[i].size() < 10)
								contours_[i].clear();
							for (j = 0; j < contours_[i].size(); j++){
								if (contours_[i][j].x < 10 || contours_[i][j].x > img.cols - 1 - 10 || contours_[i][j].y < 10 || contours_[i][j].y > img.rows - 1 - 10){
									contours_[i].erase(contours_[i].begin() + j);
									j--;
								}
								else if (cv::mean(ROI_robot[NEAR](cv::Rect(contours_[i][j].x - 5, contours_[i][j].y - 5, 10, 10)))[0] < 1){
									contours_[i].erase(contours_[i].begin() + j);
									j--;
								}
								else if (cv::mean(ROI(cv::Rect(contours_[i][j].x - 5, contours_[i][j].y - 5, 10, 10)))[0] < 1){
									contours_[i].erase(contours_[i].begin() + j);
									j--;
								}
							}
						}

						for (i = 0; i < contours_.size(); i++){
							for (j = 1; j < contours_[i].size(); j++){
								if (fabs(contours_[i][j].x - contours_[i][j - 1].x) + fabs(contours_[i][j].y - contours_[i][j - 1].y) > 2){
									contour.assign(contours_[i].begin() + j, contours_[i].end());
									contours_.push_back(contour);
									contour.clear();
									contours_[i].erase(contours_[i].begin() + j, contours_[i].end());
									break;
								}
							}
						}
						for (i = 0; i < contours_.size(); i++){
							for (j = i + 1; j < contours_.size() && contours_[i].size() > 0; j++){
								if (contours_[j].size() == 0)
									continue;
								err = contours_[i].front() - contours_[j].back();
								if (fabs(err.x) + fabs(err.y) <= 2){
									contours_[j].insert(contours_[j].end(), contours_[i].begin(), contours_[i].end());
									contours_[i].clear();
								}
							}
						}
						contours.insert(contours.end(), contours_.begin(), contours_.end());
					}
					segmentation[k].release();
				}

				/*for (i = 0; i < contours.size(); i++){
					if (contours[i].size() > 0)
						cv::drawContours(img_draw, contours, i, cv::Scalar(0, 0, 0), 2);
				}
				cv::imshow("img_draw", img_draw);
				cv::waitKey();*/

				ellipseareas = cv::Mat::zeros(contours.size(), 1, CV_32FC1);
				for (i = 0; i < contours.size(); i++){
					if (contours[i].size() < 10){
						contours[i].clear();
						continue;
					}
					else {
						cv::convexHull(contours[i], hull);
						contours[i] = hull;
					}
					points.release();
					for (j = 0; j < contours[i].size(); j++){
						if (bnw.at<uchar>(contours[i][j]) == 0){
							point5.at<float>(0, 0) = contours[i][j].x * contours[i][j].x;
							point5.at<float>(0, 1) = contours[i][j].y * contours[i][j].y;
							point5.at<float>(0, 2) = contours[i][j].x * contours[i][j].y;
							point5.at<float>(0, 3) = contours[i][j].x;
							point5.at<float>(0, 4) = contours[i][j].y;
							points.push_back(point5);
						}
					}
					if (points.rows < 10){	
						contours[i].clear();
						continue;	
					}			
						
					Q = cv::Mat::ones(points.rows, 1, CV_32FC1);
					Q = (points.t()*points).inv()*(points.t()*Q);

					points = points*Q;
					error = 0;
					for (j = 0; j < points.rows; j++)
						error += fabs(points.at<float>(j, 0) - 1);
					error /= (double)contours.size();

					if (error > 0.01)
						contours[i].clear();
					else
						ellipseareas.at<float>(i, 0) = CV_PI * getellipseprops(Q)[0] * getellipseprops(Q)[1];
				}
				cv::sortIdx(ellipseareas, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

				ellipse[0] = sortIdx1.at<int>(0, 0);
				ellipse[1] = sortIdx1.at<int>(1, 0);

				/*for (i = 0; i < 2; i++)
					cv::drawContours(img_draw, contours, sortIdx1.at<int>(i, 0), cv::Scalar(0, 0, 0), 2);
				cv::imshow("img_draw", img_draw);
				cv::waitKey();*/
			
				if (contours[ellipse[0]].size() == 0){
					printf("Circle Detection Fail\n");
					
					thetax = thetax0;
					thetay = 0;
					thetaz = 0;
					cx = 0;
					cy = -2000;
					cz = 2070;

					Rvec[NEAR] = cv::Mat::zeros(3, 1, CV_64FC1);
					Rvec[NEAR].at<double>(0, 0) = thetax*CV_PI / 180;
					Rvec[NEAR].at<double>(1, 0) = thetay*CV_PI / 180;
					Rvec[NEAR].at<double>(2, 0) = thetaz*CV_PI / 180;
					cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

					vec3.at<double>(0, 0) = cx;
					vec3.at<double>(1, 0) = cy;
					vec3.at<double>(2, 0) = cz;
					Tvec[NEAR] = -Rmat[NEAR] * vec3;

					vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
					cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[NEAR].inv();
					cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));


					IsValid = 0;
					IsWarning++;
					goto capture;
				}
				else if (contours[ellipse[1]].size() == 0)
					IsOneEllipse = true;
				else
					IsOneEllipse = false;

				IsOneEllipse = true;

				cv::convexHull(contours[ellipse[0]], hull);
				points.release();
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
			
				/*for (i = 0; i < hull.size() - 1; i++)
					cv::line(img_draw, hull[i], hull[i + 1], cv::Scalar(0, 0, 0), 1);
				cv::line(img_draw, hull[i], hull[0], cv::Scalar(0, 0, 0), 1);
				cv::imshow("img_draw", img_draw);
				cv::waitKey();*/

				if (!IsOneEllipse){
					cv::convexHull(contours[ellipse[1]], hull);
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
					Q_ = cv::Mat::ones(points.rows, 1, CV_32FC1);
					Q_ = (points.t()*points).inv()*(points.t()*Q_);
					points.release();
				
					intersections = getintersection_ellipse(Q_, cv::Point2f(0, 0), gettangentialpoint_ellipse(Q, cv::Point2f(0, 0))[0]);
					if ( intersections[2].x > 0 ){
						printf("Circle Classification Fail\n");
						IsValid = 0;
						IsWarning++;
						goto capture;
					}
				}

				/****************************************************************/

				for (i = 0; i < lines_l.size(); i++){
					for (j = 0; j < lines_r.size(); j++){
						for (k = 0; k < lines_h.size(); k++){

							van = getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]));	
							vertex[0] = getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));	
							vertex[1] = getintersection(cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));						
							
							pnts = gettangentialpoint_ellipse(Q, van);
							if (pnts[0].x > pnts[1].x){
								e2 = pnts[0];
								e1 = pnts[1];
							}
							else {
								e2 = pnts[1];
									e1 = pnts[0];
							}

							van_ = getintersection(e1, e2, cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3]));							

							pnts = gettangentialpoint_ellipse(Q, van_);
							if (pnts[0].y > pnts[1].y){
								e4 = pnts[0];
								e3 = pnts[1];
							}
							else {
								e4 = pnts[1];
								e3 = pnts[0];
							}
							
							if (!IsOneEllipse){
								//mp
							}
							else {
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
							}

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

							error = 0;
							for (ii = 0; ii < 12; ii++){
								wp.x = (Img2Wld.at<float>(0, 0)*imgpoint[ii].x + Img2Wld.at<float>(0, 1)*imgpoint[ii].y + Img2Wld.at<float>(0, 2)) / (Img2Wld.at<float>(2, 0)*imgpoint[ii].x + Img2Wld.at<float>(2, 1)*imgpoint[ii].y + Img2Wld.at<float>(2, 2));
								wp.y = (Img2Wld.at<float>(1, 0)*imgpoint[ii].x + Img2Wld.at<float>(1, 1)*imgpoint[ii].y + Img2Wld.at<float>(1, 2)) / (Img2Wld.at<float>(2, 0)*imgpoint[ii].x + Img2Wld.at<float>(2, 1)*imgpoint[ii].y + Img2Wld.at<float>(2, 2));
								err = cv::Point2f(wldpoint[ii].x - wp.x, wldpoint[ii].y - wp.y);
								error += weight[ii] * sqrt(err.x*err.x + err.y*err.y);
							}

							/*cv::solvePnP(wldpoint, imgpoint, Int[NEAR], Dist[NEAR], Rvec[NEAR], Tvec[NEAR]);
							cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);
							error = 0;
							for (ii = 0; ii < 12; ii++){
								err = cv::Point2f(wldpoint[ii].x, wldpoint[ii].y) - getwldbyimg(imgpoint[ii]);
								error += weight[ii] * sqrt(err.x*err.x + err.y*err.y);
							}
							
							vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];*/							

							if (minVal > error || (i == 0 && j == 0 && k == 0)){
								sidelines[0] = lines_l[i];
								sidelines[1] = lines_r[j];
								hogline = lines_h[k];
								minVal = error;
							}
						}
					}
				}
				if (minVal > 10000){
					printf("error : calibration is failed\n");
					printf("Calibration Error : %f\n", minVal);

					thetax = thetax0;
					thetay = 0;
					thetaz = 0;
					cx = 0;
					cy = -2000;
					cz = 2070;

					Rvec[NEAR] = cv::Mat::zeros(3, 1, CV_64FC1);
					Rvec[NEAR].at<double>(0, 0) = thetax*CV_PI / 180;
					Rvec[NEAR].at<double>(1, 0) = thetay*CV_PI / 180;
					Rvec[NEAR].at<double>(2, 0) = thetaz*CV_PI / 180;
					cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

					vec3.at<double>(0, 0) = cx;
					vec3.at<double>(1, 0) = cy;
					vec3.at<double>(2, 0) = cz;
					Tvec[NEAR] = -Rmat[NEAR] * vec3;

					vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
					cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[NEAR].inv();
					cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));
	
					IsValid = 0;
					IsWarning++;					
					goto capture;
				}

				van = getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]));	
				vertex[0] = getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]));	
				vertex[1] = getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]));

				pnts = gettangentialpoint_ellipse(Q, van);
				if (pnts[0].x > pnts[1].x){
					e2 = pnts[0];
					e1 = pnts[1];
				}
				else {
					e2 = pnts[1];
					e1 = pnts[0];
				}
	
				van_ = getintersection(e1, e2, cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]));						

				pnts = gettangentialpoint_ellipse(Q, van_);
				if (pnts[0].y > pnts[1].y){
					e4 = pnts[0];
					e3 = pnts[1];
				}
				else {
					e4 = pnts[1];
					e3 = pnts[0];
				}

				if (!IsOneEllipse){
					//mp
				}
				else {
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
				}

				cv::solvePnP(wldpoint, imgpoint, Int[NEAR], Dist[NEAR], Rvec[NEAR], Tvec[NEAR]);
				cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);
				vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
				cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
				mat3 = Rmat[NEAR].inv();
				cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));
				
				//printf("cam_pose : %f, %f, cam_dir : %f, %f\n", cam_pos[NEAR].x, cam_pos[NEAR].y, cam_dir[cam_mode].x, cam_dir[cam_mode].y);
				if (fabs(cam_pos[NEAR].x) > W || cam_pos[NEAR].y > 0 || cam_pos[NEAR].y < -5000 || cam_dir[cam_mode].y <= 0 || fabs((180/CV_PI)*atan(cam_dir[cam_mode].x / cam_dir[cam_mode].y)) > 15 || cam_pos[NEAR].x != cam_pos[NEAR].x){
					printf("error : pose is weird\n");

					thetax = thetax0;
					thetay = 0;
					thetaz = 0;
					cx = 0;
					cy = -2000;
					cz = 2070;

					Rvec[NEAR] = cv::Mat::zeros(3, 1, CV_64FC1);
					Rvec[NEAR].at<double>(0, 0) = thetax*CV_PI / 180;
					Rvec[NEAR].at<double>(1, 0) = thetay*CV_PI / 180;
					Rvec[NEAR].at<double>(2, 0) = thetaz*CV_PI / 180;
					cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);

					vec3.at<double>(0, 0) = cx;
					vec3.at<double>(1, 0) = cy;
					vec3.at<double>(2, 0) = cz;
					Tvec[NEAR] = -Rmat[NEAR] * vec3;

					vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
					cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[NEAR].inv();
					cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));
	
					IsValid = 0;
					IsWarning++;					
					goto capture;
				} 

				if (IsDisplaying){
					/*cv::line(img_draw, getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0.5, 0.5, 0.5), 3);
					cv::line(img_draw, getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0, 0, 0), 3);
					cv::line(img_draw, getintersection(cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0, 0, 1), 3);
					cv::line(img_draw, e1, e2, cv::Scalar(0, 0, 0), 2);
					cv::line(img_draw, e3, e4, cv::Scalar(0, 0, 0), 2);
					cv::Mat img_draw_resize;
					cv::resize(img_draw, img_draw_resize, cv::Size(), 0.5, 0.5);
					cv::imshow("img_draw_resize", img_draw_resize);
					cv::waitKey(msec_waitKey);*/
				}

				if (IsPositioning){
					IsValid++;

					if (cam_dir[cam_mode].y == 0 && cam_dir[cam_mode].x > 0)
						angle = -90;
					else if (cam_dir[cam_mode].y == 0 && cam_dir[cam_mode].x < 0)
						angle = 90;
					else
						angle = -(180/CV_PI)*atan(cam_dir[cam_mode].x / cam_dir[cam_mode].y);

					if (angle != angle){
						IsValid = 0;
						goto capture;
					}

					if (IsValid){
						angle_mean = angle_mean*((IsValid - 1.0) / IsValid) + angle*(1.0 / IsValid);
						robotPos.x = robotPos.x*((IsValid - 1.0) / IsValid) + cam_pos[cam_mode].x*(1.0 / IsValid);
						robotPos.y = robotPos.y*((IsValid - 1.0) / IsValid) + cam_pos[cam_mode].y*(1.0 / IsValid);
						hogDist = D_hog - cam_pos[cam_mode].y;
						if (IsPositioning && IsValid == thr_valid){
							hogDist0 = hogDist;
							printf("(x y theta) : (%f %f %f)\n", robotPos.x, robotPos.y, angle_mean);
							
						}
						else if (IsRunning){
							msec_release = getmsec_release(hogDist);
							printf("hogDist : %f\n", hogDist);
						}
					}

				}
				else {
					IsValid++;
					Rvec_mean = Rvec_mean*((IsValid - 1.0) / IsValid) + Rvec[NEAR]*(1.0 / IsValid);
					Tvec_mean = Tvec_mean*((IsValid - 1.0) / IsValid) + Tvec[NEAR]*(1.0 / IsValid);
					
									
				}

				if (IsRealtime && IsValid < thr_valid && SKIPorTHROWER == SKIP){
					framenum++;
					goto capture;
				}
				else if (IsCalibration){
					Rvec[NEAR] = Rvec_mean;
					Tvec[NEAR] = Tvec_mean;
					cv::Rodrigues(Rvec[NEAR], Rmat[NEAR]);
					vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
					cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[NEAR].inv();
					cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

					quad[0] = getimgbywld(cv::Point2f(-W + 0.5*d, -pad_back), 0);
					quad[1] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[1] + pad_hog), 0);
					quad[2] = getimgbywld(cv::Point2f(W - 0.5*d, transition[1] + pad_hog), 0);
					quad[3] = getimgbywld(cv::Point2f(W - 0.5*d, -pad_back), 0);
	
					ROI = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
					for (i = 0; i < ROI.cols; i++){
						for (j = 0; j < ROI.rows; j++){
							if (ROI_robot[NEAR].at<uchar>(j, i) && cv::pointPolygonTest(quad, cv::Point2f(i, j), false) == 1)
								ROI.at<uchar>(j, i) = 1;
						}
					}
					//cv::imshow("ROI", 255*ROI);
					//cv::waitKey();
					goto control;
				}
			}
			else if (cam_mode == THROW){

				lines_h.clear();
				lines_l.clear();
				lines_r.clear();

				if (IsCommunicating)
					msec_running = getmsec() - msec_startrun;
				else
					msec_running = 0;
					
				if (msec_running > msec_timeout + msec_blind)
					IsValid++;
				else {
				
					cv::cvtColor(img, gray, CV_BGR2GRAY);
					cv::Sobel(gray, edgex, -1, 1, 0);
					cv::Sobel(gray, edgey, -1, 0, 1);
					edgex = cv::abs(edgex);
					//edgey = cv::abs(edgey);
					cv::bitwise_or(edgex > 0.25, edgey > 0.25, edge); //changeable
					//cv::erode(edge, edge, rec3);

					cv::multiply(edge, ROI_robot[THROW], edge);

					cv::HoughLinesP(edge, lines, 1, CV_PI / 180, 150, 150, 50); //changeable
					if (!lines.size()){
						printf("warning : line search failed\n");
						IsWarning++;
						printf("\n\n\n\n");
						goto control;
					}

					/*for (i = 0; i < lines.size(); i++)
						cv::line(img_draw, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 1), 2);
					cv::imshow("img_draw", img_draw);
					cv::imshow("edge", edge);
					cv::waitKey();*/

					for (i = 0; i < lines.size(); i++){
						pt1 = getgndbytrw(cv::Point2f(lines[i][0], lines[i][1]));
						pt2 = getgndbytrw(cv::Point2f(lines[i][2], lines[i][3]));
						pt3 = pt1 - pt2;
						wp = pt1 - pt3*(pt3.dot(pt1) / pt3.dot(pt3));
						if (atan(fabs(wp.y / wp.x)) < (CV_PI / 180) * 20 && sqrt(wp.dot(wp)) < 1.5 * W && sqrt(wp.dot(wp)) > 0.5*W){
							if (wp.x < 0)
								lines_l.push_back(lines[i]);
							else
								lines_r.push_back(lines[i]);
						}
						if (atan(fabs(wp.y / wp.x)) > (CV_PI / 180) * 20){
							lines_h.push_back(lines[i]);
						}
					}
				
					if ((lines_l.size() == 0 && lines_r.size() == 0) || lines_h.size() == 0){
						printf("error : line classification failed\n");
						printf("l: %d, r: %d, h: %d\n", (int)lines_l.size(), (int)lines_r.size(), (int)lines_h.size());
						IsWarning++;
						printf("\n\n\n\n");
						goto control;
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

					if (lines_l.size()*lines_r.size() == 0)
						IsOneSideline = true;
					else
						IsOneSideline = false;

					if (lines_l.size() != 0 && lines_r.size() == 0){
						for (i = 0; i < lines_l.size(); i++){
							pt1 = getgndbytrw(cv::Point2f(lines_l[i][0], lines_l[i][1]));
							pt2 = getgndbytrw(cv::Point2f(lines_l[i][2], lines_l[i][3]));
							wp = pt1 - pt2;
							wp = cv::Point2f(wp.y, -wp.x);
							wp = wp * (1 / sqrt(wp.dot(wp)));
							if (wp.x < 0)
								wp = -wp;
							pt3 = gettrwbygnd(pt1 + 2*W*wp);
							pt4 = gettrwbygnd(pt2 + 2*W*wp);
							lines_r.push_back(cv::Vec4f(pt3.x, pt3.y, pt4.x, pt4.y));
						}
					}
					else if (lines_r.size() != 0 && lines_l.size() == 0){
						for (i = 0; i < lines_r.size(); i++){
							pt1 = getgndbytrw(cv::Point2f(lines_r[i][0], lines_r[i][1]));
							pt2 = getgndbytrw(cv::Point2f(lines_r[i][2], lines_r[i][3]));
							wp = pt1 - pt2;
							wp = cv::Point2f(wp.y, -wp.x);
							wp = wp * (1 / sqrt(wp.dot(wp)));
							if (wp.x > 0)
								wp = -wp;
							pt3 = gettrwbygnd(pt1 + 2*W*wp);
							pt4 = gettrwbygnd(pt2 + 2*W*wp);
							lines_l.push_back(cv::Vec4f(pt3.x, pt3.y, pt4.x, pt4.y));
						}
					}

					if (!IsOneSideline){
						for (i = 0; i < lines_l.size(); i++){
							for (j = 0; j < lines_r.size(); j++){
								for (k = 0; k < lines_h.size(); k++){

									pt1 = getgndbytrw(getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3])));
									pt2 = getgndbytrw(getintersection(cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3])));
									pt3 = getgndbytrw(cv::Point2f(lines_l[i][0], lines_l[i][1])) - getgndbytrw(cv::Point2f(lines_l[i][2], lines_l[i][3]));
									pt3 = pt3*(1 / sqrt(pt3.dot(pt3)));
									pt4 = pt1 + 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt2;
									t1 = pt4.dot(pt4);
									pt4 = pt1 - 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt2;
									if (t1 > pt4.dot(pt4))
										t1 = pt4.dot(pt4);
									pt3 = getgndbytrw(cv::Point2f(lines_r[j][0], lines_r[j][1])) - getgndbytrw(cv::Point2f(lines_r[j][2], lines_r[j][3]));
									pt3 = pt3*(1 / sqrt(pt3.dot(pt3)));
									pt4 = pt2 + 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt1;
									t2 = pt4.dot(pt4);
									pt4 = pt2 - 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt1;
									if (t2 > pt4.dot(pt4))
										t2 = pt4.dot(pt4);
									error = sqrt(t1) + sqrt(t2) + fabs(sqrt((pt1 - pt2).dot(pt1 - pt2)) - 2 * W);
									tri[0] = pt1;
									tri[1] = pt2;
									tri[2] = cv::Point2f(0, 0);
												
									if ((minVal > error && cv::pointPolygonTest(tri, getgndbytrw(0.5*cv::Point2f(lines_h[k][0] + lines_h[k][2], lines_h[k][1] + lines_h[k][3])), false) > 0) || (i == 0 && j == 0 && k == 0)){
										minVal = error;
										sidelines[0] = lines_l[i];
										sidelines[1] = lines_r[j];
										hogline = lines_h[k];
									}
								}
							}
						}
					}
					else {
						for (i = 0; i < lines_l.size(); i++){
							j = i;
							for (k = 0; k < lines_h.size(); k++){

								pt1 = getgndbytrw(getintersection(cv::Point2f(lines_l[i][0], lines_l[i][1]), cv::Point2f(lines_l[i][2], lines_l[i][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3])));
								pt2 = getgndbytrw(getintersection(cv::Point2f(lines_r[j][0], lines_r[j][1]), cv::Point2f(lines_r[j][2], lines_r[j][3]), cv::Point2f(lines_h[k][0], lines_h[k][1]), cv::Point2f(lines_h[k][2], lines_h[k][3])));
								pt3 = getgndbytrw(cv::Point2f(lines_l[i][0], lines_l[i][1])) - getgndbytrw(cv::Point2f(lines_l[i][2], lines_l[i][3]));
								pt3 = pt3*(1 / sqrt(pt3.dot(pt3)));
								pt4 = pt1 + 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt2;
								t1 = pt4.dot(pt4);
								pt4 = pt1 - 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt2;
								if (t1 > pt4.dot(pt4))
									t1 = pt4.dot(pt4);
								pt3 = getgndbytrw(cv::Point2f(lines_r[j][0], lines_r[j][1])) - getgndbytrw(cv::Point2f(lines_r[j][2], lines_r[j][3]));
								pt3 = pt3*(1 / sqrt(pt3.dot(pt3)));
								pt4 = pt2 + 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt1;
								t2 = pt4.dot(pt4);
								pt4 = pt2 - 2 * W*cv::Point2f(-pt3.y, pt3.x) - pt1;
								if (t2 > pt4.dot(pt4))
									t2 = pt4.dot(pt4);
								error = sqrt(t1) + sqrt(t2) + fabs(sqrt((pt1 - pt2).dot(pt1 - pt2)) - 2 * W);
								tri[0] = pt1;
								tri[1] = pt2;
								tri[2] = cv::Point2f(0, 0);
												
								if ((minVal > error && cv::pointPolygonTest(tri, getgndbytrw(0.5*cv::Point2f(lines_h[k][0] + lines_h[k][2], lines_h[k][1] + lines_h[k][3])), false) > 0) || (i == 0 && j == 0 && k == 0)){
									minVal = error;
									sidelines[0] = lines_l[i];
									sidelines[1] = lines_r[j];
									hogline = lines_h[k];
								}
							}
						}
					}

					lines_l.clear();
					lines_r.clear();
					lines_h.clear();

					pt1 = getgndbytrw(cv::Point2f(sidelines[0][0], sidelines[0][1]));
					pt2 = getgndbytrw(cv::Point2f(sidelines[0][2], sidelines[0][3]));
					pt3 = pt1 - pt2;
					pt3 = pt3*(1 / sqrt(pt3.dot(pt3)));
					wp = pt1 - pt3*pt3.dot(pt1);
					t1 = sqrt(wp.dot(wp));
	  
					pt1 = getgndbytrw(cv::Point2f(sidelines[1][0], sidelines[1][1]));
					pt2 = getgndbytrw(cv::Point2f(sidelines[1][2], sidelines[1][3]));
					pt4 = pt1 - pt2;
					pt4 = pt4*(1 / sqrt(pt4.dot(pt4))); 
					wp = pt1 - pt4*pt4.dot(pt1);
					t2 = sqrt(wp.dot(wp));

					if (fabs(t1 + t2 - 2*W) > 800){
						printf("warning : sidelines detection failed\n");
						printf("%f\n", fabs(t1 + t2 - 2*W));
						printf("\n\n\n\n");
						goto control;
					}

					wp = cv::Point2f(sin(0.5*atan(pt3.x/pt3.y) + 0.5*atan(pt4.x/pt4.y)), cos(0.5*atan(pt3.x/pt3.y) + 0.5*atan(pt4.x/pt4.y)));

					quad[0] = gettrwbygnd(getgndbytrw(getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]))) + 600*wp);
					quad[1] = gettrwbygnd(getgndbytrw(getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]))) + 600*wp);

					pt1 = getgndbytrw(quad[1]) - 2*600*wp;
					if ((Gnd2Trw.at<float>(2, 0)*pt1.x + Gnd2Trw.at<float>(2, 1)*pt1.y + Gnd2Trw.at<float>(2, 2))*Gnd2Trw.at<float>(2, 1) > 0)
						quad[2] = gettrwbygnd(pt1);
					else
						quad[2] = cv::Point2f(img.cols - 1, img.rows - 1);
	
					pt2 = getgndbytrw(quad[0]) - 2*600*wp;
					if ((Gnd2Trw.at<float>(2, 0)*pt2.x + Gnd2Trw.at<float>(2, 1)*pt2.y + Gnd2Trw.at<float>(2, 2))*Gnd2Trw.at<float>(2, 1) > 0)
						quad[3] = gettrwbygnd(pt2);
					else
						quad[3] = cv::Point2f(0, img.rows - 1);
					ROI = cv::Mat::zeros(imgsize[THROW], CV_8UC1);
					for (i = 0; i < ROI.cols; i+=4){
						for (j = 0; j < ROI.rows; j+=4){
							if (cv::pointPolygonTest(quad, cv::Point2f(i, j), true) > 2)
								ROI(cv::Range(j, j + 4), cv::Range(i, i + 4)) = cv::Mat::ones(4, 4, CV_8UC1);
						}
					}
					quad[0] = gettrwbygnd(getgndbytrw(getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]))) + 100*wp);
					quad[1] = gettrwbygnd(getgndbytrw(getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]))) + 100*wp);
					quad[2] = gettrwbygnd(getgndbytrw(quad[1]) - 2*100*wp);
					quad[3] = gettrwbygnd(getgndbytrw(quad[0]) - 2*100*wp);
					for (i = 0; i < ROI.cols; i+=4){
						for (j = 0; j < ROI.rows; j+=4){
							if (cv::pointPolygonTest(quad, cv::Point2f(i, j), true) > 2)
								ROI(cv::Range(j, j + 4), cv::Range(i, i + 4)) = cv::Mat::zeros(4, 4, CV_8UC1);
						}
					}
					cv::meanStdDev(hsv[1], mean, stddev, ROI);

					if (mean[0] < 0.14 && stddev[0] < 0.12){ //changeable

						cam_dir[THROW] = cv::Point2f(-wp.x, wp.y);
						pt1 = getgndbytrw(cv::Point2f(hogline[0], hogline[1]));
						pt2 = getgndbytrw(cv::Point2f(hogline[2], hogline[3]));
						pt4 = pt1 - pt2;
						pt4 = pt4*(1 / sqrt(pt4.dot(pt4)));
						wp = pt1 - pt4*pt4.dot(pt1);
						if (sqrt(wp.dot(wp)) < 0.5*D_hog){
							cam_pos[THROW] = cv::Point2f(0.5*(t1 - t2), D_hog - sqrt(wp.dot(wp)));
							IsValid++;
						}
						else{
							printf("warning : hogline detection failed\n");
							printf("\n\n\n\n");
							goto control;
						}
					}
					else{	
						printf("warning : hogline detection failed\n");
						printf("mean of saturation: %f, stddev of saturation: %f\n", mean[0], stddev[0]);
						printf("\n\n\n\n");
						goto control;
					}
				}
			}
		}

		if (cam_mode == NEAR && SKIPorTHROWER == SKIP){

			//cv::imshow("bnw", bnw);
	
			if (IsDisplaying){
				cv::line(img_draw, getimgbywld(cv::Point2f(W, D_hog)), getimgbywld(cv::Point2f(-W, D_hog)), cv::Scalar(0, 1, 0), 2);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, D_hog)), getimgbywld(cv::Point2f(W, 0)), cv::Scalar(0, 1, 0), 2);
				cv::line(img_draw, getimgbywld(cv::Point2f(-W, D_hog)), getimgbywld(cv::Point2f(-W, 0)), cv::Scalar(0, 1, 0), 2);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, 0)), getimgbywld(cv::Point2f(-W, 0)), cv::Scalar(0, 1, 0), 2);
			}

			if(IsReached[1] && !IsRedetecting){
				bnw_ = cv::Mat::zeros(imgsize[NEAR], CV_8UC1);
				for(k = 0; k < stonenum; k++){
					if(Collided[k] && k != thrownstonenum){
						predict = kalman[k].predict();
						ptPredicted[k] = cv::Point2f(predict.at<float>(0, 0), predict.at<float>(1, 0));	
						if (imgRect.contains(getimgbywld(ptPredicted[k]))){
							stoneRect = getStoneBoundingBox(ptPredicted[k]);
							for (i = stoneRect.x; i < stoneRect.x + stoneRect.width; i++){
								for (j = stoneRect.y; j < stoneRect.y + stoneRect.height; j++){
									if (imgRect.contains(cv::Point(i, j)))
										bnw_.at<uchar>(j, i) = bnw.at<uchar>(j, i);
								}
							}
						}
						if (IsDisplaying)
							cv::rectangle(img_draw, stoneRect, cv::Scalar(1, 0, 0), 2);
					}
				}
				predict = kalman[thrownstonenum].predict();		
				ptPredicted[thrownstonenum] = cv::Point2f(predict.at<float>(0, 0), predict.at<float>(1, 0));
				if (imgRect.contains(getimgbywld(ptPredicted[thrownstonenum]))){
					stoneRect = getStoneBoundingBox(ptPredicted[thrownstonenum]);
					for (i = stoneRect.x; i < stoneRect.x + stoneRect.width; i++){
						for (j = stoneRect.y; j < stoneRect.y + stoneRect.height; j++){
							if (imgRect.contains(cv::Point(i, j)))
								bnw_.at<uchar>(j, i) = bnw.at<uchar>(j, i);
						}
					}
					if (IsDisplaying)
						cv::rectangle(img_draw, stoneRect, cv::Scalar(1, 0, 0), 2);
				}
				cv::multiply(bnw_, ROI_reach, bnw);
				cv::findContours(bnw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			}
			else {
				if (IsSnapshot[NEAR])
					cv::multiply(bnw, ROI, bnw);			
				else
					cv::multiply(bnw, ROI_reach, bnw);			
				cv::findContours(bnw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			}

			//cv::waitKey();

			for (i = 0; i < contours.size(); i++){
				if (contours[i].size() < 10){
					contours.erase(contours.begin() + i);
					i--;
				}
				else
					coarseboxes.push_back(cv::boundingRect(contours[i]));
			}

			for (i = 0; i < coarseboxes.size(); i++){

				pad = getEllipse(getwldbyimg(coarseboxes[i].tl() + 0.5*cv::Point(coarseboxes[i].width, coarseboxes[i].height), h), d_).height;
				coarsebox = coarseboxes[i] - cv::Point(1, pad) + cv::Size(2, pad); // padding

				cv::convexHull(contours[i], hull);
				for (j = 0; j < hull.size(); j++)
					hull[j] = getimgbywld(getwldbyimg(hull[j]), 0.8*h);

				wldcontour.clear();
				for (j = 0; j < hull.size(); j++)
					wldcontour.push_back(getwldbyimg(hull[j], h));

				maxnum_Mask = cv::contourArea(wldcontour) / (CV_PI*d*d / 4);

				if (maxnum_Mask > shotnum)
					maxnum_Mask = shotnum;
				else if (maxnum_Mask < 1)
					continue;

				if (coarsebox.x < 0 || coarsebox.y < 0 || coarsebox.x + coarsebox.width >= hsv[1].cols || coarsebox.y + coarsebox.height >= hsv[1].rows)
					continue;
				else
					Mask = hsv[1](coarsebox).clone(); //when stone's saturation is larger than background.
				
				bias_Mask = cv::Point(coarsebox.x, coarsebox.y);
				cv::GaussianBlur(Mask, Mask, cv::Size(3, 3), 1);
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

				//cv::line(Mgt, getimgbywld(cv::Point2f(-W, D_hog)) - cv::Point2f(coarsebox.x, coarsebox.y), getimgbywld(cv::Point2f(W, D_hog)) - cv::Point2f(coarsebox.x, coarsebox.y), 0, 1);

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

				/*for (j = 0; j < hull.size() - 1; j++)
					cv::line(img_draw, hull[j], hull[j + 1], cv::Scalar(0.5, 0.5, 0.5), 1);
				cv::line(img_draw, hull[j], hull[0], cv::Scalar(0.5, 0.5, 0.5), 1);
				cv::rectangle(img_draw, coarsebox, cv::Scalar(0.5, 0.5, 0.5), 2);*/
				//cv::putText(img_draw, patch::to_string(maxnum_Mask), coarsebox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0.5, 0.5, 0.5), 2);

				for (x = 0; x < Vote.cols; x++){
					for (y = 0; y < Vote.rows; y++){
						if (Vote_prior.at<float>(y, x) > 0){
							pt = getimgbywld(unit*(cv::Point(x, y) + bias_Vote), h) + cv::Point2f(0.5, 0.5);
							if (coarsebox.contains(pt) && Mgt.at<float>(pt - bias_Mask) > 0.3){
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
				cv::GaussianBlur(Vote, Vote, cv::Size(3, 3), 1);

				/*cv::minMaxLoc(Vote, &minVal, &maxVal);
				cv::imshow("Vote", Vote / maxVal);
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
				if (IsSnapshot[NEAR]){
					for (j = 0; j < maxnum_Mask; j++){
						k = sortIdx1.at<int>(j, 0);
						if (V_m.at<float>(k, 0) >= ((0.65-0.35)/D_hog)*wldcenter[k].y + 0.35){
							printf("%f\n", V_m.at<float>(k, 0));
							wldcentroid.push_back(wldcenter[k]);
							cand.push_back(true);
						}
						else
							break;
					}
				}
				else {
					for (j = 0; j < maxnum_Mask; j++){
						k = sortIdx1.at<int>(j, 0);
							
						if (V_m.at<float>(k, 0) >= ((0.65-0.3)/D_hog)*wldcenter[k].y + 0.3 - fabs(wldcenter[k].x)*0.1/(0.5*W)){
							printf("%f\n", V_m.at<float>(k, 0));
							wldcentroid.push_back(wldcenter[k]);
							cand.push_back(true);
						}
						else
							break;
					}
				}
				//printf("--------%d / %d\n", j, maxnum_Mask);
				wldcenter.clear();
				V_m.release();

			}
			coarseboxes.clear();

		}
		else if (cam_mode == FFAR && SKIPorTHROWER == SKIP) {
			
			if (IsDisplaying){
				quad[0] = getimgbywld(cv::Point2f(-W + 0.5*d, D_hog_ + pad_hog_), 0);
				quad[1] = getimgbywld(cv::Point2f(W - 0.5*d, D_hog_ + pad_hog_), 0);
				quad[2] = getimgbywld(cv::Point2f(W - 0.5*d, transition[0] - pad_mid), 0);
				quad[3] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[0] - pad_mid), 0);
					
				//for (i = 0; i < 4; i++)
				//	cv::line(img_draw, quad[i%4], quad[(i+1)%4], cv::Scalar(0, 1, 0), 2);

				cv::line(img_draw, getimgbywld(cv::Point2f(W, D_hog_)), getimgbywld(cv::Point2f(-W, D_hog_)), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, D_hog_)), getimgbywld(cv::Point2f(W, transition[0])), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getimgbywld(cv::Point2f(-W, D_hog_)), getimgbywld(cv::Point2f(-W, transition[0])), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, transition[0])), getimgbywld(cv::Point2f(-W, transition[0])), cv::Scalar(0, 0, 0), 5);
			}
			
			cv::multiply(bnw, ROI_FFAR, bnw);

			//cv::imshow("bnw", bnw);
			//cv::waitKey();
			cv::findContours(bnw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			for (i = 0; i < contours.size(); i++){

				coarsebox = cv::boundingRect(contours[i]);
				if (coarsebox.height < 10 || coarsebox.width > 200){
					contours.erase(contours.begin() + i);
					i--;
				}
				else{
					coarseboxes.push_back(cv::boundingRect(contours[i]));
					cv::rectangle(img_draw, coarsebox, cv::Scalar(0.5, 0.5, 0.5), 2);
				}
			}

			for (i = 0; i < coarseboxes.size(); i++){

				pad = getEllipse(getwldbyimg(coarseboxes[i].tl() + 0.5*cv::Point(coarseboxes[i].width, coarseboxes[i].height), h), d_).height;
				coarsebox = coarseboxes[i] - cv::Point(1, pad) + cv::Size(2, pad); // padding

				cv::convexHull(contours[i], hull);
				for (j = 0; j < hull.size(); j++)
					hull[j] = getimgbywld(getwldbyimg(hull[j]), 0.8*h);

				wldcontour.clear();
				for (j = 0; j < hull.size(); j++)
					wldcontour.push_back(getwldbyimg(hull[j], h));

				maxnum_Mask = cv::contourArea(wldcontour) / (CV_PI*d*d / 4);
				if (maxnum_Mask > 1)
					maxnum_Mask = 1;
				else if (maxnum_Mask < 1)
					continue;

				if (coarsebox.x < 0 || coarsebox.y < 0 || coarsebox.x + coarsebox.width >= hsv[1].cols || coarsebox.y + coarsebox.height >= hsv[1].rows)
					continue;
				else
					Mask = hsv[1](coarsebox).clone(); //when stone's saturation is larger than background.
				cv::GaussianBlur(Mask, Mask, cv::Size(3, 3), 1);
				bias_Mask = cv::Point(coarsebox.x, coarsebox.y);
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

				/*for (j = 0; j < hull.size() - 1; j++)
					cv::line(img_draw, hull[j], hull[j + 1], cv::Scalar(0.5, 0.5, 0.5), 1);
				cv::line(img_draw, hull[j], hull[0], cv::Scalar(0.5, 0.5, 0.5), 1);
				cv::rectangle(img_draw, coarsebox, cv::Scalar(0.5, 0.5, 0.5), 2);*/

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

				for (x = 0; x < Vote.cols; x++){
					for (y = 0; y < Vote.rows; y++){
						if (Vote_prior.at<float>(y, x) > 0){
							pt = getimgbywld(unit*(cv::Point(x, y) + bias_Vote), h) + cv::Point2f(0.5, 0.5);
							if (coarsebox.contains(pt) && Mgt.at<float>(pt - bias_Mask) > 0.3){
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
				cv::GaussianBlur(Vote, Vote, cv::Size(3, 3), 1);
				
				/*cv::minMaxLoc(Vote, &minVal, &maxVal, &minLoc, &maxLoc);
				printf("%f\n", maxVal / num_c);
				if (maxVal / num_c > 0.6){
					wldcenter.push_back((maxLoc + bias_Vote)*unit);
					score.push_back(maxVal / num_c);
				}*/

				wldcenter_.clear();
				score_.clear();
				while (1){
					cv::minMaxLoc(Vote, &minVal, &maxVal, &minLoc, &maxLoc);
					if (maxVal < 0.3)
						break;
					wldcenter_.push_back((maxLoc + bias_Vote)*unit);
					score_.push_back(maxVal / num_c);
					for (k = 0; k < num_c_; k++)
						Vote.at<float>(maxLoc + circle_[k]) = 0;
				}

				if (wldcenter_.size() == 0)
					continue;
				else {
					amin = 0;					
					for (j = 0; j < wldcenter_.size(); j++){
						if (wldcenter_[j].y < wldcenter_[amin].y)
							amin = j;
					}
					wldcenter.push_back(wldcenter_[amin]);
					score.push_back(score_[amin]);
				}
			}

			for (i = 0; i < wldcenter.size(); i++)
				cv::circle(img_draw, getimgbywld(wldcenter[i], h), 2, cv::Scalar(0.5, 0.5, 0.5), -1);

			predict = kalman[thrownstonenum].predict();
			ptPredicted[thrownstonenum] = cv::Point2f(predict.at<float>(0, 0), predict.at<float>(1, 0));

			if (wldcenter.size() > 0){
				minVal = kalmanrange;
				amin = 0;
				for (i = 0; i < wldcenter.size(); i++){
					t1 = sqrt((wldcenter[i].x - ptPredicted[thrownstonenum].x)*(wldcenter[i].x - ptPredicted[thrownstonenum].x) + (wldcenter[i].y - ptPredicted[thrownstonenum].y)*(wldcenter[i].y - ptPredicted[thrownstonenum].y));
					if (minVal > t1 && wldcenter[i].y < ptPredicted[thrownstonenum].y + 500){
						amin = i;
						minVal = t1;
					}			
				}
				maxVal = 0;
				amax = amin;
				for (i = 0; i < wldcenter.size() && minVal > 2*meter; i++){
					if (maxVal < score[i]){
						amax = i;
						maxVal = score[i];
					}
				}
				if (sqrt((wldcenter[amax].x - ptPredicted[thrownstonenum].x)*(wldcenter[amax].x - ptPredicted[thrownstonenum].x) + (wldcenter[amax].y - ptPredicted[thrownstonenum].y)*(wldcenter[amax].y - ptPredicted[thrownstonenum].y)) < kalmanrange){
					ptMeasured = wldcenter[amax];
					measure.at<float>(0, 0) = ptMeasured.x;
					measure.at<float>(1, 0) = ptMeasured.y;
					wldstone[thrownstonenum][framenum] = ptMeasured;
					Identified[thrownstonenum][framenum] = true;
					kalman[thrownstonenum].correct(measure);
			
					IsValid++;
				}
				stone[thrownstonenum][framenum] = getimgbywld(wldstone[thrownstonenum][framenum], h);
			}
			else if (1){

				printf("warning : thrown stone tracking failed\n");
				IsWarning++;

				if (wldstone[thrownstonenum][framenum - 1].y != -1 && wldstone[thrownstonenum][framenum - 1].y != 0){
					stoneRect = getEllipse(wldstone[thrownstonenum][framenum - 1], d_);
					if (imgRect.contains(cv::Point(stoneRect.x, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x, stoneRect.y + stoneRect.height)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y + stoneRect.height))){
						box_cor = prv_img(stoneRect);
					}
					else
						continue;
					stoneRect = getEllipse(wldstone[thrownstonenum][framenum - 1], d_) - cv::Point(mx, my) + cv::Size(2 * mx, 2 * my);
					if (imgRect.contains(cv::Point(stoneRect.x, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x, stoneRect.y + stoneRect.height)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y + stoneRect.height))){
						box_cor_ = img(stoneRect);
					}
					else
						continue;
					cv::matchTemplate(box_cor_, box_cor, result, CV_TM_SQDIFF_NORMED);
					cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
					stone[thrownstonenum][framenum] = getimgbywld(wldstone[thrownstonenum][framenum - 1], h) + cv::Point2f(minLoc.x - mx, minLoc.y - my);
					wldstone[thrownstonenum][framenum] = getwldbyimg(stone[thrownstonenum][framenum], h);
					Identified[thrownstonenum][framenum] = true;
				}
				else {
					wldstone[thrownstonenum][framenum] = cv::Point2f(-1, -1);
					stone[thrownstonenum][framenum] = getimgbywld(wldstone[thrownstonenum][framenum], h);
					Identified[thrownstonenum][framenum] = false;
				}
			}
			else {
				printf("error : thrown stone tracking failed\n");
				IsError = 1;
				return IsError;
			}


			if (wldstone[thrownstonenum][framenum].y < transition[0] && wldstone[thrownstonenum][framenum].y > 0){
				IsReached[0] = true;
			}
			
			wldcenter.clear();
			score.clear();

			coarseboxes.clear();
		}
		else if (cam_mode == FAR && SKIPorTHROWER == SKIP) {

			if (IsDisplaying){			
				quad[0] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[1] - pad_hog), 0);
				quad[1] = getimgbywld(cv::Point2f(W - 0.5*d, transition[1] - pad_hog), 0);
				quad[2] = getimgbywld(cv::Point2f(W - 0.5*d, transition[0] + pad_mid), 0);
				quad[3] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[0] + pad_mid), 0);
						
				//for (i = 0; i < 4; i++)
				//	cv::line(img_draw, quad[i%4], quad[(i+1)%4], cv::Scalar(0, 1, 0), 2);
				
				cv::line(img_draw, getimgbywld(cv::Point2f(W, transition[0])), getimgbywld(cv::Point2f(-W, transition[0])), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, transition[0])), getimgbywld(cv::Point2f(W, transition[1])), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getimgbywld(cv::Point2f(-W, transition[0])), getimgbywld(cv::Point2f(-W, transition[1])), cv::Scalar(0, 0, 0), 5);
				cv::line(img_draw, getimgbywld(cv::Point2f(W, transition[1])), getimgbywld(cv::Point2f(-W, transition[1])), cv::Scalar(0, 0, 0), 5);
			}
			
			cv::multiply(bnw, ROI_FAR, bnw);

			//cv::imshow("bnw", bnw);
			//cv::waitKey();

			cv::findContours(bnw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			for (i = 0; i < contours.size(); i++){
	
				coarsebox = cv::boundingRect(contours[i]);

				if (coarsebox.width < 20 || coarsebox.width > 120 || coarsebox.height < 15 || coarsebox.width / coarsebox.height > 3 || cv::sum(bnw(coarsebox))[0]/(coarsebox.width*coarsebox.height*255) < 0.5){
				//if (false){				
					contours.erase(contours.begin() + i);
					i--;
				}
				else {
					coarseboxes.push_back(coarsebox);
					cv::rectangle(img_draw, coarsebox, cv::Scalar(0.5, 0.5, 0.5), 2);
				}
			}

			for (i = 0; i < coarseboxes.size(); i++){

				pad = getEllipse(getwldbyimg(coarseboxes[i].tl() + 0.5*cv::Point(coarseboxes[i].width, coarseboxes[i].height), h), d_).height;
				coarsebox = coarseboxes[i] - cv::Point(1, pad) + cv::Size(2, pad); // padding

				cv::convexHull(contours[i], hull);
				for (j = 0; j < hull.size(); j++)
					hull[j] = getimgbywld(getwldbyimg(hull[j]), 0.8*h);

				wldcontour.clear();
				for (j = 0; j < hull.size(); j++)
					wldcontour.push_back(getwldbyimg(hull[j], h));

				maxnum_Mask = cv::contourArea(wldcontour) / (CV_PI*d*d / 4);
				if (maxnum_Mask > 1)
					maxnum_Mask = 1;
				else if (maxnum_Mask < 1)
					continue;

				if (coarsebox.x < 0 || coarsebox.y < 0 || coarsebox.x + coarsebox.width >= hsv[1].cols || coarsebox.y + coarsebox.height >= hsv[1].rows)
					continue;
				else
					Mask = hsv[1](coarsebox).clone(); //when stone's saturation is larger than background.
				cv::GaussianBlur(Mask, Mask, cv::Size(3, 3), 1);
				bias_Mask = cv::Point(coarsebox.x, coarsebox.y);
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

				for (x = 0; x < Vote.cols; x++){
					for (y = 0; y < Vote.rows; y++){
						if (Vote_prior.at<float>(y, x) > 0){
							pt = getimgbywld(unit*(cv::Point(x, y) + bias_Vote), h) + cv::Point2f(0.5, 0.5);
							if (coarsebox.contains(pt) && Mgt.at<float>(pt - bias_Mask) > 0.2){
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
				cv::minMaxLoc(Vote, &minVal, &maxVal, &minLoc, &maxLoc);
				if (maxVal / num_c > 0.2){
					wldcenter.push_back((maxLoc + bias_Vote)*unit);
					score.push_back(maxVal / num_c);
				}
			}

			for (i = 0; i < wldcenter.size(); i++)
				cv::circle(img_draw, getimgbywld(wldcenter[i], h), 2, cv::Scalar(0.5, 0.5, 0.5), -1);

			predict = kalman[thrownstonenum].predict();
			ptPredicted[thrownstonenum] = cv::Point2f(predict.at<float>(0, 0), predict.at<float>(1, 0));

			if (wldcenter.size() > 0){
				minVal = kalmanrange;
				amin = 0;
				for (i = 0; i < wldcenter.size(); i++){
					t1 = sqrt((wldcenter[i].x - ptPredicted[thrownstonenum].x)*(wldcenter[i].x - ptPredicted[thrownstonenum].x) + (wldcenter[i].y - ptPredicted[thrownstonenum].y)*(wldcenter[i].y - ptPredicted[thrownstonenum].y));
					if (minVal > t1){
						amin = i;
						minVal = t1;
					}			
				}
				maxVal = 0;
				amax = amin;
				for (i = 0; i < wldcenter.size() && minVal > 1*meter; i++){
					if (maxVal < score[i]){
						amax = i;
						maxVal = score[i];
					}
				}
				if (sqrt((wldcenter[amax].x - ptPredicted[thrownstonenum].x)*(wldcenter[amax].x - ptPredicted[thrownstonenum].x) + (wldcenter[amax].y - ptPredicted[thrownstonenum].y)*(wldcenter[amax].y - ptPredicted[thrownstonenum].y)) < kalmanrange){
					ptMeasured = wldcenter[amax];
					measure.at<float>(0, 0) = ptMeasured.x;
					measure.at<float>(1, 0) = ptMeasured.y;
					wldstone[thrownstonenum][framenum] = ptMeasured;
					Identified[thrownstonenum][framenum] = true;
					kalman[thrownstonenum].correct(measure);
			
					IsValid++;
				}
				stone[thrownstonenum][framenum] = getimgbywld(wldstone[thrownstonenum][framenum], h);
			}
			else if (1){
				printf("warning : thrown stone tracking failed\n");
				IsWarning++;

				if (wldstone[thrownstonenum][framenum - 1].y != -1 && wldstone[thrownstonenum][framenum - 1].y != 0){
					stoneRect = getEllipse(wldstone[thrownstonenum][framenum - 1], d_);
					if (imgRect.contains(cv::Point(stoneRect.x, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x, stoneRect.y + stoneRect.height)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y + stoneRect.height))){
						box_cor = prv_img(stoneRect);
					}
					else
						continue;
					stoneRect = getEllipse(wldstone[thrownstonenum][framenum - 1], d_) - cv::Point(mx, my) + cv::Size(2 * mx, 2 * my);
					if (imgRect.contains(cv::Point(stoneRect.x, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x, stoneRect.y + stoneRect.height)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y + stoneRect.height))){
						box_cor_ = img(stoneRect);
					}
					else
						continue;
					cv::matchTemplate(box_cor_, box_cor, result, CV_TM_SQDIFF_NORMED);
					cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
					stone[thrownstonenum][framenum] = getimgbywld(wldstone[thrownstonenum][framenum - 1], h) + cv::Point2f(minLoc.x - mx, minLoc.y - my);
					wldstone[thrownstonenum][framenum] = getwldbyimg(stone[thrownstonenum][framenum], h);
					Identified[thrownstonenum][framenum] = true;
				}
				else {
					wldstone[thrownstonenum][framenum] = cv::Point2f(-1, -1);
					stone[thrownstonenum][framenum] = getimgbywld(wldstone[thrownstonenum][framenum], h);
					Identified[thrownstonenum][framenum] = false;
				}
			}
			else {
				printf("error : thrown stone tracking failed\n");
				IsError = 1;
				return IsError;
			}


			if (wldstone[thrownstonenum][framenum].y < transition[1] && wldstone[thrownstonenum][framenum].y > 0){
				IsReached[1] = true;
				stonenum++;
				Collided[thrownstonenum] = true;
				CollidedStonenum++;
			}

			wldcenter.clear();
			score.clear();

			coarseboxes.clear();
		}

		if ((cam_mode == FFAR || cam_mode == FAR) && SKIPorTHROWER == SKIP){
			for (i = 0; i < stonenum; i++){
				if(i!=thrownstonenum){
					wldstone[i][framenum] = wldstone[i][framenum - frameunit];
					//stone[i][framenum] = stone[i][framenum - frameunit];
					Identified[i][framenum] = true;
				}
			}
		
		}
		else if (cam_mode == NEAR && SKIPorTHROWER == SKIP){
			
			if(IsReached[1] && CollidedStonenum == 1 && !IsRedetecting){
				if (cand.size() != 0){
					minVal = kalmanrange;
					for (i = 0; i < cand.size(); i++){
						if (cand[i]){
							t1 = wldcentroid[i].y - ptPredicted[thrownstonenum].y;
							if (minVal > t1){
								amin = i;
								minVal = t1;
							}
						}
					}
					if (minVal < kalmanrange){
						ptMeasured = wldcentroid[amin];
						wldstone[thrownstonenum][framenum] = ptMeasured;
						stone[thrownstonenum][framenum] = getimgbywld(ptMeasured, h);
						Identified[thrownstonenum][framenum] = true;

						measure.at<float>(0, 0) = ptMeasured.x;
						measure.at<float>(1, 0) = ptMeasured.y;
						kalman[thrownstonenum].correct(measure);
						
						IsValid++;
					}
				}
				else{
					printf("warning : thrown stone tracking failed\n");
					IsWarning++;
					wldstone[thrownstonenum][framenum] = cv::Point2f(-1, -1);
					Identified[thrownstonenum][framenum] = false;					
				}
				
				for (i = 0; i < stonenum; i++){					
					if(i != thrownstonenum){					
						wldstone[i][framenum] = wldstone[i][framenum - frameunit];
						stone[i][framenum] = getimgbywld(wldstone[i][framenum], h);
						
						Identified[i][framenum] = true;
					}
				}
			}
			else{

				if (cand.size() != 0){

					minVals = cv::Mat(cand.size(), 1, CV_32FC1, cv::Scalar(-1));
					minIdxs = cv::Mat(cand.size(), 1, CV_32FC1, cv::Scalar(-1));
					if (stonenum > 0){
						for(i = 0; i < cand.size(); i++){
											
							if(wldstone[j][framenum-1].dot(wldstone[i][framenum])!=0)
								continue;
							if (cand[i]){
								amin = 0;						
								minVal = (wldstone[0][framenum - frameunit] - wldcentroid[i]).dot(wldstone[0][framenum - frameunit] - wldcentroid[i]);
								for (j = 1; j < stonenum; j++){
									if (minVal > (wldstone[j][framenum - frameunit] - wldcentroid[i]).dot(wldstone[j][framenum - frameunit] - wldcentroid[i])){
										minVal = (wldstone[j][framenum - frameunit] - wldcentroid[i]).dot(wldstone[j][framenum - frameunit] - wldcentroid[i]);
										amin = j;
									}
								}
								minVals.at<float>(i, 0) = minVal;
								minIdxs.at<int>(i, 0) = amin;
							}
						}
						cv::sortIdx(minVals, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

						for (j = 0; j < cand.size(); j++){
							i = sortIdx1.at<int>(j, 0);
							amin = minIdxs.at<int>(i, 0);
							if (Identified[amin][framenum]){
								minVal = minVals.at<float>(i, 0);
								for (j = 0; j < stonenum; j++){
									if (!Identified[j][framenum] && minVal > (wldstone[j][framenum - frameunit] - wldcentroid[i]).dot(wldstone[j][framenum - frameunit] - wldcentroid[i])){	
										minVal = (wldstone[j][framenum - frameunit] - wldcentroid[i]).dot(wldstone[j][framenum - frameunit] - wldcentroid[i]);
										amin = j;
									}
								}
								if (amin == minIdxs.at<int>(i, 0)){
									if (!IsChanging){
										color[stonenum] = getColor(wldcentroid[i], hsv);
										stone[stonenum][framenum] = getimgbywld(wldcentroid[i], h);
										wldstone[stonenum][framenum] = wldcentroid[i];
										Identified[stonenum][framenum] = true;
										stonenum++;
										continue;
									}
									else {
										cand[i] = 0;
										break;
									}
								}
								minVals.at<float>(i, 0) = minVal;
								minIdxs.at<int>(i, 0) = amin;
								cv::sortIdx(minVals, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
								j--;
								continue;
							}
							if (minVals.at<float>(i, 0) < errorrange*errorrange){
								//color[amin] = getColor(wldcentroid[i], hsv);								
								stone[amin][framenum] = getimgbywld(wldcentroid[i], h);
								wldstone[amin][framenum] = wldcentroid[i];
								if(Collided[amin]){
									measure.at<float>(0, 0) = wldstone[amin][framenum].x;
									measure.at<float>(1, 0) = wldstone[amin][framenum].y;
									kalman[amin].correct(measure);
								}
								Identified[amin][framenum] = true;
							}
							
					
						}//Min-min
					
				
					}
					else if (!IsThrown || IsRedetecting){	
						for (i = 0; i < cand.size(); i++){
							if (cand[i]){
								color[stonenum] = getColor(wldcentroid[i], hsv);
								if (color[stonenum][0] == 0 && color[stonenum][1] == 0 && color[stonenum][2] == 0){
									cand[i] = 0;
									continue;
								}
								stone[stonenum][framenum] = getimgbywld(wldcentroid[i], h);
								wldstone[stonenum][framenum] = wldcentroid[i];
								Identified[stonenum][framenum] = true;
								stonenum++;
							}
						}
					}
					
				}

				for (i = 0; i < stonenum && framenum > 0; i++){
					if (!Identified[i][framenum] && Collided[i]){
						stoneRect = getEllipse(wldstone[i][framenum - frameunit], d_);
						if (imgRect.contains(cv::Point(stoneRect.x, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x, stoneRect.y + stoneRect.height)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y + stoneRect.height))){
							box_cor = prv_img(stoneRect);
						}
						else
							continue;
						stoneRect = getEllipse(wldstone[i][framenum - frameunit], d_) - cv::Point(mx, my) + cv::Size(2 * mx, 2 * my);
						if (imgRect.contains(cv::Point(stoneRect.x, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y)) && imgRect.contains(cv::Point(stoneRect.x, stoneRect.y + stoneRect.height)) && imgRect.contains(cv::Point(stoneRect.x + stoneRect.width, stoneRect.y + stoneRect.height))){
							box_cor_ = img(stoneRect);
						}
						else
							continue;
						cv::matchTemplate(box_cor_, box_cor, result, CV_TM_SQDIFF_NORMED);
						cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
						stone[i][framenum] = getimgbywld(wldstone[i][framenum - frameunit], h) + cv::Point2f(minLoc.x - mx, minLoc.y - my);
						wldstone[i][framenum] = getwldbyimg(stone[i][framenum], h);
						IsWarning++;
					}
					else if (!Identified[i][framenum] && !Collided[i]){
						wldstone[i][framenum] = wldstone[i][framenum - frameunit];
						stone[i][framenum] = getimgbywld(wldstone[i][framenum], h);
						Identified[i][framenum] = true;
						if (IsDetecting)
							IsWarning++;
					}
				}

				IsRedetecting = false;
			}
				
			if (IsDetecting){
				//check detection
				num_stone = 0;
				for (i = 0; i < cand.size(); i++){
					if (cand[i])
						num_stone++;
				}
				if (num_stone > shotnum || num_stone != stonenum){
					printf("error : detection is failed\n");
					std::cout<<"num_stone : "<<num_stone<<" , stonenum : " <<stonenum<<std::endl;
					IsError = 1;
					return IsError;
				}
				else if (!IsWarning){
					num_stone_prv = num_stone;
					IsValid++;
				}
			}
			prv_img = img.clone();
			cand.clear();
			wldcentroid.clear();
				
			//check collision
			while(1){
				k = CollidedStonenum;
				for(i = 0; i < stonenum && IsThrown; i++){
					for(j = 0; j < stonenum && !Collided[i]; j++){
						if(Collided[j]){
							pt = wldstone[j][framenum]-wldstone[i][framenum];
							if(sqrt(pt.dot(pt)) < d + 100){
								Collided[i] = true;
								CollidedStonenum++;
								kalman[i].init(4, 2, 0);	
								kalman[i].transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 0.01, 0, 0, 1, 0, 0.01, 0, 0, 1, 0, 0, 0, 0, 1);
								kalman[i].measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
								cv::setIdentity(kalman[i].processNoiseCov, cv::Scalar(1e-5));
								kalman[i].processNoiseCov.at<float>(2, 2) = 0;
								kalman[i].processNoiseCov.at<float>(3, 3) = 0;
								cv::setIdentity(kalman[i].measurementNoiseCov, cv::Scalar(1e-7));
								cv::setIdentity(kalman[i].errorCovPost, cv::Scalar(1.0));
								kalman[i].statePost.at<float>(0, 0) = wldstone[i][framenum].x;
								kalman[i].statePost.at<float>(1, 0) = wldstone[i][framenum].y;
								kalman[i].statePost.at<float>(2, 0) = 0;
								kalman[i].statePost.at<float>(3, 0) = 0;
	
								std::cout<<j<<" stone maybe collided with "<<i<< " stone"<<std::endl;			
							}
						}
					}
				}
				if(k==CollidedStonenum)
					break;
			}


		}

		control:

		if (IsRealEnd == 0){
			IsEnd = true;
			if (IsReached[1]){
				for (i = 0; i < stonenum; i++){
					pt1 = wldstone[i][framenum] - wldstone[i][framenum - 3*frameunit];
					if (sqrt(pt1.dot(pt1)) <= 40 || wldstone[i][framenum].y < 0)
						Stop[i]++;
					else
						Stop[i] = 0;
				}
				for (i = 0; i < stonenum; i++){
					if (Stop[i] < 5){
						IsEnd = false;
						break;
					}
				}
			}
			else if (IsReached[0]){
				for (i = 0; i < stonenum + 1; i++){
					pt1 = wldstone[i][framenum] - wldstone[i][framenum - 3*frameunit];
					if (sqrt(pt1.dot(pt1)) <= 40 || wldstone[i][framenum].y < 0)
						Stop[i]++;
					else
						Stop[i] = 0;
				}
				for (i = 0; i < stonenum + 1; i++){
					if (Stop[i] < 5){
						IsEnd = false;
						break;
					}
				}
			}
			else
				IsEnd = false;
		}
			
		if (IsEnd){

			if (IsRealEnd == 0){
				IsDetectingSweeper = true;
				IsRealEnd = 1;
				goto capture;
			}
			else if (IsRealEnd == 1){
				IsRedetecting = true;
				IsRealEnd = 2;
				stonenum = 0;
				framenum++;
				goto capture;
			}
			else {
				IsCollided = num_stone_prv;
				for (i = 0; i < num_stone_prv; i++){
					for (j = 0; j < stonenum; j++){
						pt1 = wldstone[j][framenum] - wldstone[i][framenum_thrown];
						if (sqrt(pt1.dot(pt1)) < 40){
							IsCollided--;
							break;
						}
					}					
				}
				
			}
		}

		
		latency1 = getmsec() - msec0;
		latency2 = getmsec() - msec_frame[framenum];

		/********************************************************************************/

		printf("====current state=====\n");
		if (SKIPorTHROWER == SKIP)
			printf("SKIP mode\n");
		else if (SKIPorTHROWER == THROWER)
			printf("THROWER mode\n");
		if (cam_mode == NEAR)
			printf("Near Camera Enabled\n");
		else if (cam_mode == FAR)
			printf("Far Camera Enabled\n");
		else if (cam_mode == THROW)
			printf("Throw Camera Enabled\n");
		else if (cam_mode == FFAR)
			printf("FFAR Camera Enabled\n");
		
		if (IsChanging)
			printf("IsChanging\n");
		if (IsCalibration)
			printf("IsCalibration\n");
		if (IsPositioning)
			printf("IsPositioning\n");
		if (IsRunning)
			printf("IsRunning\n");
		if (IsOneEllipse && IsPositioning)
			printf("One Ellipse mode\n");
		if (IsOneSideline && IsRunning)
			printf("One Sideline mode\n");
		if (IsThrown)
			printf("IsThrown\n");
		if (IsEnd)
			printf("IsEnd\n");
		if (IsWarning)
			printf("Warning : %d\n", IsWarning);
		printf("======================\n");
		printf("Validation %d / %d\n", IsValid, thr_valid);
		printf("frame number : %d\n", framenum);
		printf("latency1 : %d\n", latency1);
		//printf("latency2 : %d\n", latency2);
		if (SKIPorTHROWER == SKIP && !IsCalibration){
			printf("number of stone = %d\n", stonenum);
			cv::Mat radi = cv::Mat::zeros(stonenum, 1, CV_32FC1);
			if (stonenum > 0){
				for (i = 0; i < stonenum; i++)
					radi.at<float>(i, 0) = sqrt(wldstone[i][framenum].x*wldstone[i][framenum].x + (wldstone[i][framenum].y - R_cir)*(wldstone[i][framenum].y - R_cir));	
				cv::sortIdx(radi, sortIdx1, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
				for (i = 0; i < stonenum; i++){
					if(!Collided[sortIdx1.at<int>(i, 0)])
						printf("stone %d : %d, %d, %d\n", sortIdx1.at<int>(i, 0), (int)wldstone[sortIdx1.at<int>(i, 0)][framenum].x, (int)wldstone[sortIdx1.at<int>(i, 0)][framenum].y, (int)radi.at<float>(sortIdx1.at<int>(i, 0), 0));
					else
						printf("collided stone %d : %d, %d\n", sortIdx1.at<int>(i, 0), (int)wldstone[sortIdx1.at<int>(i, 0)][framenum].x, (int)wldstone[sortIdx1.at<int>(i, 0)][framenum].y);
			}	

			}
			if (cam_mode == FAR || cam_mode == FFAR){
				printf("thrown stone %d : %d, %d\n", thrownstonenum, (int)wldstone[thrownstonenum][framenum].x, (int)wldstone[thrownstonenum][framenum].y);
				//printf("sweeper : (%.0f, %.0f) / (%.0f, %.0f)\n", wldsweeper[framenum].x, wldsweeper[framenum].y, ptPredicted_s.x, ptPredicted_s.y);

			}

		}
		else if (SKIPorTHROWER == THROWER){
			if (cam_dir[cam_mode].y == 0 && cam_dir[cam_mode].x > 0)
				angle = -90;
			else if (cam_dir[cam_mode].y == 0 && cam_dir[cam_mode].x < 0)
				angle = 90;
			else
				angle = -(180/CV_PI)*atan(cam_dir[cam_mode].x / cam_dir[cam_mode].y);

			if (IsValid){
				angle_mean = angle_mean*((IsValid - 1.0) / IsValid) + angle*(1.0 / IsValid);
				robotPos.x = robotPos.x*((IsValid - 1.0) / IsValid) + cam_pos[cam_mode].x*(1.0 / IsValid);
				robotPos.y = robotPos.y*((IsValid - 1.0) / IsValid) + cam_pos[cam_mode].y*(1.0 / IsValid);
				hogDist = D_hog - cam_pos[cam_mode].y;
				if (IsPositioning && IsValid == thr_valid){
					hogDist0 = hogDist;
					printf("(x y theta) : (%f %f %f)\n", robotPos.x, robotPos.y, angle_mean);
				}
				else if (IsRunning){
					msec_release = getmsec_release(hogDist);
					printf("hogDist : %f\n", hogDist);
				}
			}
			if (IsValid == thr_valid + 1){
				angle_mean = 0;
				robotPos.x = 0;
				robotPos.y = 0;
				hogDist = 0;
			}


			if (IsPositioning)
				printf("(x y theta) : (%f %f %f)\n", robotPos.x, robotPos.y, angle_mean);
			else if (IsRunning){
				printf("msec_running : %d\n", msec_running);
				printf("msec_release : %d\n", msec_release);
				printf("msec : %d\n", getmsec());
			}
		}
		printf("\n\n\n\n\n");
	
		/************************************ Display *************************************/
		if (IsDisplaying || IsLogging){
			
			Sheet_draw = Sheet.clone();
			for (i = 0; i < stonenum; i++){
				displaypoint = displayscale*(cv::Point(wldstone[i][framenum].y, wldstone[i][framenum].x)) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5, color[i], -1);
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5 - 1, cv::Scalar(0, 0, 0), 2);
				if (Collided[i]){
					displaypoint = displayscale*cv::Point(ptPredicted[i].y, ptPredicted[i].x) + display_origin;
					cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5, cv::Scalar(0, 0.5, 0), 2);
				}
			}
			if (IsThrown && !Collided[thrownstonenum]){
				displaypoint = displayscale*cv::Point(ptPredicted[thrownstonenum].y, ptPredicted[thrownstonenum].x) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5, cv::Scalar(0, 0.5, 0), 2);
				cv::circle(img_draw,getimgbywld(ptPredicted[thrownstonenum],h),3,cv::Scalar(0,0.5,0),-1);

				/*displaypoint = displayscale*cv::Point(ptPredicted_s.y, ptPredicted_s.x) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale*d_s*0.5, cv::Scalar(0, 0.5, 0), 2);
				cv::circle(img_draw,getimgbywld(ptPredicted_s, h_s),3,cv::Scalar(0,0.5,0),-1);
				displaypoint = displayscale*cv::Point(wldsweeper[framenum].y, wldsweeper[framenum].x) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale*d_s*0.5, cv::Scalar(0, 0, 0), -1);
				cv::circle(Sheet_draw, displaypoint, displayscale*d_s*0.5 - 1, cv::Scalar(0, 0, 0), 2);*/
			}
			
			if (IsThrown && Identified[thrownstonenum][framenum]){
				displaypoint = displayscale*cv::Point(wldstone[thrownstonenum][framenum].y, wldstone[thrownstonenum][framenum].x) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5, color[thrownstonenum], -1);
				cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5 - 1, cv::Scalar(0, 0, 0), 2);
				
			}
			if (IsValid){
				displaypoint = displayscale*cv::Point(cam_pos[cam_mode].y, cam_pos[cam_mode].x) + display_origin;
				cv::circle(Sheet_draw, displaypoint, displayscale * 100, cv::Scalar(0, 0, 0), -1);
				cv::arrowedLine(Sheet_draw, displaypoint, displaypoint + 3 * displayscale * 100 * cv::Point2f(cam_dir[cam_mode].y, cam_dir[cam_mode].x), cv::Scalar(0, 0, 0), 2);
			}
			cv::GaussianBlur(Sheet_draw, Sheet_draw, cv::Size(3, 3), 1);

			if (cam_mode == NEAR){
				if (IsSnapshot[NEAR]){
					for (i = 0; i < 12; i++)
						cv::circle(img_draw, getimgbywld(cv::Point2f(wldpoint[i].x, wldpoint[i].y)), 2, cv::Scalar(0, 0, 0), -1);
					if (!IsOneEllipse)
						cv::circle(img_draw, housecenter, 5, cv::Scalar(0, 0, 0), -1);
				}
				for (i = 0; i < stonenum; i++){
					cv::circle(img_draw, stone[i][framenum], 3, color[i], -1);
					cv::circle(img_draw, stone[i][framenum], 4, color[i], -1);
					if (Identified[i][framenum])
						cv::putText(img_draw, patch::to_string(i), cv::Point2f(stone[i][framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
					else
						cv::putText(img_draw, patch::to_string(i), cv::Point2f(stone[i][framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0.5), 2);
					if (Collided[i])
						cv::circle(img_draw,getimgbywld(ptPredicted[i],h),3,cv::Scalar(0,0.5,0),-1);			
				}
				 
			}
			else if (cam_mode == FAR || cam_mode == FFAR){
				cv::circle(img_draw, stone[thrownstonenum][framenum], 3, color[thrownstonenum], -1);
				cv::circle(img_draw, stone[thrownstonenum][framenum], 4, color[thrownstonenum], -1);
				if (IsThrown && Identified[thrownstonenum][framenum])
					cv::putText(img_draw, patch::to_string(thrownstonenum), cv::Point2f(stone[thrownstonenum][framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
				else
					cv::putText(img_draw, patch::to_string(thrownstonenum), cv::Point2f(stone[thrownstonenum][framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0.5), 2);
				/*cv::circle(img_draw, sweeper[framenum], 3, cv::Scalar(0, 0, 0), -1);
				cv::circle(img_draw, sweeper[framenum], 4, cv::Scalar(0, 0, 0), -1);
				if (Identified_s[framenum])
					cv::putText(img_draw, "S", cv::Point2f(sweeper[framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
				else
					cv::putText(img_draw, "S", cv::Point2f(sweeper[framenum]) + cv::Point2f(-1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0.5), 2);*/
			}
			else if (cam_mode == THROW && IsValid){
				cv::line(img_draw, getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(sidelines[0][0], sidelines[0][1]), cv::Point2f(sidelines[0][2], sidelines[0][3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0.5, 0.5, 0.5), 3);
				cv::line(img_draw, getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(sidelines[1][0], sidelines[1][1]), cv::Point2f(sidelines[1][2], sidelines[1][3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0, 0, 0), 3);
				cv::line(img_draw, getintersection(cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]), cv::Point2f(0, 0), cv::Point2f(0, img.rows)), getintersection(cv::Point2f(hogline[0], hogline[1]), cv::Point2f(hogline[2], hogline[3]), cv::Point2f(img.cols, 0), cv::Point2f(img.cols, img.rows)), cv::Scalar(0, 0, 1), 3);
			}
			if (img_draw.cols != 1280)
				cv::resize(img_draw, img_draw, cv::Size(1280, 720));
			img_draw.copyTo(Frame(cv::Rect(0, 0, img_draw.cols, img_draw.rows)));
			Sheet_draw.copyTo(Frame(cv::Rect(0, img_draw.rows, Sheet_draw.cols, Sheet_draw.rows)));
			cv::Mat Frame_resize;			
			//cv::resize(Frame, Frame_resize, cv::Size(), 0.33, 0.33);
			cv::imshow("Frame", Frame);
			cv::waitKey(msec_waitKey);
		}
		if (IsLogging){
			Frame.convertTo(Frame, CV_8U, 255);
			cv::imwrite("/home/nvidia/hdd/image/" + patch::to_string(framenum) + ".jpg", Frame);
		}
			
		/******************************** Control ***********************************/
		if (SKIPorTHROWER == SKIP){
			if (IsValid >= thr_valid && IsChanging){
				if (IsCalibration){
					
					Rmat[FAR] = Epi[0] * Rmat[NEAR];
					cv::Rodrigues(Rmat[FAR], Rvec[FAR]);
					Tvec[FAR] = Epi[1] + Epi[0] * Tvec[NEAR];
					vec3 = -Rmat[FAR].inv() * Tvec[FAR];
					cam_pos[FAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[FAR].inv();
					cam_dir[FAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

					cam_mode = FAR;
					quad[0] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[1] - pad_hog), 0);
					quad[1] = getimgbywld(cv::Point2f(W - 0.5*d, transition[1] - pad_hog), 0);
					quad[2] = getimgbywld(cv::Point2f(W - 0.5*d, transition[0] + pad_mid), 0);
					quad[3] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[0] + pad_mid), 0);
					ROI_FAR = cv::Mat::zeros(imgsize[FAR], CV_8UC1);
					for (i = 100; i < ROI_FAR.cols - 100; i++){
						for (j = 0; j < ROI_FAR.rows; j++){
							if (cv::pointPolygonTest(quad, cv::Point2f(i, j), false) == 1)
								ROI_FAR.at<uchar>(j, i) = 1;
						}
					}

					Rmat[FFAR] = Epi[2] * Rmat[NEAR];
					cv::Rodrigues(Rmat[FFAR], Rvec[FFAR]);
					Tvec[FFAR] = Epi[3] + Epi[2] * Tvec[NEAR];
					vec3 = -Rmat[FFAR].inv() * Tvec[FFAR];
					cam_pos[FFAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
					mat3 = Rmat[FFAR].inv();
					cam_dir[FFAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

					cam_mode = FFAR;
					quad[0] = getimgbywld(cv::Point2f(-W + 0.5*d, D_hog_ + pad_hog_), 0);
					quad[1] = getimgbywld(cv::Point2f(W - 0.5*d, D_hog_ + pad_hog_), 0);
					quad[2] = getimgbywld(cv::Point2f(W - 0.5*d, transition[0] - pad_mid), 0);
					quad[3] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[0] - pad_mid), 0);
					ROI_FFAR = cv::Mat::zeros(imgsize[FFAR], CV_8UC1);
					for (i = 0; i < ROI_FFAR.cols; i++){
						for (j = 0; j < ROI_FFAR.rows; j++){
							if (cv::pointPolygonTest(quad, cv::Point2f(i, j), false) == 1)
								ROI_FFAR.at<uchar>(j, i) = 1;
						}
					}
					
					cam_mode = NEAR;
		
					IsCalibration = false;
					thr_valid = 2;
					IsValid = 0;
					
					//is_ChangeNEAR();

				}
				else if (IsDetecting){
			
					thrownstonenum = stonenum;
					kalman[thrownstonenum].init(4, 2, 0);
					kalman[thrownstonenum].transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 0.01, 0, 0, 1, 0, 0.01, 0, 0, 1, 0, 0, 0, 0, 1);
					//kalman[thrownstonenum].transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
					kalman[thrownstonenum].measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
					cv::setIdentity(kalman[thrownstonenum].processNoiseCov, cv::Scalar(1e-5));
					kalman[thrownstonenum].processNoiseCov.at<float>(2, 2) = 0;
					kalman[thrownstonenum].processNoiseCov.at<float>(3, 3) = 0;
					cv::setIdentity(kalman[thrownstonenum].measurementNoiseCov, cv::Scalar(1e-7));
					cv::setIdentity(kalman[thrownstonenum].errorCovPost, cv::Scalar(1.0));
					kalman[thrownstonenum].statePost.at<float>(0, 0) = 0;
					kalman[thrownstonenum].statePost.at<float>(1, 0) = farrange;
					kalman[thrownstonenum].statePost.at<float>(2, 0) = 0;
					kalman[thrownstonenum].statePost.at<float>(3, 0) = 0;

					color[thrownstonenum] = cv::Scalar(0, 0, 0);

					IsSending = true;
					if (!IsCommunicating){
						cv::Mat temp = cv::Mat::ones(30, 200, CV_32FC1);
						cv::putText(temp, "Press any key to start", cv::Point(10, temp.rows-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, 2);
						cv::imshow("", temp);
						cv::waitKey();
						IsRedetecting = true;
						cam_mode = NEAR;
						IsThrown = true;	
						framenum_thrown = framenum;
						if (msec_thrown == -1){
							msec_thrown = getmsec();
							if (!is_ChangeNEAR()){
								printf("error: unable to change near camera\n");
								return 0;
							}
						}
					}
					IsDetecting = false;
					IsValid = 0;
				}
				else if (IsThrown){	
					if (msec_frame[framenum] - msec_frame[framenum_thrown] - msec_sending > period_sweeper_communicating)
						IsSending = true;
				}
			}
			if (IsThrown){
				msec_tracking = getmsec() - msec_thrown;
				if (IsRealtime && IsRealEnd){
					minVal = fabs(D_hog - wldstone[thrownstonenum][framenum_thrown + 1].y);
					framenum_reach = framenum_thrown + 1;
					for (i = framenum_thrown + 1; i < framenum; i++){
						if(minVal > fabs(D_hog - wldstone[thrownstonenum][i].y)){
							minVal = fabs(D_hog - wldstone[thrownstonenum][i].y);
							framenum_reach = i;
						}
					}
					
					if (D_hog + 500 < wldstone[thrownstonenum][framenum_reach].y || wldstone[thrownstonenum][framenum_reach].y == -1)
						msec_hog2hog = -1;
					else
						msec_hog2hog = msec_frame[framenum_reach] - msec_thrown;

					if (msec_hog2hog < 5000 || msec_hog2hog > 20000)
						msec_hog2hog = -1;

					if (IsCollided){
						pos_thrown.x = -1;
						pos_thrown.y = -1;
						printf("+++++Collision occured+++++\n");
					}
					else {
						pos_thrown.x = wldstone[thrownstonenum][framenum - frameunit].x;
						pos_thrown.y = wldstone[thrownstonenum][framenum - frameunit].y;
					}						
					trajectory.open("/home/nvidia/Robot_Data/trajectory/trajectory1.txt");
					for (i = framenum_thrown; i < framenum; i++){
						if (wldstone[thrownstonenum][i].y > 0){
							trajectory << msec_frame[i] - msec_frame[framenum_thrown];
							trajectory << " ";
							trajectory << wldstone[thrownstonenum][i].x;
							trajectory << " ";
							trajectory << wldstone[thrownstonenum][i].y;
							trajectory << "\n";
						}
					}
					trajectory << -1;
					trajectory << " ";
					trajectory << pos_thrown.x;
					trajectory << " ";
					trajectory << pos_thrown.y;
					trajectory << "\n";
					trajectory.close();
					printf("Save Trajectory\n");

					printf("pos_thrown: %d, %d\n", (int)pos_thrown.x, (int)pos_thrown.y);					
					printf("hog2hog : %d\n", msec_hog2hog);
					printf("hogline reached x : %d\n", (int)wldstone[thrownstonenum][framenum_reach].x);					
					if (IsCommunicating){
						SendPacketStoneCnt(sock, -1, pos, team);
						printf("Send Signal of Saving\n");
						usleep(400000);
						SendPacketInfoResult(sock, (long)wldstone[thrownstonenum][framenum_reach].x, (long)msec_hog2hog, pos_thrown);
						usleep(1000000);
						for (i = 0; i < stonenum; i++){
							team[i] = (float)color[i][1];
							pos[i].x = wldstone[i][framenum].x;
							pos[i].y = wldstone[i][framenum].y;
							printf("stone %d: %d, %d\n", i, (int)wldstone[i][framenum].x, (int)wldstone[i][framenum].y);
						}
						SendPacketStoneCnt(sock, stonenum, pos, team);
						printf("Send Stone Info\n");
					}
					Sheet_draw = Sheet.clone();
					for (i = framenum_thrown; i < framenum; i++){
						if (wldstone[thrownstonenum][i].y > 0){
							displaypoint = displayscale*cv::Point(wldstone[thrownstonenum][i].y, wldstone[thrownstonenum][i].x) + display_origin;
							cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5, color[thrownstonenum], -1);
							cv::circle(Sheet_draw, displaypoint, displayscale*d*0.5 - 1, cv::Scalar(0, 0, 0), 2);
						}
					}
					cv::imwrite("/home/nvidia/Robot/trajectory.png", Sheet_draw*255); 
					cv::imshow("trajectory", Sheet_draw);
					cv::waitKey(msec_waitKey);
					break;
				}
				else if (IsRealtime && !IsEnd){
					// send stone pos to sweeper
				}
			}

			if (IsDetectingSweeper){

			}
		}
		else if (SKIPorTHROWER == THROWER){
			if (IsValid >= thr_valid){
				IsSending = true;
				IsValid = 0;
				if (!IsCommunicating){
					if (IsChanging){
						IsRunning = true;
						IsPositioning = false;
						msec_startrun = getmsec();
					}
					else
						IsPositioning = true;
				}
				else
					IsPositioning = false;
			}
			if (IsRunning && msec_release < getmsec())
				IsSending = true;
			if (IsThrown){
				usleep(10000);
				break;
			}
		}
		IsWarning = 0;
		IsError = 0;
		/****************************************************************************/

		/******************************* Communication **********************************/
		if (IsCommunicating && IsSending){
			if (SKIPorTHROWER == SKIP){
				if (IsThrown){
					for (i = 0; i < 5; i++){
						if (wldstone[thrownstonenum][framenum - i].y >= 0){
							pos_sending[i].x = wldstone[thrownstonenum][framenum - i].x;
							pos_sending[i].y = wldstone[thrownstonenum][framenum - i].y;
							msec_sending_[i] = msec_frame[framenum - i] - msec_frame[framenum_thrown];
						}
						else {
							pos_sending[i].x = -1;
							pos_sending[i].y = -1;
							msec_sending_[i] = -1;
						}
					}
					msec_sending = msec_sending_[4];
					SendPacketPacStoneInfo(sock, pos_sending[0], (float)msec_sending_[0], pos_sending[1], (float)msec_sending_[1], pos_sending[2], (float)msec_sending_[2], pos_sending[3], (float)msec_sending_[3], pos_sending[4], (float)msec_sending_[4]);
				}
				else {
					for (i = 0; i < num_stone; i++){
						team[i] = (float)color[i][1];
						pos[i].x = wldstone[i][framenum].x;
						pos[i].y = wldstone[i][framenum].y;
					}
					SendPacketStoneCnt(sock, num_stone, pos, team);
					printf("Send Stone Info\n");
				}
			}
			else if (SKIPorTHROWER == THROWER){
				if (IsRunning)
					hogOffs = -1;
				else
					hogOffs = 0;
				
				if (IsRunning && msec_running > msec_timeout + msec_blind){
					SendPacketRobotInfo(sock, angle, robotPos, hogDist, hogOffs);
					printf("Send Release Flag by Timeout\n");
				}
				else if (IsRunning && hogDist < Releasedist){
					SendPacketRobotInfo(sock, angle, robotPos, hogDist, hogOffs);
					printf("Send Release Flag by Hogline Distance\n");
				}
				else if (IsRunning && msec_release < getmsec() && hogDist < hogDist0){
					SendPacketRobotInfo(sock, angle, robotPos, hogDist, hogOffs);
					printf("Send Release Flag by Prediction\n");
				}
				else if (!IsRunning){
					SendPacketRobotInfo(sock, angle_mean - angle_bias, robotPos, hogDist, hogOffs);
					printf("Send robot pos and angle\n");
				}
			}
			IsSending = false;
		}

		//printf("true latency : %d\n", getmsec() - msec0);

		/****************************************************************************/
	}

	for (i = 0; i < 3; i++){
		success = is_ExitCamera(Cams[i]);
		if (success != IS_SUCCESS)
			return 0;
	}
	pthread_join(p_thread, (void **)&stat);
	close(sock);
	
	return 0;
}

int is_OpenCamera(int CamType, HIDS* Cam, char** Mem){		//camera open

	int success, nNumCam;

	int img_width, img_height;

	UINT nPixelClock;

	double ex_time, ex_min, ex_max, ex_inc;

	int gamma;

	int bitspixel;

	int m_lMemoryId;

	double mintime, maxtime, dtime, timeinterval, maxFPS;

	UINT foc_min, foc_max, foc_inc, foc_set;

	double dEnable, mastergain;

	int zero_value;

	success = is_GetNumberOfCameras(&nNumCam);		//returns the number of uEye cameras connected to the PC
	if (success != IS_SUCCESS) {		//is_GetNumberOfCameras >> Function excuted successfully
		printf("GetNumberOfCamera fail");	
		return 0;
	}
	
	UEYE_CAMERA_LIST* pucl;		//is_GetCameraList >> Handle to the UEYE_CAMERA_LIST structure
	pucl = (UEYE_CAMERA_LIST*) new BYTE[sizeof(DWORD)+nNumCam * sizeof(UEYE_CAMERA_INFO)];	//memory size set?
	pucl->dwCount = nNumCam;		//has to initialized with the number of cameras connected to system
	success = is_GetCameraList(pucl);
	if (success != IS_SUCCESS) {		//is_GetCameraList >> Function excuted successfully
		printf("GetCameraList fail");
		return 0;
	}

	int Idx;
	for (Idx = 0; Idx < 4; Idx++) {		//near, far, thorow camera serial number setting
		if (pucl->uci[Idx].SerNo == Serialnum[CamType])		//is_GetCameraList >> uci(structure) >> serial number
			break;
	}
	
	if (Idx == 4) {			//camera number exceed 4
		printf("Camera detect error\n");
		return 0;
	}

	*Cam = pucl->uci[Idx].dwDeviceID | IS_USE_DEVICE_ID;		//is_GetCameraList, is_DeviceInfo? device id 
	success = is_InitCamera(Cam, NULL);		//starts the driver and estabilishes the connection to the camera.
	if (success != IS_SUCCESS) {
		printf("InitCamera fail");
		return 0;
	}
		
	SENSORINFO pInfo;
	success = is_GetSensorInfo(*Cam, &pInfo);
	if (success != IS_SUCCESS)
		return 0;
	img_width = pInfo.nMaxWidth;
	img_height = pInfo.nMaxHeight;
	//std::cout << "Sensor Image Size : " << img_width << ", " << img_height << std::endl;

	UINT nRange[3];
	success = is_PixelClock(*Cam, IS_PIXELCLOCK_CMD_GET_RANGE, (void*)nRange, sizeof(nRange));
	if (success != IS_SUCCESS)
		return 0;
	nRange[2] = 1;
	//std::cout << "Min Pixel Clock : " << nRange[0] << std::endl;
	//std::cout << "Max Pixel Clock : " << nRange[1] << std::endl;
	//std::cout << "Pixel Clock Unit : " << nRange[2] << std::endl;
	
	if (CamType == NEAR)
		nPixelClock = 81;
	else
		nPixelClock = 64;
	if (nRange[1] < nPixelClock || nRange[0] > nPixelClock)
		return 0;
	else
		nPixelClock = nRange[0] + ((nPixelClock - nRange[0]) / nRange[2])*nRange[2];
	success = is_PixelClock(*Cam, IS_PIXELCLOCK_CMD_SET, (void*)&nPixelClock, sizeof(nPixelClock));
	if (success != IS_SUCCESS)
		return 0;
	success = is_PixelClock(*Cam, IS_PIXELCLOCK_CMD_GET, (void*)&nPixelClock, sizeof(nPixelClock));
	if (success != IS_SUCCESS)
		return 0;
	//std::cout << "Pixel Clock : " << nPixelClock << std::endl;

	if (CamType == NEAR){

		dEnable = 1;
		success = is_SetAutoParameter(*Cam, IS_SET_ENABLE_AUTO_SENSOR_GAIN, &dEnable, 0);
		if (success != IS_SUCCESS) {						
			printf("SetAutoParameter -> auto gain set fail\n");
			return 0;
		}

		success = is_Focus(*Cam, FOC_CMD_SET_DISABLE_AUTOFOCUS, NULL, 0);			
		if (success != IS_SUCCESS)
			return 0;
		success = is_Focus(*Cam, FOC_CMD_GET_MANUAL_FOCUS_MIN, (void*)&foc_min,sizeof(UINT));
		if (success != IS_SUCCESS)
			return 0;
		success = is_Focus(*Cam, FOC_CMD_GET_MANUAL_FOCUS_MAX, (void*)&foc_max,sizeof(UINT));
		if (success != IS_SUCCESS)
			return 0;
		success = is_Focus(*Cam, FOC_CMD_GET_MANUAL_FOCUS_INC, (void*)&foc_inc,sizeof(UINT));
		if (success != IS_SUCCESS)
			return 0;
		foc_set = 68;
		success = is_Focus(*Cam, FOC_CMD_SET_MANUAL_FOCUS, (void*)&foc_set,sizeof(UINT));
		if (success != IS_SUCCESS)
			return 0;
		success = is_Focus(*Cam, FOC_CMD_GET_MANUAL_FOCUS, (void*)&foc_set,sizeof(UINT));
		if (success != IS_SUCCESS)
			return 0;

		zero_value = 0;
		success = is_Sharpness(*Cam, SHARPNESS_CMD_SET_VALUE, (void*)&zero_value, sizeof(int));
		if (success != IS_SUCCESS){
			printf("Sharpness - setting sharpness fail\n");
			return 0;
		}

		success = is_Saturation(*Cam, SATURATION_CMD_SET_VALUE, (void*)&zero_value, sizeof(int));
		if (success != IS_SUCCESS){
			printf("Saturation - setting saturation fail\n");
			return 0;
		}

		UINT count;
		UINT bytesNeeded = sizeof(IMAGE_FORMAT_LIST);
		success = is_ImageFormat(*Cam, IMGFRMT_CMD_GET_NUM_ENTRIES, &count, sizeof(count));
		if (success != IS_SUCCESS){
			printf("ImageFormat -> entries fail\n");
			return 0;
		}
		bytesNeeded += (count - 1) * sizeof(IMAGE_FORMAT_INFO);
		void* ptr = malloc(bytesNeeded);

		pformatList = (IMAGE_FORMAT_LIST*) ptr;
		pformatList->nSizeOfListEntry = sizeof(IMAGE_FORMAT_INFO);
		pformatList->nNumListElements = count;

		success = is_ImageFormat(*Cam, IMGFRMT_CMD_GET_LIST, pformatList, bytesNeeded);			//Image Format 
		//printf("format id : %d\n", nWidth);
		if (success != IS_SUCCESS){
			printf("ImageFormat fail\n");
			return 0;
		}

		int FormatID;
		if (IsSnapshot[NEAR]){
			success = is_SetExternalTrigger(*Cam, IS_SET_TRIGGER_SOFTWARE);
			if (success != IS_SUCCESS) {			//Off the trigger mode
				printf("SetExternalTrigger fail\n");
				return 0;
			}
			//printf("trigger mode on\n");
			FormatID = 0;
		}
		else {
			success = is_SetExternalTrigger(*Cam, IS_SET_TRIGGER_OFF);
			if (success != IS_SUCCESS) {			//Off the trigger mode
				printf("SetExternalTrigger fail\n");
				return 0;
			}
			//printf("freerun mode on\n");
			FormatID = 4;
		}

		formatInfo = pformatList->FormatInfo[FormatID];	
		//printf("format id : %d\n", formatInfo.nFormatID);
		img_width = formatInfo.nWidth;
		img_height = formatInfo.nHeight;
		//printf("width : %d, height : %d\n", img_width, img_height);

		//printf("format id : %d\n",formatInfo.nFormatID);
		//printf("format Capture mode : %d\n",formatInfo.nSupportedCaptureModes);
		success = is_ImageFormat(*Cam, IMGFRMT_CMD_SET_FORMAT, &formatInfo.nFormatID, sizeof(formatInfo.nFormatID));
		if (success != IS_SUCCESS){
			printf("ImageFormat -> setting fail\n");
			return 0;
		}


	}
	else {
		success = is_Exposure(*Cam, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, (void*)&ex_min, sizeof(double));
		if (success != IS_SUCCESS)
			return 0;
		success = is_Exposure(*Cam, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, (void*)&ex_max, sizeof(double));
		if (success != IS_SUCCESS)
			return 0;
		success = is_Exposure(*Cam, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, (void*)&ex_inc, sizeof(double));
		if (success != IS_SUCCESS)
			return 0;
		//std::cout << "Min Exposure Time : " << ex_min << std::endl;
		//std::cout << "Max Exposure Time : " << ex_max << std::endl;
		//std::cout << "Exposure Time Unit : " << ex_inc << std::endl;
		if (CamType == 	THROW)
			ex_time = 9;
		else
			ex_time = 10;
		if (ex_min > ex_time || ex_max < ex_time)
			return 0;
		else
			ex_time = ex_min + (int)((ex_time - ex_min) / ex_inc)*ex_inc;
		success = is_Exposure(*Cam, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*)&ex_time, sizeof(double));
		if (success != IS_SUCCESS)
			return 0;
		success = is_Exposure(*Cam, IS_EXPOSURE_CMD_GET_EXPOSURE, (void*)&ex_time, sizeof(double));
		if (success != IS_SUCCESS)
			return 0;
		//std::cout << "Exposure Time : " << ex_time << std::endl;

		success = is_SetExternalTrigger(*Cam, IS_SET_TRIGGER_SOFTWARE);
		if (success != IS_SUCCESS) {			//Off the trigger mode
			printf("SetExternalTrigger fail\n");
			return 0;
		}
		//printf("trigger mode on\n");
	
	}


	size_sensor[CamType].width = img_width;
	size_sensor[CamType].height = img_height;

	/*gamma=180;
	success = is_Gamma(*Cam, IS_GAMMA_CMD_SET, (void*)&gamma, sizeof(gamma));
	if (success != IS_SUCCESS)
		return 0;*/

	success = is_SetColorMode(*Cam, IS_CM_BGR8_PACKED);
	if (success != IS_SUCCESS)
		return 0;
	bitspixel = 8 * 3;

	success = is_AllocImageMem(*Cam, (int)img_width, (int)img_height, bitspixel, &(*Mem), &m_lMemoryId);
	if (success != IS_SUCCESS)
		return 0;
	success = is_SetImageMem(*Cam, *Mem, m_lMemoryId);
	if (success != IS_SUCCESS)
		return 0;

	success = is_GetFrameTimeRange(*Cam, &mintime, &maxtime, &dtime);
	if (success != IS_SUCCESS)
		return 0;
	//std::cout << "Max Frametime : " << maxtime << std::endl;
	//std::cout << "Min Frametime : " << mintime << std::endl;
	//std::cout << "Frametime Unit : " << dtime << std::endl;
	timeinterval = 1 / fps;
	if (timeinterval > maxtime || timeinterval < mintime)
		return 0;
	else
		timeinterval = mintime + (int)((timeinterval - mintime) / dtime)*dtime;
	//std::cout << "Max Frametime (Freerun) : " << timeinterval << std::endl;
	success = is_SetFrameRate(*Cam, 1/timeinterval, &maxFPS);
	if (success != IS_SUCCESS)
		return 0;
	//std::cout << "Max FPS (Freerun) : " << maxFPS << std::endl;

	success = is_SetHardwareGain(*Cam, gain[CamType][0], gain[CamType][1], gain[CamType][2], gain[CamType][3]);
	//std::cout << "Master Gain : " << gain[CamType][0] << std::endl;
	//std::cout << "Red Gain : " << gain[CamType][1] << std::endl;
	//std::cout << "Green Gain : " << gain[CamType][2] << std::endl;
	//std::cout << "Blue Gain : " << gain[CamType][3] << std::endl;
	if (success != IS_SUCCESS)
		return 0;

	std::cout << "Camera " << pucl->uci[Idx].SerNo << " Enabled" << std::endl;		//basic - camera load

	return 1;
}


void *RecvThreadPoint(void *param) {

	int fd_max, fd_num, str_len;

	FD_ZERO(&reads);
	FD_SET(sock, &reads);
	fd_max = sock;
	//이벤트 처리용 루프
	while (1) {
		cpy_reads = reads;
		timeout.tv_sec = 10;
		timeout.tv_usec = 10000;
		
		if ((fd_num = select(fd_max + 1, &cpy_reads, 0, 0, &timeout)) == -1)
			break;
		if (fd_num == 0)
			continue;

		for (int i = 0; i < fd_max + 1; i++){
			if (FD_ISSET(i, &cpy_reads)){
				if (i == sock){
					str_len = read(i, msg, MAX_MSG_LEN);
					if (str_len != 0){
						if (strcmp(msg, "vs") == 0)
							send(sock, msg, 256, 0);
						if ((unsigned char)msg[0] != 0xAA || (unsigned char)msg[1] != 0x00 || msg[2] < STONE_CNT || msg[2] > INFO_RESULT)
							continue;
						else if (msg[2] == PRE_MODE){

							PacketCvt tmp;
							int idx = 3;
							for (int i = 0; i < 4; i++)
								tmp.ch[i] = msg[idx++];
							if (tmp.f == 0){

								IsSendingAck = 1;
							
								for (int i = 0; i < 4; i++)
									tmp.ch[i] = msg[idx++];
								if (tmp.f == ROBOT_ID){
									std::cout << "THROWER mode!" << std::endl;
									SKIPorTHROWER = THROWER;
								}
								else {
									std::cout << "SKIP mode!" << std::endl;
									SKIPorTHROWER = SKIP;
								}

								txtfile.open("/home/nvidia/Robot/Control.txt", std::fstream::out);
								txtfile << "IsCommunicating" << std::endl;
								txtfile << IsCommunicating << std::endl;
								txtfile << "SKIPorTHROWER" << std::endl;
								txtfile << SKIPorTHROWER << std::endl;
								txtfile << "IsChanging" << std::endl;
								txtfile << IsChanging << std::endl;
								txtfile << "IsRealtime" << std::endl;
								txtfile << IsRealtime << std::endl;
								txtfile << "IsDisplaying" << std::endl;
								txtfile << IsDisplaying << std::endl;
								txtfile << "msec_waitKey" << std::endl;
								txtfile << msec_waitKey << std::endl;
								txtfile << "msec_blind" << std::endl;
								txtfile << msec_blind << std::endl;
								txtfile << "msec_timeout" << std::endl;
								txtfile << msec_timeout << std::endl;
								txtfile << "Releasedist" << std::endl;
								txtfile << Releasedist << std::endl;
								txtfile << "angle_bias" << std::endl;
								txtfile << angle_bias << std::endl;
								txtfile << "msec_cameradelay" << std::endl;
								txtfile << msec_cameradelay << std::endl;
								txtfile << "IsRecalibration" << std::endl;
								txtfile << IsRecalibration << std::endl;
								txtfile << "cam_mode_distant" << std::endl;
								txtfile << cam_mode_distant << std::endl;
							
								txtfile.close();
							}
						}
						else if (msg[2] == CALL_ROBOT_INFO && SKIPorTHROWER == THROWER){
							
							IsSendingAck = 2;
							IsPositioning = true;
							if (msg[3]){
								std::cout << "Robot Info Giveup Call!" << std::endl;
								IsLying = true;	
							}
							else
								std::cout << "Robot Info Call!" << std::endl;
						}
						else if (msg[2] == SPEED_PROF && SKIPorTHROWER == THROWER){

							IsSendingAck = 5;

							std::cout << "Speed Info Receive Success!" << std::endl;
							PacketCvt tmp;
							int idx = 3;
							for (int i = 0; i < 4; i++)
								tmp.ch[3 - i] = msg[idx++];
							Releaseangle = tmp.f;
							for (int i = 0; i < 4; i++)
								tmp.ch[3 - i] = msg[idx++];
							Releasespeed = tmp.f;
							std::cout << "Releaseangle : " << Releaseangle << std::endl;
							std::cout << "Releasespeed : " << Releasespeed << std::endl;
						}
						else if (msg[2] == START){
							if(SKIPorTHROWER == THROWER){
								std::cout << "Start Running!" << std::endl;
								msec_startrun = getmsec();
								usleep(1000*msec_blind);
								msec_release = getmsec_release(hogDist0);
								IsRunning = true;
								/*thr_id = pthread_create(&p_thread_release, NULL, ReleaseDecision, NULL);
								if (thr_id < 0)	{
									perror("thread create error : ");
									exit(0);
								}*/
							}
							else if(SKIPorTHROWER == SKIP){
								std::cout << "Start Running!" << std::endl;
								msec_startrun = getmsec();
								IsRedetecting = true;
								IsRunning = true;
							}
						}
						else if (msg[2] == CALL_STONE_INFO && SKIPorTHROWER == SKIP){
							IsSendingAck = 3;

							std::cout << "Tracking start!" << std::endl;	
							IsDetecting = true;
							stonenum = 0;
						}
						else if (msg[2] == FLAG){
							std::cout << "Stone Throwed" << std::endl;
							IsDetecting = false;	
							msec_thrown = getmsec() + msec_cameradelay;
							if (SKIPorTHROWER == SKIP){
								framenum_thrown = framenum;
								IsThrown = true;
							}
							else
								IsThrown = true;
						}
						else if (msg[2] == INFO_TIME){
							char* shotname_c = &msg[3];
							shotname = shotname_c;
							std::cout << "\n\nshotname : " << shotname << "\n\n" << std::endl;
						}
						else if (msg[2] == EMERGENCY || msg[2] == RESTART){
							std::cout << "Restart!" << std::endl;
							exit(0);
						}
					}
					else {
						close(sock);
						sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
						int re = 0;
						do {
							if (re == -1) {
								printf("\rConnect Fail!...");
								usleep(1000000);
							}
							re = connect(sock, (struct sockaddr *)&servaddr, sizeof(servaddr));//연결 요청
						} while (re == -1);
						FD_ZERO(&reads);
						FD_SET(sock, &reads);
						fd_max = sock;
					}
				}
			}
		}
	}
	close(sock);
}

void SendPacketStoneCnt(SOCKET s, int cnt, Point* posArr, float* isRedArr) {
	char* data = new char[3 + 1];
	int idx = 0;

	// Header : 0xAA, 0x00 Œ³Á€
	data[idx++] = 0xAA;
	data[idx++] = 0x00;

	// Packet Type : STONE_CNT
	data[idx++] = STONE_CNT;

	// Set Data : Stone cnt
	data[idx++] = (char)cnt;

	send(s, data, idx, 0);

	for (int i = 0; i < cnt; i++) {
		usleep(1000000);
		SendPacketStoneInfo(s, posArr[i], isRedArr[i]);
	}

	delete data;
}

void SendPacketInfoResult(SOCKET s, long release, long arrive, Point pos) {
	char* data = new char[3 + sizeof(release) + sizeof(arrive) + sizeof(pos) + 3];
	int idx = 0;

	// Header : 0xAA, 0x00 
	data[idx++] = 0xAA;
	data[idx++] = 0x00;

	// Packet Type : INFO_RESULT
	data[idx++] = INFO_RESULT;

	// Set Data : release
	char * temp = (char*)&release;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	data[idx++] = '%';

	temp = (char*)&arrive;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	data[idx++] = '%';

	temp = (char*)&pos.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	data[idx++] = '%';

	temp = (char*)&pos.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	send(s, data, idx, 0);

	delete data;
}


void SendPacketStoneInfo(SOCKET s, Point pos, float isRed) {
	char* data = new char[3 + sizeof(pos) + sizeof(isRed)];
	int idx = 0;

	// Header : 0xAA, 0x00 Œ³Á€
	data[idx++] = 0xAA;
	data[idx++] = 0x00;

	// Packet Type : STONE_INFO
	data[idx++] = STONE_INFO;

	// Set Data : Stone Info
	char * temp = (char*)&pos.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&isRed;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	send(s, data, idx, 0);

	delete data;
}

void SendPacketRobotInfo(SOCKET s, float angle, Point pos, float hogDist, float hogOffs) {
	unsigned char checkSum = 0;
	char* data = new char[3 + sizeof(angle) + sizeof(pos) + sizeof(hogDist) + sizeof(hogOffs) + sizeof(checkSum)];
	int idx = 0;

	// Header : 0xAA, 0x00 Œ³Á€
	data[idx++] = 0xAA;
	data[idx++] = 0x00;

	// Packet Type : ROBOT_INFO
	data[idx++] = ROBOT_INFO;

	// Set Data : Robot Info
	char * temp = (char*)&angle;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&pos.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&hogDist;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&hogOffs;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	for (int l = 0; l < idx; l++)
		checkSum |= data[l];

	data[idx++] = checkSum;

	send(s, data, idx, 0);

	delete data;
}

void SendPacketStoneInfoAck(SOCKET s, char type) {
	char data[4];
	int idx = 0;

	// Header : 0xAA, 0x00 Œ³Á€
	data[idx++] = 0xAA;
	data[idx++] = 0x00;

	// Packet Type : FLAG
	data[idx++] = STONE_INFO_ACK;

	// 0:premode,	1: robot info,	2 : call stone info,	3: stone info,	4: release
	data[idx++] = type;

	send(s, data, idx, 0);
}

void SendPacketPacStoneInfo(SOCKET s,Point pos1, float isRed1, Point pos2, float isRed2, Point pos3, float isRed3
						   , Point pos4, float isRed4, Point pos5, float isRed5) {
	
	char* data = new char[3 + sizeof(pos1) + sizeof(isRed1) + sizeof(pos2) + sizeof(isRed2) + sizeof(pos3) + sizeof(isRed3)
	 + sizeof(pos4) + sizeof(isRed4) + sizeof(pos5) + sizeof(isRed5)];
	int idx = 0;

	// Header : 0xAA, 0x00 Œ³Á€
	data[idx++] = 0xAA;
	data[idx++] = 0x00;

	// Packet Type : STONE_INFO
	data[idx++] = PAC_STONE_INFO;

	// Set Data : Stone Info
	char * temp = (char*)&pos1.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos1.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&isRed1;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&pos2.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos2.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&isRed2;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&pos3.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos3.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&isRed3;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&pos4.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos4.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&isRed4;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&pos5.x;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];
	temp = (char*)&pos5.y;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	temp = (char*)&isRed5;
	data[idx++] = temp[0];
	data[idx++] = temp[1];
	data[idx++] = temp[2];
	data[idx++] = temp[3];

	//data[idx++] = isRed;

	send(s, data, idx, 0);

	delete data;
}

cv::Point2f getwldbyimg(cv::Point2f imgpoint, double z){

	cv::Mat imgpoint_(1, 1, CV_64FC2);
	imgpoint_.at<cv::Vec2d>(0, 0) = cv::Vec2d(imgpoint.x, imgpoint.y);
	cv::undistortPoints(imgpoint_, imgpoint_, Int[cam_mode], Dist[cam_mode]);

	cv::Mat imgpoint_h = cv::Mat::ones(3, 1, CV_64FC1);
	imgpoint_h.at<double>(0, 0) = imgpoint_.at<double>(0, 0);
	imgpoint_h.at<double>(1, 0) = imgpoint_.at<double>(0, 1);

	double lambda = z + Rmat[cam_mode].at<double>(0, 2) * Tvec[cam_mode].at<double>(0, 0) + Rmat[cam_mode].at<double>(1, 2) * Tvec[cam_mode].at<double>(1, 0) + Rmat[cam_mode].at<double>(2, 2) * Tvec[cam_mode].at<double>(2, 0);
	lambda /= Rmat[cam_mode].at<double>(0, 2) * imgpoint_h.at<double>(0, 0) + Rmat[cam_mode].at<double>(1, 2) * imgpoint_h.at<double>(1, 0) + Rmat[cam_mode].at<double>(2, 2) * imgpoint_h.at<double>(2, 0);

	cv::Mat wldpoint = Rmat[cam_mode].inv() * (lambda * imgpoint_h - Tvec[cam_mode]);

	return cv::Point2f(wldpoint.at<double>(0, 0), wldpoint.at<double>(1, 0));
}

cv::Point2f getimgbywld(cv::Point2f wldpoint, double z){

	std::vector<cv::Point2f> imgpoint;
	std::vector<cv::Point3f> w3dpoint;
	w3dpoint.push_back(cv::Point3f(wldpoint.x, wldpoint.y, z));
	cv::projectPoints(w3dpoint, Rvec[cam_mode], Tvec[cam_mode], Int[cam_mode], Dist[cam_mode], imgpoint);
	return imgpoint[0];
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
			if (pt.x*pt.x + pt.y*pt.y < 0.25*d_*d_*0.5 && hsv[1].at<float>(y, x) > 0.1){
				if (hsv[0].at<float>(y, x) > 0.3)
					score_r++;
				else if (hsv[0].at<float>(y, x) > 0.1 && hsv[0].at<float>(y, x) < 0.3)
					score_y++;
			}
			/*if (pt.x*pt.x + pt.y*pt.y < 0.25*d_*d_){
				if (hsv[0].at<float>(y, x) > 0.1){
					printf("hsv[0]: %.2f, hsv[1]: %.2f\n", hsv[0].at<float>(y, x), hsv[1].at<float>(y, x));
				}
			}*/
		}
	}


	if (score_r >= score_y)
		return cv::Scalar(0, 0, 1);
	else if (score_r < score_y)
		return cv::Scalar(0, 1, 1);
}

cv::Point2f getintersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2){
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;
	float cross = d1.x*d2.y - d1.y*d2.x;
	if (cross == 0)
		return cv::Point2f(0, 0);
	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	return o1 + d1 * t1;
}

cv::Point2f getgndbytrw(cv::Point2f trwpoint){
	double lambda = 1 / (Trw2Gnd.at<float>(2, 0)*trwpoint.x + Trw2Gnd.at<float>(2, 1)*trwpoint.y + Trw2Gnd.at<float>(2, 2));
	return lambda * cv::Point2f(Trw2Gnd.at<float>(0, 0)*trwpoint.x + Trw2Gnd.at<float>(0, 1)*trwpoint.y + Trw2Gnd.at<float>(0, 2), Trw2Gnd.at<float>(1, 0)*trwpoint.x + Trw2Gnd.at<float>(1, 1)*trwpoint.y + Trw2Gnd.at<float>(1, 2));
}

cv::Point2f gettrwbygnd(cv::Point2f gndpoint){
	double lambda = 1 / (Gnd2Trw.at<float>(2, 0)*gndpoint.x + Gnd2Trw.at<float>(2, 1)*gndpoint.y + Gnd2Trw.at<float>(2, 2));
	return lambda * cv::Point2f(Gnd2Trw.at<float>(0, 0)*gndpoint.x + Gnd2Trw.at<float>(0, 1)*gndpoint.y + Gnd2Trw.at<float>(0, 2), Gnd2Trw.at<float>(1, 0)*gndpoint.x + Gnd2Trw.at<float>(1, 1)*gndpoint.y + Gnd2Trw.at<float>(1, 2));
}

std::vector<cv::Point2f> getintersection_ellipse(cv::Mat Q, cv::Point2f p, cv::Point2f p1){
	cv::Point2f dif = p1 - p;
	double a, b, c;
	a = Q.at<float>(0, 0)*dif.x*dif.x + Q.at<float>(1, 0)*dif.y*dif.y + Q.at<float>(2, 0)*dif.x*dif.y;
	b = 2*Q.at<float>(0, 0)*p.x*dif.x + 2*Q.at<float>(1, 0)*p.y*dif.y + Q.at<float>(2, 0)*(p.y*dif.x + p.x*dif.y) + Q.at<float>(3, 0)*dif.x + Q.at<float>(4, 0)*dif.y;
	c = Q.at<float>(0, 0)*p.x*p.x + Q.at<float>(1, 0)*p.y*p.y + Q.at<float>(2, 0)*p.x*p.y + Q.at<float>(3, 0)*p.x + Q.at<float>(4, 0)*p.y - 1;
	float lambda1;
	float lambda2;
	std::vector<cv::Point2f> intersectionpoints(3);

	if ( b*b - 4*a*c > 0)
		intersectionpoints[2] = cv::Point2f(1.0,1.0);
	else if ( b*b - 4*a*c == 0)
		intersectionpoints[2] = cv::Point2f(0.0,0.0);
	else {
		intersectionpoints[2] = cv::Point2f(-1.0,-1.0);
		return intersectionpoints;
	}

	lambda1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
	lambda2 = (-b - sqrt(b*b - 4*a*c)) / (2*a);
	intersectionpoints[0] = lambda1*dif + p;
	intersectionpoints[1] = lambda2*dif + p;
	return intersectionpoints;
}

std::vector<cv::Point2f> gettangentialpoint_ellipse(cv::Mat Q, cv::Point2f p){
	double A1 = Q.at<float>(3, 0) / 2 + Q.at<float>(0, 0)*p.x + Q.at<float>(2, 0)*p.y / 2;
	double B1 = Q.at<float>(4, 0) / 2 + Q.at<float>(1, 0)*p.y + Q.at<float>(2, 0)*p.x / 2;
	double C1 = (1 - 0.5*(Q.at<float>(3, 0)*p.x + Q.at<float>(4, 0)*p.y)) / B1;
	double A2 = Q.at<float>(0, 0)*B1*B1 + Q.at<float>(1, 0)*A1*A1 - Q.at<float>(2, 0)*A1*B1;
	double B2 = -(2 * Q.at<float>(1, 0)*A1*C1 - Q.at<float>(2, 0)*B1*C1 - Q.at<float>(3, 0)*B1 + Q.at<float>(4, 0)*A1);
	double C2 = Q.at<float>(1, 0)*C1*C1 + Q.at<float>(4, 0)*C1 - 1;
	double t1 = 0.5*(-B2 + sqrt(B2*B2 - 4 * A2*C2)) / A2;
	double t2 = 0.5*(-B2 - sqrt(B2*B2 - 4 * A2*C2)) / A2;

	std::vector<cv::Point2f> tangentialpoints(2);
	tangentialpoints[0] = cv::Point2f(B1*t1, C1 - A1*t1);
	tangentialpoints[1] = cv::Point2f(B1*t2, C1 - A1*t2);
	return tangentialpoints;
}

cv::Vec4f getellipseprops(cv::Mat Q){
	if (Q.at<float>(2, 0)*Q.at<float>(2, 0) - 4*Q.at<float>(0, 0)*Q.at<float>(1, 0) < 0){
		double q = 64 * ((-1 * (4 * Q.at<float>(0, 0) * Q.at<float>(1, 0) - Q.at<float>(2, 0) * Q.at<float>(2, 0))) - Q.at<float>(0, 0) * Q.at<float>(4, 0) * Q.at<float>(4, 0) +  Q.at<float>(2, 0) * Q.at<float>(3, 0) * Q.at<float>(4, 0) - Q.at<float>(1, 0) *Q.at<float>(3, 0) * Q.at<float>(3, 0)) / ((4 * Q.at<float>(0, 0) * Q.at<float>(1, 0) - Q.at<float>(2, 0) * Q.at<float>(2, 0)) * (4 * Q.at<float>(0, 0) * Q.at<float>(1, 0) - Q.at<float>(2, 0) * Q.at<float>(2, 0)));
		double s = (1 / 4.0) * sqrt( fabs(q) * sqrt(Q.at<float>(2, 0) * Q.at<float>(2, 0) + (Q.at<float>(0, 0) - Q.at<float>(1, 0)) * (Q.at<float>(0, 0) - Q.at<float>(1, 0))));
		double rmax = (1 / 8.0) * sqrt(2 * fabs(q) * sqrt(Q.at<float>(2, 0) * Q.at<float>(2, 0) + (Q.at<float>(0, 0) - Q.at<float>(1, 0)) * (Q.at<float>(0, 0) - Q.at<float>(1, 0))) - 2 * q * (Q.at<float>(0, 0) + Q.at<float>(1, 0)));
		double rmin = sqrt(rmax * rmax - s * s);
		double centerx = (Q.at<float>(2, 0) * Q.at<float>(4, 0) - 2 * Q.at<float>(1, 0) * Q.at<float>(3, 0)) / (4 * Q.at<float>(0, 0) * Q.at<float>(1, 0) - Q.at<float>(2, 0) * Q.at<float>(2, 0));
		double centery = (Q.at<float>(2, 0) * Q.at<float>(3, 0) - 2 * Q.at<float>(0, 0) * Q.at<float>(4, 0)) / (4 * Q.at<float>(0, 0) * Q.at<float>(1, 0) - Q.at<float>(2, 0) * Q.at<float>(2, 0));
		return cv::Vec4f(rmax, rmin, centerx, centery);
	}
	else
		return cv::Vec4f(0, 0, 0, 0);
}

int getmsec(){
	struct timeval mytime;
	gettimeofday(&mytime, NULL);
	return ((mytime.tv_sec - mytime0.tv_sec)%1000)*1000 + (mytime.tv_usec - mytime0.tv_usec)/1000.0;
}

int getmsec_release(float hogDist){
	double A, B, C, x;
	A = Releasespeed;
	B = 0.0001;
	C = 2;
	x = 20;
	double s, s0, z0, t;
	/*s0 = 0.4;
	z0 = 1 + B*exp(x);
	s = 0.001*((D_hog) - (D_hog - hogDist0)) / cos((180/CV_PI)*Releaseangle);
	t = (1 / C) * (x + log((1 / B) * ((z0 / (z0 - 1))*exp((C / A)*(s - s0)) - 1)));
	s = 0.001*((D_hog - hogDist) - (D_hog - hogDist0)) / cos((180/CV_PI)*Releaseangle);
	t -= (1 / C) * (x + log((1 / B) * ((z0 / (z0 - 1))*exp((C / A)*(s - s0)) - 1)));*/
	t = (hogDist - Releasedist) / (Releasespeed*cos((CV_PI / 180)*Releaseangle)) - msec_cameradelay;
	return t + getmsec();
}

cv::Rect getStoneBoundingBox(cv::Point2f ptPredicted){
	std::vector<cv::Point> stone_pts(6);
	cv::Rect stone_rect;
	stone_pts[0] = getimgbywld(ptPredicted+cv::Point2f(0,-0.5*d), 0);
	stone_pts[1] = getimgbywld(ptPredicted+cv::Point2f(-0.5*d,0), 0);
	stone_pts[2] = getimgbywld(ptPredicted+cv::Point2f(-0.5*d,0), h);
	stone_pts[3] = getimgbywld(ptPredicted+cv::Point2f(0,0.5*d), h);
	stone_pts[4] = getimgbywld(ptPredicted+cv::Point2f(0.5*d,0), h);
	stone_pts[5] = getimgbywld(ptPredicted+cv::Point2f(0.5*d,0), 0);
	stone_rect = cv::boundingRect(stone_pts);
	
	float rate = 2;
	stone_rect.x = stone_rect.x - (rate - 1)*stone_rect.width / 2;
	stone_rect.y = stone_rect.y - (rate - 1)*stone_rect.height / 2;
	stone_rect.width = stone_rect.width*rate;
	stone_rect.height = stone_rect.height*rate;

	return stone_rect;

}

void *ReleaseDecision(void *param) {

	float angle, hogDist, hogOffs;
	Point robotPos;

	while(1){
		if (msec_running > msec_timeout + msec_blind){
			hogOffs = -1;
			SendPacketRobotInfo(sock, angle, robotPos, hogDist, hogOffs);
			printf("Send Release Flag by Realtime Release Decision\n");
		}
		else
			usleep(1000);

	}
}

void onMouseEvent(int event, int x, int y, int flags, void* dstImage){
	cv::Mat image = *(cv::Mat *)dstImage;
	cv::Mat mask = cv::Mat::zeros(image.size(),CV_8U);
	cv::Point cancelPt;
	cv::Mat image0 = image.clone();
	switch(event){
		case CV_EVENT_LBUTTONDOWN:
			std::cout<<cv::Point(x, y)<<std::endl;
			imgpoint_pattern.push_back(cv::Point(x, y));
			break;
		case CV_EVENT_RBUTTONDOWN:
			std::cout<<"Cancel imgPoint"<<std::endl;
			imgpoint_pattern.pop_back();
			break;
	}
	for(int i = 0; i < imgpoint_pattern.size(); i++){
		cv::circle(image0, imgpoint_pattern[i], 2, cv::Scalar(0,0,0), -1);		
	}
	cv::imshow("img", image0);
}

int is_ChangeNEAR(){
	

	IsSnapshot[NEAR] = false;

		

	if (IsRealtime){
		int success;
		int m_lMemoryId;

		success = is_SetExternalTrigger(Cams[NEAR], IS_SET_TRIGGER_OFF);
		if (success != IS_SUCCESS) {		
			printf("SetExternalTrigger fail\n");
			return 0;
		}
		//printf("\n");
		//printf("freerun mode on\n");		
		formatInfo = pformatList->FormatInfo[4];		
		//printf("format id : %d\n", formatInfo.nFormatID);
		int width = formatInfo.nWidth;
		int height = formatInfo.nHeight;
		//printf("width : %d, height : %d\n", width, height);

		size_sensor[NEAR].width = width;
		size_sensor[NEAR].height = height;

		success = is_AllocImageMem(Cams[NEAR], width, height, 24, &Mems[NEAR], &m_lMemoryId);
		if (success != IS_SUCCESS) {
			printf("AllocImageMem fail\n");
			return 0;
		}
		success = is_SetImageMem(Cams[NEAR], Mems[NEAR], m_lMemoryId);
		if (success != IS_SUCCESS) {
			printf("SetImageMem fail\n");
			return 0;
		}

		//printf("format Capture mode : %d\n",formatInfo.nSupportedCaptureModes);
		success = is_ImageFormat(Cams[NEAR], IMGFRMT_CMD_SET_FORMAT, &formatInfo.nFormatID, sizeof(formatInfo.nFormatID));
		if (success != IS_SUCCESS){
			printf("ImageFormat -> setting fail\n");
			return 0;
		}

		success = is_CaptureVideo(Cams[NEAR], IS_WAIT);
		if (success != IS_SUCCESS){
			printf("CaptureVideo fail\n");
		}

		success = is_EnableEvent(Cams[NEAR], IS_SET_EVENT_FRAME);
		if (success != IS_SUCCESS){
			printf("CaptureVideo fail\n");
		}

	}

	Int[NEAR] = (cv::Mat_<double>(3, 3) << 1247.7, 0, 640.0229, 0, 1246.3, 351.5087, 0, 0, 1);
	Dist[NEAR] = (cv::Mat_<double>(5, 1) << 0.0908, -0.2617, 0, 0, 0);
	//Dist[NEAR] = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

	Rmat_temp = Rmat[NEAR];
	Rvec_temp = Rvec[NEAR];
	Tvec_temp = Tvec[NEAR];

	Rmat[NEAR] = Epi[4] * Rmat[NEAR];
	cv::Rodrigues(Rmat[NEAR], Rvec[NEAR]);
	Tvec[NEAR] = Epi[5] + Epi[4] * Tvec[NEAR];
	cv::Mat vec3 = -Rmat[NEAR].inv() * Tvec[NEAR];
	cam_pos[NEAR] = cv::Point2f(vec3.at<double>(0, 0), vec3.at<double>(1, 0));
	cv::Mat mat3 = Rmat[NEAR].inv();
	cam_dir[NEAR] = cv::Point2f(mat3.at<double>(0, 2), mat3.at<double>(1, 2)) * (1 / sqrt(mat3.at<double>(0, 2)*mat3.at<double>(0, 2) + mat3.at<double>(1, 2)*mat3.at<double>(1, 2)));

	quad[0] = getimgbywld(cv::Point2f(-W + 0.5*d, -pad_back), 0);
	quad[1] = getimgbywld(cv::Point2f(-W + 0.5*d, transition[1] + pad_hog), 0);
	quad[2] = getimgbywld(cv::Point2f(W - 0.5*d, transition[1] + pad_hog), 0);
	quad[3] = getimgbywld(cv::Point2f(W - 0.5*d, -pad_back), 0);
	ROI_reach = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < ROI_reach.cols; i++){
		for (int j = 0; j < ROI_reach.rows; j++){
			if (ROI_robot[NEAR].at<uchar>(j, i) && cv::pointPolygonTest(quad, cv::Point2f(i, j), false) == 1)
				ROI_reach.at<uchar>(j, i) = 1;
		}
	}
	//cv::imshow("ROI_reach", 255*ROI_reach);
	//cv::waitKey();
	
	/*hex[0] = getimgbywld(cv::Point2f(-W, D_hog), 0);
	hex[1] = getimgbywld(cv::Point2f(-W, D_hog), h_s);
	hex[2] = getimgbywld(cv::Point2f(-W, D_hog_), h_s);
	hex[3] = getimgbywld(cv::Point2f(W, D_hog_), h_s);
	hex[3] = getimgbywld(cv::Point2f(W, D_hog), h_s);
	hex[3] = getimgbywld(cv::Point2f(W, D_hog), 0);
	ROI_FAR_s = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < ROI_FAR_s.cols; i++){
		for (int j = 0; j < ROI_FAR_s.rows; j++){
			if (ROI_robot[NEAR].at<uchar>(j, i) && cv::pointPolygonTest(hex, cv::Point2f(i, j), false) == 1)
				ROI_FAR_s.at<uchar>(j, i) = 1;
		}
	}*/

	unit = 20;
	theta_max = 60;
	p = 1 * unit;

	cos_max = cos((CV_PI / 180)*(90 - theta_max));
	r_max = 0.5*d_ + p;
	r_min = 0.5*d_ - p;
	circle.clear();
	for (int x = -r_max; x < r_max + 1; x += unit){
		for (int y = -r_max; y < r_max + 1; y += unit){
			if (x*x + y*y > r_min*r_min && x*x + y*y < r_max*r_max)
				circle.push_back(cv::Point(x, y) * (1.0 / unit));
		}
	}
	num_c = circle.size();

	circlenorm.clear();	
	for (int i = 0; i < num_c; i++)
		circlenorm.push_back(sqrt(circle[i].x*circle[i].x + circle[i].y*circle[i].y));
	
	circle_.clear();
	for (int x = -d; x < d + 1; x += unit){
		for (int y = -d; y < d + 1; y += unit){
			if (x*x + y*y < d*d)
				circle_.push_back(cv::Point(x, y) * (1.0 / unit));
		}
	}
	num_c_ = circle_.size();

	printf("Near camera mode change\n");
	
	return 1;
}
