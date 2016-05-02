#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;


//注意，需要把"haarcascade_frontalface_alt.xml"和"haarcascade_eye.xml"这两个文件复制到工程路径下
String face_cascade_name = "D:\\NAOqiDEV\\SDKfolder\\face_recognition\\build-mytoolchain\\haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "D:\\NAOqiDEV\\SDKfolder\\face_recognition\\build-mytoolchain\\haarcascade_eye.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";


Mat tranformation(Mat face,Rect leftEye,Rect rightEye);

Mat faceDetection(Mat frame)
{
	if(!face_cascade.load(face_cascade_name))
	{ 
		printf("--(!)Error loading\n");
	};
	if(!eyes_cascade.load(eyes_cascade_name))
	{
		 printf("--(!)Error loading\n");
	};
	
	cout<<"----------------------------------------------------\n";
	std::vector<Rect> faces;
	std::vector<Rect> eyes;			
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);	//灰度化

	equalizeHist(frame_gray, frame_gray);			//直方图均衡化

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));//-- 人脸检测

	for(size_t i = 0; i < faces.size(); i++)
	{
		Mat faceROI = frame_gray(faces[i]);		//人脸图像矩阵
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20));//-- 在脸中检测眼睛
		if(eyes.size()  == 2)
		{
			if(eyes[0].x < eyes[1].x)
			{
				faceROI = tranformation(faceROI,eyes[0],eyes[1]);
			}
			else
			{
				faceROI = tranformation(faceROI,eyes[1],eyes[0]);
			}
			for(int j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2);
				rectangle(frame,Rect(faces[i].x + eyes[j].x,faces[i].y + eyes[j].y,eyes[j].width,eyes[j].height),Scalar( 255, 0, 255 ),3,0);
			}
			Point center(faces[i].x+faces[i].width/2,faces[i].y+faces[i].height/2 );
			ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 3, 0 );
			imshow("frame",frame);
			return faceROI; //需要显示的返回值
		}
		else
		{
			imshow("frame",frame);  //显示
		}

    }
 	Mat frame1;
 	return frame1;
}




Mat tranformation(Mat face,Rect leftEye,Rect rightEye)
{
	Point2f leftEyeCenter(leftEye.x+leftEye.width/2,leftEye.y+leftEye.height/2);		//左眼中心
	Point2f righrEyeCenter(rightEye.x+rightEye.width/2,rightEye.y+rightEye.height/2);	//右眼中心
	Point2f eyesCenter((leftEyeCenter.x+righrEyeCenter.x)/2,(leftEyeCenter.y+righrEyeCenter.y)/2);	//双眼中心

	double dy = rightEye.y+rightEye.height/2 - leftEye.y - leftEye.height/2;
	double dx = rightEye.x+rightEye.width/2 - leftEye.x - leftEye.width/2;
	double len = sqrt(dx*dx + dy*dy);
	
	double angle = atan2(dy,dx)*180.0 / CV_PI;	//双眼偏角

	const double DESIRED_LEFT_EYE_X = 0.16;
	const double DESIRED_RIGHT_EYE_X = 0.84;		
	const double DESIRED_LEFT_EYE_Y = 0.14;	//眼睛位置归一化

	const int DESIRED_FACE_WIDTH = 70;
	const int DESIRED_FACE_HEIGHT = 70;		//人脸尺度归一化

	double desiredLen = 0.68;			//归一化的双眼的相对位置
	double scale = desiredLen * DESIRED_FACE_WIDTH / len;	//缩放比例
	Mat rot_mat = getRotationMatrix2D(eyesCenter,angle,scale);	//变换矩阵
	double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
	double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
	rot_mat.at<double>(0,2) += ex;		//
	rot_mat.at<double>(1,2) += ey;
	
	Mat warped = Mat(DESIRED_FACE_WIDTH,DESIRED_FACE_HEIGHT,CV_8U,Scalar(0));		//初始化归一化人脸//
	
	warpAffine(face,warped,rot_mat,warped.size());

	Mat filtered = Mat(warped.size(),CV_8UC1);	

	bilateralFilter(warped,filtered,0,20.0,2.0);	//光滑

	Mat mask(filtered.size(),CV_8UC1,Scalar(255));	//
	double dw = DESIRED_FACE_WIDTH;
	double dh = DESIRED_FACE_HEIGHT;
	Point faceCenter(cvRound(dw*0.5),cvRound(dh*0.4));
	Size size(cvRound(dw*0.5),cvRound(dh*0.8));
	ellipse(mask,faceCenter,size,0,0,360,Scalar(0),CV_FILLED);
	filtered.setTo(Scalar(128),mask);

	return filtered;
}
