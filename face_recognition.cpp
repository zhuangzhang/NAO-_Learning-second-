// Aldebaran includes.
#include <alproxies/alvideodeviceproxy.h>
#include <alvision/alimage.h>
#include <alvision/alvisiondefinitions.h>
#include <alerror/alerror.h>
//˵��
#include <alproxies/altexttospeechproxy.h>
//tracker
#include <alproxies/almotionproxy.h>
#include <alproxies/altrackerproxy.h>
#include <alproxies/alrobotpostureproxy.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <string>
#include <iostream>
#include <fstream>

#include <time.h>

#include "face_detection.h"


using namespace AL;
using namespace std;
using namespace cv;

double getSimilarity( const Mat A,const Mat B);

Mat faceDetection(Mat frame);

void videoRecognition(const std::string& robotIp)
{
	//��ʼ��˵��
	ALTextToSpeechProxy textToSpeech(robotIp, 9559);
	//cout <<"the robot support the following lanuuages" << textToSpeech.getAvailableLanguages() << endl;
	//��ʼ��tracker
	ALTrackerProxy tracker(robotIp, 9559);
	//posture��ʼ��
	ALRobotPostureProxy posture(robotIp, 9559);
	//��ʼ����
	ALMotionProxy motionPrx(robotIp, 9559);
		/** Create a proxy to ALVideoDevice on the robot.*/
	ALVideoDeviceProxy camProxy(robotIp, 9559);
    /** Subscribe a client image requiring 320*240 and BGR colorspace.*/
	camProxy.setActiveCamera(kTopCamera);        // Connect to top camera.

	const std::string clientName = camProxy.subscribe("test", kQVGA, kBGRColorSpace, 30);
  	/** Create an cv::Mat header to wrap into an opencv image.*/
 	Mat imgload = Mat(cv::Size(320, 240), CV_8UC3);	
	Mat imgloa;
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->load("D:\\NAOqiDEV\\SDKfolder\\face_recognition\\build-mytoolchain\\trainedmodel.xml");
	//if(model.empty()){printf("--(!)Error loading\n");};

	int i = 0;
	int j = 0;
	while ((char) cv::waitKey(30) != 27)
	{
		i++;
		//tracker.registerTarget("Face",0.1);//

		ALValue img = camProxy.getImageRemote(clientName);	//Զ�̵�ȡͼ��
		imgload.data = (uchar*) img[6].GetBinary();//����imgloadHeader
		//
		camProxy.releaseImage(clientName);
		imshow("NAO����ͷ",imgload);
		cout<<"Camera output��"<<imgload.size()<<endl;
		if(imgload.empty())
		{
			cout<<"û�л�ȡ����Ƶ"<<endl;
			//break;		//��Ƶ������ͽ���
		}
		else
		{
		
			imgloa = faceDetection(imgload);//�����ӳ���� ����ʱ���з���ֵ����
			if(imgloa.empty())
			{
				cout<<i<<endl;
				cout<<"���������!"<<endl;
				continue;		//û��ʶ����������������
			}
			else
			{
				imshow("imgloa",imgloa);
				cout<<i<<endl;
				int identity = model->predict(imgloa);
				//cout<<identity<<endl;
				if (identity == 5)
				{
					
					if (j >= 3)
					{
						j = 0;
						cout<<"\n**********************��ã� ��׳!***********************\n"<<endl;
						tracker.setEffector("RArm");
						textToSpeech.say("hello zhuang");
						motionPrx.moveToward(1.0,0.0,0.0);
						Sleep(9000);
						motionPrx.stopMove();
						tracker.stopTracker();
						tracker.setEffector("None");

						//posture.goToPosture("StandInit", 0.8);
						//tracker.setMode("move");
						//tracker.registerTarget("Face",0.1);
						//posture.goToPosture("StandInit", 0.8);
						//tracker.setMode("move");
						
							
						/*tracker.stopTracker();
						tracker.unregisterAllTargets();
						tracker.setEffector("None");
						posture.goToPosture("Sit", 0.8);*/

					}
				/*	tracker.stopTracker();
					tracker.unregisterAllTargets();
					tracker.setEffector("None");*/
					j++;
					cout<<"������"<<j;
					cout<<"\t�����ƶ�������ȷ��!"<<endl;
				}
				if(waitKey(1)>0){break;}
			}
		}
		
	}

	  /** Cleanup.*/
  camProxy.unsubscribe(clientName);
}


double getSimilarity( const Mat A,const Mat B)   //L2
{
	double errorL2 = norm(A,B,CV_L2);
	double similarity = errorL2 / (double)(A.rows * A.cols);
	return similarity;
}
