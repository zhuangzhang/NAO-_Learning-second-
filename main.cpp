// Aldebaran includes.
#include <alproxies/alvideodeviceproxy.h>
#include <alvision/alimage.h>
#include <alvision/alvisiondefinitions.h>
#include <alerror/alerror.h>
//Ëµ»°
#include <alproxies/altexttospeechproxy.h>
//tracker
#include <alproxies/almotionproxy.h>
#include <alproxies/altrackerproxy.h>
#include <alproxies/alrobotpostureproxy.h>

// Opencv includes.
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

//other includes
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include"face_detection.h"
#include"face_recognition.h"

using namespace AL;
using namespace std;
using namespace cv;

void videoRecognition(const std::string& robotIp);

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage 'videoRecognition robotIp'" << std::endl;
    return 1;
  }

  const std::string robotIp(argv[1]);

  try
  {
    videoRecognition(robotIp);
  }
  catch (const AL::ALError& e)
  {
    std::cerr << "Caught exception " << e.what() << std::endl;
  }

  return 0;
}
