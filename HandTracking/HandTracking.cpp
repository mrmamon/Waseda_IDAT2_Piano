#include <stdlib.h>
#include<iostream>
#include<fstream>
#include <windows.h>
#include <mmsystem.h>

#include"opencv/cv.h"
#include"opencv/highgui.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "XnCppWrapper.h"
#include <SFML/Audio.hpp>


using namespace cv ;
using namespace std ;
using namespace sf;

char key = 0;
int HAND_LENGTH = 90;
vector <Point> fingertips;

int notewide =0;
int notehight =0;

/////////////////////////////////////Mouse Control///////////////////////////////////////
HCURSOR hCurs1, hCurs2;    // cursor handles 
 
POINT pt;                  // cursor location  
RECT rc;                   // client area coordinates 
static int repeat = 1;     // repeat key counter 

   int display_horizontal = 0;
   int display_vertical = 0;

// Get the horizontal and vertical screen sizes in pixel
void GetDesktopResolution(int& horizontal, int& vertical)
{
   RECT desktop;
   // Get a handle to the desktop window
   const HWND hDesktop = GetDesktopWindow();
   // Get the size of screen to the variable desktop
   GetWindowRect(hDesktop, &desktop);
   // The top left corner will have coordinates (0,0)
   // and the bottom right corner will have coordinates
   // (horizontal, vertical)
   horizontal = desktop.right;
   vertical = desktop.bottom;
}

/////////////////////////////////////End of Mouse Control///////////////////////////////////////

Point PointNormal [16];
Point PointSharp[10];

/////////////////////////////////////configuration of ROI///////////////////////////////////////
//for adjusting ROI area
bool setting_calibration_points=false;
int calibration_point_index=0; 
Point calibration_points[4];
Mat H; // perspective_transform matrix
bool state = false;
// temporary ROI boundaries
int xMin = 100;
int xMax = 540;
int yMin = 100;
int yMax = 380;
//set parameters of ROI and its trackbar
const int ROI_x_slider_max = 640, ROI_y_slider_max = 480;   //the max value of trackbar
int ROI_slider_x = 170, ROI_slider_y = 45, ROI_slider_wide = 320, ROI_slider_high = 200; //the original size of ROI
double ROI_x, ROI_y, ROI_wide, ROI_high;
char ROI_x_max[640], ROI_y_max[480], ROI_wide_max[640], ROI_high_max[480];

void ROI_X_func ( int, void*)
{
	ROI_x = (double) ROI_slider_x / ROI_x_slider_max;
}
void ROI_Y_func ( int, void*)
{
	ROI_y = (double) ROI_slider_y / ROI_y_slider_max;
}
void ROI_WIDE_func ( int, void*)
{
	ROI_wide = (double) ROI_slider_wide / ROI_x_slider_max;
}
void ROI_HIGH_func ( int, void*)
{
	ROI_high = (double) ROI_slider_high / ROI_y_slider_max;
}

/////////////////////////////////////Parameters of thresholding//////////////////////////////////////////
const int alpha_slider_max = 250;   //the max value of trackbar
int  alpha_slider = 5, alpha_slider2 = 180;   //the original value of thresholding range
double alpha, alpha2;
char upper[250];
char lower[250];

void trackbar ( int, void*)
{
	alpha = (double) alpha_slider / alpha_slider_max;
}
void trackbar2 ( int, void*)
{
	alpha2 = (double) alpha_slider2 / alpha_slider_max;
}

/////////////////////////////////check error////////////////////////////////////
static void check_eorror( XnStatus result,string status )
{
    if ( result != XN_STATUS_OK )
        cerr << status <<"error:"<<xnGetStatusString(result)<<endl ;
}

//////////////////output file of depth data matrix (short)////////////////////////
int output_file( char *filename, Mat depthdata)
	{
		FILE *fWrite;
		fWrite = fopen ( filename, "w+");
		for( int m = 0; m < XN_VGA_Y_RES; m++)
			{
				for (int n = 0; n<XN_VGA_X_RES; n++) 
					{
						fprintf( fWrite, "%4u ", depthdata.at<unsigned short>(m,n) );
					}
				fprintf( fWrite, "\n" );
			}
		fclose( fWrite );
		return *filename;
	}

//////////////////output file of depth data matrix (char)////////////////////////
int output_file_c( char *filename, Mat depthdata)
	{
		FILE *fWrite;
		fWrite = fopen ( filename, "w+");
		for( int m = 0; m < XN_VGA_Y_RES; m++)
			{
				for (int n = 0; n<XN_VGA_X_RES; n++) 
					{
						fprintf( fWrite, "%4u ", depthdata.at<unsigned char>(m,n) );
					}
				fprintf( fWrite, "\n" );
			}
		fclose( fWrite );
		return *filename;
	}

////////////////////////////////Declaraion//////////////////////////////////////////
Point hand_length_point ( Point pt1, Point pt2);
int point_in_out_rect (Point2f point0, Point2f vertex1, Point2f vertex2, Point2f vertex3, Point2f vertex4);
double line_angle( Point2f ptStart1, Point2f ptEnd1, Point2f ptStart2, Point2f ptEnd2);
void onMouse( int event, int x, int y, int, void* );
Mat find_perspective_transform();


double diffclock( clock_t clock1, clock_t clock2 ) {

        double diffticks = clock1 - clock2;
        double diffms    = diffticks / ( CLOCKS_PER_SEC / 1000 );

        return diffms;
    }

//////////////////////////////Main function////////////////////////////////////////
int main(int argc, char* argv[])
{
    //for opencv Mat  
    Mat m_srcdepth16u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_16UC1);  //updating depth image from sensor (16bit)
	Mat m_srcimage8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC3);  //updating color image from sensor (16bit)

	Mat m_depth16u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_16UC1); //depth image of backgrond (16bit)
	Mat m_ROIdepth8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //depth image of ROI area (16bit)

	Mat m_depth8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //binary image of deference from the processing of comparising background and real-time image 
	Mat m_erodedDepth8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //remove noise (after erosion)
	Mat m_dilateDepth8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //remove noise (after dilation)
	Mat m_intersectionDepth8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //remove noise (for intersection)
	Mat m_intersectionDepth8u_hand( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //backup of m_intersectionDepth8u
	Mat m_intersectionDepth8u_hand2( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //for searching minAreaRect
	
	Mat m_dilateShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //for showing the scene of dilated binary image
	Mat m_erodedShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //for showing the scene of eroded binary image
	Mat m_intersectionShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //for showing the scene of intersected binary image

	Mat m_backGrounddepth16u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_16UC1);  // depth image from sensor, which is consider as Background (16bit)
	Mat m_backGround16u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_16UC1); // depth image of Background  after smoothing
	Mat m_middepth8u( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); //temporary depth data  container when comparing backgrund and real-time images

	Mat depthShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1); // for showing processed binary depth image
	Mat imageShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC1);  // for showing original binary depth image
	Mat sceneShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC3);  // for showing color image
	Mat gameShow( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC3);
	Mat gameShow2( XN_VGA_Y_RES,XN_VGA_X_RES,CV_8UC3);


											sf::SoundBuffer buffer;
							if (!buffer.loadFromFile("c1.ogg"))
							{
								printf("sound not found");
							}
							
							sf::Sound sound1;
							sound1.setBuffer(buffer);
	
	// openni variable and initialization
     XnStatus nRet = XN_STATUS_OK ;
        
    xn::Context context ;
    nRet = context.Init() ;
    check_eorror(nRet,"context.Init" ) ;

	//Get screen resolution
	GetDesktopResolution(display_horizontal, display_vertical);

	//Set mirror ( inverse right<-->left) 
    context.SetGlobalMirror(!context.GetGlobalMirror()); 

	// Create a depth generator 
    xn::DepthGenerator depthGen ;
	xn::DepthMetaData depthMD ;
    nRet = depthGen.Create( context ) ;
    check_eorror(nRet,"depthGen.Create" ) ;

    // Create a image generator 
    xn::ImageGenerator imageGen ;
	xn::ImageMetaData imageMD;
	nRet = imageGen.Create( context ) ;
    check_eorror(nRet,"imageGen.Create" ) ;

	//Set it to VGA maps at 30 FPS
    XnMapOutputMode mapMode ;
    mapMode.nXRes = XN_VGA_X_RES ;
    mapMode.nYRes = XN_VGA_Y_RES ;
    mapMode.nFPS = 30 ;
	nRet = depthGen.SetMapOutputMode( mapMode ) ;
    check_eorror(nRet,"depthGen.SetMapOutputMode" ) ;
	nRet = imageGen.SetMapOutputMode( mapMode ) ;
    check_eorror(nRet,"imageGen.SetMapOutputMode" ) ;

	// correct view port
    depthGen.GetAlternativeViewPointCap().SetViewPoint( imageGen );
 
	// Start generating
    nRet = context.StartGeneratingAll() ;
    check_eorror(nRet,"StartGeneratingAll" );
	// Update to next frame 
    nRet = context.WaitNoneUpdateAll() ;
    check_eorror(nRet,"WaitNoneUpdateAll" );

	////////////////////////////set background//////////////////////////////////////

	for(int i=6; i>0; i--)
	{
		cout<<"Wait !!!   after: "<< i <<endl;
		//generate depth data
		depthGen.GetMetaData( depthMD ) ;
		memcpy(m_backGrounddepth16u.data,depthMD.Data(),640*480*2);
		//output_file( "background(no G).csv", m_backGrounddepth16u);
		// gaussian filter
    	GaussianBlur(m_backGrounddepth16u, m_backGround16u, Size(7,7), 0.85, 0.85);
	}
	
	cout<<"====START===="<<endl;
	//output_file( "background(O).csv", m_backGround16u);
	
	//////////////////////////hand detection/////////////////////////////////////////////
	// Update to next frame 
    nRet = context.WaitNoneUpdateAll() ;
    check_eorror(nRet,"WaitNoneUpdateAll" ) ;
	
    while ( (key != 27) && !( nRet = context.WaitNoneUpdateAll()))
	{
		//renew background
		if ( key == 'b')
		{
			cout<<"updating background image..."<<endl;
			
			//getting background image
			for(int i=5; i>0; i--)
			{
				//generate depth data
				depthGen.GetMetaData( depthMD ) ;
				memcpy(m_backGrounddepth16u.data,depthMD.Data(),640*480*2);
				//	output_file( "background2.csv", m_backGrounddepth16u);
				
				// gaussian filter
    			GaussianBlur(m_backGrounddepth16u, m_backGround16u, Size(7,7), 0.85, 0.85);
			}
			//output_file( "background(b).csv", m_backGround16u);
			cout<<"====Updated===="<<endl;
		}//if
				 
					
		//generate depth data
		depthGen.GetMetaData( depthMD ) ;
		memcpy(m_srcdepth16u.data,depthMD.Data(),640*480*2);

		//generate image data
		imageGen.GetMetaData( imageMD ) ;
		memcpy(m_srcimage8u.data,imageMD.Data(),640*480*3);
		
		// gaussian filter
   		GaussianBlur(m_srcdepth16u, m_depth16u, Size(9,9), 0.85, 0.85);
		//compare real-time image wih background
		for( int i = 0; i < XN_VGA_Y_RES; i++)
            for (int j = 0; j<XN_VGA_X_RES; j++) 
			{
                if( ( (m_depth16u.at<unsigned short>(i,j)) <( m_backGround16u.at<unsigned short>(i,j)-alpha_slider ))&&
					(m_depth16u.at<unsigned short>(i,j) > ( m_backGround16u.at<unsigned short>(i,j)-alpha_slider2)) )
                    m_middepth8u.at<unsigned char>(i,j) =255;
                else m_middepth8u.at<unsigned char>(i,j) = 0;
            }
		m_middepth8u.copyTo(m_depth8u);
		//m_middepth8u.convertTo(m_depth8u,CV_8U, 1.0); 
	
		//remove noise (erode and dilate)
		erode(m_depth8u, m_erodedDepth8u, Mat(), Point(-1,-1), 2 );
		dilate(m_erodedDepth8u, m_dilateDepth8u, Mat(), Point(-1,-1), 1 );
		bitwise_and(m_depth8u, m_dilateDepth8u, m_intersectionDepth8u);
		
		//m_intersectionDepth8u.copyTo(m_intersectionShow);
		m_dilateDepth8u.copyTo(m_dilateShow);
		m_erodedDepth8u.copyTo(m_erodedShow);

		m_intersectionDepth8u.copyTo(m_intersectionDepth8u_hand);   //backup for next processing
		m_intersectionDepth8u.copyTo(m_intersectionDepth8u_hand2); //for searching minAreaRect
		
		  cvtColor(m_intersectionShow, imageShow, CV_GRAY2BGR);        // transfer previous grey image to color 
		  cvtColor(m_depth8u, depthShow, CV_GRAY2BGR);                      // transfer previous grey image to color 
		  cvtColor(m_srcimage8u, sceneShow, CV_RGB2BGR);                    // transfer RGB image to BGR
		  cvtColor(m_srcimage8u, gameShow, CV_RGB2BGR);
		  cvtColor(m_srcimage8u, gameShow2, CV_RGB2BGR);

		//Set note size
		notehight = 240;
		notewide = 80;

		  for (int i=0;i<8;i++)
		  {
			  int tempnote =notewide*i;
			rectangle( gameShow,Point( tempnote, 120),Point( tempnote+notewide-10, 120+notehight),Scalar( 255, 255, 255 ),-1,8 );
		  }

		  rectangle( gameShow,Point( 60, 120),Point( 100, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow,Point( 140, 120),Point( 180, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow,Point( 300, 120),Point( 340, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow,Point( 380, 120),Point( 420, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow,Point( 460, 120),Point( 500, 240),Scalar( 0, 0, 0 ),-1,8 );

		  /*
		  		  for (int i=0;i<8;i++)
		  {
			  int tempnote =notewide*i;
			rectangle( gameShow2,Point( tempnote, 120),Point( tempnote+notewide-10, 120+notehight),Scalar( 255, 255, 255 ),-1,8 );
		  }

		  rectangle( gameShow2,Point( 60, 120),Point( 100, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow2,Point( 140, 120),Point( 180, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow2,Point( 300, 120),Point( 340, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow2,Point( 380, 120),Point( 420, 240),Scalar( 0, 0, 0 ),-1,8 );
		  rectangle( gameShow2,Point( 460, 120),Point( 500, 240),Scalar( 0, 0, 0 ),-1,8 );
		  */

		/////////////////////////////hand detection////////////////////////////////////////////
		//finding hand's contour
		vector<vector<Point> > contours_hand;
		vector<Vec4i> hierarchy_hand;
		RotatedRect minEllipse;
		vector<Point> available_hand_area_rect;
		vector<Point> available_hand_area_hull_rect(available_hand_area_rect.size());

		char* rect_angle_text = new char[80];  
		char* cor_index_text = new char[80];  
		//vector<RotatedRect> minEllipse(contours_hand.size());
		
		findContours( m_intersectionDepth8u_hand2, contours_hand, hierarchy_hand, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );    //finding contours of hands

		//remove arm's part
		for( int con_i = 0; con_i < contours_hand.size(); con_i++ )
			 {
				 if( contours_hand[con_i].size() > 5 )
				 {
					 minEllipse = minAreaRect( Mat(contours_hand[con_i]) ); 
				 }
				 
				 //drawing rectangle including sets of points (bounding box)
				  Point2f rect_points[4];
			      minEllipse.points( rect_points );
				  for( int rect_i = 0; rect_i < 4; rect_i++ )
					  {
						 line( imageShow, rect_points[rect_i], rect_points[(rect_i+1)%4],  Scalar(0,255,255), 1, 8 );
						 sprintf(cor_index_text,"%d", rect_i );  //the index numbers of recangle corners
						 putText (imageShow, cor_index_text ,rect_points[rect_i], CV_FONT_HERSHEY_COMPLEX_SMALL|CV_FONT_ITALIC, 1, Scalar(0,255,0));  
					  }  
				
				 if( minEllipse.size.width > 50 && minEllipse.size.height > 50) //small white areas would be ignored
				 {
					 // find out which side of bounding box is the nearest one to the end of finger
					 // find two points and connect them to divide the white shape into two parts, one is arm, another is hand
					 if( minEllipse.size.width>minEllipse.size.height)
					  {
								Point new_corner_pt03 = hand_length_point( rect_points[0],  rect_points[3]);
								Point new_corner_pt12 = hand_length_point( rect_points[1],  rect_points[2]);
								line( imageShow, new_corner_pt03, new_corner_pt12,  Scalar(0,255,255), 1, 8 );
								available_hand_area_rect.push_back( rect_points[0] );
								available_hand_area_rect.push_back( rect_points[1] );
								available_hand_area_rect.push_back( new_corner_pt12 );
								available_hand_area_rect.push_back( new_corner_pt03 );
								//convexHull( available_hand_area, available_hand_area_hull);
								//drawContours( imageShow, available_hand_area_hull, con_i, Scalar(0,0,255), -1, 8 );
					  }
					  else
				  	  {
							Point new_corner_pt01 = hand_length_point( rect_points[0],  rect_points[1]); //calculating the point coordinate which located on the line of pt1-pt0
							Point new_corner_pt32 = hand_length_point( rect_points[3],  rect_points[2]);
							line( imageShow, new_corner_pt01, new_corner_pt32,  Scalar(0,255,255), 1, 8 );
							available_hand_area_rect.push_back( rect_points[3] );
							available_hand_area_rect.push_back( rect_points[0] );
							available_hand_area_rect.push_back( new_corner_pt01 );
							available_hand_area_rect.push_back( new_corner_pt32 );
					 }	
					//scaning a frame of image 
					  for( int ni = 0; ni < XN_VGA_Y_RES; ni++)
							for (int mi = 0; mi<XN_VGA_X_RES; mi++) 
							{
								//distinguishing wether the point in or out of definied area
								int out_in_para = point_in_out_rect( Point2f(mi,ni), 
									available_hand_area_rect[0], available_hand_area_rect[1], available_hand_area_rect[2], available_hand_area_rect[3] );
																
								if ( out_in_para == -1 )//reducing points which are out of avalible area
								{
									m_intersectionDepth8u_hand.at<unsigned char>(ni,mi) = 0;
								}
							}
				 }//if(>50)
			 }//for
		  m_intersectionDepth8u_hand.copyTo(m_intersectionShow);
	  
		/////////////////////////////////set region of interest/////////////////////////////////////////////
		IplImage* p_intersectionDepth8u = &m_intersectionDepth8u_hand.operator IplImage();
		cvSetImageROI(p_intersectionDepth8u, cvRect(ROI_slider_x,ROI_slider_y,ROI_slider_wide,ROI_slider_high)); 
		m_ROIdepth8u = Mat(p_intersectionDepth8u);

		/////////////////////////////////fingertips detection////////////////////////////////////////////////
		//find hand's contour
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( m_ROIdepth8u, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(ROI_slider_x, ROI_slider_y) );    //finding contours of hands
	
		//extract convex hull and convexity defects
	    vector<vector<Point> > hull_P(contours.size());
		vector<vector<int> > hull_I(contours.size());
		vector<vector<Vec4i>> defects(contours.size());
		vector<vector<Point> > contoursPoly(contours.size());

		char* text = new char[80];  // texture for showing index numbers of fingers
		char* tempText = new char[80];  // texture for showing index numbers of fingers

		float tempDiatancePalmCenter=0;
		int tempPalmCenter_x, tempPalmCenter_y;

		for (int i = 0; i < contours.size(); i++)
		{			
			// since the edges of contours are not smooth, approximates a polygonal curves with the specified precision
			//extract approximate curve
			approxPolyDP( Mat(contours[i]), contoursPoly[i], 2, true ); 
			
				drawContours( imageShow, contours, i, Scalar(255,255,255), -1 );//fill the shape of  contours hands, //filling holes
				drawContours( sceneShow, contoursPoly, i, Scalar(0,255,0), 2, 8 );//show approximate curve
				
				//find the center point of palm
				for( int n = 0; n < XN_VGA_Y_RES; n=n+2)
					for (int m = 0; m<XN_VGA_X_RES; m=m+2) 
					{
						//find points along contour
						int k = m_intersectionShow.at<unsigned char>(n,m);
						if ( k==255)
						{
							float dis_palmCenter = pointPolygonTest( contours[i], Point2f(m,n), true );
							//calculate the minimum distance t from the contour of each point, 
							//and then pick up the point which one has the largest value
							if (dis_palmCenter>0)
							{
								//find the center point of palm, i.e. the point with the largest value in vector tempDistance
								if (dis_palmCenter>tempDiatancePalmCenter)
								{
									tempDiatancePalmCenter = dis_palmCenter;
									tempPalmCenter_x = m;
									tempPalmCenter_y = n;
									pt.x= (((tempPalmCenter_x/640)*100)/100)*1600;
									pt.y=(((tempPalmCenter_y/480)*100)/100)*900;
									//printf("&d and %d",pt.x,pt.y);
									//SetCursorPos(pt.x, pt.y); 
								}
							}//if(>0)
						}//if ( k==255)
					}//for
				circle( imageShow, Point2f(tempPalmCenter_x, tempPalmCenter_y), 4, Scalar( 0, 0, 255 ), -1 );  //center point of palm
				circle( imageShow, Point2f(tempPalmCenter_x, tempPalmCenter_y), tempDiatancePalmCenter, Scalar( 255, 50, 0 ), 2 );  // incircle of palm
				circle( sceneShow, Point2f(tempPalmCenter_x, tempPalmCenter_y), 4, Scalar( 0, 0, 255 ), -1 );  //center point of palm (scene)
				circle( sceneShow, Point2f(tempPalmCenter_x, tempPalmCenter_y), tempDiatancePalmCenter, Scalar( 255, 50, 0 ), 1 );  // incircle of palm (scene)
				
				// extract convex hull
				convexHull(Mat(contours[i]), hull_P[i],false);
				convexHull(Mat(contoursPoly[i]), hull_I[i],false);	 
				drawContours( imageShow, hull_P, i, Scalar(0,255,255), 1, 8 ); //drawing hull
				drawContours( sceneShow, hull_P, i, Scalar(0,255,255), 1, 8 ); //drawing hull
				
				//find defecfs of convex hull
				if (contoursPoly[i].size() >3 )
				{
					convexityDefects(contoursPoly[i], hull_I[i], defects[i]); 
				}//if(contours[i].size())	
		}// for ( contours.size )
		
		Point preFarthest;// preparing for calculating fingertips' angle
		
		for( int i = 0; i < defects.size(); i++)
        {
			vector <Point> tempFingertips;	
			vector <Vec4i> defects2( defects[i]);// the second dimention of defects
				for( int j=0; j<defects2.size(); j++)
				{
	        		Vec4i defects3(defects[i][j]); // the third dimension of defects, 
															   //vector <start point, end point, the farthest point, the distance hull&farthest>
					
					Point start = contoursPoly[i][defects3[0]];
					Point end = contoursPoly[i][defects3[1]];
					Point farthest = contoursPoly[i][defects3[2]];
																	
					//distance between fingertip and corresponding depth point of defect
					//double concave_convex_distance = sqrtf( (abs(start.x-farthest.x) * abs(start.x-farthest.x)) + (abs(start.y-farthest.y) * abs(start.y-farthest.y)) );
					
					//distance from palm center to fingertips
					double palm_fingertip_distance = sqrtf( (abs(start.x-tempPalmCenter_x) * abs(start.x-tempPalmCenter_x)) + 
									(abs(start.y-tempPalmCenter_y) * abs(start.y-tempPalmCenter_y)) );

					circle( imageShow, Point2f(tempPalmCenter_x,tempPalmCenter_y), tempDiatancePalmCenter+12, Scalar( 0, 0, 255 ), 1 );  //range of recognition (inside circle bound)
					circle( imageShow, Point2f(tempPalmCenter_x,tempPalmCenter_y), 90, Scalar( 0, 0, 255 ), 1 );  //range of recognition (outside circle bound)
					//line( imageShow, Point2f(tempPalmCenter_x,tempPalmCenter_y), end, Scalar( 255, 255, 0 ), 1 ); //skeleton of hand
					if( j == 0)
					{
						preFarthest = farthest;
					}
					
					if ( j > 0 && palm_fingertip_distance<90 && palm_fingertip_distance>(tempDiatancePalmCenter+12))
					{
						double fingertips_angle = line_angle(  preFarthest, start, farthest, start );
						preFarthest = farthest;
						if ( fingertips_angle>0 && fingertips_angle<82 )
						{
							tempFingertips.push_back(start); //the points of fingertips			
							
							line( imageShow, start, farthest, Scalar( 0, 255, 0 ), 2 ); //the line connected fingertips and defects
							line( imageShow, end, farthest, Scalar( 0, 255, 0 ), 2 ); //the line connected fingertips and defects
							//line( sceneShow, start, farthest, Scalar( 0, 255, 0 ), 2 ); //the line connected fingertips and defects (scene)
							//line( sceneShow, end, farthest, Scalar( 0, 255, 0 ), 2 ); //the line connected fingertips and defects (secne)

							/******************for convinient, consider the start point as fingertip's point****************************/
							/*****but it is better to compute the mid-point between end point of previous defect and start point of next defect****/

							circle( imageShow, farthest, 4, Scalar( 0, 0, 255 ), -1 );  //points of defect
							circle( imageShow, start, 4, Scalar( 255, 0, 0 ), -1 );  // points of fingertip 
							//circle( imageShow, end, 5, Scalar( 0, 0, 255 ), -1 );  // points of fingertip
							circle( sceneShow, farthest, 4, Scalar( 0, 255, 255 ), -1 );  //points of defect (scene)
							circle( sceneShow, start, 4, Scalar( 0, 0, 255 ), -1 );  // points of fingertip (scene)

							// show index number on each fingertip
							sprintf(text,"%d", j );  	//the value_text which should be displayed
							putText (imageShow, text ,start, CV_FONT_HERSHEY_COMPLEX_SMALL|CV_FONT_ITALIC, 1, Scalar(0,255,255));  //index number of fingertips
							//putText (sceneShow, text ,start, CV_FONT_HERSHEY_COMPLEX_SMALL|CV_FONT_ITALIC, 1, Scalar(0,255,255));  //index number of fingertips
							//shows angle value of each fingertip
							//sprintf(tempText, "%f", fingertips_angle);
							//putText (imageShow, tempText ,Point2f(start.x-60, start.y+170), CV_FONT_HERSHEY_COMPLEX_SMALL|CV_FONT_ITALIC, 1, Scalar(0,0,255)); 
						}	
					}//if(j>0)
				 }//for (j=0); loop
				fingertips = tempFingertips; //global valuable for numbers of fingertips
		}//for (defects.size)
		
		vector <Point> modified_fingertips(fingertips.size());
		char fingertips_Text[80];  // texture for showing index numbers of fingers
		for (int p=0; p<fingertips.size(); p++)
		{
			//circle( sceneShow, fingertips[p], 6, Scalar( 0, 0, 255 ), -1 );  // points of fingertips (scene)
			
			//adjust the coordinate value of fingertips' position because of ROI
			modified_fingertips[p].x = fingertips[p].x -ROI_slider_x; 
			modified_fingertips[p].y = ROI_slider_high - (fingertips[p].y -ROI_slider_y);
			// show numbers on fingertip
			sprintf(fingertips_Text,"%d  (%d, %d)", p+1, modified_fingertips[p].x, modified_fingertips[p].y );  	//the value_text which would be displayed
			putText (sceneShow, fingertips_Text ,fingertips[p], CV_FONT_HERSHEY_COMPLEX_SMALL|CV_FONT_ITALIC, 1, Scalar(0,255,255));  //index number of fingertip

			//detection of touch action
			int diffFingertipDepth = m_backGround16u.at<unsigned short>(fingertips[p]) - m_depth16u.at<unsigned short>(fingertips[p]); 
			int touching[13];

			if ( diffFingertipDepth <10  ) 
			{
				circle( sceneShow, fingertips[p], 8, Scalar( 255, 0, 0 ), 2 );  // points of fingertip which is touching(scene)
				circle( imageShow, fingertips[p], 8, Scalar( 255, 0, 0 ), 2 );  // points of fingertip which is touching(depth image)
				//circle( gameShow, fingertips[p], 8, Scalar( 255, 0, 0 ), 2 );
				//circle( gameShow2, fingertips[p], 8, Scalar( 255, 0, 0 ), 2 );
				int highlight=-1;


					if(fingertips[p].x>233&&fingertips[p].x<254&&fingertips[p].y>160)
					{
						if(touching[0]!=1)
						{
						PlaySound(L"csharp.wav", NULL, SND_FILENAME | SND_ASYNC);
						touching[0]=1;
						}
						rectangle( gameShow,Point( 60, 120),Point( 100, 240),Scalar( 0, 255, 0 ),-1,8 );
					}
					else if(fingertips[p].x>262&&fingertips[p].x<280&&fingertips[p].y>160)
					{
						if(touching[1]!=1)
						{
							PlaySound(L"dsharp.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[1]=1;
						}
						rectangle( gameShow,Point( 140, 120),Point( 180, 240),Scalar( 0, 0, 255 ),-1,8 );
					}
					else if(fingertips[p].x>322&&fingertips[p].x<340&&fingertips[p].y>160)
					{
						if(touching[2]!=1)
						{
							PlaySound(L"fsharp.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[2]=1;
						}
						rectangle( gameShow,Point( 300, 120),Point( 340, 240),Scalar( 255, 0, 0 ),-1,8 );
					}
					else if(fingertips[p].x>350&&fingertips[p].x<368&&fingertips[p].y>160)
					{
						if(touching[3]!=1)
						{
							PlaySound(L"gsharp.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[3]=1;
						}
						rectangle( gameShow,Point( 380, 120),Point( 420, 240),Scalar( 255, 0, 255 ),-1,8 );
						
					}
					else if(fingertips[p].x>378&&fingertips[p].x<396&&fingertips[p].y>160)
					{
						if(touching[4]!=1)
						{
							PlaySound(L"asharp.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[4]=1;
						}
						rectangle( gameShow,Point( 460, 120),Point( 500, 240),Scalar( 255, 255, 0 ),-1,8 );
						
					}
					else if(fingertips[p].x>PointNormal[0].x&&fingertips[p].x<PointNormal[1].x)
					{
						if(touching[5]!=1)
						{
							PlaySound(L"c1.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[5]=1;
						}
						highlight=0;	
					}
					else if(fingertips[p].x>PointNormal[2].x&&fingertips[p].x<PointNormal[3].x)
					{
						if(touching[6]!=1)
						{
							PlaySound(L"d.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[6]=1;
						}
						highlight=1;
						
					}
					else if(fingertips[p].x>PointNormal[4].x&&fingertips[p].x<PointNormal[5].x)
					{
						if(touching[7]!=1)
						{
							PlaySound(L"e.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[7]=1;
						}
						highlight=2;
					}
					else if(fingertips[p].x>PointNormal[6].x&&fingertips[p].x<PointNormal[7].x)
					{
						if(touching[8]!=1)
						{
							PlaySound(L"f.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[8]=1;
						}
						highlight=3;
					}
					else if(fingertips[p].x>PointNormal[8].x&&fingertips[p].x<PointNormal[9].x)
					{
						if(touching[9]!=1)
						{
						PlaySound(L"g.wav", NULL, SND_FILENAME | SND_ASYNC);
						touching[9]=1;
						}
						highlight=4;
					}
					else if(fingertips[p].x>PointNormal[10].x&&fingertips[p].x<PointNormal[11].x)
					{
						if(touching[10]!=1)
						{
							PlaySound(L"a.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[10]=1;
						}
							highlight=5;
						
					}
					else if(fingertips[p].x>PointNormal[12].x&&fingertips[p].x<PointNormal[13].x)
					{
						if(touching[11]!=1)
						{
							PlaySound(L"b.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[11]=1;
						}
							highlight=6;
					}
					else if(fingertips[p].x>PointNormal[14].x&&fingertips[p].x<PointNormal[15].x)
					{
						if(touching[12]!=1)
						{
							PlaySound(L"c2.wav", NULL, SND_FILENAME | SND_ASYNC);
							touching[12]=1;
						}
							highlight=7;
						
					}

						for (int i=0;i<8;i++)
						  {
							int tempnote =notewide*i;
							if(i==highlight)
							rectangle( gameShow,Point( tempnote, 120),Point( tempnote+notewide-10, 120+notehight),Scalar( 0, 255, 255 ),-1,8 );
						  }
				

				/*****************fingertip[p] is the coordinate value of touching fingertip*******************/
				/*****************use touching point fingertip[p] to do some application ********************/
			}
			else{
					if(fingertips[p].x>233&&fingertips[p].x<254&&fingertips[p].y>160)
					{
						touching[0]=0;
					}
					else if(fingertips[p].x>262&&fingertips[p].x<280&&fingertips[p].y>160)
					{
						touching[1]=0;
					}
					else if(fingertips[p].x>322&&fingertips[p].x<340&&fingertips[p].y>160)
					{
						touching[2]=0;
					}
					else if(fingertips[p].x>350&&fingertips[p].x<368&&fingertips[p].y>160)
					{
						touching[3]=0;
					}
					else if(fingertips[p].x>378&&fingertips[p].x<396&&fingertips[p].y>160)
					{
						touching[4]=0;
					}
					else if(fingertips[p].x>PointNormal[0].x&&fingertips[p].x<PointNormal[1].x)
					{
						touching[5]=0;
					}
					else if(fingertips[p].x>PointNormal[2].x&&fingertips[p].x<PointNormal[3].x)
					{
						touching[6]=0;
					}
					else if(fingertips[p].x>PointNormal[4].x&&fingertips[p].x<PointNormal[5].x)
					{
						touching[7]=0;
					}
					else if(fingertips[p].x>PointNormal[6].x&&fingertips[p].x<PointNormal[7].x)
					{
						touching[8]=0;
					}
					else if(fingertips[p].x>PointNormal[8].x&&fingertips[p].x<PointNormal[9].x)
					{
						touching[9]=0;
					}
					else if(fingertips[p].x>PointNormal[10].x&&fingertips[p].x<PointNormal[11].x)
					{
						touching[10]=0;
					}
					else if(fingertips[p].x>PointNormal[12].x&&fingertips[p].x<PointNormal[13].x)
					{
						touching[11]=0;
					}
					else if(fingertips[p].x>PointNormal[14].x&&fingertips[p].x<PointNormal[15].x)
					{
						//Sleep(20000);
						touching[12]=0;
					}
			}

		}
		
		//circle(depthShow, cvPoint(min_x+100,min_y+100), 10, cvScalar(0, 255, 0), -1);	// Draw Moving point
		rectangle(sceneShow, cvPoint(ROI_slider_x,ROI_slider_y), cvPoint(ROI_slider_x+ROI_slider_wide, ROI_slider_y+ROI_slider_high), cvScalar(0,0,255), 2);  // Draw rect of valuable area( ROI)
		rectangle(imageShow, cvPoint(ROI_slider_x,ROI_slider_y), cvPoint(ROI_slider_x+ROI_slider_wide, ROI_slider_y+ROI_slider_high), cvScalar(0,0,255), 2);  //ROI

		
		float notehight0 = ROI_slider_high+ROI_slider_y;
		float notehight1 = ROI_slider_y+(((notehight0-ROI_slider_y)/100)*35);
		float notehight2 = ROI_slider_y+(((notehight0-ROI_slider_y)/100)*80);

		//printf("\n @ %d:%d @",ROI_slider_y,ROI_slider_high);
		//printf("\n @ %d:%d:%d @",notehight0,notehight1,notehight2);

		printf("\n @ %d:",PointNormal[0]);
		printf("%d @",PointNormal[1]);

		float notewidth_base = ROI_slider_wide+ROI_slider_x;
		float notewidth0 = ROI_slider_x+(((notewidth_base-ROI_slider_x)/100)*11.5625);
		float notewidth1 = ROI_slider_x+(((notewidth_base-ROI_slider_x)/100)*20);

		PointNormal[0]=Point(notewidth0,notehight1);
		PointNormal[1]=Point(notewidth1,notehight2);

		float notelenght = notewidth1-notewidth0;

		for (int i =2;i<16;i+=2){
			PointNormal[i]=Point(notewidth0+(notelenght*(i/2))+(3*(i/2)),notehight1);
			PointNormal[i+1]=Point(notewidth1+(notelenght*(i/2))+(3*(i/2)),notehight2);
		}

		for (int i =0;i<16;i+=2){
			rectangle( sceneShow,PointNormal[i],PointNormal[i+1],Scalar(100, 0, 255, 255 ),0,8 );
		}
		
		setting_calibration_points=false;
		  //active callback souris
		 setMouseCallback( "Scene", onMouse, 0 );
			
		//trackerbar of thresholding
		createTrackbar ( "upper", "Image", &alpha_slider, alpha_slider_max, trackbar);
		createTrackbar ( "lower", "Image", &alpha_slider2, alpha_slider_max, trackbar2);
		//trackerbar of ROI
		createTrackbar ( "ROI_x", "Scene", &ROI_slider_x, ROI_x_slider_max, ROI_X_func);
		createTrackbar ( "ROI_y", "Scene", &ROI_slider_y, ROI_y_slider_max, ROI_Y_func);
		createTrackbar ( "ROI_wide", "Scene", &ROI_slider_wide, ROI_x_slider_max, ROI_WIDE_func);
		createTrackbar ( "ROI_high", "Scene", &ROI_slider_high, ROI_y_slider_max, ROI_HIGH_func);
		
		if (key=='r')
		{
			state = true;
			// set calibration points
            setting_calibration_points=true;
            calibration_point_index=0;
            printf("Defining 4 calibration points.... \n(draw a 4 points polygon by clicking 4 times \ninside the %s window with the mouse)\n","Scene");
			cout<<xMin<<"  "<<yMin<<"  "<<xMax<<"  "<<yMax<<endl;

			while (state)
			{
				// check consistance of (x|y)(Min|Max) values
				if ((xMin+5) > xMax)
					xMax=xMin+5;
				if ((yMin+5) > yMax)
					yMax=yMin+5;

				ROI_slider_x = xMin;
				ROI_slider_y = yMin;
				ROI_slider_wide = xMax - xMin;
				ROI_slider_high = yMax - yMin;
				
				waitKey(30);
			}
		}
		
		//imshow("eroded_Show",m_erodedShow);  //show window 
	    //imshow("dilate_Show",m_dilateShow);  //show window 
		//imshow("intersection_Show",m_intersectionShow);  //show window 
		imshow("Scene",sceneShow);  //show window of realtime scene
		imshow("Image",imageShow);  //show window of drawed image
		imshow("Depth", depthShow);    // show window of depth image
		//imshow("Game",gameShow);
		imshow("Game2",gameShow2);

		//namedWindow("Game", CV_WINDOW_NORMAL);
		//setWindowProperty("Game", CV_WND_PROP_ASPECTRATIO, CV_WINDOW_NORMAL);
		cvMoveWindow("Game", 1805, 70);
		imshow("Game",gameShow);

		key = waitKey(10) ; 
    }// while

	// destroy
	destroyWindow("Scene");
	destroyWindow("Image");
	destroyWindow("Depth");
	destroyWindow("Game");
	context.StopGeneratingAll() ;
    context.Shutdown() ;
	return 0;
}// end of main

Point hand_length_point ( Point pt1, Point pt2)
{
	int pt_x=0, pt_y=0;
	float ptx,pty;
	if(pt1.x<pt2.x)
	{
		ptx = ( sqrtf( (HAND_LENGTH*HAND_LENGTH)/(1+((pt2.y-pt1.y)*(pt2.y-pt1.y))/((pt2.x-pt1.x)*(pt2.x-pt1.x))) )+pt1.x );
		pty = ( ((ptx-pt1.x)*(pt2.y-pt1.y))/(pt2.x-pt1.x)+pt1.y);
	}
	if(pt1.x>pt2.x)
	{
		ptx = ( -sqrtf( (HAND_LENGTH*HAND_LENGTH)/(1+((pt2.y-pt1.y)*(pt2.y-pt1.y))/((pt2.x-pt1.x)*(pt2.x-pt1.x))) )+pt1.x );
		pty = ( ((ptx-pt1.x)*(pt2.y-pt1.y))/(pt2.x-pt1.x)+pt1.y );
	}
	if(pt1.x==pt2.x)
	{
		ptx = pt1.x;
		pty = pt1.y-HAND_LENGTH;
	}
	
	pt_x = (int) ptx;
	pt_y = (int) pty;

		//cout<<"  point:  "<<ptx<<"  "<<pty<<endl;
		return Point( pt_x, pt_y );
}

int point_in_out_rect (Point2f point0, Point2f vertex1, Point2f vertex2, Point2f vertex3, Point2f vertex4)
{
	float D1, D2;
	D1 = ( ((vertex1.y-vertex2.y)*point0.x) - ((vertex1.x-vertex2.x)*point0.y) +(vertex2.y*(vertex1.x-vertex2.x)) - (vertex2.x*(vertex1.y-vertex2.y)) ) *
		( ((vertex4.y-vertex3.y)*point0.x) - ((vertex4.x-vertex3.x)*point0.y) +(vertex3.y*(vertex4.x-vertex3.x)) - (vertex3.x*(vertex4.y-vertex3.y)) );
	D2 = ( ((vertex3.y-vertex2.y)*point0.x) - ((vertex3.x-vertex2.x)*point0.y) +(vertex2.y*(vertex3.x-vertex2.x)) - (vertex2.x*(vertex3.y-vertex2.y)) ) *
		( ((vertex4.y-vertex1.y)*point0.x) - ((vertex4.x-vertex1.x)*point0.y) +(vertex1.y*(vertex4.x-vertex1.x)) - (vertex1.x*(vertex4.y-vertex1.y)) );
	if (D1<0 && D2<0) return 1;
	//if ( (D1==0 && D2<0)||(D1<0 && D2==0) ) return 2;
	//if ( D1==0 && D2==0 ) return 3;
	else return -1;
}

double line_angle( Point2f ptStart1, Point2f ptEnd1, Point2f ptStart2, Point2f ptEnd2)
{
	double angle;
	double tan_alpha = (ptStart1.y - ptEnd1.y) / (ptStart1.x - ptEnd1.x);
	double tan_beta = (ptStart2.y - ptEnd2.y) / (ptStart2.x - ptEnd2.x);
	double tan_angle = abs(tan_alpha-tan_beta) / (1+tan_alpha*tan_beta);
	if ( tan_angle < 0) 
		angle = atan(tan_angle)* (180/3.1415926)+90;
	else 
		angle = atan(tan_angle)* (180/3.1415926);
	return angle;
}

Mat find_perspective_transform()
{
    vector<Point2f>pointsIn;
    vector<Point2f>pointsRes;

    for(int k=0;k<4;k++)
    pointsIn.push_back(calibration_points[k]);

    pointsRes.push_back(Point(0,0));
    pointsRes.push_back(Point(1,0));
    pointsRes.push_back(Point(1,1));
    pointsRes.push_back(Point(0,1));

    Mat m =getPerspectiveTransform(pointsIn,pointsRes);
	return(m);
}

void onMouse( int event, int x, int y, int, void* )
{
    vector<Point2f>  mouse_point(1);
    vector<Point2f>  scene_point;
    mouse_point[0]=Point(x,y);

    if( event != CV_EVENT_LBUTTONDOWN )
       return;
    if (setting_calibration_points)
	//  if( event == CV_EVENT_LBUTTONDOWN )
    {
        calibration_points[calibration_point_index] = Point(x,y);
        printf ("calibration point %d\n",calibration_point_index+1);
        calibration_point_index+=1;
        if (calibration_point_index==4)
        {
            // updating ROI boundaries
            xMin=min(min(calibration_points[0].x,calibration_points[1].x),min(calibration_points[2].x,calibration_points[3].x));
            xMax=max(max(calibration_points[0].x,calibration_points[1].x),max(calibration_points[2].x,calibration_points[3].x));
            yMin=min(min(calibration_points[0].y,calibration_points[1].y),min(calibration_points[2].y,calibration_points[3].y));
            yMax=max(max(calibration_points[0].y,calibration_points[1].y),max(calibration_points[2].y,calibration_points[3].y));

            // updating trackbars
            setTrackbarPos(	"ROI_x", "Scene" , xMin);
            setTrackbarPos(	 "ROI_wide", "Scene",xMax-xMin);
            setTrackbarPos(	"ROI_y", "Scene", yMin);
            setTrackbarPos(	"ROI_high", "Scene", yMax-yMin);

            H=find_perspective_transform();
            setting_calibration_points=false;
            printf("Matrice de transformation calculee...\n");
            //cout<<H;
        }
    }
    else
    {
        if(calibration_point_index==4)
        {
            perspectiveTransform(mouse_point,scene_point,H);
			state = false;
            printf ("mouse_new_coords=%f %f\n",scene_point[0].x,scene_point[0].y);
        }
    }
    return;
}

int handlocation()
{
	//int fingerarray[fingertips.size()+1];
	for (int p=0; p<fingertips.size(); p++)
	{
		//fingerarray[p]=fingertips[p];
	}
	return 0;
}

