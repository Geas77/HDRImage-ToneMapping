#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
void readImagesandTimes(vector<Mat> &images, vector<float> &times);
Mat  SampleIntensities(vector<Mat> images);
unsigned int weightFunction(unsigned int pixelValue);
Mat ComputeResponseCurve(Mat intensity_samples, vector<float> timesexp, float smoothing_lambda);
Mat ComputeRadianceMap(vector<Mat> images, vector<float> timesexp, Mat responseCurve);
bool AdaptHDR (Mat &scr, Mat&dst);
Mat ToneMapping(Mat& HDRImage, float intensity);
int main(int argc, char** argv)
{
	std::ofstream myfile;
	myfile.open("example.csv");
	srand(5);
	vector<Mat> sourceimages;
	vector<float> explosureTimes;
	readImagesandTimes(sourceimages, explosureTimes);
	cout << sourceimages[0].channels() << endl;
	
	Mat RadianceMap = Mat::zeros(sourceimages[0].rows, sourceimages[0].cols, CV_32FC3);
	Mat HDRImage;

	Mat samples = SampleIntensities(sourceimages);
	Mat responseCurve = ComputeResponseCurve(samples, explosureTimes, 10);
	
 	cout << responseCurve.size() << endl;
	for (int i = 0; i < 256; i++)
	{
		myfile << responseCurve.at<Vec3f>(i, 0)[0] << ",";
		myfile <<  responseCurve.at<Vec3f>(i, 0)[1] << ",";
		myfile << responseCurve.at<Vec3f>(i, 0)[2] <<endl;
	//	printf("%.2f\t", responseCurve.at<Vec3f>(i, 0)[0]);
	}
	myfile.close();
	RadianceMap = ComputeRadianceMap(sourceimages, explosureTimes, responseCurve);

	cv::normalize(RadianceMap, HDRImage, 0, 255, cv::NORM_MINMAX );
	std::ofstream myfile2;
	myfile2.open("example1.csv");
	for (int i = 0; i < sourceimages[0].rows; i++)
	{
		for (int j = 0; j < sourceimages[0].cols; j++)
		{
			myfile2 << HDRImage.at<Vec3f>(i, j)[0] << ",";
		}
		myfile2 << endl;
	}
	myfile2.close();
	
	Mat gray_image;
	cvtColor(HDRImage, gray_image, CV_BGR2GRAY);
	imwrite("./TestPhoto/HDR_Deb_image.hdr", HDRImage);
	
	Mat hdr=imread("./TestPhoto/HDR_Deb_image.hdr");
	Mat im_color;
	applyColorMap(hdr, im_color, COLORMAP_JET);
	imwrite("./TestPhoto/Radiance_image.png", im_color);//將強度圖輸出
	

	
	
	
	
	
	Mat toneImage = Mat::zeros(sourceimages[0].rows, sourceimages[0].cols, CV_8UC3 );
	toneImage = ToneMapping(RadianceMap,0.5f);
	Mat toneImagen;
	cv::normalize(toneImage, toneImagen, 0, 255, cv::NORM_MINMAX);
	imwrite("./TestPhoto/Tone_image.jpg", toneImage);
	
}
//讀圖片
void readImagesandTimes(vector<Mat> &images, vector<float> &times) {
	
	int numImages=16;
	string photopath = "./TestPhoto/";
	//static const float timesArray[] = {1/30.0f,1/15.0f,1/8.0f,1/4.0f,1/2.0f,1,2,4,8};
	static const float timesArray[] = { 1024,512,256,128,64,32,16,8,4,2,1,0.5f,0.25f,0.125f,0.0625f,0.03125f };
	//static const float timesArray[] = {1000/30000.0f,1000/16000.0f,1000/8000.0f,0.25,0.5,1,2,4,8,1000/60.0f,1000 / 30.0f, 1000/15.0f,1000/13.0f,1000/4.0f};
	//static const float timesArray[] = { 1000 / 30000.0f,1000 / 16000.0f,1000 / 8000.0f,0.25,0.5,1,2,4,8,1000 / 60.0f,1000 / 30.0f, 1000 / 15.0f,1000 / 8.0f,1000 / 4.0f };
	times.assign(timesArray, timesArray + numImages);
	//static const char* filenames[] = {"img01.jpg", "img02.jpg", "img03.jpg", "img04.jpg","img05.jpg","img06.jpg","img07.jpg","img08.jpg","img09.jpg"};
	static const char* filenames[] = { "memorial00.png", "memorial01.png", "memorial03.png", "memorial04.png","memorial05.png","memorial06.png","memorial07.png","memorial08.png", "memorial09.png", "memorial10.png", "memorial11.png","memorial12.png","memorial13.png","memorial14.png","memorial15.png" };
	for (int i = 0; i < numImages; i++)
	{
		char tmp[10];
		sprintf_s(tmp, "%02d",i);
		string number(tmp);
		Mat im = imread(photopath +"memorial"+number+".png");
		images.push_back(im); 
	}
	
}
//Weight Function
unsigned int weightFunction(unsigned int pixelValue) {
	unsigned int z_min = 0, z_max = 255;
		if (pixelValue <= (z_min + z_max) / 2)
			return pixelValue - z_min;
		else
			return z_max - pixelValue;

}
// Traverse check
bool inside(int r, int c, int rows, int cols)
{
	return r >= 0 && r < rows && c >= 0 && c < cols;
}
Mat SampleIntensities(vector<Mat> images) {

	int Zmin = 0, Zmax = 255;
	int num_intensities = 100;// 一張圖有幾個 Sample
	int num_images = images.size(); //幾張圖 
	////////////////////////////////////////////
	int rows = images[0].rows;
	int cols = images[0].cols;

	// Generate Samples 
	int sample_x[100];
	int sample_y[100];

	int col_num = (int)sqrt(1.f * num_intensities * cols / rows);
	int row_num = num_intensities / col_num;
	num_intensities = 0;
	int col = (cols / col_num) / 2;
	for (int i = 0; i < col_num; i++) {
		int row = (rows / row_num) / 2;
		for (int j = 0; j < row_num; j++) {
			if (inside(row, col, rows, cols)) {
				sample_x[num_intensities] = col;
				sample_y[num_intensities] = row;
				num_intensities++;
			}
			row += (rows / row_num);
		}
		col += (cols / col_num);
	}
	printf("%d\n", num_intensities);
	//////////////////////////////////////////////
	Mat intensity_values = Mat::zeros(num_intensities, num_images, CV_8UC3);
	Mat midImage = images[num_images / 2];
	int rowCount = images[0].rows;
	int CoulumnCount = images[0].cols;
	printf("%d %d\n", rowCount, CoulumnCount);
		for (int i = 0; i < num_intensities; i++)
		{
			int x = (rand() % (CoulumnCount));
			int y = (rand() % (rowCount));
			for (int j = 0; j < num_images; j++)
			{
				for (int ch = 0; ch < 3; ch++) {
					intensity_values.at<Vec3b>(i, j)[ch] = images[j].at<Vec3b>(sample_y[i], sample_x[i])[ch];//OpenCv Semple
					//intensity_values.at<Vec3b>(i, j)[ch] = images[j].at<Vec3b>(y, x)[ch]; //RandomSample
				}
			}
		}
	return intensity_values;
}
// f^-1 輸入8bit 數值， 找尋12bit真實HDR
Mat ComputeResponseCurve(Mat intensity_samples, vector<float> timesexp, float smoothing_lambda)
{
	int z_min = 0, z_max = 255;
	int intensity_range = 255;
	int num_samples = intensity_samples.rows;
	int num_images = timesexp.size();
	Mat x_star= Mat::zeros(256, 1, CV_32FC3);
	printf("start computeResponseCurve\n");
	printf("samples %d\n", num_samples);
	cout << x_star.size() << endl;
	for (int channel = 0; channel < 3; channel++) {
		Mat A = Mat::zeros(num_images * num_samples + intensity_range, num_samples + intensity_range + 1, CV_32F);
		Mat b = Mat::zeros(A.rows, 1, CV_32F);
		int rowCount = 0;
		for (int i = 0; i < num_samples; i++) {
			for (int j = 0; j < num_images; j++)
			{
				//Gzij-ln Ei=ln t
			    int zij = intensity_samples.at<Vec3b>(i, j)[channel];
				float wij = (float)weightFunction(zij);
				A.at<float>(rowCount, zij) =wij;//用亮度來決定g(1*weight function)
				A.at<float>(rowCount, (intensity_range + 1) + i) = -wij; //根據照片的pixel 決定 -Ei 的位置
				b.at<float>(rowCount, 0) = wij * log(timesexp[j]); //根據照片數決定
				rowCount++;
			}
		}
		A.at<float>(rowCount, 128) = 1;
		rowCount++;
		//矩陣剩下254項
		for (int i = 1; i < z_max; i++)
		{
			float wij = (float)weightFunction(i);
			A.at<float>(rowCount, i - 1) = wij * smoothing_lambda;
			A.at<float>(rowCount, i) = -2 * wij*smoothing_lambda;
			A.at<float>(rowCount, i + 1) = wij * smoothing_lambda;
			rowCount++;
		}
		Mat x;
		solve(A, b,x, DECOMP_SVD);//用svd解決x
		for (int k = 0; k < 256; k++)
		{
			x_star.at<Vec3f>(k,0)[channel]= x.at<float>(k);
		}
	}
	printf("finish computeResponseCurve\n");
	return x_star;
}
Mat ComputeRadianceMap(vector<Mat> images, vector<float> timesexp, Mat responseCurve) {
	printf("start ComputeRadianceMap\n");
	Mat imageMap = Mat::zeros(images[0].rows, images[0].cols , CV_32FC3);
	int num_images = images.size();
	for (int channels = 0; channels < 3; channels++)
	{
		for (int i = 0; i < imageMap.rows; i++)
		{
			for (int j = 0; j < imageMap.cols; j++)
			{
				float E = 0;
				float E2 = 0;
				float weightsum=0;
				
				for (int p = 0; p < num_images; p++)
				{
					E += weightFunction(images[p].at<Vec3b>(i, j)[channels])*(responseCurve.at<Vec3f>(images[p].at<Vec3b>(i, j)[channels],0)[channels]-log(timesexp[p]));
					E2+= (responseCurve.at<Vec3f>(images[p].at<Vec3b>(i, j)[channels], 0)[channels]-log(timesexp[p]));
					weightsum += weightFunction(images[p].at<Vec3b>(i, j)[channels]);
				}
				if (weightsum > 0)
				E = E / weightsum;
				else
				E = E2/num_images;
				imageMap.at<Vec3f>(i, j)[channels] =exp(E);
			}
		}
	}
	printf("finish ComputeRadianceMap\n");
	return imageMap;
}

Mat ToneMapping(Mat &HDRImage,float intensity) {
	Mat LMchannels= Mat::zeros(HDRImage.rows, HDRImage.cols, CV_32FC3);
	Mat LDchannels = Mat::zeros(HDRImage.rows, HDRImage.cols, CV_32FC3);
	cout << HDRImage.rows << endl;
	cout << HDRImage.cols << endl;
	float Lav = 0;
	float maxL[3] = {0,0,0};

	for (int i = 0; i < HDRImage.rows; i++)
	{
		for (int j = 0; j < HDRImage.cols; j++)
		{
			float gray =( 0.0722*HDRImage.at<Vec3f>(i, j)[0] + 0.7152*HDRImage.at<Vec3f>(i, j)[1] + 0.2126*HDRImage.at<Vec3f>(i, j)[2]);
			Lav += log(gray+0.0001f);
		}
	}
	float x;
	Lav =exp(Lav / (HDRImage.rows*HDRImage.cols));
	float y;
	float z;
	for (int i = 0; i < HDRImage.rows; i++)
	{
		for (int j = 0; j < HDRImage.cols; j++)
		{
			float Lw=0.0722*HDRImage.at<Vec3f>(i, j)[0] + 0.7152*HDRImage.at<Vec3f>(i, j)[1] +0.2126*HDRImage.at<Vec3f>(i, j)[2];
			float Lm = Lw * intensity / Lav;
			float Ld = Lm/ (1 + Lm);
			for (int ch = 0; ch < 3; ch++)
			{
         		LDchannels.at<Vec3f>(i, j)[ch] =255* HDRImage.at<Vec3f>(i, j)[ch]* Ld/ Lw;
				//(1+ LMchannels.at<Vec3f>(i, j)[ch]/powf(1.5,2))*(1+Lm/powf(1.5, 2))
				z = LDchannels.at<Vec3f>(i, j)[ch]*255;
			}
		}
	}
	return LDchannels;
}
bool AdaptHDR(Mat &scr,Mat&dst) {


		int row = scr.rows;
		int col = scr.cols;


		Mat ycc;                        //轉換空間到YUV；
		cvtColor(scr, ycc, COLOR_RGB2YUV);

		vector<Mat> channels(3);        //分離通道，取channels[0]；
		split(ycc, channels);


		Mat Luminance(row, col, CV_32FC1);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				Luminance.at<float>(i, j) = (float)channels[0].at<uchar>(i, j) / 255;
			}
		}


		double log_Ave = 0;
		double sum = 0;
		for (int i = 0; i < row; i++)                 //求對數均值
		{
			for (int j = 0; j < col; j++)
			{
				sum += log(0.001 + Luminance.at<float>(i, j));
			}
		}
		log_Ave = exp(sum / (row*col));

		double MaxValue, MinValue;      //獲取亮度最大值為MaxValue；
		minMaxLoc(Luminance, &MinValue, &MaxValue);

		Mat hdr_L(row, col, CV_32FC1);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				hdr_L.at<float>(i, j) = log(1 + Luminance.at<float>(i, j) / log_Ave) / log(1 + MaxValue / log_Ave);


				if (channels[0].at<uchar>(i, j) == 0)   //對應作者程式碼中的gain = Lg ./ Lw;gain(find(Lw == 0)) = 0;	
				{
					hdr_L.at<float>(i, j) = 0;
				}
				else
				{
					hdr_L.at<float>(i, j) /= Luminance.at<float>(i, j);
				}

			}
		}

		vector<Mat> rgb_channels;        //分別對RGB三個通道進行提升
		split(scr, rgb_channels);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				int r = rgb_channels[0].at<uchar>(i, j) *hdr_L.at<float>(i, j); if (r > 255) { r = 255; }
				rgb_channels[0].at<uchar>(i, j) = r;

				int g = rgb_channels[1].at<uchar>(i, j) *hdr_L.at<float>(i, j); if (g > 255) { g = 255; }
				rgb_channels[1].at<uchar>(i, j) = g;

				int b = rgb_channels[2].at<uchar>(i, j) *hdr_L.at<float>(i, j); if (b > 255) { b = 255; }
				rgb_channels[2].at<uchar>(i, j) = b;
			}
		}
		merge(rgb_channels, dst);
		return true;
}


