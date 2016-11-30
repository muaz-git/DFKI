#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "json.h"
#include "main.h"

using namespace cv;
using namespace std;

#define STROKE_LENGTH_THRESHOLD 12
#define IMAGE_HEIGHT 998
#define MAX_STROKES 6
#define MAX_DISTANCE 6

void thinningIteration(Mat & img, int iter);
void thinning(const Mat & src, Mat & dst);

Point findTopLeftMostPixel(Mat& strokeHistory, Point& searchFrom);
Point localSolver(Point a, Point b, vector<Point> possiblePoints);
double getOrientation(vector<Point> pixels);
Point globalSolver(Mat& strokeHistory, vector<Point> currentStroke, vector<Point> possiblePoints);
Point firstDecisicon(Mat& strokeHistory, vector<Point> currentStroke, vector<Point> possiblePoints);
Point findNextPixel(Mat& strokeHistory, vector<Point> currentStroke);
Point findNextPixelSimple(Mat& strokeHistory, vector<Point> currentStroke);


void thinningIteration(Mat & img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;

	uchar *pDst;

	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}

void thinning(const Mat & src, Mat & dst)
{
	dst = src.clone();
	dst /= 255;

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(dst, 0);
		thinningIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}

Point findTopLeftMostPixel(Mat& strokeHistory, Point& searchFrom)
{
	for (int i = searchFrom.y; i < strokeHistory.rows; i++)
	{
		for (int j = 1; j < strokeHistory.cols; j++) {
			if (strokeHistory.at<uchar>(i,j) == 255) {
				searchFrom = Point(j, i);
				return Point(j, i);
			}

		}
	}
	return Point(-1, -1);
}

Point localSolver(Point a, Point b, vector<Point> possiblePoints)
{
	Point ab = (b - a);
	double minAngle = 360;
	Point bestPoint;

	for (int i = 0; i < possiblePoints.size(); i++)
	{
		Point ac = (possiblePoints[i] - a);
		int angleTmp = ((acos((ab.dot(ac)) / (norm(ab) * norm(ac))) * 180) / 3.1415) - 180;
		int angle = angleTmp < 0 ? angleTmp * -1 : angleTmp;

		if (angle < minAngle)
		{
			minAngle = angle;
			bestPoint = possiblePoints[i];
		}
	}
	return bestPoint;
}

double getOrientation(vector<Point> pixels)
{
	Mat data_pts = Mat(pixels.size(), 2, CV_64FC1);
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = pixels[i].x;
		data_pts.at<double>(i, 1) = pixels[i].y;
	}
	
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; ++i)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0), pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i, 0);
	}
	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); 
	return angle;
}

Point globalSolver(Mat& strokeHistory, vector<Point> currentStroke, vector<Point> possiblePoints)
{
	vector<Point>::const_iterator first = currentStroke.end() - STROKE_LENGTH_THRESHOLD;
	vector<Point>::const_iterator last = currentStroke.end();
	vector<Point> subStroke(first, last);

	double angle1 = getOrientation(subStroke);

	int angle1Degrees = (angle1 * 180) / 3.1415;

	double minAngle = 1000;
	Point bestPoint(-1, -1);
	for (int i = 0; i < possiblePoints.size(); i++)
	{
		Mat strokeHistoryCopy = strokeHistory.clone();

		vector<Point> futureStroke = vector<Point>();
		futureStroke.push_back(possiblePoints[i]);
		strokeHistoryCopy.at<uchar>(possiblePoints[i]) = 0;


		Point nextPixel;
		while ((nextPixel = findNextPixelSimple(strokeHistoryCopy, futureStroke)).x != -1 )
		{
			futureStroke.push_back(nextPixel);
			strokeHistoryCopy.at<uchar>(nextPixel) = 0;
			if (futureStroke.size() >= STROKE_LENGTH_THRESHOLD) break;
		}
		if (futureStroke.size() < STROKE_LENGTH_THRESHOLD) continue;

		double angle2 = getOrientation(futureStroke);
		int angle2Degrees = (angle2 * 180) / 3.14159265;

		double angleDegrees = angle1Degrees - angle2Degrees < 0 ? angle2Degrees - angle1Degrees : angle1Degrees - angle2Degrees;

		if (angleDegrees < minAngle)
		{
			minAngle = angleDegrees;
			bestPoint = possiblePoints[i];
		}
	}

	return bestPoint;
}

Point firstDecisicon(Mat& strokeHistory, vector<Point> currentStroke, vector<Point> possiblePoints)
{
	Point bestPoint = Point(-1, -1);
	int maxLength = 0;
	for (int i = 0; i < possiblePoints.size(); i++)
	{
		Mat strokeHistoryCopy = strokeHistory.clone();
		vector<Point> currentStrokeCopy = vector<Point>(currentStroke);

		currentStrokeCopy.push_back(possiblePoints[i]);
		strokeHistoryCopy.at<uchar>(possiblePoints[i]) = 0;


		Point nextPixel = possiblePoints[i];
		while ((nextPixel = findNextPixel(strokeHistoryCopy, currentStrokeCopy)).x != -1)
		{
			currentStrokeCopy.push_back(nextPixel);
			strokeHistoryCopy.at<uchar>(nextPixel) = 0;
		}
		if (currentStrokeCopy.size() > maxLength)
		{
			bestPoint = possiblePoints[i];
			maxLength = currentStrokeCopy.size();
		}
	}
	return bestPoint;
}

Point findNextPixel(Mat& strokeHistory, vector<Point> currentStroke) 
{
	Point currentPixel = currentStroke.back();

	int pointCoords[16] = {
		currentPixel.x - 1, currentPixel.y - 1,
		currentPixel.x - 1, currentPixel.y,
		currentPixel.x - 1, currentPixel.y + 1,
		currentPixel.x, currentPixel.y - 1,
		currentPixel.x, currentPixel.y + 1,
		currentPixel.x + 1, currentPixel.y - 1,
		currentPixel.x + 1, currentPixel.y,
		currentPixel.x + 1, currentPixel.y + 1
	};
	int matWidth = strokeHistory.size().width;
	int matHeight = strokeHistory.size().height;

	vector<Point> possiblePoints;
	for (int i = 0; i < 8; i++)
	{

		if (strokeHistory.at<uchar>(pointCoords[i * 2 + 1], pointCoords[i * 2]) == 255)
		{
			possiblePoints.push_back(Point(pointCoords[i * 2], pointCoords[i * 2 + 1]));
		}
	}
	/*No Pixel found in neighbourhood*/
	if (possiblePoints.size() == 0) 
	{
		return Point(-1, -1);
	}
	/*Only one possible Pixel found*/
	if (possiblePoints.size() == 1)
	{
		return possiblePoints.back();
	}
	/*Multiple possible Pixels found*/
	if (possiblePoints.size() > 1)
	{
		/*Actual Stroke Length is higher or equals 2*/
		if (currentStroke.size() >= 2)
		{
			if (currentStroke.size() <= STROKE_LENGTH_THRESHOLD )
			{
				Point previousPixel = currentStroke[currentStroke.size() - 2];
				return localSolver(currentPixel, previousPixel, possiblePoints);
			}
			else
			{
				return globalSolver(strokeHistory, currentStroke, possiblePoints);
			}
		}
		/*Actual Stroke Length is less than 2*/
		else
		{
			Point best = firstDecisicon(strokeHistory, currentStroke, possiblePoints);
			return best;
		}
	}
}

Point findNextPixelSimple(Mat& strokeHistory, vector<Point> currentStroke)
{
	Point currentPixel = currentStroke.back();

	int pointCoords[16] = {
		currentPixel.x - 1, currentPixel.y - 1,
		currentPixel.x - 1, currentPixel.y,
		currentPixel.x - 1, currentPixel.y + 1,
		currentPixel.x, currentPixel.y - 1,
		currentPixel.x, currentPixel.y + 1,
		currentPixel.x + 1, currentPixel.y - 1,
		currentPixel.x + 1, currentPixel.y,
		currentPixel.x + 1, currentPixel.y + 1
	};

	int matWidth = strokeHistory.size().width;
	int matHeight = strokeHistory.size().height;

	vector<Point> possiblePoints;
	for (int i = 0; i < 8; i++)
	{
		if (pointCoords[i * 2 + 1] < 0 || pointCoords[i * 2] < 0) continue;
		if (pointCoords[i * 2 + 1] >= matHeight || pointCoords[i * 2] >= matWidth) continue;
		if (strokeHistory.at<uchar>(pointCoords[i * 2 + 1], pointCoords[i * 2]) == 255)
		{
			possiblePoints.push_back(Point(pointCoords[i * 2], pointCoords[i * 2 + 1]));
		}
	}
	/*No Pixel found in neighbourhood*/
	if (possiblePoints.size() == 0)
	{
		return Point(-1, -1);
	}
	/*Only one possible Pixel found*/
	if (possiblePoints.size() == 1)
	{
		return possiblePoints.back();
	}
	/*Multiple possible Pixels found*/
	if (possiblePoints.size() > 1)
	{
		/*Actual Stroke Length is higher or equals 2*/
		if (currentStroke.size() >= 2)
		{
			if (currentStroke.size() <= STROKE_LENGTH_THRESHOLD)
			{
				Point previousPixel = currentStroke[currentStroke.size() - 2];
				return localSolver(currentPixel, previousPixel, possiblePoints);
			}
			else
			{
				return globalSolver(strokeHistory, currentStroke, possiblePoints);
			}
		}
		/*Actual Stroke Length is less than 2*/
		else
		{
			return possiblePoints.back();
		}
	}
}


vector<vector<Point>> splitStroke(vector<Point> stroke)
{
	vector<vector<Point>> strokes = vector<vector<Point>>();

	if (stroke.size() < STROKE_LENGTH_THRESHOLD * 4)
	{
		strokes.push_back(stroke);
		return strokes;
	}
	vector<int> splitAt = vector<int>();
	splitAt.push_back(0);
	for (int i = 0; i < stroke.size() - (STROKE_LENGTH_THRESHOLD * 2); i = i + STROKE_LENGTH_THRESHOLD)
	{
		vector<Point>::const_iterator first = stroke.begin() + i;
		vector<Point>::const_iterator last = stroke.begin() + i + (STROKE_LENGTH_THRESHOLD);
		vector<Point> firstSubStroke(first, last);

		first = stroke.begin() + i + (STROKE_LENGTH_THRESHOLD);
		last = stroke.begin() + i + (STROKE_LENGTH_THRESHOLD * 2);
		vector<Point> secondSubStroke(first, last);

		int angle1 = (getOrientation(firstSubStroke) * 180 ) / 3.14159265;
		int angle2 = (getOrientation(secondSubStroke) * 180) / 3.14159265;

		int angle = angle1 < angle2 ? angle2 - angle1 : angle1 - angle2;
		if (angle > 70) {
			splitAt.push_back(i + STROKE_LENGTH_THRESHOLD);
		}
		
	}
	splitAt.push_back(stroke.size() - 1);
	for(int i = 0; i < splitAt.size() - 1; i++) 
	{
		vector<Point>::const_iterator first = stroke.begin() + splitAt[i];
		vector<Point>::const_iterator last = stroke.begin() + splitAt[i + 1];
		vector<Point> subStroke(first, last);
		strokes.push_back(subStroke);
	}
	
	return strokes;
}

void readInputFile(string path, string& content)
{
	string line;
	ifstream inputFile(path);
	if (inputFile.is_open())
	{
		while (getline(inputFile, line))
		{
			content += line + '\n';
		}
		inputFile.close();
	}

	else cout << "Unable to open file";
}

double euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

double strokeDistance(vector<Point>& stroke1, vector<Point>& stroke2) {
	double minDist = 99999;
	for (int i = 0; i < stroke1.size(); i++) {
		for (int j = 0; j < stroke2.size(); j++) {
			double dist = euclideanDist(stroke1[i], stroke2[j]);
			if (dist < minDist)
				minDist = dist;
		}
	}
	return minDist;
}

double strokeDistanceSimple(vector<Point>& stroke1, vector<Point>& stroke2) {
	double minDist = 99999;

	if (euclideanDist(stroke1[0], stroke2[0]) < minDist) 
	{
		
		minDist = euclideanDist(stroke1[0], stroke2[0]);
	}
		
	if (euclideanDist(stroke1[0], stroke2.back()) < minDist) 
	{
		cout << euclideanDist(stroke1[0], stroke2[0]) << endl;
		minDist = euclideanDist(stroke1[0], stroke2.back());
	}
		
	if (euclideanDist(stroke1.back(), stroke2[0]) < minDist) 
	{
		minDist = euclideanDist(stroke1.back(), stroke2[0]);
	}
		
	if (euclideanDist(stroke1.back(), stroke2.back()) < minDist) 
	{
		minDist = euclideanDist(stroke1.back(), stroke2.back());
	}

	return minDist;
}

vector<vector<int>> distanceMatrix(vector<vector<Point>>& strokes)
{
	vector<vector<int>> distanceMatrix = vector<vector<int>>(strokes.size(),vector<int>(strokes.size(), 0));
	for (int i = 0; i < strokes.size(); i++)
	{
		for (int j = i; j < strokes.size(); j++)
		{
			distanceMatrix[i][j] = strokeDistanceSimple(strokes[i], strokes[j]);
		}
	}
	return distanceMatrix;
}

void findStrokes(Mat source, vector<vector<Point>>& output)
{
	Mat thinned = Mat(source.size(), CV_8UC1);
	thinning(source, thinned);

	//namedWindow("Thinned", WINDOW_AUTOSIZE);
	//imshow("Thinned", thinned);

	/*
	BUILD STROKE HISTORY
	*/
	Mat strokeHistory = thinned.clone();

	/*
	REQUIRED DATA ELEMENTS
	*/
	vector<vector<Point>> strokes = vector<vector<Point>>();

	Point searchFrom = Point(0, 0);
	Point topLeftMostPixel;

	RNG rng(62345);

	/*
	ITERATIVE STROKE EXTRACTION
	*/
	while ((topLeftMostPixel = findTopLeftMostPixel(strokeHistory, searchFrom)).x != -1)
	{
		vector<Point> currentStroke = vector<Point>();
		currentStroke.push_back(topLeftMostPixel);
		strokeHistory.at<uchar>(topLeftMostPixel) = 0;

		Point nextPixel;
		while ((nextPixel = findNextPixel(strokeHistory, currentStroke)).x != -1)
		{
			currentStroke.push_back(nextPixel);
			strokeHistory.at<uchar>(nextPixel) = 0;
		}
		if (currentStroke.size() > 1)
		{
			Mat result = Mat::zeros(thinned.size(), CV_8UC3);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int j = 0; j < currentStroke.size() - 1; j++) {
				line(result, currentStroke[j], currentStroke[j + 1], color);
			}

			strokes.push_back(currentStroke);
		}
	}
	
	for (int i = 0; i < strokes.size(); i++)
	{
		vector<vector<Point>> subStrokes = splitStroke(strokes[i]);
		for (int j = 0; j < subStrokes.size(); j++)
		{
			output.push_back(subStrokes[j]);
		}
	}
}

void showDistanceGraphic(vector<Point>& stroke1, vector<Point>& stroke2)
{
	Mat groupMat = Mat::zeros(Size(800, 1200), CV_8UC3);
	
	for (int j = 0; j < stroke1.size() - 1; j++) {
		if (j == 0 || j == stroke1.size() - 2)
			line(groupMat, stroke1[j], stroke1[j + 1], Scalar(255, 255, 255));
		else
			line(groupMat, stroke1[j], stroke1[j + 1], Scalar(40, 40, 40));
	}
	for (int j = 0; j < stroke2.size() - 1; j++) {
		if (j == 0 || j == stroke2.size() - 2)
			line(groupMat, stroke2[j], stroke2[j + 1], Scalar(255, 255, 255));
		else
			line(groupMat, stroke2[j], stroke2[j + 1], Scalar(40, 40, 40));
	}
	
	cout << "Distance " << strokeDistanceSimple(stroke1, stroke2) << endl;
	namedWindow("Group Proposal", WINDOW_AUTOSIZE);
	imshow("Group Proposal", groupMat);


	waitKey(0);
}

void showGroupProposal(vector<vector<Point>> &strokes)
{
	int maxy = 0, maxx = 0;
	int miny = 100000, minx = 100000;
	if (strokes.size() == 0)
	{
		return;
	}
	for (int i = 0; i < strokes.size(); i++)
	{
		for (int j = 0; j < strokes[i].size(); j++)
		{
			if (strokes[i][j].x > maxx)
			{
				maxx = strokes[i][j].x;
			}
			if (strokes[i][j].y > maxy)
			{
				maxy = strokes[i][j].y;
			}
			if (strokes[i][j].y < miny)
			{
				miny = strokes[i][j].y;
			}
			if (strokes[i][j].x < minx)
			{
				minx = strokes[i][j].x;
			}
		}
	}
	Mat groupMat = Mat::zeros(Size(800, 1200), CV_8UC3);
	for (int i = 0; i < strokes.size(); i++)
	{
		for (int j = 0; j < strokes[i].size() - 1; j++) {
			if(j == 0 || j == strokes[i].size() - 2)
				line(groupMat, strokes[i][j], strokes[i][j + 1], Scalar(255, 255, 255));
			else
				line(groupMat, strokes[i][j], strokes[i][j + 1], Scalar(40, 40, 40));
		}
	}

	namedWindow("Group Proposal", WINDOW_AUTOSIZE);
	imshow("Group Proposal", groupMat);


	waitKey(0);

}

vector<vector<Point>> groupProposals(vector<vector<Point>>& strokes)
{
	unordered_set<int> lookUp = unordered_set<int>();
	vector<vector<int>> groups = vector<vector<int>>();
	vector<vector<int>> output = vector<vector<int>>();

	//vector<vector<int>> matrix = distanceMatrix(strokes);

	for (int i = strokes.size() - 1; i >= 0 ; i--) {
		int temp = 1 << i;
		lookUp.insert(temp);
		vector<int> group = vector<int>();
		group.push_back(i);
		groups.push_back(group);
	}

	while(groups.size() > 0)
	{
		vector<int> group = groups.back();
		groups.pop_back();
		output.push_back(group);

		for (int j = 0; j < strokes.size(); j++)
		{
			
			if (strokeDistanceSimple(strokes[j], strokes[group.back()]) > MAX_DISTANCE) continue;
			cout << strokeDistanceSimple(strokes[j], strokes[group.back()]) << endl;
			vector<int> proposedGroup = vector<int>();
			int proposedGroupBitMap = 0;
			for (int l = 0; l < group.size(); l++)
			{
				proposedGroupBitMap |= 1 << group[l];
				proposedGroup.push_back(group[l]);
			}
			proposedGroupBitMap |= 1 << j;
			proposedGroup.push_back(j);
			if (lookUp.find(proposedGroupBitMap) != lookUp.end()) continue;
			if (proposedGroup.size() >= MAX_STROKES) continue;
			lookUp.insert(proposedGroupBitMap);
			groups.push_back(proposedGroup);
		}
	}

	cout << "Number of proposed groups " << groups.size() << endl;

	for (int i = 0; i < output.size(); i++) 
	{
		vector<vector<Point>> strokeGroup = vector<vector<Point>>();
		vector<int> group = output[i];

		for (int j = 0; j < group.size(); j++)
		{
			strokeGroup.push_back(strokes[group[j]]);
		}
		showGroupProposal(strokeGroup);	
	}

	
	return vector<vector<Point>>();
}

bool rectContainsStroke(vector<Point> stroke, Rect rect)
{
	int count = 0;
	for (int i = 0; i < stroke.size(); i++)
	{
		if (rect.contains(stroke[i]))
			count++;
	}
	if (count > 20)
		return true;
	else
		return false;
}



int main(int argc, char** argv)
{
	string content;
	readInputFile("input.json", content);

	json::Object obj = json::Deserialize(content);

	json::Array data = obj["data"];

	cout << data.size() << endl;
	for (int i = 0; i < data.size(); i++)
	{
		json::Object sketchData = data[i];

		string filepath = sketchData["filePath"];
		json::Array sampleData = sketchData["samples"];
		
		Mat image;
		image = imread("C:\\Users\\zapp\\Documents\\workspaces\\workspace_sts\\indigo\\src\\main\\webapp\\resources\\img\\sketches\\" + filepath, CV_8UC1);


		if (image.empty()) // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		Mat original;
		image.copyTo(original);
		bitwise_not(original, original);
		dilate(original, original, Mat(), Point(-1,-1), 2);
		bitwise_not(original, original);

		//GaussianBlur(original, image, Size(3, 3), 0, 0, 4);
		//GaussianBlur(image, image, Size(3, 3), 0, 0, 4);

		adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 7);
		morphologyEx(image, image, MORPH_CLOSE, Mat(), Point(-1, -1), 6);
		GaussianBlur(image, image, Size(9, 9), 0, 0.0, 4);

		double scaleFactor = (double)IMAGE_HEIGHT / (double)(image.size().height);
		double width = scaleFactor * (double)(image.size().width);
		Size size = Size(width, IMAGE_HEIGHT);

		resize(image, image, size);
		resize(original, original, size);
		copyMakeBorder(image, image, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

		Mat thinned = image.clone();
		thinning(image, thinned);

		//====================================================================================================

		vector<vector<Point>> strokes = vector<vector<Point>>();
		findStrokes(image, strokes);

		RNG rng(21345);

		Mat result = Mat::zeros(image.size(), CV_8UC3);
		for (int i = 0; i < strokes.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int j = 0; j < strokes[i].size() - 1; j++) {
				line(result, strokes[i][j], strokes[i][j + 1], color);
			}
		}

		//====================================================================================================

		//namedWindow("Image", WINDOW_AUTOSIZE);
		//imshow("Image", image);
		//waitKey(0);


		/*for (int i = 0; i < strokes.size() / 24; i++)
		{
		vector<vector<Point>>::const_iterator first = strokes.begin() + 24 * i;
		vector<vector<Point>>::const_iterator last = strokes.begin() + (24 * (i + 1));

		if((24 * (i + 1)) > strokes.size())
		last = strokes.begin() + strokes.size() - 1;
		vector<vector<Point>> newVec(first, last);
		groupProposals(newVec);
		}*/

		/*for (int i = 0; i < strokes.size() / 32; i++)
		{
		vector<vector<Point>>::const_iterator first = strokes.begin() + 24 * i + 12;
		vector<vector<Point>>::const_iterator last = strokes.begin() + (24 * (i + 1)) + 12;

		if ((24 * (i + 1)) + 12 > strokes.size())
		last = strokes.begin() + strokes.size() - 1;
		vector<vector<Point>> newVec(first, last);
		groupProposals(newVec);
		}*/


		//groupProposals(strokes);

		namedWindow("Result", WINDOW_AUTOSIZE);
		for (int i = 0; i < 5; i++)
		{
			imshow("Result", original);
			waitKey(800);
			imshow("Result", image);
			waitKey(800);
			imshow("Result", thinned);
			waitKey(800);
			imshow("Result", result);
			waitKey(800);


		}
		waitKey();
		

	}

	return 0;
}