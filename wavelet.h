#pragma once
#include <opencv.hpp>
#include <iostream>
#include <vector>

namespace my_wavelet {

	enum WAVENAME {
		whaar,
		wdb1, wdb2, wdb3
	};

	class Wavelet {

	public:
		// 小波分解
		int wavedec2d(const cv::Mat &grayMat, int level, WAVENAME name, std::vector<cv::Mat> &decMats, std::vector<cv::Size> &decMatsWH);

		// 小波重构
		int waverec2d(WAVENAME name, const std::vector<cv::Mat>& decMats, std::vector<cv::Size>& decMatsWH, cv::Mat &recMat);

		// 自动阈值降噪
		int waveautoden(std::vector<cv::Mat>& decMats);

		// 小波模极大值边缘检测
		int waveedge(const cv::Mat& grayMat, int scale, float thresh, cv::Mat & edgeMat, int size = 5, float sigma = 1.0);


		// 一维小波分解
		int wavedec1d(const std::vector<float>& inputData, int level, WAVENAME name, std::vector<std::vector<float>> &decData, std::vector<int> &decSize, std::vector<std::vector<float>>& tempData);
		// 一维小波重构
		int waverec1d(WAVENAME name, const std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<float> & outputData, std::vector<std::vector<float>>& tempData);

		// 一维小波分解，正交矩阵
		int wavedec1d_matrix(const std::vector<float>& inputData, int level, WAVENAME name, std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<std::vector<float>>& tempData);
		// 一维小波重构，正交矩阵的逆
		int waverec1d_matrix(WAVENAME name, const std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<float>& outputData, std::vector<std::vector<float>>& tempData);


	};

}

