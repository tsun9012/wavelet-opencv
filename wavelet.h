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
		// С���ֽ�
		int wavedec2d(const cv::Mat &grayMat, int level, WAVENAME name, std::vector<cv::Mat> &decMats, std::vector<cv::Size> &decMatsWH);

		// С���ع�
		int waverec2d(WAVENAME name, const std::vector<cv::Mat>& decMats, std::vector<cv::Size>& decMatsWH, cv::Mat &recMat);

		// �Զ���ֵ����
		int waveautoden(std::vector<cv::Mat>& decMats);

		// С��ģ����ֵ��Ե���
		int waveedge(const cv::Mat& grayMat, int scale, float thresh, cv::Mat & edgeMat, int size = 5, float sigma = 1.0);


		// һάС���ֽ�
		int wavedec1d(const std::vector<float>& inputData, int level, WAVENAME name, std::vector<std::vector<float>> &decData, std::vector<int> &decSize, std::vector<std::vector<float>>& tempData);
		// һάС���ع�
		int waverec1d(WAVENAME name, const std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<float> & outputData, std::vector<std::vector<float>>& tempData);

		// һάС���ֽ⣬��������
		int wavedec1d_matrix(const std::vector<float>& inputData, int level, WAVENAME name, std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<std::vector<float>>& tempData);
		// һάС���ع��������������
		int waverec1d_matrix(WAVENAME name, const std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<float>& outputData, std::vector<std::vector<float>>& tempData);


	};

}

