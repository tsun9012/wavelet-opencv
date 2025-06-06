# wavelet-opencv
使用opencv，实现图像的小波分解、重构，小波降噪和模极大值边缘检测。

haar小波自动阈值降噪
```cpp
my_wavelet::Wavelet wavelet;
cv::Mat srcGray = cv::imread("D:\\pictures\\crack.jpg", 0);
if (srcGray.empty()) {
	return -1;
}
std::vector<cv::Mat> decMats;
std::vector<cv::Size> decMatsWH;
wavelet.wavedec2d(srcGray, 3, my_wavelet::wdb1, decMats, decMatsWH);
wavelet.waveautoden(decMats);
cv::Mat recMat;
wavelet.waverec2d(my_wavelet::wdb1, decMats, decMatsWH,recMat);
```


