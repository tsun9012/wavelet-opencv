#include "wavelet.h"
#include <cmath>

#pragma region 内部调用函数

void dwt_sym_stride(std::vector<float>& inp, int N, std::vector<float>& lpd, std::vector<float>& hpd, int lpd_len, std::vector<float>& cA, int len_cA, std::vector<float>& cD) {
    int i, l, t, len_avg;
    int is, os;
    len_avg = lpd_len;

    for (i = 0; i < len_cA; ++i) {
        t = 2 * i + 1; // N=9, len_cA = 5, 原始信号取偶数，下采样, 1, 3, 5, 7, 9
        os = i;
        cA[os] = 0.0;
        cD[os] = 0.0;
        for (l = 0; l < len_avg; ++l) {  // 假设len_avg=2, N=9
            if ((t - l) >= 0 && (t - l) < N) {
                is = (t - l);
                cA[os] += lpd[l] * inp[is];
                cD[os] += hpd[l] * inp[is];
            }
            else if ((t - l) < 0) {
                is = (-t + l - 1);
                cA[os] += lpd[l] * inp[is];
                cD[os] += hpd[l] * inp[is];
            }
            else if ((t - l) >= N) {
                is = (2 * N - t + l - 1);
                cA[os] += lpd[l] * inp[is];
                cD[os] += hpd[l] * inp[is];
            }
        }
    }
}



void clc_waveTransMatrix(int N, std::vector<float>& lpd, std::vector<float>& hpd, int lenKernel, cv::Mat& transMat) {

    transMat = cv::Mat::zeros(N, N, CV_32F);

    if (N % 2 != 0) {
        return;
    }
    if (lenKernel > N) {
        return;
    }

    for (int i = 0; i < N; i++) {

        if (i < N / 2) {
            int hi = i * 2;
            for (int j = 0; j < lpd.size(); j++) {
                transMat.at<float>(i, (hi + j) % N) = lpd[j];
            }
        }
        else {
            int gi = (i - N / 2) * 2;
            for (int j = 0; j < hpd.size(); j++) {
                transMat.at<float>(i, (gi + j) % N) = hpd[j];
            }
        }
    }
}



void dwt_1D(std::vector<float>& signal, int N, std::vector<float>& lpd, std::vector<float>& hpd, int lenKernel, std::vector<float>& cA, std::vector<float>& cD) {

    // 1D小波分解，由低通、高通滤波组成的系数矩阵（正交矩阵、稀疏的），与输入信号相乘，得到尺度函数系数和小波系数
    // 保证输入信号是偶数

    std::vector<float> signal_copy = signal;

    if (lenKernel > N) {
        return;
    }

    if (signal.size() != N || lpd.size() != lenKernel || hpd.size() != lenKernel) {
        return;
    }

    if (N % 2 != 0) {
        N = N + 1;
        signal_copy.push_back(0);
    }

    cv::Mat transMat;
    clc_waveTransMatrix(N, lpd, hpd, lenKernel, transMat);

    cv::Mat tempA = cv::Mat(signal_copy).reshape(1, N); // 将1行N列转成N行1列

    cv::Mat coeff = transMat * tempA;

    coeff = coeff.reshape(1, 1);

    if (!coeff.isContinuous()) {
        coeff = coeff.clone();
    }

    cA = std::vector<float>(N / 2);
    cD = std::vector<float>(N / 2);

    memcpy(cA.data(), coeff.data, sizeof(float) * N / 2);
    float* tempPtr = (float*)coeff.data + N / 2;
    memcpy(cD.data(), tempPtr, sizeof(float) * N / 2);

    /*for (int i = 0; i < N / 2; i++) {

        cA[i] = 0;
        cD[i] = 0;
        for (int j = 0; j < lenKernel; j++) {

            int sp = (i * 2 + j) % N;
            cA[i] += lpd[j] * signal_copy[sp];
            cD[i] += hpd[j] * signal_copy[sp];
        }
    }*/
}


void idwt_1D(std::vector<float>& lpd, std::vector<float>& hpd, int lenKernel, std::vector<float>& cA, std::vector<float>& cD, std::vector<float>& recSignal) {

    if (cA.size() != cD.size()) {
        return;
    }

    int N = cA.size() + cD.size();

    if (N < lenKernel) {
        return;
    }

    cv::Mat transMat;
    clc_waveTransMatrix(N, lpd, hpd, lenKernel, transMat);

    std::vector<float> cAD(N);

    memcpy(cAD.data(), cA.data(), sizeof(float) * N / 2);
    float* tempPtr = (float*)cAD.data() + N / 2;
    memcpy(tempPtr, cD.data(), sizeof(float) * N / 2);

    cv::Mat cADmat = cv::Mat(cAD).reshape(1, N);

    cv::Mat recMat = transMat.t() * cADmat;

    recMat = recMat.reshape(1, 1);

    if (!recMat.isContinuous()) {
        recMat = recMat.clone();
    }

    recSignal = std::vector<float>(N);

    memcpy(recSignal.data(), recMat.data, sizeof(float) * N);
}



void idwt_sym_stride(std::vector<float>& cA, int len_cA, std::vector<float>& cD, std::vector<float>& lpr, std::vector<float>& hpr, int lpr_len, std::vector<float>& X) {
    int len_avg, i, l, m, n, t, v;
    int ms, ns, is;
    len_avg = lpr_len;
    m = -2;
    n = -1;

    for (v = 0; v < len_cA; ++v) {
        i = v;
        m += 2;
        n += 2;
        ms = m;
        ns = n;
        X[ms] = 0.0;
        X[ns] = 0.0;
        for (l = 0; l < len_avg / 2; ++l) {
            t = 2 * l;
            if ((i - l) >= 0 && (i - l) < len_cA) {
                is = (i - l);
                X[ms] += lpr[t] * cA[is] + hpr[t] * cD[is];
                X[ns] += lpr[t + 1] * cA[is] + hpr[t + 1] * cD[is];
            }
        }
    }
}

// 小波函数
void filtcoef(my_wavelet::WAVENAME name, std::vector<float>& lp1, std::vector<float>& hp1, std::vector<float>& lp2,
    std::vector<float>& hp2) {
    if (name == my_wavelet::whaar || name == my_wavelet::wdb1) {
        lp1.push_back(0.7071); lp1.push_back(0.7071);
        hp1.push_back(-0.7071); hp1.push_back(0.7071);
        lp2.push_back(0.7071); lp2.push_back(0.7071);
        hp2.push_back(0.7071); hp2.push_back(-0.7071);
        return;
    }
    if (name == my_wavelet::wdb2) {
        float lp1_a[] = { -0.12940952255092145, 0.22414386804185735, 0.83651630373746899,
                        0.48296291314469025 };
        lp1.assign(lp1_a, lp1_a + sizeof(lp1_a) / sizeof(float));

        float hp1_a[] = { -0.48296291314469025, 0.83651630373746899, -0.22414386804185735,
                        -0.12940952255092145 };
        hp1.assign(hp1_a, hp1_a + sizeof(hp1_a) / sizeof(float));

        float lp2_a[] = { 0.48296291314469025, 0.83651630373746899, 0.22414386804185735,
                        -0.12940952255092145 };
        lp2.assign(lp2_a, lp2_a + sizeof(lp2_a) / sizeof(float));

        float hp2_a[] = { -0.12940952255092145, -0.22414386804185735, 0.83651630373746899,
                        -0.48296291314469025 };
        hp2.assign(hp2_a, hp2_a + sizeof(hp2_a) / sizeof(float));
        return;
    }
    if (name == my_wavelet::wdb3) {
        float lp1_a[] = { 0.035226291882100656, -0.085441273882241486, -0.13501102001039084,
                      0.45987750211933132, 0.80689150931333875, 0.33267055295095688 };
        lp1.assign(lp1_a, lp1_a + sizeof(lp1_a) / sizeof(float));

        float hp1_a[] = { -0.33267055295095688, 0.80689150931333875, -0.45987750211933132,
                        -0.13501102001039084, 0.085441273882241486, 0.035226291882100656 };
        hp1.assign(hp1_a, hp1_a + sizeof(hp1_a) / sizeof(float));

        float lp2_a[] = { 0.33267055295095688, 0.80689150931333875, 0.45987750211933132,
                        -0.13501102001039084, -0.085441273882241486, 0.035226291882100656 };
        lp2.assign(lp2_a, lp2_a + sizeof(lp2_a) / sizeof(float));

        float hp2_a[] = { 0.035226291882100656, 0.085441273882241486, -0.13501102001039084,
                        -0.45987750211933132, 0.80689150931333875, -0.33267055295095688 };
        hp2.assign(hp2_a, hp2_a + sizeof(hp2_a) / sizeof(float));
        return;
    }

}

// 一层2D小波分解
void dwt2(my_wavelet::WAVENAME name, cv::Mat& signal, cv::Mat& cLL, cv::Mat& cLH, cv::Mat& cHL, cv::Mat& cHH, cv::Size cAwh) {

    cv::Mat signal_copy = signal.clone();
    int raw_rows = signal.rows;
    int raw_cols = signal.cols;

    int cLL_rows = cAwh.height;
    int cLL_cols = cAwh.width;

    cLL = cv::Mat::zeros(cLL_rows, cLL_cols, CV_32F);
    cLH = cv::Mat::zeros(cLL_rows, cLL_cols, CV_32F);
    cHL = cv::Mat::zeros(cLL_rows, cLL_cols, CV_32F);
    cHH = cv::Mat::zeros(cLL_rows, cLL_cols, CV_32F);

    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(name, lp1, hp1, lp2, hp2);

    int kernelLen = lp1.size();

    // 横向扩充边界，下采样取偶数列(原始信号行列数如果为奇数，则填充)
    int pad = raw_cols % 2;
    cv::copyMakeBorder(signal_copy, signal_copy, 0, 0, (kernelLen - 2) / 2, (kernelLen - 2) / 2 + pad, cv::BORDER_REFLECT);

    int kernel_rows = 1;
    int kernel_cols = lp1.size();
    cv::Mat lp1Kernel = cv::Mat(kernel_rows, kernel_cols, CV_32F, lp1.data());
    cv::Mat hp1Kernel = cv::Mat(kernel_rows, kernel_cols, CV_32F, hp1.data());

    cv::Mat lp_dn1 = cv::Mat::zeros(raw_rows, signal_copy.cols/2, CV_32F);
    cv::Mat hp_dn1 = cv::Mat::zeros(raw_rows, signal_copy.cols/2, CV_32F);
    cv::Mat tempMat = signal_copy.clone();
    // 横向低通滤波
    tempMat = 0;

    cv::filter2D(signal_copy, tempMat, -1, lp1Kernel, cv::Point(-1, -1));

    if (!tempMat.isContinuous())
    {
        cv::Mat continuousMat;
        tempMat.copyTo(continuousMat);
        tempMat = continuousMat;
    }
    // 只取偶数列（只计算偶数列的卷积，这里计算有冗余，还可以优化）
    for (int i = 0; i < cLL_cols; i++) {
        tempMat.colRange(2 * i + 1, 2 * i + 2).copyTo(lp_dn1.colRange(i, i + 1));
    }

    // 横向高通滤波
    cv::filter2D(signal_copy, tempMat, -1, hp1Kernel, cv::Point(-1, -1));
    if (!tempMat.isContinuous())
    {
        cv::Mat continuousMat;
        tempMat.copyTo(continuousMat);
        tempMat = continuousMat;
    }
    for (int i = 0; i < cLL_cols; i++) {
        tempMat.colRange(2 * i + 1, 2 * i + 2).copyTo(hp_dn1.colRange(i, i + 1));
    }

    // 纵向填充边界
    pad = raw_rows % 2;
    cv::copyMakeBorder(lp_dn1, lp_dn1, (kernelLen - 2) / 2, (kernelLen - 2) / 2 + pad, 0, 0, cv::BORDER_REFLECT);
    cv::copyMakeBorder(hp_dn1, hp_dn1, (kernelLen - 2) / 2, (kernelLen - 2) / 2 + pad, 0, 0, cv::BORDER_REFLECT);

    //纵向低通滤波
    cv::filter2D(lp_dn1, tempMat, -1, lp1Kernel.t(), cv::Point(-1, -1));
    for (int i = 0; i < cLL_rows; i++) {
        tempMat.row(i * 2 + 1).copyTo(cLL.row(i));
    }

    //纵向高通滤波
    cv::filter2D(lp_dn1, tempMat, -1, hp1Kernel.t(), cv::Point(-1, -1));
    for (int i = 0; i < cLL_rows; i++) {
        tempMat.row(i * 2 + 1).copyTo(cLH.row(i));
    }

    // 纵向低通滤波
    cv::filter2D(hp_dn1, tempMat, -1, lp1Kernel.t(), cv::Point(-1, -1));
    for (int i = 0; i < cLL_rows; i++) {
        tempMat.row(i * 2 + 1).copyTo(cHL.row(i));
    }
    // 纵向高通滤波
    cv::filter2D(hp_dn1, tempMat, -1, hp1Kernel.t(), cv::Point(-1, -1));
    for (int i = 0; i < cLL_rows; i++) {
        tempMat.row(i * 2 + 1).copyTo(cHH.row(i));
    }
}

// 一层2D小波重构
void idwt2(my_wavelet::WAVENAME name, cv::Mat& rec_signal, cv::Size rec_signalWH, cv::Mat& cLL,
    cv::Mat& cLH, cv::Mat& cHL, cv::Mat& cHH) {

    int rows = cLL.rows * 2;
    int cols = cLL.cols * 2;

    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(name, lp1, hp1, lp2, hp2);
    int kernel_rows = 1;
    int kernel_cols = lp2.size();
    cv::Mat lp1Kernel = cv::Mat(kernel_rows, kernel_cols, CV_32F, lp2.data());
    cv::Mat hp1Kernel = cv::Mat(kernel_rows, kernel_cols, CV_32F, hp2.data());
    int kernelLen = lp1.size();

    cv::Mat temp_rec1 = cv::Mat::zeros(rows, cols / 2, CV_32F);
    cv::Mat temp_rec2 = cv::Mat::zeros(rows, cols / 2, CV_32F);
    // 纵向低通滤波，提取到偶数行
    for (int i = 0; i < cLL.rows; i++) {
        cLL.row(i).copyTo(temp_rec1.row(2 * i + 1));
    }

    cv::Mat tempK = lp1Kernel.t();
    cv::filter2D(temp_rec1, temp_rec1, -1, lp1Kernel.t(), cv::Point(0, 0));

    // 纵向高通滤波
    for (int i = 0; i < cLH.rows; i++) {
        cLH.row(i).copyTo(temp_rec2.row(2 * i + 1));
    }

    cv::filter2D(temp_rec2, temp_rec2, -1, hp1Kernel.t(), cv::Point(0, 0));
    cv::Mat lp_half_rec = temp_rec1 + temp_rec2;
    lp_half_rec = lp_half_rec(cv::Rect(0, 0, lp_half_rec.cols, rec_signalWH.height));

    //lp_half_rec = lp_half_rec(cv::Rect(0, 0, cols / 2, rec_signalWH.height));

    temp_rec1 = 0;
    temp_rec2 = 0;
    // 纵向低通滤波
    for (int i = 0; i < cHL.rows; i++) {
        cHL.row(i).copyTo(temp_rec1.row(2 * i + 1));
    }
    cv::filter2D(temp_rec1, temp_rec1, -1, lp1Kernel.t(), cv::Point(0, 0));

    // 纵向高通滤波
    for (int i = 0; i < cHH.rows; i++) {
        cHH.row(i).copyTo(temp_rec2.row(2 * i + 1));
    }
    cv::filter2D(temp_rec2, temp_rec2, -1, hp1Kernel.t(), cv::Point(0, 0));

    cv::Mat hp_half_rec = temp_rec1 + temp_rec2;
    hp_half_rec = hp_half_rec(cv::Rect(0, 0, hp_half_rec.cols, rec_signalWH.height));
    //hp_half_rec = hp_half_rec(cv::Rect(0, 0, cols / 2, rec_signalWH.height));

    if (!lp_half_rec.isContinuous())
    {
        cv::Mat continuousMat;
        lp_half_rec.copyTo(continuousMat);
        lp_half_rec = continuousMat;
    }
    if (!hp_half_rec.isContinuous())
    {
        cv::Mat continuousMat;
        hp_half_rec.copyTo(continuousMat);
        hp_half_rec = continuousMat;
    }

    cv::Mat lp_rec = cv::Mat::zeros(rec_signalWH.height, cols, CV_32F);
    cv::Mat hp_rec = cv::Mat::zeros(rec_signalWH.height, cols, CV_32F);

    for (int i = 0; i < lp_half_rec.cols; i++) {
        lp_half_rec.colRange(i, i + 1).copyTo(lp_rec.colRange(2 * i + 1, 2 * i + 2));
    }
    for (int i = 0; i < hp_half_rec.cols; i++) {
        hp_half_rec.colRange(i, i + 1).copyTo(hp_rec.colRange(2 * i + 1, 2 * i + 2));
    }

    // 横向低通滤波
    cv::filter2D(lp_rec, lp_rec, -1, lp1Kernel, cv::Point(0, 0));
    // 横向高通滤波
    cv::filter2D(hp_rec, hp_rec, -1, hp1Kernel, cv::Point(0, 0));
    rec_signal = lp_rec + hp_rec;
    rec_signal = rec_signal(cv::Rect(0, 0, rec_signalWH.width, rec_signal.rows));

}


cv::Mat generate_conv_kernel(int size, float sigma, int scale_, int type) {

    // 高斯平滑函数，exp(-(x^2 + y^2)/(2*sigma^2))
    // 对x的偏导数，-x/sigma^2 * exp(-(x^2 + y^2)/(2*sigma^2))
    // 对y的偏导数，-y/sigma^2 * exp(-(x^2 + y^2)/(2*sigma^2))
    
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    float scale = pow(2, scale_);

    if (type == 0) {
        // 对x求偏导的滤波核
        for (int i = 0; i < kernel.rows; i++) {
            for (int j = 0; j < kernel.cols; j++) {
                float x = float(j - kernel.cols / 2) / scale;
                float y = float(i - kernel.rows / 2) / scale;
                kernel.at<float>(i, j) = -x * (sigma * sigma) * exp(-(x * x + y * y) / (2 * sigma * sigma)) / (scale * scale);
            }
        }
    }
    else {
        // 对y求偏导的滤波核
        for (int i = 0; i < kernel.rows; i++) {
            for (int j = 0; j < kernel.cols; j++) {
                float x = float(j - kernel.cols / 2) / scale;
                float y = float(i - kernel.rows / 2) / scale;
                kernel.at<float>(i, j) = -y * (sigma * sigma) * exp(-(x * x + y * y) / (2 * sigma * sigma)) / (scale * scale);
            }
        }
    }

    return kernel;
}

cv::Mat calculate_modulus_maxima(cv::Mat& M_f, cv::Mat& A_f) {

    cv::Mat M_maxima = M_f.clone();
    cv::Mat neighbour = cv::Mat::zeros(3, 3, CV_32F);

    for (int i = 1; i < M_f.rows-1; i++) {
        auto pm_pre = M_f.ptr<float>(i - 1);
        auto pm_last = M_f.ptr<float>(i + 1);

        auto pm = M_f.ptr<float>(i);
        auto pa = A_f.ptr<float>(i);
        auto p_maxima = M_maxima.ptr<float>(i);

        for (int j = 1; j < M_f.cols - 1; j++) {

            neighbour.at<float>(0, 0) = pm_pre[j - 1];
            neighbour.at<float>(0, 1) = pm_pre[j];
            neighbour.at<float>(0, 2) = pm_pre[j + 1];
            neighbour.at<float>(1, 0) = pm[j - 1];
            neighbour.at<float>(1, 1) = pm[j];
            neighbour.at<float>(1, 2) = pm[j + 1];
            neighbour.at<float>(2, 0) = pm_last[j - 1];
            neighbour.at<float>(2, 1) = pm_last[j];
            neighbour.at<float>(2, 2) = pm_last[j + 1];

            float angle = pa[j];
            float modulus = pm[j];

            float left_neighbour = 0.0;
            float right_neighbour = 0.0;

            float PI = 180.0;

            if ((angle >= 0 && angle <= PI / 8) ||
                (angle >= 7 * PI / 8 && angle <= 9 * PI / 8) ||
                (angle >= 15 * PI / 8 && angle <= 2 * PI)) {
                
                left_neighbour = neighbour.at<float>(1, 0);
                right_neighbour = neighbour.at<float>(1, 2);
            }
            else if ((angle >= 3 * PI / 8 && angle <= 5 * PI / 8) ||
                (angle >= 11 * PI / 8 && angle <= 13 * PI / 8)) {

                left_neighbour = neighbour.at<float>(0, 1);
                right_neighbour = neighbour.at<float>(2, 1);
            }
            else if ((angle > PI / 8 && angle < 3 * PI / 8) ||
                (angle > 9 * PI / 8 && angle < 11 * PI / 8)) {

                left_neighbour = neighbour.at<float>(0, 0);
                right_neighbour = neighbour.at<float>(2, 2);
            }
            else if ((angle > 5 * PI / 8 && angle < 7 * PI / 8) ||
                (angle > 13 * PI / 8 && angle < 15 * PI / 8)) {

                left_neighbour = neighbour.at<float>(0, 2);
                right_neighbour = neighbour.at<float>(2, 0);
            }

            if (modulus > left_neighbour && modulus > right_neighbour) {
                p_maxima[j] = modulus;
            }

        }
    }

    return M_maxima;
}

#pragma endregion


int my_wavelet::Wavelet::wavedec2d(const cv::Mat& grayImg, int level, WAVENAME waveName, std::vector<cv::Mat>& decMats, std::vector<cv::Size>& decMatsWH)
{

    if (grayImg.channels() != 3) {
        -1;
    }

    cv::Mat srcMatcopy = grayImg.clone();
    srcMatcopy.convertTo(srcMatcopy, CV_32F, 1 / 255.0);

    int rows_n = srcMatcopy.rows; // No. of rows
    int cols_n = srcMatcopy.cols; //No. of columns

    int Max_Iter;
    Max_Iter = std::min((int)ceil(log(double(rows_n)) / log(2.0)), (int)ceil(log(double(cols_n)) / log(2.0)));
    if (Max_Iter < level) {
        return -2; // 层数超过最大可分解层数
    }

    decMatsWH.insert(decMatsWH.begin(), cv::Size(cols_n, rows_n));
    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(waveName, lp1, hp1, lp2, hp2);
    int kernelL = lp1.size();

    for (int i = 0; i < level; i++) {
        cols_n += kernelL - 2;
        rows_n += kernelL - 2;
        cols_n = (int)ceil((double)cols_n / 2.0);
        rows_n = (int)ceil((double)rows_n / 2.0);
        decMatsWH.insert(decMatsWH.begin(), cv::Size(cols_n, rows_n));
    }
    int index = decMatsWH.size() - 2;

    for (int iter = 0; iter < level; iter++) {
        cv::Size cAwh = decMatsWH[index];
        index--;

        cv::Mat cA;
        cv::Mat cV;
        cv::Mat cH;
        cv::Mat cD;

        dwt2(waveName, srcMatcopy, cA, cV, cH, cD, cAwh);
        decMats.insert(decMats.begin(), cD.clone());
        decMats.insert(decMats.begin(), cH.clone());
        decMats.insert(decMats.begin(), cV.clone());

        srcMatcopy = cA.clone();

        if (iter == level - 1) {
            decMats.insert(decMats.begin(), cA.clone());
        }
    }

    return 0;
}

int my_wavelet::Wavelet::waverec2d(WAVENAME waveName, const std::vector<cv::Mat>& decMats, std::vector<cv::Size>& decMatsWH, cv::Mat& recMat)
{
    cv::Mat cA = decMats[0];
    int index = 1;

    for (int i = 1; i < decMats.size(); i += 3) {
        cv::Mat cV = decMats[i];
        cv::Mat cH = decMats[i + 1];
        cv::Mat cD = decMats[i + 2];

        cv::Size cAwh = decMatsWH[index];
        index++;

        cv::Mat tempM;
        idwt2(waveName, tempM, cAwh, cA, cV, cH, cD);
        cA = tempM.clone();
    }

    recMat = cA.clone();
    cv::normalize(recMat, recMat, 0, 1, cv::NORM_MINMAX);
    recMat.convertTo(recMat, CV_8U, 255.0);
    return 0;
}

int my_wavelet::Wavelet::waveautoden(std::vector<cv::Mat>& decMats)
{
    if (decMats.size() < 4) {
        return -1;
    }

    int level = (decMats.size() - 1) / 3;

    for (int i = 0; i < level; i++) {

        int hhIndex = decMats.size() - 1 - i * 3;
        cv::Mat& HH = decMats[hhIndex];

        cv::Scalar mean, stddev;
        cv::meanStdDev(HH, mean, stddev);

        float N = HH.rows * HH.cols;
        float Thresh = stddev[0] * sqrt(2 * log10(N));

        cv::Mat mask = HH < Thresh;

        // 软阈值
        HH.setTo(0, mask);

        cv::Mat& HL = decMats[hhIndex - 1];
        mask = HL < Thresh;
        HL.setTo(0, mask);

        cv::Mat& LH = decMats[hhIndex - 2];
        mask = LH < Thresh;
        LH.setTo(0, mask);
    }
    return 0;
}

int my_wavelet::Wavelet::waveedge(const cv::Mat& grayMat, int scale, float thresh, cv::Mat& edgeMat, int size, float sigma)
{
    if (grayMat.channels() != 1) {
        return -1;
    }

    if (size % 2 == 0) {
        return -2;
    }

    cv::Mat grayMatCopy = grayMat.clone();
    grayMatCopy.convertTo(grayMatCopy, CV_32F, 1.0 / 255);

    cv::Mat wx_kernel = generate_conv_kernel(size, sigma, scale, 0);
    cv::Mat wy_kernel = generate_conv_kernel(size, sigma, scale, 1);
    
    cv::Mat wx_f;
    cv::filter2D(grayMatCopy, wx_f, -1, wx_kernel);
    cv::Mat wy_f;
    cv::filter2D(grayMatCopy, wy_f, -1, wy_kernel);

    // 求模
    cv::Mat wx_2f = wx_f.mul(wx_f);
    cv::Mat wy_2f = wy_f.mul(wy_f);
    cv::Mat M_f = wx_2f + wy_2f;
    cv::sqrt(M_f, M_f);

    // 求角度
    wx_f = wx_f + 1e-6;
    cv::Mat A_f = wx_f.clone();
    A_f = 0;
    for (int i = 0; i < A_f.rows; i++) {
        auto pt = A_f.ptr<float>(i);
        auto px = wx_f.ptr<float>(i);
        auto py = wy_f.ptr<float>(i);

        for (int j = 0; j < A_f.cols; j++) {
            pt[j] = cv::fastAtan2(py[j], px[j]);
        }
    }
    
    // 在3*3范围内计算模极大值
    cv::Mat local_maxima = calculate_modulus_maxima(M_f, A_f);

    // 二值化
    cv::Mat mask = local_maxima < thresh;
    cv::Mat inverseMask = mask.clone();
    inverseMask = 255;
    inverseMask = inverseMask - mask;

    edgeMat = inverseMask.clone();
    return 0;
}

int my_wavelet::Wavelet::wavedec1d(const std::vector<float>& inputData, int level, WAVENAME name, std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<std::vector<float>>& tempData)
{
    std::vector<float> dataCopy = inputData;

    int Max_Iter;
    Max_Iter = ceil(log(double(inputData.size())) / log(2.0));

    if (Max_Iter < level) {
        return -1; // 层数超过最大可分解层数
    }

    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(name, lp1, hp1, lp2, hp2);
    int kernelL = lp1.size();

    decSize.insert(decSize.begin(), dataCopy.size());

    tempData.insert(tempData.begin(), dataCopy);

    for (int i = 0; i < level; i++) {

        int len_cA = ceil(dataCopy.size() / 2.0);

        std::vector<float> cA(len_cA);
        std::vector<float> cD(len_cA);

        dwt_sym_stride(dataCopy, dataCopy.size(), lp1, hp1, kernelL, cA, len_cA, cD);

        decData.insert(decData.begin(), cD);
        decSize.insert(decSize.begin(), len_cA);

        tempData.insert(tempData.begin(), cD);
        tempData.insert(tempData.begin(), cA);

        if (i == level - 1) {
            decSize.insert(decSize.begin(), len_cA);
            decData.insert(decData.begin(), cA);
        }
        dataCopy = cA;
    }

    return 0;
}

int my_wavelet::Wavelet::waverec1d(WAVENAME name, const std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<float>& outputData, std::vector<std::vector<float>>& tempData)
{

    int level = decData.size() - 1;

    if (level < 1) {
        return -1;
    }

    std::vector<float> cA = decData[0];

    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(name, lp1, hp1, lp2, hp2);
    int kernelL = lp2.size();

    for (int i = 1; i <= level; i++) {

        std::vector<float> cD = decData[i];

        if (cD.size() < cA.size()) {
            int len = cA.size() - cD.size();
            for (int j = 0; j < len; j++) {
                cD.push_back(0);
            }
        }

        int len_cA = cA.size();

        std::vector<float> recData(len_cA*2);

        idwt_sym_stride(cA, len_cA, cD, lp2, hp2, kernelL, recData);

        cA = recData;

    }

    outputData = cA;
    return 0;
}

int my_wavelet::Wavelet::wavedec1d_matrix(const std::vector<float>& inputData, int level, WAVENAME name, std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<std::vector<float>>& tempData)
{
    std::vector<float> dataCopy = inputData;

    int Max_Iter = 0;

    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(name, lp1, hp1, lp2, hp2);
    int kernelL = lp1.size();

    int tempLen = inputData.size();

    while (tempLen > kernelL) {
        Max_Iter += 1;
        tempLen = ceil(tempLen * 1.0 / 2);
    }

    if (Max_Iter < level) {
        return -1; // 层数超过最大可分解层数
    }

    decSize.insert(decSize.begin(), dataCopy.size());

    tempData.insert(tempData.begin(), dataCopy);

    for (int i = 0; i < level; i++) {

        int len_cA = ceil(dataCopy.size() / 2.0);

        std::vector<float> cA, cD;
        dwt_1D(dataCopy, dataCopy.size(), lp1, hp1, kernelL, cA, cD);

        decData.insert(decData.begin(), cD);
        decSize.insert(decSize.begin(), len_cA);

        tempData.insert(tempData.begin(), cD);
        tempData.insert(tempData.begin(), cA);

        if (i == level - 1) {
            decSize.insert(decSize.begin(), len_cA);
            decData.insert(decData.begin(), cA);
        }
        dataCopy = cA;
    }

    return 0;
}

int my_wavelet::Wavelet::waverec1d_matrix(WAVENAME name, const std::vector<std::vector<float>>& decData, std::vector<int>& decSize, std::vector<float>& outputData, std::vector<std::vector<float>>& tempData)
{
    int level = decData.size() - 1;

    if (level < 1) {
        return -1;
    }

    std::vector<float> cA = decData[0];

    std::vector<float> lp1, hp1, lp2, hp2;
    filtcoef(name, lp1, hp1, lp2, hp2);
    int kernelL = lp2.size();

    for (int i = 1; i <= level; i++) {

        std::vector<float> cD = decData[i];

        if (cA.size() > cD.size()) {
            cA.pop_back();
        }

        std::vector<float> recData;
        idwt_1D(lp1, hp1, kernelL, cA, cD, recData);

        cA = recData;
    }

    outputData = cA;
    return 0;

}
