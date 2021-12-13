//
//  OpenCVWrapper.mm
//  OpenPose
//
//  Created by ben on 2019/7/24.
//  Copyright © 2019 ben. All rights reserved.
//

#include<opencv2/opencv.hpp>
#include<opencv2/imgcodecs/ios.h>
#include"OpenCVWrapper.h"

#include <cmath>

using namespace cv;
using namespace std;

#define COCO_COLORS \
255.f,    0.f,    0.f, \
255.f,   85.f,    0.f, \
255.f,  170.f,    0.f, \
255.f,  255.f,    0.f, \
170.f,  255.f,    0.f, \
85.f,   255.f,    0.f, \
0.f,    255.f,    0.f, \
0.f,    255.f,   85.f, \
0.f,    255.f,  170.f, \
0.f,    255.f,  255.f, \
0.f,    170.f,  255.f, \
0.f,    85.f,   255.f, \
0.f,     0.f,   255.f, \
85.f,    0.f,   255.f, \
170.f,   0.f,   255.f, \
255.f,   0.f,   255.f, \
255.f,   0.f,   170.f, \
255.f,   0.f,    85.f

#define PI 3.14159265

@implementation OpenCVWrapper

    /**
     * Convert a UIImage to grayscale.
     *
     * @param inputImage: Source image.
     * @param alphaExist: Existence of the alpha channel.
     *
     * @return A grayscale UIImage.
     */
    +(UIImage *) color2Gray: (UIImage *)inputImage alphaExist:(bool) alphaExist {
        Mat image;
        Mat gray;
        
        UIImageToMat(inputImage, image, alphaExist);
        
        if (alphaExist) {
            cvtColor(image, gray, CV_RGBA2GRAY);
        } else {
            cvtColor(image, gray, CV_RGB2GRAY);
        }
        
        UIImage *output = MatToUIImage(gray);
        return output;
    }

    /**
     * Visualize the heatmap.
     *
     * @param heatmap: Heatmap of joints.
     * @param heatmapShape: The shape of the heatmap.
     *
     * @return A image which shows the position of the joints.
     */
    +(UIImage *) visualizeHeatmap: (MLMultiArray *) heatmap
                                   heatmapShape:(const int *) heatmapShape
                                   inputImage: (UIImage *) inputImage {
        int heatmap_w = heatmapShape[0];
        int heatmap_h = heatmapShape[1];
        int keypoint_number = heatmapShape[2] - 4; // remove eyes, ears and background
        
        int sizes[] = {heatmap_w, heatmap_h, keypoint_number};
        Mat imgFloat = Mat(2, sizes, CV_32FC1, Scalar(0));
        Mat imgUchar = Mat(2, sizes, CV_8UC1, Scalar(0));
        
        // gather the confidence of each joint
        unsigned int minValue = 0;
        unsigned int maxValue = 0;
        for(int k=0; k<keypoint_number; k++) {
            for(int i=0; i<heatmap_h; i++) {
                for(int j=0; j<heatmap_w; j++) {
                    int index = k * (heatmap_h * heatmap_w) + i * (heatmap_h) + j;
                    imgFloat.at<float>(i,j) += heatmap[index].floatValue;
                    
                    minValue = imgFloat.at<float>(i,j) < minValue ? imgFloat.at<float>(i,j) : minValue;
                    maxValue = imgFloat.at<float>(i,j) > maxValue ? imgFloat.at<float>(i,j) : maxValue;
                }
            }
        }
        
        unsigned int scale = 255/(maxValue - minValue);
        for(int i=0; i<heatmap_h; i++) {
            for(int j=0; j<heatmap_w; j++) {
                imgFloat.at<float>(i,j) = (imgFloat.at<float>(i,j) - minValue) * scale;
            }
        }
        imgFloat.convertTo(imgUchar, CV_8UC1);
        
        // create mask to filter the confidence below 100
        Mat mask = Mat::zeros(imgUchar.size(), CV_8UC1);
        threshold(imgUchar, mask, 100, 255, THRESH_BINARY);
        resize(mask, mask, cv::Size(inputImage.size.width, inputImage.size.height));
        
        // filter the color map
        Mat colorMap = Mat(2, sizes, CV_8UC3, Scalar(0));
        applyColorMap(imgUchar, colorMap, COLORMAP_JET);
        cvtColor(colorMap, colorMap, COLOR_BGR2RGB);
        resize(colorMap, colorMap, cv::Size(inputImage.size.width, inputImage.size.height));
        Mat filteredColorMap;
        colorMap.copyTo(filteredColorMap, mask);
        
        // blend the heatmap with the confidence map
        Mat inputImageMat, outputImageMat;
        UIImageToMat(inputImage, inputImageMat, false);
        
        cvtColor(inputImageMat, inputImageMat, COLOR_RGBA2RGB);
        addWeighted(inputImageMat, 0.8, filteredColorMap, 0.5, 0.0, outputImageMat);
        
        UIImage *output = MatToUIImage(outputImageMat);
        return output;
    }




-(void) matrixMin: (double *) data
        data_size:(int)data_size
        data_rows:(int)data_rows
        heat_rows:(int)heat_rows {
    
    std::vector<double> vec(&data[0], data + data_size);
    cv::Mat m1(vec),m2;
    m1 = m1.reshape(0, data_rows);
    cv::reduce(m1, m2, 1, CV_REDUCE_MIN);
    for(int i = 0; i < m1.rows; i++)
    {
        m1.row(i) -= m2.row(i);
    }
    
    m1 = m1.reshape(0,data_rows*heat_rows);
    cv::reduce(m1, m2, 1, CV_REDUCE_MIN);
    for(int i = 0; i < m1.rows; i++)
    {
        m1.row(i) -= m2.row(i);
    }
    
    size_t size2 = data_size*sizeof(double);
    std::memcpy(&data[0],m1.data,size2);
    
    vector<double>().swap(vec);
    m1.release();
    m2.release();
}

-(void) matrixMinFloat: (float *) data
        data_size:(int)data_size
        data_rows:(int)data_rows
        heat_rows:(int)heat_rows {
    
    std::vector<float> vec(&data[0], data + data_size);
    cv::Mat m1(vec),m2;
    m1 = m1.reshape(0, data_rows);
    cv::reduce(m1, m2, 1, CV_REDUCE_MIN);
    for(int i = 0; i < m1.rows; i++)
    {
        m1.row(i) -= m2.row(i);
    }
    
    m1 = m1.reshape(0,data_rows*heat_rows);
    cv::reduce(m1, m2, 1, CV_REDUCE_MIN);
    for(int i = 0; i < m1.rows; i++)
    {
        m1.row(i) -= m2.row(i);
    }
    
    size_t size2 = data_size*sizeof(float);
    std::memcpy(&data[0],m1.data,size2);
    
    vector<float>().swap(vec);
    m1.release();
    m2.release();
}

-(void) maximum_filter: (double *) data
             data_size:(int)data_size
             data_rows:(int)data_rows
             mask_size:(int)mask_size
             threshold:(double)threshold {
    
    std::vector<double> vec(&data[0], data + data_size);
    cv::Mat m1(vec);
    m1 = m1.reshape(0, data_rows);
    
    cv::Mat bg;
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(mask_size,mask_size),
                                               cv::Point(-1,-1));
    cv::Mat m2;
    m1.copyTo(m2,m1 > threshold);
    
    dilate(m2, bg, kernel);
    
    cv::Mat bg2;
    m2.copyTo(bg2,m2 == bg);
    
    //    cout << "bg = "<< endl << " "  << bg  << endl << endl;
    //    cout << "bg2 = "<< endl << " "  << bg2  << endl << endl;
    
    size_t size2 = data_size*sizeof(double);
    std::memcpy(&data[0],bg2.data,size2);
    
    vector<double>().swap(vec);
    m1.release();
    m2.release();
    bg.release();
    bg2.release();
    kernel.release();
    
}

-(UIImage*) renderKeyPoint:(CGRect) bounds
                  keypoint:(int*) keypoint
             keypoint_size:(int) keypoint_size
                       pos:(CGPoint*) pos
{
    
    std::vector<int> key(&keypoint[0], keypoint + keypoint_size);
    std::vector<CGPoint> position(&pos[0], pos + keypoint_size*2);
    
    auto colors = std::vector<float>{COCO_COLORS};
    
    const auto exp = 4;
    const auto width = bounds.size.width * exp;
    const auto height = bounds.size.height * exp;
    
    cv::Mat mat(width,height,CV_8UC4);
    mat = cv::Scalar(0);
    
    int stickwidth = int(bounds.size.width * exp / 130);
    std::vector<cv::Point> polygon;
    
    for(int i = 0; i < keypoint_size; i++)
    {
        const auto colorIndex = key[i] * 3;
        const auto numberColors = colors.size();
        
        const cv::Scalar color{colors[colorIndex % numberColors],
            colors[(colorIndex+1) % numberColors],
            colors[(colorIndex+2) % numberColors],255};
        
        CGPoint p1 = position[i*2];
        p1.x *= width + 0.5;
        p1.y *= height + 0.5;
        
        CGPoint p2 = position[i*2+1];
        p2.x *= width + 0.5;
        p2.y *= height + 0.5;
        
        cv::Point point = cv::Point(int(p1.x), int(p1.y));
        cv::circle(mat,point,stickwidth,color,-1);
        
        point = cv::Point(int(p2.x), int(p2.y));
        cv::circle(mat,point,stickwidth,color,-1);
        
        auto length = pow( pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2) ,0.5);
        auto angle = atan2(p1.y - p2.y, p1.x - p2.x) * 180.0 / PI;
        
        cv::Point center = cv::Point(int((p1.x + p2.x) / 2.0),
                                     int((p1.y + p2.y) / 2.0));
        cv::ellipse2Poly(center,
                         cv::Size(int(length / 2), stickwidth),
                         int(angle), 0 , 360 , 1, polygon);
        cv::fillConvexPoly(mat, polygon, color);
        
        //        cout << color << endl;
        //        p1 = position[i*2];
        //        p2 = position[i*2+1];
        //        cv::line(mat,
        //                 cv::Point(int(p1.x * width + 0.5), int(p1.y * height + 0.5)),
        //                 cv::Point(int(p2.x * width + 0.5), int(p2.y * height + 0.5)),
        //                 color,5,CV_8UC4);
    }
    
    UIImage *preview = MatToUIImage(mat);
    vector<cv::Point>().swap(polygon);
    vector<int>().swap(key);
    vector<CGPoint>().swap(position);
    mat.release();
    
    return preview;
}

@end

