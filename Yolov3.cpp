// torchTutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <sstream>
#include <chrono>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "util.h"
#include <torch/torch.h>
#include "Yolov3_model.h"


using namespace std;



void img_inference(string path, vector<string> cls_names, const char* weightfile) {
	torch::Tensor vals = torch::rand({ 1,3,416,416 });
	YOLO yolo(vals, 80); //COCO dataset
	yolo.load_weights(weightfile);

	std::vector<cv::String> filenames;
	//std::string path = "resources";
	path = path + "/*.jpg";
	cv::glob(path, filenames, false);

	std::cout << "Start to inference ..." << std::endl;
	auto start = std::chrono::steady_clock::now();
	std::vector<cv::Mat> images;
	std::vector<std::vector<Detection>> results;

	// read images and inference
	for (auto& img : filenames) {
		cv::Mat origin_image = cv::imread(img, cv::IMREAD_COLOR);
		images.push_back(origin_image);
		auto dets = yolo.predict(origin_image);
		results.push_back(dets);
	}
	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Average time per image: " << duration / images.size() << "ms" << std::endl;

	for (size_t i = 0; i < results.size(); i++) {

		auto res = results[i];
		auto imgclone = images[i].clone();
		std::string outname(filenames[i]);
		std::cout << outname << std::endl;
		//print class_index if objects found!
		for (auto& d : res) {
			int det_conf = (int)(100 * d.scr) % 100;
			if (det_conf > 40) {
				std::cout << d.cls << " " << d.scr << std::endl;
				// draw box on images		
				draw_bbox(imgclone, d.bbox, cls_names[d.cls], color_map(d.cls));
				//draw_bbox(imgclone, d.bbox, classnames[d.cls], color_map(d.cls));
			}

			
		}
		//	cv::imwrite(outname.replace(outname.find("jpg"), 3, "bbox.jpg"), imgclone);
		cv::imshow("Detection", imgclone);
		cv::waitKey(0);
	}

	std::cout << "Done" << std::endl;
}


void cam_inference(vector<string> cls_names, const char* weightfile) {

	torch::Tensor vals = torch::rand({ 1,3,416,416 });
	YOLO yolo(vals, 80); //COCO dataset
	yolo.load_weights(weightfile);
	const char* classfile = "resources/classnames.txt";

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		throw runtime_error("Cannot open cv::VideoCapture");
	}
	cv::Mat image;
	cv::namedWindow("Real time Object Detection", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
	while (cap.read(image)) {
		//auto frame_processed = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;

		auto dets = yolo.predict(image);

		for (auto& d : dets) {
			//if (std::is_empty(d)) {
			//	std::cout << "no object found in: " << filenames[i] << std::endl;
			//}
			//std::cout << d.cls << " " << d.scr << std::endl;
			// draw box onto images	
			int det_conf = (int)(100 * d.scr) % 100;
			if (det_conf > 40) {
				draw_bbox(image, d.bbox, cls_names[d.cls] + " - " + to_string(det_conf) + "%", color_map(d.cls));
			}
			
		}
		cv::imshow("Output", image);
		switch (cv::waitKey(1) & 0xFF) {
		case 'q':
			return;
			//	case ' ':
			//		cv::imwrite(to_string(frame_processed) + ".jpg", image);
			//		break;
		default:
			break;
		}
	}
}



int main()
{

	torch::Tensor t = torch::tensor({ {316,316,516,516,1,2048,1024} });
	torch::Tensor a = torch::tensor({ {10,13}, {16,30}, {33,23}, {30,61}, {62,45}, {59,119}, {116,90}, {156,198}, {373,326} });


	auto var = YOLOUTIL::create_y_true(t, a, 2);

	std::cout << "____________________________________" << std::endl;
	std::cout << std::get<0>(var) << std::endl;

	std::cout << "____________________________________" << std::endl;
	std::cout << std::get<1>(var) << std::endl;

	std::cout << "____________________________________" << std::endl;
	std::cout << std::get<2>(var) << std::endl;








	// 1. YOLOv3 inference with COCO 80 dataset, with pretrained weights
	//const char* classfile = "resources/classnames.txt";
	//const char* weightfile = "resources/yolov3.weights";
	//std::vector<std::string> classes = YOLOUTIL::class_names(classfile);
	// video
	//cam_inference(classes, weightfile);
	// picture
	//img_inference("resources", classes, weightfile);






	// 2. YOLOv3 training with custom dataset

	return 0;
}

