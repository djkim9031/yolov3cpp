#pragma once
#ifndef YOLO_UTILS_H
#define YOLO_UTILS_H

#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#define EPSILON 1e-7

struct Detection {
	cv::Rect2f bbox;  // struct: (x, y, width, height)
	float scr; // Maximum probability of a class
	int64 cls; // class index
};

namespace PARSER {
	void ltrim(std::string& s);
	void rtrim(std::string& s);
	void trim(std::string& s);
	int split(const std::string& str, std::vector<std::string>& ret_, std::string sep);
}

namespace YOLOUTIL {
	std::vector<std::string> class_names(const char* class_file);

	torch::Tensor label_reader(const char* label_file, std::vector<std::string> classes);

	torch::Tensor binary_cross_entropy(torch::Tensor logits, torch::Tensor labels);

	std::tuple<torch::Tensor, torch::Tensor> data_reader(std::string path, std::vector<std::string> classes);

	std::vector<torch::Tensor> shuffle_inputs_labels(torch::Tensor inputs, torch::Tensor labels, torch::Tensor anchors, int num_classes);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_y_true(torch::Tensor dataset, torch::Tensor anchors, int num_classes);

	torch::Tensor find_best_anchors(torch::Tensor yboxes, torch::Tensor anchors);

	torch::Tensor broadcast_iou(torch::Tensor box_a, torch::Tensor box_b, int64_t grid_size); //DEPRECATED
}

namespace POSTPROCESSING {

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> threshold_confidence(torch::Tensor preds, float conf_thres);

	void center_to_corner(torch::Tensor& bbox);

	std::array<int64_t, 2> letterbox_dim(torch::IntArrayRef img, torch::IntArrayRef box);

	cv::Mat letterbox_img(const cv::Mat& img, torch::IntArrayRef box);

	void inv_letterbox_bbox(torch::Tensor bbox, torch::IntArrayRef box_dim, torch::IntArrayRef img_dim);

	void NMS(std::vector<Detection>& dets, float threshold);

	float IOUCalc(const cv::Rect2f& bb_test, const cv::Rect2f& bb_gt);
}

#endif // !YOLO_UTILS_H
