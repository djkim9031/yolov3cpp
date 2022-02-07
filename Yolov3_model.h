#pragma once
#ifndef YOLOV3_MODEL_H
#define YOLOV3_MODEL_H

#include <memory>
#include <torch/torch.h>
#include <iostream>
#include "yolo_utils.h"


class YOLOLoss {

public:
	YOLOLoss();
	~YOLOLoss();
protected:
	torch::Tensor calc_dim_loss(torch::Tensor true_obj, torch::Tensor true_val, torch::Tensor pred_val);
	//torch::Tensor calc_ignore_mask(torch::Tensor true_obj, torch::Tensor true_box, torch::Tensor pred_box);
	torch::Tensor calc_obj_loss(torch::Tensor true_obj, torch::Tensor pred_obj);
	torch::Tensor calc_cls_loss(torch::Tensor true_class, torch::Tensor pred_class);

private:
	float weight_bbox = 1.5;
	float weight_obj = 5.0;
	

};


class YOLO : YOLOLoss {
public:
	YOLO(torch::Tensor inputs, int num_classes);
	~YOLO();
	void load_weights(const char* weight_file);
	torch::Tensor train(torch::Tensor inputs, torch::Tensor labels, torch::Tensor val_inputs, torch::Tensor val_labels, int batch_size, int epochs);
	std::vector<Detection> predict(cv::Mat image);
	void DarknetShapeTest();
	void YOLOv3ShapeTest(torch::Tensor inputs, bool initialize);
	void LossFuncTest();
	

private:
	torch::Tensor _inputs;
	int _num_classes;
	int trackIdx;
	int64_t _num_trainable_parameters;
	torch::Tensor _anchors = torch::tensor({ {10,13}, {16,30}, {33,23}, {30,61}, {62,45}, {59,119}, {116,90}, {156,198}, {373,326} });
	const float NMS_threshold = 0.4f;
	const float confidence_threshold = 0.5f;
	std::vector<torch::Tensor> trainable_parameters;

	std::vector<std::string> YOLOseq_layers;
	std::vector<torch::nn::Sequential> YOLOseq;

	torch::Tensor DarknetConv(torch::Tensor inputs, int filters, int kernel_size, int strides, bool initialize); //padding=1 (same padding)
	torch::Tensor DarknetResidual(torch::Tensor inputs, int filter1, int filter2, bool initialize);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Darknet(torch::Tensor inputs, bool initialize);
	std::vector<torch::Tensor> YOLOv3(torch::Tensor inputs, int num_classes, bool initialize);
	torch::Tensor get_absolute_yolo_box(torch::Tensor y_pred, torch::Tensor anchors);
	torch::Tensor get_relative_yolo_box(torch::Tensor y_true, torch::Tensor anchors);
	torch::Tensor prediction_postprocess(std::vector<torch::Tensor> y_preds);

	auto loss_calc(torch::Tensor y_true, torch::Tensor y_pred, torch::Tensor anchors);
	torch::Tensor calc_train_loss(std::vector<torch::Tensor> y_true, std::vector<torch::Tensor> y_preds, int batch_size);

	void load_tensor(torch::Tensor t, std::ifstream& fs) {
		fs.read(static_cast<char*>(t.data_ptr()), t.numel() * sizeof(float));
	}

};



#endif // !YOLOV3_MODEL_H
