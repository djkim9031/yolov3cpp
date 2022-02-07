#include "yolo_utils.h"


// trim from start (in place)
void PARSER::ltrim(std::string& s) {
	s.erase(s.begin(), find_if(s.begin(), s.end(), [](char ch) {
		return !isspace(ch);
		}));
}

// trim from end (in place)
void PARSER::rtrim(std::string& s) {
	s.erase(find_if(s.rbegin(), s.rend(), [](char ch) {
		return !isspace(ch);
		}).base(), s.end());
}

// trim from both ends (in place)
void PARSER::trim(std::string& s) {
	ltrim(s);
	rtrim(s);
}

int PARSER::split(const std::string& str, std::vector<std::string>& ret_, std::string sep) {
	if (str.empty()) {
		return 0;
	}

	std::string tmp;
	std::string::size_type pos_begin = str.find_first_not_of(sep);
	std::string::size_type comma_pos = 0;

	while (pos_begin != std::string::npos) {
		comma_pos = str.find(sep, pos_begin);
		if (comma_pos != std::string::npos) {
			tmp = str.substr(pos_begin, comma_pos - pos_begin);
			pos_begin = comma_pos + sep.length();
		}
		else {
			tmp = str.substr(pos_begin);
			pos_begin = comma_pos;
		}

		if (!tmp.empty()) {
			trim(tmp);
			ret_.push_back(tmp);
			tmp.clear();
		}
	}
	return 0;
}

std::vector<std::string> YOLOUTIL::class_names(const char* class_file) {
	std::ifstream fs(class_file);
	std::string line;
	std::vector<std::string> cls;

	if (!fs) {
		throw "Fail to load class file";
	}

	while (getline(fs, line)) {

		if (line.empty()) {
			continue;
		}
		cls.push_back(line);

	}
	fs.close();
	return cls;
}

torch::Tensor YOLOUTIL::label_reader(const char* label_file, std::vector<std::string> classes) {
	std::ifstream fs(label_file);
	std::string line;
	std::vector<torch::Tensor> obj_instance;

	if (!fs) {
		throw "Fail to load class file";
	}

	while (getline(fs, line)) {

		if (line.empty()) {
			continue;
		}
		std::vector<std::string> ret;
		int val = PARSER::split(line, ret, " ");
		int cls_idx = 0;
		for (int i = 0; i < classes.size(); i++) {
			if (classes[i] == ret[4]) {
				cls_idx = i;
				break;
			}
		}
		torch::Tensor v = torch::tensor({ { stoi(ret[0]), stoi(ret[1]), stoi(ret[2]), stoi(ret[3]), cls_idx, stoi(ret[5]), stoi(ret[6]) } }); //x1, y1, x2, y2, cls, H, W
		obj_instance.push_back(v);

	}
	fs.close();
	torch::Tensor output = torch::concat(obj_instance, 0);
	return output;
}

torch::Tensor YOLOUTIL::binary_cross_entropy(torch::Tensor logits, torch::Tensor labels) {
	logits = torch::clamp(logits, EPSILON, 1 - EPSILON);
	return -(1 - labels) * ((1 - logits).log()) - (labels) * (logits.log());
}


std::tuple<torch::Tensor, torch::Tensor> YOLOUTIL::data_reader(std::string path, std::vector<std::string> classes) {
	std::vector<cv::String> filenames;
	cv::String path_img = path + "/images" + "/*.jpg";
	cv::glob(path_img, filenames, false);


	std::vector<cv::String> labelnames;
	cv::String path_labels = path + "/labels" + "/*.txt";
	cv::glob(path_labels, labelnames, false);

	//TODO: Make sure that .jpg and .txt files are ordered exactly the same when names are not numerically ordered

	std::vector<torch::Tensor> images;
	for (auto& img : filenames) {
		std::cout << img << std::endl;
		cv::Mat image = cv::imread(img, cv::IMREAD_COLOR);
		image = POSTPROCESSING::letterbox_img(image, { 416, 416 });
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		image.convertTo(image, CV_32F);
		auto imageTensors = torch::from_blob(image.data, { 1, 416, 416, 3 }).permute({ 0,3,1,2 }).div_(255.0);
		images.push_back(imageTensors);
	}

	std::vector<torch::Tensor> labels;
	for (auto& labl : labelnames) {
		std::cout << labl << std::endl;
		torch::Tensor output = label_reader(labl.c_str(), classes);
		output = output.unsqueeze_(0);
		labels.push_back(output);
	}

	torch::Tensor input_tensors = torch::concat(images, 0);
	torch::Tensor input_labels = torch::concat(labels, 0);

	return std::make_tuple(input_tensors, input_labels);
}

std::vector<torch::Tensor> YOLOUTIL::shuffle_inputs_labels(torch::Tensor inputs, torch::Tensor labels, torch::Tensor anchors, int num_classes) {
	//inputs shape = [num_images, 3, 416, 416], labels shape = [num_labels (=num_images), num_objs, 7]
	torch::Tensor perm_idx = torch::randperm(inputs.size(0));
	inputs = inputs.index_select(0, perm_idx);
	labels = labels.index_select(0, perm_idx);

	std::vector<torch::Tensor> y0, y1, y2;
	for (int64_t i = 0; i < inputs.size(0); i++) {
		auto ytrues = create_y_true(labels[i], anchors, num_classes);
		y0.push_back(std::get<0>(ytrues)); //y_large
		y1.push_back(std::get<1>(ytrues)); //y_medium
		y2.push_back(std::get<2>(ytrues)); //y_small
	}
	torch::Tensor out_0, out_1, out_2;
	out_0 = torch::concat(y0, 0); //large
	out_1 = torch::concat(y1, 0); //medium
	out_2 = torch::concat(y2, 0); //small

	std::vector<torch::Tensor> dataset_list;
	dataset_list.push_back(inputs);
	dataset_list.push_back(out_0); //large set
	dataset_list.push_back(out_1); //medium set
	dataset_list.push_back(out_2); //small set

	return dataset_list;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> YOLOUTIL::create_y_true(torch::Tensor dataset, torch::Tensor anchors, int num_classes) {
	//each y_true shape = [B, 3, X, grid, grid], X = 4 + 1 + class_num
	//y_true X = (bx,by,bw,bh,obj,cls)
	//dataset(labels) for each 3xHxW image, (num_objs, 7 )  7 = (x1, y1, x2, y2, cls, H, W)


	//1. x1,y1,x2,y2 to x_c,y_c,w,h, then scale it to match 416x416 dim
	int W, H;
	W = dataset[0][6].item().toInt();
	H = dataset[0][5].item().toInt();

	std::array<int64_t, 2> ret = POSTPROCESSING::letterbox_dim({ H, W }, {416, 416});
	auto new_h = ret[0], new_w = ret[1];

	torch::Tensor w = (dataset.select(1, 2) - dataset.select(1, 0)).toType(torch::ScalarType::Float).mul_(1.0f * new_w / W).unsqueeze_(-1);
	torch::Tensor h = (dataset.select(1, 3) - dataset.select(1, 1)).toType(torch::ScalarType::Float).mul_(1.0f * new_h / H).unsqueeze_(-1);
	torch::Tensor x_c = (dataset.select(1, 0)).toType(torch::ScalarType::Float).mul_(1.0f * new_w / W).add_((416 - new_w) / 2).unsqueeze_(-1) + w / 2;
	torch::Tensor y_c = (dataset.select(1, 1)).toType(torch::ScalarType::Float).mul_(1.0f * new_h / H).add_((416 - new_h) / 2).unsqueeze_(-1) + h / 2;
	torch::Tensor cls = dataset.slice(1, 4, 5, 1); //[num_objs, 1]
	
	//yboxes [num_objs, 4]
	torch::Tensor yboxes = torch::concat({ x_c,y_c,w,h }, 1);
	torch::Tensor yboxes_scaled = yboxes / 416; //(0~1 range)

	//find the best anchor
	torch::Tensor anchor_indices = find_best_anchors(yboxes, anchors);

	torch::Tensor y_true_small, y_true_medium, y_true_large;
	y_true_small = torch::zeros({ 1, 3, 5 + num_classes, 13, 13 });
	y_true_medium = torch::zeros({ 1, 3, 5 + num_classes, 26, 26 });
	y_true_large = torch::zeros({ 1, 3, 5 + num_classes, 52, 52 });

	for (int64_t i = 0; i < anchor_indices.size(0); i++) {
		torch::Tensor currBox = yboxes_scaled[i];
		int curr_anchor_index = anchor_indices[i].item().toInt();
		int curr_cls_index = cls[i].item().toInt();
		int grid_x, grid_y;
		double bx, by, bw, bh, rx, ry, rw, rh;
		rx = x_c[i].item().toDouble();
		ry = y_c[i].item().toDouble();
		rw = w[i].item().toDouble();
		rh = h[i].item().toDouble();

		if (curr_anchor_index < 3) {
			//case 1. grid = 52
			bx = 52 * currBox[0].item().toDouble();
			by = 52 * currBox[1].item().toDouble();

			grid_x = round(bx);
			grid_y = round(by);
			
			y_true_large[0][curr_anchor_index][0][grid_y][grid_x] = rx;
			y_true_large[0][curr_anchor_index][1][grid_y][grid_x] = ry;
			y_true_large[0][curr_anchor_index][2][grid_y][grid_x] = rw;
			y_true_large[0][curr_anchor_index][3][grid_y][grid_x] = rh;
			y_true_large[0][curr_anchor_index][4][grid_y][grid_x] = 1;
			y_true_large[0][curr_anchor_index][5+curr_cls_index][grid_y][grid_x] = 1;
		}
		else if (curr_anchor_index < 6 && curr_anchor_index >= 3) {
			//case 2. grid = 26
			bx = 26 * currBox[0].item().toDouble();
			by = 26 * currBox[1].item().toDouble();

			grid_x = round(bx);
			grid_y = round(by);

			y_true_medium[0][curr_anchor_index-3][0][grid_y][grid_x] = rx;
			y_true_medium[0][curr_anchor_index-3][1][grid_y][grid_x] = ry;
			y_true_medium[0][curr_anchor_index-3][2][grid_y][grid_x] = rw;
			y_true_medium[0][curr_anchor_index-3][3][grid_y][grid_x] = rh;
			y_true_medium[0][curr_anchor_index-3][4][grid_y][grid_x] = 1;
			y_true_medium[0][curr_anchor_index-3][5 + curr_cls_index][grid_y][grid_x] = 1;
		}
		else {
			//case 3. grid = 13
			bx = 13 * currBox[0].item().toDouble();
			by = 13 * currBox[1].item().toDouble();

			grid_x = round(bx);
			grid_y = round(by);

			y_true_small[0][curr_anchor_index - 6][0][grid_y][grid_x] = rx;
			y_true_small[0][curr_anchor_index - 6][1][grid_y][grid_x] = ry;
			y_true_small[0][curr_anchor_index - 6][2][grid_y][grid_x] = rw;
			y_true_small[0][curr_anchor_index - 6][3][grid_y][grid_x] = rh;
			y_true_small[0][curr_anchor_index - 6][4][grid_y][grid_x] = 1;
			y_true_small[0][curr_anchor_index - 6][5 + curr_cls_index][grid_y][grid_x] = 1;

		}

	}
	
	return std::make_tuple(y_true_large, y_true_medium, y_true_small);

}

torch::Tensor YOLOUTIL::find_best_anchors(torch::Tensor yboxes, torch::Tensor anchors) {

	//yboxes shape = (num_objs, 4) -> broadcast to (num_objs, 9, 4) , 9 = number of anchors
	yboxes = yboxes.unsqueeze_(1);
	yboxes = yboxes.broadcast_to({ yboxes.size(0), 9, yboxes.size(2) });
	torch::Tensor yboxes_wh = yboxes.slice(2, 2, 4, 1); //(num_objs, 9, 2)

	torch::Tensor intersections = torch::minimum(yboxes.slice(2, 2, 3, 1), anchors.slice(1, 0, 1, 1)) * torch::minimum(yboxes.slice(2, 3, 4, 1), anchors.slice(1, 1, 2, 1)); //min of (w) x min of (h)
	torch::Tensor yboxes_areas = yboxes.slice(2, 2, 3, 1) * yboxes.slice(2, 3, 4, 1);
	torch::Tensor anchors_areas = anchors.slice(1, 0, 1, 1) * anchors.slice(1, 1, 2, 1);
	//dimension reduction to (num_obj, 9)
	intersections = intersections.squeeze_(-1);
	yboxes_areas = yboxes_areas.squeeze_(-1);
	anchors_areas = anchors_areas.squeeze_(-1);

	torch::Tensor ious = intersections / (yboxes_areas + anchors_areas - intersections);

	return torch::argmax(ious, { 1 }); //return shape = [num_objs]
}


//DEPRECATED
torch::Tensor YOLOUTIL::broadcast_iou(torch::Tensor box_a, torch::Tensor box_b, int64_t grid_size) {
	/*
	box_a : a tensor full of boxes, e.g., (Batch,grid*grid*3(=N),4), box is in x1y1x2y2
    box_b : another tensor full of boxes (Batch,100,4)
	*/

	//(B,N,1,4)
	box_a = box_a.unsqueeze_(-2);
	//(B,1,100,4)
	box_b = box_b.unsqueeze_(-3);

	//broadcasted shape = [B,N,100,4]
	box_a = box_a.broadcast_to({ box_a.size(0), box_a.size(1), box_b.size(2), box_a.size(3) });
	box_b = box_b.broadcast_to({ box_a.size(0), box_a.size(1), box_b.size(2), box_a.size(3) });

	torch::Tensor al, at, ar, ab, bl, bt, br, bb;
	al = box_a.slice(3, 0, 1, 1);
	at = box_a.slice(3, 1, 2, 1);
	ar = box_a.slice(3, 2, 3, 1);
	ab = box_a.slice(3, 3, 4, 1);
	bl = box_b.slice(3, 0, 1, 1);
	bt = box_b.slice(3, 1, 2, 1);
	br = box_b.slice(3, 2, 3, 1);
	bb = box_b.slice(3, 3, 4, 1);


	torch::Tensor left = torch::maximum(al, bl); //10
	torch::Tensor right = torch::minimum(ar, br); //20
	torch::Tensor top = torch::maximum(at, bt);
	torch::Tensor bottom = torch::minimum(ab, bb);

	torch::Tensor iw = torch::clamp(right - left, 0, grid_size);
	torch::Tensor ih = torch::clamp(bottom - top, 0, grid_size);
	torch::Tensor i = iw * ih;

	torch::Tensor area_a = (ar - al) * (ab - at);
	torch::Tensor area_b = (br - bl) * (bb - bt);
	torch::Tensor union_area = area_a + area_b - i;

	torch::Tensor iou = (i / (union_area + EPSILON)).squeeze_(-1);

	return iou;
}




std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> POSTPROCESSING::threshold_confidence(torch::Tensor preds, float conf_thres) {

	auto keep = preds.select(1, 4).squeeze_() > conf_thres; //num
	auto ind = keep.nonzero().squeeze_(); //non_zeros num
	preds = preds.index_select(0, ind); //non_zeros num, bbox_attrs

	//handle no object found case
	if (ind.numel() == 0) {
		return std::make_tuple(ind, ind, ind);
	}
	//(non_zeros num, 80) x (non_zeros num, 1) = (non_zeros num, 80).amax(1) = (non_zeros num)
	auto var = preds.slice(1, 5).mul(preds.select(1, 4).unsqueeze(1));
	auto max_cls_score = var.amax(1);
	auto max_cls_indices = var.argmax(1);
	preds = preds.slice(1, 0, 5);

	return std::make_tuple(preds, max_cls_score, max_cls_indices);
}

void POSTPROCESSING::center_to_corner(torch::Tensor& bbox) {
	//top left x = centerX - w/2
	//top left y = centerY - h/2
	bbox.select(1, 0) -= bbox.select(1, 2) / 2;
	bbox.select(1, 1) -= bbox.select(1, 3) / 2;
}

std::array<int64_t, 2> POSTPROCESSING::letterbox_dim(torch::IntArrayRef img, torch::IntArrayRef box) {
	auto h = box[0], w = box[1];
	auto img_h = img[0], img_w = img[1];
	auto s = std::min(1.0f * w / img_w, 1.0f * h / img_h);
	return std::array<int64_t, 2>{ int64_t(img_h* s), int64_t(img_w* s) };
}

cv::Mat POSTPROCESSING::letterbox_img(const cv::Mat& img, torch::IntArrayRef box) {
	auto h = box[0], w = box[1];

	std::array<int64_t, 2> ret = letterbox_dim({ img.rows, img.cols }, box);
	auto new_h = ret[0], new_w = ret[1];

	cv::Mat out = (cv::Mat::zeros(h, w, CV_8UC3) + 1) * 128;

	cv::resize(img, out({ int((h - new_h) / 2), int((h - new_h) / 2 + new_h) }, { int((w - new_w) / 2), int((w - new_w) / 2 + new_w) }), { int(new_w),int(new_h) }, 0, 0, cv::INTER_CUBIC);

	return out;

}

void POSTPROCESSING::inv_letterbox_bbox(torch::Tensor bbox, torch::IntArrayRef box_dim, torch::IntArrayRef img_dim) {
	auto img_h = img_dim[0], img_w = img_dim[1];
	auto h = box_dim[0], w = box_dim[1];
	std::array<int64_t, 2> ret = letterbox_dim(img_dim, box_dim);
	auto new_h = ret[0], new_w = ret[1];

	bbox.select(1, 0).add_(-(w - new_w) / 2).mul_(1.0f * img_w / new_w); //x
	bbox.select(1, 2).mul_(1.0f * img_w / new_w); //w
	bbox.select(1, 1).add_(-(h - new_h) / 2).mul_(1.0f * img_h / new_h); //y
	bbox.select(1, 3).mul_(1.0f * img_h / new_h); //h;

}

void POSTPROCESSING::NMS(std::vector<Detection> & dets, float threshold) {
	// Sort by score //lambda expression
	std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.scr > b.scr; });

	//NMS
	for (size_t i = 0; i < dets.size(); ++i) {
		//remove_if: for [beginning, end), remove elem if op(elem) ==true
		dets.erase(std::remove_if(dets.begin() + i + 1, dets.end(), [&](const Detection& d) {
			if (dets[i].cls != d.cls) {
				return false;
			}
			//Only the highest score bbox is kept, the remainder that is above threshold is removed
			return IOUCalc(dets[i].bbox, d.bbox) > threshold;
			}), dets.end());
	}
}

float POSTPROCESSING::IOUCalc(const cv::Rect2f& bb_test, const cv::Rect2f& bb_gt) {
	auto in = (bb_test & bb_gt).area();
	auto un = (bb_test | bb_gt).area();
	return in / un;
}