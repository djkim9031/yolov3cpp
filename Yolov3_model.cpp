#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "Yolov3_model.h"
#include "yolo_utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Custom layer implementations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ResidualLayerImpl : torch::nn::Module {
    ResidualLayerImpl() = default;

    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2) {
        return x1+x2;
    }
};

TORCH_MODULE(ResidualLayer);

struct UpsampleLayerImpl : torch::nn::Module {
    int _stride;
    explicit UpsampleLayerImpl(int stride) : _stride(stride) {}

    torch::Tensor forward(torch::Tensor x) {
        auto sizes = x.sizes();
        auto w = sizes[2] * _stride;
        auto h = sizes[3] * _stride;

        return torch::upsample_nearest2d(x, { w, h });
    }

};

TORCH_MODULE(UpsampleLayer);

struct ConcatLayerImpl : torch::nn::Module {
    ConcatLayerImpl() = default;

    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2) {

        return torch::cat({ x1, x2 }, 1);;
    }

};

TORCH_MODULE(ConcatLayer);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YOLO Class definitions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::nn::Conv2dOptions conv2doptions(int64_t in_planes, int64_t filters, int64_t kerner_size, int64_t strides, int64_t padding, int64_t groups, bool with_bias) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, filters, kerner_size);
    conv_options.stride(strides);
    conv_options.padding(padding);
    conv_options.groups(groups);
    conv_options.bias(with_bias);
    return conv_options;

}

torch::nn::BatchNorm2dOptions bn2d_options(int64_t features) {
    torch::nn::BatchNorm2dOptions bn_options = torch::nn::BatchNorm2dOptions(features);
    bn_options.affine(true);
    bn_options.track_running_stats(true);
    return bn_options;
}

YOLO::YOLO(torch::Tensor inputs, int num_classes) {
    _inputs = inputs; 
    _num_classes = num_classes;
    _num_trainable_parameters = 0;
    YOLOseq.clear();
    YOLOseq_layers.clear();
    trainable_parameters.clear();
    trackIdx = 0;
    YOLOv3(_inputs, _num_classes, true);
    YOLOv3ShapeTest(_inputs, true);

}
YOLO::~YOLO() = default;

//inputs shape = [Batch, C, H, W]
torch::Tensor YOLO::DarknetConv(torch::Tensor inputs, int filters, int kernel_size, int strides, bool initialize) {

    if (initialize) {
        torch::nn::Sequential DarknetConv_seq;
        torch::nn::Conv2d conv = torch::nn::Conv2d(conv2doptions(inputs.size(1), filters, kernel_size, strides, (kernel_size - 1) / 2, 1, false));
        torch::nn::BatchNorm2d bn = torch::nn::BatchNorm2d(bn2d_options(filters));
        torch::nn::Functional lr = torch::nn::Functional(at::leaky_relu, /*slope=*/0.1);
        DarknetConv_seq->push_back(conv);
        DarknetConv_seq->push_back(bn);
        DarknetConv_seq->push_back(lr);
        YOLOseq.push_back(DarknetConv_seq);
        YOLOseq_layers.push_back("conv_bn");

        //Adding trainable parameters;
        for (int i = 0; i < conv->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq.back()[0]->parameters()[i]);
        }
        for (int i = 0; i < bn->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq.back()[1]->parameters()[i]);
        }

        return lr(bn(conv(inputs)));
    }
    
    torch::Tensor outputs = YOLOseq[trackIdx]->forward(inputs);
    //Updating trainable parameters;
    for (int i = 0; i < YOLOseq[trackIdx][0]->parameters().size(); i++) {
        trainable_parameters.push_back(YOLOseq[trackIdx][0]->parameters()[i]);
    }
    for (int i = 0; i < YOLOseq[trackIdx][1]->parameters().size(); i++) {
        trainable_parameters.push_back(YOLOseq[trackIdx][1]->parameters()[i]);
    }
    trackIdx++;
    return outputs;

}

torch::Tensor YOLO::DarknetResidual(torch::Tensor inputs, int filter1, int filter2, bool initialize) {

    
    torch::Tensor outputs = DarknetConv(DarknetConv(inputs, filter1, 1, 1, initialize), filter2, 3, 1, initialize);
    if (initialize) {
        torch::nn::Sequential DarknetResidual_seq;
        ResidualLayer residual;
        DarknetResidual_seq->push_back(residual);
        YOLOseq.push_back(DarknetResidual_seq);
        YOLOseq_layers.push_back("residual");

        return inputs + outputs;
    }

    outputs = YOLOseq[trackIdx]->forward(inputs, outputs);
    trackIdx++;
    return outputs;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> YOLO::Darknet(torch::Tensor inputs, bool initialize) {
    torch::Tensor x, y0, y1, y2;

    x = DarknetConv(inputs, 32, 3, 1, initialize);
    x = DarknetConv(x, 64, 3, 2, initialize);

    // 1 x residual block
    x = DarknetResidual(x, 32, 64, initialize);

    x = DarknetConv(x, 128, 3, 2, initialize);

    // 2 x residual blocks
    for (int i = 0; i < 2; i++) {
        x = DarknetResidual(x, 64, 128, initialize);
    }

    x = DarknetConv(x, 256, 3, 2, initialize);

    // 8 x residual blocks
    for (int i = 0; i < 8; i++) {
        x = DarknetResidual(x, 128, 256, initialize);
    }

    //First output
    y0 = x;

    x = DarknetConv(x, 512, 3, 2, initialize);

    // 8 x residual blocks
    for (int i = 0; i < 8; i++) {
        x = DarknetResidual(x, 256, 512, initialize);
    }

    //Second output
    y1 = x;

    x = DarknetConv(x, 1024, 3, 2, initialize);

    // 4 x residual blocks
    for (int i = 0; i < 4; i++) {
        x = DarknetResidual(x, 512, 1024, initialize);
    }

    //Third output
    y2 = x;

    return std::make_tuple(y0, y1, y2);
  

}

std::vector<torch::Tensor> YOLO::YOLOv3(torch::Tensor inputs, int num_classes, bool initialize) {
    _inputs = inputs;

    int final_filters = 3 * (4 + 1 + num_classes);
    torch::Tensor x;

    auto backbone = Darknet(inputs, initialize);
    auto x_large = std::get<0>(backbone); //52x52 feature map
    auto x_medium = std::get<1>(backbone); //26x26
    auto x_small = std::get<2>(backbone); // 13x13

    //small scale detection
    x = DarknetConv(x_small, 512, 1, 1, initialize);
    x = DarknetConv(x, 1024, 3, 1, initialize);
    x = DarknetConv(x, 512, 1, 1, initialize);
    x = DarknetConv(x, 1024, 3, 1, initialize);
    x = DarknetConv(x, 512, 1, 1, initialize);
    torch::Tensor y_small = DarknetConv(x, 1024, 3, 1, initialize);
    if (initialize) {
        torch::nn::Sequential Conv_seq;
        torch::nn::Conv2d conv = torch::nn::Conv2d(conv2doptions(y_small.size(1), final_filters, 1, 1, 0, 1, true));

        Conv_seq->push_back(conv);
        YOLOseq.push_back(Conv_seq);
        YOLOseq_layers.push_back("conv");

        for (int i = 0; i < conv->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq.back()[0]->parameters()[i]);
        }

        y_small = conv(y_small); //batch, 3x(4+1+num_classes), 13, 13
    }
    else {
        y_small = YOLOseq[trackIdx]->forward(y_small);
        //Updating trainable parameters;
        for (int i = 0; i < YOLOseq[trackIdx][0]->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq[trackIdx][0]->parameters()[i]);
        }
        trackIdx++;
    }

    //medium scale detection
    x = DarknetConv(x, 256, 1, 1, initialize);
    if (initialize) {
        torch::nn::Sequential Up_seq;
        torch::nn::Sequential Cat_seq;
        UpsampleLayer uplayer(2);
        ConcatLayer catlayer;

        Up_seq->push_back(uplayer);
        Cat_seq->push_back(catlayer);
        YOLOseq.push_back(Up_seq);
        YOLOseq_layers.push_back("upsampling");
        YOLOseq.push_back(Cat_seq);
        YOLOseq_layers.push_back("concatenate");

        x = torch::upsample_nearest2d(x, { x.size(2) * 2,  x.size(3) * 2 });
        x = torch::cat({ x, x_medium }, 1);
    }
    else {
        x = YOLOseq[trackIdx]->forward(x);
        trackIdx++;
        x = YOLOseq[trackIdx]->forward(x, x_medium);
        trackIdx++;
    }
    x = DarknetConv(x, 256, 1, 1, initialize);
    x = DarknetConv(x, 512, 3, 1, initialize);
    x = DarknetConv(x, 256, 1, 1, initialize);
    x = DarknetConv(x, 512, 3, 1, initialize);
    x = DarknetConv(x, 256, 1, 1, initialize);
    torch::Tensor y_medium = DarknetConv(x, 512, 3, 1, initialize);
    if (initialize) {
        torch::nn::Sequential Conv_seq;
        torch::nn::Conv2d conv = torch::nn::Conv2d(conv2doptions(y_medium.size(1), final_filters, 1, 1, 0, 1, true));

        Conv_seq->push_back(conv);
        YOLOseq.push_back(Conv_seq);
        YOLOseq_layers.push_back("conv");

        for (int i = 0; i < conv->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq.back()[0]->parameters()[i]);
        }

        y_medium = conv(y_medium); //batch, 3x(4+1+num_classes), 26, 26
    }
    else {
        y_medium = YOLOseq[trackIdx]->forward(y_medium);
        //Updating trainable parameters;
        for (int i = 0; i < YOLOseq[trackIdx][0]->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq[trackIdx][0]->parameters()[i]);
        }
        trackIdx++;
    }

    //large scale detection
    x = DarknetConv(x, 128, 1, 1, initialize);
    if (initialize) {
        torch::nn::Sequential Up_seq;
        torch::nn::Sequential Cat_seq;
        UpsampleLayer uplayer(2);
        ConcatLayer catlayer;

        Up_seq->push_back(uplayer);
        Cat_seq->push_back(catlayer);
        YOLOseq.push_back(Up_seq);
        YOLOseq_layers.push_back("upsampling");
        YOLOseq.push_back(Cat_seq);
        YOLOseq_layers.push_back("concatenate");

        x = torch::upsample_nearest2d(x, { x.size(2) * 2,  x.size(3) * 2 });
        x = torch::cat({ x, x_large }, 1);
    }
    else {
        x = YOLOseq[trackIdx]->forward(x);
        trackIdx++;
        x = YOLOseq[trackIdx]->forward(x, x_large);
        trackIdx++;
    }
    x = DarknetConv(x, 128, 1, 1, initialize);
    x = DarknetConv(x, 256, 3, 1, initialize);
    x = DarknetConv(x, 128, 1, 1, initialize);
    x = DarknetConv(x, 256, 3, 1, initialize);
    x = DarknetConv(x, 128, 1, 1, initialize);
    torch::Tensor y_large = DarknetConv(x, 256, 3, 1, initialize);
    if (initialize) {
        torch::nn::Sequential Conv_seq;
        torch::nn::Conv2d conv = torch::nn::Conv2d(conv2doptions(y_large.size(1), final_filters, 1, 1, 0, 1, true));

        Conv_seq->push_back(conv);
        YOLOseq.push_back(Conv_seq);
        YOLOseq_layers.push_back("conv");

        for (int i = 0; i < conv->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq.back()[0]->parameters()[i]);
        }

        y_large = conv(y_large); //batch, 3x(4+1+num_classes), 52, 52
    }
    else {
        y_large = YOLOseq[trackIdx]->forward(y_large);
        //Updating trainable parameters;
        for (int i = 0; i < YOLOseq[trackIdx][0]->parameters().size(); i++) {
            trainable_parameters.push_back(YOLOseq[trackIdx][0]->parameters()[i]);
        }
        trackIdx = 0;
    }

    
    //Reshape (batch, 3x(4+1+num_classes), grid, grid) to (batch, 3, X, grid, grid)
    y_small = y_small.contiguous().view({ y_small.size(0), 3, -1, y_small.size(2), y_small.size(3)});
    y_medium = y_medium.contiguous().view({ y_medium.size(0), 3, -1, y_medium.size(2), y_medium.size(3) });
    y_large = y_large.contiguous().view({ y_large.size(0), 3, -1, y_large.size(2), y_large.size(3) });


    if (initialize) {
        for (int i = 0; i < trainable_parameters.size(); i++) {
            try
            {
                _num_trainable_parameters += trainable_parameters[i].size(0) * trainable_parameters[i].size(1) * trainable_parameters[i].size(2) * trainable_parameters[i].size(3);

            }
            catch (const std::exception&)
            {
                _num_trainable_parameters += trainable_parameters[i].size(0);
            }

                
        }
    }

    std::vector<torch::Tensor> y_preds;
    y_preds.push_back(y_large);
    y_preds.push_back(y_medium);
    y_preds.push_back(y_small);

    return y_preds;

}


torch::Tensor YOLO::get_absolute_yolo_box(torch::Tensor y_pred, torch::Tensor anchors) {
    //y_pred shape = [B, 3, X, grid, grid], X = 4 + 1 + class_num
    //anchors shape = [3, 2]
    
    torch::Tensor bx, by, bw, bh, objectness, classes;
    auto grid_size = y_pred.sizes().slice(3);
    std::array<torch::Tensor, 2> grid = { torch::arange(grid_size[0],y_pred.options()), torch::arange(grid_size[1],y_pred.options()) };

    bx = y_pred.select(2, 0).sigmoid_().add_(grid[1].view({ 1,1,1,-1 })).mul_(_inputs.size(3) / grid_size[1]).view({y_pred.size(0),3,1,grid_size[0],grid_size[1]});
    by = y_pred.select(2, 1).sigmoid_().add_(grid[0].view({ 1,1,-1,1 })).mul_(_inputs.size(2) / grid_size[0]).view({ y_pred.size(0),3,1,grid_size[0],grid_size[1] });
    bw = y_pred.select(2, 2).exp_().mul_(anchors.select(1, 0).view({ 1,-1,1,1 })).view({ y_pred.size(0),3,1,grid_size[0],grid_size[1] });
    bh = y_pred.select(2, 3).exp_().mul_(anchors.select(1, 1).view({ 1,-1,1,1 })).view({ y_pred.size(0),3,1,grid_size[0],grid_size[1] });
    objectness = y_pred.select(2, 4).sigmoid_().view({ y_pred.size(0),3,1,grid_size[0],grid_size[1] });
    classes = y_pred.slice(2, 5).softmax(-1);

    return torch::cat({bx,by,bw,bh,objectness,classes},2);

}

torch::Tensor YOLO::get_relative_yolo_box(torch::Tensor y_true, torch::Tensor anchors) {
    //y_true shape = [B, 3, X, grid, grid], X = 4 + 1 + class_num, y_true (bx,by,bw,bh = values in 416 scale)
    //anchors shape = [3, 2]

    torch::Tensor tx, ty, tw, th, objectness, classes; //tx, ty = values after sigmoid operation
    auto grid_size = y_true.sizes().slice(3);
    std::array<torch::Tensor, 2> grid = { torch::arange(grid_size[0],y_true.options()), torch::arange(grid_size[1],y_true.options()) };

    tx = y_true.select(2, 0).div_(_inputs.size(3) / grid_size[1]).sub_(grid[1].view({1,1,1,-1})).view({y_true.size(0),3,1,grid_size[0],grid_size[1]});
    ty = y_true.select(2, 1).div_(_inputs.size(2) / grid_size[0]).sub_(grid[0].view({1,1,-1,1})).view({y_true.size(0),3,1,grid_size[0],grid_size[1]});
    tw = y_true.select(2, 2).div_(anchors.select(1, 0).view({ 1,-1,1,1 })).log().view({ y_true.size(0),3,1,grid_size[0],grid_size[1] });
    th = y_true.select(2, 3).div_(anchors.select(1, 1).view({ 1,-1,1,1 })).log().view({ y_true.size(0),3,1,grid_size[0],grid_size[1] });
    objectness = y_true.slice(2, 4, 5, 1);
    classes = y_true.slice(2, 5);

    return torch::cat({ tx,ty,tw,th,objectness,classes }, 2);
}

torch::Tensor YOLO::prediction_postprocess(std::vector<torch::Tensor> y_preds) {

    torch::Tensor y_large = get_absolute_yolo_box(y_preds[0], _anchors.slice(0, 0, 3, 1));
    torch::Tensor y_medium = get_absolute_yolo_box(y_preds[1], _anchors.slice(0, 3, 6, 1));
    torch::Tensor y_small = get_absolute_yolo_box(y_preds[2], _anchors.slice(0, 6, 9, 1));

    y_small = y_small.transpose(2, -1).contiguous().view({ 1, -1, y_small.size(2) }).squeeze_(0);
    y_medium = y_medium.transpose(2, -1).contiguous().view({ 1, -1, y_medium.size(2) }).squeeze_(0);
    y_large = y_large.transpose(2, -1).contiguous().view({ 1, -1, y_large.size(2) }).squeeze_(0);

    torch::Tensor outputs = torch::concat({ y_small, y_medium, y_large}, 0);

    return outputs;
}

auto YOLO::loss_calc(torch::Tensor y_true, torch::Tensor y_pred, torch::Tensor anchors) {
    //y_true (bx,by,bw,bh,obj,cls), y_pred (tx,ty,tw,th,obj,cls) => loss func has to convert first (e.g., from bx = sigmoid(tx)+cx) and calc the difference
    //Idea is to arrange values between (0~1) range to accelerate convergence.
    //y_true -> get_relative -> tx, ty, tw, th comparison
    //y_pred -> get_absolute -> obj, cls comparison

    torch::Tensor pred_abs = get_absolute_yolo_box(y_pred, anchors);
    torch::Tensor true_rel = get_relative_yolo_box(y_true, anchors);

    //x,y ->sigmoid
    torch::Tensor pred_x = y_pred.slice(2, 0, 1, 1).sigmoid_();
    torch::Tensor true_x = true_rel.slice(2, 0, 1, 1);
    torch::Tensor pred_y = y_pred.slice(2, 1, 2, 1).sigmoid_();
    torch::Tensor true_y = true_rel.slice(2, 1, 2, 1);
    //w,h
    torch::Tensor pred_w = y_pred.slice(2, 2, 3, 1);
    torch::Tensor true_w = true_rel.slice(2, 2, 3, 1);
    torch::Tensor pred_h = y_pred.slice(2, 3, 4, 1);
    torch::Tensor true_h = true_rel.slice(2, 3, 4, 1);
    //obj
    torch::Tensor pred_obj = pred_abs.slice(2, 4, 5, 1);
    torch::Tensor true_obj = y_true.slice(2, 4, 5, 1);
    //cls
    torch::Tensor pred_cls = pred_abs.slice(2, 5);
    torch::Tensor true_cls = y_true.slice(2, 5);

    torch::Tensor loss_x, loss_y, loss_w, loss_h, loss_obj, loss_cls;
        
    loss_x = calc_dim_loss(true_obj, true_x, pred_x);
    loss_y = calc_dim_loss(true_obj, true_y, pred_y);
    loss_w = calc_dim_loss(true_obj, true_w, pred_w);
    loss_h = calc_dim_loss(true_obj, true_h, pred_h);
    loss_obj = calc_obj_loss(true_obj, pred_obj);
    loss_cls = calc_cls_loss(true_cls, pred_cls);

    return std::make_tuple(loss_x, loss_y, loss_w, loss_h, loss_obj, loss_cls);
}

torch::Tensor YOLO::calc_train_loss(std::vector<torch::Tensor> y_true, std::vector<torch::Tensor> y_preds, int batch_size) {
    torch::Tensor losses_x, losses_y, losses_w, losses_h, losses_obj, losses_cls, total_losses;
    losses_x = torch::zeros({ 1 }); losses_y = torch::zeros({ 1 }); losses_w = torch::zeros({ 1 }); losses_h = torch::zeros({ 1 });
    losses_obj = torch::zeros({ 1 }); losses_cls = torch::zeros({ 1 }); total_losses = torch::zeros({ 1 });
    total_losses.set_requires_grad(true);

    for (int scale = 0; scale < 3; scale++) {
        auto losses = loss_calc(y_true[scale], y_preds[scale], _anchors.slice(0, scale * 3, (scale + 1) * 3, 1));
        torch::Tensor loss_x, loss_y, loss_w, loss_h, loss_obj, loss_cls, total_loss;
        loss_x = std::get<0>(losses);
        loss_y = std::get<1>(losses);
        loss_w = std::get<2>(losses);
        loss_h = std::get<3>(losses);
        loss_obj = std::get<4>(losses);
        loss_cls = std::get<5>(losses);
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_cls;

        losses_x += loss_x.sum() * (1. / batch_size); losses_y += loss_y.sum() * (1. / batch_size); losses_w += loss_w.sum() * (1. / batch_size); losses_h += loss_h.sum() * (1. / batch_size);
        losses_obj += loss_obj.sum() * (1. / batch_size); losses_cls += loss_cls.sum() * (1. / batch_size); total_losses += total_loss.sum() * (1. / batch_size);
    }
    losses_x /= 3; losses_y /= 3; losses_w /= 3; losses_h /= 3; losses_obj /= 3; losses_cls /= 3; total_losses /= 3;

    std::cout << ": x loss: " << losses_x << ", y loss: " << losses_y << ", w loss: " << losses_w << ", h loss: " << losses_h << ", obj loss: " << losses_obj << ", cls loss: " << losses_cls << ", total loss: " << total_losses << std::endl;

    return total_losses;

}

void YOLO::load_weights(const char* weight_file) {
    std::ifstream fs(weight_file, std::ios_base::binary);
    if (!fs) {
        throw std::runtime_error("No weight file for Darknet!");
    }

    fs.seekg(sizeof(int32_t) * 5, std::ios_base::beg);

    for (size_t i = 0; i < YOLOseq.size(); i++) {
        // only conv layers need to load weights
        if (YOLOseq_layers[i] == "conv_bn" || YOLOseq_layers[i] == "conv") {
            auto seq_module = YOLOseq[i];
            auto conv = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(seq_module[0]);
            if (YOLOseq_layers[i] == "conv_bn") {
                auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(seq_module[1]);
                load_tensor(bn->bias, fs);
                load_tensor(bn->weight, fs);
                load_tensor(bn->running_mean, fs);
                load_tensor(bn->running_var, fs);
            }
            else {
                load_tensor(conv->bias, fs);
            }
            load_tensor(conv->weight, fs);
        }
    }
}


torch::Tensor YOLO::train(torch::Tensor inputs, torch::Tensor labels, torch::Tensor val_inputs, torch::Tensor val_labels, int batch_size, int epochs) {
    std::cout << "Data preprocessing begins..." << std::endl;
    //shuffle inputs-labels pairs and put them into batches
    std::vector<torch::Tensor> dataset = YOLOUTIL::shuffle_inputs_labels(inputs, labels, _anchors, _num_classes);
    std::vector<torch::Tensor> val_dataset = YOLOUTIL::shuffle_inputs_labels(val_inputs, val_labels, _anchors, _num_classes);
    inputs = dataset[0];
    torch::Tensor y0, y1, y2;
    y0 = dataset[1]; //large set
    y1 = dataset[2]; //medium set
    y2 = dataset[3]; //small set
    int batch_num = inputs.size(0) / batch_size;

    //validation set
    val_inputs = val_dataset[0];
    int val_batch_size = int(val_inputs.size(0));
    torch::Tensor val_y0, val_y1, val_y2;
    val_y0 = val_dataset[1]; //large set
    val_y1 = val_dataset[2]; //medium set
    val_y2 = val_dataset[3]; //small set

    std::cout << "YOLOv3 training begins..." << std::endl << "Number of input images: " << inputs.size(0) << std::endl;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::cout << "Epoch " << epoch << "__________________________________________________________" << std::endl;
        float total_loss = 0.0;
        auto start = std::chrono::steady_clock::now();
        for (int batch = 0; batch < batch_num; batch++) {
            torch::Tensor batch_inputs;
            std::vector<torch::Tensor> y_true;
            try
            {
                batch_inputs = inputs.slice(0, batch_size * batch, batch_size * (batch + 1), 1);
                y_true.push_back(y0.slice(0, batch_size * batch, batch_size * (batch + 1), 1));
                y_true.push_back(y1.slice(0, batch_size * batch, batch_size * (batch + 1), 1));
                y_true.push_back(y2.slice(0, batch_size * batch, batch_size * (batch + 1), 1));
            }
            catch (const std::exception&)
            {
                batch_inputs = inputs.slice(0, batch_size * batch);
                y_true.push_back(y0.slice(0, batch_size * batch));
                y_true.push_back(y1.slice(0, batch_size * batch));
                y_true.push_back(y2.slice(0, batch_size * batch));
            }
            //y_true = one batch of the labels (to be implemented)
            std::vector<torch::Tensor> y_preds = YOLOv3(batch_inputs, _num_classes, false);
            std::cout << "Batch " << batch + 1;
            torch::Tensor total_losses = calc_train_loss(y_true, y_preds, batch_size);

            total_loss += total_losses.item().toFloat();
            total_losses.backward();
            auto adam_optim = torch::optim::Adam(trainable_parameters, torch::optim::AdamOptions(1e-4));
            adam_optim.step();
            adam_optim.zero_grad();
            trainable_parameters.clear();


        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Training Total loss: " << total_loss << ", Training speed: " << duration/(1000*batch_size) << "batch/sec __________________________________________________________" << std::endl;

        //validation
        std::vector<torch::Tensor> val_y_true;
        val_y_true.push_back(val_y0);
        val_y_true.push_back(val_y1);
        val_y_true.push_back(val_y2);

        start = std::chrono::steady_clock::now();
        std::vector<torch::Tensor> val_y_preds = YOLOv3(val_inputs, _num_classes, false);
        std::cout << "Validation set: ";
        torch::Tensor val_total_losses = calc_train_loss(val_y_true, val_y_preds, val_batch_size);
        end = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Validation speed : " << duration / (1000 * batch_size) << "batch / sec" << std::endl;
        std::cout << "Epoch " << epoch << " Training finished __________________________________________________________" << std::endl;


    }
}

std::vector<Detection> YOLO::predict(cv::Mat image) {
    //turn off the grad
    torch::NoGradGuard no_grad;
    int64_t orig_dim[] = { image.rows, image.cols };

    //cv::resize(image, image, { int(_inputs.size(2)) , int(_inputs.size(3)) }, 0, 0, cv::INTER_CUBIC);
    image = POSTPROCESSING::letterbox_img(image, { int(_inputs.size(2)) , int(_inputs.size(3)) });

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F);
    auto imageTensors = torch::from_blob(image.data, { 1,  int(_inputs.size(2)) , int(_inputs.size(3)), 3 }).permute({ 0,3,1,2 }).div_(255.0);

    std::vector<Detection> dets;

    std::vector<torch::Tensor> predictions = YOLOv3(imageTensors, _num_classes, false);
    torch::Tensor preds = prediction_postprocess(predictions);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> var = POSTPROCESSING::threshold_confidence(preds, confidence_threshold);
    auto bbox = std::get<0>(var);
    auto scr = std::get<1>(var);
    auto cls = std::get<2>(var);

    //No object found
    if (bbox.numel() == 0) {
        return dets;
    }

    //move data back to cpu
    bbox = bbox.cpu();
    cls = cls.cpu();
    scr = scr.cpu();

    POSTPROCESSING::center_to_corner(bbox);
    POSTPROCESSING::inv_letterbox_bbox(bbox, { int(_inputs.size(2)) , int(_inputs.size(3)) }, orig_dim);


    auto bbox_acc = bbox.accessor<float, 2>();
    auto scr_acc = scr.accessor<float, 1>();
    auto cls_acc = cls.accessor<int64, 1>();

    for (int64_t i = 0; i < bbox_acc.size(0); ++i) {
        // Rect2f <- [bx,by, bw, bh]
        auto d = Detection{ cv::Rect2f(bbox_acc[i][0], bbox_acc[i][1], bbox_acc[i][2], bbox_acc[i][3]),scr_acc[i],cls_acc[i] };
        dets.emplace_back(d);
    }

    POSTPROCESSING::NMS(dets, NMS_threshold);

    return dets;
}


void YOLO::DarknetShapeTest() {
    /*
    if (YOLOseq.size() != YOLOseq_layers.size()) {
        std::cout << "sizes are different!" << std::endl;
        return;
    }

;
    for (int i = 0; i < YOLOseq.size(); i++) {

        std::cout << YOLOseq_layers[i] << std::endl;
    }

    auto var = Darknet(_inputs, false);

    auto y0 = std::get<0>(var);
    auto y1 = std::get<1>(var);
    auto y2 = std::get<2>(var);

    std::cout << y0.sizes() << std::endl;
    std::cout << y1.sizes() << std::endl;
    std::cout << y2.sizes() << std::endl;
    */
}

void YOLO::YOLOv3ShapeTest(torch::Tensor inputs, bool initialize) {

    if (YOLOseq.size() != YOLOseq_layers.size()) {
        std::cout << "sizes are different!" << std::endl;
        return;
    }

    std::cout << "Model layers: " << std::endl;
    for (int i = 0; i < YOLOseq.size(); i++) {

        std::cout <<i <<": " << YOLOseq_layers[i] << std::endl;
    }

    if (initialize) {
        std::cout << "------------------------------------ " << std::endl;
        std::cout << "YOLO object created." << std::endl;
        std::cout << "number of trainable parameters: " << _num_trainable_parameters << std::endl;

        _num_trainable_parameters = 0;
        trainable_parameters.clear();
        return;
    }

    std::vector<torch::Tensor> var = YOLOv3(inputs, _num_classes, initialize);

    auto y0 = var[0];
    auto y1 = var[1];
    auto y2 = var[2];

    std::cout << "------------------------------------ " << std::endl;
    std::cout << "Input size: " << _inputs.size(1) << "," << _inputs.size(2) << "," << _inputs.size(3) << std::endl;
    std::cout << "first output size: " << y0.sizes() << std::endl;
    std::cout << "second output size: " << y1.sizes() << std::endl;
    std::cout << "third output size: " << y2.sizes() << std::endl;
    std::cout << "------------------------------------ " << std::endl;
    std::cout << "number of trainable parameters: " << _num_trainable_parameters << std::endl;

    _num_trainable_parameters = 0;
    trainable_parameters.clear();
}

void YOLO::LossFuncTest() {
   // auto var = YOLOv3(_inputs, 2);

   // auto y_small = std::get<0>(var);
   // auto y_medium = std::get<1>(var);
   // auto y_large = std::get<2>(var);

  //  torch::Tensor y_true = torch::rand({ 1, 3, 7, 13, 13 });
 //   std::cout<<"y_true: " << y_true.requires_grad() << std::endl;
  //  std::cout << "y_true grads: " << y_true.grad() << std::endl;

   // auto losses = loss_calc(y_true, y_small, _anchors.slice(0, 0, 3, 1));

   // auto loss_x = std::get<0>(losses);
  //  auto loss_obj = std::get<4>(losses);
  //  auto loss_cls = std::get<5>(losses);

  //  std::cout<<"loss_cls: " << loss_cls.requires_grad() << std::endl;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YOLO Loss Class definitions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
YOLOLoss::YOLOLoss() {
   
}
YOLOLoss::~YOLOLoss() = default;

torch::Tensor YOLOLoss::calc_dim_loss(torch::Tensor true_obj, torch::Tensor true_val, torch::Tensor pred_val) {
    //true_obj shape = [B, 3(anchors), 1, grid, grid]
    //true_val, pred_val shapes = [B, 3, 1, grid, grid];

    //Food for thought: x,y,w,h take in true_obj as a mask to only calc loss for the grid with true_obj=1.
    //This will make things faster and learn for x,y,w,h for true_obj location. But this heavily depends on it learning correctly where true obj is.
    //If pred_obj location is not correct, x,y,w,h didn't learn well so it could be very inacurrate.
    torch::Tensor val_loss = torch::sum(true_val.slice(2, 0).sub_(pred_val.slice(2, 0)).abs_(), 2);
    true_obj = true_obj.squeeze_(2);
    val_loss = true_obj * val_loss;

    val_loss = torch::sum(val_loss, { 1,2,3 }) * weight_bbox;
    //std::cout << "val loss: " << val_loss.requires_grad() << std::endl;
    return val_loss;
}


torch::Tensor YOLOLoss::calc_obj_loss(torch::Tensor true_obj, torch::Tensor pred_obj) {
    //Calculate the loss of objectness
    //true_obj, pred_obj shapes = [B, 3(anchors), 1, grid, grid]
    torch::Tensor obj_entropy = YOLOUTIL::binary_cross_entropy(pred_obj, true_obj);
    return torch::sum(obj_entropy, { 1,2,3,4 }) * weight_obj;
}

torch::Tensor YOLOLoss::calc_cls_loss(torch::Tensor true_class, torch::Tensor pred_class) {
    //true_class, pred_class shapes = [B, 3, num_classes, grid, grid]
    torch::Tensor cls_entropy = YOLOUTIL::binary_cross_entropy(pred_class, true_class);
    torch::Tensor multiclass_cross_entropy = torch::sum(cls_entropy, { 2 }, true) / true_class.size(2); //C.E = 1/n * (sigma)(BCE)
    return torch::sum(multiclass_cross_entropy, { 1,2,3,4 });
}


/* NOT NEEDED
torch::Tensor YOLOLoss::calc_ignore_mask(torch::Tensor true_obj, torch::Tensor true_box, torch::Tensor pred_box) {
    
    //    If the bounding box prior is not the best, but does overlap a ground truth object
    //    by more than some threshold we ignore the prediction. We use the threshold of .5
    //    Cacluate the iou for each pair of pred bbox and true bbox, then find the best among them
    

    //box in x1y1x2y2 dim, (0~1) range

    auto true_box_shape = true_box.sizes(); //[B, 3, 4, grid, grid]
    auto pred_box_shape = pred_box.sizes();

    true_box = true_box.contiguous().view({ true_box_shape[0],-1,4 }); //[B, 3*grid*grid, 4]
    auto var = torch::sort(true_box, 1, true); //sort in desc order
    true_box = std::get<0>(var);
    true_box = true_box.slice(1, 0, 100, 1);

    pred_box = pred_box.contiguous().view({ pred_box_shape[0],-1,4 });


    //broadcast iou; iou shape = [B, 3*grid*grid, 100]
    torch::Tensor iou = broadcast_iou(pred_box, true_box, true_box.size(4));

    return true_obj;
}
*/
