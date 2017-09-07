#include <vector>

#include "quantization/base_ristretto_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
LRNRistrettoLayer<Dtype>::LRNRistrettoLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() {
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  case QuantizationParameter_Precision_MINIFLOAT:
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(this->size_ % 2, 1) << "LRN only supports odd values for local_size";
  this->pre_pad_ = (this->size_ - 1) / 2;
  this->alpha_ = this->layer_param_.lrn_param().alpha();
  this->beta_ = this->layer_param_.lrn_param().beta();
  this->k_ = this->layer_param_.lrn_param().k();
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    this->split_top_vec_.clear();
    this->split_top_vec_.push_back(&this->product_input_);
    this->split_top_vec_.push_back(&this->square_input_);
    LayerParameter split_param;
    this->split_layer_.reset(new SplitLayer<Dtype>(split_param));
    this->split_layer_->SetUp(bottom, this->split_top_vec_);
    // Set up square_layer_ to square the inputs.
    this->square_bottom_vec_.clear();
    this->square_top_vec_.clear();
    this->square_bottom_vec_.push_back(&this->square_input_);
    this->square_top_vec_.push_back(&this->square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(Dtype(2));
    this->square_layer_.reset(new PowerLayer<Dtype>(square_param));
    this->square_layer_->SetUp(this->square_bottom_vec_, this->square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    this->pool_top_vec_.clear();
    this->pool_top_vec_.push_back(&this->pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(this->pre_pad_);
    pool_param.mutable_pooling_param()->set_kernel_size(this->size_);
    this->pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
    this->pool_layer_->SetUp(this->square_top_vec_, this->pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
    // the sum of a squared neighborhood (the output of pool_layer_).
    this->power_top_vec_.clear();
    this->power_top_vec_.push_back(&this->power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-this->beta_);
    power_param.mutable_power_param()->set_scale(this->alpha_);
    power_param.mutable_power_param()->set_shift(Dtype(1));
    this->power_layer_.reset(new PowerLayer<Dtype>(power_param));
    this->power_layer_->SetUp(this->pool_top_vec_, this->power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    this->product_bottom_vec_.clear();
    this->product_bottom_vec_.push_back(&this->product_input_);
    this->product_bottom_vec_.push_back(&this->power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    this->product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
    this->product_layer_->SetUp(this->product_bottom_vec_, top);
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    // TODO
    LOG(FATAL) << "LRNRistretto ACROSS Layer not implemented on CPU yet.";
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->split_layer_->Forward(bottom, this->split_top_vec_);
  this->square_layer_->Forward(this->square_bottom_vec_, this->square_top_vec_);
  this->pool_layer_->Forward(this->square_top_vec_, this->pool_top_vec_);
  this->power_layer_->Forward(this->pool_top_vec_, this->power_top_vec_);
  this->product_layer_->Forward(this->product_bottom_vec_, top);
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //TODO
  LOG(FATAL) << "LRNRistrettoLayer not implemented on CPU yet.";
}

#ifdef CPU_ONLY
STUB_GPU(LRNRistrettoLayer);
STUB_GPU_FORWARD(LRNRistrettoLayer, CrossChannelForward);
#endif

INSTANTIATE_CLASS(LRNRistrettoLayer);
REGISTER_LAYER_CLASS(LRNRistretto);

}  // namespace caffe

