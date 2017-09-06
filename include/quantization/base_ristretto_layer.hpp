#ifndef CAFFE_BASE_RISTRETTO_LAYER_HPP_
#define CAFFE_BASE_RISTRETTO_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

/**
 * @brief Provides quantization methods used by other Ristretto layers.
 */
template <typename Dtype>
class BaseRistrettoLayer{
 public:
  explicit BaseRistrettoLayer();
 protected:
  void QuantizeLayerOutputs_cpu(Dtype* data, const int count);
  void QuantizeLayerInputs_cpu(Dtype* data, const int count);
  void QuantizeLayerOutputs_gpu(Dtype* data, const int count);
  void QuantizeLayerInputs_gpu(Dtype* data, const int count);
  void QuantizeWeights_cpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,
      const int rounding, const bool bias_term = true);
  void QuantizeWeights_gpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,
      const int rounding, const bool bias_term = true);
  /**
   * @brief Trim data to fixed point.
   * @param fl The number of bits in the fractional part.
   */
  void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  void Trim2FixedPoint_gpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  /**
   * @brief Trim data to minifloat.
   * @param bw_mant The number of bits used to represent the mantissa.
   * @param bw_exp The number of bits used to represent the exponent.
   */
  void Trim2MiniFloat_cpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  void Trim2MiniFloat_gpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  /**
   * @brief Trim data to integer-power-of-two numbers.
   * @param min_exp The smallest quantized value is 2^min_exp.
   * @param min_exp The largest quantized value is 2^max_exp.
   */
  void Trim2IntegerPowerOf2_cpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  void Trim2IntegerPowerOf2_gpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  /**
   * @brief Generate random number in [0,1) range.
   */
  inline double RandUniform_cpu();
  // The number of bits used for dynamic fixed point parameters and layer
  // activations.
  int bw_params_, bw_layer_in_, bw_layer_out_;
  // The fractional length of dynamic fixed point numbers.
  int fl_params_, fl_layer_in_, fl_layer_out_;
  // The number of bits used to represent mantissa and exponent of minifloat
  // numbers.
  int fp_mant_, fp_exp_;
  // Integer-power-of-two numbers are in range +/- [2^min_exp, 2^max_exp].
  int pow_2_min_exp_, pow_2_max_exp_;
  // The rounding mode for quantization and the quantization scheme.
  int rounding_, precision_;
  // For parameter layers: reduced word with parameters.
  vector<shared_ptr<Blob<Dtype> > > weights_quantized_;
};

/**
 * @brief Convolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class ConvolutionRistrettoLayer : public ConvolutionLayer<Dtype>,
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit ConvolutionRistrettoLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "ConvolutionRistretto"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Deconvolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class DeconvolutionRistrettoLayer : public DeconvolutionLayer<Dtype>,
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit DeconvolutionRistrettoLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "DeconvolutionRistretto"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Inner product (fully connected) layer with quantized layer parameters
 * and activations.
 */
template <typename Dtype>
class FcRistrettoLayer : public InnerProductLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit FcRistrettoLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FcRistretto"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Local response normalization (LRN) layer with minifloat layer inputs,
 * intermediate results and outputs.
 */
template <typename Dtype>
class LRNRistrettoLayer : public LRNLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit LRNRistrettoLayer(const LayerParameter& param);
  //virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  //			  const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "LRNRistretto"; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelForwardFixedPoint_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_input_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_bottom_vec_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<PoolingLayer<Dtype> > pool_layer_;
  Blob<Dtype> pool_output_;
  vector<Blob<Dtype>*> pool_top_vec_;
  shared_ptr<PowerLayer<Dtype> > power_layer_;
  Blob<Dtype> power_output_;
  vector<Blob<Dtype>*> power_top_vec_;
  shared_ptr<EltwiseLayer<Dtype> > product_layer_;
  Blob<Dtype> product_input_;
  vector<Blob<Dtype>*> product_bottom_vec_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_RISTRETTO_LAYER_HPP_
