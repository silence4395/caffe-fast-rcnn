#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "quantization/base_ristretto_layer.hpp"
#include "quantization/base_ristretto_layer.cuh"

namespace caffe {

// Device function wrapper for quantization to minifloat numbers.
template <typename Dtype>
__device__ Dtype
toFP(Dtype data, const int mant, const int exp, const int index){
  Trim2MiniFloat_device(&data, mant, exp,
      QuantizationParameter_Rounding_NEAREST, index);
  return data;
}

// Same as LRNFillScale, but all intermediate results are quantized to
// minifloat.
template <typename Dtype>
__global__ void LRNFillScaleQ(const int nthreads, const Dtype* const in,
      const int num, const int channels, const int height, const int width,
      const int size, const Dtype alpha_over_size, const Dtype k,
      Dtype* const scale, const int mant, const int exp) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      Dtype in_off_q = toFP(in_off[head * step], mant, exp, index);
      accum_scale += toFP(in_off_q * in_off_q, mant, exp, index);
      accum_scale = toFP(accum_scale, mant, exp, index);
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      Dtype in_off_q = toFP(in_off[head * step], mant, exp, index);
      accum_scale += toFP(in_off_q * in_off_q, mant, exp, index);
      accum_scale = toFP(accum_scale, mant, exp, index);
      if (head - size >= 0) {
        Dtype in_off_q = toFP(in_off[(head - size) * step], mant, exp, index);
        accum_scale -= toFP(in_off_q * in_off_q, mant, exp, index);
        accum_scale = toFP(accum_scale, mant, exp, index);
      }
      Dtype tmp = toFP(accum_scale * alpha_over_size, mant, exp, index);
      tmp = toFP(k + tmp, mant, exp, index);
      scale_off[(head - post_pad) * step] = tmp;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        Dtype in_off_q = toFP(in_off[(head - size) * step], mant, exp, index);
        accum_scale -= toFP(in_off_q * in_off_q, mant, exp, index);
        accum_scale = toFP(accum_scale, mant, exp, index);
      }
      Dtype tmp = toFP(accum_scale * alpha_over_size, mant, exp, index);
      tmp = toFP(k + tmp, mant, exp, index);
      scale_off[(head - post_pad) * step] = tmp;
      ++head;
    }
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   // Trim layer input
  if (this->phase_ == TEST) {
    for (int i = 0; i < bottom.size(); ++i) {
      this->QuantizeLayerInputs_gpu(bottom[i]->mutable_gpu_data(),
          bottom[i]->count());
    }
  } 
  
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    if (this->precision_ == QuantizationParameter_Precision_MINIFLOAT) 
        CrossChannelForward_gpu(bottom, top);
    else if (this->precision_ == QuantizationParameter_Precision_DYNAMIC_FIXED_POINT)
        CrossChannelForwardFixedPoint_gpu(bottom, top);
    else
	LOG(FATAL) << "Unknow trimming mode: " << this->precision_;
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
  WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
  
  // Trim layer output
    if (this->phase_ == TEST) {
      this->QuantizeLayerOutputs_gpu(top[0]->mutable_gpu_data(), top[0]->count());
    }
}

// Same as LRNComputeOutput, but all intermediate results are quantized to
// minifloat
// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutputQ(const int nthreads, const Dtype* const in,
      const Dtype* const scale, const Dtype negative_beta, Dtype* const out,
      const int mant, const int exp) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype in_q = toFP(in[index], mant, exp, index);
    Dtype pow_q = toFP(pow(scale[index], negative_beta), mant, exp, index);
    out[index] = toFP(in_q * pow_q, mant, exp, index);
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::CrossChannelForward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = this->scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = this->num_ * this->height_ * this->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScaleQ<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, this->num_, this->channels_, this->height_,
      this->width_, this->size_, this->alpha_ / this->size_, this->k_,
      scale_data, this->fp_mant_, this->fp_exp_);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutputQ<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -this->beta_, top_data,
      this->fp_mant_, this->fp_exp_);
  CUDA_POST_KERNEL_CHECK;
}
template void LRNRistrettoLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNRistrettoLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

template <typename Dtype>
__device__ void FixedPointSigmoidQuan(Dtype* data, const int bit_width, const int fl, const int rounding) {
        Dtype max_data = (powf(2, bit_width - 1) - 1) * powf(2, -fl);
        Dtype min_data = -powf(2, bit_width - 1) * powf(2, -fl);
        *data = fmax(fmin(*data, max_data), min_data);
        // Round data
        *data /= powf(2, -fl);
        switch (rounding) {
        case 0: // NEAREST
          *data = rint(*data);
          break;
        case 1: // STOCHASTIC
          *data = __float2int_rd(*data + RandUniform_device(0));
          break;
        default:
          break;
        }
        *data *= powf(2, -fl);
}

template <typename Dtype>
__device__ void FixedPointQuan(Dtype* data, const int cnt,
	                       const int bit_width, const int fl, const int rounding) {
    CUDA_KERNEL_LOOP(index, cnt) {
        Dtype max_data = (powf(2, bit_width - 1) - 1) * powf(2, -fl);
        Dtype min_data = -powf(2, bit_width - 1) * powf(2, -fl);
        data[index] = fmax(fmin(data[index], max_data), min_data);
        // Round data
        data[index] /= powf(2, -fl);
        switch (rounding) {
        case 0: // NEAREST
          data[index] = rint(data[index]);
          break;
        case 1: // STOCHASTIC
          data[index] = __float2int_rd(data[index] + RandUniform_device(index));
          break;
        default:
          break;
        }
        data[index] *= powf(2, -fl);	
    }
}

// add by zhluo, 9/2/2017
template <typename Dtype>
__global__ void AREASApproximateCompute(const int n_threads, const Dtype* const bottom,
	                                const Dtype* in, Dtype* out, bool fixed_point,
					 int bit_width, int fl) {
         float variable[22] = {1.25, 1.5, 1.75, 2 , 3 , 4 , 5  , 6  , 7  , 8  ,
			       9   , 10 , 20  , 30, 40, 50, 100, 150, 200, 300,
			       400, 500};
	 float coefficient[22] = {-0.614942702877   , -0.431915810474   , -0.322019267621   , 
	                          -0.250473112162   , -0.155900736925   , -0.0851335094465  ,
				  -0.0544815774316  , -0.038221096796   , -0.0284781347549  ,
				  -0.022144085262   , -0.0177737273373  , -0.0146220568874  ,
	                          -0.00715725449729 , -0.00279313184789 , -0.00150150072144 ,
				  -0.000964051261494, -0.000428103671576, -0.000167068252181,
				  -8.98106909522e-05, -4.96696885463e-05, -2.67008781709e-05,
				  -1.71435250854e-05};
	 float const_b[22] = {1.60975075579  , 1.38285473956  , 1.21898366501  ,
	                      1.09433785502  , 0.897219434112 , 0.690525480745 ,
	                      0.569707973835 , 0.489159227786 , 0.431075802405 ,
	                      0.386945714703 , 0.352108590613 , 0.32382395031  ,
	                      0.241944064196 , 0.160070213173 , 0.122360341268 ,
	                      0.10109474241  , 0.0723581523039, 0.0478721596357,
	                      0.0365942774371, 0.0284649564246, 0.0217590875483,
	                      0.0179774698891};
	 int length = sizeof(coefficient) / sizeof(coefficient[0]);
	 
	 Dtype pass_region = 0.005;
	 Dtype lut_index;
	 if (fixed_point) {  
	    FixedPointSigmoidQuan(&pass_region, bit_width, bit_width-1, 0);
	    FixedPointQuan(variable, length, bit_width, fl, 0);
	    FixedPointQuan(coefficient, length, bit_width, bit_width-1, 0);
	    FixedPointQuan(const_b, length, bit_width, bit_width-1, 0);
	 }
	 
	 CUDA_KERNEL_LOOP(index_d, n_threads) {
	     lut_index = in[index_d];
	     if (fixed_point) {
	     	FixedPointSigmoidQuan(&lut_index, bit_width, fl, 0);
	     }
	     
	     if (lut_index > 500)
	         out[index_d] = pass_region;
	     else {
		 for(int index_v = 0; index_v < length; ++index_v) {
	             if (lut_index < variable[index_v]) {
	                 out[index_d] = coefficient[index_v] * lut_index + const_b[index_v];
	                 break;
	             }
	         }
	     }
	     if (fixed_point)
	         FixedPointSigmoidQuan(&(out[index_d]), bit_width, bit_width-1, 0);
	     out[index_d] = out[index_d] * bottom[index_d];
	 }
}

template <typename Dtype>
__global__ void LUT198ApproximateCompute(const int n_threads, const Dtype* const bottom,
	                                 const Dtype* in, Dtype* out,
					  bool fixed_point, int bit_width, int fl) {
         float variable[198] = {1.015625, 1.03125 , 1.046875, 1.0625  , 1.078125, 1.09375 , 1.109375,
	                        1.125   , 1.140625, 1.15625 , 1.171875, 1.1875  , 1.203125, 1.21875 ,
				1.234375, 1.25	  , 1.265625, 1.28125 , 1.296875, 1.3125  , 1.328125,
				1.34375 , 1.359375, 1.375   , 1.390625, 1.40625 , 1.421875, 1.4375  ,
				1.453125, 1.46875 , 1.484375, 1.5     , 1.515625, 1.53125 , 1.546875,
				1.5625  , 1.578125, 1.59375 , 1.609375, 1.625   , 1.640625, 1.65625 ,
				1.671875, 1.6875  , 1.703125, 1.71875 , 1.734375, 1.75	  , 1.765625,
				1.78125 , 1.796875, 1.8125  , 1.828125, 1.84375 , 1.859375, 1.875   ,
				1.890625, 1.90625 , 1.921875, 1.9375  , 1.953125, 1.96875 , 1.984375,
 		       		2.0     , 2.03125 , 2.0625  , 2.09375 , 2.125   , 2.15625 , 2.1875  ,
				2.21875 , 2.25    , 2.28125 , 2.3125  , 2.34375 , 2.375   , 2.40625 ,
				2.4375	, 2.46875 , 2.5     , 2.53125 , 2.5625  , 2.59375 , 2.625   ,
			        2.65625 , 2.6875  , 2.71875 , 2.75    , 2.78125 , 2.8125  , 2.84375 ,
				2.875	, 2.90625 , 2.9375  , 2.96875 , 3.0     , 3.03125 , 3.0625  ,
				3.09375 , 3.125   , 3.15625 , 3.1875  , 3.21875 , 3.25    , 3.28125 ,
				3.3125  , 3.34375 , 3.375   , 3.40625 , 3.4375	, 3.46875 , 3.5     ,
				3.53125 , 3.5625  , 3.59375 , 3.625   , 3.65625 , 3.6875  , 3.71875 ,
				3.75    , 3.78125 , 3.8125  , 3.84375 , 3.875   , 3.90625 , 3.9375  ,
 		       		3.96875 , 4.0	  , 4.125   , 4.25    , 4.375   , 4.5	  , 4.625   ,
				4.75    , 4.875   , 5.0     , 5.125   , 5.25    , 5.375   , 5.5     ,
				5.625   , 5.75    , 5.875   , 6.0     , 6.125   , 6.25	  , 6.375   ,
				6.5     , 6.625   , 6.75    , 6.875   , 7.0     , 7.125   , 7.25    ,
				7.375	, 7.5     , 7.625   , 7.75    , 7.875   , 8.0     , 8.5     ,
				9.0     , 9.5     , 10.0    , 10.5    , 11.0    , 11.5    , 12.0    ,
				12.5	, 13.0    , 13.5    , 14.0    , 14.5    , 15.0    , 15.5    ,
				16.0    , 18.0	  , 20.0    , 22.0    , 24.0    , 26.0    , 28.0    ,
				30.0    , 32.0    , 36.0    , 40.0    , 44.0    , 48.0    , 52.0    ,
				56.0    , 60.0    , 64.0    , 80.0    , 96.0    , 112.0	  , 128.0   ,
				192.0   , 256};
 
         float values[198] = {0.9942065333963459, 0.9827998414527859, 0.9716939265934808 , 0.9608765102608388 ,
          		      0.9503359874419031, 0.94006138054108   , 0.9300422970227042 , 0.9202688904668247,
          		      0.9107318247197804, 0.9014222408547905 , 0.8923317266874584 , 0.883452288617333 ,
          		      0.8747763255898964, 0.8662966049939489 , 0.8580062403276674 , 0.8498986704828934,
          		      0.8419676405117158, 0.8342071837523677 , 0.8266116052030255 , 0.8191754660424586,
          		      0.8118935692057623, 0.8047609459317389 , 0.7977728432059796 , 0.7909247120304375,
          		      0.7842121964563522, 0.7776311233228648 , 0.7711774926486059 , 0.7648474686280107,
          		      0.7586373711881655, 0.7525436680656609 , 0.7465629673662475 , 0.7406920105731243,
          		      0.7349276659724292, 0.7292669224670087 , 0.723706883751821  , 0.7182447628264019,
          		      0.7128778768217295, 0.7076036421205434 , 0.7024195697517719 , 0.697323261041165 ,
          		      0.6923124035015621, 0.6873847669474409 , 0.6825381998195154 , 0.6777706257061765,
          		      0.6730800400495128, 0.6684645070245199 , 0.6639221565809041 , 0.6594511816376277,
          		      0.6550498354210164, 0.6507164289378845 , 0.6464493285757057 , 0.6422469538223973,
          		      0.6381077750987793, 0.6340303116972268 , 0.6300131298204615 , 0.6260548407148175,
          		      0.6221540988926849, 0.6183096004391694 , 0.6145200813983247 , 0.6107843162346012,
          		      0.6071011163654301, 0.6034693287611125 , 0.5998878346084175 , 0.5963555480345142,
          		      0.5911548559969042, 0.5843725573634406 , 0.5777690928954383 , 0.5713371623411745,
          		      0.5650698658893908, 0.5589606767467984 , 0.5530034159566071 , 0.5471922292460922,
          		      0.5415215657139216, 0.5359861581879586 , 0.5305810051019025 , 0.5253013537547238,
          		      0.5201426848306566, 0.5151006980697612 , 0.5101712989899433 , 0.5053505865709982,
          		      0.5006348418198747, 0.49602051714404793, 0.49150422646677017, 0.4870827360241300,
          		      0.4827529557893637, 0.4785119314748195 , 0.4743568370664263 , 0.4702849678495203,
          		      0.4662937338884970, 0.46238065392600736, 0.4585433496703581 , 0.4547795404424364,
          		      0.4510870381558817, 0.447463742606415  , 0.4439076370482061 , 0.4404167840369663,
          		      0.4369893215210794, 0.4336234591635775 , 0.43031747487911765, 0.4270697115713567,
          		      0.4238785740572444, 0.4207425261657882 , 0.4176600879997844 , 0.4146298333498747,
          		      0.4116503872510736, 0.40872042367264294, 0.405838663332847  , 0.4030038716307417,
          		      0.4002148566877026, 0.3974704674919237 , 0.394769592139585  , 0.3921111561668325,
          		      0.3894941209671141, 0.3869174822887882 , 0.38438026880826975, 0.3818815407742914,
          		      0.3794203887191560, 0.3769959322331274 , 0.37460731879835835, 0.3722537226789900,
          		      0.3699343438642712, 0.36764840706175084, 0.36539516073777845, 0.3631738762027278,
          		      0.3609838467385128, 0.35882438676612194, 0.35669483105102984, 0.3545945339444823,
          		      0.3494863906904168, 0.34163133342002294, 0.3341762794911689 , 0.3270901221138302,
          		      0.3203449627770323, 0.31391570330181484, 0.307779699054627  , 0.3019164628299127,
          		      0.2963074109258216, 0.29093564452567355, 0.2857857607582184 , 0.2808436888157173,
          		      0.2760965473163687, 0.27153251974928294, 0.2671407453688733 , 0.2629112233364827,
          		      0.2588347282600416, 0.2549027355729341 , 0.2511073554331695 , 0.2474412740230017,
          		      0.2438977012949073, 0.24047032434842042, 0.23715326573859646, 0.2339410461147744,
          		      0.230828550670997 , 0.22781099895953155, 0.22488391767850202, 0.2220431160954325,
          		      0.2192846638119274, 0.21660487061194744, 0.2140002681681466 , 0.2114675934083249,
          		      0.2054705280934311, 0.19659567696348093, 0.1885668238622485 , 0.1812637531496289,
          		      0.1745884047270054, 0.16845994307723308, 0.16281109559430582, 0.1575853930126261,
          		      0.1527350617150679, 0.1482193942538512 , 0.1440034755294256 , 0.1400571768194182,
          		      0.1363543538650655, 0.13287220207413075, 0.12959073389005593, 0.1264923520196497,
          		      0.1195349966730957, 0.10995128763635523, 0.10198888353645309, 0.0952544456677640,
          		      0.0894742948803212, 0.08445173707615726, 0.08004153386779624, 0.0761339568761960,
          		      0.0710757235217595, 0.06537727168098752, 0.06064283525558824, 0.0566385406324998,
          		      0.0532016612090592, 0.05021524437727359, 0.04759293238224154, 0.0452694813187894,
          		      0.0405677174993462, 0.03486796140116503, 0.03074602986436756, 0.0276081607120381,
          		      0.0224271604857236, 0.01734880445246167};
         int length = sizeof(values) / sizeof(values[0]);
	 
	 Dtype pass_region = 0.01;
	 Dtype lut_index;
	 if (fixed_point) {
             FixedPointSigmoidQuan(&pass_region, bit_width, bit_width-1, 0);
	     FixedPointQuan(variable, length, bit_width, fl, 0);
	     FixedPointQuan(values, length, bit_width, bit_width-1, 0);
	 }
	     
	 CUDA_KERNEL_LOOP(index_d, n_threads) {
	     lut_index = in[index_d];
	     if (fixed_point) {
	     	FixedPointSigmoidQuan(&lut_index, bit_width, fl, 0);
	     }
	     
	     if (lut_index > 256)
	         out[index_d] = pass_region;
	     else {
	         for(int index_v = 0; index_v < length; ++index_v) {
		     if (lut_index < variable[index_v]) {
		         out[index_d] = values[index_v];
			 break;
                     }
	         }
	     }
	     if (fixed_point)
	         FixedPointSigmoidQuan(&(out[index_d]), bit_width, bit_width-1, 0);
		 
	     out[index_d] = out[index_d] * bottom[index_d];
	 }
}

template <typename Dtype>
__global__ void LUT400ApproximateCompute(const int n_threads, const Dtype* const bottom,
	                                 const Dtype* in, Dtype* out,
					  bool fixed_point, int bit_width, int fl) {
         float variable[400] = {1.0078125, 1.015625, 1.0234375, 1.03125, 1.0390625, 1.046875, 1.0546875, 1.0625, 
	                        1.0703125, 1.078125, 1.0859375, 1.09375, 1.1015625, 1.109375, 1.1171875, 1.125 ,
				1.1328125, 1.140625, 1.1484375, 1.15625, 1.1640625, 1.171875, 1.1796875, 1.1875,
				1.1953125, 1.203125, 1.2109375, 1.21875, 1.2265625, 1.234375, 1.2421875, 1.25  ,
				1.2578125, 1.265625, 1.2734375, 1.28125, 1.2890625, 1.296875, 1.3046875, 1.3125,
				1.3203125, 1.328125, 1.3359375, 1.34375, 1.3515625, 1.359375, 1.3671875, 1.375 ,
				1.3828125, 1.390625, 1.3984375, 1.40625, 1.4140625, 1.421875, 1.4296875, 1.4375,
				1.4453125, 1.453125, 1.4609375, 1.46875, 1.4765625, 1.484375, 1.4921875, 1.5   ,
				1.5078125, 1.515625, 1.5234375, 1.53125, 1.5390625, 1.546875, 1.5546875, 1.5625,
				1.5703125, 1.578125, 1.5859375, 1.59375, 1.6015625, 1.609375, 1.6171875, 1.625 ,
				1.6328125, 1.640625, 1.6484375, 1.65625, 1.6640625, 1.671875, 1.6796875, 1.6875,
				1.6953125, 1.703125, 1.7109375, 1.71875, 1.7265625, 1.734375, 1.7421875, 1.75  , 
				1.7578125, 1.765625, 1.7734375, 1.78125, 1.7890625, 1.796875, 1.8046875, 1.8125,
				1.8203125, 1.828125, 1.8359375, 1.84375, 1.8515625, 1.859375, 1.8671875, 1.875 ,
				1.8828125, 1.890625, 1.8984375, 1.90625, 1.9140625, 1.921875, 1.9296875, 1.9375,
				1.9453125, 1.953125, 1.9609375, 1.96875, 1.9765625, 1.984375, 1.9921875, 2.0   ,
				2.015625 , 2.03125 , 2.046875 , 2.0625 , 2.078125 , 2.09375 , 2.109375 , 2.125 ,
				2.140625 , 2.15625 , 2.171875 , 2.1875 , 2.203125 , 2.21875 , 2.234375 , 2.25  ,
				2.265625 , 2.28125 , 2.296875 , 2.3125 , 2.328125 , 2.34375 , 2.359375 , 2.375 ,
				2.390625 , 2.40625 , 2.421875 , 2.4375 , 2.453125 , 2.46875 , 2.484375 , 2.5   ,
				2.515625 , 2.53125 , 2.546875 , 2.5625 , 2.578125 , 2.59375 , 2.609375 , 2.625 ,
				2.640625 , 2.65625 , 2.671875 , 2.6875 , 2.703125 , 2.71875 , 2.734375 , 2.75  ,
				2.765625 , 2.78125 , 2.796875 , 2.8125 , 2.828125 , 2.84375 , 2.859375 , 2.875 ,
				2.890625 , 2.90625 , 2.921875 , 2.9375 , 2.953125 , 2.96875 , 2.984375 , 3.0   ,
				3.015625 , 3.03125 , 3.046875 , 3.0625 , 3.078125 , 3.09375 , 3.109375 , 3.125 ,
				3.140625 , 3.15625 , 3.171875 , 3.1875 , 3.203125 , 3.21875 , 3.234375 , 3.25  ,
				3.265625 , 3.28125 , 3.296875 , 3.3125 , 3.328125 , 3.34375 , 3.359375 , 3.375 ,
				3.390625 , 3.40625 , 3.421875 , 3.4375 , 3.453125 , 3.46875 , 3.484375 , 3.5   ,
				3.515625 , 3.53125 , 3.546875 , 3.5625 , 3.578125 , 3.59375 , 3.609375 , 3.625 ,
				3.640625 , 3.65625 , 3.671875 , 3.6875 , 3.703125 , 3.71875 , 3.734375 , 3.75  ,
				3.765625 , 3.78125 , 3.796875 , 3.8125 , 3.828125 , 3.84375 , 3.859375 , 3.875 ,
				3.890625 , 3.90625 , 3.921875 , 3.9375 , 3.953125 , 3.96875 , 3.984375 , 4.0   ,
				4.0625   , 4.125   , 4.1875   , 4.25   , 4.3125   , 4.375   , 4.4375   , 4.5   ,
				4.5625   , 4.625   , 4.6875   , 4.75   , 4.8125   , 4.875   , 4.9375   , 5.0   ,
				5.0625   , 5.125   , 5.1875   , 5.25   , 5.3125   , 5.375   , 5.4375   , 5.5   ,
				5.5625   , 5.625   , 5.6875   , 5.75   , 5.8125   , 5.875   , 5.9375   , 6.0   ,
				6.0625   , 6.125   , 6.1875   , 6.25   , 6.3125   , 6.375   , 6.4375   , 6.5   ,
				6.5625   , 6.625   , 6.6875   , 6.75   , 6.8125   , 6.875   , 6.9375   , 7.0   ,
				7.0625   , 7.125   , 7.1875   , 7.25   , 7.3125   , 7.375   , 7.4375   , 7.5   ,
				7.5625   , 7.625   , 7.6875   , 7.75   , 7.8125   , 7.875   , 7.9375   , 8.0   ,
				8.25     , 8.5     , 8.75     , 9.0    , 9.25     , 9.5     , 9.75     , 10.0  ,
				10.25    , 10.5    , 10.75    , 11.0   , 11.25    , 11.5    , 11.75    , 12.0  ,
				12.25    , 12.5    , 12.75    , 13.0   , 13.25    , 13.5    , 13.75    , 14.0  ,
				14.25    , 14.5    , 14.75    , 15.0   , 15.25    , 15.5    , 15.75    , 16.0  ,
				17.0     , 18.0    , 19.0     , 20.0   , 21.0     , 22.0    , 23.0     , 24.0  ,
				25.0     , 26.0    , 27.0     , 28.0   , 29.0     , 30.0    , 31.0     , 32.0  ,
				34.0     , 36.0    , 38.0     , 40.0   , 42.0     , 44.0    , 46.0     , 48.0  ,
				50.0     , 52.0    , 54.0     , 56.0   , 58.0     , 60.0    , 62.0     , 64.0  ,
				72.0     , 80.0    , 88.0     , 96.0   , 104.0    , 112.0   , 120.0    , 128.0 ,
				160.0    , 192.0   , 224.0    , 256.0  , 384.0    , 512.0   , 768.0    , 1024};
 	 float values[400] = {0.9970868949715392, 0.9913065674403334, 0.9856038305678392, 0.9799770601249655,
	                      0.9744246778971495, 0.9689451500454997, 0.9635369855380724, 0.9581987346477836,
			      0.9529289875136575, 0.9477263727623018, 0.9425895561866662, 0.9375172394793174,
			      0.9325081590176012, 0.9275610846982196, 0.9226748188188776, 0.9178481950047831,
			      0.9130800771779053, 0.9083693585670023, 0.903714960756542 , 0.8991158327727308,
	       		      0.8945709502049642, 0.8900793143610988, 0.885639951455025 , 0.881251911825104 ,
			      0.8769142691820964, 0.8726261198852896, 0.8683865822455886, 0.8641947958543972,
			      0.8600499209371801, 0.8559511377306428, 0.8518976458825271, 0.8478886638730615,
			      0.8439234284571536, 0.8400011941264622, 0.8361212325905176, 0.832282832276104 ,
			      0.8284852978441568, 0.8247279497234563, 0.821010123660439 , 0.8173311702844744,
	       		      0.8136904546879893, 0.8100873560208459, 0.8065212670984108, 0.802991594022773 ,
			      0.7994977558166008, 0.7960391840691383, 0.7926153225938795, 0.7892256270974636,
			      0.7858695648593679, 0.7825466144219813, 0.7792562652906725, 0.7759980176434698,
			      0.7727713820499982, 0.7695758791993272, 0.766411039636397 , 0.7632764035067133,
			      0.7601715203090024, 0.757095948655538 , 0.754049256039866 , 0.7510310186116522,
			      0.7480408209584084, 0.7450782558938425, 0.7421429242526022, 0.7392344346911872,
			      0.7363524034948107, 0.7334964543900045, 0.7306662183627692, 0.7278613334820739,
			      0.7250814447285265, 0.7223262038280343, 0.7195952690902856, 0.7168883052518921,
			      0.7142049833240295, 0.7115449804444304, 0.708907979733583 , 0.7062936701549943,
			      0.7037017463793859, 0.7011319086526911, 0.6985838626677319, 0.6960573194394508,
	       		      0.69355199518359  , 0.6910676111986984, 0.6886038937513661, 0.6861605739645803,
			      0.6837373877091043, 0.6813340754977827, 0.6789503823826831, 0.6765860578549826,
			      0.6742408557475158, 0.6719145341399002, 0.6696068552661598, 0.66731758542477  ,
			      0.66504649489105	, 0.6627933578318305, 0.6605579522223277, 0.6583400597651576,
			      0.6561394658114258, 0.6539559592838298, 0.6517893326017156, 0.6496393816080286,
	       		      0.6475059054981048, 0.645388706750246 , 0.6432875910580289, 0.6412023672642956,
			      0.6391328472967781, 0.6370788461053082, 0.6350401816005693, 0.633016674594343 ,
			      0.6310081487412094, 0.629014430481661 , 0.6270353489865874, 0.6250707361030946,
			      0.6231204263016187, 0.6211842566243019, 0.619262066634591 , 0.6173536983680288,
			      0.6154589962842006, 0.6135778072198101, 0.6117099803428478, 0.6098553671078262,
	       		      0.60801382121205  , 0.6061851985528972, 0.6043693571860779, 0.6025661572848504,
			      0.6007754611001657, 0.5989971329217161, 0.5972310390398656, 0.5954770477084355,
			      0.5928704331002254, 0.5894334504951046, 0.5860426029805896, 0.5826969248054941,
			      0.5793954775781195, 0.5761373492918312, 0.5729216533923329, 0.5697475278845604,
			      0.5666141344772333, 0.5635206577632164, 0.5604663044339382, 0.557450302526225 ,
	       		      0.5544719006999839, 0.5515303675452679, 0.548624990917326 , 0.5457550772983225,
			      0.5429199511844788, 0.5401189544974558, 0.5373514460188608, 0.5346168008468175,
			      0.5319144098735975, 0.5292436792833614, 0.5266040300691034, 0.5239948975679488,
			      0.5214157310139853, 0.5188659931078621, 0.5163451596024178, 0.5138527189036463,
			      0.511388171686335 , 0.5089510305237464, 0.5065408195307456, 0.5041570740198044,
	       		      0.5017993401693379, 0.4994671747038611, 0.4971601445854695, 0.4948778267161809,
			      0.4926198076506893, 0.4903856833191053, 0.4881750587592811, 0.4859875478583301,
			      0.4838227731029741, 0.4816803653383668, 0.4795599635350561, 0.4774612145637668,
			      0.4753837729776943, 0.4733273008020215, 0.4712914673303737, 0.4692759489279497,
			      0.4672804288410698, 0.4653045970128986, 0.4633481499051075, 0.4614107903252554,
	       		      0.4594922272596711, 0.4575921757116341, 0.4557103565446584, 0.4538464963306891,
			      0.4520003272030345, 0.4501715867138596, 0.4483600176960762, 0.4465653681294719,
			      0.4447873910109244, 0.4430258442285582, 0.4412804904397006, 0.4395510969525063,
			      0.4378374356111191, 0.4361392826842485, 0.4344564187570438, 0.4327886286261484,
			      0.4311357011978293, 0.4294974293890722, 0.4278736100315454, 0.4262640437783325,
	       		      0.4246685350133411, 0.4230868917632999, 0.4215189256122545, 0.4199644516184818,
			      0.4184232882337415, 0.4168952572247887, 0.4153801835970736, 0.4138778955205558,
			      0.4123882242575675, 0.4109110040926557, 0.4094460722643441, 0.4079932688987497,
			      0.4065524369449978, 0.4051234221123767, 0.4037060728091788, 0.4023002400831740,
			      0.4009057775636652, 0.3995225414050763, 0.3981503902320264, 0.3967891850858427,
	       		      0.3954387893724698, 0.3940990688117312, 0.3927698913879041, 0.3914511273015644,
			      0.3901426489226686, 0.3888443307448298, 0.3875560493407576, 0.3862776833188231,
			      0.3850091132807184, 0.3837502217801779, 0.3825008932827283, 0.3812610141264411,
			      0.3800304724836542, 0.3788091583236381, 0.3775969633761774, 0.3763937810960420,
			      0.3751995066283228, 0.3740140367746066, 0.3728372699599678, 0.3716691062007524,
	       		      0.3705094470731324, 0.3693581956824101, 0.3682152566330497, 0.3670805359994182,
			      0.3659539412972130, 0.3648353814555607, 0.3637247667897667, 0.3626220089746969,
			      0.3615270210187778, 0.3604397172385944, 0.3593600132340723, 0.358287825864227 ,
			      0.3572230732234664, 0.3561656746184307, 0.3551155505453550, 0.3540726226679437,
			      0.351501047599111 , 0.3474683404906566, 0.3435419648179993, 0.339717580103008 ,
	       		      0.3359910839478781, 0.3323585957318242, 0.3288164416400537, 0.3253611408990037,
			      0.3219893931053195, 0.3186980665479415, 0.3154841874331529, 0.3123449299317141,
			      0.3092776069754160, 0.3062796617376652, 0.3033486597391775, 0.3004822815256188,
			      0.2976783158691474, 0.2949346534503981, 0.2922492809815332, 0.2896202757346469,
			      0.2870458004430909, 0.2845240985462332, 0.282053489750807 , 0.2796323658843929,
	       		      0.2772591870187132, 0.2749324778423639, 0.2726508242643489, 0.2704128702313651,
			      0.2682173147432201, 0.2660629090520545, 0.2639484540322233, 0.2618727977087574,
			      0.2598348329332958, 0.2578334951972661, 0.2558677605728945, 0.2539366437733620,
			      0.2520391963240944, 0.2501745048377829, 0.2483416893863003, 0.2465399019631799,
			      0.2447683250308034, 0.2430261701468702, 0.2413126766651167, 0.2396271105056173,
	       		      0.2379687629903341, 0.2363369497398869, 0.2347310096277996, 0.2331503037887409,
			      0.2315942146775131, 0.2300621451757691, 0.2285535177436412, 0.2270677736136516,
			      0.2256043720244551, 0.2241627894921203, 0.2227425191168109, 0.2213430699228636,
			      0.2199639662303906, 0.2186047470566539, 0.2172649655455677, 0.2159441884237913,
			      0.2146419954819686, 0.2133579790797603, 0.2120917436733980, 0.2108429053645671,
	       		      0.2078048423303586, 0.2031342780178899, 0.1987015485653473, 0.1944881588807253,
			      0.1904775213545453, 0.1866547133232672, 0.1830062708929740, 0.1795200128864888,
			      0.1761848898750083, 0.1729908541997524, 0.1699287476384079, 0.1669902039691770,
			      0.1641675641652827, 0.1614538023401999, 0.1588424608781684, 0.1563275934407434,
			      0.1539037147499845, 0.1515657562215164, 0.1493090266633356, 0.1471291773745641,
	       		      0.1450221710769107, 0.1429842541939913, 0.1410119320627859, 0.1391019467197150,
			      0.1372512569529804, 0.1354570203544777, 0.1337165771400094, 0.1320274355367203,
			      0.1303872585624934, 0.1287938520441862, 0.1272451537406124, 0.1257392234525796,
			      0.1221729313469085, 0.1168959993414766, 0.1121220841411761, 0.1077797089704276,
			      0.1038105697776270, 0.1001666035819177, 0.0968078114404939, 0.0937006177410328,
	       		      0.0908166170599690, 0.0881316053386410, 0.0856248225150985, 0.0832783544067679,
			      0.0810766559167710, 0.0790061676553881, 0.0770550051970375, 0.0752127053314609,
			      0.0726444025655266, 0.0695067285497665, 0.0666681484675748, 0.0640861623575896,
			      0.0617261027039449, 0.05955939129586248,0.0575622447770267, 0.0557146990892005,
			      0.0539998643595468, 0.0524033488414780, 0.0509128085858269, 0.0495175918017819,
			      0.0482084553563412, 0.0469773368180631, 0.0458171696896610, 0.0447217325291977,
			      0.0422617840760525, 0.0388734891546853, 0.0360583930823247, 0.0336774368895872,
			      0.0316338645405168, 0.0298581366696813, 0.0282989053208302, 0.0269173766725865,
			      0.0241216609483099, 0.0207325862539582, 0.0182816813209909, 0.0164158988436693,
			      0.0133352473242522, 0.0103156522710535, 0.0079291833105098, 0.0061337226885514
 			   };
        int length = sizeof(values) / sizeof(values[0]);
			   
	 Dtype pass_region = 0.005;
	 Dtype lut_index;
	 if (fixed_point) {
             FixedPointSigmoidQuan(&pass_region, bit_width, bit_width-1, 0);
	     FixedPointQuan(variable, length, bit_width, fl, 0);
	     FixedPointQuan(values, length, bit_width, bit_width-1, 0);
	 }
	 
        CUDA_KERNEL_LOOP(index_d, n_threads) {
	     lut_index = in[index_d];
	     if (fixed_point) {
	     	FixedPointSigmoidQuan(&lut_index, bit_width, fl, 0);
	     }
	     
	     if (lut_index > 1024)
	         out[index_d] = pass_region;
	     else {
	         for(int index_v = 0; index_v < length; ++index_v) {
		     if (lut_index < variable[index_v]) {
		         out[index_d] = values[index_v];
			 break;
		     }
	         }
	     }
	     if (fixed_point)
	         FixedPointSigmoidQuan(&(out[index_d]), bit_width, bit_width-1, 0);
	     out[index_d] = out[index_d] * bottom[index_d];
	 }  
}


// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* const in,
    const Dtype* const scale, const Dtype negative_beta, Dtype* const out,
    bool fixed_point, int bit_width, int fl) {
      Dtype scale_tmp ;
      CUDA_KERNEL_LOOP(index, nthreads) {
          scale_tmp = scale[index];
          if (fixed_point)
      	      FixedPointSigmoidQuan(&scale_tmp, bit_width, fl, 0);
	
          out[index] = in[index] * pow(scale_tmp, negative_beta);
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::CrossChannelForwardFixedPoint_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = this->scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = this->num_ * this->height_ * this->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, this->num_, this->channels_, this->height_, this->width_, this->size_,
      this->alpha_ / this->size_, this->k_, scale_data);

  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  //LRNComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
  //    n_threads, bottom_data, scale_data, -beta_, top_data);
  enum op_types {
       POWER = 0,
       AREAS = 1,
       LUT_198 = 2,
       LUT_400 = 3
  } op_type;
  
  op_type = POWER;
  bool fixed_point = 0;
  int bit_width = 8;
  int fl = 2;
  
  switch(op_type) {
     case POWER:
      {
         LRNComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
  	     n_threads, bottom_data, scale_data, -this->beta_, top_data,
	     fixed_point, bit_width, fl);
      	 break;
      }
     case AREAS:
      {
        AREASApproximateCompute<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
          n_threads, bottom_data, scale_data, top_data, fixed_point, bit_width, fl);
	break;
      }
     case LUT_198:
      {
        LUT198ApproximateCompute<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
          n_threads, bottom_data, scale_data, top_data, fixed_point, bit_width, fl);
	break;
      }
     case LUT_400:
      {
        LUT400ApproximateCompute<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
          n_threads, bottom_data, scale_data, top_data, fixed_point, bit_width, fl);
	break;
      }
     default:
         LOG(FATAL) << "[ ERROR ] Please current set Within LRN approximate type.";
     }
  CUDA_POST_KERNEL_CHECK;
}

template void LRNRistrettoLayer<float>::CrossChannelForwardFixedPoint_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNRistrettoLayer<double>::CrossChannelForwardFixedPoint_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    this->CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    this->WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LRNRistrettoLayer);

}  // namespace caffe
