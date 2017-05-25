#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

#include "quantization/quantization.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "The trained weights.");
DEFINE_string(trimming_mode, "",
    "Available options: dynamic_fixed_point, minifloat or "
    "integer_power_of_2_weights.");
DEFINE_string(model_quantized, "",
    "The output path of the quantized net");
DEFINE_string(gpu, "",
    "Optional: Run in GPU mode on given device ID.");
DEFINE_int32(iterations, 50,
    "Optional: The number of iterations to run.");
DEFINE_double(error_margin, 2,
    "Optional: the allowed accuracy drop in %");

// for pruning by zhluo
DEFINE_bool(prun_conv, false, "Optional; pruning CONV layers");
DEFINE_bool(prun_fc, false, "Optional; pruning FC layers");
DEFINE_bool(prun_retrain, false, "Optional; retrain net after pruning");
DEFINE_bool(sparse_csc, false, "Optional; blob use CSC sparse storage");
DEFINE_int32(sparse_col, 1, "Optional; sparse column num");
DEFINE_int32(prun_fc_num, 0, "Optional; the number of FC layers");
DEFINE_int32(idx_diff_conv, 0, "Optional; conv weight diff between valid weight");
DEFINE_int32(idx_diff_fc  , 0, "Optional; fc weight diff between valid weight");
DEFINE_double(conv_ratio_0, 0, "Optional; conv layer_0 prun ratio");
DEFINE_double(conv_ratio_1, 0, "Optional; conv layer_1 prun ratio");
DEFINE_double(conv_ratio_2, 0, "Optional; conv layer_2 prun ratio");
DEFINE_double(conv_ratio_3, 0, "Optional; conv layer_3 prun ratio");
DEFINE_double(conv_ratio_4, 0, "Optional; conv layer_4 prun ratio");
DEFINE_double(conv_ratio_5, 0, "Optional; conv layer_5 prun ratio");
DEFINE_double(conv_ratio_6, 0, "Optional; conv layer_6 prun ratio");
DEFINE_double(conv_ratio_7, 0, "Optional; conv layer_7 prun ratio");
DEFINE_double(fc_ratio_0, 0, "Optional; fc layer_0 prun ratio");
DEFINE_double(fc_ratio_1, 0, "Optional; fc layer_1 prun ratio");
DEFINE_double(fc_ratio_2, 0, "Optional; fc layer_2 prun ratio");
DEFINE_double(fc_ratio_3, 0, "Optional; fc layer_3 prun ratio");
DEFINE_int32(quan_enable, 0, "Optional; enable quantization");
DEFINE_double(quan_lr, 0, "Optional; get SolverParameter learn rate");
DEFINE_int32(quan_k_min, 1, "Optional; min 2^k clusters");
DEFINE_int32(quan_k_max, 8, "Optional; max 2^k clusters");
DEFINE_int32(quan_max_iter, 256, "Optional; k-mean max iteration num");
DEFINE_bool(quan_retrain, false, "Optional; fine-tune quantization data");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available ristretto actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// ristretto commands to call by
//     ristretto <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Quantize a 32-bit FP network to smaller word width.
int quantize(){
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  CHECK_GT(FLAGS_model_quantized.size(), 0) << "Need network description "
      "output path.";
  CHECK_GT(FLAGS_trimming_mode.size(), 0) << "Need trimming mode.";
  Quantization* q = new Quantization(FLAGS_model, FLAGS_weights,
      FLAGS_model_quantized, FLAGS_iterations, FLAGS_trimming_mode,
      FLAGS_error_margin, FLAGS_gpu);
  q->QuantizeNet();
  delete q;
  return 0;
}
RegisterBrewFunction(quantize);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
			  "usage: ristretto <command> <args>\n\n"
			  "commands:\n"
			  "  quantize\n"
			  "  --model=prototxt\n"
			  "  --weights=caffemodel\n"
			  "  --model_quantized= out quantization model\n"
			  "  --iterations=iter num\n"
			  "  --gpu=GPU ID\n"
			  "  --trimming_mode=dynamic_fixed_point or minifloat or integer_power_of_2_weights\n"
			  "  --error_margin=margin num\n");
  //      "  quantize        Trim 32bit floating point net\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
      return GetBrewFunction(caffe::string(argv[1]))();
  } else {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/ristretto");
  }
}
