//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

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

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  LOG(INFO) << "Writing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    string batchFileName = input_folder + "/data_batch_"
      + caffe::format_int(fileid+1) + ".bin";
    std::ifstream data_file(batchFileName.c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
    }
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;

  if (argc != 4) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
