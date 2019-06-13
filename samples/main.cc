
#include <string>
#include <string.h>
#include <sstream>
#include <stdint.h>



#include <assert.h>
#include <stdexcept>
#include <setjmp.h>
#include <vector>
#include <memory>
#include "providers.h"
#include "local_filesystem.h"

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include "ImageLoader.h"
#include "Callback.h"
#include <fstream>

using TCharString = std::basic_string<ORTCHAR_T>;



#define ORT_ABORT_ON_ERROR(expr)                         \
  do {                                                   \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      abort();                                           \
    }                                                    \
  } while (0);

// Decompresses a JPEG file from disk.
OrtValue* LoadJpegFile(const TCharString* file_name_begin,const TCharString* file_name_end,int out_height, int out_width,
    const OrtAllocatorInfo* allocator_info,  Callback& c) {
  const int channels = 3;
  size_t output_data_len = out_width * out_height * channels;
  size_t batch_size = file_name_end - file_name_begin;
  float* output_data = new float[output_data_len*batch_size];
  for(size_t i=0;i!=batch_size;++i) {
    LoadJpegFileIntoMemory(file_name_begin[i], out_height, out_width, output_data + output_data_len * i, output_data_len);
  }
  OrtValue* input_tensor = nullptr;
  std::vector<int64_t> input_shape(4);
  input_shape[0] = batch_size;
  input_shape[1] = out_height;
  input_shape[2] = out_width;
  input_shape[3] = channels;

  ORT_ABORT_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(allocator_info, output_data, batch_size * output_data_len * sizeof(float), input_shape.data(),
                                                       input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                       &input_tensor));
  c.param = output_data;
  c.f = [](void* p){ delete[] (float*)p;};
  return input_tensor;
}



static std::vector<std::string> readFileToVec(const std::string& file_path, size_t expected_line_count) {
  std::ifstream ifs(file_path);
  if (!ifs) {
    throw std::runtime_error("open file failed");
  }
  std::string line;
  std::vector<std::string> labels;
  while (std::getline(ifs, line)) {
    if (!line.empty()) labels.push_back(line);
  }
  if (labels.size() != expected_line_count) {
    throw std::runtime_error("line count mismatch");
  }
  return labels;
}

static int ExtractImageNumberFromFileName(const TCharString &image_file) {
  size_t s = image_file.rfind('.');
  if (s == std::string::npos) throw std::runtime_error("illegal filename");
  size_t s2 = image_file.rfind('_');
  if (s2 == std::string::npos) throw std::runtime_error("illegal filename");

  const char* start_ptr = image_file.c_str() + s2 + 1;
  const char* endptr = nullptr;
  long value = strtol(start_ptr, (char**)&endptr, 10);
  if (start_ptr == endptr || value > INT32_MAX || value <= 0) throw std::runtime_error("illegal filename");
  return static_cast<int>(value);
}

// TODO: calc top 1 acccury
int main(int argc, ORTCHAR_T* argv[]) {
  if (argc < 2) return -1;
  std::vector<TCharString> image_file_paths;
  TCharString data_dir = argv[1];
  TCharString model_path = argv[2];
  // imagenet_lsvrc_2015_synsets.txt
  TCharString label_file_path = argv[3];
  TCharString validation_file_path = argv[4];
  std::vector<std::string> labels = readFileToVec(label_file_path, 1000);
  // TODO: remove the slash at the end of data_dir string
  LoopDir(data_dir, [&data_dir, &image_file_paths](const ORTCHAR_T* filename, OrtFileType filetype) -> bool {
    if (filetype != OrtFileType::TYPE_REG) return true;
    if (filename[0] == '.') return true;
    const char* p = strrchr(filename, '.');
    if (p == nullptr) return true;
    // as we tested filename[0] is not '.', p should larger than filename
    assert(p > filename);
    if (strcasecmp(p, ".JPEG") != 0) return true;
    TCharString v(data_dir);
#ifdef _WIN32
    v.append(1, '\\');
#else
    v.append(1, '/');
#endif
    v.append(filename);
    image_file_paths.emplace_back(v);
    return true;
  });
  std::vector<std::string> validation_data = readFileToVec(validation_file_path, image_file_paths.size());

  std::vector<uint8_t> data;
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  OrtSessionOptions* session_option;
  ORT_ABORT_ON_ERROR(OrtCreateSessionOptions(&session_option));
#ifdef USE_CUDA
  ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
#endif
  OrtSession* session;
  ORT_ABORT_ON_ERROR(OrtCreateSession(env, model_path.c_str(), session_option, &session));
  OrtAllocatorInfo* allocator_info;
  ORT_ABORT_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
  size_t top_1_correct_count = 0;

  const int image_size = 299;
  const int batch_size = 16;
  for (size_t completed = 0; completed!=image_file_paths.size(); completed+=batch_size) {
    //printf("loading %s\n", s.c_str());
    size_t remain = std::min<size_t>(image_file_paths.size() - completed, batch_size);
    Callback c;
    auto file_names_begin = &image_file_paths[completed];
    OrtValue* input_tensor = LoadJpegFile(file_names_begin, file_names_begin+remain, image_size,image_size,allocator_info, c);
    assert(input_tensor != NULL);
    const char* input_name = "input:0";
    const char* output_name = "InceptionV4/Logits/Predictions:0";
    OrtValue* output_tensor = NULL;
    ORT_ABORT_ON_ERROR(OrtRun(session, NULL, &input_name, &input_tensor, 1, &output_name, 1, &output_tensor));
    float* probs;
    ORT_ABORT_ON_ERROR(OrtGetTensorMutableData(output_tensor, (void**)&probs));
    for(size_t i =0;i!=remain;++i) {
      float max_prob = probs[1];
      int max_prob_index = 1;
      for (int i = max_prob_index + 1; i != 1001; ++i) {
        if (probs[i] > max_prob) {
          max_prob = probs[i];
          max_prob_index = i;
        }
      }
      // TODO:extract number from filename, to index validation_data
      auto s = file_names_begin[i];
      int test_data_id = ExtractImageNumberFromFileName(s);
      //printf("%d\n",(int)max_prob_index);
      //printf("%s\n",labels[max_prob_index - 1].c_str());
      //printf("%s\n",validation_data[test_data_id - 1].c_str());
      if (labels[max_prob_index - 1] == validation_data[test_data_id - 1]) {
        ++top_1_correct_count;
      }
      probs += 1001;
    }
    OrtReleaseValue(input_tensor);
    OrtReleaseValue(output_tensor);
    if(c.f)
      c.f(c.param);
    if((completed) % 160 == 0){
      printf("Top-1 Accuracy: %f\n", ((float)top_1_correct_count/completed));
      printf("finished %f\n", ((float)completed/image_file_paths.size()));
    }
  }
  printf("Top-1 Accuracy %f\n", ((float)top_1_correct_count/image_file_paths.size()));
  OrtReleaseSessionOptions(session_option);
  OrtReleaseSession(session);
  OrtReleaseEnv(env);
  return 0;
}