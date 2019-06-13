
#include <string>
#include <string.h>
#include <sstream>
#include <stdint.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/types.h>
#include <dirent.h>
#endif

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <jpeglib.h>
#include <stdexcept>
#include <setjmp.h>
#include <vector>
#include <sys/mman.h>

#include <memory>
#include "jpeg_mem.h"
#include "providers.h"

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include <fstream>

enum class OrtFileType { TYPE_BLK, TYPE_CHR, TYPE_DIR, TYPE_FIFO, TYPE_LNK, TYPE_REG, TYPE_SOCK, TYPE_UNKNOWN };

#ifdef _WIN32
inline OrtFileType DTToFileType(DWORD dwFileAttributes) {
  if (dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    return OrtFileType::TYPE_DIR;
  }
  // TODO: test if it is reg
  return OrtFileType::TYPE_REG;
}

template <typename T>
void LoopDir(const std::wstring& dir_name, T func) {
  std::wstring pattern = dir_name + L"\\*";
  WIN32_FIND_DATAW ffd;
  std::unique_ptr<void, decltype(&FindClose)> hFind(FindFirstFileW(pattern.c_str(), &ffd), FindClose);
  if (hFind.get() == INVALID_HANDLE_VALUE) {
    DWORD dw = GetLastError();
    std::string s = FormatErrorCode(dw);
    throw std::runtime_error(s);
  }
  do {
    if (!func(ffd.cFileName, DTToFileType(ffd.dwFileAttributes))) return;
  } while (FindNextFileW(hFind.get(), &ffd) != 0);
  DWORD dwError = GetLastError();
  if (dwError != ERROR_NO_MORE_FILES) {
    DWORD dw = GetLastError();
    std::string s = FormatErrorCode(dw);
    throw std::runtime_error(s);
  }
}
#else

struct Callback {
  void(ORT_API_CALL* f)(void* param) NO_EXCEPTION;
  void* param;
};

static void ReportSystemError(const char* operation_name, const std::string& path) {
  auto e = errno;
  char buf[1024];
  const char* msg = "";
  if (e > 0) {
#if defined(__GLIBC__) && defined(_GNU_SOURCE) && !defined(__ANDROID__)
    msg = strerror_r(e, buf, sizeof(buf));
#else
    // for Mac OS X and Android lower than API 23
    if (strerror_r(e, buf, sizeof(buf)) != 0) {
      buf[0] = '\0';
    }
    msg = buf;
#endif
  }
  std::ostringstream oss;
  oss << operation_name << " file \"" << path << "\" failed: " << msg;
  throw std::runtime_error(oss.str());
}


class UnmapFileParam {
 public:
  void* addr;
  size_t len;
  int fd;
  std::string file_path;
};

static void UnmapFile(void* param) noexcept {
  std::unique_ptr<UnmapFileParam> p(reinterpret_cast<UnmapFileParam*>(param));
  int ret = munmap(p->addr, p->len);
  if (ret != 0) {
    ReportSystemError("munmap", p->file_path);
    return;
  }
  if(close(p->fd) != 0){
    ReportSystemError("close", p->file_path);
    return;
  }
}

void ReadFileAsString(const char* fname, void*& p, size_t& len, Callback& deleter) {
  if (!fname) {
    throw std::runtime_error("ReadFileAsString: 'fname' cannot be NULL");
  }

  deleter.f = nullptr;
  deleter.param = nullptr;
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    return ReportSystemError("open", fname);
  }
  struct stat stbuf;
  if (fstat(fd, &stbuf) != 0) {
    return ReportSystemError("fstat", fname);
  }

  if (!S_ISREG(stbuf.st_mode)) {
    throw std::runtime_error("ReadFileAsString: input is not a regular file");
  }
  //TODO:check overflow
  len = static_cast<size_t>(stbuf.st_size);

  if (len == 0) {
    p = nullptr;
  } else {
    p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      ReportSystemError("mmap",fname);
    } else {
      // leave the file open
      deleter.f = UnmapFile;
      deleter.param = new UnmapFileParam{p, len, fd, fname};
      p = reinterpret_cast<char*>(p);
    }
  }

  //assert(close(fd) == 0);
}

inline OrtFileType DTToFileType(unsigned char t) {
  switch (t) {
    case DT_BLK:
      return OrtFileType::TYPE_BLK;
    case DT_CHR:
      return OrtFileType::TYPE_CHR;
    case DT_DIR:
      return OrtFileType::TYPE_DIR;
    case DT_FIFO:
      return OrtFileType::TYPE_FIFO;
    case DT_LNK:
      return OrtFileType::TYPE_LNK;
    case DT_REG:
      return OrtFileType::TYPE_REG;
    case DT_SOCK:
      return OrtFileType::TYPE_SOCK;
    default:
      return OrtFileType::TYPE_UNKNOWN;
  }
}

template <typename T>
void LoopDir(const std::string& dir_name, T func) {
  DIR* dir = opendir(dir_name.c_str());
  if (dir == nullptr) {
    auto e = errno;
    char buf[1024];
    char* msg;
#if defined(__GLIBC__) && defined(_GNU_SOURCE) && !defined(__ANDROID__)
    msg = strerror_r(e, buf, sizeof(buf));
#else
    if (strerror_r(e, buf, sizeof(buf)) != 0) {
      buf[0] = '\0';
    }
    msg = buf;
#endif
    std::ostringstream oss;
    oss << "couldn't open '" << dir_name << "':" << msg;
    std::string s = oss.str();
    throw std::runtime_error(s);
  }
  try {
    struct dirent* dp;
    while ((dp = readdir(dir)) != nullptr) {
      if (!func(dp->d_name, DTToFileType(dp->d_type))) {
        break;
      }
    }
  } catch (std::exception& ex) {
    closedir(dir);
    throw;
  }
  closedir(dir);
}
#endif

using TCharString = std::basic_string<ORTCHAR_T>;

// Error handling for JPEG decoding.
void CatchError(j_common_ptr cinfo);

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


// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64 lower;  // Lower source index used in the interpolation
  int64 upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

inline void compute_interpolation_weights(const int64 out_size,
                                          const int64 in_size,
                                          const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64 i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int64>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}


void resize_image(const float* input_b_ptr,
                  const int batch_size, const int64 in_height,
                  const int64 in_width, const int64 out_height,
                  const int64 out_width, const int channels,
                  const std::vector<CachedInterpolation>& xs_vec,
                  const std::vector<CachedInterpolation>& ys,
                  float* output_y_ptr) {
  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const CachedInterpolation* xs = xs_vec.data();

  if (channels == 3) {
    for (int b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const float* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const float* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64 x = 0; x < out_width; ++x) {
          const int64 xs_lower = xs[x].lower;
          const int64 xs_upper = xs[x].upper;
          const float xs_lerp = xs[x].lerp;

          // Read channel 0.
          const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
          const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
          const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
          const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

          // Read channel 1.
          const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
          const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
          const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
          const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

          // Read channel 2.
          const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
          const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
          const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
          const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

          // Compute output.
          output_y_ptr[x * channels + 0] =
              compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
                           xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 1] =
              compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
                           xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 2] =
              compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2,
                           xs_lerp, ys_lerp);
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  } else {
    for (int b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const float* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const float* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64 x = 0; x < out_width; ++x) {
          auto xs_lower = xs[x].lower;
          auto xs_upper = xs[x].upper;
          auto xs_lerp = xs[x].lerp;
          for (int c = 0; c < channels; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            output_y_ptr[x * channels + c] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right,
                             xs_lerp, ys_lerp);
          }
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  }
}


/**
 * CalculateResizeScale determines the float scaling factor.
 * @param in_size
 * @param out_size
 * @param align_corners If true, the centers of the 4 corner pixels of the input and output tensors are aligned,
 *                        preserving the values at the corner pixels
 * @return
 */
inline float CalculateResizeScale(int64 in_size, int64 out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}

void LoadJpegFileIntoMemory(const TCharString& file_name,int out_height, int out_width,float* output_data,size_t output_data_len){

  int width;
  int height;
  int channels;

  UncompressFlags flags;
  flags.components = 3;
  // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
  // image quality for speed.
  flags.dct_method = JDCT_IFAST;
  size_t file_len;
  void* file_data;
  Callback c;
  ReadFileAsString(file_name.c_str(),file_data,file_len, c);
  file_data = Uncompress(file_data,file_len,flags,&width,&height,&channels,nullptr);
  if(c.f)
    c.f(c.param);

  if (channels != 3) {
    std::ostringstream oss;
    oss << "input format error, expect 3 channels, got " << channels;
    throw std::runtime_error(oss.str());
  }

  std::vector<float> float_file_data(height * width*channels);
  for(size_t i=0;i!=float_file_data.size();++i){
    float_file_data[i] = static_cast<float>(((uint8_t*) file_data)[i])/255;
  }

  int in_height = height;
  int in_width = width;

  float height_scale = CalculateResizeScale(in_height, out_height, false);
  float width_scale = CalculateResizeScale(in_width, out_width, false);


  std::vector<CachedInterpolation> ys(out_height + 1);
  std::vector<CachedInterpolation> xs(out_width + 1);

  // Compute the cached interpolation weights on the x and y dimensions.
  compute_interpolation_weights(out_height, in_height, height_scale,
                                ys.data());
  compute_interpolation_weights(out_width, in_width, width_scale, xs.data());


  // Scale x interpolation weights to avoid a multiplication during iteration.
  for (int i = 0; i < xs.size(); ++i) {
    xs[i].lower *= channels;
    xs[i].upper *= channels;
  }


  resize_image(float_file_data.data(),1,in_height,in_width,out_height,out_width,channels,xs,ys,output_data);
  for(size_t i =0;i!=output_data_len;++i){
    output_data[i] = (output_data[i] - 0.5) * 2;
  }
}
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

int ExtractImageNumber(const TCharString& image_file) {
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
      int test_data_id = ExtractImageNumber(s);
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