#pragma once

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
      //TODO: assert(close(fd) == 0);
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
