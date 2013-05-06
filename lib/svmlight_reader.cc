#include "svmlight_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <string>
#include <vector>

namespace {

inline int my_getline(FILE *fp, std::string *buff) {
  if (fp == NULL || buff == NULL) {
    return -1;
  }
  buff->clear();

  int c_buff_len = 1024;
  char c_buff[c_buff_len];

  int ret = 0;
  do {
    if (fgets(c_buff, c_buff_len, fp) == NULL) {
      break;
    }
    ret = 1;
    (*buff) += c_buff;
  } while (buff->rfind('\n') == std::string::npos); 

  while (buff->size() > 0) {
    std::string::size_type nline_pos = buff->rfind('\n');
    if (nline_pos == std::string::npos) {
      break;
    }
    buff->erase(nline_pos, buff->size() - nline_pos);
  }

  return ret;
}
  
} // namespace


namespace toybox {
namespace ranking {

SVMLightReader::SVMLightReader(const char *in_file) : ifp_(NULL) {
  ifp_ = fopen(in_file, "r");
}

SVMLightReader::~SVMLightReader() {
  if (ifp_ != NULL) {
    fclose(ifp_);
  }
}

int SVMLightReader::Read(
    std::vector<std::pair<int, float> > *x, int *y, int *qid) {
  if (x == NULL || y == NULL) {
    return -1;
  }
  x->clear();

  std::vector<int>   index_vector;
  std::vector<float> value_vector;

  std::string buff;
  if (my_getline(ifp_, &buff) != 1) {
    return 0;
  }

  char cbuff[buff.size() + 1];
  memmove(cbuff, buff.c_str(), buff.size() + 1);
  char *p = strtok(cbuff, " \t");
  *y = static_cast<int>(strtol(p, NULL, 10));
  while (1) {
    char *f = strtok(NULL, ":");
    char *v = strtok(NULL, " \t");
    if (v == NULL || (f != NULL && *f == '#')) {
      break;
    }
    if (strncmp(f, "qid", 3) == 0) {
      *qid = static_cast<int>(strtol(v, NULL, 10));
      continue;
    }
    x->push_back(
        std::make_pair(static_cast<int>(strtol(f, NULL, 10)),
                       static_cast<float>(strtod(v, NULL))));
  }

  return 1;
}

int SVMLightReader::Rewind() {
  fseek(ifp_, 0, SEEK_SET);
  return 1;
}

bool SVMLightReader::IsInitialized() {
  return (ifp_ != NULL);
}


} // namespace ranking
} // namespace toybox
