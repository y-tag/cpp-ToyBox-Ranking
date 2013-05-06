#ifndef TOYBOX_RANKING_SVMLIGHT_READER_H
#define TOYBOX_RANKING_SVMLIGHT_READER_H

#include <cstdio>
#include <vector>
#include <utility>

namespace toybox {
namespace ranking {

class SVMLightReader {
  public:
    explicit SVMLightReader(const char *in_file);
    ~SVMLightReader();
    int Read(std::vector<std::pair<int, float> > *x, int *y, int *qid);
    int Rewind();
    bool IsInitialized();

  private:
    FILE *ifp_;
    SVMLightReader();
    SVMLightReader(const SVMLightReader&);
    void operator=(const SVMLightReader&);
};


} // namespace ranking 
} // namespace toybox

#endif // TOYBOX_RANKING_SVMLIGHT_READER_H

