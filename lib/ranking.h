#ifndef TOYBOX_RANKING_RANKING_H
#define TOYBOX_RANKING_RANKING_H

#include <vector>
#include <utility>

namespace toybox {
namespace ranking {

typedef std::vector<std::pair<int, float> > fv_vec;
typedef std::vector<std::pair<int, fv_vec> > query_data;
typedef std::vector<query_data> rank_data;

class Ranking {
  public:
    Ranking() {};
    virtual ~Ranking() {};
    virtual int Train(const rank_data &data) = 0;
    virtual float Predict(const fv_vec &x) const = 0;
    virtual bool IsInitialized() const = 0;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_RANKING_H

