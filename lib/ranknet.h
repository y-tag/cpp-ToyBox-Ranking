#ifndef TOYBOX_RANKING_RANKNET_H
#define TOYBOX_RANKING_RANKNET_H

#include <vector>
#include <utility>

#include "ranking.h"

namespace toybox {
namespace ranking {

class RankNet : public Ranking {
  public:
    RankNet();
    explicit RankNet(float eta0);
    ~RankNet();
    int Train(const rank_data &data);
    float Predict(const fv_vec &x) const;
    bool IsInitialized() const;

  private:
    float eta0_;
    std::vector<float> w_vec_;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_RANKNET_H

