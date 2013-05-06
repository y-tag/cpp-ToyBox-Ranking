#ifndef TOYBOX_RANKING_LAMBDARANK_H
#define TOYBOX_RANKING_LAMBDARANK_H

#include <vector>
#include <utility>

#include "ranking.h"

namespace toybox {
namespace ranking {

class LambdaRank : public Ranking {
  public:
    LambdaRank();
    LambdaRank(int T, float eta0);
    ~LambdaRank();
    int Train(const rank_data &data);
    float Predict(const fv_vec &x) const;
    bool IsInitialized() const;

  private:
    int T_;
    float eta0_;
    std::vector<float> w_vec_;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_LAMBDARANK_H

