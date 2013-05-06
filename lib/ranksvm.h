#ifndef TOYBOX_RANKING_RANKSVM_H
#define TOYBOX_RANKING_RANKSVM_H

#include <vector>
#include <utility>

#include "ranking.h"

namespace toybox {
namespace ranking {

class RankSVM : public Ranking {
  public:
    RankSVM();
    explicit RankSVM(float C);
    ~RankSVM();
    int Train(const rank_data &data);
    float Predict(const fv_vec &x) const;
    bool IsInitialized() const;

  private:
    float C_;
    std::vector<float> w_vec_;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_RANKSVM_H

