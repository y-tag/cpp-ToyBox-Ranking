#ifndef TOYBOX_RANKING_LISTMLE_H
#define TOYBOX_RANKING_LISTMLE_H

#include <vector>
#include <utility>

#include "ranking.h"

namespace toybox {
namespace ranking {

class ListMLE : public Ranking {
  public:
    ListMLE();
    explicit ListMLE(float eta0);
    ~ListMLE();
    int Train(const rank_data &data);
    float Predict(const fv_vec &x) const;
    bool IsInitialized() const;

  private:
    float eta0_;
    std::vector<float> w_vec_;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_LISTMLE_H

