#ifndef TOYBOX_RANKING_LISTNET_H
#define TOYBOX_RANKING_LISTNET_H

#include <vector>
#include <utility>

#include "ranking.h"

namespace toybox {
namespace ranking {

class ListNet : public Ranking {
  public:
    ListNet();
    explicit ListNet(float eta0);
    ~ListNet();
    int Train(const rank_data &data);
    float Predict(const fv_vec &x) const;
    bool IsInitialized() const;

  private:
    float eta0_;
    std::vector<float> w_vec_;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_LISTNET_H

