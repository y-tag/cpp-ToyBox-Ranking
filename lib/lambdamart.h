#ifndef TOYBOX_RANKING_LAMBDAMART_H
#define TOYBOX_RANKING_LAMBDAMART_H

#include <vector>
#include <utility>

#include "ranking.h"
#include "tree.h"
#include "regression_tree.h"

namespace toybox {
namespace ranking {

class LambdaMART : public Ranking {
  public:
    LambdaMART();
    LambdaMART(
        int T, int tree_num, float eta0,
        int leaf_num, float min_leaf_instance_rate,
        float data_sampling_rate, float feature_sampling_rate,
        bool is_feature_sampling_randomized);
    ~LambdaMART();
    int Train(const rank_data &data);
    float Predict(const fv_vec &x) const;
    bool IsInitialized() const;

  private:
    int T_;
    int tree_num_;
    float eta0_;
    int leaf_num_;
    float min_leaf_instance_rate_;
    float data_sampling_rate_;
    float feature_sampling_rate_;
    bool is_feature_sampling_randomized_;
    int max_fid_;
    std::vector<toybox::ensemble::RegressionTree> tree_vec_;
};


} // namespace ranking
} // namespace toybox

#endif // TOYBOX_RANKING_LAMBDAMART_H

