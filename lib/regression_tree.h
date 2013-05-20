#ifndef TOYBOX_ENSEMBLE_REGRESSIONTREE_H
#define TOYBOX_ENSEMBLE_REGRESSIONTREE_H

#include <vector>
#include <utility>

#include "tree.h"

namespace toybox {
namespace ensemble {

class RegressionTree {
  public:
    RegressionTree();
    RegressionTree(
        int leaf_num,
        double min_leaf_instance_rate,
        double feature_sampling_rate,
        bool is_feature_sampling_randomized);
    ~RegressionTree();
    int Train(
        const data &x_vec,
        const std::vector<double> &y_vec,
        const std::vector<int> &target_datum_vec,
        const std::vector<int> &target_feature_vec);
    int Train(
        const std::vector<std::vector<int> > &x_vec,
        const std::vector<double> &y_vec,
        const std::vector<std::vector<double> > &id_fv_vecvec,
        const std::vector<int> &target_datum_vec,
        const std::vector<int> &target_feature_vec);
    double Predict(const datum &x) const;
    double Predict(const std::vector<int> &x) const;
    int GetCorrespondingLeafNode(const datum &x) const;
    int GetCorrespondingLeafNode(const std::vector<int> &x) const;
    double GetLeafNodeValue(int node);
    int SetLeafNodeValue(int node, double value);
    bool IsInitialized() const { return true; };
    void Print() const;

  private:
    int leaf_num_;
    double min_leaf_instance_rate_;
    double feature_sampling_rate_;
    bool is_feature_sampling_randomized_;
    std::vector<int> feature_vec_;
    std::vector<double> id_vec_;
    std::vector<double> thresh_vec_;
    std::vector<int> left_vec_;
    std::vector<int> right_vec_;
    std::vector<double> predict_vec_;
};


} // namespace ensemble
} // namespace toybox

#endif // TOYBOX_ENSEMBLE_REGRESSIONTREE_H

