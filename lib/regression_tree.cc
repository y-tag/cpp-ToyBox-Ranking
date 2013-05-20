#include "regression_tree.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <functional>
#include <set>
#include <map>
#include <vector>
#include <utility>

namespace {

int prepare_for_training(
    const std::vector<std::vector<int> > &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<std::vector<double> > &id_fv_vecvec,
    const std::vector<int> &datum_vec,
    const std::vector<int> &feature_vec,
    std::vector<std::vector<double> > *id_y_vecvec,
    std::vector<std::vector<int> > *id_n_vecvec
    ) {

  id_y_vecvec->assign(x_vec[0].size(), std::vector<double>());
  id_n_vecvec->assign(x_vec[0].size(), std::vector<int>());

  for (size_t i = 0; i < feature_vec.size(); ++i) {
    int fid = feature_vec[i];
    /*
    fprintf(stderr, "fid: %d\n", fid);
    for (size_t j = 0; j < id_fv_vecvec[fid].size(); ++j) {
      fprintf(stderr, "%lu:%lf ", j, id_fv_vecvec[fid][j]);
    }
    fprintf(stderr, "\n");
    */

    std::vector<double> id_y_vec(id_fv_vecvec[fid].size());
    std::vector<int> id_n_vec(id_fv_vecvec[fid].size());

    for (size_t j = 0; j < datum_vec.size(); ++j) {
      int did = datum_vec[j];
      double y = y_vec[did];
      int id = x_vec[did][fid];

      //fprintf(stderr, "did:%d, y:%lf, id:%d\n", did, y, id);

      id_y_vec[id] += y;
      id_n_vec[id] += 1;
    }

    (*id_y_vecvec)[fid]  = id_y_vec;
    (*id_n_vecvec)[fid]  = id_n_vec;
  }

  return 1;
}

struct ChildInfo {
  void clear() {
    y_sum_ = 0.0;
    datum_vec_.clear();
    id_y_vecvec_.clear();
    id_n_vecvec_.clear();
  }
  double y_sum_;
  std::vector<int> datum_vec_;
  std::vector<std::vector<double> > id_y_vecvec_;
  std::vector<std::vector<int> > id_n_vecvec_;
};

struct CandidateInfo {
  void clear() {
    leaf_node_ = 0;
    feature_ = -1;
    split_id_ = -1;
    gain_ = -DBL_MAX;
    left_.clear();
    right_.clear();
  }
  int leaf_node_;
  int feature_;
  int split_id_;
  double gain_;
  ChildInfo left_;
  ChildInfo right_;
};

class SortByGain {
  public:
    SortByGain(const std::vector<CandidateInfo> &candidate_vec)
      : candidate_vec_(candidate_vec) {};

    bool operator()(int a, int b) {
      return candidate_vec_[a].gain_ < candidate_vec_[b].gain_;
    }
  
  private:
    const std::vector<CandidateInfo> &candidate_vec_;

};

/*
bool gain_less_than(const CandidateInfo &a, const CandidateInfo &b) {
  return a.gain_ < b.gain_;
}
*/

int find_best_split(
    const std::vector<std::vector<int> > &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<int> &feature_vec,
    size_t fnum_to_use,
    int min_leaf_instance,
    const ChildInfo &info,
    CandidateInfo *candidate_info
    ) {

  candidate_info->feature_ = -1;
  int best_feature  = -1;
  int best_split_id = -1;
  double best_gain  = -DBL_MAX;

  if (static_cast<int>(info.datum_vec_.size()) < std::max(2, min_leaf_instance)) {
    return -1;
  }

  //fprintf(stderr, "y_sum: %lg, data_num: %lu\n", info.y_sum_, info.datum_vec_.size());

  for (size_t i = 0; i < fnum_to_use; ++i) {
    int f = feature_vec[i];

    int left_num = 0;
    double left_sum = 0.0;
    int right_num = info.datum_vec_.size();
    double right_sum = info.y_sum_;

    //fprintf(stderr, "f: %d, i: %lu, num: %lu\n", f, i, info.id_n_vecvec_[f].size());
    for (size_t j = 0; j < info.id_n_vecvec_[f].size() - 1; ++j) {
      if (info.id_n_vecvec_[f][j] == 0) { continue; }

      left_num  += info.id_n_vecvec_ [f][j];
      left_sum  += info.id_y_vecvec_[f][j];
      right_num -= info.id_n_vecvec_[f][j];
      right_sum -= info.id_y_vecvec_[f][j];

      if (left_num  < min_leaf_instance) { continue; }
      if (right_num < min_leaf_instance) { break; }

      double gain = (left_sum * left_sum) / left_num +
                    (right_sum * right_sum) / right_num;

      //fprintf(stderr, "ln: %d, ls: %lf, rn: %d, ls: %lf, gain: %lf\n", left_num, left_sum, right_num, right_sum, gain);

      if (gain > best_gain) {
        best_feature = f;
        best_split_id = j;
        best_gain = gain;
      }
    }
  }

  if (best_feature < 0) {
    return -1;
  }

  //fprintf(stderr, "best_feature: %d, best_split_id: %d, best_gain: %lf\n", best_feature, best_split_id, best_gain);

  candidate_info->clear();
  candidate_info->feature_  = best_feature;
  candidate_info->split_id_ = best_split_id;

  for (size_t i = 0; i < info.id_n_vecvec_.size(); ++i) {
    size_t n = info.id_n_vecvec_[i].size();
    std::vector<double> y_vec(n, 0.0);
    std::vector<int> n_vec(n, 0);
    candidate_info->left_.id_y_vecvec_.push_back(y_vec);
    candidate_info->left_.id_n_vecvec_.push_back(n_vec);
    candidate_info->right_.id_y_vecvec_.push_back(y_vec);
    candidate_info->right_.id_n_vecvec_.push_back(n_vec);
  }

  for (size_t i = 0; i < info.datum_vec_.size(); ++i) {
    int didx = info.datum_vec_[i];
    double y = y_vec[didx];
    /*
    fprintf(stderr, "i: %lu, didx: %d, y: %lf\n", i, didx, y);
    if (x_vec[didx][best_feature] <= best_split_id) {
      fprintf(stderr, "%d, left\n", x_vec[didx][best_feature]);
    } else {
      fprintf(stderr, "%d, right\n", x_vec[didx][best_feature]);
    }
    */
    ChildInfo &info_ref = (x_vec[didx][best_feature] <= best_split_id) ?
                          candidate_info->left_ :
                          candidate_info->right_;

    info_ref.y_sum_ += y_vec[didx];
    info_ref.datum_vec_.push_back(didx);
    for (size_t f = 0; f < x_vec[i].size(); ++f) {
      int id = x_vec[didx][f];
      info_ref.id_y_vecvec_[f][id] += y;
      info_ref.id_n_vecvec_[f][id] += 1;
    }
  }

  /*
  fprintf(stderr, "left_sum: %lf\n", candidate_info->left_.y_sum_);
  fprintf(stderr, "left_datum\n");
  for (size_t i = 0; i < candidate_info->left_.datum_vec_.size(); ++i) {
    fprintf(stderr, "%d, ", candidate_info->left_.datum_vec_[i]);
  }
  fprintf(stderr, "\n");

  fprintf(stderr, "right_sum: %lf\n", candidate_info->right_.y_sum_);
  fprintf(stderr, "right_datum\n");
  for (size_t i = 0; i < candidate_info->right_.datum_vec_.size(); ++i) {
    fprintf(stderr, "%d, ", candidate_info->right_.datum_vec_[i]);
  }
  fprintf(stderr, "\n");
  */
  
  candidate_info->gain_ = best_gain -
                          (info.y_sum_ * info.y_sum_) / info.datum_vec_.size();
  //fprintf(stderr, "new gain: %lf\n", candidate_info->gain_);
  //fprintf(stderr, "\n");

  return 1;
}


} // namespace

namespace toybox {
namespace ensemble {

RegressionTree::RegressionTree()
  : leaf_num_(10), min_leaf_instance_rate_(1e-4),
    feature_sampling_rate_(1.0), is_feature_sampling_randomized_(false),
    feature_vec_(1, 0), id_vec_(1, INT_MAX), thresh_vec_(1, DBL_MAX),
    left_vec_(1, ~0), right_vec_(1, ~0),
    predict_vec_(1, 0.0) {
}

RegressionTree::RegressionTree(
    int leaf_num, double min_leaf_instance_rate,
    double feature_sampling_rate, bool is_feature_sampling_randomized)
  : leaf_num_(leaf_num), min_leaf_instance_rate_(min_leaf_instance_rate),
    feature_sampling_rate_(feature_sampling_rate),
    is_feature_sampling_randomized_(is_feature_sampling_randomized),
    feature_vec_(1, 0), id_vec_(1, INT_MAX), thresh_vec_(1, DBL_MAX),
    left_vec_(1, ~0), right_vec_(1, ~0),
    predict_vec_(1, 0.0) {
}

RegressionTree::~RegressionTree() {
}

int RegressionTree::Train(
    const data &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<int> &target_datum_vec,
    const std::vector<int> &target_feature_vec) {

  std::vector<std::vector<int> > converted_x_vec;
  std::vector<std::vector<double> > id_fv_vecvec;
  
  int max_fid = x_vec[0].size();
  std::set<int> fid_set;
  std::map<int, std::set<double> > fid_fv_map;

  id_fv_vecvec.assign(max_fid, std::vector<double>());
  std::map<int, std::map<double, int> > fid_fv_id_mapmap;

  for (size_t i = 0; i < x_vec.size(); ++i) {
    for (size_t f = 0; f < x_vec[i].size(); ++f) {
      float v = x_vec[i][f];
      fid_fv_map[f].insert(v);
    }
  }

  for (size_t i = 0; i < target_feature_vec.size(); ++i) {
    int fid = target_feature_vec[i];
    const std::set<double> &fv_set = fid_fv_map[fid];
    int j = 0;
    for (std::set<double>::const_iterator itr = fv_set.begin();
         itr != fv_set.end(); ++itr) {
      double v = *itr;
      id_fv_vecvec[fid].push_back(v);
      fid_fv_id_mapmap[fid][v] = j;
      ++j;
    }
  }

  for (std::map<int, std::map<double, int> >::const_iterator itr1 = fid_fv_id_mapmap.begin();
       itr1 != fid_fv_id_mapmap.end(); ++itr1) {
    for (std::map<double, int>::const_iterator itr2 = itr1->second.begin();
         itr2 != itr1->second.end(); ++itr2) {
    }
  }


  for (size_t i = 0; i < x_vec.size(); ++i) {
    std::vector<int> dx(x_vec[i].size());
    for (size_t f = 0; f < x_vec[i].size(); ++f) {
      float v = x_vec[i][f];
      int  id = fid_fv_id_mapmap[f][v];
      dx[f] = id;
    }
    converted_x_vec.push_back(dx);
  }

  Train(converted_x_vec, y_vec, id_fv_vecvec, target_datum_vec, target_feature_vec);

  return 1;
}

int RegressionTree::Train(
    const std::vector<std::vector<int> > &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<std::vector<double> > &id_fv_vecvec,
    const std::vector<int> &target_datum_vec,
    const std::vector<int> &target_feature_vec) {
  if (leaf_num_ < 2) {
    return -1;
  }

  feature_vec_.assign(leaf_num_ - 1, 0);
  id_vec_.assign(leaf_num_ - 1, 0);
  thresh_vec_.assign(leaf_num_ - 1, 0.0);
  left_vec_.assign(leaf_num_ - 1, 0);
  right_vec_.assign(leaf_num_ - 1, 0);
  predict_vec_.assign(leaf_num_, 0.0);

  std::vector<int> datum_vec(target_datum_vec);
  std::sort(datum_vec.begin(), datum_vec.end());
  std::vector<int> feature_vec(target_feature_vec);

  int min_leaf_instance = datum_vec.size() * min_leaf_instance_rate_;
  min_leaf_instance = std::max(1, min_leaf_instance);
  size_t fnum_to_use = feature_vec.size() * feature_sampling_rate_;
  if (fnum_to_use < feature_vec.size()) {
    std::random_shuffle(feature_vec.begin(), feature_vec.end());
  } else {
    std::sort(feature_vec.begin(), feature_vec.end());
  }

  std::vector<std::vector<double> > id_y_vecvec;
  std::vector<std::vector<int> > id_n_vecvec;

  prepare_for_training(
      x_vec, y_vec, id_fv_vecvec, datum_vec, feature_vec,
      &id_y_vecvec, &id_n_vecvec
  );

  // set default leaf node
  double y_sum = 0.0;
  for (size_t j = 0; j < datum_vec.size(); ++j) {
    y_sum += y_vec[datum_vec[j]];
  }
  feature_vec_[0] = 0; id_vec_[0] = INT_MAX, thresh_vec_[0] = DBL_MAX;
  left_vec_[0] = ~0; right_vec_[0] = ~0;
  predict_vec_[0] = y_sum / datum_vec.size();

  int leaf_node = 0;
  int inter_node = 0;
  std::vector<int> candidate_heap;
  std::vector<CandidateInfo> candidate_vec(leaf_num_);
  std::vector<int> leaf_node_heap;

  for (int i = 0; i < leaf_num_; ++i) {
    leaf_node_heap.push_back(i);
  }
  std::make_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());

  if (is_feature_sampling_randomized_) {
    std::random_shuffle(feature_vec.begin(), feature_vec.end());
  }

  ChildInfo root_info;
  root_info.y_sum_ = y_sum;
  root_info.datum_vec_.assign(datum_vec.begin(), datum_vec.end());
  root_info.id_y_vecvec_.assign(id_y_vecvec.begin(), id_y_vecvec.end());
  root_info.id_n_vecvec_.assign(id_n_vecvec.begin(), id_n_vecvec.end());

  CandidateInfo candidate_info;

  find_best_split(
      x_vec, y_vec, feature_vec, fnum_to_use, min_leaf_instance,
      root_info, &candidate_info
  );

  if (candidate_info.feature_ < 0) { // cannot split at all
    return 0;
  }

  // get leaf node id
  leaf_node = leaf_node_heap.front();
  std::pop_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());
  leaf_node_heap.pop_back();
  // push split candidate info
  candidate_info.leaf_node_ = leaf_node;
  candidate_vec[leaf_node] = candidate_info;
  candidate_heap.push_back(leaf_node);
  std::push_heap(candidate_heap.begin(), candidate_heap.end(), SortByGain(candidate_vec));

  // grow tree
  while (candidate_heap.size() > 0 && leaf_node_heap.size() > 0) {
    //pop the candidate that has largest gain
    CandidateInfo parent_info = candidate_vec[candidate_heap.front()];
    std::pop_heap(candidate_heap.begin(), candidate_heap.end(), SortByGain(candidate_vec));
    candidate_heap.pop_back();

    // change leaf node to internal node
    std::vector<int>::iterator itr;
    itr = std::find(left_vec_.begin(), left_vec_.end(), ~(parent_info.leaf_node_));
    if (itr != left_vec_.end())  { *itr = inter_node; }
    itr = std::find(right_vec_.begin(), right_vec_.end(), ~(parent_info.leaf_node_));
    if (itr != right_vec_.end()) { *itr = inter_node; }

    // release leaf node id
    leaf_node_heap.push_back(parent_info.leaf_node_);
    std::push_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());

    // get leaf node ids for children
    int left_leaf_node  = leaf_node_heap.front();
    std::pop_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());
    leaf_node_heap.pop_back();
    int right_leaf_node = leaf_node_heap.front();
    std::pop_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());
    leaf_node_heap.pop_back();

    // set internal node info
    feature_vec_[inter_node] = parent_info.feature_;
    id_vec_[inter_node]      = parent_info.split_id_;
    thresh_vec_[inter_node]  = id_fv_vecvec[parent_info.feature_][parent_info.split_id_];
    left_vec_[inter_node]    = ~left_leaf_node;
    right_vec_[inter_node]   = ~right_leaf_node;
    inter_node++;

    // set leaf node info
    predict_vec_[left_leaf_node]  = parent_info.left_.y_sum_  / parent_info.left_.datum_vec_.size();
    predict_vec_[right_leaf_node] = parent_info.right_.y_sum_ / parent_info.right_.datum_vec_.size();

    // for left child
    if (is_feature_sampling_randomized_) {
      std::random_shuffle(feature_vec.begin(), feature_vec.end());
    }
    find_best_split(
        x_vec, y_vec, feature_vec, fnum_to_use, min_leaf_instance,
        parent_info.left_, &(candidate_vec[left_leaf_node])
    );
    if (candidate_vec[left_leaf_node].feature_ >= 0) { // splitable
      candidate_vec[left_leaf_node].leaf_node_ = left_leaf_node;
      // push split candidate info
      candidate_heap.push_back(left_leaf_node);
      std::push_heap(candidate_heap.begin(), candidate_heap.end(), SortByGain(candidate_vec));
    }

    // for right child
    if (is_feature_sampling_randomized_) {
      std::random_shuffle(feature_vec.begin(), feature_vec.end());
    }
    find_best_split(
        x_vec, y_vec, feature_vec, fnum_to_use, min_leaf_instance,
        parent_info.right_, &(candidate_vec[right_leaf_node])
    );
    if (candidate_vec[right_leaf_node].feature_ >= 0) { // splitable
      candidate_vec[right_leaf_node].leaf_node_ = right_leaf_node;
      // push split candidate info
      candidate_heap.push_back(right_leaf_node);
      std::push_heap(candidate_heap.begin(), candidate_heap.end(), SortByGain(candidate_vec));
    }

  }

  return 1;
}

double RegressionTree::Predict(const datum &x) const {
  int node = GetCorrespondingLeafNode(x);
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return 0;
  }
  return predict_vec_[node];
}

double RegressionTree::Predict(const std::vector<int> &x) const {
  int node = GetCorrespondingLeafNode(x);
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return 0;
  }
  return predict_vec_[node];
}

int RegressionTree::GetCorrespondingLeafNode(const datum &x) const {
  if (predict_vec_.size() == 0) {
    return -1;
  }

  int x_size = x.size();
  int node = 0;
  for (size_t i = 0; i < predict_vec_.size(); ++i) {
    int feature   = feature_vec_[node];
    double thresh = thresh_vec_[node];

    double val = (feature < x_size) ? x[feature] : 0.0;
    node = (val <= thresh) ? left_vec_[node] : right_vec_[node];

    if (node < 0) { return ~node; }
  }

  return 0;
}

int RegressionTree::GetCorrespondingLeafNode(const std::vector<int> &x) const {
  if (predict_vec_.size() == 0) {
    return -1;
  }

  int x_size = x.size();
  int node = 0;
  for (size_t i = 0; i < predict_vec_.size(); ++i) {
    int feature   = feature_vec_[node];
    int id = id_vec_[node];

    double val = (feature < x_size) ? x[feature] : 0.0;
    node = (val <= id) ? left_vec_[node] : right_vec_[node];

    if (node < 0) { return ~node; }
  }

  return 0;
}

double RegressionTree::GetLeafNodeValue(int node) {
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return 0.0;
  }
  return predict_vec_[node];
}

int RegressionTree::SetLeafNodeValue(int node, double value) {
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return -1;
  }
  predict_vec_[node] = value;
  return 1;
}

void RegressionTree::Print() const {
  fprintf(stderr, "inter node\n");
  for (int i = 0; i < leaf_num_ - 1; ++i) {
    fprintf(stderr, "%d, %lf, %d, %d\n", feature_vec_[i], thresh_vec_[i], left_vec_[i], right_vec_[i]);
  }
  fprintf(stderr, "\n");

  fprintf(stderr, "leaf node\n");
  for (int i = 0; i < leaf_num_; ++i) {
    fprintf(stderr, "%lf\n", predict_vec_[i]);
  }
  fprintf(stderr, "\n");
}


} // namespace ensemble
} // namespace toybox
