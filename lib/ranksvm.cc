#include "ranksvm.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <map>
#include <vector>
#include <utility>

namespace {

void make_pairwise_diff(
    const toybox::ranking::fv_vec &fv_better,
    const toybox::ranking::fv_vec &fv_worse,
    toybox::ranking::fv_vec *fv_diff,
    int *max_fid) {

  std::map<int, float> diff(fv_better.begin(), fv_better.end());
  for (size_t i = 0; i < fv_worse.size(); ++i) {
    diff[fv_worse[i].first] -= fv_worse[i].second;
  }

  fv_diff->clear();
  fv_diff->assign(diff.begin(), diff.end());
  *max_fid = fv_diff->back().first; // since std::map is sorted by key
}

void make_diff_data(
    const toybox::ranking::rank_data &data,
    std::vector<toybox::ranking::fv_vec> *diff_vec,
    int *max_fid) {

  *max_fid = 0;
  diff_vec->clear();

  toybox::ranking::fv_vec diff;
  int diff_max_fid;

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    for (size_t i = 0; i < qdata.size()-1; ++i) {
      for (size_t j = i+1; j < qdata.size(); ++j) {
        diff_max_fid = 0;
        if (qdata[i].first > qdata[j].first) {
          make_pairwise_diff(qdata[i].second, qdata[j].second, &diff, &diff_max_fid);
          diff_vec->push_back(diff);
        } else if (qdata[j].first > qdata[i].first) {
          make_pairwise_diff(qdata[j].second, qdata[i].second, &diff, &diff_max_fid);
          diff_vec->push_back(diff);
        }

        if (diff_max_fid > *max_fid) {
          *max_fid = diff_max_fid;
        }
      }
    }
  }

}

} // namespace

namespace toybox {
namespace ranking {

RankSVM::RankSVM() : C_(1.0f) {
}

RankSVM::RankSVM(float C) : C_(C) {
}

RankSVM::~RankSVM() {
}

int RankSVM::Train(const rank_data &data) {
  w_vec_.clear();

  int max_fid = 0;
  std::vector<fv_vec> diff_data;
  make_diff_data(data, &diff_data, &max_fid);

  if (diff_data.size() == 0 || max_fid <= 0) { return -1; }
  w_vec_.assign(max_fid, 0.0f);

  // See "A Dual Coordinate Descent Method for Large-scale Linear SVM"
  // L1 loss SVM (hinge loss)
  float U = C_;
  float D = 0.0;
  // L2 loss SVM (squared hinge loss)
  //float U = DBL_MAX;
  //float D = 1.0 / (2.0 * C_);

  std::vector<float> idx_array(diff_data.size());
  std::vector<float> alpha_array(diff_data.size());
  std::vector<float> q_array(diff_data.size());

  for (size_t i = 0; i < diff_data.size(); ++i) {
    float snorm = 0.0;
    for (size_t j = 0; j < diff_data[i].size(); ++j) {
      snorm += diff_data[i][j].second * diff_data[i][j].second;
    }

    idx_array[i] = i;
    alpha_array[i] = 0.0;
    q_array[i] = snorm + D;
  }

  srand(1000);
  
  int max_iter = 100;
  for (int iter = 0; iter < max_iter; ++iter) {
    std::random_shuffle(idx_array.begin(), idx_array.end());

    float max_pg = -DBL_MAX;
    float min_pg =  DBL_MAX;

    for (size_t i = 0; i < diff_data.size(); ++i) {
      size_t idx = idx_array[i];
      const fv_vec &diff_vec = diff_data[idx];
      const int y = 1; // label of diff_vec is always 1

      float old_alpha = alpha_array[idx];
      float G = y * Predict(diff_vec) - 1.0 + D * old_alpha;

      float PG = G;
      if (old_alpha == 0.0f) {
        PG = std::min(G, 0.0f);
      } else if (old_alpha == U) {
        PG = std::max(G, 0.0f);
      }
      max_pg = std::max(PG, max_pg);
      min_pg = std::min(PG, min_pg);

      if (fabs(PG) > 1.0e-10) {
        float Q = q_array[idx];
        float alpha = std::min(std::max(old_alpha - G/Q, 0.0f), U);
        alpha_array[idx] = alpha;

        for (size_t j = 0; j < diff_data[idx].size(); ++j) {
          w_vec_[diff_data[idx][j].first - 1] +=
              (alpha - old_alpha) * y * diff_data[idx][j].second;
        }
      }

    }

    if (max_pg - min_pg < 1.0e-6) {
      break;
    }

  }

  return 1;
}

float RankSVM::Predict(const fv_vec &x) const {
  float predicted_value = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i].first >= static_cast<int>(w_vec_.size())) {
      continue;
    }
    predicted_value += w_vec_[x[i].first-1] * x[i].second;
  }

  return predicted_value;
}

bool RankSVM::IsInitialized() const {
  return true;
}


} // namespace ranking
} // namespace toybox
