#include "listnet.h"

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

void get_max_fid(
    const toybox::ranking::rank_data &data,
    int *max_fid) {

  *max_fid = 0;

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    for (size_t i = 0; i < qdata.size(); ++i) {
      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        if ((qdata[i].second)[j].first > *max_fid) {
          *max_fid = (qdata[i].second)[j].first;
        }
      }
    }
  }

}

double calc_cross_entropy(
    const toybox::ranking::rank_data &data,
    const std::vector<float> &w_vec) {

  double cross_entropy = 0.0f;
  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];

    std::vector<double> score_vec(qdata.size());

    double max_score = -DBL_MAX;
    for (size_t i = 0; i < qdata.size(); ++i) {
      double score = 0.0f;
      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        int   k = (qdata[i].second)[j].first;
        float v = (qdata[i].second)[j].second;
        score += w_vec[k - 1] * v;
      }
      score_vec[i] = score;
      max_score = std::max(max_score, score);
    }

    std::vector<double> exp_label_vec(qdata.size());
    std::vector<double> exp_score_vec(qdata.size());
    double sum_exp_label = 0.0;
    double sum_exp_score = 0.0;

    for (size_t i = 0; i < qdata.size(); ++i) {
        exp_label_vec[i] = exp(qdata[i].first);
        exp_score_vec[i] = exp(score_vec[i] - max_score);
        sum_exp_label += exp_label_vec[i];
        sum_exp_score += exp_score_vec[i];
    }

    double part_cost = 0.0; 
    for (size_t i = 0; i < qdata.size(); ++i) {
      double prob_label = exp_label_vec[i] / sum_exp_label;
      part_cost -= prob_label * (log(exp_score_vec[i]) - log(sum_exp_score));
    }

    cross_entropy += part_cost;
  }

  return cross_entropy;
}

} // namespace

namespace toybox {
namespace ranking {

ListNet::ListNet() : eta0_(0.01f) {
}

ListNet::ListNet(float eta0) : eta0_(eta0) {
}

ListNet::~ListNet() {
}

int ListNet::Train(const rank_data &data) {
  w_vec_.clear();

  int max_fid = 0;
  get_max_fid(data, &max_fid);

  if (data.size() == 0 || max_fid <= 0) { return -1; }
  w_vec_.assign(max_fid, 0.0f);

  float epsilon = 1.0e-10;
  float prev_cross_entropy = -FLT_MAX;
  float cross_entropy = -FLT_MAX; 

  int max_iter = 1000;
  int iter = 0;

  double eta = eta0_;

  do {
    prev_cross_entropy = cross_entropy;

    for (size_t q = 0; q < data.size(); ++q) {
      const query_data &qdata = data[q];

      std::vector<double> score_vec(qdata.size());

      double max_score = -DBL_MAX;
      for (size_t i = 0; i < qdata.size(); ++i) {
        double score = 0.0f;
        for (size_t j = 0; j < qdata[i].second.size(); ++j) {
          int   k = (qdata[i].second)[j].first;
          float v = (qdata[i].second)[j].second;
          score += w_vec_[k - 1] * v;
        }
        score_vec[i] = score;
        max_score = std::max(max_score, score);
      }

      std::vector<double> exp_label_vec(qdata.size());
      std::vector<double> exp_score_vec(qdata.size());
      double sum_exp_label = 0.0;
      double sum_exp_score = 0.0;

      for (size_t i = 0; i < qdata.size(); ++i) {
        exp_label_vec[i] = exp(qdata[i].first);
        exp_score_vec[i] = exp(score_vec[i] - max_score);
        sum_exp_label += exp_label_vec[i];
        sum_exp_score += exp_score_vec[i];
      }

      for (size_t i = 0; i < qdata.size(); ++i) {
        double prob_label = exp_label_vec[i] / sum_exp_label;
        double prob_score = exp_score_vec[i] / sum_exp_score;
        float coef = eta * (prob_score - prob_label);
        for (size_t j = 0; j < qdata[i].second.size(); ++j) {
          int   f = (qdata[i].second)[j].first;
          float v = (qdata[i].second)[j].second;
          w_vec_[f - 1] -= coef * v;
        }
      }

    }

    cross_entropy = calc_cross_entropy(data, w_vec_);
    ++iter;

    if (cross_entropy < prev_cross_entropy) {
      eta *= 0.8;
    }
    fprintf(stderr, "iter: %d, cross_entropy: %f, prev_cross_entropy: %f\n",
            iter, cross_entropy, prev_cross_entropy);
  } while(fabs(cross_entropy - prev_cross_entropy) > epsilon && iter < max_iter);

  return 1;
}

float ListNet::Predict(const fv_vec &x) const {
  float predicted_value = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i].first >= static_cast<int>(w_vec_.size())) {
      continue;
    }
    predicted_value += w_vec_[x[i].first-1] * x[i].second;
  }

  return predicted_value;
}

bool ListNet::IsInitialized() const {
  return true;
}


} // namespace ranking
} // namespace toybox
