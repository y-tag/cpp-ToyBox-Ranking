#include "ranknet.h"

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

void make_diff_index(
    const toybox::ranking::rank_data &data,
    std::vector<std::vector<std::pair<int, int> > > *diff_index_vec,
    int *max_fid) {

  *max_fid = 0;
  diff_index_vec->clear();

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    std::vector<std::pair<int, int> > qidxvec;
    for (size_t i = 0; i < qdata.size(); ++i) {
      for (size_t j = i+1; j < qdata.size(); ++j) {
        if (qdata[i].first > qdata[j].first) {
          qidxvec.push_back(std::make_pair(i, j));
        } else if (qdata[j].first > qdata[i].first) {
          qidxvec.push_back(std::make_pair(j, i));
        }
      }

      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        if ((qdata[i].second)[j].first > *max_fid) {
          *max_fid = (qdata[i].second)[j].first;
        }
      }
    }
    diff_index_vec->push_back(qidxvec);
  }

}

double calc_cross_entropy(
    const toybox::ranking::rank_data &data,
    const std::vector<std::vector<std::pair<int, int> > > &diff_index_vec,
    float sigma,
    const std::vector<float> &w_vec) {
  double cross_entropy = 0.0;

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    const std::vector<std::pair<int, int> > qidxvec = diff_index_vec[q];

    std::vector<double> score_vec(qdata.size());

    for (size_t i = 0; i < qdata.size(); ++i) {
      double score = 0.0f;
      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        int   k = (qdata[i].second)[j].first;
        float v = (qdata[i].second)[j].second;
        score += w_vec[k - 1] * v;
      }
      score_vec[i] = score;
    }

    double part_cost = 0.0f;

    for (size_t k = 0; k < qidxvec.size(); ++k) {
      int i = qidxvec[k].first;
      int j = qidxvec[k].second;

      double diff = -sigma * (score_vec[i] - score_vec[j]);
      if (diff > 35.0) { // this is magic number...
        part_cost += diff;
      } else {
        part_cost += log(1.0 + exp(diff));
      }
    }

    cross_entropy += part_cost;
  }


  return cross_entropy;
}

} // namespace

namespace toybox {
namespace ranking {

RankNet::RankNet() : eta0_(0.001f) {
}

RankNet::RankNet(float eta0) : eta0_(eta0) {
}

RankNet::~RankNet() {
}

int RankNet::Train(const rank_data &data) {
  w_vec_.clear();

  int max_fid = 0;
  float sigma = 1.0f; // parameter which determines the shape of the sigmoid
  std::vector<std::vector<std::pair<int, int> > > diff_index_vec;

  make_diff_index(data, &diff_index_vec, &max_fid);

  if (data.size() == 0 || max_fid <= 0) { return -1; }
  w_vec_.assign(max_fid, 0.0f);

  float epsilon = 1.0e-6;
  float prev_cross_entropy = FLT_MAX;
  float cross_entropy = FLT_MAX; 

  int max_iter = 1000;
  int iter = 0;

  double eta = eta0_;

  do {
    prev_cross_entropy = cross_entropy;

    for (size_t q = 0; q < data.size(); ++q) {
      const query_data &qdata = data[q];
      const std::vector<std::pair<int, int> > qidxvec = diff_index_vec[q];

      std::vector<double> score_vec(qdata.size());

      for (size_t i = 0; i < qdata.size(); ++i) {
        double score = 0.0f;
        for (size_t j = 0; j < qdata[i].second.size(); ++j) {
          int   k = (qdata[i].second)[j].first;
          float v = (qdata[i].second)[j].second;
          score += w_vec_[k - 1] * v;
        }
        score_vec[i] = score;
      }

      std::vector<double> lambda_vec(qdata.size());
      for (size_t k = 0; k < qidxvec.size(); ++k) {
        int i = qidxvec[k].first;
        int j = qidxvec[k].second;

        double lambda_ij = 0.0;
        double t = sigma * (score_vec[i] - score_vec[j]);
        if (t > 35.0) { // This is magic number...
          double et = exp(-t);
          lambda_ij = -(sigma * et) / (1.0 + et);
        } else {
          lambda_ij = -sigma / (1.0 + exp(t));
        }

        lambda_vec[i] += lambda_ij;
        lambda_vec[j] -= lambda_ij;
      }

      for (size_t i = 0; i < qdata.size(); ++i) {
        float coef = eta * lambda_vec[i];
        for (size_t j = 0; j < qdata[i].second.size(); ++j) {
          int   f = (qdata[i].second)[j].first;
          float v = (qdata[i].second)[j].second;
          w_vec_[f - 1] -= coef * v;
        }
      }

    }

    cross_entropy = calc_cross_entropy(data, diff_index_vec, sigma, w_vec_);
    ++iter;

    if (cross_entropy > prev_cross_entropy) {
      eta *= 0.8;
    }
    fprintf(stderr, "iter: %d, cross_entropy: %f, prev_cross_entropy: %f\n",
            iter, cross_entropy, prev_cross_entropy);

  } while(fabs(cross_entropy - prev_cross_entropy) > epsilon && iter < max_iter);

  return 1;
}

float RankNet::Predict(const fv_vec &x) const {
  float predicted_value = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i].first >= static_cast<int>(w_vec_.size())) {
      continue;
    }
    predicted_value += w_vec_[x[i].first-1] * x[i].second;
  }

  return predicted_value;
}

bool RankNet::IsInitialized() const {
  return true;
}


} // namespace ranking
} // namespace toybox
