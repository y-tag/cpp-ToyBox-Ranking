#include "listmle.h"

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

bool second_greater_than(
    const std::pair<int, int> &a,
    const std::pair<int, int> &b) {
  return a.second > b.second;
}

void copy_and_sort_by_label(
    const toybox::ranking::rank_data &data,
    std::vector<std::vector<int> > *sort_index_vec,
    int *max_fid) {

  *max_fid = 0;
  sort_index_vec->clear();
  std::vector<std::pair<int, int> > index_label_vec;

  srand(1000);

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    index_label_vec.clear();
    for (size_t i = 0; i < qdata.size(); ++i) {
      index_label_vec.push_back(std::make_pair(i, qdata[i].first));
      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        if ((qdata[i].second)[j].first > *max_fid) {
          *max_fid = (qdata[i].second)[j].first;
        }
      }
    }
    std::random_shuffle(index_label_vec.begin(), index_label_vec.end());
    std::sort(
        index_label_vec.begin(), index_label_vec.end(), second_greater_than);

    std::vector<int> index_vec;
    for (size_t i = 0; i < index_label_vec.size(); ++i) {
      index_vec.push_back(index_label_vec[i].first);
    }

    sort_index_vec->push_back(index_vec);
  }

}

double calc_loglikelihood(
    const toybox::ranking::rank_data &data,
    const std::vector<std::vector<int> > &sort_index_vec,
    const std::vector<float> &w_vec) {

  double loglikelihood = 0.0f;
  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    const std::vector<int> qidxvec = sort_index_vec[q];
    std::vector<double> score_vec(qdata.size());
    double partial_ll = 0.0f;

    double max_score = -DBL_MAX;
    for (size_t i = 0; i < qdata.size(); ++i) {
      int idx = qidxvec[i];
      double score = 0.0f;
      for (size_t j = 0; j < qdata[idx].second.size(); ++j) {
        int   k = (qdata[idx].second)[j].first;
        float v = (qdata[idx].second)[j].second;
        score += w_vec[k - 1] * v;
      }
      score_vec[idx] = score;
      max_score = std::max(max_score, score);
    }

    std::vector<double> exp_vec(qdata.size());
    std::vector<double> sumexp_vec(qdata.size());
    double sumexp = 0.0f;

    for (int i = qdata.size() - 1; i >= 0; --i) {
        int idx = qidxvec[i];
        exp_vec[idx] = exp(score_vec[idx] - max_score);
        sumexp += exp_vec[idx];
        sumexp_vec[idx] = sumexp;
    }
    for (size_t i = 0; i < qdata.size(); ++i) {
      int idx = qidxvec[i];
      partial_ll += log(exp_vec[idx]) - log(sumexp_vec[idx]);
      //fprintf(stderr, "q: %lu, i: %lu, idx: %d, exp:%g, sumexp: %g, log(exp): %g, log(sumexp): %g\n", q, i, idx, exp_vec[idx], sumexp_vec[i], log(exp_vec[idx]), log(sumexp_vec[i]));
    }

    loglikelihood += partial_ll;
  }

  return loglikelihood;
}

} // namespace

namespace toybox {
namespace ranking {

ListMLE::ListMLE() : eta0_(0.01f) {
}

ListMLE::ListMLE(float eta0) : eta0_(eta0) {
}

ListMLE::~ListMLE() {
}

int ListMLE::Train(const rank_data &data) {
  w_vec_.clear();

  int max_fid = 0;
  std::vector<std::vector<int> > sort_index_vec;
  copy_and_sort_by_label(data, &sort_index_vec, &max_fid);

  if (data.size() == 0 || max_fid <= 0) { return -1; }
  w_vec_.assign(max_fid, 0.0f);

  float epsilon = 1.0e-10;
  float prev_loglikelihood = -FLT_MAX;
  float loglikelihood = -FLT_MAX; 

  int max_iter = 1000;
  int iter = 0;

  double eta = eta0_;

  do {
    prev_loglikelihood = loglikelihood;

    for (size_t q = 0; q < data.size(); ++q) {
      const query_data &qdata = data[q];
      const std::vector<int> qidxvec = sort_index_vec[q];

      std::vector<double> score_vec(qdata.size());
      double max_score = -DBL_MAX;

      for (size_t i = 0; i < qdata.size(); ++i) {
        int idx = qidxvec[i];
        double score = 0.0f;
        for (size_t j = 0; j < qdata[idx].second.size(); ++j) {
          int   k = (qdata[idx].second)[j].first;
          float v = (qdata[idx].second)[j].second;
          score += w_vec_[k - 1] * v;
        }
        score_vec[idx] = score;
        max_score = std::max(max_score, score);
      }

      double sumexp = 0.0;
      std::vector<double> exp_vec(qdata.size());
      std::vector<double> sumexp_vec(qdata.size());

      for (int i = qdata.size() - 1; i >= 0; --i) {
        int idx = qidxvec[i];
        exp_vec[idx] = exp(score_vec[idx] - max_score);
        sumexp += exp_vec[idx];
        sumexp_vec[idx] = sumexp;
      }

      double suminvsumexp = 0.0;
      std::vector<double> suminvsumexp_vec(qdata.size());

      for (size_t i = 0; i < qdata.size(); ++i) {
        int idx = qidxvec[i];
        suminvsumexp += 1.0 / sumexp_vec[idx];
        suminvsumexp_vec[idx] = suminvsumexp;
      }

      for (size_t i = 0; i < qdata.size(); ++i) {
        int idx = qidxvec[i];
        float coef = eta * (exp_vec[idx] * suminvsumexp_vec[idx] - 1.0f);
        //fprintf(stderr, "q: %lu, i: %lu, idx: %d, label: %d, coef: %g, exp: %g, sumexp:%g, suminvsumexp: %g\n", q, i, idx, qdata[idx].first, coef, exp_vec[idx], sumexp_vec[i], suminvsumexp_vec[idx]);
        for (size_t j = 0; j < qdata[idx].second.size(); ++j) {
          int   f = (qdata[idx].second)[j].first;
          float v = (qdata[idx].second)[j].second;
          w_vec_[f - 1] -= coef * v;
        }
      }

    }

    loglikelihood = calc_loglikelihood(data, sort_index_vec, w_vec_);
    ++iter;

    if (loglikelihood < prev_loglikelihood) {
      eta *= 0.8;
    }
    fprintf(stderr, "iter: %d, loglikelihood: %f, prev_loglikelihood: %f\n",
            iter, loglikelihood, prev_loglikelihood);
  } while(fabs(loglikelihood - prev_loglikelihood) > epsilon && iter < max_iter);

  return 1;
}

float ListMLE::Predict(const fv_vec &x) const {
  float predicted_value = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i].first >= static_cast<int>(w_vec_.size())) {
      continue;
    }
    predicted_value += w_vec_[x[i].first-1] * x[i].second;
  }

  return predicted_value;
}

bool ListMLE::IsInitialized() const {
  return true;
}


} // namespace ranking
} // namespace toybox
