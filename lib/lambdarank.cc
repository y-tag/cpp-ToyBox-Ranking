#include "lambdarank.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <functional>
#include <map>
#include <vector>
#include <utility>

namespace {

bool second_double_greater_than(
    const std::pair<int, double> &a,
    const std::pair<int, double> &b) {
  return a.second > b.second;
}

double log2(double x) {
  return log(x) / log(2.0);
}

double calc_dcg(
    const std::vector<int> sorted_label_vec,
    int T) {
  double dcg = 0;
  int max_rank = static_cast<int>(sorted_label_vec.size());
  max_rank = std::min(max_rank, T);

  for (int i = 0; i < max_rank; ++i) {
    int rank  = i + 1;
    int label = sorted_label_vec[i];
    dcg += (pow(2.0, label) - 1) / log2(rank + 1);
  }

  return dcg;
}

void calc_idcg(
    const toybox::ranking::rank_data &data,
    int T,
    std::vector<float> *idcg_vec) {
  idcg_vec->clear();

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];

    std::vector<int> label_vec;
    for (size_t i = 0; i < qdata.size(); ++i) {
      label_vec.push_back(qdata[i].first);
    }
    std::sort(label_vec.begin(), label_vec.end(), std::greater<int>()); 

    double idcg = calc_dcg(label_vec, T);
    idcg_vec->push_back(idcg);
  }
}

double calc_ndcg(
    const toybox::ranking::rank_data &data,
    const std::vector<float> &w_vec,
    const std::vector<float> &idcg_vec,
    int T) {

  double sum_ndcg = 0.0;

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];

    std::vector<std::pair<int, double> > label_score_vec(qdata.size());
    for (size_t i = 0; i < qdata.size(); ++i) {
      double score = 0.0f;
      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        int   k = (qdata[i].second)[j].first;
        float v = (qdata[i].second)[j].second;
        score += w_vec[k - 1] * v;
      }
      label_score_vec.push_back(std::make_pair(qdata[i].first, score));
    }
    std::sort(
        label_score_vec.begin(),
        label_score_vec.end(),
        second_double_greater_than);

    std::vector<int> label_vec;
    for (size_t i = 0; i < label_score_vec.size(); ++i) {
      label_vec.push_back(label_score_vec[i].first);
    }

    double dcg = calc_dcg(label_vec, T);
    double ndcg = idcg_vec[q] > 0.0 ? dcg / idcg_vec[q] : 0.0;
    sum_ndcg += ndcg;
  }

  return sum_ndcg / data.size();
}

double calc_delta_dcg(
    const std::vector<std::pair<int, int> > &rank_label_vec,
    int T,
    int i,
    int j) {
  int ranki  = rank_label_vec[i].first;
  int labeli = rank_label_vec[i].second;
  int rankj  = rank_label_vec[j].first;
  int labelj = rank_label_vec[j].second;

  int pow2_array[] = {0, 1, 3, 7, 15, 31};
  double delta1 = pow2_array[labeli] - pow2_array[labelj];
  //double delta2 = ((ranki <= T) ? (1.0 / log2(ranki + 1)) : 0.0);
  //delta2 -= ((rankj <= T) ? (1.0 / log2(rankj + 1)) : 0.0);
  double delta2 = 1.0 / log2(ranki + 1) - 1.0 / log2(rankj + 1);

  return delta1 * delta2;
}


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


} // namespace

namespace toybox {
namespace ranking {

LambdaRank::LambdaRank() : T_(10), eta0_(0.001f) {
}

LambdaRank::LambdaRank(int T, float eta0) : T_(T), eta0_(eta0) {
}

LambdaRank::~LambdaRank() {
}

int LambdaRank::Train(const rank_data &data) {
  w_vec_.clear();

  int max_fid = 0;
  float sigma = 1.0f; // parameter which determines the shape of the sigmoid
  std::vector<std::vector<std::pair<int, int> > > diff_index_vec;

  make_diff_index(data, &diff_index_vec, &max_fid);
  
  if (data.size() == 0 || max_fid <= 0) { return -1; }
  w_vec_.assign(max_fid, 0.0f);

  std::vector<float> idcg_vec;
  calc_idcg(data, T_, &idcg_vec);

  float epsilon = 1.0e-10;
  float prev_ndcg = 0.0;
  float ndcg = 0.0; 

  int max_iter = 1000;
  int iter = 0;

  double eta = eta0_;

  do {
    prev_ndcg = ndcg;

    for (size_t q = 0; q < data.size(); ++q) {
      const query_data &qdata = data[q];
      const std::vector<std::pair<int, int> > qidxvec = diff_index_vec[q];

      std::vector<double> score_vec(qdata.size());
      std::vector<std::pair<int, double> > index_score_vec(qdata.size());

      for (size_t i = 0; i < qdata.size(); ++i) {
        double score = 0.0f;
        for (size_t j = 0; j < qdata[i].second.size(); ++j) {
          int   k = (qdata[i].second)[j].first;
          float v = (qdata[i].second)[j].second;
          score += w_vec_[k - 1] * v;
        }
        score_vec[i] = score;
        index_score_vec[i].first  = i;
        index_score_vec[i].second = score;
      }

      std::sort(
          index_score_vec.begin(),
          index_score_vec.end(),
          second_double_greater_than);

      std::vector<std::pair<int, int> > rank_label_vec(qdata.size());
      for (size_t i = 0; i < index_score_vec.size(); ++i) {
        int idx = index_score_vec[i].first;
        rank_label_vec[idx].first  = i + 1;
        rank_label_vec[idx].second = qdata[idx].first;
      }

      std::vector<double> lambda_vec(qdata.size());
      for (size_t k = 0; k < qidxvec.size(); ++k) {
        int i = qidxvec[k].first;
        int j = qidxvec[k].second;
        float delta = calc_delta_dcg(rank_label_vec, T_, i, j) / idcg_vec[q];

        double lambda_ij = 0.0;
        double t = sigma * (score_vec[i] - score_vec[j]);
        if (t > 35.0) { // This is magic number...
          double et = exp(-t);
          lambda_ij = (sigma * et) / (1.0 + et);
        } else {
          lambda_ij = sigma / (1.0 + exp(t));
        }

        lambda_vec[i] += lambda_ij * fabs(delta);
        lambda_vec[j] -= lambda_ij * fabs(delta);
      }

      for (size_t i = 0; i < qdata.size(); ++i) {
        float coef = eta * lambda_vec[i];
        if (coef == 0.0f) {
          continue;
        }
        for (size_t j = 0; j < qdata[i].second.size(); ++j) {
          int   f = (qdata[i].second)[j].first;
          float v = (qdata[i].second)[j].second;
          w_vec_[f - 1] += coef * v;
        }
      }

    }

    ndcg = calc_ndcg(data, w_vec_, idcg_vec, T_);
    ++iter;

    if (ndcg < prev_ndcg) {
      eta *= 0.8;
    }
    fprintf(stderr, "iter: %d, ndcg: %f, prev_ndcg: %f\n",
            iter, ndcg, prev_ndcg);
  } while(fabs(ndcg - prev_ndcg) > epsilon && iter < max_iter);

  return 1;
}

float LambdaRank::Predict(const fv_vec &x) const {
  float predicted_value = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i].first >= static_cast<int>(w_vec_.size())) {
      continue;
    }
    predicted_value += w_vec_[x[i].first-1] * x[i].second;
  }

  return predicted_value;
}

bool LambdaRank::IsInitialized() const {
  return true;
}


} // namespace ranking
} // namespace toybox
