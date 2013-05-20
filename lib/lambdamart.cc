#include "lambdamart.h"

#include <cfloat>
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

bool second_double_greater_than(
    const std::pair<int, double> &a,
    const std::pair<int, double> &b) {
  return a.second > b.second;
}

inline double log2(double x) {
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
    const std::vector<double> &fx_vec,
    const std::vector<int> &label_vec,
    const std::vector<std::pair<int, int> > &query_index_vec,
    const std::vector<float> idcg_vec,
    int T) {

  double sum_ndcg = 0.0;

  for (size_t q = 0; q < query_index_vec.size(); ++q) {
    int qstart = query_index_vec[q].first;
    int qend   = query_index_vec[q].second;

    std::vector<std::pair<int, double> > label_score_vec(qend - qstart);
    for (int i = qstart; i < qend; ++i) {
      label_score_vec.push_back(std::make_pair(label_vec[i], fx_vec[i]));
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

  return sum_ndcg / query_index_vec.size();
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
  double delta2 = 1.0 / log2(ranki + 2) - 1.0 / log2(rankj + 2);

  return delta1 * fabs(delta2);
}

void prepare_for_training(
    const toybox::ranking::rank_data &data,
    std::vector<std::vector<int> > *x_vec,
    std::vector<int> *label_vec,
    std::vector<int> *datum_vec,
    std::vector<int> *feature_vec,
    std::vector<std::vector<std::pair<int, int> > > *diff_index_vec,
    std::vector<std::pair<int, int> > *query_index_vec,
    std::vector<std::vector<double> > *id_fv_vecvec,
    int *max_fid) {

  x_vec->clear();
  datum_vec->clear();
  feature_vec->clear();
  diff_index_vec->clear();
  query_index_vec->clear();
  *max_fid = 0;

  std::set<int> fid_set;
  std::map<int, std::set<double> > fid_fv_map;

  int qstart = 0;
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
        int    f = (qdata[i].second)[j].first;
        double v = (qdata[i].second)[j].second;
        fid_set.insert(f - 1);
        fid_fv_map[f - 1].insert(v);
        if (f > *max_fid) {
          *max_fid = f;
        }
      }

      label_vec->push_back(qdata[i].first);
      datum_vec->push_back(qstart + i);
    }
    diff_index_vec->push_back(qidxvec);

    int qend = qstart + qdata.size();
    query_index_vec->push_back(std::make_pair(qstart, qend));

    qstart = qend;
  }

  feature_vec->assign(fid_set.begin(), fid_set.end());

  id_fv_vecvec->assign(*max_fid, std::vector<double>());
  std::map<int, std::map<double, int> > fid_fv_id_mapmap;

  for (size_t i = 0; i < feature_vec->size(); ++i) {
    int fid = (*feature_vec)[i];
    const std::set<double> &fv_set = fid_fv_map[fid];
    int j = 0;
    for (std::set<double>::const_iterator itr = fv_set.begin();
         itr != fv_set.end(); ++itr) {
      double v = *itr;
      ((*id_fv_vecvec)[fid]).push_back(v);
      fid_fv_id_mapmap[fid][v] = j;
      ++j;
    }
  }

  for (size_t q = 0; q < data.size(); ++q) {
    const toybox::ranking::query_data &qdata = data[q];
    for (size_t i = 0; i < qdata.size(); ++i) {
      std::vector<int> dx(*max_fid);
      for (size_t j = 0; j < qdata[i].second.size(); ++j) {
        int   f = (qdata[i].second)[j].first;
        float v = (qdata[i].second)[j].second;
        int  id = fid_fv_id_mapmap[f - 1][v];
        dx[f - 1] = id;
      }
      x_vec->push_back(dx);
    }
  }

  /*
  for (size_t i = 0; i < id_fv_vecvec->size(); ++i) {
    fprintf(stderr, "%lu ", i);
    for (size_t j = 0; j < (*id_fv_vecvec)[i].size(); ++j) {
      fprintf(stderr, "%lu:%lf ", j, (*id_fv_vecvec)[i][j]);
    }
    fprintf(stderr, "\n");
  }
  */


}


} // namespace

namespace toybox {
namespace ranking {

LambdaMART::LambdaMART()
  : T_(10), tree_num_(10), eta0_(0.001f),
    leaf_num_(10), min_leaf_instance_rate_(2.5e-4), 
    data_sampling_rate_(1.0), feature_sampling_rate_(1.0),
    is_feature_sampling_randomized_(false) {
}

LambdaMART::LambdaMART(
    int T, int tree_num, float eta0,
    int leaf_num, float min_leaf_instance_rate,
    float data_sampling_rate, float feature_sampling_rate,
    bool is_feature_sampling_randomlized)
  : T_(T), tree_num_(tree_num), eta0_(eta0),
    leaf_num_(leaf_num),
    min_leaf_instance_rate_(min_leaf_instance_rate), 
    data_sampling_rate_(data_sampling_rate),
    feature_sampling_rate_(feature_sampling_rate),
    is_feature_sampling_randomized_(is_feature_sampling_randomlized) {
}

LambdaMART::~LambdaMART() {
}

int LambdaMART::Train(const rank_data &data) {
  if (leaf_num_ < 2) {
    return -1;
  }

  tree_vec_.clear();

  std::vector<std::vector<int> > x_vec;
  std::vector<int> label_vec;
  std::vector<int> datum_vec;
  std::vector<int> feature_vec;
  std::vector<std::vector<std::pair<int, int> > > diff_index_vec;
  std::vector<std::pair<int, int> > query_index_vec;
  std::vector<std::vector<double> > id_fv_vecvec;

  prepare_for_training(
      data, &x_vec, &label_vec, &datum_vec, &feature_vec,
      &diff_index_vec, &query_index_vec, &id_fv_vecvec, &max_fid_);

  std::vector<float> idcg_vec;
  calc_idcg(data, T_, &idcg_vec);

  size_t sampled_num = datum_vec.size() * data_sampling_rate_;

  std::vector<double> fx_vec(datum_vec.size(), 0.0);
  double sigma = 1.0;

  for (int iter = 0; iter < tree_num_; ++iter) {

    std::vector<double> lambda_vec(datum_vec.size(), 0.0);
    std::vector<double> w_vec(datum_vec.size(), 0.0);
    for (size_t q = 0; q < query_index_vec.size(); ++q) {
      int qstart = query_index_vec[q].first;
      int qend   = query_index_vec[q].second;

      std::vector<std::pair<int, double> > index_score_vec(qend - qstart);
      for (size_t i = 0; i < index_score_vec.size(); ++i) {
        index_score_vec[i].first  = i;
        index_score_vec[i].second = fx_vec[qstart + i];
      }

      std::sort(
          index_score_vec.begin(),
          index_score_vec.end(),
          second_double_greater_than);

      std::vector<std::pair<int, int> > rank_label_vec(index_score_vec.size());
      for (size_t i = 0; i < index_score_vec.size(); ++i) {
        int idx = index_score_vec[i].first;
        rank_label_vec[idx].first  = i;
        rank_label_vec[idx].second = label_vec[qstart + idx];
      }

      const std::vector<std::pair<int, int> > qidxvec = diff_index_vec[q];
      for (size_t k = 0; k < diff_index_vec[q].size(); ++k) {
        int i = qidxvec[k].first;
        int j = qidxvec[k].second;
        float delta = calc_delta_dcg(rank_label_vec, T_, i, j) / idcg_vec[q];

        double rho_ij = 0.0;
        double t = sigma * (fx_vec[qstart + i] - fx_vec[qstart + j]);
        if (t > 35.0) { // This is magic number...
          double et = exp(-t);
          rho_ij = (sigma * et) / (1.0 + et);
        } else {
          rho_ij = sigma / (1.0 + exp(t));
        }
        double lambda_ij = rho_ij * delta;
        lambda_vec[qstart + i] += lambda_ij;
        lambda_vec[qstart + j] -= lambda_ij;

        double w_ij = rho_ij * (1.0 - rho_ij) * delta;
        w_vec[qstart + i] += w_ij;
        w_vec[qstart + j] += w_ij;

        //fprintf(stderr, "rho: %lf, lambda: %lf, w: %lf\n", rho_ij, lambda_ij, w_ij);
      }
    }
    
    /*
    for (size_t i = 0; i < lambda_vec.size(); ++i) {
      fprintf(stderr, "%lu:%lf\n", i, lambda_vec[i]);
    }
    */

    toybox::ensemble::RegressionTree tree(
        leaf_num_, min_leaf_instance_rate_,
        feature_sampling_rate_, is_feature_sampling_randomized_
    );

    std::vector<int> sampled_datum_vec;
    if (sampled_num == datum_vec.size()) {
      sampled_datum_vec.assign(datum_vec.begin(), datum_vec.end());
    } else {
      for (size_t j = 0; j < sampled_num; ++j) {
        int idx = rand() % datum_vec.size();
        sampled_datum_vec.push_back(datum_vec[idx]);
      }
      std::sort(sampled_datum_vec.begin(), sampled_datum_vec.end());
    }

    std::random_shuffle(feature_vec.begin(), feature_vec.end());

    tree.Train(x_vec, lambda_vec, id_fv_vecvec, sampled_datum_vec, feature_vec);

    std::vector<double> nume_vec(leaf_num_, 0.0);
    std::vector<double> deno_vec(leaf_num_, 0.0);
    std::vector<int> num_vec(leaf_num_, 0);
    for (size_t i = 0; i < datum_vec.size(); ++i) {
      int idx = datum_vec[i];
      int node = tree.GetCorrespondingLeafNode(x_vec[idx]);
      nume_vec[node] += lambda_vec[idx];
      deno_vec[node] += w_vec[idx];
      num_vec[node] += 1;
    }

    for (size_t node = 0; node < nume_vec.size(); ++node) {
      double gamma = nume_vec[node] / deno_vec[node];
      tree.SetLeafNodeValue(node, eta0_ * gamma);
    }

    for (size_t i = 0; i < datum_vec.size(); ++i) {
      int idx = datum_vec[i];
      fx_vec[idx] += tree.Predict(x_vec[idx]);
    }

    //tree.Print();
    tree_vec_.push_back(tree);

    double ndcg = calc_ndcg(fx_vec, label_vec, query_index_vec, idcg_vec, T_);
    fprintf(stderr, "iter: %d, ndcg: %f\n", iter, ndcg);

  }


  return 1;
}

float LambdaMART::Predict(const fv_vec &x) const {
  toybox::ensemble::datum dx(max_fid_);

  for (size_t i = 0; i < x.size(); ++i) {
    int   f = x[i].first;
    float v = x[i].second;
    if (f < max_fid_) {
      dx[f] = v;
    }
  }

  double predicted_value = 0.0f;
  for (size_t i = 0; i < tree_vec_.size(); ++i) {
    predicted_value += tree_vec_[i].Predict(dx);
  }

  return predicted_value;
}

bool LambdaMART::IsInitialized() const {
  return true;
}


} // namespace ranking
} // namespace toybox
