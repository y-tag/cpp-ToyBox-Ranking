#include "ranksvm.h"
#include "ranknet.h"
#include "lambdarank.h"
#include "lambdamart.h"
#include "listnet.h"
#include "listmle.h"
#include "svmlight_reader.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <map>

//#include <google/profiler.h>

int main(int argc, char **argv) {

  if (argc < 3) {
    fprintf(stderr, "Usage: %s train_file test_file\n", argv[0]);
    return 1;
  }

  const char *train_file = argv[1];
  const char *test_file  = argv[2];

  toybox::ranking::SVMLightReader train_reader(train_file);
  toybox::ranking::SVMLightReader test_reader(test_file);
  if (! train_reader.IsInitialized()) {
    fprintf(stderr, "Fail to open train_file: %s\n", train_file);
    return 1;
  }
  if (! test_reader.IsInitialized()) {
    fprintf(stderr, "Fail to open test_file: %s\n", test_file);
    return 1;
  }

  std::vector<std::pair<int, float> > x;
  int y = 0;
  int qid = 0;

  /*
  float C = 1.0f;
  //float eta0 = 0.001;
  //float eta0 = 0.01;
  //float eta0 = 0.1;
  float eta0 = 0.05;
  int T = 10;

  //toybox::ranking::RankSVM ranking(C);
  //toybox::ranking::RankNet ranking(eta0);
  toybox::ranking::LambdaRank ranking(T, eta0);
  //toybox::ranking::ListNet ranking(eta0);
  //toybox::ranking::ListMLE ranking(eta0);
  */


  int T = 10;
  int tree_num = 10;
  float eta0 = 0.05;
  int leaf_num = 7;
  float min_leaf_instance_rate = 0.0025;
  float data_sampling_rate = 1.0;
  float feature_sampling_rate = 1.0;
  bool is_feature_sampling_randomized = false;
  toybox::ranking::LambdaMART ranking(
      T, tree_num, eta0, leaf_num, min_leaf_instance_rate,
      data_sampling_rate, feature_sampling_rate, is_feature_sampling_randomized);

  std::map<int, int> qid_index_map; 
  toybox::ranking::rank_data train_data;
  while (train_reader.Read(&x, &y, &qid) == 1) {
    if (qid_index_map.find(qid) == qid_index_map.end()) {
      qid_index_map[qid] = train_data.size();
      train_data.push_back(toybox::ranking::query_data());
    }

    int i = qid_index_map[qid];
    train_data[i].push_back(std::make_pair(y, x));
  }

  /*
  for (size_t i = 0; i < train_data.size(); ++i) {
    for (size_t j = 0; j < train_data[i].size(); ++j) {
      int label = train_data[i][j].first;
      const toybox::ranking::fv_vec &fvv = train_data[i][j].second;
      fprintf(stdout, "qid:%lu label:%d", i, label);
      for (size_t k = 0; k < fvv.size(); ++k) {
        fprintf(stdout, " %d:%f", fvv[k].first, fvv[k].second);
      }
      fprintf(stdout, "\n");
    }
  }
  */

  //ProfilerStart ("prof.out");
  ranking.Train(train_data);
  //ProfilerStop ();

  while (test_reader.Read(&x, &y, &qid) == 1) {
    float predicted_value = ranking.Predict(x);
    fprintf(stdout, "%f\n", predicted_value);
  }

  return 0;
}
