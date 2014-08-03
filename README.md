cpp-ToyBox-Ranking
==================

Now, under developing.
There are insufficient unit tests.

This library is a tiny package for learning-to-rank problems.
This library currently supports:
- RankSVM \[1\]
- RankNet \[2,7\]
- ListNet \[3\]
- ListMLE \[4\]
- LambdaRank \[5,7\]
- LambdaMART \[6,7\]

These implemented models are currently linear model, except for LambdaMART.
No non-linear kernel in SVM, no hidden layer in neural networks.

REFERENCES
----------

- \[1\] T. Joachims. (2002). Optimizing Search Engines Using Clickthrough Data. KDD 2002.
- \[2\] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. Hullender. (2005). Learning to Rank using Gradient Descent. ICML 2005.
- \[3\] Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, and H. Li. (2007). Learning to Rank: From Pairwise Approach to Listwise Approach. ICML 2007.
- \[4\] F. Xia, T.-Y. Liu, J. Wang, W. Zhang, and H. Li. (2008). Listwise Approach to Learning to Rank - Theory and Algorithm. ICML 2008.
- \[5\] C. J. C. Burges, R. Ragno, and Q. V. Le. (2006). Learning to Rank with Nonsmooth Cost Functions. NIPS 2006.
- \[6\] Q. Wu, C. Burges, K. Svore, and J. Gao. (2008). Ranking, Boosting and Model Adaptation. Microsoft Technical Report MSR-TR-2008-109.
- \[7\] C. J.C. Burges. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Technical Report MSR-TR-2010-82.

LINKS
-----

- [RankLib](http://sourceforge.net/p/lemur/wiki/RankLib/)
- [jforests](http://code.google.com/p/jforests/)


AUTHOR
------

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

LICENSE
-------

This library is distributed under the term of the MIT license.
<http://opensource.org/licenses/mit-license.php>
