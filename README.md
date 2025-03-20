# INPO
A PyTorch implementation of "[Stable Fair Graph Representation Learning with Lipschitz Constraint](https://openreview.net/pdf?id=oJQWvsStNh)"

## Overview
**INPO** is a PO-based graph unlearning method to improving the robustness of the model utility to the unlearning process. **The core idea behind INPO is to amplify the influence of unlearned edges and mitigate the tight topological coupling between the forget set and the retain set**, reducing impact on model utility when unlearning specific edges.

## Datasets
We employ standard and commonly used datasets, which you can download directly according to given links.

| Data       | Link                                                         |
| ---------- | ------------------------------------------------------------ |
| Cora       | https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz |
| DBLP       | https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz |
| PubMed     | https://github.com/abojchevski/graph2gauss/raw/master/data/pubmed.npz |
| CS         | https://github.com/shchur/gnn-benchmark/blob/master/data/npz/ms_academic_cs.npz |
| OGB-Collab | https://ogb.stanford.edu/docs/linkprop/#ogbl-collab          |

## Reproduction
To reproduce our results, please run:
```shell
bash run.sh
```

## Hyper-parameter Setting
For easy reproduction, we provide detailed hyper-parameter setting.

