# Sparseout: Controlling Sparsity in Deep Networks

## Example
```python
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sparseout import sparseout

input = Variable(torch.randn(10, 5))
# Dropout
output_dropout = F.dropout(input, p=0.5)
# Sparsout
output_sparseout = sparseout(input, p=0.5, q=2.0)
```
For `q=2` Sparseout and Dropout are equivalent. For `q<2` Sparseout results in sparser activations, while for `q>2` Sparseout results in dense/non-sparse activations compared to Dropout. 
## Citation
```
@ARTICLE{khan2019sparsout,
       author = {Khan, Najeeb and Stavness, Ian},
        title = "{Sparseout: Controlling Sparsity in Deep Networks}",
      journal = {arXiv e-prints},
         year = "2019",
      url = {https://arxiv.org/abs/1904.08050}
}
```
