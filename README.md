# SNNs-Self-Normalizing-Neural-Networks-Caffe-Reimplementation
Personal Caffe reimplementation of SNNs([arXiv pre-print Link](https://arxiv.org/abs/1706.02515)).

## SeLu Layer
*if x>0*  <br>
  *`selu(x) = lambda*x `*  <br>
*else*  <br>
  *`selu(x) = lambda*alpha*(exp(x)-1) `* <br>
<br>
## SeLu Dropout Layer
**dropout_ratio = 1 - q** <br>
*if random > dropout_ratio* <br>
*`selu_drop(x) = a*(x)+b`* <br>
*else*<br>
*`selu_drop(x) = a*(alpha)+b`*

## How to use?
**.prototxt**
```
layer {
  name: "SNNlayer"
  type: "SeLu"
  bottom: "conv"
  top: "conv"
  selu_param {
    alpha : xxxxx # default 1.67326324
    lambda : xxxxx # default 1.050700987
  }
}

layer {
  name: "SNNDropoutlayer"
  type: "SeLuDropout"
  bottom: "conv"
  top: "conv"
  selu_dropout_param {
    alpha : xxxxx # default -1.75809934
    dropout_ratio : xxxxx # default 0.1
  }
}

```

**Edit caffe.proto**
```
# EDIT
message LayerParameter {
...
optional SliceParameter slice_param = 126;
optional SeLuParameter selu_param = 161; # HERE !!
optional SeLuDropoutParameter selu_dropout_param = 162; # HERE !!
optional TanHParameter tanh_param = 127;
...
}

...
# ADD 
message SeLuParameter {
  optional float lambda = 1 [default = 1.050700987];
  optional float alpha = 2 [default = 1.67326324];
}

message SeLuDropoutParameter {
  optional float dropout_ratio = 1 [default = 0.1]; // dropout ratio  recommend 0.05 or 0.1
  optional float alpha = 2 [default = -1.75809934];
}
...
# EDIT (omissible)
message V1LayerParameter {
...
SPLIT = 22;
SLICE = 33;
SELU = 53; # HERE !!
SELU_DROPOUT = 54; # HERE !!
TANH = 23;
...
}
```
