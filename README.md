# CoSMoS: Controllable Single Motion Synthesis

Project Page : https://childtoy.github.io/csms.github.io/



## Environment 
* Ubuntu 18.04.5 LTS
* Python 3.11
* Pytorch 
Setup conda env:
```shell
conda env create -f environment.yml
```

Install ganimator-eval-kernel by following [these](https://github.com/PeizhuoLi/ganimator-eval-kernel) instructions,  OR by running: 
```shell 
pip install git+https://github.com/PeizhuoLi/ganimator-eval-kernel.git
```

## Preparations
### Get Data and Unconditional Model
Download by following [these](https://github.com/SinMDM/SinMDM) instructions for mixamo and humanML3D dataset and pretrained models
```shell 
cd ../
git clone https://github.com/SinMDM/SinMDM
```

### Training Process
Our method consists of two stages:
Stage 1: Training Dense Label Classification Model and Creating Synthesis Dataset
Train the dense label classification model:
```shell
python3 -m densecls.train_dense_cls --dataset humanml --sin_path HUMANML3D-NPY-PATH --output_path save/densecls/humanml/0000 --device 0 

```
Create synthesis dataset with dense labels:
```shell
python3 -m sample.create_synthesis_dataset --dataset humanml --sin_path HUMANML3D-NPY-PATH --model_path ../SinMDM/save/humanml/0000/model000020000.pt --cls_model_path DENSECLS-MODEL-PATH --num_samples 100 --batch_size 100 --device 0
```

Stage 2: Training Densely Conditioned UNet for Dense Conditioned Generation

```shell
python3 -m train.train_cosmos --dataset humanml --pkl_path save/humanml/0000/synthesis_dataset.pkl --save_dir save/humanml/cosmos/0000 --device 0 --lr_method ExponentialLR --lr_gamma 0.99998 --use_scale_shift_norm
```

### Run Inference command
Our model allows for flexible motion generation with precise control over timing and structure. To generate motions using dense labels, we provide the sample.generate_cosmos.py script. This powerful tool enables you to create diverse motion sequences with varying lengths and characteristics.
Key features of our generation process:

Dense Label Control: Use detailed frame-by-frame labels to guide the motion generation.
Flexible Sequence Length: Generate motions of any desired length, not limited by the original sample.
Timing Adjustments: Fine-tune the timing of specific motion segments within the sequence.
Semantic Consistency: Maintain the intended meaning and structure of the motion while introducing variations.
```shell
python3 -m sample.generate_cosmos --model_path DENSELYCONDTIONEDUNET-MODEL-PATH --sin_path HUMANML3D-NPY-PATH --num_samples 100 --batch_size 100 
```
### Additional Info
For more details, please refer to the following shell scripts:

Please see the shell files. 
train_densecls.sh
create_synthesis_dataset.sh
train_cosmos.sh




## Acknowledgments
We extend our heartfelt thanks to the creators of SinMDM (https://github.com/SinMDM/SinMDM) for their inspiring work and open-source contributions, which greatly influenced our research. 
[SinMDM](https://github.com/SinMDM/SinMDM).

## License
This code is distributed under the [MIT LICENSE](LICENSE).
