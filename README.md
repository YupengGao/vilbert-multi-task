# Multi-Task Vision and Language Representation Learning (ViLBERT-MT)

Please cite the following if you use this code. Code and pre-trained models for [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315):

```
@article{lu201912,
  title={12-in-1: Multi-Task Vision and Language Representation Learning},
  author={Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
  journal={arXiv preprint arXiv:1912.02315},
  year={2019}
}
```

and [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265):

```
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Data Setup

Check `README.md` under `data` for more details.  

## Visiolinguistic Pre-training and Multi Task Training

### Pretraining on Conceptual Captions

```
python train_concap.py --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --train_batch_size 512 --objective 1 --file_path <path_to_extracted_cc_features>
```
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin)

### Multi-task Training

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <pretrained_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1-2-4-7-8-9-10-11-12-13-15-17 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name multi_task_model
```

[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin)


### Fine-tune from Multi-task trained model

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model
```
 
## License

vilbert-multi-task is licensed under MIT license available in [LICENSE](LICENSE) file.



## vqa-maskrcnn-benchmark has already included the maskrcnn_benchmark repo, so we don’t need to download a new maskrcnn-benchmark. If we download a new maskrcnn-benchmark from master branch git clone https://github.com/facebookresearch/maskrcnn-benchmark.git. It will raise error when we run script/extract_features.py
```
conda create --name maskrcnn_benchmark -y

conda activate maskrcnn_benchmark

Conda install python==3.7

conda install ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python

#install pytorch1.4 torchvivion==0.5.0 cudatoolkit==10.1

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

export INSTALL_DIR=$PWD

#install pycocotools

cd $INSTALL_DIR

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

python setup.py build_ext install

#install cityscapesScripts

cd $INSTALL_DIR

git clone https://github.com/mcordts/cityscapesScripts.git

cd cityscapesScripts/

python setup.py build_ext install

#install apex

cd $INSTALL_DIR

git clone https://github.com/NVIDIA/apex.git

cd apex

python setup.py install --cuda_ext --cpp_ext

#install PyTorch Detection

cd $INSTALL_DIR

git clone https://github.com/facebookresearch/maskrcnn-benchmark.git #we don't need to do this if we are installing vqa-maskrcnn-benchmark

cd maskrcnn-benchmark

#the following will install the lib with
#symbolic links, so that you can modify
#the files if you want and won't need to
#re-build it

python setup.py build develop


unset INSTALL_DIR





python script/extract_features.py --model_file /dccstor/yupeng_storage/conceptual-captions/detectron_model.pth --config_file /dccstor/yupeng_storage/conceptual-captions/detectron_config.yaml --image_dir /dccstor/yupeng_storage/conceptual-captions/validation --output_folder /dccstor/yupeng_storage/conceptual-captions/extract_feature_validation/

jbsub -cores 8+1 -mem 64G -queue x86_6h -interactive -require “v100” bash
```
