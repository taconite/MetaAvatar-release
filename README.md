# MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images
This repository contains the implementation of our paper
[MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images](https://arxiv.org/abs/2106.11944).

You can find detailed usage instructions for training your own models and using pretrained models below.

If you find our code useful, please cite:

```bibtex
@InProceedings{MetaAvatar:NeurIPS:2021,
  title = {MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images},
  author = {Shaofei Wang and Marko Mihajlovic and Qianli Ma and Andreas Geiger and Siyu Tang},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2021}
}
```

## Installation
This repository has been tested on the following platform:
1) Python 3.7, PyTorch 1.7.1 with CUDA 10.2 and cuDNN 7.6.5, Ubuntu 20.04

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `meta-avatar` using
```
conda env create -f environment.yml
conda activate meta-avatar
```

(Optional) if you want to use the evaluation code under `evaluation/`, then you need to install kaolin. Download the code from the [kaolin repository](https://github.com/NVIDIAGameWorks/kaolin.git), checkout to commit e7e513173bd4159ae45be6b3e156a3ad156a3eb9 and install it according to the instructions.

## Build the dataset
To prepare the dataset for training/fine-tuning/evaluation, you have to first download the CAPE dataset from the [CAPE website](https://cape.is.tue.mpg.de/dataset).

0. Download [SMPL v1.0](https://smpl.is.tue.mpg.de/), clean-up the chumpy objects inside the models using [this code](https://github.com/vchoutas/smplx/tree/master/tools), and rename the files and extract them to `./body_models/smpl/`, eventually, the `./body_models` folder should have the following structure:
   ```
   body_models
    └-- smpl
		├-- male
		|   └-- model.pkl
		└-- female
		    └-- model.pkl

   ```
(Optional) if you want to use the evaluation code under `evaluation/`, then you need to download all the .pkl files from [IP-Net repository](https://github.com/bharat-b7/IPNet/tree/master/assets) and put them under `./body_models/misc/`. 

Finally, run the following script to extract necessary SMPL parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be save into `./body_models/misc/`.

1. Extract CAPE dataset to an arbitrary path, denoted as ${CAPE_ROOT}. The extracted dataset should have the following structure:
   ```
   ${CAPE_ROOT}
    ├-- 00032
	├-- 00096
	|   ...
	├-- 03394
	└-- cape_release

   ```
2. Create `data` directory under the project directory.
3. Modify the parameters in `preprocess/build_dataset.sh` accordingly (i.e. modify the --dataset_path to ${CAPE_ROOT}) to extract training/fine-tuning/evaluation data.
4. Run `preprocess/build_dataset.sh` to preprocess the CAPE dataset.

(Optional) if you want evaluate performance on interpolation task, then you need to process CAPE data again in order to generate processed data at full framerate. Simply comment the first command and uncomment the second command in `preprocess/build_dataset.sh` and run the script.

## Pre-trained models
We provide pre-trained [models](https://drive.google.com/drive/folders/17Mikk3KwNB5TQMLexiMw_x5o0sg-cArh?usp=sharing), including 1) forward/backward skinning networks for full pointcloud (stage 0) 2) forward/backward skinning networks for depth pointcloud (stage 0) 3) meta-learned static SDF (stage 1) 3) meta-learned hypernetwork (stage 2) . After downloading them, please put them in respective folders under `./out/metaavatar`.

## Fine-tuning fromt the pre-trained model 
We provide script to fine-tune subject/cloth-type specific avatars in batch. Simply run:
```
bash run_fine_tuning.sh
```
And it will conduct fine-tuning with default setting (subject 00122 with shortlong). You can comment/uncomment/add lines in jobs/splits to modify data splits.

## Training
To train new networks from scratch, run
```
python train.py --num-workers 8 configs/meta-avatar/${config}.yaml
```
You can train the two stage 0 models in parallel, while stage 1 model depends on stage 0 models and stage 2 model depends on stage 1 model.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ${OUTPUT_DIR}/logs --port 6006
```
where you replace ${OUTPUT_DIR} with the respective output directory.

## Evaluation
To evaluate the generated meshes, use the following script:
```
bash run_evaluation.sh
```
Again, it will conduct evaluation with default setting (subject 00122 with shortlong). You can comment/uncomment/add lines in jobs/splits to modify data splits.

## License
We employ [MIT License](LICENSE.md) for the MetaAvatar code, which covers
```
extract_smpl_parameters.py
run_fine_tuning.py
train.py
configs
jobs/
depth2mesh/
preprocess/
```
The SIREN networks are borrowed from the official [SIREN repository](https://github.com/vsitzmann/siren). Mesh extraction code is borrowed from the [DeeSDF repository](https://github.com/facebookresearch/DeepSDF).

Modules not covered by our license are:
1) Modified code from [IP-Net](https://github.com/bharat-b7/IPNet) (`./evaluation`);
2) Modified code from [SMPL-X](https://github.com/nghorbani/human_body_prior) (`./human_body_prior`);
for these parts, please consult their respective licenses and cite the respective papers.
