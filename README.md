
<!-- ## [[CVPR22] SimVQA: Exploring Simulated Environments for Visual Question Answering.](https://arxiv.org/abs/2203.17219)
Paola Cascante-Bonilla, Hui Wu, Letao Wang, Rogerio Feris, Vicente Ordonez. -->

<h1 align="center"><a href="https://arxiv.org/abs/2203.17219"> :saxophone:SimVQA: Exploring Simulated Environments<br> for Visual Question Answering</a></h1>
<h3 align="center"><a href="https://paolacascante.com/">Paola Cascante-Bonilla</a>, <a href="https://www.spacewu.com/">Hui Wu</a>, <a href="https://dw61.github.io/">Letao Wang</a>, <a href="https://www.rogerioferis.org/">Rogerio Feris</a>, <a href="https://www.cs.rice.edu/~vo9/">Vicente Ordonez</a></h3>
<h4 align="center"><a href="https://www.rice.edu/">Rice University</a>  •  <a href="https://mitibmwatsonailab.mit.edu/">MIT-IBM Watson AI Lab</a>  •  <a href="https://www.virginia.edu/">University of Virginia</a></h4>
<h5 align="center">In the 33th IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2022).</h5> 

<p align="center">
  <a href="#about">About</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#dataset-preparation">Dataset Preparation</a> •
  <a href="#train">Train/Eval</a> •
  <a href="#bibtex">Bibtex</a>
</p>


## About

<p align="center">
  <img src="https://github.com/uvavision/SimVQA/blob/main/img/showcase_reversed.gif?raw=true"  width="250">
  <img src="https://www.cs.rice.edu/~pc51/simvqa/images/drum_air2.png"  width="250">
  <img src="https://www.cs.rice.edu/~pc51/simvqa/images/drum_air_id.png"  width="250">
</p>

 We explore using synthetic computer-generated data to fully control the visual and language space, allowing us to provide more diverse scenarios for VQA. By exploiting 3D and physics simulation platforms, we provide a pipeline to generate synthetic data to expand and replace type-specific questions and answers without risking the exposure of sensitive or personal data that might be present in real images. We quantify the effect of synthetic data in real-world VQA benchmarks and to which extent it produces results that generalize to real data.

Project page: [https://www.cs.rice.edu/~pc51/simvqa/](https://www.cs.rice.edu/~pc51/simvqa/)

<br/>

## Requirements
- python >= 3.7.7 
- pytorch > 1.5.0
- torchvision
- tensorflow-gpu==1.14
- torchcontrib
- spacy >= 2.0.18
- numpy >= 1.16.2
- initialize GloVe vectors: `wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz` and then `pip install en_vectors_web_lg-2.1.0.tar.gz` <br>

*Please note: the base work of this repository is built on top of the amazing work of Zhou Yu et al. with their [Deep Modular Co-Attention Networks for Visual Question Answering](https://github.com/MILVLG/mcan-vqa) public implementation.*

<br/>

## Dataset Preparation

Following [MCAN](https://github.com/MILVLG/mcan-vqa), please use [this repo (bottom-up-attention)](https://github.com/peteanderson80/bottom-up-attention) to get a dynamic number (from 10 to 100) of 2048-D features. 

- VQAv2 Dataset:
You can download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) provided by MCAN. The downloaded files contains three files: train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz, corresponding to the features of the train/val/test images for VQA-v2, respectively.
To download the VQA splits, Questions/Answers you can run their [setup.sh](https://github.com/uvavision/SimVQA/download.sh) script.

- Synthetic Dataset:
Our generated ThreeDWorld-VQA dataset and the Question/Answers for the Hypersim dataset are stored in [this link](https://drive.google.com/drive/u/2/folders/1h-UkmjjP6jOjqF1-55HA9kmBdG1g3DDB). Please download both folders, extract their content and keep their folder structure.
To get the Hypersim Images, please follow [the dataset intructions](https://github.com/apple/ml-hypersim).    



<br/>

## Train

To train a base model using only the real images from VQAv2 please run:
```
python run.py --RUN='train' --VERSION='vqaonly_novalset' --GPU='0,1,2,3' --BS=256 --DATA_PATH='~/data/VQAv2/' --SPLIT train
```
Note that the original repository uses both the train+val sets to boost the final VQAv2 accuracy on the test split. <br>
We use only the train set and evaluate on the val set. Please make sure to follow this setup when running your experiments.

**All splits for real, counting, no counting, and additional attributes are further defined [here](https://github.com/uvavision/SimVQA/cfgs/path_cfgs.py).**

To train a model using only the counting VQA set synthetically generated using [TDW](https://github.com/threedworld-mit/tdw), you should train using the `train_no_counting_v3`**+**`tdw_count_train_33264` split as follows:
```
python run.py --RUN='train' --VERSION='vqa_nocountingv3_and_tdw' --GPU='0,1,2,3' --BS=256 --DATA_PATH='~/data/VQAv2/' --SPLIT train_no_counting_v3+tdw_count_train_33264
```

### Evaluate
For evaluation, please run:
```
python run.py --RUN='val' --CKPT_V='vqaonly_novalset' --CKPT_E=13 --BS=400 --GPU='0,1,2,3'
```
to evaluate the trained model on the VQAv2 validation set only.

<br/>

## Bibtex

If you use SimVQA for your research or projects, please cite [SimVQA: Exploring Simulated Environments for Visual Question Answering](https://arxiv.org/abs/2203.17219).

```bibtex
@InProceedings{simvqa,
    author = {Cascante-Bonilla, Paola and Wu, Hui and Wang, Letao and Feris, Rogerio and Ordonez, Vicente},
    title = {SimVQA: Exploring Simulated Environments for Visual Question Answering},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022}
}
```

Please also cite the [MCAN paper](https://arxiv.org/abs/1906.10770) and check [their work](https://github.com/MILVLG/mcan-vqa):
```bibtex
@inProceedings{yu2019mcan,
    author = {Yu, Zhou and Yu, Jun and Cui, Yuhao and Tao, Dacheng and Tian, Qi},
    title = {Deep Modular Co-Attention Networks for Visual Question Answering},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {6281--6290},
    year = {2019}
}
```

along with the hyper-realistic dataset [Hypersim](https://github.com/apple/ml-hypersim):
```bibtex
@inproceedings{roberts:2021,
    author    = {Mike Roberts AND Jason Ramapuram AND Anurag Ranjan AND Atulit Kumar AND
                 Miguel Angel Bautista AND Nathan Paczan AND Russ Webb AND Joshua M. Susskind},
    title     = {{Hypersim}: {A} Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding},
    booktitle = {International Conference on Computer Vision (ICCV) 2021},
    year      = {2021}
}
```

and the [TDW multi-modal physical simulation platform](https://www.threedworld.org/):
```bibtex
@inproceedings{NEURIPS DATASETS AND BENCHMARKS2021_735b90b4,
  author = {Gan, Chuang and Schwartz, Jeremy and Alter, Seth and Mrowca, Damian and Schrimpf, Martin and Traer, James and De Freitas, Julian and Kubilius, Jonas and Bhandwaldar, Abhishek and Haber, Nick and Sano, Megumi and Kim, Kuno and Wang, Elias and Lingelbach, Michael and Curtis, Aidan and Feigelis, Kevin and Bear, Daniel and Gutfreund, Dan and Cox, David and Torralba, Antonio and DiCarlo, James J and Tenenbaum, Josh and McDermott, Josh and Yamins, Dan},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  title = {ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation},
  year = {2021}
}
```

