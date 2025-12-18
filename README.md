# FedBiP: Heterogeneous One-Shot Federated Learning with Personalized Latent Diffusion Models

## Installation Instructions

This code has been verified with python 3.9 and CUDA version 11.7. To get started, navigate to the `InterpretDiffusion` directory and install the necessary packages using the following commands:

```bash
git clone git@github.com:HaokunChen245/FedBiP.git
cd FedBiP
pip install -r requirements.txt
pip install -e diffusers
```

## Data Preparation
Download the datasets and put them in the ``/data`` der
* DomainNet: https://ai.bu.edu/M3SDA/
* PACS: https://huggingface.co/datasets/flwrlabs/pacs
* UCM: https://huggingface.co/datasets/blanchon/UC_Merced
* OfficeHome: https://huggingface.co/datasets/flwrlabs/office-home

## Training
* Concept-level personalization
```bash
bash train.sh
```

* Image Generation
```bash
bash generate.sh
```

* Classification Model Training
```bash
bash clf_train.sh
```

---

查看中间过程图片：

```bash
 tensorboard --logdir exps_domainnet/prompt_d_clipart_multiclient/logs/FedBiP
```



