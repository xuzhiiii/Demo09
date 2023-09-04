

### main_files:
./configs/Distill.yaml  <br/>
./dataset/\__init__.py          (test_xmai_file) <br/>
./models/model_distill.py       (KD + EMD) <br/>
./models/vit.py                 (noise) <br/>
./models/xbert.py               (noise) <br/>
Distill.py                      (main file) <br/>


### need to change:
1. download dataset (and move to ./data/)
   1. [flickr30k dataset](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/flickr30k.zip)
2. download pretrain_models
   1. teacher model: [flickr30k.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth)
      2. change Distill.py --teacher_ckpt and --student_ckpt
   2. bert ("bert-base-uncased"), and change Distill.yaml ("text_encoder":XXXX)


### run command:
python -m torch.distributed.launch --nproc_per_node=2 --use_env Distill.py

