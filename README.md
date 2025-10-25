# Fork implementation of 'Detecting Adversarial Faces Using Only Real Face Self-Perturbations' for thesis


### Generate adversarial faces

Provide original image dataset.
Use conda environment 'adv' and run with only one GPU, since otherwise a deadlock occurs.
```
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -m make_adv --file_list /path/to/list.txt --server False --batch_size 16 --dataset LFW
```

### Train Detector

Donwload model checkpoint and place in results/checkpoints folder. Do the same for pretrained XceptionNet checkpoint.
Add path to checkpoint in SP_train.sh or in configs/pipelines/test/SP_train.yml.
```
sh SP_train.sh
```

### Test

Use conda 'base' environment.
Add paths to adversarial faces and specify which attacks to test in configs/datasets/SP_LFW.yml.

```
sh SP_test.sh
```


```
## Dependencies
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
```


From the original repository:

### Citation

If you find our repository useful for your research, please consider citing our paper:
```
@inproceedings{ijcai2023p165,
  title     = {Detecting Adversarial Faces Using Only Real Face Self-Perturbations},
  author    = {Wang, Qian and Xian, Yongqin and Ling, Hefei and Zhang, Jinyuan and Lin, Xiaorui and Li, Ping and Chen, Jiazhong and Yu, Ning},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {1488--1496},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/165},
  url       = {https://doi.org/10.24963/ijcai.2023/165},
}
```
