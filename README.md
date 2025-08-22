## CovidEnvelop Implementation

<img width="20%" src="pic/CovidEnvelop.jpg" />

## Dependencies
```bash
pip install librosa pylab audio2numpy signal_envelope
```

## Datasets
To get Russian dataset:

```bash
git clone https://github.com/covid19-cough/dataset.git
```
## Training & Evaluation

Train the model on the dataset:   

```bash
python train.py
```

## Citation
If you find this work useful for your research, please cite [paper](https://ieeexplore.ieee.org/document/9718501):
```
@inproceedings{hossain2021covidenvelope,
  title={Covidenvelope: an automated fast approach to diagnose covid-19 from cough signals},
  author={Hossain, Md Zakir and Uddin, Md Bashir and Yang, Yan and Ahmed, Khandaker Asif},
  booktitle={2021 IEEE Asia-Pacific Conference on Computer Science and Data Engineering (CSDE)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
