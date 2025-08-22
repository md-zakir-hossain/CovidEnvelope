## CovidEnvelop Implementation
CovidEnvelope is an end-to-end pipeline that extracts features from cough recordings and applies a machine learning model to distinguish between COVID-19 positive and negative samples.

### Features

* Extracts cough signal envelope using Hilbert transform
* Computes envelope-based features from raw audio
* Trains a lightweight logistic regression classifier
* Includes example datasets and training scripts
* Outputs performance metrics (sensitivity, specificity, ROC curves)

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
