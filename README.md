# b-DCCA for Bird Sound Detection

This repository contains the code base used to implement experiments described in the paper:  
BALANCED DEEP CANONICAL CORRELATION ANALYSIS FOR BIRD SOUND DETECTION by Sumit Kumar et al.

## Dataset

The real-time bird sound dataset named TwoRadioBird is collected using the synchronized microphone and accelerometer sensors for the bird sound detection task. Accelerometers are mounted on a male and female Zebra Finch bird. Instances of the sound events are labeled by annotators, i.e, the dataset is strongly labeled. The dataset consists of eleven data files of 1 hr. duration each, out of which three are la-
beled and eight are unlabeled. Each file contains data from a microphone (audio) and two accelerometer channels (sound-pressure).The data in both channels is sampled at a rate of 24000 Hz.

## Structure

The following tree shows the structure of this file system:

```bash
.
├── data
│   ├── binning                         # contains binning results in form of csv
│   │   └── spec_binned_linear.csv
│   ├── embeddings                      # contains combined spectrograms or DCCA embeddings
│   │   ├── balanced_acc_spectrograms.npy
...
│   │   └── new_feat_1410_self_.npy
│   ├── labelled                        # contains some samples of labelled dataset used
│   │   ├── labels
│   │   │   ├── b8p2male-b10o15female_87_SdrChannels_870.npy
...
│   │   │   └── b8p2male-b10o15female_87_SdrChannels_879.npy
│   │   ├── acc
│   │   │   ├── b8p2male-b10o15female_87_SdrChannels_870.npy
...
│   │   │   └── b8p2male-b10o15female_87_SdrChannels_879.npy
│   │   └── mic
│   │       ├── b8p2male-b10o15female_87_SdrChannels_870.npy
...
│   │       └── b8p2male-b10o15female_87_SdrChannels_879.npy
│   └── unlabelled                      # contains some samples of unlabelled dataset used
│       ├── acc
│       │   ├── b8p2male-b10o15female_88_SdrChannels_870.npy
...
│       │   └── b8p2male-b10o15female_88_SdrChannels_879.npy
│       └── mic
│           ├── b8p2male-b10o15female_88_SdrChannels_870.npy
...
│           └── b8p2male-b10o15female_88_SdrChannels_879.npy
├── models                              # models mentioned in the paper
│   ├── CRNN_binning_cls.h5                 # CRNN trained with DCCA embeddings using single modal (mic only)
│   ├── CRNN_cls_dcca_83.h5                 # CRNN trained with DCCA embeddings using dual modal
│   ├── dcca_with_binning.h5                # DCCA trained using binning
│   └── dcca_without_binning.h5             # DCCA trained without binning
├── Classifier
│   ├── __init__.py
│   ├── classifier_train_test.py
│   ├── model.py
│   └── utils.py
├── DCCA
│   ├── __init__.py
│   ├── binning.py
│   ├── create_embeddings.py
│   ├── dcca_train_test.py
│   ├── group_spectrograms.py
│   ├── model.py
│   ├── objective.py
│   └── utils.py
├── env.py                                  # contains environment variables
└── README.md
```

### DCCA

- `binning.py [path to trained CRNN .h5]` : uses accelerometer data trained CRNN to bin unlabelled spectrograms. You can change the binning distribution by changing `binning_type`. We've used linear mapping (from logit distribution to binning distribution) distribution throughout
- `create_embeddings.py [path to trained DCCA .h5] [path to merged mic spectrograms file] [path to merged acc spectrograms file]` : extracts and saves DCCA embeddings by taking in 2 modalities
- `dcca_train_test.py [path to binning details csv]` : change unlabelled files' paths mentioned in `env.py`. Check out `metadata_to_file` function in `DCCA/utils.py` for matching file paths and name of your spectrograms. Trains DCCA using unlabelled 2 modal data
- `group_spectrograms.py` : creates merged spectrograms and labels files. Adjust paths in `env.py`. It also increases number of spectrograms by augmenting spectrograms having most positive labels.
- `model.py` : defines the architecture of DCCA used to extract "better" embeddings
- `objective.py` : defines the loss function used to train DCCA
- `utils.py` : contains helper functions

### Classifier

- `classifier_train_test.py [path to merged features file] [path to merged labels file]` : Accepts either spectrograms or DCCA embeddings, with all files clumped together as a single file loading the entire dataset into RAM.
- `model.py` : defines the architecture of CRNN used as the classifier for bird sound detection
- `utils.py` : contains helper functions

Recommended: Run each file by being in that corresponding directory. Paths are adjusted according to that.

## Contributors

- [sumit7692](https://github.com/sumit7692)
- [ba-13](https://github.com/ba-13/)
