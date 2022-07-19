# On the Practicality of Deterministic Epistemic Uncertainty

This is the code for our ICML 2022 paper **On the Practicality of Deterministic Epistemic Uncertainty** ([paper](https://arxiv.org/abs/2107.00649)).

## Setup

### Environment

Install conda environment:
```shell script
conda env create -f environment.yml
```

### Datasets

For training and evaluation of semantic segmentation follow
[the official instructions](https://www.tensorflow.org/datasets/catalog/cityscapes)
to manually download cityscapes. This typically means downloading the 
dataset manually and placing it in ~/tensorflow_datasets/downloads/manual


## Model training

We refer to the paper for details on each method.

### Image Classification 

Train model on image classification using:

```shell script
python train_image_classifier.py 
    --data_root your_data_root_folder 
    --exp_root your_experiment_root_folder 
    --dataset cifar10/cifar100 
    --backbone resnet 
    --method softmax/dropout/duq/mir/ddu/sngp
```

### Semantic Segmentation

Train model on image classification using:

```shell script
python train_semantic_segmentation.py 
    --data_root your_data_root_folder 
    --exp_root your_experiment_root_folder 
    --dataset cityscapes 
    --backbone drn_a_50 
    --method softmax/dropout/mir/ddu/sngp
```

## Evaluation

We evaluate two different characteristics of epistemic uncertainty - OOD detection and calibration. 

### Image Classification

```shell script
python evaluate_image_classifier.py 
    --exp_name experiment_name 
    --exp_root experiment_root
    --evaluate_calibration=true/false
    --use_corrupted=true/false
    --normalize_features=true/false
    --evaluate_ood_detection=true/false
```

**experiment_root**: Your root folder containing all experiments.

**experiment_name**: Folder of the experiment which you want to evaluate. Located in experiment_root.

**evaluate_calibration**: Whether to evaluate calibration

**use_corrupted**: Whether we use standard corruption datasets (e.g. CIFAR10/100-C).

**normalize_features**: Whether we normalize features to unit length for MIR/DDU. Was found to generally yield better performance.

**evaluate_ood_detection**: Whether to evaluate OOD detection

The evaluation script yields OOD detection performance on variety of OOD datasets and calibration results. The results will be saved in the "resutls" folder located in experiment_root.


### Semantic Segmentation

Evaluates predictive performance and calibration on semantic segmentation.

```shell script
python evaluate_image_classifier.py 
    --exp_name experiment_name 
    --exp_root experiment_root
    --mcd=true/false
    --mir=true/false
```

**mcd**: Whether to use MC dropout for uncertainty quantification.

**mir**: Whether to use MIR for uncertainty quantification.

## Citation

```
@article{postels2021practicality,
  title={On the practicality of deterministic epistemic uncertainty},
  author={Postels, Janis and Segu, Mattia and Sun, Tao and Van Gool, Luc and Yu, Fisher and Tombari, Federico},
  journal={International Conference on Machine Learning},
  year={2022}
}
```
