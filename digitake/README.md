# Thyroid research
------

Data Preprocessing and Auxiliary functions

------
0.12
- Refactor code and add ShowPredCallBack, Resnet_multichannel and their accompany functions

0.11
- Add with_alpha_channel option to ThyroidDataSet class to specify weather RGB or RGBA mode

0.10
- Add masking enable data loader
- Fix KeyError handling when mask is not present
- Fix ProgressMeter __str__ formater to use meters.batch

0.9
- Add `eval` for a custom validation(test)
- Refactor before stable API bump(v1)
- Reduce redundancy specification
- Make build dataset taking both path(string) and list of path

0.8
- Add build_dataset function
- Add save_model and load_model
- Adjust image augmentation to include perspective random
- Send out preds from one batch as well
- Update ThyroidDataset to return class_index
- Fix formatter
- Fix callback on start batch
- Add ProgressBar `tqdm` to train method
- Add `data_points` to AverageMeter
- Create a ModelTrainer class to wrap train/val functions

0.7
- Update __str__ for AverageMeter class to show only avg value
- Update preprocess __getitem__ to return class_num instead of class name
- Add test case on average meter
- Add extra parameter to take in transform function for Dataset
- Add set_reproducible function
- Add AverageMeter and ProgressMeter from Imagenet example
- Make the in-place fully-connected layer change weight and bias as well
- Add model last Linear layer search and replace function

0.6
- Remove InterpolationMode warning
- Minor bug fixing
- Fix ThyroidDataset loader, make it keep track of label
- Add display_sample function 

0.3.0
- expose submodule for import

0.2.0
- restructure and add preprocess module 

0.1.0
- add labnote package
- add preprocess package
