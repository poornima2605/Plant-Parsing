# Data Prepare

This document covers how to prepare training and evaluating data for QANet, for Green AI dataset.



# Data Structure

  Recommended downloading the data format prepared.

  - [Green AI] with  files in data_prepare

  Make sure to put the files as the following structure:

  ```
  ├─cfgs
  ├─ckpts
  ├─data
  │  ├─CIHP
  │  │  ├─Training
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─Validation
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Plants
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─annotations
  │  │  │  ├─GreenAI_train.json
  │  │  │  ├─GreenAI_val.json
  |  |  |  |─GreenAI_test.json
  ├─docs
  ├─instance
  ├─lib
  ├─tools
  ├─weights
     ├─resnet50c-pretrained.pth

  ```
