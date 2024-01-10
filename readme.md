# CBPF:Class Balanced Power Focal Loss
CBPF is an improved loss function for class imbalanced small datasets.

This project is an open-source content related to the prediction model of postoperative hepatic encephalopathy in TIPS surgery.Please read our published paper for detailed information.
## Install
Clone the repo:

```bash
git clone https://github.com/flashxin/CBPF-TIPS-HE.git
```
## File directory organization
```bash
└── Train
          ├── Alice
              ├── Liver
                ├──XXX1.npy
                ├──XXX2.npy
                        :
                └──XXXn.npy
              ├── Lumbar
                ├──XXX1.npy
                ├──XXX2.npy
                        :
                └──XXXn.npy
          ├── Bob
                    :
          └── name
```
## Custom datasets
The original code is an algorithm designed for thin layer CT scanning,if you want to use your own dataset or generalize our methodology, you need to complete follow steps.
+ Adjust the file directory to meet the code requirements
+ Modify the path to the dataset in yaml
+ The dcm/png/jpg file needs to be converted to an npy file in advance. The script for converting DCM to NPY already includes

## Citation
- If you found our work useful in your research, please consider citing our work at:
```
    #No published paper, will be updated later
```