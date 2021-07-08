# MS^2L
MS^2L: Multi-Task Self-Supervised Learning for Skeleton Based Action Recognition in ACMMM 2020
     
# Training & Testing

Pretrain with Multi-Task Self-Supervised Learning (MS^2L).

    `python procedure with 'train_mode="pretrain"'`

Finetune with labeled data: 

    `python procedure with 'train_mode="loadweight_linear"'`

    `python procedure with 'train_mode="loadweight_finetune"'`
     
# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{lin2020ms2l,
        title       = {MS2L: Multi-Task Self-Supervised Learning for Skeleton Based Action Recognition},
        author      = {Lin, Lilang and Song, Sijie and Yang, Wenhan and Liu, Jiaying},
        booktitle   = {Proceedings of the 28th ACM International Conference on Multimedia},
        pages       = {2490--2498},
        year        = {2020}
    }
    

# Contact
For any questions, feel free to contact: `linlilang@pku.edu.cn`