# Conceptual Compression via Deep Structure and Texture Synthesis
The codes of paper "Conceptual Compression via Deep Structure and Texture Synthesis", which is accepted by Transactions on Image Processing. The paper address is <a href="https://arxiv.org/pdf/2011.04976.pdf">Conceptual Compression via Deep Structure and
Texture Synthesis
</a>.

## Citation
If you find it useful for your research, please cite as following:

>@article{chang2022conceptual,
  title={Conceptual compression via deep structure and texture synthesis},
  author={Chang, Jianhui and Zhao, Zhenghui and Jia, Chuanmin and Wang, Shiqi and Yang, Lingbo and Mao, Qi and Zhang, Jian and Ma, Siwei},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={2809--2823},
  year={2022},
  publisher={IEEE}s
}
 
## Pretrained models 
The pretrained models for edges2shoes&handbags, celebaHD, alps_seasons three datasets can be down loaded at onedrive [here](https://pkueducn-my.sharepoint.com/:f:/g/personal/jhchang_pku_edu_cn/Eus_4gwN3MtAlzu5Rh3CVTQBJHl-wPy5aI41Wtf9W7rLDA?e=UdXrdy).


## About edge map
The edge maps are extracted with canny algorithm. Related parameters are setting as follows.

```
import cv2

def edge_extract(fname):
    img=cv2.imread(fname)
    if img is None:
        return None
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges=cv2.GaussianBlur(img_gray,(5,5),0)
    edges=cv2.Canny(edges,50,150)
    return edges
```
