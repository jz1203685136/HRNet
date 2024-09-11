# HRNet

This is the implementation of our paper "HighlightRemover: Spatially Valid Pixel Learning for ImageSpecular Highlight Removal" 

Abstract: we introduce a context-aware fusion block(CFBlock) that aggregates information in four directions, effectively capturing global contextual information. Additionally, we introduce a location-aware feature transformation module (LFTModule) to adaptively learn the valid pixels for feature reconstruction, there by avoiding information errors caused by invalid pixels. With these modules, our method can produce high-quality highlight-free re-sults without color distortion and highlight residual. Furthermore,we develop a multiple light image-capturing system to construct a large-scale highlight dataset called NSH, which exhibits minimal misalignment in image pairs and minimal brightness variation in non-highlight regions. 


## Requisites
* Python =3.8, PyTorch = 1.13.1 py3.8_cuda11.6_cudnn8.3.2_0


## Quick Start
#### Training dataset
* Modify gpu id, dataset path, and checkpoint path. Adjusting some other parameters if you like.
  
* Please run the following code: 

  ```
  sh train-refined.sh
  ```

#### Testing dataset
* Modify test dataset path and result path.
* Please run the following code: 

```
sh test-refined.sh
```



## Citation

If you find our code helpful in your research or work please cite our paper.

```
@inproceedings{zhang2024highlightremover,
  title={HighlightRemover: Spatially Valid Pixel Learning for Image Specular Highlight Removal},
  author={Zhang, Ling and Ma, Yidong and Jiang, Zhi and He, Weilei and Bao, Zhongyun and Fu, Gang and Xu, Wenju and Xiao, Chunxia},
  booktitle={ACM Multimedia 2024}
}
```

