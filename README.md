# An effective multimodal representation and fusion method for multimodal intent recognition


# Dataset

You can download the full data from [Google Drive](https://drive.google.com/drive/folders/18iLqmUYDDOwIiiRbgwLpzw76BD62PK0p?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/1lAHdQ_RRaMw-DugtqRnEDg) (code：dbhc)

Dataset Description:

| Contents                       | Description                                                                                                                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| audio_data/audio_feats.pkl     | This directory includes the "audio_feats.pkl" file. It contains audio<br />feature tensors in each video segment.                                                                                                                                          |
| video_data/video_feats.pkl     | This directory includes the "video_feats.pkl" file. It contains video<br />feature tensors for all keyframes in each video segment.                                                                                                                         |
| train.tsv / dev.tsv / test.tsv | These files contain (1) video segment indexes (season, episode, clip)<br />(2) clean text utterances (3) multimodal annotations (among 20 intent<br />classes) for training, validation, and testing.                                                       |
| raw_data                       | It contains the original video segment files in .mp4 format. Among<br />directory names, "S" means season id, "E" means episode id.                                                                                                                      |
| speaker_annotations            | It contains 12,228 keyframes and corresponding manual-annotated<br />bounding box information for speakers.<br />The speaker annotations are obtained by using a pre-trained Faster <br />R-CNN to predict "person" on images and select speaker index. |

## Dependencies
* easydict==1.9
* librosa==0.9.1
* matplotlib==3.3.4
* mmcv==1.6.1
* mmdet==2.25.1
* moviepy==1.0.3
* pandas==1.1.5
* pillow==8.4.0
* scikit-learn==0.24.2
* scipy==1.5.4
* sklearn==0.0
* tokenizers==0.11.6
* tqdm==4.63.1
* transformers==4.17.0
* urllib3==1.26.9

# Citation


```
@article{huang2023effective,
  title={An effective multimodal representation and fusion method for multimodal intent recognition},
  author={Huang, Xuejian and Ma, Tinghuai and Jia, Li and Zhang, Yuanjian and Rong, Huan and Alnabhan, Najla},
  journal={Neurocomputing},
  volume={548},
  pages={126373},
  year={2023},
  publisher={Elsevier}
}
```
