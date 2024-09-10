# LFAS-CFMMF
This is a lightweight multi-modal FAS model, the theory comes from the paper "Lightweight Face Anti-Spoofing Method based on Cross-Fused Multi-modal Features".

# Dependencies
```bash
yacs == 0.1.8  
timm == 0.5.4  
imgaug == 0.4.0   
numpy == 1.23.3  
python == 3.8.13    
torch == 1.8.2+cu111  
torchaudio == 0.8.2  
torchsummary == 1.5.1  
torchvision == 0.9.2+cu111  
opencv-python == 4.6.0.66  
opencv-python-headless == 4.6.0.66  
opencv-contrib-python-headless == 4.6.0.66
```

# To train
```bash
# Multimodal
python main_FAS.py --model=ShffleNetV2_hd_v1 --batch_size=64 --image_size=64 --dataset_name=WMCA --prot=prints --is_Multi=True
# Single-modal
python main_FAS.py --model=ShffleNetV2_hd_v1 --batch_size=64 --image_size=64 --dataset_name=WMCA --prot=prints--image_modality=thermal
```

# To Test
Specify the name of the saved parameter file.  
```bash
# Multimodal
python main_FAS.py --model=ShffleNetV2_hd_v1 --batch_size=64 --image_size=64 --dataset_name=WMCA --prot=fakehead --is_Multi=True --mode=infer_test --pretrained_model = r'test_min_acer_model_20230726_06_48_22
Testing all the saved models under this protocol.
# Single-modal
python main_FAS.py --model=ShffleNetV2_hd_v1 --batch_size=64 --image_size=64 --dataset_name=WMCA --prot=rigidmask --is_Multi=True --mode=infer_test
```

## The code refers to:
https://github.com/SoftwareGift/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019  
https://github.com/SeuTao/CVPR19-Face-Anti-spoofing  

## Citation
If you find our work useful, please cite us:
```bibtex
@article{He2024LightweightFA,
  title={Lightweight face anti-spoofing method based on cross-fused multi-modal features},
  author={Xiping He and Yi Li and Dan He and Rui Yuan and Ling Huang},
  journal={Journal of Electronic Imaging},
  year={2024},
  volume={33},
  pages={023033 - 023033},
  url={https://api.semanticscholar.org/CorpusID:268646302}
}
```
    
If you have any questions, feel free to E-mail me via: hedan123987@163.com
