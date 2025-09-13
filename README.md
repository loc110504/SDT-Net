# Scribble-Supervised Learning for Medical Image Segmentation

This repository provides re-implementations of some papers about scribble-supervised  for medical image segmentation:

## Related Papers

1. [DMPLS](https://arxiv.org/pdf/2203.02106) — *MICCAI 2022*  
   ✅ Status: Done  

2. [ShapePU](https://arxiv.org/pdf/2206.02118) — *MICCAI 2022*  
   ⚠️ Status: Bug  

3. [UAMT](https://www.sciencedirect.com/science/article/pii/S0031320321005215) — *Pattern Recognition 2022*  
   ✅ Status: Code done, not yet run  

4. [ScribbleVC](https://arxiv.org/pdf/2307.16226) — *ACM MM 2023*  
   ✅ Status: Done  

5. [ScribFormer](https://arxiv.org/pdf/2402.02029) — *IEEE TMI 2024*  
   ✅ Status: Done  

6. [DMSPS](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid=author) — *MedIA 2024*  
   ✅ Status: Stage1 done, Stage2 pending  

7. [ScribbleVS](https://arxiv.org/pdf/2411.10237) — *arXiv 2024*  
   ✅ Status: Done  

8. [TABNet](https://arxiv.org/pdf/2507.02399) — *arXiv 2025*  
   ✅ Status: Done  


## 📊  Benchmark on ACDC


| Method        | LV Dice ↑ | LV HD95 ↓ | RV Dice ↑ | RV HD95 ↓ | MYO Dice ↑ | MYO HD95 ↓ | **Mean Dice ↑** | **Mean HD95 ↓** |
|---------------|-----------|-----------|-----------|-----------|-------------|-------------|-----------------|-----------------|
| **TABNet**    | 88.18     | 1.82      | 86.78     | 1.24      | 92.78       | 2.48        | **89.25**       | **1.85**        |
| **ScribbleVS**| 87.97     | 1.47      | 86.17     | 5.17      | 92.80       | 1.21        | **88.98**       | **2.62**        |
| **DMSPS**     | 87.98     | 1.50      | 85.07     | 5.90      | 92.31       | 6.55        | **88.45**       | **4.65**        |
| **DMPLS**     | 87.17     | 1.76      | 84.22     | 9.31      | 91.69       | 6.60        | **87.69**       | **5.89**        |





### Tasks
- Test ScribbleVC, Scribformer, DMSPS stage2

### Acknowledgement
This repo partially uses code from [Hilab-WSL4MIS](https://github.com/HiLab-git/WSL4MIS) and [ShapePU](https://github.com/BWGZK/ShapePU)