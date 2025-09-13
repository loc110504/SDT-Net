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


<table style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 6px;">Method</th>
      <th style="border: 1px solid black; padding: 6px;">LV<br>(Dice / HD95)</th>
      <th style="border: 1px solid black; padding: 6px;">RV<br>(Dice / HD95)</th>
      <th style="border: 1px solid black; padding: 6px;">MYO<br>(Dice / HD95)</th>
      <th style="border: 1px solid black; padding: 6px;">Mean<br>(Dice / HD95)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>TABNet</b></td>
      <td style="border: 1px solid black; padding: 6px;">88.18 / 1.82</td>
      <td style="border: 1px solid black; padding: 6px;">86.78 / 1.24</td>
      <td style="border: 1px solid black; padding: 6px;">92.78 / 2.48</td>
      <td style="border: 1px solid black; padding: 6px;"><b>89.25 / 1.85</b></td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>ScribbleVS</b></td>
      <td style="border: 1px solid black; padding: 6px;">87.97 / 1.47</td>
      <td style="border: 1px solid black; padding: 6px;">86.17 / 5.17</td>
      <td style="border: 1px solid black; padding: 6px;">92.80 / 1.21</td>
      <td style="border: 1px solid black; padding: 6px;">88.98 / 2.62</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>DMSPS</b></td>
      <td style="border: 1px solid black; padding: 6px;">87.98 / 1.50</td>
      <td style="border: 1px solid black; padding: 6px;">85.07 / 5.90</td>
      <td style="border: 1px solid black; padding: 6px;">92.31 / 6.55</td>
      <td style="border: 1px solid black; padding: 6px;">88.45 / 4.65</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>DMPLS</b></td>
      <td style="border: 1px solid black; padding: 6px;">87.17 / 1.76</td>
      <td style="border: 1px solid black; padding: 6px;">84.22 / 9.31</td>
      <td style="border: 1px solid black; padding: 6px;">91.69 / 6.60</td>
      <td style="border: 1px solid black; padding: 6px;">87.69 / 5.89</td>
    </tr>
  </tbody>
</table>





### Tasks
- Test ScribbleVC, Scribformer, DMSPS stage2

### Acknowledgement
This repo partially uses code from [Hilab-WSL4MIS](https://github.com/HiLab-git/WSL4MIS) and [ShapePU](https://github.com/BWGZK/ShapePU)