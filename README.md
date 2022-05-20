# ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation 
This repository provides source code for ECONet, an online likelihood method for scribble-based interactive segmentation. If you use this code, please cite the following paper:

- **Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. ["ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation."](https://openreview.net/forum?id=9xtE2AgD_Cc), Medical Imaging with Deep Learning (MIDL) 2022.**

#  Introduction
A challenge when looking at annotating lung lesions associated with COVID-19 is that the lung lesions have large inter-patient variations, with some pathologies having similar visual appearance as healthy lung tissues. This poses a challenge when applying existing semi-automatic interactive segmentation methods for data labelling. To address this, we propose an efficient convolutional neural networks (CNNs) that can be learned online while the annotator provides scribble-based interaction. 

![econet-flowchart](https://raw.githubusercontent.com/masadcv/ECONet-MONAILabel/main/data/model-ECONetFlowchart.png)

The flowchart above shows (a) patch-based online training of ECONet where a patch of size K $\times$ K $\times$ K, extracted around a scribble voxel, is used. The loss function in paper is used along with label from scribble to learn the model parameters. (b) shows online likelihood inference using ECONet as fully convolutional network on full image volume.

Further details about ECONet can be found in our paper linked above.

# Methods Included
In addition to ECONet, we include all comparison methods used in our paper. These are summarised in table below:
| Method Name                | Description                      |
|----------------------------|----------------------------------|
| ECONet + GraphCut          | ECONet (proposed) based likelihood inference               |
| ECONet-Haar + GraphCut     | ECONet with Haar-Like features for likelihood inference             |
| DybaORF-Haar + GraphCut    | DybaORF [1] with Haar-Like features for likelihood inference|
| GMM + GraphCut             | GMM-based [2] likelihood generation      |
| Histogram + GraphCut       | Histogram-based [3] likelihood generation |

[1] Wang, Guotai, et al. "Dynamically balanced online random forests for interactive scribble-based segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2016.

[2] Boykov, Yuri Y., and M-P. Jolly. "Interactive graph cuts for optimal boundary & region segmentation of objects in ND images." Proceedings eighth IEEE international conference on computer vision. ICCV 2001. Vol. 1. IEEE, 2001.

[3] Rother, Carsten, Vladimir Kolmogorov, and Andrew Blake. "" GrabCut" interactive foreground extraction using iterated graph cuts." ACM transactions on graphics (TOG) 23.3 (2004): 309-314.

# Installation Instructions
ECONet is implemented using [MONAI Label](https://github.com/Project-MONAI/MONAILabel), which is an AI-Assisted tool for developing interactive segmentation methods. We provide the ECONet MONAI Label app that can be run with following steps:

- Clone ECONet repo: `git clone https://github.com/masadcv/ECONet-MONAILabel`
- Install requirements: `pip install -r requirements.txt`
- Download and install **3D Slicer Preview Release** from: [https://download.slicer.org/](https://download.slicer.org/)
- Install **MONAILabel extension** from 3D Slicer Extension Manager

More detailed documentation on setting up MONAI Label can be found at: [https://docs.monai.io/projects/label/en/latest/installation.html](https://docs.monai.io/projects/label/en/latest/installation.html)

## Slicer Plugin Installation
If using older than the current released version of MONAI Label, you will need to install corresponding MONAI Label Slicer plugin using pip package instead of using Slicer Extension Manager. The following steps guide on how this can be done:

1. Download MONAI Label Slicer plugins from pip package into a known location: `monailabel plugins --download --name slicer --output ./MONAILabelPlugin`
2. Open 3D Slicer and add path `./MONAILabelPlugin/slicer/MONAILabel` to Modules (Edit->Application Settings->Modules->Additional Module Paths->Add)
3. Save and restart 3D Slicer 

Note: make sure to remove any MONAI Label plugins that were installed from Extension Manager before following the steps above for manual installation

See [this post](https://discourse.slicer.org/t/how-to-start-with-monailabel-for-new-models/21063/31) for step by step guide on how to do the above Slicer plugin installation

# Running the ECONet App
The ECONet MONAI Label App runs as MONAI Label server and connects to a MONAI Label client plugin (3D Slicer/OHIF)

## Server: Running ECONet Server App
ECONet MONAI Label server can be started using MONAI Label CLI as:
```
monailabel start_server --app /path/to/this/github/clone --studies /path/to/dataset/images
```

e.g. command to run with sample data from root of this directory
```
monailabel start_server --app . --studies ./data/
```

> By default, MONAI Label server for ECONet will be up and serving at: https://127.0.0.1:8000

## Client: Annotating CT Volumes using ECONet on Client Plugin
On the client side, run slicer and load MONAILabel extension:
- Load ECONet server at: https://127.0.0.1:8000
- Click Next Sample to load an input CT volume
- Scribbles-based likelihood functionality is inside Scribbles section
- To add scribbles select scribbles-based likelihood method then Painter/Eraser Tool and appropriate label Foreground/Background
- Painting/Erasing tool will be activated, add scribbles to each slice/view
- Once done, click Update to send scribbles to server for applying the selected scribbles-based likelihood method
- Iterate as many times as needed, then click submit to save final segmentation

 Demo video showing ECONet usage can be accessed here: [https://www.youtube.com/watch?v=lQEDCx9Mp7A](https://www.youtube.com/watch?v=lQEDCx9Mp7A)

![econet-preview](https://raw.githubusercontent.com/masadcv/ECONet-MONAILabel/main/data/econet_preview.png)

# Citing ECONet
Pre-print of ECONet can be found at: [ECONet: ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation](https://openreview.net/forum?id=9xtE2AgD_Cc)

If you use ECONet in your research, then please cite:

> Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. 
>"ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation." 
>arXiv preprint arXiv:2201.04584 (2022).

BibTeX:
```
@inproceedings{
asad2022econet,
title={{ECON}et: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation},
author={Muhammad Asad and Lucas Fidon and Tom Vercauteren},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=9xtE2AgD_Cc}
}
```
