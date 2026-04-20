# SAM-EM
## Comprehensive application and framework for multiple particle segmentation and tracking in liquid phase TEM
![Banner](./images/banner.jpg)
* * * * * *
## Abstract

The absence of robust segmentation frameworks for noisy liquid phase transmission electron microscopy (LPTEM) videos prevents reliable extraction of particle trajectories, creating a major barrier to quantitative analysis and to connecting observed dynamics with materials characterization and design. To address this challenge, we present Segment Anything Model for Electron Microscopy (SAM-EM), a domain-adapted foundation model that unifies segmentation, tracking, and statistical analysis for LPTEM data. Built on Segment Anything Model 2 (SAM-2), SAM-EM is derived through full-model fine-tuning on 46,600 curated LPTEM synthetic video frames, substantially improving mask quality and temporal identity stability compared to zero-shot SAM-2 and existing baselines. Beyond segmentation, SAM-EM integrates particle tracking with statistical tools, including mean-squared displacement and trajectory distribution analysis, providing an end-to-end framework for extracting and interpreting nanoscale dynamics. Crucially, full fine-tuning allows SAM-EM to remain robust under low signal-to-noise conditions, such as those caused by increased liquid sample thickness in LPTEM experiments. By establishing a reliable analysis pipeline, SAM-EM transforms LPTEM into a quantitative single-particle tracking platform and accelerates its integration into data-driven materials discovery and design.
* * * * * *

## Installation

### Prerequisites

- **Miniconda** (or Anaconda) must be installed on your machine. Download it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
- Download the SAM-EM model checkpoint from HuggingFace: [https://huggingface.co/sam-em-paper/finetuned-checkpoint/tree/main](https://huggingface.co/sam-em-paper/finetuned-checkpoint/tree/main) and place it in the `checkpoints/` folder inside the repository.
- Clone this repository (SAM-EM): `git clone https://github.com/JamaliLab/SAM-EM.git`. Subsequent installation steps should be completed within the resulting directory unless otherwise specified.

### Quick Launch (Recommended)

The launchers below will automatically create the `SAM-EM-app` conda environment and install all dependencies on the first run (5–15 minutes). Subsequent launches open the app directly.

**Windows**

Double-click `launch_app.bat`. A terminal window will appear showing setup progress on the first run, then the application will open.

**Mac**

Open a terminal in the repository folder and run:

```bash
chmod +x launch_app.sh
./launch_app.sh
```

On Mac, `chmod +x` only needs to be done once. After that you can also right-click the file in Finder → Open With → Terminal.

### Manual Installation

If you prefer to set up the environment yourself:

```bash
conda env create -f environment_app.yml
conda activate SAM-EM-app
```

Then launch the application from the `application/` directory:

```bash
cd application
python app.py
```

This is the main screen.

![Main Screen](./images/main.png)

The checkpoint and config paths are detected automatically. If you followed the steps above (checkpoint in `checkpoints/`, config yaml in the repo root), everything should work out of the box. The gear icon (top-right) can be used to override these paths if needed.

Specify the video directory which contains the video frames and the output directory. Press load video and initialize. Then select annotate frame 0, and press annotate frame.

![Prompt](./images/prompt.png)


For each particle prompt annotation, enter the particle ID starting from 0 then 1, etc. Then drag a box prompt around the particle. Press generate mask, then move on to select prompts for all other particles. When you are done, select close and save prompts. On the main screen, press propogate masks, with results being stored in the output folder. 

An example video of the drawn masklets can be seen as following:

![Example Video](./images/exampleanimation.gif)

For the particle tracking portion of the application, click on particle tracking on the top element to get to the main screen of particle tracking.

![Particle Tracking Main Screen](./images/main_traj.png)

Then browse the output folder for the output csv, and press run motion analysis.

![Particle Tracking Main Screen](./images/main_traj_csv.png)

Finally, press view graphs.

![Particle Tracking Main Screen](./images/traj_dist.png)


<!--
## Acknowledgements

This research was supported by the NSF, Division of Chemical, Bioengineering, Environmental, and Transport Systems under award 2338466, Georgia Tech Institute for Matter and Systems, Exponential Electronics seed grant, the American Chemical Society Petroleum Research Fund under award 67239-DNI5, and the Exponential Electronics Seed grant of the Institute for Matter and Systems at Georgia Tech. The authors acknowledge the support of the Material Characterization Facility and the Electron Microscopy Facility of the Institute for Matter and Systems at Georgia Tech.

## Citation

If you are using this code, please reference our paper:
```
@inproceedings{wang2025samem,
  title        = {SAM-EM: Real-Time Segmentation for Automated Liquid Phase Transmission Electron Microscopy},
  author       = {Wang, Alexander and Xu, Max and Goel, Risha and Shabeeb, Zain and Panicker, Isabel and Jamali, Vida},
  booktitle    = {NeurIPS 2025 Workshop on AI4Mat: AI for Accelerated Materials Design},
  year         = {2025},
  organization = {Neural Information Processing Systems Foundation},
  url          = {https://github.com/JamaliLab/SAM-EM}
}
-->

* * * * * *

## Model Training
To conduct training in accordance with the procedure established in the paper, utilize the 'finalconfig.yaml' configuration file (within main branch of repository) in the training process to establish training parameters and other configurations appropriately.
