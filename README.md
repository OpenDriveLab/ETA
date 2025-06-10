# 🤔 ETA

## Highlight
We propose "Efficiency through Thinking Ahead" (ETA), an asynchronous dual-system that pre-processes information from past frames using a large model in tandem with processing the current information with a small model to enable real-time decisions with strong performance.
<img width="800" alt="CarFormer overview" src="assets/overview.png">

## Abstract
How can we benefit from large models without sacrificing inference speed, a common dilemma in self-driving systems? A prevalent solution is a dual-system architecture, employing a small model for rapid, reactive decisions and a larger model for slower but more informative analyses. Existing dual-system designs often implement parallel architectures where inference is either directly conducted using the large model at each current frame or retrieved from previously stored inference results. However, these works still struggle to enable large models for a timely response to every online frame. Our key insight is to shift intensive computations of the current frame to previous time steps and perform a batch inference of multiple time steps to make large models respond promptly to each time step. To achieve the shifting, we introduce Efficiency through Thinking Ahead (ETA), an asynchronous system designed to: (1) propagate informative features from the past to the current frame using future predictions from the large model, (2) extract current frame features using a small model for real-time responsiveness, and (3) integrate these dual features via an action mask mechanism that emphasizes action-critical image regions. Evaluated on the Bench2Drive CARLA Leaderboard-v2 benchmark, ETA advances state-of-the-art performance by 8% with a driving score of 69.53 while maintaining a near-real-time inference speed at 50 ms.

## Results
<img width="800" alt="CarFormer overview" src="assets/results.png">

## Training and Inference

Please refer to [TRAIN_EVAL.md](docs/TRAIN_EVAL.md) for detailed instructions on how to train and evaluate the model.

## Acknowledgements

This codebase builds on open sourced code from [carla_garage](git@github.com:autonomousvision/carla_garage.git) among others. We thank the authors for their contributions. This project is funded by the European Union (ERC, ENSURE, 101116486) with additional compute support from Leonardo Booster (EuroHPC Joint Undertaking, EHPC-AI-2024A01-060). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. This study is also supported by National Natural Science Foundation of China (62206172) and Shanghai Committee of Science and Technology (23YF1462000).

## Citation
If you find our project useful for your research, please consider citing our paper with the following BibTeX:


```bibtex
@article{hamdan2025eta,
  title={ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models},
  author={Hamdan, Shadi and Sima, Chonghao and Yang, Zetong and Li, Hongyang and G{\"u}ney, Fatma},
  journal={arXiv preprint arXiv:2506.07725},
  year={2025}
}
```
