<div align="center">

# OpenMAMS: Open-Sourced Multi-Agent Memory System

### Memory in the Sky: Low-Altitude Question Answering with Multi-Agent Memory Aggregation

Chengyang Li<sup>1</sup>, Yujie Wan<sup>2</sup>, Shuai Wang<sup>3</sup>, Kejiang Ye<sup>3</sup>, Weijie Yuan<sup>2</sup>, Boyu Zhou<sup>2</sup>, Yik-Chung Wu<sup>1</sup>, Chengzhong Xu<sup>4</sup>, and Huseyin Arslan<sup>5</sup>

<sup>1</sup>The University of Hong Kong · <sup>2</sup>Southern University of Science and Technology<br>
<sup>3</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences<br>
<sup>4</sup>University of Macau · <sup>5</sup>Istanbul Medipol University

[Overview](#overview) · [Architecture](#architecture) · [CARLA Simulation](#carla-simulation) · [Real-World Experiments](#real-world-experiments) · [MemNTN](#memory-native-non-terrestrial-networks) · [Citation](#citation)

[arXiv:2609.35431](https://arxiv.org/abs/2609.35431) · [Paper PDF](https://arxiv.org/pdf/2609.35431)

</div>

![Town04 demo: four UAV views with captions](assets/town04_4uavs.gif)


> OpenMAMS aggregates distributed UAV memories for long-horizon question answering. Our memory-centric framework measures what each candidate memory adds, then jointly selects UAVs and allocates transmit power under communication constraints.



## Overview

OpenMAMS is the multi-agent memory system and benchmarking platform developed for **low-altitude question answering (LAQA)**. It connects aerial observations with a ground memory server so that users can ask about objects, locations, and events observed over time.

The platform described in the paper supports:

- **Multi-agent data collection:** images, timestamps, 6D poses, and LiDAR point clouds in CARLA.
- **Memory construction and retrieval:** VLM captioning, text embeddings, and a vector database for spatiotemporal queries.
- **Memory quality evaluation:** a generative adversarial exam (GAE) measures the knowledge gap between candidate observations and the current memory.
- **Memory-centric resource allocation:** MemCen jointly selects UAV memories and allocates transmit power, with penalty successive optimization (PSO) and learning to memorize (L2M) solvers.

## Architecture

![LAQA architecture: distributed UAV observations are uploaded to a ground server for memory construction and question answering.](assets/architecture.png)

Selected observations are captioned and stored with their timestamps and poses. The resulting global memory supports retrieval-augmented question answering.

<details>
<summary><strong>How does GAE evaluate a candidate memory?</strong></summary>

![GAE pipeline with pilot upload, exam generation, and practice testing against the current memory.](assets/generative-adversarial-exam.png)

GAE generates questions grounded in candidate observations and tests whether the current global memory can answer them. Unanswered questions reveal missing knowledge and quantify the value of acquiring that candidate memory. MemCen combines this task utility with payload sizes, channel conditions, interference, and power constraints.

</details>

## CARLA Simulation

### Town04: multi-UAV inspection

[![CARLA Town04 ten-UAV demo showing UAV locations, image frames, and their associated captions.](assets/carla-town04.png)](assets/carla-town04.png)

Ten UAVs inspect different regions of Town04 and collect complementary observations for memory construction. The demo shows their locations, sample image frames, and associated captions. Questions ask whether an object is present, where it is located, and which UAV observed it.

### Town05: dynamic and heterogeneous UAVs

<table>
<tr>
<td width="58%" align="center" valign="middle"><a href="assets/carla-town05.png"><img src="assets/carla-town05.png" width="100%" alt="CARLA Town05 building geometry, ground station, and flight trajectories of four fixed-wing and six multirotor UAVs."></a></td>
<td width="42%" align="center" valign="middle"><a href="assets/carla-town05-blockage.png"><img src="assets/carla-town05-blockage.png" width="100%" alt="Time-varying blockage across the ten UAV links in Town05, with per-UAV blockage ratios."></a></td>
</tr>
<tr>
<td align="center">UAV trajectories and ground station</td>
<td align="center">Time-varying link blockage</td>
</tr>
</table>

Four fixed-wing and six multirotor UAVs conduct a 200-second search-and-rescue mission in Town05. Their trajectories, image workloads, and building blockage create varying communication conditions. MemCen selects complementary memories and adapts transmit power to support downstream question answering.

## Real-World Experiments

### Panoramic multi-agent system (PMAS)

![PMAS field experiment with three UAVs, reconstructed trajectories, onboard observations, and spatial question answering.](assets/pmas-benchmark.png)

Three panoramic UAVs collect complementary observations along distinct routes. Their observations are registered in a shared 3D coordinate frame for object-presence and spatial-grounding questions. The aerial data are collected in the field, while communication is evaluated through offline channel replay.

### UAV-to-ground-robot memory reuse

![A robot dog answers questions and navigates to a basketball court using previously acquired UAV memory.](assets/uav-ground-robot.png)

A robot dog reuses aerial memory to answer environmental questions and navigate to a queried location.

## Memory-Native Non-Terrestrial Networks

MemNTN extends memory-based remote question answering to satellite networks through memory management, fusion, and valuation.

[![MemNTN: UAV perception in Pittsburgh, a satellite constellation, and remote question answering in Istanbul.](assets/memntn_fig1.png)](assets/memntn_fig1.png)

![MemNTN evaluation results: LEOPath and CARLA setup, the 400-satellite benchmark, and constellation-size comparisons.](assets/memntn_fig4.png)

The LEOPath and CARLA evaluation compares remote question-answering accuracy and end-to-end throughput across satellite constellation sizes. MemNTN achieves 97.8% QA accuracy in the 400-satellite case, outperforming the compared baselines.

## Selected Results

| Evaluation | Setting | MemCen QA accuracy |
| --- | --- | ---: |
| CARLA Town04 | Static communication conditions | **92.4%** |
| CARLA Town05 | Dynamic channels, heterogeneous UAVs, and building blockage | **84.0%** |
| PMAS  | Real aerial observations with offline channel replay | **88.5%** |

## Code

The following modules are available and install independently:

| Module | Contents | Usage |
| --- | --- | --- |
| `uav_data_recorder/` | Synchronized CARLA RGB images, camera poses, object ground truth | [Recorder guide](uav_data_recorder/README.md) |
| `ntn/` | Satellite geometry, uplink/downlink scheduling, and FIFO image-delivery replay | [NTN guide](ntn/README.md) |

### Demos


![Town05 demo: ten UAV views with captions](assets/town05_10uavs.webp)

| Demo | Duration | Frame rate | Resolution | Size |
| --- | --- | --- | --- | --- |
| [Town04, K=4](assets/town04_4uavs.gif) | 15 s | 10 FPS | 960 × 540 | 25.02 MB |
| [Town05, K=10](assets/town05_10uavs.webp) | 15 s | 10 FPS | 1600 × 360 | 8.28 MB |



The Town05 demo shows ten virtual UAV cameras following a closed road route in a
5 × 2 layout. Captions update every three seconds. See the
[asset descriptions](assets/README.md) for recording details.

## Citation

If you find this work useful, please cite:

### Journal version

```bibtex
@misc{li2026memoryinthesky,
  title  = {Memory in the Sky: Low-Altitude Question Answering with Multi-Agent Memory Aggregation},
  author = {Li, Chengyang and Wan, Yujie and Wang, Shuai and Ye, Kejiang and Yuan, Weijie and Zhou, Boyu and Wu, Yik-Chung and Xu, Chengzhong and Arslan, Huseyin},
  year   = {2026},
  eprint = {2609.35431},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url    = {https://arxiv.org/abs/2609.35431}
}
```

### Conference version

```bibtex
@inproceedings{li2026memory,
  title={Memory centric power allocation for multi-agent embodied question answering},
  author={C. Li and S. Wang and K. Ye and W. Yuan and B. Zhou and Y.-C. Wu and C. Xu and H. Arslan},
  booktitle={Proc. GLOBECOM},
  year={2026}
}
```

### Magazine version

```bibtex
@article{li2026memntn,
  title={Memory-Native Non-Terrestrial Networks for Embodied Intelligence},
  author={Li, Chengyang and Wang, Yikun and He, Jiahui and Wan, Yujie and Wang, Shuai and Wu, Yuan and Wu, Yik-Chung and Xu, Chengzhong and Arslan, Huseyin},
  journal={IEEE Communications Standards Magazine},
  year={2026}
}
```

## Contact

- **Chengyang Li**: [KevinLADLee](https://github.com/KevinLADLee)
- **Shuai Wang**: [bearswang](https://github.com/bearswang)
