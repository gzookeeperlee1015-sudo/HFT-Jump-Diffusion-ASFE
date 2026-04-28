# HFT-Jump-Diffusion-ASFE
# ASFE: Adaptive Stochastic Framework for Execution

> ⚠️ **Disclaimer: Academic Purpose Only**
> This repository is created strictly for academic research and educational purposes focusing on market microstructure and computer architecture. The models and codes provided herein do not constitute financial advice. The author assumes no responsibility for any financial losses incurred from applying these algorithms in live trading environments.

## Overview
This repository contains the simulation environment and core architectural skeleton for the **ASFE (Adaptive Stochastic Framework for Execution)** model. [cite_start]It is designed to manage execution friction and adverse selection risks in High-Frequency Trading (HFT) environments[cite: 5]. 

[cite_start]Unlike traditional Almgren-Chriss models, ASFE incorporates non-linear market impacts and Jump-Diffusion dynamics to handle fat-tail risks and self-exciting order flow volatility[cite: 6].

## Key Features
* [cite_start]**Hawkes Endogeneity Filtering:** Separates personal order "echoes" from external market noise using a recursive online update with a memory forgetting factor[cite: 16, 17].
* [cite_start]**Hardware-Accelerated Optimality:** Designed for Xilinx Alveo U50 FPGA via AXI4-Stream, achieving a theoretical pipeline latency of 185ns (approx. 60 clock cycles at 322 MHz)[cite: 26, 28].
* [cite_start]**Wasserstein Knowledge Distillation:** Transfers high-dimensional HJB policies to a lightweight INT8 quantized neural network, preserving non-Gaussian tail-risk defense mechanisms with less than 0.018% quantization error[cite: 18, 23, 27].
* [cite_start]**Differential Privacy:** Injects Laplace noise ($Lap(0, \Delta f / \epsilon)$) into execution timing to mask algorithmic footprints against inverse reinforcement learning (IRL) attacks[cite: 33, 34].

## Performance (Backtest)
[cite_start]During a simulated Flash Crash scenario (e.g., LUNA collapse / May 2021), ASFE successfully capped the maximum slippage at **-4.2 bps**, compared to **-24.2 bps** for a naive limit model[cite: 40, 43, 44].

*(Insert the Flash Crash graph image here: `![Flash Crash Defense](assets/flash_crash_graph.png)`)*

## Repository Structure
* `/src`: Contains the market simulator and ASFE agent skeleton. *(Note: Proprietary parameters such as exact Hawkes kernels and INT8 weights are withheld.)*
* `/docs`: Contains the full research paper (Korean).

## License
This project is licensed under the **GPL-3.0 License**. Any commercial use or modification requires the derivative work to be open-sourced under the same terms.
