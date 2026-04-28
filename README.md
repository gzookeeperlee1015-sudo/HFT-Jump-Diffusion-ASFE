<img width="1200" height="800" alt="Figure_4" src="https://github.com/user-attachments/assets/35ea084e-f978-4f57-ad13-1bc76da426b1" />

# HFT-Jump-Diffusion-ASFE
# ASFE: Adaptive Stochastic Framework for Execution

> ⚠️ **Disclaimer: Academic Purpose Only**
> This repository is created strictly for academic research and educational purposes focusing on market microstructure and computer architecture. The models and codes provided herein do not constitute financial advice. The author assumes no responsibility for any financial losses incurred from applying these algorithms in live trading environments.

## Overview
This repository contains the simulation environment and core architectural skeleton for the **ASFE (Adaptive Stochastic Framework for Execution)** model. It is designed to manage execution friction and adverse selection risks in High-Frequency Trading (HFT) environments. 

Unlike traditional Almgren-Chriss models, ASFE incorporates non-linear market impacts and Jump-Diffusion dynamics to handle fat-tail risks and self-exciting order flow volatility.

## Key Features
* **Hawkes Endogeneity Filtering:** Separates personal order "echoes" from external market noise using a recursive online update with a memory forgetting factor.
* **Hardware-Accelerated Optimality:** Designed for Xilinx Alveo U50 FPGA via AXI4-Stream, achieving a theoretical pipeline latency of 185ns (approx. 60 clock cycles at 322 MHz).
* **Wasserstein Knowledge Distillation:** Transfers high-dimensional HJB policies to a lightweight INT8 quantized neural network, preserving non-Gaussian tail-risk defense mechanisms with less than 0.018% quantization error.
* **Differential Privacy:** Injects Laplace noise ($Lap(0, \Delta f / \epsilon)$) into execution timing to mask algorithmic footprints against inverse reinforcement learning (IRL) attacks.

## Performance (Backtest)
During a simulated Flash Crash scenario (e.g., LUNA collapse / May 2021), ASFE successfully capped the maximum slippage at **-4.2 bps**, compared to **-24.2 bps** for a naive limit model.

<img width="1200" height="800" alt="Figure_4" src="https://github.com/user-attachments/assets/70de7af6-1047-4fcc-ae83-df4bc6663ca2" />

## Repository Structure
* `/src`: Contains the market simulator and ASFE agent skeleton. *(Note: Proprietary parameters such as exact Hawkes kernels and INT8 weights are withheld.)*
* `/docs`: Contains the full research paper (Korean).

## License
This project is licensed under the **GPL-3.0 License**. Any commercial use or modification requires the derivative work to be open-sourced under the same terms.
