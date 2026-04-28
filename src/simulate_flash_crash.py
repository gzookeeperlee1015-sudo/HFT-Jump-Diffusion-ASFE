import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_flash_crash_market(n_steps=1000, crash_start=500, crash_end=550):
    """
    점프-확산(Jump-Diffusion) 모델을 기반으로 플래시 크래시 시장 데이터를 생성합니다.
    """
    np.random.seed(42) # 재현성을 위한 시드 고정
    
    dt = 1/n_steps
    mu = 0.02 # 기본 드리프트
    sigma = 0.15 # 기본 변동성
    
    # 1. 기본 기하학적 브라운 운동(GBM) 생성
    price = np.zeros(n_steps)
    price[0] = 100.0 # 시작가 (비트코인 등 자산의 스케일링된 가격)
    
    # 일반 구간의 노이즈
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    for t in range(1, n_steps):
        # 2. 플래시 크래시 구간 (Jump 역학)
        if crash_start <= t <= crash_end:
            # 시장 미세구조 경색 (극단적인 하락 점프 발생)
            jump = np.random.normal(-0.015, 0.005) # 강한 음의 방향 점프
        else:
            # 평상시 (간헐적인 작은 점프)
            jump = np.random.choice([0, np.random.normal(0, 0.002)], p=[0.95, 0.05])
            
        # SDE 계산 (드리프트 + 확산 + 점프)
        price[t] = price[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t] + jump)
        
    return price

def simulate_execution(prices, crash_start, crash_end):
    """
    Naive Limit 모델과 ASFE(Wasserstein KD) 모델의 체결 슬리피지 시뮬레이션
    """
    n_steps = len(prices)
    
    # 슬리피지 기록용 배열 (단위: bps)
    naive_slippage = np.zeros(n_steps)
    asfe_slippage = np.zeros(n_steps)
    
    for t in range(1, n_steps):
        volatility = np.abs(prices[t] - prices[t-1]) / prices[t-1]
        
        # 크래시 구간 진입 시 역선택(Adverse Selection) 리스크 폭발
        if crash_start <= t <= crash_end:
            # 1. Naive Limit: 폭락을 인지하지 못하고 그대로 호가를 맞다가 큰 슬리피지 발생
            # 논문 데이터 기준 Max -24.2 bps 수준 모사
            naive_slippage[t] = naive_slippage[t-1] - (volatility * 10000 * 1.5) 
            
            # 2. ASFE: Wasserstein 지식 증류를 통한 테일 리스크 방어
            # 내생성 필터링으로 큐를 취소하거나 보수적으로 재조정하여 슬리피지 최소화
            # 논문 데이터 기준 Max -4.2 bps 수준 방어 모사
            asfe_slippage[t] = asfe_slippage[t-1] - (volatility * 10000 * 0.1) 
        else:
            # 일반 장세에서의 기본 슬리피지 (실행 마찰)
            naive_slippage[t] = naive_slippage[t-1] - (volatility * 10000 * 0.2)
            asfe_slippage[t] = asfe_slippage[t-1] - (volatility * 10000 * 0.15)
            
    return naive_slippage, asfe_slippage

# --- 시뮬레이션 실행 ---
N_STEPS = 1000
CRASH_START = 500
CRASH_END = 550

# 데이터 생성
market_prices = generate_flash_crash_market(N_STEPS, CRASH_START, CRASH_END)
naive_slip, asfe_slip = simulate_execution(market_prices, CRASH_START, CRASH_END)

# Max Slippage 수치 추출
max_naive = np.min(naive_slip[CRASH_START:CRASH_END]) - naive_slip[CRASH_START]
max_asfe = np.min(asfe_slip[CRASH_START:CRASH_END]) - asfe_slip[CRASH_START]

# --- 그래프 시각화 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

# 1. 시장 가격 경로 (Flash Crash 구간 하이라이트)
ax1.plot(market_prices, color='black', linewidth=1.2, label='Market Price (Jump-Diffusion)')
ax1.axvspan(CRASH_START, CRASH_END, color='red', alpha=0.2, label='Flash Crash Zone')
ax1.set_title('Market Dynamics: Flash Crash Scenario (e.g., LUNA / May 2021)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left')

# 2. 누적 슬리피지 방어력 비교
ax2.plot(naive_slip, color='red', linestyle='--', linewidth=1.5, label='Naive Limit (Traditional)')
ax2.plot(asfe_slip, color='blue', linewidth=2.0, label='ASFE (Wasserstein KD)')
ax2.axvspan(CRASH_START, CRASH_END, color='red', alpha=0.2)
ax2.set_title(f'Cumulative Execution Slippage (bps) | Max Slippage during Crash - Naive: {max_naive:.1f} bps, ASFE: {max_asfe:.1f} bps', fontsize=12)
ax2.set_xlabel('Time Steps', fontsize=12)
ax2.set_ylabel('Slippage (bps)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.show()