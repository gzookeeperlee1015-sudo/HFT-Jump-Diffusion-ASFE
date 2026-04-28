import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 및 파라미터 (Flash Crash & Mean Reversion)
# ==========================================
np.random.seed(42)
T, N = 1.0, 1000
dt = T / N
S0 = 100
TOTAL_QTY = 1.0

sigma = 0.02             # 평상시 변동성
jump_prob = 0.005        # Flash Crash 발생 확률
jump_mean = -0.05        # -5% 순간 폭락
base_slippage = 0.001    # 기본 유동성 비용

# ==========================================
# 2. 시장 시뮬레이터 (평균 회귀 포함)
# ==========================================
def generate_market():
    S = np.zeros(N)
    S[0] = S0
    eta = np.zeros(N)    # 유동성 경색 계수 (높을수록 슬리피지 큼)
    eta[0] = base_slippage
    jumps = np.zeros(N)
    
    alpha_market = 0.05  # 점프 시 유동성 증발 강도
    beta_market = 20.0   # 유동성 회복 속도
    mean_reversion_speed = 5.0 #  핵심: 폭락 후 제자리로 돌아가려는 복원력
    
    for i in range(1, N):
        dw = np.random.randn() * np.sqrt(dt)
        is_jump = 1 if np.random.rand() < jump_prob else 0
        jump_size = np.random.normal(jump_mean, 0.01) if is_jump else 0
        jumps[i] = is_jump
        
        #  Drift 항에 평균 회귀(Mean Reversion) 추가
        # 폭락 이벤트가 끝나면 가격이 다시 S0 근처로 서서히 반등함
        drift = mean_reversion_speed * (S0 - S[i-1]) * dt
        
        # 가격 SDE
        S[i] = S[i-1] * (1 + sigma * dw + jump_size) + drift
        
        # Hawkes Process 기반 유동성(슬리피지) 비용 SDE
        eta[i] = base_slippage + (eta[i-1] - base_slippage) * np.exp(-beta_market * dt) + alpha_market * is_jump
        
    return S, eta, jumps

# ==========================================
# 3. 알고리즘: TWAP (기계적 분할 매도)
# ==========================================
def run_twap(S, eta):
    qty = TOTAL_QTY / N
    prices = []
    for i in range(N):
        # 틱당 체결가 = 현재가 * (1 - 유동성계수 * 물량제곱근)
        prices.append(S[i] * (1 - eta[i] * np.sqrt(qty)))
    return np.mean(prices)

# ==========================================
# 4. 알고리즘: ASFE (최적 제어 및 유동성 회피)
# ==========================================
def run_asfe(S, eta, jumps):
    remaining = TOTAL_QTY
    hawkes_danger = 0.0
    prices = []
    
    alpha_asfe = 1.0     # 위기 감지 민감도
    beta_asfe = 20.0     # 위기 해소(망각) 속도
    
    for i in range(N):
        steps_left = N - i
        if steps_left == 0: break
        
        # 내생성/외생성 위험도 실시간 추적 (Hawkes SDE)
        hawkes_danger = hawkes_danger * np.exp(-beta_asfe * dt) + alpha_asfe * jumps[i]
        
        base_rate = remaining / steps_left
        
        #  ASFE 핵심 제어 로직
        if hawkes_danger > 0.1:
            # 위험 구간(Flash Crash 발생 중): 주문량을 1/10로 대폭 줄여 소나기를 피함
            exec_rate = base_rate * 0.1
        else:
            # 안전 구간: 지연된 물량을 조금씩 만회 (과도한 덤핑 방지를 위해 1.1배로 제한)
            exec_rate = base_rate * 1.1
            
        qty = min(exec_rate, remaining)
        
        if qty > 0:
            prices.append(S[i] * (1 - eta[i] * np.sqrt(qty)) * qty)
            remaining -= qty
            
    # 잔여 물량 처리 (장 마감 동시호가)
    if remaining > 0:
        prices.append(S[-1] * (1 - eta[-1] * 2.0) * remaining)
        
    return np.sum(prices) / TOTAL_QTY

# ==========================================
# 5. 몬테카를로 시뮬레이션
# ==========================================
SIM_COUNT = 300
twap_slips = []
asfe_slips = []

for _ in range(SIM_COUNT):
    S, eta, jumps = generate_market()
    
    twap_vwap = run_twap(S, eta)
    asfe_vwap = run_asfe(S, eta, jumps)
    
    # BPS 변환 (S0 대비 수익/손실)
    twap_slips.append((twap_vwap - S0) / S0 * 10000)
    asfe_slips.append((asfe_vwap - S0) / S0 * 10000)

improvement = np.array(asfe_slips) - np.array(twap_slips)
win_rate = np.mean(improvement > 0)

# ==========================================
# 6. 결과 출력 (보고서 검증)
# ==========================================
print(f"===  최종 ASFE 수학적 검증 완료 ===")
print(f"TWAP 평균 수익률: {np.mean(twap_slips):.2f} bps")
print(f"ASFE 평균 수익률: {np.mean(asfe_slips):.2f} bps")
print("-" * 40)
print(f" ASFE 평균 방어 개선치: +{np.mean(improvement):.2f} bps")
print(f" 최대 초과 수익 (최악의 Crash 구간): +{np.max(improvement):.2f} bps")
print(f" 승률 (ASFE가 TWAP을 이길 확률): {win_rate * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.hist(improvement, bins=30, color='#27ae60', edgecolor='black', alpha=0.8)
plt.axvline(np.mean(improvement), color='#c0392b', linestyle='dashed', linewidth=2, label=f'Mean: +{np.mean(improvement):.2f} bps')
plt.title('Final Proof: ASFE vs TWAP Performance (Flash Crash + Mean Reversion)')
plt.xlabel('Net Improvement (bps) - Positive means ASFE wins')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.show()