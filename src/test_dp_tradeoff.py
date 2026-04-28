import numpy as np
import pandas as pd

class PrivacySimulator:
    def __init__(self, delta_f=1.0):
        # 민감도(Sensitivity) 파라미터
        self.delta_f = delta_f

    def simulate_tradeoff(self, epsilons, num_orders=10000):
        """
        다양한 Epsilon 값에 대해 라플라스 노이즈를 주입하고,
        그 노이즈가 체결 가격 오차율에 미치는 영향을 시뮬레이션합니다.
        """
        results = []
        # 보고서에 언급된 비트코인 65,000불 수준 가정
        base_price = 65000.0 
        
        for eps in epsilons:
            # 라플라스 노이즈 생성: Lap(0, delta_f / epsilon)
            scale = self.delta_f / eps
            noise = np.random.laplace(0, scale, size=num_orders)
            
            # 노이즈(타이밍 섭동)로 인해 발생하는 가격 오차 모사
            # 실제 오더북의 호가 잔량(Liquidity) 구조에 따라 다르나, 
            # 여기서는 노이즈의 절대값에 비례해 슬리피지가 발생한다고 선형 가정
            price_impact_bps = np.abs(noise) * 1.5 
            
            # 오차율(%) 계산
            mean_error_bps = np.mean(price_impact_bps)
            mean_error_percent = mean_error_bps / 10000 * 100
            
            results.append({
                'Epsilon': eps,
                'Noise_Scale (초)': round(scale, 4),
                'Max_Delay (초)': round(np.max(np.abs(noise)), 4),
                'Price_Error_Rate (%)': round(mean_error_percent, 5)
            })
            
        return pd.DataFrame(results)

def run_privacy_test():
    print("="*65)
    print("ASFE 전략 은닉(Differential Privacy) 트레이드오프 검증")
    print("="*65)
    
    sim = PrivacySimulator(delta_f=1.0)
    
    # 엡실론(epsilon) 값이 작을수록 보안성(노이즈)은 강해지지만 오차(비용)는 커짐
    test_epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    df_results = sim.simulate_tradeoff(test_epsilons)
    
    print(df_results.to_string(index=False))
    print("-" * 65)
    print("[검증 포인트]")
    print("보고서 주장: Epsilon = 1.0 일 때, 가격 오차율 0.05% 미만 통제 ")
    print("위 표에서 Epsilon이 1.0일 때의 Price_Error_Rate(%)를 확인하십시오.")

if __name__ == "__main__":
    run_privacy_test()