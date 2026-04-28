import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class JitterSimulator:
    def __init__(self, base_latency_ns=185, jitter_std_ns=50):
        # 보고서에 명시된 기본 하드웨어 파이프라인 지연 시간 [cite: 27, 29]
        self.base_latency_ns = base_latency_ns
        self.jitter_std_ns = jitter_std_ns

    def apply_jitter(self, original_signals: pd.DataFrame) -> pd.DataFrame:
        signals = original_signals.copy()
        
        # 지터 생성 (물리적 계층 변동성 모사) [cite: 11]
        random_jitter = np.random.normal(loc=0, scale=self.jitter_std_ns, size=len(signals))
        random_jitter = np.abs(random_jitter) 
        
        total_latency = self.base_latency_ns + random_jitter
        signals['total_latency_ns'] = total_latency
        
        # 나노초 단위 타임스탬프 계산
        signals['executed_timestamp'] = signals['timestamp'] + pd.to_timedelta(total_latency, unit='ns')
        return signals

def run_test():
    print("="*50)
    print(" ASFE 하드웨어 지연 시간(185ns) 및 지터 검증")
    print("="*50)

    # 1. 테스트용 가상 데이터 생성 (5개의 연속 주문 시나리오)
    now = datetime.now()
    data = {
        'timestamp': [now + timedelta(milliseconds=i*100) for i in range(5)],
        'target_price': [65000.5, 65001.0, 64998.5, 64997.0, 65002.5]
    }
    asfe_signals = pd.DataFrame(data)

    # 2. 시뮬레이터 인스턴스 생성 (지터 강도를 500ns로 설정하여 테스트)
    # 실제 FPGA 타겟 클럭은 322 MHz이며 파이프라인은 60 Cycle 이내입니다. [cite: 27, 29]
    jitter_sim = JitterSimulator(base_latency_ns=185, jitter_std_ns=500)
    
    # 3. 지터 적용 실행
    results = jitter_sim.apply_jitter(asfe_signals)

    # 4. 결과 터미널 출력
    print(f"{'Original Timestamp':<30} | {'Executed (Delayed)':<30} | {'Latency (ns)':<12}")
    print("-" * 80)
    
    for _, row in results.iterrows():
        orig = row['timestamp'].strftime('%H:%M:%S.%f')
        exec_t = row['executed_timestamp'].strftime('%H:%M:%S.%f')
        lat = f"{row['total_latency_ns']:.2f}"
        print(f"{orig:<30} | {exec_t:<30} | {lat:<12}")

    print("\n [검증 알림] 위와 같이 나노초 단위의 체결 밀림이 발생합니다.")
    print("이 지연 시간 동안 가격 변동(Slippage)을 계산하면 실전 방어력이 도출됩니다.")

if __name__ == "__main__":
    run_test()