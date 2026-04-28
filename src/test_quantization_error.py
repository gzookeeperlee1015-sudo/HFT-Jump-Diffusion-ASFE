import numpy as np
import pandas as pd

class QuantizationSimulator:
    def __init__(self, input_dim=64, hidden_dim=128):
        # 컴퓨터 구조 관점에서 메모리 정렬에 유리한 2의 제곱수 차원 설정
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 가상의 HJB 솔루션 증류(Distillation) 모델 가중치 (Float32)
        # 평균 0, 표준편차 0.1 정규분포로 가중치 초기화
        self.weights_fp32 = np.random.normal(0, 0.1, (input_dim, hidden_dim)).astype(np.float32)

    def quantize_int8(self, weights):
        """
        Float32 가중치를 INT8로 양자화(Quantization)하는 과정 모사
        """
        # Symmetric Quantization (Min-Max 기반)
        max_val = np.max(np.abs(weights))
        
        # INT8 표현 범위: -127 ~ 127 (대칭성 유지를 위해 -128은 제외)
        scale = 127.0 / max_val
        
        # 양자화 (클리핑 및 반올림)
        weights_int8 = np.round(weights * scale).astype(np.int8)
        weights_int8 = np.clip(weights_int8, -127, 127)
        
        # 역양자화 (Dequantization) - 실제 추론 결과 비교를 위해 FP32 스케일로 복원
        weights_dequantized = (weights_int8 / scale).astype(np.float32)
        
        return weights_dequantized

    def evaluate_error(self, num_samples=1000):
        # 시장 시그널 입력 (Float32)
        market_signals = np.random.normal(0, 1, (num_samples, self.input_dim)).astype(np.float32)
        
        # 1. 원본 가중치(FP32) 연산 결과 (Teacher Model)
        output_fp32 = np.dot(market_signals, self.weights_fp32)
        
        # 2. 양자화 가중치(INT8) 연산 결과 (Student Model)
        weights_dq = self.quantize_int8(self.weights_fp32)
        output_int8 = np.dot(market_signals, weights_dq)
        
        # 3. 오차율(%) 계산
        # 분모가 0이 되는 것을 방지하기 위해 아주 작은 값(epsilon) 추가
        abs_diff = np.abs(output_fp32 - output_int8)
        error_rate = (abs_diff / (np.abs(output_fp32) + 1e-8)) * 100
        
        # 전체 평균 오차율 반환
        return np.mean(error_rate)

def run_quantization_test():
    print("="*65)
    print(" ASFE 가중치 양자화(INT8) 오차율 검증 (FPGA 최적화)")
    print("="*65)
    
    sim = QuantizationSimulator()
    
    # 5번의 시나리오(에포크) 테스트
    error_rates = []
    for i in range(5):
        err = sim.evaluate_error()
        error_rates.append(err)
        print(f"[{i+1}/5] 시뮬레이션 평균 오차율: {err:.5f}%")
        
    final_mean_error = np.mean(error_rates)
    print("-" * 65)
    print(f" 최종 평균 오차율: {final_mean_error:.5f}%")
    print("보고서 주장: 가중치 INT8 양자화 오차는 0.018% 미만으로 억제")
    
    if final_mean_error < 0.018:
        print("\n 결론: 오차율 0.018% 미만이 성공적으로 재현되었습니다.")
        print("         모델 경량화(FPGA 실장) 시에도 전략의 수학적 최적성이 훼손되지 않음을 증명합니다.")
    else:
        print("\n 결론: 오차율이 기준치를 초과합니다. 극단적 이상치(Outlier)를 제어하는")
        print("         지식 증류 기법(Sinkhorn-regularized KD)이 추가로 적용되어야 달성 가능할 것으로 보입니다.")

if __name__ == "__main__":
    run_quantization_test()