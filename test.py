import torch

# 임의의 데이터로 간단한 테스트
x_aug = torch.tensor([0.4, 0.6, 0.3, 0.8, 0.2, 0.5], device='cuda')

mask_in = x_aug[:, None, None]  # 차원 확장을 통해 원래 코드와 유사한 형태 유지
comparison_result = mask_in < 0.5  # 비교 연산 결과

# 비교 연산 결과 출력
print("Comparison Result:")
print(comparison_result)

# torch.where 적용
mask_aug = torch.where(mask_in < 0.5, torch.tensor(0.0, device='cuda'), mask_in)

# 결과 출력
print("Mask Augmented:")
print(mask_aug)

# NaN 검사
if torch.isnan(mask_aug).any():
    print("NaN values present after processing.")
else:
    print("No NaN values detected.")
