import torch
import torch.nn.functional as F

def custom_softmax(x, dim):
    """
    Compute softmax along a specific dimension.
    
    Parameters:
    x (torch.Tensor): 입력 텐서
    dim (int): Softmax를 적용할 차원

    Returns:
    torch.Tensor: Softmax가 적용된 텐서
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)  # 안정적인 계산을 위해 최대값을 빼줍니다.
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

# 사용 예시
scores = torch.rand(5,5,5)  # 예제 스코어 텐서
dim = -2  # softmax를 적용할 차원
t1 = custom_softmax(scores, dim=dim)
t2 = F.softmax(scores, -2)

print(t1)
print(t2)
print(t1 - t2)
print(torch.sum(t1 - t2))
