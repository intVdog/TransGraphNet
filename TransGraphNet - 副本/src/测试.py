import torch


def get_cuda_info():
    """获取 CUDA 版本和 GPU 型号"""
    print("=== CUDA & GPU 信息 ===")

    # 1. 检查 CUDA 是否可用
    if torch.cuda.is_available():
        # 获取 GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU 数量: {gpu_count}")

        # 获取每个 GPU 的型号
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i} 型号: {gpu_name}")

        # 获取 PyTorch 编译时的 CUDA 版本（核心版本）
        torch_cuda_version = torch.version.cuda
        print(f"PyTorch 编译的 CUDA 版本: {torch_cuda_version}")

        # 获取当前设备的 CUDA 运行时版本（可选）
        current_device = torch.cuda.current_device()
        cuda_runtime_version = torch.cuda.get_device_properties(current_device).cuda_version
        print(f"CUDA 运行时版本: {cuda_runtime_version}")
    else:
        print("⚠️  当前环境无可用 CUDA（GPU 未识别/驱动未安装）")




# 执行查询
if __name__=="__main__": get_cuda_info()