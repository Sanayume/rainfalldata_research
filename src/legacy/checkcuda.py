import cupy as cp

# 测试GPU功能
def test_gpu():
    try:
        # 创建测试数组
        x = cp.array([1, 2, 3])
        print("CuPy 测试数组:", x)
        
        # 显示CUDA信息
        print(f"CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"可用GPU数量: {cp.cuda.runtime.getDeviceCount()}")
        
        # 显示设备信息
        for i in range(cp.cuda.runtime.getDeviceCount()):
            device = cp.cuda.Device(i)
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nGPU {i}: {props['name'].decode()}")
            print(f"总内存: {props['totalGlobalMem'] / 1024**3:.2f} GB")
            print(f"计算能力: {props['major']}.{props['minor']}")
            
        return True
    except Exception as e:
        print(f"GPU测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_gpu()