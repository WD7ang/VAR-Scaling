import os

def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    import glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    
    # 检查是否有 50,000 个 .png 文件
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    samples = np.stack(samples)
    
    # 确保图片形状符合预期
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    
    # 输出 .npz 文件路径
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path

def main():
    sample_folder = '/home/wangkai/wdtang/output'
    
    # 调用 create_npz_from_sample_folder 函数
    npz_file = create_npz_from_sample_folder(sample_folder)
    
    # 进一步处理 npz 文件（如有需要）
    print(f'.npz file created at: {npz_file}')

if __name__ == "__main__":
    main()
