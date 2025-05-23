from utils.metrics import compute_ssim
import numpy as np

def test_ssim_identity():
    img = np.random.rand(40, 40)
    score = compute_ssim(img, img)
    assert abs(score - 1.0) < 1e-5, "SSIM of identical images should be 1.0"

def test_ssim_different():
    img1 = np.ones((40, 40))
    img2 = np.zeros((40, 40))
    score = compute_ssim(img1, img2)
    assert score < 0.1, "SSIM of completely different images should be low"
