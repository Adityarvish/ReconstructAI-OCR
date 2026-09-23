
import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    size = max(length, 3)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(
        kernel,
        (center - length // 2, center),
        (center + length // 2, center),
        color=1.0,
        thickness=1,
    )
    rot = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (size, size))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def wiener_deconvolve(gray_image: np.ndarray, psf: np.ndarray, balance: float = 0.02) -> np.ndarray:
    img = gray_image.astype(np.float32)
    h, w = img.shape

    psf_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf


    psf_padded = np.roll(psf_padded, -(kh // 2), axis=0)
    psf_padded = np.roll(psf_padded, -(kw // 2), axis=1)

    G = np.fft.fft2(img)
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)

    denom = (H * H_conj) + balance
    F_hat = (H_conj / denom) * G

    result = np.fft.ifft2(F_hat)
    result = np.abs(result)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def richardson_lucy(gray_image: np.ndarray, psf: np.ndarray, iterations: int = 12) -> np.ndarray:
    img = gray_image.astype(np.float32) / 255.0
    img = np.clip(img, 1e-6, 1.0)

    kh, kw = psf.shape
    psf_norm = psf / (psf.sum() + 1e-8)
    psf_mirror = psf_norm[::-1, ::-1]

    estimate = img.copy()
    for _ in range(iterations):
        conv = cv2.filter2D(estimate, -1, psf_norm, borderType=cv2.BORDER_REPLICATE)
        conv = np.clip(conv, 1e-6, None)
        relative_blur = img / conv
        correction = cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REPLICATE)
        estimate = estimate * correction
        estimate = np.clip(estimate, 0, 1)

    return (estimate * 255).astype(np.uint8)


def generate_deblur_candidates(gray_image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []


    for sigma in (1.5, 3.0):
        psf = gaussian_psf(size=max(9, int(sigma * 6) | 1), sigma=sigma)
        for balance in (0.01, 0.05):
            deconv = wiener_deconvolve(gray_image, psf, balance=balance)
            candidates.append((f"wiener_gaussian_s{sigma}_k{balance}", deconv))


    rl_psf = gaussian_psf(size=9, sigma=2.0)
    candidates.append(("richardson_lucy_defocus", richardson_lucy(gray_image, rl_psf, iterations=10)))


    for angle in (0, 45, 90, 135):
        m_psf = motion_psf(length=9, angle_deg=angle)
        deconv = wiener_deconvolve(gray_image, m_psf, balance=0.02)
        candidates.append((f"wiener_motion_{angle}deg", deconv))

    return candidates
