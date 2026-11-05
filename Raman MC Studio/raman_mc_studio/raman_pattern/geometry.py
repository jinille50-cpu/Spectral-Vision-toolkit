# -*- coding: utf-8 -*-
"""시뮬레이션 기하 구조 (단위 mm).

좌표: x = CCD offset 축(가로), y = 슬릿 방향, z = 깊이 (0 = 조직 표면, +z 아래).
PET 정육면체는 격자 가로/세로 중앙, 윗면이 depth_top 에 오도록 놓는다.
광원은 큐브 중심에서 x 방향으로 p 만큼 떨어진 표면 위 한 점.
"""
from dataclasses import dataclass, field

import numpy as np

# 광학 계수 (mm⁻¹) — sors_mc_studio/core/presets.py 와 동일
TISSUE = dict(mua=0.03, mus=11.0, g=0.92, n=1.40)     # Pork muscle (830 nm)
PET = dict(mua=0.005, mus=1.5, g=0.7, n=1.58)         # PET (transparent)
FAT = dict(mua=0.01, mus=12.0, g=0.90, n=1.44)        # Fat (subcutaneous, 830 nm)
AIR = dict(mua=0.0, mus=0.0, g=1.0, n=1.0)

# 광원: 2" f = 100 mm achromat, NA ≈ 0.23, 초점 빔 100 µm (= 복셀 1개) → 원뿔 발산 광원
LASER_NA = 0.23


@dataclass
class FatStrand:
    """+x 쪽 근육 안을 y 방향으로 지나는 구불구불한 지방 원기둥 (무작위 흔들림)."""
    x_center: float = 3.0        # 큐브 중심에서 x 방향 위치 (mm, +x 쪽)
    depth_center: float = 2.0    # 원기둥 축의 깊이 (mm)
    diameter: float = 1.5        # mm
    wobble: float = 0.8          # 축이 좌우/상하로 흔들리는 폭 (mm)
    seed: int = 3


@dataclass
class Geometry:
    voxel: float = 0.1            # mm / voxel
    x_extent: tuple = (-8.0, 24.0)   # 큐브 중심 0, 광원 최대 16 mm + 여유 8 mm
    y_extent: tuple = (-8.0, 8.0)
    air_mm: float = 1.0
    tissue_mm: float = 10.0
    cube_mm: float = 2.0
    depth_top: float = 2.0        # 조직 표면부터 큐브 윗면까지 (mm)
    tissue: dict = field(default_factory=lambda: dict(TISSUE))
    pet: dict = field(default_factory=lambda: dict(PET))
    fat: dict = field(default_factory=lambda: dict(FAT))
    fat_strand: object = None     # FatStrand 이면 지방 줄기 포함
    source: str = 'pencil'        # 'pencil' | 'cone'
    na: float = LASER_NA

    # ---- 격자 크기 ----
    @property
    def nx(self):
        return int(round((self.x_extent[1] - self.x_extent[0]) / self.voxel))

    @property
    def ny(self):
        return int(round((self.y_extent[1] - self.y_extent[0]) / self.voxel))

    @property
    def nz(self):
        return int(round((self.air_mm + self.tissue_mm) / self.voxel))

    @property
    def z_surface_vox(self):
        return int(round(self.air_mm / self.voxel))

    # ---- 좌표 변환 ----
    def x_to_vox(self, x_mm):
        return int(round((x_mm - self.x_extent[0]) / self.voxel))

    def y_to_vox(self, y_mm):
        return int(round((y_mm - self.y_extent[0]) / self.voxel))

    def z_to_vox(self, z_mm):
        """z_mm: 조직 표면 기준 깊이"""
        return int(round(z_mm / self.voxel)) + self.z_surface_vox

    def cube_slices(self):
        n = int(round(self.cube_mm / self.voxel))
        x0 = self.x_to_vox(-self.cube_mm / 2)
        y0 = self.y_to_vox(-self.cube_mm / 2)
        z0 = self.z_to_vox(self.depth_top)
        return slice(x0, x0 + n), slice(y0, y0 + n), slice(z0, z0 + n)

    def fat_mask(self):
        """지방 줄기 복셀 마스크 (nx, ny, nz). 축은 y 를 따라가며 x, z 가 무작위로 흔들린다."""
        fs = self.fat_strand
        rng = np.random.default_rng(fs.seed)
        ny = self.ny
        # 부드러운 무작위 흔들림: 난수 누적합을 이동평균으로 다듬고 폭을 wobble 로 맞춤
        def smooth_noise():
            w = np.cumsum(rng.normal(size=ny))
            k = max(3, ny // 8)
            w = np.convolve(w, np.ones(k) / k, mode='same')
            w -= w.mean()
            return w / (np.abs(w).max() + 1e-12) * fs.wobble
        xs = fs.x_center + smooth_noise()
        zs = fs.depth_center + 0.5 * smooth_noise()
        r = fs.diameter / 2
        X = (np.arange(self.nx) + 0.5) * self.voxel + self.x_extent[0]
        Z = (np.arange(self.nz) + 0.5) * self.voxel - self.air_mm
        mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        for j in range(ny):
            dx = (X - xs[j])[:, None]
            dz = (Z - zs[j])[None, :]
            mask[:, j, :] = (dx ** 2 + dz ** 2) <= r ** 2
        mask[:, :, :self.z_surface_vox] = False
        return mask

    def build_volume(self, with_cube=True):
        """0 = air, 1 = tissue, 2 = PET, 3 = fat"""
        vol = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        vol[:, :, self.z_surface_vox:] = 1
        if self.fat_strand is not None:
            vol[self.fat_mask()] = 3
        if with_cube:
            sx, sy, sz = self.cube_slices()
            if sz.stop > self.nz:
                raise ValueError("Cube extends below the tissue slab.")
            vol[sx, sy, sz] = 2
        return vol

    def prop(self):
        return [[AIR['mua'], AIR['mus'], AIR['g'], AIR['n']],
                [self.tissue['mua'], self.tissue['mus'], self.tissue['g'], self.tissue['n']],
                [self.pet['mua'], self.pet['mus'], self.pet['g'], self.pet['n']],
                [self.fat['mua'], self.fat['mus'], self.fat['g'], self.fat['n']]]

    def source_cfg(self):
        """MCX 광원 추가 설정. cone: 반각 = asin(NA) (공기 중), 초점은 표면 복셀 하나."""
        if self.source == 'cone':
            return {'srctype': 'cone', 'srcparam1': [float(np.arcsin(self.na)), 0, 0, 0]}
        return {}

    def source_voxel(self, p_mm):
        """큐브 중심에서 x 방향 p_mm 떨어진 표면 점 (voxel 좌표, 0-based)"""
        xv = self.x_to_vox(p_mm)
        if not (0 <= xv < self.nx):
            raise ValueError(f"Source at {p_mm} mm is outside the grid.")
        return [xv, self.y_to_vox(0.0), self.z_surface_vox]
