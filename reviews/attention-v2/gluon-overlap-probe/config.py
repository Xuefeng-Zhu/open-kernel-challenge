"""CPU-only fixed configuration for a standalone compile-gated mechanism probe."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProbeConfig:
    batch: int = 32
    hkv: int = 2
    page_table_stride: int = 86008
    sm_scale: float = 0.125
    qk_factor: float = 5.4730064863875

    def __post_init__(self):
        for name in ('batch', 'hkv', 'page_table_stride'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name+' must be a positive integer configuration')
        if self.page_table_stride < 8:
            raise ValueError('Page table must have room for the eight query tokens')
        for name in ('sm_scale', 'qk_factor'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')

    @property
    def grid(self):
        return (self.batch, self.hkv)

    def compile_kwargs(self, overlap):
        if type(overlap) is not bool:
            raise ValueError('OVERLAP must be a literal boolean')
        return dict(HKV=self.hkv, HQ=self.hkv*8,
                    PT_STRIDE=self.page_table_stride, SM_SCALE=self.sm_scale,
                    QK_FACTOR=self.qk_factor, OVERLAP=overlap,
                    num_warps=4, num_stages=1)


def output_shape(config):
    return (config.batch, 8, config.hkv*8, 64)


def page_table_shape(config):
    return (config.batch, config.page_table_stride)
