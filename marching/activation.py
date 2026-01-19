import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Union, Any

jax.config.update("jax_enable_x64", True)

class Activation:
    breakpoints: jax.Array
    values: jax.Array
    slopes: jax.Array
    intercepts: jax.Array
    eps: jax.Array
    name: str

    def __init__(self, breakpoints: jax.Array, values: jax.Array, slopes: jax.Array, intercepts: jax.Array, eps: jax.Array, name: str):
        self.breakpoints = breakpoints
        self.values = values
        self.slopes = slopes
        self.intercepts = intercepts
        self.eps = eps
        self.name = name

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def query(self, lower: jax.Array, upper: jax.Array) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    def linear_enclosure(
        self, lower: jax.Array, upper: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError

    def collapse(
        self, W: jax.Array, b: jax.Array, segment_idx: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        m = self.slopes[segment_idx]
        c = self.intercepts[segment_idx]
        return W * m.astype(jnp.float64), b * m.astype(jnp.float64) + c.astype(jnp.float64)


class ReluActivation(Activation):
    def __init__(self, eps: jax.Array=jnp.array(1e-5, dtype=jnp.float64), float_dtype=jnp.float64):
        assert float_dtype in [jnp.float32, jnp.float64]
        assert eps.dtype == float_dtype
        self.float_dtype = float_dtype
        super().__init__(
            breakpoints=jnp.array([0.0], dtype=float_dtype),
            values=jnp.array([0.0], dtype=float_dtype),
            slopes=jnp.array([0.0, 1.0], dtype=float_dtype),
            intercepts=jnp.array([0.0, 0.0], dtype=float_dtype),
            eps=eps,
            name='relu'
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.maximum(0, x)

    def query(self, lower: jax.Array, upper: jax.Array) -> Tuple[jax.Array, jax.Array]:
        mid = (lower+upper)/2
        seg_idx = jnp.where(mid<0, 0, 1)
        # bps = jnp.where((lower+self.eps < 0) & (0 < upper-self.eps), 1, 0)
        bps = jnp.sum((lower+self.eps<self.breakpoints) & (self.breakpoints<upper-self.eps), axis=0).astype(jnp.int32)
        return seg_idx, bps, jnp.array(0.0, dtype=self.float_dtype)
    
    def query_single(self, lower: jax.Array, upper: jax.Array) -> Tuple[jax.Array, jax.Array]:
        mid = (lower+upper)/2
        seg_idx = jnp.where(mid<0, 0, 1)
        # bps = jnp.where((lower+self.eps < 0) & (0 < upper-self.eps), 1, 0)
        bps = jnp.sum((lower+self.eps<self.breakpoints) & (self.breakpoints<upper-self.eps), axis=0).astype(jnp.int32)
        return seg_idx, bps, jnp.array(0.0, dtype=self.float_dtype)


class SinActivation(Activation):
    
    def __init__(self,
                 eps: jnp.ndarray = jnp.array(1e-5, dtype=jnp.float64),
                 float_dtype      = jnp.float64):

        assert float_dtype in (jnp.float32, jnp.float64)
        assert eps.dtype == float_dtype

        breakpoints = jnp.array([jnp.pi/4, jnp.pi/2, 3*jnp.pi/4, 5*jnp.pi/4, 3*jnp.pi/2, 7*jnp.pi/4])
        values = jnp.sin(breakpoints).astype(float_dtype)

        knot_x = jnp.array([breakpoints[-1]-2*jnp.pi, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4, 5*jnp.pi/4, 3*jnp.pi/2, 7*jnp.pi/4, breakpoints[0]+2*jnp.pi])
        y = jnp.sin(knot_x)

        slopes = (y[1:]-y[:-1])/(knot_x[1:]-knot_x[:-1])
        intercepts = y[:-1]-slopes*knot_x[:-1]

        super().__init__(
            breakpoints = breakpoints,
            values      = values,
            slopes      = slopes,
            intercepts  = intercepts,
            eps         = eps,
            name        = 'sin'
        )

    # ---------- evaluation -------------------------------------------------
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Fast piece-wise-linear surrogate of sin(x).

        Any real input is wrapped into [0,2π).  Vectorised over x.
        """
        # x is vector of shape (N,)
        x_wrapped = jnp.remainder(x, 2 * jnp.pi)

        # which segment?  (searchsorted returns the index of the right edge)
        seg_idx = jnp.searchsorted(self.breakpoints,
                                   x_wrapped,
                                   side="left")

        return self.slopes[seg_idx] * x_wrapped + self.intercepts[seg_idx]
    
    def get_slopes_intercepts(self, segment_idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        remainder_idx = jnp.remainder(segment_idx, self.breakpoints.shape[0])
        offset = segment_idx//self.breakpoints.shape[0]

        slopes = self.slopes[remainder_idx]
        intercept = self.intercepts[remainder_idx] - slopes*offset*2*jnp.pi

        return self.slopes[remainder_idx], intercept

    def query_single(self, lower: jnp.ndarray, upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lower_wrapped = jnp.remainder(lower, 2 * jnp.pi)
        upper_wrapped = jnp.remainder(upper, 2 * jnp.pi)

        lower_interval_end = lower+2*jnp.pi-lower_wrapped
        upper_interval_start = upper-upper_wrapped
        periods = jnp.rint((upper_interval_start-lower_interval_end)//(2*jnp.pi)).astype(jnp.int32)
        
        # >0 case, different period: 0
        def period_between(periods, lower_wrapped, upper_wrapped):
            lower_bps = jnp.sum(lower_wrapped+self.eps<self.breakpoints, axis=0).astype(jnp.int32)
            upper_bps = jnp.sum(self.breakpoints<upper_wrapped-self.eps, axis=0).astype(jnp.int32)
            periods_bps = periods*self.breakpoints.shape[0]
            bp_sum = periods_bps+lower_bps+upper_bps

            return bp_sum

        # -1 case, in the same period: 1
        def same_period(periods, lower_wrapped, upper_wrapped):
            # TODO: check if eps handling is correct
            lower_bps = jnp.sum(self.breakpoints<lower_wrapped+self.eps, axis=0).astype(jnp.int32)
            upper_bps = jnp.sum(self.breakpoints<upper_wrapped-self.eps, axis=0).astype(jnp.int32)
            bp_sum = upper_bps-lower_bps

            return bp_sum
        
        # 0 case, different period: 2
        def different_period(periods, lower_wrapped, upper_wrapped):
            lower_bps = jnp.sum(lower_wrapped+self.eps<self.breakpoints, axis=0).astype(jnp.int32)
            upper_bps = jnp.sum(self.breakpoints<upper_wrapped-self.eps, axis=0).astype(jnp.int32)
            bp_sum = lower_bps+upper_bps
            
            return bp_sum

        case = jnp.where(
            periods>0, 
            0,
            jnp.where(
                periods==-1,
                1,
                2
            )
        )
        
        bp_sum = jax.lax.switch(
            case,
                (
                period_between,
                same_period,
                different_period
            ),
            periods, lower_wrapped, upper_wrapped
        )

        mid = (lower+upper)/2
        mid_wrapped = jnp.remainder(mid, 2*jnp.pi)
        dist = mid_wrapped-self.breakpoints[:, None]
        idx = jnp.argmin(jnp.abs(dist))
        dist = dist[idx]
        offset = mid+dist

        periods_before = (jnp.floor(mid/(2*jnp.pi)).astype(jnp.int32))*self.breakpoints.shape[0]
        breakpoints_after = jnp.sum(self.breakpoints<mid_wrapped).astype(jnp.int32)
        segment_idx = breakpoints_after+periods_before
        segment_idx = jnp.where(bp_sum==0, segment_idx, -99)           

        return bp_sum, segment_idx, offset[0]


    # ---------- helper identical to your ReLU implementation --------------
    def query(self,
              lower: jnp.ndarray,
              upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        tells you which segment a
        box [lower, upper] falls into and how many break-points it crosses.
        if it contains multiple breakpoints it returns -1
        """
        lower_wrapped = jnp.remainder(lower, 2 * jnp.pi)
        upper_wrapped = jnp.remainder(upper, 2 * jnp.pi)

        lower_interval_end = lower+2*jnp.pi-lower_wrapped
        upper_interval_start = upper-upper_wrapped
        periods = jnp.rint((upper_interval_start-lower_interval_end)//(2*jnp.pi)).astype(jnp.int32)

        # >0 case, different period: 0
        def period_between(periods, lower_wrapped, upper_wrapped):
            lower_bps = jnp.sum(lower_wrapped+self.eps<self.breakpoints, axis=0).astype(jnp.int32)
            upper_bps = jnp.sum(self.breakpoints<upper_wrapped-self.eps, axis=0).astype(jnp.int32)
            periods_bps = periods*self.breakpoints.shape[0]
            bp_sum = periods_bps+lower_bps+upper_bps

            return bp_sum

        # -1 case, in the same period: 1
        def same_period(periods, lower_wrapped, upper_wrapped):
            lower_bps = jnp.sum(self.breakpoints<lower_wrapped+self.eps, axis=0).astype(jnp.int32)
            upper_bps = jnp.sum(self.breakpoints<upper_wrapped-self.eps, axis=0).astype(jnp.int32)
            bp_sum = upper_bps-lower_bps

            return bp_sum
        
        # 0 case, different period: 2
        def different_period(periods, lower_wrapped, upper_wrapped):
            lower_bps = jnp.sum(lower_wrapped+self.eps<self.breakpoints, axis=0).astype(jnp.int32)
            upper_bps = jnp.sum(self.breakpoints<upper_wrapped-self.eps, axis=0).astype(jnp.int32)
            bp_sum = lower_bps+upper_bps
            
            return bp_sum

        case = jnp.where(
            periods>0, 
            0,
            jnp.where(
                periods==-1,
                1,
                2
            )
        )
        
        
        def switch_v(case, periods, lower_wrapped, upper_wrapped, lower, upper):
            bp_sum = jax.lax.switch(
                case,
                    (
                    period_between,
                    same_period,
                    different_period
                ),
                periods, lower_wrapped, upper_wrapped
            )

            mid = (lower+upper)/2
            mid_wrapped = jnp.remainder(mid, 2*jnp.pi)
            dist = mid_wrapped-self.breakpoints[:, None]
            idx = jnp.argmin(jnp.abs(dist))
            dist = dist[idx]
            offset = mid+dist

            periods_before = (jnp.floor(mid/(2*jnp.pi)).astype(jnp.int32))*self.breakpoints.shape[0]
            breakpoints_after = jnp.sum(self.breakpoints<mid_wrapped).astype(jnp.int32)
            segment_idx = breakpoints_after+periods_before
            segment_idx = jnp.where(bp_sum==0, segment_idx, -99)           

            return bp_sum, segment_idx, offset

        bp_sum, segment_idx, offset = jax.vmap(switch_v)(case, periods, lower_wrapped, upper_wrapped, lower, upper)

        return segment_idx, bp_sum, offset

    
    def collapse(
        self, W: jax.Array, b: jax.Array, segment_idx: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        m, c = self.get_slopes_intercepts(segment_idx)
        return W * m.astype(jnp.float64), b * m.astype(jnp.float64) + c.astype(jnp.float64)
