// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	check "launchpad.net/gocheck"
	"math"
)

func (s *S) TestEigen(c *check.C) {
	for _, t := range []struct {
		a *Dense

		epsilon float64

		e, d []float64
		v    *Dense
	}{
		{
			a: make_dense(3, 3, []float64{
				1, 2, 1,
				6, -1, 0,
				-1, -2, -1,
			}),

			epsilon: math.Pow(2, -52.0),

			d: []float64{3.0000000000000044, -4.000000000000003, -1.0980273383714707e-16},
			e: []float64{0, 0, 0},
			v: make_dense(3, 3, []float64{
				-0.48507125007266627, 0.41649656391752204, 0.11785113019775795,
				-0.7276068751089995, -0.8329931278350428, 0.7071067811865481,
				0.48507125007266627, -0.4164965639175216, -1.5320646925708532,
			}),
		},
		{
			a: make_dense(3, 3, []float64{
				1, 6, -1,
				6, -1, -2,
				-1, -2, -1,
			}),

			epsilon: math.Pow(2, -52.0),

			d: []float64{-6.240753470718579, -1.3995889142010132, 6.640342384919599},
			e: []float64{0, 0, 0},
			v: make_dense(3, 3, []float64{
				-0.6134279348516111, -0.31411097261113, -0.7245967607083111,
				0.7697297716508223, -0.03251534945303795, -0.6375412384185983,
				0.17669818159240022, -0.9488293044247931, 0.2617263908869383,
			}),
		},
		{ // Jama pvals
			a: make_dense(3, 3, []float64{
				4, 1, 1,
				1, 2, 3,
				1, 3, 6,
			}),

			epsilon: math.Pow(2, -52.0),
		},
		{ // Jama evals
			a: make_dense(4, 4, []float64{
				0, 1, 0, 0,
				1, 0, 2e-7, 0,
				0, -2e-7, 0, 1,
				0, 0, 1, 0,
			}),

			epsilon: math.Pow(2, -52.0),
		},
		{ // Jama badeigs
			a: make_dense(5, 5, []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 1,
				0, 0, 0, 1, 0,
				1, 1, 0, 0, 1,
				1, 0, 1, 0, 1,
			}),

			epsilon: math.Pow(2, -52.0),
		},
	} {
		ef := Eigen(DenseCopyOf(t.a), t.epsilon)
		if t.d != nil {
			c.Check(ef.d, check.DeepEquals, t.d)
		}
		if t.e != nil {
			c.Check(ef.e, check.DeepEquals, t.e)
		}

		if t.v != nil {
			c.Check(ef.V.Equals(t.v), check.Equals, true)
		}

		t.a.Mul(t.a, ef.V)
		ef.V.Mul(ef.V, ef.D())
		c.Check(t.a.EqualsApprox(ef.V, 1e-12), check.Equals, true)
	}
}
