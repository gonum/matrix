// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	check "launchpad.net/gocheck"
)

func (s *S) TestLUD(c *check.C) {
	for _, t := range []struct {
		a *Dense

		l *Dense
		u *Dense

		pivot []int
		sign  int
	}{
		{ // This is a hard coded equivalent of the approach used in the Jama LU test.
			a: make_dense(3, 3, []float64{
				0, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),

			l: make_dense(3, 3, []float64{
				1, 0, 0,
				0, 1, 0,
				0.5714285714285714, 0.2142857142857144, 1,
			}),
			u: make_dense(3, 3, []float64{
				7, 8, 9,
				0, 2, 3,
				0, 0, 0.2142857142857144,
			}),
			pivot: []int{
				2, // 0 0 1
				0, // 1 0 0
				1, // 0 1 0
			},
			sign: 1,
		},
	} {
		lf := LU(Clone(t.a))
		if t.pivot != nil {
			c.Check(lf.Pivot, check.DeepEquals, t.pivot)
			c.Check(lf.Sign, check.Equals, t.sign)
		}

		l := lf.L()
		if t.l != nil {
			c.Check(l.Equals(t.l), check.Equals, true)
		}
		u := lf.U()
		if t.u != nil {
			c.Check(u.Equals(t.u), check.Equals, true)
		}

		l = Mult(l, u, nil)
		c.Check(l.EqualsApprox(pivotRows(Clone(t.a), lf.Pivot), 1e-12), check.Equals, true)

		x := lf.Solve(eye())
		t.a = Mult(t.a, x, nil)
		c.Check(t.a.EqualsApprox(eye(), 1e-12), check.Equals, true)
	}
}

func (s *S) TestLUDGaussian(c *check.C) {
	for _, t := range []struct {
		a *Dense

		l *Dense
		u *Dense

		pivot []int
		sign  int
	}{
		{ // This is a hard coded equivalent of the approach used in the Jama LU test.
			a: make_dense(3, 3, []float64{
				0, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),

			l: make_dense(3, 3, []float64{
				1, 0, 0,
				0, 1, 0,
				0.5714285714285714, 0.2142857142857144, 1,
			}),
			u: make_dense(3, 3, []float64{
				7, 8, 9,
				0, 2, 3,
				0, 0, 0.2142857142857144,
			}),
			pivot: []int{
				2, // 0 0 1
				0, // 1 0 0
				1, // 0 1 0
			},
			sign: 1,
		},
	} {
		lf := LUGaussian(Clone(t.a))
		if t.pivot != nil {
			c.Check(lf.Pivot, check.DeepEquals, t.pivot)
			c.Check(lf.Sign, check.Equals, t.sign)
		}

		l := lf.L()
		if t.l != nil {
			c.Check(l.Equals(t.l), check.Equals, true)
		}
		u := lf.U()
		if t.u != nil {
			c.Check(u.Equals(t.u), check.Equals, true)
		}

		l = Mult(l, u, nil)
		c.Check(l.EqualsApprox(pivotRows(Clone(t.a), lf.Pivot), 1e-12), check.Equals, true)

		aInv := Inverse(t.a)
		aInv = Mult(aInv, t.a, nil)
		c.Check(aInv.EqualsApprox(eye(), 1e-12), check.Equals, true)

		x := lf.Solve(eye())
		t.a = Mult(t.a, x, nil)
		c.Check(t.a.EqualsApprox(eye(), 1e-12), check.Equals, true)
	}
}
