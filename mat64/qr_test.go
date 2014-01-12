// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	check "launchpad.net/gocheck"
)

func (s *S) TestQRD(c *check.C) {
	for _, test := range []struct {
		a    [][]float64
		name string
	}{
		{
			name: "Square",
			a: [][]float64{
				{1.3, 2.4, 8.9},
				{-2.6, 8.7, 9.1},
				{5.6, 5.8, 2.1},
			},
		},
		{
			name: "Skinny",
			a: [][]float64{
				{1.3, 2.4, 8.9},
				{-2.6, 8.7, 9.1},
				{5.6, 5.8, 2.1},
				{19.4, 5.2, -26.1},
			},
		},
	} {

		a := flatten2dense(test.a)
		qf := QR(Clone(a))
		r := qf.R()
		q := qf.Q()

		newA := Mult(q, r, nil)

		c.Check(isOrthogonal(q), check.Equals, true, check.Commentf("Test %v: Q not orthogonal", test.name))
		c.Check(isUpperTriangular(r), check.Equals, true, check.Commentf("Test %v: R not upper triangular", test.name))
		c.Check(a.EqualsApprox(newA, 1e-13), check.Equals, true, check.Commentf("Test %v: Q*R != A", test.name))
	}
}
