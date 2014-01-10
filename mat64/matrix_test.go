// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"github.com/gonum/blas/cblas"
	"github.com/gonum/floats"

	"fmt"
	check "launchpad.net/gocheck"
	"math/rand"
	"testing"
)

// Tests
func Test(t *testing.T) { check.TestingT(t) }

type S struct{}

var _ = check.Suite(&S{})

func leaksPanic(fn Panicker) (panicked bool) {
	defer func() {
		r := recover()
		panicked = r != nil
	}()
	Maybe(fn)
	return
}

func panics(fn func()) (panicked bool, message string) {
	defer func() {
		r := recover()
		panicked = r != nil
		message = fmt.Sprint(r)
	}()
	fn()
	return
}

func (s *S) SetUpSuite(c *check.C) { blasEngine = cblas.Blas{} }

func eye() *Dense {
	return NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
}

func (s *S) TestMaybe(c *check.C) {
	for i, test := range []struct {
		fn     Panicker
		panics bool
	}{
		{
			func() {},
			false,
		},
		{
			func() { panic("panic") },
			true,
		},
		{
			func() { panic(Error("panic")) },
			false,
		},
	} {
		c.Check(leaksPanic(test.fn), check.Equals, test.panics, check.Commentf("Test %d", i))
	}
}

func (s *S) TestNewDense(c *check.C) {
	for i, test := range []struct {
		a          []float64
		rows, cols int
		min, max   float64
		fro        float64
		mat        *Dense
	}{
		{
			[]float64{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			},
			3, 3,
			0, 0,
			0,
			&Dense{BlasMatrix{
				Order: BlasOrder,
				Rows:  3, Cols: 3,
				Stride: 3,
				Data:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			}},
		},
		{
			[]float64{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,
			},
			3, 3,
			1, 1,
			3,
			&Dense{BlasMatrix{
				Order: BlasOrder,
				Rows:  3, Cols: 3,
				Stride: 3,
				Data:   []float64{1, 1, 1, 1, 1, 1, 1, 1, 1},
			}},
		},
		{
			[]float64{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1,
			},
			3, 3,
			0, 1,
			1.7320508075688772,
			&Dense{BlasMatrix{
				Order: BlasOrder,
				Rows:  3, Cols: 3,
				Stride: 3,
				Data:   []float64{1, 0, 0, 0, 1, 0, 0, 0, 1},
			}},
		},
		{
			[]float64{
				-1, 0, 0,
				0, -1, 0,
				0, 0, -1,
			},
			3, 3,
			-1, 0,
			1.7320508075688772,
			&Dense{BlasMatrix{Order: BlasOrder,
				Rows: 3, Cols: 3,
				Stride: 3,
				Data:   []float64{-1, 0, 0, 0, -1, 0, 0, 0, -1},
			}},
		},
		{
			[]float64{
				1, 2, 3,
				4, 5, 6,
			},
			2, 3,
			1, 6,
			9.539392014169456,
			&Dense{BlasMatrix{Order: BlasOrder,
				Rows: 2, Cols: 3,
				Stride: 3,
				Data:   []float64{1, 2, 3, 4, 5, 6},
			}},
		},
		{
			[]float64{
				1, 2,
				3, 4,
				5, 6,
			},
			3, 2,
			1, 6,
			9.539392014169456,
			&Dense{BlasMatrix{
				Order: BlasOrder,
				Rows:  3, Cols: 2,
				Stride: 2,
				Data:   []float64{1, 2, 3, 4, 5, 6},
			}},
		},
	} {
		m := NewDense(test.rows, test.cols, test.a)
		rows, cols := m.Dims()
		c.Check(rows, check.Equals, test.rows, check.Commentf("Test %d", i))
		c.Check(cols, check.Equals, test.cols, check.Commentf("Test %d", i))
		c.Check(m.Min(), check.Equals, test.min, check.Commentf("Test %d", i))
		c.Check(m.Max(), check.Equals, test.max, check.Commentf("Test %d", i))
		c.Check(m.Norm(0), check.Equals, test.fro, check.Commentf("Test %d", i))
		c.Check(m, check.DeepEquals, test.mat, check.Commentf("Test %d", i))
		c.Check(m.Equals(test.mat), check.Equals, true, check.Commentf("Test %d", i))
	}
}

func (s *S) TestRowCol(c *check.C) {
	for i, af := range [][][]float64{
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
		{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
	} {
		a := NewDense(flatten(af))
		for ri, row := range af {
			c.Check(a.Row(nil, ri), check.DeepEquals, row, check.Commentf("Test %d", i))
		}
		for ci := range af[0] {
			col := make([]float64, a.mat.Rows)
			for j := range col {
				col[j] = float64(ci + 1 + j*a.mat.Cols)
			}
			c.Check(a.Col(nil, ci), check.DeepEquals, col, check.Commentf("Test %d", i))
		}
	}
}

func (s *S) TestSetRowColumn(c *check.C) {
	for _, as := range [][][]float64{
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
		{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
	} {
		for ri, row := range as {
			a := NewDense(flatten(as))
			t := &Dense{}
			t.Clone(a)
			a.SetRow(ri, make([]float64, a.mat.Cols))
			t.Sub(t, a)
			c.Check(t.Norm(0), check.Equals, floats.Norm(row, 2))
		}

		for ci := range as[0] {
			a := NewDense(flatten(as))
			t := &Dense{}
			t.Clone(a)
			a.SetCol(ci, make([]float64, a.mat.Rows))
			col := make([]float64, a.mat.Rows)
			for j := range col {
				col[j] = float64(ci + 1 + j*a.mat.Cols)
			}
			t.Sub(t, a)
			c.Check(t.Norm(0), check.Equals, floats.Norm(col, 2))
		}
	}
}

func (s *S) TestAdd(c *check.C) {
	for i, test := range []struct {
		a, b, r [][]float64
	}{
		{
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}},
		},
		{
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}},
		},
		{
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-2, 0, 0}, {0, -2, 0}, {0, 0, -2}},
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{2, 4, 6}, {8, 10, 12}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		temp := &Dense{}
		temp.Add(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		zero(temp.mat.Data)
		temp.Add(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		c.Check(func() { temp.Add(a, b) }, check.PanicMatches, "runtime error: index out of range", check.Commentf("Test %d"))

		a.Add(a, b)
		c.Check(a.Equals(r), check.Equals, true, check.Commentf("Test %d: %v sub %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data)))
	}
}

func (s *S) TestSub(c *check.C) {
	for i, test := range []struct {
		a, b, r [][]float64
	}{
		{
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{0, 0, 0}, {0, 0, 0}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		temp := &Dense{}
		temp.Sub(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		zero(temp.mat.Data)
		temp.Sub(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		c.Check(func() { temp.Sub(a, b) }, check.PanicMatches, "runtime error: index out of range", check.Commentf("Test %d"))

		a.Sub(a, b)
		c.Check(a.Equals(r), check.Equals, true, check.Commentf("Test %d: %v sub %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data)))
	}
}

func (s *S) TestMulElem(c *check.C) {
	for i, test := range []struct {
		a, b, r [][]float64
	}{
		{
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 4, 9}, {16, 25, 36}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		temp := &Dense{}
		temp.MulElem(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		zero(temp.mat.Data)
		temp.MulElem(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		c.Check(func() { temp.MulElem(a, b) }, check.PanicMatches, "runtime error: index out of range", check.Commentf("Test %d"))

		a.MulElem(a, b)
		c.Check(a.Equals(r), check.Equals, true, check.Commentf("Test %d: %v sub %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data)))
	}
}

func (s *S) TestMul(c *check.C) {
	for i, test := range []struct {
		a, b, r [][]float64
	}{
		{
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{3, 3, 3}, {3, 3, 3}, {3, 3, 3}},
		},
		{
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 2}, {3, 4}, {5, 6}},
			[][]float64{{22, 28}, {49, 64}},
		},
		{
			[][]float64{{0, 1, 1}, {0, 1, 1}, {0, 1, 1}},
			[][]float64{{0, 1, 1}, {0, 1, 1}, {0, 1, 1}},
			[][]float64{{0, 2, 2}, {0, 2, 2}, {0, 2, 2}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		temp := &Dense{}
		temp.Mul(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v add %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data)))

		zero(temp.mat.Data)
		temp.Mul(a, b)
		c.Check(temp.Equals(r), check.Equals, true, check.Commentf("Test %d: %v sub %v expect %v got %v",
			i, test.a, test.b, test.r, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data)))

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		c.Check(func() { temp.Mul(a, b) }, check.PanicMatches, "cblas: index of c out of range", check.Commentf("Test %d"))
	}
}

func randDense(size int, rho float64, rnd func() float64) (*Dense, error) {
	if size == 0 {
		return nil, ErrZeroLength
	}
	d := &Dense{BlasMatrix{
		Order: BlasOrder,
		Rows:  size, Cols: size, Stride: size,
		Data: make([]float64, size*size),
	}}
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if rand.Float64() < rho {
				d.Set(i, j, rnd())
			}
		}
	}
	return d, nil
}

func (s *S) TestLU(c *check.C) {
	for i := 0; i < 100; i++ {
		size := rand.Intn(100)
		r, err := randDense(size, rand.Float64(), rand.NormFloat64)
		if size == 0 {
			c.Check(err, check.Equals, ErrZeroLength)
			continue
		}
		c.Assert(err, check.Equals, nil)

		var (
			u, l Dense
			rc   *Dense
		)

		u.U(r)
		l.L(r)
		for m := 0; m < size; m++ {
			for n := 0; n < size; n++ {
				switch {
				case m < n: // Upper triangular matrix.
					c.Check(u.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				case m == n: // Diagonal matrix.
					c.Check(u.At(m, n), check.Equals, l.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
					c.Check(u.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				case m < n: // Lower triangular matrix.
					c.Check(l.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				}
			}
		}

		rc = DenseCopyOf(r)
		rc.U(rc)
		for m := 0; m < size; m++ {
			for n := 0; n < size; n++ {
				switch {
				case m < n: // Upper triangular matrix.
					c.Check(rc.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				case m == n: // Diagonal matrix.
					c.Check(rc.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				case m > n: // Lower triangular matrix.
					c.Check(rc.At(m, n), check.Equals, 0., check.Commentf("Test #%d At(%d, %d)", i, m, n))
				}
			}
		}

		rc = DenseCopyOf(r)
		rc.L(rc)
		for m := 0; m < size; m++ {
			for n := 0; n < size; n++ {
				switch {
				case m < n: // Upper triangular matrix.
					c.Check(rc.At(m, n), check.Equals, 0., check.Commentf("Test #%d At(%d, %d)", i, m, n))
				case m == n: // Diagonal matrix.
					c.Check(rc.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				case m > n: // Lower triangular matrix.
					c.Check(rc.At(m, n), check.Equals, r.At(m, n), check.Commentf("Test #%d At(%d, %d)", i, m, n))
				}
			}
		}
	}
}

func (s *S) TestTranspose(c *check.C) {
	for i, test := range []struct {
		a, t [][]float64
	}{
		{
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 4}, {2, 5}, {3, 6}},
		},
	} {
		a := NewDense(flatten(test.a))
		t := NewDense(flatten(test.t))

		var r, rr Dense

		r.TCopy(a)
		c.Check(r.Equals(t), check.Equals, true, check.Commentf("Test %d: %v transpose = %v", i, test.a, test.t))

		rr.TCopy(&r)
		c.Check(rr.Equals(a), check.Equals, true, check.Commentf("Test %d: %v transpose = I", i, test.a, test.t))

		zero(r.mat.Data)
		r.TCopy(a)
		c.Check(r.Equals(t), check.Equals, true, check.Commentf("Test %d: %v transpose = %v", i, test.a, test.t))

		zero(rr.mat.Data)
		rr.TCopy(&r)
		c.Check(rr.Equals(a), check.Equals, true, check.Commentf("Test %d: %v transpose = I", i, test.a, test.t))
	}
}

func (s *S) TestNorm(c *check.C) {
	for i, test := range []struct {
		a    [][]float64
		ord  float64
		norm float64
	}{
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  0,
			norm: 25.495097567963924,
		},
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  1,
			norm: 30,
		},
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  -1,
			norm: 22,
		},
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  2,
			norm: 25.46240743603639,
		},
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  -2,
			norm: 9.013990486603544e-16,
		},
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  inf,
			norm: 33,
		},
		{
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  -inf,
			norm: 6,
		},
	} {
		a := NewDense(flatten(test.a))
		c.Check(a.Norm(test.ord), check.Equals, test.norm, check.Commentf("Test %d: %v norm = %f", i, test.a, test.norm))
	}
}

func identity(r, c int, v float64) float64 { return v }

func (s *S) TestApply(c *check.C) {
	for i, test := range []struct {
		a, t [][]float64
		fn   ApplyFunc
	}{
		{
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			identity,
		},
		{
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			identity,
		},
		{
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			identity,
		},
		{
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float64{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			identity,
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			identity,
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{2, 4, 6}, {8, 10, 12}},
			func(r, c int, v float64) float64 { return v * 2 },
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{0, 2, 0}, {0, 5, 0}},
			func(r, c int, v float64) float64 {
				if c == 1 {
					return v
				}
				return 0
			},
		},
		{
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{0, 0, 0}, {4, 5, 6}},
			func(r, c int, v float64) float64 {
				if r == 1 {
					return v
				}
				return 0
			},
		},
	} {
		a := NewDense(flatten(test.a))
		t := NewDense(flatten(test.t))

		var r Dense

		r.Apply(test.fn, a)
		c.Check(r.Equals(t), check.Equals, true, check.Commentf("Test %d: obtained %v expect: %v", i, r.mat.Data, t.mat.Data))

		a.Apply(test.fn, a)
		c.Check(a.Equals(t), check.Equals, true, check.Commentf("Test %d: obtained %v expect: %v", i, a.mat.Data, t.mat.Data))
	}
}

func (s *S) TestSolve(c *check.C) {
	for _, test := range []struct {
		name   string
		panics bool
		a      [][]float64
		b      [][]float64
		x      [][]float64
	}{
		{
			name:   "OneElement",
			panics: false,
			a:      [][]float64{{6}},
			b:      [][]float64{{3}},
			x:      [][]float64{{0.5}},
		},
		{
			name:   "SquareIdentity",
			panics: false,
			a: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			b: [][]float64{
				{3},
				{2},
				{1},
			},
			x: [][]float64{
				{3},
				{2},
				{1},
			},
		},
		{
			name:   "Square",
			panics: false,
			a: [][]float64{
				{0.8147, 0.9134, 0.5528},
				{0.9058, 0.6324, 0.8723},
				{0.1270, 0.0975, 0.7612},
			},
			b: [][]float64{
				{0.278},
				{0.547},
				{0.958},
			},
			x: [][]float64{
				{-0.932687281002860},
				{0.303963920182067},
				{1.375216503507109},
			},
		},
		{
			name:   "ColumnMismatch",
			panics: true,
			a: [][]float64{
				{0.6046602879796196, 0.9405090880450124, 0.6645600532184904},
				{0.4377141871869802, 0.4246374970712657, 0.6868230728671094},
			},
			b: [][]float64{
				{0.30091186058528707},
				{0.5152126285020654},
				{0.8136399609900968},
				{0.12345},
			},
			x: [][]float64{
				{-26.618512183136257},
				{8.730387239011677},
				{12.316510032082446},
				{0.1234},
			},
		},
		{
			name:   "WideMatrix",
			panics: true,
			a: [][]float64{
				{0.8147, 0.9134, 0.5528},
				{0.9058, 0.6324, 0.8723},
			},
			b: [][]float64{
				{0.278},
				{0.547},
			},
			x: [][]float64{
				{1.037422650449745},
				{-0.620963688768783},
			},
		},

		{
			name:   "Skinny1",
			panics: false,
			a: [][]float64{
				{0.8147, 0.9134, 0.9},
				{0.9058, 0.6324, 0.9},
				{0.1270, 0.0975, 0.1},
				{1.6, 2.8, -3.5},
			},
			b: [][]float64{
				{0.278},
				{0.547},
				{-0.958},
				{1.452},
			},
			x: [][]float64{
				{0.820970340787782},
				{-0.218604626527306},
				{-0.212938815234215},
			},
		},
		{
			name:   "Skinny2",
			panics: false,
			a: [][]float64{
				{0.8147, 0.9134, 0.231, -1.65},
				{0.9058, 0.6324, 0.9, 0.72},
				{0.1270, 0.0975, 0.1, 1.723},
				{1.6, 2.8, -3.5, 0.987},
				{7.231, 9.154, 1.823, 0.9},
			},
			b: [][]float64{
				{0.278, 8.635},
				{0.547, 9.125},
				{-0.958, -0.762},
				{1.452, 1.444},
				{1.999, -7.234},
			},
			x: [][]float64{
				{1.863006789511373, 44.467887791812750},
				{-1.127270935407224, -34.073794226035126},
				{-0.527926457947330, -8.032133759788573},
				{-0.248621916204897, -2.366366415805275},
			},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))

		var x *Dense

		fn := func() {
			x = Solve(a, b)
		}

		panicked, message := panics(fn)
		if panicked {
			c.Check(panicked, check.Equals, test.panics, check.Commentf("Test %v panicked: %s", test.name, message))
			continue
		}

		trueX := NewDense(flatten(test.x))
		c.Check(x.EqualsApprox(trueX, 1e-13), check.Equals, true, check.Commentf("Test %v solution mismatch: Found %v, expected %v ", test.name, x, trueX))
	}
}

var (
	wd *Dense
)

func BenchmarkMulDense100Half(b *testing.B)        { denseMulBench(b, 100, 0.5) }
func BenchmarkMulDense100Tenth(b *testing.B)       { denseMulBench(b, 100, 0.1) }
func BenchmarkMulDense1000Half(b *testing.B)       { denseMulBench(b, 1000, 0.5) }
func BenchmarkMulDense1000Tenth(b *testing.B)      { denseMulBench(b, 1000, 0.1) }
func BenchmarkMulDense1000Hundredth(b *testing.B)  { denseMulBench(b, 1000, 0.01) }
func BenchmarkMulDense1000Thousandth(b *testing.B) { denseMulBench(b, 1000, 0.001) }
func denseMulBench(b *testing.B, size int, rho float64) {
	b.StopTimer()
	a, _ := randDense(size, rho, rand.NormFloat64)
	d, _ := randDense(size, rho, rand.NormFloat64)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		var n Dense
		n.Mul(a, d)
		wd = &n
	}
}

func BenchmarkPreMulDense100Half(b *testing.B)        { denseMulBench(b, 100, 0.5) }
func BenchmarkPreMulDense100Tenth(b *testing.B)       { denseMulBench(b, 100, 0.1) }
func BenchmarkPreMulDense1000Half(b *testing.B)       { denseMulBench(b, 1000, 0.5) }
func BenchmarkPreMulDense1000Tenth(b *testing.B)      { denseMulBench(b, 1000, 0.1) }
func BenchmarkPreMulDense1000Hundredth(b *testing.B)  { denseMulBench(b, 1000, 0.01) }
func BenchmarkPreMulDense1000Thousandth(b *testing.B) { denseMulBench(b, 1000, 0.001) }
func densePreMulBench(b *testing.B, size int, rho float64) {
	b.StopTimer()
	a, _ := randDense(size, rho, rand.NormFloat64)
	d, _ := randDense(size, rho, rand.NormFloat64)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		wd.Mul(a, d)
	}
}
