// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file must be kept in sync with index_no_bound_checks.go.

//+build bounds

package mat64

import "github.com/gonum/blas"

func (m *Dense) At(r, c int) float64 {
	return m.at(r, c)
}

func (m *Dense) at(r, c int) float64 {
	if r >= m.mat.Rows || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= m.mat.Cols || c < 0 {
		panic(ErrColAccess)
	}
	return m.mat.Data[r*m.mat.Stride+c]
}

func (m *Dense) Set(r, c int, v float64) {
	m.set(r, c, v)
}

func (m *Dense) set(r, c int, v float64) {
	if r >= m.mat.Rows || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= m.mat.Cols || c < 0 {
		panic(ErrColAccess)
	}
	m.mat.Data[r*m.mat.Stride+c] = v
}

func (m *Vector) At(r, c int) float64 {
	if c != 0 {
		panic(ErrColAccess)
	}
	return m.at(r)
}

func (m *Vector) at(r int) float64 {
	if r < 0 || r >= m.n {
		panic(ErrRowAccess)
	}
	return m.mat.Data[r*m.mat.Inc]
}

func (m *Vector) Set(r, c int, v float64) {
	if c != 0 {
		panic(ErrColAccess)
	}
	m.set(r, v)
}

func (m *Vector) set(r int, v float64) {
	if r < 0 || r >= m.n {
		panic(ErrRowAccess)
	}
	m.mat.Data[r*m.mat.Inc] = v
}

// At returns the element at row r and column c.
func (t *Triangular) At(r, c int) float64 {
	return t.at(r, c)
}

func (t *Triangular) at(r, c int) float64 {
	if r >= t.mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if t.mat.Uplo == blas.Upper {
		if r > c {
			return 0
		}
		return t.mat.Data[r*t.mat.Stride+c]
	}
	if r < c {
		return 0
	}
	return t.mat.Data[r*t.mat.Stride+c]
}

// Set sets the element at row r and column c. Set panics if the location is outside
// the appropriate half of the matrix.
func (t *Triangular) Set(r, c int, v float64) {
	t.set(r, c, v)
}

func (t *Triangular) set(r, c int, v float64) {
	if r >= t.mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if t.mat.Uplo == blas.Upper && r > c {
		panic("mat64: triangular set out of bounds")
	}
	if t.mat.Uplo == blas.Lower && r < c {
		panic("mat64: triangular set out of bounds")
	}
	t.mat.Data[r*t.mat.Stride+c] = v
}
