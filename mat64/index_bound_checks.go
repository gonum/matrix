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
		panic("index error: row access out of bounds")
	}
	if c >= m.mat.Cols || c < 0 {
		panic("index error: column access out of bounds")
	}
	return m.mat.Data[r*m.mat.Stride+c]
}

func (m *Dense) Set(r, c int, v float64) {
	m.set(r, c, v)
}

func (m *Dense) set(r, c int, v float64) {
	if r >= m.mat.Rows || r < 0 {
		panic("index error: row access out of bounds")
	}
	if c >= m.mat.Cols || c < 0 {
		panic("index error: column access out of bounds")
	}
	m.mat.Data[r*m.mat.Stride+c] = v
}

// At returns the element at row r and column c.
func (t *Symmetric) At(r, c int) float64 {
	return t.at(r, c)
}
func (t *Symmetric) at(r, c int) float64 {
	if r >= t.mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if t.mat.Uplo == blas.Upper {
		if r > c {
			r, c = c, r
		}
		return t.mat.Data[r*t.mat.Stride+c]
	}
	if r < c {
		r, c = c, r
	}
	return t.mat.Data[r*t.mat.Stride+c]
}

// Set sets the element at row r and column c. Set panics if the location is outside
// the appropriate half of the matrix.
func (t *Symmetric) Set(r, c int, v float64) {
	t.set(r, c, v)
}
func (t *Symmetric) set(r, c int, v float64) {
	if r >= t.mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if t.mat.Uplo == blas.Upper && r > c {
		panic("mat64: symmetric set out of bounds")
	}
	if t.mat.Uplo == blas.Lower && r < c {
		panic("mat64: symmetric set out of bounds")
	}
	t.mat.Data[r*t.mat.Stride+c] = v
}
