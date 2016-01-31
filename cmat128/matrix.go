// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmat128

// Matrix is the basic matrix interface type.
type Matrix interface {
	// Dims returns the dimensions of a Matrix.
	Dims() (r, c int)

	// At returns the value of a matrix element at (r, c). It will panic if r or c are
	// out of bounds for the matrix.
	At(r, c int) complex128

	// T returns the transpose of the Matrix. Whether T returns a copy of the
	// underlying data is implementation dependent.
	// This method may be implemented using the Transpose type, which
	// provides an implicit matrix transpose.
	T() Matrix

	// H returns the conjugate transpose of the Matrix. Whether H returns a copy of the
	// underlying data is implementation dependent.
	// This method may be implemented using the Conjugate type, which
	// provides an implicit matrix conjugate transpose.
	H() Matrix
}

var (
	_ Matrix       = Transpose{}
	_ Matrix       = Conjugate{}
	_ Untransposer = Transpose{}
	_ Untransposer = Conjugate{}
)

// Transpose is a type for performing an implicit matrix transpose. It implements
// the Matrix interface, returning values from the transpose of the matrix within.
type Transpose struct {
	Matrix Matrix
}

// At returns the value of the element at row i and column j of the transposed
// matrix, that is, row j and column i of the Matrix field.
func (t Transpose) At(i, j int) complex128 {
	return t.Matrix.At(j, i)
}

// Dims returns the dimensions of the transposed matrix. The number of rows returned
// is the number of columns in the Matrix field, and the number of columns is
// the number of rows in the Matrix field.
func (t Transpose) Dims() (r, c int) {
	c, r = t.Matrix.Dims()
	return r, c
}

// T performs an implicit transpose by returning the Matrix field.
func (t Transpose) T() Matrix {
	return t.Matrix
}

// H performs an implicit conjugate transpose.
func (t Transpose) H() Matrix {
	return Conjugate{t}
}

// Untranspose returns the Matrix field.
func (t Transpose) Untranspose() Matrix {
	return t.Matrix
}

// Transpose is a type for performing an implicit matrix transpose. It implements
// the Matrix interface, returning values from the transpose of the matrix within.
type Conjugate struct {
	Matrix Matrix
}

// At returns the value of the element at row i and column j of the transposed
// matrix, that is, row j and column i of the Matrix field.
func (h Conjugate) At(i, j int) complex128 {
	return h.Matrix.At(j, i) * complex(0, -1)
}

// Dims returns the dimensions of the transposed matrix. The number of rows returned
// is the number of columns in the Matrix field, and the number of columns is
// the number of rows in the Matrix field.
func (h Conjugate) Dims() (r, c int) {
	c, r = h.Matrix.Dims()
	return r, c
}

// T performs an implicit transpose.
func (h Conjugate) T() Matrix {
	return Transpose{h}
}

// T performs an implicit conjugate transpose by returning the Matrix field.
func (h Conjugate) H() Matrix {
	return h.Matrix
}

// Untranspose returns the Matrix field.
func (h Conjugate) Untranspose() Matrix {
	return h.Matrix
}

// Untransposer is a type that can undo an implicit transpose.
type Untransposer interface {
	// Note: This interface is needed to unify all of the Transpose types. In
	// the cmat128 methods, we need to test if the Matrix has been implicitly
	// transposed. If this is checked by testing for the specific Transpose type
	// then the behavior will be different if the user uses T() or TTri() for a
	// triangular matrix.

	// Untranspose returns the underlying Matrix stored for the implicit transpose.
	Untranspose() Matrix
}
