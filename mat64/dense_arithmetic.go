// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"math"
	"runtime"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

// Min returns the smallest element value of the receiver.
func (m *Dense) Min() float64 {
	min := m.mat.Data[0]
	for k := 0; k < m.mat.Rows; k++ {
		for _, v := range m.rowView(k) {
			min = math.Min(min, v)
		}
	}
	return min
}

// Max returns the largest element value of the receiver.
func (m *Dense) Max() float64 {
	max := m.mat.Data[0]
	for k := 0; k < m.mat.Rows; k++ {
		for _, v := range m.rowView(k) {
			max = math.Max(max, v)
		}
	}
	return max
}

// Trace returns the trace of the matrix.
//
// See the Tracer interface for more information.
func (m *Dense) Trace() float64 {
	if m.mat.Rows != m.mat.Cols {
		panic(ErrSquare)
	}
	var t float64
	for i := 0; i < len(m.mat.Data); i += m.mat.Stride + 1 {
		t += m.mat.Data[i]
	}
	return t
}

var inf = math.Inf(1)

const (
	epsilon = 2.2204e-16
	small   = math.SmallestNonzeroFloat64
)

// Norm returns the specified matrix p-norm of the receiver.
//
// See the Normer interface for more information.
func (m *Dense) Norm(ord float64) float64 {
	var n float64
	switch {
	case ord == 1:
		col := make([]float64, m.mat.Rows)
		for i := 0; i < m.mat.Cols; i++ {
			var s float64
			for _, e := range m.Col(col, i) {
				s += math.Abs(e)
			}
			n = math.Max(s, n)
		}
	case math.IsInf(ord, +1):
		row := make([]float64, m.mat.Cols)
		for i := 0; i < m.mat.Rows; i++ {
			var s float64
			for _, e := range m.Row(row, i) {
				s += math.Abs(e)
			}
			n = math.Max(s, n)
		}
	case ord == -1:
		n = math.MaxFloat64
		col := make([]float64, m.mat.Rows)
		for i := 0; i < m.mat.Cols; i++ {
			var s float64
			for _, e := range m.Col(col, i) {
				s += math.Abs(e)
			}
			n = math.Min(s, n)
		}
	case math.IsInf(ord, -1):
		n = math.MaxFloat64
		row := make([]float64, m.mat.Cols)
		for i := 0; i < m.mat.Rows; i++ {
			var s float64
			for _, e := range m.Row(row, i) {
				s += math.Abs(e)
			}
			n = math.Min(s, n)
		}
	case ord == 0:
		for i := 0; i < len(m.mat.Data); i += m.mat.Stride {
			for _, v := range m.mat.Data[i : i+m.mat.Cols] {
				n = math.Hypot(n, v)
			}
		}
		return n
	case ord == 2, ord == -2:
		s := SVD(m, epsilon, small, false, false).Sigma
		if ord == 2 {
			return s[0]
		}
		return s[len(s)-1]
	default:
		panic(ErrNormOrder)
	}

	return n
}

// Add adds a and b element-wise, placing the result in the receiver.
//
// See the Adder interface for more information.
func (m *Dense) Add(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		if b, ok := b.(RawMatrixer); ok {
			amat, bmat := a.RawMatrix(), b.RawMatrix()
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v + bmat.Data[i+jb]
				}
			}
			return
		}
	}

	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			rowa := make([]float64, ac)
			rowb := make([]float64, bc)
			for r := 0; r < ar; r++ {
				a.Row(rowa, r)
				for i, v := range b.Row(rowb, r) {
					rowa[i] += v
				}
				copy(m.rowView(r), rowa)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)+b.At(r, c))
		}
	}
}

// Sub subtracts the matrix b from a, placing the result in the receiver.
//
// See the Suber interface for more information.
func (m *Dense) Sub(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		if b, ok := b.(RawMatrixer); ok {
			amat, bmat := a.RawMatrix(), b.RawMatrix()
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v - bmat.Data[i+jb]
				}
			}
			return
		}
	}

	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			rowa := make([]float64, ac)
			rowb := make([]float64, bc)
			for r := 0; r < ar; r++ {
				a.Row(rowa, r)
				for i, v := range b.Row(rowb, r) {
					rowa[i] -= v
				}
				copy(m.rowView(r), rowa)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)-b.At(r, c))
		}
	}
}

// MulElem performs element-wise multiplication of a and b, placing the result
// in the receiver.
//
// See the ElemMuler interface for more information.
func (m *Dense) MulElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		if b, ok := b.(RawMatrixer); ok {
			amat, bmat := a.RawMatrix(), b.RawMatrix()
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v * bmat.Data[i+jb]
				}
			}
			return
		}
	}

	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			rowa := make([]float64, ac)
			rowb := make([]float64, bc)
			for r := 0; r < ar; r++ {
				a.Row(rowa, r)
				for i, v := range b.Row(rowb, r) {
					rowa[i] *= v
				}
				copy(m.rowView(r), rowa)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)*b.At(r, c))
		}
	}
}

// DivElem performs element-wise division of a by b, placing the result
// in the receiver.
//
// See the ElemDiver interface for more information.
func (m *Dense) DivElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		if b, ok := b.(RawMatrixer); ok {
			amat, bmat := a.RawMatrix(), b.RawMatrix()
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v / bmat.Data[i+jb]
				}
			}
			return
		}
	}

	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			rowa := make([]float64, ac)
			rowb := make([]float64, bc)
			for r := 0; r < ar; r++ {
				a.Row(rowa, r)
				for i, v := range b.Row(rowb, r) {
					rowa[i] /= v
				}
				copy(m.rowView(r), rowa)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)/b.At(r, c))
		}
	}
}

// Dot returns the sum of the element-wise products of the elements of the
// receiver and b.
//
// See the Dotter interface for more information.
func (m *Dense) Dot(b Matrix) float64 {
	mr, mc := m.Dims()
	br, bc := b.Dims()

	if mr != br || mc != bc {
		panic(ErrShape)
	}

	nCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(nCPU)

	var d float64
	if b, ok := b.(RawMatrixer); ok {

		bmat := b.RawMatrix()

		part := int(math.Ceil(float64(mr*mc) / float64(nCPU)))

		ch := make(chan float64, nCPU)
		for i := 0; i < nCPU; i++ {
			go func(i int) {
				lb := part * i
				ub := int(math.Min(float64(part*(i+1)),
					float64(mr*mc)))

				var res float64
				for k := lb; k < ub; k++ {
					res += m.mat.Data[k] * bmat.Data[k]
				}

				ch <- res
			}(i)
		}
		for i := 0; i < nCPU; i++ {
			d += <-ch
		}

		return d
	}

	if b, ok := b.(Vectorer); ok {
		row := make([]float64, bc)

		part := int(math.Ceil(float64(br) / float64(nCPU)))

		ch := make(chan float64, nCPU)
		for i := 0; i < nCPU; i++ {
			go func(i int) {
				lb := part * i
				ub := int(math.Min(float64(part*(i+1)),
					float64(br)))

				var res float64
				for r := lb; r < ub; r++ {
					for c, v := range b.Row(row, r) {
						res += m.mat.Data[r*m.mat.Stride+c] * v
					}
				}

				ch <- res
			}(i)
		}
		for i := 0; i < nCPU; i++ {
			d += <-ch
		}

		return d
	}

	part := int(math.Ceil(float64(mr) / float64(nCPU)))

	ch := make(chan float64, nCPU)
	for i := 0; i < nCPU; i++ {
		go func(i int) {
			lb := part * i
			ub := int(math.Min(float64(part*(i+1)),
				float64(mr)))

			var res float64
			for r := lb; r < ub; r++ {
				for c := 0; c < mc; c++ {
					res += m.At(r, c) * b.At(r, c)
				}
			}

			ch <- res
		}(i)
	}
	for i := 0; i < nCPU; i++ {
		d += <-ch
	}

	return d
}

// Mul takes the matrix product of a and b, placing the result in the receiver.
//
// See the Muler interface for more information.
func (m *Dense) Mul(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	var w Dense
	if m != a && m != b {
		w = *m
	}
	w.reuseAs(ar, bc)

	if a, ok := a.(RawMatrixer); ok {
		if b, ok := b.(RawMatrixer); ok {
			amat, bmat := a.RawMatrix(), b.RawMatrix()
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, amat, bmat, 0, w.mat)
			*m = w
			return
		}
	}

	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			row := make([]float64, ac)
			col := make([]float64, br)
			for r := 0; r < ar; r++ {
				dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
				for c := 0; c < bc; c++ {
					dataTmp[c] = blas64.Dot(ac,
						blas64.Vector{Inc: 1, Data: a.Row(row, r)},
						blas64.Vector{Inc: 1, Data: b.Col(col, c)},
					)
				}
			}
			*m = w
			return
		}
	}

	row := make([]float64, ac)
	for r := 0; r < ar; r++ {
		for i := range row {
			row[i] = a.At(r, i)
		}
		for c := 0; c < bc; c++ {
			var v float64
			for i, e := range row {
				v += e * b.At(i, c)
			}
			w.mat.Data[r*w.mat.Stride+c] = v
		}
	}
	*m = w
}

// MulTrans takes the matrix product of a and b, optionally transposing each,
// and placing the result in the receiver.
//
// See the MulTranser interface for more information.
func (m *Dense) MulTrans(a Matrix, aTrans bool, b Matrix, bTrans bool) {
	ar, ac := a.Dims()
	if aTrans {
		ar, ac = ac, ar
	}

	br, bc := b.Dims()
	if bTrans {
		br, bc = bc, br
	}

	if ac != br {
		panic(ErrShape)
	}

	var w Dense
	if m != a && m != b {
		w = *m
	}
	w.reuseAs(ar, bc)

	if a, ok := a.(RawMatrixer); ok {
		if b, ok := b.(RawMatrixer); ok {
			amat := a.RawMatrix()
			if a == b && aTrans != bTrans {
				var op blas.Transpose
				if aTrans {
					op = blas.Trans
				} else {
					op = blas.NoTrans
				}
				blas64.Syrk(op, 1, amat, 0, blas64.Symmetric{N: w.mat.Rows, Stride: w.mat.Stride, Data: w.mat.Data, Uplo: blas.Upper})

				// Fill lower matrix with result.
				// TODO(kortschak): Investigate whether using blas64.Copy improves the performance of this significantly.
				for i := 0; i < w.mat.Rows; i++ {
					for j := i + 1; j < w.mat.Cols; j++ {
						w.set(j, i, w.at(i, j))
					}
				}
			} else {
				var aOp, bOp blas.Transpose
				if aTrans {
					aOp = blas.Trans
				} else {
					aOp = blas.NoTrans
				}
				if bTrans {
					bOp = blas.Trans
				} else {
					bOp = blas.NoTrans
				}
				bmat := b.RawMatrix()
				blas64.Gemm(aOp, bOp, 1, amat, bmat, 0, w.mat)
			}

			*m = w
			return
		}
	}

	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			row := make([]float64, ac)
			col := make([]float64, br)
			if aTrans {
				if bTrans {
					for r := 0; r < ar; r++ {
						dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
						for c := 0; c < bc; c++ {
							dataTmp[c] = blas64.Dot(ac,
								blas64.Vector{Inc: 1, Data: a.Col(row, r)},
								blas64.Vector{Inc: 1, Data: b.Row(col, c)},
							)
						}
					}
					*m = w
					return
				}
				// TODO(jonlawlor): determine if (b*a)' is more efficient
				for r := 0; r < ar; r++ {
					dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
					for c := 0; c < bc; c++ {
						dataTmp[c] = blas64.Dot(ac,
							blas64.Vector{Inc: 1, Data: a.Col(row, r)},
							blas64.Vector{Inc: 1, Data: b.Col(col, c)},
						)
					}
				}
				*m = w
				return
			}
			if bTrans {
				for r := 0; r < ar; r++ {
					dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
					for c := 0; c < bc; c++ {
						dataTmp[c] = blas64.Dot(ac,
							blas64.Vector{Inc: 1, Data: a.Row(row, r)},
							blas64.Vector{Inc: 1, Data: b.Row(col, c)},
						)
					}
				}
				*m = w
				return
			}
			for r := 0; r < ar; r++ {
				dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
				for c := 0; c < bc; c++ {
					dataTmp[c] = blas64.Dot(ac,
						blas64.Vector{Inc: 1, Data: a.Row(row, r)},
						blas64.Vector{Inc: 1, Data: b.Col(col, c)},
					)
				}
			}
			*m = w
			return
		}
	}

	row := make([]float64, ac)
	if aTrans {
		if bTrans {
			for r := 0; r < ar; r++ {
				dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
				for i := range row {
					row[i] = a.At(i, r)
				}
				for c := 0; c < bc; c++ {
					var v float64
					for i, e := range row {
						v += e * b.At(c, i)
					}
					dataTmp[c] = v
				}
			}
			*m = w
			return
		}

		for r := 0; r < ar; r++ {
			dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
			for i := range row {
				row[i] = a.At(i, r)
			}
			for c := 0; c < bc; c++ {
				var v float64
				for i, e := range row {
					v += e * b.At(i, c)
				}
				dataTmp[c] = v
			}
		}
		*m = w
		return
	}
	if bTrans {
		for r := 0; r < ar; r++ {
			dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
			for i := range row {
				row[i] = a.At(r, i)
			}
			for c := 0; c < bc; c++ {
				var v float64
				for i, e := range row {
					v += e * b.At(c, i)
				}
				dataTmp[c] = v
			}
		}
		*m = w
		return
	}
	for r := 0; r < ar; r++ {
		dataTmp := w.mat.Data[r*w.mat.Stride : r*w.mat.Stride+bc]
		for i := range row {
			row[i] = a.At(r, i)
		}
		for c := 0; c < bc; c++ {
			var v float64
			for i, e := range row {
				v += e * b.At(i, c)
			}
			dataTmp[c] = v
		}
	}
	*m = w
}

// Exp calculates the exponential of the matrix a, e^a, placing the result
// in the receiver.
//
// See the Exper interface for more information.
//
// Exp uses the scaling and squaring method described in section 3 of
// http://www.cs.cornell.edu/cv/researchpdf/19ways+.pdf.
func (m *Dense) Exp(a Matrix) {
	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}
	switch {
	case m.isZero():
		m.mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   useZeroed(m.mat.Data, r*r),
		}
		m.capRows = r
		m.capCols = c
		for i := 0; i < r*r; i += r + 1 {
			m.mat.Data[i] = 1
		}
	case r == m.mat.Rows && c == m.mat.Cols:
		for i := 0; i < r; i++ {
			zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
			m.mat.Data[i*m.mat.Stride+i] = 1
		}
	default:
		panic(ErrShape)
	}

	const (
		terms   = 10
		scaling = 4
	)

	var small, power Dense
	small.Scale(math.Pow(2, -scaling), a)
	power.Clone(&small)

	var (
		tmp   = NewDense(r, r, nil)
		factI = 1.
	)
	for i := 1.; i < terms; i++ {
		factI *= i

		// This is OK to do because power and tmp are
		// new Dense values so all rows are contiguous.
		// TODO(kortschak) Make this explicit in the NewDense doc comment.
		for j, v := range power.mat.Data {
			tmp.mat.Data[j] = v / factI
		}

		m.Add(m, tmp)
		if i < terms-1 {
			power.Mul(&power, &small)
		}
	}

	for i := 0; i < scaling; i++ {
		m.Mul(m, m)
	}
}

// Pow calculates the integral power of the matrix a to n, placing the result
// in the receiver.
//
// See the Power interface for more information.
func (m *Dense) Pow(a Matrix, n int) {
	if n < 0 {
		panic("matrix: illegal power")
	}
	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}

	m.reuseAs(r, c)

	// Take possible fast paths.
	switch n {
	case 0:
		for i := 0; i < r; i++ {
			zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
			m.mat.Data[i*m.mat.Stride+i] = 1
		}
		return
	case 1:
		m.Copy(a)
		return
	case 2:
		m.Mul(a, a)
		return
	}

	// Perform iterative exponentiation by squaring in work space.
	var w, s, x Dense
	w.Clone(a)
	s.Clone(a)
	for n--; n > 0; n >>= 1 {
		if n&1 != 0 {
			x.Mul(&w, &s)
			w, x = x, w
		}
		if n != 1 {
			x.Mul(&s, &s)
			s, x = x, s
		}
	}
	m.Copy(&w)
}

// Scale multiplies the elements of a by f, placing the result in the receiver.
//
// See the Scaler interface for more information.
func (m *Dense) Scale(f float64, a Matrix) {
	ar, ac := a.Dims()

	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		amat := a.RawMatrix()
		for ja, jm := 0, 0; ja < ar*amat.Stride; ja, jm = ja+amat.Stride, jm+m.mat.Stride {
			for i, v := range amat.Data[ja : ja+ac] {
				m.mat.Data[i+jm] = v * f
			}
		}
		return
	}

	if a, ok := a.(Vectorer); ok {
		row := make([]float64, ac)
		for r := 0; r < ar; r++ {
			for i, v := range a.Row(row, r) {
				row[i] = f * v
			}
			copy(m.rowView(r), row)
		}
		return
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, f*a.At(r, c))
		}
	}
}

// Apply applies the function f to each of the elements of a, placing the
// resulting matrix in the receiver.
//
// See the Applyer interface for more information.
func (m *Dense) Apply(f ApplyFunc, a Matrix) {
	ar, ac := a.Dims()

	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		amat := a.RawMatrix()
		for j, ja, jm := 0, 0, 0; ja < ar*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+m.mat.Stride {
			for i, v := range amat.Data[ja : ja+ac] {
				m.mat.Data[i+jm] = f(j, i, v)
			}
		}
		return
	}

	if a, ok := a.(Vectorer); ok {
		row := make([]float64, ac)
		for r := 0; r < ar; r++ {
			for i, v := range a.Row(row, r) {
				row[i] = f(r, i, v)
			}
			copy(m.rowView(r), row)
		}
		return
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, f(r, c, a.At(r, c)))
		}
	}
}

// Sum returns the sum of the elements of the matrix.
//
// See the Sumer interface for more information.
func (m *Dense) Sum() float64 {
	l := m.mat.Cols
	var s float64
	for i := 0; i < len(m.mat.Data); i += m.mat.Stride {
		for _, v := range m.mat.Data[i : i+l] {
			s += v
		}
	}
	return s
}

// Equals returns true if b and the receiver have the same size and contain all
// equal elements.
//
// See the Equaler interface for more information.
func (m *Dense) Equals(b Matrix) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}

	if b, ok := b.(RawMatrixer); ok {
		bmat := b.RawMatrix()
		for jb, jm := 0, 0; jm < br*m.mat.Stride; jb, jm = jb+bmat.Stride, jm+m.mat.Stride {
			for i, v := range m.mat.Data[jm : jm+bc] {
				if v != bmat.Data[i+jb] {
					return false
				}
			}
		}
		return true
	}

	if b, ok := b.(Vectorer); ok {
		rowb := make([]float64, bc)
		for r := 0; r < br; r++ {
			rowm := m.mat.Data[r*m.mat.Stride : r*m.mat.Stride+m.mat.Cols]
			for i, v := range b.Row(rowb, r) {
				if rowm[i] != v {
					return false
				}
			}
		}
		return true
	}

	for r := 0; r < br; r++ {
		for c := 0; c < bc; c++ {
			if m.At(r, c) != b.At(r, c) {
				return false
			}
		}
	}
	return true
}

// EqualsApprox compares the matrices represented by b and the receiver, with
// tolerance for element-wise equality specified by epsilon.
//
// See the ApproxEqualer interface for more information.
func (m *Dense) EqualsApprox(b Matrix, epsilon float64) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}

	if b, ok := b.(RawMatrixer); ok {
		bmat := b.RawMatrix()
		for jb, jm := 0, 0; jm < br*m.mat.Stride; jb, jm = jb+bmat.Stride, jm+m.mat.Stride {
			for i, v := range m.mat.Data[jm : jm+bc] {
				if math.Abs(v-bmat.Data[i+jb]) > epsilon {
					return false
				}
			}
		}
		return true
	}

	if b, ok := b.(Vectorer); ok {
		rowb := make([]float64, bc)
		for r := 0; r < br; r++ {
			rowm := m.mat.Data[r*m.mat.Stride : r*m.mat.Stride+m.mat.Cols]
			for i, v := range b.Row(rowb, r) {
				if math.Abs(rowm[i]-v) > epsilon {
					return false
				}
			}
		}
		return true
	}

	for r := 0; r < br; r++ {
		for c := 0; c < bc; c++ {
			if math.Abs(m.At(r, c)-b.At(r, c)) > epsilon {
				return false
			}
		}
	}
	return true
}

// RankOne performs a rank-one update to the matrix b and stores the result
// in the receiver
//  m = a + alpha * x * y'
func (m *Dense) RankOne(a Matrix, alpha float64, x, y []float64) {
	ar, ac := a.Dims()

	var w Dense
	if m == a {
		w = *m
	}
	w.reuseAs(ar, ac)

	// Copy over to the new memory if necessary
	if m != a {
		w.Copy(a)
	}
	if len(x) != ar {
		panic(ErrShape)
	}
	if len(y) != ac {
		panic(ErrShape)
	}
	blas64.Ger(alpha, blas64.Vector{Inc: 1, Data: x}, blas64.Vector{Inc: 1, Data: y}, w.mat)
	*m = w
	return
}
