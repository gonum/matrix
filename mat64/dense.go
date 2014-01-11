// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"github.com/gonum/blas"
	"math"
)

var blasEngine blas.Float64

func Register(b blas.Float64) { blasEngine = b }

func Registered() blas.Float64 { return blasEngine }

const BlasOrder = blas.RowMajor


type Dense struct {
	mat BlasMatrix
}


// NewDense creates a Dense of required dimensions
// and returns the pointer to it.
func NewDense(r, c int) *Dense {
	return &Dense{BlasMatrix{
		Order:  BlasOrder,
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   make([]float64, r * c),
	}}
}



// MakeDense returns a Dense (not *Dense) that wraps the provided
// data, the length of which must be compatible with
// the required dimensions of the Dense.
func MakeDense(r, c int, data []float64) *Dense {
    if len(data) != r * c {
        panic(ErrInLength)
    }
	return &Dense{BlasMatrix{
		Order:  BlasOrder,
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   data,
	}}
}


func (m *Dense) LoadBlas(b BlasMatrix) {
	if b.Order != BlasOrder {
		panic(ErrIllegalOrder)
	}
	m.mat = b
}


func (m *Dense) isZero() bool {
	return m.mat.Cols == 0 || m.mat.Rows == 0
}


func (m *Dense) Dims() (r, c int) { return m.mat.Rows, m.mat.Cols }

func (m *Dense) Rows() int { return m.mat.Rows }

func (m *Dense) Cols() int { return m.mat.Cols }

func (m *Dense) Size() int { return m.mat.Rows * m.mat.Cols }



func (m *Dense) validate_row_idx(r int) {
	if r >= m.mat.Rows || r < 0 {
		panic(ErrIndexOutOfRange)
	}
}




func (m *Dense) validate_col_idx(c int) {
	if c >= m.mat.Cols || c < 0 {
		panic(ErrIndexOutOfRange)
	}
}




// Contiguous reports whether the data of the matrix is stored
// in a contiguous segment of a slice.
// The returned value is false if and only if the matrix is
// the submatrix view of another matrix and has fewer columns
// than its parent matrix; otherwise, the value is true.
// If this function returns true, one may subsequently
// call DataView to get a view of the data slice and work on it directly.
func (m *Dense) Contiguous() bool { return m.mat.Cols == m.mat.Stride }




func (m *Dense) At(r, c int) float64 {
	return m.mat.Data[r*m.mat.Stride+c]
}




func (m *Dense) Set(r, c int, v float64) {
	m.mat.Data[r*m.mat.Stride+c] = v
}




func (m *Dense) RowView(r int) []float64 {
    m.validate_row_idx(r)
    k := r * m.mat.Stride
    return m.mat.Data[k : k + m.mat.Cols]
}




func (m *Dense) RowCopy(r int, row []float64) []float64 {
    row = use_slice(row, m.mat.Cols, ErrOutLength)
	copy(row, m.RowView(r))
	return row
}




func (m *Dense) SetRow(r int, v []float64) {
    if len(v) != m.mat.Cols {
        panic(ErrInLength)
    }
	copy(m.RowView(r), v)
}



// DataView returns the slice in the matrix object
// that holds the data, in row major.
// Subsequent changes to the returned slice is reflected
// in the original matrix, and vice versa.
// This is possible only when Contiguous() is true;
// if Contiguous() is false, nil is returned.
func (m *Dense) DataView() []float64 {
    if m.Contiguous() {
        return m.mat.Data
    }
    return nil
        // TODO: return nil here or panic?
}



// Data copies out all elements of the matrix, row by row.
// If out is nil, a slice is allocated;
// otherwise out must have the right length.
// The copied slice is returned.
func (m *Dense) Data(out []float64) []float64 {
    if out == nil {
        out = make([]float64, m.Size())
    } else {
        if len(out) != m.Size() {
            panic(ErrOutLength)
        }
    }
    if m.Contiguous() {
        copy(out, m.DataView())
    } else {
        r, c := m.Dims()
        for row, k := 0, 0; row < r; row++ {
            copy(out[k : k + c], m.RowView(row))
            k += c
        }
    }
    return out
}


func (m *Dense) Col(col []float64, c int) []float64 {
	if c >= m.mat.Cols || c < 0 {
		panic(ErrIndexOutOfRange)
	}

	if col == nil {
		col = make([]float64, m.mat.Rows)
	}
	col = col[:min(len(col), m.mat.Rows)]
	if blasEngine == nil {
		panic(ErrNoEngine)
	}
	blasEngine.Dcopy(len(col), m.mat.Data[c:], m.mat.Stride, col, 1)

	return col
}

func (m *Dense) SetCol(c int, v []float64) int {
	if c >= m.mat.Cols || c < 0 {
		panic(ErrIndexOutOfRange)
	}

	if blasEngine == nil {
		panic(ErrNoEngine)
	}
	blasEngine.Dcopy(min(len(v), m.mat.Rows), v, 1, m.mat.Data[c:], m.mat.Stride)

	return min(len(v), m.mat.Rows)
}



// View returns a view on the receiver.
func (m *Dense) View(i, j, r, c int) {
	m.mat.Data = m.mat.Data[i*m.mat.Stride+j : (i+r-1)*m.mat.Stride+(j+c)]
	m.mat.Rows = r
	m.mat.Cols = c
}

/*
func (m *Dense) Submatrix(a Matrix, i, j, r, c int) {
	// This is probably a bad idea, but for the moment, we do it.
	v := *m
	v.View(i, j, r, c)
	m.Clone(&Dense{v.BlasMatrix()})
}
*/

func (m *Dense) Clone(a *Dense) {
	r, c := a.Dims()
	m.mat = BlasMatrix{
		Order: BlasOrder,
		Rows:  r,
		Cols:  c,
	}
	data := make([]float64, r*c)
    for i := 0; i < r; i++ {
        copy(data[i*c:(i+1)*c], a.mat.Data[i*a.mat.Stride:i*a.mat.Stride+c])
    }
    m.mat.Stride = c
    m.mat.Data = data
}

func (m *Dense) Copy(a *Dense) (r, c int) {
	r, c = a.Dims()
	r = min(r, m.mat.Rows)
	c = min(c, m.mat.Cols)

    for i := 0; i < r; i++ {
        copy(m.mat.Data[i*m.mat.Stride:i*m.mat.Stride+c], a.mat.Data[i*a.mat.Stride:i*a.mat.Stride+c])
    }

	return r, c
}

func (m *Dense) Min() float64 {
	min := m.mat.Data[0]
	for k := 0; k < m.mat.Rows; k++ {
		for _, v := range m.mat.Data[k*m.mat.Stride : k*m.mat.Stride+m.mat.Cols] {
			min = math.Min(min, v)
		}
	}
	return min
}

func (m *Dense) Max() float64 {
	max := m.mat.Data[0]
	for k := 0; k < m.mat.Rows; k++ {
		for _, v := range m.mat.Data[k*m.mat.Stride : k*m.mat.Stride+m.mat.Cols] {
			max = math.Max(max, v)
		}
	}
	return max
}

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

// Norm(±2) depends on SVD, and so m must be tall or square.
func (m *Dense) Norm(ord float64) float64 {
	var n float64
	switch {
	case ord == 1:
		col := make([]float64, m.mat.Rows)
		for i := 0; i < m.mat.Cols; i++ {
			var s float64
			for _, e := range m.Col(col, i) {
				s += e
			}
			n = math.Max(math.Abs(s), n)
		}
	case math.IsInf(ord, +1):
		for i := 0; i < m.mat.Rows; i++ {
			var s float64
			for _, e := range m.RowView(i) {
				s += e
			}
			n = math.Max(math.Abs(s), n)
		}
	case ord == -1:
		n = math.MaxFloat64
		col := make([]float64, m.mat.Rows)
		for i := 0; i < m.mat.Cols; i++ {
			var s float64
			for _, e := range m.Col(col, i) {
				s += e
			}
			n = math.Min(math.Abs(s), n)
		}
	case math.IsInf(ord, -1):
		n = math.MaxFloat64
		for i := 0; i < m.mat.Rows; i++ {
			var s float64
			for _, e := range m.RowView(i) {
				s += e
			}
			n = math.Min(math.Abs(s), n)
		}
	case ord == 0:
		for i := 0; i < len(m.mat.Data); i += m.mat.Stride {
			for _, v := range m.mat.Data[i : i+m.mat.Cols] {
				n += v * v
			}
		}
		return math.Sqrt(n)
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

func (m *Dense) Add(a, b *Dense) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	if m.isZero() {
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	}

    for ja, jb, jm := 0, 0, 0; ja < ar*a.mat.Stride; ja, jb, jm = ja+a.mat.Stride, jb+b.mat.Stride, jm+m.mat.Stride {
        for i, v := range a.mat.Data[ja : ja+ac] {
            m.mat.Data[i+jm] = v + b.mat.Data[i+jb]
        }
    }
    return
}


func (m *Dense) Sub(a, b *Dense) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	if m.isZero() {
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	}

    for ja, jb, jm := 0, 0, 0; ja < ar*a.mat.Stride; ja, jb, jm = ja+a.mat.Stride, jb+b.mat.Stride, jm+m.mat.Stride {
        for i, v := range a.mat.Data[ja : ja+ac] {
            m.mat.Data[i+jm] = v - b.mat.Data[i+jb]
        }
    }
    return
}

func (m *Dense) MulElem(a, b *Dense) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	if m.isZero() {
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	}

    for ja, jb, jm := 0, 0, 0; ja < ar*a.mat.Stride; ja, jb, jm = ja+a.mat.Stride, jb+b.mat.Stride, jm+m.mat.Stride {
        for i, v := range a.mat.Data[ja : ja+ac] {
            m.mat.Data[i+jm] = v * b.mat.Data[i+jb]
        }
    }
    return
}

func (m *Dense) Dot(b *Dense) float64 {
	mr, mc := m.Dims()
	br, bc := b.Dims()

	if mr != br || mc != bc {
		panic(ErrShape)
	}

	var d float64

    if m.mat.Order != BlasOrder || b.mat.Order != BlasOrder {
        panic(ErrIllegalOrder)
    }
    for jm, jb := 0, 0; jm < mr*m.mat.Stride; jm, jb = jm+m.mat.Stride, jb+b.mat.Stride {
        for i, v := range m.mat.Data[jm : jm+mc] {
            d += v * b.mat.Data[i+jb]
        }
    }
    return d
}

func (m *Dense) Mul(a, b *Dense) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	var w Dense
	if m != a && m != b {
		w = *m
	}
	if w.isZero() {
		w.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   bc,
			Stride: bc,
			Data:   use(w.mat.Data, ar*bc),
		}
	} else if ar != w.mat.Rows || bc != w.mat.Cols {
		panic(ErrShape)
	}

    if blasEngine == nil {
        panic(ErrNoEngine)
    }
    blasEngine.Dgemm(
        BlasOrder,
        blas.NoTrans, blas.NoTrans,
        ar, bc, ac,
        1.,
        a.mat.Data, a.mat.Stride,
        b.mat.Data, b.mat.Stride,
        0.,
        w.mat.Data, w.mat.Stride)
    *m = w
    return
}


func (m *Dense) Scale(f float64, a *Dense) {
	ar, ac := a.Dims()

	if m.isZero() {
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	}

    for ja, jm := 0, 0; ja < ar*a.mat.Stride; ja, jm = ja+a.mat.Stride, jm+m.mat.Stride {
        for i, v := range a.mat.Data[ja : ja+ac] {
            m.mat.Data[i+jm] = v * f
        }
    }
    return
}


// An ApplyFunc takes a row/column index and element value and returns some function of that tuple.
type ApplyFunc func(r, c int, v float64) float64

func (m *Dense) Apply(f ApplyFunc, a *Dense) {
	ar, ac := a.Dims()

	if m.isZero() {
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	}

    for j, ja, jm := 0, 0, 0; ja < ar*a.mat.Stride; j, ja, jm = j+1, ja+a.mat.Stride, jm+m.mat.Stride {
        for i, v := range a.mat.Data[ja : ja+ac] {
            m.mat.Data[i+jm] = f(j, i, v)
        }
    }
    return
}

func (m *Dense) U(a *Dense) {
	ar, ac := a.Dims()
	if ar != ac {
		panic(ErrSquare)
	}

	switch {
	case m == a:
		m.zeroLower()
		return
	case m.isZero():
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	case ar != m.mat.Rows || ac != m.mat.Cols:
		panic(ErrShape)
	}

		copy(m.mat.Data[:ac], a.mat.Data[:ac])
		for j, ja, jm := 1, a.mat.Stride, m.mat.Stride; ja < ar*a.mat.Stride; j, ja, jm = j+1, ja+a.mat.Stride, jm+m.mat.Stride {
			zero(m.mat.Data[jm : jm+j])
			copy(m.mat.Data[jm+j:jm+ac], a.mat.Data[ja+j:ja+ac])
		}
		return
}

func (m *Dense) zeroLower() {
	for i := 1; i < m.mat.Rows; i++ {
		zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+i])
	}
}

func (m *Dense) L(a *Dense) {
	ar, ac := a.Dims()
	if ar != ac {
		panic(ErrSquare)
	}

	switch {
	case m == a:
		m.zeroUpper()
		return
	case m.isZero():
		m.mat = BlasMatrix{
			Order:  BlasOrder,
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   use(m.mat.Data, ar*ac),
		}
	case ar != m.mat.Rows || ac != m.mat.Cols:
		panic(ErrShape)
	}

    copy(m.mat.Data[:ar], a.mat.Data[:ar])
    for j, ja, jm := 1, a.mat.Stride, m.mat.Stride; ja < ac*a.mat.Stride; j, ja, jm = j+1, ja+a.mat.Stride, jm+m.mat.Stride {
        zero(m.mat.Data[jm : jm+j])
        copy(m.mat.Data[jm+j:jm+ar], a.mat.Data[ja+j:ja+ar])
    }
    return
}

func (m *Dense) zeroUpper() {
	for i := 0; i < m.mat.Rows-1; i++ {
		zero(m.mat.Data[i*m.mat.Stride+i+1 : (i+1)*m.mat.Stride])
	}
}

func (m *Dense) TCopy(a *Dense) {
	ar, ac := a.Dims()

	var w Dense
	if m != a {
		w = *m
	}
	if w.isZero() {
		w.mat = BlasMatrix{
			Order: BlasOrder,
			Rows:  ac,
			Cols:  ar,
			Data:  use(w.mat.Data, ar*ac),
		}
		w.mat.Stride = ar
	} else if ar != m.mat.Cols || ac != m.mat.Rows {
		panic(ErrShape)
	}
    for i := 0; i < ac; i++ {
        for j := 0; j < ar; j++ {
            w.Set(i, j, a.At(j, i))
        }
    }
	*m = w
}

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

func (m *Dense) Equals(b *Dense) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}

    for jb, jm := 0, 0; jm < br*m.mat.Stride; jb, jm = jb+b.mat.Stride, jm+m.mat.Stride {
        for i, v := range m.mat.Data[jm : jm+bc] {
            if v != b.mat.Data[i+jb] {
                return false
            }
        }
    }
    return true
}

func (m *Dense) EqualsApprox(b *Dense, epsilon float64) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}

    for jb, jm := 0, 0; jm < br*m.mat.Stride; jb, jm = jb+b.mat.Stride, jm+m.mat.Stride {
        for i, v := range m.mat.Data[jm : jm+bc] {
            if math.Abs(v-b.mat.Data[i+jb]) > epsilon {
                return false
            }
        }
    }
    return true
}



// BlasMatrix represents a cblas native representation of a matrix.
type BlasMatrix struct {
	Order      blas.Order
	Rows, Cols int
	Stride     int
	Data       []float64
}

// Matrix converts a BlasMatrix to a Matrix, writing the data to the matrix represented by c. If c is a
// BlasLoader, that method will be called, otherwise the matrix must be the correct shape.
func (b BlasMatrix) Matrix(c *Dense) {
    c.LoadBlas(b)
}

// A BlasLoader can directly load a BlasMatrix representation. There is no restriction on the shape of the
// receiver.
type BlasLoader interface {
	LoadBlas(a BlasMatrix)
}

// A Blasser can return a BlasMatrix representation of the receiver. Changes to the BlasMatrix.Data
// slice will be reflected in the original matrix, changes to the Rows, Cols and Stride fields will not.
type Blasser interface {
	BlasMatrix() BlasMatrix
}

// Det returns the determinant of the matrix a.
func Det(a *Dense) float64 {
	return LU(DenseCopyOf(a)).Det()
}

// Inverse returns the inverse or pseudoinverse of the matrix a.
func Inverse(a *Dense) *Dense {
	m, _ := a.Dims()
	d := make([]float64, m*m)
	for i := 0; i < m*m; i += m + 1 {
		d[i] = 1
	}
	eye := MakeDense(m, m, d)
	return Solve(a, eye)
}

// Solve returns a matrix x that satisfies ax = b.
func Solve(a, b *Dense) (x *Dense) {
	m, n := a.Dims()
	if m == n {
		return LU(DenseCopyOf(a)).Solve(DenseCopyOf(b))
	}
	return QR(DenseCopyOf(a)).Solve(DenseCopyOf(b))
}


// DenseCopyOf returns a newly allocated copy of the elements of a.
func DenseCopyOf(a *Dense) *Dense {
	d := &Dense{}
	d.Clone(a)
	return d
}
