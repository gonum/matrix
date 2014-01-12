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
		Data:   make([]float64, r*c),
	}}
}

// MakeDense returns a Dense (not *Dense) that wraps the provided
// data, the length of which must be compatible with
// the required dimensions of the Dense.
func MakeDense(r, c int, data []float64) *Dense {
	if len(data) != r*c {
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
	return m.mat.Data[k : k+m.mat.Cols]
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

// ColView
// There is no ColView b/c of row-major.

func (m *Dense) ColCopy(c int, col []float64) []float64 {
	m.validate_col_idx(c)
	col = use_slice(col, m.mat.Rows, ErrOutLength)

	if blasEngine == nil {
		panic(ErrNoEngine)
	}
	blasEngine.Dcopy(m.mat.Rows, m.mat.Data[c:], m.mat.Stride, col, 1)

	return col
}

func (m *Dense) SetCol(c int, v []float64) {
	m.validate_col_idx(c)

	if len(v) != m.mat.Rows {
		panic(ErrInLength)
	}

	if blasEngine == nil {
		panic(ErrNoEngine)
	}
	blasEngine.Dcopy(m.mat.Rows, v, 1, m.mat.Data[c:], m.mat.Stride)
}

func (m *Dense) SubmatrixView(i, j, r, c int) *Dense {
	if i < 0 || i >= m.mat.Rows || r <= 0 || i+r > m.mat.Rows {
		panic(ErrIndexOutOfRange)
	}
	if j < 0 || j >= m.mat.Cols || c <= 0 || j+c > m.mat.Cols {
		panic(ErrIndexOutOfRange)
	}

	out := Dense{}
	out.mat.Rows = r
	out.mat.Cols = c
	out.mat.Stride = m.mat.Stride
	out.mat.Data = m.mat.Data[i*m.mat.Stride+j : (i+r-1)*m.mat.Stride+(j+c)]
	return &out
}

func (m *Dense) SubmatrixCopy(i, j, r, c int, out *Dense) *Dense {
	out = use_dense(out, r, c, ErrOutShape)
	Copy(out, m.SubmatrixView(i, j, r, c))
	return out
}

func (m *Dense) SetSubmatrix(i, j, r, c int, v []float64) {
	m.SubmatrixView(i, j, r, c).SetData(v)
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

// DataCopy copies out all elements of the matrix, row by row.
// If out is nil, a slice is allocated;
// otherwise out must have the right length.
// The copied slice is returned.
func (m *Dense) DataCopy(out []float64) []float64 {
	out = use_slice(out, m.mat.Rows*m.mat.Cols, ErrOutLength)
	if m.Contiguous() {
		copy(out, m.DataView())
	} else {
		r, c := m.mat.Rows, m.mat.Cols
		for row, k := 0, 0; row < r; row++ {
			copy(out[k:k+c], m.RowView(row))
			k += c
		}
	}
	return out
}

// SetData copies the values of v into the matrix.
// Values in v are supposed to be in row major, that is,
// values for the first row of the matrix, followed by
// values for the second row, and so on.
// Length of v must be equal to the total number of elements in the
// matrix.
func (m *Dense) SetData(v []float64) {
	r, c := m.mat.Rows, m.mat.Cols
	if len(v) != r*c {
		panic(ErrInLength)
	}
	if m.Contiguous() {
		copy(m.DataView(), v)
	} else {
		for k, row := 0, 0; row < r; row++ {
			copy(m.RowView(row), v[k:k+c])
			k += c
		}
	}
}

func (m *Dense) Fill(v float64) {
	if m.Contiguous() {
		fill(m.DataView(), v)
	} else {
		for row := 0; row < m.mat.Rows; row++ {
			fill(m.RowView(row), v)
		}
	}
}

func Copy(dest *Dense, src *Dense) {
	if dest.mat.Rows != src.mat.Rows || dest.mat.Cols != src.mat.Cols {
		panic(ErrShape)
	}
	if dest.Contiguous() && src.Contiguous() {
		copy(dest.DataView(), src.DataView())
	} else {
		for row := 0; row < src.mat.Rows; row++ {
			copy(dest.RowView(row), src.RowView(row))
		}
	}
}

// Clone creates a new Dense and copies the elements of src into it.
// The new Dense is returned.
// Note that while src could be a submatrix of a larger matrix,
// the cloned matrix is always freshly allocated and is its own
// entirety.
func Clone(src *Dense) *Dense {
	out := NewDense(src.mat.Rows, src.mat.Cols)
	Copy(out, src)
	return out
}

func Shift(m *Dense, v float64, out *Dense) *Dense {
	r, c := m.mat.Rows, m.mat.Cols
	out = use_dense(out, r, c, ErrOutShape)
	if m.Contiguous() && out.Contiguous() {
		shift(m.DataView(), v, out.DataView())
		return out
	}
	for row := 0; row < m.mat.Rows; row++ {
		shift(m.RowView(row), v, out.RowView(row))
	}
	return out
}

func (m *Dense) Shift(v float64) {
	Shift(m, v, m)
}

func Scale(m *Dense, v float64, out *Dense) *Dense {
	r, c := m.mat.Rows, m.mat.Cols
	out = use_dense(out, r, c, ErrOutShape)
	if m.Contiguous() && out.Contiguous() {
		scale(m.DataView(), v, out.DataView())
		return out
	}
	for row := 0; row < m.mat.Rows; row++ {
		scale(m.RowView(row), v, out.RowView(row))
	}
	return out
}

func (m *Dense) Scale(v float64) {
	Scale(m, v, m)
}

func element_wise_binary(a, b, out *Dense,
	f func(a, b, out []float64) []float64) *Dense {

	if a.mat.Rows != b.mat.Rows || a.mat.Cols != b.mat.Cols {
		panic(ErrShape)
	}
	out = use_dense(out, a.mat.Rows, a.mat.Cols, ErrOutShape)
	if a.Contiguous() && b.Contiguous() && out.Contiguous() {
		f(a.DataView(), b.DataView(), out.DataView())
		return out
	}
	for row := 0; row < a.mat.Rows; row++ {
		f(a.RowView(row), b.RowView(row), out.RowView(row))
	}
	return out
}

func Add(a, b, out *Dense) *Dense {
	return element_wise_binary(a, b, out, add)
}

func (m *Dense) Add(X *Dense) {
	Add(m, X, m)
}

func AddScaled(a, b *Dense, s float64, out *Dense) *Dense {
	if a.mat.Rows != b.mat.Rows || a.mat.Cols != b.mat.Cols {
		panic(ErrShape)
	}
	out = use_dense(out, a.mat.Rows, a.mat.Cols, ErrOutShape)
	if a.Contiguous() && b.Contiguous() && out.Contiguous() {
		add_scaled(a.DataView(), b.DataView(), s, out.DataView())
		return out
	}
	for row := 0; row < a.mat.Rows; row++ {
		add_scaled(a.RowView(row), b.RowView(row), s, out.RowView(row))
	}
	return out
}

func (m *Dense) AddScaled(X *Dense, s float64) {
	AddScaled(m, X, s, m)
}

func Subtract(a, b, out *Dense) *Dense {
	return element_wise_binary(a, b, out, subtract)
}

func (m *Dense) Subtract(X *Dense) {
	Subtract(m, X, m)
}

func Elemult(a, b, out *Dense) *Dense {
	return element_wise_binary(a, b, out, multiply)
}

func (m *Dense) Elemult(X *Dense) {
	Elemult(m, X, m)
}

func Mult(a, b, out *Dense) *Dense {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	out = use_dense(out, ar, bc, ErrOutShape)

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
		out.mat.Data, out.mat.Stride)

	return out
}

func Dot(a, b *Dense) float64 {
	if a.mat.Rows != b.mat.Rows || a.mat.Cols != b.mat.Cols {
		panic(ErrShape)
	}
	if a.Contiguous() && b.Contiguous() {
		return dot(a.DataView(), b.DataView())
	}
	d := 0.0
	for row := 0; row < a.mat.Rows; row++ {
		d += dot(a.RowView(row), b.RowView(row))
	}
	return d
}

func (m *Dense) Dot(b *Dense) float64 {
	return Dot(m, b)
}

func (m *Dense) Min() float64 {
	if m.Contiguous() {
		return min(m.DataView())
	}
	v := min(m.RowView(0))
	for row := 1; row < m.mat.Rows; row++ {
		z := min(m.RowView(row))
		if z < v {
			v = z
		}
	}
	return v
}

func (m *Dense) Max() float64 {
	if m.Contiguous() {
		return max(m.DataView())
	}
	v := max(m.RowView(0))
	for row := 1; row < m.mat.Rows; row++ {
		z := max(m.RowView(row))
		if z > v {
			v = z
		}
	}
	return v
}

func (m *Dense) Sum() float64 {
	if m.Contiguous() {
		return sum(m.DataView())
	}
	v := 0.0
	for row := 0; row < m.mat.Rows; row++ {
		v += sum(m.RowView(row))
	}
	return v
}

func (m *Dense) Trace() float64 {
	if m.mat.Rows != m.mat.Cols {
		panic(ErrSquare)
	}
	var t float64
	for i, n := 0, m.mat.Rows*m.mat.Cols; i < n; i += m.mat.Stride + 1 {
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
			for _, e := range m.ColCopy(i, col) {
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
			for _, e := range m.ColCopy(i, col) {
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

// Function f takes a row/column index and element value
// and returns some function of that tuple.
func Apply(
	m *Dense,
	f func(r, c int, v float64) float64,
	out *Dense) *Dense {

	out = use_dense(out, m.mat.Rows, m.mat.Cols, ErrOutShape)
	for row := 0; row < m.mat.Rows; row++ {
		in_row := m.RowView(row)
		out_row := out.RowView(row)
		for col, z := range in_row {
			out_row[col] = f(row, col, z)
		}
	}
	return out
}

func (m *Dense) Apply(f func(int, int, float64) float64) {
	Apply(m, f, m)
}

func T(m, out *Dense) *Dense {
	out = use_dense(out, m.mat.Cols, m.mat.Rows, ErrOutShape)
	for row := 0; row < m.mat.Rows; row++ {
		z := m.RowView(row)
		for col, val := range z {
			out.Set(col, row, val)
		}
	}
	return out
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
	return LU(Clone(a)).Det()
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
		return LU(Clone(a)).Solve(Clone(b))
	}
	return QR(Clone(a)).Solve(Clone(b))
}
