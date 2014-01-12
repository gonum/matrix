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

// This package uses row-major storage.
// Every operation is affected by it.
// Do not change it.
const BlasOrder = blas.RowMajor

type Dense struct {
	rows, cols, stride int
	data               []float64
}

// NewDense creates a Dense of required dimensions
// and returns the pointer to it.
func NewDense(r, c int) *Dense {
	return &Dense{
		rows:   r,
		cols:   c,
		stride: c,
		data:   make([]float64, r*c),
	}
}

func (m *Dense) LoadData(data []float64, r, c int) {
	if len(data) != r*c {
		panic(ErrInLength)
	}
	m.rows = r
	m.cols = c
	m.stride = c
	m.data = data
}

func (m *Dense) isZero() bool {
	return m.cols == 0 || m.rows == 0
}

func (m *Dense) Dims() (r, c int) { return m.rows, m.cols }

func (m *Dense) Rows() int { return m.rows }

func (m *Dense) Cols() int { return m.cols }

func (m *Dense) validate_row_idx(r int) {
	if r >= m.rows || r < 0 {
		panic(ErrIndexOutOfRange)
	}
}

func (m *Dense) validate_col_idx(c int) {
	if c >= m.cols || c < 0 {
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
func (m *Dense) Contiguous() bool { return m.cols == m.stride }

func (m *Dense) Get(r, c int) float64 {
	return m.data[r*m.stride+c]
}

func (m *Dense) Set(r, c int, v float64) {
	m.data[r*m.stride+c] = v
}

func (m *Dense) RowView(r int) []float64 {
	m.validate_row_idx(r)
	k := r * m.stride
	return m.data[k : k+m.cols]
}

func (m *Dense) GetRow(r int, row []float64) []float64 {
	row = use_slice(row, m.cols, ErrOutLength)
	copy(row, m.RowView(r))
	return row
}

func (m *Dense) SetRow(r int, v []float64) {
	if len(v) != m.cols {
		panic(ErrInLength)
	}
	copy(m.RowView(r), v)
}

// ColView
// There is no ColView b/c of row-major.

func (m *Dense) GetCol(c int, col []float64) []float64 {
	m.validate_col_idx(c)
	col = use_slice(col, m.rows, ErrOutLength)

	if blasEngine == nil {
		panic(ErrNoEngine)
	}
	blasEngine.Dcopy(m.rows, m.data[c:], m.stride, col, 1)

	return col
}

func (m *Dense) SetCol(c int, v []float64) {
	m.validate_col_idx(c)

	if len(v) != m.rows {
		panic(ErrInLength)
	}

	if blasEngine == nil {
		panic(ErrNoEngine)
	}
	blasEngine.Dcopy(m.rows, v, 1, m.data[c:], m.stride)
}

func (m *Dense) SubmatrixView(i, j, r, c int) *Dense {
	if i < 0 || i >= m.rows || r <= 0 || i+r > m.rows {
		panic(ErrIndexOutOfRange)
	}
	if j < 0 || j >= m.cols || c <= 0 || j+c > m.cols {
		panic(ErrIndexOutOfRange)
	}

	out := Dense{}
	out.rows = r
	out.cols = c
	out.stride = m.stride
	out.data = m.data[i*m.stride+j : (i+r-1)*m.stride+(j+c)]
	return &out
}

func (m *Dense) GetSubmatrix(i, j, r, c int, out *Dense) *Dense {
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
		return m.data
	}
	return nil
	// TODO: return nil here or panic?
}

// GetData copies out all elements of the matrix, row by row.
// If out is nil, a slice is allocated;
// otherwise out must have the right length.
// The copied slice is returned.
func (m *Dense) GetData(out []float64) []float64 {
	out = use_slice(out, m.rows*m.cols, ErrOutLength)
	if m.Contiguous() {
		copy(out, m.DataView())
	} else {
		r, c := m.rows, m.cols
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
	r, c := m.rows, m.cols
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

func (m *Dense) GetDiag(out []float64) []float64 {
	if m.rows != m.cols {
		panic(ErrSquare)
	}
	out = use_slice(out, m.rows, ErrOutLength)
	for i, j := 0, 0; i < m.rows; i += m.stride + 1 {
		out[j] = m.data[i]
		j++
	}
	return out
}

func (m *Dense) SetDiag(v []float64) {
	if m.rows != m.cols {
		panic(ErrSquare)
	}
	if len(v) != m.rows {
		panic(ErrInLength)
	}
	for i, j := 0, 0; i < m.rows; i += m.stride + 1 {
		m.data[i] = v[j]
		j++
	}
}

func (m *Dense) FillDiag(v float64) {
	if m.rows != m.cols {
		panic(ErrSquare)
	}
	for row, k := 0, 0; row < m.rows; row++ {
		m.data[k] = v
		k += m.stride + 1
	}
}

func (m *Dense) Fill(v float64) {
	element_wise_unary(m, v, m, fill)
}

func Copy(dest *Dense, src *Dense) {
	if dest.rows != src.rows || dest.cols != src.cols {
		panic(ErrShape)
	}
	if dest.Contiguous() && src.Contiguous() {
		copy(dest.DataView(), src.DataView())
	} else {
		for row := 0; row < src.rows; row++ {
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
	out := NewDense(src.rows, src.cols)
	Copy(out, src)
	return out
}

func element_wise_unary(a *Dense, val float64, out *Dense,
	f func(a []float64, val float64, out []float64) []float64) *Dense {

	out = use_dense(out, a.rows, a.cols, ErrOutShape)
	if a.Contiguous() && out.Contiguous() {
		f(a.DataView(), val, out.DataView())
		return out
	}
	for row := 0; row < a.rows; row++ {
		f(a.RowView(row), val, out.RowView(row))
	}
	return out
}

func Shift(m *Dense, v float64, out *Dense) *Dense {
	return element_wise_unary(m, v, out, shift)
}

func (m *Dense) Shift(v float64) {
	Shift(m, v, m)
}

func Scale(m *Dense, v float64, out *Dense) *Dense {
	return element_wise_unary(m, v, out, scale)
}

func (m *Dense) Scale(v float64) {
	Scale(m, v, m)
}

func element_wise_binary(a, b, out *Dense,
	f func(a, b, out []float64) []float64) *Dense {

	if a.rows != b.rows || a.cols != b.cols {
		panic(ErrShape)
	}
	out = use_dense(out, a.rows, a.cols, ErrOutShape)
	if a.Contiguous() && b.Contiguous() && out.Contiguous() {
		f(a.DataView(), b.DataView(), out.DataView())
		return out
	}
	for row := 0; row < a.rows; row++ {
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
	if a.rows != b.rows || a.cols != b.cols {
		panic(ErrShape)
	}
	out = use_dense(out, a.rows, a.cols, ErrOutShape)
	if a.Contiguous() && b.Contiguous() && out.Contiguous() {
		add_scaled(a.DataView(), b.DataView(), s, out.DataView())
		return out
	}
	for row := 0; row < a.rows; row++ {
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
		a.data, a.stride,
		b.data, b.stride,
		0.,
		out.data, out.stride)

	return out
}

func Dot(a, b *Dense) float64 {
	if a.rows != b.rows || a.cols != b.cols {
		panic(ErrShape)
	}
	if a.Contiguous() && b.Contiguous() {
		return dot(a.DataView(), b.DataView())
	}
	d := 0.0
	for row := 0; row < a.rows; row++ {
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
	for row := 1; row < m.rows; row++ {
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
	for row := 1; row < m.rows; row++ {
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
	for row := 0; row < m.rows; row++ {
		v += sum(m.RowView(row))
	}
	return v
}

func (m *Dense) Trace() float64 {
	if m.rows != m.cols {
		panic(ErrSquare)
	}
	var t float64
	for i, n := 0, m.rows*m.cols; i < n; i += m.stride + 1 {
		t += m.data[i]
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
		col := make([]float64, m.rows)
		for i := 0; i < m.cols; i++ {
			var s float64
			for _, e := range m.GetCol(i, col) {
				s += e
			}
			n = math.Max(math.Abs(s), n)
		}
	case math.IsInf(ord, +1):
		for i := 0; i < m.rows; i++ {
			var s float64
			for _, e := range m.RowView(i) {
				s += e
			}
			n = math.Max(math.Abs(s), n)
		}
	case ord == -1:
		n = math.MaxFloat64
		col := make([]float64, m.rows)
		for i := 0; i < m.cols; i++ {
			var s float64
			for _, e := range m.GetCol(i, col) {
				s += e
			}
			n = math.Min(math.Abs(s), n)
		}
	case math.IsInf(ord, -1):
		n = math.MaxFloat64
		for i := 0; i < m.rows; i++ {
			var s float64
			for _, e := range m.RowView(i) {
				s += e
			}
			n = math.Min(math.Abs(s), n)
		}
	case ord == 0:
		for i := 0; i < len(m.data); i += m.stride {
			for _, v := range m.data[i : i+m.cols] {
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

	out = use_dense(out, m.rows, m.cols, ErrOutShape)
	for row := 0; row < m.rows; row++ {
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
	out = use_dense(out, m.cols, m.rows, ErrOutShape)
	for row := 0; row < m.rows; row++ {
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
		m.rows = ar
		m.cols = ac
		m.stride = ac
		m.data = use_slice(m.data, ar*ac, ErrInLength)
	case ar != m.rows || ac != m.cols:
		panic(ErrShape)
	}

	copy(m.data[:ac], a.data[:ac])
	for j, ja, jm := 1, a.stride, m.stride; ja < ar*a.stride; j, ja, jm = j+1, ja+a.stride, jm+m.stride {
		zero(m.data[jm : jm+j])
		copy(m.data[jm+j:jm+ac], a.data[ja+j:ja+ac])
	}
	return
}

func (m *Dense) zeroLower() {
	for i := 1; i < m.rows; i++ {
		zero(m.data[i*m.stride : i*m.stride+i])
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
		m.rows = ar
		m.cols = ac
		m.stride = ac
		m.data = use_slice(m.data, ar*ac, ErrInLength)
	case ar != m.rows || ac != m.cols:
		panic(ErrShape)
	}

	copy(m.data[:ar], a.data[:ar])
	for j, ja, jm := 1, a.stride, m.stride; ja < ac*a.stride; j, ja, jm = j+1, ja+a.stride, jm+m.stride {
		zero(m.data[jm : jm+j])
		copy(m.data[jm+j:jm+ar], a.data[ja+j:ja+ar])
	}
	return
}

func (m *Dense) zeroUpper() {
	for i := 0; i < m.rows-1; i++ {
		zero(m.data[i*m.stride+i+1 : (i+1)*m.stride])
	}
}

func (m *Dense) Equals(b *Dense) bool {
	br, bc := b.Dims()
	if br != m.rows || bc != m.cols {
		return false
	}

	for jb, jm := 0, 0; jm < br*m.stride; jb, jm = jb+b.stride, jm+m.stride {
		for i, v := range m.data[jm : jm+bc] {
			if v != b.data[i+jb] {
				return false
			}
		}
	}
	return true
}

func (m *Dense) EqualsApprox(b *Dense, epsilon float64) bool {
	br, bc := b.Dims()
	if br != m.rows || bc != m.cols {
		return false
	}

	for jb, jm := 0, 0; jm < br*m.stride; jb, jm = jb+b.stride, jm+m.stride {
		for i, v := range m.data[jm : jm+bc] {
			if math.Abs(v-b.data[i+jb]) > epsilon {
				return false
			}
		}
	}
	return true
}

// Det returns the determinant of the matrix a.
func Det(a *Dense) float64 {
	return LU(Clone(a)).Det()
}

func (m *Dense) Det() float64 {
	return Det(m)
}

// Inv returns the inverse or pseudoinverse of the matrix a.
func Inv(a *Dense, out *Dense) *Dense {
	eye := NewDense(a.rows, a.rows)
	eye.FillDiag(1.0)
	return Solve(a, eye, out)
}

// Solve returns a matrix x that satisfies ax = b.
// TODO: check LU and QR to see if output allocation can be avoided
// when output receiving matrix is provided.
func Solve(a, b, out *Dense) *Dense {
	out = use_dense(out, a.cols, b.cols, ErrOutShape)
	if a.rows == a.cols {
		Copy(out, LU(Clone(a)).Solve(Clone(b)))
	} else {
		Copy(out, QR(Clone(a)).Solve(Clone(b)))
	}
	return out
}
