package mat64

// add returns slice out whose elements are
// element-wise sums of x and y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func add(x, y, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), ErrOutLength)
	for i, v := range x {
		out[i] = v + y[i]
	}
	return out
}

// add_scaled returns slice out whose elements are
// element-wise sums of x and scaled y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func add_scaled(x, y []float64, s float64, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), ErrOutLength)
	for i, v := range x {
		out[i] = v + y[i]*s
	}
	return out
}

// subtract returns slice out whose elements are
// element-wise differences of x and y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func subtract(x, y, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), ErrOutLength)
	for i, v := range x {
		out[i] = v - y[i]
	}
	return out
}

// multiply returns slice out whose elements are
// element-wise products of x and y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func multiply(x, y, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), ErrOutLength)
	for i, v := range x {
		out[i] = v * y[i]
	}
	return out
}

func dot(x, y []float64) float64 {
	if len(x) != len(y) {
		panic(ErrLength)
	}
	d := 0.0
	for i, v := range x {
		d += v * y[i]
	}
	return d
}


// This signature is made to be consistent with shift and scale.
func fill(_ []float64, v float64, out []float64) []float64 {
	for i := range out {
		out[i] = v
	}
    return out
}

/*
func fill(_ []float64, v float64, out []float64) []float64 {
	out[0] = v
	for i, n := 1, len(out); i < n; {
		i += copy(out[i:], out[:i])
	}
}
*/

func zero(x []float64) {
	fill(nil, 0.0, x)
}

// shift adds constant v to every element of x,
// store the result in out and returns out.
// If out is nil, a new slice will be allocated;
// otherwise, out must have the same length as x.
// out can be x itself, in which case elements
// of x are incremented by the amount v.
func shift(x []float64, v float64, out []float64) []float64 {
	out = use_slice(out, len(x), ErrOutLength)
	for i, val := range x {
		out[i] = val + v
	}
	return out
}

// scale multiplies constant v to every element of x,
// store the result in out and returns out.
// If out is nil, a new slice will be allocated;
// otherwise, out must have the same length as x.
// out can be x itself, in which case elements
// of x are scaled by the amount v.
func scale(x []float64, v float64, out []float64) []float64 {
	out = use_slice(out, len(x), ErrOutLength)
	for i, val := range x {
		out[i] = val * v
	}
	return out
}

func min(x []float64) float64 {
	v := x[0]
	for _, val := range x {
		if val < v {
			v = val
		}
	}
	return v
}

func max(x []float64) float64 {
	v := x[0]
	for _, val := range x {
		if val > v {
			v = val
		}
	}
	return v
}

func sum(x []float64) float64 {
	v := 0.0
	for _, val := range x {
		v += val
	}
	return v
}

func smaller(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func larger(a, b int) int {
	if a > b {
		return a
	}
	return b
}


// use_slice takes a slice x and required length,
// returns x if it is of correct length,
// returns a newly created slice if x is nil,
// and panic if x is non-nil but has wrong length.
func use_slice(x []float64, n int, err error) []float64 {
	if x == nil {
		return make([]float64, n)
	}
	if len(x) != n {
		panic(err)
	}
	return x
}

// use_dense takes a Dense x and required shape,
// returns x if it is of correct shape,
// returns a newly created Dense if x is nil,
// and panic if x is non-nil but has wrong shape.
func use_dense(x *Dense, r, c int, err error) *Dense {
	if x == nil {
		return NewDense(r, c)
	}
	m, n := x.Dims()
	if m != r || n != c {
		panic(err)
	}
	return x
}


func eye(k int) *Dense {
    x := NewDense(k, k)
    x.FillDiag(1.0)
    return x
}

// A Panicker is a function that may panic.
type Panicker func()

// Maybe will recover a panic with a type matrix.Error from fn, and return this error.
// Any other error is re-panicked.
func Maybe(fn Panicker) (err error) {
	defer func() {
		if r := recover(); r != nil {
			var ok bool
			if err, ok = r.(Error); ok {
				return
			}
			panic(r)
		}
	}()
	fn()
	return
}

// A FloatPanicker is a function that returns a float64 and may panic.
type FloatPanicker func() float64

// MaybeFloat will recover a panic with a type matrix.Error from fn, and return this error.
// Any other error is re-panicked.
func MaybeFloat(fn FloatPanicker) (f float64, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(Error); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	return fn(), nil
}

// Must can be used to wrap a function returning an error.
// If the returned error is not nil, Must will panic.
func Must(err error) {
	if err != nil {
		panic(err)
	}
}

// Type Error represents matrix package errors. These errors can be recovered by Maybe wrappers.
type Error string

func (err Error) Error() string { return string(err) }

const (
	ErrIndexOutOfRange = Error("mat64: index out of range")
	ErrZeroLength      = Error("mat64: zero length in matrix definition")
	ErrRowLength       = Error("mat64: row length mismatch")
	ErrColLength       = Error("mat64: col length mismatch")
	ErrSquare          = Error("mat64: expect square matrix")
	ErrNormOrder       = Error("mat64: invalid norm order for matrix")
	ErrSingular        = Error("mat64: matrix is singular")
	ErrLength          = Error("mat64: length mismatch")
	ErrShape           = Error("mat64: dimension mismatch")
	ErrIllegalStride   = Error("mat64: illegal stride")
	ErrPivot           = Error("mat64: malformed pivot list")
	ErrIllegalOrder    = Error("mat64: illegal order")
	ErrNoEngine        = Error("mat64: no blas engine registered: call Register()")
	ErrInLength        = Error("mat64: input data has wrong length")
	ErrOutLength       = Error("mat64: output receiving slice has wrong length")
	ErrOutShape        = Error("mat64: output receiving matrix has wrong shape")
)
