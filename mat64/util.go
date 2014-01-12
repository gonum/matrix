package mat64



func fill(x []float64, v float64) {
    for i := range x {
        x[i] = v
    }
}



func zero(x []float64) {
    fill(x, 0.0)
}


/*
func zero(f []float64) {
	f[0] = 0
	for i := 1; i < len(f); {
		i += copy(f[i:], f[:i])
	}
}
*/




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
        out[i] = v + y[i] * s
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




func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// use returns a float64 slice with l elements, using f if it
// has the necessary capacity, otherwise creating a new slice.
func use(f []float64, l int) []float64 {
	if l < cap(f) {
		return f[:l]
	}
	return make([]float64, l)
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
	ErrShape           = Error("mat64: dimension mismatch")
	ErrIllegalStride   = Error("mat64: illegal stride")
	ErrPivot           = Error("mat64: malformed pivot list")
	ErrIllegalOrder    = Error("mat64: illegal order")
	ErrNoEngine        = Error("mat64: no blas engine registered: call Register()")
    ErrInLength        = Error("mat64: input data has wrong length")
    ErrOutLength       = Error("mat64: output receiving slice has wrong length")
    ErrOutShape        = Error("mat64: output receiving matrix has wrong shape")
)
