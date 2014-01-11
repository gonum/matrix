package mat64


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
    ErrOutLength       = Error("mat64: output receiver has wrong length")
)
