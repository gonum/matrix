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
