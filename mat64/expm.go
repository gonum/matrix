package mat64

import "math"

// Ref: https://github.com/poliu2s/MKL/blob/master/matrix_exponential.cpp
// Computes the exponential of an nxn matrix concurrently
// Could make such that the receiver is the result
// But this allows chaining so whichever works out neat
func (A *Dense) ExpMC() *Dense {
	nrows, ncols := A.Dims()
	nelem := A.dataQuan()
	const (
		acc = 10.0

		// Scaling
		N = 0.0
	)

	var (
		msmall = new(Dense)
		mpwr   = zeros(nrows, ncols)
		mpwr1  = zeros(nrows, ncols)
		mexp1  = zeros(nrows, ncols)
		mexp2  = zeros(nrows, ncols)
		temp   = zeros(nrows, ncols)
		res    = eye(nrows)

		mpwrD  = mpwr.data()
		mpwr1D = mpwr1.data()
		mexp1D = mexp1.data()
		mexp2D = mexp2.data()
		tempD  = temp.data()
		resD   = res.data()

		emchan  = make(chan *Dense, 1)
		sqmchan = make(chan *Dense, 1)
	)

	// M*(1/(2^N))
	msmall.ScaleDense(1.0/math.Pow(2.0, N), M)
	msmallD := msmall.data()
	copy(mpwrD, msmallD)

	// Exponentiation Part
	fact_i := 0.0
	exp := func(i float64) {
		fact_i = FactorialMemoized(i)
		for i := 0; i < nelem; i++ {
			tempD[i] = mpwrD[i] / fact_i
		}
		copy(mpwr1D, mpwrD)
		res.Add(temp, res)

		mpwr.Mul(mpwr1, msmall)
		emchan <- res
	}

	// Squaring Part
	sqm := func(r *Dense) {
		for i := 0; i < N; i++ {
			copy(mexp1D, resD)
			copy(mexp2D, resD)
			r.Mul(mexp1, mexp2)
		}
		sqmchan <- r
	}

	for i := 1.0; i < acc; i++ {
		go exp(i)
	}

	select {
	case r := <-emchan:
		{
			go sqm(r)
			select {
			case retm := <-sqmchan:
				{
					return retm
				}
			}
		}
	}
	return nil
}

// Computes the exponential of an nxn matrix serially
// Again could make the receiver the result
func (A *Dense) ExpMS() *Dense {
	nrows, ncols := A.Dims()
	nelem := A.dataQuan()
	const (
		acc = 10.0

		// Scaling
		N = 0.0
	)

	var (
		msmall = new(Dense)
		mpwr   = zeros(nrows, ncols)
		mpwr1  = zeros(nrows, ncols)
		mexp1  = zeros(nrows, ncols)
		mexp2  = zeros(nrows, ncols)
		temp   = zeros(nrows, ncols)
		res    = eye(nrows)

		mpwrD  = mpwr.data()
		mpwr1D = mpwr1.data()
		mexp1D = mexp1.data()
		mexp2D = mexp2.data()
		tempD  = temp.data()
		resD   = res.data()
	)

	// M*(1/(2^N))
	msmall.ScaleDense(1.0/math.Pow(2.0, N), M)
	msmallD := msmall.data()
	copy(mpwrD, msmallD)

	fact_i := 0.0
	for i := 1.0; i < acc; i++ {
		// Exponentiation Part
		fact_i = FactorialMemoized(i)
		for i := 0; i < nelem; i++ {
			tempD[i] = mpwrD[i] / fact_i
		}
		copy(mpwr1D, mpwrD)
		res.Add(temp, res)

		mpwr.Mul(mpwr1, msmall)
	}

	for i := 0; i < N; i++ {
		// Squaring Part
		copy(mexp1D, resD)
		copy(mexp2D, resD)
		res.Mul(mexp1, mexp2)
	}
	return res
}

// Create a zero matrix
func zeros(r, c int) *Dense {
	return NewDense(r, c, nil)
}

// Create an identity matrix
func eye(span int) *Dense {
	A := NewDense(span, span, nil)
	for i := 0; i < span; i++ {
		A.Set(i, i, 1)
	}
	return A
}

// Gives the underlying elements array
func (A *Dense) data() []float64 {
	RawM := A.RawMatrix()
	return len(RawM)
}

// gives number of elements in matrix
func (A *Dense) dataQuan() int {
	return len(A.data())
}
