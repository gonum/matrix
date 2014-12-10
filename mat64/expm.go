package mat64

import (
	"math"
)

// array to cache the factorials upto 20!
var facts = make([]float64, 20)

// calculate the factorials and cache it
func factorialMemoized(n float64) float64 {
	if facts[int(n)] != 0 {
		res := facts[int(n)]
		return res
	}

	if n > 0 {
		res := n * factorialMemoized(n-1)
		return res
	}
	return 1.0
}

// Taylor Series Optimum Value Pair
type taylor struct {
	k, j float64
}

// www.cs.cornell.edu/cv/researchpdf/19ways+.pdf
// https://github.com/poliu2s/MKL/blob/master/matrix_exponential.cpp
// ExpMC computes the exponential of an nxn matrix concurrently using
// Taylor Series + Scaling and Squaring Method
// a and id should be same size
func (m *Dense) ExpMC(a, id *Dense) {
	// Use as many CPUs as available (good?)
	// runtime.GOMAXPROCS(runtime.NumCPU())
	ar, ac := a.Dims()
	idr, idc := id.Dims()
	if ar != idr && ac != idc {
		panic(ErrShape)
	}

	// hardcoded optimum value pairs
	// may not be needed but keeps it clean
	// incase some update to the code is added
	// just use a switch statement for different vals
	// taylor.j can be taken as 0.0
	tay := taylor{10, 1.0}

	var (
		// A divided by 2 power j (A / 2^j)
		Ascl = new(Dense)
		// Aj = A/2^j
		Aj = new(Dense)
		// jth power of A (A^j) divided by j factorial(j!)
		Ajj = new(Dense)
		// temporary Aj
		Ajt = new(Dense)
		// result matrix
		Ar = new(Dense)
	)

	Ascl.Scale(1/math.Pow(2, tay.j), a)

	Aj.Clone(Ascl)
	AjD := Aj.RawMatrix().Data

	Ajj.Clone(id)
	AjjD := Ajj.RawMatrix().Data

	// initialize result matrix with Eye()
	Ar.Clone(id)

	emChan := make(chan *Dense, 1)

	// Exponentiation
	fact_i := 0.0
	exp := func() {
		go func() {
			for j := 1.0; j < tay.k; j++ {
				fact_i = factorialMemoized(j)

				for i := 0; i < ar*ac; i++ {
					AjjD[i] = AjD[i] / fact_i
				}

				Ajt.Clone(Aj)

				// I + Mjj
				Ar.Add(Ajj, Ar)

				Aj.Mul(Ajt, Ascl)
			}
			emChan <- Ar
		}()
	}

	// need further investigation into the performance difference
	// since they produce the same result except go exp() runs independently
	// while the other one stalls until it is finished
	go exp()
	// exp()

	if r, ok := <-emChan; ok {
		// squaring
		for i := 0.0; i < tay.j; i++ {
			m.Clone(r)
			m.Mul(m, r)
		}
	}
}

// Same as ExpMC
// Serial Implementation
func (m *Dense) ExpMS(a, id *Dense) {
	ar, ac := a.Dims()
	idr, idc := id.Dims()
	if ar != idr && ac != idc {
		panic(ErrShape)
	}

	tay := taylor{10, 1.0}

	var (
		// A divided by 2 power j (A / 2^j)
		Ascl = new(Dense)
		// Aj = A/2^j
		Aj = new(Dense)
		// jth power of A (A^j) divided by j factorial(j!)
		Ajj = new(Dense)
		// temporary Aj
		Ajt = new(Dense)
		// result matrix
		Ar = new(Dense)
	)

	// A/(2^j)
	Ascl.Scale(1/math.Pow(2, tay.j), a)

	Aj.Clone(Ascl)
	AjD := Aj.RawMatrix().Data

	Ajj.Clone(id)
	AjjD := Ajj.RawMatrix().Data

	// initialize result matrix with Eye()
	Ar.Clone(id)

	// Exponentiation
	fact_i := 0.0
	for j := 1.0; j < tay.k; j++ {
		fact_i = factorialMemoized(j)

		for i := 0; i < ar*ac; i++ {
			AjjD[i] = AjD[i] / fact_i
		}

		Ajt.Clone(Aj)

		// I + Ajj
		Ar.Add(Ajj, Ar)

		Aj.Mul(Ajt, Ascl)
	}

	// Squaring
	for i := 0.0; i < tay.j; i++ {
		m.Clone(Ar)
		m.Mul(m, Ar)
	}
}
