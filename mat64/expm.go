package mat64

// Taylor Series Constants
const (
	tk = 10
	tj = 1.0
)

// www.cs.cornell.edu/cv/researchpdf/19ways+.pdf
// https://github.com/poliu2s/MKL/blob/master/matrix_exponential.cpp
// ExpM calculates the exponential of matrix and stores result into receiver
// a is the input matrix of size nxn
// at input m is an identity matrix and on output it is filled with the result
func (m *Dense) ExpM(a *Dense) {

	var (
		// A/tj
		as = new(Dense)

		// As^tj
		asj = new(Dense)

		// Asj.Clone()
		asjc = new(Dense)

		chk = new(Dense)
	)

	// if m is not identity then panic
	// if checking is removed and identity is created
	// inside, then another check will have to be made
	// to check if incoming matrix is identity matrix anyways
	// to avoid creating one if not needed
	if chk.Mul(m, m); !chk.Equals(m) {
		panic("m must be identity matrix of size a")
	}

	ar, _ := a.Dims()

	// scaling here
	as.Scale(1/tj, a)

	asj.Clone(as)

	asjc.Clone(m)

	// Exponentiation here
	fact_i := 1.0
	for j := 1.0; j < tk; j++ {
		fact_i *= j

		// asjc.Scale(1/fact_i, asj)
		for i := 0; i < ar*ar; i++ {
			asjc.mat.Data[i] = asj.mat.Data[i] / fact_i
		}

		m.Add(m, asjc)
		asj.Mul(asj, as)
	}
}
