package mat64

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

var (
	triangular *Triangular

	_ Matrix  = triangular
	_ Mutable = triangular
)

type TriType int

const (
	Upper TriType = TriType(blas.Upper)
	Lower TriType = TriType(blas.Lower)
)

// A triangular matrix has the same underlying data representation as a Dense matrix
// but the entries that aren't in the populated half are completely ignored.

// Triangular represents an upper or lower triangular matrix.
type Triangular struct {
	mat blas64.Triangular
}

func NewTriangular(n int, t TriType, mat []float64) *Triangular {
	if n < 0 {
		panic("mat64: negative dimension")
	}
	if mat != nil && n*n != len(mat) {
		panic(ErrShape)
	}
	if mat == nil {
		mat = make([]float64, n*n)
	}
	if t != Upper && t != Lower {
		panic("mat64: bad TriSide")
	}
	return &Triangular{blas64.Triangular{
		N:      n,
		Stride: n,
		Data:   mat,
		Uplo:   blas.Uplo(t),
		Diag:   blas.NonUnit,
	}}
}

func (t *Triangular) Dims() (r, c int) {
	return t.mat.N, t.mat.N
}

func (t *Triangular) RawTriangular() blas64.Triangular {
	return t.mat
}
