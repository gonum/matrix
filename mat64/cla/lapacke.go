package cla

// #cgo linux LDFLAGS: -llapacke -lblas
// #cgo darwin LDFLAGS: -L/usr/local/Cellar/openblas/0.2.8/lib -lopenblas
// #include <stdlib.h>
// #include "lapacke.h"
import "C"
import (
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/matrix/mat64/la"
	"unsafe"
)

//import "fmt"

func Cholesky(a *mat64.Dense) la.CholeskyFactor {
	// Initialize.
	l := mat64.DenseCopyOf(a)
	L := l.BlasMatrix()
	spd := L.Rows == L.Cols
	if !spd {
		return la.CholeskyFactor{L: l, SPD: spd}
	}

	info := C.LAPACKE_dpotrf(C.int(L.Order), 'L', C.int(L.Rows),
		(*C.double)(unsafe.Pointer(&L.Data[0])), C.int(L.Stride))

	l.L(l)

	spd = info == 0
	return la.CholeskyFactor{L: l, SPD: spd}
}

func SVD(a *mat64.Dense, epsilon, small float64, wantu, wantv bool) la.SVDFactors {
	m, n := a.Dims()
	nu := m
	if n < m {
		nu = n
	}

	s := make([]float64, nu)
	A := a.BlasMatrix()
	var u, v *mat64.Dense
	if wantu || wantv {
		u, _ = mat64.NewDense(m, nu, make([]float64, m*nu))
		v, _ = mat64.NewDense(nu, n, make([]float64, nu*n))

		U := u.BlasMatrix()
		V := v.BlasMatrix()

		info := C.LAPACKE_dgesdd(C.int(A.Order), C.char('S'),
			C.int(m), C.int(n),
			(*C.double)(unsafe.Pointer(&A.Data[0])), C.int(A.Stride),
			(*C.double)(unsafe.Pointer(&s[0])),
			(*C.double)(unsafe.Pointer(&U.Data[0])), C.int(U.Stride),
			(*C.double)(unsafe.Pointer(&V.Data[0])), C.int(V.Stride))
		if info != 0 {
			panic("Lapacke error")
		}
	} else {
		nmax := m
		if n > m {
			nmax = n
		}
		info := C.LAPACKE_dgesdd(C.int(A.Order), C.char('N'),
			C.int(m), C.int(n),
			(*C.double)(unsafe.Pointer(&A.Data[0])), C.int(A.Stride),
			(*C.double)(unsafe.Pointer(&s[0])),
			(*C.double)(nil), C.int(nmax),
			(*C.double)(nil), C.int(nmax))
		if info != 0 {
			panic("Lapacke error")
		}
	}

	if wantv {
		v.TCopy(v)
	} else {
		v = nil
	}

	if !wantu {
		u = nil
	}

	return la.SVDFactors{u, s, v}
}

/*func procUplo(uplo blas.Uplo) C.char {
	if uplo == blas.Upper {
		return 'U'
	}
	if uplo == blas.Lower {
		return 'L'
	}
	return 0
}

func (Lapack) Dgesvd(order blas.Order, jobu lapack.Job, jobvt lapack.Job, m int, n int,
	a []float64, lda int, s []float64, u []float64, ldu int,
	vt []float64, ldvt int, superb []float64) lapack.Info {
	info := C.LAPACKE_dgesvd(C.int(order), C.char(jobu), C.char(jobvt),
		C.int(m), C.int(n),
		(*C.double)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.double)(unsafe.Pointer(&s[0])),
		(*C.double)(unsafe.Pointer(&u[0])), C.int(ldu),
		(*C.double)(unsafe.Pointer(&vt[0])), C.int(ldvt),
		(*C.double)(unsafe.Pointer(&superb[0])))
	return lapack.Info(info)
}

func (Lapack) Dpotrf(order blas.Order, uplo blas.Uplo, n int, a []float64,
	lda int) lapack.Info {
	info := C.LAPACKE_dpotrf(C.int(order), procUplo(uplo), C.int(n),
		(*C.double)(unsafe.Pointer(&a[0])), C.int(lda))
	return lapack.Info(info)
}

func (Lapack) Dsytrf(order blas.Order, uplo blas.Uplo, n int, a []float64,
	lda int, ipiv []int) lapack.Info {
	info := C.LAPACKE_dsytrf(C.int(order), procUplo(uplo), C.int(n),
		(*C.double)(unsafe.Pointer(&a[0])),
		C.int(lda), (*C.int)(unsafe.Pointer(&ipiv[0])))
	return lapack.Info(info)
}*/
