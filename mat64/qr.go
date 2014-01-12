// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the QRDecomposition class from Jama 1.0.3.

package mat64

import (
	"math"
)

type QRFactor struct {
	QR    *Dense
	rDiag []float64
}

// QR computes a QR Decomposition for an m-by-n matrix a with m >= n by Householder
// reflections, the QR decomposition is an m-by-n orthogonal matrix q and an n-by-n
// upper triangular matrix r so that a = q.r. QR will panic with ErrShape if m < n.
//
// The QR decomposition always exists, even if the matrix does not have full rank,
// so QR will never fail unless m < n. The primary use of the QR decomposition is
// in the least squares solution of non-square systems of simultaneous linear equations.
// This will fail if QRIsFullRank() returns false. The matrix a is overwritten by the
// decomposition.
func QR(a *Dense) QRFactor {
	// Initialize.
	m, n := a.Dims()
	if m < n {
		panic(ErrShape)
	}

	qr := a
	rDiag := make([]float64, n)

	// Main loop.
	for k := 0; k < n; k++ {
		// Compute 2-norm of k-th column without under/overflow.
		var norm float64
		for i := k; i < m; i++ {
			norm = math.Hypot(norm, qr.Get(i, k))
		}

		if norm != 0 {
			// Form k-th Householder vector.
			if qr.Get(k, k) < 0 {
				norm = -norm
			}
			for i := k; i < m; i++ {
				qr.Set(i, k, qr.Get(i, k)/norm)
			}
			qr.Set(k, k, qr.Get(k, k)+1)

			// Apply transformation to remaining columns.
			for j := k + 1; j < n; j++ {
				var s float64
				for i := k; i < m; i++ {
					s += qr.Get(i, k) * qr.Get(i, j)
				}
				s /= -qr.Get(k, k)
				for i := k; i < m; i++ {
					qr.Set(i, j, qr.Get(i, j)+s*qr.Get(i, k))
				}
			}
		}
		rDiag[k] = -norm
	}

	return QRFactor{qr, rDiag}
}

// IsFullRank returns whether the R matrix and hence a has full rank.
func (f QRFactor) IsFullRank() bool {
	for _, v := range f.rDiag {
		if v == 0 {
			return false
		}
	}
	return true
}

// H returns the Householder vectors in a lower trapezoidal matrix
// whose columns define the reflections.
func (f QRFactor) H() *Dense {
	qr := f.QR
	m, n := qr.Dims()
	h := NewDense(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i >= j {
				h.Set(i, j, qr.Get(i, j))
			}
		}
	}
	return h
}

// R returns the upper triangular factor for the QR decomposition.
func (f QRFactor) R() *Dense {
	qr, rDiag := f.QR, f.rDiag
	_, n := qr.Dims()
	r := NewDense(n, n)
	for i, v := range rDiag[:n] {
		for j := 0; j < n; j++ {
			if i < j {
				r.Set(i, j, qr.Get(i, j))
			} else if i == j {
				r.Set(i, j, v)
			}
		}
	}
	return r
}

// Q generates and returns the (economy-sized) orthogonal factor.
func (f QRFactor) Q() *Dense {
	qr := f.QR
	m, n := qr.Dims()
	q := NewDense(m, n)

	for k := n - 1; k >= 0; k-- {
		q.Set(k, k, 1)
		for j := k; j < n; j++ {
			if qr.Get(k, k) != 0 {
				var s float64
				for i := k; i < m; i++ {
					s += qr.Get(i, k) * q.Get(i, j)
				}
				s /= -qr.Get(k, k)
				for i := k; i < m; i++ {
					q.Set(i, j, q.Get(i, j)+s*qr.Get(i, k))
				}
			}
		}
	}

	return q
}

// Solve computes a least squares solution of a.x = b where b has as many rows as a.
// A matrix x is returned that minimizes the two norm of Q*R*X-B. Solve will panic
// if a is not full rank. The matrix b is overwritten during the call.
func (f QRFactor) Solve(b *Dense) (x *Dense) {
	qr := f.QR
	rDiag := f.rDiag
	m, n := qr.Dims()
	bm, bn := b.Dims()
	if bm != m {
		panic(ErrShape)
	}
	if !f.IsFullRank() {
		panic("mat64: matrix is rank deficient")
	}

	// Compute Y = transpose(Q)*B
	for k := 0; k < n; k++ {
		for j := 0; j < bn; j++ {
			var s float64
			for i := k; i < m; i++ {
				s += qr.Get(i, k) * b.Get(i, j)
			}
			s /= -qr.Get(k, k)

			for i := k; i < m; i++ {
				b.Set(i, j, b.Get(i, j)+s*qr.Get(i, k))
			}
		}
	}

	// Solve R*X = Y;
	for k := n - 1; k >= 0; k-- {
		for j := 0; j < bn; j++ {
			b.Set(k, j, b.Get(k, j)/rDiag[k])
		}
		for i := 0; i < k; i++ {
			for j := 0; j < bn; j++ {
				b.Set(i, j, b.Get(i, j)-b.Get(k, j)*qr.Get(i, k))
			}
		}
	}

	return b.SubmatrixView(0, 0, n, bn)
}
