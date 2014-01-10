// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the SingularValueDecomposition class from Jama 1.0.3.

package mat64

import (
	"math"
)

type SVDFactors struct {
	U     *Dense
	Sigma []float64
	V     *Dense
}

// SVD performs singular value decomposition for an m-by-n matrix a with m >= n,
// the singular value decomposition is an m-by-n orthogonal matrix u, an n-by-n
// diagonal matrix s, and an n-by-n orthogonal matrix v so that a = u*s*v'. The
// matrix a is overwritten during the decomposition and u and v are only returned
// when wantu and wantv are true respectively.
//
// The singular values, sigma[k] = s[k][k], are ordered so that
//
//  sigma[0] >= sigma[1] >= ... >= sigma[n-1].
//
// The matrix condition number and the effective numerical rank can be computed from
// this decomposition.
func SVD(a *Dense, epsilon, small float64, wantu, wantv bool) SVDFactors {
	m, n := a.Dims()

	// Apparently the failing cases are only a proper subset of (m<n),
	// so let's not panic. Correct fix to come later?
	// if m < n {
	// 	panic(ErrShape)
	// }

	sigma := make([]float64, min(m+1, n))
	nu := min(m, n)
	var u, v *Dense
	if wantu {
		u = NewDense(m, nu)
	}
	if wantv {
		v = NewDense(n, n)
	}

	var (
		e    = make([]float64, n)
		work = make([]float64, m)
	)

	// Reduce a to bidiagonal form, storing the diagonal elements
	// in sigma and the super-diagonal elements in e.
	nct := min(m-1, n)
	nrt := max(0, min(n-2, m))
	for k := 0; k < max(nct, nrt); k++ {
		if k < nct {
			// Compute the transformation for the k-th column and
			// place the k-th diagonal in sigma[k].
			// Compute 2-norm of k-th column without under/overflow.
			sigma[k] = 0
			for i := k; i < m; i++ {
				sigma[k] = math.Hypot(sigma[k], a.At(i, k))
			}
			if sigma[k] != 0 {
				if a.At(k, k) < 0 {
					sigma[k] = -sigma[k]
				}
				for i := k; i < m; i++ {
					a.Set(i, k, a.At(i, k)/sigma[k])
				}
				a.Set(k, k, a.At(k, k)+1)
			}
			sigma[k] = -sigma[k]
		}

		for j := k + 1; j < n; j++ {
			if k < nct && sigma[k] != 0 {
				// Apply the transformation.
				var t float64
				for i := k; i < m; i++ {
					t += a.At(i, k) * a.At(i, j)
				}
				t = -t / a.At(k, k)
				for i := k; i < m; i++ {
					a.Set(i, j, a.At(i, j)+t*a.At(i, k))
				}
			}

			// Place the k-th row of a into e for the
			// subsequent calculation of the row transformation.
			e[j] = a.At(k, j)
		}

		if wantu && k < nct {
			// Place the transformation in u for subsequent back
			// multiplication.
			for i := k; i < m; i++ {
				u.Set(i, k, a.At(i, k))
			}
		}

		if k < nrt {
			// Compute the k-th row transformation and place the
			// k-th super-diagonal in e[k].
			// Compute 2-norm without under/overflow.
			e[k] = 0
			for i := k + 1; i < n; i++ {
				e[k] = math.Hypot(e[k], e[i])
			}
			if e[k] != 0 {
				if e[k+1] < 0 {
					e[k] = -e[k]
				}
				for i := k + 1; i < n; i++ {
					e[i] /= e[k]
				}
				e[k+1] += 1
			}
			e[k] = -e[k]
			if k+1 < m && e[k] != 0 {
				// Apply the transformation.
				for i := k + 1; i < m; i++ {
					work[i] = 0
				}
				for j := k + 1; j < n; j++ {
					for i := k + 1; i < m; i++ {
						work[i] += e[j] * a.At(i, j)
					}
				}
				for j := k + 1; j < n; j++ {
					t := -e[j] / e[k+1]
					for i := k + 1; i < m; i++ {
						a.Set(i, j, a.At(i, j)+t*work[i])
					}
				}
			}
			if wantv {
				// Place the transformation in v for subsequent
				// back multiplication.
				for i := k + 1; i < n; i++ {
					v.Set(i, k, e[i])
				}
			}
		}
	}

	// Set up the final bidiagonal matrix or order p.
	p := min(n, m+1)
	if nct < n {
		sigma[nct] = a.At(nct, nct)
	}
	if m < p {
		sigma[p-1] = 0
	}
	if nrt+1 < p {
		e[nrt] = a.At(nrt, p-1)
	}
	e[p-1] = 0

	// If requested, generate u.
	if wantu {
		for j := nct; j < nu; j++ {
			for i := 0; i < m; i++ {
				u.Set(i, j, 0)
			}
			u.Set(j, j, 1)
		}
		for k := nct - 1; k >= 0; k-- {
			if sigma[k] != 0 {
				for j := k + 1; j < nu; j++ {
					var t float64
					for i := k; i < m; i++ {
						t += u.At(i, k) * u.At(i, j)
					}
					t /= -u.At(k, k)
					for i := k; i < m; i++ {
						u.Set(i, j, u.At(i, j)+t*u.At(i, k))
					}
				}
				for i := k; i < m; i++ {
					u.Set(i, k, -u.At(i, k))
				}
				u.Set(k, k, 1+u.At(k, k))
				for i := 0; i < k-1; i++ {
					u.Set(i, k, 0)
				}
			} else {
				for i := 0; i < m; i++ {
					u.Set(i, k, 0)
				}
				u.Set(k, k, 1)
			}
		}
	}

	// If requested, generate v.
	if wantv {
		for k := n - 1; k >= 0; k-- {
			if k < nrt && e[k] != 0 {
				for j := k + 1; j < nu; j++ {
					var t float64
					for i := k + 1; i < n; i++ {
						t += v.At(i, k) * v.At(i, j)
					}
					t /= -v.At(k+1, k)
					for i := k + 1; i < n; i++ {
						v.Set(i, j, v.At(i, j)+t*v.At(i, k))
					}
				}
			}
			for i := 0; i < n; i++ {
				v.Set(i, k, 0)
			}
			v.Set(k, k, 1)
		}
	}

	// Main iteration loop for the singular values.
	pp := p - 1
	for iter := 0; p > 0; {
		var k, kase int

		// Here is where a test for too many iterations would go.

		// This section of the program inspects for
		// negligible elements in the sigma and e arrays.  On
		// completion the variables kase and k are set as follows.
		//
		// kase = 1     if sigma(p) and e[k-1] are negligible and k<p
		// kase = 2     if sigma(k) is negligible and k<p
		// kase = 3     if e[k-1] is negligible, k<p, and
		//              sigma(k), ..., sigma(p) are not negligible (qr step).
		// kase = 4     if e(p-1) is negligible (convergence).
		//
		for k = p - 2; k >= -1; k-- {
			if k == -1 {
				break
			}
			if math.Abs(e[k]) <= small+epsilon*(math.Abs(sigma[k])+math.Abs(sigma[k+1])) {
				e[k] = 0
				break
			}
		}

		if k == p-2 {
			kase = 4
		} else {
			var ks int
			for ks = p - 1; ks >= k; ks-- {
				if ks == k {
					break
				}
				var t float64
				if ks != p {
					t = math.Abs(e[ks])
				}
				if ks != k+1 {
					t += math.Abs(e[ks-1])
				}
				if math.Abs(sigma[ks]) <= small+epsilon*t {
					sigma[ks] = 0
					break
				}
			}
			if ks == k {
				kase = 3
			} else if ks == p-1 {
				kase = 1
			} else {
				kase = 2
				k = ks
			}
		}
		k++

		switch kase {
		// Deflate negligible sigma(p).
		case 1:
			f := e[p-2]
			e[p-2] = 0
			for j := p - 2; j >= k; j-- {
				t := math.Hypot(sigma[j], f)
				cs := sigma[j] / t
				sn := f / t
				sigma[j] = t
				if j != k {
					f = -sn * e[j-1]
					e[j-1] *= cs
				}
				if wantv {
					for i := 0; i < n; i++ {
						t = cs*v.At(i, j) + sn*v.At(i, p-1)
						v.Set(i, p-1, -sn*v.At(i, j)+cs*v.At(i, p-1))
						v.Set(i, j, t)
					}
				}
			}

		// Split at negligible sigma(k).
		case 2:
			f := e[k-1]
			e[k-1] = 0
			for j := k; j < p; j++ {
				t := math.Hypot(sigma[j], f)
				cs := sigma[j] / t
				sn := f / t
				sigma[j] = t
				f = -sn * e[j]
				e[j] *= cs
				if wantu {
					for i := 0; i < m; i++ {
						t = cs*u.At(i, j) + sn*u.At(i, k-1)
						u.Set(i, k-1, -sn*u.At(i, j)+cs*u.At(i, k-1))
						u.Set(i, j, t)
					}
				}
			}

		// Perform one qr step.
		case 3:
			// Calculate the shift.
			scale := math.Max(math.Max(math.Max(math.Max(
				math.Abs(sigma[p-1]), math.Abs(sigma[p-2])), math.Abs(e[p-2])), math.Abs(sigma[k])), math.Abs(e[k]),
			)
			sp := sigma[p-1] / scale
			spm1 := sigma[p-2] / scale
			epm1 := e[p-2] / scale
			sk := sigma[k] / scale
			ek := e[k] / scale
			b := ((spm1+sp)*(spm1-sp) + epm1*epm1) / 2
			c := (sp * epm1) * (sp * epm1)

			var shift float64
			if b != 0 || c != 0 {
				shift = math.Sqrt(b*b + c)
				if b < 0 {
					shift = -shift
				}
				shift = c / (b + shift)
			}
			f := (sk+sp)*(sk-sp) + shift
			g := sk * ek

			// Chase zeros.
			for j := k; j < p-1; j++ {
				t := math.Hypot(f, g)
				cs := f / t
				sn := g / t
				if j != k {
					e[j-1] = t
				}
				f = cs*sigma[j] + sn*e[j]
				e[j] = cs*e[j] - sn*sigma[j]
				g = sn * sigma[j+1]
				sigma[j+1] *= cs
				if wantv {
					for i := 0; i < n; i++ {
						t = cs*v.At(i, j) + sn*v.At(i, j+1)
						v.Set(i, j+1, -sn*v.At(i, j)+cs*v.At(i, j+1))
						v.Set(i, j, t)
					}
				}
				t = math.Hypot(f, g)
				cs = f / t
				sn = g / t
				sigma[j] = t
				f = cs*e[j] + sn*sigma[j+1]
				sigma[j+1] = -sn*e[j] + cs*sigma[j+1]
				g = sn * e[j+1]
				e[j+1] *= cs
				if wantu && j < m-1 {
					for i := 0; i < m; i++ {
						t = cs*u.At(i, j) + sn*u.At(i, j+1)
						u.Set(i, j+1, -sn*u.At(i, j)+cs*u.At(i, j+1))
						u.Set(i, j, t)
					}
				}
			}
			e[p-2] = f
			iter++

		// Convergence.
		case 4:
			// Make the singular values positive.
			if sigma[k] <= 0 {
				if sigma[k] < 0 {
					sigma[k] = -sigma[k]
				} else {
					sigma[k] = 0
				}
				if wantv {
					for i := 0; i <= pp; i++ {
						v.Set(i, k, -v.At(i, k))
					}
				}
			}

			// Order the singular values.
			for k < pp {
				if sigma[k] >= sigma[k+1] {
					break
				}
				sigma[k], sigma[k+1] = sigma[k+1], sigma[k]
				if wantv && k < n-1 {
					for i := 0; i < n; i++ {
						t := v.At(i, k+1)
						v.Set(i, k+1, v.At(i, k))
						v.Set(i, k, t)
					}
				}
				if wantu && k < m-1 {
					for i := 0; i < m; i++ {
						t := u.At(i, k+1)
						u.Set(i, k+1, u.At(i, k))
						u.Set(i, k, t)
					}
				}
				k++
			}
			iter = 0
			p--
		}
	}

	return SVDFactors{u, sigma, v}
}

// S returns a newly allocated S matrix from the sigma values held by the
// factorisation.
func (f SVDFactors) S() *Dense {
	s := NewDense(len(f.Sigma), len(f.Sigma))
	for i, v := range f.Sigma {
		s.Set(i, i, v)
	}
	return s
}

// Rank returns the number of non-negligible singular values in the sigma held by
// the factorisation with the given epsilon.
func (f SVDFactors) Rank(epsilon float64) int {
	if len(f.Sigma) == 0 {
		return 0
	}
	m, _ := f.U.Dims()
	tol := float64(max(m, len(f.Sigma))) * f.Sigma[0] * epsilon
	var r int
	for _, v := range f.Sigma {
		if v > tol {
			r++
		}
	}
	return r
}

// Cond returns the 2-norm condition number for the S matrix.
func (f SVDFactors) Cond() float64 {
	m, _ := f.U.Dims()
	n, _ := f.V.Dims()
	return f.Sigma[0] / f.Sigma[min(m, n)-1]
}
