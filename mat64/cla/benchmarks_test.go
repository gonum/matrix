package cla

import (
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/matrix/mat64/la"
	"math"
	"math/rand"
	"testing"
)

func BenchmarkCholeskyLa(b *testing.B) {
	n := 20
	for iter := 0; iter < b.N; iter++ {
		b.StopTimer()
		a, _ := mat64.NewDense(n, n, make([]float64, n*n))
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				v := rand.NormFloat64()
				a.Set(i, j, v)
				a.Set(j, i, v)
			}
		}
		b.StartTimer()

		_ = la.Cholesky(a)
	}
}

func BenchmarkCholeskyLAPACKE(b *testing.B) {
	n := 20
	for iter := 0; iter < b.N; iter++ {
		b.StopTimer()
		a, _ := mat64.NewDense(n, n, make([]float64, n*n))
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				v := rand.NormFloat64()
				a.Set(i, j, v)
				a.Set(j, i, v)
			}
		}
		b.StartTimer()

		_ = Cholesky(a)
	}
}

func BenchmarkSVDLa(b *testing.B) {
	n := 100
	epsilon := math.Pow(2, -52.0)
	small := math.Pow(2, -966.0)

	for iter := 0; iter < b.N; iter++ {
		b.StopTimer()
		a, _ := mat64.NewDense(n, n, make([]float64, n*n))
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				v := rand.NormFloat64()
				a.Set(i, j, v)
			}
		}
		b.StartTimer()

		_ = la.SVD(a, epsilon, small, true, true)
	}
}

func BenchmarkSVDLAPACKE(b *testing.B) {
	n := 100

	for iter := 0; iter < b.N; iter++ {
		b.StopTimer()
		a, _ := mat64.NewDense(n, n, make([]float64, n*n))
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				v := rand.NormFloat64()
				a.Set(i, j, v)
			}
		}
		b.StartTimer()

		_ = SVD(a, 0, 0, true, true)
	}
}
