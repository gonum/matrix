package mat64

import (
	"math"
	"testing"
	
	check "gopkg.in/check.v1"
)

var _ = check.Suite(&S{})

func iden(span int) *Dense {
	d := make([]float64, span*span)
	for i := 0; i < span*span; i += span + 1 {
		d[i] = 1
	}
	return NewDense(span, span, d)
}

func skewMatrix(m *Dense) *Dense {
	vchan := make(chan []float64, 1)
	v := m.RawMatrix().Data
	return newStackedDense([][]float64{
		{0, -v[2], v[1]},
		{v[2], 0, -v[0]},
		{-v[1], v[0], 0}})
}

func newStackedDense(mv [][]float64) *Dense {
	r := len(mv)
	c := len(mv[0])
	d := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			d[i*c+j] = mv[i][j]
		}
	}
	return NewDense(r, c, d)
}

// func (s *S) SetUpSuite(c *check.C) { blasEngine = goblas.Blas{} }

/*
   Trivial tests but the cases when the function fails are all known and deliberate to favor performance.
   Expect function to fail if either parameters "a" or "id" are nil. It will panic anyways.
   If "a" or "id" are not equal size, wrong answer will be produced but will not complain.
*/
func (s *S) TestExpMC(c *check.C) {
	var testvals = []struct {
		a, m, id, res *Dense
	}{
		{
			// m == res
			a:   skewMatrix(NewDense(3, 1, []float64{0, 0, 1})),
			m:   new(Dense),
			id:  iden(3),
			res: NewDense(3, 3, []float64{0.7071067812302496, -0.707106781206505, 0, 0.707106781206505, 0.7071067812302496, 0, 0, 0, 1}),
		},
		{
			//   identity matrix
			a:   skewMatrix(NewDense(3, 1, []float64{0, 0, 0})),
			m:   new(Dense),
			id:  iden(3),
			res: iden(3),
		},
	}

	for i, t := range testvals {
		t.a.Scale(math.Pi/4, t.a)
		t.m.ExpMC(t.a, t.id)
		c.Check(t.m, check.DeepEquals, t.res, check.Commentf("Test %d", i))
	}

	var knownfailval = []struct {
		a, m, id, res *Dense
	}{

		{
			// will produce wrong result => fail (known reason)
			// a and id == same size (required)
			a:   skewMatrix(NewDense(3, 1, []float64{0, 0, 0})),
			m:   new(Dense),
			id:  iden(4),
			res: NewDense(3, 1, nil),
		},
	}

	for i, t := range knownfailval {
		t.a.Scale(math.Pi/4, t.a)
		t.m.ExpMC(t.a, t.id)
		c.Check(t.m, check.DeepEquals, t.res, check.Commentf("Test %d", i))
		c.ExpectFailure("a and id must be of equal size!")
	}
}

// Benchmarking not completed yet - need more work
// Concurrent Benchmark
func BenchmarkExpMCUnits(b *testing.B) { exponenMat(b, 2, "c") }
func BenchmarkExpMCTens(b *testing.B)  { exponenMat(b, 18, "c") }
func BenchmarkExpMCHuns(b *testing.B)  { exponenMat(b, 108, "c") }
func BenchmarkExpMCThous(b *testing.B) { exponenMat(b, 1008, "c") }

func initParams(s int) (ba, bid, bm *Dense) {
	ba = iden(s)
	bid = new(Dense)
	bid.Clone(ba)
	bm = new(Dense)
	ba.Scale(math.Pi/4, ba)
	return
}

func exponenMat(b *testing.B, size int, mode string) {
	a, id, m := initParams(size)

	b.StopTimer()
	if mode == "c" {
		b.StartTimer()
		for i := 0; i < 100; i++ {
			m.ExpMC(a, id)
		}
	} else {
		b.StartTimer()
		for i := 0; i < 100; i++ {
			m.ExpMS(a, id)
		}
	}
}
