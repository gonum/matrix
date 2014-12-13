package mat64

import (
	"math"
	"math/rand"
	"testing"

	check "launchpad.net/gocheck"
)

func iden(span int) *Dense {
	d := make([]float64, span*span)
	for i := 0; i < span*span; i += span + 1 {
		d[i] = 1
	}
	return NewDense(span, span, d)
}

func skewMatrix(m *Dense) *Dense {
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

// Suite
var _ = check.Suite(&S{})

var (
	ta, tm *Dense
)

func (s *S) SetUpTest(c *check.C) {
	ta = skewMatrix(NewDense(3, 1, []float64{0, 0, 1}))
	tm = iden(3)
}

func (s *S) TearDownTest(c *check.C) {
	ta = new(Dense)
	tm = new(Dense)
}

// It should panic if m is not identity matrix
func (s *S) TestExpMP(c *check.C) {
	tm.Scale(2, tm)
	c.Check(func() { tm.ExpM(ta) }, check.PanicMatches, "m must be identity matrix of size a")
}

// It should be a Z axis rotation matrix
func (s *S) TestExpMZ(c *check.C) {
	tr := newStackedDense([][]float64{
		{0.707106, -0.707106, 0},
		{0.707106, 0.707106, 0},
		{0, 0, 1},
	})
	ta.Scale(math.Pi/4, ta)
	tm.ExpM(ta)
	c.Check(tm.EqualsApprox(tr, 1e-6), check.DeepEquals, true)
}

// It should be identity matrix
func (s *S) TestExpMI(c *check.C) {
	ta = NewDense(3, 3, nil)
	tm.ExpM(ta)
	c.Check(tm, check.DeepEquals, iden(3))
}

// It should be a 90deg matrix
func (s *S) TestExpM90(c *check.C) {
	ta.Scale(math.Pi/2, ta)
	tm.ExpM(ta)
	for i, v := range tm.RawMatrix().Data {
		tm.RawMatrix().Data[i] = math.Trunc(v)
	}
	c.Check(tm, check.DeepEquals, newStackedDense([][]float64{
		{0, -1, 0},
		{1, 0, 0},
		{0, 0, 1},
	}))
}

func BenchmarkExpMRandUnits(b *testing.B) { bexp(b, 3, math.Pi/4) }
func BenchmarkExpMRandTens(b *testing.B)  { bexp(b, 30, math.Pi/4) }
func BenchmarkExpMRandHuns(b *testing.B)  { bexp(b, 300, math.Pi/4) }
func BenchmarkExpMRandThous(b *testing.B) { bexp(b, 3000, math.Pi/4) }

func bexp(b *testing.B, s int, f float64) {
	b.StopTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ta, _ = randDense(s, f, rand.Float64)
		tm = iden(s)
		tm.ExpM(ta)
	}
}
