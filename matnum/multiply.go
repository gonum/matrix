package matnum

import (
	"github.com/gonum/matrix"
	"math"
)

type multiplier struct {
	// factors is the ordered set of
	// factors to multiply.
	factors []Matrix2D_I
	// dims is the chain of factor
	// dimensions.
	dims []int

	// table contains the dynamic
	// programming costs and subchain
	// division indices.
	table table
}

func newMultiplier(m Matrix2D_I,factors ...Matrix2D_I) *multiplier {
	ar,ac := m.Shape()
	if ar == 0 || ac == 0 {
		panic(matrix.ErrShape)
	}


	dims := make([]int,len(factors)+2)
	dims[0] = ar
	dims[1] = ac
	for i:=0;i<len(factors);i++ {
		br,bc := factors[i].Shape()
		if br != dims[i+1] {
			panic(matrix.ErrShape)
		}
		dims[i+2] = bc
	}
	factor_head := []Matrix2D_I{m}
	factor_end := append(factor_head,factors...)

	return &multiplier{
		factors: factor_end,
		dims : dims,
		table: newTable(len(factor_end)),
	}

}

func (p *multiplier) optimize() {
	const maxInt = math.MaxInt64
	for f := 1; f < len(p.factors); f++ {
		for i := 0; i < len(p.factors)-f; i++ {
			j := i + f
			p.table.set(i, j, entry{cost: maxInt})
			for k := i; k < j; k++ {
				cost := p.table.at(i, k).cost + p.table.at(k+1, j).cost + p.dims[i]*p.dims[k+1]*p.dims[j+1]
				if cost < p.table.at(i, j).cost {
					p.table.set(i, j, entry{cost: cost, k: k})
				}
			}
		}
	}
}
func (p *multiplier) multiply() Matrix2D_I {
	result, _ := p.multiplySubchain(0, len(p.factors)-1)
	return result
}

func (p *multiplier) multiplySubchain(i, j int) (m Matrix2D_I, intermediate bool) {
	if i == j {
		return p.factors[i], false
	}

	a, _ := p.multiplySubchain(i, p.table.at(i, j).k)
	b, _ := p.multiplySubchain(p.table.at(i, j).k+1, j)

	_, ac := a.Shape()
	br, _ := b.Shape()
	if ac != br {
		// Panic with a string since this
		// is not a user-facing panic.
		panic(matrix.ErrShape.Error())
	}

	r := a.Mul(b)
	return r.(Matrix2D_I), true
}

type entry struct {
	k    int // is the chain subdivision index.
	cost int // cost is the cost of the operation.
}

// table is a row major n×n dynamic programming table.
type table struct {
	n       int
	entries []entry
}

func newTable(n int) table {
	return table{n: n, entries: make([]entry, n*n)}
}

func (t table) at(i, j int) entry     { return t.entries[i*t.n+j] }
func (t table) set(i, j int, e entry) { t.entries[i*t.n+j] = e }


