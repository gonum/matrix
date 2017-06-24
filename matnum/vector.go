package matnum

var(
	vec *Vector
	_ Matrix = vec
)

type Vector struct {

}

func (*Vector) Dims() (dim int) {
	panic("implement me")
}

func (*Vector) Shape() int {
	panic("implement me")
}

func (*Vector) shape() (shapes []int) {
	panic("implement me")
}

func (*Vector) at(paras ...int) float64 {
	panic("implement me")
}
func (*Vector) At(i int) float64 {
	panic("implement me")
}

func (*Vector) set(val float64, paras ...int) {
	panic("implement me")
}

func (*Vector) T() Matrix {
	panic("implement me")
}

func (*Vector) Mul(matrixs ...Matrix) Matrix {
	panic("implement me")
}

func (*Vector) Add(matrixs ...Matrix) Matrix {
	panic("implement me")
}

func (*Vector) Scale(scale float64) Matrix {
	panic("implement me")
}

func (*Vector) Clone() Matrix {
	panic("implement me")
}

func (*Vector) copyTo(mat Matrix) {

}

func (*Vector) Slice(paras ...[2]int) Matrix {
	panic("implement me")
}

func (*Vector) apply(fn func(fn_e func(elem float64) float64, elems ...int) float64,fn2 func(elem float64) float64, val float64) {
	panic("implement me")
}

func (*Vector) appendMatrix(matrix Matrix, flags ...int) Matrix {
	panic("implement me")
}

func (*Vector) Err() error {
	panic("implement me")
}

