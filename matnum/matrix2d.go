//powered by BruceMarcus
package matnum

import (
	"github.com/gonum/blas/blas64"
	"github.com/gonum/matrix"
)

var (
	matrix2d *Matrix2D_I
	_        Matrix = matrix2d
)

type Matrix2D_I interface {
	// implement the Matrix interface
	Matrix

	At(r,c int)  float64

	CopyTo(mat Matrix2D_I)
	Set(val float64,r,c int)
	//return the shape
	Shape() (r, c int)

	Map(fn func(x float64) float64,val float64)

	//Append for append the matrix2d
	//0 indicate the row
	//1 indicate the col
	AppendMatrix(vector *Vector, flag int) Matrix2D_I
}

type Matrix2D struct {
	mat blas64.General

	capRows, capCols int

	err error
}

func (mat2d *Matrix2D) Dims() (dim int) {
	return 2
}

func (mat2d *Matrix2D) shape() (shapes []int) {
	shapes = make([]int, 2)
	shapes[0] = mat2d.mat.Rows
	shapes[1] = mat2d.mat.Cols

	return shapes
}

func (mat2d *Matrix2D) At(r,c int) float64 {
	return mat2d.at(r,c)
}

func (mat2d *Matrix2D) at(paras ...int) float64 {
	if len(paras) != 2 {
		panic(matrix.ErrShape)
	}

	row_length := mat2d.mat.Rows
	col_length := mat2d.mat.Cols
	row := paras[0]
	col := paras[1]

	if row_length < row || col_length < col {
		panic(matrix.ErrIndexOutOfRange)

	}
	return mat2d.mat.Data[row*mat2d.mat.Stride+col]
}

func (mat2d *Matrix2D) Set(val float64, r,c int) {
	mat2d.set(val,r,c)
}

func (mat2d *Matrix2D) set(val float64, paras ...int) {
	if len(paras) != 2 {
		panic(matrix.ErrShape)
	}

	r, c := paras[0], paras[1]

	mat2d.mat.Data[r*mat2d.mat.Stride+c] = val
}

func (mat2d *Matrix2D) T() Matrix {
	r, c := mat2d.Shape()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[j*r+i] = mat2d.At(r, c)
		}
	}
	return &Matrix2D{
		mat: blas64.General{
			Rows:   c,
			Cols:   r,
			Stride: r,
			Data:   data,
		},
		capRows: c,
		capCols: r,
		err:     nil,
	}
}

func (mat2d *Matrix2D) Mul(matrixs ...Matrix) Matrix {
	switch len(matrixs) {
	case 0:
		panic(matrix.ErrShape)
	case 1:
		if mat, ok := matrixs[0].(Matrix2D_I); ok {
			mat2d.mulElem(mat)
		} else if vec,ok := matrix[0].(*Vector);ok {
			mat2d.mulElem(vec)
			panic(matrix.ErrTypeAssert)
		}
	}
	matrixs2d_slice := make([]Matrix2D_I, len(matrixs))
	for i, arg := range matrixs {
		matrixs2d_slice[i] = arg.(Matrix2D_I)
	}
	p := newMultiplier(mat2d, matrixs2d_slice...)
	p.optimize()
	result := p.multiply()

	return result
}

func (mat2d *Matrix2D) Add(matrixs ...Matrix) Matrix {
	ar, ac := mat2d.Shape()
	data := make([]float64, ar*ac)
	for _, e := range matrixs {
		if mat, ok := e.(Matrix2D_I); ok {
			br, bc := mat.Shape()
			if br != ar || bc != ac {
				panic(matrix.ErrShape)
			}
			for r := 0; r < ar; r++ {
				for c := 0; c < ac; c++ {
					data[r*ac+ar] += mat.At(r, c)
				}
			}
		} else {
			panic(matrix.ErrTypeAssert)
		}
	}
	return &Matrix2D{
		mat: blas64.General{
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   data,
		},
		capRows: ar,
		capCols: ac,
		err:     nil,
	}
}

func (mat2d *Matrix2D) Scale(scale float64) Matrix {
	ar, ac := mat2d.Shape()
	data := make([]float64, ar*ac)
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			data[r*ac+c] = mat2d.At(r, c) * scale
		}
	}
	return &Matrix2D{
		mat: blas64.General{
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   data,
		},
		capRows: ar,
		capCols: ac,
		err:     nil,
	}
}

func (mat2d *Matrix2D) Clone() Matrix {
	ar, ac := mat2d.Shape()
	data := make([]float64, ar*ac)
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			data[r*ac+c] = mat2d.At(r, c)
		}
	}
	return &Matrix2D{
		mat: blas64.General{
			Rows:   ar,
			Cols:   ac,
			Stride: ac,
			Data:   data,
		},
		capRows: ar,
		capCols: ac,
		err:     nil,
	}
}

func (mat2d *Matrix2D) copyTo(mat Matrix) {
	if bm,ok := mat.(Matrix2D_I);ok {
		ar,ac := mat2d.Shape()
		br,bc := bm.Shape()
		if ar != br || ac != bc {
			panic(matrix.ErrShape)
		}
		for r:=0;r<ar;r++ {
			for c:=0;c<ac;c++{
				bm.At(r,c) = mat2d.At(r,c)
			}
		}
	}
}

func (mat2d *Matrix2D) CopyTo(mat Matrix2D_I) {
	mat2d.copyTo(mat)
}

func (mat2d *Matrix2D) Slice(paras ...[2]int) Matrix {
	if len(paras) != 2 {
		panic(matrix.ErrShape)
	}
	ar, ac := mat2d.Shape()
	br, brl := paras[0][0], paras[0][1]
	bc, bcl := paras[1][0], paras[1][1]
	if br < 0 || bc < 0 || brl < 1 || bcl < 1 || br+brl > ar || bc+bcl > ac {
		panic(matrix.ErrShape)
	}
	t := *mat2d
	t.mat.Data = mat2d.mat.Data[br*mat2d.mat.Stride+bc : (br+brl)*mat2d.mat.Stride+bc+bcl]
	t.mat.Rows = brl
	t.mat.Cols = bcl
	t.capRows -= br
	t.capCols -= bc

	return &t
}

func (mat2d *Matrix2D) Map(fn func(x float64) float64,val float64) {
	ar,ac := mat2d.Shape()
	for r:=0;r<ar;r++ {
		for c:=0;c<ac;c++{
			mat2d.Set(fn(val),r,c)
		}
	}
}

func (mat2d *Matrix2D) appendMatrix(mat Matrix, flags ...int) Matrix {
	if vec,ok := mat.(*Vector);ok {
		ar,ac := mat2d.Shape()
		if flags[0] == 0 {
			if vec.Shape() != ac {
				panic(matrix.ErrShape)
			}
			data := make([]float64,(ar+1)*ac)
			for r:=0;r<ar;r++ {
				for c:=0;c<ac;c++ {
					data[r*ac+c] = mat2d.At(r,c)
				}
			}
			for index:=0;index<ac;index++ {
				data[ar*ac + index] = vec.At(index)
			}
			return &Matrix2D{
				mat: blas64.General{
					Rows:   ar+1,
					Cols:   ac,
					Stride: ac,
					Data:   data,
				},
				capRows: ar+1,
				capCols: ac,
				err:     nil,
			}
		} else if flags[0] == 1{
			if vec.Shape() != ar {
				panic(matrix.ErrShape)
			}
			data := make([]float64,ar*(ac+1))
			for r:=0;r<ar;r++ {
				for c:=0;c<ac;c++ {
					data[r*ac+c] = mat2d.At(r,c)
				}
			}
			for index:=0;index<ar;index++ {
				data[index*(ac+1) + ac+1] = vec.At(index)
			}
			return &Matrix2D{
				mat: blas64.General{
					Rows:   ar,
					Cols:   ac+1,
					Stride: ac+1,
					Data:   data,
				},
				capRows: ar,
				capCols: ac+1,
				err:     nil,
			}

		} else {
			panic(matrix.ErrShape)
		}
	}else{
		panic(matrix.ErrTypeAssert)
	}

}

func (mat2d *Matrix2D) Err() error {
	return mat2d.err
}

func (mat2d *Matrix2D) Shape() (r, c int) {
	return mat2d.shape()[0],mat2d.shape()[1]
}

func (mat2d *Matrix2D) AppendMatrix(vector *Vector, flag int) Matrix2D_I{
	return mat2d.appendMatrix(vector,flag).(Matrix2D_I)
}

func (mat2d *Matrix2D) mulElem(mat Matrix) *Matrix2D {
	ar, ac := mat2d.Shape()
	var br,bc int
	if vec,ok := mat.(*Vector);ok {
		br = vec.Shape()
		bc = 1
	}
	if mat,ok := mat.(Matrix2D_I);ok {
		br,bc = mat.Shape()
	} else {
		panic(matrix.ErrTypeAssert)
	}
	if ac != br {
		panic(matrix.ErrShape)
	}
	data := make([]float64, ar*bc)
	row := make([]float64, ac)
	for r := 0; r < ar; r++ {
		for i := range row {
			row[i] = mat2d.At(r, i)
		}
		for c := 0; c < bc; c++ {
			var v float64
			for i, e := range row {
				v += e * mat.at(i, c)
			}
			data[r*bc+c] = v
		}
	}
	return &Matrix2D{
		mat: blas64.General{
			Rows:   ar,
			Cols:   bc,
			Stride: bc,
			Data:   data,
		},
		capRows: ar,
		capCols: ac,
		err:     nil,
	}
}
