package matnum


type Matrix interface {

	// return the dimensions of matrix
	Dims() (dim int)

	// return the shape of matrix(rows and cols)
	shape() (shapes []int)

	// return the value of element from high dimension to low dimension
	at(paras ...int) float64

	// Set will give a number for certain position
	set(val float64,paras ...int)

	//copyto one matrix to another matrix
	copyTo(mat Matrix)

	//the Matrix transpose
	T() Matrix

	//multiply the matrix
	Mul(matrixs ...Matrix) Matrix

	//Add the matrix
	Add(matrixs ...Matrix) Matrix

	//Scale will scale the matrix
	Scale(scale float64) Matrix

	//Clone the Matrix
	Clone() Matrix

	//Slice return the sub matrix
	//if the len(paras) == the len(dim) it will return error
	Slice(paras ...[2]int) Matrix

	//Append the Matrix dynamic
	appendMatrix(matrix Matrix,flags ...int) Matrix

	//Error indicator for chain rule(do not return the error from func)
	Err() error

}
