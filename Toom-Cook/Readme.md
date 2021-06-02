## Description
#### Toom3KaratRec
The folder Toom3KaratRec contains the source code of the multiplication of polynomials of degree >= 24191. First the operands are splitted using Toom-Cook3 algorithm, then elementary multiplications are done using recursive classical Karatsuba algorithm. When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Toom3Karat3
The folder Toom3Karat3 contains the source code of the multiplication of polynomials of degree >= 18049. First the operands are splitted using Toom-Cook3 algorithm, then elementary multiplications are done using the 3-way split Karatsuba algorithm. When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Toom3Karat5
The folder Toom3Karat5 contains the source code of the multiplication of polynomials of degree >= 14975. First the operands are splitted using Toom-Cook3 algorithm, then elementary multiplications are done using the 5-way split Karatsuba algorithm. When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

