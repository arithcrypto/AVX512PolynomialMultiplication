## Description 
#### KaratRec
The folder KaratRec contains the source code of the multiplication of polynomials of degree up to 255 using AVX512 instruction set or AVX2 instruction set for **schoolbook and Karatsuba** methods.  For degree > 255, classical recursive Karatsuba is used.

#### Karat3KaratRec
The folder Karat3KaratRec contains the AVX512 and AVX2 source codes of the multiplication of polynomials of degree >= 1535.   First a 3-way split is applied to the operands, and a recursive call to classical Karatsuba is done. 
When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Karat5KaratRec
The folder Karat5KaratRec contains the AVX512 and AVX2 source codes of the multiplication of polynomials of degree >= 2559.   First a 5-way split is applied to the operands, and a recursive call to classical Karatsuba is done. 
When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Karat3Karat5KaratRec
The folder Karat3Karat5KaratRec contains the AVX512 and AVX2 source codes of the multiplication of polynomials of degree >= 7679.   First a 3-way split is applied to the operands, than the elementary multiplications of the Karatsuba algorithm are done using the 5-way split Karatsuba algorithm. 
When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Karat5Karat3KaratRec
The folder Karat5Karat3KaratRec contains the AVX512 and AVX2 source codes of the multiplication of polynomials of degree >= 7679.   First a 5-way split is applied to the operands, than the elementary multiplications of the Karatsuba algorithm are done using the 3-way split Karatsuba algorithm. 
When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Karat3Karat3KaratRec
The folder Karat3Karat3KaratRec contains the AVX512 and AVX2 source codes of the multiplication of polynomials of degree >= 4607.   First a 3-way split is applied to the operands, than the elementary multiplications of the Karatsuba algorithm are done using the 3-way split Karatsuba algorithm. 
When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

#### Karat5Karat5KaratRec
The folder Karat5Karatr5KaratRec contains the AVX512 and AVX2 source codes of the multiplication of polynomials of degree >= 12799.   First a 5-way split is applied to the operands, than the elementary multiplications of the Karatsuba algorithm are done using the 5-way split Karatsuba algorithm. 
When the recursive step reaches operands of degree < 256 (resp. of degree < 128), the AVX512 schoolbook multiplication (resp. the AVX2 Karatsuba multiplication) is applied. 

