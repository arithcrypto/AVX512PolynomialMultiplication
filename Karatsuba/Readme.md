This folder contains AVX512 implementations using the **VPCLMULQDQ** instruction for the 
multiplication in <img src="https://render.githubusercontent.com/render/math?math=\mathbb F_{2}[X]/(X^n-1)" valign="middle"> of polynomials of degree up to 131071.

#### KaratRec
The folder KaratRec contains the source code of the multiplication of polynomials of degree up to 255 using AVX512 instruction set or AVX2 instruction set for schoolbook and Karatsuba methods.  For degree > 255, classical recursive Karatsuba is used.

#### Karat3KaratRec
The folder Karat3KaratRec contains the source code of the **schoolbook** multiplication of polynomials of degree up to 255 using AVX512 instruction set or AVX2 instruction set.  For degree > 255, a 3-way split is used and a recursive call to classical Karatsuba is done. 

#### Karat5KaratRec
The folder Karat5KaratRec contains the source code of the **schoolbook** multiplication of polynomials of degree up to 255 using AVX512 instruction set or AVX2 instruction set.  For degree > 255, a 5-way split is used and a recursive call to classical Karatsuba is done. 

#### Karat3Karat5KaratRec
The folder Karat3Karat5KaratRec contains the source code of the multiplication of polynomials of degree 7607 to 122879 using AVX512 instruction set or AVX2 instruction set. First a 3-way split is applied, than the elementary multiplications of the Karatsuba algorithm are done using the 5-way split Karatsuba algorithm.

#### Karat5Karat3KaratRec
The folder Karat5Karat3KaratRec contains the source code of the multiplication of polynomials of degree 7607 to 122879 using AVX512 instruction set or AVX2 instruction set. First a 5-way split is applied, than the elementary multiplications of the Karatsuba algorithm are done using the 3-way split Karatsuba algorithm.

#### Karat3Karat3KaratRec
The folder Karat3Karat3KaratRec contains the source code of the multiplication of polynomials of degree 4607 to 73727 using AVX512 instruction set or AVX2 instruction set. First a 3-way split is applied, than the elementary multiplications of the Karatsuba algorithm are done using the 3-way split Karatsuba algorithm.

#### Karat5Karat5KaratRec
The folder Karat5Karat5KaratRec contains the source code of the multiplication of polynomials of degree 12799 to 102399 using AVX512 instruction set or AVX2 instruction set. First a 5-way split is applied, than the elementary multiplications of the Karatsuba algorithm are done using the 5-way split Karatsuba algorithm.

