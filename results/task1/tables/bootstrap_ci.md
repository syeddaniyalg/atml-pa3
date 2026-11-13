| comparison | a | b | metric | diff pp | 95% CI | significant |
|---|---|---|---|---|---|---|
| fullft_mid vs baseline | fullft_mid | baseline | GSM8K | 13.42 | [10.16, 16.76] | True |
| fullft_mid vs baseline | fullft_mid | baseline | HellaSwag | -1.76 | [-2.35, -1.2] | True |
| fullft_mid vs baseline | fullft_mid | baseline | BoolQ | -5.32 | [-7.46, -3.24] | True |
| fullft_mid vs baseline | fullft_mid | baseline | ARC-E | 6.02 | [4.5, 7.62] | True |
| fullft_final vs baseline | fullft_final | baseline | GSM8K | 21.15 | [18.04, 24.18] | True |
| fullft_final vs baseline | fullft_final | baseline | HellaSwag | -1.28 | [-1.87, -0.7] | True |
| fullft_final vs baseline | fullft_final | baseline | BoolQ | -6.24 | [-8.41, -4.16] | True |
| fullft_final vs baseline | fullft_final | baseline | ARC-E | 7.53 | [6.02, 9.13] | True |
| l2sp_final vs baseline | l2sp_final | baseline | GSM8K | 20.09 | [17.13, 23.2] | True |
| l2sp_final vs baseline | l2sp_final | baseline | HellaSwag | -0.96 | [-1.51, -0.4] | True |
| l2sp_final vs baseline | l2sp_final | baseline | BoolQ | 2.57 | [0.61, 4.4] | True |
| l2sp_final vs baseline | l2sp_final | baseline | ARC-E | 7.37 | [5.89, 8.88] | True |
| lora4_final vs baseline | lora4_final | baseline | GSM8K | 15.24 | [12.13, 18.35] | True |
| lora4_final vs baseline | lora4_final | baseline | HellaSwag | 0.03 | [-0.41, 0.5] | False |
| lora4_final vs baseline | lora4_final | baseline | BoolQ | 5.66 | [3.91, 7.37] | True |
| lora4_final vs baseline | lora4_final | baseline | ARC-E | 6.65 | [5.18, 8.08] | True |
| lora16_final vs baseline | lora16_final | baseline | GSM8K | 18.2 | [15.09, 21.3] | True |
| lora16_final vs baseline | lora16_final | baseline | HellaSwag | -0.04 | [-0.54, 0.42] | False |
| lora16_final vs baseline | lora16_final | baseline | BoolQ | 6.48 | [4.68, 8.32] | True |
| lora16_final vs baseline | lora16_final | baseline | ARC-E | 6.69 | [5.26, 8.08] | True |
| oplora16_final vs baseline | oplora16_final | baseline | GSM8K | 18.73 | [15.54, 21.99] | True |
| oplora16_final vs baseline | oplora16_final | baseline | HellaSwag | 0.01 | [-0.48, 0.5] | False |
| oplora16_final vs baseline | oplora16_final | baseline | BoolQ | 6.57 | [4.83, 8.44] | True |
| oplora16_final vs baseline | oplora16_final | baseline | ARC-E | 6.65 | [5.22, 8.08] | True |
| rank | lora16_final | lora4_final | GSM8K | 2.96 | [0.76, 5.0] | True |
| rank | lora16_final | lora4_final | HellaSwag | -0.07 | [-0.38, 0.25] | False |
| rank | lora16_final | lora4_final | BoolQ | 0.83 | [0.0, 1.62] | False |
| rank | lora16_final | lora4_final | ARC-E | 0.04 | [-0.76, 0.88] | False |
| explicit regularization | l2sp_final | fullft_final | GSM8K | -1.06 | [-2.88, 0.83] | False |
| explicit regularization | l2sp_final | fullft_final | HellaSwag | 0.33 | [-0.03, 0.69] | False |
| explicit regularization | l2sp_final | fullft_final | BoolQ | 8.81 | [7.49, 10.18] | True |
| explicit regularization | l2sp_final | fullft_final | ARC-E | -0.17 | [-1.05, 0.72] | False |
| orthogonal protection | oplora16_final | lora16_final | GSM8K | 0.53 | [-1.14, 2.05] | False |
| orthogonal protection | oplora16_final | lora16_final | HellaSwag | 0.05 | [-0.17, 0.26] | False |
| orthogonal protection | oplora16_final | lora16_final | BoolQ | 0.09 | [-0.61, 0.83] | False |
| orthogonal protection | oplora16_final | lora16_final | ARC-E | -0.04 | [-0.67, 0.59] | False |
| full vs low-rank | lora16_final | fullft_final | GSM8K | -2.96 | [-5.16, -0.68] | True |
| full vs low-rank | lora16_final | fullft_final | HellaSwag | 1.24 | [0.76, 1.75] | True |
| full vs low-rank | lora16_final | fullft_final | BoolQ | 12.72 | [10.82, 14.65] | True |
| full vs low-rank | lora16_final | fullft_final | ARC-E | -0.84 | [-2.02, 0.34] | False |
| full FT progression | fullft_final | fullft_mid | GSM8K | 7.73 | [5.31, 10.01] | True |
| full FT progression | fullft_final | fullft_mid | HellaSwag | 0.48 | [0.14, 0.8] | True |
| full FT progression | fullft_final | fullft_mid | BoolQ | -0.92 | [-1.62, -0.28] | True |
| full FT progression | fullft_final | fullft_mid | ARC-E | 1.52 | [0.72, 2.27] | True |
