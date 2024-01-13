# The Complete Mathematics of Neural Networks and Deep Learning

Inspired by [this video](https://www.youtube.com/watch?v=Ixl3nykKG9M).

## 1. Basic Concepts

### 1.1. Derivative of the Function

```latex
\displaystyle{f'(x) = \frac{df}{dx}=\lim\limits_{\Delta\to0}\frac{f(x + \Delta) - f(x)}{\Delta}}
```

<details><summary><b>Examples</b></summary>
```latex
\displaystyle{(x^2)' = \frac{d(x^2)}{dx}=\lim\limits_{\Delta\to0}\frac{(x + \Delta)^2 - x^2}{\Delta} = }
```
```latex
\displaystyle{\lim\limits_{\Delta\to0}\frac{x^2 + 2x\Delta + \Delta^2 - x^2}{\Delta} = \lim\limits_{\Delta\to0}(2x + \Delta) = 2x}
```
</details>

### 1.2. Derivative of the Function with Vector Input (Gradient)

```latex
\displaystyle{
	\nabla f(\vec x) =
	\nabla f(x_1, x_2, ..., x_n) =
	\begin{bmatrix}
		\frac{df}{dx_1} \\ \\
		\frac{df}{dx_2} \\ \\ 
		\frac{df}{dx_3} \\ \\
		... \\  \\
		\frac{df}{dx_n}
	\end{bmatrix}
}
```

<details><summary><b>Examples</b></summary>
```latex
\displaystyle{
	\nabla (x_1^2 + cos(x_2)) =
	\begin{bmatrix}
		\frac{d(x_1^2 + cos(x_2))}{dx_1} \\ \\
		\frac{d(x_1^2 + cos(x_2))}{dx_2}
	\end{bmatrix} = \begin{bmatrix}
		2x_1 \\ \\
		-sin(x_2)
	\end{bmatrix}
}
```
</details>

### 1.2. Derivative of the Vector of Functions with Multiple Variables (with Vector Input) (Jacobian)

```latex
\displaystyle{
	J \begin{bmatrix}
		f_1(\vec x) \\ \\
		f_2(\vec x) \\ \\
		... \\ \\
		f_n(\vec x)
	\end{bmatrix} =
	J \begin{bmatrix}
		f_1(x_1, x_2, ..., x_n) \\ \\
		f_2(x_1, x_2, ..., x_n) \\ \\
		... \\ \\
		f_n(x_1, x_2, ..., x_n)
	\end{bmatrix} =
	\begin{bmatrix}
		\nabla^T f_1(x_1, x_2, ..., x_n) \\ \\
		\nabla^T f_2(x_1, x_2, ..., x_n) \\ \\
		... \\ \\
		\nabla^T f_n(x_1, x_2, ..., x_n)
	\end{bmatrix} = 

}
```

```latex
\displaystyle{
	\begin{bmatrix}
		\frac {df_1(x_1, x_2, ..., x_n)}{dx_1} & \frac {df_1(x_1, x_2, ..., x_n)}{dx_2} & ... & \frac {df_1(x_1, x_2, ..., x_n)}{dx_n} \\ \\
		\frac {df_2(x_1, x_2, ..., x_n)}{dx_1} & \frac {df_2(x_1, x_2, ..., x_n)}{dx_2} & ... & \frac {df_2(x_1, x_2, ..., x_n)}{dx_n} \\ \\
		. & . & ... & . \\
		. & . & ... & . \\
		. & . & ... & . \\
		\frac {df_n(x_1, x_2, ..., x_n)}{dx_1} & \frac {df_n(x_1, x_2, ..., x_n)}{dx_2} & ... & \frac {df_n(x_1, x_2, ..., x_n)}{dx_n}
	\end{bmatrix}
}
```

<details><summary><b>Examples</b></summary>
```latex
\displaystyle{
	J \begin{bmatrix}
		2x_1 + x_2^3 \\ \\
		-13x_1 + e^x_2
	\end{bmatrix} =
	\begin{bmatrix}
		\nabla^T (2x_1 + x_2^3) \\ \\
		\nabla^T (-13x_1 + e^x_2)
	\end{bmatrix} = 
	\begin{bmatrix}
		\frac{d(2x_1 + x_2^3)}{dx_1} & \frac{d(2x_1 + x_2^3)}{dx_2} \\ \\
		\frac{d(-13x_1 + e^x_2)}{(dx_1)} & \frac{d(-13x_1 + e^x_2)}{(dx_2)}
	\end{bmatrix}
} =
```
```latex
\displaystyle{
	\begin{bmatrix}
		2 & 3x_2^2 \\ \\
		-13 & e^{x_2}
	\end{bmatrix}
}
```
</details>

### 1.3. Chain Rule for Scalar Functions

```latex
\displaystyle{
	\frac{df_1(f_2(f_3(\medspace ... \medspace f_n(x))))}{dx} = \frac{df_n}{dx} \frac{df_{n-1}}{df_n} \medspace ...  \frac{df_2}{df_3} \frac{df_1}{df_2}
}
```

<details><summary><b>Examples</b></summary>
```latex
\displaystyle{
	\frac{d(sin(x^2))}{dx} = \frac{d(x^2)}{dx} \frac{d(sin(x^2))}{d(x^2)} = 2xcos(x^2)
}
```
</details>

