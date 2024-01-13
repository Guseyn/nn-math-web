# The Complete Mathematics of Neural Networks and Deep Learning

Inspired by [this video](https://www.youtube.com/watch?v=Ixl3nykKG9M).

## 1. Basic Concepts

### 1.1. Derivative of the Function

```latex
\displaystyle{f'(x) = \frac{df}{dx}=\lim\limits_{\Delta\to0}\frac{f(x + \Delta) - f(x)}{\Delta}}
```

<details><summary><b>Example</b></summary>
```latex
\displaystyle{f(x) = x^2}
```
```latex
\displaystyle{f'(x) = (x^2)' = \frac{d(x^2)}{dx}=\lim\limits_{\Delta\to0}\frac{(x + \Delta)^2 - x^2}{\Delta} = }
```
```latex
\displaystyle{\lim\limits_{\Delta\to0}\frac{x^2 + 2x\Delta + \Delta^2 - x^2}{\Delta} = \lim\limits_{\Delta\to0}(2x + \Delta) = 2x}
```
</details>

### 1.2. Derivative of the Function with Vector Input (Gradient)

```latex
\displaystyle{
	\nabla f(\vec x) =
	\nabla f(x_1, x_2, \ldots, x_n) =
	\begin{bmatrix}
		\frac{df(x_1, x_2, \ldots, x_n)}{dx_1} \\ \\
		\frac{df(x_1, x_2, \ldots, x_n)}{dx_2} \\ \\ 
		\frac{df(x_1, x_2, \ldots, x_n)}{dx_3} \\ \\
		\ldots \\  \\
		\frac{df(x_1, x_2, \ldots, x_n)}{dx_n}
	\end{bmatrix}
}
```

<details><summary><b>Example</b></summary>
```latex
\displaystyle{f(x_1, x_2) = x_1^2 + cos(x_2)}
```
```latex
\displaystyle{
	\nabla f(x_1, x_2) =
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
		\ldots \\ \\
		f_n(\vec x)
	\end{bmatrix} =
	J \begin{bmatrix}
		f_1(x_1, x_2, \ldots, x_n) \\ \\
		f_2(x_1, x_2, \ldots, x_n) \\ \\
		\ldots \\ \\
		f_n(x_1, x_2, \ldots, x_n)
	\end{bmatrix} =
	\begin{bmatrix}
		\nabla^T f_1(x_1, x_2, \ldots, x_n) \\ \\
		\nabla^T f_2(x_1, x_2, \ldots, x_n) \\ \\
		\ldots \\ \\
		\nabla^T f_n(x_1, x_2, \ldots, x_n)
	\end{bmatrix} = 

}
```

```latex
\displaystyle{
	= \begin{bmatrix}
		\frac {df_1(x_1, x_2, \ldots, x_n)}{dx_1} & \frac {df_1(x_1, x_2, \ldots, x_n)}{dx_2} & \ldots & \frac {df_1(x_1, x_2, \ldots, x_n)}{dx_n} \\ \\
		\frac {df_2(x_1, x_2, \ldots, x_n)}{dx_1} & \frac {df_2(x_1, x_2, \ldots, x_n)}{dx_2} & \ldots & \frac {df_2(x_1, x_2, \ldots, x_n)}{dx_n} \\ \\
		\vdots & \vdots & \cdots & \vdots \\ \\
		\frac {df_n(x_1, x_2, \ldots, x_n)}{dx_1} & \frac {df_n(x_1, x_2, \ldots, x_n)}{dx_2} & \ldots & \frac {df_n(x_1, x_2, \ldots, x_n)}{dx_n}
	\end{bmatrix}
}
```

<details><summary><b>Example</b></summary>
```latex
\displaystyle{
	f_1(x_1, x_2) = 2x_1 + x_2^3
}
```
```latex
\displaystyle{
	f_2(x_1, x_2) = -13x_1 + e^{x_2}
}
```
```latex
\displaystyle{
	J \begin{bmatrix}
		f_1(x_1, x_2) \\ \\
		f_2(x_1, x_2)
	\end{bmatrix} =
	J \begin{bmatrix}
		2x_1 + x_2^3 \\ \\
		-13x_1 + e^{x_2}
	\end{bmatrix} =
	\begin{bmatrix}
		\nabla^T (2x_1 + x_2^3) \\ \\
		\nabla^T (-13x_1 + e^{x_2})
	\end{bmatrix} =
}
```
```latex
\displaystyle{
	= \begin{bmatrix}
		\frac{d(2x_1 + x_2^3)}{dx_1} & \frac{d(2x_1 + x_2^3)}{dx_2} \\ \\
		\frac{d(-13x_1 + e^{x_2})}{(dx_1)} & \frac{d(-13x_1 + e^{x_2})}{(dx_2)}
	\end{bmatrix}
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
	\frac{df_1(f_2(f_3(\medspace \cdots \medspace f_m(x))))}{dx} = \frac{df_1}{df_2} \frac{df_2}{df_3} \medspace \cdots \medspace  \frac{df_{m-1}}{df_m} \frac{df_m}{dx}
}
```

<details><summary><b>Example</b></summary>
```latex
f_1(x) = sin(x)
```
```latex
f_2(x) = x^2
```
```latex
f_1(f_2(x)) = sin(x^2)
```
```latex
\displaystyle{
	\frac{d(f_1(f_2(x)))}{dx} = \frac{d(sin(x^2))}{dx} = \frac{d(sin(x^2))}{d(x^2)} \frac{d(x^2)}{dx} = cos(x^2) * 2x
}
```

</details>

### 1.4. Gradient Chain Rule

```latex
\displaystyle{
	\nabla f_1(f_2(f_3(\medspace \cdots \medspace f_m(\vec x)))) =
	\nabla f_1(f_2(f_3(\medspace \cdots \medspace f_m(x_1, x_2, \ldots, x_n)))) =
}
```
```latex
\displaystyle{
	\begin{bmatrix}
		\frac{df_1f_2f_3\medspace \cdots \medspace f_m}{dx_1} \\ \\
		\frac{df_1f_2f_3\medspace \cdots \medspace f_m}{dx_2} \\ \\ 
		\frac{df_1f_2f_3\medspace \cdots \medspace f_m}{dx_3} \\ \\
		\ldots \\  \\
		\frac{df_1f_2f_3\medspace \cdots \medspace f_m}{dx_n}
	\end{bmatrix} =
	\begin{bmatrix}
		\frac{df_1}{df_2} \frac{df_2}{df_3} \medspace \cdots \frac{df_{m-1}}{df_m} \frac{df_m}{dx_1} \\ \\
		\frac{df_1}{df_2} \frac{df_2}{df_3} \medspace \cdots \frac{df_{m-1}}{df_m} \frac{df_m}{dx_2} \\ \\ 
		\frac{df_1}{df_2} \frac{df_2}{df_3} \medspace \cdots \frac{df_{m-1}}{df_m} \frac{df_m}{dx_3} \\ \\
		\ldots \\ \\
		\frac{df_1}{df_2} \frac{df_2}{df_3} \medspace \cdots \frac{df_{m-1}}{df_m} \frac{df_m}{dx_n}
	\end{bmatrix} =
	\frac{df_1}{df_2}
	\frac{df_2}{df_3} \medspace \cdots \medspace
	\frac{df_{m-1}}{df_m}
	\begin{bmatrix}
		\frac{df_m}{dx_1} \\ \\
		\frac{df_m}{dx_2} \\ \\
		\frac{df_m}{dx_3} \\ \\
		\ldots \\ \\
		\frac{df_m}{dx_n}
	\end{bmatrix} =
}
```
```latex
	= \frac{df_1}{df_2}
	\frac{df_2}{df_3} \medspace \cdots \medspace
	\frac{df_{m-1}}{df_m}
	\nabla f_m(x_1, x_2, \ldots, x_n)
```

<details><summary><b>Example</b></summary>
```latex
\displaystyle{f_1(x_1, x_2) = x_1^2 + x_2^2}
```
```latex
\displaystyle{f_2(x) = \sqrt{x}}
```
```latex
\displaystyle{f_2(f_1(x_1, x_2)) = \sqrt{(x_1^2 + x_2^2)}}
```
```latex
\displaystyle{
	\nabla f_2(f_1(x_1, x_2)) = \frac{d\sqrt{(x_1^2 + x_2^2)}}{d(x_1^2 + x_2^2)}
	\begin{bmatrix}
		\frac{d(x_1^2 + x_2^2)}{dx_1} \\ \\
		\frac{d(x_1^2 + x_2^2)}{dx_2}
	\end{bmatrix} =
	\frac{1}{2\sqrt{x_1^2 + x_2^2}}
	\begin{bmatrix}
		2x_1 \\ \\
		2x_2
	\end{bmatrix} =
}
```
```latex
\displaystyle{
	= \frac{1}{\sqrt{x_1^2 + x_2^2}}
	\begin{bmatrix}
		x_1 \\ \\
		x_2
	\end{bmatrix}
}
```

</details>

### 1.5. Jacobian Chain Rule

```latex
\displaystyle{
	J \begin{bmatrix}
		f_{11}(f_{12}(f_{13}(\medspace \ldots \medspace f_{1m}(\vec x)))) \\ \\
		f_{21}(f_{22}(f_{23}(\medspace \ldots \medspace f_{2m}(\vec x)))) \\ \\
		\ldots \\ \\
		f_{n1}(f_{n2}(f_{n3}(\medspace \ldots \medspace f_{nm}(\vec x))))
	\end{bmatrix} =
	J \begin{bmatrix}
		f_{11}(f_{12}(f_{13}(\medspace \ldots \medspace f_{1m}(x_1, x_2, \ldots, x_n)))) \\ \\
		f_{21}(f_{22}(f_{23}(\medspace \ldots \medspace f_{2m}(x_1, x_2, \ldots, x_n)))) \\ \\
		\ldots \\ \\
		f_{n1}(f_{n2}(f_{n3}(\medspace \ldots \medspace f_{nm}(x_1, x_2, \ldots, x_n))))
	\end{bmatrix}
}
```
```latex
\displaystyle{
	= \begin{bmatrix}
		\nabla^T f_{11}(f_{12}(f_{13}(\medspace \ldots \medspace f_{1m}(x_1, x_2, \ldots, x_n)))) \\ \\
		\nabla^T f_{21}(f_{22}(f_{23}(\medspace \ldots \medspace f_{2m}(x_1, x_2, \ldots, x_n)))) \\ \\
		\ldots \\ \\
		\nabla^T f_{n1}(f_{n2}(f_{n3}(\medspace \ldots \medspace f_{nm}(x_1, x_2, \ldots, x_n))))
	\end{bmatrix} =
}
```
```latex
\displaystyle{
	\begin{bmatrix}
		\frac{df_{11}}{df_{12}} \frac{df_{12}}{df_{13}} \medspace \cdots \frac{df_{{1m-1}}}{df_{1m}} \frac{df_{1m}}{dx_1} &
		\frac{df_{11}}{df_{12}} \frac{df_{12}}{df_{13}} \medspace \cdots \frac{df_{{1m-1}}}{df_{1m}} \frac{df_{1m}}{dx_2} &
		\ldots &
		\frac{df_{11}}{df_{12}} \frac{df_{12}}{df_{13}} \medspace \cdots \frac{df_{{1m-1}}}{df_{1m}} \frac{df_{1m}}{dx_m} \\ \\
		\frac{df_{21}}{df_{22}} \frac{df_{22}}{df_{23}} \medspace \cdots \frac{df_{{2m-1}}}{df_{2m}} \frac{df_{2m}}{dx_1} &
		\frac{df_{21}}{df_{22}} \frac{df_{22}}{df_{23}} \medspace \cdots \frac{df_{{2m-1}}}{df_{2m}} \frac{df_{2m}}{dx_2} &
		\ldots &
		\frac{df_{21}}{df_{22}} \frac{df_{22}}{df_{23}} \medspace \cdots \frac{df_{{2m-1}}}{df_{2m}} \frac{df_{2m}}{dx_m} \\ \\
		\cdots & \cdots & \vdots & \cdots \\ \\ 
		\frac{df_{n1}}{df_{n2}} \frac{df_{n2}}{df_{n3}} \medspace \cdots \frac{df_{{nm-1}}}{df_{nm}} \frac{df_{nm}}{dx_1} &
		\frac{df_{n1}}{df_{n2}} \frac{df_{n2}}{df_{n3}} \medspace \cdots \frac{df_{{nm-1}}}{df_{nm}} \frac{df_{nm}}{dx_2} &
		\ldots &
		\frac{df_{n1}}{df_{n2}} \frac{df_{n2}}{df_{n3}} \medspace \cdots \frac{df_{{nm-1}}}{df_{nm}} \frac{df_{nm}}{dx_m}
	\end{bmatrix}
}
```
```latex
\displaystyle{
	= \begin{bmatrix}
		\frac{df_{11}}{df_{12}}
		\frac{df_{12}}{df_{13}} \medspace \cdots \medspace
		\frac{df_{1m-1}}{df_{1m}}
		\nabla^T f_{1m}(x_1, x_2, \ldots, x_n) \\ \\
		\frac{df_{21}}{df_{22}}
		\frac{df_{22}}{df_{23}} \medspace \cdots \medspace
		\frac{df_{2m-1}}{df_{2m}}
		\nabla^T f_{2m}(x_1, x_2, \ldots, x_n) \\ \\
		\vdots \\ \\ 
		\frac{df_{n1}}{df_{n2}}
		\frac{df_{n2}}{df_{n3}} \medspace \cdots \medspace
		\frac{df_{nm-1}}{df_{nm}}
		\nabla^T f_{nm}(x_1, x_2, \ldots, x_n) \\ \\
	\end{bmatrix}
}
```

<details><summary><b>Example</b></summary>
```latex
\displaystyle{f_{11}(x_1, x_2) = x_1^2 + x_2}
```
```latex
\displaystyle{f_{12}(x) = sin{x}}
```
```latex
\displaystyle{f_{12}(f_{11}(x_1, x_2)) = sin(x_1^2 + x_2)}
```
```latex
\displaystyle{f_{21}(x_1, x_2) = x_2^3}
```
```latex
\displaystyle{f_{22}(x) = ln(x)}
```
```latex
\displaystyle{f_{22}(f_{21}(x_1, x_2)) = ln(x_2^3)}
```
```latex
\displaystyle{
	\nabla f_{12}(f_{11}(x_1, x_2)) = 
	\frac{df_{12}}{df_{11}}\nabla f_{11} =
	\frac{d(sin(x_1^2 + x_2))}{d(x_1^2 + x_2)}
	\begin{bmatrix}
		\frac{d(x_1^2 + x_2)}{x_1} \\ \\
		\frac{d(x_1^2 + x_2)}{x_2}
	\end{bmatrix}
} =
```
```latex
\displaystyle{
	= cos(x_1^2 + x_2)
	\begin{bmatrix}
		2x_1 \\ \\
		1
	\end{bmatrix}
}
```
```latex
\displaystyle{
	\nabla f_{22}(f_{21}(x_1, x_2)) = 
	\frac{df_{22}}{df_{21}}\nabla f_{21} =
	\frac{d(ln(x_2^3))}{d(x_2^3)}
	\begin{bmatrix}
		\frac{d(x_2^3)}{x_1} \\ \\
		\frac{d(x_2^3)}{x_2}
	\end{bmatrix}
} = \frac{1}{(x_2^3)}
\begin{bmatrix}
	0 \\ \\
	3x_2^2
\end{bmatrix}
```
```latex
\displaystyle{
	J \begin{bmatrix}
		f_{12}(f_{11}(x_1, x_2) \\ \\
		f_{22}(f_{21}(x_1, x_2)
	\end{bmatrix} =
	\begin{bmatrix}
		\nabla f_{12}(f_{11}(x_1, x_2)) \\ \\
		\nabla f_{22}(f_{21}(x_1, x_2))
	\end{bmatrix} =
	\begin{bmatrix}
		\frac{df_{12}}{df_{11}}\nabla^T f_{11} \\ \\
		\frac{df_{22}}{df_{21}}\nabla^T f_{21} \\
	\end{bmatrix} =
}
```
```latex
\displaystyle{
	= \begin{bmatrix}
		2x_1cos(x_1^2 + x_2) & cos(x_1^2 + x_2) \\ \\
		0 & 3/{x_2}
	\end{bmatrix}
}
```
</details>


## 2. Forward Propagation
### 2.1. The Neuron Function

```latex
\sigma(\sum_{i=1}^{n} x_iw_i + b)
```

