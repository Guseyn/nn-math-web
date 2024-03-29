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


### 1.6. Dot Product of Two Vectors

```latex
A = [a_1, a_2, \ldots, a_n]^T, B = [b_1, b_2, \ldots, b_n]^T
```
```latex
P = \mathbf{A}^T \mathbf{B} = \sum_{i=1}^{n} a_ib_i
```

### 1.7. Product Rule For Scalar Functions

```latex
(u \odot v)' = u' \odot v + u \odot v'
```

### 1.8. Product Rule For Gradients

```latex
\nabla (f \odot g) = \nabla f \odot g + f \odot \nabla g,
```
where $$f$$ and $$g$$ produce scalars or vectors.

## 2. Forward Propagation
### 2.1. The Neuron Function

Let's consider ReLU as activation function

```latex
\sigma(x) = max(0, x)
```
```latex
\sigma(\sum_{i=1}^{n} x_iw_i + b) = max(0, \sum_{i=1}^{n} x_iw_i + b) = max(0, x^Tw + b)
```
```latex
z = x^Tw + b
```
Let's call the result or activation of the activation function as $$a$$
```latex
a = \sigma(x^Tw + b) = \sigma(z)
```

For each nueron, we have two vectors $$x$$(input) and $$w$$(weight), and we get $$a$$ as a scalar result of the activation for the neuron.

### 2.2. Weight and Bias Indexing
- $$w_{jk}^l$$ (as scalar)
	- $$l$$ - current layer index, where we are calculating our activation for the neuron
	- $$j$$ - neuron index in the layer ($$l$$), where we are calculating our activation for the neuron
	- $$k$$ - neuron index in the previous layer ($$l-1$$), which is connected to the current neuron where we are calculating our activation

- $$b^l$$ - bias (as scalar) for current layer
- $$b^l_j$$ - bias (as scalar) for neuron with index $$j$$ current layer
- $$b^l$$ - bias (as vector) for current layer for each corresponding neuron

### 2.3. A Layer of Neurons

- $$a^l$$ - actiovation vector of the layer $$l$$ that can be used as an input vector $$x^{l+1}$$ for the next later ($$l+1$$)
- $$x^l$$ - input vector for the layer $$l$$, $$x^0$$ - the very first input vector for the whole neural network
- $$W^l$$ - matrix of weights for the layer $$l$$, which can be represented as follows:

```latex
\displaystyle{
	\begin{bmatrix}
		w_{11}^l & w_{12}^l & \cdots & w_{1k}^l & \cdots & w_{1m}^l \\ \\
		w_{21}^l & w_{22}^l & \cdots & w_{2k}^l & \cdots & w_{2m}^l \\ \\
		\vdots & \vdots & \cdots & \vdots & \cdots & \vdots \\ \\
		w_{j1}^l & w_{j2}^l & \cdots & w_{jk}^l & \cdots & w_{jm}^l \\ \\
		\vdots & \vdots & \cdots & \vdots & \cdots & \vdots \\ \\
		w_{n1}^l & w_{j2}^l & \cdots & w_{nk}^l & \cdots & w_{nm}^l
	\end{bmatrix}
}
```

$$n$$ - number of neurons in current layer
$$m$$ - number of neurons in previous layer
$$w_{jk}^l$$ - weight for the neuron with index $$j$$ in current layer $$l$$ coming from a neuron with index $$k$$ in previous layer $$l - 1$$

```latex
a^{l - 1} = x^{l}
```
```latex
a^l = \sigma(a^{l - 1}W^l + b^l) = \sigma(x^{l}W^l + b^l)
```
```latex
z^l = a^{l - 1}W^l + b^l = x^{l}W^l + b^l
```

Eventually, to calculate an entire layer, we do this:

```latex
a^l = \sigma(z^l)
```

## 3. Derivatives of Neural Networks and Gradient Descent
### 3.1. Cost Function

Let's say $$m$$ - number of of training examples, $$\hat y_m$$ is a total output of the neural network (is the last layer activation $$a^l$$) in the training example $$m$$.

Then the cost can be calculated as:

```latex
\displaystyle{
	C = \frac{1}{2m}\sum_{i = 0}^{m} (y - \hat y_m)^2
}
```

### 3.2. Differentiating a Neuron's Operations
#### 3.2.1. Derivative of a Binary Elementwise Function

Binary Elementwise Function or Hadamard Product is a function $$f(\vec v, \vec w) \rightarrow \vec b$$, which takes two vectors on input and produces a vector on output.

Let's assume that the product $$f$$ can be expressed as

```latex
f(\vec v, \vec w) = f_1(\vec v) \odot f_2(\vec w)
```

For the most part, $$f_1(\vec v) = \vec v$$ and $$f_2(\vec w) = \vec w$$, but let's take more geral approach and assume that those functions can take other forms as well. Also, it's worth noting that $$f_1$$ and $$f_2$$ generally are not elementwise functions. Each of them produces a vector, but each element of that vector can be calculated using other elements of the input vector  $$\vec v$$ or $$\vec w$$ respectively. It means that $$f_1$$ and $$f_2$$ can be expressed as vectors of functions $$f_1 = [f_{11}, f_{12}, ..., f_{1n}]^T$$ and $$f_2 = [f_{21}, f_{22}, ..., f_{2n}]^T$$.

Let's find a derivative of such function or Jacobian. We will start with gradients, so that we can use them to calculate our Jacobian.

```latex
f_{j}(\vec v, \vec w) = f_{1j}(\vec v) \odot f_{2j}(\vec w), 
```
where $$1 \le j \le n$$. Using the product rule, we can conclude:
```latex
\nabla f_{j}(\vec v, \vec w) = \nabla (f_{1j}(\vec v) \odot f_{2j}(\vec w)) =
```
```latex
\displaystyle{
	=
	\nabla f_{1j}(\vec v) \odot f_{2j}(\vec w) + f_{1j}(\vec v) \odot \nabla f_{2j}(\vec w) =
}
```
```latex
\displaystyle{
	\begin{bmatrix}
		\frac{df_{1j}}{v_{1}} \odot f_{2j}(w_{1}) + f_{1j}(v_{1}) \odot \frac{df_{2j}}{dw_{1}} \\ \\
		\frac{df_{1j}}{v_{2}} \odot f_{2j}(w_{2}) + f_{1j}(v_{2}) \odot \frac{df_{2j}}{dw_{2}} \\ \\
		\vdots \\ \\
		\frac{df_{1j}}{v_{n}} \odot f_{2j}(w_{n}) + f_{1j}(v_{n}) \odot \frac{df_{2j}}{dw_{n}}
	\end{bmatrix}
}
```

Now, let's find Jacobian:
```latex
J f(\vec v, \vec w) = J (f_1(\vec v) \odot f_2(\vec w)) =
```
```latex
\displaystyle{
	= J
	\begin{bmatrix}
		f_{11}(\vec v) \odot f_{21}(\vec w) \\ \\
		f_{12}(\vec v) \odot f_{22}(\vec w) \\ \\
		\vdots \\ \\
		f_{1n}(\vec v) \odot f_{2n}(\vec w)
	\end{bmatrix} =
	\begin{bmatrix}
		\nabla^T f_{11}(\vec v) \odot f_{21}(\vec w) \\ \\
		\nabla^T f_{12}(\vec v) \odot f_{22}(\vec w) \\ \\
		\vdots \\ \\
		\nabla^T f_{1n}(\vec v) \odot f_{2n}(\vec w)
	\end{bmatrix} =
}
```
```latex
\displaystyle{
	=
	\begin{bmatrix}
		\frac{df_{11}}{v_{1}} \odot f_{21}(w_{1}) + f_{11}(v_{1}) \odot \frac{df_{21}}{dw_{1}} &
		\cdots &
		\frac{df_{11}}{v_{n}} \odot f_{21}(w_{n}) + f_{11}(v_{n}) \odot \frac{df_{21}}{dw_{n}}
		\\ \\
		\frac{df_{12}}{v_{1}} \odot f_{22}(w_{1}) + f_{12}(v_{1}) \odot \frac{df_{22}}{dw_{1}} &
		\cdots &
		\frac{df_{12}}{v_{n}} \odot f_{22}(w_{n}) + f_{12}(v_{n}) \odot \frac{df_{22}}{dw_{n}}
		\\ \\
		\vdots & \cdots & \vdots
		\\ \\
		\frac{df_{1n}}{v_{1}} \odot f_{2n}(w_{1}) + f_{1n}(v_{1}) \odot \frac{df_{2n}}{dw_{1}} &
		\cdots &
		\frac{df_{1n}}{v_{n}} \odot f_{2n}(w_{n}) + f_{1n}(v_{n}) \odot \frac{df_{2n}}{dw_{n}}
	\end{bmatrix}
}
```

#### 3.2.2. Derivative of a Hadamard Product

If $$f_{1j}$$ and $$f_{2j}$$ are elementwise functions, meaning that $$f_{1j}$$ affects only $$v_j$$, and $$f_{2j}$$ affects only $$w_j$$, then our Jacobian will be a diagonal matrix:

```latex
\displaystyle{
	\begin{bmatrix}
		\frac{df_{11}}{v_{1}} \odot f_{21}(w_{1}) + f_{11}(v_{1}) \odot \frac{df_{21}}{dw_{1}} &
		\cdots &
		0
		\\ \\
		0 &
		\cdots &
		0
		\\ \\
		\vdots & \cdots & \vdots
		\\ \\
		0 &
		\cdots &
		\frac{df_{1n}}{v_{n}} \odot f_{2n}(w_{n}) + f_{1n}(v_{1}) \odot \frac{df_{2n}}{dw_{n}}
	\end{bmatrix}
}

```

In general,

```latex
J (f_{1j}(\vec v) \odot f_{2j}(\vec w))_{ij} = \frac{df_{1j}}{v_{j}} \odot f_{2j}(w_{j}) + f_{1j}(v_{j}) \odot \frac{df_{2j}}{dw_{j}}, i = j
```
```latex
J (f_{1j}(\vec v) \odot f_{2j}(\vec w))_{ij} = 0, i \neq j
```

If the product operation is just multiplication and $$f_{1j} = v_{j}$$ and $$f_{2j} = w_{j}$$, then $$\frac{df_{1j}(v_j)}{v_j} = \frac{d(v_j)}{v_j} = 1$$; $$\frac{df_{2j}(w_j)}{w_j} = \frac{d(w_j)}{w_j} = 1$$ and our Jacobian would be:
```latex
\displaystyle{
	\begin{bmatrix}
		w_1 + v_1 &
		\cdots &
		0
		\\ \\
		0 &
		\cdots &
		0
		\\ \\
		\vdots & \cdots & \vdots
		\\ \\
		0 &
		\cdots &
		w_n + v_n
	\end{bmatrix}
}
```

And again in general,

```latex
J (\vec v \vec w) = diag(w_j + v_j)
```

#### 3.2.3 Derivative of a Scalar Expansion

```latex
\displaystyle{
	n \odot \vec v =
	n \odot \begin{bmatrix}
		v_1 \\ \\
		v_2 \\ \\
		\vdots \\ \\
		v_n
	\end{bmatrix} =
	\begin{bmatrix}
		n \odot v_1 \\ \\
		n \odot v_2 \\ \\
		\vdots \\ \\
		n \odot v_n
	\end{bmatrix}
}
```
```latex
\displaystyle{
	(n \odot \vec v)' = \nabla (n \odot \vec v) = \nabla n \odot \vec v + n \odot \nabla \vec v =
}
```
```latex
\displaystyle{
	=
	\begin{bmatrix}
		\frac{dn}{dv_1} \odot v_1 + n \odot \frac{dv_1}{v_1} \\ \\
		\frac{dn}{dv_2} \odot v_2 + n \odot \frac{dv_2}{v_2} \\ \\
		\vdots \\ \\
		\frac{dn}{dv_n} \odot v_n + n \odot \frac{dv_n}{v_n}
	\end{bmatrix}
	=
	\begin{bmatrix}
		n \odot \frac{dv_1}{v_1} \\ \\
		n \odot \frac{dv_2}{v_2} \\ \\
		\vdots \\ \\
		n \odot \frac{dv_n}{v_n}
	\end{bmatrix}
	=
	n \odot \nabla \vec v
}
```

If the prodcut operation is multiplication or addition, then

```latex
\displaystyle{
	n \nabla \vec v = n \vec 1; n + \nabla \vec v = n + \vec 1;
}
```

If we have some $$f(\vec v)$$ which is a vector of functions $$[f_1(\vec v), f_2(\vec v), \ldots, f_n(\vec v)]$$, then we can calculate a Jacobian of the product of that function and some scalar $$s$$:

```latex
\displaystyle{
	J (s \odot f(\vec v)) =
	\begin{bmatrix}
		\nabla^T (s \odot f_1(\vec v)) \\ \\
		\nabla^T (s \odot f_2(\vec v)) \\ \\
		\vdots \\ \\
		\nabla^T (s \odot f_n(\vec v))
	\end{bmatrix}
	=
	\begin{bmatrix}
		s \odot \nabla^T f_1(\vec v) \\ \\
		s \odot \nabla^T f_2(\vec v) \\ \\
		\vdots \\ \\
		s \odot \nabla^T f_n(\vec v)
	\end{bmatrix}
}
```

Again, if $$f_i(\vec v)$$ is elementwise function, meaning that $$f_i(\vec v)$$ affects only $$v_i$$, then

```latex
J (s \odot f(\vec v)) = diag(s\frac{df}{dv_i}),
```
if $$\odot$$ is multiplication,

or

```latex
J (s \odot f(\vec v)) = diag(s + \frac{df}{dv_i}),
```
if $$\odot$$ is addition.

If $$f(v_i) = v_i$$, then

```latex
J (s \odot f(\vec v)) = diag(s),
```
if $$\odot$$ is multiplication,

or

```latex
J (s \odot f(\vec v)) = diag(s + 1),
```
if $$\odot$$ is addition.

#### 3.2.4 Derivative of a Sum

```latex
\displaystyle{
	S = \sum_{i = 1}^n f_i(\vec v)
}
```
```latex
\displaystyle{
	d S = d(\sum_{i = 1}^n f_i(\vec v)) = \sum_{i = 1}^n \nabla f_i(\vec v) =
	\sum_{i = 1}^n \begin{bmatrix}
		\frac{df_i}{dv_1} \\ \\
		\frac{df_i}{dv_2} \\ \\
		\vdots \\ \\
		\frac{df_i}{dv_n}
	\end{bmatrix}
}
```

If $$f_i(\vec v)$$ is elementwise function regarding $$v_i$$, then

```latex
\nabla f_i(\vec v) = \vec {v_{i}}, {\vec {v_{i}}}_{i} = \frac{df_i}{dv_i}; {\vec {v_{i}}}_{j \neq i} = 0;
```

```latex
d S = \sum_{i = 1}^n \vec {v_{i}} = \vec{\frac{df}{dv}} =
\begin{bmatrix}
	\frac{df_1}{dv_1} \\ \\
	\frac{df_2}{dv_2} \\ \\
	\vdots \\ \\
	\frac{df_n}{dv_n}
\end{bmatrix}
```

If $$f_i(v_i) = v_i$$, then

```latex
d S = \vec 1
```

### 3.3 Derivative of a Neuron's Activation (Recosdier from this point)

```latex
\displaystyle{a = \sigma(w^Tx + b) = \sigma(z)}
```
```latex
\displaystyle{\frac{da}{dw} = \frac{d\sigma}{dz}\frac{dz}{dw}}
```
```latex
\displaystyle{\frac{da}{db} = \frac{d\sigma}{dz}\frac{dz}{db}}
```
```latex
\displaystyle{
	a = \sigma(\sum(w\odot x) + b)
}
```
```latex
\displaystyle{
	a = \sigma(z(s)); z(s) = s(h) + b; s(h) = \sum(h(w)); h(w) = w \odot x
}
```
```latex
\displaystyle{
	\frac{da}{dw} = \frac{d\sigma}{dz}\frac{dz}{ds}\frac{ds}{dh}\frac{dh}{dw}
}
```
```latex
\sigma(x) = max(0, x)
```
```latex
\displaystyle{
	\frac{dh}{dw} = \frac{d(w \odot x)}{dw} = diag(w_i + x_i) \medspace (from \medspace 3.2.2.)
}
```
```latex
\displaystyle{
	\frac{ds}{dh} = \frac{d(\sum_{i=1}^n f_i(w, x))}{dh(w)}; f_i(w, x) = w_{j = i} + x_{j = i}, f_i(w, x) = 0_{j \neq i}
}
```
```latex
\displaystyle{
	\frac{ds}{dh} = \vec 1 \medspace (from \medspace 3.2.4.)
}
```
```latex
\displaystyle{
	\frac{dz}{ds} = \frac{dz(\vec s \odot \vec b)}{ds} = \frac{dz(\vec s + \vec b)}{ds}
}
```
Since, in this case $$b$$ is a scalar vector, then from (3.2.3.)
```latex
\displaystyle{
	\frac{dz}{ds} = diag(b_i + 1)
}
```
```latex
\displaystyle{
	\frac{d\sigma}{dz} = \vec{\hat z}, \hat z_i = 0 (z_i < 0), \hat z_i = 1 (z_i > 0)
}
```

Finally,

```latex
\displaystyle{
	\frac{da}{dw} = \vec{\hat z} diag(b_i + 1) \vec{1} diag(w_i + x_i) = \vec{\hat z} diag(b_i + 1) diag (w_i + x_i) = 
}
```
```latex
\displaystyle{
	= \vec{\hat z} diag((b_i + 1)(w_i + x_i)) = \vec{\hat z} diag(b_iw_i + b_ix_i + w_i + x_i)
}
```
```latex
\displaystyle{
	\frac{da}{dw} = diag(\hat a) , \hat a_i = b_iw_i + b_ix_i + w_i + x_i, z_i > 0; \hat a_i = 0, z_i < 0;
}
```
```latex
\displaystyle{
	\frac{da}{db} = \vec{(\hat b)}, \hat b_i = 1, z_i > 0; \hat b_i = 0, z_i < 0;
}
```

### 3.4. Derivative of the Cost for a Simple Network

```latex
\displaystyle{
	C = \frac{1}{2m}\sum_{e=1}^m(y_e - \hat y_e)^2 = \frac{1}{2m}\sum_{e=1}^m(y_e - a_e^l)^2
}
```
```latex
\displaystyle{
	\frac{dC}{dw} = \frac{dC}{da}\frac{da}{dw}
}
```
```latex
\displaystyle{
	\frac{dC}{da} = \frac{1}{2m}\sum_{e=1}^m\frac{d((y_e - a_e^l)^2)}{da} = \frac{1}{2m}\sum_{e=1}^m\frac{d((y_e - a_e^l)^2)}{da}
}
```
```latex
\displaystyle{
	\frac{dC}{da} = \frac{1}{m}\sum_{e=1}^m(a_e^l - y_e)
}
```

$$\frac{da}{dw}$$ is calculated in the previous step.


```latex
\displaystyle{
	\frac{dC}{dw} = \frac{1}{m}\sum_{i=e}^m(a_e^l - y_e) diag({\hat a^l})_e
}
```
```latex
\displaystyle{
	\frac{dC}{dw} = \frac{1}{m}\sum_{e=1}^m(a_e^l - y_e) diag(b_i^lw_i^l + b_i^lx_i^l + w_i^l + x_i^l)_e, z_i^l > 0
}
```
```latex
\displaystyle{
	\frac{dC}{dw} = 0, z_i^l < 0
}
```
```latex
\displaystyle{
	\frac{dC}{db} = \frac{dC}{da}\frac{da}{db}
}
```
```latex
\displaystyle{
	\frac{dC}{db} = \frac{1}{m}\sum_{e=1}^m(a_e^l - y_e) \hat b, \hat b_i = 1, z_i > 0; \hat b_i = 0, z_i < 0;
}
```

### 3.5. Gradient Descent Algorithm and SGD 

```latex
\Theta_w = \Theta_w - \alpha \nabla_w C
```
```latex
\Theta_b = \Theta_b - \alpha \nabla_b C
```

$$\Theta_w$$, $$\Theta_b$$ - vectors of new calculated weight and biases, $$\nabla_w C$$ - presumably our diag from our $$\frac{dC}{dw}$$ matrix, $$\nabla_b C$$ - bias derivatives, $$\alpha$$ is a learning rate.


## 4. Backpropagation
### 4.1. The Error of a Node

```latex
\displaystyle{
	\delta_j^l = \frac{dC}{dz^l_j}
}
```

### 4.2. The Four Equations of Backpropagation 
#### 4.2.1. Equation 1: The Error of the last Layer

```latex
\displaystyle{
	\delta_j^L = \frac{dC}{da^L_j}\frac{da^L_j}{dz^L_j}, L - last \medspace layer
}
```
```latex
\displaystyle{
	\delta_j^L = \frac{dC}{da^L_j}\sigma'(z^L_j)
}
```

For entire layer,

```latex
\displaystyle{
	\delta^L = \nabla_{a^L}{C}\sigma'(z^L)
}
```

#### 4.2.2. Equation 2: The Error of any layer

```latex
\displaystyle{
	\delta^l = ((W^{l+1})^T\delta^{l+1})\odot\sigma'(z^l)
}
```

#### 4.2.3. Equation 3: The Derivative of the Cost w.r.t any bias

```latex
\displaystyle{
	\frac{dC}{db^l} = \delta^l
}
```

#### 4.2.4. Equation 4: The Derivative of the Cost w.r.t any weight 

```latex
\displaystyle{
	\frac{dC}{dw_{ij}^l} = a_k^{l-1} \delta_j^l
}
```

#### 4.2.5. Vectorizing Equation 4

```latex
\displaystyle{
	\frac{dC}{dW^l} = b^l \otimes (a^{l-1})^T
}
```

### 4.3 Tying Part III and Part IV together



