# Mathematics of Neural Network

## 1. Neuron in Isolation

```latex
\displaystyle{
	a = \sigma(\sum_{i=1}^{n}{x_i}{w_i} + b)
}
```

$$a$$ - rusult or activation of a neuron (scalar)
$$x$$ - vector with inputs with length $${n}$$
$$w$$ - vector of weights with length $${n}$$
$$b$$ - scalar bias
$$\sigma$$ - activation function, for example $$ReLU(x) = max(0, x)$$

## 2. Neuron in Terms of Layers in a Neuron Network

```latex
\displaystyle{
	a_j^l = \sigma(\sum_{k=1}^{n^{l-1}}{x^l_k}{w^l_{jk}} + b^l)
}
```

or

```latex
\displaystyle{
	a_j^l = \sigma(\sum_{k=1}^{n^{l - 1}}{a^{l-1}_k}{w^l_{jk}} + b^l)
}
```

$$n$$ - vector of number of neurons in each layer.
$$n^l$$ - number of neurons in layer with index $$l$$.
$$n^{l-1}$$ - number of neurons in layer with index $$l-1$$ (previous layer).
$$a_j^l$$ - result of a neuron in current layer $$l$$ with index $$j$$ where $$1 \leq j \leq n^l$$, $$n^l$$ - number of neurons in the current layer.
$$a^l$$ - result as vector that conbines all $$a_j^l$$, $$1 \leq j \leq n^l$$, $$n^l$$ - number of neurons in the current layer $$l$$.
$$a^{l - 1}$$ - result as a vector of nerons in previous layer, which is used as input for current layer with elements $$a_k^{l-1}$$, $$1 \leq k \leq n^{l-1}$$, $$n^{l-1}$$ - number of neurons in the previous layer.
$$a_k^{l-1}$$ - result of neuron with index k in the previous layer $$l-1$$, $$1 \leq k \leq n^{l-1}$$, $$n^{l-1}$$ - number of neurons in the previous layer.
$$x^1=a^0$$ - very first input for neural network can be considered as activation output with index $$0$$.
$$w_{jk}^l$$ - weight, leading in current layer $$l$$ for a neuron with index $$j$$ and comming from a neuron with index $$k$$ in the previous layer $$l-1$$, $$1 \leq j \leq n^l$$, $$n^l$$ - number of neurons in the current layer and $$1 \leq k \leq n^{l-1}$$, $$n^{l-1}$$ - number of neurons in the previous layer.
$$b^l$$ - bias, which is used for current layer. It also can be different for each neuron like $$b_j^l$$, $$0 \leq j \leq n^l$$, $$n^l$$ - number of neurons in the current layer.

Let's define $$z^l_j$$ as

```latex
\displaystyle{
	z_j^l = \sum_{k=1}^{n^{l - 1}}{a^{l-1}_k}{w^l_{jk}} + b^l
}
```

```latex
\displaystyle{
	a_j^l = \sigma(z^l_j)
}
```

## 3. Layer of Neurons

```latex
\displaystyle{
	a^l = \sigma(z^l)
}
```

$$a^l$$ can be used as input vector for next layer $$l+1$$. Let's expand further vector notation:

```latex
\displaystyle{
	a^l = \sigma(a^{l-1}{W^l} + b^l)
}
```

$$a^l$$ - result vector of current layer
$$a^{l-1}$$ - result vector of previous layer or input for current layer, $$a^0=x^1$$ - input for the whole network
$$W^l$$ - matrix of weights between previous layer and current layer, which contains elements $$w_{jk}^l$$, where $$l$$ - index of current layer, $$j$$ - index of the neuron in current layer $$l$$ where the weight is leading to, and $$k$$ - index of the neuron in previous layer $$l - 1$$ where the weight is coming from. $$1 \leq j \leq n^l$$; $$n^l$$ - number of neurons in current layer, $$0 \leq k \leq n^{l-1}$$, $$n^{l-1}$$ - number of neurons in previous layer $$l-1$$

```latex
\displaystyle{
	W^l =
	\begin{bmatrix}
		w^l_{11} & w^l_{12} & \cdots & w^l_{1n^{l-1}} \\ \\
		w^l_{21} & w^l_{22} & \cdots & w^l_{2n^{l-1}} \\ \\
		\vdots & \vdots & w^l_{jk} & \vdots \\ \\
		w^l_{n^l1} & w^l_{n^l2} & \cdots & w^l_{n^ln^{l-1}}
	\end{bmatrix}
}
```
$$b^l$$ - vector of biases used for current layer $$l$$, $$1 \leq l \leq L$$; $$L$$ - index of the last layer in neural network

## 4. Expanding Last layer

Let $$L$$ be an index of last layer in the neural network. Then,

```latex
\displaystyle{
	a^L = \sigma(a^{L-1}{W^L} + b^L)
}
```

We know that

```latex
\displaystyle{
	a^{L-1} = \sigma(a^{L-2}{W^{L-1}} + b^{L-1})
}
```

Then

```latex
\displaystyle{
	a^L = \sigma(\sigma(a^{L-2}{W^{L-1}} + b^{L-1}){W^L} + b^L)
}
```

Going further,

```latex
\displaystyle{
	a^L = \sigma(\sigma(\sigma(a^{L-3}{W^{L-2}} + b^{L-2}){W^{L-1}} + b^{L-1}){W^L} + b^L)
}
```

We can go and on till L - (L - 1). It's just a demonstration of the recursive nature of the argument in the activation function.

## 5. Cost Function

```latex
\displaystyle{
	C(W, b) = \frac{1}{2E}\sum_{e = 1}^E(y_e - a^L_e)^2
}
```

$$C(W, b)$$ - result of cost function is vector of scalars.
$$e$$ - index of the current learning example.
$$E$$ - number of learning examples.
$$y_e$$ - desired result for the example with index $$e$$.
$$a^L_e$$ - output of the last layer $$L$$ (or, in another words, an entire network) for example with index $$e$$.

Here we consider $$C$$ as $$C(W, b)$$, because vector $$W$$ of matrices $$W^l$$ determines weights and vector $$b$$ determines biases that affect our cost function.

## 6. Learning Goal

If $$C$$ is close to $$\vec 0$$, then it means that our network is configured in such a way, that it's very close for each learning example with index $$e$$. Thefore, we can use some custom input and expect to get the right output. 

## 7. What Goes on Initial Input in Neural Network

Having discussed everything above, we can conclude what we need for our input for neural network:

- We set number of layers in our network $$L + 1$$, where $$L$$ - index of last layer, and $$0$$ is used for input vector.
- We set number of neurons in each layer with index $$l$$. Let's have a vector $$n$$ which represents number of neurons in each layer. Then $$n^l$$ is a number of neurons in each layer with index $$l$$.
- We set initial input vector $$x^1 = a^0$$.
- We set matrices of weights between each pair of neighbouring layers (or initial input $$x^1=a^0$$). Let's express it as $$W$$ - vector of matrices, where $$W^l$$ is a matrix of weights between layers $$l$$ and $$l - 1$$. Weights themselves can be setup randomly, since we have a goal to get right weights for our network.
- We set a vector of biases for each layer $$b$$, where $$b^l$$ is a bias for a layer $$l$$. They can also be some random numbers.
- We set a vector of desired results for a vector of learning examples $$y$$, where $$y_e$$ is a desired result (as a vector of scalars like $$a^L_e$$) for an example with index $$e$$, $$1 \leq e \leq E$$, $$E$$ - total number of examples.

## 8. Finding Minimum of Cost Function

Let's use Gradient Descent for our cost function:


```latex
\displaystyle{
	P_{i+1} = P_{i} - \alpha C'(P_i)
}
```

$$i$$ - index of iteration that we do to find the minimum of $$P_i$$.
$$P_i$$ - result of cost function for iteration with index $${i}$$.
$$P_{i+1}$$ - result of cost function for iteration with index $${i+1}$$.
$$\alpha$$ - learning rate, some constant scalar
$$C'(P_i)=\frac{dC(P_i)}{dP_i}$$ - derivative of cost function for iteration with index $$i$$.

In our case $$P = W\|b$$ for all iterations. $$\|$$ is logical "or".

We can rewrite Gradient Descent for our case:

```latex
\displaystyle{
	\begin{cases}
		W_{i+1} = W_{i} - \alpha C'(W_i) \\ \\
		b_{i+1} = b_{i} - \alpha C'(b_i)
	\end{cases}
}
```

Basically, once we found $$\alpha C'(W_i\|b_i)$$, then we can compute $$P_{i+1}$$ or next $$W_{i+1}\|b_{i+1}$$ for next iteration. Then we use those new $$W_{i+1}\|b_{i+1}$$ vectors for all examples $$1 \leq e \leq E$$, then we find new cost function $$C(W_{i+1}\|b_{i+1})$$, if it's not small enough, then we reapeat process of finding new $$P_{i+1}$$. So, basically it goes like this:

1. Go through all examples $$e$$, $$1 \leq e \leq E$$ and find all $$a^L_e$$ for each example
2. Find cost function $$C(W_i, b_i)$$ for iteration with index $$i$$, $$i=0$$ for first iteration
3. If $$C(W_i, b_i)$$ is close enough to $$\vec 0$$, then it's all good, you got correct $$W$$ and $$b$$. Use them for custom inputs $$x$$. If not, go to 4.
4. Find $$P_{i+1}=W_{i+1}\|b_{i+1}$$ and go again to 1.


Now, all we need is to calculate $$C'(W_i|b_i)$$. Naturally, we can split this task into two separate tasks of finding $$C'(W_i)$$ and $$C'(b_i)$$. We can also just consider them in isolation: $$C'(W)$$ and $$C'(b)$$.

It's also worth noting that finding $$C'(W)\|C'(b)$$ means finding all $$\frac{dC}{dw_{jk}^l}\|\frac{dC}{db^l}$$, therefore Gradient Descent for each iteration with index $$i$$ on element level would look like:

```latex
\displaystyle{
	\begin{cases}
		(w_{jk}^l)_{i+1} = (w_{jk}^l)_{i} - \alpha C'((w_{jk}^l)_i) \\ \\
		(b^l)_{i+1} = (b^l)_{i} - \alpha C'((b^l)_i)
	\end{cases}
}
```

## 9. Finding Derivative of Cost Function with Regard to Weights

When we try to find $$C'(W)$$, it means that we need to find $$C$$ with regard to each $$W^l$$, therefore for each $$w^l_{jk}$$. In another words, we need to find $$\frac{dC}{dw_{jk}^l}$$ for each layer $$l$$, for each $$w^l_{kl}$$ in each matrix $$W^l$$ between layer $$l-1$$ and $$l$$.

```latex
\displaystyle{
	\frac{dC}{W^l} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d(y_e - a^L_e)^2}{dW^l}
}
```

Let $$\delta_e = (y_e - a^L_e)^2$$, then

```latex
\displaystyle{
	\frac{dC}{W^l} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{dW^l}
}
```

Now, let's explore more about $$a_e^L$$:

```latex
\displaystyle{
	a_e^L = \sigma(a_e^{L-1}{W^L} + b^L)
}
```

As we discussed before $$z_e^L=a_e^{L-1}{W^L} + b^L$$, so

```latex
\displaystyle{
	a_e^L = \sigma(z_e^L)
}
```

Let's now explore $$z^L_e$$:

```latex
\displaystyle{
	z_e^L=a_e^{L-1}{W^L} + b^L
}
```

Just for clarity, let's recall how it can be written on element level:

```latex
\displaystyle{
	(z_j^L)_e=\sum_{k=1}^{n^{l-1}}(a_k^{L-1})_e{w_{jk}^L} + b^L
}
```

Here indices $$_e$$ just indicate current learning example with index $$_e$$, we just keep it here for consistency.

So, in the end we have following chain of derivatives:

```latex
\displaystyle{
	\frac{dC}{dW^l} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{da_e^L}\frac{da_e^L}{dz_e^L}\frac{dz_e^L}{dW^l}
}
```

We can notice here very interesting detail: we are trying to find $${dC}/{dW^l}$$ for any layer $$l$$, but our chain seems like heavily depends on layer with index $$L$$ - last layer in the neural network. In fact, this chain goes ever further or rather all the way back to the layer $$l=1$$. In another words, $${dW^L}$$ relies on $${dW^l}$$.

It seems, the best approach would be to find $${dC}/{dW^L}$$ first, and only then try to figure out other layers.

### 9.1. Finding Derivative of Cost Function with Regard to Weights In The Last Layer 

So we can find derivative $${dC}/{dW^L}$$,

```latex
\displaystyle{
	\frac{dC}{dW^L} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{da_e^L}\frac{da_e^L}{dz_e^L}\frac{dz_e^L}{dW^L}
}
```

Let's start with $${d\delta_e}/{da_e^L}$$ or $${d((y_e - a^L_e)^2)}/{d((y_e - a^L_e))}$$. $$(y_e - a^L_e)$$ - is a vector, therefore $$(y_e - a^L_e)^2$$ is a vector, so the result of it will be a gradient $$\nabla\delta_e$$.

Let $$f_1(x)=x^2$$ and $$f_2(a^L_e) = y_e - a^L_e$$ (c - scalar), then $$f_1(f_2(a^L_e)) = (y_e - a^L_e)^2$$. Using chain rule, we will get:

```latex
\displaystyle{
	\frac{df_1(f_2(a^L_e))}{da^L} = \frac{df_1}{df_2}\frac{df_2}{da^L} = \frac{df_1((y_e - a^L_e)^2)}{d(y_e - a^L_e)}\frac{d(y_e - a^L_e)}{da^L} = 
}
```
```latex
\displaystyle{
	= 2(y_e - a^L_e)\otimes(-\vec 1) = 2(a^L_e - y_e)
}
```
```latex
\displaystyle{
	\frac{d\delta_e}{da_e^L} = 2(a^L_e - y_e) \medspace \medspace \medspace [1]
}
```

Next, we need to find $$da^L_e/dz^L_e$$, or $$d\sigma(z^L_e)/d(z^L_e)$$. Basically, we need to calculate $$d\sigma(z)/dz$$ = $$d(max(0, z))/dz$$. It can be solved in following way:

```latex
\Bigl(\frac{da_e^L}{dz_e^L}\Bigr)_j =
\displaystyle{
	\begin{cases}
		0; (z_j^L)_e < 0 \\ \\
		1; (z_j^L)_e \geq 0
	\end{cases}
}
```
```latex
	0 \leq j \leq n^L
```

$${da_e^L}/{dz_e^L}$$ is a vector, therefore we need to consider all $$z_j$$ separately to construct whole vector, $$1 \leq j \leq n^L$$.

Actually, $$max(0, z)$$ is not defind at $$z=0$$, but we can just assume that it very close to some very small $$\epsilon$$, so we can rewrite the solution:

```latex
\Bigl(\frac{da_e^L}{dz_e^L}\Bigr)_j =
\displaystyle{
	\begin{cases}
		0; (z_j^L)_e < 0 \\ \\
		1; (z_j^L)_e > 0 \\ \\
		\epsilon; (z_j^L)_e = 0
	\end{cases}
} \medspace \medspace \medspace [2]
```
```latex
	0 \leq j \leq n^L
```

Finally, let's find $${dz_e^L}/{dW^L}$$ or $${d(a_e^{L-1}W^L + b^L)}/{dW^L}$$. When we multiply a vector to a matrix, we transpose our vector in order to apply the vector to all rows in the matrix. Therefore,

```latex
\displaystyle{
	\frac{dz_e^L}{dW^L} =
	(a^{L-1}_e)^T
} \medspace \medspace \medspace [3]
```

Now, let's combine results from [1], [2], [3] and write down $${dC}/{dW^L}$$:

```latex
\displaystyle{
	\Bigl(\frac{dC}{dW^L}\Bigr)_{jk} = \frac{1}{2E}\sum_{e=1}^{E}
	\begin{cases}
		0; (z_j^L)_e < 0 \\ \\
		2((a^L_j)_e - (y_j)_e) \cdot (a^{L-1}_k)_e; (z_j^L)_e > 0 \\ \\
		\epsilon; (z_j^L)_e = 0
	\end{cases}
}
```
```latex
	0 \leq j \leq n^L, 0 \leq k \leq n^{L-1}
```

```latex
\displaystyle{
	\Bigl(\frac{dC}{dW^L}\Bigr)_{jk} = \frac{1}{E}\sum_{e=1}^{E}
	\begin{cases}
		0; (z_j^L)_e < 0 \\ \\
		((a^L_j)_e - (y_j)_e) \cdot (a^{L-1}_k)_e; (z_j^L)_e > 0 \\ \\
		\epsilon; (z_j^L)_e = 0
	\end{cases}
}
```
```latex
	0 \leq j \leq n^L, 0 \leq k \leq n^{L-1}
```

On element level, it would lool like this:

```latex
\displaystyle{
	\frac{dC}{dw_{jk}^L} = \frac{1}{E}\sum_{e=1}^{E}
	\begin{cases}
		0; (z_j^L)_e < 0 \\ \\
		((a^L_j)_e - (y_j)_e) \cdot (a^{L-1}_k)_e; (z_j^L)_e > 0 \\ \\
		\epsilon; (z_j^L)_e = 0
	\end{cases}
}
```
```latex
	0 \leq j \leq n^L, 0 \leq k \leq n^{L-1}
```

So, to calculate next weights for the next iteration with index $$i+1$$, we do following:


```latex
	(w_{jk}^L)_{i+1} = (w_{jk}^L)_{i} - \alpha C'((w_{jk}^L)_i)
```

```latex
	(w_{jk}^L)_{i+1} = (w_{jk}^L)_{i} -
	\alpha\frac{1}{E}\sum_{e=1}^{E}
	\begin{cases}
		0; (z_j^L)_e < 0 \\ \\
		((a^L_j)_e - (y_j)_e) \cdot (a^{L-1}_k)_e; (z_j^L)_e > 0 \\ \\
		\epsilon; (z_j^L)_e = 0
	\end{cases}
```
```latex
	0 \leq j \leq n^L, 0 \leq k \leq n^{L-1}
```

### 9.2. Finding Derivative of Cost Function with Regard to Weights in Any Layer

So, how can we find derivative of cost function for any layer? We can assume that we can find the derivative in the layer with index $$l$$ with given derivative in the layer with index $$l+1$$, because as we discussed earlier, any given layer depends on previous ones.

And we already calculated the derivative for layer $$L$$. If we find the derivative for layer $$L-1$$, we can use the same pattern or iterative method to find the derivative for all layers $$L-2$$, $$L-3$$, ..., 1.

First, let's explore $${dz_e^L}/{dW^{L-1}}$$.

```latex
\displaystyle{
	\frac{dz_e^L}{dW^{L-1}} = \frac{d(a_e^{L-1}{W^L} + b^L)}{dW^{L-1}} = \frac{d(\sigma(a_e^{L-2}{W^{L-1}} + b^{L-1}){W^L} + b^L)}{dW^{L-1}} =
}
```
```latex
\displaystyle{
	= \frac{d(\sigma(a_e^{L-2}{W^{L-1}} + b^{L-1}){W^L})}{dW^{L-1}} =
}
```
```latex
\displaystyle{
	= W^L\frac{d(\sigma(a_e^{L-2}{W^{L-1}} + b^{L-1}))}{d(a_e^{L-2}{W^{L-1}} + b^{L-1})}\frac{d(a_e^{L-2}{W^{L-1}} + b^{L-1})}{W^{L-1}} =
}
```
```latex
\displaystyle{
	= W^L\frac{d(\sigma(a_e^{L-1}))}{d(a_e^{L-1})}(a^{L-2})^T_e = W^L\frac{da^{L-1}}{dz^{L-1}}(a^{L-2})^T_e
}
```

Let's now write down whole $${dC}/{dW^{L-1}}$$:

```latex
\displaystyle{
	\frac{dC}{dW^{L-1}} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{da_e^L}\frac{da_e^L}{dz_e^L}\frac{dz_e^L}{dW^{L-1}} =
}
```
```latex
\displaystyle{
	=
	\frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{da_e^L}\frac{da_e^L}{dz_e^L}W^L\frac{da^{L-1}}{dz^{L-1}}(a^{L-2})^T_e
}
```
```latex
\displaystyle{
	\frac{dC}{dw_{jk}^{L-1}} = \frac{1}{E}\sum_{e=1}^{E}
	\begin{cases}
		0; (z_j^L)_e < 0 | (z_j^{L-1})_e < 0 \\ \\
		((a^L_j)_e - (y_j)_e) \cdot w^L_{jk} \cdot (a^{L-2}_j)_e; (z_j^L)_e > 0 \& (z_j^{L-1})_e > 0 \\ \\
		\epsilon; (z_j^L)_e = 0 | (z_j^{L-1})_e = 0
	\end{cases}
}
```
