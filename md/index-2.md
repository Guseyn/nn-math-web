# Mathematics of Neural Network

## 1. Neuron in Isolation

```latex
\displaystyle{

}
```

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
$$a^{l-1}$$ - input vector of previous layer or $$a^0=x^1$$ - which is input for the whole network
$$W^l$$ - matrix of weights between previous layer and current layer, which contains elements $$w_{jk}^l$$, where $$l$$ - index of current layer, $$j$$ - index of the neuron in current layer $$l$$ where the weight is leading to, and $$k$$ - index of the neuron in previous layer $$l - 1$$ where the weight is coming from. $$1 \leq j \leq n^l$$; $$n^l$$ - number of neurons in current layer, $$0 \leq k \leq n^{l-1}$$, $$n^{l-1}$$ - number of neurons in previous layer $$l-1$$

```latex
\displaystyle{
	W^l =
	\begin{bmatrix}
		w^l_{11} & w^l_{12} & \cdots & w^l_{1n^{l-1}} \\ \\
		w^l_{21} & w^l_{22} & \cdots & w^l_{2^{l-1}} \\ \\
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
	C(W, b) = \frac{1}{2E}\sum_{e = 1}^E(a^L_e - y_e)^2
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
- We set matrices of weights between each pair of neighbouring layers (or initial input $$x^1=a^0$$). Let's express it as $$W$$ - vector of matrices, where $$W^l$$ is a matrix of weights between layers $$l$$ and $$l - 1$$. Weight themselves can be setup randomly, since we have a goal to get right weights for our network.
- We set a vector of biases for each layer $$b$$, where $$b^l$$ is a bias for a layer $$l$$.
- We set a vector of desired results for learning examples $$y$$, where $$y_e$$ is a desired result (as a vector of scalars like $$a^L_e$$) for an example with index $$e$$, $$1 \leq e \leq E$$, $$E$$ - total number of examples.

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

## 9. Finding Derivative of Cost Function With Regard to Weights

When we try to find $$C'(W)$$, it means that we need to find $$C$$ with regard to each $$W^l$$, therefore for each $$w^l_{jk}$$. In another words, we need to find $$\frac{dC}{dw_{jk}^l}$$ for each layer $$l$$, for each $$w^l_{kl}$$ in each matrix $$W^l$$ between layer $$l-1$$ and $$l$$.

```latex
\displaystyle{
	\frac{dC}{w_{jk}^l} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d(a^L_e - y_e)^2}{dw_{jk}^l}
}
```

Let $$\delta_e = (a^L_e - y_e)^2$$, then

```latex
\displaystyle{
	\frac{dC}{w_{jk}^l} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{dw_{jk}^l}
}
```

Now, let's explore more about $$\delta_e$$:

```latex
\displaystyle{
	\delta_e = \sigma(a_e^{L-1}{W^L} + b^L)
}
```

We can omit index $$e$$ and explore $$\delta$$ in isolation, since it does not really affect any calculations but at the same time it will simplify equations:

```latex
\displaystyle{
	\delta = \sigma(a^{L-1}{W^L} + b^L)
}
```

As we discussed before $$z^l=a^{L-1}{W^L} + b^L$$, so

```latex
\displaystyle{
	\delta = \sigma(z^l)
}
```

Let's now explore $$z^l$$:

```latex
\displaystyle{
	z^l=a^{L-1}{W^L} + b^L
}
```

We can see that $$z^l$$ depends on inputs from previous layer $$a^{L-1}$$, which also depends on all previous layers. And it's quite difficult to calculate. This is why there is such thing as **backpropagation** that allows to simplify calculations.
