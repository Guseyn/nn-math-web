'use strict'

const path = require('path')
const { Backend, RestApi, ServingFilesEndpoint, NotFoundEndpoint } = require('@cuties/rest')
const CustomIndexEndpoint = require('./endpoints/CustomIndexEndpoint')

const mapperForStatic = (url) => {
  const parts = url.split('?')[0].split('/').filter(part => part !== '')
  return path.join(...parts)
}

const mapperForSrc = (url) => {
  const mainPart = `${url.split('.html')[0]}.html`
  const parts = mainPart.split('/').filter(part => part !== '')
  return path.join(...parts)
}

new Backend(
  'http',
  4200,
  '127.0.0.1',
  new RestApi(
    new CustomIndexEndpoint(
      'index.html',
      new NotFoundEndpoint(new RegExp(/\/not-found/))
    ),
    new ServingFilesEndpoint(
      new RegExp(/^\/(html|js|json|image|css|md)/),
      mapperForStatic,
      {},
      new NotFoundEndpoint(new RegExp(/\/not-found/))
    ),
    new ServingFilesEndpoint(
      new RegExp(/[^\s]+.html([/?][^\s]*)?$/),
      mapperForSrc,
      {},
      new NotFoundEndpoint(new RegExp(/\/not-found/))
    ),
    new NotFoundEndpoint(new RegExp(/\/not-found/))
  )
).call()


// ### 9.2. Finding Derivative of Cost Function with Regard to Weights in Any Layer

// So, how can we find derivative of cost function for any layer? We can assume that we can find the derivative in the layer with index $$l$$ with given derivative in the layer with index $$l+1$$, because as we discussed earlier, any given layer depends on previous ones.

// And we already calculated the derivative for layer $$L$$. If we find the derivative for layer $$L-1$$, we can use the same pattern or iterative method to find the derivative for all layers $$L-2$$, $$L-3$$, ..., 1.

// If we decide to find derivative $${dz_e^L}/{dW^{l}}$$ for any layer with index $$l$$, it will be extremely challenging because of recursive nature of the arguments for the activation function. We would need to express $$z_e^L$$ recursively via layer any given $$l$$.

// We can do it via interesting trick that would simplify the calculations. In previous section, we found new weight for new iteration in the last layer. We can recalculate all activation 













// First, let's explore $${dz_e^L}/{dW^{L-1}}$$.

// ```latex
// \displaystyle{
//   \frac{dz_e^L}{dW^{L-1}} = \frac{d(a_e^{L-1}{W^L} + b^L)}{dW^{L-1}} = \frac{d(\sigma(a_e^{L-2}{W^{L-1}} + b^{L-1}){W^L} + b^L)}{dW^{L-1}} =
// }
// ```
// ```latex
// \displaystyle{
//   = \frac{d(\sigma(a_e^{L-2}{W^{L-1}} + b^{L-1}){W^L})}{dW^{L-1}} =
// }
// ```
// ```latex
// \displaystyle{
//   = W^L\frac{d(\sigma(a_e^{L-2}{W^{L-1}} + b^{L-1}))}{d(a_e^{L-2}{W^{L-1}} + b^{L-1})}\frac{d(a_e^{L-2}{W^{L-1}} + b^{L-1})}{W^{L-1}} =
// }
// ```
// ```latex
// \displaystyle{
//   = W^L\frac{d(\sigma(a_e^{L-1}))}{d(a_e^{L-1})}(a^{L-2})^T_e = W^L\frac{da^{L-1}}{dz^{L-1}}(a^{L-2})^T_e
// }
// ```

// Let's now write down whole $${dC}/{dW^{L-1}}$$:

// ```latex
// \displaystyle{
//   \frac{dC}{dW^{L-1}} = \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{da_e^L}\frac{da_e^L}{dz_e^L}\frac{dz_e^L}{dW^{L-1}} =
// }
// ```
// ```latex
// \displaystyle{
//   =
//   \frac{1}{2E}\sum_{e=1}^{E}\frac{d\delta_e}{da_e^L}\frac{da_e^L}{dz_e^L}W^L\frac{da^{L-1}}{dz^{L-1}}(a^{L-2})^T_e
// }
// ```
// ```latex
// \displaystyle{
//   \frac{dC}{dw_{jk}^{L-1}} = \frac{1}{E}\sum_{e=1}^{E}
//   \begin{cases}
//     0; (z_j^L)_e < 0 | (z_j^{L-1})_e < 0 \\ \\
//     ((a^L_j)_e - (y_j)_e) \cdot w^L_{jk} \cdot (a^{L-2}_j)_e; (z_j^L)_e > 0 \& (z_j^{L-1})_e > 0 \\ \\
//     \epsilon; (z_j^L)_e = 0 | (z_j^{L-1})_e = 0
//   \end{cases}
// }
// ```







