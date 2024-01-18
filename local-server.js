'use strict'

const path = require('path')
const { Backend, RestApi, ServingFilesEndpoint, NotFoundEndpoint } = require('@cuties/rest')
const CustomIndexEndpoint = require('./endpoints/CustomIndexEndpoint')

const mapperForStatic = (url) => {
  const parts = url.split('?')[0].split('/').filter(part => part !== '').slice(1)
  if (parts.length === 0) {
    return 'index.html'
  }
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
      new RegExp(/^\/nn-math-web\/?(html|js|json|image|css|md)?/),
      mapperForStatic,
      {},
      new NotFoundEndpoint(new RegExp(/\/not-found/))
    ),
    new NotFoundEndpoint(new RegExp(/\/not-found/))
  )
).call()

