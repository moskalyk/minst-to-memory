import React, {useEffect, useState} from 'react';

import logo from './logo.svg';
import './App.css';

import MnistData from './generator/data.js'
import GM from './generator/weights.js'
import * as tf from '@tensorflow/tfjs';

// import {seed, trainBatch, gen} from './generator/gan'
import ImageUtil from './generator/image-util'
// import {getVariableValues} from './generator/misc'


// Zora

import { Zora } from '@zoralabs/zdk'
import { Wallet } from 'ethers'

import { addresses } from '@zoralabs/zdk'

import {
  constructBidShares,
  constructMediaData,
  sha256FromBuffer,
  generateMetadata,
  isMediaDataVerified
} from '@zoralabs/zdk'

import { Web3ReactProvider } from '@web3-react/core'
import { Web3Provider } from '@ethersproject/providers'
import { ethers } from "ethers";
import { useWeb3React } from '@web3-react/core'

export const MARKET_ADDRESS = addresses.rinkeby.market
export const MEDIA_ADDRESS = addresses.rinkeby.media

export const injectedConnector = new InjectedConnector({
  supportedChainIds: [
    1, // Mainet
    3, // Ropsten
    4, // Rinkeby
    5, // Goerli
    42, // Kovan
  ],
})

let ethersProvider;


/////

const wallet = Wallet.createRandom()
console.log(wallet)

const zora = new Zora(wallet, 50, MEDIA_ADDRESS, MARKET_ADDRESS)

const IPFS = require('ipfs-core')
const uint8ArrayConcat = require('uint8arrays/concat')
const all = require('it-all')
var ipfs;

function getLibrary(provider) {
  console.log(provider)
  // const provider = new ethers.providers.AlchemyProvider
  // const provider = 
  ethersProvider = new ethers.providers.Web3Provider(window.ethereum)
  console.log(ethersProvider.getSigner())

  const library = new Web3Provider(provider)
  library.pollingInterval = 12000
  return library
}


async function boot(){
  ipfs = await IPFS.create()
  const { cid } = await ipfs.add('Hello world')
  console.log(cid.string)

  // console.log(await zora.totalSupply())
}

//////// Generator

const mnistData = new MnistData();

// Input params
const BATCH = 200;
const SIZE = 28;
const INPUT_SIZE = SIZE*SIZE;
const SEED_SIZE = 40;
const SEED_STD = 3.5; 
const ONES = tf.ones([BATCH, 1]);
const ONES_PRIME = tf.ones([BATCH, 1]).mul(tf.scalar(0.98));
const ZEROS = tf.zeros([BATCH, 1]);

// Generator and discrimantor params
const DISCRIMINATOR_LEARNING_RATE = 0.025;
const GENERATOR_LEARNING_RATE = 0.025;
const dOptimizer = tf.train.sgd(DISCRIMINATOR_LEARNING_RATE);
const gOptimizer = tf.train.sgd(GENERATOR_LEARNING_RATE);

// Helper functions
const varInitNormal = (shape, mean=0, std=0.1) => tf.variable(tf.randomNormal(shape, mean, std));
const varLoad = (shape, data) => tf.variable(tf.tensor(shape, data));
const seed  = (s=BATCH) => tf.randomNormal([s, SEED_SIZE], 0, SEED_STD);


// Network arch for generator
let G1w = varInitNormal([SEED_SIZE, 140]);
let G1b = varInitNormal([140]);
let G2w = varInitNormal([140, 80]);
let G2b = varInitNormal([80]);
let G3w = varInitNormal([80, INPUT_SIZE]);
let G3b = varInitNormal([INPUT_SIZE]);

// Network arch for discriminator
let D1w = varInitNormal([INPUT_SIZE, 200]);
let D1b = varInitNormal([200]);
let D2w = varInitNormal([200, 90]);
let D2b = varInitNormal([90]);
let D3w = varInitNormal([90, 1]);
let D3b = varInitNormal([1]);

var Buffer = require('buffer/').Buffer

////////////////////////////////////////////////////////////////////////////////
// GAN functions
////////////////////////////////////////////////////////////////////////////////
function gen(xs) {
  const l1 = tf.leakyRelu(xs.matMul(G1w).add(G1b));
  const l2 = tf.leakyRelu(l1.matMul(G2w).add(G2b));
  const l3 = tf.tanh(l2.matMul(G3w).add(G3b));
  return l3;
}

function disReal(xs) {
  const l1 = tf.leakyRelu(xs.matMul(D1w).add(D1b));
  const l2 = tf.leakyRelu(l1.matMul(D2w).add(D2b));
  const logits = l2.matMul(D3w).add(D3b);
  const output = tf.sigmoid(logits);
  return [logits, output];
}

function disFake(xs) {
  return disReal(gen(xs));
}

// Copied from tensorflow core
function sigmoidCrossEntropyWithLogits(target, output) {
  return tf.tidy(function () {
    let maxOutput = tf.maximum(output, tf.zerosLike(output));
    let outputXTarget = tf.mul(output, target);
    let sigmoidOutput = tf.log(tf.add(tf.scalar(1.0), tf.exp(tf.neg(tf.abs(output)))));
    let result = tf.add(tf.sub(maxOutput, outputXTarget), sigmoidOutput);
    return result;
  });
}

// Single batch training
async function trainBatch(realBatch, fakeBatch) {
  const dcost = dOptimizer.minimize(() => {
    const [logitsReal, outputReal] = disReal(realBatch);
    const [logitsFake, outputFake] = disFake(fakeBatch);

    const lossReal = sigmoidCrossEntropyWithLogits(ONES_PRIME, logitsReal);
    const lossFake = sigmoidCrossEntropyWithLogits(ZEROS, logitsFake);
    return lossReal.add(lossFake).mean();
  }, true, [D1w, D1b, D2w, D2b, D3w, D3b]);
  await tf.nextFrame();

  // console.log([G1w, G1b, G2w, G2b, G3w, G3b])

  const gcost = gOptimizer.minimize(() => {
    const [logitsFake, outputFake] = disFake(fakeBatch);

    const lossFake = sigmoidCrossEntropyWithLogits(ONES, logitsFake);
    return lossFake.mean();
  }, true, [G1w, G1b, G2w, G2b, G3w, G3b]);
  await tf.nextFrame();

  return [dcost, gcost];
}

var cidVal;

async function catValue() {
  console.log('call on me')
  // const cat = await ipfs.cat(cidVal)
  // console.log(cid.string)
  // const file = ipfs.catReadableStream(cidVal)

  const data = uint8ArrayConcat(await all(ipfs.cat(cidVal)))
  console.log(data)
  var string = new TextDecoder().decode(data)
  console.log(JSON.parse(string))
  // for await (const chunk of ipfs.cat(cidVal)) {
  //   console.info(chunk)
  // }
//   for await (

//   const file of ipfs.get(cidVal)
//   console.log(file.type, file.path)

//   if (!file.content) continue;

//   const content = []

//   for await (const chunk of file.content) {
//     content.push(chunk)
//   }

//   console.log(content)
// }
}

function mint(mediaData) {

  const bidShares = constructBidShares(
    10, // creator share
    90, // owner share
    0 // prevOwner share
  )
  console.log(bidShares)

  zora.mint(mediaData, bidShares).then((tx) => {
    console.log(tx)
  })
}

async function getVariableValues() {

  const obj = {
    G1w: { shape: G1w.shape, data: Array.from(G1w.dataSync()) },
    G1b: { shape: G1b.shape, data: Array.from(G1b.dataSync()) },
    G2w: { shape: G2w.shape, data: Array.from(G2w.dataSync()) },
    G2b: { shape: G2b.shape, data: Array.from(G2b.dataSync()) },
    G3w: { shape: G3w.shape, data: Array.from(G3w.dataSync()) },
    G3b: { shape: G3b.shape, data: Array.from(G3b.dataSync()) },
    D1w: { shape: D1w.shape, data: Array.from(D1w.dataSync()) },
    D1b: { shape: D1b.shape, data: Array.from(D1b.dataSync()) },
    D2w: { shape: D2w.shape, data: Array.from(D2w.dataSync()) },
    D2b: { shape: D2b.shape, data: Array.from(D2b.dataSync()) },
    D3w: { shape: D3w.shape, data: Array.from(D3w.dataSync()) },
    D3b: { shape: D3b.shape, data: Array.from(D3b.dataSync()) }
  }
  
  console.log(JSON.stringify(obj))

  const b = Buffer(JSON.stringify(obj))
  let { cid } = await ipfs.add(b)
  // const { cid } = await ipfs.add(JSON.stringify(obj))
  const cat = await ipfs.cat(cid.string)
  console.log(cid.string)
  console.log(cat)
  cidVal = cid.string

  // zora

  const metadataJSON = generateMetadata('zora-20210101', {
    description: '',
    mimeType: 'text/plain',
    name: '',
    version: 'zora-20210101',
  })

  const bMetaData = Buffer(JSON.stringify(metadataJSON))

  let metaAdd = await ipfs.add(bMetaData)
  console.log(metaAdd.cid.string)
  const contentHash = sha256FromBuffer(b)
  const metadataHash = sha256FromBuffer(bMetaData)
  const mediaData = constructMediaData(
    `https://ipfs.io/ipfs/${cidVal}`,
    `https://ipfs.io/ipfs/${metaAdd.cid.string}`,
    contentHash,
    metadataHash
  )

  console.log(mediaData)

  // const contentHash = sha256FromBuffer(Buffer.from('Ours Truly,'))
  // const metadataHash = sha256FromBuffer(Buffer.from(metadataJSON))
  // const mediaData = constructMediaData(
  //   'https://ipfs.io/ipfs/bafybeifyqibqlheu7ij7fwdex4y2pw2wo7eaw2z6lec5zhbxu3cvxul6h4',
  //   'https://ipfs.io/ipfs/bafybeifpxcq2hhbzuy2ich3duh7cjk4zk4czjl6ufbpmxep247ugwzsny4',
  //   contentHash,
  //   metadataHash
  // )

  mint(mediaData)


  // return obj;
}

let canvasCount = 0;

function loadCachedModel() {
  console.log(GM)
  G1w = tf.variable(tf.tensor(GM.G1w.data).reshape(GM.G1w.shape));
  G1b = tf.variable(tf.tensor(GM.G1b.data).reshape(GM.G1b.shape));
  G2w = tf.variable(tf.tensor(GM.G2w.data).reshape(GM.G2w.shape));
  G2b = tf.variable(tf.tensor(GM.G2b.data).reshape(GM.G2b.shape));
  G3w = tf.variable(tf.tensor(GM.G3w.data).reshape(GM.G3w.shape));
  G3b = tf.variable(tf.tensor(GM.G3b.data).reshape(GM.G3b.shape));
}

async function loadMnist() {
  console.log('Start loading...');
  document.querySelectorAll('button').forEach( d => d.disabled = true);
  await mnistData.load();
  console.log('Done loading...');
  document.querySelectorAll('button').forEach(d => d.disabled = false);
  // document.querySelector('#load-status').style.display = 'none';
}

async function train(num=1500) {
  console.log('starting....');
  document.querySelector('#train').disabled = true;

  for (let i=0; i < num; i++) {
    document.querySelector('#train').innerHTML = i + '/' + num;
    const real = mnistData.nextTrainBatch(BATCH);
    const fake = seed();

    const [dcost, gcost] = await trainBatch(real.xs, fake);
    if (i % 50 === 0 || i === (num-1)) {
      console.log('i', i);
      console.log('discriminator cost', dcost.dataSync());
      console.log('generator cost', gcost.dataSync());
    }
  }
  document.querySelector('#train').innerHTML = 'Train';
  document.querySelector('#train').disabled = false;
  console.log('done...');
}

async function sampleImage() {
  await tf.nextFrame();
  const options = {
    width: SIZE,
    height: SIZE 
  };

  const canvas = document.createElement('canvas');
  canvas.setAttribute("id", `canvas-${canvasCount++}`)
  canvas.width = options.width;
  canvas.height = options.height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(options.width, options.height);
  const data = gen(seed(1)).dataSync();
  console.log(data)
  
  // Undo tanh
  /*
  for (let i=0; i < data.length; i++) {
    data[i] = 0.5 * (data[i]+1.0);
  }
  */

  const unflat = ImageUtil.unflatten(data, options);
  for (let i=0; i < unflat.length; i++) {
    imageData.data[i] = unflat[i];
  }
  // await ipfs.add(imageData)
  console.log(imageData)
  // const { cid } = await ipfs.add(imageData)
  // console.log(cid.string)

  ctx.putImageData(imageData, 0, 0);
  document.body.querySelector('#samples-container').appendChild(canvas);
  const button = document.createElement('BUTTON');

  // var btn = document.createElement("BUTTON");   // Create a <button> element
  button.innerHTML = "Mint"; 

  document.body.querySelector(`#canvas-${(canvasCount-1)}`).appendChild(button);
}


export const WalletEl = (props) => {
  const { chainId, account, activate, active } = useWeb3React()

    useEffect(async () => {
      console.log(`use effect -- account: ${account}`)
      console.log(ethersProvider)
      if(ethersProvider){
        props.setAccount(account)
      }else {
        console.log('provider NOT_SET')
      }

    }, [account,active,ethersProvider])

  const onActivateClick = async () => {
      activate(injectedConnector)
  }

  return (
    <div className="simple-form">
      {active ? (
        <>
          <div >account: {account ? account.substring(0,5)+'...' : ''}</div>
          <div >dai balance: {props.balance}</div>
          <div> ✅ </div>
        </>
      ) : (
        <Button name="connect" style={{marginLeft: '-9px'}} onClick={() => onActivateClick()}>⎈</Button>
      )}
    </div>
  )
}

const Account = (props) => {
  const [account, setAccount ] = useState('')
  const [balance, setBalance ] = useState(0)
  const [bigbalance, setBigbalance ] = useState(0)
  const [approved, setApproved ] = useState(false)

  console.log(props.id)
  console.log(account)
  return (
    <Web3ReactProvider getLibrary={getLibrary}>
      {props.base ? '' : <p>{'⇪'}</p> }
      <Wallet balance={balance} setBigbalance={setBigbalance} setBalance={setBalance} setAccount={setAccount}/>
      {/*wallet*/}
    </Web3ReactProvider>
  )
}


function App() {

  useEffect(() => {
    boot()
    loadMnist();
  })

  return (
    <div className="App">
      <h1>MNIST to Memory</h1>
      <WalletEl />
      <button id="train" onClick={async () => await train()}>Train</button>
      <button onClick={loadCachedModel}>Load weights</button>
      <button onClick={sampleImage}>Sample image</button>
      <button onClick={() => getVariableValues()}>mint model</button>
      <button onClick={async () => await catValue()}>get cat</button>
      {/*<button onClick={getVariableValues}>download</button>*/}
      <div id="samples-container"></div>
    </div>
  );
}

export default App;
