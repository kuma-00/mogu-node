import { MoguImageProcessor, MoguFoodDetector } from './index.js'
import { readFileSync, existsSync } from 'fs'

console.log('Verifying mogu-node...')

// 1. Instantiate Processor
console.log('Creating MoguImageProcessor...')
const processor = new MoguImageProcessor(224, 224)
console.log(' Processor created.')

// 2. Instantiate Detector (Mock path or check if real model exists)
// Note: We need a model file for this to really work.
// If one doesn't exist, we expect an error, but that proves the binding calls the Rust code.
const modelPath = 'model.onnx'
console.log(`Creating MoguFoodDetector with path ${modelPath}...`)

if (!existsSync(modelPath)) {
  console.warn(`Model not found at ${modelPath}. Note that postinstall script should have downloaded it.`)
}

try {
  const detector = new MoguFoodDetector(modelPath)
  console.log(' Detector created successfully (unexpected if model missing).')
} catch (e) {
  console.log(' Caught expected error (or real error) creating detector:', e.message)
  if (e.message.includes('No such file') || e.message.includes('Failed to load model')) {
    console.log('  -> This confirms the binding is trying to load the model.')
  } else {
    console.error('  -> Unexpected error type.')
  }
}

// 3. Test Preprocessing with dummy buffer
console.log('Testing preprocess with dummy buffer...')
// Create a fake small image buffer (this might fail if image crate expects valid header)
// Actually image crate `load_from_memory` needs valid format.
// Let's try to read one of the assets if available.
const assetPath = 'mogu/assets/goldfish.jpg'

if (existsSync(assetPath)) {
  console.log(`Reading ${assetPath}...`)
  const buffer = readFileSync(assetPath)
  try {
    const tensor = processor.preprocess(buffer)
    console.log(' Preprocess successful, got tensor:', tensor)
  } catch (e) {
    console.error(' Preprocess failed:', e)
  }
} else {
  console.log(' No asset found to test preprocessing.')
}

console.log('Verification finished.')
