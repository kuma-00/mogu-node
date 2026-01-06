const { downloadFile } = require('@huggingface/hub')
const fs = require('fs')
const path = require('path')

async function main() {
  console.log('Downloading model from Hugging Face...')
  try {
    const response = await downloadFile({
      repo: 'onnx-community/mobilenetv4_conv_small.e2400_r224_in1k',
      path: 'onnx/model.onnx',
    })

    // In some environments/versions, downloadFile returns a WebBlob/Response
    // Check if it's a blob-like object
    let buffer
    if (response && typeof response.arrayBuffer === 'function') {
      console.log('Response is a Blob, converting to buffer...')
      const arrayBuffer = await response.arrayBuffer()
      buffer = Buffer.from(arrayBuffer)
    } else if (typeof response === 'string') {
      // It's a path
      console.log('Response is a path, reading file...')
      buffer = fs.readFileSync(response)
    } else {
      throw new Error(`Unexpected response type: ${typeof response}`)
    }

    const targetPath = path.join(__dirname, '../model.onnx')
    fs.writeFileSync(targetPath, buffer)
    console.log(`Model downloaded to ${targetPath}`)
  } catch (error) {
    console.error('Failed to download model:', error)
    process.exit(1)
  }
}

main()
